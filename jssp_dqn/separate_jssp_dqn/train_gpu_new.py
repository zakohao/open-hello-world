import os
import re
import ast
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
import concurrent.futures
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
import psutil
import GPUtil

# 获取当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 详细检查CUDA可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    #print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    #print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    #print("使用CPU")

#print(f"PyTorch版本: {torch.__version__}")
#print(f"CUDA可用: {torch.cuda.is_available()}")
#if torch.cuda.is_available():
    #print(f"CUDA版本: {torch.version.cuda}")
    #print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# Hyperparameters
EPISODES = 1000  # 减少总episode数，但提高每个episode的效率
GAMMA = 0.95
LR = 0.0005  # 增加学习率
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 128  # 增加批次大小
MEMORY_SIZE = 50000  # 增加回放缓冲区大小
UPDATE_FREQUENCY = 2  # 更新频率
NUM_ENVS = 8  # 并行环境数量

class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape
        
        # 获取所有机器ID并创建映射
        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}
        
        self.reset()
    
    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0] * len(self.all_machines)  # 为每个机器创建时间表
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        
        # 跟踪机器利用率
        self.machine_total_time = [0] * len(self.all_machines)  # 机器总可用时间
        self.machine_working_time = [0] * len(self.all_machines)  # 机器实际加工时间
        
        return self._get_state()
    
    def _get_state(self):
        # 简化状态表示
        state = []
        
        # 1. 当前可执行操作的机器和处理时间
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])
        
        # 2. 机器当前负载
        for machine_time in self.time_table:
            state.append(machine_time)
        
        # 3. 作业完成情况
        for job_end_time in self.job_end_times:
            state.append(job_end_time)
        
        # 4. 机器利用率信息
        for i in range(len(self.all_machines)):
            if self.machine_total_time[i] > 0:
                util = self.machine_working_time[i] / self.machine_total_time[i]
            else:
                util = 0
            state.append(util)
            
        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self):
        """返回当前可执行的动作（作业）列表"""
        valid_actions = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            # 如果作业还有工序未完成，且前一道工序已完成（如果是第一道工序则总是可执行）
            if op < self.num_machines and (op == 0 or self.current_step[job] > 0):
                valid_actions.append(job)
        return valid_actions
    
    def step(self, action):
        job = action
        op = self.current_step[job]
        
        # 检查动作是否有效
        if self.done or op >= self.num_machines or job not in self.get_valid_actions():
            return self._get_state(), -10, self.done, {}
        
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time
        
        # 更新机器总时间和工作时间
        machine_available_time = max(self.time_table[machine_idx], self.job_end_times[job])
        self.machine_total_time[machine_idx] = end_time
        self.machine_working_time[machine_idx] += proc_time
        
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        
        self.schedule.append((job, op, machine_id, start_time, end_time))
        
        self.current_step[job] += 1
        self.done = all(step >= self.num_machines for step in self.current_step)
        
        # 计算机器利用率奖励
        utilization_reward = 0
        if self.machine_total_time[machine_idx] > 0:
            utilization = self.machine_working_time[machine_idx] / self.machine_total_time[machine_idx]
            utilization_reward = utilization * 5
        
        # 组合奖励：负的处理时间 + 机器利用率奖励
        reward = -proc_time + utilization_reward
        # 如果所有作业完成，添加最终奖励
        if self.done:
            makespan = max(self.job_end_times)
            reward += -makespan * 0.1
        
        return self._get_state(), reward, self.done, {
            "schedule": self.schedule,
            "machine_utilization": [self.machine_working_time[i] / self.machine_total_time[i] 
                                   if self.machine_total_time[i] > 0 else 0 
                                   for i in range(len(self.all_machines))]
        }

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class ExperienceDataset(Dataset):
    def __init__(self, memory):
        self.memory = memory
        
    def __len__(self):
        return len(self.memory)
        
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.memory[idx]
        return (
            torch.FloatTensor(state),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state),
            torch.BoolTensor([done])
        )

def parse_tuple(cell):
    """专门解析元组格式 ('1', '110.0')"""
    try:
        # 移除可能的空白字符和引号
        cell = cell.strip().replace("'", "").replace('"', "")
        
        # 检查是否包含括号
        if '(' in cell and ')' in cell:
            # 提取括号内的内容
            content = cell[cell.find('(')+1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                machine = int(parts[0].strip())
                time = float(parts[1].strip())
                return machine, time
    except Exception as e:
        pass
    
    # 如果解析失败，返回0值
    return 0, 0.0

def load_single_csv(file_path):
    """从CSV文件加载数据并解析为机器分配矩阵和处理时间矩阵"""
    try:
        # 尝试不同的分隔符
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    # 读取前几行来检测分隔符
                    lines = [f.readline() for _ in range(5)]
                    f.seek(0)
                    
                    # 创建reader对象
                    reader = csv.reader(f, delimiter=delimiter)
                    data = list(reader)
                    
                    # 如果成功读取多行数据，使用此分隔符
                    if len(data) > 1 and len(data[0]) > 1:
                        print(f"使用分隔符 '{delimiter}' 读取文件: {os.path.basename(file_path)}")
                        break
            except:
                continue
        
        # 如果仍然无法读取，使用默认分隔符
        if 'data' not in locals():
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                data = list(reader)
            print(f"使用默认分隔符读取文件: {os.path.basename(file_path)}")
        
        # 获取标题行，确定工序数量
        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0  # 减去作业名列
        
        # 解析数据行
        machine_data = []
        time_data = []
        
        for row in data[1:]:
            # 跳过空行
            if not row or len(row) == 0:
                continue
            
            job_machines = []
            job_times = []
            
            # 跳过作业名列（第一列）
            for cell in row[1:1+num_processes]:
                machine, time = parse_tuple(cell)
                job_machines.append(machine)
                job_times.append(time)
            
            machine_data.append(job_machines)
            time_data.append(job_times)
        
        # 转换为NumPy数组
        machine_array = np.array(machine_data, dtype=int)
        time_array = np.array(time_data, dtype=float)
        
        # 检查数据有效性
        if machine_array.size == 0 or time_array.size == 0:
            print(f"文件 {os.path.basename(file_path)} 数据为空")
            return None, None
        if machine_array.shape != time_array.shape:
            print(f"文件 {os.path.basename(file_path)} 机器分配和处理时间形状不匹配")
            return None, None
        
        # 打印成功消息
        print(f"成功加载文件: {os.path.basename(file_path)}, 形状: {machine_array.shape}")
        return machine_array, time_array
    
    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None, None

def load_folder_data(folder_path):
    """从文件夹加载所有CSV文件数据"""
    datasets = []
    valid_files = 0
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return datasets
    
    # 打印文件夹信息
    print(f"扫描文件夹: {folder_path}")
    print(f"找到文件: {len(os.listdir(folder_path))}")
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否为CSV文件
        if filename.lower().endswith('.csv') and 'gene' in filename.lower():
            file_path = os.path.join(folder_path, filename)
            
            # 加载文件数据
            machine_assignments, processing_times = load_single_csv(file_path)
            
            # 验证数据有效性
            if machine_assignments is not None and processing_times is not None:
                datasets.append((machine_assignments, processing_times))
                valid_files += 1
    
    print(f"总共加载了 {valid_files} 个有效数据集")
    return datasets

def run_episode(model, dataset, epsilon, device):
    """运行一个episode并返回经验"""
    ma, pt = dataset
    env = JSSPEnv(ma, pt)
    state = env.reset()
    experiences = []
    total_reward = 0
    done = False
    
    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                valid_q_values = q_values.clone()
                for i in range(env.num_jobs):
                    if i not in valid_actions:
                        valid_q_values[i] = -float('inf')
                action = valid_q_values.argmax().item()
        
        next_state, reward, done, info = env.step(action)
        experiences.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
    return experiences, total_reward, info if done else None

def parallel_collect_experience(model, datasets, epsilon, device, num_envs):
    """并行收集经验"""
    results = []
    
    # 使用线程池并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_envs) as executor:
        futures = []
        for i in range(num_envs):
            dataset = random.choice(datasets)
            futures.append(executor.submit(run_episode, model, dataset, epsilon, device))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"并行收集经验时出错: {e}")
    
    return results

def train_optimized(training_datasets):
    """优化后的训练函数"""
    if not training_datasets:
        print("没有可用的训练数据!")
        return
    
    # 初始化模型
    sample_env = JSSPEnv(*training_datasets[0])
    state_dim = len(sample_env.reset())
    action_dim = sample_env.num_jobs
    
    print(f"状态维度: {state_dim}, 动作空间: {action_dim}")
    
    model = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
    criterion = nn.MSELoss()
    scaler = amp.GradScaler()  # 混合精度
    
    # 使用更大的回放缓冲区
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    utilizations = []
    losses = []
    
    # 预热GPU
    if torch.cuda.is_available():
        print("预热GPU...")
        dummy_input = torch.randn(32, state_dim).to(device)
        dummy_target = torch.randn(32, action_dim).to(device)
        for _ in range(10):
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("GPU预热完成")
    
    # 训练循环
    start_time = time.time()
    for ep in tqdm(range(EPISODES), desc="训练进度"):
        # 监控资源使用情况
        if ep % 50 == 0:
            print(f"\n=== Episode {ep} 资源使用情况 ===")
            #monitor_resources()
            print(f"探索率: {epsilon:.4f}")
            print(f"回放缓冲区大小: {len(memory)}")
            if makespans:
                print(f"最近makespan: {makespans[-1]:.2f}")
        
        # 并行收集经验
        results = parallel_collect_experience(model, training_datasets, epsilon, device, NUM_ENVS)
        
        # 将经验添加到回放缓冲区
        for experiences, total_reward, info in results:
            memory.extend(experiences)
            
            if info and 'schedule' in info:
                makespan = max([end_time for _, _, _, _, end_time in info['schedule']])
                makespans.append(makespan)
                
                if 'machine_utilization' in info:
                    avg_utilization = np.mean(info["machine_utilization"])
                    utilizations.append(avg_utilization)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = info["schedule"].copy()
        
        # 经验回放
        if len(memory) >= BATCH_SIZE and ep % UPDATE_FREQUENCY == 0:
            # 使用DataLoader进行批量处理
            dataset = ExperienceDataset(list(memory))
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
            
            for batch in dataloader:
                states, actions, rewards, next_states, dones = batch
                
                # 将数据移动到GPU
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                # 使用混合精度训练
                with amp.autocast():
                    current_q = model(states).gather(1, actions)
                    next_q = model(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + GAMMA * next_q * (~dones)
                    loss = criterion(current_q, target_q.detach())
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                losses.append(loss.item())
        
        # 衰减探索率和学习率调度
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        scheduler.step()
    
    # 训练结束统计
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\n训练完成! 总时间: {training_time:.2f}秒")
    print(f"平均每秒处理: {EPISODES/training_time:.2f} episodes")
    print(f"最佳makespan: {best_makespan}")
    
    # 保存模型
    torch.save(model.state_dict(), 'new_GPU_R(p+u)_LR0.0005_1000ep_barch128_jssp_model_npcb_(10,2)_(3,1)_(13)_(3).pth')
    print("模型已保存")
    
    return model

if __name__ == "__main__":
    #print("正在使用", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", "训练")
    
    # 指定文件夹路径
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    
    # 加载训练数据
    training_datasets = load_folder_data(folder_path)
    
    # 训练模型
    if training_datasets:
        model = train_optimized(training_datasets)
    else:
        print("未找到有效的训练数据，请检查文件夹路径和文件格式。")