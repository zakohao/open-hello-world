import os
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

# 获取当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 详细检查CUDA可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = torch.device("cpu")
    print("使用CPU")

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

# Hyperparameters
EPISODES = 3000
GAMMA = 0.95
LR = 0.0005
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 128  
MEMORY_SIZE = 20000 
UPDATE_FREQUENCY = 2 
NUM_ENVS = 4 

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
            return self._get_state(), -20, self.done, {}
        
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time
        
        # 计算空闲时间（机器等待时间）
        machine_idle_time = start_time - self.time_table[machine_idx]
        job_waiting_time = start_time - self.job_end_times[job]

         # 更新机器总时间和工作时间
        self.machine_total_time[machine_idx] = end_time
        self.machine_working_time[machine_idx] += proc_time
    
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
    
        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.current_step[job] += 1
        self.done = all(step >= self.num_machines for step in self.current_step)
        
        # ========== 改进的奖励计算 ==========
    
        # 1. 基础奖励（负的处理时间）
        base_reward = -proc_time * 0.1  # 降低基础惩罚

        # 2. 机器利用率奖励
        utilization_reward = 0
        if self.machine_total_time[machine_idx] > 0:
            utilization = self.machine_working_time[machine_idx] / self.machine_total_time[machine_idx]
            utilization_reward = utilization * 3
        
        # 3. 空闲时间惩罚（鼓励减少机器空闲）
        idle_penalty = -machine_idle_time * 0.05

#        4. 作业等待时间惩罚（鼓励减少作业等待）
        waiting_penalty = -job_waiting_time * 0.03

        # 5. 进度平衡奖励（鼓励均衡推进所有作业）
        progress_reward = 0
        if not self.done:
            # 计算所有作业的平均进度
            avg_progress = sum(self.current_step) / (self.num_jobs * self.num_machines)
            # 当前作业进度与平均进度的差异
            current_progress = self.current_step[job] / self.num_machines
            progress_diff = abs(current_progress - avg_progress)
            progress_reward = (1 - progress_diff) * 2  # 越接近平均进度奖励越高
        
        # 6. 完成奖励（当作业完成时）
        completion_reward = 0
        if self.current_step[job] == self.num_machines:
            completion_reward = 50  # 作业完成给予较大奖励

        # 7. 最终makespan奖励（当所有作业完成时）
        final_reward = 0
        if self.done:
            makespan = max(self.job_end_times)
            # makespan越小奖励越大（负的makespan作为奖励）
            final_reward = -makespan * 0.2
            # 额外奖励：基于与最优解的相对性能
            estimated_optimal = sum(self.processing_times.flatten()) / len(self.all_machines)
            if makespan < estimated_optimal * 1.5:  # 如果在最优解的1.5倍内
                final_reward += 100

        # 组合所有奖励成分
        reward = (base_reward + utilization_reward + idle_penalty + 
                  waiting_penalty + progress_reward + completion_reward + final_reward)
        
        return self._get_state(), reward, self.done, {
            "schedule": self.schedule,
            "machine_utilization": [self.machine_working_time[i] / self.machine_total_time[i] 
                                   if self.machine_total_time[i] > 0 else 0 
                                   for i in range(len(self.all_machines))],
            "makespan": max(self.job_end_times) if self.done else None
        }

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
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

def run_episode(model, dataset, epsilon):
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

def parallel_collect_experience(model, datasets, epsilon):
    """并行收集经验"""
    results = []
    for dataset in datasets:
        experiences, total_reward, info = run_episode(model, dataset, epsilon)
        results.append((experiences, total_reward, info))
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
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # 使用更大的回放缓冲区
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    utilizations = []
    
    # 预加载数据集到列表
    datasets_list = list(training_datasets)
    
    for ep in tqdm(range(EPISODES), desc="训练进度"):
        # 并行收集经验
        selected_datasets = random.sample(datasets_list, min(NUM_ENVS, len(datasets_list)))
        results = parallel_collect_experience(model, selected_datasets, epsilon)
        
        # 将经验添加到回放缓冲区
        for experiences, total_reward, info in results:
            memory.extend(experiences)
            
            if info:
                makespan = max([end_time for _, _, _, _, end_time in info['schedule']])
                makespans.append(makespan)
                
                avg_utilization = np.mean(info["machine_utilization"])
                utilizations.append(avg_utilization)
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = info["schedule"].copy()
        
        # 经验回放 - 使用更大的批次和更频繁的更新
        if len(memory) >= BATCH_SIZE and ep % UPDATE_FREQUENCY == 0:
            batch = random.sample(memory, BATCH_SIZE)
            
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # 使用预分配的张量
            states_tensor = torch.zeros((BATCH_SIZE, state_dim), dtype=torch.float32, device=device)
            next_states_tensor = torch.zeros((BATCH_SIZE, state_dim), dtype=torch.float32, device=device)
            rewards_tensor = torch.zeros((BATCH_SIZE, 1), dtype=torch.float32, device=device)
            actions_tensor = torch.zeros((BATCH_SIZE, 1), dtype=torch.long, device=device)
            dones_tensor = torch.zeros((BATCH_SIZE, 1), dtype=torch.bool, device=device)
            
            # 填充数据
            for i in range(BATCH_SIZE):
                states_tensor[i] = torch.FloatTensor(states[i])
                next_states_tensor[i] = torch.FloatTensor(next_states[i])
                rewards_tensor[i] = torch.FloatTensor([rewards[i]])
                actions_tensor[i] = torch.LongTensor([actions[i]])
                dones_tensor[i] = torch.BoolTensor([dones[i]])
            
            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda'):
                current_q = model(states_tensor).gather(1, actions_tensor)
                next_q = model(next_states_tensor).max(1)[0].unsqueeze(1)
                target_q = rewards_tensor + GAMMA * next_q * (~dones_tensor)
                loss = criterion(current_q, target_q.detach())
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 衰减探索率和更新学习率
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        scheduler.step()
    
    # 保存模型
    torch.save(model.state_dict(), 'new_GPU_LR0.0005_3000ep_brach128_jssp_model_npcb_(10,2)_(3,1)_(13)_(3).pth')
    print(f"最佳makespan: {best_makespan}")
    
    return model

if __name__ == "__main__":
    print("正在使用", torch.cuda.get_device_name(0), "训练")
    # 指定文件夹路径
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    
    # 加载训练数据
    training_datasets = load_folder_data(folder_path)
    
    # 训练模型
    if training_datasets:
        train_optimized(training_datasets)
    else:
        print("未找到有效的训练数据，请检查文件夹路径和文件格式。")