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

# Hyperparameters
EPISODES = 3000
GAMMA = 0.95
LR = 0.0001
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")

a=input()

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
        return self._get_state()
    
    def _get_state(self):
        state = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.append([machine, proc_time])
            else:
                state.append([0, 0])
        return np.array(state).flatten()
    
    def step(self, action):
        job = action
        if self.done or self.current_step[job] >= self.num_machines:
            return self._get_state(), -10, self.done, []
        
        op = self.current_step[job]
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time
        
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.schedule.append((job, op, machine_id, start_time, proc_time))
        
        self.current_step[job] += 1
        self.done = all(step >= self.num_machines for step in self.current_step)
        
        #reward verson1
        #reward = -end_time if self.done else 0 
        
        #reward verson2
        reward = -proc_time 
        self.current_step[job] += 1 
        self.done = all(step >= self.num_machines for step in self.current_step) 
        
        return self._get_state(), reward, self.done, self.schedule

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

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

def train(training_datasets):
    """使用多个数据集训练模型"""
    if not training_datasets:
        print("没有可用的训练数据!")
        return
    
    # 检查所有数据集是否具有相同的维度
    sample_env = JSSPEnv(*training_datasets[0])
    state_dim = len(sample_env.reset())
    action_dim = sample_env.num_jobs
    
    print(f"状态维度: {state_dim}, 动作空间: {action_dim}")
    
    # 初始化模型并移动到GPU
    model = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    
    # 训练循环
    for ep in tqdm(range(EPISODES), desc="训练进度"):
        # 随机选择一个训练数据集
        ma, pt = random.choice(training_datasets)
        env = JSSPEnv(ma, pt)
        state = env.reset()
        state = torch.FloatTensor(state).to(device)  # 移动到GPU
        total_reward = 0
        done = False
        
        while not done:
            # 选择动作
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()
            
            # 执行动作
            next_state, reward, done, schedule = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)  # 移动到GPU
            memory.append((state.cpu(), action, reward, next_state_tensor.cpu(), done))  # 存储时移回CPU
            state = next_state_tensor
            total_reward += reward
            
            # 记录完成时的数据
            if done:
                makespan = max(env.job_end_times)
                makespans.append(makespan)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule.copy()
            
            # 经验回放
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # 将批次数据移动到GPU
                states = torch.stack(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.stack(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)
                
                current_q = model(states).gather(1, actions)
                next_q = model(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * next_q * (~dones)
                
                loss = criterion(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # 衰减探索率
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    
    # 保存模型
    torch.save(model.state_dict(), 'GPU_newR_LR0.0001_3000ep_jssp_model_npcb_(10,2)_(3,1)_(13)_(3).pth')
    print(f"模型已保存")
    
    return model

if __name__ == "__main__":
    # 指定文件夹路径
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    #folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[20],odder_std_dev[5]\lot_mean[7],lot_std_dev[3]\machine[13]\seed[3]"
    #folder_path = r"D:\pysrc\wang_data\jobset\double normal\j1[5,1],j2[8,1]\p1[5,1],p2[8,1]\machine[4]\t1[10,1],t2[15,1]\seed[3]"
    
    # 加载训练数据
    training_datasets = load_folder_data(folder_path)
    
    # 训练模型
    if training_datasets:
        train(training_datasets)
    else:
        print("未找到有效的训练数据，请检查文件夹路径和文件格式。")