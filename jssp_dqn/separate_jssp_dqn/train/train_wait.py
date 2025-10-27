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
else:
    device = torch.device("cpu")
    print("使用CPU")

# Hyperparameters
EPISODES = 2000  
GAMMA = 0.95           # 越接近1表示越重视长期回报
LR = 0.0005           # 较小的学习率使训练更稳定但收敛较慢
EPSILON_DECAY = 0.9995  # 控制从探索(随机选择)到利用(选择最优动作)的过渡速度
MIN_EPSILON = 0.01     # 最小探索率 越大越随机选择，越小越选择当前最适
BATCH_SIZE = 128        # 固定批量大小
MEMORY_SIZE = 100000   # 存储(state, action, reward, next_state)经验元组的个数
UPDATE_TARGET_FREQUENCY = 30  # 每UPDATE_TARGET_FREQUENCY步更新一次目标网络
MIN_COMPLETE_EPISODES = 2     # 完成MIN_COMPLETE_EPISODES个数据集的完整调度后才开始训练

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
        self.time_table = [0] * len(self.all_machines)
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        self.total_processing_time = np.sum(self.processing_times)
        self.completed_ops = 0  # 初始化累计完成工序数
        return self._get_state()
    
    def _get_state(self):
        state = []
        
        # 1. 每个作业的下一道工序信息
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])  # 作业已完成
        
        # 2. 机器当前时间
        state.extend(self.time_table)
        
        # 3. 作业完成时间
        state.extend(self.job_end_times)
        
        # 4. 进度信息
        completed_ops = sum(self.current_step)
        total_ops = self.num_jobs * self.num_machines
        progress = completed_ops / total_ops if total_ops > 0 else 0
        state.append(progress)

        # 5. 机器相对负载（帮助识别瓶颈）
        if len(self.time_table) > 0:
            max_machine_time = max(self.time_table)
            if max_machine_time > 0:
                machine_relative_load = [t / max_machine_time for t in self.time_table]
                state.extend(machine_relative_load)
            else:
                state.extend([0] * len(self.time_table))
        else:
            state.extend([0] * len(self.all_machines))

        # 6. 作业剩余工作量
        for job in range(self.num_jobs):
            remaining_ops = self.num_machines - self.current_step[job]
            if remaining_ops > 0:
                remaining_time = sum(self.processing_times[job, self.current_step[job]:])
                state.append(remaining_time)
            else:
                state.append(0)

        # 7. 可选动作的特征（处理时间分布）
        valid_actions = self.get_valid_actions()
        if valid_actions:
            valid_proc_times = []
            for job in valid_actions:
                op = self.current_step[job]
                proc_time = self.processing_times[job, op]
                valid_proc_times.append(proc_time)
        
            if valid_proc_times:
                state.append(min(valid_proc_times))  # 最短处理时间
                state.append(max(valid_proc_times))  # 最长处理时间
                state.append(np.mean(valid_proc_times))  # 平均处理时间
            else:
                state.extend([0, 0, 0])
        else:
            state.extend([0, 0, 0])

        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self):
        """返回当前可执行的动作（作业）列表"""
        valid_actions = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:  # 作业还有工序未完成
                valid_actions.append(job)
        return valid_actions
    
    def step(self, action):
        job = action
        op = self.current_step[job]
        
        # 检查动作是否有效
        if self.done or op >= self.num_machines or job not in self.get_valid_actions():
            return self._get_state(), -50, self.done, {}  # 增加无效动作惩罚
        
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        # 计算开始时间（作业完成时间和机器空闲时间的最大值）
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time

        # 保存状态用于计算延迟相关奖励
        prev_machine_idle_time = self.time_table[machine_idx] - max(self.time_table)
        prev_job_wait_time = self.time_table[machine_idx] - self.job_end_times[job]
        
        # 更新状态
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.current_step[job] += 1
        
        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.done = all(step >= self.num_machines for step in self.current_step)
        
        # 奖励函数
        
        time_penalty = -proc_time * 0.05   

        # 进度差分奖励
        total_ops = self.num_jobs * self.num_machines
        progress_prev = self.completed_ops / total_ops if total_ops > 0 else 0
        self.completed_ops += 1
        progress_now = self.completed_ops / total_ops
        progress_reward = (progress_now - progress_prev) * 30  

        # 机器负载平衡
        machine_loads = [t for t in self.time_table]
        if len(machine_loads) > 1:
            load_balance_reward = -np.std(machine_loads) * 0.05
        else:
            load_balance_reward = 0

        # 单作业完成奖励
        completion_reward = 15 if self.current_step[job] == self.num_machines else 0

        # 机会成本惩罚：如果有其他作业可以更快完成但没被选择
        opportunity_cost_penalty = 0
        valid_actions = self.get_valid_actions()
        if len(valid_actions) > 1:
            # 计算当前选择作业的处理时间
            current_proc_time = proc_time
        
            # 找到最快可完成的作业（处理时间最短）
            fastest_job = None
            fastest_time = float('inf')
            for other_job in valid_actions:
                if other_job != job:
                    other_op = self.current_step[other_job]
                    other_proc_time = self.processing_times[other_job, other_op]
                    if other_proc_time < fastest_time:
                        fastest_time = other_proc_time
                        fastest_job = other_job
        
            # 如果当前选择的不是最快的，给予轻微惩罚
            if fastest_job is not None and current_proc_time > fastest_time:
                opportunity_cost_penalty = -0.1 * (current_proc_time - fastest_time)

        # 机器空闲时间利用奖励
        machine_utilization_reward = 0
        current_machine_busy_time = self.time_table[machine_idx]
        max_machine_time = max(self.time_table) if self.time_table else 0
    
        # 如果这个选择减少了机器的空闲时间
        if prev_machine_idle_time > 0 and start_time == self.time_table[machine_idx]:
            # 立即使用空闲机器，给予奖励
            machine_utilization_reward = 0.5
        elif prev_job_wait_time > 0:
            # 作业等待了一段时间，轻微惩罚
            machine_utilization_reward = -0.1

        # 关键路径启发式奖励
        critical_path_reward = 0
        if not self.done:
            # 计算剩余工作量最大的作业
            remaining_work = []
            for j in range(self.num_jobs):
                remaining_ops = self.num_machines - self.current_step[j]
                if remaining_ops > 0:
                    remaining_time = sum(self.processing_times[j, self.current_step[j]:])
                    remaining_work.append((j, remaining_time))
        
            if remaining_work:
                max_remaining_job, max_remaining_time = max(remaining_work, key=lambda x: x[1])
                # 如果选择了剩余工作量最大的作业，给予奖励
                if job == max_remaining_job:
                    critical_path_reward = 0.3

        # 瓶颈机器识别奖励
        bottleneck_reward = 0
        if len(self.time_table) > 0:
            # 找出当前最忙的机器（可能的瓶颈）
            max_load_machine_idx = np.argmax(self.time_table)
            max_load = self.time_table[max_load_machine_idx]
            avg_load = np.mean(self.time_table)
        
            # 如果这个机器明显比其他机器忙，且我们选择了使用这个机器的作业
            if max_load > avg_load * 1.2 and machine_idx == max_load_machine_idx:
                # 避免进一步加重瓶颈，轻微惩罚
                bottleneck_reward = -0.2
            elif machine_idx != max_load_machine_idx:
                # 选择了非瓶颈机器，给予奖励
                bottleneck_reward = 0.2
    

        # 最终效率奖励
        final_reward = 0
        if self.done:
            makespan = max(self.job_end_times)
            theoretical_lower_bound = self.total_processing_time / len(self.all_machines)
            efficiency = theoretical_lower_bound / makespan if makespan > 0 else 0
            final_reward = efficiency * 100

        # 组合奖励
        reward = (time_penalty + load_balance_reward + progress_reward + 
                 completion_reward + final_reward + 
                 opportunity_cost_penalty + machine_utilization_reward + 
                 critical_path_reward + bottleneck_reward)
        
        info = {
            "schedule": self.schedule.copy(),
            "makespan": max(self.job_end_times) if self.done else None,
            "machine_utilization": [t/max(self.time_table) if max(self.time_table) > 0 else 0 
                               for t in self.time_table],
            "reward_components": {
                "time_penalty": time_penalty,
                "progress_reward": progress_reward,
                "opportunity_cost": opportunity_cost_penalty,
                "machine_utilization": machine_utilization_reward,
                "critical_path": critical_path_reward,
                "bottleneck": bottleneck_reward
            }
        }
        
        return self._get_state(), reward, self.done, info

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

def train_target(training_datasets):
    if not training_datasets:
        print("没有可用的训练数据!")
        return None, None
    
    # 初始化模型
    sample_env = JSSPEnv(*training_datasets[0])
    state_dim = len(sample_env.reset())
    action_dim = sample_env.num_jobs
    
    print(f"状态维度: {state_dim}, 动作空间: {action_dim}")
    print(f"可用数据集数量: {len(training_datasets)}")
    
    # 创建主网络和目标网络
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    
    # 初始化目标网络与主网络相同
    target_model.load_state_dict(model.state_dict())
    target_model.eval()  # 目标网络设置为评估模式
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    
    # 添加训练控制变量
    total_steps = 0
    complete_datasets_count = 0  # 完成的完整数据集调度次数
    training_started = False     # 标记是否开始训练
    
    print("第一阶段：收集经验")
    
    selected_datasets = random.sample(training_datasets, MIN_COMPLETE_EPISODES)
    print(f"已选择 {len(selected_datasets)} 个不同的数据集用于第一阶段")
    
    pbar_stage1 = tqdm(total=MIN_COMPLETE_EPISODES, desc="收集经验进度")
    
    for i, (ma, pt) in enumerate(selected_datasets):
        env = JSSPEnv(ma, pt)
        state = env.reset()
        
        # 在当前数据集上完成完整调度
        while not env.done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # 纯探索策略（不利用模型）
            action = random.choice(valid_actions)
            
            next_state, reward, done, info = env.step(action)
            
            # 存储到全局记忆库
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_steps += 1
        
        # 记录完成的调度
        if env.done:
            complete_datasets_count += 1
            makespan = info["makespan"]
            makespans.append(makespan)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = info["schedule"].copy()
            
            pbar_stage1.update(1)
            pbar_stage1.set_postfix({
                '当前数据集': f'{i+1}/{MIN_COMPLETE_EPISODES}',
                'best_makespan': f'{best_makespan:.1f}',
                'memory_size': len(memory)
            })
    
    pbar_stage1.close()
    print(f"第一阶段完成！使用了 {len(selected_datasets)} 个不同数据集")
    print("开始第二阶段：训练模型...")
    training_started = True
    
    # 第二阶段：开始训练，每个episode都是一个完整的数据集调度
    progress_bar = tqdm(range(EPISODES), desc="训练进度")
    
    # 用于跟踪第二阶段使用的数据集
    stage2_used_datasets = set()
    
    for ep in progress_bar:
        # 随机选择一个训练数据集
        ma, pt = random.choice(training_datasets)
        
        # 记录使用的数据集
        dataset_hash = hash((ma.tobytes(), pt.tobytes()))
        stage2_used_datasets.add(dataset_hash)
        
        env = JSSPEnv(ma, pt)
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # 在当前数据集上完成完整调度
        while not env.done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
                
            # ε-贪婪策略
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    
                    # 创建有效动作mask
                    mask = torch.ones_like(q_values, dtype=torch.bool)
                    for i in range(action_dim):
                        if i not in valid_actions:
                            mask[0, i] = False
                    
                    q_values_masked = q_values.clone()
                    q_values_masked[~mask] = -float('inf')
                    action = q_values_masked.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            
            # 存储到全局记忆库
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1
            
            # 经验回放（在完整调度过程中进行）
            #if len(memory) % BATCH_SIZE == 0:
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)
                
                # 使用目标网络计算目标Q值
                with torch.no_grad():
                    next_q = target_model(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q = rewards_tensor + GAMMA * next_q * (~dones_tensor)
                
                # 使用主网络计算当前Q值
                current_q = model(states_tensor).gather(1, actions_tensor)
                
                # 计算损失并更新主网络
                loss = criterion(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 定期更新目标网络
            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                target_model.load_state_dict(model.state_dict())
        
        # 记录结果（一个完整数据集调度完成）
        if env.done:
            complete_datasets_count += 1
            makespan = info["makespan"]
            makespans.append(makespan)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = info["schedule"].copy()
        
        # 更新探索率（每个episode结束后更新）
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        
        # 更新进度条
        avg_makespan = np.mean(makespans[-100:]) if makespans else 0
        progress_bar.set_postfix({
            'epsilon': f'{epsilon:.3f}',
            'best_makespan': f'{best_makespan:.1f}',
            'avg_makespan': f'{avg_makespan:.1f}',
            'episode_steps': steps,
            'memory_size': len(memory),
            'stage2_datasets': len(stage2_used_datasets)
        })
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'dynamic_jssp_model_with_target.pth')
    print(f"最佳makespan: {best_makespan}")
    print(f"第一阶段使用了 {MIN_COMPLETE_EPISODES} 个不同数据集")
    print(f"第二阶段使用了 {len(stage2_used_datasets)} 个不同数据集")
    print(f"总共完成了 {complete_datasets_count} 个完整数据集调度")
    
    return model, target_model, best_schedule

if __name__ == "__main__":
    print("开始训练JSSP模型")
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    
    # 加载训练数据
    training_datasets = load_folder_data(folder_path)
    
    # 训练模型
    if training_datasets:
        model, target_model, best_schedule = train_target(training_datasets)
        print("训练完成！")
        
        # 测试最佳调度
        if best_schedule:
            makespan = max([end for _, _, _, _, end in best_schedule])
            print(f"最佳调度方案 (makespan: {makespan:.1f})")
    else:
        print("未找到有效的训练数据")