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
LR = 0.0005            # 较小的学习率使训练更稳定但收敛较慢
EPSILON_DECAY = 0.9995 # 控制从探索(随机选择)到利用(选择最优动作)的过渡速度
MIN_EPSILON = 0.01     # 最小探索率 越大越随机选择，越小越选择当前最适
BATCH_SIZE = 128       # 固定批量大小
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
        self.current_step = [0] * self.num_jobs              # 每个作业当前做到第几道工序
        self.time_table = [0] * len(self.all_machines)       # 每台机器被排到的结束时间
        self.job_end_times = [0] * self.num_jobs             # 每个作业目前排到的结束时间
        self.done = False
        self.schedule = []
        self.total_processing_time = np.sum(self.processing_times)
        self.completed_ops = 0  # 累计完成工序数
        return self._get_state()

    def _get_state(self):
        state = []

        # 1. 每个作业的下一道工序信息 (目标机器, 处理时间)
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])  # 作业已完成

        # 2. 机器当前时间（每台机器调度到的最晚结束时刻）
        state.extend(self.time_table)

        # 3. 作业当前已排产到的结束时间
        state.extend(self.job_end_times)

        # 4. 全局进度比例
        completed_ops = sum(self.current_step)
        total_ops = self.num_jobs * self.num_machines
        progress = completed_ops / total_ops if total_ops > 0 else 0
        state.append(progress)

        # 5. 机器相对负载（归一化占用，找瓶颈）
        if len(self.time_table) > 0:
            max_machine_time = max(self.time_table)
            if max_machine_time > 0:
                machine_relative_load = [t / max_machine_time for t in self.time_table]
                state.extend(machine_relative_load)
            else:
                state.extend([0] * len(self.time_table))
        else:
            state.extend([0] * len(self.all_machines))

        # 6. 每个作业的剩余总工作量（后续所有还没做的工序所需时间之和）
        for job in range(self.num_jobs):
            remaining_ops = self.num_machines - self.current_step[job]
            if remaining_ops > 0:
                remaining_time = sum(self.processing_times[job, self.current_step[job]:])
                state.append(remaining_time)
            else:
                state.append(0)

        # 7. 当前可行调度动作(除等待外)的处理时间统计特征
        valid_actions = self.get_valid_actions()
        valid_proc_times = []
        for a in valid_actions:
            if a == 0:
                continue
            job_idx = a - 1  # action=k -> job=k-1
            op = self.current_step[job_idx]
            if op < self.num_machines:
                proc_time = self.processing_times[job_idx, op]
                valid_proc_times.append(proc_time)

        if valid_proc_times:
            state.append(min(valid_proc_times))   # 最短处理时间
            state.append(max(valid_proc_times))   # 最长处理时间
            state.append(np.mean(valid_proc_times))  # 平均处理时间
        else:
            state.extend([0, 0, 0])

        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        """
        返回当前可执行的动作列表:
        - 0 表示等待 (允许始终存在)
        - k>0 表示选择 job = k-1, 如果这个作业还有未完成的工序
        """
        valid_actions = [0]  # 等待动作永远允许被选
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                valid_actions.append(job + 1)
        return valid_actions

    def step(self, action):
        """
        action = 0        -> 等待（推进时间，不新增工序）
        action = k > 0    -> 选择 job = k-1 调度它的下一道工序
        """

        # 如果已经全部完成
        if self.done:
            return self._get_state(), 0, True, {
                "schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.done else None,
            }

        valid_actions = self.get_valid_actions()

        # 非法动作：比如调度已完成作业，或给了不在valid_actions的索引
        if action not in valid_actions:
            return self._get_state(), -50, self.done, {}

        # 分支1：action == 0 等待
        if action == 0:
            # 等待含义：时间往前推进一个离散步长 dt，没有新工序插入
            dt = 1.0

            # 检查是否除了等待以外还有其他可执行动作
            has_other_actions = (len(valid_actions) > 1)

            # 时间流逝：所有机器时间线和作业时间线整体加 dt
            self.time_table = [t + dt for t in self.time_table]
            self.job_end_times = [t + dt for t in self.job_end_times]

            # 等待不会直接推进工序步骤，所以 self.current_step 不变
            # 环境完成判定（一般来说不会在等待中直接完成）
            self.done = all(step >= self.num_machines for step in self.current_step)

            # 奖励规则：
            # 1. 如果只能等（没有其他合法调度动作），不给惩罚
            # 2. 如果有活可干但你选择等 -> 轻罚
            # 3. 如果有活可干且你等的这段时间 dt 已经足够去做一个最短工序 -> 额外重罚

            if not has_other_actions:
                # 系统客观上卡住，只能等
                wait_penalty = 0.0
            else:
                # 有工可以做但你选择等
                # 先找当前可执行工序中处理时间最短的那个
                candidate_proc_times = []
                for a in valid_actions:
                    if a == 0:
                        continue
                    job_idx = a - 1
                    op2 = self.current_step[job_idx]
                    if op2 < self.num_machines:
                        candidate_proc_times.append(self.processing_times[job_idx, op2])

                if candidate_proc_times:
                    shortest_time = min(candidate_proc_times)
                else:
                    shortest_time = None  # 理论上不会None，因为has_other_actions为True

                # 轻罚基线
                wait_penalty = -1.0

                # 判断这次等待是不是“过度浪费”
                if shortest_time is not None and dt >= shortest_time:
                    # 你本可以立刻开一个很短的工序，而你却空等了这么久
                    wait_penalty -= 5.0  # 额外重罚

            # 最终奖励里仍可保留episode完成时的makespan激励
            final_reward = 0.0
            if self.done:
                makespan = max(self.job_end_times)
                theoretical_lower_bound = self.total_processing_time / len(self.all_machines)
                efficiency = theoretical_lower_bound / makespan if makespan > 0 else 0
                final_reward = efficiency * 100

            reward = wait_penalty + final_reward

            info = {
                "schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.done else None,
                "machine_utilization": [
                    t / max(self.time_table) if max(self.time_table) > 0 else 0
                    for t in self.time_table
                ],
                "reward_components": {
                    "wait_penalty": wait_penalty,
                    "final_reward": final_reward,
                }
            }

            return self._get_state(), reward, self.done, info

        # 分支2：调度具体作业 job = action-1
        job = action - 1
        op = self.current_step[job]

        # 再次安全校验：这个作业是否还有工序
        if op >= self.num_machines:
            return self._get_state(), -50, self.done, {}

        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]

        # 计算这道工序的开工/完工时间
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time

        # 记录之前状态，用来给奖励（利用率、等待等）
        prev_machine_idle_time = self.time_table[machine_idx] - max(self.time_table)
        prev_job_wait_time = self.time_table[machine_idx] - self.job_end_times[job]

        # 更新环境：这道工序被正式排入甘特图
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.current_step[job] += 1

        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.done = all(step >= self.num_machines for step in self.current_step)

        # ----------------- 奖励函数 -----------------
        # 目标：缩短makespan (最小化最后完工时间)
        # 下面基本沿用你的原本设计，只保持语义一致

        # 1. 基础时间惩罚：鼓励选择处理时间短的任务
        time_penalty = -proc_time * 0.05

        # 2. 进度差分奖励：工序推进越多越好
        total_ops = self.num_jobs * self.num_machines
        progress_prev = self.completed_ops / total_ops if total_ops > 0 else 0
        self.completed_ops += 1
        progress_now = self.completed_ops / total_ops if total_ops > 0 else 0
        progress_reward = (progress_now - progress_prev) * 30

        # 3. 机器负载平衡奖励：鼓励不要形成过强瓶颈
        machine_loads = [t for t in self.time_table]
        if len(machine_loads) > 1:
            load_balance_reward = -np.std(machine_loads) * 0.05
        else:
            load_balance_reward = 0

        # 4. 作业完成奖励：完成一个job就给奖励
        completion_reward = 15 if self.current_step[job] == self.num_machines else 0

        # 5. 机会成本惩罚：如果没选最快的可执行工序，轻微惩罚
        opportunity_cost_penalty = 0
        valid_actions_now = self.get_valid_actions()
        if len(valid_actions_now) > 2:  # 至少有等待+两个以上的job可选
            current_proc_time = proc_time
            fastest_time = float('inf')
            for a in valid_actions_now:
                if a == 0:
                    continue
                j2 = a - 1
                op2 = self.current_step[j2]
                if op2 < self.num_machines:
                    pt2 = self.processing_times[j2, op2]
                    if pt2 < fastest_time:
                        fastest_time = pt2
            if fastest_time < float('inf') and current_proc_time > fastest_time:
                opportunity_cost_penalty = -0.1 * (current_proc_time - fastest_time)

        # 6. 机器利用奖励：鼓励减少空转
        machine_utilization_reward = 0
        if prev_machine_idle_time > 0 and start_time == self.time_table[machine_idx]:
            # 立即拿闲置机器开工 -> 鼓励
            machine_utilization_reward = 0.5
        elif prev_job_wait_time > 0:
            # 作业自己等了很久才上机 -> 轻微惩罚
            machine_utilization_reward = -0.1

        # 7. 关键路径奖励：优先推进剩余工作量最大的作业
        critical_path_reward = 0
        if not self.done:
            remaining_work = []
            for j in range(self.num_jobs):
                remain_ops = self.num_machines - self.current_step[j]
                if remain_ops > 0:
                    remain_time = sum(self.processing_times[j, self.current_step[j]:])
                    remaining_work.append((j, remain_time))
            if remaining_work:
                max_remaining_job, max_remaining_time = max(remaining_work, key=lambda x: x[1])
                if job == max_remaining_job:
                    critical_path_reward = 0.3

        # 8. 瓶颈机器奖励：避免继续压死最忙的机器
        bottleneck_reward = 0
        if len(self.time_table) > 0:
            max_load_machine_idx = np.argmax(self.time_table)
            max_load = self.time_table[max_load_machine_idx]
            avg_load = np.mean(self.time_table)
            if max_load > avg_load * 1.2 and machine_idx == max_load_machine_idx:
                bottleneck_reward = -0.2
            elif machine_idx != max_load_machine_idx:
                bottleneck_reward = 0.2

        # 9. 最终效率奖励（episode结束时）
        final_reward = 0
        if self.done:
            makespan = max(self.job_end_times)
            theoretical_lower_bound = self.total_processing_time / len(self.all_machines)
            efficiency = theoretical_lower_bound / makespan if makespan > 0 else 0
            final_reward = efficiency * 100

        reward = (
            time_penalty
            + load_balance_reward
            + progress_reward
            + completion_reward
            + final_reward
            + opportunity_cost_penalty
            + machine_utilization_reward
            + critical_path_reward
            + bottleneck_reward
        )

        info = {
            "schedule": self.schedule.copy(),
            "makespan": max(self.job_end_times) if self.done else None,
            "machine_utilization": [
                t / max(self.time_table) if max(self.time_table) > 0 else 0
                for t in self.time_table
            ],
            "reward_components": {
                "time_penalty": time_penalty,
                "progress_reward": progress_reward,
                "opportunity_cost": opportunity_cost_penalty,
                "machine_utilization": machine_utilization_reward,
                "critical_path": critical_path_reward,
                "bottleneck": bottleneck_reward,
                "final_reward": final_reward
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
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(')+1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                machine = int(parts[0].strip())
                time = float(parts[1].strip())
                return machine, time
    except Exception as e:
        pass
    return 0, 0.0


def load_single_csv(file_path):
    """从CSV文件加载数据并解析为机器分配矩阵和处理时间矩阵"""
    try:
        # 尝试不同的分隔符
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = [f.readline() for _ in range(5)]
                    f.seek(0)

                    reader = csv.reader(f, delimiter=delimiter)
                    data = list(reader)

                    if len(data) > 1 and len(data[0]) > 1:
                        print(f"使用分隔符 '{delimiter}' 读取文件: {os.path.basename(file_path)}")
                        break
            except:
                continue

        if 'data' not in locals():
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                data = list(reader)
            print(f"使用默认分隔符读取文件: {os.path.basename(file_path)}")

        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0  # 减去作业名列

        machine_data = []
        time_data = []

        for row in data[1:]:
            if not row or len(row) == 0:
                continue

            job_machines = []
            job_times = []

            for cell in row[1:1+num_processes]:
                machine, time = parse_tuple(cell)
                job_machines.append(machine)
                job_times.append(time)

            machine_data.append(job_machines)
            time_data.append(job_times)

        machine_array = np.array(machine_data, dtype=int)
        time_array = np.array(time_data, dtype=float)

        if machine_array.size == 0 or time_array.size == 0:
            print(f"文件 {os.path.basename(file_path)} 数据为空")
            return None, None
        if machine_array.shape != time_array.shape:
            print(f"文件 {os.path.basename(file_path)} 机器分配和处理时间形状不匹配")
            return None, None

        print(f"成功加载文件: {os.path.basename(file_path)}, 形状: {machine_array.shape}")
        return machine_array, time_array

    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None, None


def load_folder_data(folder_path):
    """从文件夹加载所有CSV文件数据"""
    datasets = []
    valid_files = 0

    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return datasets

    print(f"扫描文件夹: {folder_path}")
    print(f"找到文件: {len(os.listdir(folder_path))}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv') and 'gene' in filename.lower():
            file_path = os.path.join(folder_path, filename)

            machine_assignments, processing_times = load_single_csv(file_path)

            if machine_assignments is not None and processing_times is not None:
                datasets.append((machine_assignments, processing_times))
                valid_files += 1

    print(f"总共加载了 {valid_files} 个有效数据集")
    return datasets


def train_target(training_datasets):
    if not training_datasets:
        print("没有可用的训练数据!")
        return None, None

    # 初始化环境，推导状态空间与动作空间
    sample_env = JSSPEnv(*training_datasets[0])
    state_dim = len(sample_env.reset())

    # 新动作空间: 等待(0) + 每个job
    action_dim = sample_env.num_jobs + 1

    print(f"状态维度: {state_dim}, 动作空间: {action_dim}")
    print(f"可用数据集数量: {len(training_datasets)}")

    # 创建主网络和目标网络
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    best_makespan = float('inf')
    best_schedule = None
    makespans = []

    total_steps = 0
    complete_datasets_count = 0
    training_started = False

    print("第一阶段：收集经验")

    selected_datasets = random.sample(training_datasets, MIN_COMPLETE_EPISODES)
    print(f"已选择 {len(selected_datasets)} 个不同数据集用于第一阶段")

    pbar_stage1 = tqdm(total=MIN_COMPLETE_EPISODES, desc="收集经验进度")

    # 第一阶段：纯探索，跑完完整调度，把经验塞进 memory
    for i, (ma, pt) in enumerate(selected_datasets):
        env = JSSPEnv(ma, pt)
        state = env.reset()

        while not env.done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # 纯探索：随机合法动作（包括等待0）
            action = random.choice(valid_actions)

            next_state, reward, done, info = env.step(action)

            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_steps += 1

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

    # 第二阶段：带训练
    progress_bar = tqdm(range(EPISODES), desc="训练进度")

    stage2_used_datasets = set()

    for ep in progress_bar:
        ma, pt = random.choice(training_datasets)

        dataset_hash = hash((ma.tobytes(), pt.tobytes()))
        stage2_used_datasets.add(dataset_hash)

        env = JSSPEnv(ma, pt)
        state = env.reset()
        total_reward = 0
        steps = 0

        while not env.done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # ε-贪婪
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)  # shape [1, action_dim]

                    # mask掉非法动作（包括那些不在valid_actions中的动作索引）
                    mask = torch.ones_like(q_values, dtype=torch.bool)
                    for i_act in range(action_dim):
                        if i_act not in valid_actions:
                            mask[0, i_act] = False

                    q_values_masked = q_values.clone()
                    q_values_masked[~mask] = -float('inf')
                    action = q_values_masked.argmax().item()

            next_state, reward, done, info = env.step(action)

            memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1

            # 经验回放训练
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                next_states_tensor = torch.FloatTensor(np.array(next_states)).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)

                # 目标Q值，用目标网络
                with torch.no_grad():
                    next_q_all = target_model(next_states_tensor)  # [B, action_dim]

                    # 需要对 next_state 也做合法动作mask
                    # 对于batch里的每个 next_state 我们需要合法动作集 => 这比较麻烦
                    # 简化方案（与原版一致的近似）：直接用 max(所有动作)
                    # 如果你想要mask版target，你就必须在memory里也存valid_actions_next
                    next_q = next_q_all.max(1)[0].unsqueeze(1)

                    target_q = rewards_tensor + GAMMA * next_q * (~dones_tensor)

                # 当前Q值，用主网络
                current_q_all = model(states_tensor)           # [B, action_dim]
                current_q = current_q_all.gather(1, actions_tensor)  # [B,1]

                loss = criterion(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # 同步目标网络
            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                target_model.load_state_dict(model.state_dict())

        # 回合结束记录
        if env.done:
            complete_datasets_count += 1
            makespan = info["makespan"]
            makespans.append(makespan)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = info["schedule"].copy()

        # epsilon 衰减
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

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
