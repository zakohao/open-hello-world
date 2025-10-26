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

# CUDA检查
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")

# 超参数
EPISODES = 2000
GAMMA = 0.95
LR = 0.0005
EPSILON_DECAY = 0.998
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQUENCY = 30
MIN_COMPLETE_EPISODES = 10

# 环境定义
class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}

        # 永久有效WAIT动作
        self.WAIT_ACTION = self.num_jobs
        self.action_dim = self.num_jobs + 1
        self.reset()

    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0] * len(self.all_machines)
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        self.total_processing_time = np.sum(self.processing_times)
        self.completed_ops = 0
        self.last_makespan = 0
        self.current_time = 0
        return self._get_state()

    def _get_state(self):
        state = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])
        state.extend(self.time_table)
        state.extend(self.job_end_times)
        total_ops = self.num_jobs * self.num_machines
        progress = sum(self.current_step) / total_ops if total_ops > 0 else 0
        state.append(progress)
        state.append(self.current_time)
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        valid_actions = [job for job in range(self.num_jobs)
                         if self.current_step[job] < self.num_machines]
        valid_actions.append(self.WAIT_ACTION)
        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done, {}

        # WAIT动作
        if action == self.WAIT_ACTION:
            next_event_time = min(self.time_table) if any(self.time_table) else self.current_time + 1
            time_advance = max(1e-6, next_event_time - self.current_time)
            self.current_time += time_advance
            wait_penalty = -time_advance * 0.2
            reward = wait_penalty
            info = {"action": "WAIT", "schedule": self.schedule.copy()}
            return self._get_state(), reward, self.done, info

        # 非WAIT动作
        if action >= self.num_jobs or self.current_step[action] >= self.num_machines:
            return self._get_state(), -50, self.done, {}

        job = action
        op = self.current_step[job]
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]

        start_time = max(self.job_end_times[job], self.time_table[machine_idx], self.current_time)
        end_time = start_time + proc_time
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.current_step[job] += 1
        self.current_time = min(self.time_table)

        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.done = all(step >= self.num_machines for step in self.current_step)

        # 奖励函数
        time_penalty = -proc_time * 0.05
        total_ops = self.num_jobs * self.num_machines
        progress_prev = self.completed_ops / total_ops if total_ops > 0 else 0
        self.completed_ops += 1
        progress_now = self.completed_ops / total_ops
        progress_reward = (progress_now - progress_prev) * 30
        machine_loads = [t for t in self.time_table]
        load_balance_reward = -np.std(machine_loads) * 0.05 if len(machine_loads) > 1 else 0
        completion_reward = 15 if self.current_step[job] == self.num_machines else 0
        prev_makespan = self.last_makespan
        self.last_makespan = max(self.job_end_times)
        delta_makespan = prev_makespan - self.last_makespan
        defer_gain = delta_makespan * 10 if delta_makespan > 0 else 0

        final_reward = 0
        if self.done:
            makespan = max(self.job_end_times)
            theoretical_lower_bound = self.total_processing_time / len(self.all_machines)
            efficiency = theoretical_lower_bound / makespan if makespan > 0 else 0
            final_reward = efficiency * 100

        reward = (time_penalty + progress_reward + load_balance_reward +
                  completion_reward + defer_gain + final_reward)

        info = {"action": job, "schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.done else None}
        return self._get_state(), reward, self.done, info


# DQN模型
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



# CSV加载函数
def parse_tuple(cell):
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(')+1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                return int(parts[0].strip()), float(parts[1].strip())
    except:
        pass
    return 0, 0.0


def load_single_csv(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            data = list(reader)
        machine_data, time_data = [], []
        for row in data[1:]:
            if not row:
                continue
            job_machines, job_times = [], []
            for cell in row[1:]:
                machine, time = parse_tuple(cell)
                job_machines.append(machine)
                job_times.append(time)
            machine_data.append(job_machines)
            time_data.append(job_times)
        return np.array(machine_data, int), np.array(time_data, float)
    except Exception as e:
        print(f"加载失败: {file_path}, {e}")
        return None, None


def load_folder_data(folder_path):
    datasets = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv') and 'gene' in filename.lower():
            file_path = os.path.join(folder_path, filename)
            ma, pt = load_single_csv(file_path)
            if ma is not None and pt is not None:
                datasets.append((ma, pt))
    print(f"总共加载 {len(datasets)} 个数据集")
    return datasets


# 训练函数
def train_reward_wait(training_datasets):
    if not training_datasets:
        print("无可用训练数据")
        return

    sample_env = JSSPEnv(*training_datasets[0])
    state_dim = len(sample_env.reset())
    action_dim = sample_env.action_dim
    print(f"状态维度={state_dim}, 动作维度={action_dim}")

    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    best_makespan = float('inf')
    makespans = []

    progress_bar = tqdm(range(EPISODES))
    for ep in progress_bar:
        ma, pt = random.choice(training_datasets)
        env = JSSPEnv(ma, pt)
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    mask = torch.full_like(q_values, -float('inf'))
                    mask[0, valid_actions] = 0
                    q_values = q_values + mask
                    action = q_values.argmax().item()

            next_state, reward, done, info = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states_tensor = torch.FloatTensor(states).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_q = target_model(next_states_tensor).max(1)[0].unsqueeze(1)
                    target_q = rewards_tensor + GAMMA * next_q * (~dones_tensor)

                current_q = model(states_tensor).gather(1, actions_tensor)
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if done and info["makespan"] is not None:
            makespans.append(info["makespan"])
            if info["makespan"] < best_makespan:
                best_makespan = info["makespan"]

        if ep % UPDATE_TARGET_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        progress_bar.set_postfix({"epsilon": f"{epsilon:.3f}",
                                  "best_makespan": f"{best_makespan:.1f}"})

    torch.save(model.state_dict(), "wait_0.pth")
    print(f"最佳makespan={best_makespan}")



if __name__ == "__main__":
    print("开始训练JSSP模型")
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    
    # 加载训练数据
    datasets = load_folder_data(folder_path)

    # 训练模型
    if datasets:
        train_reward_wait(datasets)
    else:
        print("未找到有效训练数据。")
