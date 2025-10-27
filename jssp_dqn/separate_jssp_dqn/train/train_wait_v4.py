import os
import csv
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt


#                     超参数定义

EPISODES = 2000
GAMMA = 0.95
LR = 0.0005
EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
BATCH_SIZE = 128
MEMORY_SIZE = 100000
UPDATE_TARGET_FREQUENCY = 30
MIN_COMPLETE_EPISODES = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")


#                    神经网络结构
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
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

#                     JSSP 环境定义

class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape
        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {m: i for i, m in enumerate(self.all_machines)}
        self.index_to_machine = {i: m for i, m in enumerate(self.all_machines)}
        self.reset()

    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0] * len(self.all_machines)
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        self.total_processing_time = np.sum(self.processing_times)
        self.completed_ops = 0
        return self._get_state()

    def _get_state(self):
        state = []
        # 每个作业的下一工序信息
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])

        # 机器时间表
        state.extend(self.time_table)
        # 作业完成时间
        state.extend(self.job_end_times)
        # 进度
        total_ops = self.num_jobs * self.num_machines
        progress = sum(self.current_step) / total_ops if total_ops > 0 else 0
        state.append(progress)
        # 机器负载比例
        if len(self.time_table) > 0:
            max_t = max(self.time_table)
            state.extend([t / max_t if max_t > 0 else 0 for t in self.time_table])
        else:
            state.extend([0] * len(self.all_machines))
        # 剩余工时
        for job in range(self.num_jobs):
            remain = self.num_machines - self.current_step[job]
            state.append(sum(self.processing_times[job, self.current_step[job]:]) if remain > 0 else 0)
        # 有效动作统计
        valid_actions = self.get_valid_actions()
        valid_proc_times = []
        for a in valid_actions:
            if a == 0:
                continue
            job = a - 1
            op = self.current_step[job]
            if op < self.num_machines:
                valid_proc_times.append(self.processing_times[job, op])
        if valid_proc_times:
            state += [min(valid_proc_times), max(valid_proc_times), np.mean(valid_proc_times)]
        else:
            state += [0, 0, 0]
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        valid_actions = [0]  # 等待动作
        for job in range(self.num_jobs):
            if self.current_step[job] < self.num_machines:
                valid_actions.append(job + 1)
        return valid_actions

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {"makespan": max(self.job_end_times)}

        valid_actions = self.get_valid_actions()
        if action not in valid_actions:
            return self._get_state(), -50, self.done, {}

        # ---------------- 等待逻辑 ----------------
        if action == 0:
            # 判断是否真的无法执行
            has_executable = False
            for job in range(self.num_jobs):
                op = self.current_step[job]
                if op < self.num_machines:
                    machine = self.machine_assignments[job, op]
                    m_idx = self.machine_to_index[machine]
                    if self.job_end_times[job] <= self.time_table[m_idx]:
                        has_executable = True
                        break

            # 若有工序可执行却等待 => 轻惩罚
            reward = -10.0 if has_executable else -1.0

            # 时间推进到下一个可执行工序
            min_next_start = float("inf")
            for job in range(self.num_jobs):
                op = self.current_step[job]
                if op < self.num_machines:
                    machine = self.machine_assignments[job, op]
                    m_idx = self.machine_to_index[machine]
                    ready_time = max(self.job_end_times[job], self.time_table[m_idx])
                    min_next_start = min(min_next_start, ready_time)
            dt = min_next_start - min(self.time_table) if min_next_start < float("inf") else 1.0
            dt = max(0.1, dt)
            self.time_table = [t + dt for t in self.time_table]
            self.job_end_times = [t + dt for t in self.job_end_times]
            self.done = all(s >= self.num_machines for s in self.current_step)
            return self._get_state(), reward, self.done, {"makespan": max(self.job_end_times)}
        # ---------------- 等待逻辑结束 ----------------

        # 执行动作（调度作业）
        job = action - 1
        op = self.current_step[job]
        if op >= self.num_machines:
            return self._get_state(), -50, self.done, {}

        machine_id = self.machine_assignments[job, op]
        midx = self.machine_to_index[machine_id]
        pt = self.processing_times[job, op]
        start = max(self.job_end_times[job], self.time_table[midx])
        end = start + pt
        self.job_end_times[job] = end
        self.time_table[midx] = end
        self.current_step[job] += 1
        self.schedule.append((job, op, machine_id, start, end))
        self.done = all(s >= self.num_machines for s in self.current_step)
        # 奖励函数
        reward = -pt * 0.05
        return self._get_state(), reward, self.done, {"makespan": max(self.job_end_times)}


#                     数据加载函数

def parse_tuple(cell):
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(') + 1:cell.find(')')]
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
        header = data[0]
        num_processes = len(header) - 1
        machine_data, time_data = [], []
        for row in data[1:]:
            if not row:
                continue
            jm, jt = [], []
            for cell in row[1:1 + num_processes]:
                m, t = parse_tuple(cell)
                jm.append(m)
                jt.append(t)
            machine_data.append(jm)
            time_data.append(jt)
        machine_array = np.array(machine_data, dtype=int)
        time_array = np.array(time_data, dtype=float)
        return machine_array, time_array
    except Exception as e:
        print(f"加载文件出错: {e}")
        return None, None


def load_folder_data(folder_path):
    datasets = []
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".csv") and "gene" in fn.lower():
            ma, pt = load_single_csv(os.path.join(folder_path, fn))
            if ma is not None:
                datasets.append((ma, pt))
    print(f"总共加载了 {len(datasets)} 个数据集")
    return datasets


#                     训练主函数

def train_target(training_datasets):
    if not training_datasets:
        print("没有数据")
        return
    env0 = JSSPEnv(*training_datasets[0])
    state_dim = len(env0.reset())
    action_dim = env0.num_jobs + 1

    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = 1.0
    best_makespan = float('inf')
    makespans = []

    # 阶段1: 收集经验
    selected = random.sample(training_datasets, MIN_COMPLETE_EPISODES)
    for ma, pt in selected:
        env = JSSPEnv(ma, pt)
        s = env.reset()
        while not env.done:
            a = random.choice(env.get_valid_actions())
            ns, r, d, _ = env.step(a)
            memory.append((s, a, r, ns, d))
            s = ns

    # 阶段2: 训练
    progress = tqdm(range(EPISODES))
    for ep in progress:
        ma, pt = random.choice(training_datasets)
        env = JSSPEnv(ma, pt)
        s = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid = env.get_valid_actions()
            if random.random() < epsilon:
                a = random.choice(valid)
            else:
                st = torch.FloatTensor(s).unsqueeze(0).to(device)
                with torch.no_grad():
                    q = model(st)
                    mask = torch.ones_like(q, dtype=torch.bool)
                    for i in range(action_dim):
                        if i not in valid:
                            mask[0, i] = False
                    q[~mask] = -float("inf")
                    a = q.argmax().item()

            ns, r, done, info = env.step(a)
            memory.append((s, a, r, ns, done))
            s = ns
            total_reward += r

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_q = target_model(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + GAMMA * next_q * (~dones)
                current_q = model(states).gather(1, actions)
                loss = criterion(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        if ep % UPDATE_TARGET_FREQUENCY == 0:
            target_model.load_state_dict(model.state_dict())

        makespan = info["makespan"]
        makespans.append(makespan)
        if makespan < best_makespan:
            best_makespan = makespan

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        avg_m = np.mean(makespans[-50:])
        progress.set_postfix({"eps": f"{epsilon:.3f}", "best": f"{best_makespan:.1f}", "avg": f"{avg_m:.1f}"})

    torch.save({
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, "model/wait_v4.pth")
    print(f"训练完成,最佳makespan={best_makespan:.1f}")

if __name__ == "__main__":
    folder = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    data = load_folder_data(folder)
    train_target(data)
