import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hyperparameters
EPISODES = 300
GAMMA = 0.95
LR = 0.001
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000

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
        
        reward = -end_time if self.done else 0
        return self._get_state(), reward, self.done, self.schedule

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def plot_gantt_chart(schedule, num_jobs, all_machines, title="Gantt Chart"):
    fig, ax = plt.subplots(figsize=(12, 8))
    color_map = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    
    # 创建机器ID到y轴位置的映射
    machine_to_ypos = {machine: idx for idx, machine in enumerate(sorted(all_machines))}
    
    for job_id in range(num_jobs):
        job_tasks = [t for t in schedule if t[0] == job_id]
        for task in job_tasks:
            job_id, op_index, machine_id, start, duration = task
            label = f"J{job_id+1}-P{op_index+1}"
            ypos = machine_to_ypos[machine_id]
            ax.barh(ypos, duration, left=start, height=0.6,
                    color=color_map(job_id), edgecolor='black')
            ax.text(start + duration / 2, ypos, label,
                    va='center', ha='center', color='black', fontsize=8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(title)
    
    # 设置y轴刻度和标签
    sorted_machines = sorted(all_machines)
    ax.set_yticks(range(len(sorted_machines)))
    ax.set_yticklabels([f"M{m}" for m in sorted_machines])
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def train(machine_assignments, processing_times):
    env = JSSPEnv(machine_assignments, processing_times)
    state_dim = len(env.reset())
    action_dim = env.num_jobs  # 动作空间大小等于作业数
    
    model = DQN(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0
    
    best_makespan = float('inf')
    best_schedule = None
    makespans = []
    
    for ep in range(EPISODES):
        state = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = q_values.argmax().item()
            
            next_state, reward, done, schedule = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state)
            memory.append((state, action, reward, next_state_tensor, done))
            state = next_state_tensor
            total_reward += reward
            
            if done:
                makespan = max(env.job_end_times)
                makespans.append(makespan)
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_schedule = schedule.copy()
            
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.stack(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.stack(next_states)
                dones = torch.BoolTensor(dones).unsqueeze(1)
                
                current_q = model(states).gather(1, actions)
                next_q = model(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * next_q * (~dones)
                
                loss = criterion(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        print(f"Episode {ep+1}: Reward={total_reward}, Makespan={makespan}")
    
    # Plot final result
    print(f"Best makespan: {best_makespan}")
    plot_gantt_chart(best_schedule, env.num_jobs, env.all_machines)
    plt.figure()
    plt.plot(makespans)
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 示例输入矩阵 - 可以是任意形状
    machine_assignments = np.array([
        [1, 2, 3, 4, 5, 6],
        [2, 3, 6, 1, 5, 4],
        [3, 6, 1, 2, 4, 5],
        [6, 2, 1, 4, 5, 3],
        [1, 3, 5, 6, 2, 4],
        [2, 6, 4, 1, 5, 3],
    ])
    
    processing_times = np.array([
        [1, 3, 6, 7, 3, 6],
        [8, 5, 10, 10, 10, 4],
        [5, 4, 8, 9, 1, 7],
        [5, 5, 5, 3, 8, 9],
        [9, 3, 5, 4, 3, 1],
        [3, 3, 9, 10, 4, 1],
    ])
    
    train(machine_assignments, processing_times)