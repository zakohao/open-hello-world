import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# ====== 环境类，带限制条件和甘特图数据记录 ======
class JSSPEnv:
    def __init__(self, processing_times, machine_assignments):
        self.processing_times = processing_times
        self.machine_assignments = machine_assignments
        self.num_jobs = len(processing_times)
        self.num_machines = max(max(m) for m in machine_assignments) + 1
        self.reset()

    def reset(self):
        self.current_time = 0
        self.job_step = [0 for _ in range(self.num_jobs)]             # 每个job当前执行的工序索引
        self.done_ops = [[False]*len(job) for job in self.processing_times]  # 工序完成标记
        self.machine_available_time = [0 for _ in range(self.num_machines)] # 机器空闲时间
        self.job_available_time = [0 for _ in range(self.num_jobs)]          # 任务可执行时间（上一工序完成时间）
        self.schedule = []          # 甘特图数据： [{'job': , 'step': , 'machine': , 'start': , 'end': }, ...]
        return self._get_state()

    def _get_state(self):
        state = []
        for j in range(self.num_jobs):
            step = self.job_step[j]
            if step < len(self.processing_times[j]):
                state.append(self.job_available_time[j])
                state.append(self.machine_available_time[self.machine_assignments[j][step]])
                state.append(self.processing_times[j][step])
            else:
                state.extend([0, 0, 0])
        return np.array(state, dtype=np.float32)

    def _get_valid_actions(self):
        valid_actions = []
        for j in range(self.num_jobs):
            step = self.job_step[j]
            if step >= len(self.processing_times[j]):
                continue  # 工序全部完成
            if self.done_ops[j][step]:
                continue  # 当前工序已完成
            # 这里可加入机器忙碌判断，训练时动作选择时过滤非法动作即可
            valid_actions.append(j)
        return valid_actions

    def step(self, action):
        j = action
        step = self.job_step[j]

        # 违反顺序或重复操作判罚
        if step >= len(self.processing_times[j]) or self.done_ops[j][step]:
            return self._get_state(), -10, False, {}

        machine = self.machine_assignments[j][step]
        proc_time = self.processing_times[j][step]

        # 当前作业与机器的最早可用时间，保证加工不重叠且顺序执行
        start_time = max(self.job_available_time[j], self.machine_available_time[machine])
        end_time = start_time + proc_time

        # 更新状态
        self.machine_available_time[machine] = end_time
        self.job_available_time[j] = end_time
        self.done_ops[j][step] = True
        self.job_step[j] += 1

        # 记录甘特图信息
        self.schedule.append({
            'job': j,
            'step': step,
            'machine': machine,
            'start': start_time,
            'end': end_time
        })

        done = all(self.job_step[j] >= len(self.processing_times[j]) for j in range(self.num_jobs))
        reward = -max(self.machine_available_time) if done else -1

        return self._get_state(), reward, done, {}

    def get_makespan(self):
        return max(self.machine_available_time)

    def render_gantt(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.get_cmap('tab20', self.num_jobs)

        for task in self.schedule:
            ax.barh(
                y=f"Machine {task['machine']}",
                width=task['end'] - task['start'],
                left=task['start'],
                height=0.4,
                color=colors(task['job']),
                edgecolor='black'
            )
            ax.text(
                x=task['start'] + 0.1,
                y=f"Machine {task['machine']}",
                s=f"J{task['job']}O{task['step']}",
                va='center',
                ha='left',
                fontsize=8,
                color='black'
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        ax.set_title("JSSP Final Schedule Gantt Chart")
        plt.tight_layout()
        plt.show()

# ====== DQN网络 ======
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ====== 训练函数 ======
def train_dqn(env, episodes=300):
    input_dim = len(env._get_state())
    output_dim = env.num_jobs
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=10000)
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.1
    target_update_freq = 10

    rewards_all = []
    makespans_all = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            valid_actions = env._get_valid_actions()

            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    sorted_actions = q_values.argsort(descending=True)
                    # 选择合法动作中q值最高的
                    for a in sorted_actions:
                        if a.item() in valid_actions:
                            action = a.item()
                            break

            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.BoolTensor(dones).unsqueeze(1)

                q_values = policy_net(states).gather(1, actions)
                q_next = target_net(next_states).max(1)[0].detach().unsqueeze(1)
                q_target = rewards + gamma * q_next * (~dones)

                loss = nn.MSELoss()(q_values, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_all.append(total_reward)
        makespan = env.get_makespan()
        makespans_all.append(makespan)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Total Reward={total_reward:.2f}, Makespan={makespan}")

    torch.save(policy_net.state_dict(), 'jssp_dqn_model.pt')
    print("\n模型保存至 jssp_dqn_model.pt")

    # 训练过程makespan曲线
    plt.figure(figsize=(10,4))
    plt.plot(makespans_all)
    plt.xlabel("Episode")
    plt.ylabel("Makespan")
    plt.title("Makespan over Episodes")
    plt.grid(True)
    plt.show()

    # 最终甘特图
    env.render_gantt()

    return policy_net, rewards_all, makespans_all

# ====== 评估函数 ======
def evaluate(env, model_path='jssp_dqn_model.pt'):
    input_dim = len(env._get_state())
    output_dim = env.num_jobs
    policy_net = DQN(input_dim, output_dim)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    state = env.reset()
    done = False

    while not done:
        valid_actions = env._get_valid_actions()
        with torch.no_grad():
            q_values = policy_net(torch.FloatTensor(state))
            sorted_actions = q_values.argsort(descending=True)
            for a in sorted_actions:
                if a.item() in valid_actions:
                    action = a.item()
                    break

        state, _, done, _ = env.step(action)

    print("评估完成，最终Makespan:", env.get_makespan())
    env.render_gantt()

# ====== 主程序示例 ======
if __name__ == "__main__":
    # 示例任务：每个 job 对应各工序的加工时间和机器编号
    processing_times = [
        [3, 2, 2],
        [2, 1, 4],
        [4, 3, 2]
    ]
    machine_assignments = [
        [0, 1, 2],
        [1, 2, 0],
        [2, 0, 1]
    ]

    env = JSSPEnv(processing_times, machine_assignments)
    train_dqn(env, episodes=300)
    evaluate(env)
