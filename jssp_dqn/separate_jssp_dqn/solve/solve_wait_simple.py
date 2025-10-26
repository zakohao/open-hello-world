import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os
import time
from collections import defaultdict


# DQN 模型定义（与训练时一致）
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)



# JSSP 环境（与训练时完全一致）
class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}

        # 永久有效 WAIT 动作
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

        # WAIT 动作逻辑
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
        load_balance_reward = -np.std(self.time_table) * 0.05 if len(self.time_table) > 1 else 0
        completion_reward = 15 if self.current_step[job] == self.num_machines else 0
        prev_makespan = self.last_makespan
        self.last_makespan = max(self.job_end_times)
        delta_makespan = prev_makespan - self.last_makespan
        defer_gain = delta_makespan * 10 if delta_makespan > 0 else 0
        final_reward = 0
        if self.done:
            makespan = max(self.job_end_times)
            lower_bound = self.total_processing_time / len(self.all_machines)
            efficiency = lower_bound / makespan if makespan > 0 else 0
            final_reward = efficiency * 100
        reward = time_penalty + progress_reward + load_balance_reward + completion_reward + defer_gain + final_reward
        info = {"action": job, "schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.done else None}
        return self._get_state(), reward, self.done, info



# CSV 读取函数
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



# 绘制甘特图
def plot_gantt_chart(schedule, num_jobs, all_machines, title="Gantt Chart", filename="wait_gantt.png"):
    if not schedule:
        print("无调度数据")
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    color_map = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    machine_tasks = defaultdict(list)
    for task in schedule:
        job, op, machine, start, end = task
        machine_tasks[machine].append((start, end - start, job, op))
    sorted_machines = sorted(machine_tasks.keys())
    machine_to_y = {m: i for i, m in enumerate(sorted_machines)}
    for m, tasks in machine_tasks.items():
        y = machine_to_y[m]
        for start, dur, job, op in sorted(tasks, key=lambda x: x[0]):
            ax.barh(y, dur, left=start, color=color_map(job), edgecolor='black')
            ax.text(start + dur / 2, y, f"J{job+1}-O{op+1}", ha='center', va='center', fontsize=8)
    ax.set_yticks(range(len(sorted_machines)))
    ax.set_yticklabels([f"M{m}" for m in sorted_machines])
    ax.set_xlabel("时间")
    ax.set_ylabel("机器")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"甘特图已保存为 {filename}")



# 推理求解函数（兼容 WAIT 训练）
def solve_jssp_wait(machine_assignments, processing_times, model_path, device='cpu'):
    if machine_assignments is None or processing_times is None:
        print("输入无效")
        return
    env = JSSPEnv(machine_assignments, processing_times)
    state_dim = len(env.reset())
    action_dim = env.action_dim
    print(f"状态维度={state_dim}, 动作维度={action_dim}")

    model = DQN(state_dim, action_dim)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    state = env.reset()
    done = False
    step = 0
    schedule = []
    start_time = time.time()

    while not done and step < 5000:
        step += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            valid_actions = env.get_valid_actions()
            q_mask = q_values.clone()
            for i in range(action_dim):
                if i not in valid_actions:
                    q_mask[0, i] = -float('inf')
            action = q_mask.argmax().item()
        next_state, reward, done, info = env.step(action)
        state = next_state
        schedule = info["schedule"]
    elapsed = time.time() - start_time
    makespan = max(env.job_end_times)
    print(f"求解完成，步数={step}, Makespan={makespan:.1f}, 用时={elapsed:.2f}秒")
    plot_gantt_chart(schedule, env.num_jobs, env.all_machines, f"JSSP_DQN_WAIT (Makespan={makespan:.1f})")
    return schedule, makespan


# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    problem_file = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]\[6]r[17]c,99gene.csv"
    model_path = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_0.pth"

    print(f"加载文件: {problem_file}")
    ma, pt = load_single_csv(problem_file)
    if ma is not None and pt is not None:
        solve_jssp_wait(ma, pt, model_path=model_path, device=device)
    else:
        print("无法加载问题数据")
