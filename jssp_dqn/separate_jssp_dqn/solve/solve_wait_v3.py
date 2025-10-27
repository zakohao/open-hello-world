import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import time


# ======================== DQN模型定义 ========================
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
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


# ======================== CSV加载函数 ========================
def parse_tuple(cell):
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            c = cell[cell.find('(') + 1:cell.find(')')]
            p = c.split(',')
            if len(p) >= 2:
                return int(p[0]), float(p[1])
    except:
        pass
    return 0, 0.0


def load_single_csv(file_path):
    try:
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    data = list(reader)
                    if len(data) > 1 and len(data[0]) > 1:
                        break
            except:
                continue
        header = data[0]
        num_p = len(header) - 1
        mach, time = [], []
        for row in data[1:]:
            if not row:
                continue
            jm, jt = [], []
            for cell in row[1:1 + num_p]:
                m, t = parse_tuple(cell)
                jm.append(m)
                jt.append(t)
            mach.append(jm)
            time.append(jt)
        ma = np.array(mach, int)
        pt = np.array(time, float)
        print(f"成功加载: {os.path.basename(file_path)}, shape={ma.shape}")
        return ma, pt
    except Exception as e:
        print("文件错误:", e)
        return None, None


# ======================== JSSP环境定义 ========================
class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape
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
        self.completed_ops = 0
        return self._get_state()

    def _get_state(self):
        state = []
        # 1. 每个作业的下一工序(machine, time)
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])

        # 2. 当前机器时间表
        state.extend(self.time_table)

        # 3. 各作业完成时间
        state.extend(self.job_end_times)

        # 4. 全局进度
        total_ops = self.num_jobs * self.num_machines
        progress = sum(self.current_step) / total_ops if total_ops > 0 else 0
        state.append(progress)

        # 5. 机器负载标准化
        if len(self.time_table) > 0:
            max_t = max(self.time_table)
            if max_t > 0:
                state.extend([t / max_t for t in self.time_table])
            else:
                state.extend([0] * len(self.time_table))
        else:
            state.extend([0] * len(self.all_machines))

        # 6. 剩余工时
        for job in range(self.num_jobs):
            remain = self.num_machines - self.current_step[job]
            state.append(sum(self.processing_times[job, self.current_step[job]:]) if remain > 0 else 0)

        # 7. 当前可执行动作的工序时间统计
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

        # ---------- 改进的等待逻辑 ----------
        if action == 0:
            # 判断是否真的无法执行任何工序
            has_executable = False
            for job in range(self.num_jobs):
                op = self.current_step[job]
                if op < self.num_machines:
                    machine = self.machine_assignments[job, op]
                    m_idx = self.machine_to_index[machine]
                    if self.job_end_times[job] <= self.time_table[m_idx]:
                        has_executable = True
                        break

            # 若仍有工序可执行而选择等待 -> 惩罚
            if has_executable:
                reward = -10.0
            else:
                reward = -1.0

            # 推进时间到下一个可执行工序的最早时刻
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
        # ---------- 等待逻辑结束 ----------

        # 调度动作
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
        return self._get_state(), -pt * 0.05, self.done, {"makespan": max(self.job_end_times), "schedule": self.schedule.copy()}


# ======================== 绘制甘特图 ========================
def plot_gantt_chart(schedule, num_jobs, all_machines, title="Gantt Chart"):
    if not schedule:
        print("无调度数据")
        return
    color_map = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    machines = sorted([m for m in all_machines if m > 0])
    machine_to_y = {m: i for i, m in enumerate(machines)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for job, op, m, s, e in schedule:
        if m <= 0:
            continue
        y = machine_to_y[m]
        ax.barh(y, e - s, left=s, height=0.5, color=color_map(job), edgecolor="black")
        ax.text(s + (e - s) / 2, y, f"J{job+1}-O{op+1}", ha="center", va="center", fontsize=8)

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.invert_yaxis()
    ax.set_xlabel("时间")
    ax.set_ylabel("机器")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("solve_gantt.png", dpi=200)
    plt.show()


# ======================== 模型推理求解 ========================
def solve_with_trained_model(model_path, csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    ma, pt = load_single_csv(csv_path)
    if ma is None:
        print("无法加载CSV数据")
        return

    env = JSSPEnv(ma, pt)
    state_dim = len(env.reset())
    action_dim = env.num_jobs + 1  

    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    state = env.reset()
    done = False
    steps = 0
    total_reward = 0

    start_time = time.time()
    while not done and steps < 20000:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            print("无可执行动作，提前结束。")
            break

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            mask = torch.ones_like(q_values, dtype=torch.bool)
            for i in range(q_values.shape[1]):
                if i not in valid_actions:
                    mask[0, i] = False
            q_values[~mask] = -float("inf")

            if torch.all(q_values == -float("inf")):
                action = random.choice(valid_actions)
            else:
                action = q_values.argmax().item()

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        if steps > 10000 and len(env.schedule) < 10:
            print("模型可能陷入等待循环，强制退出。")
            break

    elapsed = time.time() - start_time
    makespan = max(env.job_end_times)
    print(f"\n求解完成! makespan={makespan:.1f}, 步数={steps}, 耗时={elapsed:.2f}s")

    plot_gantt_chart(env.schedule, env.num_jobs, env.all_machines,
                     title=f"JSSP Solve (makespan={makespan:.1f})")


# ======================== 主函数 ========================
if __name__ == "__main__":
    csv_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]\[6]r[17]c,9gene.csv"
    model_path = r'D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_v3_1.pth' 
    solve_with_trained_model(model_path, csv_path)
