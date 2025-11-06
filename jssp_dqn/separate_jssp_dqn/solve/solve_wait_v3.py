# solve_target_model.py
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random

# -------------------------
# DQN（与训练一致）
# -------------------------
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
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# CSV 解析（与训练一致）
def parse_tuple(cell):
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(')+1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                machine = int(parts[0].strip())
                t = float(parts[1].strip())
                return machine, t
    except:
        pass
    return 0, 0.0

def load_single_csv(file_path):
    try:
        data = None
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    tmp = list(reader)
                if len(tmp) > 1 and len(tmp[0]) > 1:
                    data = tmp
                    # print(f"使用分隔符 '{delimiter}' 读取: {os.path.basename(file_path)}")
                    break
            except:
                continue
        if data is None:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                data = list(reader)

        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0

        machine_data, time_data = [], []
        for row in data[1:]:
            if not row:
                continue
            job_machines, job_times = [], []
            for cell in row[1:1+num_processes]:
                m, t = parse_tuple(cell)
                job_machines.append(m)
                job_times.append(t)
            machine_data.append(job_machines)
            time_data.append(job_times)

        ma = np.array(machine_data, dtype=int)
        pt = np.array(time_data, dtype=float)

        if ma.size == 0 or pt.size == 0 or ma.shape != pt.shape:
            print(f"[X] 数据异常: ma.shape={ma.shape}, pt.shape={pt.shape}")
            return None, None

        print(f"加载成功: {os.path.basename(file_path)}, 形状={ma.shape}")
        return ma, pt
    except Exception as e:
        print(f"读取失败: {e}")
        return None, None


# 环境（与训练一致：含等待动作0）
class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {m: idx for idx, m in enumerate(self.all_machines)}
        self.index_to_machine = {idx: m for idx, m in enumerate(self.all_machines)}
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
        # 1) 下一道工序信息
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                state.extend([self.machine_assignments[job, op], self.processing_times[job, op]])
            else:
                state.extend([0, 0])
        # 2) 机器当前时间线
        state.extend(self.time_table)
        # 3) 作业已排至的结束时间
        state.extend(self.job_end_times)
        # 4) 全局进度
        total_ops = self.num_jobs * self.num_machines
        progress = sum(self.current_step) / total_ops if total_ops > 0 else 0
        state.append(progress)
        # 5) 机器相对负载
        if self.time_table:
            mx = max(self.time_table)
            state.extend([t / mx if mx > 0 else 0 for t in self.time_table])
        else:
            state.extend([0] * len(self.all_machines))
        # 6) 每作业剩余工作量
        for job in range(self.num_jobs):
            remain_ops = self.num_machines - self.current_step[job]
            state.append(sum(self.processing_times[job, self.current_step[job]:]) if remain_ops > 0 else 0)
        # 7) 当前可行动作处理时间统计
        valid_actions = self.get_valid_actions()
        vals = []
        for a in valid_actions:
            if a == 0:  # 跳过等待
                continue
            j = a - 1
            op = self.current_step[j]
            if op < self.num_machines:
                vals.append(self.processing_times[j, op])
        if vals:
            state.extend([min(vals), max(vals), float(np.mean(vals))])
        else:
            state.extend([0, 0, 0])
        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        # 0 表示等待；k>0 表示 job=k-1
        valid = [0]
        for job in range(self.num_jobs):
            if self.current_step[job] < self.num_machines:
                valid.append(job + 1)
        return valid

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {"schedule": self.schedule.copy(),
                                                  "makespan": max(self.job_end_times)}
        valid = self.get_valid_actions()
        if action not in valid:
            return self._get_state(), -50.0, self.done, {}

        # 等待动作
        if action == 0:
            dt = 1.0
            has_other = (len(valid) > 1)
            self.time_table = [t + dt for t in self.time_table]
            self.job_end_times = [t + dt for t in self.job_end_times]
            self.done = all(step >= self.num_machines for step in self.current_step)
            if not has_other:
                wait_penalty = 0.0
            else:
                # 轻罚 + 若 dt ≥ 当前最短工序时间再加重罚
                cand = []
                for a in valid:
                    if a == 0: continue
                    j = a - 1
                    op = self.current_step[j]
                    if op < self.num_machines:
                        cand.append(self.processing_times[j, op])
                shortest = min(cand) if cand else None
                wait_penalty = -1.0
                if shortest is not None and dt >= shortest:
                    wait_penalty -= 5.0
            final_reward = 0.0
            if self.done:
                makespan = max(self.job_end_times)
                lb = self.total_processing_time / len(self.all_machines)
                eff = lb / makespan if makespan > 0 else 0
                final_reward = eff * 100
            reward = wait_penalty + final_reward
            info = {"schedule": self.schedule.copy(),
                    "makespan": max(self.job_end_times) if self.done else None}
            return self._get_state(), reward, self.done, info

        # 调度具体作业
        job = action - 1
        op = self.current_step[job]
        if op >= self.num_machines:
            return self._get_state(), -50.0, self.done, {}

        m_id = self.machine_assignments[job, op]
        m_idx = self.machine_to_index[m_id]
        p = self.processing_times[job, op]

        start = max(self.job_end_times[job], self.time_table[m_idx])
        end = start + p

        prev_machine_idle_time = self.time_table[m_idx] - max(self.time_table)
        prev_job_wait_time = self.time_table[m_idx] - self.job_end_times[job]

        self.job_end_times[job] = end
        self.time_table[m_idx] = end
        self.current_step[job] += 1
        self.schedule.append((job, op, m_id, start, end))
        self.done = all(step >= self.num_machines for step in self.current_step)

        # 奖励（与训练一致的组合）
        time_penalty = -p * 0.05
        total_ops = self.num_jobs * self.num_machines
        prev = self.completed_ops / total_ops if total_ops > 0 else 0
        self.completed_ops += 1
        now = self.completed_ops / total_ops if total_ops > 0 else 0
        progress_reward = (now - prev) * 30
        load_balance_reward = -np.std(self.time_table) * 0.05 if len(self.time_table) > 1 else 0
        completion_reward = 15 if self.current_step[job] == self.num_machines else 0

        # 机会成本惩罚
        opp = 0.0
        va = self.get_valid_actions()
        if len(va) > 2:
            fastest = float('inf')
            for a in va:
                if a == 0: continue
                j2 = a - 1
                op2 = self.current_step[j2]
                if op2 < self.num_machines:
                    fastest = min(fastest, self.processing_times[j2, op2])
            if fastest < float('inf') and p > fastest:
                opp = -0.1 * (p - fastest)

        # 利用率/瓶颈/关键路径
        util_reward = 0.5 if (prev_machine_idle_time > 0 and start == self.time_table[m_idx]) else (-0.1 if prev_job_wait_time > 0 else 0)
        critical_reward = 0.0
        if not self.done:
            rem = []
            for j in range(self.num_jobs):
                r_ops = self.num_machines - self.current_step[j]
                if r_ops > 0:
                    rem.append((j, sum(self.processing_times[j, self.current_step[j]:])))
            if rem:
                max_job, _ = max(rem, key=lambda x: x[1])
                if job == max_job:
                    critical_reward = 0.3

        bottleneck_reward = 0.0
        if self.time_table:
            mx_idx = int(np.argmax(self.time_table))
            mx = self.time_table[mx_idx]
            avg = float(np.mean(self.time_table))
            if mx > avg * 1.2 and m_idx == mx_idx:
                bottleneck_reward = -0.2
            elif m_idx != mx_idx:
                bottleneck_reward = 0.2

        final_reward = 0.0
        if self.done:
            mk = max(self.job_end_times)
            lb = self.total_processing_time / len(self.all_machines)
            eff = lb / mk if mk > 0 else 0
            final_reward = eff * 100

        reward = (time_penalty + progress_reward + load_balance_reward + completion_reward +
                  opp + util_reward + critical_reward + bottleneck_reward + final_reward)

        info = {"schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.done else None}
        return self._get_state(), reward, self.done, info

# 兜底完成调度（与训练一致）
def complete_schedule_with_fallback(env):
    steps = 0
    max_steps = env.num_jobs * env.num_machines * 5
    while not env.done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            break
        # 最短处理时间优先（避免选等待0）
        best = None
        best_t = float('inf')
        for a in valid:
            if a == 0:  # 避免等待
                continue
            j = a - 1
            op = env.current_step[j]
            if op < env.num_machines:
                t = env.processing_times[j, op]
                if t < best_t:
                    best_t = t
                    best = a
        if best is None:
            # 只能等
            best = 0
        _, _, _, _ = env.step(best)
        steps += 1
    return max(env.job_end_times) if env.job_end_times else float('inf')


# 甘特图
def plot_gantt(schedule, num_jobs, all_machines, title="Gantt", filename="gantt.png"):
    sched = [t for t in schedule if t[2] != 0]
    if not sched:
        print("无可绘制任务")
        return
    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    machines = sorted([m for m in np.unique([t[2] for t in sched]) if m != 0])
    m2y = {m: i for i, m in enumerate(machines)}
    buckets = defaultdict(list)
    for job, op, m, s, e in sched:
        buckets[m].append((s, e-s, job, op))
    for m, tasks in buckets.items():
        y = m2y[m]
        tasks.sort(key=lambda x: x[0])
        for s, dur, j, op in tasks:
            ax.barh(y, dur, left=s, height=0.6, edgecolor='black', color=cmap(j))
            ax.text(s + dur/2, y, f"J{j+1}-O{op+1}", ha='center', va='center', fontsize=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"甘特图已保存: {filename}")


# 求解
def solve_with_trained_model(machine_assignments, processing_times, model_path, device=None,
                             max_steps_factor=10, gantt_name="new_gantt_chart.png"):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = JSSPEnv(machine_assignments, processing_times)
    state_dim = len(env.reset())
    action_dim = env.num_jobs + 1  # 含等待动作0
    print(f"状态维度={state_dim}, 动作数={action_dim} (含等待0)")

    # 构建模型并加载权重
    model = DQN(state_dim, action_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("从checkpoint['model_state_dict']加载")
    else:
        model.load_state_dict(ckpt)
        print(" 从纯state_dict加载")
    model.eval()

    state = env.reset()
    done = False
    steps = 0
    max_steps = env.num_jobs * env.num_machines * max_steps_factor
    t0 = time.time()

    while not done and steps < max_steps:
        steps += 1
        valid = env.get_valid_actions()
        if not valid:
            print(" 无有效动作")
            break

        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)
            if torch.isnan(q).any() or torch.isinf(q).any():
                # 退化为最短处理时间启发式
                best, best_t = None, float('inf')
                for a in valid:
                    if a == 0:  # 尽量不等
                        continue
                    j = a - 1
                    op = env.current_step[j]
                    if op < env.num_machines:
                        t_proc = env.processing_times[j, op]
                        if t_proc < best_t:
                            best_t, best = t_proc, a
                action = best if best is not None else 0
            else:
                # mask非法动作
                mask = torch.full_like(q, fill_value=False, dtype=torch.bool)
                mask[0, valid] = True
                q_masked = q.clone()
                q_masked[~mask] = -float('inf')
                action = int(q_masked.argmax().item())

        next_state, _, done, info = env.step(action)
        state = next_state

    if not done:
        print(f"未在 {max_steps} 步内完成，启动兜底策略...")
        makespan = complete_schedule_with_fallback(env)
        done = True
    else:
        makespan = info["makespan"]

    dt = time.time() - t0
    print(f"求解结束:done={done}, steps={steps}, makespan={makespan:.2f}, 用时={dt:.2f}s")

    # 画甘特图
    plot_gantt(env.schedule, env.num_jobs, env.all_machines, title=f"Makespan={makespan:.1f}", filename=gantt_name)
    return env.schedule, makespan

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # === 修改成你的实际路径 ===
    problem_file = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]\[6]r[17]c,9gene.csv"
    model_path = r'D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_v3_1.pth' 

    ma, pt = load_single_csv(problem_file)
    if ma is None or pt is None:
        raise SystemExit("数据加载失败")

    solve_with_trained_model(ma, pt, model_path=model_path, device=device, gantt_name="new_gantt_chart.png")
