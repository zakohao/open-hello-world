# solve_target_model_v2.py
import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

# =========================
# Device
# =========================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")


# =========================
# DQN（与训练一致）
# =========================
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


# =========================
# CSV 解析（与训练一致）
# =========================
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
    """从CSV文件加载数据并解析为机器分配矩阵和处理时间矩阵"""
    try:
        data = None
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    tmp = list(reader)
                if len(tmp) > 1 and len(tmp[0]) > 1:
                    data = tmp
                    print(f"使用分隔符 '{delimiter}' 读取文件: {os.path.basename(file_path)}")
                    break
            except:
                continue

        if data is None:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                data = list(reader)
            print(f"使用默认分隔符读取文件: {os.path.basename(file_path)}")

        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0

        machine_data, time_data = [], []
        for row in data[1:]:
            if not row:
                continue
            job_m, job_t = [], []
            for cell in row[1:1+num_processes]:
                m, t = parse_tuple(cell)
                job_m.append(m)
                job_t.append(t)
            machine_data.append(job_m)
            time_data.append(job_t)

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


# =========================
# 环境（复制你“现在 train 代码”的 JSSPEnv）
# =========================
class JSSPEnv:
    """
    事件驱动(event-based) + 对齐等待 + valid_actions 仅允许最早空闲机器
    Reward: 主项 -ΔCmax + missed/skip/idle 等
    """
    def __init__(self, machine_assignments, processing_times,
                 idle_weight=0.02,
                 miss_weight=2.0,
                 skip_weight=0.2,
                 immediate_bonus=0.5):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}

        self.idle_weight = idle_weight
        self.miss_weight = miss_weight
        self.skip_weight = skip_weight
        self.immediate_bonus = immediate_bonus

        self.reset()

    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0.0] * len(self.all_machines)
        self.job_end_times = [0.0] * self.num_jobs
        self.done = False
        self.schedule = []
        self.total_processing_time = float(np.sum(self.processing_times))
        self.completed_ops = 0
        return self._get_state()

    def _earliest_machine(self):
        min_t = min(self.time_table) if self.time_table else 0.0
        m_idx = int(np.argmin(self.time_table)) if self.time_table else 0
        return m_idx, min_t

    def get_valid_actions(self):
        valid = [0]
        if self.done:
            return valid
        m_idx, t_free = self._earliest_machine()
        m_id = self.index_to_machine[m_idx]

        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op >= self.num_machines:
                continue
            if self.machine_assignments[j, op] != m_id:
                continue
            if self.job_end_times[j] <= t_free + 1e-9:
                valid.append(j + 1)
        return valid

    def _get_state(self):
        state = []

        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines:
                machine = self.machine_assignments[j, op]
                proc_time = self.processing_times[j, op]
                state.extend([float(machine), float(proc_time)])
            else:
                state.extend([0.0, 0.0])

        state.extend([float(x) for x in self.time_table])
        state.extend([float(x) for x in self.job_end_times])

        total_ops = self.num_jobs * self.num_machines
        completed_ops = sum(self.current_step)
        progress = completed_ops / total_ops if total_ops > 0 else 0.0
        state.append(float(progress))

        if self.time_table:
            mx = max(self.time_table)
            if mx > 0:
                state.extend([float(t / mx) for t in self.time_table])
            else:
                state.extend([0.0] * len(self.time_table))
        else:
            state.extend([0.0] * len(self.all_machines))

        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines:
                rem = float(np.sum(self.processing_times[j, op:]))
                state.append(rem)
            else:
                state.append(0.0)

        m_idx, t_free = self._earliest_machine()
        state.append(float(m_idx))
        state.append(float(t_free))

        return np.array(state, dtype=np.float32)

    def _align_wait(self, m_idx, t_free):
        times = []
        m_id = self.index_to_machine[m_idx]

        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines and self.machine_assignments[j, op] == m_id:
                times.append(self.job_end_times[j])

        for t in self.time_table:
            if t > t_free + 1e-9:
                times.append(t)

        for t in self.job_end_times:
            if t > t_free + 1e-9:
                times.append(t)

        if not times:
            dt = 1e-3
            self.time_table[m_idx] += dt
            return dt

        t_next = min(times)
        dt = max(0.0, t_next - t_free)
        self.time_table[m_idx] = t_free + dt
        return dt

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {
                "schedule": self.schedule.copy(),
                "makespan": max(self.job_end_times) if self.job_end_times else 0.0
            }

        prev_signature = (
            tuple(self.current_step),
            tuple(self.time_table),
            tuple(self.job_end_times),
        )

        m_idx, t_free = self._earliest_machine()
        valid_actions = self.get_valid_actions()

        if action not in range(0, self.num_jobs + 1):
            action = 0

        cmax_prev = max(self.job_end_times) if self.job_end_times else 0.0

        if action == 0:
            if len(valid_actions) > 1:
                missed_penalty = -self.miss_weight
                skip_penalty = -self.skip_weight

                best_action = min(
                    [a for a in valid_actions if a != 0],
                    key=lambda a: self.processing_times[a-1, self.current_step[a-1]]
                )
                action = best_action
            else:
                idle = self._align_wait(m_idx, t_free)
                idle_penalty = -self.idle_weight * idle
                reward = idle_penalty
                info = {
                    "schedule": self.schedule.copy(),
                    "makespan": None,
                    "reward_components": {"idle_penalty": idle_penalty}
                }
                return self._get_state(), reward, self.done, info
        else:
            missed_penalty = 0.0
            skip_penalty = 0.0
            if len(valid_actions) > 1 and action not in valid_actions:
                missed_penalty = -self.miss_weight

        job = action - 1
        op = self.current_step[job]

        if op >= self.num_machines:
            idle = self._align_wait(m_idx, t_free)
            idle_penalty = -self.idle_weight * idle
            reward = idle_penalty - self.miss_weight
            info = {"schedule": self.schedule.copy(), "makespan": None,
                    "reward_components": {"idle_penalty": idle_penalty, "invalid_job_penalty": -self.miss_weight}}
            return self._get_state(), reward, self.done, info

        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = float(self.processing_times[job, op])

        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time

        extra_idle_penalty = 0.0
        if machine_idx != m_idx and len(valid_actions) > 1:
            extra_idle_penalty = -0.5

        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.current_step[job] += 1
        self.completed_ops += 1

        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.done = all(s >= self.num_machines for s in self.current_step)

        cmax_now = max(self.job_end_times) if self.job_end_times else 0.0
        delta_cmax = cmax_now - cmax_prev
        reward_main = -delta_cmax

        immediate_reward = 0.0
        if machine_idx == m_idx and start_time <= t_free + 1e-9:
            immediate_reward = self.immediate_bonus

        local_idle_penalty = -0.0

        reward = (reward_main
                  + immediate_reward
                  + missed_penalty
                  + skip_penalty
                  + extra_idle_penalty
                  + local_idle_penalty)

        info = {
            "schedule": self.schedule.copy(),
            "makespan": cmax_now if self.done else None,
            "reward_components": {
                "reward_main(-dCmax)": reward_main,
                "immediate_reward": immediate_reward,
                "missed_penalty": missed_penalty,
                "skip_penalty": skip_penalty,
                "extra_idle_penalty": extra_idle_penalty,
            }
        }

        new_signature = (
            tuple(self.current_step),
            tuple(self.time_table),
            tuple(self.job_end_times),
        )
        if new_signature == prev_signature:
            self.time_table[m_idx] += 1e-2

        return self._get_state(), reward, self.done, info


# =========================
# 兜底完成调度（适配新 env 的 valid_actions 机制）
# =========================
def complete_schedule_with_fallback(env):
    """
    新 env 下：valid_actions 可能常常只有 [0]
    - 若 valid_actions 有 job：选 SPT（最短处理时间）
    - 若只有 0：就执行 0（对齐等待）
    """
    steps = 0
    max_steps = env.num_jobs * env.num_machines * 20  # 比旧版放大些，事件驱动下步数可能多
    while not env.done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            break

        if len(valid) > 1:
            best = min(
                [a for a in valid if a != 0],
                key=lambda a: env.processing_times[a-1, env.current_step[a-1]]
            )
            action = best
        else:
            action = 0

        env.step(action)
        steps += 1

    if env.done:
        return max(env.job_end_times) if env.job_end_times else 0.0
    return max(env.job_end_times) if env.job_end_times else float("inf")


# =========================
# 甘特图（与旧 solve 同等功能）
# =========================
def plot_gantt(schedule, num_jobs, title="Gantt", filename="new_gantt_chart.png"):
    sched = [t for t in schedule if t[2] != 0]
    if not sched:
        print("无可绘制任务")
        return

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = plt.colormaps.get_cmap("tab20").resampled(num_jobs)

    machines = sorted(list(set([t[2] for t in sched if t[2] != 0])))
    m2y = {m: i for i, m in enumerate(machines)}

    buckets = defaultdict(list)
    for job, op, m, s, e in sched:
        buckets[m].append((s, e - s, job, op))

    for m, tasks in buckets.items():
        y = m2y[m]
        tasks.sort(key=lambda x: x[0])
        for s, dur, j, op in tasks:
            ax.barh(y, dur, left=s, height=0.6, edgecolor='black', color=cmap(j))
            ax.text(s + dur / 2, y, f"J{j+1}-O{op+1}", ha='center', va='center', fontsize=8)

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


# =========================
# 推理求解（核心：mask valid_actions）
# =========================
def solve_with_trained_model(ma, pt, model_path,
                            env_params=None,
                            max_steps_factor=50,
                            gantt_name="new_gantt_chart.png"):
    """
    功能对齐旧 solve：
    - 载入模型（支持 checkpoint dict: model_state_dict）
    - 贪婪推理 + mask 非法动作
    - 失败则 fallback 完成
    - 输出 makespan + 甘特图
    """
    if env_params is None:
        # 与 train_target 中创建 env 的参数保持一致
        env_params = dict(miss_weight=3.0, idle_weight=0.05, skip_weight=0.2, immediate_bonus=0.5)

    env = JSSPEnv(ma, pt, **env_params)
    state = env.reset()

    state_dim = len(state)
    action_dim = env.num_jobs + 1  # 0..num_jobs
    print(f"状态维度={state_dim}, 动作数={action_dim} (含0)")

    model = DQN(state_dim, action_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("从 checkpoint['model_state_dict'] 加载模型")
    else:
        model.load_state_dict(ckpt)
        print("从纯 state_dict 加载模型")

    model.eval()

    steps = 0
    max_steps = env.num_jobs * env.num_machines * max_steps_factor
    t0 = time.time()

    done = False
    info = {}

    while not done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            print("无有效动作，提前结束")
            break

        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)

            if torch.isnan(q).any() or torch.isinf(q).any():
                # Q 值异常：退化为启发式（若有job则SPT，否则0）
                if len(valid) > 1:
                    action = min(
                        [a for a in valid if a != 0],
                        key=lambda a: env.processing_times[a-1, env.current_step[a-1]]
                    )
                else:
                    action = 0
            else:
                # ✅ 关键：只在 valid_actions 里选 argmax
                mask = torch.full_like(q, False, dtype=torch.bool)
                mask[0, valid] = True
                q_masked = q.clone()
                q_masked[~mask] = -float("inf")
                action = int(q_masked.argmax().item())

        state, _, done, info = env.step(action)
        steps += 1

    if not env.done:
        print(f"未在 {max_steps} 步内完成，启动 fallback...")
        makespan = complete_schedule_with_fallback(env)
        done = env.done
    else:
        makespan = info.get("makespan", max(env.job_end_times) if env.job_end_times else 0.0)

    dt = time.time() - t0
    print(f"求解结束: done={env.done}, steps={steps}, makespan={makespan:.2f}, 用时={dt:.2f}s")

    plot_gantt(env.schedule, env.num_jobs, title=f"Makespan={makespan:.1f}", filename=gantt_name)
    return env.schedule, makespan


# =========================
# Main
# =========================
if __name__ == "__main__":
    # === 修改成你的实际路径 ===
    problem_file = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]\[6]r[17]c,30gene.csv"
    model_path = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_v5_1.pth"

    ma, pt = load_single_csv(problem_file)
    if ma is None or pt is None:
        raise SystemExit("数据加载失败")

    solve_with_trained_model(
        ma, pt,
        model_path=model_path,
        env_params=dict(miss_weight=3.0, idle_weight=0.05, skip_weight=0.2, immediate_bonus=0.5),
        gantt_name="new_gantt_chart.png"
    )
