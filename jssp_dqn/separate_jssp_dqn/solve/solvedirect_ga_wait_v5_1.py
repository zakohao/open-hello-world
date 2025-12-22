# ga_vs_dqn_compare_gantt_from_matrices.py
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from matplotlib import gridspec
from matplotlib.patches import Patch

# =========================================================
# Device
# =========================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")


# =========================================================
# Helpers: sanitize -1 padded matrices
# =========================================================
def sanitize_matrices(machine_matrix: np.ndarray, time_matrix: np.ndarray):
    """
    Accepts:
      - machine_matrix: int, -1 means no operation
      - time_matrix: float/int, -1 means no operation
    Returns:
      - ma: int matrix where invalid ops set to 0, valid are 1-based machine ids
      - pt: float matrix where invalid ops set to 0.0, valid > 0
    Also checks shapes and consistency.
    """
    if machine_matrix.shape != time_matrix.shape:
        raise ValueError(f"Shape mismatch: machine_matrix{machine_matrix.shape} vs time_matrix{time_matrix.shape}")

    ma = np.array(machine_matrix, dtype=int).copy()
    pt = np.array(time_matrix, dtype=float).copy()

    invalid = (ma == -1) | (pt == -1)
    ma[invalid] = 0
    pt[invalid] = 0.0

    # Disallow negative values other than -1
    if (ma < 0).any():
        raise ValueError("machine_matrix contains values < -1")
    if (pt < 0).any():
        raise ValueError("time_matrix contains values < -1")

    # If one says valid and the other says invalid -> force invalid
    mismatch = ((ma == 0) & (pt > 0)) | ((ma > 0) & (pt == 0))
    if mismatch.any():
        # safest: invalidate them
        ma[mismatch] = 0
        pt[mismatch] = 0.0
        print("[WARN] Found ma/pt mismatch; those cells were invalidated to 0.")

    # If a row has valid ops after an invalid, you can allow it, but usually padding is contiguous.
    # We'll allow it, GA/DQN can still handle because we stop at first 0 for GA conversion.

    return ma, pt


# =========================================================
# DQN model (same as your solve)
# =========================================================
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


# =========================================================
# DQN Env (same as your solve)
# =========================================================
class JSSPEnv:
    """
    Event-based + align-wait + valid_actions only earliest free machine.

    IMPORTANT:
    - Assumes ma uses 1-based machine ids for valid ops and 0 for invalid/padding
    - Here we treat num_machines = ma.shape[1] (like your solve),
      but if some jobs have fewer ops (padding), then those padded ops are still present.
      To support varying op counts per job, we compute per-job "effective length"
      and stop jobs at that length.
    """
    def __init__(self, machine_assignments, processing_times,
                 idle_weight=0.02,
                 miss_weight=2.0,
                 skip_weight=0.2,
                 immediate_bonus=0.5):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.max_cols = machine_assignments.shape

        # effective ops per job: count until first 0 (padding)
        self.job_ops = []
        for j in range(self.num_jobs):
            row = self.machine_assignments[j]
            k = 0
            for s in range(self.max_cols):
                if int(row[s]) == 0 or float(self.processing_times[j, s]) == 0.0:
                    break
                k += 1
            self.job_ops.append(k)

        self.all_machines = np.unique(np.concatenate([self.machine_assignments.flatten(), np.array([0])]))
        if self.all_machines.size == 0:
            raise ValueError("No valid machines found in machine_assignments (>0).")

        self.machine_to_index = {int(machine): idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: int(machine) for idx, machine in enumerate(self.all_machines)}

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
        self.completed_ops = 0
        self.total_ops = sum(self.job_ops)
        self.total_processing_time = float(np.sum(self.processing_times))
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
            if op >= self.job_ops[j]:
                continue
            if int(self.machine_assignments[j, op]) != int(m_id):
                continue
            if self.job_end_times[j] <= t_free + 1e-9:
                valid.append(j + 1)
        return valid

    def _get_state(self):
        state = []

        # per job next op (machine, proc_time) or (0,0)
        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.job_ops[j]:
                machine = float(self.machine_assignments[j, op])
                proc_time = float(self.processing_times[j, op])
                state.extend([machine, proc_time])
            else:
                state.extend([0.0, 0.0])

        # machine available times + job end times
        state.extend([float(x) for x in self.time_table])
        state.extend([float(x) for x in self.job_end_times])

        progress = (sum(self.current_step) / self.total_ops) if self.total_ops > 0 else 1.0
        state.append(float(progress))

        # normalized machine times
        mx = max(self.time_table) if self.time_table else 0.0
        if mx > 0:
            state.extend([float(t / mx) for t in self.time_table])
        else:
            state.extend([0.0] * len(self.all_machines))

        # remaining time per job
        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.job_ops[j]:
                rem = float(np.sum(self.processing_times[j, op:self.job_ops[j]]))
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
            if op < self.job_ops[j] and int(self.machine_assignments[j, op]) == int(m_id):
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

        prev_signature = (tuple(self.current_step), tuple(self.time_table), tuple(self.job_end_times))
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
                    key=lambda a: self.processing_times[a - 1, self.current_step[a - 1]]
                )
                action = best_action
            else:
                idle = self._align_wait(m_idx, t_free)
                idle_penalty = -self.idle_weight * idle
                return self._get_state(), idle_penalty, self.done, {"schedule": self.schedule.copy(), "makespan": None}
        else:
            missed_penalty = 0.0
            skip_penalty = 0.0
            if len(valid_actions) > 1 and action not in valid_actions:
                missed_penalty = -self.miss_weight

        job = action - 1
        op = self.current_step[job]
        if op >= self.job_ops[job]:
            idle = self._align_wait(m_idx, t_free)
            idle_penalty = -self.idle_weight * idle
            reward = idle_penalty - self.miss_weight
            return self._get_state(), reward, self.done, {"schedule": self.schedule.copy(), "makespan": None}

        machine_id = int(self.machine_assignments[job, op])
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
        self.done = all(self.current_step[j] >= self.job_ops[j] for j in range(self.num_jobs))

        cmax_now = max(self.job_end_times) if self.job_end_times else 0.0
        reward_main = -(cmax_now - cmax_prev)

        immediate_reward = 0.0
        if machine_idx == m_idx and start_time <= t_free + 1e-9:
            immediate_reward = self.immediate_bonus

        reward = reward_main + immediate_reward + missed_penalty + skip_penalty + extra_idle_penalty

        new_signature = (tuple(self.current_step), tuple(self.time_table), tuple(self.job_end_times))
        if new_signature == prev_signature:
            self.time_table[m_idx] += 1e-2

        info = {"schedule": self.schedule.copy(), "makespan": cmax_now if self.done else None}
        return self._get_state(), reward, self.done, info


def complete_schedule_with_fallback(env):
    steps = 0
    max_steps = max(1, env.total_ops) * 20
    while not env.done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            break
        if len(valid) > 1:
            best = min(
                [a for a in valid if a != 0],
                key=lambda a: env.processing_times[a - 1, env.current_step[a - 1]]
            )
            action = best
        else:
            action = 0
        env.step(action)
        steps += 1
    return max(env.job_end_times) if env.job_end_times else float("inf")


def solve_with_trained_model(ma, pt, model_path,
                            env_params=None,
                            max_steps_factor=50):
    if env_params is None:
        env_params = dict(miss_weight=3.0, idle_weight=0.05, skip_weight=0.2, immediate_bonus=0.5)

    env = JSSPEnv(ma, pt, **env_params)
    state = env.reset()

    state_dim = len(state)
    action_dim = env.num_jobs + 1
    print(f"[DQN] 状态维度={state_dim}, 动作数={action_dim} (含0)")

    model = DQN(state_dim, action_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[DQN] 从 checkpoint['model_state_dict'] 加载模型")
    else:
        model.load_state_dict(ckpt)
        print("[DQN] 从纯 state_dict 加载模型")
    model.eval()

    steps = 0
    max_steps = max(1, env.total_ops) * max_steps_factor
    t0 = time.time()

    done = False
    info = {}
    while not done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            print("[DQN] 无有效动作，提前结束")
            break

        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)

            if torch.isnan(q).any() or torch.isinf(q).any():
                if len(valid) > 1:
                    action = min(
                        [a for a in valid if a != 0],
                        key=lambda a: env.processing_times[a - 1, env.current_step[a - 1]]
                    )
                else:
                    action = 0
            else:
                mask = torch.full_like(q, False, dtype=torch.bool)
                mask[0, valid] = True
                q_masked = q.clone()
                q_masked[~mask] = -float("inf")
                action = int(q_masked.argmax().item())

        state, _, done, info = env.step(action)
        steps += 1

    if not env.done:
        print(f"[DQN] 未在 {max_steps} 步内完成，启动 fallback...")
        makespan = complete_schedule_with_fallback(env)
    else:
        makespan = info.get("makespan", max(env.job_end_times) if env.job_end_times else 0.0)

    dt = time.time() - t0
    print(f"[DQN] done={env.done}, steps={steps}, makespan={makespan:.2f}, 用时={dt:.2f}s")
    return env.schedule, float(makespan), float(dt)


# =========================================================
# GA solver (adapted to -1 padding)
# =========================================================
def build_jobs_from_matrices(ma_1based: np.ndarray, pt: np.ndarray):
    """
    Convert ma/pt to GA job steps list.
    Stop at first 0 per job (padding).
    Machine stored as 0-based internally.
    """
    jobs = []
    num_jobs, num_cols = pt.shape
    for j in range(num_jobs):
        steps = []
        for s in range(num_cols):
            m = int(ma_1based[j, s])
            t = float(pt[j, s])
            if m == 0 or t == 0.0:
                break
            steps.append({"machine": m - 1, "duration": t})  # 0-based
        jobs.append(steps)
    return jobs


def ga_solve_and_schedule(ma_1based, pt,
                          POP_SIZE=100, MAX_GEN=200,
                          CX_PROB=0.8, MUT_PROB=0.1,
                          ELITE_SIZE=2, TOURNAMENT_SIZE=5,
                          seed=0):
    random.seed(seed)
    np.random.seed(seed)

    jobs = build_jobs_from_matrices(ma_1based, pt)
    num_jobs = len(jobs)
    total_ops = sum(len(j) for j in jobs)
    if total_ops == 0:
        raise ValueError("GA: total_ops==0 (no valid operations)")

    class Chromosome:
        def __init__(self, genes):
            self.genes = genes
            self._makespan = None

        def validate(self):
            job_progress = defaultdict(int)
            for job_id, step_id in self.genes:
                if step_id != job_progress[job_id]:
                    return False
                job_progress[job_id] += 1
            required = {(j, s) for j in range(num_jobs) for s in range(len(jobs[j]))}
            return set(self.genes) == required

        def repair(self):
            required_genes = {(j, s) for j in range(num_jobs) for s in range(len(jobs[j]))}
            new_genes = [g for g in self.genes if g in required_genes]
            present = set(new_genes)
            missing = list(required_genes - present)
            random.shuffle(missing)
            new_genes.extend(missing)

            sorted_genes = []
            job_progress = defaultdict(int)
            remaining = new_genes.copy()
            while remaining:
                available = [g for g in remaining if g[1] == job_progress[g[0]]]
                if not available:
                    available = remaining
                selected = random.choice(available)
                sorted_genes.append(selected)
                job_progress[selected[0]] += 1
                remaining.remove(selected)

            self.genes = sorted_genes
            self._makespan = None
            return self

        def makespan(self):
            if self._makespan is None:
                self._makespan = self._calc()
            return self._makespan

        def _calc(self):
            machine_times = defaultdict(float)
            job_times = defaultdict(float)
            for job_id, step_id in self.genes:
                machine = jobs[job_id][step_id]["machine"]
                dur = jobs[job_id][step_id]["duration"]
                start = max(job_times[job_id], machine_times[machine])
                end = start + dur
                job_times[job_id] = end
                machine_times[machine] = end
            return max(job_times.values(), default=0.0)

    def initialize_population():
        pop = []
        for _ in range(POP_SIZE):
            genes = []
            job_progress = defaultdict(int)
            remaining = total_ops
            while remaining > 0:
                avail_jobs = [j for j in range(num_jobs) if job_progress[j] < len(jobs[j])]
                jsel = random.choice(avail_jobs)
                step = job_progress[jsel]
                genes.append((jsel, step))
                job_progress[jsel] += 1
                remaining -= 1
            pop.append(Chromosome(genes))
        return pop

    def strict_precedence_crossover(p1, p2):
        pool = defaultdict(list)
        for gene in p1.genes + p2.genes:
            pool[gene[0]].append(gene)

        child = []
        job_progress = defaultdict(int)
        while len(child) < total_ops:
            avail_jobs = [j for j in range(num_jobs) if job_progress[j] < len(jobs[j])]
            jsel = random.choice(avail_jobs)
            candidates = [g for g in pool[jsel] if g[1] == job_progress[jsel]]
            if candidates:
                gsel = random.choice(candidates)
                child.append(gsel)
                pool[jsel].remove(gsel)
                job_progress[jsel] += 1
        return Chromosome(child).repair()

    def enhanced_safe_mutation(chrom):
        genes = chrom.genes.copy()

        swap_candidates = []
        for i in range(len(genes)):
            j, s = genes[i]
            if s == len(jobs[j]) - 1:
                continue
            prev_same = (i > 0 and genes[i - 1][0] == j)
            next_same = (i < len(genes) - 1 and genes[i + 1][0] == j)
            if not prev_same and not next_same:
                swap_candidates.append(i)

        if len(swap_candidates) >= 2 and random.random() < MUT_PROB:
            i1, i2 = random.sample(swap_candidates, 2)
            if genes[i1][0] != genes[i2][0]:
                genes[i1], genes[i2] = genes[i2], genes[i1]
        return Chromosome(genes).repair()

    def tournament_selection(pop):
        selected = []
        for _ in range(2):
            contestants = random.sample(pop, TOURNAMENT_SIZE)
            contestants.sort(key=lambda x: x.makespan())
            selected.append(deepcopy(contestants[0]))
        return selected

    t0 = time.time()
    pop = initialize_population()
    best = min(pop, key=lambda x: x.makespan())

    for _ in range(MAX_GEN):
        pop = [c.repair() for c in pop]
        fits = [c.makespan() if c.validate() else float("inf") for c in pop]

        cur_best = min(fits)
        if cur_best < best.makespan():
            best = deepcopy(pop[int(np.argmin(fits))])

        elite = sorted(pop, key=lambda x: x.makespan())[:ELITE_SIZE]
        new_pop = elite.copy()

        while len(new_pop) < POP_SIZE:
            p1, p2 = tournament_selection(pop)
            if random.random() < CX_PROB:
                child = strict_precedence_crossover(p1, p2)
            else:
                child = random.choice([p1, p2])
            child = enhanced_safe_mutation(child)
            new_pop.append(child)

        pop = new_pop[:POP_SIZE]

    ga_time = time.time() - t0
    ga_makespan = float(best.makespan())

    # convert to schedule: (job, op, machine_id(1-based), start, end)
    machine_times = defaultdict(float)
    job_times = defaultdict(float)
    schedule = []
    for job_id, step_id in best.genes:
        machine0 = jobs[job_id][step_id]["machine"]  # 0-based
        dur = jobs[job_id][step_id]["duration"]
        start = max(job_times[job_id], machine_times[machine0])
        end = start + dur
        job_times[job_id] = end
        machine_times[machine0] = end
        schedule.append((job_id, step_id, machine0 + 1, start, end))

    print(f"[GA] makespan={ga_makespan:.2f}, 用时={ga_time:.2f}s, valid={best.validate()}")
    return schedule, ga_makespan, float(ga_time)


# =========================================================
# Plotting: two gantts aligned + right-side legend & stats
# =========================================================
def _draw_gantt_on_ax(ax, schedule, num_jobs, title, cmap, xlim_max):
    sched = [t for t in schedule if int(t[2]) != 0 and (float(t[4]) - float(t[3])) > 0]
    if not sched:
        ax.set_title(title + " (no tasks)")
        ax.set_xlim(0, xlim_max)
        ax.set_ylabel("Machine")
        ax.grid(True, linestyle='--', alpha=0.6)
        return

    machines = sorted(list(set([int(t[2]) for t in sched])))
    m2y = {m: i for i, m in enumerate(machines)}

    buckets = defaultdict(list)
    for job, op, m, s, e in sched:
        buckets[int(m)].append((float(s), float(e - s), int(job), int(op)))

    for m, tasks in buckets.items():
        y = m2y[m]
        tasks.sort(key=lambda x: x[0])
        for s, dur, j, op in tasks:
            ax.barh(y, dur, left=s, height=0.6, edgecolor='black', color=cmap(j))
            ax.text(s + dur / 2, y, f"J{j+1}-O{op+1}", ha='center', va='center', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.invert_yaxis()
    ax.set_xlim(0, xlim_max)
    ax.grid(True, linestyle='--', alpha=0.6)


def plot_compare_canvas(ga_schedule, dqn_schedule,
                        num_jobs,
                        ga_makespan, dqn_makespan,
                        ga_time_s, dqn_time_s,
                        out_png="ga_vs_dqn_compare.png",
                        suptitle="GA vs DQN (JSSP)"):
    cmap = plt.colormaps.get_cmap("tab20").resampled(num_jobs)

    xlim_max = max(ga_makespan, dqn_makespan) * 1.05 + 1e-6

    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        width_ratios=[4.8, 1.6],
        height_ratios=[1, 1],
        wspace=0.12, hspace=0.18
    )

    ax_ga = fig.add_subplot(gs[0, 0])
    ax_dqn = fig.add_subplot(gs[1, 0], sharex=ax_ga)
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    _draw_gantt_on_ax(ax_ga, ga_schedule, num_jobs,
                      title=f"GA Gantt (makespan={ga_makespan:.1f})",
                      cmap=cmap, xlim_max=xlim_max)
    _draw_gantt_on_ax(ax_dqn, dqn_schedule, num_jobs,
                      title=f"DQN Gantt (makespan={dqn_makespan:.1f})",
                      cmap=cmap, xlim_max=xlim_max)

    ratio = (dqn_makespan / ga_makespan) if ga_makespan > 0 else float("inf")

    lines = [
        "=== Metrics ===",
        f"GA solve time     : {ga_time_s:.2f} s",
        f"DQN solve time    : {dqn_time_s:.2f} s",
        "",
        f"GA makespan       : {ga_makespan:.2f}",
        f"DQN makespan      : {dqn_makespan:.2f}",
        f"DQN / GA ratio    : {ratio:.4f}",
        "",
        "=== Job Color Legend ===",
    ]
    ax_info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=12)

    patches = [Patch(facecolor=cmap(j), edgecolor='black', label=f"Job {j+1}") for j in range(num_jobs)]
    ax_info.legend(
        handles=patches,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        ncol=1 if num_jobs <= 12 else 2,
        frameon=True,
        fontsize=10
    )

    fig.suptitle(suptitle, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_png, dpi=220)
    plt.show()
    print(f"[OK] 对比画布已保存: {out_png}")


# =========================================================
# Main: YOU PROVIDE MATRICES HERE
# =========================================================
if __name__ == "__main__":
    # 你在这里给矩阵：
    # - machine_matrix: -1 表示没有该工序；有效值为机器编号(从1开始)
    # - time_matrix: -1 表示没有该工序；有效值为加工时间(>0)
    #
    # ✅ 你可以随意改变形状（jobs数、最大工序列数），代码会自适应。

    # 示例：你需要自己提供 machine_matrix（此处仅占位）
    # !!! 请把下面这个 machine_matrix 替换成你的真实矩阵 !!!
    machine_matrix = np.array([
        [ 1,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1, -1, -1],
        [ 2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1, -1, -1],
        [ 1,  3,  4,  6,  5,  6,  7,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1],
        [ 2,  3,  5,  6,  4,  6,  7,  8,  9, 10, 11, 12, 13, -1, -1, -1, -1],
        [ 1,  3,  5,  6,  4,  6,  7,  8,  9,  4,  6, 10, 11, 12, 13, -1, -1],
        [ 2,  3,  5,  6,  4,  6,  7,  8,  9,  4,  6,  5,  6, 10, 11, 12, 13],
    ], dtype=int)

    time_matrix = np.array([
                     [300,300,300,300,300,300,300,300,300,300,300,-1,-1,-1,-1,-1,-1],
    [300,300,300,300,300,300,300,300,300,300,300,-1,-1,-1,-1,-1,-1],
    [300,300,300,300,300,300,300,300,300,300,300,300,300,-1,-1,-1,-1],
    [300,300,300,300,300,300,300,300,300,300,300,300,300,-1,-1,-1,-1],
    [300,300,300,300,300,300,300,300,300,300,300,300,300,300,300,-1,-1],
    [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
                     ], dtype=int)

    # DQN 模型路径（你改成自己的）
    MODEL_PATH = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_v5_1.pth"

    # 1) sanitize -1 padding => ma (0/1-based), pt (0/float)
    ma, pt = sanitize_matrices(machine_matrix, time_matrix)
    num_jobs = ma.shape[0]

    # 2) GA solve
    ga_sched, ga_ms, ga_t = ga_solve_and_schedule(
        ma, pt,
        POP_SIZE=100, MAX_GEN=200,
        CX_PROB=0.8, MUT_PROB=0.1,
        ELITE_SIZE=2, TOURNAMENT_SIZE=5,
        seed=0
    )

    # 3) DQN solve
    dqn_sched, dqn_ms, dqn_t = solve_with_trained_model(
        ma, pt,
        model_path=MODEL_PATH,
        env_params=dict(miss_weight=3.0, idle_weight=0.05, skip_weight=0.2, immediate_bonus=0.5),
        max_steps_factor=50
    )

    # 4) compare canvas (aligned x-axis)
    plot_compare_canvas(
        ga_sched, dqn_sched,
        num_jobs=num_jobs,
        ga_makespan=ga_ms, dqn_makespan=dqn_ms,
        ga_time_s=ga_t, dqn_time_s=dqn_t,
        out_png="ga_vs_dqn_compare_from_matrices.png",
        suptitle="GA vs DQN (from matrices)"
    )
