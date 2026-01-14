# ga_vs_dqn_compare_gantt.py
import os
import csv
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
# Global plot style (even bigger fonts)
# =========================================================
plt.rcParams.update({
    "font.size": 22,
    "axes.titlesize": 30,
    "axes.labelsize": 26,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 22,
})

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
# CSV loader (same as your solve)
# =========================================================
def parse_tuple(cell):
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(') + 1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                machine = int(parts[0].strip())
                t = float(parts[1].strip())
                return machine, t
    except:
        pass
    return 0, 0.0


def load_single_csv(file_path):
    """Load CSV => machine assignment matrix (int) + processing time matrix (float)."""
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
            for cell in row[1:1 + num_processes]:
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


# =========================================================
# DQN model (for wait_1.pth style: state_dim=33, action_dim=6)
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
# DQN Env (for wait_1.pth: no "wait action", action=job_index 0..5)
# =========================================================
class JSSPEnv:
    def __init__(self, machine_assignments, processing_times, **kwargs):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {m: idx for idx, m in enumerate(self.all_machines)}
        self.index_to_machine = {idx: m for idx, m in enumerate(self.all_machines)}

        # count valid ops per job (machine_id != 0)
        self.actual_ops_per_job = []
        for j in range(self.num_jobs):
            self.actual_ops_per_job.append(int(np.sum(self.machine_assignments[j, :] != 0)))

        self.reset()

    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0.0] * len(self.all_machines)
        self.job_end_times = [0.0] * self.num_jobs
        self.done = False
        self.schedule = []

        # utilization bookkeeping (same as your solve_jssp code)
        self.machine_total_time = [0.0] * len(self.all_machines)
        self.machine_working_time = [0.0] * len(self.all_machines)

        return self._get_state()

    def _get_state(self):
        state = []

        # 1) 6 jobs × (machine, proc_time) => 12 dims
        for job in range(6):
            op = self.current_step[job]
            if op < self.num_machines:
                m = float(self.machine_assignments[job, op])
                p = float(self.processing_times[job, op])
                state.extend([m, p])
            else:
                state.extend([0.0, 0.0])

        # 2) machine loads => 13 dims
        need_m = 13
        for i in range(min(len(self.time_table), need_m)):
            state.append(float(self.time_table[i]))
        while len(state) < 12 + need_m:
            state.append(0.0)

        # 3) job end times => 6 dims
        for i in range(min(6, len(self.job_end_times))):
            state.append(float(self.job_end_times[i]))
        while len(state) < 12 + need_m + 6:
            state.append(0.0)

        # 4) utilization => 2 dims
        util_count = 2
        for i in range(min(util_count, len(self.all_machines))):
            tot = self.machine_total_time[i]
            util = (self.machine_working_time[i] / tot) if tot > 0 else 0.0
            state.append(float(util))
        while len(state) < 12 + need_m + 6 + util_count:
            state.append(0.0)

        # ensure 33 dims
        if len(state) != 33:
            if len(state) > 33:
                state = state[:33]
            else:
                state.extend([0.0] * (33 - len(state)))

        return np.array(state, dtype=np.float32)

    def get_valid_actions(self):
        valid = []
        for job in range(self.num_jobs):
            if self.current_step[job] < self.actual_ops_per_job[job]:
                valid.append(job)  # 0..num_jobs-1
        return valid

    def step(self, action):
        job = int(action)
        op = self.current_step[job]

        if self.done or job not in self.get_valid_actions() or op >= self.actual_ops_per_job[job]:
            return self._get_state(), -10.0, self.done, {}

        machine_id = int(self.machine_assignments[job, op])
        machine_idx = self.machine_to_index[machine_id]
        proc_time = float(self.processing_times[job, op])

        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time

        # utilization bookkeeping
        self.machine_total_time[machine_idx] = end_time
        self.machine_working_time[machine_idx] += proc_time

        # update
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.schedule.append((job, op, machine_id, start_time, end_time))

        self.current_step[job] += 1
        self.done = all(self.current_step[j] >= self.actual_ops_per_job[j] for j in range(self.num_jobs))

        # reward (kept same as your simplified solve_jssp)
        reward = -proc_time
        if self.current_step[job] == self.actual_ops_per_job[job]:
            reward += 10.0
        if self.done:
            makespan = max(self.job_end_times) if self.job_end_times else 0.0
            reward += -makespan * 0.01

        info = {
            "schedule": self.schedule.copy(),
            "makespan": (max(self.job_end_times) if self.done else None)
        }
        return self._get_state(), reward, self.done, info


def solve_with_trained_model(ma, pt, model_path, max_steps_factor=200):
    """
    For wait_1.pth style model:
    - state_dim must be 33
    - action_dim must be num_jobs (6)
    """
    env = JSSPEnv(ma, pt)
    state = env.reset()

    state_dim = len(state)
    action_dim = env.num_jobs
    print(f"[DQN(wait_1)] state_dim={state_dim}, action_dim={action_dim}")

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
    max_steps = env.num_jobs * max(env.actual_ops_per_job) * max_steps_factor
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
            q = model(st)  # [1, action_dim]

            # mask invalid actions
            q_masked = q.clone()
            for a in range(action_dim):
                if a not in valid:
                    q_masked[0, a] = -float("inf")

            action = int(q_masked.argmax().item())

        state, _, done, info = env.step(action)
        steps += 1

    makespan = max(env.job_end_times) if env.job_end_times else 0.0
    dt = time.time() - t0
    print(f"[DQN] done={env.done}, steps={steps}, makespan={makespan:.2f}, 用时={dt:.2f}s")
    return env.schedule, float(makespan), float(dt)


# =========================================================
# GA solver (UNCHANGED; keep as-is)
# =========================================================
def build_jobs_from_matrices(machine_matrix, time_matrix):
    """
    Convert ma/pt into jobs list for GA.
    Stop per job when machine==0 or time==0 (padding).
    Machine stored as 0-based index internally.
    """
    jobs = []
    num_jobs, num_cols = time_matrix.shape
    for j in range(num_jobs):
        steps = []
        for s in range(num_cols):
            m = int(machine_matrix[j, s])
            t = float(time_matrix[j, s])
            if m == 0 or t == 0.0:
                break
            steps.append({"machine": m - 1, "duration": t})  # 0-based
        jobs.append(steps)
    return jobs


def ga_solve_and_schedule(machine_matrix, time_matrix,
                          POP_SIZE=100, MAX_GEN=200, CX_PROB=0.8, MUT_PROB=0.1,
                          ELITE_SIZE=2, TOURNAMENT_SIZE=5,
                          seed=0):
    random.seed(seed)
    np.random.seed(seed)

    jobs = build_jobs_from_matrices(machine_matrix, time_matrix)
    num_jobs = len(jobs)
    total_ops = sum(len(j) for j in jobs)

    class Chromosome:
        def __init__(self, genes):
            self.genes = genes  # [(job_id, step_id), ...]
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
                self._makespan = self._calculate_makespan()
            return self._makespan

        def _calculate_makespan(self):
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
        population = []
        for _ in range(POP_SIZE):
            genes = []
            job_progress = defaultdict(int)
            remaining = total_ops
            while remaining > 0:
                available_jobs = [j for j in range(num_jobs) if job_progress[j] < len(jobs[j])]
                jsel = random.choice(available_jobs)
                step = job_progress[jsel]
                genes.append((jsel, step))
                job_progress[jsel] += 1
                remaining -= 1
            population.append(Chromosome(genes))
        return population

    def strict_precedence_crossover(p1, p2):
        pool = defaultdict(list)
        for gene in p1.genes + p2.genes:
            pool[gene[0]].append(gene)

        child = []
        job_progress = defaultdict(int)
        while len(child) < total_ops:
            available_jobs = [j for j in range(num_jobs) if job_progress[j] < len(jobs[j])]
            jsel = random.choice(available_jobs)
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

    def tournament_selection(population):
        selected = []
        for _ in range(2):
            contestants = random.sample(population, TOURNAMENT_SIZE)
            contestants.sort(key=lambda x: x.makespan())
            selected.append(deepcopy(contestants[0]))
        return selected

    t0 = time.time()
    population = initialize_population()
    best = min(population, key=lambda x: x.makespan())
    for _ in range(MAX_GEN):
        population = [c.repair() for c in population]
        fitnesses = [c.makespan() if c.validate() else float("inf") for c in population]

        current_best = min(fitnesses)
        if current_best < best.makespan():
            best = deepcopy(population[int(np.argmin(fitnesses))])

        elite = sorted(population, key=lambda x: x.makespan())[:ELITE_SIZE]
        new_pop = elite.copy()
        while len(new_pop) < POP_SIZE:
            p1, p2 = tournament_selection(population)
            if random.random() < CX_PROB:
                child = strict_precedence_crossover(p1, p2)
            else:
                child = random.choice([p1, p2])
            child = enhanced_safe_mutation(child)
            new_pop.append(child)
        population = new_pop[:POP_SIZE]

    ga_time = time.time() - t0
    ga_makespan = float(best.makespan())

    machine_times = defaultdict(float)
    job_times = defaultdict(float)
    schedule = []
    for job_id, step_id in best.genes:
        machine0 = jobs[job_id][step_id]["machine"]     # 0-based
        dur = jobs[job_id][step_id]["duration"]
        start = max(job_times[job_id], machine_times[machine0])
        end = start + dur
        job_times[job_id] = end
        machine_times[machine0] = end
        schedule.append((job_id, step_id, machine0 + 1, start, end))  # machine_id => 1-based

    print(f"[GA] makespan={ga_makespan:.2f}, 用时={ga_time:.2f}s, valid={best.validate()}")
    return schedule, ga_makespan, float(ga_time)


# =========================================================
# Plotting: two gantts + bottom metrics/legend (unchanged content)
# =========================================================
def _draw_gantt_on_ax(ax, schedule, num_jobs, title, cmap, xlim_max):
    """
    schedule: list of (job, op, machine_id, start, end)
    machine_id is 1-based; ignore machine_id==0
    """
    sched = [t for t in schedule if int(t[2]) != 0 and (t[4] - t[3]) > 0]
    if not sched:
        ax.set_title(title + " (no tasks)")
        ax.set_xlim(0, xlim_max)
        ax.set_ylabel("Machine")
        ax.grid(True, linestyle='--', alpha=0.6)
        return

    machines = sorted(list(set([int(t[2]) for t in sched])))
    m2y = {m: i for i, m in enumerate(machines)}

    # ✅ 让行距更大，但保持“刻度=条”的严格对齐
    Y_GAP = 1.35

    # ✅ 条更粗
    BAR_H = 1.05

    buckets = defaultdict(list)
    for job, op, m, s, e in sched:
        buckets[int(m)].append((float(s), float(e - s), int(job), int(op)))

    for m, tasks in buckets.items():
        y = m2y[m] * Y_GAP
        tasks.sort(key=lambda x: x[0])
        for s, dur, j, op in tasks:
            ax.barh(
                y, dur,
                left=s,
                height=BAR_H,
                edgecolor='black',
                color=cmap(j)
            )

            # ✅ 标签更大；短条旋转避免重叠
            if dur >= 260:
                rot = 0
                fs = 20
            elif dur >= 180:
                rot = 20
                fs = 19
            else:
                rot = 60
                fs = 18

            ax.text(
                s + dur / 2, y,
                f"J{j+1}-O{op+1}",
                ha='center', va='center',
                fontsize=fs,
                rotation=rot,
                rotation_mode='anchor'
            )

    # ✅ 关键：刻度位置必须与 y 使用同一套 y_positions（否则就会你说的“对不上”）
    y_positions = [i * Y_GAP for i in range(len(machines))]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"M{m}" for m in machines])

    ax.set_title(title, pad=12)
    ax.set_xlabel("Time", labelpad=10)
    ax.set_ylabel("Machine", labelpad=10)
    ax.invert_yaxis()

    # 让上下不贴边（但不改变对齐关系）
    ax.margins(y=0.10)

    ax.set_xlim(0, xlim_max)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 再保险一次字体（只影响显示，不影响数据/legend）
    ax.title.set_fontsize(30)
    ax.xaxis.label.set_fontsize(26)
    ax.yaxis.label.set_fontsize(26)
    ax.tick_params(axis='both', labelsize=22)


def plot_compare_canvas(ga_schedule, dqn_schedule,
                        num_jobs,
                        ga_makespan, dqn_makespan,
                        ga_time_s, dqn_time_s,
                        out_png="ga_vs_dqn_compare.png",
                        suptitle="GA vs DQN (JSSP)"):
    cmap = plt.colormaps.get_cmap("tab20").resampled(num_jobs)

    xlim_max = max(ga_makespan, dqn_makespan) * 1.06 + 1e-6

    # ✅ 画布再放大 + 两张甘特图区域更高（metrics/legend 不动）
    fig = plt.figure(figsize=(36, 22))
    gs = gridspec.GridSpec(
        nrows=3, ncols=1,
        height_ratios=[1.55, 1.55, 0.60],
        hspace=0.40
    )

    ax_ga = fig.add_subplot(gs[0, 0])
    ax_dqn = fig.add_subplot(gs[1, 0], sharex=ax_ga)
    ax_info = fig.add_subplot(gs[2, 0])
    ax_info.axis("off")

    _draw_gantt_on_ax(
        ax_ga, ga_schedule, num_jobs,
        title=f"GA Gantt (makespan={ga_makespan:.1f})",
        cmap=cmap, xlim_max=xlim_max
    )
    _draw_gantt_on_ax(
        ax_dqn, dqn_schedule, num_jobs,
        title=f"DQN Gantt (makespan={dqn_makespan:.1f})",
        cmap=cmap, xlim_max=xlim_max
    )

    ratio = (dqn_makespan / ga_makespan) if ga_makespan > 0 else float("inf")

    # ===== 下方信息区（内容不动，只是字体更大）=====
    metrics_lines = [
        "=== Metrics ===",
        f"GA solve time     : {ga_time_s:.2f} s",
        f"DQN solve time    : {dqn_time_s:.2f} s",
        "",
        f"GA makespan       : {ga_makespan:.2f}",
        f"DQN makespan      : {dqn_makespan:.2f}",
        f"DQN / GA ratio    : {ratio:.4f}",
    ]

    ax_info.text(
        0.02, 0.92,
        "\n".join(metrics_lines),
        va="top", ha="left",
        fontsize=28
    )

    patches = [Patch(facecolor=cmap(j), edgecolor='black', label=f"Job {j+1}") for j in range(num_jobs)]
    ax_info.legend(
        handles=patches,
        loc="upper left",
        bbox_to_anchor=(0.55, 0.98),
        ncol=2 if num_jobs > 10 else 1,
        frameon=True,
        fontsize=26,
        title="Job Color",
        title_fontsize=28
    )

    fig.suptitle(suptitle, fontsize=34)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_png, dpi=260)
    plt.close(fig)
    print(f"[OK] 对比画布已保存: {out_png}")


# =========================================================
# Batch main
# =========================================================
if __name__ == "__main__":
    FOLDER = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    FILE_PREFIX = r"[6]r[17]c,"
    GENE_START = 0
    GENE_END = 120

    MODEL_PATH = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_1.pth"

    OUT_DIR = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\gantt_chart\complex_dqn_ga_1"
    os.makedirs(OUT_DIR, exist_ok=True)

    GA_PARAMS = dict(
        POP_SIZE=100, MAX_GEN=200, CX_PROB=0.8, MUT_PROB=0.1,
        ELITE_SIZE=2, TOURNAMENT_SIZE=5,
        seed=0
    )

    print(f"[Batch] gene range: {GENE_START}..{GENE_END}")
    print(f"[Batch] input folder: {FOLDER}")
    print(f"[Batch] output dir  : {OUT_DIR}")
    print(f"[Batch] model       : {MODEL_PATH}")

    success = 0
    failed = 0
    t_batch0 = time.time()

    for gene in range(GENE_START, GENE_END + 1):
        csv_name = f"{FILE_PREFIX}{gene}gene.csv"
        problem_file = os.path.join(FOLDER, csv_name)

        if not os.path.exists(problem_file):
            print(f"[SKIP] not found: {csv_name}")
            failed += 1
            continue

        print("\n" + "=" * 80)
        print(f"[RUN] gene={gene} file={csv_name}")

        try:
            ma, pt = load_single_csv(problem_file)
            if ma is None or pt is None:
                print(f"[FAIL] load failed: {csv_name}")
                failed += 1
                continue

            num_jobs = ma.shape[0]

            ga_sched, ga_ms, ga_t = ga_solve_and_schedule(ma, pt, **GA_PARAMS)

            dqn_sched, dqn_ms, dqn_t = solve_with_trained_model(
                ma, pt,
                model_path=MODEL_PATH,
                max_steps_factor=200
            )

            out_png = os.path.join(OUT_DIR, f"{gene}_ga_vs_wait_1.png")
            plot_compare_canvas(
                ga_sched, dqn_sched,
                num_jobs=num_jobs,
                ga_makespan=ga_ms, dqn_makespan=dqn_ms,
                ga_time_s=ga_t, dqn_time_s=dqn_t,
                out_png=out_png,
                suptitle=f"GA vs DQN (gene={gene}, file={csv_name})"
            )

            success += 1

        except Exception as e:
            print(f"[FAIL] gene={gene} error: {e}")
            failed += 1
            continue

    dt_batch = time.time() - t_batch0
    print("\n" + "=" * 80)
    print(f"[DONE] success={success}, failed={failed}, total={success+failed}, time={dt_batch:.2f}s")
