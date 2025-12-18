import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")


# ============================================================
# DQN（与训练一致）
# ============================================================
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
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


# ============================================================
# CSV 解析
# ============================================================
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
    data = None
    for delimiter in [',', '\t', ';']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                tmp = list(csv.reader(f, delimiter=delimiter))
            if len(tmp) > 1 and len(tmp[0]) > 1:
                data = tmp
                break
        except:
            continue

    if data is None:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = list(csv.reader(f))

    header = data[0] if data else []
    num_processes = len(header) - 1 if header else 0

    machine_data, time_data = [], []
    for row in data[1:]:
        if not row:
            continue
        jm, jt = [], []
        for cell in row[1:1+num_processes]:
            m, t = parse_tuple(cell)
            jm.append(m)
            jt.append(t)
        machine_data.append(jm)
        time_data.append(jt)

    ma = np.array(machine_data, dtype=int)
    pt = np.array(time_data, dtype=float)

    if ma.size == 0 or pt.size == 0 or ma.shape != pt.shape:
        return None, None

    return ma, pt


# ============================================================
# 环境（与你最新 train 的 JSSPEnv 对齐）
# ============================================================
class JSSPEnv:
    def __init__(self, ma, pt,
                 idle_weight=0.05,
                 miss_weight=3.0,
                 skip_weight=0.2,
                 immediate_bonus=0.5):
        self.ma = ma
        self.pt = pt
        self.num_jobs, self.num_machines = ma.shape

        self.all_machines = np.unique(ma)
        self.m2i = {m: i for i, m in enumerate(self.all_machines)}
        self.i2m = {i: m for m, i in self.m2i.items()}

        self.idle_weight = idle_weight
        self.miss_weight = miss_weight
        self.skip_weight = skip_weight
        self.immediate_bonus = immediate_bonus

        self.reset()

    def reset(self):
        self.cur = [0] * self.num_jobs
        self.mt = [0.0] * len(self.all_machines)
        self.jt = [0.0] * self.num_jobs
        self.done = False
        self.schedule = []
        return self._get_state()

    def _earliest_machine(self):
        idx = int(np.argmin(self.mt)) if self.mt else 0
        return idx, self.mt[idx] if self.mt else 0.0

    def get_valid_actions(self):
        valid = [0]
        if self.done:
            return valid
        m_idx, t_free = self._earliest_machine()
        m_id = self.i2m[m_idx]

        for j in range(self.num_jobs):
            op = self.cur[j]
            if op >= self.num_machines:
                continue
            if self.ma[j, op] != m_id:
                continue
            if self.jt[j] <= t_free + 1e-9:
                valid.append(j + 1)
        return valid

    def _get_state(self):
        s = []
        # 1) next op (machine, time)
        for j in range(self.num_jobs):
            op = self.cur[j]
            if op < self.num_machines:
                s.extend([float(self.ma[j, op]), float(self.pt[j, op])])
            else:
                s.extend([0.0, 0.0])

        # 2) machine time
        s.extend([float(x) for x in self.mt])

        # 3) job ready time
        s.extend([float(x) for x in self.jt])

        # 4) progress
        total_ops = self.num_jobs * self.num_machines
        s.append(float(sum(self.cur) / total_ops if total_ops > 0 else 0.0))

        # 5) relative load
        mx = max(self.mt) if self.mt else 0.0
        if mx > 0:
            s.extend([float(x / mx) for x in self.mt])
        else:
            s.extend([0.0] * len(self.mt))

        # 6) remaining work
        for j in range(self.num_jobs):
            op = self.cur[j]
            s.append(float(np.sum(self.pt[j, op:])) if op < self.num_machines else 0.0)

        # 7) earliest machine info
        m_idx, t_free = self._earliest_machine()
        s.append(float(m_idx))
        s.append(float(t_free))

        return np.array(s, dtype=np.float32)

    def _align_wait(self, m_idx, t_free):
        times = []
        m_id = self.i2m[m_idx]

        for j in range(self.num_jobs):
            op = self.cur[j]
            if op < self.num_machines and self.ma[j, op] == m_id:
                times.append(self.jt[j])

        for t in self.mt:
            if t > t_free + 1e-9:
                times.append(t)

        for t in self.jt:
            if t > t_free + 1e-9:
                times.append(t)

        if not times:
            self.mt[m_idx] += 1e-3
            return

        self.mt[m_idx] = min(times)

    def step(self, action):
        if self.done:
            return self._get_state(), 0.0, True, {"makespan": max(self.jt) if self.jt else 0.0}

        m_idx, t_free = self._earliest_machine()
        valid = self.get_valid_actions()
        cmax_prev = max(self.jt) if self.jt else 0.0

        # action=0: heuristic or wait
        if action == 0:
            if len(valid) > 1:
                action = min([a for a in valid if a != 0],
                             key=lambda a: self.pt[a-1, self.cur[a-1]])
            else:
                self._align_wait(m_idx, t_free)
                return self._get_state(), 0.0, False, {"makespan": None}

        job = action - 1
        op = self.cur[job]
        if op >= self.num_machines:
            self._align_wait(m_idx, t_free)
            return self._get_state(), 0.0, False, {"makespan": None}

        machine_id = self.ma[job, op]
        mi = self.m2i[machine_id]
        p = float(self.pt[job, op])

        st = max(self.jt[job], self.mt[mi])
        ed = st + p

        self.jt[job] = ed
        self.mt[mi] = ed
        self.cur[job] += 1
        self.schedule.append((job, op, machine_id, st, ed))
        self.done = all(x >= self.num_machines for x in self.cur)

        cmax_now = max(self.jt) if self.jt else 0.0
        return self._get_state(), -(cmax_now - cmax_prev), self.done, {"makespan": cmax_now if self.done else None}


# ============================================================
# Fallback（SPT + action=0 wait）
# ============================================================
def complete_schedule_with_fallback(env):
    steps = 0
    max_steps = env.num_jobs * env.num_machines * 20
    while not env.done and steps < max_steps:
        v = env.get_valid_actions()
        if not v:
            break
        if len(v) > 1:
            a = min([x for x in v if x != 0],
                    key=lambda x: env.pt[x-1, env.cur[x-1]])
        else:
            a = 0
        env.step(a)
        steps += 1
    return max(env.jt) if env.jt else float("inf")


# ============================================================
# 推理单个实例（统计 DQN 次数 / action0 次数 / fallback）
# ============================================================
def solve_one(model, ma, pt, env_params, max_steps_factor=50):
    env = JSSPEnv(ma, pt, **env_params)
    state = env.reset()

    steps = 0
    dqn_decisions = 0
    action0_count = 0
    fallback_used = False

    max_steps = env.num_jobs * env.num_machines * max_steps_factor
    t0 = time.time()

    info = {}
    while not env.done and steps < max_steps:
        valid = env.get_valid_actions()
        if not valid:
            break

        dqn_decisions += 1

        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            q = model(st)

            # ✅ 修复点：mask 必须是 bool
            mask = torch.zeros_like(q, dtype=torch.bool)
            mask[0, valid] = True

            q_masked = q.clone()
            q_masked[~mask] = -float("inf")
            action = int(q_masked.argmax().item())

        if action == 0:
            action0_count += 1

        state, _, _, info = env.step(action)
        steps += 1

    if not env.done:
        fallback_used = True
        makespan = complete_schedule_with_fallback(env)
    else:
        makespan = info.get("makespan", max(env.jt) if env.jt else 0.0)

    return {
        "makespan": float(makespan),
        "steps": int(steps),
        "dqn_decisions": int(dqn_decisions),
        "action0_count": int(action0_count),
        "fallback_used": bool(fallback_used),
        "solve_seconds": float(time.time() - t0),
    }


# ============================================================
# Batch
# ============================================================
def batch_solve(folder_path, model_path, out_csv,
                prefix="[6]r[17]c,", gene_start=0, gene_end=120):
    env_params = dict(miss_weight=3.0, idle_weight=0.05, skip_weight=0.2, immediate_bonus=0.5)

    # 1) 用第一个文件推导 state_dim/action_dim
    first_file = os.path.join(folder_path, f"{prefix}{gene_start}gene.csv")
    ma0, pt0 = load_single_csv(first_file)
    if ma0 is None:
        raise RuntimeError(f"首个数据读取失败: {first_file}")

    tmp_env = JSSPEnv(ma0, pt0, **env_params)
    state_dim = len(tmp_env.reset())
    action_dim = tmp_env.num_jobs + 1

    # 2) 只加载一次模型
    model = DQN(state_dim, action_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("从 checkpoint['model_state_dict'] 加载模型")
    else:
        model.load_state_dict(ckpt)
        print("从纯 state_dict 加载模型")
    model.eval()

    # 3) 批处理
    results = []
    for g in range(gene_start, gene_end + 1):
        fn = f"{prefix}{g}gene.csv"
        fp = os.path.join(folder_path, fn)

        row = {"gene": g, "filename": fn}
        try:
            ma, pt = load_single_csv(fp)
            if ma is None:
                raise RuntimeError("数据加载失败(空/shape不一致)")

            stat = solve_one(model, ma, pt, env_params, max_steps_factor=50)
            row.update(stat)

            print(f"[OK] {fn} mk={stat['makespan']:.1f} "
                  f"dqn={stat['dqn_decisions']} a0={stat['action0_count']} fb={stat['fallback_used']}")

        except Exception as e:
            row["error"] = str(e)
            print(f"[ERR] {fn} -> {e}")

        results.append(row)

    # 4) 输出 CSV
    # 统一字段顺序
    fieldnames = [
        "gene", "filename",
        "makespan", "steps", "dqn_decisions", "action0_count", "fallback_used", "solve_seconds",
        "error"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    print(f"\n✅ 结果已保存: {out_csv}")


# ============================================================
# Main（只改这里）
# ============================================================
if __name__ == "__main__":
    DATA_DIR = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"
    MODEL_PATH = r"D:\vscode\open-hello-world\jssp_dqn\separate_jssp_dqn\model\wait_v5_1.pth"
    OUT_CSV = "batch_wait_v5_results.csv"

    batch_solve(DATA_DIR, MODEL_PATH, OUT_CSV, prefix="[6]r[17]c,", gene_start=0, gene_end=120)
