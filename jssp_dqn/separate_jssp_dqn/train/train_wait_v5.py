import os
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from tqdm import tqdm

# 获取当前脚本所在的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 详细检查CUDA可用性
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("使用CPU")

# Hyperparameters
EPISODES = 1000
GAMMA = 0.9           # 越接近1表示越重视长期回报
LR = 0.0005            # 较小的学习率使训练更稳定但收敛较慢
EPSILON_DECAY = 0.998 # 控制从探索(随机选择)到利用(选择最优动作)的过渡速度
MIN_EPSILON = 0.01     # 最小探索率 越大越随机选择，越小越选择当前最适
BATCH_SIZE = 64       # 固定批量大小
MEMORY_SIZE = 100000   # 存储(state, action, reward, next_state)经验元组的个数
UPDATE_TARGET_FREQUENCY = 30  # 每UPDATE_TARGET_FREQUENCY步更新一次目标网络
MIN_COMPLETE_EPISODES = 2     # 完成MIN_COMPLETE_EPISODES个数据集的完整调度后才开始训练
EVALUATION_FREQUENCY = 10      # 每EVALUATION_FREQUENCY次episode评估一次固定测试集

class JSSPEnv:
    """
    改进点：
    1) 事件驱动(event-based)：每一步优先处理“最早空闲的机器”上的可加工工序
    2) 等待(action=0)不再是dt=1全局平移,而是“对齐等待”：直接跳到该机器下一次有工序可干的时刻
    3) 如果该机器当前有可加工工序,你却选了别的job(或等待),则立即惩罚(missed_dispath_penalty)
    4) Reward函数更新 -ΔCmax(makespan增量) + 行动的奖励以及惩罚
    """

    def __init__(self, machine_assignments, processing_times,
                 idle_weight=0.02,
                 miss_weight=2.0,
                 skip_weight=0.2,
                 immediate_bonus=0.5):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape

        # 获取所有机器ID并创建映射
        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}

        # reward参数
        self.idle_weight = idle_weight      # 强化机器待机的代价
        self.miss_weight = miss_weight      # “有可做任务但不做”的惩罚强度
        self.skip_weight = skip_weight      # action=0（交给启发式）时的小惩罚
        self.immediate_bonus = immediate_bonus      

        self.reset()

    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0.0] * len(self.all_machines)      # machine available time
        self.job_end_times = [0.0] * self.num_jobs            # job ready time
        self.done = False
        self.schedule = []
        self.total_processing_time = float(np.sum(self.processing_times))
        self.completed_ops = 0
        return self._get_state()

    # 事件驱动：选“最早空闲机器”
    def _earliest_machine(self):
        min_t = min(self.time_table) if self.time_table else 0.0
        # 多台机器同一最早时间时，固定选最小idx
        m_idx = int(np.argmin(self.time_table)) if self.time_table else 0
        return m_idx, min_t

    # 在“最早空闲机器”上，此刻可开工的job集合
    def get_valid_actions(self):
        """
        动作空间仍保持：0(等待/交给启发式) + 1..num_jobs (选job)
        但合法动作只允许：
        - 0 永远允许（表示对齐等待/或启发式选择）
        - job+1：该job下一道工序的机器 == 最早空闲机器 且 job已就绪(job_end<=machine_free)
        """
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

    # state（加入“最早机器信息”）
    def _get_state(self):
        state = []

        # 1) 每个job下一工序（machine_id, proc_time）
        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines:
                machine = self.machine_assignments[j, op]
                proc_time = self.processing_times[j, op]
                state.extend([float(machine), float(proc_time)])
            else:
                state.extend([0.0, 0.0])

        # 2) machine available time
        state.extend([float(x) for x in self.time_table])

        # 3) job ready time
        state.extend([float(x) for x in self.job_end_times])

        # 4) global progress
        total_ops = self.num_jobs * self.num_machines
        completed_ops = sum(self.current_step)
        progress = completed_ops / total_ops if total_ops > 0 else 0.0
        state.append(float(progress))

        # 5) machine relative load
        if self.time_table:
            mx = max(self.time_table)
            if mx > 0:
                state.extend([float(t / mx) for t in self.time_table])
            else:
                state.extend([0.0] * len(self.time_table))
        else:
            state.extend([0.0] * len(self.all_machines))

        # 6) remaining work per job
        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines:
                rem = float(np.sum(self.processing_times[j, op:]))
                state.append(rem)
            else:
                state.append(0.0)

        # 7) 加入“最早空闲机器信息”
        m_idx, t_free = self._earliest_machine()
        state.append(float(m_idx))
        state.append(float(t_free))

        return np.array(state, dtype=np.float32)


    # 对齐等待：跳到该机器下一次可能开工的时刻
    def _align_wait(self, m_idx, t_free):
        """
        强制时间推进到下一个“任何事件”：
        - 该机器可加工 job 就绪
        - 或 任意 job 完成上一道工序
        """
        times = []

        # 1) 该机器相关 job 的就绪时间
        m_id = self.index_to_machine[m_idx]
        for j in range(self.num_jobs):
            op = self.current_step[j]
            if op < self.num_machines and self.machine_assignments[j, op] == m_id:
                times.append(self.job_end_times[j])

        # 2) 其他机器未来完成时间（全局事件）
        for t in self.time_table:
            if t > t_free + 1e-9:
                times.append(t)

        # 3) 兜底：所有 job 的未来完成时间
        for t in self.job_end_times:
            if t > t_free + 1e-9:
                times.append(t)

        if not times:
            # 实在没事件，强制 +ε
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
        
        # anti-stall：记录执行前状态
        prev_signature = (
            tuple(self.current_step),
            tuple(self.time_table),
            tuple(self.job_end_times),
        )

        # 当前应关注的机器：最早空闲机器
        m_idx, t_free = self._earliest_machine()
        valid_actions = self.get_valid_actions()

        # 非法动作处理
        if action not in range(0, self.num_jobs + 1):
            action = 0

        # 计算当前Cmax
        cmax_prev = max(self.job_end_times) if self.job_end_times else 0.0

        # action == 0：等待 or 启发式替代
        if action == 0:
            # 情况1：其实有可调度 job（action=0 是在逃避决策）
            if len(valid_actions) > 1:
                # 明确惩罚
                missed_penalty = -self.miss_weight
                skip_penalty = -self.skip_weight

                # 用 SPT 启发式选一个 job（这是一次“真实调度”）
                best_action = min(
                    [a for a in valid_actions if a != 0],
                    key=lambda a: self.processing_times[a-1, self.current_step[a-1]]
                )

                action = best_action

            # 情况2：真的没可执行 job，只能等
            else:
                idle = self._align_wait(m_idx, t_free)
                idle_penalty = -self.idle_weight * idle
                reward = idle_penalty

                info = {
                    "schedule": self.schedule.copy(),
                    "makespan": None,
                    "reward_components": {
                        "idle_penalty": idle_penalty
                    }
                }
                return self._get_state(), reward, self.done, info

        else:
            missed_penalty = 0.0
            skip_penalty = 0.0
            # 如果此刻有“应该在这台最早空闲机器上开工”的job，但你选了别的job：惩罚
            if len(valid_actions) > 1 and action not in valid_actions:
                missed_penalty = -self.miss_weight

        # 执行选择的job（可能是启发式选择后的）
        job = action - 1
        op = self.current_step[job]

        # 安全：若选到已完成job，当作等待
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

        # 如果你选的不是最早空闲机器，而导致“最早机器”继续闲着，也给一点小惩罚
        # 注意：我们用“最早机器闲置到它下一次被更新”的信息不容易即时得到，所以用近似：
        extra_idle_penalty = 0.0
        if machine_idx != m_idx and len(valid_actions) > 1:
            # 有机会在最早机器上开工，但你跑去别的机器安排，视为浪费机会
            extra_idle_penalty = -0.5

        # 更新
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.current_step[job] += 1
        self.completed_ops += 1

        self.schedule.append((job, op, machine_id, start_time, end_time))
        self.done = all(s >= self.num_machines for s in self.current_step)

        # Reward：主项 -ΔCmax + 辅助惩罚/奖励
        cmax_now = max(self.job_end_times) if self.job_end_times else 0.0
        delta_cmax = cmax_now - cmax_prev

        # 主项：让Cmax增长越慢越好
        reward_main = -delta_cmax

        # 奖励：如果确实在“最早空闲机器”上立刻开工
        immediate_reward = 0.0
        if machine_idx == m_idx and start_time <= t_free + 1e-9:
            immediate_reward = self.immediate_bonus

        ## 若该机器本可以更早开工但你选的job要等很久，也惩罚等待（局部空转）
        #local_idle = max(0.0, start_time - self.time_table[machine_idx] + proc_time) 
        ## 更合理的 local idle：start_time - prev_machine_available
        ## 但我们没存 prev_machine_available，这里用轻惩罚即可
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

        # anti-stall：如果状态完全没变，强制推进时间
        new_signature = (
            tuple(self.current_step),
            tuple(self.time_table),
            tuple(self.job_end_times),
        )

        if new_signature == prev_signature:
            # 强制让系统时间前进一点（防止无限循环）
            self.time_table[m_idx] += 1e-2

        return self._get_state(), reward, self.done, info


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
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


def parse_tuple(cell):
    """专门解析元组格式 ('1', '110.0')"""
    try:
        cell = cell.strip().replace("'", "").replace('"', "")
        if '(' in cell and ')' in cell:
            content = cell[cell.find('(')+1:cell.find(')')]
            parts = content.split(',')
            if len(parts) >= 2:
                machine = int(parts[0].strip())
                time = float(parts[1].strip())
                return machine, time
    except Exception as e:
        pass
    return 0, 0.0


def load_single_csv(file_path):
    """从CSV文件加载数据并解析为机器分配矩阵和处理时间矩阵"""
    try:
        # 尝试不同的分隔符
        for delimiter in [',', '\t', ';']:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = [f.readline() for _ in range(5)]
                    f.seek(0)

                    reader = csv.reader(f, delimiter=delimiter)
                    data = list(reader)

                    if len(data) > 1 and len(data[0]) > 1:
                        print(f"使用分隔符 '{delimiter}' 读取文件: {os.path.basename(file_path)}")
                        break
            except:
                continue

        if 'data' not in locals():
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.reader(f)
                data = list(reader)
            print(f"使用默认分隔符读取文件: {os.path.basename(file_path)}")

        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0  # 减去作业名列

        machine_data = []
        time_data = []

        for row in data[1:]:
            if not row or len(row) == 0:
                continue

            job_machines = []
            job_times = []

            for cell in row[1:1+num_processes]:
                machine, time = parse_tuple(cell)
                job_machines.append(machine)
                job_times.append(time)

            machine_data.append(job_machines)
            time_data.append(job_times)

        machine_array = np.array(machine_data, dtype=int)
        time_array = np.array(time_data, dtype=float)

        if machine_array.size == 0 or time_array.size == 0:
            print(f"文件 {os.path.basename(file_path)} 数据为空")
            return None, None
        if machine_array.shape != time_array.shape:
            print(f"文件 {os.path.basename(file_path)} 机器分配和处理时间形状不匹配")
            return None, None

        print(f"成功加载文件: {os.path.basename(file_path)}, 形状: {machine_array.shape}")
        return machine_array, time_array

    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None, None


def load_folder_data(folder_path):
    """从文件夹加载所有CSV文件数据"""
    datasets = []
    valid_files = 0

    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 - {folder_path}")
        return datasets

    print(f"扫描文件夹: {folder_path}")
    print(f"找到文件: {len(os.listdir(folder_path))}")

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv') and 'gene' in filename.lower():
            file_path = os.path.join(folder_path, filename)

            machine_assignments, processing_times = load_single_csv(file_path)

            if machine_assignments is not None and processing_times is not None:
                datasets.append((machine_assignments, processing_times))
                valid_files += 1

    print(f"总共加载了 {valid_files} 个有效数据集")
    return datasets


def complete_schedule_with_fallback(env):
    """使用备用策略完成调度，确保所有任务都被调度，并返回 makespan 与额外 reward"""
    # 记录当前状态
    steps_after_fallback = 0
    max_fallback_steps = env.num_jobs * env.num_machines * 5

    total_reward = 0.0
    
    while not env.done and steps_after_fallback < max_fallback_steps:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break
            
        # 使用最短处理时间优先策略
        shortest_time = float('inf')
        best_action = valid_actions[0]
        
        for action in valid_actions:
            if action == 0:  # 避免等待动作
                continue
                
            job_idx = action - 1
            op = env.current_step[job_idx]
            if op < env.num_machines:
                proc_time = env.processing_times[job_idx, op]
                if proc_time < shortest_time:
                    shortest_time = proc_time
                    best_action = action
        
        # 如果所有非等待动作都不可用，才选择等待
        if best_action == valid_actions[0] and len(valid_actions) > 1:
            # 选择第一个非等待动作
            for action in valid_actions:
                if action != 0:
                    best_action = action
                    break
        
        _, reward, done, info = env.step(best_action)
        total_reward += reward
        steps_after_fallback += 1
    
    # 返回最终的makespan
    if env.done:
        return max(env.job_end_times), total_reward
    else:
        # 如果备用策略也无法完成，返回当前最大时间
        return max(env.job_end_times) if env.job_end_times else float('inf'), total_reward


def evaluate_model_on_datasets(model, datasets, device):
    """使用给定模型在多个数据集上评估，返回平均makespan，确保完成完整调度"""
    makespans = []
    total_rewards = []
    
    for i, (ma, pt) in enumerate(datasets):
        env = JSSPEnv(
            ma, pt,
            miss_weight=3.0,
            idle_weight=0.05,
            skip_weight=0.2,
            immediate_bonus=0.5
        )
        state = env.reset()
        steps = 0
        max_eval_steps = env.num_jobs * env.num_machines * 10  # 基于问题规模动态设置最大步数
        
        # 记录初始状态
        initial_ops = env.num_jobs * env.num_machines
        completed_ops = 0

        total_reward = 0.0
        
        while not env.done and steps < max_eval_steps:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                print(f"评估数据集 {i+1}: 无有效动作，提前结束")
                break
                
            # 使用贪婪策略（不探索）
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                
                # 检查Q值是否异常
                if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                    # 如果Q值异常，使用最短处理时间启发式
                    shortest_time = float('inf')
                    best_action = valid_actions[0]
                    for action in valid_actions:
                        if action == 0:  # 等待动作
                            continue
                        job_idx = action - 1
                        op = env.current_step[job_idx]
                        if op < env.num_machines:
                            proc_time = env.processing_times[job_idx, op]
                            if proc_time < shortest_time:
                                shortest_time = proc_time
                                best_action = action
                    action = best_action
                else:
                    # mask掉非法动作
                    mask = torch.ones_like(q_values, dtype=torch.bool)
                    for i_act in range(q_values.shape[1]):
                        if i_act not in valid_actions:
                            mask[0, i_act] = False
                    
                    q_values_masked = q_values.clone()
                    q_values_masked[~mask] = -float('inf')
                    action = q_values_masked.argmax().item()
            
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1
            
            # 跟踪进度
            current_ops = sum(env.current_step)
            if current_ops > completed_ops:
                completed_ops = current_ops
                # 每完成10%的进度打印一次
                if completed_ops % max(1, initial_ops // 10) == 0:
                    progress = completed_ops / initial_ops * 100
                    print(f"评估数据集 {i+1}: 进度 {progress:.1f}% ({completed_ops}/{initial_ops} 工序)")
        
        # 检查是否完成所有任务
        if env.done:
            makespan = info["makespan"]
            makespans.append(makespan)
            total_rewards.append(total_reward)
            #print(f"评估数据集 {i+1}: 成功完成调度，makespan = {makespan:.2f}, 步数 = {steps}")
        else:
            # 如果未完成，使用备用策略完成调度
            #print(f"评估数据集 {i+1}: 模型未能在{max_eval_steps}步内完成调度，使用备用策略...")
            makespan_fallback, reward_fallback = complete_schedule_with_fallback(env)
            makespans.append(makespan_fallback)
            total_rewards.append(total_reward + reward_fallback)
            #print(f"评估数据集 {i+1}: 备用策略完成调度，makespan = {makespan:.2f}")
    
    avg_makespan = np.mean(makespans) if makespans else float('inf')
    avg_reward = np.mean(total_rewards) if total_rewards else 0.0
    #print(f"评估完成: 平均makespan = {avg_makespan:.2f}")
    
    return avg_makespan, avg_reward


def plot_evaluation_results(evaluation_results, episode_losses, hyperparams):
    if not evaluation_results and not episode_losses:
        print("没有评估结果或loss值可绘制")
        return
        
    # 创建2×2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_tl = axes[0, 0]  # 左上：平均makespan
    ax_tr = axes[0, 1]  # 右上：平均总reward
    ax_bl = axes[1, 0]  # 左下：loss
    ax_br = axes[1, 1]  # 右下：超参数文本
    
    # 左上：评估结果（平均makespan）
    if evaluation_results:
        # evaluation_results: [(episode, avg_makespan, avg_reward), ...]
        episodes, makespans, rewards = zip(*evaluation_results)
        
        ax_tl.plot(episodes, makespans, linewidth=2, marker='o', markersize=4, label='Average Makespan')
        ax_tl.set_xlabel('Episode')
        ax_tl.set_ylabel('Average Makespan')
        ax_tl.set_title('Evaluation on Fixed Jobsets: Makespan')
        ax_tl.grid(True, alpha=0.3)
        ax_tl.legend()
    
    # 右上：平均总reward
        ax_tr.plot(episodes, rewards, linewidth=2, marker='^', markersize=4, label='Average Total Reward')
        ax_tr.set_xlabel('Episode')
        ax_tr.set_ylabel('Average Total Reward')
        ax_tr.set_title('Evaluation on Fixed Jobsets: Total Reward')
        ax_tr.grid(True, alpha=0.3)
        ax_tr.legend()
    else:
        ax_tl.set_visible(False)
        ax_tr.set_visible(False)

    # 左下：loss曲线 
    if episode_losses:
        episodes_loss, losses = zip(*episode_losses)
        ax_bl.plot(episodes_loss, losses, 'r-', linewidth=2, markersize=3, label='Loss')
        ax_bl.set_xlabel('Episode')
        ax_bl.set_ylabel('Loss')
        ax_bl.set_yscale('log')  # 使用对数坐标，因为loss值可能变化很大
        ax_bl.set_title('Training Loss')
        ax_bl.grid(True, alpha=0.3)
        
        if losses:
            avg_loss = np.mean(losses)
            ax_bl.axhline(y=avg_loss, linestyle='--', alpha=0.7, 
                          label=f'Average Loss: {avg_loss:.4f}')
        ax_bl.legend()
    else:
        ax_bl.set_visible(False)
    
    # 右下：超参数文本
    ax_br.axis('off')  
    hyperparam_text = "\n".join([f"{key}: {value}" for key, value in hyperparams.items()])
    ax_br.text(
        0.02, 0.98, hyperparam_text,
        fontsize=9,
        va='top', ha='left',
        transform=ax_br.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7)
    )
    ax_br.set_title('Hyperparameters', pad=10)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"训练结果图已保存为 'training_results.png'")


def train_target(training_datasets):
    if not training_datasets:
        print("没有可用的训练数据!")
        return None, None, None

    # 初始化环境，推导状态空间与动作空间
    sample_env = JSSPEnv(
        *training_datasets[0],
        miss_weight=3.0,
        idle_weight=0.05,
        skip_weight=0.2,
        immediate_bonus=0.5
    )
    state_dim = len(sample_env.reset())

    # 新动作空间: 等待(0) + 每个job
    action_dim = sample_env.num_jobs + 1

    print(f"状态维度: {state_dim}, 动作空间: {action_dim}")
    print(f"可用数据集数量: {len(training_datasets)}")

    # 创建主网络和目标网络
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = 1.0

    best_makespan = float('inf')
    best_schedule = None
    makespans = []

    total_steps = 0
    complete_datasets_count = 0

    # 选择5个固定测试集
    fixed_test_datasets = random.sample(training_datasets, min(5, len(training_datasets)))
    evaluation_results = []  # 存储评估结果 (episode_index, avg_makespan)
    episode_losses = []      # 存储每个episode的loss值 (episode_index, loss)
    
    print(f"选择了 {len(fixed_test_datasets)} 个固定测试集用于定期评估")

    print("第一阶段：收集经验")

    selected_datasets = random.sample(training_datasets, MIN_COMPLETE_EPISODES)
    print(f"已选择 {len(selected_datasets)} 个不同数据集用于第一阶段")

    pbar_stage1 = tqdm(total=MIN_COMPLETE_EPISODES, desc="收集经验进度")

    # 第一阶段：纯探索，跑完完整调度，把经验塞进 memory
    for i, (ma, pt) in enumerate(selected_datasets):
        env = JSSPEnv(
            ma, pt,
            miss_weight=3.0,
            idle_weight=0.05,
            skip_weight=0.2,
            immediate_bonus=0.5
        )
        state = env.reset()
        steps = 0
        max_steps_per_episode = 10000  # 添加最大步数限制

        while not env.done and steps < max_steps_per_episode:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # 纯探索：随机合法动作（包括等待0）
            action = random.choice(valid_actions)

            next_state, reward, done, info = env.step(action)

            memory.append((state, action, reward, next_state, done))

            state = next_state
            steps += 1
            total_steps += 1

        # 检查是否达到最大步数
        if steps >= max_steps_per_episode:
            print(f"数据集 {i+1}: 达到最大步数限制 {max_steps_per_episode}")

        if env.done:
            complete_datasets_count += 1
            makespan = info["makespan"]
            makespans.append(makespan)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = info["schedule"].copy()

            pbar_stage1.update(1)
            pbar_stage1.set_postfix({
                '当前数据集': f'{i+1}/{MIN_COMPLETE_EPISODES}',
                'best_makespan': f'{best_makespan:.1f}',
                'memory_size': len(memory)
            })

    pbar_stage1.close()
    print(f"第一阶段完成！使用了 {len(selected_datasets)} 个不同数据集")
    print("开始第二阶段：训练模型...")

    # 第二阶段：训练
    progress_bar = tqdm(range(EPISODES), desc="训练进度")

    stage2_used_datasets = set()

    for ep in progress_bar:
        ma, pt = random.choice(training_datasets)

        dataset_hash = hash((ma.tobytes(), pt.tobytes()))
        stage2_used_datasets.add(dataset_hash)

        env = JSSPEnv(
            ma, pt,
            miss_weight=3.0,
            idle_weight=0.05,
            skip_weight=0.2,
            immediate_bonus=0.5
        )
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps_per_episode = 10000  # 添加最大步数限制
        
        # 记录当前episode的loss
        episode_loss_values = []

        while not env.done and steps < max_steps_per_episode:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            # ε-贪婪
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)  # shape [1, action_dim]

                    # 检查Q值是否异常
                    if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                        print(f"Episode {ep}: Q值异常，使用随机动作")
                        action = random.choice(valid_actions)
                    else:
                        # mask掉非法动作（包括那些不在valid_actions中的动作索引）
                        mask = torch.ones_like(q_values, dtype=torch.bool)
                        for i_act in range(action_dim):
                            if i_act not in valid_actions:
                                mask[0, i_act] = False

                        q_values_masked = q_values.clone()
                        q_values_masked[~mask] = -float('inf')
                        action = q_values_masked.argmax().item()

            next_state, reward, done, info = env.step(action)

            # 检查状态是否有效
            if np.isnan(state).any() or np.isinf(state).any():
                print(f"Episode {ep}: 状态包含NaN或Inf，跳过经验存储")
            else:
                memory.append((state, action, reward, next_state, done))

            state = next_state
            total_reward += reward
            steps += 1
            total_steps += 1

            # 经验回放训练
            if len(memory) >= BATCH_SIZE and len(memory) >= BATCH_SIZE:
                try:
                    batch = random.sample(memory, BATCH_SIZE)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    # 检查批次数据
                    states_array = np.array(states)
                    next_states_array = np.array(next_states)
                    
                    if (np.isnan(states_array).any() or np.isinf(states_array).any() or
                        np.isnan(next_states_array).any() or np.isinf(next_states_array).any()):
                        print(f"Episode {ep}: 批次数据包含NaN或Inf,跳过训练")
                        continue

                    states_tensor = torch.FloatTensor(states_array).to(device)
                    next_states_tensor = torch.FloatTensor(next_states_array).to(device)
                    actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                    dones_tensor = torch.BoolTensor(dones).unsqueeze(1).to(device)

                    # 前向传播
                    current_q_all = model(states_tensor)
                    current_q = current_q_all.gather(1, actions_tensor)

                    # 目标网络前向传播
                    with torch.no_grad():
                        next_q_all = target_model(next_states_tensor)
                        next_q = next_q_all.max(1)[0].unsqueeze(1)
                        target_q = rewards_tensor + GAMMA * next_q * (~dones_tensor)

                    # 检查损失
                    loss = criterion(current_q, target_q.detach())
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Episode {ep}: 损失为NaN或Inf，跳过梯度更新")
                        continue

                    # 逆传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 检查梯度
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    
                    if total_norm > 1000:  # 梯度爆炸
                        print(f"Episode {ep}: 梯度爆炸 (norm={total_norm:.2f})，跳过更新")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    
                    # 记录loss值
                    episode_loss_values.append(loss.item())

                except Exception as e:
                    print(f"Episode {ep}: 训练过程中出错: {str(e)}")
                    continue

            # 同步目标网络
            if total_steps % UPDATE_TARGET_FREQUENCY == 0:
                target_model.load_state_dict(model.state_dict())

        # 检查是否达到最大步数
        if steps >= max_steps_per_episode:
            print(f"Episode {ep}: 达到最大步数限制 {max_steps_per_episode}")
        
        # 记录当前episode的平均loss
        if episode_loss_values:
            avg_episode_loss = np.mean(episode_loss_values)
            episode_losses.append((ep + 1, avg_episode_loss))
            # 在进度条中显示loss
            progress_bar.set_postfix({
                'epsilon': f'{epsilon:.3f}',
                'best_makespan': f'{best_makespan:.1f}',
                'avg_loss': f'{avg_episode_loss:.4f}',
                'episode_steps': steps,
                'memory_size': len(memory),
                'stage2_datasets': len(stage2_used_datasets)
            })
        else:
            progress_bar.set_postfix({
                'epsilon': f'{epsilon:.3f}',
                'best_makespan': f'{best_makespan:.1f}',
                'avg_loss': 'N/A',
                'episode_steps': steps,
                'memory_size': len(memory),
                'stage2_datasets': len(stage2_used_datasets)
            })

        # 回合结束记录
        if env.done:
            complete_datasets_count += 1
            makespan = info["makespan"]
            makespans.append(makespan)
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedule = info["schedule"].copy()

        # epsilon 衰减
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        # 每10个episode评估一次固定测试集
        if (ep + 1) % EVALUATION_FREQUENCY == 0:
            try:
                #print(f"开始评估固定测试集...")
                avg_makespan_test, avg_reward_test = evaluate_model_on_datasets(target_model, fixed_test_datasets, device)
                evaluation_results.append((ep + 1, avg_makespan_test, avg_reward_test))
                #print(f"Episode {ep+1}: 固定测试集平均makespan = {avg_makespan_test:.2f}")
            except Exception as e:
                print(f"评估过程中出错: {str(e)}")
                # 如果评估失败，使用一个默认值
                evaluation_results.append((ep + 1, float('inf'), 0.0))

        avg_makespan = np.mean(makespans[-100:]) if makespans else 0
        
        if 'avg_episode_loss' in locals() and episode_loss_values:
            progress_bar.set_postfix({
                'epsilon': f'{epsilon:.3f}',
                'avg_makespan': f'{avg_makespan:.1f}',
                'avg_loss': f'{avg_episode_loss:.4f}',
                'episode_steps': steps,
                'memory_size': len(memory),
                'stage2_datasets': len(stage2_used_datasets)
            })
        else:
            progress_bar.set_postfix({
                'epsilon': f'{epsilon:.3f}',
                'avg_makespan': f'{avg_makespan:.1f}',
                'avg_loss': 'N/A',
                'episode_steps': steps,
                'memory_size': len(memory),
                'stage2_datasets': len(stage2_used_datasets)
            })

    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'dynamic_jssp_model_with_target.pth')

    print(f"最佳makespan: {best_makespan}")
    print(f"第一阶段使用了 {MIN_COMPLETE_EPISODES} 个不同数据集")
    print(f"第二阶段使用了 {len(stage2_used_datasets)} 个不同数据集")
    print(f"总共完成了 {complete_datasets_count} 个完整数据集调度")
        
    # 绘制评估平均makespan结果和loss值图
    plot_evaluation_results(evaluation_results, episode_losses, {
        'EPISODES': EPISODES,
        'GAMMA': GAMMA,
        'LR': LR,
        'EPSILON_DECAY': EPSILON_DECAY,
        'MIN_EPSILON': MIN_EPSILON,
        'BATCH_SIZE': BATCH_SIZE,
        'MEMORY_SIZE': MEMORY_SIZE,
        'UPDATE_TARGET_FREQUENCY': UPDATE_TARGET_FREQUENCY,
        'MIN_COMPLETE_EPISODES': MIN_COMPLETE_EPISODES,
        'EVALUATION_FREQUENCY': EVALUATION_FREQUENCY
    })

    return model, target_model, best_schedule


if __name__ == "__main__":
    print("开始训练JSSP模型")
    folder_path = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]"

    # 加载训练数据
    training_datasets = load_folder_data(folder_path)

    # 训练模型
    if training_datasets:
        model, target_model, best_schedule = train_target(training_datasets)
        print("训练完成！")

        # 测试最佳调度
        if best_schedule:
            makespan = max([end for _, _, _, _, end in best_schedule])
            print(f"最佳调度方案 (makespan: {makespan:.1f})")
    else:
        print("未找到有效的训练数据")