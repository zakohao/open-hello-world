import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os
import time
from collections import defaultdict

# 重新定义与训练代码一致的DQN模型
class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def plot_gantt_chart(schedule, num_jobs, all_machines, title="Gantt Chart", filename="new_gantt_chart.png"):
    if not schedule:
        print("无调度数据，无法绘制甘特图")
        return
        
    fig, ax = plt.subplots(figsize=(15, 10))
    color_map = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    
    # 过滤掉M0，仅保留大于0的机器
    filtered_machines = sorted([m for m in all_machines if m > 0])
    if not filtered_machines:
        print("没有可用的非零机器数据")
        return
        
    # 创建机器ID到y轴位置的映射（仅非零机器）
    machine_to_ypos = {machine: idx for idx, machine in enumerate(filtered_machines)}
    
    # 按机器和开始时间排序任务（忽略M0）
    machine_tasks = defaultdict(list)
    for task in schedule:
        job_id, op_index, machine_id, start, end = task
        if machine_id > 0:  # 仅处理非零机器
            duration = end - start
            machine_tasks[machine_id].append((start, duration, job_id, op_index))
    
    # 绘制每个机器上的任务
    for machine_id, tasks in machine_tasks.items():
        ypos = machine_to_ypos[machine_id]
        tasks.sort(key=lambda x: x[0])  # 按开始时间排序
        
        for start, duration, job_id, op_index in tasks:
            label = f"J{job_id+1}-O{op_index+1}"
            ax.barh(ypos, duration, left=start, height=0.6,
                    color=color_map(job_id), edgecolor='black')
            ax.text(start + duration / 2, ypos, label,
                    va='center', ha='center', color='black', fontsize=8)
    
    ax.set_xlabel("时间")
    ax.set_ylabel("机器")
    ax.set_title(title)
    
    # 设置y轴刻度和标签（仅非零机器）
    ax.set_yticks(range(len(filtered_machines)))
    ax.set_yticklabels([f"M{m}" for m in filtered_machines])
    
    # 反转y轴使标签从上往下增大
    ax.invert_yaxis()
    
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"甘特图已保存为 '{filename}'")

def parse_tuple(cell):
    """解析元组格式 ('1', '110.0')"""
    try:
        # 移除可能的空白字符和引号
        cell = cell.strip().replace("'", "").replace('"', "").replace("(", "").replace(")", "")
        
        # 分割字符串
        parts = cell.split(',')
        if len(parts) >= 2:
            machine = int(parts[0].strip())
            time = float(parts[1].strip())
            return machine, time
    except Exception as e:
        print(f"解析元组错误: {cell}, 错误: {e}")
    
    # 如果解析失败，返回(0, 0)表示无效操作
    return 0, 0.0

def load_single_csv(file_path):
    """从CSV文件加载数据并解析为机器分配矩阵和处理时间矩阵"""
    try:
        print(f"加载文件: {os.path.basename(file_path)}")
        
        # 尝试不同的分隔符
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            # 读取所有行
            lines = f.readlines()
            
            # 确定分隔符
            first_line = lines[0].strip()
            if '\t' in first_line:
                delimiter = '\t'
            elif ',' in first_line:
                delimiter = ','
            elif ';' in first_line:
                delimiter = ';'
            else:
                delimiter = None
                
            f.seek(0)
            reader = csv.reader(f, delimiter=delimiter) if delimiter else csv.reader(f)
            data = list(reader)
        
        # 获取标题行，确定工序数量
        header = data[0] if data else []
        num_processes = len(header) - 1 if header else 0  # 减去作业名列
        
        # 解析数据行
        machine_data = []
        time_data = []
        
        for row_idx, row in enumerate(data[1:]):
            # 跳过空行
            if not row or len(row) == 0:
                continue
            
            job_machines = []
            job_times = []
            
            # 跳过作业名列（第一列）
            for cell_idx, cell in enumerate(row[1:1+num_processes]):
                machine, time = parse_tuple(cell)
                
                # 只添加有效操作（机器号不为0）
                if machine != 0:
                    job_machines.append(machine)
                    job_times.append(time)
            
            # 只添加有有效操作的作业
            if job_machines:
                machine_data.append(job_machines)
                time_data.append(job_times)
        
        # 检查是否所有作业都被过滤掉了
        if not machine_data or not time_data:
            print(f"警告: 文件 {os.path.basename(file_path)} 没有有效操作")
            return None, None
        
        # 找到最大工序数量
        max_ops = max(len(ops) for ops in machine_data)
        
        # 填充矩阵使所有作业有相同数量的工序
        padded_machine_data = []
        padded_time_data = []
        
        for i in range(len(machine_data)):
            # 填充0表示无效操作
            padded_machine = machine_data[i] + [0] * (max_ops - len(machine_data[i]))
            padded_time = time_data[i] + [0.0] * (max_ops - len(time_data[i]))
            padded_machine_data.append(padded_machine)
            padded_time_data.append(padded_time)
        
        # 转换为NumPy数组
        machine_array = np.array(padded_machine_data, dtype=int)
        time_array = np.array(padded_time_data, dtype=float)
        
        # 检查数据有效性
        if machine_array.size == 0 or time_array.size == 0:
            print(f"文件 {os.path.basename(file_path)} 数据为空")
            return None, None
        if machine_array.shape != time_array.shape:
            print(f"文件 {os.path.basename(file_path)} 机器分配和处理时间形状不匹配")
            print(f"机器分配形状: {machine_array.shape}, 处理时间形状: {time_array.shape}")
            return None, None
        
        # 打印成功消息
        print(f"成功加载文件: {os.path.basename(file_path)}, 形状: {machine_array.shape}")
        print(f"有效作业数: {len(machine_data)}, 最大工序数: {max_ops}")
        print(f"机器分配示例:\n{machine_array[:2]}")
        print(f"处理时间示例:\n{time_array[:2]}")
        return machine_array, time_array
    
    except Exception as e:
        print(f"加载文件 {os.path.basename(file_path)} 时出错: {str(e)}")
        return None, None

class JSSPEnv:
    def __init__(self, machine_assignments, processing_times):
        self.machine_assignments = machine_assignments
        self.processing_times = processing_times
        self.num_jobs, self.num_machines = machine_assignments.shape
        
        # 获取所有机器ID并创建映射
        self.all_machines = np.unique(machine_assignments)
        self.machine_to_index = {machine: idx for idx, machine in enumerate(self.all_machines)}
        self.index_to_machine = {idx: machine for idx, machine in enumerate(self.all_machines)}
        
        # 计算每个作业的实际操作数
        self.actual_ops_per_job = []
        for job in range(self.num_jobs):
            valid_ops = 0
            for op in range(self.num_machines):
                if machine_assignments[job, op] != 0:
                    valid_ops += 1
            self.actual_ops_per_job.append(valid_ops)
        
        print(f"实际操作数: {self.actual_ops_per_job}")
        
        self.reset()
    
    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0] * len(self.all_machines)
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        
        # 初始化利用率相关变量
        self.machine_total_time = [0] * len(self.all_machines)
        self.machine_working_time = [0] * len(self.all_machines)
            
        return self._get_state()
    
    def _get_state(self):
        # 使用与训练代码相同的状态表示
        state = []
        
        # 1. 当前可执行操作的机器和处理时间
        for job in range(self.num_jobs):
            op = self.current_step[job]
            if op < self.num_machines:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.extend([machine, proc_time])
            else:
                state.extend([0, 0])
        
        # 2. 机器当前负载
        for machine_time in self.time_table:
            state.append(machine_time)
        
        # 3. 作业完成情况
        for job_end_time in self.job_end_times:
            state.append(job_end_time)
        
        # 4. 机器利用率信息
        for i in range(len(self.all_machines)):
            if self.machine_total_time[i] > 0:
                util = self.machine_working_time[i] / self.machine_total_time[i]
            else:
                util = 0
            state.append(util)
            
        return np.array(state, dtype=np.float32)
    
    def get_valid_actions(self):
        """返回当前可执行的动作（作业）列表"""
        valid_actions = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            # 如果作业还有工序未完成
            if op < self.actual_ops_per_job[job]:
                valid_actions.append(job)
        return valid_actions
    
    def step(self, action):
        job = action
        op = self.current_step[job]
        
        # 检查动作是否有效
        if self.done or op >= self.actual_ops_per_job[job] or job not in self.get_valid_actions():
            return self._get_state(), -10, self.done, {}
        
        machine_id = self.machine_assignments[job, op]
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time
        
        # 更新机器总时间和工作时间
        self.machine_total_time[machine_idx] = end_time
        self.machine_working_time[machine_idx] += proc_time
        
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        
        self.schedule.append((job, op, machine_id, start_time, end_time))
        
        self.current_step[job] += 1
        self.done = all(step >= self.actual_ops_per_job[i] for i, step in enumerate(self.current_step))
        
        # 计算机器利用率奖励
        utilization_reward = 0
        if self.machine_total_time[machine_idx] > 0:
            utilization = self.machine_working_time[machine_idx] / self.machine_total_time[machine_idx]
            utilization_reward = utilization * 5
        
        # 组合奖励：负的处理时间 + 机器利用率奖励
        reward = -proc_time + utilization_reward
        # 如果所有作业完成，添加最终奖励
        if self.done:
            makespan = max(self.job_end_times)
            reward += -makespan * 0.1
        
        return self._get_state(), reward, self.done, {
            "schedule": self.schedule,
            "machine_utilization": [self.machine_working_time[i] / self.machine_total_time[i] 
                                   if self.machine_total_time[i] > 0 else 0 
                                   for i in range(len(self.all_machines))]
        }

def solve_jssp(machine_assignments, processing_times, model_path, device='cpu'):
    if machine_assignments is None or processing_times is None:
        print("输入数据无效，无法求解")
        return None, None
        
    print(f"问题维度: {machine_assignments.shape[0]} 作业, {machine_assignments.shape[1]} 工序")
    
    # 创建环境
    env = JSSPEnv(machine_assignments, processing_times)
    state_dim = len(env.reset())
    action_dim = env.num_jobs
    
    print(f"状态维度: {state_dim}, 动作空间大小: {action_dim}")
    
    # 加载模型
    model = DQN(state_dim, action_dim)
    try:
        # 加载模型时忽略dropout层的参数（如果存在）
        state_dict = torch.load(model_path, map_location=device)
        
        # 处理可能的键名不匹配
        new_state_dict = {}
        for key, value in state_dict.items():
            # 重命名键以匹配当前模型结构
            if key == 'fc.6.weight':
                new_key = 'fc.8.weight'
            elif key == 'fc.6.bias':
                new_key = 'fc.8.bias'
            elif key == 'fc.3.weight':
                new_key = 'fc.4.weight'
            elif key == 'fc.3.bias':
                new_key = 'fc.4.bias'
            elif key == 'fc.4.weight':
                new_key = 'fc.6.weight'
            elif key == 'fc.4.bias':
                new_key = 'fc.6.bias'
            elif key == 'fc.8.weight':
                new_key = 'fc.3.weight'
            elif key == 'fc.8.bias':
                new_key = 'fc.3.bias'
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"模型加载成功，使用设备: {device}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 尝试直接加载（不重命名键）
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model.to(device)
            model.eval()
            print(f"使用宽松模式加载模型成功")
        except Exception as e2:
            print(f"宽松模式加载也失败: {e2}")
            return None, None
    
    # 使用训练好的模型求解
    state = env.reset()
    state_tensor = torch.FloatTensor(state).to(device)
    done = False
    schedule = []
    
    step_count = 0
    max_steps = 5000
    total_operations = sum(env.actual_ops_per_job)
    
    start_time = time.time()
    
    while not done and step_count < max_steps:
        step_count += 1
        
        with torch.no_grad():
            q_values = model(state_tensor)
            
            # 获取有效动作
            valid_actions = env.get_valid_actions()
            
            if not valid_actions:
                print("无有效动作可用")
                break
                
            # 只考虑有效动作
            valid_q_values = q_values.clone()
            for i in range(action_dim):
                if i not in valid_actions:
                    valid_q_values[i] = -float('inf')
            
            action = valid_q_values.argmax().item()
        
        next_state, reward, done, info = env.step(action)
        state_tensor = torch.FloatTensor(next_state).to(device)
        
        if 'schedule' in info:
            schedule = info['schedule']
        
        if step_count % 50 == 0:
            elapsed = time.time() - start_time
            completed_ops = sum(env.current_step)
            progress = completed_ops / total_operations
            ops_per_sec = step_count / elapsed if elapsed > 0 else 0
            print(f"步数 {step_count}: 完成 {completed_ops}/{total_operations} 操作 "
                  f"({progress*100:.1f}%), {ops_per_sec:.1f} 操作/秒")
        
        if done:
            break
    
    elapsed = time.time() - start_time
    
    if done:
        makespan = max(env.job_end_times)
        print(f"求解成功! 步数: {step_count}, 最大完工时间: {makespan}, 耗时: {elapsed:.2f}秒")
    else:
        completed_ops = sum(min(env.current_step[job], env.actual_ops_per_job[job]) for job in range(env.num_jobs))
        makespan = max(env.job_end_times)
        print(f"求解未完成! 步数: {step_count}, 完成 {completed_ops}/{total_operations} 操作")
    
    # 绘制甘特图
    if schedule:
        print(f"调度包含 {len(schedule)} 个任务")
        plot_gantt_chart(schedule, env.num_jobs, env.all_machines, title=f"JSSP_DQN (Makespan={makespan})")
    else:
        print("无调度数据可绘制")
    
    return schedule, makespan

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 使用训练好的模型求解新问题
    problem_file = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[10],odder_std_dev[2]\lot_mean[3],lot_std_dev[1]\machine[13]\seed[3]\[6]r[17]c,505gene.csv"
    
    # 选择模型
    model_path = 'new_GPU_LR0.0005_3000ep_brach128_jssp_model_npcb_(10,2)_(3,1)_(13)_(3).pth'  # 使用优化后的模型
    
    print(f"开始加载问题文件: {problem_file}")
    ma, pt = load_single_csv(problem_file)
    
    if ma is not None and pt is not None:
        print("成功加载问题数据，开始求解...")
        solve_jssp(ma, pt, model_path=model_path, device=device)
    else:
        print("无法加载问题数据")