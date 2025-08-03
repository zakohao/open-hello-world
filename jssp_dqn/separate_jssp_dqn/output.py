import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
import os
import time
from collections import defaultdict
from train import DQN, JSSPEnv  # 确保train.py中有DQN和JSSPEnv的定义

def plot_gantt_chart(schedule, num_jobs, all_machines, title="Gantt Chart", filename="gantt_chart.png"):
    if not schedule:
        print("无调度数据，无法绘制甘特图")
        return
        
    fig, ax = plt.subplots(figsize=(15, 10))
    color_map = plt.colormaps.get_cmap("tab20").resampled(num_jobs)
    
    # 创建机器ID到y轴位置的映射
    machine_to_ypos = {machine: idx for idx, machine in enumerate(sorted(all_machines))}
    
    # 按机器和开始时间排序任务
    machine_tasks = defaultdict(list)
    for task in schedule:
        job_id, op_index, machine_id, start, duration = task
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
    
    # 设置y轴刻度和标签
    sorted_machines = sorted(all_machines)
    ax.set_yticks(range(len(sorted_machines)))
    ax.set_yticklabels([f"M{m}" for m in sorted_machines])
    
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

class FixedJSSPEnv(JSSPEnv):
    """修复原始环境类的问题，处理0值操作"""
    def __init__(self, machine_assignments, processing_times):
        # 先计算每个作业的实际操作数
        num_jobs, num_machines = machine_assignments.shape
        self.actual_ops_per_job = []
        for job in range(num_jobs):
            valid_ops = 0
            for op in range(num_machines):
                if machine_assignments[job, op] != 0:
                    valid_ops += 1
            self.actual_ops_per_job.append(valid_ops)
        
        print(f"实际操作数: {self.actual_ops_per_job}")
        
        # 然后调用父类初始化
        super().__init__(machine_assignments, processing_times)
    
    def reset(self):
        self.current_step = [0] * self.num_jobs
        self.time_table = [0] * len(self.all_machines)
        self.job_end_times = [0] * self.num_jobs
        self.done = False
        self.schedule = []
        return self._get_state()
    
    def _get_state(self):
        state = []
        for job in range(self.num_jobs):
            op = self.current_step[job]
            # 只包含有效操作的状态
            if op < self.actual_ops_per_job[job]:
                machine = self.machine_assignments[job, op]
                proc_time = self.processing_times[job, op]
                state.append([machine, proc_time])
            else:
                state.append([0, 0])
        return np.array(state).flatten()
    
    def step(self, action):
        job = action
        if self.done:
            return self._get_state(), 0, self.done, self.schedule
        
        # 检查作业是否已完成
        if self.current_step[job] >= self.actual_ops_per_job[job]:
            # 作业已完成，跳过
            return self._get_state(), -1, self.done, self.schedule
        
        op = self.current_step[job]
        machine_id = self.machine_assignments[job, op]
        
        # 跳过无效操作（机器号为0）
        while machine_id == 0 and self.current_step[job] < self.num_machines - 1:
            self.current_step[job] += 1
            op = self.current_step[job]
            machine_id = self.machine_assignments[job, op]
        
        # 检查是否跳过所有操作
        if machine_id == 0 or self.current_step[job] >= self.actual_ops_per_job[job]:
            self.current_step[job] += 1
            self.done = all(step >= self.actual_ops_per_job[i] for i, step in enumerate(self.current_step))
            return self._get_state(), 0, self.done, self.schedule
        
        machine_idx = self.machine_to_index[machine_id]
        proc_time = self.processing_times[job, op]
        
        start_time = max(self.job_end_times[job], self.time_table[machine_idx])
        end_time = start_time + proc_time
        
        self.job_end_times[job] = end_time
        self.time_table[machine_idx] = end_time
        self.schedule.append((job, op, machine_id, start_time, proc_time))
        self.current_step[job] += 1
        
        # 检查所有作业是否完成
        self.done = all(step >= self.actual_ops_per_job[i] for i, step in enumerate(self.current_step))
        
        # 仅在完成时提供奖励
        reward = -end_time if self.done else 0
        return self._get_state(), reward, self.done, self.schedule

def solve_jssp(machine_assignments, processing_times, model_path='jssp_model.pth'): 
    if machine_assignments is None or processing_times is None:
        print("输入数据无效，无法求解")
        return None, None
        
    print(f"问题维度: {machine_assignments.shape[0]} 作业, {machine_assignments.shape[1]} 工序")
    
    # 创建修复后的环境
    env = FixedJSSPEnv(machine_assignments, processing_times)
    state_dim = len(env.reset())
    action_dim = env.num_jobs
    
    print(f"状态维度: {state_dim}, 动作空间大小: {action_dim}")
    
    # 加载模型
    model = DQN(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None
    
    # 使用训练好的模型求解
    state = env.reset()
    state_tensor = torch.FloatTensor(state)
    done = False
    schedule = []
    
    step_count = 0
    max_steps = 5000  # 合理的最大步数
    total_operations = sum(env.actual_ops_per_job)
    
    # 跟踪进度
    progress = []
    start_time = time.time()
    
    while not done and step_count < max_steps:
        step_count += 1
        
        with torch.no_grad():
            q_values = model(state_tensor)
            
            # 创建有效动作掩码（只选择未完成的作业）
            valid_actions = [job for job in range(env.num_jobs) 
                            if env.current_step[job] < env.actual_ops_per_job[job]]
            
            if not valid_actions:
                print("无有效动作可用")
                break
                
            # 只考虑有效动作
            valid_q_values = q_values[valid_actions]
            best_valid_idx = torch.argmax(valid_q_values).item()
            action = valid_actions[best_valid_idx]
        
        next_state, reward, done, current_schedule = env.step(action)
        state_tensor = torch.FloatTensor(next_state)
        schedule = current_schedule
        
        # 记录进度
        completed_ops = sum(env.current_step)
        progress.append(completed_ops / total_operations)
        
        if step_count % 100 == 0:
            elapsed = time.time() - start_time
            ops_per_sec = step_count / elapsed if elapsed > 0 else 0
            print(f"步数 {step_count}: 完成 {completed_ops}/{total_operations} 操作 "
                  f"({progress[-1]*100:.1f}%), {ops_per_sec:.1f} 操作/秒")
        
        if done:
            break
    
    elapsed = time.time() - start_time
    
    if done:
        makespan = max(env.job_end_times) if hasattr(env, 'job_end_times') and env.job_end_times else 0
        print(f"求解成功! 步数: {step_count}, 最大完工时间: {makespan}, 耗时: {elapsed:.2f}秒")
    else:
        completed_ops = sum(min(env.current_step[job], env.actual_ops_per_job[job]) for job in range(env.num_jobs))
        makespan = max(env.job_end_times) if hasattr(env, 'job_end_times') and env.job_end_times else 0
        print(f"求解未完成! 步数: {step_count}, 完成 {completed_ops}/{total_operations} 操作")
    
    # 绘制甘特图
    if schedule:
        print(f"调度包含 {len(schedule)} 个任务")
        plot_gantt_chart(schedule, env.num_jobs, env.all_machines, title=f"调度方案 (Makespan={makespan})")
    else:
        print("无调度数据可绘制")
    
    return schedule, makespan

if __name__ == "__main__":
    # 使用训练好的模型求解新问题
    problem_file = r"D:\pysrc\wang_data\jobset\normal Printed Circuit Board\odder_mean[20],odder_std_dev[5]\lot_mean[7],lot_std_dev[3]\machine[13]\seed[3]\[6]r[17]c,14gene.csv"
    
    print(f"开始加载问题文件: {problem_file}")
    ma, pt = load_single_csv(problem_file)
    
    if ma is not None and pt is not None:
        print("成功加载问题数据，开始求解...")
        solve_jssp(ma, pt)
    else:
        print("无法加载问题数据")