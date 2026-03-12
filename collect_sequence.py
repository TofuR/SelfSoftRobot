import numpy as np
import os
import time
from tqdm import tqdm
from elastica_env import ContinuousSoftArmEnv # 导入刚才添加的类

def generate_random_walk_actions(seq_length, dim=2, min_val=-0.005, max_val=0.005, step_size=0.001):
    """
    生成一个“随机游走”的动作序列，模拟连续变化的控制指令。
    这样数据之间是平滑过渡的，更符合物理实际。
    """
    actions = np.zeros((seq_length, dim))
    current_val = np.zeros(dim)
    
    for i in range(seq_length):
        # 随机扰动
        noise = np.random.uniform(-step_size, step_size, size=dim)
        current_val += noise
        
        # 裁剪到物理限制范围内
        current_val = np.clip(current_val, min_val, max_val)
        actions[i] = current_val
        
    return actions

def collect_continuous_data():
    """从 `ContinuousSoftArmEnv` 采集连续时序数据集。

    流程：
        1) 通过随机游走生成平滑动作序列；
        2) 连续推进仿真；
        3) 按固定间隔记录图像与动作；
        4) 保存为 `data/sequence_data/*.npz`。

    Returns:
        无返回值，结果写入磁盘文件。
    """
    # --- 配置参数 ---
    NUM_SEQUENCES = 10       # 采集几段独立的连续轨迹
    ACTIONS_PER_SEQ = 50    # 每段轨迹包含多少个不同的动作目标
    STEPS_PER_ACTION = 500  # 每个动作保持多少个仿真步 (0.05秒 @ 1e-4 dt)
    RECORD_INTERVAL = 50    # 每隔多少步录制一帧 (降采样，防止数据量过大)
    
    # 保存路径
    save_dir = "data/sequence_data"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f">>> 开始连续时间数据采集")
    print(f"    序列数: {NUM_SEQUENCES}, 动作数/序列: {ACTIONS_PER_SEQ}")
    print(f"    每动作仿真步数: {STEPS_PER_ACTION} (总时长: {ACTIONS_PER_SEQ * STEPS_PER_ACTION * 1e-4:.2f}s)")
    
    # 初始化环境 (只初始化一次！)
    env = ContinuousSoftArmEnv(dt=1e-4)
    
    total_frames = 0
    
    for seq_idx in range(NUM_SEQUENCES):
        print(f"\n--- 正在采集序列 {seq_idx + 1}/{NUM_SEQUENCES} ---")
        
        # 1. 生成一段平滑变化的动作指令
        action_schedule = generate_random_walk_actions(ACTIONS_PER_SEQ)
        
        seq_images = []
        seq_actions = []
        
        # 使用进度条显示当前序列进度
        pbar = tqdm(total=ACTIONS_PER_SEQ * STEPS_PER_ACTION)
        
        # 2. 执行动作序列
        for target_action in action_schedule:
            # 改变驱动力
            env.set_action(target_action)
            
            # 在这个力的作用下运行一段时间
            for _ in range(STEPS_PER_ACTION):
                # 往前跑一步
                env.step(steps=1)
                pbar.update(1)
                
                # 按照间隔录制数据
                if env.step_count % RECORD_INTERVAL == 0:
                    img, act = env.get_observation()
                    seq_images.append(img)
                    seq_actions.append(act)
        
        pbar.close()
        
        # 3. 保存该序列数据
        # 格式建议：每段轨迹保存为一个单独的文件，或者带上序列ID
        timestamp = int(time.time())
        filename = os.path.join(save_dir, f"seq_{seq_idx}_{timestamp}.npz")
        
        # 转换为 numpy 数组
        # image shape: (Time, H, W)
        # action shape: (Time, Action_Dim)
        np.savez_compressed(
            filename,
            images=np.array(seq_images),
            actions=np.array(seq_actions),
            dt=env.dt * RECORD_INTERVAL # 记录两帧之间的时间间隔
        )
        
        frames_count = len(seq_images)
        total_frames += frames_count
        print(f"    已保存序列 {seq_idx}: {frames_count} 帧 -> {filename}")

    print(f"\n>>> 采集全部完成！共采集 {total_frames} 帧数据。")

if __name__ == "__main__":
    collect_continuous_data()
