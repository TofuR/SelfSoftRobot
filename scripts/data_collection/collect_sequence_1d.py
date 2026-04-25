"""collect_sequence_1d.py — 单维度动作序列数据采集。

只改变第一个动作维度（垂直于相机平面的弯曲），第二个维度恒为 0。
保存到 data/sequence_data_1d/。

用法:
  python scripts/data_collection/collect_sequence_1d.py
"""

import numpy as np
import os
import time
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from elastica_env import ContinuousSoftArmEnv


def generate_1d_random_walk(seq_length, min_val=-0.005, max_val=0.005, step_size=0.001):
    """生成单维度随机游走动作序列。第二个维度恒为 0。"""
    actions = np.zeros((seq_length, 2))
    current_val = 0.0

    for i in range(seq_length):
        current_val += np.random.uniform(-step_size, step_size)
        current_val = np.clip(current_val, min_val, max_val)
        actions[i, 0] = current_val
        # actions[i, 1] 保持 0

    return actions


def collect_1d_data(
    num_sequences=10,
    actions_per_seq=50,
    steps_per_action=500,
    record_interval=50,
    save_dir="data/sequence_data_1d",
):
    os.makedirs(save_dir, exist_ok=True)

    print(f">>> 开始 1D 动作数据采集")
    print(f"    序列数: {num_sequences}, 动作数/序列: {actions_per_seq}")
    print(f"    只改变 dim-0, dim-1 恒为 0")

    env = ContinuousSoftArmEnv(dt=1e-4)
    total_frames = 0

    for seq_idx in range(num_sequences):
        print(f"\n--- 序列 {seq_idx + 1}/{num_sequences} ---")

        action_schedule = generate_1d_random_walk(actions_per_seq)

        seq_images = []
        seq_actions = []
        pbar = tqdm(total=actions_per_seq * steps_per_action)

        for target_action in action_schedule:
            env.set_action(target_action)

            for _ in range(steps_per_action):
                env.step(steps=1)
                pbar.update(1)

                if env.step_count % record_interval == 0:
                    img, act = env.get_observation()
                    seq_images.append(img)
                    seq_actions.append(act)

        pbar.close()

        filename = os.path.join(save_dir, f"seq_1d_{seq_idx}.npz")
        np.savez_compressed(
            filename,
            images=np.array(seq_images),
            actions=np.array(seq_actions),
            dt=env.dt * record_interval,
        )

        frames_count = len(seq_images)
        total_frames += frames_count
        print(f"    已保存: {frames_count} 帧 -> {filename}")

    print(f"\n>>> 采集完成！共 {total_frames} 帧 -> {save_dir}/")


if __name__ == "__main__":
    collect_1d_data()
