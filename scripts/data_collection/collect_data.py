"""
collect_data.py — 统一数据采集入口。

支持两种模式：
  batch     — 每次独立仿真，适用于静态数据采集
  sequence  — 连续仿真 + 随机游走动作序列，适用于时序数据采集

用法:
  python collect_data.py --mode batch --count 100
  python collect_data.py --mode sequence --sequences 10 --actions-per-seq 50
"""

import sys
import os
import argparse
import time
import numpy as np
from tqdm import tqdm

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from elastica_env import (
    ContinuousSoftArmEnv,
    get_simulation_data_pair,
    CAMERA_EYE,
    CAMERA_CENTER,
    CAMERA_UP,
)


def generate_random_walk_actions(seq_length, action_dim=2,
                                 min_val=-0.005, max_val=0.005,
                                 step_size=0.001):
    """生成平滑随机游走动作序列。

    Args:
        seq_length: 序列长度。
        action_dim: 动作维度。
        min_val: 动作下限。
        max_val: 动作上限。
        step_size: 每步最大扰动量。

    Returns:
        动作数组，形状 (seq_length, action_dim)。
    """
    actions = np.zeros((seq_length, action_dim))
    current_val = np.zeros(action_dim)

    for i in range(seq_length):
        noise = np.random.uniform(-step_size, step_size, size=action_dim)
        current_val = np.clip(current_val + noise, min_val, max_val)
        actions[i] = current_val

    return actions


def generate_random_actions(count, action_dim=2, min_val=-0.005, max_val=0.005):
    """生成均匀随机动作。

    Args:
        count: 样本数量。
        action_dim: 动作维度。
        min_val: 动作下限。
        max_val: 动作上限。

    Returns:
        动作数组，形状 (count, action_dim)。
    """
    return np.random.uniform(min_val, max_val, size=(count, action_dim))


# =============================================================================
# Batch 模式
# =============================================================================

def collect_batch(count, action_dim, action_range, save_dir, image_size):
    """静态批量采集：每次仿真独立创建。

    Args:
        count: 采集样本数。
        action_dim: 动作维度。
        action_range: (min_val, max_val) 元组。
        save_dir: 保存目录。
        image_size: 渲染图像尺寸 (W, H)。
    """
    print(f"\n>>> [Batch 模式] 开始采集 {count} 组数据")

    actions_list = generate_random_actions(
        count, action_dim, action_range[0], action_range[1],
    )

    images_list = []
    angles_list = []

    for i, params in enumerate(tqdm(actions_list, desc="采集进度")):
        _, binary_img = get_simulation_data_pair(params, verbose=False, visualize=False)
        images_list.append(binary_img)
        angles_list.append(params)

        if (i + 1) % 10 == 0:
            print(f"  已采集: {i + 1}/{count}")

    images = np.array(images_list)
    actions = np.array(angles_list)

    timestamp = int(time.time())
    filename = os.path.join(save_dir, f"batch_{count}_{timestamp}.npz")
    np.savez_compressed(
        filename,
        images=images,
        actions=actions,
        focal=1.0,
        dt=1.0,
        camera_eye=np.array(CAMERA_EYE),
        camera_center=np.array(CAMERA_CENTER),
        camera_up=np.array(CAMERA_UP),
    )

    print(f"\n>>> Batch 采集完成！")
    print(f"    文件: {filename}")
    print(f"    Images: {images.shape}, Actions: {actions.shape}")


# =============================================================================
# Sequence 模式
# =============================================================================

def collect_sequence(num_sequences, actions_per_seq, steps_per_action,
                     record_interval, action_dim, action_range, save_dir):
    """连续时序采集：保持仿真状态推进。

    Args:
        num_sequences: 独立轨迹数量。
        actions_per_seq: 每段轨迹的动作目标数。
        steps_per_action: 每个动作的仿真步数。
        record_interval: 录制间隔（步数）。
        action_dim: 动作维度。
        action_range: (min_val, max_val) 元组。
        save_dir: 保存目录。
    """
    dt = 1e-4
    total_frames = 0

    print(f"\n>>> [Sequence 模式] 开始连续时序数据采集")
    print(f"    序列数: {num_sequences}, 动作数/序列: {actions_per_seq}")
    print(f"    每动作仿真步数: {steps_per_action} "
          f"(总时长: {actions_per_seq * steps_per_action * dt:.2f}s)")

    env = ContinuousSoftArmEnv(dt=dt)

    for seq_idx in range(num_sequences):
        print(f"\n--- 序列 {seq_idx + 1}/{num_sequences} ---")

        action_schedule = generate_random_walk_actions(
            actions_per_seq, action_dim,
            min_val=action_range[0], max_val=action_range[1],
        )

        seq_images = []
        seq_actions = []

        pbar = tqdm(total=actions_per_seq * steps_per_action,
                     desc=f"Seq {seq_idx + 1}")

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

        timestamp = int(time.time())
        filename = os.path.join(save_dir, f"seq_{seq_idx}_{timestamp}.npz")

        np.savez_compressed(
            filename,
            images=np.array(seq_images),
            actions=np.array(seq_actions),
            dt=dt * record_interval,
            focal=1.0,
            camera_eye=np.array(CAMERA_EYE),
            camera_center=np.array(CAMERA_CENTER),
            camera_up=np.array(CAMERA_UP),
        )

        frames_count = len(seq_images)
        total_frames += frames_count
        print(f"    保存: {frames_count} 帧 -> {filename}")

    print(f"\n>>> Sequence 采集完成！共 {total_frames} 帧。")


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="软体机器人数据采集工具")
    parser.add_argument("--mode", choices=["batch", "sequence"], required=True,
                        help="采集模式：batch（静态批量）或 sequence（连续时序）")

    # 通用参数
    parser.add_argument("--save-dir", default="data/sequence_data",
                        help="数据保存目录")
    parser.add_argument("--action-dim", type=int, default=2,
                        help="动作维度（驱动参数数量）")
    parser.add_argument("--action-min", type=float, default=-0.005,
                        help="动作下限")
    parser.add_argument("--action-max", type=float, default=0.005,
                        help="动作上限")

    # Batch 模式参数
    parser.add_argument("--count", type=int, default=100,
                        help="[batch] 采集样本数")

    # Sequence 模式参数
    parser.add_argument("--sequences", type=int, default=10,
                        help="[sequence] 独立轨迹数量")
    parser.add_argument("--actions-per-seq", type=int, default=50,
                        help="[sequence] 每段轨迹的动作目标数")
    parser.add_argument("--steps-per-action", type=int, default=500,
                        help="[sequence] 每个动作的仿真步数")
    parser.add_argument("--record-interval", type=int, default=50,
                        help="[sequence] 录制间隔（步数）")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    action_range = (args.action_min, args.action_max)

    if args.mode == "batch":
        collect_batch(args.count, args.action_dim, action_range,
                      args.save_dir, image_size=(100, 100))
    else:
        collect_sequence(
            args.sequences, args.actions_per_seq, args.steps_per_action,
            args.record_interval, args.action_dim, action_range, args.save_dir,
        )


if __name__ == "__main__":
    main()
