"""collect_canonical.py — 采集零动作（静止态）数据用于 Canonical Field 训练。

用法:
  python scripts/data_collection/collect_canonical.py
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from elastica_env import ContinuousSoftArmEnv


def collect_canonical_data(
    n_sequences=5,
    steps_per_seq=2000,
    record_interval=50,
    warmup_steps=2000,
    save_dir="data/canonical_data",
):
    """采集零动作数据。

    机器人在零扭矩下保持静止，采集多段数据（格式与 sequence_data 一致）。
    SoftSequenceDataset 可以直接加载。

    Args:
        n_sequences: 采集段数（每段保存为一个 .npz）。
        steps_per_seq: 每段采集的仿真步数。
        record_interval: 录制间隔（步）。
        warmup_steps: 首次采集前的预热步数，确保杆稳定。
        save_dir: 保存目录。
    """
    os.makedirs(save_dir, exist_ok=True)

    env = ContinuousSoftArmEnv(dt=1e-4)

    # 预热：让杆在零扭矩下充分稳定
    print(f">>> 预热 {warmup_steps} 步 ({warmup_steps * 1e-4:.2f}s)...")
    env.set_action(np.array([0.0, 0.0]))
    for _ in range(warmup_steps):
        env.step(steps=1)

    total_frames = 0

    for seq_idx in range(n_sequences):
        seq_images = []
        seq_actions = []

        # 重置到稳定状态（每段重新创建环境确保独立）
        if seq_idx > 0:
            env = ContinuousSoftArmEnv(dt=1e-4)
            env.set_action(np.array([0.0, 0.0]))
            for _ in range(warmup_steps):
                env.step(steps=1)

        for step in range(steps_per_seq):
            env.step(steps=1)
            if step % record_interval == 0:
                img, act = env.get_observation()
                seq_images.append(img)
                seq_actions.append(act)

        filename = os.path.join(save_dir, f"canonical_{seq_idx}.npz")
        np.savez_compressed(
            filename,
            images=np.array(seq_images),
            actions=np.array(seq_actions),
            dt=env.dt * record_interval,
        )

        frames_count = len(seq_images)
        total_frames += frames_count
        print(f"  序列 {seq_idx}: {frames_count} 帧 -> {filename}")

    print(f"\n>>> 采集完成！共 {total_frames} 帧 -> {save_dir}/")


if __name__ == "__main__":
    collect_canonical_data()
