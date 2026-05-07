"""collect_multiview.py — 多视角数据采集。

采集双视角（正面+侧面）图像，同时保存 3D GT 用于评估。
数据格式:
  npz:
    images_front, images_side, actions, positions (可选)
    depth_maps_front, depth_maps_side (可选，--depth 时)
    camera_*_front, camera_*_side, focal, H, W

用法:
    python scripts/data_collection/collect_multiview.py
    python scripts/data_collection/collect_multiview.py --sequences 5 --save-dir data/multiview_rr
    python scripts/data_collection/collect_multiview.py --depth  # 含深度图
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from elastica_env import (
    ContinuousSoftArmEnv, CAMERA_EYE, CAMERA_CENTER, CAMERA_UP,
    CAMERA_EYE_SIDE, CAMERA_CENTER_SIDE, CAMERA_UP_SIDE,
    DEFAULT_IMAGE_SIZE, render_to_binary_with_depth,
)
from collect_utils import ActionSchedule, load_defaults


def run_multiview_collection(schedule, args, defaults):
    cam = defaults["camera"]
    dt = defaults["dt"]
    save_dir = args.save_dir
    record_depth = args.record_depth
    os.makedirs(save_dir, exist_ok=True)

    n_seqs = args.sequences
    actions_per_seq = args.actions_per_seq
    steps_per_action = args.steps_per_action
    record_interval = args.record_interval
    warmup_steps = args.warmup_steps

    total_dur = actions_per_seq * steps_per_action * dt
    print(f"\n>>> 多视角采集: [{schedule.mode_tag}]" + (" + Depth" if record_depth else ""))
    print(f"    序列: {n_seqs}, 动作/序列: {actions_per_seq}, 时长: {total_dur:.2f}s")
    print(f"    正面: eye={CAMERA_EYE}")
    print(f"    侧面: eye={CAMERA_EYE_SIDE}")
    print(f"    保存: {save_dir}")

    env = ContinuousSoftArmEnv(dt=dt)

    print(f"    预热 {warmup_steps} 步...")
    env.set_action(np.array([0.0, 0.0]))
    for _ in range(warmup_steps):
        env.step(steps=1)

    total_frames = 0

    for seq_idx in range(n_seqs):
        if seq_idx > 0:
            env = ContinuousSoftArmEnv(dt=dt)
            env.set_action(np.array([0.0, 0.0]))
            for _ in range(warmup_steps):
                env.step(steps=1)

        actions = schedule.generate()
        seq_imgs_front, seq_imgs_side = [], []
        seq_actions, seq_positions, seq_radii = [], [], []
        seq_depths_front, seq_depths_side = [], []

        pbar = tqdm(total=actions_per_seq * steps_per_action,
                     desc=f"Seq {seq_idx + 1}/{n_seqs}")

        for target_action in actions:
            env.set_action(target_action)
            for _ in range(steps_per_action):
                env.step(steps=1)
                pbar.update(1)

                if env.step_count % record_interval == 0:
                    if record_depth:
                        img_f, dep_f, img_s, dep_s, act, pos, rad = \
                            env.get_observation_multiview_with_depth()
                        seq_depths_front.append(dep_f)
                        seq_depths_side.append(dep_s)
                    else:
                        img_f, img_s, act, pos, rad = env.get_observation_multiview()
                    seq_imgs_front.append(img_f)
                    seq_imgs_side.append(img_s)
                    seq_actions.append(act)
                    seq_positions.append(pos)
                    seq_radii.append(rad)

        pbar.close()

        H, W = DEFAULT_IMAGE_SIZE
        data = {
            "images_front": np.array(seq_imgs_front),
            "images_side": np.array(seq_imgs_side),
            "actions": np.array(seq_actions),
            "positions": np.array(seq_positions),
            "radii": np.array(seq_radii),
            "dt": dt * record_interval,
            "focal": cam["focal"],
            "H": H, "W": W,
            "camera_eye_front": np.array(CAMERA_EYE),
            "camera_center_front": np.array(CAMERA_CENTER),
            "camera_up_front": np.array(CAMERA_UP),
            "camera_eye_side": np.array(CAMERA_EYE_SIDE),
            "camera_center_side": np.array(CAMERA_CENTER_SIDE),
            "camera_up_side": np.array(CAMERA_UP_SIDE),
        }

        if record_depth:
            data["depth_maps_front"] = np.array(seq_depths_front, dtype=np.float32)
            data["depth_maps_side"] = np.array(seq_depths_side, dtype=np.float32)

        filepath = os.path.join(save_dir, f"seq_{seq_idx:03d}_{schedule.mode_tag}_mv.npz")
        np.savez_compressed(filepath, **data)

        frames = len(seq_imgs_front)
        total_frames += frames
        print(f"    {os.path.basename(filepath)}: {frames} 帧")

    print(f"\n>>> 采集完成！共 {total_frames} 帧 -> {save_dir}/")


def main():
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="多视角数据采集")
    parser.add_argument("--action-x", choices=["zero", "random", "hold", "file"], default="random")
    parser.add_argument("--action-y", choices=["zero", "random", "hold", "file"], default="random")
    parser.add_argument("--action-file", type=str, default=None)
    parser.add_argument("--sequences", type=int, default=defaults["num_sequences"])
    parser.add_argument("--actions-per-seq", type=int, default=defaults["actions_per_seq"])
    parser.add_argument("--steps-per-action", type=int, default=defaults["steps_per_action"])
    parser.add_argument("--record-interval", type=int, default=defaults["record_interval"])
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--save-dir", type=str, default="data/multiview_rr")
    parser.add_argument("--depth", dest="record_depth", action="store_true",
                        help="保存双视角深度图（z-buffer depth）")
    args = parser.parse_args()

    schedule = ActionSchedule(
        dim_modes=[args.action_x, args.action_y],
        n_actions=args.actions_per_seq,
        min_val=defaults["action_min"],
        max_val=defaults["action_max"],
        step_size=defaults["step_size"],
        file_path=args.action_file,
    )

    run_multiview_collection(schedule, args, defaults)


if __name__ == "__main__":
    main()
