"""collect.py — 统一数据采集入口（支持任意数量相机）。

每个动作维度独立控制（zero/random/hold/file）。相机数量由
config/camera.json 中的 primary + extra_cameras 决定：
  - 无 extra_cameras → 单相机模式 (c1)
  - 有 N 个 extra_cameras → N+1 相机模式 (cN+1)

数据格式根据相机数自动适配：
  c1: images (N, H, W), camera_eye/center/up（单视角训练兼容）
  c2+: images (N, V, H, W), camera_params (V, 10), view_names

用法:
  # 两维随机游走，单相机（默认）
  python scripts/data_collection/collect.py

  # 两维随机游走，多相机
  python scripts/data_collection/collect.py --sk

  # canonical（两维都为零）
  python scripts/data_collection/collect.py --action-x zero --action-y zero

  # 含深度图（用于 Depth-CMSTNF / 多视角训练）
  python scripts/data_collection/collect.py --depth

  # 从文件读取轨迹
  python scripts/data_collection/collect.py --action-x file --action-file traj.npz

  # 完整自定义
  python scripts/data_collection/collect.py \\
      --action-x random --action-y zero \\
      --sk --depth --sequences 10 --actions-per-seq 50 \\
      --save-dir data/my_experiment
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config.params import get_camera_params, get_all_camera_params
from elastica_env import (
    ContinuousSoftArmEnv,
    ALL_CAMERAS, N_VIEWS, DEFAULT_IMAGE_SIZE,
)
from collect_utils import (
    ActionSchedule, load_defaults, save_collection,
    make_filename, infer_save_dir,
)
from src.utils.camera_system import MultiCameraSystem


# =============================================================================
# 统一采集循环
# =============================================================================

def run_collection(schedule, args, defaults):
    """统一采集循环——单相机和多相机统一处理。"""
    cam_cfg = get_camera_params()
    all_cam = get_all_camera_params()
    dt = defaults["dt"]
    n_seqs = args.sequences
    actions_per_seq = args.actions_per_seq
    steps_per_action = args.steps_per_action
    record_interval = args.record_interval
    warmup_steps = args.warmup_steps
    record_sk = args.record_sk
    record_depth = args.record_depth
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    mode_tag = schedule.mode_tag
    total_dur = actions_per_seq * steps_per_action * dt
    tags = [f"[{mode_tag}]"]
    if N_VIEWS > 1:
        tags.append(f"c{N_VIEWS}")
    if record_sk:
        tags.append("+SK")
    if record_depth:
        tags.append("+Depth")
    print(f"\n>>> 采集: {' '.join(tags)}")
    print(f"    序列: {n_seqs}, 动作/序列: {actions_per_seq}, 时长/序列: {total_dur:.2f}s")
    print(f"    相机数: {N_VIEWS}")
    for i, c in enumerate(all_cam):
        print(f"      [{i}] {c['name']}: eye={c['eye']}")
    print(f"    动作模式: {', '.join(f'dim{i}={m}' for i, m in enumerate(schedule.dim_modes))}")
    print(f"    保存: {save_dir}")

    # 多相机时构建 MultiCameraSystem
    H, W = DEFAULT_IMAGE_SIZE
    if N_VIEWS > 1:
        cam_configs = []
        for c in all_cam:
            cam_configs.append({
                'eye': c['eye'], 'center': c['center'], 'up': c['up'],
                'focal': cam_cfg['focal'], 'H': H, 'W': W,
            })
        cam_system = MultiCameraSystem(cam_configs)

    env = ContinuousSoftArmEnv(dt=dt)

    # 预热
    print(f"    预热 {warmup_steps} 步 ({warmup_steps * dt:.2f}s)...")
    env.set_action(np.array([0.0, 0.0]))
    for _ in range(warmup_steps):
        env.step(steps=1)

    total_frames = 0

    for seq_idx in range(n_seqs):
        # 每段序列重新创建环境（保证独立性）
        if seq_idx > 0:
            env = ContinuousSoftArmEnv(dt=dt)
            env.set_action(np.array([0.0, 0.0]))
            for _ in range(warmup_steps):
                env.step(steps=1)

        actions = schedule.generate()
        seq_images_per_view = [[] for _ in range(N_VIEWS)]
        seq_depths_per_view = [[] for _ in range(N_VIEWS)] if record_depth else None
        seq_actions, seq_positions, seq_radii = [], [], []

        pbar = tqdm(total=actions_per_seq * steps_per_action,
                     desc=f"Seq {seq_idx + 1}/{n_seqs}")

        for target_action in actions:
            env.set_action(target_action)
            for _ in range(steps_per_action):
                env.step(steps=1)
                pbar.update(1)

                if env.step_count % record_interval == 0:
                    if record_depth:
                        images_list, depths_list, act, pos, rad = \
                            env.get_observation_multiview_with_depth()
                        for v in range(N_VIEWS):
                            seq_depths_per_view[v].append(depths_list[v])
                    elif N_VIEWS == 1 and not record_sk:
                        # 单相机 + 无骨架：用最快的方法
                        img, act = env.get_observation()
                        seq_images_per_view[0].append(img)
                        seq_actions.append(act)
                        continue
                    else:
                        images_list, act, pos, rad = env.get_observation_multiview()

                    for v in range(N_VIEWS):
                        seq_images_per_view[v].append(images_list[v])
                    seq_actions.append(act)
                    if record_sk:
                        seq_positions.append(pos)
                        seq_radii.append(rad)

        pbar.close()

        filename = make_filename(seq_idx, mode_tag, N_VIEWS, record_sk)
        filepath = os.path.join(save_dir, filename)

        # --- 保存 ---
        if N_VIEWS == 1:
            # 单相机：向后兼容格式
            save_collection(
                filepath, seq_images_per_view[0], seq_actions,
                dt * record_interval, defaults["camera"],
                positions=seq_positions if record_sk else None,
                radii=seq_radii if record_sk else None,
                depth_maps=seq_depths_per_view[0] if record_depth else None,
            )
        else:
            # 多相机：(N, V, H, W) 格式 + camera_params
            view_names = [c['name'] for c in all_cam]
            images = np.stack(seq_images_per_view, axis=1)  # (N, V, H, W)

            data = {
                "images": np.array(images, dtype=np.float32),
                "actions": np.array(seq_actions),
                "dt": dt * record_interval,
                "focal": cam_cfg["focal"],
                "H": H, "W": W,
                "camera_params": cam_system.get_camera_params_array(),
                "view_names": np.array(view_names),
            }
            if record_sk:
                data["positions"] = np.array(seq_positions)
                data["radii"] = np.array(seq_radii)
            if record_depth:
                depths = np.stack(seq_depths_per_view, axis=1)  # (N, V, H, W)
                data["depths"] = np.array(depths, dtype=np.float32)

            # 保留旧格式字段（前两个视角）以兼容
            legacy_suffixes = ['front', 'side']
            for v in range(min(N_VIEWS, 2)):
                suffix = legacy_suffixes[v]
                data[f"images_{suffix}"] = np.array(seq_images_per_view[v])
                eye, center, up = ALL_CAMERAS[v]
                data[f"camera_eye_{suffix}"] = np.array(eye)
                data[f"camera_center_{suffix}"] = np.array(center)
                data[f"camera_up_{suffix}"] = np.array(up)
            if record_depth:
                for v in range(min(N_VIEWS, 2)):
                    data[f"depth_maps_{legacy_suffixes[v]}"] = \
                        np.array(seq_depths_per_view[v], dtype=np.float32)

            np.savez_compressed(filepath, **data)

        frames = len(seq_actions)
        total_frames += frames
        print(f"    {filename}: {frames} 帧")

    print(f"\n>>> 采集完成！共 {total_frames} 帧 -> {save_dir}/")


# =============================================================================
# CLI
# =============================================================================

def build_parser(defaults):
    parser = argparse.ArgumentParser(
        description="软体机器人数据采集（支持多相机）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  collect.py                                    # 两维 random, 单/多相机
  collect.py --sk                               # 两维 random + sk
  collect.py --action-x zero --action-y zero    # canonical (两维零)
  collect.py --action-x random --action-y zero  # 只变 x
  collect.py --action-x hold --action-y hold    # batch (保持随机值)
  collect.py --action-x file --action-file t.npz  # 文件轨迹
  collect.py --depth                            # 含深度图""")

    # 动作控制
    act = parser.add_argument_group("动作控制")
    act.add_argument("--action-x", choices=["zero", "random", "hold", "file"],
                     default="random", help="dim0 模式（默认 random）")
    act.add_argument("--action-y", choices=["zero", "random", "hold", "file"],
                     default="random", help="dim1 模式（默认 random）")
    act.add_argument("--action-file", type=str, default=None,
                     help="file 模式的数据源路径（npz，含 actions 字段）")
    act.add_argument("--action-min", type=float, default=defaults["action_min"],
                     help="动作下限（默认 from config）")
    act.add_argument("--action-max", type=float, default=defaults["action_max"],
                     help="动作上限（默认 from config）")
    act.add_argument("--step-size", type=float, default=defaults["step_size"],
                     help="随机游走步长（默认 from config）")

    # 采集参数
    col = parser.add_argument_group("采集参数")
    col.add_argument("--sequences", type=int, default=defaults["num_sequences"],
                     help="序列数量（默认 from config）")
    col.add_argument("--actions-per-seq", type=int, default=defaults["actions_per_seq"],
                     help="每段动作目标数（默认 from config）")
    col.add_argument("--steps-per-action", type=int, default=defaults["steps_per_action"],
                     help="每动作仿真步数（默认 from config）")
    col.add_argument("--record-interval", type=int, default=defaults["record_interval"],
                     help="录制间隔（默认 from config）")
    col.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"],
                     help="预热步数（默认 from config）")

    # sk / 深度 / 输出
    parser.add_argument("--sk", dest="record_sk", action="store_true",
                        help="保存 sk 节点坐标和半径")
    parser.add_argument("--depth", dest="record_depth", action="store_true",
                        help="保存深度图（z-buffer depth）")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="保存目录（默认自动推断，含相机数后缀如 _c2）")

    return parser


def main():
    defaults = load_defaults()
    parser = build_parser(defaults)
    args = parser.parse_args()

    schedule = ActionSchedule(
        dim_modes=[args.action_x, args.action_y],
        n_actions=args.actions_per_seq,
        min_val=args.action_min,
        max_val=args.action_max,
        step_size=args.step_size,
        file_path=args.action_file,
    )

    args.save_dir = infer_save_dir(
        schedule.mode_tag, N_VIEWS, args.record_sk, args.save_dir)

    run_collection(schedule, args, defaults)


if __name__ == "__main__":
    main()
