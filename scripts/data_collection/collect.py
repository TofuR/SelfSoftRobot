"""collect.py — 统一数据采集入口。

每个动作维度独立控制（zero/random/hold/file），不再区分 canonical/sequence/batch 模式。
所有默认值从 simulation.json + camera.json 读取，相机参数始终写入 npz。

用法:
  # 两维随机游走（默认）
  python scripts/data_collection/collect.py
  python scripts/data_collection/collect.py --3d

  # canonical（两维都为零）
  python scripts/data_collection/collect.py --action-x zero --action-y zero
  python scripts/data_collection/collect.py --action-x zero --action-y zero --3d

  # 单维度随机
  python scripts/data_collection/collect.py --action-x random --action-y zero

  # batch（每段保持一个随机值）
  python scripts/data_collection/collect.py --action-x hold --action-y hold

  # 从文件读取轨迹
  python scripts/data_collection/collect.py --action-x file --action-file traj.npz

  # 含深度图采集（用于 Depth-CMSTNF 训练）
  python scripts/data_collection/collect.py --depth
  python scripts/data_collection/collect.py --3d --depth

  # 完整自定义
  python scripts/data_collection/collect.py \\
      --action-x random --action-y zero \\
      --3d --depth --sequences 10 --actions-per-seq 50 \\
      --save-dir data/my_experiment
"""

import sys
import os
import argparse
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from elastica_env import ContinuousSoftArmEnv
from collect_utils import (
    ActionSchedule, load_defaults, save_collection,
    make_filename, infer_save_dir,
)


# =============================================================================
# 统一采集循环
# =============================================================================

def run_collection(schedule, args, defaults):
    """统一采集循环——不区分 canonical/sequence/batch。"""
    cam = defaults["camera"]
    dt = defaults["dt"]
    n_seqs = args.sequences
    actions_per_seq = args.actions_per_seq
    steps_per_action = args.steps_per_action
    record_interval = args.record_interval
    warmup_steps = args.warmup_steps
    record_3d = args.record_3d
    record_depth = args.record_depth
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    mode_tag = schedule.mode_tag
    total_dur = actions_per_seq * steps_per_action * dt
    print(f"\n>>> 采集开始: [{mode_tag}]" + (" + 3D" if record_3d else "") + (" + Depth" if record_depth else ""))
    print(f"    序列: {n_seqs}, 动作/序列: {actions_per_seq}, "
          f"时长/序列: {total_dur:.2f}s")
    print(f"    动作模式: {', '.join(f'dim{i}={m}' for i, m in enumerate(schedule.dim_modes))}")
    print(f"    保存: {save_dir}")

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
        seq_images, seq_actions = [], []
        seq_positions, seq_radii, seq_depths = [], [], []

        pbar = tqdm(total=actions_per_seq * steps_per_action,
                     desc=f"Seq {seq_idx + 1}/{n_seqs}")

        for target_action in actions:
            env.set_action(target_action)
            for _ in range(steps_per_action):
                env.step(steps=1)
                pbar.update(1)

                if env.step_count % record_interval == 0:
                    # 选择合适的 observation 方法
                    if record_depth and record_3d:
                        img, depth, act, pos, rad = env.get_observation_with_depth()
                        seq_depths.append(depth)
                        seq_positions.append(pos)
                        seq_radii.append(rad)
                    elif record_depth:
                        img, depth, act, pos, rad = env.get_observation_with_depth()
                        seq_depths.append(depth)
                    elif record_3d:
                        img, act, pos, rad = env.get_observation_3d()
                        seq_positions.append(pos)
                        seq_radii.append(rad)
                    else:
                        img, act = env.get_observation()
                    seq_images.append(img)
                    seq_actions.append(act)

        pbar.close()

        filename = make_filename(seq_idx, mode_tag, record_3d)
        filepath = os.path.join(save_dir, filename)
        save_collection(
            filepath, seq_images, seq_actions,
            dt * record_interval, cam,
            positions=seq_positions if record_3d else None,
            radii=seq_radii if record_3d else None,
            depth_maps=seq_depths if record_depth else None,
        )

        frames = len(seq_images)
        total_frames += frames
        print(f"    {filename}: {frames} 帧")

    print(f"\n>>> 采集完成！共 {total_frames} 帧 -> {save_dir}/")


# =============================================================================
# CLI
# =============================================================================

def build_parser(defaults):
    parser = argparse.ArgumentParser(
        description="软体机器人数据采集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  collect.py                                    # 两维 random
  collect.py --3d                               # 两维 random + 3D
  collect.py --action-x zero --action-y zero    # canonical (两维零)
  collect.py --action-x random --action-y zero  # 只变 x
  collect.py --action-x hold --action-y hold    # batch (保持随机值)
  collect.py --action-x file --action-file t.npz  # 文件轨迹""")

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

    # 3D / 深度 / 输出
    parser.add_argument("--3d", dest="record_3d", action="store_true",
                        help="保存 3D 节点坐标和半径")
    parser.add_argument("--depth", dest="record_depth", action="store_true",
                        help="保存深度图（z-buffer depth，与 3D 可独立使用）")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="保存目录（默认自动推断）")

    return parser


def main():
    defaults = load_defaults()
    parser = build_parser(defaults)
    args = parser.parse_args()

    dim_modes = [args.action_x, args.action_y]

    schedule = ActionSchedule(
        dim_modes=dim_modes,
        n_actions=args.actions_per_seq,
        min_val=args.action_min,
        max_val=args.action_max,
        step_size=args.step_size,
        file_path=args.action_file,
    )

    args.save_dir = infer_save_dir(schedule.mode_tag, args.record_3d, args.save_dir)

    run_collection(schedule, args, defaults)


if __name__ == "__main__":
    main()
