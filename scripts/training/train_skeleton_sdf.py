"""train_skeleton_sdf.py — 方案 B: 参数化骨架 + SDF 截面 训练入口。

用法:
    # 默认: GPU 1, bspline 骨架, 两阶段训练
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py

    # 指定骨架模式
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py --skeleton_mode fourier

    # 覆盖 loss 权重
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py \
        --w_skeleton_fine 1.0 --w_sdf 3000 --w_eikonal 50
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.args import add_common_args, resolve_training_config
from src.training.trainer_skeleton_sdf import SkeletonSDFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="SkeletonSDF Training")
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--phase1_epochs", type=int, default=50)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--skeleton_mode", type=str, default=None,
                    choices=["point", "fourier", "bspline", "catmullrom"])
parser.add_argument("--rod_radius", type=float, default=0.015)
parser.add_argument("--w_skeleton_fine", type=float, default=1.0)
parser.add_argument("--w_skeleton_medium", type=float, default=0.3)
parser.add_argument("--w_skeleton_coarse", type=float, default=0.1)
parser.add_argument("--w_smooth", type=float, default=0.01)
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_eikonal", type=float, default=50.0)
parser.add_argument("--n_surface", type=int, default=None)
parser.add_argument("--n_near_surface", type=int, default=None)
parser.add_argument("--n_off_surface", type=int, default=None)
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "sdf.w_sdf": args.w_sdf,
    "sdf.w_normal": args.w_normal,
    "sdf.w_eikonal": args.w_eikonal,
    "sdf.n_surface": args.n_surface,
    "sdf.n_near_surface": args.n_near_surface,
    "sdf.n_off_surface": args.n_off_surface,
    "ms_scnf.w_skeleton_fine": args.w_skeleton_fine,
    "ms_scnf.w_skeleton_medium": args.w_skeleton_medium,
    "ms_scnf.w_skeleton_coarse": args.w_skeleton_coarse,
    "ms_scnf.w_smooth": args.w_smooth,
})

ms = config.get("ms_scnf", {})
skeleton_mode = args.skeleton_mode or ms.get("skeleton_mode", "bspline")

trainer = SkeletonSDFTrainer(
    device=device,
    config=config,
    skeleton_mode=skeleton_mode,
    rod_radius=args.rod_radius,
    phase1_epochs=args.phase1_epochs,
)
trainer.train(args.data_dir)
