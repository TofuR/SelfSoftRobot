"""train_ms_scnf.py — MS-SCNF 两阶段训练入口。

Usage:
    # 完整训练 (Phase 1: Skeleton + Phase 2: Joint)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py

    # 仅 Phase 1（骨架回归）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 1

    # 仅 Phase 2（联合训练）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 2 \
        --exp_dir train_log/train_ms_scnf/001 \
        --skeleton_path train_log/train_ms_scnf/001/phase1/model/skeleton_best.pt

    # 覆盖学习率和骨架 loss 权重
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py \
        --lr 1e-4 --w_skeleton_fine 2.0

未指定的参数自动从 src/config/training.json 读取。
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.params import load_config
from src.utils.config_utils import resolve_config
from src.training.trainer_ms_scnf import MSSCNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/seq_rz_3d")
parser.add_argument("--skeleton_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--phase1_epochs", type=int, default=None)
parser.add_argument("--phase2_epochs", type=int, default=None)
parser.add_argument("--w_skeleton_fine", type=float, default=None)
parser.add_argument("--w_skeleton_medium", type=float, default=None)
parser.add_argument("--w_skeleton_coarse", type=float, default=None)
parser.add_argument("--w_render", type=float, default=None)
args = parser.parse_args()

defaults = load_config("training")
config = resolve_config(defaults, {
    "optimization.lr": args.lr,
    "optimization.n_epochs": args.n_epochs,
    "canonical.phase1_epochs": args.phase1_epochs,
    "canonical.phase2_epochs": args.phase2_epochs,
    "ms_scnf.w_skeleton_fine": args.w_skeleton_fine,
    "ms_scnf.w_skeleton_medium": args.w_skeleton_medium,
    "ms_scnf.w_skeleton_coarse": args.w_skeleton_coarse,
    "ms_scnf.w_render": args.w_render,
})

print(f"Device: {device}")
trainer = MSSCNFTrainer(device=device, config=config)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, skeleton_path=args.skeleton_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir)
