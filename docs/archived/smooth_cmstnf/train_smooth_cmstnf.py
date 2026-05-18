"""train_smooth_cmstnf.py — Smooth-CMSTNF 两阶段训练（正则化光滑变形）。

Usage:
    # 完整训练 (Phase 1 + Phase 2)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_smooth_cmstnf.py

    # 仅 Phase 1（canonical field）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_smooth_cmstnf.py --phase 1

    # 仅 Phase 2（deformation field）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_smooth_cmstnf.py --phase 2 \
        --exp_dir train_log/train_smooth_cmstnf/001 \
        --canonical_path train_log/train_smooth_cmstnf/001/phase1/model/canonical_best.pt

    # 覆盖正则化权重
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_smooth_cmstnf.py \
        --w_jacobian 0.05 --w_temporal_grad 0.05

未指定的参数自动从 config/training.json 读取。
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from config.params import load_config
from src.utils.config_utils import resolve_config
from src.training.trainer_smooth_cmstnf import SmoothCMSTNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/sequence_data")
parser.add_argument("--canonical_data_dir", type=str, default="data/canonical_data")
parser.add_argument("--canonical_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--phase1_epochs", type=int, default=None)
parser.add_argument("--phase2_epochs", type=int, default=None)
parser.add_argument("--deform_lr", type=float, default=None)
parser.add_argument("--w_jacobian", type=float, default=None)
parser.add_argument("--w_temporal_grad", type=float, default=None)
args = parser.parse_args()

defaults = load_config("training")
config = resolve_config(defaults, {
    "optimization.lr": args.lr,
    "optimization.n_epochs": args.n_epochs,
    "canonical.phase1_epochs": args.phase1_epochs,
    "canonical.phase2_epochs": args.phase2_epochs,
    "canonical.deform_lr": args.deform_lr,
    "canonical.w_jacobian": args.w_jacobian,
    "canonical.w_temporal_grad": args.w_temporal_grad,
})

print(f"Device: {device}")
trainer = SmoothCMSTNFTrainer(device=device, config=config)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.canonical_data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, canonical_path=args.canonical_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir, canonical_data_dir=args.canonical_data_dir)
