"""train_cmstnf.py — CMSTNF 两阶段训练入口。

Usage:
    # 完整训练 (Phase 1 + Phase 2)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_cmstnf.py

    # 仅 Phase 1（canonical field）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_cmstnf.py --phase 1

    # 仅 Phase 2（deformation field）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_cmstnf.py --phase 2 \
        --exp_dir train_log/train_cmstnf/001 \
        --canonical_path train_log/train_cmstnf/001/phase1/model/phase1_best.pt
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.args import add_common_args, add_two_phase_args, resolve_training_config
from src.training.trainer_cmstnf import CMSTNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
add_common_args(parser)
add_two_phase_args(parser)
parser.add_argument("--canonical_data_dir", type=str, default="data/canonical_data")
parser.add_argument("--canonical_path", type=str, default=None)
parser.add_argument("--deform_lr", type=float, default=None)
args = parser.parse_args()

config = resolve_training_config({
    "canonical.deform_lr": args.deform_lr,
})

print(f"Device: {device}")
trainer = CMSTNFTrainer(device=device, config=config)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.canonical_data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, phase1_path=args.canonical_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir, canonical_data_dir=args.canonical_data_dir)
