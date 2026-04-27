"""train_cmstnf.py — CMSTNF 两阶段训练入口。

Usage:
    # 完整训练 (Phase 1 + Phase 2)
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_cmstnf.py

    # 仅 Phase 1（canonical field）
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_cmstnf.py --phase 1

    # 仅 Phase 2（deformation field），需指定实验目录和 canonical 权重
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_cmstnf.py --phase 2 \
        --exp_dir train_log/train_cmstnf/001 \
        --canonical_path train_log/train_cmstnf/001/phase1/model/canonical_best.pt \
        --data_dir data/sequence_data

    # 指定数据路径
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_cmstnf.py \
        --data_dir data/sequence_data_1d \
        --canonical_data_dir data/canonical_data
"""

import os
import sys

CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.trainer_cmstnf import CMSTNFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/sequence_data")
parser.add_argument("--canonical_data_dir", type=str, default="data/canonical_data")
parser.add_argument("--canonical_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
args = parser.parse_args()

print(f"Device: {device}")
trainer = CMSTNFTrainer(device=device)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.canonical_data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, canonical_path=args.canonical_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir, canonical_data_dir=args.canonical_data_dir)
