"""train_ms_scnf.py — MS-SCNF 两阶段训练入口。

Usage:
    # 完整训练 (Phase 1: Skeleton + Phase 2: Joint)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py

    # 仅 Phase 1（骨架回归）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 1

    # 仅 Phase 2（联合训练），需指定实验目录和骨架权重
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 2 \
        --exp_dir train_log/train_ms_scnf/001 \
        --skeleton_path train_log/train_ms_scnf/001/phase1/model/skeleton_best.pt \
        --data_dir data/sequence_data_3d
"""

import os
import sys

CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.trainer_ms_scnf import MSSCNFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/seq_rz_3d")
parser.add_argument("--skeleton_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
args = parser.parse_args()

print(f"Device: {device}")
trainer = MSSCNFTrainer(device=device)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, skeleton_path=args.skeleton_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir)
