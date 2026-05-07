"""train_rgbd.py — RGB-D Neural Field 单阶段训练入口。

Usage:
    # 完整训练
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_rgbd.py

    # 调整深度损失权重
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_rgbd.py --depth_weight 0.5

    # 指定数据和 epoch 数
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_rgbd.py \
        --data_dir data/sequence_data --n_epochs 200
"""

import os
import sys

CUDA_DEVICE = 2
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.trainer_rgbd import RGBDTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/sequence_data")
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--depth_weight", type=float, default=1.0,
                    help="Depth loss weight (default: 1.0)")
parser.add_argument("--smooth_weight", type=float, default=0.01,
                    help="Smoothness loss weight (default: 0.01)")
parser.add_argument("--no_guided_sampling", action="store_true",
                    help="Disable depth-guided ray sampling")
args = parser.parse_args()

print(f"Device: {device}")
trainer = RGBDTrainer(
    device=device,
    depth_weight=args.depth_weight,
    smooth_weight=args.smooth_weight,
    use_guided_sampling=not args.no_guided_sampling,
)

trainer.train(data_dir=args.data_dir, n_epochs=args.n_epochs)
