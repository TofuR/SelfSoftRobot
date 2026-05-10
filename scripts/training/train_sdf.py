"""train_sdf.py — SDF 3D 监督训练入口。

用法:
    # 使用 3D 采集数据训练 SDF 模型
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_sdf.py

    # 指定数据和 epoch 数
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --data_dir data/seq_rr_3d --n_epochs 1000

    # 调整 loss 权重
    CUDA_VISIBLE_DEVICES=2 python scripts/training/train_sdf.py \
        --w_sdf 3e3 --w_normal 1e2

默认 GPU 0，可通过 CUDA_VISIBLE_DEVICES 环境变量指定。
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from src.training.trainer_sdf import SDFTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/seq_rr_3d")
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--w_sdf", type=float, default=3e3)
parser.add_argument("--w_normal", type=float, default=1e2)
parser.add_argument("--w_grad", type=float, default=5e1)
args = parser.parse_args()

print(f"Device: {device}")
trainer = SDFTrainer(
    device=device,
    w_sdf=args.w_sdf,
    w_normal=args.w_normal,
    w_grad=args.w_grad,
)
trainer.train(data_dir=args.data_dir, n_epochs=args.n_epochs)
