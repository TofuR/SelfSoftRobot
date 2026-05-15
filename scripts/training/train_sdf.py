"""train_sdf.py — SDF 3D 监督训练入口。

用法:
    # 默认参数训练
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py

    # 指定数据和 epoch 数
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --data_dir data/seq_rz_3d --n_epochs 1000

    # 覆盖学习率和 loss 权重
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --lr 1e-4 --w_sdf 3e3 --w_normal 1e2

默认 GPU 0，可通过 CUDA_VISIBLE_DEVICES 环境变量指定。
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
from src.training.trainer_sdf import SDFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/seq_rz_3d")
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_grad", type=float, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--n_surface", type=int, default=None,
                    help="表面采样点数 (default: 300)")
parser.add_argument("--n_near_surface", type=int, default=None,
                    help="近表面采样点数 (default: 200)")
parser.add_argument("--n_off_surface", type=int, default=None,
                    help="远表面均匀采样点数 (default: 200)")
args = parser.parse_args()

defaults = load_config("training")
config = resolve_config(defaults, {
    "optimization.lr": args.lr,
    "optimization.n_epochs": args.n_epochs,
    "temporal.window_size": args.window_size,
    "temporal.n_scales": args.n_scales,
    "temporal.hidden_dim": args.hidden_dim,
    "sdf.w_sdf": args.w_sdf,
    "sdf.w_normal": args.w_normal,
    "sdf.w_grad": args.w_grad,
    "sdf.n_surface": args.n_surface,
    "sdf.n_near_surface": args.n_near_surface,
    "sdf.n_off_surface": args.n_off_surface,
})

print(f"Device: {device}")
trainer = SDFTrainer(device=device, config=config)
trainer.train(data_dir=args.data_dir)
