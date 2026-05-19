"""train_sdf.py — SDF 3D 监督训练入口。

用法:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --data_dir data/seq_rz_3d --n_epochs 1000
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --lr 1e-4 --w_sdf 3e3 --w_normal 1e2
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.args import add_common_args, resolve_training_config
from src.training.trainer_sdf import SDFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_3d")
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_grad", type=float, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--n_surface", type=int, default=None, help="表面采样点数")
parser.add_argument("--n_near_surface", type=int, default=None, help="近表面采样点数")
parser.add_argument("--n_off_surface", type=int, default=None, help="远表面采样点数")
args = parser.parse_args()

config = resolve_training_config({
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
