"""train_mstnf.py — MSTNF 单阶段训练入口。

Usage:
    # 默认参数训练
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py

    # 覆盖学习率和数据目录
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py \
        --lr 1e-4 --data_dir data/sequence_data_1d

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
from src.training.trainer_mstnf import MSTNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data/sequence_data")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
args = parser.parse_args()

defaults = load_config("training")
config = resolve_config(defaults, {
    "optimization.lr": args.lr,
    "optimization.n_epochs": args.n_epochs,
    "optimization.batch_size": args.batch_size,
    "temporal.window_size": args.window_size,
})

print(f"Device: {device}")
trainer = MSTNFTrainer(device=device, config=config)
trainer.train(data_dir=args.data_dir)
