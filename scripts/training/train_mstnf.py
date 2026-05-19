"""train_mstnf.py — MSTNF 单阶段训练入口。

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py \
        --lr 1e-4 --data_dir data/sequence_data_1d
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.args import add_common_args, resolve_training_config
from src.training.trainer_mstnf import MSTNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
args = parser.parse_args()

config = resolve_training_config({
    "optimization.batch_size": args.batch_size,
    "temporal.window_size": args.window_size,
})

print(f"Device: {device}")
trainer = MSTNFTrainer(device=device, config=config)
trainer.train(data_dir=args.data_dir)
