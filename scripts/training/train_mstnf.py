"""train_mstnf.py — MSTNF 单阶段训练入口。

Usage:
    # 完整训练
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py
"""

import os
import sys

# ═══════════════════════════════════════════════════════════
# 常用配置（直接在这里修改）
# ═══════════════════════════════════════════════════════════
DATA_DIR = "data/sequence_data"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from src.training.trainer_mstnf import MSTNFTrainer

trainer = MSTNFTrainer(device=device)
trainer.train(data_dir=DATA_DIR)
