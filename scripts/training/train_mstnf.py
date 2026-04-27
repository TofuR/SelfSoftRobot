"""train_mstnf.py — MSTNF 单阶段训练入口。

Usage:
    # 完整训练
    CUDA_VISIBLE_DEVICES=3 python scripts/training/train_mstnf.py

    # 指定数据路径（在文件内修改 CUDA_DEVICE 和 DATA_DIR）
"""

import os
import sys
import torch

# ═══════════════════════════════════════════════════════════
# 常用配置（直接在这里修改）
# ═══════════════════════════════════════════════════════════
CUDA_DEVICE = 3
DATA_DIR = "data/sequence_data"

os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from src.training.trainer_mstnf import MSTNFTrainer

trainer = MSTNFTrainer(device=device)
trainer.train(data_dir=DATA_DIR)
