"""train_mstnf.py — MSTNF 训练入口。"""

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
