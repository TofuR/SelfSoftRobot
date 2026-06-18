"""train_state_transition.py — StateTransitionSpatialModel 训练 via UnifiedTrainer。

闭环状态转移模型：前一步骨架 + 当前动作 → 当前骨架（预测增量 s_t = s_{t-1} + Δ），
带可学习迟滞潜变量 z（方案 A，无 GT，端到端学）。

Stage 0：3D 纯监督，teacher forcing（训练时 prev_skeleton 取 GT）。
UnifiedTrainer 逐帧独立训练，零 trainer 改动。

Usage:
    # 默认（fractional 编码器，cuda1）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition.py

    # 切换编码器 + z_dim
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition.py \
        --data_dir data/seq_rz_c2_sk --encoder gamma --z_dim 32

    # 短 epoch 冒烟测试（如 5 个 epoch 验证管线）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition.py --n_epochs 5
"""

import os
import sys

# 默认 cuda1（按用户要求：测试实验用 cuda1 跑短 epoch）
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from src.config.args import (add_common_args, resolve_training_config,
                             build_common_overrides)
from src.utils.data_detect import detect_action_dim, detect_n_nodes
from src.training.trainer_unified import UnifiedTrainer


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--encoder", type=str, default="fractional",
                    choices=["ema", "fractional", "gamma", "gru", "transformer", "tcn"],
                    help="Temporal encoder type")
parser.add_argument("--n_nodes", type=int, default=None,
                    help="Number of skeleton nodes (auto-detect if None)")
parser.add_argument("--z_dim", type=int, default=16,
                    help="Dimension of learnable hysteretic latent z (recommend 16-32)")
args = parser.parse_args()

config = resolve_training_config(build_common_overrides(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = detect_action_dim(args.data_dir)
n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)

from src.models.model_state_transition import StateTransitionSpatialModel

temp_cfg = config["temporal"]
hidden_dim = temp_cfg["hidden_dim"]

model = StateTransitionSpatialModel(
    action_dim=action_dim,
    n_nodes=n_nodes,
    hidden_dim=hidden_dim,
    window_size=temp_cfg["window_size"],
    n_orders=temp_cfg["n_scales"],
    encoder_type=args.encoder,
    z_dim=args.z_dim,
).to(device)

spec = model.training_spec
n_params = sum(p.numel() for p in model.parameters())
print("\nModel: StateTransition (closed-loop, learnable latent z)")
print(f"  Action dim: {action_dim}")
print(f"  N nodes: {n_nodes}")
print(f"  Encoder: {args.encoder}")
print(f"  z_dim: {args.z_dim}")
print(f"  Parameters: {n_params:,}")
print(f"  Active losses: {spec.phases[0].active_losses}")

# 用 StateTransitionDataset 读取归一化参数（与训练用同一数据集类）
from src.data.dataset_spatial import StateTransitionDataset
norm_dataset = StateTransitionDataset(
    args.data_dir,
    seq_len=temp_cfg["window_size"],
)
pc_center, pc_scale = norm_dataset.get_normalization_params()
action_norm_factor = norm_dataset.norm_factor

model.set_normalization(pc_center, pc_scale, action_norm_factor)
print(f"  Action norm factor: {action_norm_factor:.4f}")
print(f"  PC center: {pc_center}")
print(f"  PC scale: {pc_scale}")

data_dirs = {"sequence": args.data_dir}
trainer = UnifiedTrainer(model, view_strategy=None, config=config,
                         model_tag="state_transition")
trainer.train(data_dirs)
