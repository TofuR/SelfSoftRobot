"""train_gt_transition.py — 全 GT 驱动单步状态转移模型训练（GTObservedTransitionModel）。

定位（与 train_state_transition.py / train_state_transition_s1.py 的区别）:
  这是"全 GT 驱动"框架——前一状态 s_{t-1} 永远来自真实观测（仿真 GT / 实物图像骨架化），
  模型做单步转移 ŝ_t = F(真实 s_{t-1}, z_{t-1}, a_t)。train 与 inference 完全一致。

  - episode 模式：z 在序列内逐步演化（BPTT），s 每步取 GT
  - teacher_forcing_ratio = 1.0（s 总是真实，无需 scheduled sampling）
  - 不涉及纯自回归 rollout（那是 model_state_transition 的未来扩展方向）

z 跨帧演化：z 是可学习迟滞潜变量，无 GT，编码位置+动作之外的深度历史。

Usage:
    # 默认（cuda1）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py

    # 短 epoch 冒烟测试
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py --n_epochs 5

    # 调 episode_len / z_dim
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py \
        --episode_len 16 --z_dim 32
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
                    help="Dimension of learnable hysteretic latent z")
parser.add_argument("--episode_len", type=int, default=40,
                    help="State window length K (= z evolution steps); default aligns action_window")
parser.add_argument("--dense_step_weight", type=str, default="uniform",
                    choices=["uniform", "linear"],
                    help="Dense supervision weighting: uniform (等权) or linear (递增，最后步权重大)")
args = parser.parse_args()

config = resolve_training_config(build_common_overrides(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = detect_action_dim(args.data_dir)
n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)

from src.models.model_gt_transition import GTObservedTransitionModel

temp_cfg = config["temporal"]
hidden_dim = temp_cfg["hidden_dim"]

model = GTObservedTransitionModel(
    action_dim=action_dim,
    n_nodes=n_nodes,
    hidden_dim=hidden_dim,
    window_size=temp_cfg["window_size"],
    n_orders=temp_cfg["n_scales"],
    encoder_type=args.encoder,
    z_dim=args.z_dim,
    episode_len=args.episode_len,
).to(device)

spec = model.training_spec
# 透传 dense 监督权重模式到 spec（trainer 的 _compute_sequence_losses 读取）
spec.phases[0].dense_step_weight = args.dense_step_weight
n_params = sum(p.numel() for p in model.parameters())
print("\nModel: GTObservedTransition (全 GT 驱动窗口, z 跨帧演化, dense supervision)")
print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, Encoder: {args.encoder}, z_dim: {args.z_dim}")
print(f"  episode_len(K): {args.episode_len}, teacher_forcing_ratio: {spec.phases[0].teacher_forcing_ratio}, dense_step_weight: {args.dense_step_weight}")
print(f"  Parameters: {n_params:,}")
print(f"  Active losses: {spec.phases[0].active_losses}")

# 归一化参数（episode 模式数据集，与训练一致）
from src.data.dataset_spatial import StateTransitionDataset
norm_dataset = StateTransitionDataset(
    args.data_dir,
    seq_len=temp_cfg["window_size"],
    episode_mode=True,
    episode_len=args.episode_len,
)
pc_center, pc_scale = norm_dataset.get_normalization_params()
model.set_normalization(pc_center, pc_scale, norm_dataset.norm_factor)

data_dirs = {"sequence": args.data_dir}
trainer = UnifiedTrainer(model, view_strategy=None, config=config,
                         model_tag="gt_transition")
trainer.train(data_dirs)
