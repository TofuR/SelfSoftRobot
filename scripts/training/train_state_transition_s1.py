"""train_state_transition_s1.py — Stage 1 序列级训练（z 跨帧演化 + scheduled sampling）。

相对 Stage 0 (train_state_transition.py) 的升级:
  - episode 模式：每个样本是同一 episode 内连续 episode_len 步的序列
  - z 跨帧演化：z 在序列内逐步递推（而非每步从 cond 重初始化），真正成为迟滞潜变量
  - scheduled sampling：按 teacher_forcing_ratio 概率，每步的 prev_skeleton 取 GT
    （teacher forcing）或模型上一步预测（闭环），弥合 train/inference gap
  - BPTT：梯度穿过 T 步，训练 z 的转移动力学

目标：解决 Stage 0 的 rollout 漂移（实测漂移比 1170×），让闭环 rollout 误差可控。

可选：从 Stage 0 checkpoint 热启动（先训好单步转移，再序列级精调 z）。

Usage:
    # 默认（cuda1）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition_s1.py

    # 从 Stage 0 checkpoint 热启动
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition_s1.py \
        --init_from train_log/state_transition/exp_xxx/phase_state_transition/model/best_model.pt

    # 短 epoch 冒烟测试
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_state_transition_s1.py --n_epochs 5
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
from src.training.spec import TrainingSpec, PhaseSpec


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--encoder", type=str, default="fractional",
                    choices=["ema", "fractional", "gamma", "gru", "transformer", "tcn"],
                    help="Temporal encoder type")
parser.add_argument("--n_nodes", type=int, default=None,
                    help="Number of skeleton nodes (auto-detect if None)")
parser.add_argument("--z_dim", type=int, default=16,
                    help="Dimension of learnable hysteretic latent z")
parser.add_argument("--episode_len", type=int, default=20,
                    help="Sequence length per episode (time steps)")
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5,
                    help="Probability of using GT prev_skeleton (1.0=teacher forcing, 0.0=closed-loop)")
parser.add_argument("--init_from", type=str, default=None,
                    help="Stage 0 checkpoint to warm-start from (optional)")
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

# 热启动：从 Stage 0 checkpoint 加载（可选，strict=False 容忍 buffer 差异）
if args.init_from is not None:
    sd = torch.load(args.init_from, map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=False)
    print(f"Warm-started from: {args.init_from}")

# ── 构建 Stage 1 training_spec：克隆模型默认 spec，开启 episode 模式 ──
base_phase = model.training_spec.phases[0]
s1_phase = PhaseSpec(
    name="state_transition_s1",
    dataset_type="state_transition",
    supervision_mode="spatial_sequence",
    active_losses=list(base_phase.active_losses),
    forward_attr="forward",
    use_episode_mode=True,                   # 关键：开启序列级训练
    teacher_forcing_ratio=args.teacher_forcing_ratio,
    episode_len=args.episode_len,
)
model.training_spec = TrainingSpec(phases=[s1_phase])

n_params = sum(p.numel() for p in model.parameters())
print("\nModel: StateTransition (Stage 1, sequence-level, z evolution)")
print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, Encoder: {args.encoder}, z_dim: {args.z_dim}")
print(f"  episode_len: {args.episode_len}, teacher_forcing_ratio: {args.teacher_forcing_ratio}")
print(f"  Parameters: {n_params:,}")

# 归一化参数（用 episode 模式数据集读取，与训练一致）
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
                         model_tag="state_transition_s1")
trainer.train(data_dirs)
