"""train_open_loop_transition.py — 窗口开环状态转移模型训练（OpenLoopTransitionModel）。

定位（详见 docs/directions/15_open_loop_windowed_transition.md）:
  每个窗口 K 步：仅以 1 帧 GT 种子（init_skeleton）锚定绝对位姿，窗口内剩余 K 步把模型自身
  预测喂回（teacher_forcing 退火到 0.0；s 与 z 在窗口内自演化）。窗口结束重新观测种子。
  部署语义："观测一次 → 开环预测 K 步"。迟滞由窗口内累积的潜轨迹 z 编码。

  与姊妹训练脚本区别:
    train_gt_transition.py        — 每步真实 s（tf=1.0），s 不漂移（方向 14 主线）
    train_state_transition_s1.py  — 序列级 + 固定 tf scheduled sampling（方向 13）
    本脚本                        — 窗口开环（tf 退火到 0），漂移约束在 K 步内（方向 15）

推荐训练流程（由 direction 15 验证工作流确认）:
  1. 从 gt_transition checkpoint 热启动（单步动力学已学好，56× 优于 copy）。
     ⚠️ 必须 route through _migrate_gru_keys（gt_transition ckpt 用旧 GRUCell 键名，
        直接 strict=False 会静默丢弃整个训练好的空间 GRU → 蓝点）。
  2. 先试 tf=0.0 直接纯闭环（成本最低；per-frame 误差 ~1e-8 下漂移缓慢，退火未必必要）。
  3. 仅当 rollout 漂移比 > ~50× 才升级退火（--tf_anneal_epochs，staircase 优先）。

Usage:
    # 默认（cuda1）：热启动 gt_transition + 纯闭环 tf=0
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_open_loop_transition.py

    # 短 epoch 冒烟
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_open_loop_transition.py --n_epochs 5

    # 升级：staircase 退火 tf 1.0→0.0 over 15 epochs（前 7 epoch GT，后 8 epoch 纯闭环）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_open_loop_transition.py \
        --tf_ratio 1.0 --tf_anneal_epochs 15 --tf_schedule staircase

    # 指定热启动 checkpoint（默认自动检测最新 gt_transition best_model.pt）
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_open_loop_transition.py \
        --init_from train_log/gt_transition/exp_20260616_3/phase_gt_transition/model/best_model.pt
"""

import os
import sys
import glob

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
from src.utils.model_loader import _migrate_gru_keys


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
                    help="Open-loop window length K (steps fed back per window)")
parser.add_argument("--init_from", type=str, default=None,
                    help="gt_transition checkpoint to warm-start from "
                         "(default: auto-detect latest train_log/gt_transition/*/phase_gt_transition/model/best_model.pt)")
# ── teacher-forcing 退火（默认纯闭环 tf=0；详见 direction 15）──
parser.add_argument("--tf_ratio", type=float, default=0.0,
                    help="Steady-state / anneal-start teacher forcing ratio (0.0=pure closed-loop)")
parser.add_argument("--tf_anneal_epochs", type=int, default=0,
                    help="Anneal tf_ratio→tf_min over this many epochs (0=no anneal, fixed tf_ratio)")
parser.add_argument("--tf_min", type=float, default=0.0,
                    help="Anneal floor teacher forcing ratio")
parser.add_argument("--tf_schedule", type=str, default="staircase",
                    choices=["linear", "staircase"],
                    help="Anneal shape: staircase (前半 nominal/后半 tf_min，规避中段速度混入) or linear")
parser.add_argument("--dense_step_weight", type=str, default="uniform",
                    choices=["uniform", "linear"],
                    help="Dense supervision weighting: uniform or linear (后段权重大)")
args = parser.parse_args()

config = resolve_training_config(build_common_overrides(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = detect_action_dim(args.data_dir)
n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)

from src.models.model_open_loop_transition import OpenLoopTransitionModel

temp_cfg = config["temporal"]
hidden_dim = temp_cfg["hidden_dim"]

model = OpenLoopTransitionModel(
    action_dim=action_dim,
    n_nodes=n_nodes,
    hidden_dim=hidden_dim,
    window_size=temp_cfg["window_size"],
    n_orders=temp_cfg["n_scales"],
    encoder_type=args.encoder,
    z_dim=args.z_dim,
    episode_len=args.episode_len,
).to(device)

# ── 热启动：从 gt_transition checkpoint 加载单步动力学 ──
# ⚠️ 必须 _migrate_gru_keys（gt_transition ckpt 用旧 GRUCell 键 gru.weight_ih，
#    新模型 self.gru 是 nn.GRU 期望 *_l0；strict=False 会静默丢弃整层 GRU）。
init_from = args.init_from
if init_from is None:
    cands = sorted(glob.glob(os.path.join(
        "train_log", "gt_transition", "*", "phase_gt_transition", "model", "best_model.pt")))
    if cands:
        init_from = cands[-1]
        print(f"[warm-start] auto-detected gt_transition checkpoint: {init_from}")
if init_from is not None:
    if not os.path.exists(init_from):
        raise FileNotFoundError(f"--init_from not found: {init_from}")
    sd = torch.load(init_from, map_location=device, weights_only=True)
    sd = _migrate_gru_keys(sd)
    incompatible = model.load_state_dict(sd, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    # 仅允许已知非参数 buffer 未匹配；任何 trained module 缺失 = 静默丢权重 = BLOCKER
    safe = {"gt_observed_mode", "open_loop_mode"}
    real_missing = [k for k in missing if k not in safe]
    assert not real_missing, (
        f"[warm-start BLOCKER] trained keys dropped (GRU etc.): {real_missing}. "
        f"Did _migrate_gru_keys run? missing={missing}, unexpected={unexpected}")
    print(f"[warm-start] loaded {init_from}")
    print(f"  missing(should be only mode buffer)={missing}")
    print(f"  unexpected(should be only gt_observed_mode)={unexpected}")
else:
    print("[warm-start] no gt_transition checkpoint found — training from scratch (cold).")

# ── 覆盖 spec 的 tf 退火参数（CLI → PhaseSpec）──
spec = model.training_spec
p0 = spec.phases[0]
p0.teacher_forcing_ratio = args.tf_ratio
p0.tf_anneal_epochs = args.tf_anneal_epochs
p0.tf_min = args.tf_min
p0.tf_schedule = args.tf_schedule
p0.dense_step_weight = args.dense_step_weight

n_params = sum(p.numel() for p in model.parameters())
print("\nModel: OpenLoopTransition (窗口开环: 1 帧 GT 种子 + K 步自回归 rollout)")
print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, Encoder: {args.encoder}, z_dim: {args.z_dim}")
print(f"  episode_len(K): {args.episode_len}")
print(f"  tf_ratio(start)={args.tf_ratio}, tf_anneal_epochs={args.tf_anneal_epochs}, "
      f"tf_min={args.tf_min}, tf_schedule={args.tf_schedule}")
print(f"  dense_step_weight: {args.dense_step_weight}")
print(f"  Parameters: {n_params:,}")
print(f"  Active losses: {p0.active_losses}")

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
                         model_tag="open_loop_transition")
trainer.train(data_dirs)
