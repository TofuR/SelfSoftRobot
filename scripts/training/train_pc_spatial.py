"""train_pc_spatial.py — 预测-修正空间序列模型训练 via UnifiedTrainer。

两阶段训练:
  Phase 1 (Predictive): 仅训练预测分支（FractionalMemory + GRU）
  Phase 2 (Corrective): 解冻修正分支（CNN 图像编码器 + 残差头）

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_pc_spatial.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_pc_spatial.py \
        --data_dir data/seq_rz_c2_sk --n_epochs 500 --encoder fractional
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import glob
import numpy as np
import torch

from src.config.args import add_common_args, resolve_training_config
from src.training.trainer_unified import UnifiedTrainer


def detect_action_dim(data_dir):
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")
    sample = np.load(npz_files[0])
    if 'actions' in sample:
        return sample['actions'].shape[-1]
    raise ValueError(f"No 'actions' field in {npz_files[0]}")


def detect_n_nodes(data_dir):
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    sample = np.load(npz_files[0])
    if 'positions' in sample:
        return sample['positions'].shape[-1]
    raise ValueError(f"No 'positions' field in {npz_files[0]}")


def detect_n_views(data_dir):
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    sample = np.load(npz_files[0])
    images = sample.get('images')
    if images is not None and images.ndim == 4:
        return images.shape[1]
    return 2


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--encoder", type=str, default="fractional",
                    choices=["ema", "fractional"])
parser.add_argument("--n_nodes", type=int, default=None)
parser.add_argument("--phase", type=str, default=None,
                    choices=["predictive", "corrective", "none"],
                    help="Train specific phase only (None=all phases)")
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "temporal.n_scales": args.n_scales,
    "temporal.hidden_dim": args.hidden_dim,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = detect_action_dim(args.data_dir)
n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)
n_views = detect_n_views(args.data_dir)

from src.models.model_pc_spatial import PCSpatialSequenceModel

temp_cfg = config["temporal"]
hidden_dim = temp_cfg["hidden_dim"]

model = PCSpatialSequenceModel(
    action_dim=action_dim,
    n_nodes=n_nodes,
    hidden_dim=hidden_dim,
    window_size=temp_cfg["window_size"],
    n_orders=temp_cfg["n_scales"],
    encoder_type=args.encoder,
    n_views=n_views,
).to(device)

spec = model.training_spec
n_params = sum(p.numel() for p in model.parameters())
print("\nModel: PCSpatialSequence (Predictive-Corrective)")
print(f"  Action dim: {action_dim}")
print(f"  N nodes: {n_nodes}")
print(f"  N views: {n_views}")
print(f"  Encoder: {args.encoder}")
print(f"  Parameters: {n_params:,}")

pred_params = sum(p.numel() for n, p in model.named_parameters()
                  if not n.startswith("correction"))
corr_params = sum(p.numel() for n, p in model.named_parameters()
                  if n.startswith("correction"))
print(f"  Predictive branch: {pred_params:,}")
print(f"  Correction branch: {corr_params:,}")
print(f"  Phases: {[p.name for p in spec.phases]}")

from src.data.dataset_spatial import SpatialSequenceDataset
norm_dataset = SpatialSequenceDataset(
    args.data_dir, seq_len=temp_cfg["window_size"])
pc_center, pc_scale = norm_dataset.get_normalization_params()
action_norm_factor = norm_dataset.norm_factor
model.set_normalization(pc_center, pc_scale, action_norm_factor)

print(f"  Action norm factor: {action_norm_factor:.4f}")
print(f"  PC center: {pc_center}")
print(f"  PC scale: {pc_scale}")

data_dirs = {"sequence": args.data_dir}
trainer = UnifiedTrainer(model, view_strategy=None, config=config)

if args.phase == "predictive":
    n_epochs_per_phase = {0: config.get("optimization", {}).get("n_epochs", 500)}
elif args.phase == "corrective":
    n_epochs_per_phase = {1: config.get("optimization", {}).get("n_epochs", 500)}
else:
    n_epochs_per_phase = None

trainer.train(data_dirs, n_epochs_per_phase=n_epochs_per_phase)
