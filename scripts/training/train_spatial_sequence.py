"""train_spatial_sequence.py — SpatialSequenceModel 训练 via UnifiedTrainer。

空间序列生成模型：用 GRU 沿 Z 轴预测中心线节点坐标，
替代 Flow Matching 的无结构点云生成。

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_spatial_sequence.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_spatial_sequence.py \
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


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--encoder", type=str, default="fractional",
                    choices=["ema", "fractional"],
                    help="Temporal encoder type")
parser.add_argument("--n_nodes", type=int, default=None,
                    help="Number of skeleton nodes (auto-detect if None)")
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "temporal.n_scales": args.n_scales,
    "temporal.hidden_dim": args.hidden_dim,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

action_dim = detect_action_dim(args.data_dir)
n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)

from src.models.model_spatial_sequence import SpatialSequenceModel

temp_cfg = config["temporal"]
hidden_dim = temp_cfg["hidden_dim"]

model = SpatialSequenceModel(
    action_dim=action_dim,
    n_nodes=n_nodes,
    hidden_dim=hidden_dim,
    window_size=temp_cfg["window_size"],
    n_orders=temp_cfg["n_scales"],
    encoder_type=args.encoder,
).to(device)

spec = model.training_spec
n_params = sum(p.numel() for p in model.parameters())
print("\nModel: SpatialSequence")
print(f"  Action dim: {action_dim}")
print(f"  N nodes: {n_nodes}")
print(f"  Encoder: {args.encoder}")
print(f"  Parameters: {n_params:,}")
print(f"  Active losses: {spec.phases[0].active_losses}")

from src.data.dataset_spatial import SpatialSequenceDataset
norm_dataset = SpatialSequenceDataset(
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
trainer = UnifiedTrainer(model, view_strategy=None, config=config)
trainer.train(data_dirs)
