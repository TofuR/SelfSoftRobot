"""train_flowmatch.py -- Flow Matching 点云生成训练 via UnifiedTrainer。

改进:
  - action_recon loss: 强制编码器保留 action 信息
  - compactness loss: 惩罚 z-band 内 x,y 扩散（攻击扇形）
  - action_norm_factor 保存到 checkpoint（推理时自动恢复）

Usage:
    CUDA_VISIBLE_DEVICES=3 python scripts/training/train_flowmatch.py
    CUDA_VISIBLE_DEVICES=3 python scripts/training/train_flowmatch.py \
        --data_dir data/seq_rr_3d --n_epochs 500
    CUDA_VISIBLE_DEVICES=3 python scripts/training/train_flowmatch.py \
        --sigma 0.3 --ode_steps 50 --velocity_net_hidden 256
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


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_c6_sk")
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--sigma", type=float, default=None, help="Source noise std")
parser.add_argument("--ode_steps", type=int, default=None, help="ODE integration steps")
parser.add_argument("--ode_solver", type=str, default=None,
                    choices=["euler", "rk4"], help="ODE solver type")
parser.add_argument("--velocity_net_hidden", type=int, default=None)
parser.add_argument("--velocity_net_layers", type=int, default=None)
parser.add_argument("--time_embed_dim", type=int, default=None)
parser.add_argument("--n_surface_points", type=int, default=None)
parser.add_argument("--encoder", type=str, default="ema",
                    choices=["ema", "fractional"],
                    help="Temporal encoder: ema or fractional")
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "temporal.n_scales": args.n_scales,
    "temporal.hidden_dim": args.hidden_dim,
    "pointcloud.sigma": args.sigma,
    "pointcloud.ode_steps": args.ode_steps,
    "pointcloud.ode_solver": args.ode_solver,
    "pointcloud.velocity_net_hidden": args.velocity_net_hidden,
    "pointcloud.velocity_net_layers": args.velocity_net_layers,
    "pointcloud.time_embed_dim": args.time_embed_dim,
    "pointcloud.n_surface_points": args.n_surface_points,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Auto-detect action_dim --
action_dim = detect_action_dim(args.data_dir)

# -- Create model --
from src.models.model_flowmatch import FlowMatchPointCloudModel

temp_cfg = config["temporal"]
pc_cfg = config.get("pointcloud", {})

model = FlowMatchPointCloudModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
    velocity_net_hidden=pc_cfg.get("velocity_net_hidden", 256),
    velocity_net_layers=pc_cfg.get("velocity_net_layers", 6),
    time_embed_dim=pc_cfg.get("time_embed_dim", 64),
    sigma=pc_cfg.get("sigma", 1.0),
    ode_steps=pc_cfg.get("ode_steps", 50),
    ode_solver=pc_cfg.get("ode_solver", "euler"),
    n_points=pc_cfg.get("n_surface_points", 1000),
    encoder_type=args.encoder,
).to(device)

spec = model.training_spec
n_params = sum(p.numel() for p in model.parameters())
print("\nModel: FlowMatchPointCloud")
print(f"  Action dim: {action_dim}")
print(f"  Encoder: {args.encoder}")
print(f"  Parameters: {n_params:,}")
print(f"  Sigma: {model.sigma}")
print(f"  ODE steps: {model.ode_steps} ({model.ode_solver})")
print(f"  Active losses: {spec.phases[0].active_losses}")
print(f"  Loss weights: { {k: v for k, v in config.get('loss_weights', {}).items() if not k.startswith('_')} }")

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

# -- Set normalization from dataset (pc + action) --
from src.data.dataset_pointcloud import PointCloudDataset
norm_dataset = PointCloudDataset(
    args.data_dir,
    seq_len=temp_cfg["window_size"],
    n_surface_points=pc_cfg.get("n_surface_points", 1000),
)
pc_center, pc_scale = norm_dataset.get_normalization_params()
action_norm_factor = norm_dataset.norm_factor  # max(|all_actions|)

# 保存到模型 buffer → 随 checkpoint 一起保存
model.set_normalization(pc_center, pc_scale, action_norm_factor)
print(f"  Action norm factor: {action_norm_factor:.4f}")
print(f"  PC center: {pc_center}")
print(f"  PC scale: {pc_scale}")

# -- Train (no ViewStrategy -- pointcloud supervision) --
trainer = UnifiedTrainer(model, view_strategy=None, config=config)
trainer.train(data_dirs)
