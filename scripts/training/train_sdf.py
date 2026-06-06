"""train_sdf.py -- SDF 3D supervision training via UnifiedTrainer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --data_dir data/seq_rz_3d --n_epochs 1000
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_sdf.py \
        --lr 1e-4 --w_sdf 3e3 --w_normal 1e2
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from src.config.args import add_common_args, resolve_training_config
from src.utils.data_detect import detect_action_dim
from src.training.trainer_unified import UnifiedTrainer


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_3d")
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_grad", type=float, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--n_scales", type=int, default=None)
parser.add_argument("--hidden_dim", type=int, default=None)
parser.add_argument("--n_surface", type=int, default=None, help="Surface sample count")
parser.add_argument("--n_near_surface", type=int, default=None, help="Near-surface sample count")
parser.add_argument("--n_off_surface", type=int, default=None, help="Off-surface sample count")
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "temporal.n_scales": args.n_scales,
    "temporal.hidden_dim": args.hidden_dim,
    "sdf.w_sdf": args.w_sdf,
    "sdf.w_normal": args.w_normal,
    "sdf.w_grad": args.w_grad,
    "sdf.n_surface": args.n_surface,
    "sdf.n_near_surface": args.n_near_surface,
    "sdf.n_off_surface": args.n_off_surface,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Auto-detect action_dim --
action_dim = detect_action_dim(args.data_dir)

# -- Create model --
from src.models.model_sdf import TemporalSDFModel

temp_cfg = config["temporal"]
sdf_cfg = config.get("sdf", {})

model = TemporalSDFModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
    w_sdf=sdf_cfg.get("w_sdf", 3e3),
    w_normal=sdf_cfg.get("w_normal", 1e2),
    w_eikonal=sdf_cfg.get("w_eikonal", 5e1),
).to(device)

spec = model.training_spec
print("\nModel: SDF")
print(f"  Action dim: {action_dim}")
print(f"  Phases: {[p.name for p in spec.phases]}")

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

# -- Train (no ViewStrategy -- direct_3d supervision) --
trainer = UnifiedTrainer(model, view_strategy=None, config=config)
trainer.train(data_dirs)
