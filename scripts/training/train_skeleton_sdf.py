"""train_skeleton_sdf.py -- Skeleton+SDF two-phase training via UnifiedTrainer.

Usage:
    # Default: GPU 1, bspline skeleton, two-phase training
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py

    # Specify skeleton mode
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py --skeleton_mode fourier

    # Override loss weights
    CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py \
        --w_skeleton_fine 1.0 --w_sdf 3000 --w_eikonal 50
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from src.config.args import add_common_args, resolve_training_config
from src.utils.data_detect import detect_action_dim
from src.training.trainer_unified import UnifiedTrainer


parser = argparse.ArgumentParser(description="SkeletonSDF Training")
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
parser.add_argument("--phase1_epochs", type=int, default=50)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--skeleton_mode", type=str, default=None,
                    choices=["point", "fourier", "bspline", "catmullrom"])
parser.add_argument("--rod_radius", type=float, default=0.015)
parser.add_argument("--w_skeleton_fine", type=float, default=1.0)
parser.add_argument("--w_skeleton_medium", type=float, default=0.3)
parser.add_argument("--w_skeleton_coarse", type=float, default=0.1)
parser.add_argument("--w_smooth", type=float, default=0.01)
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_eikonal", type=float, default=50.0)
parser.add_argument("--n_surface", type=int, default=None)
parser.add_argument("--n_near_surface", type=int, default=None)
parser.add_argument("--n_off_surface", type=int, default=None)
args = parser.parse_args()

config = resolve_training_config({
    "temporal.window_size": args.window_size,
    "sdf.w_sdf": args.w_sdf,
    "sdf.w_normal": args.w_normal,
    "sdf.w_eikonal": args.w_eikonal,
    "sdf.n_surface": args.n_surface,
    "sdf.n_near_surface": args.n_near_surface,
    "sdf.n_off_surface": args.n_off_surface,
    "ms_scnf.w_skeleton_fine": args.w_skeleton_fine,
    "ms_scnf.w_skeleton_medium": args.w_skeleton_medium,
    "ms_scnf.w_skeleton_coarse": args.w_skeleton_coarse,
    "ms_scnf.w_smooth": args.w_smooth,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Auto-detect action_dim --
action_dim = detect_action_dim(args.data_dir)

# -- Determine skeleton_mode --
ms_cfg = config.get("ms_scnf", {})
skeleton_mode = args.skeleton_mode or ms_cfg.get("skeleton_mode", "bspline")

# -- Create model --
from src.models.model_skeleton_sdf import SkeletonSDFModel

temp_cfg = config["temporal"]
sdf_cfg = config.get("sdf", {})

model = SkeletonSDFModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
    skeleton_mode=skeleton_mode,
    rod_radius=args.rod_radius,
    w_skel_fine=ms_cfg.get("w_skeleton_fine", 1.0),
    w_skel_medium=ms_cfg.get("w_skeleton_medium", 0.3),
    w_skel_coarse=ms_cfg.get("w_skeleton_coarse", 0.1),
    w_skel_smooth=ms_cfg.get("w_smooth", 0.01),
    w_sdf=sdf_cfg.get("w_sdf", 3e3),
    w_normal=sdf_cfg.get("w_normal", 10.0),
    w_eikonal=sdf_cfg.get("w_eikonal", 50.0),
).to(device)

spec = model.training_spec
print("\nModel: SkeletonSDF")
print(f"  Action dim: {action_dim}, skeleton_mode: {skeleton_mode}")
print(f"  Phases: {[p.name for p in spec.phases]}")

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

# -- Two-phase epoch allocation --
n_epochs_per_phase = {}
for p in spec.phases:
    if p.name == "skeleton":
        n_epochs_per_phase[p.name] = args.phase1_epochs
    else:
        n_epochs_per_phase[p.name] = config["optimization"]["n_epochs"]

# -- Train (no ViewStrategy -- direct_3d supervision) --
trainer = UnifiedTrainer(model, view_strategy=None, config=config)
trainer.train(data_dirs, n_epochs_per_phase=n_epochs_per_phase)
