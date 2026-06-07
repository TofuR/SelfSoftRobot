"""train_skeleton_sdf.py -- Skeleton+SDF two-phase training via UnifiedTrainer.

Usage:
    # Default: bspline skeleton, two-phase training
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_skeleton_sdf.py

    # Specify skeleton mode
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_skeleton_sdf.py --skeleton_mode fourier

    # Override loss weights
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_skeleton_sdf.py \
        --w_skeleton_fine 1.0 --w_sdf 3000 --w_eikonal 50

    # Run only phase 2
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_skeleton_sdf.py \
        --phase 2 --exp_dir train_log/train_skeleton_sdf/exp_xxx
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from src.config.args import (add_common_args, add_two_phase_args,
                              resolve_training_config, build_common_overrides,
                              resolve_phase_epochs)
from src.utils.data_detect import detect_action_dim
from src.training.trainer_unified import UnifiedTrainer


parser = argparse.ArgumentParser(description="SkeletonSDF Training")
add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
add_two_phase_args(parser)
parser.add_argument("--skeleton_mode", type=str, default=None,
                    choices=["point", "fourier", "bspline", "catmullrom"])
parser.add_argument("--rod_radius", type=float, default=None)
parser.add_argument("--w_skeleton_fine", type=float, default=None)
parser.add_argument("--w_skeleton_medium", type=float, default=None)
parser.add_argument("--w_skeleton_coarse", type=float, default=None)
parser.add_argument("--w_sdf", type=float, default=None)
parser.add_argument("--w_normal", type=float, default=None)
parser.add_argument("--w_eikonal", type=float, default=None)
parser.add_argument("--n_surface", type=int, default=None)
parser.add_argument("--n_near_surface", type=int, default=None)
parser.add_argument("--n_off_surface", type=int, default=None)
args = parser.parse_args()

overrides = build_common_overrides(args)
overrides.update({
    "ms_scnf.w_skeleton_fine": args.w_skeleton_fine,
    "ms_scnf.w_skeleton_medium": args.w_skeleton_medium,
    "ms_scnf.w_skeleton_coarse": args.w_skeleton_coarse,
    "ms_scnf.rod_radius": args.rod_radius,
    "sdf.w_sdf": args.w_sdf,
    "sdf.w_normal": args.w_normal,
    "sdf.w_eikonal": args.w_eikonal,
    "sdf.n_surface": args.n_surface,
    "sdf.n_near_surface": args.n_near_surface,
    "sdf.n_off_surface": args.n_off_surface,
})
config = resolve_training_config(overrides)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_dim = detect_action_dim(args.data_dir)

ms_cfg = config.get("ms_scnf", {})
sdf_cfg = config.get("sdf", {})
skeleton_mode = args.skeleton_mode or ms_cfg.get("skeleton_mode", "bspline")
rod_radius = args.rod_radius or ms_cfg.get("rod_radius", 0.015)

from src.models.model_skeleton_sdf import SkeletonSDFModel

temp_cfg = config["temporal"]
model_cfg = config.get("model", {})
model = SkeletonSDFModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
    n_coarse=ms_cfg.get("n_coarse", 4),
    n_medium=ms_cfg.get("n_medium", 10),
    n_fine=ms_cfg.get("n_fine", 31),
    skeleton_mode=skeleton_mode,
    rod_radius=rod_radius,
    fourier_n_freq=ms_cfg.get("fourier_n_freq", 8),
    bspline_n_ctrl=ms_cfg.get("bspline_n_ctrl", 10),
    catmullrom_n_ctrl=ms_cfg.get("catmullrom_n_ctrl", 10),
).to(device)

spec = model.training_spec
print("\nModel: SkeletonSDF")
print(f"  Action dim: {action_dim}, skeleton_mode: {skeleton_mode}")
print(f"  Phases: {[p.name for p in spec.phases]}")

n_epochs_per_phase = resolve_phase_epochs(
    spec, config, phase=args.phase, n_epochs_override=args.n_epochs)

trainer = UnifiedTrainer(model, view_strategy=None, config=config)
trainer.train({"sequence": args.data_dir}, exp_dir=args.exp_dir,
              n_epochs_per_phase=n_epochs_per_phase)
