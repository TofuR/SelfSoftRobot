"""train_ms_scnf.py -- MS-SCNF two-phase training via UnifiedTrainer.

Usage:
    # Full training (Phase 1: Skeleton + Phase 2: Joint)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py

    # Phase 1 only (skeleton regression)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 1

    # Phase 2 only (joint training)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 2 \
        --exp_dir train_log/train_ms_scnf/001 \
        --skeleton_path train_log/train_ms_scnf/001/phase1/model/phase1_best.pt
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from src.config.args import (add_common_args, add_two_phase_args, resolve_training_config, build_common_overrides, resolve_phase_epochs)
from src.utils.data_detect import detect_action_dim
from src.training.trainer_unified import UnifiedTrainer
from src.rendering.view_strategy import create_view_strategy


parser = argparse.ArgumentParser()
add_common_args(parser, data_dir_default="data/seq_rz_3d")
add_two_phase_args(parser)
parser.add_argument("--skeleton_path", type=str, default=None)
parser.add_argument("--w_skeleton_fine", type=float, default=None)
parser.add_argument("--w_skeleton_medium", type=float, default=None)
parser.add_argument("--w_skeleton_coarse", type=float, default=None)
parser.add_argument("--w_render", type=float, default=None)
args = parser.parse_args()

overrides = build_common_overrides(args)
overrides.update({
    "ms_scnf.w_skeleton_fine": args.w_skeleton_fine,
    "ms_scnf.w_skeleton_medium": args.w_skeleton_medium,
    "ms_scnf.w_skeleton_coarse": args.w_skeleton_coarse,
    "ms_scnf.w_render": args.w_render,
})
config = resolve_training_config(overrides)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Auto-detect action_dim --
action_dim = detect_action_dim(args.data_dir)

# -- Create model --
from src.models.model_ms_scnf import MSSCNFModel

temp_cfg = config["temporal"]
model_cfg = config["model"]
ms_cfg = config.get("ms_scnf", {})
canon_cfg = config.get("canonical", {})

model = MSSCNFModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
    d_filter=model_cfg["d_filter"],
    n_freqs=model_cfg["n_freqs"],
    n_coarse=ms_cfg.get("n_coarse", 4),
    n_medium=ms_cfg.get("n_medium", 10),
    n_fine=ms_cfg.get("n_fine", 31),
    deform_n_freqs=canon_cfg.get("deform_n_freqs", 6),
    skeleton_mode=ms_cfg.get("skeleton_mode", "point"),
    fourier_n_freq=ms_cfg.get("fourier_n_freq", 8),
    bspline_n_ctrl=ms_cfg.get("bspline_n_ctrl", 10),
    catmullrom_n_ctrl=ms_cfg.get("catmullrom_n_ctrl", 10),
).to(device)

spec = model.training_spec
print("\nModel: MS-SCNF")
print(f"  Action dim: {action_dim}")
print(f"  Phases: {[p.name for p in spec.phases]}")

# -- ViewStrategy for rendering phases --
from src.training.dataset_factory import create_dataset

rendering_phase = next((p for p in spec.phases if p.supervision_mode == "rendering"), None)
view_strat = None
if rendering_phase is not None:
    ds = create_dataset(rendering_phase.dataset_type, args.data_dir, config, rendering_phase)
    view_strat = create_view_strategy(ds, set(rendering_phase.active_losses))

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

n_epochs_per_phase = resolve_phase_epochs(spec, config, phase=args.phase, n_epochs_override=args.n_epochs)

# -- Train --
trainer = UnifiedTrainer(model, view_strat, config=config)
exp_dir = args.exp_dir
trainer.train(data_dirs, exp_dir=exp_dir, n_epochs_per_phase=n_epochs_per_phase)
