"""train_mstnf.py -- MSTNF single-phase training via UnifiedTrainer.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py \
        --lr 1e-4 --data_dir data/sequence_data_1d
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
from src.rendering.view_strategy import create_view_strategy


parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
args = parser.parse_args()

config = resolve_training_config({
    "optimization.n_epochs": args.n_epochs,
    "optimization.lr": args.lr,
    "optimization.batch_size": args.batch_size,
    "temporal.window_size": args.window_size,
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Auto-detect action_dim --
action_dim = detect_action_dim(args.data_dir)

# -- Create model --
from src.models.model_mstnf import MSTNFModel

temp_cfg = config["temporal"]

model = MSTNFModel(
    action_dim=action_dim,
    window_size=temp_cfg["window_size"],
    n_scales=temp_cfg["n_scales"],
    hidden_dim=temp_cfg["hidden_dim"],
).to(device)

spec = model.training_spec
print("\nModel: MSTNF")
print(f"  Action dim: {action_dim}")
print(f"  Phases: {[p.name for p in spec.phases]}")

# -- ViewStrategy --
from src.training.dataset_factory import create_dataset

phase_spec = spec.phases[0]
ds = create_dataset(phase_spec.dataset_type, args.data_dir, config, phase_spec)
view_strat = create_view_strategy(ds, set(phase_spec.active_losses))

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

# -- Train --
trainer = UnifiedTrainer(model, view_strat, config=config)
trainer.train(data_dirs)
