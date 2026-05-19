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
import glob
import numpy as np
import torch

from src.config.args import add_common_args, resolve_training_config
from src.training.trainer_unified import UnifiedTrainer
from src.training.view_strategy import SingleViewStrategy


def detect_action_dim(data_dir):
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")
    sample = np.load(npz_files[0])
    if 'actions' in sample:
        return sample['actions'].shape[-1]
    raise ValueError(f"No 'actions' field in {npz_files[0]}")


parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
args = parser.parse_args()

config = resolve_training_config({
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

view_strat = None
if hasattr(ds, 'get_camera_params'):
    params = ds.get_camera_params()
    if params:
        view_strat = SingleViewStrategy(
            params.get('H', 64), params.get('W', 64), params.get('focal', 130.0),
            {'eye': params['eye'], 'center': params['center'], 'up': params['up']})
elif hasattr(ds, 'H') and hasattr(ds, 'W'):
    view_strat = SingleViewStrategy(
        ds.H, ds.W, ds.focal,
        ds.get_camera_params() if hasattr(ds, 'get_camera_params') else None)

# -- data_dirs --
data_dirs = {"sequence": args.data_dir}

# -- Train --
trainer = UnifiedTrainer(model, view_strat, config=config)
trainer.train(data_dirs)
