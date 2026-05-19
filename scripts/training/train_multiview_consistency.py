"""train_multiview_consistency.py -- Multi-view consistency training via UnifiedTrainer.

Extends Plan A (multi-view rendering + depth supervision) with cross-view
consistency and reprojection constraints.

Usage:
    # CMSTNF multi-view consistency training
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk

    # CMSTNF multi-view consistency training (with depth)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk --depth

    # Override parameters
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview_consistency.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk \
        --lr 1e-4 --n_epochs 300 --w_consist 0.1 --w_reproj 0.2

Supported models: mstnf, cmstnf
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from config.params import load_config
from src.utils.config_utils import resolve_config
from src.training.trainer_unified import UnifiedTrainer
from src.training.view_strategy import MultiViewStrategy


def create_model(model_type, action_dim, config):
    """Create model instance by type."""
    temp_cfg = config["temporal"]
    model_cfg = config["model"]

    if model_type == "mstnf":
        from src.models.model_mstnf import MSTNFModel
        return MSTNFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
        )
    elif model_type == "cmstnf":
        from src.models.model_cmstnf import CMSTNFModel
        return CMSTNFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
            d_filter=model_cfg["d_filter"],
            n_freqs=model_cfg["n_freqs"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Multi-view consistency training (Plan B)")
    parser.add_argument("--model", type=str, default="mstnf",
                        choices=["mstnf", "cmstnf"])
    parser.add_argument("--data_dir", type=str, default="data/exp7_multiview")
    parser.add_argument("--depth", action="store_true", help="Enable depth supervision")
    parser.add_argument("--depth-guided", action="store_true",
                        help="Enable depth-guided sampling (coarse-to-fine)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--w_depth", type=float, default=None, help="Depth loss weight")
    parser.add_argument("--w_consist", type=float, default=None, help="Cross-view consistency weight")
    parser.add_argument("--w_reproj", type=float, default=None, help="Reprojection consistency weight")
    parser.add_argument("--n_rays_per_view", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    defaults = load_config("training")
    config = resolve_config(defaults, {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "optimization.batch_size": args.batch_size,
        "multiview.w_depth": args.w_depth,
        "multiview.w_consist": args.w_consist,
        "multiview.w_reproj": args.w_reproj,
        "multiview.n_rays_per_view": args.n_rays_per_view,
        "multiview.n_samples": args.n_samples,
        "multiview.use_depth_guided_sampling": args.depth_guided,
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    temp_cfg = config["temporal"]
    mv_cfg = config["multiview"]

    # -- Load dataset to get cam_system and action_dim --
    from src.data.dataset_multiview_depth import MultiViewDepthDataset

    full_ds = MultiViewDepthDataset(
        args.data_dir,
        seq_len=temp_cfg["window_size"],
        return_depth=args.depth,
        return_pairs=True,
    )
    cam_system = full_ds.cam_system
    n_views = full_ds.n_views
    action_dim = full_ds.action_dim

    print(f"\nData: {args.data_dir}")
    print(f"  Views: {n_views}, Action dim: {action_dim}")
    cam_system.summary()

    # -- Create model --
    model = create_model(args.model, action_dim, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}, Params: {n_params:,}")

    # -- ViewStrategy: MultiView with consistency + reprojection --
    view_strat = MultiViewStrategy(
        cam_system, with_depth=args.depth,
        with_consistency=True, with_reprojection=True)

    print("  ViewStrategy: MultiView + Consistency + Reprojection")
    print(f"    w_consist={mv_cfg.get('w_consist', 0.05):.3f}, "
          f"w_reproj={mv_cfg.get('w_reproj', 0.1):.3f}")

    # -- data_dirs --
    data_dirs = {"sequence": args.data_dir}

    # -- Train --
    trainer = UnifiedTrainer(model, view_strat, config=config)
    trainer.train(data_dirs)


if __name__ == "__main__":
    main()
