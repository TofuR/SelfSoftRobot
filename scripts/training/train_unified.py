"""train_unified.py — 统一训练入口（支持全部 5 个模型）。

用法:
    # MSTNF (单阶段, rendering)
    python scripts/training/train_unified.py --model mstnf --data_dir data/sequence_data

    # CMSTNF (两阶段, rendering)
    python scripts/training/train_unified.py --model cmstnf --data_dir data/sequence_data \\
        --canonical_data_dir data/canonical_data

    # MS-SCNF (两阶段, skeleton+rendering)
    python scripts/training/train_unified.py --model ms_scnf --data_dir data/sequence_data_3d

    # SDF (单阶段, direct_3d)
    python scripts/training/train_unified.py --model sdf --data_dir data/seq_rr_3d

    # SkeletonSDF (两阶段, direct_3d)
    python scripts/training/train_unified.py --model skeleton_sdf --data_dir data/seq_rr_3d

    # 多视角 + 一致性
    python scripts/training/train_unified.py --model cmstnf --data_dir data/seq_rz_c2_sk \\
        --multiview --depth --consistency
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
from src.training.view_strategy import (
    SingleViewStrategy, MultiViewStrategy,
)


def create_model(model_type, action_dim, config):
    """根据模型类型创建模型实例。"""
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

    elif model_type == "ms_scnf":
        from src.models.model_ms_scnf import MSSCNFModel
        ms_cfg = config.get("ms_scnf", {})
        canon_cfg = config.get("canonical", {})
        return MSSCNFModel(
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
        )

    elif model_type == "sdf":
        from src.models.model_sdf import TemporalSDFModel
        sdf_cfg = config.get("sdf", {})
        return TemporalSDFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
            w_sdf=sdf_cfg.get("w_sdf", 3e3),
            w_normal=sdf_cfg.get("w_normal", 1e2),
            w_eikonal=sdf_cfg.get("w_eikonal", 5e1),
        )

    elif model_type == "skeleton_sdf":
        from src.models.model_skeleton_sdf import SkeletonSDFModel
        ms_cfg = config.get("ms_scnf", {})
        sdf_cfg = config.get("sdf", {})
        return SkeletonSDFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
            skeleton_mode=ms_cfg.get("skeleton_mode", "bspline"),
            rod_radius=ms_cfg.get("rod_radius", 0.015),
            w_skel_fine=ms_cfg.get("w_skeleton_fine", 1.0),
            w_skel_medium=ms_cfg.get("w_skeleton_medium", 0.3),
            w_skel_coarse=ms_cfg.get("w_skeleton_coarse", 0.1),
            w_skel_smooth=ms_cfg.get("w_smooth", 0.01),
            w_sdf=sdf_cfg.get("w_sdf", 3e3),
            w_normal=sdf_cfg.get("w_normal", 10.0),
            w_eikonal=sdf_cfg.get("w_eikonal", 50.0),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def detect_action_dim(model_type, data_dir, config):
    """从数据中探测 action_dim。"""
    import numpy as np
    import glob

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        raise FileNotFoundError(f"No data in {data_dir}")

    sample = np.load(npz_files[0])
    if 'actions' in sample:
        return sample['actions'].shape[-1]
    raise ValueError(f"No 'actions' field in {npz_files[0]}")


def train(args):
    defaults = load_config("training")
    config = resolve_config(defaults, {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "optimization.batch_size": args.batch_size,
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 探测 action_dim ──
    action_dim = detect_action_dim(args.model, args.data_dir, config)

    # ── 创建模型 ──
    model = create_model(args.model, action_dim, config).to(device)
    spec = model.training_spec

    print(f"\n模型: {args.model}")
    print(f"  Action dim: {action_dim}")
    print(f"  Phases: {[p.name for p in spec.phases]}")
    print(f"  is_two_phase: {spec.is_two_phase}")

    # ── ViewStrategy：仅 rendering 模式需要 ──
    view_strat = None
    needs_rendering = any(p.supervision_mode == "rendering" for p in spec.phases)

    if needs_rendering:
        from src.training.dataset_factory import create_dataset
        from src.training.spec import PhaseSpec

        rendering_phase = next(p for p in spec.phases if p.supervision_mode == "rendering")
        ds = create_dataset(rendering_phase.dataset_type, args.data_dir, config, rendering_phase)

        if args.multiview and hasattr(ds, 'cam_system') and ds.cam_system.n_views >= 2:
            from src.utils.camera_system import MultiCameraSystem
            view_strat = MultiViewStrategy(
                ds.cam_system, with_depth=args.depth,
                with_consistency=args.consistency,
                with_reprojection=args.consistency)
            print(f"  ViewStrategy: MultiView{' + Consistency' if args.consistency else ''}")
        elif hasattr(ds, 'get_camera_params'):
            params = ds.get_camera_params()
            if params:
                cam = params
                view_strat = SingleViewStrategy(
                    cam.get('H', 64), cam.get('W', 64), cam.get('focal', 130.0),
                    {'eye': cam['eye'], 'center': cam['center'], 'up': cam['up']})
                print(f"  ViewStrategy: SingleView")
        elif hasattr(ds, 'H') and hasattr(ds, 'W'):
            view_strat = SingleViewStrategy(
                ds.H, ds.W, ds.focal,
                ds.get_camera_params() if hasattr(ds, 'get_camera_params') else None)
            print(f"  ViewStrategy: SingleView (from dataset)")

        if view_strat is None:
            print("  WARNING: rendering phase needs ViewStrategy but couldn't create one")

    else:
        print("  ViewStrategy: None (non-rendering model)")

    # ── data_dirs ──
    data_dirs = {"sequence": args.data_dir}
    if args.canonical_data_dir:
        data_dirs["canonical"] = args.canonical_data_dir

    # ── epochs per phase ──
    n_epochs_per_phase = None
    if spec.is_two_phase:
        can_cfg = config.get("canonical", {})
        n_epochs_per_phase = {}
        for p in spec.phases:
            if p.name in ("canonical", "skeleton"):
                n_epochs_per_phase[p.name] = can_cfg.get("phase1_epochs", 50)
            else:
                n_epochs_per_phase[p.name] = args.n_epochs or config["optimization"]["n_epochs"]

    # ── 训练 ──
    trainer = UnifiedTrainer(model, view_strat, config=config)
    trainer.train(data_dirs, n_epochs_per_phase=n_epochs_per_phase)


def main():
    parser = argparse.ArgumentParser(description="统一训练入口（支持 5 个模型）")
    parser.add_argument("--model", type=str, default="mstnf",
                        choices=["mstnf", "cmstnf", "ms_scnf", "sdf", "skeleton_sdf"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--canonical_data_dir", type=str, default=None,
                        help="canonical 数据目录（两阶段模型需要）")
    parser.add_argument("--multiview", action="store_true",
                        help="启用多视角")
    parser.add_argument("--depth", action="store_true",
                        help="启用深度监督")
    parser.add_argument("--consistency", action="store_true",
                        help="启用跨视角一致性约束")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
