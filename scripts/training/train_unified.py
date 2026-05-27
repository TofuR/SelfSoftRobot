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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch

from config.params import load_config
from src.utils.config_utils import resolve_config
from src.training.trainer_unified import UnifiedTrainer
from src.rendering.view_strategy import (
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
            n_coarse=ms_cfg["n_coarse"],
            n_medium=ms_cfg["n_medium"],
            n_fine=ms_cfg["n_fine"],
            deform_n_freqs=canon_cfg["deform_n_freqs"],
            skeleton_mode=ms_cfg["skeleton_mode"],
            fourier_n_freq=ms_cfg["fourier_n_freq"],
            bspline_n_ctrl=ms_cfg["bspline_n_ctrl"],
            catmullrom_n_ctrl=ms_cfg["catmullrom_n_ctrl"],
        )

    elif model_type == "sdf":
        from src.models.model_sdf import TemporalSDFModel
        sdf_cfg = config.get("sdf", {})
        return TemporalSDFModel(
            action_dim=action_dim,
            window_size=temp_cfg["window_size"],
            n_scales=temp_cfg["n_scales"],
            hidden_dim=temp_cfg["hidden_dim"],
            w_sdf=sdf_cfg["w_sdf"],
            w_normal=sdf_cfg["w_normal"],
            w_eikonal=sdf_cfg["w_grad"],
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
            skeleton_mode=ms_cfg["skeleton_mode"],
            rod_radius=ms_cfg["rod_radius"],
            w_skel_fine=ms_cfg["w_skeleton_fine"],
            w_skel_medium=ms_cfg["w_skeleton_medium"],
            w_skel_coarse=ms_cfg["w_skeleton_coarse"],
            w_skel_smooth=ms_cfg.get("w_smooth", 0.01),
            w_sdf=sdf_cfg["w_sdf"],
            w_normal=sdf_cfg["w_normal"],
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
        "optimization.num_workers": args.num_workers,
        "ms_scnf.skeleton_mode": args.skeleton_mode,
        "multiview.chunk_size": args.chunk_size,
        "multiview.n_rays_per_view": args.n_rays,
        "multiview.n_samples": args.n_samples,
        "sdf.n_surface": args.n_surface,
        "sdf.w_sdf": args.w_sdf,
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
        active_losses = set(rendering_phase.active_losses)
        ds = create_dataset(rendering_phase.dataset_type, args.data_dir, config, rendering_phase)

        if hasattr(ds, 'cam_system') and ds.cam_system.n_views >= 2:
            view_strat = MultiViewStrategy(
                ds.cam_system,
                with_depth="depth" in active_losses or args.depth,
                with_consistency="consist" in active_losses or args.consistency,
                with_reprojection="reproj" in active_losses or args.consistency)
            print(f"  ViewStrategy: MultiView ({ds.cam_system.n_views} views)")
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
                n_epochs_per_phase[p.name] = can_cfg["phase1_epochs"]
            else:
                n_epochs_per_phase[p.name] = args.n_epochs or config["optimization"]["n_epochs"]

    # ── 训练 ──
    skip_phases = None
    if args.phase is not None and spec.is_two_phase:
        skip_phases = [p.name for i, p in enumerate(spec.phases) if i + 1 != args.phase]
        print(f"  Skipping phases: {skip_phases}")

    trainer = UnifiedTrainer(model, view_strat, config=config,
                             model_tag=args.model)
    trainer.train(data_dirs, n_epochs_per_phase=n_epochs_per_phase,
                  skip_phases=skip_phases)


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
    parser.add_argument("--skeleton_mode", type=str, default=None,
                        choices=["point", "fourier", "bspline", "catmullrom"],
                        help="骨架参数化方式")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2],
                        help="只运行指定阶段（跳过其他阶段）")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader 进程数")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="渲染查询分块大小")
    parser.add_argument("--n_rays", type=int, default=None,
                        help="每视角采样射线数")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="每射线采样点数")
    parser.add_argument("--n_surface", type=int, default=None,
                        help="SDF 表面采样点数")
    parser.add_argument("--w_sdf", type=float, default=None,
                        help="SDF loss 权重")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
