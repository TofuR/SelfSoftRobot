"""train_unified.py — 统一训练入口。

用法:
    # MSTNF + 多视角 (单阶段)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
        --model mstnf --data_dir data/seq_rz_c2_sk --multiview

    # CMSTNF + 多视角 + 一致性 (两阶段)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
        --model cmstnf --data_dir data/seq_rz_c2_sk --multiview --depth --consistency \
        --canonical_data_dir data/canonical_data

    # CMSTNF + 单视角 (两阶段)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
        --model cmstnf --data_dir data/sequence_data \
        --canonical_data_dir data/canonical_data
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
from src.data.dataset_multiview_depth import MultiViewDepthDataset


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
    elif model_type == "smooth_cmstnf":
        raise ValueError("smooth_cmstnf 已归档到 docs/archived/smooth_cmstnf/，如需使用请恢复文件")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(args):
    defaults = load_config("training")
    config = resolve_config(defaults, {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "optimization.batch_size": args.batch_size,
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 探测数据集 ──
    temp_cfg = config["temporal"]
    ds = MultiViewDepthDataset(
        args.data_dir, seq_len=temp_cfg["window_size"],
        return_depth=args.depth, return_pairs=True)
    action_dim = ds.action_dim
    cam_system = ds.cam_system
    n_views = ds.n_views
    print(f"数据: {args.data_dir}, 视角: {n_views}, 动作维度: {action_dim}")
    cam_system.summary()

    # ── 模型 ──
    model = create_model(args.model, action_dim, config).to(device)
    spec = model.training_spec
    print(f"\n模型: {args.model}")
    print(f"  Phases: {[p.name for p in spec.phases]}")
    print(f"  is_two_phase: {spec.is_two_phase}")

    # ── ViewStrategy ──
    if args.multiview and n_views >= 2:
        view_strat = MultiViewStrategy(
            cam_system, with_depth=args.depth,
            with_consistency=args.consistency,
            with_reprojection=args.consistency)
        print(f"  ViewStrategy: MultiView{' + Consistency' if args.consistency else ''}")
    else:
        cam = cam_system.cameras[0]
        view_strat = SingleViewStrategy(
            cam['H'], cam['W'], cam['focal'],
            {'eye': cam['eye'], 'center': cam['center'], 'up': cam['up']})
        print("  ViewStrategy: SingleView")

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
            if p.name == "canonical":
                n_epochs_per_phase["canonical"] = can_cfg.get("phase1_epochs", 50)
            else:
                n_epochs_per_phase[p.name] = args.n_epochs or config["optimization"]["n_epochs"]

    # ── 训练 ──
    trainer = UnifiedTrainer(model, view_strat, config=config)
    trainer.train(data_dirs, n_epochs_per_phase=n_epochs_per_phase)


def main():
    parser = argparse.ArgumentParser(description="统一训练入口 (解耦架构)")
    parser.add_argument("--model", type=str, default="mstnf",
                        choices=["mstnf", "cmstnf", "smooth_cmstnf"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--canonical_data_dir", type=str, default=None,
                        help="canonical 数据目录（两阶段模型需要）")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--consistency", action="store_true",
                        help="启用跨视角一致性约束")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
