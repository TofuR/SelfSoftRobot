"""train_depth_cmstnf.py — Depth-supervised CMSTNF 两阶段训练入口。

Usage:
    # 完整训练 (Phase 1 + Phase 2 with depth supervision)
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_depth_cmstnf.py

    # 仅 Phase 2（depth-supervised deformation）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_depth_cmstnf.py --phase 2 \
        --exp_dir train_log/train_depth_cmstnf/001 \
        --canonical_path train_log/train_depth_cmstnf/001/phase1/model/canonical_best.pt

    # 调整深度损失权重
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_depth_cmstnf.py --depth_weight 0.5

    # 关闭深度引导采样（只用深度损失）
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_depth_cmstnf.py --no_guided_sampling

未指定的参数自动从 src/config/training.json 读取。
"""

import os
import sys

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import torch
from src.config.params import load_config
from src.utils.config_utils import resolve_config
from src.training.trainer_depth_cmstnf import DepthCMSTNFTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
parser.add_argument("--data_dir", type=str, default="data/sequence_data")
parser.add_argument("--canonical_data_dir", type=str, default="data/canonical_data")
parser.add_argument("--canonical_path", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--depth_weight", type=float, default=0.1,
                    help="Depth loss weight (default: 0.1)")
parser.add_argument("--no_guided_sampling", action="store_true",
                    help="Disable depth-guided ray sampling")
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--n_epochs", type=int, default=None)
parser.add_argument("--phase1_epochs", type=int, default=None)
parser.add_argument("--phase2_epochs", type=int, default=None)
parser.add_argument("--deform_lr", type=float, default=None)
args = parser.parse_args()

defaults = load_config("training")
config = resolve_config(defaults, {
    "optimization.lr": args.lr,
    "optimization.n_epochs": args.n_epochs,
    "canonical.phase1_epochs": args.phase1_epochs,
    "canonical.phase2_epochs": args.phase2_epochs,
    "canonical.deform_lr": args.deform_lr,
})

print(f"Device: {device}")
trainer = DepthCMSTNFTrainer(
    device=device, config=config,
    depth_weight=args.depth_weight,
    use_guided_sampling=not args.no_guided_sampling,
)

if args.phase == 1:
    trainer.train_phase1(exp_dir=args.exp_dir, data_dir=args.canonical_data_dir)
elif args.phase == 2:
    trainer.train_phase2(exp_dir=args.exp_dir, canonical_path=args.canonical_path,
                         data_dir=args.data_dir)
else:
    trainer.train(data_dir=args.data_dir, canonical_data_dir=args.canonical_data_dir)
