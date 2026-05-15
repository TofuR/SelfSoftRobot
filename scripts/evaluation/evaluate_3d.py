"""evaluate_3d.py — 3D 几何评估脚本（支持 MS-SCNF 和其他模型）。

评估指标：
  - Mean Node Error: 所有人节点平均 L2 误差
  - Endpoint Error: 末端节点 L2 误差
  - Curve Smoothness: 预测骨架的平滑度
  - Chamfer Distance: 点云双向最近邻距离

用法:
  # 评估 MS-SCNF 模型
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
      --model_type ms_scnf \
      --checkpoint train_log/train_ms_scnf/001/phase2/model/best_model.pt \
      --data_dir data/sequence_data_3d

  # 仅评估 Phase 1 骨架
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
      --model_type ms_scnf \
      --checkpoint train_log/train_ms_scnf/001/phase1/model/skeleton_best.pt \
      --data_dir data/sequence_data_3d \
      --phase 1
"""

import os
import sys
import argparse
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CUDA_DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_DEVICE)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.training.metrics_3d import chamfer_distance, endpoint_error, mean_node_error, curve_smoothness
from src.data.dataset import SoftSequenceDataset


def evaluate_ms_scnf(checkpoint, data_dir, phase, device):
    """评估 MS-SCNF 模型。"""
    from src.models.model_ms_scnf import MSSCNFModel
    from config.params import load_config

    train_cfg = load_config("training")
    ms_cfg = train_cfg.get("ms_scnf", {})

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    ds = SoftSequenceDataset(
        data_dir, seq_len=train_cfg["temporal"]["window_size"],
        file_list=all_files, return_3d=True,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    model = MSSCNFModel(
        action_dim=ds.action_dim,
        window_size=train_cfg["temporal"]["window_size"],
        n_scales=train_cfg["temporal"]["n_scales"],
        hidden_dim=train_cfg["temporal"]["hidden_dim"],
        d_filter=train_cfg["model"]["d_filter"],
        n_freqs=train_cfg["model"]["n_freqs"],
        n_coarse=ms_cfg.get("n_coarse", 4),
        n_medium=ms_cfg.get("n_medium", 10),
        n_fine=ms_cfg.get("n_fine", 31),
        deform_n_freqs=train_cfg["canonical"]["deform_n_freqs"],
    ).to(device)

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    if phase == 1:
        model.temporal.load_state_dict(state_dict['temporal'])
        model.skeleton_head.load_state_dict(state_dict['skeleton_head'])
    else:
        model.load_state_dict(state_dict)
    model.eval()

    all_metrics = {'mne': [], 'epe': [], 'smooth': [], 'cd': []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            seq = batch[0].to(device)
            positions = batch[2].to(device)
            gt = positions.permute(0, 2, 1)  # (B, N, 3)

            pred_dict = model.predict_skeleton(seq)
            pred = pred_dict['fine']

            all_metrics['mne'].append(mean_node_error(pred, gt).item())
            all_metrics['epe'].append(endpoint_error(pred, gt).item())
            all_metrics['smooth'].append(curve_smoothness(pred).item())
            all_metrics['cd'].append(chamfer_distance(pred, gt).item())

    print(f"\n{'='*50}")
    print(f"  Model: MS-SCNF (Phase {phase})")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Data: {data_dir} ({len(ds)} samples)")
    print(f"{'='*50}")
    for name, vals in all_metrics.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        print(f"  {name:>8s}: {mean_val:.6f} ± {std_val:.6f}")
    print(f"{'='*50}")

    return {k: np.mean(v) for k, v in all_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="3D 几何评估")
    parser.add_argument("--model_type", type=str, default="ms_scnf",
                        choices=["ms_scnf"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/sequence_data_3d")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model_type == "ms_scnf":
        evaluate_ms_scnf(args.checkpoint, args.data_dir, args.phase, device)


if __name__ == "__main__":
    main()
