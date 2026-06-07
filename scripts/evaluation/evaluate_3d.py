"""evaluate_3d.py — 3D 几何评估脚本。

支持模型：
  - MS-SCNF (ms_scnf) — 骨架条件密度场
  - SpatialSequence (spatial_sequence) — GRU 空间序列骨架
  - PCSpatial (pc_spatial) — 预测-修正骨架

评估指标：
  绝对误差: Mean node error, Endpoint error, Max node error
  相对误差: % of arm length (500mm), % of rod radius (15mm)
  逐节点分析: base/mid/tip 三段误差分布（--per_node）

用法:
  # 评估 MS-SCNF 模型
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
      --model_type ms_scnf \
      --checkpoint train_log/train_ms_scnf/001/phase2/model/best_model.pt \
      --data_dir data/sequence_data_3d

  # 评估 SpatialSequence / PCSpatial
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
      --model_type spatial_sequence \
      --checkpoint train_log/spatialsequence/exp_20260606_1/phase_spatial/model/best_model.pt \
      --data_dir data/seq_rz_c2_sk

  # 逐节点误差分析 + 保存图表
  python scripts/evaluation/evaluate_3d.py \
      --model_type spatial_sequence \
      --checkpoint train_log/spatialsequence/.../best_model.pt \
      --data_dir data/seq_rz_c2_sk --per_node
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

from src.training.metrics_3d import (
    chamfer_distance, endpoint_error, mean_node_error,
    curve_smoothness, evaluate_skeleton,
)
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


def evaluate_skeleton_model(checkpoint, data_dir, device, per_node=False):
    """评估骨架模型（SpatialSequence / PCSpatial）。"""
    from src.utils.model_loader import load_model
    from src.evaluation.query import query_skeleton_direct

    info = load_model(checkpoint, data_dir=data_dir, device=device)
    model = info['model']
    model_type = info['model_type']
    norm_factor = info['norm_factor']
    window_size = info['window_size']

    # 物理参数
    arm_length = 0.5    # m
    rod_radius = 0.015   # m

    # 加载数据
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    n_seqs = len(all_files)
    n_windows = 0

    # 累积指标
    agg = {
        'mean_node_err': [], 'endpoint_err': [], 'max_node_err': [],
        'chamfer_distance': [], 'mean_pct_arm': [], 'endpoint_pct_arm': [],
        'mean_pct_radius': [], 'endpoint_pct_radius': [],
    }
    per_node_all = None

    for f in tqdm(all_files, desc="Evaluating"):
        d = np.load(f)
        actions = d['actions']
        positions = d['positions']  # (T, 3, 31)
        T = len(actions)

        for start in range(0, T - window_size - 1, 20):
            end = start + window_size
            act = actions[start:end] / norm_factor
            aw = torch.FloatTensor(act).unsqueeze(0).to(device)
            gt = positions[end].T  # (31, 3)

            with torch.no_grad():
                if hasattr(model, 'forward_predictive'):
                    pred = model.forward_predictive({"action_window": aw}).squeeze(0)
                else:
                    pred = model(aw).squeeze(0)

            # 反归一化到世界坐标
            center = model.pc_center.cpu().squeeze().numpy()
            scale = model.pc_scale.cpu().squeeze().numpy()
            pred_world = pred.cpu().numpy() * scale + center

            pred_t = torch.from_numpy(pred_world).float().unsqueeze(0)
            gt_t = torch.from_numpy(gt).float().unsqueeze(0)

            r = evaluate_skeleton(pred_t, gt_t, arm_length, rod_radius)

            for k in agg:
                agg[k].append(r[k])

            if per_node:
                if per_node_all is None:
                    per_node_all = r['per_node_err'].copy()
                else:
                    per_node_all += r['per_node_err']

            n_windows += 1

    # 汇总
    avg = {k: np.mean(v) for k, v in agg.items()}

    # 打印结果
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Skeleton Evaluation: {model_type}")
    print(f"  Checkpoint: {os.path.relpath(checkpoint)}")
    print(f"  Data: {data_dir} ({n_seqs} sequences, {n_windows} windows)")
    print(f"  Arm: {arm_length*1000:.0f}mm, Radius: {rod_radius*1000:.0f}mm")
    print(f"{sep}")
    print(f"  Absolute Error:")
    print(f"    Mean node:    {avg['mean_node_err']*1000:>8.2f} mm")
    print(f"    Endpoint:     {avg['endpoint_err']*1000:>8.2f} mm")
    print(f"    Max node:     {avg['max_node_err']*1000:>8.2f} mm")
    print(f"")
    print(f"  Relative to Arm ({arm_length*1000:.0f}mm):")
    print(f"    Mean:         {avg['mean_pct_arm']:>8.2f}%")
    print(f"    Endpoint:     {avg['endpoint_pct_arm']:>8.2f}%")
    print(f"")
    print(f"  Relative to Radius ({rod_radius*1000:.0f}mm):")
    print(f"    Mean:         {avg['mean_pct_radius']:>8.1f}%  "
          f"(≈ {avg['mean_node_err']/rod_radius:.1f} × radius)")
    print(f"    Endpoint:     {avg['endpoint_pct_radius']:>8.1f}%  "
          f"(≈ {avg['endpoint_err']/rod_radius:.1f} × radius)")
    print(f"")
    print(f"  Chamfer Distance: {avg['chamfer_distance']:.6f}")

    # 逐节点分析
    if per_node and per_node_all is not None:
        per_node_avg = per_node_all / n_windows  # (N,)
        N = len(per_node_avg)
        n_base = N // 3
        n_mid = 2 * N // 3

        print(f"\n  Per-node Analysis (base → tip):")
        print(f"    Base  (node  1-{n_base:>2d}):  {np.mean(per_node_avg[:n_base])*1000:.2f} mm  "
              f"({np.mean(per_node_avg[:n_base])/arm_length*100:.2f}% arm)")
        print(f"    Mid   (node {n_base+1:>2d}-{n_mid:>2d}):  {np.mean(per_node_avg[n_base:n_mid])*1000:.2f} mm  "
              f"({np.mean(per_node_avg[n_base:n_mid])/arm_length*100:.2f}% arm)")
        print(f"    Tip   (node {n_mid+1:>2d}-{N:>2d}):  {np.mean(per_node_avg[n_mid:])*1000:.2f} mm  "
              f"({np.mean(per_node_avg[n_mid:])/arm_length*100:.2f}% arm)")

        # 保存图表
        try:
            from src.utils.skeleton_viz import plot_error_along_arm
            out_dir = os.path.join("output", "evaluation")
            os.makedirs(out_dir, exist_ok=True)
            plot_error_along_arm(
                per_node_avg, title=f"{model_type} Node-wise Error",
                save_path=os.path.join(out_dir, f"{model_type}_per_node.png"),
                show=False, arm_length=arm_length, rod_radius=rod_radius,
            )
            print(f"\n  Chart saved: {out_dir}/{model_type}_per_node.png")
        except Exception as e:
            print(f"\n  Chart failed: {e}")

    print(f"{sep}")

    return avg


def main():
    parser = argparse.ArgumentParser(description="3D 几何评估")
    parser.add_argument("--model_type", type=str, default="ms_scnf",
                        choices=["ms_scnf", "spatial_sequence", "pc_spatial"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/sequence_data_3d")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2])
    parser.add_argument("--per_node", action="store_true",
                        help="逐节点误差分析 + 保存图表")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model_type == "ms_scnf":
        evaluate_ms_scnf(args.checkpoint, args.data_dir, args.phase, device)
    elif args.model_type in ("spatial_sequence", "pc_spatial"):
        evaluate_skeleton_model(args.checkpoint, args.data_dir, device,
                                per_node=args.per_node)


if __name__ == "__main__":
    main()
