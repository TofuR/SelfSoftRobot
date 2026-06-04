"""evaluate_shape.py — 统一 3D 形状评估脚本。

对所有模型类型（MSTNF, C-MSTNF, MS-SCNF, SDF, SkeletonSDF, FlowMatch）
计算统一的形状指标：Chamfer Distance, F-Score, Hausdorff Distance。

用法:
    python scripts/evaluation/evaluate_shape.py \
        --checkpoint train_log/.../best_model.pt \
        --data_dir data/seq_rr_3d

    # 交互式选择（不指定 --checkpoint）
    python scripts/evaluation/evaluate_shape.py
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_loader import load_model
from src.evaluation.query import query_density_field, query_sdf_field, query_pointcloud
from src.evaluation.shape_metrics import chamfer_distance, f_score, hausdorff_distance
from src.evaluation.surface_sampling import sample_gt_surface, model_output_to_pointcloud
from src.evaluation.projection_metrics import projection_f1
from src.utils.camera_system import MultiCameraSystem


# ── 交互式选择工具 ──────────────────────────────────────────────

def select_from_list(items, prompt, allow_custom=False):
    """交互式列表选择。"""
    if not items:
        print(f"  {prompt}: 无可用选项")
        return None
    print(f"\n{prompt}:")
    for i, item in enumerate(items):
        print(f"  [{i}] {os.path.relpath(item, PROJECT_ROOT) if item.startswith('/') else item}")
    if allow_custom:
        print(f"  [c] 自定义路径")
    choice = input("  > ").strip()
    if allow_custom and choice == 'c':
        path = input("  路径: ").strip()
        return path if path else None
    try:
        return items[int(choice)]
    except (ValueError, IndexError):
        return items[0]


def scan_checkpoints():
    """扫描所有可用 checkpoint。"""
    patterns = [
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'best_model.pt'),
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'final_model.pt'),
    ]
    ckpts = []
    for pat in patterns:
        ckpts.extend(glob.glob(pat, recursive=True))
    return sorted(set(ckpts))


def scan_data_dirs():
    """扫描 data/ 下有 npz 文件的目录。"""
    data_root = os.path.join(PROJECT_ROOT, 'data')
    dirs = []
    for d in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, d)
        if os.path.isdir(full) and glob.glob(os.path.join(full, '*.npz')):
            dirs.append(full)
    return dirs


# ── 核心评估逻辑 ──────────────────────────────────────────────────

def get_query_bounds(data):
    """从数据自动计算查询边界。"""
    positions = data["positions"]  # (T, 3, N)
    all_pos = positions.reshape(-1, 3)
    margin = 0.03
    bounds = []
    for dim in range(3):
        lo, hi = all_pos[:, dim].min(), all_pos[:, dim].max()
        bounds.extend([lo - margin, hi + margin])
    return tuple(bounds)


def evaluate_single_sample(model, model_type, data, t, window_size,
                           device, config, eval_cfg):
    """评估单个样本。

    Args:
        model:       加载的模型。
        model_type:  str，模型类型。
        data:        npz 数据 dict。
        t:           int，时间步索引。
        window_size: int，时序窗口长度。
        device:      torch device。
        config:      模型保存的配置 dict。
        eval_cfg:    评估配置 dict。

    Returns:
        dict: 各项指标值，或 None（样本无效时）。
    """
    actions = data["actions"]  # (T, action_dim)
    positions = data["positions"]  # (T, 3, N)
    radii = data.get("radii")

    # 构建 action_window
    start = max(0, t - window_size)
    aw = actions[start:t + 1]
    if len(aw) < window_size:
        pad = np.tile(aw[0:1], (window_size - len(aw), 1))
        aw = np.concatenate([pad, aw], axis=0)
    aw = aw[-window_size:]

    aw_tensor = torch.from_numpy(aw).float().unsqueeze(0).to(device)

    # 归一化 action
    norm_factor = config.get("norm_factor", 1.0)
    aw_tensor = aw_tensor / max(norm_factor, 1e-8)

    # GT 表面点云
    gt_pos = positions[t]
    if radii is not None:
        gt_rad = float(np.mean(radii[t])) if radii.ndim > 0 else float(radii)
    else:
        gt_rad = 0.015

    n_gt_points = eval_cfg.get("n_gt_points", 1000)
    gt_pc = sample_gt_surface(gt_pos, gt_rad, n_points=n_gt_points)

    if len(gt_pc) == 0:
        return None

    # 查询模型输出
    if model_type == "flowmatch":
        query_result = query_pointcloud(
            model, aw_tensor,
            n_points=eval_cfg.get("n_pred_points", 1000))

    elif model_type in ("mstnf", "cmstnf", "ms_scnf"):
        bounds = get_query_bounds(data)
        grid_res = eval_cfg.get("grid_res", 30)
        query_result = query_density_field(
            model, aw_tensor, bounds, grid_res, device)

    elif model_type in ("sdf", "skeleton_sdf"):
        bounds = get_query_bounds(data)
        grid_res = eval_cfg.get("grid_res", 30)
        query_result = query_sdf_field(
            model, aw_tensor, bounds, grid_res, device)

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 转换为点云
    pred_pc = model_output_to_pointcloud(model_type, query_result, model, eval_cfg)

    if len(pred_pc) == 0:
        return None

    # 计算指标
    thresholds = eval_cfg.get("fscore_thresholds", [0.005, 0.01, 0.02])

    result = {
        "chamfer_distance": chamfer_distance(pred_pc, gt_pc),
        "hausdorff_distance": hausdorff_distance(pred_pc, gt_pc),
    }
    for tau in thresholds:
        result[f"f_score_{int(tau*1000)}mm"] = f_score(pred_pc, gt_pc, tau)

    # 投影 F1：3D 点云投影到相机视角 vs GT 图像（惩罚扇形扩散）
    if "images" in data and "camera_params" in data:
        try:
            cam_sys = MultiCameraSystem.from_npz(data)
            n_views = cam_sys.n_views
            dilation = eval_cfg.get("projection_dilation", 1)
            proj_results = []
            for v in range(n_views):
                gt_img = data["images"][t, v]  # (H, W)
                res = projection_f1(pred_pc, gt_img, cam_sys.cameras[v],
                                    dilation=dilation)
                proj_results.append(res)
            # 多视角平均
            result["proj_precision"] = float(np.mean([r['precision'] for r in proj_results]))
            result["proj_recall"] = float(np.mean([r['recall'] for r in proj_results]))
            result["proj_f1"] = float(np.mean([r['f1'] for r in proj_results]))
            result["proj_iou"] = float(np.mean([r['iou'] for r in proj_results]))
        except Exception:
            pass  # 相机参数不完整时静默跳过

    return result


def evaluate_model(checkpoint_path, data_dir, eval_cfg=None):
    """主评估函数：加载模型，遍历数据，计算指标。"""
    eval_cfg = eval_cfg or {}
    device = torch.device(eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model_info = load_model(checkpoint_path, device=device)
    model = model_info["model"]
    model_type = model_info["model_type"]
    config = model_info.get("saved_config", {})
    window_size = config.get("window_size", 20)

    print(f"\n{'='*50}")
    print(f"Shape Evaluation Report")
    print(f"  Model: {type(model).__name__} ({model_type})")
    print(f"  Checkpoint: {os.path.relpath(checkpoint_path, PROJECT_ROOT)}")

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    n_eval_samples = eval_cfg.get("n_eval_samples", 0)

    if n_eval_samples > 0 and len(npz_files) > n_eval_samples:
        indices = np.linspace(0, len(npz_files) - 1, n_eval_samples, dtype=int)
        npz_files = [npz_files[i] for i in indices]

    print(f"  Data: {os.path.relpath(data_dir, PROJECT_ROOT)} ({len(npz_files)} samples)")
    print(f"{'='*50}")

    all_results = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        T = len(data["actions"])
        t = T // 2

        try:
            result = evaluate_single_sample(
                model, model_type, data, t, window_size,
                device, config, eval_cfg)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"  Warning: failed {os.path.basename(npz_path)}: {e}")

    if not all_results:
        print("  No valid samples evaluated.")
        return {}

    n_points_gt = eval_cfg.get("n_gt_points", 1000)
    print(f"  Points: {n_points_gt} (pred) vs {n_points_gt} (GT)")
    print(f"{'='*50}")

    report = {"model": type(model).__name__, "model_type": model_type,
              "data": data_dir, "n_samples": len(all_results)}
    metrics_keys = all_results[0].keys()

    for key in sorted(metrics_keys):
        values = [r[key] for r in all_results]
        mean, std = np.mean(values), np.std(values)
        report[key] = {"mean": float(mean), "std": float(std)}

        if "f_score" in key or key.startswith("proj_"):
            print(f"  {key:20s}: {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {key:20s}: {mean:.5f} ± {std:.5f}  (m)")

    print(f"{'='*50}")
    return report


def main():
    parser = argparse.ArgumentParser(description="统一 3D 形状评估")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型 checkpoint 路径（不指定则交互选择）")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="数据目录路径（不指定则交互选择）")
    parser.add_argument("--n_gt_points", type=int, default=1000)
    parser.add_argument("--n_pred_points", type=int, default=1000)
    parser.add_argument("--fscore_thresholds", type=str, default="0.005,0.01,0.02")
    parser.add_argument("--density_threshold", type=float, default=0.5)
    parser.add_argument("--grid_res", type=int, default=30)
    parser.add_argument("--n_eval_samples", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 路径")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        ckpts = scan_checkpoints()
        if ckpts:
            checkpoint = select_from_list(ckpts, "选择 checkpoint", allow_custom=True)
        if checkpoint is None:
            print("未选择 checkpoint，退出")
            return

    data_dir = args.data_dir
    if data_dir is None:
        dirs = scan_data_dirs()
        if dirs:
            data_dir = select_from_list(dirs, "选择数据目录", allow_custom=True)
        if data_dir is None:
            print("未选择数据目录，退出")
            return

    eval_cfg = {
        "n_gt_points": args.n_gt_points,
        "n_pred_points": args.n_pred_points,
        "fscore_thresholds": [float(t) for t in args.fscore_thresholds.split(",")],
        "density_threshold": args.density_threshold,
        "grid_res": args.grid_res,
        "n_eval_samples": args.n_eval_samples,
    }
    if args.device:
        eval_cfg["device"] = args.device

    report = evaluate_model(checkpoint, data_dir, eval_cfg)

    if args.output and report:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
