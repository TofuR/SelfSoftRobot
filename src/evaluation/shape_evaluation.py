"""训练中形状评估工具。

从 UnifiedTrainer._evaluate_shape() 提取出来，
保持训练器代码简洁。
"""

import glob
import json
import os

import numpy as np


# 支持形状评估的模型类型白名单
_EVAL_SUPPORTED_MODELS = (
    "flowmatch", "mstnf", "cmstnf", "ms_scnf", "sdf", "skeleton_sdf",
)

# 支持骨架评估的模型类型
_SKELETON_EVAL_MODELS = (
    "spatial_sequence", "pc_spatial",
)


def evaluate_shape_during_training(model, model_tag, config, device,
                                    phase_name, data_dir, epoch, exp_dir):
    """在训练中运行形状评估，结果追加到 shape_metrics.json。

    Args:
        model: 神经场模型。
        model_tag: 模型标签字符串（如 "flowmatch"）。
        config: 训练配置 dict。
        device: torch device。
        phase_name: 当前阶段名。
        data_dir: 评估数据目录。
        epoch: 当前 epoch 编号。
        exp_dir: 实验日志目录。
    """
    if model_tag not in _EVAL_SUPPORTED_MODELS:
        return

    eval_cfg = config.get("evaluation", {})
    n_eval = eval_cfg.get("n_eval_samples", 100)
    n_gt = eval_cfg.get("n_gt_points", 1000)
    thresholds = eval_cfg.get("fscore_thresholds", [0.005, 0.01, 0.02])

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        return
    if n_eval > 0 and len(npz_files) > n_eval:
        indices = np.linspace(0, len(npz_files) - 1, n_eval, dtype=int)
        npz_files = [npz_files[i] for i in indices]

    model.eval()
    all_results = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        T = len(data["actions"])
        t = T // 2
        try:
            from scripts.evaluation.evaluate_shape import evaluate_single_sample
            result = evaluate_single_sample(
                model, model_tag, data, t,
                config.get("temporal", {}).get("window_size", 20),
                device, {},
                {"n_gt_points": n_gt, "n_pred_points": n_gt,
                 "fscore_thresholds": thresholds,
                 "density_threshold": eval_cfg.get("density_threshold", 0.5),
                 "grid_res": eval_cfg.get("grid_res", 30)})
            if result is not None:
                all_results.append(result)
        except Exception:
            pass

    model.train()

    if not all_results:
        return

    metrics = {"phase": phase_name, "epoch": epoch,
               "n_samples": len(all_results)}
    for key in all_results[0]:
        values = [r[key] for r in all_results]
        metrics[key] = {"mean": float(np.mean(values)),
                        "std": float(np.std(values))}

    metrics_path = os.path.join(exp_dir, "shape_metrics.json")
    history = {"model": type(model).__name__,
               "data": data_dir, "evaluations": []}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            history = json.load(f)
    history["evaluations"].append(metrics)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    cd = metrics.get("chamfer_distance", {}).get("mean", 0)
    hd = metrics.get("hausdorff_distance", {}).get("mean", 0)
    mid_key = [k for k in metrics if "f_score" in k]
    mid_fs = metrics[mid_key[len(mid_key)//2]]["mean"] if mid_key else 0
    pf1 = metrics.get("proj_f1", {}).get("mean", 0)
    print(f"  [Eval] Epoch {epoch} | CD={cd:.5f} | F@10mm={mid_fs:.3f} | "
          f"HD={hd:.5f} | ProjF1={pf1:.3f}")


def evaluate_skeleton_during_training(model, model_tag, config, device,
                                      phase_name, data_dir, epoch, exp_dir):
    """在训练中运行骨架评估，结果追加到 skeleton_metrics.json。

    评估绝对误差(mm)、相对误差(%arm, %radius)、逐节点分布(base/mid/tip)。

    Args:
        model: 骨架预测模型（SpatialSequence / PCSpatial）。
        model_tag: 模型标签（如 "spatial_sequence"）。
        config: 训练配置 dict。
        device: torch device。
        phase_name: 当前阶段名。
        data_dir: 评估数据目录。
        epoch: 当前 epoch 编号。
        exp_dir: 实验日志目录。
    """
    if model_tag not in _SKELETON_EVAL_MODELS:
        return

    import torch as _torch
    from src.evaluation.query import query_skeleton_direct
    from src.training.metrics_3d import evaluate_skeleton as _eval_skel

    eval_cfg = config.get("evaluation", {})
    n_eval = eval_cfg.get("n_eval_samples", 5)
    arm_length = 0.5
    rod_radius = 0.015
    window_size = config.get("temporal", {}).get("window_size", 40)

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        return
    if n_eval > 0 and len(npz_files) > n_eval:
        indices = np.linspace(0, len(npz_files) - 1, n_eval, dtype=int)
        npz_files = [npz_files[i] for i in indices]

    # 获取归一化参数
    norm_factor = 1.0
    if hasattr(model, 'action_norm_factor'):
        nf = model.action_norm_factor
        norm_factor = nf.item() if isinstance(nf, _torch.Tensor) else float(nf)

    model.eval()
    agg_keys = ['mean_node_err', 'endpoint_err', 'max_node_err', 'chamfer_distance',
                'mean_pct_arm', 'endpoint_pct_arm', 'mean_pct_radius', 'endpoint_pct_radius']
    all_metrics = {k: [] for k in agg_keys}
    per_node_sum = None
    n_windows = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        actions = data['actions']
        positions = data['positions']  # (T, 3, 31)
        T = len(actions)

        for start in range(0, T - window_size - 1, max(1, (T - window_size) // 5)):
            end = start + window_size
            act = actions[start:end] / norm_factor
            aw = _torch.FloatTensor(act).unsqueeze(0).to(device)
            gt = positions[end].T  # (31, 3)

            with _torch.no_grad():
                if hasattr(model, 'forward_predictive'):
                    pred = model.forward_predictive({"action_window": aw}).squeeze(0)
                else:
                    pred = model(aw).squeeze(0)

            center = model.pc_center.cpu().squeeze().numpy()
            scale = model.pc_scale.cpu().squeeze().numpy()
            pred_world = pred.cpu().numpy() * scale + center

            pred_t = _torch.from_numpy(pred_world).float().unsqueeze(0)
            gt_t = _torch.from_numpy(gt).float().unsqueeze(0)
            r = _eval_skel(pred_t, gt_t, arm_length, rod_radius)

            for k in agg_keys:
                all_metrics[k].append(r[k])

            if per_node_sum is None:
                per_node_sum = r['per_node_err'].copy()
            else:
                per_node_sum += r['per_node_err']
            n_windows += 1

    model.train()

    if n_windows == 0:
        return

    # 汇总
    metrics = {"phase": phase_name, "epoch": epoch, "n_windows": n_windows}
    for k in agg_keys:
        vals = all_metrics[k]
        metrics[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "unit": "m" if "pct" not in k else "%"}

    # 逐节点三段
    if per_node_sum is not None:
        pn = per_node_sum / n_windows
        N = len(pn)
        n_base, n_mid = N // 3, 2 * N // 3
        metrics["per_node"] = {
            "base_mm": float(np.mean(pn[:n_base]) * 1000),
            "mid_mm": float(np.mean(pn[n_base:n_mid]) * 1000),
            "tip_mm": float(np.mean(pn[n_mid:]) * 1000),
            "all": [float(v * 1000) for v in pn.tolist()],
        }

    # 追加到 JSON
    metrics_path = os.path.join(exp_dir, "skeleton_metrics.json")
    history = {"model": type(model).__name__,
               "data": data_dir, "arm_length_mm": arm_length * 1000,
               "rod_radius_mm": rod_radius * 1000, "evaluations": []}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            history = json.load(f)
    history["evaluations"].append(metrics)
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # 打印摘要
    mean_mm = metrics["mean_node_err"]["mean"] * 1000
    ep_mm = metrics["endpoint_err"]["mean"] * 1000
    max_mm = metrics["max_node_err"]["mean"] * 1000
    arm_pct = metrics["mean_pct_arm"]["mean"]
    ep_arm_pct = metrics["endpoint_pct_arm"]["mean"]
    r_pct = metrics["mean_pct_radius"]["mean"]
    ep_r_pct = metrics["endpoint_pct_radius"]["mean"]

    line = (f"  [Skeleton] Epoch {epoch} | "
            f"Mean={mean_mm:.2f}mm ({arm_pct:.1f}%arm, {r_pct:.0f}%R) | "
            f"Endpoint={ep_mm:.2f}mm ({ep_arm_pct:.1f}%arm) | Max={max_mm:.2f}mm")
    if "per_node" in metrics:
        pn = metrics["per_node"]
        line += f"\n    Base={pn['base_mm']:.1f} Mid={pn['mid_mm']:.1f} Tip={pn['tip_mm']:.1f} mm"
    print(line)
