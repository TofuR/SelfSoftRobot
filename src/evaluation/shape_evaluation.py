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
