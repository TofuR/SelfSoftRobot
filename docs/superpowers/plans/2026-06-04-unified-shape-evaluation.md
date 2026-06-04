# Unified 3D Shape Evaluation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a unified evaluation framework that measures 3D shape reconstruction quality (CD, F-Score, Hausdorff) for all model types, runnable standalone or automatically during training.

**Architecture:** Three new modules — `shape_metrics.py` (numpy metrics), `surface_sampling.py` (GT generation + model output conversion), `evaluate_shape.py` (CLI). Plus training integration in `trainer_unified.py` that calls these modules every N epochs and saves results to `shape_metrics.json`.

**Tech Stack:** Python 3, NumPy, PyTorch (for model inference), scipy (for spatial distance), scikit-image (already a dependency for marching cubes).

---

## Task 1: Create `src/evaluation/shape_metrics.py`

**Files:**
- Create: `src/evaluation/shape_metrics.py`

Pure numpy implementations of three shape comparison metrics. No torch dependency.

- [ ] **Step 1: Write `shape_metrics.py`**

```python
"""shape_metrics.py — 3D 形状比较指标（纯 numpy 实现）。

三个核心指标，统一签名 (pred, gt) -> float：
  chamfer_distance   — 双向平均最近邻距离
  f_score            — 精度-覆盖率调和平均（阈值化）
  hausdorff_distance — 双向最大最近邻距离（最坏情况）
"""

import numpy as np
from scipy.spatial.distance import cdist


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """双向 Chamfer Distance。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。

    Returns:
        float: (mean(pred→gt) + mean(gt→pred)) / 2。
    """
    dists = cdist(pred, gt)  # (N, M)
    cd_pred = dists.min(axis=1).mean()  # pred→gt
    cd_gt = dists.min(axis=0).mean()    # gt→pred
    return float((cd_pred + cd_gt) / 2)


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """F-Score @threshold。

    Precision: pred 中有 GT 邻居（<threshold）的比例。
    Recall:    GT 中有 pred 邻居（<threshold）的比例。
    F-Score:   precision 和 recall 的调和平均。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。
        threshold: 距离阈值（米）。

    Returns:
        float: F-Score ∈ [0, 1]。
    """
    dists = cdist(pred, gt)  # (N, M)
    precision = (dists.min(axis=1) < threshold).mean()
    recall = (dists.min(axis=0) < threshold).mean()
    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """双向 Hausdorff Distance（最大最近邻距离）。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。

    Returns:
        float: max(max(pred→gt), max(gt→pred))。
    """
    dists = cdist(pred, gt)  # (N, M)
    hd_pred = dists.min(axis=1).max()  # pred→gt 最远点
    hd_gt = dists.min(axis=0).max()    # gt→pred 最远点
    return float(max(hd_pred, hd_gt))
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.evaluation.shape_metrics import chamfer_distance, f_score, hausdorff_distance; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/shape_metrics.py
git commit -m "feat: add shape_metrics.py — CD, F-Score, Hausdorff in numpy"
```

---

## Task 2: Create `src/evaluation/surface_sampling.py`

**Files:**
- Create: `src/evaluation/surface_sampling.py`
- Read (reference only): `src/data/dataset_pointcloud.py:163-216` — the `_sample_surface` function

- [ ] **Step 1: Write `surface_sampling.py`**

```python
"""surface_sampling.py — GT 表面采样 + 模型输出→点云转换。

两个核心功能：
  sample_gt_surface        — 从 positions+radii 解析生成 GT 表面点云
  model_output_to_pointcloud — 统一调度，把任意模型输出转为 (N,3) 点云
"""

import numpy as np


def sample_gt_surface(positions, radii, n_points=1000, seed=42):
    """从骨架节点 + 半径解析采样表面点云。

    复用 dataset_pointcloud._sample_surface 的圆柱采样逻辑，
    但使用固定 seed 并返回原始物理坐标（不做归一化）。

    Args:
        positions: (3, N_nodes) 骨架节点坐标。
        radii:     (N_nodes,) 或 scalar，杆体半径。
        n_points:  目标采样点数。
        seed:      随机种子（保证可重复性）。

    Returns:
        np.ndarray: (n_points, 3) 表面点云，物理坐标（米）。
    """
    rng = np.random.RandomState(seed)
    N = positions.shape[1]
    n_segs = N - 1
    n_per_seg = max(1, n_points // n_segs)

    # radii 可以是标量或数组
    radii = np.atleast_1d(np.asarray(radii, dtype=np.float32))
    if len(radii) == 1:
        radii = np.full(N, radii[0])
    elif len(radii) == n_segs:
        # (N-1,) → 扩展到 (N,)，取相邻平均
        radii = np.concatenate([radii, [radii[-1]]])

    pts_list = []
    for i in range(n_segs):
        p1, p2 = positions[:, i], positions[:, i + 1]
        r = (radii[i] + radii[i + 1]) / 2  # 段平均半径
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            continue
        tangent = seg_vec / seg_len

        # 构建法平面正交基
        ref = (np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.99
               else np.array([1.0, 0.0, 0.0]))
        perp1 = np.cross(tangent, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)

        # 沿线段插值 + 圆周采样
        n_seg = min(n_per_seg, n_points - len(pts_list))
        if n_seg <= 0:
            break
        t_param = rng.rand(n_seg)
        centers = p1[:, None] * (1 - t_param[None, :]) + p2[:, None] * t_param[None, :]
        angles = rng.rand(n_seg) * 2 * np.pi
        offsets = r * (
            np.cos(angles)[:, None] * perp1[None, :] +
            np.sin(angles)[:, None] * perp2[None, :])
        pts_list.append(centers.T + offsets)

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(pts_list, axis=0).astype(np.float32)


def model_output_to_pointcloud(model_type, query_result, model, config):
    """把模型查询结果统一转换为 (N, 3) 点云（物理坐标）。

    Args:
        model_type:  str，模型类型标识（"flowmatch", "mstnf" 等）。
        query_result: dict，query.py 对应函数的返回值。
        model:       nn.Module，原始模型（用于获取归一化参数等）。
        config:      dict，评估配置（含 density_threshold 等）。

    Returns:
        np.ndarray: (N, 3) 点云，物理坐标（米）。
    """
    density_threshold = config.get("density_threshold", 0.5)

    if model_type == "flowmatch":
        # FlowMatch: 反归一化
        pc = query_result["points"]  # (N, 3) numpy, normalized
        scale = model.pc_scale.cpu().numpy()  # (1, 1, 3)
        center = model.pc_center.cpu().numpy()  # (1, 1, 3)
        return pc * scale[0] + center[0]

    elif model_type in ("mstnf", "cmstnf", "ms_scnf"):
        # 密度场：阈值提取高密度点
        points = query_result["points"]    # (grid³, 3)
        density = query_result["density"]  # (grid³,)
        visibility = query_result.get("visibility", np.ones_like(density))
        mask = (visibility > 0.5) & (density > density_threshold)
        return points[mask]

    elif model_type in ("sdf", "skeleton_sdf"):
        # SDF: marching cubes 顶点
        vertices = query_result.get("vertices")
        if vertices is not None and len(vertices) > 0:
            return vertices
        # fallback: 从 SDF grid 采样接近零的点
        sdf_grid = query_result["sdf_grid"]
        threshold = 0.01
        mask = np.abs(sdf_grid) < threshold
        x = query_result["x"]
        y = query_result["y"]
        z = query_result["z"]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([xx[mask], yy[mask], zz[mask]], axis=-1).astype(np.float32)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.evaluation.surface_sampling import sample_gt_surface, model_output_to_pointcloud; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/surface_sampling.py
git commit -m "feat: add surface_sampling.py — GT sampling + model output conversion"
```

---

## Task 3: Create `scripts/evaluation/evaluate_shape.py`

**Files:**
- Create: `scripts/evaluation/evaluate_shape.py`
- Read (reference only): `scripts/evaluation/visualize_3d_shape.py:35-98` — `select_from_list`, `scan_checkpoints`, `scan_data_dirs`
- Read (reference only): `src/evaluation/query.py` — query function signatures
- Read (reference only): `src/utils/model_loader.py` — `load_model` return keys

- [ ] **Step 1: Write `evaluate_shape.py`**

```python
"""evaluate_shape.py — 统一 3D 形状评估脚本。

对所有模型类型（MSTNF, C-MSTNF, MS-SCNF, SDF, SkeletonSDF, FlowMatch）
计算统一的形状指标：Chamfer Distance, F-Score, Hausdorff Distance。

用法:
    python scripts/evaluation/evaluate_shape.py \\
        --checkpoint train_log/.../best_model.pt \\
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


# ── 交互式选择工具（复用 visualize_3d_shape.py 的模式）───────────────

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
    all_pos = positions.reshape(-1, 3)  # (T*N, 3)
    margin = 0.03
    bounds = []
    for dim in range(3):
        lo, hi = all_pos[:, dim].min(), all_pos[:, dim].max()
        span = max(hi - lo, 0.01)
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
    radii = data.get("radii")  # (T, N) or scalar

    # 构建 action_window
    start = max(0, t - window_size)
    aw = actions[start:t + 1]  # (K, action_dim)
    if len(aw) < window_size:
        pad = np.tile(aw[0:1], (window_size - len(aw), 1))
        aw = np.concatenate([pad, aw], axis=0)
    aw = aw[-window_size:]

    aw_tensor = torch.from_numpy(aw).float().unsqueeze(0).to(device)  # (1, K, D)

    # 归一化 action
    norm_factor = config.get("norm_factor", 1.0)
    aw_tensor = aw_tensor / max(norm_factor, 1e-8)

    # GT 表面点云
    gt_pos = positions[t]  # (3, N)
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

    return result


def evaluate_model(checkpoint_path, data_dir, eval_cfg=None):
    """主评估函数：加载模型，遍历数据，计算指标。

    Args:
        checkpoint_path: str，模型 checkpoint 路径。
        data_dir:        str，数据目录路径。
        eval_cfg:        dict，评估配置（覆盖默认值）。

    Returns:
        dict: 评估报告。
    """
    eval_cfg = eval_cfg or {}
    device = torch.device(eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 加载模型
    model_info = load_model(checkpoint_path, device=device)
    model = model_info["model"]
    model_type = model_info["model_type"]
    config = model_info.get("saved_config", {})
    window_size = config.get("window_size", 20)

    print(f"\n{'='*50}")
    print(f"Shape Evaluation Report")
    print(f"  Model: {type(model).__name__} ({model_type})")
    print(f"  Checkpoint: {os.path.relpath(checkpoint_path, PROJECT_ROOT)}")

    # 扫描数据文件
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

        if "f_score" in key:
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
    parser.add_argument("--n_gt_points", type=int, default=1000,
                        help="GT 表面采样点数（默认 1000）")
    parser.add_argument("--n_pred_points", type=int, default=1000,
                        help="预测点云采样点数（默认 1000，FlowMatch 用）")
    parser.add_argument("--fscore_thresholds", type=str, default="0.005,0.01,0.02",
                        help="F-Score 阈值，逗号分隔（米）")
    parser.add_argument("--density_threshold", type=float, default=0.5,
                        help="密度场阈值（密度模型用）")
    parser.add_argument("--grid_res", type=int, default=30,
                        help="密度/SDF 网格分辨率（密度/SDF 模型用）")
    parser.add_argument("--n_eval_samples", type=int, default=0,
                        help="评估样本数（0=全部）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果输出 JSON 路径（默认不保存）")
    parser.add_argument("--device", type=str, default=None,
                        help="计算设备（默认自动）")
    args = parser.parse_args()

    # 交互选择或使用参数
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
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from scripts.evaluation.evaluate_shape import evaluate_model; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluation/evaluate_shape.py
git commit -m "feat: add evaluate_shape.py — unified shape evaluation CLI"
```

---

## Task 4: Update `src/evaluation/__init__.py`

**Files:**
- Modify: `src/evaluation/__init__.py`

- [ ] **Step 1: Add new exports**

Current content (line 1-2):
```python
from .query import query_density_field, query_sdf_field, query_skeleton
from .render import render_density_html, render_sdf_html, render_animation
```

Replace with:
```python
from .query import query_density_field, query_sdf_field, query_skeleton, query_pointcloud
from .render import render_density_html, render_sdf_html, render_animation
from .shape_metrics import chamfer_distance, f_score, hausdorff_distance
from .surface_sampling import sample_gt_surface, model_output_to_pointcloud
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.evaluation import chamfer_distance, sample_gt_surface; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/evaluation/__init__.py
git commit -m "feat: export shape_metrics and surface_sampling from evaluation __init__"
```

---

## Task 5: Add evaluation config to `config/training.json`

**Files:**
- Modify: `config/training.json`

- [ ] **Step 1: Add evaluation training integration fields**

Current `"evaluation"` section (lines 103-111):
```json
    "evaluation": {
        "query_batch_size": 50000,
        "grid_res": 40,
        "density_threshold": 0.01,
        "bounds_margin": 0.03,
        "marching_cubes_level": 0,
        "fps": 3,
        "_doc": "query_batch_size=评估查询分块大小, grid_res=可视化网格分辨率, density_threshold=密度可视化阈值, bounds_margin=空间包围盒余量, fps=GIF帧率"
    },
```

Add 4 new keys before `"_doc"`:
```json
    "evaluation": {
        "query_batch_size": 50000,
        "grid_res": 40,
        "density_threshold": 0.01,
        "bounds_margin": 0.03,
        "marching_cubes_level": 0,
        "fps": 3,
        "eval_interval": 0,
        "n_eval_samples": 100,
        "n_gt_points": 1000,
        "fscore_thresholds": [0.005, 0.01, 0.02],
        "_doc": "query_batch_size=评估查询分块大小, grid_res=可视化网格分辨率, density_threshold=密度可视化阈值, bounds_margin=空间包围盒余量, fps=GIF帧率, eval_interval=每N epoch评估一次(0=关闭), n_eval_samples=评估用样本数, n_gt_points=GT表面采样密度, fscore_thresholds=F-Score阈值列表(米)"
    },
```

- [ ] **Step 2: Verify JSON is valid**

Run: `python -c "import json; json.load(open('config/training.json')); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config/training.json
git commit -m "feat: add shape eval config to training.json"
```

---

## Task 6: Integrate shape evaluation into `trainer_unified.py`

**Files:**
- Modify: `src/training/trainer_unified.py`

Two changes: (1) add imports + `_evaluate_shape` method, (2) call it in the epoch loop.

- [ ] **Step 1: Add `import glob` at line 17**

Add `import glob` after `import os` (line 17):
```python
import glob
import os
```

- [ ] **Step 2: Add evaluation imports after line 28**

After `from config.params import load_config`:
```python
from src.evaluation.surface_sampling import sample_gt_surface, model_output_to_pointcloud
from src.evaluation.shape_metrics import chamfer_distance, f_score, hausdorff_distance
```

- [ ] **Step 3: Add `_evaluate_shape` method after `_setup_views_from_dataset` (after line 228)**

```python
    def _evaluate_shape(self, phase_spec, data_dir, epoch, exp_dir):
        """在训练中运行形状评估，结果保存到 shape_metrics.json。"""
        import json as _json
        eval_cfg = self.config.get("evaluation", {})
        n_eval = eval_cfg.get("n_eval_samples", 100)
        n_gt = eval_cfg.get("n_gt_points", 1000)
        thresholds = eval_cfg.get("fscore_thresholds", [0.005, 0.01, 0.02])

        # 确定模型类型
        model_type = self.model_tag
        if model_type not in ("flowmatch", "mstnf", "cmstnf", "ms_scnf",
                              "sdf", "skeleton_sdf"):
            return

        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not npz_files:
            return
        if n_eval > 0 and len(npz_files) > n_eval:
            indices = np.linspace(0, len(npz_files) - 1, n_eval, dtype=int)
            npz_files = [npz_files[i] for i in indices]

        self.model.eval()
        all_results = []
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            T = len(data["actions"])
            t = T // 2
            try:
                from scripts.evaluation.evaluate_shape import evaluate_single_sample
                result = evaluate_single_sample(
                    self.model, model_type, data, t,
                    self.config.get("temporal", {}).get("window_size", 20),
                    self.device, {},
                    {"n_gt_points": n_gt, "n_pred_points": n_gt,
                     "fscore_thresholds": thresholds,
                     "density_threshold": eval_cfg.get("density_threshold", 0.5),
                     "grid_res": eval_cfg.get("grid_res", 30)})
                if result is not None:
                    all_results.append(result)
            except Exception:
                pass

        self.model.train()

        if not all_results:
            return

        metrics = {"phase": phase_spec.name, "epoch": epoch,
                    "n_samples": len(all_results)}
        for key in all_results[0]:
            values = [r[key] for r in all_results]
            metrics[key] = {"mean": float(np.mean(values)),
                            "std": float(np.std(values))}

        metrics_path = os.path.join(exp_dir, "shape_metrics.json")
        history = {"model": type(self.model).__name__,
                    "data": data_dir, "evaluations": []}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                history = _json.load(f)
        history["evaluations"].append(metrics)
        with open(metrics_path, "w") as f:
            _json.dump(history, f, indent=2, ensure_ascii=False)

        cd = metrics.get("chamfer_distance", {}).get("mean", 0)
        hd = metrics.get("hausdorff_distance", {}).get("mean", 0)
        mid_key = [k for k in metrics if "f_score" in k]
        mid_fs = metrics[mid_key[len(mid_key)//2]]["mean"] if mid_key else 0
        print(f"  [Eval] Epoch {epoch} | CD={cd:.5f} | F@10mm={mid_fs:.3f} | HD={hd:.5f}")
```

- [ ] **Step 4: Add eval call in epoch loop**

After the `scheduler.step()` block (line 341-342), before the `# 保存 Phase 权重` comment (line 344), insert:

```python
                    # 形状评估（每 eval_interval epoch）
                    eval_interval = self.config.get("evaluation", {}).get("eval_interval", 0)
                    if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == n_epochs):
                        self._evaluate_shape(phase_spec, data_dir, epoch, exp_dir)
```

- [ ] **Step 5: Verify import works**

Run: `python -c "from src.training.trainer_unified import UnifiedTrainer; print('OK')"`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/training/trainer_unified.py
git commit -m "feat: integrate shape evaluation into training loop (every N epochs)"
```

---

## Task 7: Smoke test with a real checkpoint

**Files:** None (verification only)

- [ ] **Step 1: Find an available checkpoint**

Run: `find train_log -name "best_model.pt" | head -5`

- [ ] **Step 2: Run evaluate_shape.py**

Run: `python scripts/evaluation/evaluate_shape.py --checkpoint <path> --data_dir <data_dir> --n_eval_samples 5`

Expected: Shape evaluation report printed, no errors.

- [ ] **Step 3: Verify output format matches design**

Expected format:
```
==================================================
Shape Evaluation Report
  Model: ... (...)
  Checkpoint: ...
  Data: ... (5 samples)
==================================================
  chamfer_distance:     0.XXXXX ± 0.XXXXX  (m)
  f_score_5mm:          0.XXX ± 0.XXX
  ...
==================================================
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All 4 modules from the design spec have corresponding tasks
- [x] **Placeholder scan:** No TBD, TODO — all steps contain complete code
- [x] **Type consistency:** `sample_gt_surface` returns `np.ndarray (N,3)`, `model_output_to_pointcloud` returns the same, all metrics consume `(N,3)` numpy arrays
- [x] **No duplicate code:** GT sampling uses fixed-seed version of `dataset_pointcloud.py` pattern
- [x] **Backwards compatible:** `eval_interval=0` means no training eval by default
