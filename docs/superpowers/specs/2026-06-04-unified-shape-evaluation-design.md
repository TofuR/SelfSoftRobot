# Unified 3D Shape Evaluation Framework

**Date:** 2026-06-04
**Status:** Approved

## Problem

The project has 5+ model types (MSTNF, C-MSTNF, MS-SCNF, TemporalSDF, SkeletonSDF, FlowMatch) that all predict 3D robot shape from actuator inputs. There is no unified quantitative evaluation:

- `evaluate_3d.py` only covers MS-SCNF skeleton metrics (MNE, EPE)
- No evaluation for FlowMatch point cloud models
- No standard metrics (F-Score, Hausdorff) for any model
- Cannot compare models against each other

Training loss is not a reliable shape quality indicator (the fan-shape issue proved this).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Common representation | Surface point cloud | Universal — all model outputs can be converted to points |
| GT source | Analytical from positions+radii | Exact, deterministic, available in every npz with 3D data |
| Coordinate system | Physical (meters) | Metrics in meters enable physical interpretation |
| Architecture | Functional (not OOP) | Model differences handled by one dispatch function, not 4 classes |
| First version scope | Core metrics + text report | CD, F-Score, Hausdorff; extensible later |

## Architecture

```
scripts/evaluation/evaluate_shape.py     (CLI entry point)
src/evaluation/shape_metrics.py          (metric functions: CD, F-Score, Hausdorff)
src/evaluation/surface_sampling.py       (GT sampling + model output conversion)
src/evaluation/query.py                  (existing — model querying)
src/utils/model_loader.py                (existing — model loading)
```

## Module 1: `src/evaluation/shape_metrics.py`

Pure numpy metric functions. All share the signature `(pred: np.ndarray, gt: np.ndarray) -> float`.

### `chamfer_distance(pred, gt) -> float`
Bidirectional mean nearest-neighbor distance.
Reuse logic from `src/losses/pointcloud_losses.py` but in numpy.

### `f_score(pred, gt, threshold) -> float`
Harmonic mean of precision and recall at distance threshold.
- Precision: fraction of pred points within `threshold` of any GT point
- Recall: fraction of GT points within `threshold` of any pred point
Reported at multiple thresholds: 5mm, 10mm, 20mm.

### `hausdorff_distance(pred, gt) -> float`
Bidirectional maximum nearest-neighbor distance. Detects worst-case outliers.

## Module 2: `src/evaluation/surface_sampling.py`

### `sample_gt_surface(positions, radii, n_points=1000, seed=42) -> np.ndarray`
Analytical cylindrical surface sampling from skeleton nodes + radii.
- Input: `positions` (3, N_nodes), `radii` (N_nodes,)
- Output: (n_points, 3) in physical coordinates (meters)
- Fixed seed for reproducibility
- Reuse sampling logic from `src/data/dataset_pointcloud.py`

### `model_output_to_pointcloud(model_type, query_result, model, config) -> np.ndarray`
Dispatch function converting model-specific query output to (N, 3) point cloud in physical coordinates.

| Model type | Query output | Conversion |
|------------|-------------|------------|
| `flowmatch` | (N, 3) normalized points | Denormalize: `points * pc_scale + pc_center` |
| `mstnf` / `cmstnf` / `ms_scnf` | density grid + visibility | Threshold density > threshold, extract high-density points |
| `sdf` / `skeleton_sdf` | SDF grid + mesh (vertices, faces) | Use marching cubes vertices directly |

Note: MS-SCNF skeleton is NOT the shape output. Final output is the density field, same as MSTNF/CMSTNF. Skeleton is intermediate visualization only.

## Module 3: `scripts/evaluation/evaluate_shape.py`

### Pipeline
```
1. Load model (model_loader auto-detects type)
2. Scan data directory for npz files
3. For each sample:
   a. Load positions (3, N_nodes) + radii (N_nodes,) from npz
   b. Generate GT surface: sample_gt_surface(positions, radii)
   c. Build action_window from actions history
   d. Query model: query_density_field / query_sdf_field / query_pointcloud
   e. Convert output: model_output_to_pointcloud(...)
   f. Compute: CD, F-Score @thresholds, Hausdorff
4. Aggregate mean ± std across all samples
5. Print report
```

### CLI Interface
```bash
python scripts/evaluation/evaluate_shape.py \
    --checkpoint <path> \        # Model checkpoint (required, or interactive)
    --data_dir <path> \          # Data directory with npz files (required)
    --n_gt_points 1000 \         # GT surface sampling density (default: 1000)
    --n_pred_points 1000 \       # Prediction sampling density (default: 1000)
    --fscore_thresholds 0.005,0.01,0.02 \  # F-Score thresholds in meters
    --density_threshold 0.5 \    # Density threshold for field models (default: 0.5)
    --device cuda:0              # Device (default: auto)
```

### Output Format
```
==================================================
Shape Evaluation Report
  Model: FlowMatchPointCloud (flowmatch)
  Checkpoint: train_log/.../best_model.pt
  Data: data/seq_rr_3d (437 samples)
  Points: 1000 (pred) vs 1000 (GT)
==================================================
  Chamfer Distance:    0.00523 ± 0.00217  (m)
  F-Score @5mm:        0.847 ± 0.071
  F-Score @10mm:       0.932 ± 0.043
  F-Score @20mm:       0.981 ± 0.019
  Hausdorff Distance:  0.0312 ± 0.0124  (m)
==================================================
```

### Interactive Mode
If `--checkpoint` is omitted, interactively list available experiments and checkpoints (reuse pattern from `visualize_3d_shape.py`).

## Module 4: Training Integration

### Evaluation Timing

During `UnifiedTrainer.train()`, shape evaluation runs every `eval_interval` epochs and at the final epoch. This provides a shape quality curve alongside the training loss curve.

### Implementation

In `trainer_unified.py`, add a method and call it in the epoch loop:

```python
# New method on UnifiedTrainer:
def _evaluate_shape(self, phase_spec, data_dir, epoch, exp_dir):
    """Run shape evaluation on a subset of data and save results."""

# Called in train() epoch loop:
eval_interval = self.config.get("evaluation", {}).get("eval_interval", 0)
if eval_interval > 0 and (epoch % eval_interval == 0 or epoch == n_epochs):
    self._evaluate_shape(phase_spec, data_dir, epoch, exp_dir)
```

### Result Storage

Results are appended to `<exp_dir>/shape_metrics.json`:

```json
{
  "model": "FlowMatchPointCloud",
  "data": "data/seq_rr_3d",
  "evaluations": [
    {
      "phase": "flowmatch",
      "epoch": 50,
      "n_samples": 100,
      "chamfer_distance": {"mean": 0.00523, "std": 0.00217},
      "f_score_5mm": {"mean": 0.847, "std": 0.071},
      "f_score_10mm": {"mean": 0.932, "std": 0.043},
      "f_score_20mm": {"mean": 0.981, "std": 0.019},
      "hausdorff_distance": {"mean": 0.0312, "std": 0.0124}
    }
  ]
}
```

### Performance

- Uses a subset of data (`n_eval_samples`, default 100) to avoid slowing training
- Entire evaluation runs under `torch.no_grad()`
- Prints one-line summary per eval: `[Eval] Epoch 50 | CD=0.00523 | F@10mm=0.932 | HD=0.0312`

### Configuration

Added to `config/training.json` under `"evaluation"`:

```json
"evaluation": {
    "eval_interval": 50,
    "n_eval_samples": 100,
    "n_gt_points": 1000,
    "fscore_thresholds": [0.005, 0.01, 0.02],
    "_doc": "eval_interval=每N epoch评估一次(0=关闭), n_eval_samples=评估用样本数, n_gt_points=GT表面采样密度"
}
```

When `eval_interval=0` (default), no evaluation runs during training — backwards compatible.

## Files to Create

| File | Lines (est.) | Purpose |
|------|-------------|---------|
| `src/evaluation/shape_metrics.py` | ~100 | CD, F-Score, Hausdorff in numpy |
| `src/evaluation/surface_sampling.py` | ~120 | GT sampling + model output conversion |
| `scripts/evaluation/evaluate_shape.py` | ~200 | CLI entry point, pipeline, report |

## Files to Modify

| File | Change | Lines changed |
|------|--------|---------------|
| `src/training/trainer_unified.py` | Add `_evaluate_shape()` method + call in epoch loop | ~40 |
| `config/training.json` | Add eval_interval, n_eval_samples, n_gt_points, fscore_thresholds to `evaluation` section | ~6 |

## Files to Reuse (no changes)

- `src/evaluation/query.py` — model querying
- `src/utils/model_loader.py` — model loading
- `src/losses/pointcloud_losses.py` — reference for CD logic

## Not in Scope (v1)

- Per-action CD breakdown
- Temporal consistency metrics
- HTML/plot visualization of metrics
- Normal consistency metric
- Skeleton metrics (MNE, EPE) — kept in existing `evaluate_3d.py`
- Automatic model comparison table
