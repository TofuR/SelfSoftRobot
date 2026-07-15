# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **新接手本项目?先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)** — 5 分钟接手指南:当前真实状态、别破坏的不变量、怎么跑最新控制/规划、诚实边界。本文件(CLAUDE.md)是完整规范,HANDOFF 是快速定向。

## Project Overview

SelfSoftRobot is a research project implementing **neural field-based 3D self-modeling for soft robots**. The robot learns to predict its own 3D shape from 2D camera observations conditioned on actuator inputs, using NeRF-inspired volume rendering as the training signal.

The project builds on the FBV-SM (Field-Based Vision Soft Manipulation) codebase from Hu et al. 2025, extending it from rigid arms to soft continuum arms simulated via PyElastica.

The project follows **two data routes**:
- **(A) Simulation** — PyElastica 3D arm + PyVista rendering (the original main line; all `train_unified` / MS-SCNF / SkeletonSDF work below stays valid).
- **(B) Real-world** (newer, calibration-free) — a physical 1-DOF two-segment soft arm driven by a TwinCAT PLC + syringe motors, single Intel RealSense camera, **no camera calibration**: the 2D image skeleton `[col,row,0]` is used directly as `state` (predicted output is de-normalized back to pixels), and an NDI 6DOF tracker provides end-effector ground truth in mm for metric validation. The full real-data pipeline is documented in [`docs/real_data/workflow.md`](docs/real_data/workflow.md).

## Running the Code

### Environment Setup
```bash
pip install -r requirements.txt
```
Key dependencies: PyTorch 2.6, PyElastica (via `elastica`), PyVista, OpenCV. Requires CUDA GPU for training.

### Data Collection
```bash
# Soft arm data collection (PyElastica)
python scripts/data_collection/collect.py

# With 3D ground truth + depth
python scripts/data_collection/collect.py --3d --depth

# Canonical data (zero actions, for two-phase models)
python scripts/data_collection/collect.py --action-x zero --action-y zero
```

### Training
All models use the unified entry point:
```bash
# MSTNF (single-phase, rendering)
python scripts/training/train_unified.py --model mstnf --data_dir data/sequence_data

# C-MSTNF (two-phase, rendering)
python scripts/training/train_unified.py --model cmstnf --data_dir data/sequence_data \
    --canonical_data_dir data/canonical_data

# MS-SCNF (two-phase, skeleton+rendering)
python scripts/training/train_unified.py --model ms_scnf --data_dir data/seq_rr_3d

# TemporalSDF (single-phase, direct 3D)
python scripts/training/train_unified.py --model sdf --data_dir data/seq_rr_3d

# SkeletonSDF (two-phase, direct 3D)
python scripts/training/train_unified.py --model skeleton_sdf --data_dir data/seq_rr_3d

# Multi-view + depth
python scripts/training/train_unified.py --model cmstnf --data_dir data/exp7_multiview \
    --multiview --depth --consistency
```

Individual training scripts (`train_mstnf.py`, `train_cmstnf.py`, etc.) are thin wrappers around UnifiedTrainer.

### Evaluation & Visualization
```bash
python scripts/evaluation/evaluate_3d.py
python scripts/evaluation/visualize_predictions.py compare   # Side-by-side comparison
python scripts/evaluation/visualize_predictions.py animate   # GIF animation
python scripts/evaluation/visualize_3d_shape.py              # 3D SDF/mesh visualization
```

There is no formal test suite. Validation is done through notebooks and the evaluation scripts.

### Real-Data Pipeline (route B, calibration-free 2D)

Data prep → training → evaluation. Full workflow in [`docs/real_data/workflow.md`](docs/real_data/workflow.md).

```bash
# --- scripts/real/ : mask + skeleton prep ---
python scripts/real/masks_to_transition_npz.py   # white_on_blue mask → 2D skeleton [col,row,0] + tip_fix + action norm[0,1]
python scripts/real/clean_transition_npz.py      # static-segment consensus cleaning
python scripts/real/repair_masks.py              # mask-level repair (independent track), outputs masks_repaired/
python scripts/real/composite_frames.py          # original + mask + skeleton overlay
python scripts/real/compare_skeleton_methods.py  # 7-method end-effector corner comparison
python scripts/real/skeleton_to_shape.py         # node→shape baseline (skeleton + constant radius)

# --- scripts/training/ : unified entry, replaces old train_gt/open_loop_transition ---
python scripts/training/train_transition.py --mode gt         # ground-truth state
python scripts/training/train_transition.py --mode open_loop  # open-loop rollout

# --- scripts/evaluation/ : real-data metrics + visualization ---
python scripts/evaluation/eval_real_quant.py        # end-effector NDI mm + shape px + drift_by_k
python scripts/evaluation/visualize_real_overlay.py # model prediction overlaid on real photo
python scripts/evaluation/inspect_real_data.py      # skeleton grid diagnostics
```

Note: in the real-data pipeline `state` is in image pixels `[col,row,0]`; no camera matrix / intrinsic projection is used (no metric 3D or intrinsics in this calibration-free route). Whole-shape error is reported in px; end-effector error in both px and mm (mm via NDI affine self-calibration against GT `node0` px). Backing utilities: `src/utils/skeleton_2d.py` (2D skeleton + `_perpendicular_tip_fix`), `src/evaluation/transition_metrics.py` (rollout + `drift_by_k`), `src/evaluation/shape_metrics.py` (chamfer/hausdorff).

## Architecture

### Simulation Backend

- **PyElastica** (`elastica_env.py`): Soft continuum arm (Cosserat rod). Two modes: static `get_simulation_data_pair()` for independent episodes, and `ContinuousSoftArmEnv` for stateful sequential simulation. Renders via PyVista to binary images.
- **PyBullet** (reference only): The original rigid arm simulation code from the FBV-SM paper is preserved in `docs/ref/SelfSimRobot/` for reference.

### Source Layout (`src/`)

```
src/
  encoders/          # Temporal encoders (MultiScaleEMA, GammaLaguerre, TemporalGRU, TemporalTransformer, TemporalTCN)
  fields/            # Neural fields (CanonicalField, DeformationField, SkeletonConditionedDensity)
  heads/             # Skeleton regression heads (point/fourier/bspline/catmullrom)
  rendering/         # View strategies (SingleView, MultiView)
  evaluation/        # Query & render utilities for evaluation
  models/            # Model definitions
  training/          # Training infrastructure (UnifiedTrainer, PhaseStrategy, spec)
  data/              # Dataset classes
  utils/             # Shared utilities
  config/            # CLI argument definitions (args.py)
```

### Model Architecture (`src/models/`)

| Model | File | Key Idea |
|-------|------|----------|
| FBV_SM | `model.py` | Legacy base model from FBV-SM paper |
| MSTNF | `model_mstnf.py` | Multi-scale EMA for temporal action history |
| C-MSTNF | `model_cmstnf.py` | D-NeRF paradigm: separate canonical + deformation fields |
| MS-SCNF | `model_ms_scnf.py` | Predicts 3D skeleton at multiple scales, skeleton-conditioned density |
| TemporalSDF | `model_sdf.py` | SIREN coordinate encoding + direct 3D SDF supervision |
| SkeletonSDF | `model_skeleton_sdf.py` | Parametric skeleton + tubular SDF prior + SIREN residual |

ODE-CMSTNF and Smooth-CMSTNF are archived in `docs/archived/`.

Shared components:
- `layers.py`: `PositionalEncoder`, `ActuatorMLPEncoder`, `MLPDecoder`, `TemporalLSTMEncoder`
- `mixins.py`: Shared mixin classes for model composition

### Extracted Modules

- `src/encoders/multi_scale_ema.py`: `MultiScaleEMA` — multi-scale exponential moving average
- `src/encoders/gamma_laguerre.py`: `GammaLaguerreMemory` — Gamma distribution kernels with delayed peak
- `src/encoders/temporal_gru.py`: `TemporalGRU` — order-sensitive GRU over action window
- `src/encoders/temporal_transformer.py`: `TemporalTransformer` — self-attention with CLS token
- `src/encoders/temporal_tcn.py`: `TemporalTCN` — causal dilated 1D convolution
- `src/fields/`: `CanonicalField`, `DeformationField`, `SkeletonConditionedDensity`
- `src/heads/skeleton_heads.py`: 4 skeleton parameterizations (point/fourier/bspline/catmullrom), factory function `create_skeleton_head()`
- `src/rendering/view_strategy.py`: `SingleViewStrategy` / `MultiViewStrategy`
- `src/evaluation/`: `query.py` (model querying), `render.py` (visualization rendering)

### Training Infrastructure (`src/training/`)

All models use **declarative Spec-based training** via `UnifiedTrainer`:

- `spec.py`: `PhaseSpec` / `TrainingSpec` — declare training requirements per model
- `phase_strategy.py`: Interprets spec, manages freeze/unfreeze/forward per phase
- `trainer_unified.py`: `UnifiedTrainer` — combines PhaseStrategy + ViewStrategy
- `dataset_factory.py`: Creates dataset + collate function based on spec
- `base.py`: `BaseTrainer` — shared camera setup, rendering, validation loops (legacy)

Three supervision modes: `"rendering"` (volume rendering), `"direct_3d"` (SDF query), `"skeleton"` (skeleton regression).

### Data (`src/data/`)

- `dataset.py`: `SoftSequenceDataset` — loads action-image sequences, supports 3D positions + depth
- `dataset_sdf.py`: `SDFDataset` — 3D SDF supervision sampling (surface/near/off-surface points)
- `dataset_skeleton_sdf.py`: `SkeletonSDFDataset` — skeleton + SDF joint sampling
- `dataset_multiview.py`: `MultiViewDataset` — dual-view with 2D skeleton extraction (legacy)
- `dataset_multiview_depth.py`: `MultiViewDepthDataset` — multi-view + depth, auto-detects old/new npz format

### Configuration

- `config/training.json`: Shared training hyperparameters (all models)
- `config/camera.json`: Camera parameters
- `config/simulation.json`: Simulation parameters
- `config/params.py`: YAML-based config loading
- `src/config/args.py`: CLI argument definitions for unified training

### Key Utilities (`src/utils/`)

- `rendering.py`: Volume rendering (OM rendering, ray sampling, depth-guided sampling)
- `camera.py`: Camera ray generation (`get_rays`)
- `camera_system.py`: `MultiCameraSystem` — multi-camera management, projection/reprojection
- `model_loader.py`: Auto-detect model type and load checkpoint
- `sdf_utils.py`: GT SDF generation (analytical tubular structure computation)
- `config_utils.py`: CLI parameter override + config merge utilities
- `experiment.py`: Experiment directory management + GIF saving
- `skeleton_2d.py`: 2D skeleton extraction from binary images
- `skeleton_viz.py`: 3D skeleton visualization and animation
- `visualization.py`: General visualization utilities

### Data Layout

```
data/
  seq_zz/            # canonical (both dims zero)
  seq_zz_3d/         # canonical + 3D
  seq_rr/            # sequence (both dims random)
  seq_rr_3d/         # sequence + 3D
  seq_rz/            # x random, y zero
  seq_hh/            # batch (both dims hold)
  exp7_multiview/    # multi-view experiment data
```

Real-data layout (route B):

```
real_capture/data/
  raw/<seq>/                     # raw capture: cam0/<NNNNN>.png, actions6.csv, ndi.csv, frame_times.txt, meta.json
  derived/<seq>/                 # masks, masks_repaired (repair_masks output), overlay
data/real_seq/<seq>/             # transition npz (train/val): positions(T,3,N), actions(T,1)
data/real_seq/<seq>_clean/       # after clean_transition_npz
```

Each transition npz carries metadata (incl. `n_points`, `tip_fix`) under `data_prep`; training experiments save `config.json` (n_nodes/z_dim/episode_len + data_prep) and a one-line `model_card.txt` at the experiment root.

### Code Language

Comments, docstrings, and variable names are a mix of English and Chinese. The project documentation (`docs/`) is primarily in Chinese.

## Key Conventions

- **No formal test framework** — validation uses Jupyter notebooks (`notebooks/`) and evaluation scripts
- **Experiment logging**: Training outputs go to `train_log/<model_name>/exp_<date>_<n>/` with images, best model weights, and loss logs
- **Config-driven**: `config/training.json` for all hyperparameters; CLI args in `src/config/args.py` can override defaults
- **Two-phase training**: C-MSTNF/MS-SCNF/SkeletonSDF models train canonical/skeleton first, then deformation/SDF jointly
- **Unified training**: All models use `UnifiedTrainer` via `training_spec` class attributes; no per-model Trainer subclasses needed
- **Model loading**: Use `src/utils/model_loader.py` which auto-detects model type from checkpoint
- **Model input convention**: Models take only actuator inputs + 3D query points. Images/depth are supervision signals only, never model inputs
