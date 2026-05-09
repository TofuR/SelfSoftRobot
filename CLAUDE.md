# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SelfSoftRobot is a research project implementing **neural field-based 3D self-modeling for soft robots**. The robot learns to predict its own 3D shape from 2D camera observations conditioned on actuator inputs, using NeRF-inspired volume rendering as the training signal.

The project builds on the FBV-SM (Field-Based Vision Soft Manipulation) codebase from Hu et al. 2025, extending it from rigid arms to soft continuum arms simulated via PyElastica.

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

# Multi-view data collection with 3D ground truth (PyElastica soft arm)
python scripts/data_collection/collect_multiview.py
```

### Training
```bash
# Model training scripts (soft arm, sequence data)
python scripts/training/train_mstnf.py
python scripts/training/train_cmstnf.py        # Two-phase: canonical + deformation
python scripts/training/train_ode_cmstnf.py     # Neural ODE temporal encoder
python scripts/training/train_smooth_cmstnf.py  # Spectral normalization variant
python scripts/training/train_ms_scnf.py        # Skeleton-conditioned, two-phase
```

### Evaluation & Visualization
```bash
python scripts/evaluation/evaluate_3d.py
python scripts/evaluation/visualize_predictions.py compare   # Side-by-side comparison
python scripts/evaluation/visualize_predictions.py animate   # GIF animation
```

There is no formal test suite. Validation is done through notebooks and the evaluation scripts.

## Architecture

### Two Simulation Backends

- **PyElastica** (`elastica_env.py`): Soft continuum arm (Cosserat rod). Two modes: static `get_simulation_data_pair()` for independent episodes, and `ContinuousSoftArmEnv` for stateful sequential simulation. Renders via PyVista to binary images.
- **PyBullet** (reference only): The original rigid arm simulation code from the FBV-SM paper is preserved in `docs/ref/SelfSimRobot/` for reference. See `docs/papers/hu2025_paper_understanding.md` for detailed analysis.

### Model Architecture (`src/models/`)

| Model | File | Key Idea |
|-------|------|----------|
| FBV_SM | `model.py` | Legacy base model from FBV-SM paper |
| MSTNF | `model_mstnf.py` | Multi-scale EMA for temporal action history |
| CMSTNF | `model_cmstnf.py` | D-NeRF paradigm: separate canonical + deformation fields, two-phase training |
| ODE-CMSTNF | `model_ode_cmstnf.py` | Neural ODE (damped spring) replaces EMA for temporal encoding |
| Smooth-CMSTNF | `model_smooth_cmstnf.py` | Spectral normalization + Jacobian/gradient penalties for smooth deformation |
| MS-SCNF | `model_ms_scnf.py` | Predicts 3D skeleton at multiple scales, conditioned on physics state |

Shared layers in `layers.py`: `PositionalEncoder`, `ActuatorMLPEncoder`, `MLPDecoder`, `TemporalLSTMEncoder`.

### Training Infrastructure (`src/training/`)

- `base.py`: `BaseTrainer` — shared camera setup, rendering, validation loops
- `two_phase_trainer.py`: Base for CMSTNF-family models (Phase 1: canonical field, Phase 2: deformation)
- Model-specific trainers inherit from these: `MSTNFTrainer`, `CMSTNFTrainer`, `ODECMSTNFTrainer`, `SmoothCMSTNFTrainer`, `MSSCNFTrainer`

### Data (`src/data/`)

- `dataset.py`: `SoftSequenceDataset` — loads action-image sequences from `.npz` files (supports 3D positions, camera params)
- `dataset_multiview.py`: `MultiViewDataset` — dual-view (front + side) with 2D skeleton extraction

### Key Utilities

- `src/utils/rendering.py`: Volume rendering utilities (OM rendering, ray sampling)
- `src/utils/camera.py`: Camera ray generation for the new pipeline
- `src/utils/skeleton_2d.py`: 2D skeleton extraction from binary images
- `src/utils/skeleton_viz.py`: 3D skeleton visualization and animation
- `src/utils/model_loader.py`: Auto-detect model type and load checkpoint
- `src/config/params.py`: YAML-based config loading for camera/simulation/training params

### Data Layout

```
data/
  sim_data/         # PyBullet rigid arm simulation data (.npz)
  action/           # Action sequences
  canonical_data/   # Canonical field training data
  sequence_data/    # Sequence training data
  sequence_data_1d/ # 1D sequence variant
  seq_rr_3d/        # 3D sequence data (rotation-rotation)
  seq_rz_3d/        # 3D sequence data (rotation-zero)
  seq_zz_3d/        # 3D sequence data (zero-zero)
  exp7_multiview/   # Multi-view experiment data
  processed/        # Preprocessed data
  raw/              # Raw collected data
```

### Code Language

Comments, docstrings, and variable names are a mix of English and Chinese. The project documentation (`docs/`) is primarily in Chinese.

## Key Conventions

- **No formal test framework** — validation uses Jupyter notebooks (`notebooks/01-09`) and evaluation scripts
- **Experiment logging**: Training outputs go to `train_log/<model_name>/exp_<date>_<n>/` with images, best model weights, and loss logs
- **Config-driven**: New pipeline uses `src/config/params.py` for YAML config loading; legacy code uses hardcoded constants
- **Two-phase training**: CMSTNF/MS-SCNF models train canonical field first, then deformation/skeleton jointly
- **Model loading**: Use `src/utils/model_loader.py` which auto-detects model type from checkpoint
