# Decoupled Training Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple training phase strategy (single/two-phase) from view strategy (single/multi-view) so any model × phase × view combination works through a single `UnifiedTrainer`.

**Architecture:** Models declare a `TrainingSpec` (phases, freeze, forward method, active losses). `PhaseStrategy` reads the spec to manage freeze/unfreeze. `ViewStrategy` handles ray sampling and rendering (single or multi-view). `UnifiedTrainer` composes both. Existing trainers and scripts are untouched.

**Tech Stack:** Python 3.10, PyTorch 2.6, no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-15-decoupled-training-architecture-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/training/spec.py` | Create | `PhaseSpec`, `TrainingSpec` dataclasses |
| `src/training/phase_strategy.py` | Create | `PhaseStrategy` — freeze/unfreeze, forward selection |
| `src/training/view_strategy.py` | Create | `ViewStrategy` ABC + `SingleViewStrategy` + `MultiViewStrategy` + `MultiViewConsistencyStrategy` |
| `src/training/trainer_unified.py` | Create | `UnifiedTrainer` — combines phase + view, loss routing |
| `src/models/model_mstnf.py` | Modify | Add `training_spec` class attribute |
| `src/models/model_cmstnf.py` | Modify | Add `training_spec` class attribute |
| `src/models/model_smooth_cmstnf.py` | Modify | Add `training_spec` class attribute |
| `src/models/model_ode_cmstnf.py` | Modify | Add `training_spec` class attribute |
| `src/models/model_ms_scnf.py` | Modify | Add `training_spec` class attribute |
| `scripts/training/train_unified.py` | Create | CLI entry point using `UnifiedTrainer` |

---

### Task 1: Create `TrainingSpec` and `PhaseSpec`

**Files:**
- Create: `src/training/spec.py`

- [ ] **Step 1: Write `spec.py`**

```python
"""spec.py — 模型训练需求声明。"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PhaseSpec:
    """单个训练阶段配置。"""
    name: str
    freeze_modules: list[str] = field(default_factory=list)
    forward_attr: str = "forward"
    data_mode: str = "sequence"
    lr: Optional[float] = None
    active_losses: list[str] = field(default_factory=lambda: ["recon", "smooth"])

@dataclass
class TrainingSpec:
    """模型的训练需求声明。"""
    phases: list[PhaseSpec]
    supports_smoothness: bool = True

    @property
    def is_two_phase(self) -> bool:
        return len(self.phases) > 1

    @property
    def needs_canonical_data(self) -> bool:
        return any(p.data_mode == "canonical" for p in self.phases)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.training.spec import PhaseSpec, TrainingSpec; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/spec.py
git commit -m "feat: add TrainingSpec + PhaseSpec dataclasses"
```

---

### Task 2: Create `PhaseStrategy`

**Files:**
- Create: `src/training/phase_strategy.py`

- [ ] **Step 1: Write `phase_strategy.py`**

```python
"""phase_strategy.py — 根据 TrainingSpec 管理模型训练阶段切换。"""

import torch
from src.training.spec import PhaseSpec, TrainingSpec


class PhaseStrategy:
    """管理模型的多阶段训练：冻结/解冻子模块、选择 forward 方法。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.spec: TrainingSpec = model.training_spec
        self.current_phase_idx = 0

    @property
    def current_phase(self) -> PhaseSpec:
        return self.spec.phases[self.current_phase_idx]

    @property
    def is_last_phase(self) -> bool:
        return self.current_phase_idx == len(self.spec.phases) - 1

    def enter_phase(self, phase_idx: int):
        self.current_phase_idx = phase_idx
        for p in self.model.parameters():
            p.requires_grad = True
        for mod_name in self.current_phase.freeze_modules:
            module = getattr(self.model, mod_name)
            for p in module.parameters():
                p.requires_grad = False

    def get_forward_fn(self):
        return getattr(self.model, self.current_phase.forward_attr)

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def iterate_phases(self):
        for i, phase in enumerate(self.spec.phases):
            self.enter_phase(i)
            yield i, phase
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.training.phase_strategy import PhaseStrategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/phase_strategy.py
git commit -m "feat: add PhaseStrategy for multi-phase training management"
```

---

### Task 3: Create `ViewStrategy` — SingleView + MultiView + Consistency

**Files:**
- Create: `src/training/view_strategy.py`

This is the largest task. Logic extracted from `BaseTrainer`, `MultiViewTrainer`, and `MultiViewConsistencyTrainer`.

- [ ] **Step 1: Write `view_strategy.py`**

Full file content is ~250 lines. The code is provided in the spec file's Section 3.3 and extracted from:
- `src/training/base.py` lines 45-59 (`sample_fg_rays`) → `SingleViewStrategy._sample_rays`
- `src/training/base.py` lines 61-67 (`render_points`) → `_query_chunked` helper
- `src/training/trainer_multiview.py` lines 77-96 (`_sample_rays_for_view`) → `MultiViewStrategy._sample_rays_for_view`
- `src/training/trainer_multiview.py` lines 98-135 (`_render_view`) → `MultiViewStrategy._render_view`
- `src/training/trainer_multiview_consistency.py` lines 50-143 (`_compute_reprojection_loss`) → `MultiViewStrategy._compute_reprojection_loss`
- `src/training/trainer_multiview_consistency.py` lines 147-244 (`_compute_consistency_loss`) → `MultiViewStrategy._compute_consistency_loss`

Key design: `ViewStrategy.compute_losses()` receives `forward_fn` (not model), `active_losses` list, and returns `dict[str, Tensor]`.

The complete code for this file is provided in the plan's Task 3 Step 1 in the brainstorming conversation. Copy the full `view_strategy.py` content from there.

- [ ] **Step 2: Verify import**

Run: `python -c "from src.training.view_strategy import SingleViewStrategy, MultiViewStrategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/view_strategy.py
git commit -m "feat: add ViewStrategy — SingleView + MultiView + consistency"
```

---

### Task 4: Create `UnifiedTrainer`

**Files:**
- Create: `src/training/trainer_unified.py`

- [ ] **Step 1: Write `trainer_unified.py`**

Key design:
- `_compute_losses(forward_fn, batch, phase_spec)` routes losses by `phase_spec.active_losses`
- `_compute_smoothness()` tries `model.temporal` then `model.deform.temporal`
- `train(data_dirs, n_epochs_per_phase)` iterates phases, creates loaders, runs training loops
- `_create_loader()` uses `MultiViewDepthDataset` with config from `phase_spec`

The complete code for this file is provided in the plan's Task 4 Step 1 in the brainstorming conversation.

- [ ] **Step 2: Verify import**

Run: `python -c "from src.training.trainer_unified import UnifiedTrainer; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/training/trainer_unified.py
git commit -m "feat: add UnifiedTrainer composing PhaseStrategy + ViewStrategy"
```

---

### Task 5: Add `training_spec` to all models

**Files:**
- Modify: `src/models/model_mstnf.py`
- Modify: `src/models/model_cmstnf.py`
- Modify: `src/models/model_smooth_cmstnf.py`
- Modify: `src/models/model_ode_cmstnf.py`
- Modify: `src/models/model_ms_scnf.py`

For each model file, add `from src.training.spec import PhaseSpec, TrainingSpec` after existing imports, then add a class attribute `training_spec = TrainingSpec(...)`.

- [ ] **Step 1: MSTNFModel** — single phase, full forward, recon+depth+smooth

```python
class MSTNFModel(nn.Module):
    training_spec = TrainingSpec(
        phases=[PhaseSpec("full", forward_attr="forward", data_mode="sequence",
                          active_losses=["recon", "depth", "smooth"])],
    )
```

- [ ] **Step 2: CMSTNFModel** — two phase, canonical then deformation

```python
class CMSTNFModel(nn.Module):
    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("canonical", freeze_modules=["deform"],
                      forward_attr="forward_canonical", data_mode="canonical",
                      active_losses=["recon"]),
            PhaseSpec("deformation", freeze_modules=["canonical"],
                      forward_attr="forward", data_mode="sequence",
                      active_losses=["recon", "depth", "smooth"]),
        ],
    )
```

- [ ] **Step 3: SmoothCMSTNFModel** — same as CMSTNF

```python
class SmoothCMSTNFModel(nn.Module):
    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("canonical", freeze_modules=["deform"],
                      forward_attr="forward_canonical", data_mode="canonical",
                      active_losses=["recon"]),
            PhaseSpec("deformation", freeze_modules=["canonical"],
                      forward_attr="forward", data_mode="sequence",
                      active_losses=["recon", "depth", "smooth"]),
        ],
    )
```

- [ ] **Step 4: ODECMSTNFModel** — check sub-module names first

Read `src/models/model_ode_cmstnf.py` `__init__` to confirm module names (`self.canonical`, `self.ode_encoder`, etc.). Then add spec with correct `freeze_modules`.

- [ ] **Step 5: MSSCNFModel** — skeleton then joint, no freeze

```python
class MSSCNFModel(nn.Module):
    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("skeleton", forward_attr="forward", data_mode="sequence",
                      active_losses=["skeleton"]),
            PhaseSpec("joint", forward_attr="forward", data_mode="sequence",
                      active_losses=["skeleton", "recon", "smooth"]),
        ],
    )
```

- [ ] **Step 6: Verify all models**

Run: `python -c "
from src.models.model_mstnf import MSTNFModel
from src.models.model_cmstnf import CMSTNFModel
from src.models.model_ms_scnf import MSSCNFModel
for cls in [MSTNFModel, CMSTNFModel, MSSCNFModel]:
    spec = cls.training_spec
    print(f'{cls.__name__}: {len(spec.phases)} phases, is_two_phase={spec.is_two_phase}')
"`

- [ ] **Step 7: Commit**

```bash
git add src/models/model_mstnf.py src/models/model_cmstnf.py src/models/model_smooth_cmstnf.py src/models/model_ode_cmstnf.py src/models/model_ms_scnf.py
git commit -m "feat: add training_spec class attribute to all models"
```

---

### Task 6: Create `train_unified.py` entry script

**Files:**
- Create: `scripts/training/train_unified.py`

- [ ] **Step 1: Write `train_unified.py`**

CLI flags: `--model`, `--data_dir`, `--canonical_data_dir`, `--multiview`, `--depth`, `--consistency`, `--lr`, `--n_epochs`, `--batch_size`.

The script:
1. Loads config, creates dataset to get action_dim and cam_system
2. Creates model, reads `training_spec` to display phase info
3. Selects ViewStrategy based on `--multiview` and `--consistency` flags
4. Builds `data_dirs` dict and `n_epochs_per_phase` from spec
5. Creates `UnifiedTrainer` and calls `train()`

Complete code provided in the plan's Task 6 Step 1 in the brainstorming conversation.

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/training/train_unified.py').read()); print('Syntax OK')"`

- [ ] **Step 3: Commit**

```bash
git add scripts/training/train_unified.py
git commit -m "feat: add train_unified.py entry script"
```

---

### Task 7: Smoke test — MSTNF + MultiView (single phase)

- [ ] **Step 1: Run 2-epoch smoke test**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
    --model mstnf --data_dir data/seq_rz_c2_sk --multiview --depth \
    --n_epochs 2 --batch_size 2
```

Expected: Runs to completion. Prints single phase "full" with MultiViewStrategy.

- [ ] **Step 2: Clean up test logs**

```bash
rm -rf train_log/train_unified/
```

---

### Task 8: Smoke test — CMSTNF + MultiView (two phase)

- [ ] **Step 1: Check canonical data availability**

```bash
ls data/canonical_data/*.npz 2>/dev/null | head -3
```

- [ ] **Step 2: Run smoke test** (adapt based on canonical data availability)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
    --model cmstnf --data_dir data/seq_rz_c2_sk --multiview --depth \
    --n_epochs 2 --batch_size 2
```

Expected: Two phases printed ("canonical" then "deformation"), both complete without error.

- [ ] **Step 3: Clean up and final commit**

```bash
rm -rf train_log/train_unified/
git add -f docs/superpowers/plans/2026-05-15-decoupled-training-architecture.md
git commit -m "docs: add implementation plan for decoupled training architecture"
```
