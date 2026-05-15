# 设计文档：解耦训练架构 — Phase Strategy + View Strategy

> 日期：2026-05-15
> 分支：`feat/multiview-depth-planB`
> 状态：设计阶段

## 1. 问题

当前训练架构中，**训练阶段策略**（单阶段 vs 两阶段）和**视角策略**（单视角 vs 多视角）是耦合在各个 trainer 类里的：

| Trainer | 阶段策略 | 视角策略 | 支持模型 |
|---------|---------|---------|---------|
| `MSTNFTrainer` | 单阶段 | 单视角 | MSTNF |
| `TwoPhaseTrainer` + 子类 | 两阶段 (canonical→deform) | 单视角 | CMSTNF / Smooth / ODE |
| `MSSCNFTrainer` | 两阶段 (skeleton→joint) | 单视角 | MS-SCNF |
| `MultiViewTrainer` | 单阶段 | 多视角 | 任意 |
| `MultiViewConsistencyTrainer` | 单阶段 | 多视角+一致性 | 任意 |

**无法组合**：CMSTNF 想做多视角两阶段训练 — 当前没有 trainer 能同时支持两者。

## 2. 设计目标

1. **阶段策略与视角策略解耦**：任意模型 × 任意阶段 × 任意视角可自由组合
2. **模型声明式**：模型通过 `TrainingSpec` 声明自己需要几个 phase、每个 phase freeze 什么、用什么 forward 方法
3. **向后兼容**：现有 trainer 和脚本不动，新架构通过新文件引入

## 3. 架构

```
┌─────────────────────────────────────────────────────┐
│                   Training Script                    │
│  选择 model + view_strategy → 构建 UnifiedTrainer   │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │  Model   │ │  Phase   │ │  View    │
   │ + Spec   │ │ Strategy │ │ Strategy │
   └──────────┘ └──────────┘ └──────────┘
       声明          管理            管理
    训练需求     freeze/phase     射线/loss
```

### 3.1 TrainingSpec — 模型训练声明

**文件**：`src/training/spec.py`（新建）

```python
from dataclasses import dataclass, field
from typing import Optional

# 内置 loss 名称：
#   "recon"    — MSE 重建 loss（每个视角各算一次）
#   "depth"    — L1 深度 loss（需要深度数据）
#   "smooth"   — 时序平滑 loss（需要 model.temporal）
#   "reproj"   — 跨视角重投影一致性（仅多视角）
#   "consist"  — 跨视角 density 一致性（仅多视角）
# 模型特定 loss 通过 extra_losses_fn 回调注入，名称自定义

@dataclass
class PhaseSpec:
    """单个训练阶段配置。"""
    name: str                                          # "canonical", "deformation", "full", "joint"
    freeze_modules: list[str] = field(default_factory=list)  # 要冻结的子模块名
    forward_attr: str = "forward"                      # 该阶段调用的模型方法名
    data_mode: str = "sequence"                        # "canonical" | "sequence"
    lr: Optional[float] = None                         # 专用学习率 (None=用全局)
    active_losses: list[str] = field(default_factory=lambda: ["recon", "smooth"])
    # 该阶段启用的 loss 列表。内置 loss 用名称引用，模型特定 loss 通过 extra_losses_fn 添加

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

**各模型声明**（作为类属性）：

| 模型 | Phase | freeze | forward | active_losses |
|------|-------|--------|---------|---------------|
| MSTNF | `full` | — | `forward` | `["recon", "depth", "smooth"]` |
| CMSTNF | `canonical` | deform | `forward_canonical` | `["recon"]` |
| CMSTNF | `deformation` | canonical | `forward` | `["recon", "depth", "smooth"]` |
| Smooth-CMSTNF | 同 CMSTNF + extra: `jacobian` | 同上 | 同上 | 同上 |
| ODE-CMSTNF | 同 CMSTNF | 同上 | 同上 | 同上 |
| MS-SCNF | `skeleton` | — | `forward` | `["skeleton"]` |
| MS-SCNF | `joint` | — | `forward` | `["skeleton", "recon", "smooth"]` |

多视角训练时，active_losses 自动追加 `"reproj"`, `"consist"`（由 ViewStrategy 控制）。

模型特定 loss（`skeleton`, `jacobian`）通过 `extra_losses_fn` 回调注入到 UnifiedTrainer。

### 3.2 PhaseStrategy — 训练阶段管理

**文件**：`src/training/phase_strategy.py`（新建）

```python
class PhaseStrategy:
    """根据 TrainingSpec 管理模型的训练阶段切换。"""

    def __init__(self, model):
        self.model = model
        self.spec: TrainingSpec = model.training_spec
        self.current_phase_idx = 0

    @property
    def current_phase(self) -> PhaseSpec: ...

    @property
    def is_last_phase(self) -> bool: ...

    def enter_phase(self, phase_idx: int):
        """切换阶段：解冻全部 → 冻结指定模块。"""
        for p in self.model.parameters():
            p.requires_grad = True
        for mod_name in self.current_phase.freeze_modules:
            for p in getattr(self.model, mod_name).parameters():
                p.requires_grad = False

    def get_forward_fn(self):
        """返回当前阶段的 forward 函数。"""
        return getattr(self.model, self.current_phase.forward_attr)

    def get_trainable_params(self) -> list:
        return [p for p in self.model.parameters() if p.requires_grad]

    def iterate_phases(self):
        """迭代所有阶段，每个 yield (phase_idx, PhaseSpec)。"""
        for i, phase in enumerate(self.spec.phases):
            self.enter_phase(i)
            yield i, phase
```

### 3.3 ViewStrategy — 视角策略

**文件**：`src/training/view_strategy.py`（新建）

```python
class ViewStrategy(ABC):
    """视角策略：管理射线采样、渲染、loss 聚合。"""

    @abstractmethod
    def setup(self, camera_system, device, config): ...

    @abstractmethod
    def compute_losses(self, forward_fn, action_window,
                       target_images, target_depths=None) -> dict:
        """采样射线 → 查询模型 → 渲染 → 返回 losses dict。"""
        ...

class SingleViewStrategy(ViewStrategy):
    """单视角：从 BaseTrainer 提取的射线采样 + 渲染逻辑。"""
    # 复用 BaseTrainer 的 sample_fg_rays + render_points

class MultiViewStrategy(ViewStrategy):
    """多视角：从 MultiViewTrainer 提取的 per-view 循环。"""
    # 复用 _sample_rays_for_view + _render_view
    # 可选：with_depth, with_consistency, with_reprojection

class MultiViewConsistencyStrategy(ViewStrategy):
    """多视角 + 跨视角一致性约束。"""
    # 继承 MultiViewStrategy，追加 reprojection + consistency loss
```

**关键设计**：ViewStrategy 接收 `forward_fn`（由 PhaseStrategy 提供），不直接引用 model。这样 phase 和 view 完全解耦。

### 3.4 UnifiedTrainer — 统一训练器

**文件**：`src/training/trainer_unified.py`（新建）

```python
class UnifiedTrainer:
    """组合 PhaseStrategy + ViewStrategy 的通用训练器。"""

    def __init__(self, model, view_strategy, config=None, extra_losses_fn=None):
        self.model = model
        self.phase = PhaseStrategy(model)
        self.views = view_strategy
        self.config = config or load_config("training")
        self.extra_losses_fn = extra_losses_fn  # 模型特定 loss 回调

    def _compute_losses(self, forward_fn, batch, phase_spec):
        """根据 phase_spec.active_losses 选择性计算 loss。"""
        active = phase_spec.active_losses
        losses = {}

        # 1. 渲染相关 loss（由 ViewStrategy 管理）
        view_losses = self.views.compute_losses(forward_fn, batch, active)
        losses.update(view_losses)  # recon, depth, reproj, consist

        # 2. smoothness loss（如果 active 且模型支持）
        if "smooth" in active and self.phase.spec.supports_smoothness:
            losses["smooth"] = self._compute_smoothness(batch)

        # 3. 模型特定 loss（通过回调注入）
        if self.extra_losses_fn:
            extra = self.extra_losses_fn(self.model, batch, phase_spec, active)
            losses.update(extra)  # skeleton, jacobian, etc.

        losses["total"] = sum(losses.values())
        return losses

    def train(self, data_dirs: dict[str, str],
              exp_dir: str = None,
              n_epochs_per_phase: dict[str, int] = None):
        """统一训练入口。

        Args:
            data_dirs: {"canonical": "path/", "sequence": "path/"}
            n_epochs_per_phase: {"canonical": 50, "deformation": 500}
        """
        for phase_idx, phase_spec in self.phase.iterate_phases():
            data_dir = data_dirs[phase_spec.data_mode]
            loader = self._create_loader(data_dir, phase_spec)

            lr = phase_spec.lr or self.config["optimization"]["lr"]
            optimizer = torch.optim.Adam(self.phase.get_trainable_params(), lr=lr)

            n_epochs = (n_epochs_per_phase or {}).get(phase_spec.name,
                        self.config["optimization"]["n_epochs"])

            for epoch in range(1, n_epochs + 1):
                for batch in loader:
                    forward_fn = self.phase.get_forward_fn()
                    losses = self._compute_losses(forward_fn, batch, phase_spec)

                    optimizer.zero_grad()
                    losses["total"].backward()
                    optimizer.step()
```

## 4. 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/training/spec.py` | 新建 | TrainingSpec + PhaseSpec |
| `src/training/phase_strategy.py` | 新建 | PhaseStrategy |
| `src/training/view_strategy.py` | 新建 | ViewStrategy + 三个实现 |
| `src/training/trainer_unified.py` | 新建 | UnifiedTrainer |
| `src/models/model_mstnf.py` | 修改 | 添加 `training_spec` 类属性 |
| `src/models/model_cmstnf.py` | 修改 | 添加 `training_spec` 类属性 |
| `src/models/model_smooth_cmstnf.py` | 修改 | 添加 `training_spec` 类属性 |
| `src/models/model_ode_cmstnf.py` | 修改 | 添加 `training_spec` 类属性 |
| `src/models/model_ms_scnf.py` | 修改 | 添加 `training_spec` 类属性 |
| `scripts/training/train_unified.py` | 新建 | 使用 UnifiedTrainer 的训练入口 |
| 现有 trainer 脚本 | **不动** | 向后兼容 |

## 5. 使用示例

```python
# 例1: CMSTNF + 多视角 + 一致性约束 (两阶段)
model = CMSTNFModel(action_dim=2, ...)
cam_system = MultiCameraSystem.from_npz(data)
view_strat = MultiViewConsistencyStrategy(cam_system, with_depth=True)

trainer = UnifiedTrainer(model, view_strat)
trainer.train(
    data_dirs={"canonical": "data/canonical_data/", "sequence": "data/seq_rz_c2_sk/"},
    n_epochs_per_phase={"canonical": 50, "deformation": 500},
)

# 例2: MSTNF + 多视角 (单阶段)
model = MSTNFModel(action_dim=2, ...)
view_strat = MultiViewStrategy(cam_system, with_depth=True)

trainer = UnifiedTrainer(model, view_strat)
trainer.train(data_dirs={"sequence": "data/seq_rz_c2_sk/"})

# 例3: CMSTNF + 单视角 (两阶段，等价于现有 TwoPhaseTrainer)
view_strat = SingleViewStrategy(H=100, W=100, focal=136.4, camera_pose=...)
trainer = UnifiedTrainer(model, view_strat)
trainer.train(data_dirs={"canonical": "...", "sequence": "..."})
```

## 6. 验证

1. `UnifiedTrainer` + MSTNF + `MultiViewStrategy` 应产生与 `MultiViewTrainer` 相同的 loss
2. `UnifiedTrainer` + CMSTNF + `SingleViewStrategy` 应产生与 `TwoPhaseTrainer` 相同的 loss
3. 现有所有训练脚本运行不受影响

## 7. 实施优先级

1. `spec.py` + `phase_strategy.py` — 基础声明和管理
2. `view_strategy.py` — 从现有 trainer 提取视角逻辑
3. `trainer_unified.py` — 组合
4. 各模型添加 `training_spec`
5. `train_unified.py` — 入口脚本
6. 验证：对比 loss 值
