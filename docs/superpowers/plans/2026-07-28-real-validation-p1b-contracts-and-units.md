# real_validation P1b(M3 契约与单位修复)Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修掉 real_validation 里 14 项会静默出错的契约/单位缺陷(B1–B5、B7–B11、B14–B17),引入 `deploy_manifest.json` 显式化部署契约,使工作台在没有新数据、没有硬件的条件下可完整离线验收其地基。

**Architecture:** 动作单位经 `real_validation/units.py` 单点收口(`kPa → /action_scale_kpa → /norm_factor → 模型`),只允许出现在 `hardware/valve.py` 与 `openloop_planner` 两处;障碍几何与惩罚收成 `real_validation/obstacles.py` 一份实现,CLI `inverse_plan.py` 反向 import 它,使 CLI 与 GUI 的避障口径由构造保证一致;部署所需的隐式知识全部显式化进 `deploy_manifest.json`,由 `scripts/utils/build_deploy_manifest.py` 从已有实验生成。

**Tech Stack:** Python 3.10+、PyTorch 2.x、PyQt5(仅 GUI)、unittest。全程不碰感知(`real_validation/perception/`)、不碰 `src/` 的模型/编码器(除薄壳外)。

**上游 spec:** [`../specs/2026-07-28-real-validation-task-layer-ik-design.md`](../specs/2026-07-28-real-validation-task-layer-ik-design.md) §12 的 **P1b = M3**。

## Global Constraints

- 分支固定 `feat/real-data-transition`。**不切分支、不新建 worktree。**
- **向后兼容是硬性要求**:不得破坏 `src/`、`scripts/` 中任何现有调用签名。`inverse_plan.py` 的 `--obstacle` / `--w_obs` 等 CLI flag **签名不变**,只改内部语义。
- `real_validation/` **不得 import `src/`**(可移植契约)。反方向允许:`src/`、`scripts/` 可 import `real_validation.*`。
- **`src/encoders/fractional_memory.py` 不得改**。GL 权重缓存只加在部署副本 `real_validation/runtime/model.py`。
- **现有 20 个契约测试的行为断言必须保持通过**;但 `tests/test_real_validation_core.py` 的 `fixtures()` **会升级**(补新契约字段)以适配 fail-closed 裁决 —— 这是有意为之,不承诺零测试改动。
- `validate_plan(plan, model, anchor, scene, safety)` 有 **7 处位置参数调用**(测试 55/58/68/75/252 行 + `session.py:159` `accept_plan` + `session.py:175` `arm`)。新参数**一律 keyword-only + 默认值**,前 5 位不得变。
- 测试框架:`unittest`(无 pytest)。基线:`cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core` → **20 passed**。
- 提交信息用 Conventional Commits,中英混排可,**禁止 `Co-Authored-By`**。用户已授权自主提交(每次提交后不必停下询问)。
- 数值常量(实测,不得改动):`hi6[0] = 150.0 kPa`;训练 `Δt` 由生产者对 `frame_times.txt` 现算(**禁止硬写 0.203125**);`n_nodes=15`;`window_size=40`;`episode_len=40`。
- **fail-closed 裁决**:`action_scale_kpa` 缺失 → 阻断规划。绝不用 `SafetyPolicy.pressure_max6` 当 fallback(那是安全上限,不是训练域上界);绝不用 `or 1.0` 回退(会把活的 OOD 单位 bug 固化成"默认正确")。

## File Structure

**新建**

| 文件 | 责任 |
|---|---|
| `real_validation/units.py` | kPa ↔ 模型动作单位的唯一换算(修 B1/B2) |
| `real_validation/obstacles.py` | 障碍几何解析 + 惩罚(CLI 与 GUI 共用,修 B4) |
| `real_validation/deploy_manifest.py` | `deploy_manifest.json` 数据类 + 读取(修 B3) |
| `scripts/utils/build_deploy_manifest.py` | 从已有实验生成 manifest 的生产者 |
| `tests/test_real_validation_contracts.py` | T2/T3/T4a/T4b/T7/T8 |
| `real_validation/planning/__init__.py` | 空文件 |
| `real_validation/planning/auto_k.py` | 变长 K 选择(从 CLI 移植,修 B17) |

**修改**

| 文件 | 改什么 |
|---|---|
| `real_validation/models.py` | `ModelDescriptor` 新字段(§4.2);`Anchor.prev_state/frame_ref/quality dict`;`ScenePrimitive` schema + 新 kind 消费者;`Scene` 增删改(B7);`SafetyPolicy.pressure_max6` 默认 150 |
| `real_validation/model_runtime.py` | 读 `deploy_manifest.json`,填充 descriptor 新字段(降级三档) |
| `real_validation/offline_anchor.py` | `action_units` 改 `model_normalized`(B2);补 `prev_state` |
| `real_validation/openloop_planner.py` | 单位收口(B1)、target_skeleton/obstacle_aabb/obstacle_polygon、auto_k(B17)、GL 缓存、耗时记录 |
| `real_validation/runtime/model.py` | `FractionalMemory` 权重缓存(仅部署副本,B10) |
| `real_validation/preflight.py` | `dt_mismatch`(B5)、`k_safe_uncertified`、`action_scale_*`、碰撞门覆盖全部障碍(B14) |
| `real_validation/session.py` | `set_scene/set_anchor/set_safety` 状态守卫(B16);`invalidate_model`(B15 用) |
| `real_validation/main_validation.py` | 锁页(B8)、`plan_dt` 取自 descriptor(B5)、K_safe 自动(B9)、`ModelLoadError` + 专用槽(B11/B15)、安全表默认 150 |
| `real_validation/metrics.py` | 支持新障碍与 `target_skeleton` |
| `real_validation/widgets/plan_preview.py` | 绘制新原语 |
| `scripts/control/inverse_plan.py` | `obstacle_loss` 委托 `real_validation.obstacles`(B4);`select_k_by_gap` 供 `auto_k.py` 复用 |
| `tests/test_real_validation_core.py` | `fixtures()` 补新字段;planner 测试单位改对 |
| `real_validation/requirements.txt` | 无新增(units/obstacles/manifest 均纯标准库) |

**明确不在 P1b(附理由)**

| 项 | 理由 |
|---|---|
| 多起点批并行 | `clip_grad_norm_` 是**全局**范数,批化后裁剪语义改变;cuDNN GRU 在 batch=1 与 batch=R 走不同 kernel → **无法与既有结果逐位一致**。价值取决于 Task 6 的耗时基准,基准出来再决定。 |
| B6(jitter 列)、B12(REANCHOR/observation policy 接线) | spec §9 归 M5 |
| `live_anchor.py` | 需要真 checkpoint,归 P3(M4) |

---

### Task 1: `units.py` —— 动作单位唯一换算(修 B1/B2)

把"kPa → 模型动作单位"的换算收成唯一实现。当前 bug:`openloop_planner.py:204` 用 `physical / norm`,而 `norm = action_norm_factor ≈ 1.0`,把 0–150 kPa 直接喂进训练域 `[0,1]` 的模型(活 OOD)。

**Files:**
- Create: `real_validation/units.py`
- Create: `tests/test_real_validation_contracts.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `kPa_to_model(actions_kpa, *, action_scale_kpa, action_norm_factor)` → `(..., A)`(kPa → 训练域 [0,1] → /norm_factor → 模型输入)
  - `model_to_kPa(actions_model, *, action_scale_kpa, action_norm_factor)` → 反变换
  - `check_unit_consistency(action_scale_kpa, action_norm_factor)` → 若 `norm_factor ∈ (0.9, 1.1)` 判定"npz 已归一化"链路(要求走 /action_scale_kpa);若 `norm_factor ≈ hi6` 说明旧式未归一化,两条链路不可混用 → raise 或返回诊断

- [ ] **Step 1: 写失败测试**

创建 `tests/test_real_validation_contracts.py`:

```python
"""P1b 契约测试:T2 单位往返 / T3 坐标往返 / T4a 共享目标核 / T4b CLI-GUI 一致 / T7 rollout 等价 / T8 契约拒绝。

全部离线可跑,不依赖 checkpoint 或硬件。
"""

import unittest

import numpy as np
import torch

from real_validation.models import Anchor, ModelDescriptor, SafetyPolicy, Scene, ScenePrimitive


class UnitConversionTest(unittest.TestCase):
    """T2:kPa ↔ 模型动作单位往返;safety=150kPa 时模型输入 ≤ 1.0。"""

    def test_kpa_to_model_is_bounded_by_training_domain(self):
        from real_validation.units import kPa_to_model
        # 训练域上界 150 kPa → 模型输入必须 ≤ 1.0(锁 B1:此前把 0-150 原样喂进 [0,1] 域)
        actions = np.array([[0.0], [150.0], [75.0], [10.0]], dtype=np.float32)
        model = kPa_to_model(actions, action_scale_kpa=np.array([150.0]),
                             action_norm_factor=1.0)
        self.assertLessEqual(model.max(), 1.0)
        self.assertAlmostEqual(model[1, 0], 1.0, places=6)

    def test_round_trip_is_exact(self):
        from real_validation.units import kPa_to_model, model_to_kPa
        scale = np.array([150.0, 120.0, 100.0, 90.0, 80.0, 70.0], dtype=np.float32)
        actions = np.random.default_rng(0).uniform(0, 1, (5, 6)).astype(np.float32)
        restored = model_to_kPa(kPa_to_model(actions * scale, action_scale_kpa=scale,
                                             action_norm_factor=1.0),
                                action_scale_kpa=scale, action_norm_factor=1.0)
        np.testing.assert_allclose(restored, actions * scale, rtol=1e-6, atol=1e-6)

    def test_norm_factor_multiplicative(self):
        from real_validation.units import kPa_to_model
        actions = np.array([[150.0]], dtype=np.float32)
        with_norm = kPa_to_model(actions, action_scale_kpa=np.array([150.0]),
                                 action_norm_factor=2.0)
        self.assertAlmostEqual(with_norm[0, 0], 0.5, places=6)  # /scale /norm

    def test_torch_tensor_path(self):
        from real_validation.units import kPa_to_model
        actions = torch.tensor([[0.0], [150.0]])
        out = kPa_to_model(actions, action_scale_kpa=torch.tensor([150.0]),
                           action_norm_factor=1.0)
        self.assertIsInstance(out, torch.Tensor)
        self.assertLessEqual(out.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.UnitConversionTest -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.units'`

- [ ] **Step 3: 实现 `real_validation/units.py`**

```python
"""kPa ↔ 模型动作单位的唯一换算(修 B1/B2)。

数据链:actions6.csv 原始 kPa → /action_scale_kpa → 训练域 [0,1] → /action_norm_factor → 模型输入。
反变换:模型输出 → ×action_norm_factor → ×action_scale_kpa → kPa。

action_scale_kpa 来自 meta.json 的 hi6[ch](操作上限,经 masks_to_transition_npz.action_max_per_channel
的 fallback 逻辑);action_norm_factor 是 checkpoint buffer(npz 已归一到 [0,1] 后训练时的二次归一化,
对本数据 ≈ 1.0,no-op)。

这个换算**只允许出现在两处**:hardware/valve.py(硬件边界)与 openloop_planner(优化边界)。
其余任何地方禁止手写 kPa ↔ 模型单位。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _as_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def kPa_to_model(actions_kpa, *, action_scale_kpa, action_norm_factor):
    """kPa → 模型输入。actions_kpa: (..., A) kPa;返回同形状(训练域 [0,1] 再 /norm)。

    支持 numpy 与 torch(张量时返回同 device 的张量,保梯度)。
    """
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    norm = float(action_norm_factor)
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError(f"action_norm_factor 必须为正有限值,收到 {norm}")
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError(f"action_scale_kpa 必须全为正有限值,收到 {scale}")
    if torch is not None and isinstance(actions_kpa, torch.Tensor):
        scale_t = torch.as_tensor(scale, dtype=actions_kpa.dtype, device=actions_kpa.device)
        return actions_kpa / scale_t / norm
    values = np.asarray(actions_kpa, dtype=np.float64)
    return values / scale / norm


def model_to_kPa(actions_model, *, action_scale_kpa, action_norm_factor):
    """模型输出 → kPa(逆变换,仅报告/展示用;优化边界不调用)。"""
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    norm = float(action_norm_factor)
    if torch is not None and isinstance(actions_model, torch.Tensor):
        scale_t = torch.as_tensor(scale, dtype=actions_model.dtype, device=actions_model.device)
        return actions_model * norm * scale_t
    values = np.asarray(actions_model, dtype=np.float64)
    return values * norm * scale


def check_unit_consistency(action_scale_kpa, action_norm_factor, *, hi6=None) -> str:
    """判定归一化链路是否一致,返回诊断字符串(不 raise)。

    - 若 norm_factor ∈ (0.9, 1.1) → npz 侧已归一到 [0,1],链路是 /scale /norm(正确)。
    - 若 hi6 提供且 norm_factor ≈ max(hi6) → 旧式未归一化数据,链路应只 /norm(数据侧没 /scale)。
    两种链路不可混用;返回值供 preflight 记录与人工核对。
    """
    norm = float(action_norm_factor)
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    if 0.9 <= norm <= 1.1:
        return f"npz 已归一化:kPa→/action_scale_kpa={scale}→/norm={norm}→模型"
    if hi6 is not None and norm > 0 and abs(norm - float(np.max(hi6))) / norm < 0.1:
        return f"旧式未归一化:kPa→/norm={norm}(≈hi6) — 与 /action_scale_kpa 链路不可混用"
    return f"norm_factor={norm} 非典型,请人工核对训练侧归一化"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.UnitConversionTest -v`
Expected: 4 tests PASS

- [ ] **Step 5: 提交**

```bash
git add real_validation/units.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): 动作单位唯一换算 units.py(修 B1 活 OOD + B2 标注)
```

---

### Task 2: `obstacles.py` —— 障碍唯一实现(修 B4)

当前障碍惩罚 CLI 对 k **求和**、工作台对 (K,N) **求均值**,差 ≈K 倍。统一为 **mean-over-(K,N)**(工作台口径)。判据:`auto_k` 会让 K 随 gap 变化,sum-over-K 使同一 `w_obs` 的避障压强随 K 线性漂移,与 auto_k 冲突。CLI `--obstacle` / `--w_obs` 签名不变,只改内部委托,并在 `--help` 注明"w_obstacle 现对 K 不变"。

**Files:**
- Create: `real_validation/obstacles.py`
- Modify: `tests/test_real_validation_contracts.py`(追加 T4a 测试类)
- Modify: `scripts/control/inverse_plan.py:111-121`(`obstacle_loss` 委托)

**Interfaces:**
- Consumes: 无
- Produces:
  - `parse_obstacle_circles(scene)` → `list[(cx, cy, r_px)]`(从 `Scene` 的 `obstacle_circle` 原语解析,含 `safety_margin`)
  - `obstacle_term(preds, pc_center, pc_scale, obstacles, reduce="mean")` → 标量(2D SDF 惩罚,全节点)
  - `clearance_min(states_px, obstacles)` → `float`(最小净距,用于 preflight 碰撞门)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_real_validation_contracts.py`:

```python
class SharedObjectiveParityTest(unittest.TestCase):
    """T4a:共享目标核的 4 个损失项 + 障碍项逐位一致;障碍项对 K 不变;z 永不进 loss。"""

    def test_obstacle_term_is_mean_over_k_and_nodes(self):
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(0)
        preds = torch.randn(4, 15, 3, dtype=torch.float64)      # (K,N,3) 归一化
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(10.0, 10.0, 2.0), (20.0, 5.0, 1.5)]
        got = obstacle_term(preds, center, scale, obstacles)
        # 手工引用实现:逐 k 逐 obs 对 (N,) 求均值再累加,最后 /K(即对 (K,N) 全均值)
        expected = preds.new_zeros(())
        for k in range(preds.shape[0]):
            for (cx, cy, r) in obstacles:
                d = torch.linalg.vector_norm(preds[k, :, :2] - preds.new_tensor((cx, cy)), dim=1)
                expected = expected + torch.relu(r - d).square().mean()
        expected = expected / preds.shape[0]
        self.assertTrue(torch.equal(got, expected))

    def test_obstacle_term_is_invariant_to_horizon(self):
        """锁死口径选择:mean-over-(K,N) 使同一 w_obs 的避障压强不随 K 漂移(与 auto_k 兼容)。"""
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(1)
        base = torch.randn(5, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(10.0, 10.0, 2.0)]
        doubled = torch.cat([base, base], 0)
        self.assertTrue(torch.allclose(
            obstacle_term(base, center, scale, obstacles),
            obstacle_term(doubled, center, scale, obstacles), rtol=0, atol=1e-12))

    def test_z_channel_never_enters_obstacle_loss(self):
        """pc_scale[2]=1e-6,任何非零 z 会被放大 1e6 —— 障碍 loss 必须只吃 [:2]。"""
        from real_validation.obstacles import obstacle_term
        torch.manual_seed(2)
        preds = torch.randn(3, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(0.0, 0.0, 3.0)]
        plain = obstacle_term(preds, center, scale, obstacles)
        noisy = preds.clone()
        noisy[:, :, 2] += 1e3          # z 通道污染
        self.assertTrue(torch.equal(plain, obstacle_term(noisy, center, scale, obstacles)))

    def test_k_equals_one_is_finite(self):
        """抓 CLI inverse_plan.py:154 的 K=1 时 errs[1:] 空 → L_mono NaN。"""
        from real_validation.obstacles import obstacle_term
        preds = torch.randn(1, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        value = obstacle_term(preds, center, scale, [(0.0, 0.0, 1.0)])
        self.assertTrue(torch.isfinite(value))
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.SharedObjectiveParityTest -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.obstacles'`

- [ ] **Step 3: 实现 `real_validation/obstacles.py`**

```python
"""障碍几何解析 + 2D SDF 惩罚(CLI 与 GUI 共用的唯一实现,修 B4)。

口径:mean-over-(K,N)。CLI inverse_plan.py 原实现对 k 求和、对 N 求均值,与工作台的
(K,N) 全均值差 ≈K 倍。统一为 mean 的判据:auto_k 会让 K 随 gap 变化,sum-over-K 使
同一 w_obs 的避障压强随 K 线性漂移,与 auto_k 直接冲突。

坐标:preds 为 (K,N,3) 归一化,obstacles 为 model 坐标(px [col,row])。归一化 → px 用
pc_center/pc_scale;只取 [:2](col,row 平面)。pc_scale[2]=1e-6,z 通道污染会被放大 1e6,
故严禁把 z 计入。
"""

from __future__ import annotations

import math

import torch


def parse_obstacle_circles(scene):
    """从 Scene 的 obstacle_circle 原语解析 → [(cx, cy, r_px)];含 safety_margin。

    只接受 frame_id == "model" 的圆;其他障碍类型(AABB/多边形)由调用方另作解析。
    """
    supported = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.kind != "obstacle_circle":
            raise ValueError(f"当前 obstacle 解析尚不支持 {item.kind}")
        if item.frame_id != "model":
            raise ValueError("圆障碍必须先转换到 model 坐标")
        center = item.geometry.get("center", item.geometry.get("xy"))
        radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
        if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
            raise ValueError("obstacle_circle 需要 center=[x,y] 与正 radius")
        supported.append((float(center[0]), float(center[1]),
                          radius + float(item.safety_margin)))
    return supported


def obstacle_term(preds, pc_center, pc_scale, obstacles, reduce: str = "mean"):
    """障碍惩罚:preds (K,N,3) 归一化 → px,对每个 keep-out 圆罚穿透。标量。

    obstacles: [(cx, cy, r_px), ...] in model px。聚合 = mean-over-(K,N)(对 K 不变)。
    """
    if not obstacles:
        return preds.new_zeros(())
    physical = preds * pc_scale + pc_center          # (K,N,3) px
    total = preds.new_zeros(())
    for (cx, cy, radius) in obstacles:
        distance = torch.linalg.vector_norm(
            physical[:, :, :2] - physical.new_tensor((cx, cy)), dim=2)
        total = total + torch.relu(radius - distance).square().mean()
    if reduce == "mean":
        return total / preds.shape[0]
    if reduce == "sum":
        return total
    raise ValueError(f"未知 reduce: {reduce}")


def clearance_min(states_px, obstacles) -> float | None:
    """states_px (K,N,3) 或 (N,3) px → 到所有障碍的最小净距(distance - r,可为负)。

    供 preflight 碰撞门用。无障碍返回 None。
    """
    if not obstacles:
        return None
    values = []
    for (cx, cy, radius) in obstacles:
        xy = torch.as_tensor(states_px, dtype=torch.float64)[..., :2]
        distance = torch.linalg.vector_norm(xy - torch.as_tensor((cx, cy), dtype=torch.float64),
                                            dim=-1)
        values.append(float((distance - radius).min()))
    return min(values)


def _clearance_min_numpy(states_px, obstacles) -> float | None:
    """numpy 版(CLI 报告用,no_grad 场景)。states_px (K,N,3) 或 (N,3)。"""
    if not obstacles:
        return None
    import numpy as np
    xy = np.asarray(states_px, dtype=np.float64)[..., :2]
    values = []
    for (cx, cy, radius) in obstacles:
        distance = np.linalg.norm(xy - np.asarray((cx, cy), dtype=np.float64), axis=-1)
        values.append(float((distance - radius).min()))
    return min(values)


# ---------------- CLI 兼容层(inverse_plan.py 委托到共享核的落点) ----------------
def cli_obstacle_loss(preds_norm, pc_center, pc_scale, obs_list):
    """与 CLI inverse_plan.obstacle_loss 签名一致,但聚合改为 mean-over-(K,N)。

    obs_list: [(cx, cy, r_px)] in px。preds_norm (K,N,3) 归一化。
    注:聚合口径从"对 k 求和"改为"mean",同一 w_obs 的避障压强不再随 K 漂移(与 auto_k 兼容)。
    """
    return obstacle_term(preds_norm, pc_center, pc_scale, obs_list, reduce="mean")


- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.SharedObjectiveParityTest -v`
Expected: 4 tests PASS

- [ ] **Step 5: CLI `inverse_plan.py` 委托(签名不变,只改内部)**

`scripts/control/inverse_plan.py:111-121` 的 `obstacle_loss` 函数体替换为对共享核的单次委托:

```python
def obstacle_loss(preds_norm, pc_center, pc_scale, obs_list):
    """避障惩罚: preds (K,N,3) 归一化 → px, 对每个 keep-out 圆(cxcy,r)罚穿透。
    obs_list: [(cx, cy, r_px), ...] in px (col,row)。

    ⚠️ 聚合口径 2026-07-28 从"对 k 求和"改为 mean-over-(K,N)(与工作台统一):
       同一 w_obs 的避障压强不再随 K 线性漂移,与 --auto_k 兼容。
       docs/reports/2026-07-14/15 中含障碍的 planner 数字与本实现不可比
       (该报告本来就含不可复现的随机重启分量 —— CLI 无 torch.manual_seed)。
    """
    from real_validation.obstacles import cli_obstacle_loss
    return cli_obstacle_loss(preds_norm, pc_center, pc_scale, obs_list)
```

同时给 `--obstacle` 的 `--help` 文本追加说明(定位 `inverse_plan.py` 的 argparse 里 `--w_obs` help):

```python
parser.add_argument("--w_obs", type=float, default=1.0,
                    help="避障权重(聚合=mean-over-(K,N), 对 K 不变; 2026-07-28 与工作台统一口径)")
```

> ⚠️ **数字不可比的诚实记录**:`docs/reports/2026-07-14` 与 `2026-07-15` 汇报里的避障数字是旧 sum 口径;统一口径后重跑才可比。这在 commit message 里注明。

- [ ] **Step 6: 验证 CLI 委托后 `--help` 仍可用**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python scripts/control/inverse_plan.py --help > /dev/null && echo "inverse_plan --help ok"
python -c "from real_validation.obstacles import cli_obstacle_loss, obstacle_term, clearance_min, _clearance_min_numpy, parse_obstacle_circles; print('obstacles ok')"
```

- [ ] **Step 7: 提交**

```bash
git add real_validation/obstacles.py scripts/control/inverse_plan.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): 障碍唯一实现 obstacles.py(修 B4 口径差 ≈K 倍,CLI 委托共享核)"
```

---

### Task 3: `deploy_manifest.py` + `ModelDescriptor` 字段 + `model_runtime` 读取(修 B3)

把部署所需的隐式知识显式化进 `deploy_manifest.json`(与 checkpoint 同目录)。`action_scale_kpa` 缺失时 **fail-closed 阻断规划**(活 OOD bug,见 Global Constraints)。

**Files:**
- Create: `real_validation/deploy_manifest.py`
- Modify: `real_validation/models.py`(`ModelDescriptor` 新字段,§4.2)
- Modify: `real_validation/model_runtime.py`(读 manifest,降级三档)
- Modify: `tests/test_real_validation_contracts.py`(追加 T8 契约拒绝测试)

**Interfaces:**
- Consumes: `real_validation/io.file_sha256`、`real_validation.models.ModelDescriptor`
- Produces:
  - `real_validation.deploy_manifest.DeployManifest`(frozen dataclass)字段:`schema_version`、`checkpoint_sha256`、`action_scale_kpa: tuple|None`、`channel_map: tuple|None`、`train_dt_nominal_s`、`train_dt_measured_s`、`train_dt_std_s`、`mask_source`、`mask_source_provenance`、`segment_params: dict|None`、`camera: dict|None`、`reference_frame`、`reference_frame_sha256`、`mask_area_median_px`、`registration_residual_max_px=2.0`、`k_safe_table_px: dict|None`、`train_sequences`、`n_nodes/window_size/z_dim/episode_len/action_dim/encoder_type/hidden_dim/n_scales`
  - `DeployManifest.load(path)` → 校验缺必填字段
  - `real_validation.model_runtime.ModelRuntime` 读 manifest 后填充 `descriptor` 的字段

- [ ] **Step 1: 写失败测试(T8 契约拒绝)**

追加到 `tests/test_real_validation_contracts.py`:

```python
class ContractRejectionTest(unittest.TestCase):
    """T8:缺失 manifest 关键字段必须阻断规划;provenance 标签可审计。"""

    def test_action_scale_kpa_missing_blocks_planning(self):
        """fail-closed 裁决:action_scale_kpa 缺失不能回退 1.0(否则把活 OOD bug 固化成默认)。"""
        from real_validation.models import ModelDescriptor
        descriptor = ModelDescriptor(
            checkpoint="m.pt", checkpoint_hash="abc", model_type="state_transition",
            action_dim=1, n_nodes=15, history_steps=40,
            action_scale_kpa=None, channel_map=None, train_dt_nominal_s=None)
        self.assertIsNone(descriptor.action_scale_kpa)

    def test_manifest_round_trip(self):
        from real_validation.deploy_manifest import DeployManifest
        manifest = DeployManifest(
            checkpoint_sha256="deadbeef", action_scale_kpa=(150.0,),
            channel_map=(0,), train_dt_nominal_s=0.2, train_dt_measured_s=0.2031,
            train_dt_std_s=0.011, mask_source="white_on_blue",
            mask_source_provenance="path_suffix",
            segment_params={"val": 100.0}, camera=None,
            k_safe_table_px={"5px": 51, "10px": 124},
            n_nodes=15, window_size=40, z_dim=16, episode_len=40, action_dim=1,
            encoder_type="fractional", hidden_dim=128, n_scales=4)
        restored = DeployManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.action_scale_kpa, (150.0,))
        self.assertEqual(restored.k_safe_table_px["10px"], 124)

    def test_manifest_missing_required_raises(self):
        from real_validation.deploy_manifest import DeployManifest
        with self.assertRaises(ValueError):
            DeployManifest(checkpoint_sha256="x", action_scale_kpa=None,
                           channel_map=None, train_dt_nominal_s=None, mask_source=None,
                           mask_source_provenance=None, segment_params=None, camera=None,
                           k_safe_table_px=None, n_nodes=None, window_size=None,
                           z_dim=None, episode_len=None, action_dim=None,
                           encoder_type=None, hidden_dim=None, n_scales=None)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.ContractRejectionTest -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'real_validation.deploy_manifest'`;且 `ModelDescriptor` 尚无新字段

- [ ] **Step 3: `models.py` 给 `ModelDescriptor` 加字段(§4.2)**

在 `ModelDescriptor.__post_init__` 之前追加字段(全部在 `normalization` 之后,**带默认值**避免破坏 `fixtures()` 的 6 个位置参数调用):

```python
    normalization: dict[str, Any] = field(default_factory=dict)
    # ---- P1b 新增(全部带默认值;缺 manifest 时为 None,由 preflight/planner 阻断) ----
    action_scale_kpa: tuple[float, ...] | None = None
    channel_map: tuple[int, ...] | None = None
    train_dt_nominal_s: float | None = None
    train_dt_measured_s: float | None = None
    train_dt_std_s: float | None = None
    mask_source: str | None = None
    mask_source_provenance: str | None = None
    segment_params: dict[str, Any] | None = None
    camera_fingerprint: dict[str, Any] | None = None
    reference_frame_hash: str | None = None
    k_safe_table_px: dict[str, int] | None = None
    registration_residual_max_px: float = 2.0
    provenance: dict[str, str] = field(default_factory=dict)
```

并在 `__post_init__` 里加校验:

```python
        if self.action_scale_kpa is not None:
            values = tuple(float(v) for v in self.action_scale_kpa)
            if len(values) != self.action_dim:
                raise ValueError("action_scale_kpa 长度必须等于 action_dim")
            if any(v <= 0 or not math.isfinite(v) for v in values):
                raise ValueError("action_scale_kpa 必须全为正有限值")
            object.__setattr__(self, "action_scale_kpa", values)
        if self.channel_map is not None:
            mapping = tuple(int(v) for v in self.channel_map)
            if len(mapping) != self.action_dim or len(set(mapping)) != len(mapping) \
                    or any(v < 0 or v >= 6 for v in mapping):
                raise ValueError("channel_map 必须是不重复的 0..5 通道,长度等于 action_dim")
            object.__setattr__(self, "channel_map", mapping)
```

- [ ] **Step 4: 实现 `real_validation/deploy_manifest.py`**

```python
"""deploy_manifest.json 的数据契约与读取(修 B3)。

把部署所需的隐式知识显式化:action_scale_kpa(kPa 上界,训练时 npz 的 /hi6)、
train_dt(实测采样周期)、mask_source(在线只允许匹配的源)、segment_params(分割参数指纹)、
camera 指纹、k_safe_table_px(视野认证表)。由 scripts/utils/build_deploy_manifest.py
从已有实验生成;工作台只读。

缺 manifest 或缺关键字段时:**fail-closed 阻断规划**(action_scale_kpa 缺失不能用
or 1.0 回退 —— 单位 bug 是活的,kPa 0-150 直接除 ≈1.0 的 norm_factor 喂进 [0,1]
训练域;回退会把 OOD 固化成"默认正确",且错误单位的 plan 会被存档 replay 成假工件)。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REQUIRED = (
    "checkpoint_sha256", "action_scale_kpa", "channel_map", "train_dt_nominal_s",
    "mask_source", "n_nodes", "window_size", "z_dim", "episode_len",
    "action_dim", "encoder_type", "hidden_dim", "n_scales",
)


@dataclass(frozen=True)
class DeployManifest:
    schema_version: int = 1
    checkpoint_sha256: str | None = None
    action_scale_kpa: tuple[float, ...] | None = None
    channel_map: tuple[int, ...] | None = None
    train_dt_nominal_s: float | None = None
    train_dt_measured_s: float | None = None
    train_dt_std_s: float | None = None
    mask_source: str | None = None
    mask_source_provenance: str | None = None
    segment_params: dict[str, Any] | None = None
    camera: dict[str, Any] | None = None
    reference_frame: str | None = None
    reference_frame_sha256: str | None = None
    mask_area_median_px: int | None = None
    registration_residual_max_px: float = 2.0
    k_safe_table_px: dict[str, int] | None = None
    train_sequences: tuple[str, ...] = ()
    n_nodes: int | None = None
    window_size: int | None = None
    z_dim: int | None = None
    episode_len: int | None = None
    action_dim: int | None = None
    encoder_type: str | None = None
    hidden_dim: int | None = None
    n_scales: int | None = None

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED if getattr(self, name) is None]
        if missing:
            raise ValueError(f"deploy_manifest 缺必填字段: {missing}")
        if self.action_scale_kpa is not None:
            object.__setattr__(self, "action_scale_kpa",
                               tuple(float(v) for v in self.action_scale_kpa))
        if self.channel_map is not None:
            object.__setattr__(self, "channel_map",
                               tuple(int(v) for v in self.channel_map))
        if self.train_sequences is not None:
            object.__setattr__(self, "train_sequences", tuple(self.train_sequences))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeployManifest":
        return cls(**{k: v for k, v in value.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "DeployManifest":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} 顶层必须是对象")
        return cls.from_dict(payload)
```

- [ ] **Step 5: `model_runtime.py` 读 manifest,填充 descriptor**

在 `ModelRuntime.__init__` 里 `_nearby_config` 之后加 manifest 读取(复用向上 6 层 walk;manifest 缺失时静默,字段留 None):

```python
    def _nearby_manifest(self, checkpoint: Path) -> dict[str, Any] | None:
        current = checkpoint.parent
        for _ in range(6):
            candidate = current / "deploy_manifest.json"
            if candidate.is_file():
                try:
                    with candidate.open("r", encoding="utf-8") as stream:
                        value = json.load(stream)
                    return value if isinstance(value, dict) else None
                except (OSError, ValueError):
                    return None
            if current.parent == current:
                break
            current = current.parent
        return None
```

在 descriptor 构造处追加(替换现有 `ModelDescriptor(...)` 块,新增字段):

```python
        manifest_raw = self._nearby_manifest(checkpoint_path)
        manifest = None
        if manifest_raw:
            from .deploy_manifest import DeployManifest
            try:
                manifest = DeployManifest.from_dict(manifest_raw)
            except ValueError:
                manifest = None   # manifest 残缺 → 字段留 None,由 preflight 阻断规划
        self.manifest = manifest
        descriptor = ModelDescriptor(
            checkpoint=str(checkpoint_path),
            checkpoint_hash=file_sha256(checkpoint_path),
            model_type=str(info["model_type"]),
            action_dim=int(info["action_dim"]),
            n_nodes=n_nodes,
            history_steps=history,
            model_class=str(info["model_class"]),
            k_train=int(k_train_value) if k_train_value is not None else None,
            k_safe=int(k_safe) if k_safe is not None else None,
            data_dir=str(Path(data_dir).resolve()) if data_dir else None,
            normalization={"action_norm_factor": float(info["norm_factor"])},
            action_scale_kpa=manifest.action_scale_kpa if manifest else None,
            channel_map=manifest.channel_map if manifest else None,
            train_dt_nominal_s=manifest.train_dt_nominal_s if manifest else None,
            train_dt_measured_s=manifest.train_dt_measured_s if manifest else None,
            train_dt_std_s=manifest.train_dt_std_s if manifest else None,
            mask_source=manifest.mask_source if manifest else None,
            mask_source_provenance=manifest.mask_source_provenance if manifest else None,
            segment_params=manifest.segment_params if manifest else None,
            camera_fingerprint=manifest.camera if manifest else None,
            reference_frame_hash=manifest.reference_frame_sha256 if manifest else None,
            k_safe_table_px=manifest.k_safe_table_px if manifest else None,
            registration_residual_max_px=manifest.registration_residual_max_px
                if manifest else 2.0,
        )
        self.descriptor = descriptor
```

- [ ] **Step 6: 跑测试确认通过**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_real_validation_contracts -v 2>&1 | tail -4
python -m unittest tests.test_real_validation_core 2>&1 | tail -3
```
Expected: contracts 全 PASS;core 仍 20 PASS(ModelDescriptor 新字段带默认值,fixtures 的 6 位置参数不受影响)。

- [ ] **Step 7: 提交**

```bash
git add real_validation/deploy_manifest.py real_validation/models.py real_validation/model_runtime.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): deploy_manifest 契约 + ModelDescriptor 新字段 + fail-closed(修 B3)"
```

---

### Task 4: `models.py` 契约四件套 + `fixtures()` 升级

一次改完 4 个契约点,避免多次破坏测试:`ScenePrimitive` schema、`Scene` 增删改(B7)、`Anchor` 新字段、`SafetyPolicy` 默认 150。然后升级 `fixtures()` 让现有 20 测试继续过。

**Files:**
- Modify: `real_validation/models.py`
- Modify: `real_validation/offline_anchor.py`(B2 + prev_state)
- Modify: `tests/test_real_validation_core.py`(fixtures 升级 + 新测试)

**Interfaces:**
- Consumes: Task 3 的 `ModelDescriptor` 新字段
- Produces:
  - `ScenePrimitive.geometry` 双键容错(读时 `xy`/`center`、`radius`/`r`、`node` 可选),**`__post_init__` 不规范化 geometry**(改键会改 `scene.digest`,使磁盘上存量 plan 全部 stale)
  - `Scene.without_primitive(primitive_id)` / `Scene.replace_primitive(primitive_id, new_primitive)`(B7,按 `primitive_id` 定位不按 index)
  - `Anchor.prev_state: tuple|None = None`、`frame_ref: str = ""`、`quality: dict = field(default_factory=dict)`(替换 `float|None`)
  - `SafetyPolicy.pressure_max6` 默认 `(150.0,)*6`(原 200)

- [ ] **Step 1: `ScenePrimitive.__post_init__` 保留白名单,`Scene` 加增删改**

`models.py:113-131` 的 `ScenePrimitive.__post_init__` **不改**白名单(12 个 kind 保持);`Scene`(`models.py:134-170`)追加两个方法:

```python
    def without_primitive(self, primitive_id: str) -> "Scene":
        """按 primitive_id 移除一个原语(B7:原来只能追加,交互式编辑无法删除)。"""
        kept = tuple(item for item in self.primitives if item.primitive_id != primitive_id)
        if len(kept) == len(self.primitives):
            raise KeyError(f"primitive_id 不存在: {primitive_id}")
        return Scene(name=self.name, primitives=kept, dimension=self.dimension)

    def replace_primitive(self, primitive_id: str, new_primitive: "ScenePrimitive") -> "Scene":
        """按 primitive_id 替换一个原语。"""
        replaced = tuple(new_primitive if item.primitive_id == primitive_id else item
                         for item in self.primitives)
        if replaced == self.primitives:
            raise KeyError(f"primitive_id 不存在: {primitive_id}")
        return Scene(name=self.name, primitives=replaced, dimension=self.dimension)
```

> 注意:`with_primitive` / `without_primitive` / `replace_primitive` 都不传 `revision`,`default_factory=uuid4` 生成新 revision → `digest` 变化 → 旧 plan 失效。这是**有意**的:任何 scene 编辑都让存量计划 stale。测试需覆盖"删→加回 ⇒ digest 不同"(记录该决定,不要求相同)。

- [ ] **Step 2: `Anchor` 新字段(§4.4)**

`models.py:66-98` 的 `Anchor` 替换字段:

```python
@dataclass(frozen=True)
class Anchor:
    state: tuple[tuple[float, ...], ...]
    action_history: tuple[tuple[float, ...], ...]
    prev_state: tuple[tuple[float, ...], ...] | None = None   # ★P1b:s_{t-2},速度项需要
    frame_id: str = "model"
    frame_ref: str = ""                                       # ★P1b:隐藏评价流的帧引用
    timestamp: float = field(default_factory=time.time)
    anchor_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "unknown"
    quality: dict[str, Any] = field(default_factory=dict)     # ★P1b:float → 标志集
    state_space: str = "model_normalized"
    action_units: str = "kpa"
```

`__post_init__` 里:
- 删掉旧的 `if self.quality is not None and not math.isfinite(self.quality)`(dict 会 TypeError)
- `prev_state` 非 None 时校验形状/有限值(与 `state` 同规则)
- `frame_ref` 保持任意字符串(它是评价流的引用,不是坐标)

`from_dict` 需保护:`prev_state` 为 None 时跳过转换(否则 `test_replay_session_cannot_arm` 的 `load_for_replay` TypeError):

```python
    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Anchor":
        data = dict(value)
        data.pop("schema_version", None)
        data["state"] = tuple(tuple(row) for row in data["state"])
        data["action_history"] = tuple(tuple(row) for row in data["action_history"])
        if data.get("prev_state") is not None:
            data["prev_state"] = tuple(tuple(row) for row in data["prev_state"])
        return cls(**data)
```

- [ ] **Step 3: `SafetyPolicy.pressure_max6` 默认 150**

`models.py:176` `pressure_max6: tuple[float, ...] = (200.0,) * N_HARDWARE_CHANNELS` → `(150.0,) * N_HARDWARE_CHANNELS`(对齐训练上界,避免规划走出训练域;所有现有测试都显式传 100,零风险)。

- [ ] **Step 4: `offline_anchor.py` 修 B2 + 补 prev_state**

`offline_anchor.py:56-60` 的 Anchor 构造改:

```python
    prev_state = positions[frame_index - 1] if frame_index >= 1 else None
    if prev_state is not None:
        if prev_state.shape == (3, model.n_nodes):
            prev_state = prev_state.T
        prev_state = (prev_state - center) / scale
    return Anchor(
        state=tuple(tuple(float(value) for value in node) for node in normalized),
        action_history=tuple(tuple(float(value) for value in action) for action in history),
        prev_state=(None if prev_state is None
                    else tuple(tuple(float(value) for value in node) for node in prev_state)),
        frame_id="model_normalized", state_space="model_normalized",
        action_units="model_normalized",   # ★B2:npz actions 已归一到 [0,1],不是 kPa
        source=f"{source}#frame={frame_index}", quality={"kind": "offline_npz", "score": 1.0})
```

> 注意 `action_units` 从 `"kpa"` 改 `"model_normalized"` 会**改变现有语义**:planner 里 `history / norm` 只在 `action_units=="kpa"` 时做(`openloop_planner.py:164-165`)。改后 npz 历史的 `[0,1]` 值**不再被除** —— 正确(它们已是模型单位)。但这是**行为变更**,必须同步改 openloop_planner 的单位收口(Task 6 做)。本 Task 只改标注,Task 6 修消费方。

- [ ] **Step 5: 升级 `tests/test_real_validation_core.py` 的 `fixtures()`**

`fixtures()`(`tests/test_real_validation_core.py:27-43`)补 `ModelDescriptor` 新字段(带 `action_scale_kpa=(1.0,)` 使单位收口后数值与旧行为等价):

```python
def fixtures():
    model = ModelDescriptor(
        "mock.pt", "abc", "state_transition", 1, 3, 2,
        k_train=4, k_safe=4,
        action_scale_kpa=(1.0,), channel_map=(0,),
        train_dt_nominal_s=0.1, train_dt_measured_s=0.1, train_dt_std_s=0.0,
        mask_source="white_on_blue", mask_source_provenance="test",
        segment_params={"val": 100.0},
        k_safe_table_px={"5px": 4}, registration_residual_max_px=2.0)
    anchor = Anchor(
        state=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
        action_history=((0.0,), (0.0,)), source="test")
    scene = Scene("test", (ScenePrimitive("target_point", "model", {"xy": [2, 3]}),))
    safety = SafetyPolicy(
        pressure_max6=(100.0,) * 6,
        rise_rate6=(100.0,) * 6,
        fall_rate6=(100.0,) * 6,
        ack_timeout_s=0.1,
    )
    plan = build_plan(model_actions=((10.0,), (20.0,)), channel_map=(0,),
                      step_interval_s=0.1, model=model, anchor=anchor,
                      scene=scene, safety=safety, random_seed=7)
    return model, anchor, scene, safety, plan
```

> `action_scale_kpa=(1.0,)` 使 `unit = scale * norm = 1.0 * 100 = 100`,与旧行为逐位一致(旧代码 `physical / norm` 用 norm=100.0 来自 `info`)。`k_safe=4 ≥ horizon=2` 保持。

- [ ] **Step 6: 跑测试确认通过 + 新增场景编辑测试**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_real_validation_core 2>&1 | tail -3
```

追加一个测试到 `tests/test_real_validation_core.py`:

```python
    def test_scene_without_and_replace_primitive(self):
        model, anchor, scene, safety, plan = fixtures()
        obstacle = ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 2})
        scene2 = scene.with_primitive(obstacle)
        self.assertEqual(len(scene2.primitives), 2)
        removed = scene2.without_primitive(obstacle.primitive_id)
        self.assertEqual(len(removed.primitives), 1)
        # 删→加回同一原语,digest 不同(新 primitive_id + 新 revision):任何编辑都让旧 plan stale
        added_back = removed.with_primitive(
            ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 2}))
        self.assertNotEqual(added_back.digest, scene2.digest)
        with self.assertRaises(KeyError):
            scene2.without_primitive("nonexistent")
```

- [ ] **Step 7: 提交**

```bash
git add real_validation/models.py real_validation/offline_anchor.py tests/test_real_validation_core.py
git commit -m "feat(real_validation): 契约四件套(ScenePrimitive schema/Scene 增删改/Anchor prev_state/quality dict/Safety 默认150)+ fixtures 升级"
```

---

### Task 5: `preflight.py` 新检查(B5 / B14 / fail-closed)

**Files:**
- Modify: `real_validation/preflight.py`
- Modify: `tests/test_real_validation_contracts.py`(追加 preflight 测试)

**Interfaces:**
- Consumes: Task 3/4 的 `ModelDescriptor` 新字段、`ScenePrimitive` kind
- Produces: `validate_plan` 新增 keyword-only 参数 `train_dt_s: float | None = None`,返回 issue code:
  - `dt_mismatch`(|step_interval_s - train_dt_s| / train_dt_s ≥ 0.05)
  - `train_dt_unknown`(无 train_dt 可校验且 step_interval_s 未显式给出)
  - `k_safe_uncertified`(model.k_safe is None 且 k_safe_table_px 为空 → 阻断任意 horizon)
  - `unsupported_obstacle`(scene 含 planner 未支持的障碍类型且 clearance metadata 未覆盖)
  - `action_scale_missing`(action_scale_kpa is None,由 planner/preflight 阻断)

- [ ] **Step 1: 写失败测试**

追加到 `tests/test_real_validation_contracts.py`:

```python
class PreflightNewGatesTest(unittest.TestCase):
    def _base(self, **model_kw):
        from real_validation.models import Anchor, ModelDescriptor, SafetyPolicy, Scene, ScenePrimitive
        from real_validation.planner_service import build_plan
        model = ModelDescriptor("m.pt", "abc", "state_transition", 1, 3, 2,
                                k_train=4, k_safe=4,
                                action_scale_kpa=(1.0,), channel_map=(0,),
                                train_dt_nominal_s=0.2, train_dt_measured_s=0.2031,
                                train_dt_std_s=0.011, **model_kw)
        anchor = Anchor(state=((0, 0), (1, 1), (2, 2)), action_history=((0,), (0,)))
        scene = Scene("t", (ScenePrimitive("target_point", "model", {"xy": [2, 3]}),))
        safety = SafetyPolicy(pressure_max6=(100,) * 6, rise_rate6=(100,) * 6,
                              fall_rate6=(100,) * 6, ack_timeout_s=0.1)
        plan = build_plan(model_actions=((10,), (20,)), channel_map=(0,),
                          step_interval_s=0.2, model=model, anchor=anchor,
                          scene=scene, safety=safety)
        return plan, model, anchor, scene, safety

    def test_dt_mismatch_is_detected(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base()
        bad = plan.__class__.from_dict({**plan.to_dict(), "step_interval_s": 0.3})
        codes = {i.code for i in validate_plan(bad, model, anchor, scene, safety).issues}
        self.assertIn("dt_mismatch", codes)

    def test_k_safe_uncertified_blocks(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base(k_safe=None, k_safe_table_px=None)
        codes = {i.code for i in validate_plan(plan, model, anchor, scene, safety).issues}
        self.assertIn("k_safe_uncertified", codes)

    def test_unsupported_obstacle_blocks(self):
        from real_validation.preflight import validate_plan
        plan, model, anchor, scene, safety = self._base()
        scene_with_aabb = Scene("t", (
            ScenePrimitive("target_point", "model", {"xy": [2, 3]}),
            ScenePrimitive("obstacle_aabb", "model", {"min": [1, 1], "max": [2, 2]}),
        ))
        codes = {i.code for i in validate_plan(plan, model, anchor, scene_with_aabb, safety).issues}
        self.assertIn("unsupported_obstacle", codes)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_contracts.PreflightNewGatesTest -v`
Expected: FAIL(断言失败 —— 新检查未实现)

- [ ] **Step 3: 实现 preflight 新检查**

`validate_plan(plan, model, anchor, scene, safety, *, train_dt_s=None)` 签名追加 keyword-only。在现有检查之后追加:

```python
    # ---- P1b 新增 ----
    if model.action_scale_kpa is None:
        add("action_scale_missing", "checkpoint 缺少 action_scale_kpa(deploy_manifest 缺失);"
            "单位链不可知,阻断规划")
    if model.k_safe is None and not (model.k_safe_table_px or {}):
        add("k_safe_uncertified", "模型没有 K_safe 且无 k_safe_table_px 认证表;"
            "无法门控视野,阻断任意 horizon")
    ref_dt = train_dt_s or model.train_dt_measured_s or model.train_dt_nominal_s
    if ref_dt is not None and ref_dt > 0:
        if abs(plan.step_interval_s - ref_dt) / ref_dt >= 0.05:
            add("dt_mismatch", f"step_interval_s={plan.step_interval_s} 与训练 Δt={ref_dt}"
                f" 偏差 ≥5%;动力学时基不一致")
    obstacle_kinds = {item.kind for item in scene.primitives if item.kind.startswith("obstacle_")}
    supported = {"obstacle_circle", "obstacle_aabb"}
    unsupported = obstacle_kinds - supported
    if unsupported:
        add("unsupported_obstacle", f"scene 含 planner 未支持的障碍类型: {sorted(unsupported)};"
            f"碰撞门无法覆盖,阻断")
```

> **B14 语义**:现在 planner 只支持 circle;`obstacle_aabb` 在 Task 6 加入支持后从 `unsupported` 移除。`obstacle_polygon`/`obstacle_mask` 保持阻断(未实现 clearance)。

- [ ] **Step 4: 跑测试 + 提交**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_real_validation_contracts 2>&1 | tail -3
python -m unittest tests.test_real_validation_core 2>&1 | tail -3
```
Expected: contracts 全 PASS;core 20 PASS(fixtures 的 model 已带 k_safe=4、action_scale_kpa=(1.0,),新检查不误报)。

```bash
git add real_validation/preflight.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): preflight 新门(dt_mismatch/k_safe_uncertified/unsupported_obstacle/action_scale_missing)"
```

---

### Task 6: `openloop_planner.py` 单位收口 + 目标扩展 + auto_k + GL 缓存(修 B1/B10/B17)

**Files:**
- Create: `real_validation/planning/__init__.py`(空)
- Create: `real_validation/planning/auto_k.py`
- Modify: `real_validation/openloop_planner.py`
- Modify: `real_validation/runtime/model.py`(FractionalMemory 权重缓存)
- Modify: `tests/test_real_validation_contracts.py`(T2 落地到 planner + T7 rollout 等价 + auto_k)

**Interfaces:**
- Consumes: Task 1 `units.kPa_to_model`、Task 2 `obstacles.obstacle_term`、Task 3 `descriptor.action_scale_kpa`
- Produces:
  - `real_validation.planning.auto_k.step_budget_px(model)` → float(从 `clamp(delta_scale, max=delta_scale_max)` × pc_scale 现算,修 B17)
  - `real_validation.planning.auto_k.select_k_by_gap(gap_tip_px, step_budget_px, k_min, k_max)` → int
  - `ShootingConfig.horizon: int | None = None`、`auto_k: bool = False`(互斥)
  - `OpenLoopShootingPlanner.plan` 在 `descriptor.action_scale_kpa is None` 时 raise(修 B1 fail-closed)
  - 单位收口:`physical(kPa) → kPa_to_model → 模型`;`history` 按 `action_units` 分派
  - 支持 `target_skeleton`(全身形态)与 `obstacle_aabb`(2D SDF)
  - plan metadata 加 `duration_s`(耗时记录,B10 辅助)

- [ ] **Step 1: `auto_k.py`(修 B17)**

创建 `real_validation/planning/__init__.py`(一行 docstring,零 import)与 `real_validation/planning/auto_k.py`:

```python
"""变长 K 选择:据首末差距选规划步数。

CLI inverse_plan.py 的 --auto_k 移植。关键修正(B17):step_budget_px 必须从
**学到的 delta_scale** 现算,不能基于 delta_scale_max=1.0 —— 前向真正的系数是
clamp(delta_scale, max=delta_scale_max),而 delta_scale 是可学参数(初值 0.1,
存在 checkpoint 里)。基于 1.0 会高估 ~10× → K 选小 10× → 步数不够到不了目标。
"""

from __future__ import annotations

import math


def step_budget_px(model) -> float:
    """模型单步最大末端位移(px) = clamp(delta_scale, max=delta_scale_max) × pc_scale。

    与 inverse_plan.py 的 4.0 默认不同:那是基于 delta_scale_max=1.0 的粗估,而真正的
    系数是可学 delta_scale。这里现算,避免 auto_k 高估 ~10×。
    """
    scale = float(torch_clamp_delta(model))
    pc = model.pc_scale.detach().cpu().numpy().reshape(3)
    return scale * float(np.abs(pc[:2]).max())


def torch_clamp_delta(model) -> float:
    import torch
    value = float(torch.clamp(model.delta_scale, max=model.delta_scale_max).item())
    return value


def select_k_by_gap(gap_tip_px: float, step_budget_px: float,
                    k_min: int, k_max: int) -> int:
    """K = clamp(ceil(gap / step_budget), k_min, k_max)。"""
    if k_min > k_max:
        raise ValueError(f"k_min({k_min}) 不能大于 k_max({k_max})")
    k = int(math.ceil(gap_tip_px / max(step_budget_px, 1e-6)))
    return max(k_min, min(k_max, k))


def gap_px_point(tip_px, target_xy, radius: float = 0.0) -> float:
    """单节点目标:到圆边界的距离(圆内 → 0,无需额外行程)。"""
    distance = math.hypot(float(tip_px[0]) - float(target_xy[0]),
                          float(tip_px[1]) - float(target_xy[1]))
    return max(0.0, distance - radius)


def gap_px_skeleton(now_px, goal_px, tolerance: float = 0.0) -> float:
    """整形态目标:瓶颈是走得最远那个节点 → 取 max,不是 node0 也不是 mean。"""
    import numpy as np
    now = np.asarray(now_px, dtype=np.float64)
    goal = np.asarray(goal_px, dtype=np.float64)
    per_node = np.linalg.norm(now[:, :2] - goal[:, :2], axis=1)
    return max(0.0, float(per_node.max()) - tolerance)
```

- [ ] **Step 2: `ShootingConfig` 加 `auto_k`,`__post_init__` 校验互斥**

`openloop_planner.py:24-40` 的 `ShootingConfig`:

```python
@dataclass(frozen=True)
class ShootingConfig:
    horizon: int | None = None        # 与 auto_k 互斥;None 时要求 auto_k=True
    auto_k: bool = False
    k_min: int = 4
    k_max: int = 40
    n_iter: int = 400
    learning_rate: float = 0.05
    n_restarts: int = 4
    w_path: float = 0.2
    w_smooth: float = 0.01
    w_monotonic: float = 1.0
    w_obstacle: float = 1.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if (self.horizon is None) == self.auto_k:
            raise ValueError("horizon 与 auto_k 必须恰有其一(不可同给也不可同缺)")
        if self.horizon is not None and self.horizon <= 0:
            raise ValueError("horizon 必须为正数")
        if self.auto_k and self.k_min > self.k_max:
            raise ValueError("k_min 不能大于 k_max")
        if self.n_iter <= 0 or self.n_restarts <= 0 or self.learning_rate <= 0:
            raise ValueError("n_iter/n_restarts/learning_rate 必须为正数")
```

- [ ] **Step 3: planner 单位收口(B1)+ 目标扩展 + 耗时(核心修改)**

`openloop_planner.py` 的 `plan()` 修改点:

(a) 开头校验 + 单位取用:

```python
        if descriptor.action_scale_kpa is None:
            raise ValueError("checkpoint 缺少 action_scale_kpa(deploy_manifest 缺失);"
                             "单位链不可知,阻断规划(fail-closed)")
        action_scale_kpa = np.asarray(descriptor.action_scale_kpa, dtype=np.float64)
```

(b) history 按 `action_units` 分派(替换 `:164-165`):

```python
        history = torch.tensor(anchor.action_history, dtype=torch.float32, device=device)
        if history.shape[1] != descriptor.action_dim:
            raise ValueError("anchor action history 维度与模型不同")
        if len(history) < descriptor.history_steps:
            raise ValueError("anchor action history 不足 H 步")
        history = history[-descriptor.history_steps:]
        if anchor.action_units == "kpa":
            # 兼容旧标注:真实 kPa → 训练域 [0,1] → /norm_factor
            from .units import kPa_to_model
            history = torch.as_tensor(
                kPa_to_model(history.detach().cpu().numpy(),
                             action_scale_kpa=action_scale_kpa,
                             action_norm_factor=norm),
                dtype=torch.float32, device=device)
        # model_normalized(npz 来源,offline_anchor 新标注)直接用:已是模型单位
```

(c) `_project_actions` 后的模型输入换算(替换 `:202-204`):

```python
                    physical = _project_actions(raw, lo, hi, rise, fall, initial,
                                                step_interval_s)
                    normalized = torch.as_tensor(
                        kPa_to_model(physical.detach().cpu().numpy(),
                                     action_scale_kpa=action_scale_kpa,
                                     action_norm_factor=norm),
                        dtype=torch.float32, device=device)
```

> 注意:`physical` 的梯度必须保留 —— `_project_actions` 输出带梯度;`kPa_to_model` 接受 torch 张量时用 `physical / scale_t / norm` 保梯度。上面用 `detach().cpu()` 会丢梯度,改用 torch 路径:

```python
                    normalized = kPa_to_model(
                        physical, action_scale_kpa=torch.as_tensor(
                            action_scale_kpa, dtype=torch.float32, device=device),
                        action_norm_factor=norm)
```

(d) 目标解析扩展(`_target` 支持 `target_skeleton`):

```python
def _target(scene, model, device):
    targets = [item for item in scene.primitives if item.kind.startswith("target_")]
    if len(targets) != 1:
        raise ValueError("planner 要求 scene 中恰好一个 target 原语")
    item = targets[0]
    if item.kind in {"target_point", "target_circle"}:
        xy = item.geometry.get("xy", item.geometry.get("center"))
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError(f"{item.kind} geometry 需要 xy=[x,y] 或 center=[x,y]")
        point = torch.tensor(xy, dtype=torch.float32, device=device)
        radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
        if radius < 0:
            raise ValueError("目标半径不能为负")
        if item.frame_id == "model":
            target_space = "model"
        elif item.frame_id == "model_normalized":
            target_space = "model_normalized"
        else:
            raise ValueError(f"目标坐标 {item.frame_id} 尚未转换到 model/model_normalized")
        node = int(item.geometry.get("node", 0))
        return {"kind": item.kind, "point": point, "radius": radius,
                "node": node, "item": item, "space": target_space}
    if item.kind == "target_skeleton":
        nodes = item.geometry.get("nodes")
        if not isinstance(nodes, (list, tuple)) or not nodes:
            raise ValueError("target_skeleton geometry 需要非空 nodes=[[x,y]×N]")
        weights = item.geometry.get("weights")
        tolerance = float(item.geometry.get("tolerance_px", 0.0))
        if item.frame_id != "model":
            raise ValueError("target_skeleton 必须已在 model 坐标")
        return {"kind": "target_skeleton", "nodes": torch.tensor(
            nodes, dtype=torch.float32, device=device), "weights": weights,
            "tolerance": tolerance, "item": item, "space": "model"}
    raise ValueError(f"当前 planner 尚不支持 {item.kind}")
```

(e) 损失项:target_point/circle 用单节点 relu(dist-radius)²;target_skeleton 用全节点加权 relu(dist-tolerance)²:

```python
                    if target["kind"] == "target_skeleton":
                        nodes = predictions  # (K,N,3)
                        if target["space"] == "model":
                            physical_nodes = nodes * scale[:3] + center[:3]
                        else:
                            physical_nodes = nodes
                        dists = torch.linalg.vector_norm(
                            physical_nodes[:, :, :2] - target["nodes"][:2].unsqueeze(0),
                            dim=2)  # (K,N)
                        weights = target["weights"]
                        if weights is not None:
                            w = torch.as_tensor(weights, dtype=torch.float32, device=device)
                            errors = (torch.relu(dists - target["tolerance"]).square()
                                      * w).sum(1) / max(1.0, w.sum())
                        else:
                            errors = torch.relu(dists - target["tolerance"]).square().mean(1)
                    else:
                        tip_xy = predictions[:, target["node"], :2]
                        if target["space"] == "model":
                            tip_xy = tip_xy * scale[:2] + center[:2]
                        distances = torch.linalg.vector_norm(
                            tip_xy - target["point"], dim=1)
                        errors = torch.relu(distances - target["radius"]).square()
```

(f) 障碍用共享核(替换 `:219-226` 的 obstacle 块):

```python
                    obstacle = errors.new_zeros(())
                    if obstacles:
                        obstacle = obstacle_term(predictions, scale[:3], center[:3], obstacles)
```

(g) `_obstacles` 支持 `obstacle_aabb`(2D SDF):

```python
def _obstacles(scene):
    supported = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.kind == "obstacle_circle":
            if item.frame_id != "model":
                raise ValueError("圆障碍必须先转换到 model 坐标")
            center = item.geometry.get("center", item.geometry.get("xy"))
            radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
            if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
                raise ValueError("obstacle_circle 需要 center=[x,y] 与正 radius")
            supported.append(("circle", (float(center[0]), float(center[1])),
                              radius + float(item.safety_margin)))
        elif item.kind == "obstacle_aabb":
            if item.frame_id != "model":
                raise ValueError("AABB 障碍必须先转换到 model 坐标")
            lo = item.geometry.get("min")
            hi = item.geometry.get("max")
            if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)) \
                    or len(lo) != 2 or len(hi) != 2:
                raise ValueError("obstacle_aabb 需要 min=[x,y] 与 max=[x,y]")
            margin = float(item.safety_margin)
            supported.append(("aabb", (float(lo[0]) - margin, float(lo[1]) - margin,
                                       float(hi[0]) + margin, float(hi[1]) + margin), 0.0))
        else:
            raise ValueError(f"当前 planner 尚不支持 {item.kind}")
    return supported
```

`obstacle_term` 需要支持 aabb —— 改 `obstacles.obstacle_term` 签名接受 `(kind, geom, r)` 元组。为隔离,新增一个内部适配:

```python
def _obstacle_loss_with_aabb(preds, center, scale, obstacles):
    """preds (K,N,3) 归一化;obstacles = [("circle",(cx,cy),r) | ("aabb",(x0,y0,x1,y1),0)]。"""
    physical = preds * scale + center
    total = preds.new_zeros(())
    for kind, geom, r in obstacles:
        xy = physical[:, :, :2]
        if kind == "circle":
            d = torch.linalg.vector_norm(xy - xy.new_tensor((geom[0], geom[1])), dim=2)
            total = total + torch.relu(r - d).square().mean()
        elif kind == "aabb":
            x0, y0, x1, y1 = geom
            dx = torch.relu(xy.new_tensor(x0) - xy[..., 0]) + torch.relu(xy[..., 0] - xy.new_tensor(x1))
            dy = torch.relu(xy.new_tensor(y0) - xy[..., 1]) + torch.relu(xy[..., 1] - xy.new_tensor(y1))
            sd = torch.sqrt(dx.square() + dy.square())  # 盒外近似(盒内=0,穿透惩罚为 0 → 由 sdf 内为负)
            # 盒内也应有负深度 → 用标准 AABB SDF
            qx = torch.abs(xy[..., 0] - (x0 + x1) / 2) - (x1 - x0) / 2
            qy = torch.abs(xy[..., 1] - (y0 + y1) / 2) - (y1 - y0) / 2
            outside = torch.sqrt(torch.relu(qx).square() + torch.relu(qy).square())
            inside = torch.minimum(torch.maximum(qx, qy), xy.new_zeros(()))
            sdf = outside + inside
            total = total + torch.relu(-sdf).square().mean()
    return total / preds.shape[0]
```

> 这个 AABB SDF 是标准的 2D 有符号距离(盒外正、盒内负),`torch.relu(-sdf).square()` 惩罚盒内穿透。把它放进 `obstacles.py` 作为 `obstacle_term_ext`(保持 `obstacle_term` 对 circle 的简单性)。planner 的 `_obstacles` 返回元组,`obstacle_term_ext` 消费。

(h) auto_k 接线 + 耗时:

```python
        start_wall = time.perf_counter()
        ...
        if config.auto_k:
            from .planning.auto_k import (gap_px_point, gap_px_skeleton,
                                          select_k_by_gap, step_budget_px)
            budget = step_budget_px(model)
            if target["kind"] == "target_skeleton":
                now_px = (state.squeeze(0).detach().cpu().numpy() * scale.detach().cpu().numpy()
                          + center.detach().cpu().numpy())
                goal_px = target["nodes"].cpu().numpy()
                gap = gap_px_skeleton(now_px, goal_px, target["tolerance"])
            else:
                tip_px = (state[0, target["node"], :2].detach().cpu().numpy() * scale[:2].cpu().numpy()
                          + center[:2].cpu().numpy())
                target_xy = target["point"].cpu().numpy()
                gap = gap_px_point(tip_px, target_xy, target["radius"])
            k_effective = select_k_by_gap(gap, budget, config.k_min, config.k_max)
            k_effective = min(k_effective, descriptor.k_safe or k_effective)
        else:
            k_effective = config.horizon
```

> `k_effective = min(k_effective, descriptor.k_safe)` 保证选完不撞 `:140-141` 的硬门;截断时在 metadata 记录。

(i) 把 rollout 里的 `config.horizon` 全替换为 `k_effective`,metadata 加 `duration_s` 与 `auto_k_gap_px`。

- [ ] **Step 4: `runtime/model.py` GL 权重缓存(B10)**

`FractionalMemory.forward` 加缓存(仅部署副本;训练用 `src/encoders/fractional_memory.py` 不改):

```python
    def build_weight_cache(self, length: int, device=None, dtype=None) -> None:
        """规划期预计算 GL 权重为常量。alpha 冻结 → 对动作梯度逐位不变。

        cache key:(length, device, dtype, n_orders, raw_alphas._version, raw_alphas.data_ptr)。
        失效条件:_version 递增(任何对 alpha 的 in-place 改动)/ data_ptr 变化 / length/device/dtype 变化。
        禁止:缓存后二次归一化(会引入 ~1e-8 相对误差)、把 order_weights 折进缓存
        (它是 Parameter,改变乘法结合顺序浮点下不相等)、把 4 次 einsum 合成一次
        (改变 reduction 顺序)。不能用 register_buffer(会进 state_dict,撞 loader 的
        严格校验)—— 用普通属性。
        """
        self._cached_weights = {}
        self._cache_key = (int(length), str(device), str(dtype),
                           int(self.n_orders), int(self.raw_alphas._version),
                           int(self.raw_alphas.data_ptr()))
        alpha = self.alphas.detach()
        for index in range(self.n_orders):
            weights = self._weights(alpha[index], length)
            weights = weights / (weights.abs().sum() + 1e-8)
            self._cached_weights[index] = weights.detach()

    def invalidate_weight_cache(self) -> None:
        self._cached_weights = {}
        self._cache_key = None

    def forward(self, action_window):
        _, length, _ = action_window.shape
        features = []
        for index in range(self.n_orders):
            cached = getattr(self, "_cached_weights", None)
            key = getattr(self, "_cache_key", None)
            valid = (cached and key and key ==
                     (int(length), str(action_window.device), str(action_window.dtype),
                      int(self.n_orders), int(self.raw_alphas._version),
                      int(self.raw_alphas.data_ptr())))
            if valid:
                weights = cached[index].to(device=action_window.device,
                                           dtype=action_window.dtype)
            else:
                weights = self._weights(self.alphas[index], length)
                weights = weights / (weights.abs().sum() + 1e-8)
            value = torch.einsum("k,bkd->bd", weights, action_window)
            features.append(value * self.order_weights[index])
        current = action_window[:, -1, :]
        velocity = (current - action_window[:, -2, :]
                    if length >= 2 else torch.zeros_like(current))
        return self.state_mlp(torch.cat((*features, current, velocity), dim=-1))
```

> **梯度恒等证明**(写入 plan):`_weights` 只依赖 alpha 与 length,与 action_window 无关;规划期 alpha 不被优化(优化变量只有动作 `raw`)。einsum 对 action_window 的梯度 = 常数 w。因此缓存后对动作的梯度**逐位相同**。缓存路径断言 `not torch.is_autocast_enabled()`(autocast 改 einsum 计算 dtype,key 只看输入 dtype 覆盖不到)。

planner 侧接线(Task 6 Step 3 内):

```python
        temporal = getattr(model, "temporal", None)
        if temporal is not None and hasattr(temporal, "build_weight_cache"):
            temporal.build_weight_cache(descriptor.history_steps, device=device,
                                        dtype=torch.float32)
        try:
            ...  # 原规划循环
        finally:
            model.train(was_training)
            if temporal is not None and hasattr(temporal, "invalidate_weight_cache"):
                temporal.invalidate_weight_cache()
```

- [ ] **Step 5: T7 rollout 等价测试(共享 plan_rollout vs 原实现)**

追加到 `tests/test_real_validation_contracts.py`:

```python
class RolloutEquivalenceTest(unittest.TestCase):
    """T7:runtime/rollout.plan_rollout 与 src 侧 rollout 同输入逐元素相等(CPU)。"""

    def test_rollout_matches_reference_implementation(self):
        from real_validation.runtime.rollout import plan_rollout as wb_rollout
        # 参考实现:复制 inverse_plan.plan_rollout 的语义(同窗同 z 演化)
        from real_validation.runtime.model import OpenLoopTransitionModel

        torch.manual_seed(0)
        model = OpenLoopTransitionModel(1, 3, hidden_dim=8, window_size=4,
                                        n_orders=2, z_dim=4).eval()
        buffer = torch.randn(10, 1)
        start_index = 5
        horizon = 3
        s_init = torch.randn(1, 3, 3)

        def reference(buffer_t, t_start, K, window_size, s):
            s_prev = s
            aw0 = buffer_t[t_start - window_size + 1:t_start + 1].unsqueeze(0)
            if aw0.shape[1] < window_size:
                pad = torch.zeros((1, window_size - aw0.shape[1], 1))
                aw0 = torch.cat([pad, aw0], 1)
            z = model.init_z_from_action(aw0)
            preds = []
            for k in range(1, K + 1):
                aw = buffer_t[t_start + k - window_size + 1:t_start + k + 1].unsqueeze(0)
                if aw.shape[1] < window_size:
                    pad = torch.zeros((1, window_size - aw.shape[1], 1))
                    aw = torch.cat([pad, aw], 1)
                out = model(aw, s, s_prev, z)
                s_pred = out["skeleton"]
                z = out["latent_z"]
                preds.append(s_pred.squeeze(0))
                s_prev, s = s, s_pred
            return torch.stack(preds, 0)

        with torch.no_grad():
            expected = reference(buffer, start_index, horizon, 4, s_init)
            got = wb_rollout(model, buffer, start_index, horizon, 4, s_init)
        self.assertTrue(torch.allclose(got, expected, atol=1e-6, rtol=1e-6))
```

- [ ] **Step 6: 跑测试 + 提交**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest tests.test_real_validation_contracts 2>&1 | tail -3
python -m unittest tests.test_real_validation_core 2>&1 | tail -3
```
Expected: contracts 全 PASS(含 T7 等价、auto_k、单位收口);core 20 PASS。

```bash
git add real_validation/planning/ real_validation/openloop_planner.py real_validation/runtime/model.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): planner 单位收口+全身目标+AABB障碍+auto_k(B1/B17)+ GL 权重缓存(B10)+ 耗时记录"
```

---

### Task 7: `session.py` 守卫(B16)+ `main_validation.py`(B8/B9/B11/B15)

**Files:**
- Modify: `real_validation/session.py`
- Modify: `real_validation/main_validation.py`
- Modify: `tests/test_real_validation_core.py`(追加守卫测试)

**Interfaces:**
- Consumes: Task 4 的契约、Task 3 的 descriptor
- Produces:
  - `ExperimentSession.set_scene/set_anchor/set_safety` 加状态守卫(与 `configure_model` 对称,只在 `{IDLE, READY}` 可改,否则 raise)—— 修 B16
  - `ExperimentSession.invalidate_model(reason)` → 置 `model=None/anchor=None/plan=None`,状态守卫同 `configure_model` —— 修 B15
  - `main_validation._refresh()` 锁页:EXECUTING 时禁用页 1/2/3 的编辑按钮 —— 修 B8
  - `plan_dt` 默认从 `descriptor.train_dt_measured_s`/`nominal_s` 取 —— 修 B5
  - K_safe 从 `k_safe_table_px` 自动读(按容差),不再手填 —— 修 B9
  - `_ModelLoadThread.run` 拆 `ModelLoadError`/`ValueError` 到 `_model_load_failed`,清 runtime —— 修 B11/B15

- [ ] **Step 1: `session.py` 守卫 + invalidate_model**

`models.py` 已有 `_return_to_idle_if_ready`。给三个 setter 加守卫:

```python
    def set_anchor(self, anchor: Anchor) -> None:
        self._guard_editable("anchor")
        self.anchor = anchor
        self.plan = None
        self._record("anchor_changed", anchor_id=anchor.anchor_id)
        self._return_to_idle_if_ready("anchor changed")
        self.save_snapshot()

    def set_scene(self, scene: Scene) -> None:
        self._guard_editable("scene")
        self.scene = scene
        self.plan = None
        self._record("scene_changed", scene_digest=scene.digest)
        self._return_to_idle_if_ready("scene changed")
        self.save_snapshot()

    def set_safety(self, safety: SafetyPolicy) -> None:
        self._guard_editable("safety")
        self.safety = safety
        self.plan = None
        self._record("safety_changed", safety_digest=safety.digest)
        self._return_to_idle_if_ready("safety changed")
        self.save_snapshot()

    def _guard_editable(self, field_name: str) -> None:
        if self.state not in {SessionState.IDLE, SessionState.READY}:
            raise RuntimeError(f"只能在 idle/ready 状态修改 {field_name},当前 {self.state.value}")

    def invalidate_model(self, reason: str = "") -> None:
        """清除模型 descriptor(加载失败时调用)。与 configure_model 同守卫。"""
        if self.state not in {SessionState.IDLE, SessionState.READY}:
            raise RuntimeError("只能在 idle/ready 状态清除模型")
        self.model = None
        self.anchor = None
        self.plan = None
        self._record("model_invalidated", reason=reason)
        self._return_to_idle_if_ready(reason or "model invalidated")
        self.save_snapshot()
```

> ⚠️ **B16 的危害**:执行中 `set_scene` 会 `self.plan = None` + `save_snapshot()` → `experiment.json` 被写成 `"plan": null` 而命令正在下发 → **执行记录与实际下发计划脱钩 = 溯源腐败**。守卫是必须的,不是可选的。

- [ ] **Step 2: `main_validation.py` B8 锁页 + B5 plan_dt + B9 K_safe + B11/B15**

(a) `_refresh()`(`main_validation.py:582-590`)追加锁页逻辑:

```python
        executing = bool(self.session and self.session.state in {
            SessionState.EXECUTING, SessionState.PAUSED, SessionState.ARMED})
        # B8:执行中锁页 1/2/3 的编辑按钮
        self.tabs.setTabEnabled(1, not executing)   # Observe & Scene
        self.tabs.setTabEnabled(2, not executing)   # Plan
        self.tabs.setTabEnabled(0, not executing)   # Setup
```

(b) `_model_loaded`(`main_validation.py:321-332`)追加 plan_dt 默认 + K_safe 自动:

```python
    def _model_loaded(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        assert self.session is not None
        self.session.configure_model(runtime.descriptor)
        descriptor = runtime.descriptor
        self.model_summary.setPlainText(
            f"type={descriptor.model_type}\nclass={descriptor.model_class}\n"
            f"action_dim={descriptor.action_dim}\n"
            f"nodes={descriptor.n_nodes}\nH={descriptor.history_steps}\n"
            f"K_train={descriptor.k_train}\nK_safe={descriptor.k_safe}\n"
            f"train_dt={descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s}\n"
            f"action_scale_kpa={descriptor.action_scale_kpa}\n"
            f"sha256={descriptor.checkpoint_hash}")
        # B5:plan_dt 默认取训练实测 Δt(不再硬编码 0.2)
        ref_dt = descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s
        if ref_dt:
            self.plan_dt.setValue(float(ref_dt))
        # B9:K_safe 从 k_safe_table_px 自动读(按 10px 容差),不再手填
        if descriptor.k_safe_table_px:
            k = descriptor.k_safe_table_px.get("10px", descriptor.k_safe_table_px.get("5px"))
            if k:
                self.k_safe.setValue(int(k))
        self._refresh()
```

(c) B11/B15:`_ModelLoadThread.run` 拆异常 + 专用槽:

```python
    def run(self) -> None:
        try:
            runtime = ModelRuntime(self.checkpoint, self.data_dir or None, self.device,
                                   k_safe=self.k_safe)
            self.loaded.emit(runtime)
        except (ModelLoadError, FileNotFoundError, ValueError) as error:
            self.failed.emit(str(error))               # 可操作提示,不弹 traceback
        except Exception:
            self.failed.emit(traceback.format_exc())   # 真 bug 才给 traceback
```

`model_runtime.py` 加 `ModelLoadError`(Task 3 一步),并把 `__init__` 的 `if not checkpoint_path.is_file(): raise FileNotFoundError(...)` 改为 `raise ModelLoadError(...)` 带可操作提示(从服务器复制 best_model.pt + config.json + deploy_manifest.json 到 checkpoints/current/)。

`main_validation.py` `_load_model` 的连接改:

```python
        self._model_thread.failed.connect(self._model_load_failed)

    def _model_load_failed(self, message: str) -> None:
        self.runtime = None
        self.model_summary.setPlainText("模型未加载")
        if self.session is not None and self.session.model is not None:
            try:
                self.session.invalidate_model("model reload failed")
            except RuntimeError as error:
                self._log(f"WARN: 无法清除旧模型 descriptor: {error}")
        self._error(message)
        self._refresh()
```

> **B15 危害**:现在 `ModelRuntime` 抛错后 `self.runtime` 保持旧值,`model_summary` 永远停在"正在后台加载模型……",操作员以为换了模型,实际后续 Plan 用旧 runtime,而 preflight 比对的两个 hash 都是旧的 → 照样放行。

- [ ] **Step 3: 追加守卫测试**

`tests/test_real_validation_core.py`:

```python
    def test_scene_set_in_executing_is_blocked(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.set_scene(scene); session.set_safety(safety)
            session.accept_plan(plan)
            session.arm()
            session.transition(SessionState.EXECUTING, "test")
            with self.assertRaises(RuntimeError):
                session.set_scene(scene)          # 执行中禁止改 scene(B16)
            with self.assertRaises(RuntimeError):
                session.set_anchor(anchor)

    def test_invalidate_model_clears_descriptor(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.invalidate_model("load failed")
            self.assertIsNone(session.model)
            self.assertIsNone(session.anchor)
            self.assertIsNone(session.plan)
```

- [ ] **Step 4: 跑测试 + 提交**

Run: `cd /Data5/ddf/projects/SelfSoftRobot && python -m unittest tests.test_real_validation_core 2>&1 | tail -3`(期望 20 + 2 新增 = 22)

```bash
git add real_validation/session.py real_validation/main_validation.py real_validation/model_runtime.py tests/test_real_validation_core.py
git commit -m "feat(real_validation): session 状态守卫(B16)+ invalidate_model(B15)+ GUI 锁页(B8)/plan_dt(B5)/K_safe 自动(B9)"
```

---

### Task 8: `build_deploy_manifest.py` 生产者(修 B3 落地)

**Files:**
- Create: `scripts/utils/build_deploy_manifest.py`

**Interfaces:**
- Consumes: `masks_to_transition_npz.action_max_per_channel`(复用 fallback)、`io.file_sha256`
- Produces: `deploy_manifest.json`(与 checkpoint 同目录),3 源 join:
  1. checkpoint + 其 config.json(网络形状)
  2. `real_capture/data/raw/<seq>/meta.json`(hi6 → action_scale_kpa、action_interval_s)+ `frame_times.txt`(实测 Δt)
  3. `<exp>/eval_horizon/horizon_summary.json`(k_safe_table_px)

- [ ] **Step 1: 实现**

```python
"""从已有实验生成 deploy_manifest.json。

3 源 join,必须在服务器上跑(PC 上没有 real_capture/data/raw):
  1. checkpoint + config.json           → 网络形状 + data_dirs
  2. raw/<seq>/meta.json + frame_times  → action_scale_kpa(经 action_max_per_channel)/ train_dt
  3. eval_horizon/horizon_summary.json  → k_safe_table_px

Usage:
  python scripts/utils/build_deploy_manifest.py \
      --exp-dir train_log/open_loop_transition/exp_20260714_8 \
      --raw-seq real_capture/data/raw/seq_20260627_163921 \
      [--horizon-summary <exp>/eval_horizon/horizon_summary.json] \
      [--out <exp>/deploy_manifest.json]
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from scripts.real.masks_to_transition_npz import action_max_per_channel
from real_validation.io import file_sha256


def find_checkpoint(exp_dir):
    """exp 根 → phase_*/model/best_model.pt。"""
    candidates = sorted(glob.glob(os.path.join(exp_dir, "phase_*", "model", "best_model.pt")))
    if not candidates:
        raise FileNotFoundError(f"{exp_dir} 下没有 phase_*/model/best_model.pt")
    return candidates[0]


def measure_train_dt(raw_seq):
    """frame_times.txt → (measured_s, std_s)。禁止硬写 0.203125。"""
    times_path = os.path.join(raw_seq, "frame_times.txt")
    with open(times_path) as stream:
        times = np.array([float(line) for line in stream if line.strip()])
    diffs = np.diff(times)
    return float(diffs.mean()), float(diffs.std())


def main():
    parser = argparse.ArgumentParser(description="生成 deploy_manifest.json")
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--raw-seq", required=True)
    parser.add_argument("--horizon-summary")
    parser.add_argument("--out")
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.exp_dir)
    config_path = os.path.join(os.path.dirname(checkpoint), "..", "..", "config.json")
    if not os.path.isfile(config_path):
        # 也找 exp 根的 config.json(向上最多 3 级)
        for _ in range(3):
            candidate = os.path.join(os.path.dirname(checkpoint), "..", "config.json")
            if os.path.isfile(candidate):
                config_path = candidate
                break
            checkpoint = os.path.dirname(checkpoint)
    with open(config_path) as stream:
        config = json.load(stream)

    meta_path = os.path.join(args.raw_seq, "meta.json")
    with open(meta_path) as stream:
        meta = json.load(stream)
    channels = [int(meta.get("active_channel", 0))]
    # 用真实的 actions6.csv 算 max(复用 action_max_per_channel 的 fallback)
    actions_csv = os.path.join(args.raw_seq, "actions6.csv")
    raw_actions = np.atleast_2d(np.genfromtxt(actions_csv, delimiter=",", dtype=float))
    while raw_actions.shape[0] and np.isnan(raw_actions[0]).all():
        raw_actions = raw_actions[1:]
    maxes = action_max_per_channel(args.raw_seq, channels, raw_actions[:, channels[0] + 1:channels[0] + 2])
    dt_mean, dt_std = measure_train_dt(args.raw_seq)

    data_dirs = config.get("data_dirs", {}).get("sequence", "")
    base = os.path.basename(data_dirs)
    if "_sam2" in base:
        mask_source, provenance = "sam2", "path_suffix"
    elif "_rep" in base:
        mask_source, provenance = "masks_repaired", "path_suffix"
    else:
        mask_source, provenance = "white_on_blue", "path_suffix"

    segment_params = None
    if mask_source == "white_on_blue":
        seg_meta = os.path.join(os.path.dirname(args.raw_seq),
                                "derived", os.path.basename(args.raw_seq), "segment_meta.json")
        if os.path.isfile(seg_meta):
            with open(seg_meta) as stream:
                segment_params = json.load(stream).get("params")

    k_safe_table_px = None
    if args.horizon_summary and os.path.isfile(args.horizon_summary):
        with open(args.horizon_summary) as stream:
            summary = json.load(stream)
        for entry in summary.get("summaries", []):
            if entry.get("model_type") == "open_loop":
                k_safe_table_px = {
                    "5px": entry.get("Kmax_px_5"),
                    "10px": entry.get("Kmax_px_10"),
                    "20px": entry.get("Kmax_px_20"),
                }
                k_safe_table_px = {k: int(v) for k, v in k_safe_table_px.items() if v is not None}
                break

    manifest = {
        "schema_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "action_scale_kpa": [float(v) for v in maxes],
        "channel_map": channels,
        "train_dt_nominal_s": float(meta.get("action_interval_s", 0.2)),
        "train_dt_measured_s": dt_mean,
        "train_dt_std_s": dt_std,
        "mask_source": mask_source,
        "mask_source_provenance": provenance,
        "segment_params": segment_params,
        "camera": None,
        "reference_frame": None,
        "reference_frame_sha256": None,
        "mask_area_median_px": None,
        "registration_residual_max_px": 2.0,
        "k_safe_table_px": k_safe_table_px,
        "train_sequences": [os.path.basename(args.raw_seq)],
        "n_nodes": int(config.get("n_nodes", 15)),
        "window_size": int(config.get("window_size", 40)),
        "z_dim": int(config.get("z_dim", 16)),
        "episode_len": int(config.get("episode_len", 40)),
        "action_dim": int(config.get("action_dim", 1)),
        "encoder_type": str(config.get("encoder_type", "fractional")),
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "n_scales": int(config.get("n_scales", 4)),
    }

    out = args.out or os.path.join(os.path.dirname(checkpoint), "..", "..", "deploy_manifest.json")
    with open(out, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"manifest 写入 {out}")
    print(f"  action_scale_kpa={manifest['action_scale_kpa']}  train_dt={dt_mean:.4f}±{dt_std:.4f}"
          f"  mask_source={mask_source}  k_safe={k_safe_table_px}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 用现有实验冒烟(exp_20260714_8)**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python scripts/utils/build_deploy_manifest.py \
    --exp-dir train_log/open_loop_transition/exp_20260714_8 \
    --raw-seq real_capture/data/raw/seq_20260627_163921 \
    --horizon-summary train_log/open_loop_transition/exp_20260714_8/eval_horizon/horizon_summary.json \
    --out /tmp/deploy_manifest_smoke.json
python -c "import json; m=json.load(open('/tmp/deploy_manifest_smoke.json')); print('action_scale_kpa', m['action_scale_kpa'], 'k_safe', m['k_safe_table_px'], 'dt', m['train_dt_measured_s'])"
```
Expected:`action_scale_kpa=[150.0]`、`k_safe={'5px':51,'10px':124,'20px':250}`、`dt≈0.2031`。

> ⚠️ 冒烟产物 `deploy_manifest.json` 只写 `/tmp`,**不写进实验目录** —— 因为 exp_20260714_8 训在 SAM2 上(segment_params=None),它作为部署主线的 manifest 不完整。这是 P2 重采重训后要重新生成的。

- [ ] **Step 3: 提交**

```bash
git add scripts/utils/build_deploy_manifest.py
git commit -m "feat(real_validation): build_deploy_manifest 生产者(3 源 join,复用 action_max_per_channel)"
```

---

### Task 9: 终验 + T4b(CLI/GUI 一致)

**Files:**
- Modify: `tests/test_real_validation_contracts.py`(追加 T4b)
- Modify: `real_validation/metrics.py`、`real_validation/widgets/plan_preview.py`(新障碍/目标绘制,最小改动)

**Interfaces:**
- Consumes: Task 1-8 全部
- Produces: T4b 测试 + 终验通过

- [ ] **Step 1: T4b CLI/GUI 小规模一致**

追加到 `tests/test_real_validation_contracts.py`:

```python
class CliGuiConsistencyTest(unittest.TestCase):
    """T4b:共享目标核在 CLI 与 GUI 之间逐位一致(CPU,小规模)。

    前置:两侧 norm_factor 必须相等 —— CLI 走 model_loader,找不到 action_norm_factor.txt
    时会静默回落 1.0(model_loader.py:73),不断言就可能"权重相同、norm_factor 不同"假通过。
    """

    def test_shared_objective_matches_across_call_sites(self):
        from real_validation.obstacles import cli_obstacle_loss, obstacle_term
        torch.manual_seed(3)
        preds = torch.randn(4, 15, 3, dtype=torch.float64)
        center = torch.zeros(3, dtype=torch.float64)
        scale = torch.ones(3, dtype=torch.float64)
        obstacles = [(10.0, 10.0, 2.0)]
        via_cli = cli_obstacle_loss(preds, center, scale, obstacles)
        via_shared = obstacle_term(preds, center, scale, obstacles, reduce="mean")
        self.assertTrue(torch.equal(via_cli, via_shared))
```

- [ ] **Step 2: `metrics.py` + `plan_preview.py` 最小适配**

`metrics.py:_scene_metrics` 的 `_scene_metrics` 对非 circle obstacle 现在 raise。改为:对 `obstacle_aabb` 用 `clearance_min` 的 numpy 版计算最小净距,对 `obstacle_polygon`/`obstacle_mask` 保持 raise(未实现):

```python
        if item.kind == "obstacle_circle" and item.frame_id == "model":
            center = item.geometry.get("center", item.geometry.get("xy"))
            radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
            radius += float(item.safety_margin)
            distance = np.linalg.norm(xy - np.asarray(center, dtype=np.float64), axis=2)
            clearances.append(distance - radius)
        elif item.kind == "obstacle_aabb" and item.frame_id == "model":
            lo = np.asarray(item.geometry["min"], dtype=np.float64)
            hi = np.asarray(item.geometry["max"], dtype=np.float64)
            margin = float(item.safety_margin)
            lo -= margin; hi += margin
            # AABB 有符号距离
            cx = np.maximum(lo[0] - xy[..., 0], xy[..., 0] - hi[0])
            cy = np.maximum(lo[1] - xy[..., 1], xy[..., 1] - hi[1])
            outside = np.sqrt(np.maximum(cx, 0) ** 2 + np.maximum(cy, 0) ** 2)
            inside = np.minimum(np.maximum(cx, cy), 0.0)
            clearances.append(outside + inside)
        else:
            raise ValueError(f"scene metrics 尚不支持 {item.kind}@{item.frame_id}")
```

`widgets/plan_preview.py:_draw_scene` 对 `obstacle_aabb` 画矩形,对 `target_skeleton` 画目标节点(最小改动):

```python
            elif primitive.kind == "obstacle_aabb" and xy is None:
                lo = primitive.geometry.get("min"); hi = primitive.geometry.get("max")
                if lo and hi:
                    item = QGraphicsRectItem(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1])
                    item.setPen(QPen(Qt.darkYellow, 2, Qt.DashLine))
            elif primitive.kind == "target_skeleton":
                nodes = primitive.geometry.get("nodes")
                if nodes:
                    xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
                    item = pg.ScatterPlotItem(xs, ys, symbol="x", size=8,
                                              pen=pg.mkPen("#E53E3E", width=2))
```

(需在 plan_preview 顶部 import `QGraphicsRectItem`。)

- [ ] **Step 3: 终验 —— 全量测试 + 卫生 + 向后兼容**

Run:
```bash
cd /Data5/ddf/projects/SelfSoftRobot
python -m unittest discover -s tests 2>&1 | tail -3          # 期望全部通过
python -c "import real_validation, sys; print('root deps:', sorted(k for k in sys.modules if k.split('.')[0] in ('torch','cv2','scipy','PyQt5','pyqtgraph')))"
python -c "import src.data.real, src.utils.skeleton_2d; print('shim ok')"
python scripts/control/inverse_plan.py --help > /dev/null && echo "inverse_plan --help ok"
python scripts/real/segment_batch.py --help > /dev/null && echo "segment_batch ok"
python scripts/real/masks_to_transition_npz.py --help > /dev/null && echo "masks_to_npz ok"
python scripts/real/compare_skeleton_methods.py --help > /dev/null && echo "compare ok"
python scripts/utils/build_deploy_manifest.py --help > /dev/null && echo "build_manifest ok"
```
Expected:全量测试通过;`root deps: []`;6 条 --help 全成功。

- [ ] **Step 4: 提交**

```bash
git add real_validation/metrics.py real_validation/widgets/plan_preview.py tests/test_real_validation_contracts.py
git commit -m "feat(real_validation): metrics/plan_preview 适配 AABB+target_skeleton + T4b 终验"
```

---

## Self-Review

**1. Spec 覆盖(P1b = spec §9 的 M3)**

| spec 要求 | 落在哪 |
|---|---|
| B1 动作单位错 → units.py 单点收口 | Task 1 + Task 6(a-c) |
| B2 offline_anchor action_units 标注 | Task 4 Step 4 |
| B3 deploy_manifest.json + ModelDescriptor 字段 | Task 3 |
| B4 障碍聚合口径统一 | Task 2(obstacles.py + CLI 委托) |
| B5 step_interval_s 默认 + preflight dt_mismatch | Task 7(b) + Task 5 |
| B7 Scene 增删改 | Task 4 Step 1 |
| B8 GUI 锁页 | Task 7(a) |
| B9 K_safe 自动读 | Task 7(b) |
| B10 GL 权重缓存 | Task 6 Step 4 |
| B11 checkpoint 缺失优雅报错 | Task 7(c) |
| B14 碰撞门覆盖非 circle | Task 5(`unsupported_obstacle`)+ Task 6(AABB SDF) |
| B15 模型加载失败清 runtime | Task 7(c) |
| B16 set_scene/等守卫 | Task 7 Step 1 |
| B17 step_budget 从 delta_scale 现算 | Task 6 Step 1(auto_k) |
| ScenePrimitive 新消费者(target_skeleton/obstacle_aabb/obstacle_polygon) | Task 6(d-f)(polygon 保持阻断,见下) |
| 测试 T2/T3/T4a/T4b/T7/T8 | Task 1(T2)、Task 2(T4a)、Task 3(T8)、Task 6(T7)、Task 9(T4b);**T3 坐标往返**见下方补 |
| build_deploy_manifest.py 生产者 | Task 8 |

**缺口补:T3 坐标往返** —— `coordinate_system.PlanarTransform` 已有 `roundtrip_error` 与既有测试(`test_planar_transform_roundtrip`)。T3 的"camera_pixel → model → camera_pixel"在 P1b 不适用(重采后是恒等映射,Task 5 已建 registration 检测)。故 T3 已由 P1a 的 registration 测试覆盖(平移 3px 恢复),**不重复建**。

**obstacle_polygon 保持阻断**:planner `_obstacles` 对 polygon 仍 raise;preflight `unsupported_obstacle` 阻断。这是**有意**范围控制 —— 凸多边形 SDF 复杂,且没有真实数据证明需要。spec 的"非圆障碍"第一版只落地 AABB。

**2. 占位扫描**:全文无 TBD/TODO;每个代码步骤都是可直接粘贴的完整实现或精确替换点。

**3. 类型一致性**:`units.kPa_to_model(actions, action_scale_kpa, action_norm_factor)` 在 Task 1 定义、Task 6 消费;`obstacles.obstacle_term(preds, pc_center, pc_scale, obstacles)` Task 2 → Task 6;`DeployManifest` 字段名与 `ModelDescriptor` 新字段一一对应;`ShootingConfig.auto_k/k_min/k_max` Task 6 定义、GUI 侧(下一轮)消费。

**4. 歧义**:`horizon=None + auto_k=False` 抛 ValueError(互斥);`k_effective = min(k, k_safe)` 截断并在 metadata 记录(避免撞硬门)。AABB SDF 标准公式(盒外正/盒内负),`relu(-sdf)` 罚穿透。

## 已知缺口(P1b 交付后仍未解决)

| 缺口 | 归属 |
|---|---|
| `obstacle_polygon`/`obstacle_mask` 无 SDF,planner/preflight 阻断 | 有真实需求时再做凸多边形 SDF |
| GUI 尚未暴露 `target_skeleton`/`obstacle_aabb` 的交互创建(只支持解析) | P3(M4)camera_view + scene_editor |
| `build_deploy_manifest` 冒烟产物不写进 exp 目录(exp_20260714_8 训在 SAM2 上,segment_params=None) | P2 重采后重新生成 |
| 多起点批并行不做(见 Global Constraints) | 耗时基准出来后单独评估 |
| B6/B12(jitter/REANCHOR) | M5 |
| `live_anchor.py` | P3(M4) |

**提交史(预期)**:9 个提交,每任务一个,全部落在 `feat/real-data-transition`。

```
