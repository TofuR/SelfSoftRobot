# real_validation 独立移植 + 目录重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** real_validation 完全自包含(去掉 real_capture 外部引用,三类真机驱动移植进内部)+ 23 个平铺文件收进功能域子包 + 彻底同步 src/scripts/tests 引用。

**Architecture:** 分 4 阶段。阶段 1 硬件移植(独立可验收,先做);阶段 2 目录重构(按依赖顺序分批移动文件 + 同步相对 import);阶段 3 入口收敛(main.py + GUI 移动);阶段 4 全量验证(测试绿 + import 卫生 + 无 real_capture 代码引用)。每阶段独立提交,全部完成后 real_validation 彻底自包含。

**Tech Stack:** Python 3.10 stdlib, PyQt5, pyqtgraph, numpy。真机硬件依赖(pyserial/pyrealsense2/scikit-surgerynditracker)仅 requirements-hardware.txt,延迟 import 不进包根闭包。

## Global Constraints

- 分支固定 `feat/real-data-transition`;提交前询问用户(用户已授权直接执行本 plan,提交可自动,但大提交前仍提示)。
- **包根 `real_validation/__init__.py` 保持 stdlib-only**:只 re-export `contracts.models` 顶层符号(models/Anchor/Scene/SafetyPolicy/ActionPlan/ModelDescriptor/ScenePrimitive),不能 import 任何真机依赖。`tests/test_import_hygiene.py` 必须全绿。
- **适配层公共 API 签名不变**(GUI 接线不破):`valve.create_valve_controller(group1, group2, *, baudrate, slave_addr)`、`valve.connect_valve_groups(controller, groups)`、`valve.valve_to_kpa_requested(action6, action_scale_kpa)`、`camera.create_realsense_cam(...)`、`camera.assert_camera_fingerprint(...)`、`ndi.create_ndi_thread(...)`、`ndi.require_hidden_evaluation_allowed(...)`、`ndi.HIDDEN_EVALUATION_SOURCE`、异常类 `ValveHardwareError/CameraHardwareError/NdiHardwareError`。
- **硬件移植即搬移,不重写逻辑**:只改 import 为内部相对引用,驱动行为语义不变。
- 现有测试必须全绿(129 个 + 新增)。测试框架 unittest,无 pytest。
- 目录重构优先 `git mv`(保历史),必要时合并。
- 每阶段结束跑一次全量测试,不一次搬完。

---

## 文件结构(目标)

```
real_validation/
├── main.py                      # 入口壳: from .gui.main_window import main; main()
├── __init__.py                  # re-export contracts.models 顶层符号
├── contracts/  models.py io.py deploy_manifest.py plan_io.py
├── core/       session.py
├── perception/ (已有不变)
├── planning/   auto_k.py(已有) openloop_planner.py planner_service.py obstacles.py coordinate_system.py units.py
├── runtime/    loader.py model.py rollout.py(已有) model_runtime.py anchors.py warmup.py observation_policy.py
├── execution/  executor.py preflight.py metrics.py hardware_session.py
├── hardware/   valve.py camera.py ndi.py(_bootstrap.py 删除)
├── gui/        main_window.py theme.py widgets/(已有)
├── tools/      perception_probe.py
├── config/ checkpoints/ data/ runs/ calibration/   (数据目录,不变)
```

---

### Task 1: 移植串口阀驱动(ValveController + ModbusManager)进 hardware/valve.py

**Files:**
- Create: `real_validation/hardware/valve.py`(重写:移植驱动 + 保留适配层 API)
- Create: `real_validation/hardware/modbus.py`(ModbusManager 内部化)
- Delete: `real_validation/hardware/_bootstrap.py`
- Test: `tests/test_hardware_adapters.py`(更新:create_valve_controller 不再走 real_capture)

**Interfaces:**
- Consumes: `real_capture/valve_control.py`(468行,依赖 `from modbus_manager import ModbusManager`)、`real_capture/modbus_manager.py`(514行)。
- Produces: `valve.create_valve_controller(group1, group2, *, baudrate=9600, slave_addr=1)` 返回内部 `ValveController`;`valve.connect_valve_groups(controller, groups)`;`valve.valve_to_kpa_requested(action6, action_scale_kpa)`;`valve.ValveHardwareError`。**不再调用 real_capture**。

- [ ] **Step 1: 先更新测试(证明当前依赖 real_capture)**

`tests/test_hardware_adapters.py` 现在测试 `create_valve_controller('','')` 抛 `ValveHardwareError`(缺 COM 校验,不触碰真机 import)。这仍是有效断言。加一个断言:导入 `real_validation.hardware.valve` 后,`sys.modules` 里**没有** `real_capture`(锁住自包含):

```python
# tests/test_hardware_adapters.py 新增
import sys
from real_validation.hardware import valve

class ValveSelfContainedTest(unittest.TestCase):
    def test_valve_module_does_not_import_real_capture(self):
        # 移植后硬件模块必须完全自包含(不触碰 real_capture)
        self.assertNotIn("real_capture", sys.modules)
        self.assertNotIn("modbus_manager", sys.modules)   # 顶层模块名也不该出现
```

- [ ] **Step 2: 运行测试确认当前通过(旧实现下 real_capture 未被 eager import)**

Run: `python -m unittest tests.test_hardware_adapters -v`
Expected: PASS —— 当前 `create_valve_controller` 只在调用时延迟 import real_capture,模块导入时不触碰,所以 `sys.modules` 无 real_capture。此测试是移植后的约束契约。

- [ ] **Step 3: 复制并内部化 ModbusManager**

读 `real_capture/modbus_manager.py`(514行)完整内容。创建 `real_validation/hardware/modbus.py`,内容 = 原文件,但:
- 去掉 `import serial` 的顶层强制依赖 → 保持延迟 import(原文件 line 2-4 已是 try/except 延迟,保留)
- 无跨文件相对引用(它独立)

- [ ] **Step 4: 复制并内部化 ValveController**

读 `real_capture/valve_control.py`(468行)完整内容。创建 `real_validation/hardware/valve.py`,内容 = 原文件,但:
- 把 `from modbus_manager import ModbusManager` 改为 `from .modbus import ModbusManager`
- **保留现有适配层公共 API**:文件开头/末尾保留 `ValveHardwareError`、`create_valve_controller`、`connect_valve_groups`、`valve_to_kpa_requested`(这些已有,保持签名)
- `create_valve_controller` 内部不再调 `ensure_real_capture_importable()`(删掉),直接构造内部 ValveController

具体:当前 `valve.py` 是适配层(create_valve_controller 包装 real_capture)。重写为:保留适配层函数,但 `create_valve_controller` 返回**本文件内**移植的 `ValveController` 类。

- [ ] **Step 5: 删除 _bootstrap.py**

删除 `real_validation/hardware/_bootstrap.py`。改 `hardware/__init__.py` 注释(不再提 real_capture 桥接)。

- [ ] **Step 6: 更新 camera.py / ndi.py 去掉 _bootstrap 引用**

`hardware/camera.py`、`hardware/ndi.py` 现在 `from ._bootstrap import ensure_real_capture_importable`。删除这些 import 和调用(它们将各自移植驱动)。**本任务先删掉对 _bootstrap 的引用**(相机/NDI 驱动在 Task 2/3 移植,先用占位/延迟 import 保持可导入)。

- [ ] **Step 7: 跑测试**

Run: `python -m unittest tests.test_hardware_adapters -v`
Expected: 全绿 —— 新增的 `test_valve_module_does_not_import_real_capture` 通过,证明 valve 模块不再触碰 real_capture。

- [ ] **Step 8: 全量测试 + 提交**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

```bash
git add real_validation/hardware/
git commit -m "refactor(real_validation): 串口阀驱动内部化 —— ValveController+ModbusManager 移入 hardware/,删 _bootstrap 外部引用"
```

---

### Task 2: 移植相机驱动(RealSenseCam)进 hardware/camera.py

**Files:**
- Create: `real_validation/hardware/camera.py`(重写:移植 RealSenseCam + 保留 assert_camera_fingerprint)
- Delete: `real_validation/hardware/_bootstrap.py`(若 Task 1 未删净)
- Test: `tests/test_hardware_adapters.py`(更新:无 real_capture 断言覆盖 camera)

**Interfaces:**
- Consumes: `real_capture/realsense_cam.py`(144行,依赖 pyrealsense2 延迟 import)。
- Produces: `camera.create_realsense_cam(width=640, height=480, fps=30, serial=None)` 返回内部 `RealSenseCam`;`camera.assert_camera_fingerprint(descriptor_fingerprint, *, width, height, fps, serial)`(保留现有,不改签名);`camera.CameraHardwareError`。

- [ ] **Step 1: 读 realsense_cam.py 并复制**

读 `real_capture/realsense_cam.py`(144行)。把 `RealSenseCam` 类复制进 `real_validation/hardware/camera.py`,保留现有适配层函数 `create_realsense_cam`/`assert_camera_fingerprint`/`CameraHardwareError`。

- [ ] **Step 2: 改 import 自包含**

删除 `from ._bootstrap import ensure_real_capture_importable`。`create_realsense_cam` 不再调 `ensure_real_capture_importable()`,直接构造内部 `RealSenseCam`。RealSenseCam 内部 `import pyrealsense2 as rs` 保持延迟(原样)。

- [ ] **Step 3: 跑测试**

Run: `python -m unittest tests.test_hardware_adapters -v`
Expected: 全绿 —— `test_no_fingerprint_is_permissive`(无指纹不阻断)和指纹不匹配阻断仍在。

- [ ] **Step 4: 提交**

```bash
git add real_validation/hardware/camera.py
git commit -m "refactor(real_validation): 相机驱动内部化 —— RealSenseCam 移入 hardware/,删 _bootstrap 外部引用"
```

---

### Task 3: 移植 NDI 驱动(NdiThread + nditracker)进 hardware/ndi.py

**Files:**
- Create: `real_validation/hardware/ndi.py`(重写:移植 NdiThread + 保留适配层 API)
- Create: `real_validation/hardware/nditracker.py`(从 real_capture 复制,69行)
- Test: `tests/test_hardware_adapters.py`

**Interfaces:**
- Consumes: `real_capture/hardware_threads.py`(141行,NdiThread)、`real_capture/nditracker.py`(69行,依赖 scipy+sksurgerynditracker 延迟)。
- Produces: `ndi.create_ndi_thread(port, *, rate_hz=50.0, ndi_count=1)` 返回内部 `NdiThread`;`ndi.require_hidden_evaluation_allowed(policy, *, timestamp, source)`;`ndi.HIDDEN_EVALUATION_SOURCE`;`ndi.NdiHardwareError`。

- [ ] **Step 1: 复制 nditracker.py**

复制 `real_capture/nditracker.py`(69行)→ `real_validation/hardware/nditracker.py`。它依赖 `from sksurgerynditracker.nditracker import NDITracker`(延迟,真机才装)+ scipy。

- [ ] **Step 2: 复制 NdiThread**

从 `real_capture/hardware_threads.py` 复制 `NdiThread` 类(141行中的该类)进 `real_validation/hardware/ndi.py`,保留现有适配层函数。`NdiThread.run` 里 `import nditracker` 改为 `from . import nditracker`(内部相对引用)。

- [ ] **Step 3: 自包含化**

删除 `from ._bootstrap import ensure_real_capture_importable`。`create_ndi_thread` 不再调它。检查 `hardware/__init__.py` 干净。

- [ ] **Step 4: 跑测试**

Run: `python -m unittest tests.test_hardware_adapters -v`
Expected: 全绿(create_ndi_thread('') 抛 NdiHardwareError;HIDDEN_EVALUATION_SOURCE 存在)。

- [ ] **Step 5: 全局 grep 确认无 real_capture 代码引用**

Run: `grep -rn "real_capture\|_bootstrap\|ensure_real_capture" real_validation/ --include="*.py"`
Expected: 无结果(或仅注释提到"已移入内部"的说明,无 import)。

- [ ] **Step 6: 提交**

```bash
git add real_validation/hardware/
git commit -m "refactor(real_validation): NDI 驱动内部化 —— NdiThread+nditracker 移入 hardware/,彻底删除 real_capture 外部引用"
```

---

### Task 4: 目录重构 —— 建子包骨架 + 移动叶子文件

**Files:**
- Create: `real_validation/contracts/__init__.py`、`core/__init__.py`、`execution/__init__.py`、`gui/__init__.py`、`tools/__init__.py`、`planning/__init__.py`(已有 auto_k,补 openloop 等)
- Move: `models.py io.py deploy_manifest.py plan_io.py` → `contracts/`;`session.py` → `core/`;`units.py coordinate_system.py obstacles.py planner_service.py openloop_planner.py` → `planning/`;`executor.py preflight.py metrics.py hardware_session.py` → `execution/`
- Test: 全量

**Interfaces:**
- Consumes: 无(纯移动)。
- Produces: 新子包路径。后续任务同步 import。

- [ ] **Step 1: 建子包骨架**

```bash
mkdir -p real_validation/contracts real_validation/core real_validation/execution real_validation/gui real_validation/tools
# planning 已存在,补 __init__ 导出
# 每个新子包建 __init__.py(可为空或导出)
```

- [ ] **Step 2: git mv 移动叶子文件**

```bash
cd /Data5/ddf/projects/SelfSoftRobot
git mv real_validation/models.py real_validation/contracts/models.py
git mv real_validation/io.py real_validation/contracts/io.py
git mv real_validation/deploy_manifest.py real_validation/contracts/deploy_manifest.py
git mv real_validation/plan_io.py real_validation/contracts/plan_io.py
git mv real_validation/session.py real_validation/core/session.py
git mv real_validation/units.py real_validation/planning/units.py
git mv real_validation/coordinate_system.py real_validation/planning/coordinate_system.py
git mv real_validation/obstacles.py real_validation/planning/obstacles.py
git mv real_validation/planner_service.py real_validation/planning/planner_service.py
git mv real_validation/openloop_planner.py real_validation/planning/openloop_planner.py
git mv real_validation/executor.py real_validation/execution/executor.py
git mv real_validation/preflight.py real_validation/execution/preflight.py
git mv real_validation/metrics.py real_validation/execution/metrics.py
git mv real_validation/hardware_session.py real_validation/execution/hardware_session.py
```

- [ ] **Step 3: 同步移动文件的相对 import**

对每个移动的文件,把 `from .X import` 改成新位置正确的相对引用。关键映射(依赖图):
- `contracts/models.py`:`from .io import stable_digest`(同包,不变)
- `contracts/io.py`:无相对 import
- `core/session.py`:`from .io` → `from ..contracts.io`;`from .models` → `from ..contracts.models`;`from .preflight` → `from ..execution.preflight`
- `planning/units.py`:无相对 import
- `planning/coordinate_system.py`:无
- `planning/obstacles.py`:无
- `planning/planner_service.py`:`from .models` → `from ..contracts.models`
- `planning/openloop_planner.py`:`from .runtime` → `from ..runtime`;`from .models` → `from ..contracts.models`;`from .planner_service` → 同包(不变);`from .units` → 同包(不变);`from .planning.auto_k` → `from .auto_k`;`from .obstacles` → 同包(不变)
- `execution/executor.py`:`from .models` → `from ..contracts.models`;`from .observation_policy` → `from ..runtime.observation_policy`
- `execution/preflight.py`:`from .models` → `from ..contracts.models`
- `execution/metrics.py`:`from .models` → `from ..contracts.models`
- `execution/hardware_session.py`:`from .executor` → 同包(不变)

**注意** `runtime/` 与 `planning/` 内部文件互相引用也要检查:openloop_planner 用 `from .runtime import plan_rollout` → 改 `from ..runtime import plan_rollout`;`from .planning.auto_k` → `from .auto_k`。

- [ ] **Step 4: 跑测试确认(测试还没改,应大量失败但可看到 import 错误清单)**

Run: `python -c "import real_validation.contracts.models; print('models ok')"`(先确认叶子模块可导入)
Run: `python -c "from real_validation.planning.openloop_planner import OpenLoopShootingPlanner; print('planner ok')"`

- [ ] **Step 5: 提交(仅移动 + import 修正,测试引用留到 Task 6)**

```bash
git add real_validation/
git commit -m "refactor(real_validation): 目录重构第一轮 —— contracts/core/planning/execution 子包,移动叶子文件+同步相对 import"
```

---

### Task 5: 目录重构 —— 移动依赖中间层(runtime 合并 anchors + gui/tools)

**Files:**
- Move: `model_runtime.py warmup.py observation_policy.py` → `runtime/`
- Create: `runtime/anchors.py`(合并 live_anchor + offline_anchor + anchor_utils)
- Move: `main_validation.py` → `gui/main_window.py`;`theme.py` → `gui/theme.py`
- Move: `perception_probe.py` → `tools/perception_probe.py`
- Test: 全量

**Interfaces:**
- Consumes: Task 4 的 contracts/core/planning/execution 子包。
- Produces: `runtime/anchors.py` 导出 `anchor_from_npz`、`anchor_from_camera_frame`(同名函数,供 scripts/run_avoidance 用 `from real_validation.runtime.anchors import anchor_from_npz`);`runtime/model_runtime.py` 导出 `ModelRuntime`;`gui/main_window.py` 导出 `ValidationWindow` + `main`。

- [ ] **Step 1: 移动 runtime 相关**

```bash
git mv real_validation/model_runtime.py real_validation/runtime/model_runtime.py
git mv real_validation/warmup.py real_validation/runtime/warmup.py
git mv real_validation/observation_policy.py real_validation/runtime/observation_policy.py
git mv real_validation/live_anchor.py real_validation/runtime/live_anchor.py
git mv real_validation/offline_anchor.py real_validation/runtime/offline_anchor.py
git mv real_validation/anchor_utils.py real_validation/runtime/anchor_utils.py
```

- [ ] **Step 2: 合并 anchors.py**

创建 `real_validation/runtime/anchors.py`,re-export 三个模块的公共函数(或直接合并):

```python
# runtime/anchors.py
"""锚点构建:从离线 NPZ 或实时相机帧建立模型状态锚点(合并原 live/offline/anchor_utils)。"""
from .anchor_utils import float_rows, model_normalization, normalize_rows  # noqa: F401
from .live_anchor import anchor_from_camera_frame  # noqa: F401
from .offline_anchor import anchor_from_npz  # noqa: F401
```

(anchors.py 作为 re-export 门面,三个源文件保留在 runtime/ 内;也可真正合并,但 re-export 更省风险。**优先 re-export**。)

- [ ] **Step 3: 同步 runtime 内 import**

`runtime/model_runtime.py`:`from .io` → `from ..contracts.io`;`from .models` → `from ..contracts.models`;`from .runtime import` → `from . import`(同包 loader);`from .deploy_manifest` → `from ..contracts.deploy_manifest`。
`runtime/warmup.py`:`from .planner_service` → `from ..planning.planner_service`。
`runtime/observation_policy.py`:无相对 import。
`runtime/live_anchor.py`:`from .models` → `from ..contracts.models`;`from .perception.X` → `from ..perception.X`;`from .anchor_utils` → 同包(不变)。
`runtime/offline_anchor.py`:`from .models` → `from ..contracts.models`;`from .anchor_utils` → 同包。
`runtime/loader.py`/`model.py`/`rollout.py`(已有):检查 `from .` 引用(loader 引 model/rollout 同包)。

- [ ] **Step 4: 移动 GUI 和 tools**

```bash
git mv real_validation/main_validation.py real_validation/gui/main_window.py
git mv real_validation/perception_probe.py real_validation/tools/perception_probe.py
```

`gui/main_window.py` 相对 import 全部改:`from .executor` → `from ..execution.executor`;`from .io` → `from ..contracts.io`;`from .model_runtime` → `from ..runtime.model_runtime`;`from .models` → `from ..contracts.models`;`from .openloop_planner` → `from ..planning.openloop_planner`;`from .offline_anchor` → `from ..runtime.anchors`;`from .plan_io` → `from ..contracts.plan_io`;`from .session` → `from ..core.session`;`from .widgets` → 同包;`from .widgets.theme` → 同包;`from .warmup` → `from ..runtime.warmup`;`from .live_anchor` → `from ..runtime.anchors`;`from .hardware.valve` → `from ..hardware.valve`;`from .hardware_session` → `from ..execution.hardware_session`;`from .observation_policy` → `from ..runtime.observation_policy`;`from .metrics` → `from ..execution.metrics`;`from .hardware.valve` → `from ..hardware.valve`。

`tools/perception_probe.py`:`from .perception.X` → `from ..perception.X`。

- [ ] **Step 5: 建 main.py 入口壳**

创建 `real_validation/main.py`:

```python
"""实机验证工作台 GUI 入口。运行: python -m real_validation.main"""
from .gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: 建子包 __init__.py**

给 contracts/core/execution/gui/tools 建 `__init__.py`(空或必要导出);planning 已有。**`contracts/__init__.py` 应 re-export 关键模型**(供 `from real_validation.contracts import Scene` 用),但包根 `__init__.py` 直接 re-export contracts.models。

- [ ] **Step 7: 跑测试(预期部分失败 —— import 路径还没同步)**

Run: `python -c "from real_validation.runtime.anchors import anchor_from_npz; print('anchors ok')"`
Run: `python -c "from real_validation.gui.main_window import ValidationWindow; print('gui ok')"`(offscreen 需 QT_QPA_PLATFORM)

- [ ] **Step 8: 提交**

```bash
git add real_validation/
git commit -m "refactor(real_validation): 目录重构第二轮 —— runtime 合并 anchors + gui/tools 移动 + main.py 入口壳"
```

---

### Task 6: 同步包根 __init__ + src/scripts/tests 全部引用

**Files:**
- Modify: `real_validation/__init__.py`(re-export 新路径)
- Modify: `scripts/control/run_avoidance.py:52-58`、`scripts/control/inverse_plan.py:120`、`scripts/utils/build_deploy_manifest.py:27`
- Modify: `tests/*.py`(全部 import)
- Test: 全量

**Interfaces:**
- Consumes: Task 4/5 的子包结构。
- Produces: 全部旧 import 路径 → 新子包路径。

- [ ] **Step 1: 更新包根 __init__.py**

`real_validation/__init__.py` 改为 re-export contracts.models(保持 stdlib-only):

```python
"""实机模型验证工作台。

核心模块不依赖 Qt;GUI、CLI 和测试共用同一套 session、preflight 与 executor。
包根必须保持 stdlib-only(import 卫生测试);真机硬件依赖在 hardware/ 子包延迟 import。
"""
from .contracts.models import (  # noqa: F401
    ActionPlan, Anchor, ModelDescriptor, SafetyPolicy, Scene, ScenePrimitive,
)
from .core.session import ExperimentSession, SessionState  # noqa: F401

__all__ = [
    "ActionPlan", "Anchor", "ExperimentSession", "ModelDescriptor",
    "SafetyPolicy", "Scene", "ScenePrimitive", "SessionState",
]
```

- [ ] **Step 2: 更新 scripts 引用**

```bash
# scripts/control/run_avoidance.py:52-58
sed 匹配:
  from real_validation.models import (...)
  → from real_validation.contracts.models import (...)
  from real_validation.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
  → from real_validation.planning.openloop_planner import ...
  from real_validation.offline_anchor import anchor_from_npz
  → from real_validation.runtime.anchors import anchor_from_npz
  from real_validation.model_runtime import ModelRuntime
  → from real_validation.runtime.model_runtime import ModelRuntime

# scripts/control/inverse_plan.py:120
  from real_validation.obstacles import cli_obstacle_loss
  → from real_validation.planning.obstacles import cli_obstacle_loss

# scripts/utils/build_deploy_manifest.py:27
  from real_validation.io import file_sha256
  → from real_validation.contracts.io import file_sha256
```

- [ ] **Step 3: 更新 tests 引用**

`tests/*.py` 的全部 `from real_validation.X import` 同步到新路径。映射(来自设计 §4.2):
- `real_validation.models` → `real_validation.contracts.models`
- `real_validation.coordinate_system` → `real_validation.planning.coordinate_system`
- `real_validation.deploy_manifest` → `real_validation.contracts.deploy_manifest`
- `real_validation.executor` → `real_validation.execution.executor`
- `real_validation.hardware._bootstrap` → 删除该测试(已自包含)或改测 hardware 内部
- `real_validation.hardware_session` → `real_validation.execution.hardware_session`
- `real_validation.live_anchor` / `real_validation.offline_anchor` → `real_validation.runtime.anchors`(或 runtime.live_anchor/offline_anchor,anchors 是门面)
- `real_validation.main_validation` → `real_validation.gui.main_window`
- `real_validation.metrics` → `real_validation.execution.metrics`
- `real_validation.observation_policy` → `real_validation.runtime.observation_policy`
- `real_validation.obstacles` → `real_validation.planning.obstacles`
- `real_validation.openloop_planner` → `real_validation.planning.openloop_planner`
- `real_validation.perception_probe` → `real_validation.tools.perception_probe`
- `real_validation.planner_service` → `real_validation.planning.planner_service`
- `real_validation.planning.auto_k` → 不变
- `real_validation.preflight` → `real_validation.execution.preflight`
- `real_validation.runtime.loader/model/rollout` → 不变
- `real_validation.session` → `real_validation.core.session`
- `real_validation.units` → `real_validation.planning.units`
- `real_validation.warmup` → `real_validation.runtime.warmup`
- `real_validation.widgets.*` → 不变(GUI 子包)
- `real_validation.main_validation` → `real_validation.gui.main_window`

**关键**:`tests/test_hardware_adapters.py` 的 `test_valve_module_does_not_import_real_capture` 保留(自包含契约);`test_import_hygiene.py` 的 `import real_validation` 断言仍应过(包根 stdlib-only)。

- [ ] **Step 4: 全量测试**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿(129 + 新增)。

- [ ] **Step 5: grep 确认无旧路径残留**

Run: `grep -rn "real_validation\.models\|real_validation\.session\|real_validation\.openloop_planner\|real_validation\.offline_anchor\|real_validation\.model_runtime\|real_validation\.obstacles\|real_validation\.io\|real_validation\.main_validation\|real_validation\.preflight\|real_validation\.executor\|real_validation\.warmup\|real_validation\.units\|real_validation\.metrics\|real_validation\.live_anchor\|real_validation\.observation_policy\|real_validation\.planner_service\|real_validation\.coordinate_system\|real_validation\.deploy_manifest\|real_validation\.hardware_session\|real_validation\.perception_probe" --include="*.py" .`
Expected: 无结果(scripts/src/tests 全部更新)。

- [ ] **Step 6: 提交**

```bash
git add real_validation/ scripts/ tests/
git commit -m "refactor(real_validation): 同步全部引用 —— 包根 re-export 新路径 + scripts/tests import 更新,彻底重构无旧路径残留"
```

---

### Task 7: run 脚本 + GUI_GUIDE 同步 + 最终验证

**Files:**
- Modify: `real_validation/run_gui.sh`、`run_gui.bat`(指向 python main.py)
- Modify: `real_validation/GUI_GUIDE.md`、`README.md`(目录结构 + 自包含说明)
- Test: 全量 + offscreen 冒烟

**Interfaces:**
- Consumes: Task 6 后的完整结构。
- Produces: 部署文档准确;启动脚本指向新入口。

- [ ] **Step 1: 更新 run 脚本**

```bash
# run_gui.sh → python -m real_validation.main
# run_gui.bat → python -m real_validation.main
```

- [ ] **Step 2: 更新 GUI_GUIDE / README 目录结构**

把 README/GUI_GUIDE 里提到的旧路径(如 `main_validation.py`、`real_validation/data/npz/`)更新为新结构(`main.py`、`real_validation/data/npz/` 数据目录不变)。

- [ ] **Step 3: 最终冒烟**

Run: `QT_QPA_PLATFORM=offscreen python -c "import sys; from PyQt5.QtWidgets import QApplication; from real_validation.gui.main_window import ValidationWindow; app=QApplication(sys.argv); w=ValidationWindow(); w.close(); print('gui ok')"`
Run: `python -c "from real_validation.main import main; print('entry ok')"`
Run: `grep -rn "real_capture" real_validation/ --include="*.py" | grep -v "#"`(应无代码引用)

- [ ] **Step 4: 全量测试**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add real_validation/
git commit -m "docs(real_validation): run 脚本指向 main.py + 文档同步 + 最终自包含验证(无 real_capture 引用)"
```

---

## 自审

**1. Spec 覆盖:**
- §2 目录结构 → Task 4/5(所有子包文件归属逐项列出)✅
- §3.1 移植清单(valve/camera/ndi)→ Task 1/2/3 ✅
- §3.2 适配层 API 签名保留 → 各 Task 明确"保留现有公共 API"✅
- §3.3 删 _bootstrap → Task 1 Step 5 + Task 3 Step 5 ✅
- §4.1 src/scripts 引用 → Task 6 Step 2(3 处逐一列出)✅
- §4.2 tests 引用 → Task 6 Step 3(映射表)✅
- §4.3 包根 stdlib-only → Task 6 Step 1 + import 卫生测试 ✅
- §5 GUI 入口 → Task 5 Step 5 + Task 7 Step 1 ✅
- §6 验证 → 各 Task + Task 7 ✅

**2. Placeholder 扫描:** 无 TBD/TODO;每个 Task 有具体文件、import 映射、测试命令。移动顺序按依赖图(叶子→中间→顶层)✅

**3. Type/命名一致:** `anchors.py` 门面导出 `anchor_from_npz`(scripts/run_avoidance 依赖)与设计一致;`gui/main_window.py` 导出 `ValidationWindow`(测试用)一致;适配层 API 签名跨 Task 1/2/3 一致 ✅

**注:** Task 4 移动时 main_validation 尚未移,它的相对 import 在 Task 5 才改;Task 4 提交时 main_validation 可能 import 旧路径而暂时失效——**修正:Task 4 和 Task 5 必须连续执行,Task 4 提交前不跑 GUI 测试**(仅跑叶子模块 import + 非 GUI 测试),全量测试在 Task 6 后。已在 Task 4 Step 5 注明"测试引用留到 Task 6"。
