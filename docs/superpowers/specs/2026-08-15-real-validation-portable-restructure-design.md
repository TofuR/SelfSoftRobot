# 设计:real_validation 完全独立移植 + 目录层级重构

> 日期:2026-08-15 · 分支 `feat/real-data-transition`
> 目标:①去掉对 `real_capture/` 的全部外部引用(sys.path 桥接),三类真机驱动移植进内部,real_validation **完全自包含可独立移植**;②23 个平铺文件 → 功能域子包,主目录只剩入口。
> 用户决策(2026-08-15):硬件移植 + 目录重构一起做;功能域子包风格;全部三驱动移植;彻底重构 + 同步仓库内引用,不留顶层薄壳。设计获批准后直接执行,无需逐节确认。

---

## 1. 目标与非目标

### 1.1 要实现

1. **完全自包含**:删除 `real_validation/hardware/_bootstrap.py`(sys.path 桥接),不再引用 `real_capture/` 任何模块。串口阀(ValveController + ModbusManager)、相机(RealSenseCam)、NDI(NdiThread + nditracker)驱动移植进 `real_validation/hardware/`。
2. **目录层级**:23 个平铺文件按功能域收进子包;主目录只剩入口 + 包根 `__init__.py`。
3. **彻底重构**:同步更新 `src/`、`scripts/`、`tests/` 的全部 import 路径,不留顶层薄壳转发文件。
4. 真机执行链路不变:GUI 的 `_make_transport`/硬件连接面板继续工作(适配层公共 API 签名保留)。

### 1.2 非目标

- 不改动 `perception/`(已是子包,结构合理)。
- 不改动已有子包 `runtime/`(loader/model/rollout)与 `planning/`(auto_k)内部结构,只移动平铺文件进新子包。
- 不改真机驱动的行为语义(移植即搬移,不重写逻辑;只改 import 路径为内部相对引用)。
- 不做与移植/重构无关的代码优化。

---

## 2. 目标目录结构

```
real_validation/
├── main.py                      # ★入口(原 main_validation.py 移动 + 减到最小启动壳)
├── __init__.py                  # 包根 re-export contracts 顶层符号(保持 stdlib-only)
│
├── contracts/                   # 数据契约
│   ├── __init__.py
│   ├── models.py                # ← 平铺 models.py(核心,src/scripts 依赖)
│   ├── io.py                    # ← io.py
│   ├── deploy_manifest.py       # ← deploy_manifest.py
│   └── plan_io.py               # ← plan_io.py
│
├── core/                        # 会话/状态
│   ├── __init__.py
│   └── session.py               # ← session.py
│
├── perception/                  # 已有,不变(骨架/分割/背景/配准/质量)
│
├── planning/                    # 规划(已有 auto_k,并入平铺规划文件)
│   ├── __init__.py
│   ├── auto_k.py                # 已有
│   ├── openloop_planner.py      # ← openloop_planner.py
│   ├── planner_service.py       # ← planner_service.py
│   ├── obstacles.py             # ← obstacles.py(scripts 依赖 cli_obstacle_loss)
│   ├── coordinate_system.py     # ← coordinate_system.py
│   └── units.py                 # ← units.py
│
├── runtime/                     # 模型运行时 + 锚点(已有 loader/model/rollout)
│   ├── __init__.py
│   ├── loader.py  model.py  rollout.py     # 已有
│   ├── model_runtime.py         # ← model_runtime.py(scripts 依赖)
│   ├── anchors.py               # ★合并 live_anchor + offline_anchor + anchor_utils
│   ├── warmup.py                # ← warmup.py
│   └── observation_policy.py    # ← observation_policy.py
│
├── execution/                   # 执行/安全/评价
│   ├── __init__.py
│   ├── executor.py              # ← executor.py
│   ├── preflight.py             # ← preflight.py
│   ├── metrics.py               # ← metrics.py
│   └── hardware_session.py      # ← hardware_session.py(QtValveTransport)
│
├── hardware/                    # ★真机驱动,完全自包含(不再引用 real_capture)
│   ├── __init__.py
│   ├── valve.py                 # ★移植 ValveController + ModbusManager
│   ├── camera.py                # ★移植 RealSenseCam + 保留 assert_camera_fingerprint
│   ├── ndi.py                   # ★移植 NdiThread + nditracker
│   └── _bootstrap.py            # ★删除(不再需要 sys.path 桥接)
│
├── gui/                         # GUI
│   ├── __init__.py
│   ├── main_window.py           # ★原 main_validation.py(GUI 主窗口)
│   ├── theme.py                 # ★平铺 theme.py 移入(或留 widgets/)
│   └── widgets/                 # 已有(camera_view/plan_preview/scene_editor/primitive_items)
│
├── tools/
│   ├── __init__.py
│   └── perception_probe.py      # ← perception_probe.py
│
├── config/  checkpoints/  data/  runs/  calibration/   # 数据/配置目录(不变)
```

**移动策略**:优先**移动文件**(git mv 保历史),必要时合并(anchors.py)。

---

## 3. 硬件移植(核心)

### 3.1 移植清单

| 移植目标 | 来源 | 依赖 | 说明 |
|---|---|---|---|
| `hardware/valve.py` | `real_capture/valve_control.py`(468行) + `modbus_manager.py`(514行) | pyserial | ValveController 类 + ModbusManager 类;**内部 import**(`from .modbus import ModbusManager` 或合并) |
| `hardware/camera.py` | `real_capture/realsense_cam.py`(144行) | pyrealsense2(延迟 import) | RealSenseCam 类;保留已实现的 `assert_camera_fingerprint` |
| `hardware/ndi.py` | `real_capture/hardware_threads.py`(141行 NdiThread) + `nditracker.py`(69行) | scipy + sksurgerynditracker(延迟 import) | NdiThread 类 + nditracker 辅助;保留 `HIDDEN_EVALUATION_SOURCE` |

### 3.2 保留的适配层 API(签名不变,GUI 接线不破)

- `valve.create_valve_controller(group1, group2, *, baudrate, slave_addr) -> ValveController`
- `valve.connect_valve_groups(controller, groups) -> dict[int, tuple[bool, str]]`
- `valve.valve_to_kpa_requested(action6, action_scale_kpa) -> tuple[float, ...]`
- `camera.create_realsense_cam(...)`, `camera.assert_camera_fingerprint(...)`
- `ndi.create_ndi_thread(...)`, `ndi.require_hidden_evaluation_allowed(...)`, `ndi.HIDDEN_EVALUATION_SOURCE`
- 异常类:`ValveHardwareError` / `CameraHardwareError` / `NdiHardwareError`

### 3.3 删除

- `hardware/_bootstrap.py` + `ensure_real_capture_importable()`
- `perception_probe.py` 里的 `sys.path.append(real_capture)` 分支(改为内部驱动或标记 Mock)

---

## 4. 引用同步(彻底重构,不留薄壳)

### 4.1 src/scripts 反向依赖(6 模块 → 新路径)

| 现路径 | 新路径 | 调用方 |
|---|---|---|
| `real_validation.models` | `real_validation.contracts.models` | run_avoidance.py:52 |
| `real_validation.openloop_planner` | `real_validation.planning.openloop_planner` | run_avoidance.py:54 |
| `real_validation.offline_anchor` | `real_validation.runtime.anchors` | run_avoidance.py:55 |
| `real_validation.model_runtime` | `real_validation.runtime.model_runtime` | run_avoidance.py:58 |
| `real_validation.obstacles` | `real_validation.planning.obstacles` | inverse_plan.py:120 |
| `real_validation.io` | `real_validation.contracts.io` | build_deploy_manifest.py:27 |

### 4.2 测试 imports

`tests/*.py` 全部 `from real_validation.X import ...` 同步到新路径(contracts/core/planning/runtime/execution/hardware/gui/tools)。

### 4.3 包根 __init__ 保持 stdlib-only

`real_validation/__init__.py` 只 re-export `contracts.models` 的顶层符号(models/Anchor/Scene/...),维持 import 卫生测试(`test_import_hygiene`)。

---

## 5. GUI 入口

- `main_validation.py` → `gui/main_window.py`(类 `ValidationWindow` 移动)
- 新增 `main.py`:`from .gui.main_window import main; main()`(减到最小启动壳)
- `run_gui.sh` / `run_gui.bat` 指向 `python main.py`
- 移动后 `gui/main_window.py` 内部 `from .widgets import ...`、`from ..contracts.models import ...` 等相对引用

---

## 6. 验证

1. 全量测试:`python -m unittest discover -s tests -v` 全绿(129 + 新增驱动测试)
2. import 卫生:`tests/test_import_hygiene.py` 全绿(包根 stdlib-only 不破)
3. 移植后硬件:**无真机也能测纯逻辑**(create_valve_controller 缺 COM 报错、kPa 换算、指纹断言);真机串口只能在 PC 实测
4. `python main.py`(或 offscreen 构造)冒烟
5. `grep -rn "real_capture" real_validation/` → 仅剩文档/注释,无代码引用

---

## 7. 风险与约束

| 项 | 说明 |
|---|---|
| 大 diff | 目录重构涉及大量 import 路径改动 → 分任务逐步移动 + 每步跑测试,不一次搬完 |
| import 卫生 | 包根 `__init__.py` 改动必须保持 stdlib-only;硬件的真机依赖(pyserial 等)延迟 import,不进包根闭包 |
| 硬件行为 | 移植即搬移,不重写逻辑;只改 import 为内部相对引用,驱动行为语义不变 |
| 向后兼容 | 适配层公共 API 签名不变(create_valve_controller/assert_camera_fingerprint/create_ndi_thread);GUI `_make_transport`/硬件面板不破 |
| 真机验证 | 移植后的真机串口只能在 PC 实测(本环境无硬件);纯逻辑测试在本环境 |
| 仓库引用 | src/scripts 同步更新;`scripts/control/run_avoidance.py`、`inverse_plan.py`、`scripts/utils/build_deploy_manifest.py` 3 处 |

---

## 8. 实施计划拆分(后续 writing-plans)

建议按依赖顺序拆分:

1. **硬件移植**(先做,独立可验收):移植 valve/camera/ndi 驱动进 `hardware/`,删 `_bootstrap`,保留适配层 API
2. **目录重构**(后做):文件 → 子包 + 同步 src/scripts/tests 引用
3. **入口收敛**:main.py + GUI 移动 + run 脚本
4. **验证**:全量测试 + import 卫生 + 无 real_capture 引用 grep
