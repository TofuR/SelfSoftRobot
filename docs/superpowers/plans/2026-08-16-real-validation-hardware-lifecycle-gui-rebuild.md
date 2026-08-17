# real_validation 硬件生命周期与实验工作流 GUI 重构实施方案

> 日期：2026-08-16
>
> 分支：`feat/real-data-transition`
>
> 状态：实施依据
>
> 目标：保留 `real_validation/` 可整目录复制到实验 PC 的部署方式，以
> `real_capture/` 已验证的硬件语义为参考，通过“复制复用”统一相机、阀、NDI、
> Mock 和安全关闭逻辑；随后重建符合稀疏观察 OpenLoop 实验流程的 GUI。

## 1. 背景与问题

`real_capture` 面向同步数据采集，其界面虽然朴素，但硬件对象、操作顺序和状态反馈
比较直接：相机、阀和 NDI 各有明确配置、连接/断开动作与状态；Mock 可以按组件组合；
退出时统一停止生产者、阀归零、关闭串口并等待写盘。

当前 `real_validation` 已有较好的后端基础：不可变契约、实验状态机、OpenLoop
shooting、Preflight、ACK 感知执行和结果计算。但 GUI 与硬件编排存在以下结构性问题：

1. Mock/Real 不是显式实验配置。相机默认真机、NDI 自动启动 Mock、阀失败后执行静默
   回退 Mock；按钮文案与实际 transport 可能相反。
2. 相机、NDI、阀分别由 GUI 手工创建和销毁，没有统一生命周期；关闭窗口不能可靠
   停止真实 NDI、归零并关闭阀。
3. 从 `real_capture` 复制的驱动没有保留同一套调用语义：相机设备枚举、唯一 serial、
   Apply/Reconnect、MockValveController 等未在验证 GUI 中统一使用。
4. Setup 页把实验、硬件、模型、K_safe、安全矩阵塞入狭窄滚动栏，字段前后依赖不清，
   控件尺寸随窗口压缩，操作员无法快速判断下一步。
5. 主视图、气压图、NDI 图和日志同时常驻，摄像头这一主要工作区反而不突出。
6. 实验安全语义没有完整反映到界面：所需阀组未按 `channel_map` 推导，归零 Pause
   之后仍显示 Resume，隐藏评价流可在执行期直接展示。

本次不重写已经通过测试的模型、规划和契约层；重点替换硬件编排和 GUI 协调层。

## 2. 必须保持的不变量

### 2.1 可移植性

- `real_validation/` 必须可以作为完整目录复制到 Windows/Linux 实验 PC。
- 运行时不得依赖仓库中的 `real_capture/`、`src/`、`scripts/` 或训练目录。
- 允许且要求把 `real_capture` 的稳定驱动代码复制到 `real_validation/hardware/`。
- 复制来源和版本要可追溯；验证专用逻辑放在相邻模块，不混入复制驱动主体。

### 2.2 科学实验边界

- 部署主线是一个真实 Anchor 后的窗口化 OpenLoop：模型连续自馈 K 步预测，不得把
  执行期隐藏相机/NDI 真值送回 planner。
- NDI 是隐藏评价流，不是模型输入。隐藏阶段默认不向操作员显示 NDI 数值，避免人工
  闭环泄漏；只在执行结束后显示和评价。
- GT/实时观察只用于 Anchor、指定观察检查点或诊断，不替代 OpenLoop 主线。
- 真实执行必须使用 `deploy_manifest.json` 中的动作单位、通道映射、训练时基、
  K_safe、相机指纹和感知参数；缺失关键契约时 fail-closed。
- 当前阀无压力反馈，GUI 和结果中只能称为 requested/applied command，不得写成实测压力。

### 2.3 安全不变量

- 真实硬件连接失败不得静默回退 Mock。
- 计划必须通过 Preflight，并由操作员显式 Arm 后才能执行。
- Abort/Zero 永远可用；真实执行中关闭窗口必须先 Abort、归零、等待 ACK，再释放串口。
- `required_groups` 由实际 `channel_map` 推导；所有必需阀组未就绪时禁止 Arm。
- 默认 Pause 策略为归零。归零后旧计划失效，进入 Re-anchor/Replan，不允许 Resume。

## 3. 目标实验操作流程

### Step 1：建立实验

1. 选择或创建 `run_<timestamp>`。
2. 选择运行配置：`全 Mock`、`真机验证`、`自定义混合`。
3. 运行配置明确列出相机、阀和 NDI 的 backend，执行期间不可修改。

### Step 2：加载部署包

选择一个目录，而不是让操作员独立猜测多个文件：

```text
checkpoints/<name>/
├── best_model.pt
├── config.json
└── deploy_manifest.json
```

加载成功后只读显示模型类型、节点数、H、K_train、K_safe、Δt、动作单位上界、
channel_map、相机指纹和 checkpoint hash。K_safe、channel_map、Δt 默认不手填。

### Step 3：连接硬件并做健康检查

1. Camera：选择 Disabled/Mock/Real；Real 模式枚举设备并显式选择唯一 serial。
2. Valve：选择 Mock/Real；Real 模式分别连接所需组，连接后先确认零压命令 ACK。
3. NDI：选择 Disabled/Mock/Real；Real 模式选择端口与探头数，收到有效帧后才算 READY。
4. 顶部状态灯独立显示 Camera、Valve、NDI，不使用一个会被覆盖的共享状态文本。

### Step 4：建立 Anchor 和任务

- 离线：从 transition NPZ 选取一帧以及完整 H 步动作历史。
- 在线：检查相机 serial/尺寸/fps、参考背景、registration、frame age、分割/骨架质量，
  再冻结 Anchor。
- 真机 Warmup 必须实际下发训练分布内动作，并用 ACK 后 `applied6` 填历史；全零历史只
  允许 Mock/调试，真机默认禁用。
- 在主画面创建目标点/区域、目标骨架与障碍。

### Step 5：规划和审核

- Planner 自动使用 manifest 中的 channel_map、Δt 和 K_safe。
- 预览预测全身轨迹、六通道命令、目标误差、最小障碍间距和约束违反点。
- Preflight PASS 后计划进入 READY；Anchor、Scene、Safety、Model 任一变化使计划失效。

### Step 6：执行与隐藏评价

- 按 backend 显示明确按钮：`运行 Mock 计划` 或 `执行真机计划`。
- 真机确认框显示 required groups、K、总时长、最大命令、最大变化率和记录源状态。
- 执行时独立记录命令/ACK、相机、NDI和预测状态。隐藏观测不进入模型，也不在执行期
  暴露给操作员。
- Abort 或归零 Pause 后进入 Re-anchor，不恢复旧计划。

### Step 7：结果

- 计划侧：目标距离、任务成功、最小障碍间距、预测碰撞。
- 执行侧：ACK、jitter、requested/applied、命令安全。
- 真值侧：prediction-to-execution error-by-k、末端误差、全身 MNE/p90/max、NDI 指标。
- 保存完整 run，可只读 Replay 和离线复算。

## 4. 自包含复制复用结构

复制复用不等于在两个 GUI 中重新发明生命周期。目标结构：

```text
real_validation/
├── hardware/
│   ├── camera.py          # 复制 RealSenseCam/Mock 与 list_devices
│   ├── ndi.py             # 复制 NdiThread/MockNdiThread
│   ├── nditracker.py      # 复制 NDI 底层
│   ├── valve.py           # 复制 ValveController/MockValveController
│   ├── modbus.py          # 复制 ModbusManager
│   ├── profile.py         # 新增：backend/profile 不可变配置
│   ├── session.py         # 新增：统一对象生命周期、状态和安全关闭
│   └── compatibility.py   # 新增：manifest/设备指纹检查
├── execution/
│   ├── hardware_session.py
│   ├── validation_recorder.py
│   └── executor.py
└── gui/
    ├── main_window.py     # 只协调页面、session 和 HardwareSession
    ├── pages/
    └── widgets/
```

复制驱动的验证专用扩展，如相机 fingerprint 和隐藏评价标签，应从驱动主体移到
`compatibility.py` 或 adapter，降低以后从 `real_capture` 同步代码时的冲突。

## 5. HardwareProfile

新增不可变配置：

```python
HardwareProfile(
    name="all_mock | real | custom",
    camera_backend="disabled | mock | real",
    camera_count=1,
    camera_serials=(),
    valve_backend="disabled | mock | real",
    group1_port="COM3",
    group2_port="COM46",
    baudrate=9600,
    slave_addr=1,
    ndi_backend="disabled | mock | real",
    ndi_port="COM9",
    ndi_count=1,
)
```

规则：

- backend 值显式；空端口不能暗示 Mock。
- Real 相机必须使用唯一 serial；设备不足进入 ERROR。
- profile 只在非 ARMED/EXECUTING 状态可应用。
- 配置保存到 `config/hardware.json`，但每个 run 还要保存一份快照。

## 6. HardwareSession

`HardwareSession(QObject)` 是 GUI 唯一硬件入口：

```text
signals:
  camera_frames(list)
  camera_frame(index, image, timestamp)
  ndi_data(values, timestamp)
  valve_command(applied6, timestamp)
  device_state_changed(device, state, message)

methods:
  apply_profile(profile)
  connect_camera()/disconnect_camera()
  connect_valve_groups(groups)/disconnect_valves()
  connect_ndi()/disconnect_ndi()
  command_transport(required_groups)
  safe_zero()
  shutdown()
```

设备状态统一为：

```text
DISABLED → OFF → CONNECTING → READY
                         └→ ERROR
Mock 连接成功同样是 READY，但 backend 标签始终显示 MOCK。
```

不允许 GUI 通过 `controller is None` 猜测 Mock/Real。transport 必须由 profile 明确决定；
Real backend 未 READY 时抛错并阻止执行。

## 7. GUI 目标布局

采用 `real_capture` 易理解的“左控制、右主画面”，但按验证工作流重新组织：

```text
┌──────────────────────────────────────────────────────────────┐
│ Run | Model | Camera ● | Valve ● | NDI ● | Zero | Abort     │
├────────────────────────────┬─────────────────────────────────┤
│ ① 实验与部署               │                                 │
│ ② 硬件连接                 │      主相机 / 场景 / 轨迹       │
│ ③ Anchor 与任务            │                                 │
│ ④ 规划与审核               │                                 │
│ ⑤ 执行与结果               ├─────────────────────────────────┤
│                            │ [动作] [NDI] [日志] [结果]       │
└────────────────────────────┴─────────────────────────────────┘
```

### 7.1 Setup 页面

按实际前后依赖排列：

1. 实验目录与 New/Open。
2. 运行配置（全 Mock/真机/自定义）。
3. Camera/Valve/NDI 三张独立卡，各自有 backend、配置、连接按钮、状态。
4. 部署包选择和 Load。
5. 模型摘要与“安全配置…”按钮。

六通道安全矩阵不再常驻挤压 Setup；放入独立对话框，一次显示 6×5，Apply 后关闭。

### 7.2 主工作区

- 摄像头/场景视图是唯一大画面。
- 图层开关保留，但收敛为一行工具栏。
- 气压、NDI、日志和结果放入下方 Tab，按需查看；无数据时显示空状态说明。
- 隐藏评价期间自动隐藏 NDI Tab 和实际骨架层。

### 7.3 文案

- 全部按钮使用中文，英文技术词保留在说明中。
- 执行按钮严格区分 Mock/Real。
- 状态显示具体设备，不再只显示 `Hardware: MOCK`。
- 不使用“连接失败后回退 Mock”文案和行为。

## 8. 文件级实施步骤

### Phase A：硬件配置和生命周期

1. 新增 `hardware/profile.py`，实现 Backend、DeviceState、HardwareProfile、preset 和 JSON。
2. 新增 `hardware/session.py`，集中创建/停止复制来的驱动。
3. Camera 真/假都使用同一个 `RealSenseCam(mock=...)`；复制 `real_capture` 的设备枚举和
   unique serial 逻辑。
4. Valve 真/假都使用 Controller 接口，再进入同一 transport；删除执行时临时构造
   MockCommandTransport 的隐式回退。
5. NDI 使用 Disabled/Mock/Real 显式选择和连接/断开 toggle。
6. 复制 `real_capture` shutdown 顺序并增加真实执行窗口关闭测试。

### Phase B：GUI 重建

1. 保留现有业务槽函数，先重建 `_build_ui()` 与 Setup 页面，避免同时改 planner。
2. 顶部改为 run/model/device 独立 badge。
3. 左侧步骤页固定合理最小宽度；路径字段使用 stretch，COM/数字字段使用短固定宽度。
4. 右侧只保留主画面，下方建立 viz tabs。
5. 新增 `SafetyPolicyDialog`，移出 Setup 常驻安全网格。
6. 将硬件控件连接到 HardwareSession，不直接创建线程。

### Phase C：执行安全接线

1. required groups 从 plan/channel_map 推导。
2. `_make_transport()` 改为 HardwareSession 显式返回；Real 未就绪直接报错。
3. 动态更新执行按钮文案和确认信息。
4. 修复 Pause=zero 后仍可 Resume 的矛盾。
5. Zero/Abort 不依赖当前页面和 executor 是否存在。

### Phase D：在线实验闭环

1. 新增 `ValidationRecorder`，复制 SaveThread、最新值缓存、frame-age、全视角原子采样。
2. 由 executor step/ACK 事件驱动采样，而不是复制随机采集时钟。
3. 接入真实 Warmup、registration、background、camera fingerprint 和质量门。
4. 写 hidden ground truth、observations audit 和 prediction-to-execution metrics。

## 9. 测试与验收

### 9.1 自动测试

- 全 Mock profile 能完整连接、规划、执行、归零和关闭。
- Real 连接失败进入 ERROR，执行按钮禁用，不产生 Mock 命令。
- Mock/Real 按钮和日志与实际 transport 一致。
- N 台 RealSense 使用 N 个唯一 serial；设备不足明确失败。
- 模式切换先停止旧线程，执行中禁止切换。
- required groups 未全部 READY 时不能 Arm。
- 关闭窗口会停止相机/NDI、归零阀并释放串口。
- Pause zero 后旧计划失效，Resume 不可用。
- fingerprint/manifest 不匹配阻止真实 Anchor/执行。
- 复制驱动的接口/Mock 行为与 `real_capture` parity 测试一致。

### 9.2 人工 GUI 验收

- 1400×860 下 Setup 不横向截断，常用内容无需水平滚动。
- 路径输入长、COM/数量输入短，标签和控件对齐。
- 操作员能在 10 秒内判断每个设备是真/假、是否连接、是否健康。
- 相机主画面占主要区域；曲线和日志不会同时挤压主画面。
- 从 New Experiment 到 Execute 的下一步始终明确。
- 真机执行前界面明确显示“真机”、阀组、最大压力和总时长。

## 10. 本轮实施范围

本轮直接完成 Phase A、Phase B 和 Phase C，并为 Phase D 建立可接入的 recorder/lifecycle
接口。Phase D 中需要真实设备和部署 manifest 验证的在线感知、同步评价将在结构就绪后
继续实现，不能用 Mock 结果冒充真机闭环完成。

## 11. 2026-08-16 实施记录

### 11.1 已完成

- `hardware/profile.py`：完成显式 backend、preset、唯一 serial、数量范围、JSON
  round-trip 与 `channel_map -> required_groups`。
- `hardware/session.py`：完成 Camera/Valve/NDI 创建、状态、连接、断开和
  shutdown 的唯一入口；Real 失败保持 ERROR，不更换 controller 类型。
- `gui/main_window.py`：Setup 按真实前后依赖重排；安全矩阵移入对话框；相机为
  右侧唯一大画面；动作/NDI/日志改为 Tab；三个设备拥有独立 badge。
- GUI 不再保存 `_real_cams`/`_camera_thread`/`_mock_ndi_thread`；所有设备操作
  改为调用 `HardwareSession`。
- `_make_transport()` 只能从已 READY 的 profile/controller 创建；Arm 前按 plan
  `channel_map` 检查所需阀组；执行文案严格区分 Mock/Real。
- zero-pause 改为“归零并重新锚定”，状态直接进入 `REANCHOR`，Resume 不再
  出现；Zero 通过非 GUI 线程等待 ACK。
- `config/hardware.json` 使用 `{"profile": ...}` 新格式，启动时仍可迁移旧
  `camera_input/group1/group2` 格式；每个 run 保存独立 profile 快照。
- 已增加模式、无 fallback、shutdown、GUI 独立 backend/badge、应用配置门禁与
  Resume 不可见的回归测试。

### 11.2 本轮有意不宣称完成

- Phase D `ValidationRecorder`、多视图全视角原子新鲜度、真实 Warmup、隐藏评价流
  执行期封锁和 prediction-to-execution 实测指标尚未完成。
- 真实 RealSense/Modbus/NDI 的端到端验收必须在实验 PC 与真设备上进行；离线
  Mock 通过不等于真机实验已完成。
