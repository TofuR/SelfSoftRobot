# 实机验证工作台 GUI 使用指南

> 对应设计文档:`docs/experiments/real_robot_validation_workbench_todo.md`、`docs/superpowers/specs/2026-07-28-real-validation-task-layer-ik-design.md`。

## 0. 启动与运行配置

**启动**:在 `real_validation/` 目录内 `python main.py`(或 Windows 双击 run_gui.bat)。

启动后先在 Setup 选择并点击**应用配置**:

- `全 Mock`:相机、阀、NDI 都是软件替身，但仍需分别点击连接，用于跑通完整生命周期。
- `真机验证`:三类设备都是 Real；真实连接失败会显式进入 ERROR，**不会回退 Mock**。
- `自定义混合`:相机、阀、NDI 可各自选 Disabled/Mock/Real，例如 Real 相机 + Mock 阀 + Disabled NDI。

RealSense 支持 1–8 台。真机模式可留空 serial 让系统枚举，也可用逗号显式填入与数量一致的唯一 serial。真机依赖安装见 `requirements-hardware.txt`。

---

## 1. 总体:五页 = 一条验证流水线

GUI 用五个 Tab 页串起"**加载模型 → 建立状态锚点 → 定义目标/障碍 → 规划 → 安全执行 → 查看结果**"的完整闭环。**使用顺序就是页面顺序,通常从左到右、逐页推进**:

```
1 Setup  ──▶  2 Observe & Scene  ──▶  3 Plan  ──▶  4 Execute  ──▶  5 Results
建立实验       建立锚点+场景            规划+预检       安全执行         看指标
加载模型       定义目标/障碍            预览预测轨迹     Arm→执行         导出
安全配置
```

顶部始终显示 Run/State、模型状态，以及**相机、阀、NDI 三个独立状态灯**。状态灯同时显示 backend 和 `OFF/CONNECTING/READY/ERROR/DISABLED`，不再用一个会相互覆盖的 Hardware 文字。session 颜色如下:
- 灰 未建实验(`no_session`)· 蓝 `idle`(已建实验待机)· 绿 `ready`/`completed`/`zeroed` · 橙 `armed`/`paused`/`planning`/`reanchor` · 红 `executing`/`aborting`/`error`

**关键约束**:各页之间有前置依赖(见下),第 2~4 页在不满足前置时会弹窗拒绝并提示。`执行中(armed/executing/paused)会自动锁定第 1/2/3 页`,防止执行中改场景破坏记录溯源。

---

## 2. 各页详解

### 2.0 整体布局 —— 控制台在左,可视化在右(参考 real_capture)

主窗口(顶部全局安全栏之下)为**左右两栏**:

- **左栏 · 控制台(5 页 Tab)**:Setup → Observe & Scene → Plan → Execute → Results(使用顺序即页面顺序,见 §1)。
- **右栏 · 可视化面板(常驻,跨 Tab 可见)**:
  - **右上 · 摄像头画面**(正常横屏比例)+ 多层叠加(骨架 / 场景 / 预测轨迹 / 实际骨架 / NDI 位姿);图层开关复选框在其下方,可逐层独立开合。
  - **气压命令曲线**(6 通道 kPa,执行/下发时实时滚动)。
  - 下方 Tab 按需切换**六通道动作曲线 / NDI 末端坐标 / 运行日志**，不同时挤压主画面。NDI 只进评价，不进模型。
  - **状态栏**:相机来源 / 骨架节点 / NDI 状态。

**Observe 锚定交互在右上面板主摄像头**:Observe 页(第 2 页)是**纯控制面板**(离线锚定 / 目标与障碍 / 相机锚定工具 / 场景编辑列表),不再页内复制摄像头。`从相机取流锚定`、`点加目标`/`点出目标骨架`/`点加障碍` 等工具选中后,直接在**右上面板主画面**上点击操作(该页激活时主摄像头进入可交互模式;切到其它页自动恢复只读显示)。

**场景编辑多选删除**:scene_editor 原语列表支持 **Ctrl 多选 / Shift 范围 / Ctrl+A 全选**,点 `删除` 按钮或按 `Del`(Backspace)键批量删除选中原语。

### 2.1 Page 1 · Setup — 建立实验、加载模型、配置安全

| 卡片 | 控件 | 作用 |
|---|---|---|
| **实验与运行** | Run 根目录 + `新建实验` | 创建唯一 `run_<时间戳>`，并写入当前 `hardware_profile.json` 快照 |
| | `打开 Run（只读回放）` | 回看历史实验，不能 Arm |
| **运行配置** | 全 Mock / 真机验证 / 自定义混合 + `应用配置` | 确定三类设备的 backend。有任何设备对象存在时配置锁定，须全部断开后才能重新应用 |
| **相机** | Backend / 数量 / serials / 主显示 + `连接相机` | 1–8 路 Mock 或 RealSense；真机 serial 必须唯一，设备不足直接 ERROR |
| **六通道比例阀** | Backend / 组1/2 端口 / baud / slave | Mock 与 Real 共用 Controller/ACK 路径；两组可单独连接。Arm 会按 plan `channel_map + channel_equalities` 推导必需阀组并 fail-closed |
| **NDI** | Backend / 串口 / 探头数 + `连接 NDI` | 显式 Disabled/Mock/Real；启动时不再自动跑 Mock NDI；末端 mm 只进评价 |
| **模型与部署契约** | Checkpoint / 训练数据目录 / K_safe / 设备 | 指定模型权重路径;K_safe(规划视野上限)在加载后按认证表自动填充,并在模型摘要标注来源(`认证表(10px 容差)` 或 `手动`),悬停 K_safe 可见说明 |
| | `加载部署模型` | 后台加载 checkpoint,显示 type/action_dim/nodes/H/K_train/K_safe/train_dt/action_scale_kpa/hash |
| **安全配置…** | 对话框中 min/max/rise/fall/initial × 6 | 安全矩阵不再挤占 Setup；Apply 后写入 session 并使旧计划失效 |

**这一步要产出**:一个 run + 已应用的显式 profile + 已加载模型 + 已连接的必需设备。配置保存到 `config/hardware.json`，每个 run 另存 `hardware_profile.json`。

> **真机接线需要什么**:串口阀两组 Modbus(组1=ch0-2,组2=ch3-5,4-20mA)+ 依赖 `requirements-hardware.txt`(pyserial/pyrealsense2/scikit-surgerynditracker)。相机/NDI 适配层在 `real_validation/hardware/`,Mock 流程不触碰。

### 2.2 Page 2 · Observe & Scene — 建立锚点、定义目标与障碍

> 前置:已 `Load Model`。锚点(s_{t-1} + 动作历史)是模型规划的起点,必须有。
> 布局:Observe 页是**紧凑满宽控制面板**(离线锚定 / 目标与障碍 / 相机锚定工具 / 场景编辑列表);摄像头画面在右上面板主显示区(见 §2.0),锚定/点选工具直接在主画面上操作。

| 卡片 | 控件 | 作用 |
|---|---|---|
| **离线锚定** | `加载 anchor.json` / `加载 scene.json` | 载入历史 run 的锚点/场景 |
| | Transition NPZ + 帧索引 + `从 NPZ 建立 Anchor` | 从离线数据帧建立锚点(帧索引处必须已有完整 H=40 步动作历史) |
| **目标与障碍** | 目标 x/y/半径 + `设置目标` | 指定末端目标点(半径>0 则为目标区域) |
| | 障碍 x/y/半径 + `添加障碍` | 指定 keep-out 圆形障碍 |
| **相机锚定与工具** | `从相机取流锚定` | 将“当前相机骨架 + 最近 H 步动作”冻结为 planner 起点，不是创建目标。界面会逐项显示尚缺的前置条件 |
| | `生成 Mock Warmup 历史` | 只在 Mock 阀模式用于算法调试，生成 H 步历史，不下发真阀。真机 Warmup 必须使用 ACK `applied6`，当前未接入时 fail-closed |
| | `零历史起步` | Mock/Real 都可由操作员显式选择。前 H 步为 OOD，初始预测可能偏差较大；后续用 ACK `applied6` 逐步替换零填充历史 |
| | `点加目标` / **`点出目标骨架`** | 目标是“单一活动目标”：新目标会替换旧目标。目标骨架需按末端 node0 到基座 nodeN-1 顺序点出与模型 N 一致的节点，再点 `完成目标骨架` |
| | `点加障碍` | 障碍可以有多个，planner 对所有障碍累计碰撞代价 |
| **场景编辑** | scene_editor 原语列表 | 目标骨架作为一个 `target_skeleton` 对象显示，可重命名、删除或重画；支持 Ctrl 多选 / Del 批量删除 |

**这一步要产出**:一个锚点(anchor)+ 一份场景(目标 + 障碍)。第 3 页规划必需 anchor。

**Planner 的目标语义**:

- `target_point/target_circle`：优化指定节点（当前 GUI 为末端 node 0）到目标点/圆区域的距离。
- `target_skeleton`：对每个预测骨架节点与对应目标节点计算距离，优化全身形状；节点数和顺序必须与模型一致。
- 当前 planner 只支持一个活动目标，不会自动求多目标折中；GUI 会用新目标替换旧目标。

#### Anchor 是什么、怎么建(离线 NPZ 方式,新手从这里开始)

**Anchor = 模型规划的起点**:OpenLoop 模型是状态转移模型 `s_t = F(s_{t-1}, 最近 H 步动作, z)`,它需要知道"现在软臂在什么形状 + 最近 H 步怎么动的",才能预测下一步。这个"现在的形状 + 最近 H 步动作"就是 anchor。**它不是界面填出来的,是从 transition NPZ 数据里选一帧提取的。**

**最快跑通步骤(示例数据已内置)**:
```text
1. Setup 页:  New Experiment → Load Model(checkpoint 已放 checkpoints/current/)
2. Observe 页: 左下『离线锚定』卡,Transition NPZ 已默认指向 data/npz/seq_20260627_163921_train.npz
3. 帧索引保持 39(≥H-1=39,保证有完整 40 步历史)→ 点『从 NPZ 建立 Anchor』
4. 底部 anchor_status 显示『已锚定 …』+ 右侧 scene_summary 更新 → 可去第 3 页规划
```

**规则(报错时会提示怎么改)**:
- NPZ 必须含 `positions(T,3,N)` + 已归一化的 `actions(T,D)`。旧数据可令 D 等于模型维度；
  六腔受约束数据保留 `D=6`，工作台按部署 `channel_map=[0,1,3,4]` 投影为四维模型历史。
- 帧索引往前必须凑满 H=40 步历史 —— 选太靠前会报"缺少 N 步历史",把帧索引调大即可(示例数据 8172 帧,用 39+)。
- 其它序列:把 transition npz 拷入 `real_validation/data/npz/`,或点 `…` 选择任意位置的文件。

**另一种锚定方式(实时相机，当前用于 Mock/感知调试)**:
```text
1. Setup 页:  选择 profile → 应用配置 → 连接相机 → 新建实验 → 加载模型
2. Observe 页: 确认右侧主画面为所选 cam
3. Mock 模式可生成 Mock Warmup 历史，或显式勾选『零历史起步』
   → 点『从相机取流锚定』
   —— 用全 0 历史起步(⚠️ OOD,首窗口可能不准),运行几步后自动用本次真实动作累积历史
4. anchor_status 显示『已锚定 … 零历史起步(OOD)』
```

Real 阀模式下仍允许操作员显式选择零历史，但会保留 OOD 警告和 Anchor 标记；“生成 Mock Warmup”不会被冒充为真机历史。

**关于"运行后自动累积历史"**:执行时(第 4 页)每次实际下发动作都会累积进本次实验的历史(`_history_buffer`),供后续滚动重锚定/重规划使用 —— 你不再需要手动从历史加载,历史由本次实验自己产出。

### 2.3 Page 3 · Plan — 逆规划 + 预检 + 预览

> 前置:已加载模型 + 已建立 anchor。

| 卡片 | 控件 | 作用 |
|---|---|---|
| **规划参数** | K / 优化迭代 / 多起点 / 动作周期(s) / 模型维度→硬件通道 | shooting planner 的视野、优化预算、随机重启数;动作周期自动取训练实测 Δt;通道映射默认 `0` |
| **规划与预检** | `运行 OpenLoop Planner` | 后台求解 K 步动作序列使预测末端/全身达到目标 |
| | `取消规划` / `导入 plan.json` / `运行 Preflight` | 取消后台任务 / 载入历史计划 / 运行安全门禁 |
| | plan_summary | 显示规划进度或 Preflight 的 PASS / BLOCKED 明细 |
| **预览(页底部)** | 预测全身轨迹 + 六通道动作曲线 | 逐 k 步拖动查看预测形态,核对动作是否落在安全上下界内 |

**这一步要产出**:一份通过 Preflight 的计划(plan)。`Preflight: PASS` 是进入第 4 页的必要条件;BLOCKED 会列出具体失败门(单位链缺失/k_safe 未认证/Δt 不匹配/障碍类型不支持/碰撞等)。

### 2.4 Page 4 · Execute — 确认并执行

> 前置:计划已 Preflight PASS 且 session 处于 `ready`。

| 卡片 | 控件 | 作用 |
|---|---|---|
| **执行控制** | `Arm / Confirm` | 操作员确认,进入 `armed`(唯一放行执行的门) |
| | `运行 Mock 计划` / `执行真机计划` | 文案由已应用的阀 backend 决定；两者都逐条等待 ACK |
| | `归零并重新锚定` | 中止计划并归零，进入 `reanchor`；旧计划不可 Resume |
| **执行日志** | execution_log | 命令下发/ACK/状态流转记录 |

**安全行为**:Arm 之前必须连好模型通道及 equality follower 涉及的所有阀组；Abort/归零全通道归零；执行中锁定 1/2/3 页；关窗时统一中止执行、停相机/NDI、归零关阀。

### 2.5 Page 5 · Results — 查看结果

| 卡片 | 控件 | 作用 |
|---|---|---|
| **结果与指标** | results | 执行摘要 + 指标,见下 |

**执行完成自动展示(实测指标)**:
- **计划侧场景指标**(用预测轨迹 + 场景,离线下即可算):末端目标距离、目标达成 ✓/✗、最小障碍间距、是否碰撞;
- **命令安全**:压力越界数、速率越界数(`evaluate_command_safety`);
- **jitter 统计**:mean/max(执行时基偏差);
- `prediction-to-execution gap`:当前标为 `待真机闭环(M5)` —— 需要执行期观测骨架,Mock 链路暂缺,不假装有数。

完整执行记录落在 `run_<时间戳>/execution.csv`(requested/applied/ACK/时间),可离线重放。

---

## 3. 标准操作流程(从零开始,五分钟走通 Mock)

```text
1. Setup:   全 Mock → 应用配置 → 连接相机/所需阀组/NDI
            → 新建实验 → 加载模型 → 应用安全配置
2. Observe: 从 NPZ 建立 Anchor(选一帧) → 设置末端目标 → (可选)添加圆障碍
            -- 或实时:Start Camera → Warmup → 从相机取流锚定 --
3. Plan:    运行 OpenLoop Planner → 看 Preflight → 确认 PASS → 预览轨迹
4. Execute: Arm / Confirm → 运行 Mock 计划
5. Results: 查看执行摘要 → run 目录下找 execution.csv
```

---

## 4. 状态机与页面锁(内部逻辑,了解即可)

session 状态机:`IDLE → PLANNING → READY → ARMED → EXECUTING → PAUSED/REANCHOR/COMPLETED → ZEROED/ERROR`。

- **ready** = 有模型 + 有锚点 + 计划通过 Preflight → 第 4 页 Arm 按钮亮起。
- **armed/executing/paused** 期间,第 1/2/3 页自动锁定(灰掉),防执行中改 scene 破坏记录溯源。
- **Abort/归零**始终可用;归零失败会二次重试并显式报 `ERROR`。

---

## 5. 当前能力边界(诚实声明)

| 能力 | 状态 |
|---|---|
| 执行链路 | Mock/Real 显式可选且共用 Controller + QtValveTransport + ACK 路径；未 READY 或必需阀组缺失时禁止 Arm，不回退 |
| 相机 | 1–8 路 Mock/RealSense，支持枚举或唯一 serial 选择。当前主显示可切换；多视图原子新鲜度门禁仍属后续在线同步评价阶段 |
| NDI | Disabled/Mock/Real 显式生命周期；当前可展示评价曲线，执行期完整隐藏/解封和在线同步落盘属 Phase D |
| 在线锚定/warmup | 可用;**零历史起步**(默认勾选,免 warmup,⚠️ OOD 标注);Warmup 可选填真实历史;真机需带 manifest 的 checkpoint |
| 场景编辑 | camera_view + scene_editor 已接;**骨架叠加显示**(青线+圆点,与训练同源);`obstacle_aabb` 支持解析但交互创建待后续 |
| 目标类型 | 末端 target_point/circle、**全身 target_skeleton(可『点出目标骨架』交互创建)** + 圆障碍;多边形/AABB 障碍、多检查点重锚定待 P3/M4+ |
| 执行累积历史 | 执行时实际下发动作自动累积进本次实验历史(供滚动重锚定),不需手动加载历史 |
| 离线锚定 | 示例数据已内置 `data/npz/`(15 节点,8172 帧);`checkpoints/current/best_model.pt` 已放置,可直接走通 Mock 流程 |
| K_safe | 加载带认证表的模型后自动填充;`未认证` 时规划会被 preflight 阻断(fail-closed) |

> 目标取自真实录制帧可保证物理可达;planner 优化出的动作**尚未在真机执行过** —— 真机闭环是下一个里程碑。
