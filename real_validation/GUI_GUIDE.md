# 实机验证工作台 GUI 使用指南

> 运行:`python -m real_validation.main`(或双击 `run_gui.bat`)。本目录随部署整体拷贝到 PC,**完全自包含**(不需 real_capture 并排部署),操作者机器上直接可用。
> 对应设计文档:`docs/experiments/real_robot_validation_workbench_todo.md`、`docs/superpowers/specs/2026-07-28-real-validation-task-layer-ik-design.md`。

---

## 1. 总体:五页 = 一条验证流水线

GUI 用五个 Tab 页串起"**加载模型 → 建立状态锚点 → 定义目标/障碍 → 规划 → 安全执行 → 查看结果**"的完整闭环。**使用顺序就是页面顺序,通常从左到右、逐页推进**:

```
1 Setup  ──▶  2 Observe & Scene  ──▶  3 Plan  ──▶  4 Execute  ──▶  5 Results
建立实验       建立锚点+场景            规划+预检       安全执行         看指标
加载模型       定义目标/障碍            预览预测轨迹     Arm→执行         导出
安全配置
```

顶部始终显示全局安全栏:`Run: <实验名>    State: <状态>    Hardware: MOCK`,颜色随状态变化:
- 灰 未建实验(`no_session`)· 蓝 `idle`(已建实验待机)· 绿 `ready`/`completed`/`zeroed` · 橙 `armed`/`paused`/`planning`/`reanchor` · 红 `executing`/`aborting`/`error`

**关键约束**:各页之间有前置依赖(见下),第 2~4 页在不满足前置时会弹窗拒绝并提示。`执行中(armed/executing/paused)会自动锁定第 1/2/3 页`,防止执行中改场景破坏记录溯源。

---

## 2. 各页详解

### 2.0 整体布局 —— 主窗口左右两栏

主窗口(顶部全局安全栏之下)为**左右两栏**:

- **左栏 · 主显示区(常驻)**:摄像头画面 + 多层叠加 + 图层开关。跨 Tab 常驻,切换页面不消失。
  - 叠加层:骨架(青线+圆点,与训练同源)/ 场景(目标、障碍)/ 预测轨迹 / 实际骨架 / NDI 位姿。
  - 下方**图层开关**复选框:骨架 / 场景 / 预测 / 实际 / NDI(默认关 NDI),可逐层独立开合。
- **右栏 · 5 页 Tab**:Setup → Observe & Scene → Plan → Execute → Results(使用顺序即页面顺序,见 §1)。

**Observe 锚定交互仍在右栏**:Observe 页(第 2 页)内部仍为左右两栏 —— 左控制面板,右场景编辑(camera_view 锚定视图 + scene_editor 原语列表)。`从相机取流锚定`、`点加目标`/`点出目标骨架`/`点加障碍` 等锚定/点选交互在其右栏 camera_view 上操作;同一路摄像头流同时送入左栏主显示区与 Observe 锚定视图,两侧均显示实时骨架。

**场景编辑多选删除**:scene_editor 原语列表支持 **Ctrl 多选 / Shift 范围 / Ctrl+A 全选**,点 `删除` 按钮或按 `Del`(Backspace)键批量删除选中原语。

### 2.1 Page 1 · Setup — 建立实验、加载模型、配置安全

| 卡片 | 控件 | 作用 |
|---|---|---|
| **实验与运行** | Run 根目录 + `New Experiment` | 在指定目录下创建唯一 `run_<时间戳>` 实验目录(所有产物写这里) |
| | `Open Run (Replay)` | 只读打开历史 run 目录,回看当时的模型/场景/计划/结果(**不能 Arm**) |
| **模型与部署契约** | Checkpoint / 训练数据目录 / K_safe / 设备 | 指定模型权重路径;K_safe(规划视野上限)在加载后按认证表自动填充,并在模型摘要标注来源(`认证表(10px 容差)` 或 `手动`),悬停 K_safe 可见说明 |
| | `Load Model` | 后台加载 checkpoint,显示 type/action_dim/nodes/H/K_train/K_safe/train_dt/action_scale_kpa/hash |
| **安全配置(六通道)** | min/max/rise/fall/initial × 6 通道 | kPa 上下界 + 升/降压速率 + 初始动作,默认 max=150 kPa(对齐训练域) |
| | `应用安全配置并使旧计划失效` | 写入 session 并落盘 safety.json,任何安全改动都会让旧计划失效 |
| **硬件连接(真机)** | 组1 串口 / 组2 串口 / baudrate + `连接阀` | **真机执行必需**。填两组串口(COM)后点连接(后台线程,不卡 GUI)。连接成功 → 状态栏 `Hardware: REAL VALVE`,执行走真阀(QtValveTransport);失败/不连 → 回退 `Mock` |
| | `断开阀` | 释放串口,执行回退 Mock |
| | 配置持久化 | 串口/baudrate 保存到 `config/hardware.json`(gitignore 不入库),下次启动自动回填 |

**这一步要产出**:一个 run 目录 + 一个已加载的模型 + 一份安全配置(+ 可选:真机阀已连接)。

> **真机接线需要什么**:串口阀两组 Modbus(组1=ch0-2,组2=ch3-5,4-20mA)+ 依赖 `requirements-hardware.txt`(pyserial/pyrealsense2/scikit-surgerynditracker)。相机/NDI 适配层在 `real_validation/hardware/`,Mock 流程不触碰。

### 2.2 Page 2 · Observe & Scene — 建立锚点、定义目标与障碍

> 前置:已 `Load Model`。锚点(s_{t-1} + 动作历史)是模型规划的起点,必须有。
> 布局:Observe 页内部**左右两栏** —— 左栏控制面板(离线锚定/目标与障碍/实时相机),右栏场景编辑(camera_view 锚定视图 + 原语列表)。注意:主窗口左栏已有常驻**主显示区**(摄像头 + 多层叠加 + 图层开关,见 §2.0);Observe 锚定交互仍可用 —— `从相机取流锚定` 与点选工具在其右栏 camera_view 上操作,同一路摄像头流同时送入主显示区与锚定视图。

| 卡片 | 控件 | 作用 |
|---|---|---|
| **离线锚定** | `加载 anchor.json` / `加载 scene.json` | 载入历史 run 的锚点/场景 |
| | Transition NPZ + 帧索引 + `从 NPZ 建立 Anchor` | 从离线数据帧建立锚点(帧索引处必须已有完整 H=40 步动作历史) |
| **目标与障碍** | 目标 x/y/半径 + `设置末端目标` | 指定末端目标点(半径>0 则为目标区域) |
| | 障碍 x/y/半径 + `添加圆障碍` | 指定 keep-out 圆形障碍 |
| **实时相机与 Warmup** | `Start Camera (Mock)` | 启动实时取流(当前为合成演示帧;真机换 RealSense) |
| | `Warmup(填动作历史)` | 用训练分布内动作填满 H 步历史(可选;勾了零历史起步可跳过) |
| | `零历史起步(免 warmup,首窗口 OOD)` | ⚠️ 勾选后用全 0 历史直接锚定,不需先 Warmup。模型训练从没见过零填充窗口,首窗口预测可能不准;运行几步后自动用本次真实动作累积历史,误差收敛。**默认勾选** |
| | `从相机取流锚定` | 从当前帧经质量门控 → 分割 → 骨架 → 归一化,建立实时锚点(可零历史起步) |
| | 工具:`select` / `点加目标` / **`点出目标骨架`** / `点加障碍` | 点选/绘制目标、障碍;**点出目标骨架**:依次点 N 个点连成期望软臂形状,双击完成(选中工具呈青底高亮) |
| **场景编辑** | camera_view + scene_editor | 图像/骨架/目标/障碍可视化(青线+圆点 = 实时骨架,与训练同源),下方原语列表(增删改/锁定;**Ctrl 多选 / Del 批量删**) |

**这一步要产出**:一个锚点(anchor)+ 一份场景(目标 + 障碍)。第 3 页规划必需 anchor。

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
- NPZ 必须含 `positions(T,3,N)` + `actions(T,D)`;N = 模型节点数(15),D = action_dim(1),动作已归一 [0,1]。
- 帧索引往前必须凑满 H=40 步历史 —— 选太靠前会报"缺少 N 步历史",把帧索引调大即可(示例数据 8172 帧,用 39+)。
- 其它序列:把 transition npz 拷入 `real_validation/data/npz/`,或点 `…` 选择任意位置的文件。

**另一种锚定方式(实时相机,零历史起步,免 warmup)**:
```text
1. Setup 页:  New Experiment → Load Model
2. Observe 页: 实时相机卡 → Start Camera (Mock)
3. 保持『零历史起步』勾选(默认)→ 直接点『从相机取流锚定』
   —— 用全 0 历史起步(⚠️ OOD,首窗口可能不准),运行几步后自动用本次真实动作累积历史
4. anchor_status 显示『已锚定 … 零历史起步(OOD)』
```

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
| | `Mock Execute` | 逐条下发动作并等待 ACK(当前为 Mock 传输,真机换 QtValveTransport) |
| | `Pause` / `Resume` | 执行中暂停(零压)/恢复 |
| **执行日志** | execution_log | 命令下发/ACK/状态流转记录 |

**安全行为**:Abort/归零(顶部)永远可用且全部归零;执行中锁定 1/2/3 页;Pause 采用零压策略,恢复需重规划。

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
1. Setup:   New Experiment → Load Model(checkpoint) → 应用安全配置(默认即可)
2. Observe: 从 NPZ 建立 Anchor(选一帧) → 设置末端目标 → (可选)添加圆障碍
            -- 或实时:Start Camera → Warmup → 从相机取流锚定 --
3. Plan:    运行 OpenLoop Planner → 看 Preflight → 确认 PASS → 预览轨迹
4. Execute: Arm / Confirm → Mock Execute → (Pause/Resume)
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
| 执行链路 | **Mock 默认**;Setup 页『硬件连接』填两组串口连真机阀后 → `Hardware: REAL VALVE`,执行走 QtValveTransport(线程安全桥接 `hardware/valve.py` 的 ValveController,已自包含移植);失败/未连回退 Mock |
| 相机 | **合成演示帧**(Mock);真机换 RealSenseCam(`hardware/camera.py` 适配层已就位,含指纹断言) |
| NDI | `hardware/ndi.py` 适配层已就位(隐藏评价流,只进评价不进模型);真机接 COM 后启用 |
| 在线锚定/warmup | 可用;**零历史起步**(默认勾选,免 warmup,⚠️ OOD 标注);Warmup 可选填真实历史;真机需带 manifest 的 checkpoint |
| 场景编辑 | camera_view + scene_editor 已接;**骨架叠加显示**(青线+圆点,与训练同源);`obstacle_aabb` 支持解析但交互创建待后续 |
| 目标类型 | 末端 target_point/circle、**全身 target_skeleton(可『点出目标骨架』交互创建)** + 圆障碍;多边形/AABB 障碍、多检查点重锚定待 P3/M4+ |
| 执行累积历史 | 执行时实际下发动作自动累积进本次实验历史(供滚动重锚定),不需手动加载历史 |
| 离线锚定 | 示例数据已内置 `data/npz/`(15 节点,8172 帧);`checkpoints/current/best_model.pt` 已放置,可直接走通 Mock 流程 |
| K_safe | 加载带认证表的模型后自动填充;`未认证` 时规划会被 preflight 阻断(fail-closed) |

> 目标取自真实录制帧可保证物理可达;planner 优化出的动作**尚未在真机执行过** —— 真机闭环是下一个里程碑。
