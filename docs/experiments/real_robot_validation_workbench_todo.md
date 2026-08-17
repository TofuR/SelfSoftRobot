# 实机模型验证工作台：功能设计与 TODO

> 状态：Phase 0 已启动（2026-07-20）；首批核心实现见 `real_validation/`
> 目标：把“加载模型 → 获取状态锚点 → 定义目标与环境 → OpenLoop IK/规划 → 安全执行 → 同步采集 → 独立评价”打通为可复现的实机实验闭环
> 上游方案：[`openloop_sparse_observation_validation_plan.md`](openloop_sparse_observation_validation_plan.md)
> 可复用基础：`real_capture/`、`src/utils/model_loader.py`、`scripts/control/inverse_plan.py`、`scripts/evaluation/eval_real_quant.py`
>
> 当前边界：数据契约、状态机、OpenLoop shooting、preflight、Mock ACK 执行、Qt 真阀
> 线程桥接、基础离线指标和五页 GUI 骨架已实现。GUI 仍只开放 Mock 执行；在线视觉
> 锚定、多相机场景交互、同步隐藏真值记录和任务插件尚未完成。
>
> PC 部署：`real_validation/` 已内置当前 OpenLoop fractional 推理结构和 rollout，运行时
> 不再导入仓库的 `src/`、`scripts/`、`real_capture/` 或 `config/`。checkpoint、数据、
> 标定、硬件配置和 run 均使用该目录内的固定相对位置。

---

## 1. 工作台要解决的问题

现有 `real_capture/main_capture.py` 面向数据采集，能够连接六通道阀、相机和 NDI，并按动作时钟同步落盘。但论文后续需要的是另一条闭环：

```text
加载训练模型与标定
  -> 获取当前真实形态和最近动作历史
  -> 在场景中指定任务条件
  -> 求解可达、满足安全约束的动作序列
  -> 预览预测全身轨迹并人工确认
  -> 实机执行，遮挡期间不向模型泄漏隐藏真值
  -> 相机/NDI继续独立记录真值
  -> 自动评价预测、规划与实机执行差距
```

工作台不是把采集 GUI 改成“自动控制按钮集合”，而是统一四类状态：

1. **硬件状态**：阀、相机、NDI、通信 ACK、急停；
2. **模型状态**：checkpoint、归一化、H、`K_train`、`K_safe`、action dimension；
3. **任务状态**：目标、障碍、门区、观察口和安全边界；
4. **实验状态**：计划版本、执行进度、有效观测、隐藏评价流和结果文件。

---

## 2. 设计原则

### 2.1 新建验证工作台，复用底层服务

不建议继续把规划、执行和任务状态堆进现有 `CaptureWindow`。推荐新增 `real_validation/` 应用，并逐步抽取可复用服务：

```text
real_validation/
  main_validation.py          # GUI 入口
  session.py                  # 全局实验状态机
  hardware_session.py         # 共享阀/相机/NDI生命周期与健康状态
  model_runtime.py            # checkpoint + normalization + anchor + rollout
  scene.py                    # 目标、障碍、门区、观察口的数据模型
  planner_service.py          # 异步 IK / 约束轨迹规划
  executor.py                 # ACK 驱动的动作序列执行器
  validation_recorder.py      # 计划、执行、隐藏真值和指标落盘
  coordinate_system.py        # pixel / model-normalized / NDI / world 转换
  observation_policy.py       # 哪些观测允许送入模型
  tasks/
    base.py                   # TaskPlugin 接口
    reach.py                  # 调试性末端到达
    path_dependent_ik.py      # 路径依赖 IK
    occluded_trajectory.py    # 不可见区间全身约束规划
    checkpoint_navigation.py  # 多窗口重锚定
    channel_inspection.py     # 不透明通道组合应用
  widgets/
    scene_view.py
    model_panel.py
    execution_panel.py
    result_panel.py
```

`real_capture/` 保留采集职责；阀控制、相机、NDI 和 recorder 通过稳定接口被两个 GUI 共享。规划核心应从 `scripts/control/inverse_plan.py` 抽成库，CLI 只做参数解析和结果输出。

### 2.2 规划、执行、评价三条数据流必须分开

```text
控制流：允许观测 + 动作历史 -> 模型 -> planner -> valve
评价流：全部相机/NDI隐藏真值 -> recorder -> metrics
显示流：允许显示给操作员的信息 + 独立安全状态
```

在模拟遮挡时，相机和 NDI仍可继续记录，但禁止送入模型或规划器。否则得到的是视觉闭环结果，不是稀疏观测 OpenLoop 结果。

### 2.3 默认“先预览、再授权、后执行”

规划完成不能自动下发。每个计划必须经过：

```text
模型兼容检查 -> 可达性检查 -> 轨迹/障碍检查
-> 压力与速率检查 -> 操作员 Arm/Confirm -> 执行
```

急停、暂停和全部归零始终可用，不受当前页面、规划线程或录制状态影响。

---

## 3. 现有能力与缺口

| 能力 | 当前状态 | 复用/新增 |
|---|---|---|
| 六通道阀、分组连接、归零 | 已有 | 复用 `ValveController` |
| 压力上下界与 rise/fall 速率限制 | 已有 | 复用；执行前再做整段静态检查 |
| 命令 ID、`t_command/t_ack`、通信状态 | 已有 | 复用；执行器改为 ACK 感知 |
| 1–8 相机预览与同步保存 | 已有 | 复用相机与有界保存队列 |
| 1–8 NDI 探头 | 已有 | 复用；区分模型输入和隐藏评价 |
| actions6 Replay | 已有 | 复用为计划执行底层，但需执行状态机 |
| checkpoint 自动识别与归一化 | 已有 | 封装成 GUI 模型运行时 |
| OpenLoop rollout | 已有 | 抽成统一可微/无梯度接口 |
| 离线 shooting IK、多起点、圆障碍 | 部分已有 | 从 CLI 抽库并扩展目标/约束 |
| 从当前实时图像得到模型状态 | 缺失 | 新增在线骨架提取和锚定 |
| 当前动作历史 H 的实时缓存 | 缺失 | 新增运行时 ring buffer |
| pixel / normalized / NDI / world 统一坐标 | 缺失 | 新增坐标系统与标定管理 |
| 场景和目标交互编辑 | 缺失 | 新增 Scene Editor |
| 可达性残差与不可达提示 | 缺失 | 新增 reachability 层 |
| 安全计划执行、暂停、重锚定、重规划 | 缺失 | 新增 executor + session state machine |
| 规划预测与实机执行自动对齐评价 | 缺失 | 新增 validation recorder/metrics |
| 机制/任务/应用实验模板 | 缺失 | 新增 TaskPlugin |

---

## 4. GUI 总体布局与主流程

### 4.1 建议页面

顶部保留跨页面的全局安全栏：硬件连接摘要、模型状态、当前实验状态、暂停、急停/归零。

主区域使用五个步骤页：

1. **Setup**：硬件、checkpoint、标定、通道映射和安全配置；
2. **Observe & Scene**：获取状态锚点，选择目标、障碍、门区和观察口；
3. **Plan**：设置 H/K、目标权重，异步求解并预览候选轨迹；
4. **Execute**：确认、下发、暂停、重锚定和重规划；
5. **Results**：预测—执行对齐、成功判据、指标和导出。

中心视图复用固定总尺寸的多相机预览，并支持叠加：

- 当前观测骨架；
- OpenLoop 预测骨架序列；
- 目标点/区域；
- 障碍和安全膨胀边界；
- 允许观察的检查点；
- NDI 评价点；
- 当前 `k_since_observation / K_safe`。

### 4.2 标准操作流程

```text
New Experiment
  -> Connect Hardware
  -> Load Model
  -> Validate Compatibility
  -> Load/Create Scene
  -> Observe & Anchor
  -> Select Task
  -> Plan
  -> Preview + Preflight
  -> Arm + Execute
  -> Re-anchor/Replan when policy allows
  -> Stop
  -> Evaluate + Export
```

每一步都允许保存草稿并恢复，不依赖 GUI 内存中的临时对象。

---

## 5. 通用基础功能

### C1. 项目、配置与模型管理

- [ ] 新建/打开实验目录，禁止覆盖已有 run；
- [ ] checkpoint 文件选择与最近使用列表；
- [ ] 显示模型类型、action_dim、节点数、H、`K_train`、归一化参数、训练数据和 checkpoint hash；
- [ ] 加载与模型关联的 `K_safe(ε)` 认证结果；
- [ ] 模型 action_dim 与当前有效阀通道映射检查；
- [ ] 1/3/6 通道模型都能工作，未使用通道显式锁零；
- [x] 模型设备选择 CPU/GPU，加载失败可恢复；
- [x] 模型切换时清空旧 anchor、z、动作历史和计划，防止跨模型污染；
- [ ] 配置快照写入实验目录，不依赖之后被修改的全局 ini。

验收：加载错误维度 checkpoint 时必须阻止规划和执行；加载正确模型后能用 mock state 完成一次 rollout。

### C2. 坐标系与标定

统一声明四类坐标：

```text
camera pixel
model world/pixel state
model normalized state
NDI/world millimeter
```

- [ ] 集中实现坐标转换，禁止各 widget 自己换算；
- [x] 目标、障碍和预测轨迹均携带 `frame_id`；
- [ ] 相机点击点能转换到模型状态空间；
- [ ] NDI teach-in 点能转换到显示/任务空间；
- [ ] 多相机模式支持选择工作平面或三角化 3D 点；
- [ ] 标定文件数量、相机序列号和视角顺序严格检查；
- [ ] 显示转换残差和有效范围，超范围目标禁止执行；
- [ ] 2D 模型只允许声明平面避障，界面不得标成 3D 无碰撞。

验收：同一点经 pixel→model→pixel 往返误差低于预设阈值；NDI 标定残差在界面可见并写入结果。

### C3. 实时状态锚定与动作历史

- [ ] 从当前相机帧运行在线分割/骨架化，输出与训练一致的 N 节点；
- [ ] 提供 white-on-blue 快速模式和可选学习分割模式；
- [ ] 显示骨架质量、时间戳、frame age 和处理延迟；
- [ ] “Observe & Anchor”按钮冻结一个经过质量检查的状态；
- [x] 维护最近 H 步实际下发动作 ring buffer；
- [x] 冷启动时要求先建立足够历史，或显式选择 padding 策略；
- [ ] 初始化/恢复 OpenLoop 的 `s`、`z` 和动作窗口；
- [ ] 允许在指定检查点重新观察并重锚定；
- [ ] 显示距最近有效观察的步数和秒数；
- [ ] 超过 `K_safe` 时阻止继续自动执行，要求观察或人工授权停止。

验收：mock 相机下锚定状态与离线数据加载结果一致；遮挡模式开启后，隐藏评价帧不能进入 model runtime。

### C4. 场景编辑器与通用任务原语

目标和约束不应写死在具体任务中。统一支持：

- [ ] 末端目标点；
- [ ] 末端目标区域：圆、矩形、多边形；
- [ ] 从相机画面点击目标物/框选 ROI，并把其 mask/轮廓转成目标区域；
- [ ] 指定节点/指定段进入目标区域；
- [ ] pass-through gate / waypoint；
- [ ] keep-out 障碍：圆、AABB、多边形、二值 mask；
- [ ] 从画面选择障碍物轮廓并生成带安全膨胀的 keep-out 区域；
- [ ] 可选视觉目标描述符/目标轮廓，不要求精确复制一个可能不可达的全身形状；
- [ ] 工作空间边界；
- [ ] 全身安全半径/障碍膨胀余量；
- [x] 压力上下界、每通道 rise/fall 和固定锁零通道；
- [ ] 允许观察的检查点/观察口；
- [ ] 鼠标点击、拖拽、数值输入、NDI teach-in 和文件导入；
- [ ] Undo/Redo、删除、锁定和 scene JSON 保存/加载。

每个 scene primitive 必须包含类型、坐标系、几何参数、安全余量和名称。任何未转换到规划坐标系的对象不得参与 loss。

第一版只支持执行期间不移动的静态目标和障碍。动态环境需要在线环境感知、运动预测和更高频重规划，不能仅靠当前软臂 OpenLoop 模型解决。

### C5. 规划器服务

- [ ] 将 `inverse_plan.py` 的 rollout、multi-start 和 loss 拆到可导入模块（rollout 已抽出，GUI shooting 已接入）；
- [ ] 保留 CLI，CLI 和 GUI 调用同一 planner API；
- [x] 支持目标区域而非只能从离线数据选择完整目标骨架；
- [ ] 支持多目标/多 waypoint 和全程约束；
- [ ] 支持 2D 圆/AABB/多边形障碍，后续扩展 3D；
- [ ] 增加可达性投影和 residual gap，明确显示“不可达”；
- [ ] K 自动建议必须受当前模型 `K_safe` 限制（越界阻断已完成，自动建议未完成）；
- [x] 规划动作经过压力和速率投影；
- [x] 规划在后台线程/进程运行，GUI 不阻塞；
- [ ] 支持取消、超时、多起点进度和候选排序（取消和多起点择优已完成）；
- [x] 优化过程限制计算图生命周期，候选输出使用 `no_grad`，避免 GPU 内存持续增长；
- [ ] 输出各 loss 分量、梯度异常、迭代时间和随机种子；
- [ ] 保存完整 plan JSON 和动作 CSV，可离线复算。

验收：同一输入和 seed 的 CLI/GUI 计划一致；取消规划后线程和 GPU 内存可回收；不可达目标不会生成“成功”状态。

### C6. 计划预览与审批

- [ ] 动画播放预测全身轨迹；
- [x] 同时显示动作 6 通道曲线和上下界（速率曲线待补）；
- [ ] 显示每一步目标误差、最小障碍间距和 `k/K_safe`；
- [ ] 对违反约束的时间点高亮；
- [ ] 可比较多个候选计划；
- [x] 操作员选择候选后生成不可变 plan version；
- [x] 修改 scene、anchor、模型或安全参数后自动使旧计划失效；
- [x] 只有通过 preflight 的计划才允许 Arm。

### C7. 安全执行器

执行状态机：

```text
IDLE -> PLANNING -> READY -> ARMED -> EXECUTING
                         -> PAUSED -> REANCHOR/REPLAN
                         -> COMPLETED
任意状态 -> ABORTING -> ZEROED/ERROR
```

- [x] 使用现有 `ValveController`，不得复制 Modbus 实现；
- [x] 每个动作记录 requested/applied、command_id、ACK 和时间；
- [x] ACK 超时、queue full、串口错误立即中止并归零；
- [x] 执行时再次应用 rise/fall limiter，计划值与 applied 值不同时记录偏差；
- [x] 禁止规划任务使用 `bypass_rate=True`；
- [ ] Pause 保持还是归零必须作为明确安全策略，不可隐式决定；
- [x] Abort/E-stop 永远归零所有已连接组；
- [ ] 看门狗监测 GUI、执行线程和硬件通信；
- [ ] 支持逐步执行、低速 dry-run 和完整序列执行；
- [ ] 支持执行 N 步后进入观察/重规划检查点；
- [ ] 关闭窗口时先停执行、归零、等待队列，再释放硬件。

验收：mock 模式可注入 ACK timeout/queue full 并验证安全转移；真机先完成低压单步和急停测试，再开放序列执行。

### C8. 记录与独立评价

每次 run 建议保存：

```text
run_xxxx/
  experiment.json       # 任务、观测策略、成功阈值
  model.json            # checkpoint/hash/H/K/K_safe/normalization
  calibration.json
  scene.json
  anchor.npz            # 初始状态、动作历史、时间
  plan.json
  planned_actions6.csv
  execution.csv         # requested/applied/ack/status
  predicted_states.npz
  observations.csv      # 哪些观测允许进入模型
  hidden_ground_truth/  # 相机/NDI评价流
  metrics.json
  summary.md
```

- [ ] 复用 `ValveRecorder` 的相机/NDI/命令生命周期记录；
- [ ] 增加计划版本、task step 和 observation_allowed 字段；
- [ ] 明确区分 commanded pressure 与 measured pressure；当前阀无压力反馈，不得把命令写成实测气压；
- [ ] 对齐 predicted state、command ACK、frame grab 和 NDI；
- [x] 自动计算 prediction-to-execution error-by-k；
- [x] 自动计算末端误差、全身 MNE、p90、最大误差；
- [x] 自动计算最小安全间距、碰撞和压力/速率违反；
- [ ] 自动计算观测次数、最长不可见时间、重规划次数和任务成功；
- [ ] 支持一键导出论文图表所需 CSV/JSON，而不是只保存截图；
- [ ] 支持从 run 目录离线 replay GUI 和重新计算指标。

---

## 6. 可集成的任务模块

所有任务实现统一接口：

```text
TaskPlugin
  required_capabilities()
  build_constraints(scene, anchor)
  build_observation_policy()
  preflight(plan, hardware, model)
  next_step(execution_state)
  success_criteria(result)
  compute_metrics(result)
```

### T1. 调试性末端到达

- 选择一个末端点/区域；
- 求解并预览 K 步动作；
- 实机执行后用 NDI 判断成功；
- 用于调通坐标、planner、executor 和 recorder；
- 不作为论文主要应用贡献。

### T2. 路径依赖 IK

- 选择/加载 loading、unloading、循环和变速初始历史；
- 对相同目标分别规划；
- 保存各历史下动作差异；
- 支持计划交叉执行到错误历史；
- 自动统计 history-aware 与 history-reduced 成功率；
- 直接服务实验方案的 `T_IK*` 验证。

### T3. 不可见区间全身约束轨迹

- 窗口开始允许一次锚定；
- 执行期间关闭模型观测输入，但持续记录隐藏真值；
- 同时约束目标、全身障碍距离和动作安全；
- 扫描遮挡长度 K；
- 比较 full-body、endpoint-only 和 history-reduced 计划。

### T4. 多检查点重锚定

- Scene 中配置观察口；
- 到达观察口后暂停、重新分割/锚定；
- 更新动作历史和模型状态；
- 保留剩余全局任务，重新规划下一窗口；
- 记录每次重锚定消除的误差和额外耗时。

### T5. 不透明通道巡检应用

任务状态机：

```text
ENTRY_OBSERVE
-> BEND_1_PLAN/EXECUTE
-> CHECKPOINT_REANCHOR
-> BEND_2_PLAN/EXECUTE
-> INSPECTION_HOLD
-> RETURN_PLAN/EXECUTE
-> EXIT_VERIFY
```

- 支持透明真值通道和不透明验证通道使用同一 scene；
- 支持进入、检查点、巡检保持和回撤成功条件；
- 回撤从当前加载历史重新规划，不直接反转进入动作；
- 自动统计完整流程成功、观察预算、碰撞和失败阶段。

---

## 7. 分阶段实现 TODO

### Phase 0：接口与安全基线

- [ ] 冻结 `ValveController`、相机、NDI、recorder 的共享接口；
- [x] 定义 `ExperimentSession` 状态机和不可变计划版本；
- [x] 定义 model/scene/plan/run JSON schema；
- [ ] 从 `inverse_plan.py` 抽出 planner 库，保持 CLI 兼容；
- [x] 建立 mock 全链路和错误注入；
- [x] 实现独立于 GUI 的 preflight；
- [x] 明确 Pause、Abort、ACK timeout 和窗口关闭安全策略。

完成标志：无相机、无 NDI、无真阀时，mock 能完成“加载模型—规划—执行—记录—评价”，并通过异常注入安全测试。

### Phase 1：离线/Mock 验证工作台

- [x] 建立五步 GUI 页面和全局安全栏骨架；
- [x] checkpoint 后台加载、兼容性元数据和 K_safe 显示；
- [ ] Scene Editor：点、区域、圆/AABB/多边形障碍（数值输入点/圆与圆障碍已完成）；
- [ ] 从离线 npz 或 mock 当前状态建立 anchor（离线 NPZ 已完成）；
- [ ] 异步规划、取消、进度和候选预览（异步规划与取消已完成）；
- [ ] 动作曲线和预测骨架动画；
- [ ] run 目录和离线 replay。

完成标志：不连接硬件也能在 GUI 中复现 `inverse_plan.py` 当前结果，且 CLI/GUI 输出一致。

### Phase 2：真机安全执行与同步评价

- [ ] 复用采集 GUI 的硬件连接组件；
- [x] 实现 ACK 感知的计划执行器；
- [x] 计划静态压力/速率检查；
- [ ] 低压逐步执行、Pause、Abort、归零；
- [ ] 执行期间同步保存相机、NDI、命令和预测；
- [x] 自动计算预测—执行误差的离线核心；
- [ ] 真机急停、断线、queue full 和关闭窗口测试。

完成标志：完成小范围末端到达任务；每一步命令、ACK、图像、NDI 和预测可重放对齐；任何通信失败都安全归零。

### Phase 3：实时观察与稀疏观测

- [ ] 在线分割/骨架化与质量门控；
- [ ] 当前动作历史 H ring buffer；
- [ ] Observe & Anchor / Re-anchor；
- [x] observation policy 隔离控制流和隐藏评价流；
- [ ] `k_since_observation`、K_safe 和过期阻断；
- [ ] 周期、burst、检查点三类观测策略；
- [ ] 多窗口预测和重规划。

完成标志：隐藏评价流开启时，模型输入日志仍只包含允许观察的帧；重锚定后累计误差可见地回落。

### Phase 4：论文任务插件

- [ ] 路径依赖 IK 配对与交叉执行；
- [ ] 目标区域和可达性 residual；
- [ ] 全身障碍/门区约束；
- [ ] endpoint-only 与 history-reduced 对照；
- [ ] K 扫描和 H–K 任务可行域；
- [ ] 自动成功判据与论文表格导出。

完成标志：同一目标的不同历史会产生并实机验证不同动作；不可见区间内可评价完整轨迹和全身安全，而非只看末态。

### Phase 5：不透明通道组合应用

- [ ] 通道 scene 模型、观察口和任务状态机；
- [ ] 透明真值通道实验；
- [ ] 不透明同构通道实验；
- [ ] 进入—多窗口推进—巡检保持—回撤；
- [ ] 观察口数量/位置扫描；
- [ ] 多次独立重复和失败分类；
- [ ] 生成应用层成功率—观测预算曲线。

完成标志：机制层 H–K 可行域能够解释应用成功/失败边界；完整流程结果来自多次独立实机试验，不是单次演示。

---

## 8. 必须提前处理的风险

| 风险 | 工作台要求 |
|---|---|
| 当前模型主要为 1-DOF，实机是 6 通道 | action_dim/通道映射不匹配时硬阻断，不能自动补零后假装兼容 |
| 当前阀只有命令、没有压力反馈 | UI 和日志统一称 command/applied command；未来压力传感器作为新 observation source |
| 当前实物状态主要是 2D 像素骨架 | 障碍约束只声明平面有效；3D 应用需多相机三角化模型与标定 |
| SAM2 可能无法实时 | 在线采用低延迟管线；SAM2可作离线隐藏真值或未来优化，记录实际延迟 |
| 规划约 400 次迭代，未必“实时” | 先称 online planning；记录 latency，达到控制周期预算后再称 real-time IK |
| OpenLoop K_safe 依赖模型和数据 | K_safe 绑定 checkpoint/容限，不做全局常数 |
| 常数半径碰撞近似 | 使用保守膨胀余量并显示近似边界；接触任务暂不开放 |
| 优化器利用模型误差 | 必须实机执行并报告 prediction-to-execution gap、不可达残差和失败案例 |
| GUI/规划/保存竞争内存 | 规划与图像队列有界；结果 tensor 及时 detach；只保留降采样预览轨迹 |
| 隐藏真值泄漏 | 每个观测写 `observation_allowed`，model runtime 只接受 policy 授权的数据 |

---

## 9. 第一版最小可用范围

第一版不要直接实现不透明通道。最小闭环应只包含：

1. 加载一个 OpenLoop checkpoint；
2. 从离线/实时单相机骨架建立状态锚点和动作历史；
3. 在图像上点击一个末端目标点，并可添加一个圆形障碍；
4. 后台运行现有 shooting planner；
5. 预览预测轨迹和 6 通道动作；
6. 通过 preflight 后人工确认执行；
7. 同步记录相机、NDI、命令 ACK 和预测；
8. 自动输出末端误差、全身误差、最小间距和 prediction-to-execution gap；
9. Pause/Abort/归零和 mock 错误注入通过。

这个闭环稳定后，再增加路径依赖配对、稀疏重锚定和通道应用。否则应用任务失败时无法判断是坐标、规划、通信、模型还是任务状态机的问题。
