# 设计:real_validation 任务层 IK 工作台

> 日期:2026-07-28 · 分支 `feat/real-data-transition`
> 目标:把 `real_validation/` 从"Mock 离线骨架"做成**实时形态采集 → 图上指定目标/障碍 → 真机 IK 控制 → 独立评价**的闭环工作台
> 对应论文层级:实验方案的**任务层**(`docs/experiments/openloop_sparse_observation_validation_plan.md` §5 T1/T2/T3)
> 上游功能清单:[`real_robot_validation_workbench_todo.md`](../../experiments/real_robot_validation_workbench_todo.md)(本 spec 是其中"任务层 IK 闭环"的具体实现设计)

---

## 1. 目标与非目标

### 1.1 要实现的能力

1. 实时从单相机图像提取软臂 15 节点 2D 骨架,作为模型状态锚点
2. 在图像上交互指定:末端目标点/区域、**全身目标形态**、keep-out 障碍
3. 在 GUI 中一键规划(OpenLoop shooting)、预览预测轨迹、preflight、真机执行
4. 执行期同步记录隐藏真值(相机 + NDI),自动算 prediction-to-execution gap

### 1.2 明确的非目标(第一版不做)

- 3D 形态 / 多相机三角化(当前 state 定义为 2D 像素,`z ≡ 0`)
- 六腔道双段协同(模型 `action_dim=1`;先把链路验证完再加维度)
- 接触力控制、动态环境、移动障碍
- 学习型逆网络 / diffusion IK(主方法是可微模型上的轨迹优化)
- 实时 MPC 级重规划(先做"执行 N 步 → 重锚定 → 重规划"的滚动窗口)

### 1.3 已确认的前置决定

| 决定 | 取值 | 影响 |
|---|---|---|
| 相机位姿 | 不确定 → **重新配置硬件后重采数据重训** | 消除 homography warp 需求,只保留漂移检测 |
| 第一版目标类型 | 末端到达 + 全身形态**两种都建**,首次真机只跑末端 | Scene/loss 一次建到位,风险逐步暴露 |
| 运行环境 | 真机 PC **有 NVIDIA GPU** | 本地规划,保留"整目录拷走"可移植设计 |
| 首跑安全包线 | `0–150 kPa` 上界 + `100 kPa/s` 速率,首跑可临时降到 60–80 kPa | 对齐训练上界不越域;速率比训练窄(是训练分布子集) |
| 代码共享 | **单一实现 + 离线改薄壳** | 在线==离线由构造保证 |

---

## 2. 现状盘点(设计的事实基础)

### 2.1 三条链的完成度

```
① 感知链  相机取流 → mask → 骨架15点 → 归一化 state      real_validation 里 0 行
② 意图链  图上点选 → 目标/障碍 → planner 约束             断在 pixel↔model;目标只支持单节点圆
③ 执行链  plan → preflight → 真阀 → 记录 → 评价           已实现 ~90%,差接线 + 若干 bug
```

### 2.2 已有且直接复用(不重写)

| 能力 | 位置 |
|---|---|
| 5 类 frozen dataclass 数据契约 + 四重 stale 绑定(model_hash / scene_digest / anchor_id / safety_digest) | `models.py` |
| 11 态状态机 + run 目录 + experiment.json 快照 | `session.py` |
| preflight 12 项纯函数 | `preflight.py` |
| 绝对时基 ACK 执行器 + 失败即归零 + zero-pause 强制重规划 | `executor.py` |
| Qt 真阀 transport(真实现,与 `CommandTransport` 协议兼容) | `hardware_session.py` |
| 自包含 OpenLoop 前向 + GRU key 迁移 + 严格加载(已核对与 `src/` 逐块数值等价) | `runtime/` |
| **图内可微**压力/速率投影(比 CLI 强:CLI 只有图外 clamp、无速率约束) | `openloop_planner._project_actions` |
| 计划预览(6 通道动作曲线 + 逐 k 骨架,坐标约定已与 `viz_control` 一致) | `widgets/plan_preview.py` |
| 22 个单元测试 | `tests/test_real_validation_core.py` |

### 2.3 已实现但零接线的孤儿模块(必须接上)

`metrics.py`、`observation_policy.py`(含 `ActionHistoryBuffer`)、`coordinate_system.py`、`validation_recorder.py` —— 四个文件都是真实现,但只被单测引用,生产路径零调用。

### 2.4 完全缺失(要新写)

相机取流、相机预览控件、在线分割、在线骨架化 + tip_fix、实时锚定、**图上点击交互**、pixel↔model 校验、NDI 接入、Scene 增删改、**全身形态目标**、非圆障碍、auto-K、从文件读 K_safe、re-anchor / 滚动重规划、结果页指标、warmup 冷启动。

### 2.5 必须修的既有缺陷

| # | 缺陷 | 证据 | 后果 |
|---|---|---|---|
| **B1** | **动作单位错**:`safety.pressure_min6/max6` 是 kPa(默认 max=200),`_project_actions` 在 kPa 空间投影,随后 `normalized = physical / norm`,而 `norm = action_norm_factor ≈ 1.0`。模型要的是 `kPa/150` | `openloop_planner.py:168-204`;`action_norm_factor` 是**第二次**归一化(≈0.99999,no-op),不是 kPa 换算 | 接真硬件设 0–150 kPa 那一刻,模型输入超训练域 150×,规划出垃圾动作而 preflight 全绿 |
| **B2** | `offline_anchor` 把已是 `[0,1]` 的 npz actions 标成 `action_units="kpa"` | `offline_anchor.py:59` | 标注是假的;两个错误互相抵消才没炸 |
| **B3** | `[0,1]↔kPa` 换算因子(`hi6[0]=150`)**不在 checkpoint、不在 npz、不在 config.json**,只在 `real_capture/data/raw/<seq>/meta.json` | — | 部署时无从获得 |
| **B4** | 障碍惩罚聚合口径:CLI 对 k **求和**,工作台对 (K,N) **求均值** | `inverse_plan.py:117-120` vs `openloop_planner.py:222-226` | 同一 `w_obs` 实际避障压强差 ≈K 倍,CLI/GUI 不可比 |
| **B5** | `step_interval_s` GUI 默认 0.2,训练实测 Δt = **0.203125 s**;preflight 不校验 | `frame_times.txt` 实测(0.187–0.219 量化,均值 0.2031) | 动力学时基不一致且无告警 |
| **B6** | `execution.csv` 无"期望 t vs 实际 t"列;ACK 等待超 `step_interval_s` 时连续补发无告警 | `executor.py:98-154` | prediction-to-execution gap 无法归因 |
| **B7** | `Scene.with_primitive` 只能追加,无删除/替换;`_set_target` 靠"清掉所有旧 target"实现替换 | `models.py:150`;`main_validation.py:374-377` | 交互式编辑无法实现 |
| **B8** | GUI 执行中不锁页 1/2/3,点"设置目标"会因非法状态转移抛 RuntimeError | `main_validation.py:582-590` 只 enable/disable 4 个按钮 | 运行时崩 |
| **B9** | `K_safe` 靠手填 QSpinBox;`horizon_summary.json` 里有现成认证数字但无人读 | `main_validation.py:145-146,314-316` | 人为错填即失去唯一安全门 |
| **B10** | 分数阶 GL 权重每次 forward 在计算图内用 Python 循环重建(`n_orders×window = 160` 标量 op) | `runtime/model.py:31-36` | 规划 = `n_restarts×n_iter×K` = 32,000 次 forward;**全仓无任何耗时记录** |
| **B11** | `checkpoints/current/best_model.pt` 不存在,GUI 默认路径指向空文件 | `checkpoints/current/` 只有 config.json | 开箱即崩 |
| **B12** | `SessionState.REANCHOR` 已定义但**无代码进入**;`ObservationPolicy.require_allowed` 无调用点 | `session.py:22,36-41` | 滚动重锚定不可用;观测隔离只是审计不是强制 |
| **B13** | `tip_fix` 的三个门控是**静默跳过**,不产生"tip 未修"的可观测信号 | `src/utils/skeleton_2d.py:31-45` | 在线时 node0 可能落在 cap 角落而调用方无从得知 |

### 2.6 模型侧的在线约束(不可协商的物理事实)

| 约束 | 依据 |
|---|---|
| 需要 `s_{t-1}`、`s_{t-2}`(速度项)、`z_{t-1}`、**最近 40 个真实动作** | `model_state_transition.py:209-280` |
| 动作窗口 `(B,40,1)`,**最新在 index -1**;训练时 padding 分支**永不触发**(`start_t ≥ 39`)→ 模型从没见过零填充窗口 | `dataset_spatial.py:151-158,227` |
| 分数阶 GL 核 `w_0` 幅值最大,乘的是窗口**最旧**那格 → 零填充落在权重最大位置 | `fractional_memory.py:74-94,116` |
| `z` 无 GT、无传感器,**物理上无法凭空构造**;接管瞬间只能 `init_z_from_action` 重置,该误差不可消除 | `model_state_transition.py:198-207` |
| `delta_scale_max` 是普通 float,**不在 checkpoint**,只靠构造出 `OpenLoopTransitionModel` 类才恢复为 1.0 | `model_state_transition.py:175`;`model_open_loop_transition.py:84` |
| `pc_scale[2] = 1e-6` → 模型**结构性地无法输出平面外形变** | `dataset_spatial.py:139-140` |

**冷启动结论**:必须先用训练分布内的动作跑 ≥40 步(≈8.1 s)积累真实历史,才允许锚定。这段时间预测不可信。

---

## 3. 架构

### 3.1 代码共享规则

**谁是部署产物,谁持有唯一实现。**

- **感知**的部署产物是工作台 → 实现放 `real_validation/perception/`,`src/` 改薄壳
- **硬件**的部署产物是已在真机上用的采集程序 → 实现留在 `real_capture/`,工作台加适配层

部署方式:把 `real_capture/` 与 `real_validation/` **并排**拷到 PC。两个"整目录拷走"契约都不破。

依赖方向:`real_validation` 不 import `src/`(保持契约);`src/` 反向 import `real_validation.perception`(它是部署产物,感知是训练与部署的共享契约,方向倒置是对的)。

### 3.2 目录结构

```
real_validation/
├── perception/                    ★全新 —— 感知的唯一实现
│   ├── segmentation.py            ← 从 src/data/real/segmentation.py 移入
│   ├── skeleton.py                ← 从 src/utils/skeleton_2d.py 移入(+ 暴露 tip_fix 是否生效)
│   ├── background.py              中值背景:加载 / 重建 / 与基准帧比对
│   ├── registration.py            ★基准帧 ORB+RANSAC + 残差(只做检测,不 warp)
│   ├── quality.py                 ★在线帧质量门控(§6.1)
│   └── live_anchor.py             ★frame → mask → 骨架 → [col,row,0] → 归一化 → Anchor
├── hardware/                      ★全新 —— 适配层,不复制驱动
│   ├── _bootstrap.py              sys.path.append(兄弟目录 real_capture/);用 append 不用 insert 以免遮蔽 stdlib
│   ├── valve.py                   包装 ValveController;**kPa 单位边界在这里收口**
│   ├── camera.py                  包装 RealSenseCam;断言分辨率/序列号与 manifest 一致
│   └── ndi.py                     包装 NdiThread;标记为隐藏评价流,禁止进模型
├── widgets/
│   ├── plan_preview.py            已有,保留
│   ├── camera_view.py             ★pg.ImageItem 图像层 + 骨架/目标/障碍/门区图层 + 鼠标点选拖拽
│   ├── scene_editor.py            ★primitive 列表:增删改、锁定、坐标系标签、Undo
│   └── results_view.py            ★指标表 + error-by-k 曲线 + 预测vs实测叠加
├── warmup.py                      ★冷启动:训练分布内 ≥window_size 步预热 + 就绪判据
├── calibration/
│   ├── reference_frame.png        ★训练期基准帧
│   ├── registration.json          ★配准结果 + 残差 px + 时间戳 + 基准帧 hash
│   ├── segment_params.json        ★分割参数指纹(注意实际产物用 val=100,不是 CLI 默认 120)
│   └── px_to_mm.json              ★NDI↔px 仿射 + 残差(仅评价,不进控制)
├── checkpoints/current/
│   ├── best_model.pt              ★补上(现在缺失)
│   ├── config.json                已有
│   └── deploy_manifest.json       ★新:部署契约(§4.1)
├── models.py            改:契约变更(§4)
├── openloop_planner.py  改:修 B1/B4、+全身骨架损失、+AABB/多边形障碍、+auto_k、GL 权重缓存、多起点批并行、记耗时
├── preflight.py         改:+Δt / action_scale / 配准残差 / anchor 新鲜度 / warmup 就绪 / mask_source
├── executor.py          改:+jitter 列、接 ObservationPolicy、接 ActionHistoryBuffer、归零二次重试
├── session.py           改:接通 REANCHOR、执行中锁页
├── metrics.py / validation_recorder.py / coordinate_system.py   改:接线
└── main_validation.py   改:五页填实

src/utils/skeleton_2d.py          → 薄壳 re-export(签名不变)
src/data/real/segmentation.py     → 薄壳 re-export(签名不变)
real_capture/                     → 完全不动
```

`real_validation/requirements.txt` 补 `opencv-python`、`scipy`。

### 3.3 坐标系(严格四分,枚举 + 白名单校验)

| frame_id | 含义 | 谁产生 | 谁消费 |
|---|---|---|---|
| `camera_pixel` | live 相机原始像素 | 相机 / 鼠标点击 | **只有 registration** |
| `model` | 训练期相机像素 `[col,row]` | registration 校验通过后直接等同 | planner / metrics / 显示 |
| `model_normalized` | `(model - pc_center)/pc_scale` | `live_anchor` | 模型前向 |
| `ndi_mm` | NDI 毫米 | NDI | **只进评价,永不进控制** |

因为重采后采集位姿 == 部署位姿,`camera_pixel → model` 是**恒等映射 + 一个残差门**,不是 warp。`registration.py` 的职责是**证明这个恒等映射仍成立**,残差超阈值即全局阻断。

任何未经 registration 校验的 `camera_pixel` 对象禁止进入 loss。`frame_id` 从自由字符串改为枚举 + 白名单。

---

## 4. 数据契约变更

### 4.1 新增 `deploy_manifest.json`(与 checkpoint 同目录)

修 B3 的核心:把部署所需的全部隐式知识显式化。由数据准备/训练流程生成(M1/M2),工作台只读。

```json
{
  "schema_version": 1,
  "checkpoint_sha256": "<hex>",
  "action_scale_kpa": [150.0],
  "channel_map": [0],
  "train_dt_s": 0.203125,
  "train_dt_std_s": 0.011,
  "mask_source": "white_on_blue",
  "segment_params": {"sat": 100, "val": 100, "diff": 25, "dil": 35,
                     "open_k": 5, "close_k": 15,
                     "min_area_frac": 0.003, "min_h_frac": 0.15},
  "camera": {"serial": "<serial>", "width": 640, "height": 480, "fps": 30},
  "reference_frame": "calibration/reference_frame.png",
  "reference_frame_sha256": "<hex>",
  "mask_area_median_px": 8562,
  "registration_residual_max_px": 2.0,
  "k_safe_table": {"3.0mm": 41, "5.0mm": 51, "10.0mm": 124},
  "train_sequences": ["seq_YYYYmmdd_HHMMSS"],
  "n_nodes": 15, "window_size": 40, "z_dim": 16, "episode_len": 40
}
```

### 4.2 `ModelDescriptor` 新增字段

```python
action_scale_kpa: tuple[float, ...]   # 每模型维的 kPa 上界
train_dt_s: float                     # 训练采样周期实测均值
train_dt_std_s: float
mask_source: str                      # 在线只允许匹配的源
segment_params: dict
camera_fingerprint: dict              # serial / width / height / fps
reference_frame_hash: str
k_safe_table: dict[str, int]          # 容差 → 步数;k_safe 由选定容差决定
```

### 4.3 `ScenePrimitive` 新增真正的消费者

现有 12 个 `kind` 只有 3 个有实现(`target_point` / `target_circle` / `obstacle_circle`),其余是"名字白名单"。新增:

| kind | geometry | 消费者 |
|---|---|---|
| `target_skeleton` | `{"nodes": [[x,y] × N], "weights": null 或 [w × N], "tolerance_px": float}` | planner 全身损失、metrics、预览 |
| `obstacle_aabb` | `{"min": [x,y], "max": [x,y]}` | planner 2D SDF、metrics 间距、预览 |
| `obstacle_polygon` | `{"points": [[x,y], ...]}` | 同上 |

`geometry` 增加 schema 校验(现在是自由 dict,各处靠 `xy`/`center`、`radius`/`r` 双键容错)。统一键名并保留旧键读取以向后兼容。

### 4.4 `Anchor` 变更

```python
quality: dict     # float → 标志集
# {"mask_area_ratio": 0.98, "top_row": 12, "second_blob_ratio": 0.01,
#  "tip_fix_applied": true, "frame_age_s": 0.03,
#  "registration_residual_px": 0.8, "max_node_step_px": 2.1,
#  "verdict": "ok" | "degraded" | "reject"}
frame_ref: str                                    # ★新:隐藏评价流中的帧引用(事后对齐用)
prev_state: tuple[tuple[float, ...], ...] | None  # ★新:s_{t-2},速度项需要
action_units: "kpa" | "model_normalized"          # "kpa" 必须真是 kPa;npz 来源改标 model_normalized(修 B2)
```

`prev_state` 是新增的必要字段:模型 forward 需要 `prev_prev_skeleton` 算速度,现在 `Anchor` 只带一个 state,`plan_rollout` 用 `s_prev = s_init` 起步(等价 v=0),那是训练分布外。

### 4.5 `SafetyPolicy` 变更

`pressure_max6` 默认 `200.0 → 150.0`(对齐训练上界,避免规划走出训练域)。

### 4.6 单位收口(修 B1)

```
kPa  --/action_scale_kpa[i]-->  [0,1] 训练域  --/norm_factor-->  模型输入
```

只允许出现在两处:`hardware/valve.py`(硬件边界)与 `openloop_planner`(优化边界)。其余任何地方禁止手写。配往返单测锁死(§7.2 T2)。

---

## 5. 数据流

```
[Setup]
  载 checkpoint + deploy_manifest
  连硬件 → 断言 camera_fingerprint 匹配(序列号/分辨率/fps)
  载 reference_frame → ORB+RANSAC 配准 → 残差 px → 写 registration.json
    残差 > 阈值 → 全局阻断规划与执行
  安全配置:0–150 kPa / 100 kPa/s(首跑可临时降到 60–80 kPa)
  K_safe:从 manifest.k_safe_table 按选定容差取值(不再手填)

[Warmup]  ★新
  归零 → 回放**训练序列中的一段真实动作**(≥window_size=40 步)@ train_dt
    默认:从 train npz 随机截一段(保证分布内);可选:0→0.8×scale 慢速三角波
  ActionHistoryBuffer 累积 applied 值(不是 requested)
  未满 40 步 → 禁止 Anchor,UI 显示进度与剩余秒数

[Observe & Anchor]
  取连续两帧(间隔必须 ≈ train_dt ± 容差)
  逐帧:质量门控 → mask → 骨架 15 点 → [col,row,0] → 归一化
  Anchor = {state(s_{t-1}), prev_state(s_{t-2}), action_history(40, applied),
            quality 标志集, frame_ref, timestamp}
  ObservationPolicy.decide(force=True, reason="operator_anchor")
  z:接管时用 init_z_from_action(动作窗口) —— 不可消除的一次重置误差,UI 标注

[Scene]
  camera_view 上:
    点击 → target_point;拖拽半径 → target_circle
    "取当前观测形态为目标" / "从录制序列选一帧" → target_skeleton
    画圆/矩形/多边形 → obstacle_*,带保守 safety_margin
  scene_editor:列表增删改锁定、Undo、scene.json 存取
  所有 primitive 携带 frame_id;camera_pixel 经 registration 校验后等同 model

[Plan]
  ShootingConfig:horizon / n_iter / n_restarts / lr / w_path / w_mono / w_smooth / w_obs 全部 GUI 暴露
  auto_k(从 CLI 移植):K = clamp(ceil(gap_tip_px / step_budget_px), k_min, k_max)
    step_budget_px 绑定 delta_scale_max(1.0) × pc_scale
  K ≤ K_safe 硬门
  损失 = terminal + w_path·path + w_mono·mono + w_smooth·smooth + w_obs·obstacle
    target_point / circle:单节点 relu(dist - radius)²
    target_skeleton:     全节点加权平均 relu(dist - tolerance)²
    obstacle:            全节点对所有障碍的 2D SDF;**聚合口径与 CLI 统一**(修 B4)
  动作在 kPa 空间图内可微投影(clamp + rise/fall 递推) → /action_scale_kpa/norm_factor → 模型
  后台线程 + cancel_event + 进度 + 候选排序 + **记录耗时**
  输出 ActionPlan(四重 hash 绑定) + predicted_states.npz(states_normalized + states_model)

[Preflight]
  现有 12 项 + 新增:
    dt_mismatch      |step_interval_s - train_dt_s| / train_dt_s < 5%
    action_scale     manifest.action_scale_kpa 与 descriptor 一致
    registration     残差 < manifest.registration_residual_max_px(默认 2.0 px ≈ 0.6 mm)
                     且 registration.json 与当前会话同源(未过期)
    anchor_stale     anchor 年龄 ≤ 5 步(≈1 s),且期间未发生归零 / Abort / 模型切换
    warmup_ready     ActionHistoryBuffer.ready
    mask_source      在线分割源 == manifest.mask_source
  任一失败 → 不允许 Arm

[Execute]
  Arm → Confirm → 三档执行(逐步 / 低速 dry-run / 整段)
  每步:等 deadline(绝对时基) → set_pressures → 等 ACK → 记录
    requested / applied / command_id / t_command / t_ack / **t_expected / t_actual / jitter_s**
    ActionHistoryBuffer.append_applied6(applied)      ← 接线
    ObservationPolicy:执行期相机/NDI 继续录但 allowed=False(隐藏评价流)
  Pause(zero → 强制重规划;hold → 保持) / Abort → 全部归零(失败二次重试 + ERROR 态)
  执行 N 步 → REANCHOR:重观测 → 更新 anchor + 动作历史 → 重规划下一窗口

[Results]
  对齐 predicted_states[k] ↔ 隐藏 GT 骨架[k](按 frame_ref + t_actual)
  指标:末端误差 px/mm、全身 MNE/p90/max、error-by-k、最小障碍间距、碰撞次数、
        压力/速率违反、jitter 统计、观测次数、最长不可见时间、重规划次数
  prediction-to-execution gap 曲线
  一键导出 CSV/JSON;run 目录可离线 replay 重算(结果须一致)
```

### 5.1 run 目录(按 todo §C8,补齐 jitter 与 observation)

```
runs/run_YYYYmmdd_HHMMSS/
  experiment.json  model.json  deploy_manifest.json  registration.json
  scene.json  safety.json  anchor.npz
  plan.json  planned_actions6.csv  predicted_states.npz
  execution.csv          # + t_expected, t_actual, jitter_s
  observations.csv       # 每帧:allowed / reason / quality / frame_ref
  hidden_ground_truth/   # camN/*.png + ndi.csv(执行期全程)
  metrics.json  summary.md  events.jsonl
```

---

## 6. 错误处理与诚实边界

### 6.1 在线质量门控

离线那套"坏帧时间插值修复"在线**全部不可用**(需要未来帧)。在线只能**拒帧**。

| 判据 | 阈值 | 来源 | verdict |
|---|---|---|---|
| mask 最大连通区面积 / 采集期中位 | <0.7 或 >1.3 | manifest.mask_area_median_px | reject |
| 最大连通区 height / H | <0.15 | 与离线同 | reject |
| top_row(臂是否连到基座) | >20 | 与离线同 | reject(手入画 / 臂缺失) |
| 次大连通区面积 / 最大 | >0.15 | ★新 | degraded |
| `tip_fix` 是否被静默跳过 | 被跳过 | ★新(修 B13:`skeleton.py` 需返回该标志) | degraded |
| 与上一被接受帧的最大节点位移 | > `delta_scale_max × pc_scale × 余量` ≈ 4 px/步 | ★新 | reject |
| frame_age | >0.5 s | 与采集同 | reject |
| 配准残差 | > `registration_residual_max_px`(默认 2.0 px) | registration | **全局阻断** |

- `reject` 帧:不进模型、不更新 anchor,**但仍写入隐藏评价流**
- **连续 5 帧**(≈1 s @ 5 fps)`reject` → 阻断继续执行,要求人工介入
- `degraded` 帧:允许锚定但在 UI 与 run 记录里标记,metrics 分层报告

### 6.2 执行期

- ACK 超时 / queue full / 串口错误 → 立即归零 + 中止(已有)
- **归零本身失败**:现在会覆盖原异常并跳过后续处理 → 改为二次重试 + 显式 `ERROR` 态 + 醒目告警
- jitter > `train_dt × 20%` → 记录 + 警告;> 50% → **中止**(动力学时基假设已破)
- 看门狗:GUI 线程 / 执行线程 / 硬件通信 三路心跳
- 关窗:停执行 → 归零 → 等队列 → 释放硬件(已有)

### 6.3 必须在 UI 上明示的诚实边界(不能只写文档)

| 边界 | UI 要求 |
|---|---|
| 障碍是**图像平面近似**(NDI 实测平面外跨度 4.35 mm vs 平面内 24.2 mm,约 15–18%) | 障碍标签必须写 `planar approx, margin=X mm`;**禁止**显示"3D 无碰撞" |
| 阀无压力反馈 | 一律称 `command` / `applied command`;**禁止**写"实测气压" |
| warmup 期间预测不可信 | 预览与执行按钮灰掉,显示剩余步数 / 秒数 |
| 开环视野有限 | `k_since_observation / K_safe` 常驻显示;超限阻断自动执行 |
| planner 可能钻模型盲区 | 执行后必须报 prediction-to-execution gap;超容差报"**不可达**",不硬称成功 |
| z 接管重置 | 锚定时提示"迟滞潜变量已重置,首窗口精度略降" |

---

## 7. 测试策略

### 7.1 保留

现有 22 个单测全部保留并保持通过。

### 7.2 新增

| # | 测试 | 断言 |
|---|---|---|
| T1 | **parity**:固定 50 张 mask,`real_validation.perception.skeleton` vs 迁移前 `src/utils/skeleton_2d.py` 的行为快照 | 逐点 px 差 == 0;`segmentation` 逐像素 == 0 |
| T2 | **单位往返** | `kPa → model → kPa` 误差 < 1e-6;且 `safety.max=150 kPa` 时模型输入 ≤ 1.0(**直接锁 B1**) |
| T3 | **坐标往返** | `camera_pixel → model → camera_pixel` 误差 < 阈值(接上已有 `PlanarTransform.roundtrip_error`) |
| T4 | **CLI/GUI 一致性** | 同 anchor / scene / seed 下 `inverse_plan.py` 与工作台 planner 结果一致(前提:先统一 B4 的障碍聚合与损失定义) |
| T5 | **Mock 错误注入** | ACK timeout / queue full / 串口错 / 坏帧 / 配准漂移 / warmup 未满 → 全部安全转移并归零 |
| T6 | **回放确定性** | run 目录离线 replay 重算指标 == 在线记录 |
| T7 | **rollout 等价** | `runtime/rollout.plan_rollout` 与 `src/` 侧 rollout 同输入逐元素相等 |
| T8 | **契约拒绝** | 错 action_dim / 缺 manifest / mask_source 不匹配 / K > K_safe → 阻断规划与执行 |

---

## 8. 采集协议(因重采而新增)

重采是消除训练/部署差异的机会。协议本身必须**为部署设计**,否则会重复踩同样的坑。

1. **mask 源 = 在线可复现**:主模型训练用 `white_on_blue` 单帧阈值 mask。SAM2 仅作离线 QC 对照,**不训主模型**
   - 依据:SAM2 是分块 200 帧 + **双向**传播、锚帧还来自启发式修复,根本不可实时
   - 参考量级:同一 80 px 离群判据下,离群帧 raw 98 / repaired 32 / SAM2 3(共 10214 帧)→ 采集时须主动压低坏帧率(固定背景、避免手入画、稳定光照)
2. **固定并记录一切**:相机序列号/分辨率/曝光锁定;拍 ≥20 帧背景存基准帧;分割参数固定并写 manifest;**记录实测 Δt 均值与标准差**
3. **清洗策略对齐在线能力**:只用逐帧可判定的门控。时间插值**仅用于保持时序连续**(状态转移模型必须连续帧),被插值的帧打标记并在 loss 里降权/剔除,不让模型学插值产物
4. **多条独立序列**(≥3–5 条),按**序列**划分 train/val/test(不按帧),覆盖 loading / unloading / hold / 动作反转 / 变速
5. **先 1-DOF(ch0)**:链路未验证前不加维度(与实验方案 §7 一致)
6. **序列即目标库**:目标形态取自录制帧 → 保证物理可达
7. **障碍物**:虚拟障碍先做;真机避障验证需一个**在臂运动平面内**的物理障碍,采集时拍一段"障碍在场景内但臂不碰它"的参考帧供 scene 标定
8. **NDI 全程录**:末端 mm 真值 + px→mm 仿射重拟合(写 `calibration/px_to_mm.json`)

---

## 9. 里程碑

| 里程碑 | 内容 | 验收产物 |
|---|---|---|
| **M0** | **感知迁移**(`src/data/real/segmentation.py` + `src/utils/skeleton_2d.py` → `real_validation/perception/`,`src/` 改薄壳,修 B13 暴露 tip_fix 标志)+ T1 parity 测试 + 命令行感知探针(无 GUI):抓帧 → 分割 → 骨架 → 叠加图 + 配准残差 + 逐算子耗时 | T1 全绿;叠加图、残差数字、逐算子耗时表 |
| **M1** | 按 §8 协议重采集 + 生成 npz + `deploy_manifest.json` | 新 seq;离群帧率 < 1%;门控通过率报告 |
| **M2** | 重训 gt + open_loop + 视野认证 | `k_safe_table`(容差→步数);error-by-k 曲线 |
| **M3** | 契约与单位修复(B1–B5、B7–B11)+ T2/T3/T4/T7/T8 测试 | 测试全绿;CLI/GUI 同输入同结果;规划耗时基准 |
| **M4** | `camera_view` + `scene_editor` + 实时锚定 + warmup | 点目标 → 规划 → 预测骨架动画;Mock 执行通过 |
| **M5** | 真机安全执行 + 同步隐藏评价(jitter、observation policy、录制、metrics 接线、B6/B12) | 低压小幅末端到达真机成功 + prediction-to-execution gap 曲线 |
| **M6** | 全身形态目标 + 非圆障碍 + REANCHOR 滚动窗口 | 给目标形态 + 真实障碍 → 真机执行 → 全身误差与最小间距报告 |

**M0 先于一切**:它不需要 GUI,却把在线感知链每个算子都跑通,并给出 M1 采集协议所需的全部参数(实际分割参数、单帧耗时、坏帧率)。

依赖:`M0 → M1 → M2 → M3 → M4 → M5 → M6`。**M3 可与 M1/M2 并行**(只改代码,不依赖新数据)。

---

## 10. 性能

规划一次 = `n_restarts × n_iter × K` 次模型前向(GUI 默认 4×400×20 = **32,000**)。**全仓没有任何耗时记录**,M3 必须先测基准。

已识别的优化(纯收益,不改数值):

1. **缓存分数阶 GL 权重**(修 B10):规划期 `alpha` 冻结,权重可在 `torch.no_grad()` 下预计算一次并作为常量;对动作的梯度完全不受影响。预期 2–5×
2. **多起点批并行**:现在 `n_restarts` 是串行 for 循环;可 batch 到 `(n_restarts, ...)` 一次前向。预期接近 `n_restarts` 倍
3. **降 n_iter**:先测收敛曲线,400 可能过量

瓶颈性质:模型只有 241,260 参数、batch=1、K≤40 → FLOPs 与显存都极小,**瓶颈是 `K × O(200)` 个微 kernel 的启动延迟 × n_iter × n_restarts**。杠杆在批并行与减少 kernel 数,不在省显存。

---

## 11. 开放风险

| 风险 | 处理 |
|---|---|
| 模型-现实鸿沟:planner 优化出的动作可能在真机复现不出预测形状 | 三层防线:① 动作 clamp 到训练真实执行范围;② 目标取自录制帧(保证可达);③ 执行后报 gap,超容差报"不可达"不硬称成功。这是**核心开放问题**,学习模型无法单方面保证 |
| 在线 `z` 重置误差不可消除 | 承认并标注;首窗口精度略降。长期解法是 episode 序列训练让 z 跨帧演化 |
| `white_on_blue` 阈值分割对光照敏感 | 采集与部署锁定曝光;质量门控兜底;背景基准帧漂移检测 |
| 2D 平面障碍近似 | 保守膨胀 margin + UI 明示;真机障碍放在臂运动平面内 |
| 重采后需重新拟合 NDI px→mm 仿射 | 纳入 M1 验收 |
| `real_capture` 加入 `sys.path` 的模块名冲突 | 用 `sys.path.append` 而非 `insert`;当前两目录无同名模块(`recorder` vs `validation_recorder`),新增文件时须检查 |

---

## 12. 实现计划的拆分

本 spec 覆盖 7 个里程碑,**跨度过大不适合单一实现计划**。按"可独立验收、依赖清晰"拆成 4 个计划,每个计划单独走 writing-plans:

| 计划 | 覆盖 | 前置条件 | 为什么是一个独立单元 |
|---|---|---|---|
| **P1**(先做) | **M0 + M3** | 无 —— 纯代码,不需要硬件、不需要新数据 | 感知迁移 + 契约/单位修复 + 全部可离线跑的测试。做完之后地基是对的,且 CLI/GUI 结果一致。这是唯一现在就能完整验收的部分 |
| **P2** | **M1 + M2** | 硬件重新配置完毕;P1 的 `perception/` 已就位(采集要用同一套分割参数) | 数据采集协议 + 训练 + 视野认证。产物是新 checkpoint + `deploy_manifest.json` |
| **P3** | **M4** | P1 + P2 | GUI 感知/场景/锚定层。可用 P2 的 checkpoint 在 Mock 下完整验收 |
| **P4** | **M5 + M6** | P1 + P2 + P3 | 真机执行 + 隐藏评价 + 全身形态/避障/重锚定 |

P1 与 P2 的硬件准备可并行:P2 的硬件配置和 P1 的编码互不阻塞,但 P2 的**采集**必须等 P1 的 `perception/` 定稿(否则采集用的分割参数与在线不一致,就白费了这次重采)。

**下一步:为 P1 写实现计划。**
