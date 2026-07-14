# SelfSoftRobot 项目帮助文档

本文档说明每个文件的作用、如何使用，以及每个模型从数据采集到训练评估的完整运行流程。

---

## 1. 项目整体逻辑

```
PyElastica 物理仿真 → PyVista 渲染图像 → 数据采集 (.npz) → 模型训练 → 形态预测 + 3D 评估
```

项目目标：**仅用驱动参数（扭矩）预测软体机器人的完整 3D 形态**。

当前有十套模型管线，全部通过 Spec 声明式训练架构统一管理：

| 管线 | 模型 | 核心创新 | 监督模式 | 数据需求 | 输出 |
|------|------|---------|---------|---------|------|
| C-MSTNF 系列 | MSTNF / C-MSTNF | 多尺度时序编码 + D-NeRF 变形场 | rendering | 2D 图像 + 动作 | 2D 渲染图 |
| MS-SCNF | MS-SCNF | 显式骨架回归 + 骨架条件密度场 | skeleton→rendering | 2D 图像 + 动作 + 3D 节点坐标 | 3D 骨架 + 2D 渲染图 |
| SDF 3D | TemporalSDF | SIREN 坐标编码 + 3D SDF 直接监督 | direct_3d | 3D 节点坐标 + 动作（无需图像） | 3D SDF 场 |
| SkeletonSDF | SkeletonSDF | 参数化骨架 + 管状 SDF 先验 + SIREN 残差 | direct_3d | 3D 节点坐标 + 动作（无需图像） | 3D SDF 场 + 3D 骨架 |
| 多视角+深度 | MSTNF / C-MSTNF (多视角训练) | 多视角 rendering + 深度监督融合 | rendering (MultiView) | 多视角 2D 图像 + 深度图 + 相机参数 | 多视角 2D 渲染图 |
| **分数阶记忆** | FractionalMemory | 幂律记忆核替代指数衰减 EMA | —（编码器，非独立模型） | 同宿主模型 | 物理状态向量 |
| **空间序列** | SpatialSequence | GRU 沿 Z 轴空间传播 + 分数阶记忆 | spatial_sequence | 3D 节点坐标 + 动作 | 3D 中心线 (31 节点) |
| **预测-修正** | PCSpatial | 预测(驱动历史) + 修正(视觉观测) 两阶段 | spatial_sequence | 3D 节点坐标 + 动作 + 图像（Phase 2） | 3D 中心线 (31 节点) |
| **闭环状态转移** | StateTransition | 前一步状态 + 可学习迟滞潜变量 z，学转移不学状态（自回归 rollout） | spatial_sequence (单帧) | 3D 节点坐标 + 动作 | 3D 中心线 (31 节点) |
| **全 GT 驱动转移（主线）** | GTObservedTransition | 前一状态恒真实 + z 窗口演化 + dense supervision | spatial_sequence (窗口) | 3D 节点坐标 + 动作 | 3D 中心线 (31 节点) |

---

## 2. 目录结构

```
SelfSoftRobot/
├── elastica_env.py                  # PyElastica 仿真环境 + 渲染
│
├── config/                          # 配置文件（根目录）
│   ├── training.json                #   训练超参数（所有模型共享）
│   ├── camera.json                  #   相机参数
│   ├── simulation.json              #   仿真参数
│   └── params.py                    #   YAML 配置加载工具
│
├── src/
│   ├── encoders/                    # 时序编码器（从 models/ 提取）
│   │   ├── multi_scale_ema.py       #   MultiScaleEMA（多尺度指数移动平均）
│   │   ├── fractional_memory.py     #   FractionalMemory（分数阶幂律记忆编码器）
│   │   └── gamma_laguerre.py        #   GammaLaguerreMemory（Gamma 延迟峰记忆核）
│   ├── fields/                      # 神经场模块（从 models/ 提取）
│   │   ├── canonical.py             #   CanonicalField（规范场）
│   │   ├── deformation.py           #   DeformationField（变形场）
│   │   └── skeleton_density.py      #   SkeletonConditionedDensity（骨架条件密度场）
│   ├── heads/                       # 回归头（从 models/ 提取）
│   │   └── skeleton_heads.py        #   骨架回归头（point/fourier/bspline/catmullrom）
│   ├── rendering/                   # 渲染策略（从 training/ 提取）
│   │   └── view_strategy.py         #   ViewStrategy（单视角/多视角策略）
│   ├── evaluation/                  # 评估工具
│   │   ├── query.py                 #   模型查询工具（预测骨架/SDF/密度）
│   │   └── render.py                #   可视化渲染（mesh/pointcloud/animation）
│   ├── models/                      # 模型定义
│   │   ├── layers.py                #   通用层（PositionalEncoder, MLPDecoder 等）
│   │   ├── mixins.py                #   共享 mixin 类
│   │   ├── model.py                 #   FBV_SM 基线（原始论文方法）
│   │   ├── model_mstnf.py           #   MSTNF（MultiScaleEMA 时序编码）
│   │   ├── model_cmstnf.py          #   C-MSTNF（Canonical + Deformation）
│   │   ├── model_ms_scnf.py         #   MS-SCNF（骨架条件神经场）
│   │   ├── model_skeleton_sdf.py    #   SkeletonSDF（参数化骨架 + 管状 SDF）
│   │   ├── model_sdf.py             #   TemporalSDF（SIREN + EMA 时序 SDF）
│   │   ├── model_spatial_sequence.py #  SpatialSequence（GRU 空间序列中心线）
│   │   ├── model_pc_spatial.py      #   PCSpatial（预测-修正空间序列）
│   │   ├── model_state_transition.py #  StateTransition（闭环状态转移 + 可学习潜变量 z）
│   │   └── model_gt_transition.py   #   GTObservedTransition（全 GT 驱动窗口，当前主线）
│   ├── data/
│   │   ├── dataset.py               #   SoftSequenceDataset（支持 2D/3D/深度）
│   │   ├── dataset_sdf.py           #   SDFDataset（3D SDF 监督采样）
│   │   ├── dataset_skeleton_sdf.py  #   SkeletonSDFDataset（骨架 + SDF 采样）
│   │   ├── dataset_spatial.py       #   SpatialSequenceDataset（中心线回归）
│   │   ├── dataset_pointcloud.py    #   PointCloudDataset（表面点云采样）
│   │   ├── dataset_multiview.py     #   MultiViewDataset（旧版双视角数据集）
│   │   └── dataset_multiview_depth.py # MultiViewDepthDataset（新版多视角+深度）
│   ├── training/
│   │   ├── base.py                  #   BaseTrainer（渲染、射线采样等共享工具）
│   │   ├── spec.py                  #   PhaseSpec / TrainingSpec（训练需求声明，核心）
│   │   ├── phase_strategy.py        #   PhaseStrategy（解析 spec，管理冻结/解冻/forward）
│   │   ├── dataset_factory.py       #   数据集工厂 + collate 函数（dict batch 统一）
│   │   ├── trainer_unified.py       #   UnifiedTrainer（统一训练器，支持所有模型）
│   │   ├── metrics_3d.py            #   3D 评估指标
│   │   └── rendering.py             #   旧版渲染工具
│   ├── config/
│   │   └── args.py                  #   CLI 参数定义（统一训练入口参数）
│   └── utils/
│       ├── camera.py                #   get_rays（射线生成）
│       ├── camera_system.py         #   MultiCameraSystem（多相机管理/投影/反投影）
│       ├── rendering.py             #   OM_rendering, sample_stratified, OM_rendering_with_depth, sample_depth_guided
│       ├── experiment.py            #   实验目录管理 + GIF 保存
│       ├── config_utils.py          #   CLI 参数覆盖 + 配置合并工具
│       ├── sdf_utils.py             #   GT SDF 生成工具（管状结构解析计算 + 采样）
│       ├── model_loader.py          #   自动检测模型类型并加载 checkpoint
│       ├── skeleton_2d.py           #   2D 骨架提取
│       ├── skeleton_viz.py          #   3D 骨架可视化与动画
│       └── visualization.py         #   通用可视化工具
│
├── scripts/
│   ├── data_collection/             # 数据采集
│   │   ├── collect.py               #   统一采集入口（per-dim 动作控制 + --3d + --depth + 多视角）
│   │   └── collect_utils.py         #   动作策略、保存、命名工具函数
│   ├── training/                    # 训练入口脚本
│   │   ├── train_unified.py         #   统一入口（支持全部 5 个模型，推荐）
│   │   ├── train_search.py          #   超参数网格搜索（子进程调用 train_unified.py）
│   │   ├── train_mstnf.py           #   MSTNF 薄包装（→ UnifiedTrainer）
│   │   ├── train_cmstnf.py          #   C-MSTNF 薄包装
│   │   ├── train_ms_scnf.py         #   MS-SCNF 薄包装
│   │   ├── train_sdf.py             #   TemporalSDF 薄包装
│   │   ├── train_skeleton_sdf.py    #   SkeletonSDF 薄包装
│   │   ├── train_depth_cmstnf.py    #   Depth-CMSTNF 薄包装
│   │   ├── train_multiview.py       #   多视角+深度训练薄包装
│   │   ├── train_multiview_consistency.py  # 多视角一致性薄包装
│   │   ├── train_flowmatch.py       #   FlowMatch 点云薄包装
│   │   ├── train_spatial_sequence.py #  SpatialSequence 薄包装
│   │   ├── train_pc_spatial.py      #   PCSpatial 薄包装
│   │   ├── train_open_loop_transition.py # OpenLoopTransition 窗口开环（热启动 gt_transition + tf 退火）
│   │   ├── train_gt_transition.py   #   GTObservedTransition 薄包装（主线）
│   │   ├── train_transition.py      #   ★ 实物统一入口（--mode gt|open_loop，取代旧 train_gt/open_loop_transition）
│   │       # (train_state_transition.py / _s1.py 已归档至 docs/archived/trainers/，被 train_gt/open_loop_transition 取代)
│   ├── evaluation/
│   │   ├── evaluate_3d.py           #   3D 几何评估脚本
│   │   ├── evaluate_shape.py        #   形态评估（chamfer/hausdorff/f-score）
│   │   ├── visualize_3d_shape.py    #   3D SDF/mesh 可视化
│   │   ├── visualize_predictions.py #   预测对比/动画可视化
│   │   ├── eval_rollout.py          #   StateTransition 闭环 rollout 评估（自回归，未来扩展）
│   │   ├── eval_gt_transition.py    #   GTObservedTransition 观测驱动评估（主线）
│   │   ├── eval_real_quant.py       #   ★ 实物定量评估（末端 NDI mm + 形态 px + open_loop drift_by_k）
│   │   ├── visualize_real_overlay.py #  ★ 实物：模型预测叠真实照片（原图+mask+GT+预测骨架）
│   │   └── inspect_real_data.py     #   ★ 实物：骨架数据网格诊断（9 帧 2D + 3D 预览）
│   ├── real/                        # ★ 实物数据流水线（免标定 2D，详见 §8）
│   │   ├── masks_to_transition_npz.py #  mask → 免标定 2D npz（tip_fix + 离群插值 + action 归一[0,1]）
│   │   ├── clean_transition_npz.py  #   静态段共识清洗（绝对位置锚定，修分割抖动/关节偏移）
│   │   ├── repair_masks.py          #   mask 级修复（独立轨道，逐行宽共识，产 masks_repaired/）
│   │   ├── composite_frames.py      #   批量 原图+mask+骨架 叠图（含 montage）
│   │   ├── compare_skeleton_methods.py # 7 法末端 corner 对比（独立真值+bend 分层）
│   │   ├── skeleton_to_shape.py     #   node→形态 半径偏移基线（骨架+常数半径 r→IoU 对比 mask）
│   │   ├── segment_rd.py / segment_batch.py # white_on_blue 分割（R&D / 批量）
│   │   ├── capture_to_npz.py        #   采集原始数据 → npz
│   │   ├── inspect_capture.py       #   采集数据检查
│   │   └── calibrate_cameras.py     #   多视角标定（另一条标定→3D 三角化路径，非本工作流主线）
│   ├── testing/
│   │   └── verify_unified_trainer.py #  统一训练器验证（5 模型 forward+backward）
│   ├── experiments/                 # 实验脚本
│   │   ├── exp1_skeleton_from_2d.py      # 从 2D 提取骨架
│   │   ├── exp2_pure_2d_comparison.py    # 纯 2D 方法对比
│   │   ├── exp3_multi_camera.py          # 多相机实验
│   │   ├── exp4_domain_randomization.py  # 域随机化
│   │   ├── exp5_hysteresis_analysis.py   # 迟滞分析
│   │   ├── exp6_comprehensive_report.py  # 综合报告
│   │   ├── exp7_3d_occupancy.py          # 3D 占据场实验
│   │   └── ...                           # 其他实验变体
│   └── visualization/               # 可视化工具
│       ├── view_data.py             #   数据查看
│       ├── save_gif.py              #   GIF 保存
│       ├── test_3d_seq.py           #   3D 序列测试
│       └── verify_simulation_3d.py  #   3D 仿真验证
│
├── notebooks/                       # Jupyter 验证笔记本
│
├── data/                            # 数据目录（gitignore）
│   ├── seq_zz/                      #   canonical（两维 zero）
│   ├── seq_zz_3d/                   #   canonical + 3D
│   ├── seq_rr/                      #   时序（两维 random）
│   ├── seq_rr_3d/                   #   时序 + 3D
│   ├── seq_rz/                      #   x random, y zero
│   ├── seq_hh/                      #   batch（两维 hold）
│   ├── seq_rz_c2_sk/                #   单维度随机 + 3D 节点（空间序列模型训练用）
│   └── exp7_multiview/              #   多视角实验数据
│
├── train_log/                       # 训练日志与模型权重
└── docs/                            # 文档
    ├── PROJECT_HELP.md              #   文件说明与运行流程（本文件）
    ├── project_status_report.md     #   项目状态报告
    ├── soft_robot_pipeline.md       #   技术管线与模型演进
    ├── literature_innovations.md    #   文献创新点总结
    ├── papers/                      #   论文 PDF 及分析
    ├── directions/                  #   研究方向文档
    ├── experiments/                 #   实验分析与改进
    └── archived/                    #   已归档代码
        ├── ode_cmstnf/              #   ODE-CMSTNF 模型（已归档）
        ├── smooth_cmstnf/           #   Smooth-CMSTNF 模型（已归档）
        └── trainers/                #   旧版 Trainer（hook_based/standalone/multiview）
```

---

## 3. 核心文件详解

### 3.1 仿真环境：`elastica_env.py`

PyElastica 软体臂仿真 + PyVista 渲染，所有数据采集的源头。

**关键类和函数**：

| 名称 | 作用 |
|------|------|
| `ContinuousSoftArmEnv` | 连续仿真环境，保持状态逐步推进 |
| `env.get_observation()` | 获取二值图像 + 当前扭矩。返回 `(binary_img, action)` |
| `env.get_observation_3d()` | 获取二值图像 + 扭矩 + **3D 节点坐标** + 半径。返回 `(img, action, positions(3,31), radii(31,))` |
| `env.get_observation_with_depth()` | 获取二值图像 + **深度图** + 扭矩 + 3D 数据。返回 `(img, depth, action, pos, radii)` |
| `env.get_observation_multiview_with_depth()` | 获取双视角二值图像 + **双视角深度图** + 扭矩 + 3D 数据 |
| `render_depth_map()` | 从 PyVista z-buffer 渲染深度图（float32，单位米） |
| `render_to_binary_with_depth()` | 同时返回二值图像和深度图 |
| `env.set_action(torque)` | 设置驱动扭矩 |
| `env.step(steps=N)` | 推进 N 步物理仿真 |
| `create_simulation()` | 创建独立仿真实例（静态采集用） |
| `get_simulation_data_pair()` | 一次性仿真 + 渲染（批量采集用） |

**物理参数**：30 单元 CosseratRod，0.5m 长，0.015m 半径，2D 扭矩驱动。

### 3.2 数据集：`src/data/dataset.py`

`SoftSequenceDataset` — 加载 .npz 数据，支持 2D 和 3D 两种模式。

| 参数 | 作用 |
|------|------|
| `seq_len=20` | 时序窗口长度 |
| `return_pairs=True` | 返回相邻帧对 `(seq_t, seq_t1, img_t, img_t1)`，用于 smoothness loss |
| `return_3d=True` | **额外返回 3D 节点坐标**。自动检测 npz 是否含 `positions` 字段 |
| `return_depth=True` | **额外返回深度图**。自动检测 npz 是否含 `depth_maps` 字段 |

暴露的属性：`H, W, focal, action_dim`，以及 `get_camera_params()`（优先返回数据自带的相机参数，无则返回 None）。

**返回格式**（随参数不同）：

```
默认:                  (seq, img)
return_pairs:          (seq_t, seq_t1, img_t, img_t1)
return_pairs+3d:       (seq_t, seq_t1, img_t, img_t1, pos_t, pos_t1)
return_pairs+3d+depth: (seq_t, seq_t1, img_t, img_t1, pos_t, pos_t1, depth_t, depth_t1)
return_3d:             (seq, img, positions)
return_3d+depth:       (seq, img, positions, depth)
```

其中 `positions` 形状为 `(3, 31)`，即 31 个节点的 xyz 坐标。

### 3.3 数据采集工具：`scripts/data_collection/collect_utils.py`

动作策略、数据保存、文件命名的工具函数，被 `collect.py` 调用。

| 类/函数 | 作用 |
|---------|------|
| `ActionSchedule` | 每个维度独立生成动作序列（zero/random/hold/file） |
| `save_collection()` | 保存 npz，始终嵌入相机参数。可选保存 `depth_maps`（float32） |
| `make_filename()` | 生成自描述文件名（含模式标签、3D 标记） |
| `infer_save_dir()` | 根据模式自动推断保存目录 |
| `load_defaults()` | 从 simulation.json + camera.json 读取默认参数 |

**ActionSchedule 模式说明**：

| 模式 | 行为 | 对应旧模式 |
|------|------|-----------|
| `zero` | 始终输出 0 | canonical |
| `random` | 平滑随机游走 | sequence |
| `hold` | 采样一个随机值后整段保持 | batch |
| `file` | 从 npz 文件的 actions 字段读取 | 自定义轨迹 |

**文件命名规则**：`seq_{序号}_{模式标签}[_3d]_{时间戳}.npz`

```
seq_000_zz_1748000000.npz        # 两维 zero (canonical)
seq_000_rr_3d_1748000000.npz     # 两维 random + 3D
seq_000_rz_1748000000.npz        # x random, y zero
seq_000_hh_1748000000.npz        # 两维 hold (batch)
```

模式标签：每个维度取首字母（z=zero, r=random, h=hold, f=file），拼接成 2 字符串。

### 3.4 模型文件

所有模型的核心思想相同：**输入驱动参数（actuator inputs），在 3D 空间中查询神经场，输出该点的属性（可见度/密度/SDF），通过体渲染（volume rendering）生成 2D 图像作为监督信号**。

```
驱动参数 ──→ 时序编码器 ──→ 物理状态向量
                              │
3D 查询点 ──→ 位置编码 ──────→  空间解码器 ──→ [vis, density] 或 SDF
                              │              │
                         当前动作 ──────────→│
                                           ↓
                                    体渲染 → 2D 图像（训练监督）
```

> **核心约定**：模型输入只有驱动参数和 3D 查询点。图像、深度图等仅作为训练时的监督信号，不直接输入模型。

---

#### `model.py` — FBV_SM（Field-Based Volumetric Soft body Model）

**名字来源**：FBV-SM 论文（Hu et al. 2025）的原始基线模型。

**核心思想**：最简单的神经场——输入一个点的完整特征（位置 + 动作拼接），直接 MLP 预测该点的可见度和密度。

**数据流**：
```
输入 x: (N, 5)  [xyz(3) + action(2)]
  ├── x[:, :3] → PositionalEncoder(n_freqs=10) → (N, 63)
  ├── x[:, 3:] → ActuatorMLPEncoder → (N, 63)
  └── concat → MLPDecoder → (N, 2) [visibility, density]
```

**训练**：单阶段，MSE 重建 loss。

---

#### `model_mstnf.py` — MSTNF（Multi-Scale Temporal Neural Field）

**名字来源**：**M**ulti-**S**cale **T**emporal **N**eural **F**ield。多尺度时序神经场。

**核心思想**：引入时序建模。用一个动作历史窗口（最近 K 帧）代替单帧动作，通过 MultiScaleEMA 编码器提取物理状态。EMA（指数移动平均）用多个可学习衰减率捕获不同时间尺度的历史影响。

**数据流**：
```
action_window: (B, K=20, D=2)  — 最近 20 帧的驱动参数
  │
  ↓ MultiScaleEMA (4 个可学习衰减率)
  │ → 加权平均 → MLP
  ↓
physics_state: (B, 128)

3D 查询点 points: (N_rays, N_samples, 3)
  │
  ↓ PositionalEncoder(n_freqs=10)
  ↓
  pos_enc: (N_rays, N_samples, 63)

融合: [pos_enc, physics_state(广播), current_action] → MLPDecoder
  ↓
output: (N_rays, N_samples, 2) [visibility, density]
  ↓
体渲染 → 2D 图像
```

**训练**：单阶段，MSE 重建 + 下一帧预测 + 时序平滑 loss。

---

#### `model_cmstnf.py` — C-MSTNF（Canonical MSTNF）

**名字来源**：**C**anonical **M**ulti-**S**cale **T**emporal **N**eural **F**ield。引入规范场（canonical field）概念的 MSTNF。

**核心思想**：D-NeRF 范式——将神经场分解为两个部分：
- **Canonical Field**（规范场）：零动作下的静止形态，是机器人"默认形状"
- **Deformation Field**（变形场）：根据当前动作将查询点映射回规范空间

查询时先变形再查规范场：`world_point → deformation MLP → canonical_point → canonical MLP → [vis, density]`

**数据流**：
```
Phase 1 — Canonical Field:
  points: (N, N_samples, 3)
    → PositionalEncoder → MLPDecoder → [vis, density]
  （用零动作数据训练，学习默认形状）

Phase 2 — Deformation Field:
  action_window: (B, K, D) → MultiScaleEMA → physics_state: (B, 128)

  points: (N, N_samples, 3)
    → PositionalEncoder(n_freqs=6) → deform_features: (N, N_samples, 39)

  [deform_features, physics_state, current_action] → deform_MLP → displacement: (N, N_samples, 3)

  canonical_points = points + displacement
  → canonical_field(canonical_points) → [vis, density]
  → 体渲染 → 2D 图像
```

**训练**：两阶段。Phase 1 用零动作数据训练规范场；Phase 2 冻结规范场，只训练变形场。

---

#### ODE-CMSTNF / Smooth-CMSTNF（已归档）

ODE-CMSTNF 用 Neural ODE（阻尼弹簧模型）替代 EMA 做时序编码，Smooth-CMSTNF 对变形场施加 Spectral Normalization + Jacobian/Gradient 平滑约束。

两个模型已归档到 `docs/archived/ode_cmstnf/` 和 `docs/archived/smooth_cmstnf/`，对应的训练脚本也已移除。

---

#### `model_ms_scnf.py` — MS-SCNF（Multi-Scale Skeleton-Conditioned Neural Field）

**名字来源**：**M**ulti-**S**cale **S**keleton-**C**onditioned **N**eural **F**ield。多尺度骨架条件神经场。

**核心思想**：所有前面的模型都是在隐空间中学习形状——没有显式的 3D 输出。MS-SCNF 引入**骨架回归头**，直接从物理状态预测软体臂的 3D 骨架曲线，然后用骨架作为几何先验条件化密度场。

**数据流**：
```
action_window: (B, K=20, D=2) → MultiScaleEMA → physics_state: (B, 128)
  │
  ↓ SkeletonHead
  │   physics_state → 共享 trunk MLP → feat: (B, 256)
  │   ├── coarse_head:  Linear(256, 12)  → reshape → (B, 4, 3)   粗骨架
  │   ├── medium_head:  Linear(256, 30)  → reshape → (B, 10, 3)  中骨架
  │   └── fine_head:    Linear(256, 93)  → reshape → (B, 31, 3)  细骨架
  │   （三个并行线性头共享 trunk，输出不同粒度的 3D 节点坐标）
  ↓
  skeleton = fine: (B, 31, 3)

3D 查询点 points: (N_rays, N_samples, 3)
  │
  ↓ SkeletonConditionedDensity（骨架局部柱坐标）
  │   1. point_to_skeleton_coords 计算局部柱坐标:
  │      skeleton 相邻节点构成 30 条线段
  │      对每个查询点，找最近线段 → 柱坐标 (dist, t_axial, theta)
  │        dist    = 到最近线段的径向距离
  │        t_axial = 归一化轴向参数 in [0, 1]（沿骨架位置比例）
  │        theta   = 环向角度 in [-π, π]（当前不使用，留供未来扩展）
  │
  │   2. 双路位置编码:
  │      dist    → PE(d_input=1, n_freqs=6)  → dist_enc:  (B*N, N_samples, 13)
  │      t_axial → PE(d_input=1, n_freqs=8)  → axial_enc: (B*N, N_samples, 17)
  │
  │   3. 拼接 + 解码:
  │      [dist_enc, axial_enc] → MLPDecoder → (B*N, N_samples, 2) [vis, density]
  ↓
  体渲染 → 2D 图像
```

**独特优势**：
- `model.predict_skeleton(action_window)` 可直接输出 31 个 3D 节点坐标，无需渲染
- 多尺度预测：三个并行线性头共享 trunk，coarse-to-fine 课程式学习
- 骨架局部柱坐标编码：查询点的密度由径向距离 `dist` 和轴向位置 `t_axial` 决定，不再使用 3D 绝对坐标
  - 径向距离决定"距骨架多远"（距骨架近 → 密度高）
  - 轴向位置决定"在骨架的哪一段"，允许截面半径沿长度变化
  - 环向角度 `theta` 已计算但当前不输入网络（假设圆形截面，留供未来非圆截面扩展）
- `SkeletonConditionedDensity` 不直接用 `physics_state` 或 `action`——所有动作信息都通过骨架间接传递
- 辅助函数 `point_to_skeleton_coords()` 返回 `(dist, t_axial, theta)`，替代旧的 `point_to_segment_distance()`（仅返回 dist）

**训练**：两阶段。Phase 1 骨架回归（3D L2 loss）；Phase 2 联合训练（渲染 loss + 骨架 loss）。

---

#### `model_sdf.py` — TemporalSDF（Temporal SDF）

**名字来源**：**Temporal** **S**igned **D**istance **F**ield。带时序编码的有符号距离场。

**核心思想**：与前面所有模型不同，TemporalSDF **不通过体渲染和 2D 图像做监督**，而是直接用 3D 点云的 SDF 值做监督。模型用 SIREN（周期性激活函数）作为坐标编码器，天然适合学习光滑的距离场。

**独特优势**：
- 不需要 2D 图像作为训练数据，只需要 3D 节点坐标
- 直接输出 SDF 值，SDF=0 的零等值面就是机器人表面
- SIREN 的 sin 激活天然保证 SDF 场的光滑性和可微性

**数据流**：
```
3D 查询坐标 coords: (N, 3)
  │
  ↓ SIREN Coordinate Encoder (3 层 sin 激活)
  │   SirenLayer(3 → 128, is_first=True)  — 第一层用特殊初始化
  │   SirenLayer(128 → 128)
  │   SirenLayer(128 → 128)
  ↓
  spatial_feat: (N, 128)

action_window: (B, K=20, D=2)
  │
  ↓ MultiScaleEMA → temporal_state: (B, 128)
  ↓ Linear(128 → 128)
  ↓
  state_feat: (B, 128)

融合:
  [spatial_feat(广播), state_feat(广播)] → concat → (B*N, 256)
  ↓
  SIREN Fusion MLP (3 层 + 输出层)
  SirenLayer(256 → 256) × 3
  SirenLayer(256 → 1, is_last=True)  — 线性输出
  ↓
  sdf: (B*N, 1)  — 有符号距离值
```

**Loss 组合**：
```
1. SDF L1 regression: |pred_sdf - gt_sdf|（表面点=0，off-surface 点=真实距离）
2. Normal constraint:  法向量一致性（仅表面点，cosine similarity）
3. Eikonal constraint: |∇SDF| = 1（梯度模等于 1，SDF 的基本性质）
```

**训练数据采样**（`dataset_sdf.py`）：
```
每帧数据:
  - on-surface 点 (300): 杆体表面采样，SDF = 0
  - near-surface 点 (200): 表面附近偏置采样，有精确 SDF 值
  - off-surface 点 (200): 扩大空间均匀采样，有精确 SDF 值
  - 坐标归一化到 [-1, 1]^3
```

**训练**：单阶段，直接 3D 监督，不需要体渲染。

---

#### `model_skeleton_sdf.py` — SkeletonSDF（参数化骨架 + 管状 SDF）

**名字来源**：**Skeleton**（参数化骨架曲线）+ **SDF**（有符号距离场）。

**核心思想**：结合骨架回归和 SDF 的优势。先用参数化曲线（B-spline/Fourier 等）预测骨架，保证拓扑连通；再用管状 SDF 先验（`dist_to_skeleton - radius`）提供完整 3D 形状，最后用 SIREN 网络学习残差修正截面形状。

**数据流**：
```
action_window: (B, K=20, D=2) → MultiScaleEMA → physics_state: (B, 128)
  │
  ↓ SkeletonHead (bspline/fourier/point/catmullrom)
  ↓   预测控制点/系数 → 求值 → skeleton: (B, 31, 3)
  ↓
  对每个查询点 x:
    1. dist = point_to_segment_distance(x, skeleton)  — 到骨架最近线段距离
    2. sdf_prior = dist - rod_radius                    — 管状 SDF 先验
    3. pos_enc = positional_encode(x, n_freqs=4)        — 空间位置编码
    4. state_feat = Linear(physics_state) → 32d         — 物理状态投影
    5. residual = SIREN([sdf_prior, pos_enc, state_feat]) — 残差修正
    6. final_sdf = sdf_prior + residual
  ↓
  sdf: (B*N, n_samples, 1)  — 有符号距离值
```

**独特优势**：
- 参数化骨架保证拓扑连通（曲线不会断裂）
- 管状 SDF 先验提供强几何先验，加速收敛
- SIREN 残差修正截面形状（不只是圆柱，可以学习变截面）
- 同时输出 3D 骨架和完整 SDF 场

**Loss 组合**：
```
1. 骨架 loss: 多尺度 (coarse/medium/fine) L2，GT 来自 PyElastica
2. SDF loss: L1 |pred_sdf - gt_sdf|（GT 解析计算: dist_to_skeleton - radius）
3. Normal loss: 表面法向量一致性
4. Eikonal loss: ||grad SDF|| = 1
```

**训练**：两阶段。Phase 1 骨架预热（仅骨架 loss）；Phase 2 联合训练（骨架 + SDF loss）。

**GT SDF 生成**（`sdf_utils.py`）：
```
compute_gt_sdf(query_points, skeleton, radius)
  → 解析计算: GT SDF(x) = dist_to_skeleton(x) - radius
  → 对管状结构精确，不需要 directors

sample_sdf_training_data(positions, radius)
  → 采样三种点: 表面(SDF=0) + 近表面(有精确距离) + 远表面(均匀空间)
  → 返回 query_points, gt_sdf, gt_normals
```

---

#### `fractional_memory.py` — FractionalMemory（分数阶记忆编码器）

**名字来源**：**Fractional**-order **Memory**。用分数阶微积分替代整数阶 EMA 做时序记忆。

**动机**：传统 MultiScaleEMA 用指数衰减核 G(t)∝e^(-t/τ) 做历史加权。但软体材料（硅胶、聚合物）的粘弹性实验表明，记忆核是**幂律衰减** G(t)∝t^(-α)（Rabotnov 1969），而非指数衰减。分数阶导数的 Grünwald-Letnikov（GL）离散化天然给出幂律权重序列。

**核心假设**：
- 软体机器人的当前形态不只取决于最近几步动作，而是受**整个历史轨迹**影响（迟滞效应）
- 历史影响的衰减方式是幂律的（长尾），而非指数的（短尾）
- 不同时间尺度可以用不同的分数阶参数 α 捕获

**输入输出**：
```
输入: action_window (B, K, D) — 最近 K 帧的动作序列
输出: physics_state (B, hidden_dim) — 物理状态向量
```
接口与 MultiScaleEMA **完全一致**，可无缝替换。

**数据流**：
```
action_window: (B, K=40, D=2)
  │
  ├─ α₀=0.11 → GL weights → 加权求和 → feat₀: (B, 2)  # 极短记忆（~无记忆）
  ├─ α₁=0.35 → GL weights → 加权求和 → feat₁: (B, 2)  # 短期记忆
  ├─ α₂=0.41 → GL weights → 加权求和 → feat₂: (B, 2)  # 中期记忆
  └─ α₃=0.69 → GL weights → 加权求和 → feat₃: (B, 2)  # 长期记忆
       │
       ↓ 拼接 + 当前动作 + 速度
       [feat₀, feat₁, feat₂, feat₃, action_now, velocity]: (B, 12)
       │
       ↓ MLP(12 → 128 → 128)
       physics_state: (B, 128)
```

**GL 权重递推**：
```
w₀ = 1
wₖ = wₖ₋₁ × (k - 1 - α) / k

α ∈ (0, 1):
  α → 0: 权重快速衰减，只有最近几帧有效（纯弹性，无记忆）
  α → 1: 权重缓慢衰减，长历史都有效（纯粘性，完全记忆）
  α ≈ 0.3~0.5: 软体材料典型范围（幂律衰减，长尾记忆）
```

**可学习参数**：
- `raw_alphas` (4,) — 4 个分数阶参数，通过 sigmoid 映射到 (0,1)
- `order_weights` (4,) — 4 个尺度混合权重
- `state_mlp` — 状态 MLP 权重

**训练结果**（30 epoch, spatial_sequence 数据）：
- 4 个 α 值自动分化：[0.11, 0.35, 0.41, 0.69]
- scale 0 近似无记忆（快速响应），scale 3 长期记忆（大变形历史）
- 与 EMA 编码器对比：在 SpatialSequence 模型上精度相当，但 α 值可解释

**与 MultiScaleEMA 的关系**：
FractionalMemory 是 MultiScaleEMA 的**物理驱动替代品**。两者接口完全兼容（`forward`、`compute_smoothness`、`decays` 属性），任何使用 MultiScaleEMA 的模型都可通过 `--encoder fractional` 切换。

---

#### `model_spatial_sequence.py` — SpatialSequence（空间序列中心线模型）

**名字来源**：**Spatial Sequence** — 沿空间维度（Z 轴）用序列模型生成中心线。

**动机**：之前的所有模型（MSTNF、C-MSTNF、MS-SCNF、SDF 等）都试图学习"3D 空间中任意点的属性"（密度/SDF），这是一个非常高维的映射。但对于软体臂这种**拓扑固定的管状结构**，真正的自由度只是中心线的弯曲形状（31 个 3D 节点 = 93 DOF）。直接预测中心线比间接通过密度/SDF 恢复要简单得多。

**核心假设**：
- 软体臂的形态可以完全由中心线节点坐标描述（管状结构假设）
- 中心线沿 Z 轴具有因果性：底部节点决定顶部节点（悬臂梁物理）
- 时序历史决定全局弯曲方向，空间传播决定局部形状细节

**输入输出**：
```
输入: action_window (B, K=40, D=2) — 最近 40 帧的驱动参数历史
输出: skeleton_pred (B, 31, 3) — 归一化空间的 31 个中心线节点坐标
```

**数据流**：
```
action_window: (B, 40, 2)
  │
  ↓ FractionalMemory (或 MultiScaleEMA)
  ↓ encode()
  physics_state / cond: (B, 128)
  │
  ↓ init_hidden MLP
  h₀: (B, 128)  — 初始空间隐藏状态
  │
  │  沿 Z 轴从底部到顶部，逐节点传播:
  │  for i in range(31):
  │    z_i = linspace(-1, 1, 31)[i]
  │    z_emb = z_embed(z_i): (B, 128)    — Z 位置嵌入
  │    h = GRU(cond + z_emb, h_{i-1})     — 空间状态传播
  │    node_i = slice_head(h): (B, 3)     — 预测该节点 xyz
  │
  ↓
  skeleton: (B, 31, 3)
```

**监督信号**：
```
L_skeleton:      MSE(pred, gt_skeleton)               — 节点坐标回归
L_spatial_smooth: MSE(Δpred, Δgt)                      — 相邻节点位移连续性
L_smooth:        MSE(state_t, state_{t+1})             — 时序平滑性（TemporalMixin）
```

**独特设计**：
- **GRU 沿 Z 轴传播**：每个节点基于前一个节点的隐藏状态生成，保证拓扑连通和空间连续性
- **Z 位置嵌入**：每个节点知道自己在臂上的位置（底部 vs 尖端），悬臂梁因果性自然编码
- **扇形问题从架构层面消失**：FlowMatch 等方法可能出现"扇形"（预测点云不确定中心线位置），GRU 传播保证节点有序
- **直接 3D 监督**：不需要体渲染，不需要图像，只需要 GT 中心线坐标
- **参数量极小**：183,947 参数（对比 FlowMatch 的 ~300K+），训练快

**训练**：单阶段，端到端。数据用 `SpatialSequenceDataset`（`dataset_spatial.py`），返回归一化的 action_window + gt_skeleton。

**训练结果**（44 epoch, fractional encoder）：
- 全部 10 条序列 CD mean=0.001184（对比 FlowMatch CD≈0.40，提升 350×）
- 测试序列（未见过）CD=0.001416，泛化良好

---

#### `model_pc_spatial.py` — PCSpatial（预测-修正空间序列模型）

**名字来源**：**P**redictive-**C**orrective **Spatial** Sequence。两阶段架构：先预测（Predictive），再修正（Corrective）。

**动机**：SpatialSequence 纯靠驱动历史预测中心线，效果已经很好（CD≈0.001）。但在 real-world 部署时：
1. 仿真模型与真实机器人有差异（sim-to-real gap）
2. 驱动器信号可能有噪声或延迟
3. 外部扰动（碰撞、负载变化）无法仅从驱动历史推断

PCSpatial 借鉴**Kalman 滤波**思想：**模型预测（基于驱动历史）+ 观测修正（基于视觉图像）**。

**核心假设**：
- 驱动历史提供一个强先验（预测分支），在大多数情况下已经足够准确
- 图像观测仅用于修正预测误差（修正分支学习残差），不需要从零学习形状
- 修正信号是低维的（Δxyz per node = 93 DOF），可以用简单的 CNN 从图像提取

**输入输出**：
```
Phase 1 (Predictive):
  输入: action_window (B, K, D) — 驱动历史
  输出: skeleton_pred (B, 31, 3) — 预测中心线

Phase 2 (Corrective):
  输入: action_window + images — 驱动历史 + 多视角图像
  输出: skeleton_final (B, 31, 3) — 修正后中心线
```

**数据流**：
```
Phase 1 — 预测分支（与 SpatialSequence 完全相同）:
  action_window → FractionalMemory → GRU(Z) → pred_skeleton: (B, 31, 3)

Phase 2 — 修正分支:
  pred_skeleton: (B, 31, 3)
  images: (B, V, H, W)  — V 个视角的图像
    │
    ↓ Conv2d(V→32→64) → AdaptiveAvgPool → Linear(64→128) → img_feat: (B, 128)
    ↓ Linear(128→128→93) → reshape → delta: (B, 31, 3)
    │
  final = pred_skeleton + delta: (B, 31, 3)
```

**监督信号**（两阶段相同）：
```
L_skeleton:      MSE(final, gt_skeleton)
L_spatial_smooth: MSE(Δfinal, Δgt)
L_smooth:        时序平滑（TemporalMixin）
```

**独特设计**：
- **两阶段解耦**：Phase 1 冻结修正分支，只训练预测分支；Phase 2 联合训练
- **残差修正**：修正分支只学习 Δ（图像与预测的差异），而非完整形状，收敛更快
- **修正分支参数量小**：89,725 参数（CNN + MLP），仅占总参数 33%
- **sim-to-real 设计**：预测分支用仿真数据训练，修正分支用真实图像微调
- **可选修正**：无图像时退化为纯预测（Phase 1），有图像时加修正（Phase 2）

**训练结果**（48 epoch, fractional encoder, Phase 1 only）：
- 全部 10 条序列 CD mean=0.001114（比 SpatialSequence 好 ~6%）
- 测试序列 CD=0.001154（比 SpatialSequence 的 0.001416 好 ~18%）
- Phase 2（修正分支）尚未训练，待有图像数据后启用

---

#### `model_state_transition.py` — StateTransition（闭环状态转移 + 可学习迟滞潜变量 z）

**名字来源**：**State Transition** — 学习一步状态转移 s_t = F(s_{t-1}, a_t, z_{t-1})，而非前馈状态推断。

**动机**：SpatialSequence 隐含**稳态假设**——"给定动作历史，机器人已到达该历史决定的稳态形态"。但软体材料的迟滞效应使"同一动作、不同历史 → 不同形态"，稳态假设失效（详见 [direction 13](directions/13_closed_loop_state_transition.md) §〇）。StateTransition 把前一步真实状态作为显式输入，**学转移不学状态**。

**核心思想**：
- **闭环**：前一步骨架 s_{t-1} + 当前动作 → 当前骨架 s_t
- **可学习迟滞潜变量 z（无 GT）**：z 自演化 `z_t = Φ_z(z_{t-1}, a_t, s_{t-1})`（GRUCell），编码位置+动作之外的深度历史（如内部应力方向、充/放气方向）
- **Δ 预测**：`s_t = s_{t-1} + delta_scale·tanh(Δ_raw)`，输出增量、天然连续，delta_scale·tanh 提供收缩约束（控制 rollout 误差累积）

**信息流（forward）**：
```
action_window (B, K, D) → TemporalEncoder → cond (B, hidden)   # 动作编码，每步重算、无记忆
                                                          │
s_{t-1} (B,N,3) ─┐  v = s_{t-1} - s_{t-2}                  │
                 └→ StateEncoder([s_{t-1}, v]) → state_seed  # warm-start 的 GRU 种子
                                                          │
z_{t-1} (B, z_dim) → z_cell(GRUCell,[cond,s_{t-1}]) → z_t → z_proj
                                                          ↓
   沿 Z 轴逐节点: cond + z_pos_embed + z_proj → GRU(z₀→z_K) → 每节点 Δ_raw
                                                          │
   s_t = s_{t-1} + delta_scale·tanh(Δ_raw)   （冷启动首帧 s_t = Δ）
```

**关键模块**（相对 SpatialSequence 的增量）：
| 模块 | 作用 |
|------|------|
| `z_init` | 冷启动 z_0 = z_init(cond) |
| `z_cell` (GRUCell) | 演化 z_t = Φ_z(z_{t-1}, [cond, s_{t-1}])，门控利于迟滞建模 |
| `z_proj` | 将 z_t 投影到 hidden，加性注入 per-node GRU |
| `state_encoder` | [s_{t-1}, v] → GRU 种子（替代 init_hidden） |
| `delta_head` + `delta_scale` | 增量预测 + 可学习收缩标量（防 NaN、控累积） |

**关键约定（z 与 cond 的职责分离）**：cond 是"当前动作编码"（每步重算、无记忆）；z 是"演化中的迟滞潜状态"（带历史记忆）。z 无物理真值（实物上无 z 传感器），端到端从 skeleton loss 学。

**向后兼容**：forward 的 `prev_skeleton/prev_z` 默认 None → 冷启动回退 `init_hidden(cond) + z_init(cond)`，退化为带 z 初始化的前馈预测，旧单参调用（只传 action_window）完全兼容。不修改 SpatialSequenceModel / PCSpatial。

**训练（Stage 0，单帧 per-frame）**：teacher forcing，`prev_skeleton = positions[t-1]`（GT，已在 .npz 无需重采）。此模式下 z 每步从 z_init 重置（无跨帧记忆，退化为 cond 的函数）——z 成长为真正迟滞潜变量需窗口训练（见 GTObservedTransition）。

**定位**：**自回归闭环**——推理时把模型自身预测喂回（rollout），适用于无法每步观测真实状态、需一路推下去的场景。**当前已退为未来扩展**（方向 13），主线是 GTObservedTransition。

**Loss**：skeleton（MSE）+ spatial_smooth（相邻节点位移连续）+ smooth（时序平滑，TemporalMixin）。z 无 GT，不加 loss。

**评估**：`eval_rollout.py` 闭环 rollout（s 和 z 都喂模型预测），报告 rollout/onestep 漂移比 + ‖z‖ 范数轨迹 + 发散步检测。

---

#### `model_gt_transition.py` — GTObservedTransition（全 GT 驱动窗口框架，当前主线）

**名字来源**：**GT**-Observed **Transition** — 前一状态永远来自真实观测（仿真 GT / 实物图像骨架化）的状态转移。

**动机**：实际部署中每步都能采集图像 → 骨架化得到真实 s_{t-1}，**不必一路自回归推下去**（那会导致 s 误差累积，漂移比可达 1000×）。GTObservedTransition 固化"前一状态恒真实"的设定，是当前主线。

**与 StateTransition 的关系**：继承 `StateTransitionSpatialModel`，**复用全部 forward / z_module**，仅固化 `training_spec`（`use_episode_mode=True` + `teacher_forcing_ratio=1.0` + `episode_len=40`）。

**核心思想（窗口模式）**：
- 每步：`ŝ_t = F(真实 s_{t-1}, z_{t-1}, a_t)`，s_{t-1} 恒真实
- z 在状态窗口 `[s_{t-K}...s_{t-1}]` 内 K 步演化（每步喂真实 s），**z 不跨样本携带**
- K = episode_len（默认 40，对齐 action_window），可调（`--episode_len`）
- 样本自包含 → **样本间可打乱**（shuffle，解决"必须按顺序"的顾虑）

**关键设计**（详见 [direction 14](directions/14_gt_observed_transition.md) §4–6）：
1. **z 窗口演化可打乱**：z 只在样本内 K 步演化，不跨样本叠加
2. **z_0 = cond-only 初始化**：K=40 演化下 z_0 留存 ≈ 0.9^40 ≈ 2%，初始化方式影响小；先简后消融（zero-init baseline 对比）
3. **dense supervision**：窗口内每步预测 ŝ_j 都算 loss，给无 GT 的 z 每步直接梯度（关键）；可选递增权重 `--dense_step_weight linear`（最后几步权重大）
4. **部署/评估只看最后一步 ŝ_t**：dense 是训练手段（帮 z 学），部署无 GT 不算 loss

**信息流（窗口训练，trainer._compute_sequence_losses）**：
```
样本: action_windows (B, K, seq_len, D) + gt_skeletons (B, K, N, 3) + init_skeleton (B, N, 3)
z_0 = init_z_from_action(action_windows[:, 0])       # cond-only
for t in range(K):
    out = model.forward(action_windows[:, t], s_prev=gt_skeletons[:, t-1], ..., z_t)  # s 恒 GT
    ŝ_t = out["skeleton"]; z_t = out["latent_z"]      # z 跨步演化，s_prev 永远真实
loss = Σ_t w_t · MSE(ŝ_t, gt_skeletons[:, t])         # dense（w_t 等权或递增）
```

**无数据泄漏**：预测 `ŝ_{j+1}=F(s_j, z_j, a_{j+1})` 的 GT 是 s_{j+1}，而 s_{j+1} 从未出现在预测路径（z_j 只依赖 ≤s_j 的历史）。状态窗口双重使用（既作输入又作 GT label）是标准 teacher forcing，合法。

**训练**：单阶段 episode 窗口，TF=1.0，dense supervision。

**验证（eval_gt_transition.py）**：观测驱动 rollout（s 每步真实 + z 跨帧演化），报告：
- **部署指标**：最后一步 ŝ_t 的 MSE（无 loss 时的预测精度）
- per-step MSE（诊断，dense supervision 的逐步精度）
- ‖z‖ 范数轨迹 + z drift ratio（z 漂移监测）

**部署流程**：
```
循环每步 t:
  1. 采集图像 → 骨架化 → 真实 s_{t-1}
  2. model.forward(action_window_t, s_{t-1}, z_{t-1}) → ŝ_t, z_t
  3. z_t 内部维护（跨步演化），下一步喂回
  （无需 GT，不计算 loss，直接用 ŝ_t 作为当前形态预测）
```

**冒烟验证**（cuda3，episode_len=12，3 epoch）：z drift ratio=2.06x（z 从 0.58 温和演化到 1.19，收敛有界），对比纯自回归 rollout 漂移比 1170× → **s 不漂移（每步重置真实），仅 z 是风险源**，确认全 GT 窗口框架是当前部署的合理选择。

**与 StateTransition / SpatialSequence 的定位对比**：
| 模型 | 前一状态来源 | z | s 误差累积 | 定位 |
|------|------------|---|-----------|------|
| SpatialSequence | 无（纯前馈稳态） | 无 | — | 基线（稳态假设） |
| StateTransition | 自身预测（rollout） | 跨步演化 | 严重（漂移 1000×） | 未来扩展（无法每步观测） |
| **GTObservedTransition** | **真实观测** | **窗口内演化** | **无（s 每步真实）** | **当前主线** |

---

#### Depth-CMSTNF（无独立模型文件）

**名字来源**：**Depth**-supervised **C**anonical **M**STNF。用深度图增强监督的 C-MSTNF。

**核心思想**：Depth-CMSTNF **没有独立的模型文件**，它直接复用 `model_cmstnf.py` 的 `CMSTNFModel`，模型架构与 C-MSTNF 完全相同。创新点在训练策略层面——利用深度图提供额外的三维监督信号。

**与 C-MSTNF 的区别**（仅在训练策略中）：
```
额外训练策略:
  1. 深度 L1 loss:  |E[depth] - depth_gt|（渲染期望深度与 GT 深度图的差异）
  2. 深度引导采样:  用 GT 深度图集中采样点在物体表面附近（coarse-to-fine）
```

**关键设计**：深度图仅用于训练时的 loss 和采样引导，**推理时只需要驱动参数**，不需要深度图或任何传感器输入。

---

#### 多视角+深度训练（无独立模型文件）

**核心思想**：多视角训练不修改模型架构，直接复用 MSTNF / C-MSTNF 等现有模型。创新点在训练策略——同时从多个固定相机视角做 volume rendering，利用多视角图片 + 深度图提供更强的 3D 几何约束。

**与单视角训练的关键区别**（在 `ViewStrategy` 的 `MultiViewStrategy` 中）：
```
每个 training step:
  for each view_i in [front, side, ...]:
    采样该视角射线 → 查询模型 → 体渲染 → MSE(rendered_i, gt_i)
    如果有深度: L1(rendered_depth_i, gt_depth_i)
  loss = Σ view_losses / V + w_depth × depth_loss + w_smooth × smoothness
```

**新增基础设施**：

| 文件 | 作用 |
|------|------|
| `src/utils/camera_system.py` | `MultiCameraSystem` — 多相机射线生成、投影/反投影，兼容新旧 npz 格式 |
| `src/data/dataset_multiview_depth.py` | `MultiViewDepthDataset` — 自动识别新旧格式，返回多视角图片+深度列表 |
| `src/rendering/view_strategy.py` | `MultiViewStrategy` — 多视角 rendering + 深度 + 一致性约束 |

---

#### `model.py` — FBV_SM（遗留基线）

来自 FBV-SM 原始论文，输入 (xyz + action) 拼接后直接 MLP，无时序建模。仅作参考对比。

---

#### 共享层：`layers.py` + `skeleton_heads.py`

**`layers.py` 通用层**：

| 组件 | 作用 |
|------|------|
| `PositionalEncoder` | 正弦余弦位置编码：`x → [x, sin(2^0·x), cos(2^0·x), ..., sin(2^{L-1}·x), cos(2^{L-1}·x)]` |
| `ActuatorMLPEncoder` | 动作参数 MLP 编码器，将低维动作映射到高维特征空间 |
| `MLPDecoder` | 通用解码 MLP：`input → 2d → 2d → d → d/2 → output`，density 用 softplus 激活 |
| `MultiScaleEMA` | 多尺度指数移动平均：用 N 个可学习衰减率分别做 EMA，再加权拼接 |
| `FractionalMemory` | 分数阶记忆：用 Grünwald-Letnikov 离散化实现幂律衰减记忆核，接口与 EMA 兼容 |
| `GammaLaguerreMemory` | Gamma/Laguerre 延迟核：`w_t = t^(k-1) * λ^t / Z`，钟罩形权重捕获延迟峰值响应，k=1 退化为 EMA |
| `TemporalLSTMEncoder` | LSTM 时序编码器（旧版，已被 EMA 替代） |

**`skeleton_heads.py` 骨架回归头**（从 model_ms_scnf.py 提取，供 MS-SCNF 和 SkeletonSDF 复用）：

| 类 | 参数化方式 | 参数量 | 特点 |
|----|-----------|--------|------|
| `SkeletonHead` | point — 独立预测每个节点 | 135 | 原始方案，最灵活 |
| `FourierSkeletonHead` | fourier — 截断 Fourier 级数 | 51 (@ n_freq=8) | 天然光滑，带限 |
| `BSplineSkeletonHead` | bspline — 三次 B-spline | 30 (@ n_ctrl=10) | 局部控制 + C² 连续 |
| `CatmullRomSkeletonHead` | catmullrom — Catmull-Rom 样条 | 30 (@ n_ctrl=10) | 插值型，通过控制点 |

统一接口：`forward(physics_state) → dict('coarse', 'medium', 'fine')` 各为 `(B, N, 3)`。

辅助函数：`downsample_skeleton()`（均匀下采样）、`point_to_segment_distance()`（可微点到线段距离，仅返回 dist）、`point_to_skeleton_coords()`（可微骨架局部柱坐标，返回 dist + t_axial + theta）、`create_skeleton_head()`（工厂函数）。


### 3.5 提取模块（`src/` 子目录分层）

代码经过分层重构，共享组件从 `models/` 提取到独立子目录：

| 目录 | 文件 | 内容 |
|------|------|------|
| `src/encoders/` | `multi_scale_ema.py` | `MultiScaleEMA` — 多尺度指数移动平均时序编码 |
| | `fractional_memory.py` | `FractionalMemory` — 分数阶幂律记忆核（GL 离散化） |
| | `gamma_laguerre.py` | `GammaLaguerreMemory` — Gamma 延迟峰记忆核（k/λ 可学习） |
| `src/fields/` | `canonical.py` | `CanonicalField` — 规范场 MLP |
| | `deformation.py` | `DeformationField` — 变形场 MLP |
| | `skeleton_density.py` | `SkeletonConditionedDensity` — 骨架局部柱坐标条件密度场 (dist + t_axial) |
| `src/heads/` | `skeleton_heads.py` | 4 种骨架参数化头 + 工厂函数 |
| `src/rendering/` | `view_strategy.py` | `SingleViewStrategy` / `MultiViewStrategy` |
| `src/evaluation/` | `query.py` | 模型查询工具（预测骨架/SDF/密度场） |
| | `render.py` | 可视化渲染（mesh/pointcloud/animation） |

### 3.6 训练架构：Spec 声明式系统

所有模型通过 `training_spec` 类属性声明训练需求，`UnifiedTrainer` 统一解释执行。**无需为每个模型写独立的 Trainer 子类**。

**三个正交维度**：

| 维度 | 机制 | 负责 |
|------|------|------|
| **Phase 策略** | `PhaseSpec` + `PhaseStrategy` | 阶段数、冻结、forward、epochs |
| **监督模式** | `supervision_mode` | `"rendering"` / `"direct_3d"` / `"skeleton"` |
| **视角策略** | `ViewStrategy` | 单视角 / 多视角 / 跨视角约束 |

**三种监督模式**：

| 模式 | 前向流程 | 适用模型 |
|------|---------|---------|
| `"rendering"` | rays → 3D points → model(pts, action) → 体渲染 → 像素对比 | MSTNF, CMSTNF, MS-SCNF Phase 2 |
| `"direct_3d"` | coords → model(coords, action) → 值对比 (SDF/法向量) | SDF, SkeletonSDF |
| `"skeleton"` | action → model.predict_skeleton(action) → 骨架对比 | MS-SCNF Phase 1, SkeletonSDF Phase 1 |

**Loss 组织**：所有 loss 在 `active_losses` 中声明，分两层计算：
- **渲染层**（ViewStrategy）：recon, depth, reproj, consist
- **模型层**（`model.compute_losses()`）：smooth, skeleton, sdf, normal, eikonal

**各模型的 training_spec**：

| 模型 | 阶段 | 监督模式 | 活跃 Loss |
|------|------|---------|----------|
| MSTNF | 1 phase: full | rendering | recon, smooth |
| CMSTNF | 2 phase: canonical → deformation | rendering → rendering | [recon] → [recon, smooth] |
| MS-SCNF | 2 phase: skeleton → joint | skeleton → rendering | [skeleton] → [skeleton, recon, smooth] |
| TemporalSDF | 1 phase: full | direct_3d | sdf, normal, eikonal |
| SkeletonSDF | 2 phase: skeleton → joint | direct_3d → direct_3d | [skeleton] → [skeleton, sdf, normal, eikonal] |
| SpatialSequence | 1 phase: spatial | spatial_sequence | skeleton, spatial_smooth, smooth |
| PCSpatial | 2 phase: predictive → corrective | spatial_sequence → spatial_sequence | [skeleton, spatial_smooth, smooth] → [skeleton, spatial_smooth, smooth] |
| StateTransition | 1 phase: state_transition (单帧) | spatial_sequence | skeleton, spatial_smooth, smooth |
| GTObservedTransition | 1 phase: gt_transition (窗口, TF=1.0) | spatial_sequence (窗口) | skeleton(dense), spatial_smooth, smooth |

**核心文件**：

| 文件 | 作用 |
|------|------|
| `src/training/spec.py` | `PhaseSpec` / `TrainingSpec` 数据类 |
| `src/training/phase_strategy.py` | 解析 spec，管理冻结/解冻/forward |
| `src/rendering/view_strategy.py` | `SingleViewStrategy` / `MultiViewStrategy` |
| `src/training/dataset_factory.py` | 根据 `dataset_type` 创建数据集 + collate → dict batch |
| `src/training/trainer_unified.py` | 统一训练器，组合 PhaseStrategy + ViewStrategy |
| `scripts/training/train_unified.py` | 统一入口脚本，支持 5 个模型 |

**旧 Trainer 文件**已归档到 `docs/archived/trainers/`（hook_based / standalone / multiview 三类）。各训练脚本（`train_*.py`）已改为 UnifiedTrainer 的薄包装，保持原有 CLI 接口不变。

### 3.7 配置文件：`config/training.json`

所有模型共享，关键参数：

| 节 | 用途 |
|----|------|
| `optimization` | 学习率、batch_size、epoch 数 |
| `normalization` | 动作归一化方式（max_abs） |
| `temporal` | 时序窗口大小 (40)、EMA 尺度数 (4)、隐层维度 (128) |
| `loss_weights` | 重建/预测/平滑权重 |
| `model` | 位置编码频率数 (10)、隐层维度 (128) |
| `canonical` | Phase 1/2 epoch 数、变形学习率 |
| `ms_scnf` | **MS-SCNF 专用**：多尺度节点数 (4/10/31)、骨架 loss 权重 |
| `multiview` | **多视角训练**：每视角射线数、深度/reproj/consist loss 权重、warmup 课程 epoch 数 |
| `sdf` | **SDF 专用**：表面/近表面/离表面采样点数、SDF/法向量/Eikonal 权重 |
| `evaluation` | **评估**：网格分辨率、动画 FPS |

CLI 参数在 `src/config/args.py` 定义，运行时可覆盖 `training.json` 默认值。共享 CLI 基础设施：

| 工具函数 | 作用 |
|---------|------|
| `add_common_args(parser)` | 10 个通用参数（data_dir, lr, n_epochs, batch_size, num_workers, window_size, n_scales, hidden_dim, eval_interval, seed） |
| `add_two_phase_args(parser)` | 两阶段参数（phase, exp_dir, phase1_epochs, phase2_epochs） |
| `build_common_overrides(args)` | 从 args 自动提取覆盖项，避免每个脚本重复写字典 |
| `resolve_training_config(overrides)` | 合并 CLI 覆盖到 training.json 默认值（跳过 None 值） |
| `resolve_phase_epochs(spec, config, ...)` | 统一两阶段 epoch 分配逻辑 |

### 3.8 3D 评估指标：`src/training/metrics_3d.py`

| 函数 | 含义 |
|------|------|
| `mean_node_error(pred, gt)` | 所有节点平均 L2 误差 |
| `endpoint_error(pred, gt)` | 末端节点 L2 误差（最关心的指标） |
| `chamfer_distance(pred, gt)` | 双向最近邻点云距离 |
| `curve_smoothness(skeleton)` | 二阶差分 L2 范数（越小越平滑） |

---

## 4. 数据格式说明

### 4.0 动作-仿真-模型输入的时序对应关系

这是理解整个管线的关键：一个动作值在仿真中维持多久、产生多少帧、模型看到什么。

```
仿真层：
  dt = 0.0001s (100μs)          ← config/simulation.json
  steps_per_action = 500         ← 每个动作值维持 500 步
  record_interval = 50           ← 每 50 步记录一帧

时间换算：
  1 个动作值 = 500 × 0.0001s = 0.05s (50ms)
  1 帧记录  = 50 × 0.0001s  = 0.005s (5ms)
  每个动作值产生 500/50 = 10 帧记录

数据集层：
  window_size = 40 帧 (默认)     ← config/training.json
  40 帧 × 0.005s = 0.2s (200ms)
  40 帧 / 10 帧/动作 ≈ 4 个不同动作值

模型输入：
  action_window 形状: (batch, 40, 2)
  包含 ~4 个不同动作值，每个重复 ~10 次
  时间跨度：200ms 的历史
```

**关键细节**：
- 动作值在 hold 期间是**重复的**——同一个动作值产生 10 帧，但每帧的机器人物理状态在演化（因为 50ms 内材料还在变形）
- 模型的 action_window 包含这些**重复的动作值**——即模型看到的输入中有大量重复
- 如果改用 `ContinuousSoftArmEnv`（连续仿真），动作值可以每帧都不同，没有 hold 概念
- 当前所有数据（`seq_rr`, `seq_zz` 等）使用 `get_simulation_data_pair()` 模式，每次独立仿真，每个动作值 hold 500 步

```
可视化（一个动作值的生命周期）：

时间:  0ms          10ms    20ms    30ms    40ms    50ms
       |------------|-------|-------|-------|-------|
动作:  a=0.3        a=0.3   a=0.3   a=0.3   a=0.3   → 切换到 a=0.5
       ↑            ↑       ↑       ↑       ↑
帧:   frame_0     frame_1  frame_2 frame_3 ... frame_9
记录:  ✓           ✓       ✓       ✓       ...   ✓

物理:  刚施加力    开始变形  继续变形  接近平衡  基本稳定
```

**对模型设计的影响**：
- EMA/Gamma 等时序编码器对重复动作值做加权平均 → 实际效果类似于对 4 个不同值做平均（因为重复 10 次的权重被合并）
- 如果要研究速率效应（变速加载），需要修改 `steps_per_action` 或使用 `ContinuousSoftArmEnv`
- `--window_size` 参数控制历史窗口长度，但物理含义取决于 `steps_per_action` 和 `record_interval`

---

### 4.1 数据字段

所有采集数据（`collect.py`）均为 `.npz` 文件，**始终包含相机参数**，使数据自描述：

```
# 基础字段（所有数据都有）
images:    (T, H, W)        二值渲染图像序列
actions:   (T, 2)           驱动扭矩序列 [torque_x, torque_y]
dt:        float            帧间时间间隔
focal:     float            焦距（从 camera.json 计算）
H, W:      int              图像尺寸
camera_eye:    (3,)         相机位置
camera_center: (3,)         相机注视点
camera_up:     (3,)         相机上方向

# 3D 数据额外字段（--3d 模式）
positions: (T, 3, 31)       每帧的 31 个 3D 节点坐标
radii:     (T, 31)          每帧的节点半径

# 深度数据额外字段（--depth 模式）
depth_maps: (T, H, W)       每帧的深度图（float32，单位米，无物体区域为 0.0）

# 多视角深度数据
depth_maps_front: (T, H, W) 正面深度图
depth_maps_side:  (T, H, W) 侧面深度图
```

**多视角数组格式**（新版优先保存，可扩展到任意视角数）：
```
# 新格式（优先）
images:        (T, V, H, W)   V 个视角的二值图，index 0=front, 1=side, ...
depths:        (T, V, H, W)   V 个视角深度图（--depth 时），索引与 images 对齐
camera_params: (V, 10)        每行 [eye(3), center(3), up(3), focal(1)]
view_names:    ['front', 'side', ...]   视角名称映射表

# 同时保留旧格式字段（兼容现有代码）
images_front, images_side, camera_eye_front, ...（与旧格式完全一致）
```

`MultiViewDepthDataset` 自动兼容新旧格式，优先读取数组字段。`MultiCameraSystem.from_npz()` 也按相同优先级自动构建。

**向后兼容**：旧数据不含 camera_eye/center/up 字段，训练器会自动回退到 `camera.json` 配置。

---

## 5. 运行流程

所有数据采集通过统一的 `collect.py` 脚本完成，每个动作维度独立控制。

```bash
# 查看所有参数
python scripts/data_collection/collect.py --help
```

### 5.1 采集数据

`collect.py` 不再区分模式，而是通过 `--action-x` 和 `--action-y` 独立控制每个维度：

```bash
# canonical（两维都为零）— 用于 C-MSTNF 系列 Phase 1
python scripts/data_collection/collect.py --action-x zero --action-y zero

# 时序数据（两维随机游走）— 用于 MSTNF 训练或 C-MSTNF 系列 Phase 2
python scripts/data_collection/collect.py

# 单维度随机（x 随机，y 固定为零）
python scripts/data_collection/collect.py --action-x random --action-y zero

# batch（每段序列保持一个随机值）
python scripts/data_collection/collect.py --action-x hold --action-y hold

# 从文件读取轨迹
python scripts/data_collection/collect.py --action-x file --action-file traj.npz

# 带 3D 标注的时序数据 — 用于 MS-SCNF / SDF / SkeletonSDF 训练
python scripts/data_collection/collect.py --3d

# 带 3D 标注的 canonical 数据
python scripts/data_collection/collect.py --action-x zero --action-y zero --3d

# 含深度图的时序数据 — 用于 Depth-CMSTNF / RGB-D 训练
python scripts/data_collection/collect.py --depth

# 含 3D + 深度图的数据
python scripts/data_collection/collect.py --3d --depth
```

**保存目录自动推断**：`data/seq_{模式标签}[_3d]/`

| 命令 | 保存目录 |
|------|---------|
| `--action-x zero --action-y zero` | `data/seq_zz/` |
| 默认（两维 random） | `data/seq_rr/` |
| `--3d` | `data/seq_rr_3d/` |
| `--action-x zero --action-y zero --3d` | `data/seq_zz_3d/` |
| `--action-x random --action-y zero` | `data/seq_rz/` |
| `--action-x hold --action-y hold` | `data/seq_hh/` |

可用 `--save-dir` 覆盖自动推断。

### 5.2 统一训练入口（推荐）

所有 5 个模型均可通过 `train_unified.py` 训练：

```bash
# MSTNF（单阶段，rendering）
python scripts/training/train_unified.py --model mstnf --data_dir data/sequence_data

# C-MSTNF（两阶段，rendering）
python scripts/training/train_unified.py --model cmstnf --data_dir data/sequence_data \
    --canonical_data_dir data/canonical_data

# MS-SCNF（两阶段，skeleton+rendering）
python scripts/training/train_unified.py --model ms_scnf --data_dir data/seq_rr_3d

# TemporalSDF（单阶段，direct_3d，无需图像）
python scripts/training/train_unified.py --model sdf --data_dir data/seq_rr_3d

# SkeletonSDF（两阶段，direct_3d，无需图像）
python scripts/training/train_unified.py --model skeleton_sdf --data_dir data/seq_rr_3d

# 多视角 + 一致性
python scripts/training/train_unified.py --model cmstnf --data_dir data/exp7_multiview \
    --multiview --depth --consistency
```

各模型原有脚本（`train_mstnf.py` 等）仍可使用，内部已迁移到 UnifiedTrainer。

### 5.2.2 超参数搜索（`train_search.py`）

对任意模型、任意参数做网格搜索，直接调用 `train_unified.py` 作为子进程。

```bash
# 搜索学习率（4 组）
python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rr_3d \
    --search lr=1e-4,3e-4,1e-3,3e-3

# 多参数网格搜索（2×3=6 组）
python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rr_3d \
    --search lr=1e-4,1e-3 --search batch_size=2,4,8

# 只打印命令不执行（手动复制或保存为 .sh）
python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rr_3d \
    --search lr=1e-4,1e-3 --dry_run

# 跳过 Phase 1，直接搜索 Phase 2
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_search.py \
    --model ms_scnf --data_dir data/seq_rr_3d --search lr=1e-4,1e-3 --phase 2

# 中断后续跑（跳过已有 best_model.pt 的实验）
python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rr_3d \
    --search lr=1e-4,1e-3 --resume

# 汇总已完成搜索结果
python scripts/training/train_search.py --model ms_scnf --data_dir data/seq_rr_3d \
    --search lr=1e-4,1e-3 --summarize
```

支持搜索的参数：`lr`、`batch_size`、`n_epochs`、`phase1_epochs`、`phase2_epochs`、`skeleton_mode`、`n_freqs`、`d_filter`、`deform_n_freqs`、`n_rays`、`n_samples`、`chunk_size`。

### 5.2.1 C-MSTNF 系列（MSTNF / C-MSTNF）

**第一步：采集数据**

```bash
# 零动作数据（Phase 1 用，仅 C-MSTNF 需要）
python scripts/data_collection/collect.py --action-x zero --action-y zero

# 时序动作数据（Phase 2 / MSTNF 训练用）
python scripts/data_collection/collect.py
```

**第二步：训练**

```bash
# 推荐方式（统一入口）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py --model mstnf --data_dir data/sequence_data
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py --model cmstnf --data_dir data/sequence_data \
    --canonical_data_dir data/canonical_data

# 或使用原有脚本（薄包装，CLI 不变）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_cmstnf.py
```

> **注意**：ODE-CMSTNF 和 Smooth-CMSTNF 已归档到 `docs/archived/`，对应训练脚本已移除。

### 5.3 MS-SCNF（推荐）

**第一步：采集带 3D 标注的数据**

```bash
# 带 3D 标注的时序数据
python scripts/data_collection/collect.py --3d

# （可选）带 3D 标注的零动作数据
python scripts/data_collection/collect.py --action-x zero --action-y zero --3d
```

**第二步：训练**

```bash
# 完整训练（统一入口）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py --model ms_scnf --data_dir data/seq_rr_3d

# 或使用薄包装脚本
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py

# 分阶段运行：
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 1  # 仅骨架回归
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ms_scnf.py --phase 2 \
    --exp_dir train_log/train_ms_scnf/001 \
    --skeleton_path train_log/train_ms_scnf/001/phase1/model/skeleton_best.pt
```

**第三步：3D 评估**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
    --model_type ms_scnf \
    --checkpoint train_log/train_ms_scnf/<timestamp>/phase2/model/best_model.pt \
    --data_dir data/seq_rr_3d
```

输出 4 个定量指标：Mean Node Error、Endpoint Error、Chamfer Distance、Curve Smoothness。

### 5.3.1 SkeletonSDF（参数化骨架 + 管状 SDF）

**核心思路**：参数化骨架保证拓扑连通 + 管状 SDF 先验提供完整 3D 体积 + SIREN 残差修正截面。不需要 2D 图像，仅用 3D 节点坐标 + 动作驱动参数。

**第一步：采集带 3D 标注的数据**

```bash
# 与 MS-SCNF 相同的 3D 数据
python scripts/data_collection/collect.py --3d
```

**第二步：训练**

```bash
# 推荐方式（统一入口）
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_unified.py --model skeleton_sdf --data_dir data/seq_rr_3d

# 或使用薄包装脚本
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py

# 指定骨架模式
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py --skeleton_mode fourier

# 自定义参数
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_skeleton_sdf.py \
    --skeleton_mode bspline \
    --n_epochs_phase1 100 \
    --n_epochs_phase2 200 \
    --rod_radius 0.015 \
    --w_sdf 1.0 --w_eikonal 0.1 --w_normal 0.1
```

**训练过程**：
- Phase 1: 仅骨架回归（与 MS-SCNF Phase 1 相同），预热骨架头
- Phase 2: 联合训练骨架 + SDF（SDF GT 由 `sdf_utils.py` 解析计算）

**第三步：3D 评估**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_3d.py \
    --model_type skeleton_sdf \
    --checkpoint train_log/train_skeleton_sdf/<timestamp>/model/best_model.pt \
    --data_dir data/seq_rr_3d
```

### 5.3.2 空间序列模型（SpatialSequence / PCSpatial）

**核心思路**：直接预测中心线节点坐标（31 × 3D），不走体渲染或 SDF 管线。GRU 沿 Z 轴空间传播保证拓扑连通，FractionalMemory 提供物理驱动的时序记忆。

**第一步：采集带 3D 标注的数据**

```bash
# 与 MS-SCNF 相同的 3D 数据
python scripts/data_collection/collect.py --3d
```

**第二步：训练 SpatialSequence**

```bash
# 分数阶记忆编码器（推荐）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_spatial_sequence.py \
    --data_dir data/seq_rz_c2_sk --encoder fractional --n_epochs 500

# 传统 EMA 编码器
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_spatial_sequence.py \
    --data_dir data/seq_rz_c2_sk --encoder ema --n_epochs 500
```

**第二步（备选）：训练 PCSpatial**

```bash
# Phase 1: 预测分支（与 SpatialSequence 相同）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_pc_spatial.py \
    --data_dir data/seq_rz_c2_sk --encoder fractional --n_epochs 500

# Phase 2 自动在 Phase 1 完成后开始（修正分支）
# 修正分支需要图像数据，无图像时退化为纯预测
```

**第三步：可视化**

```bash
# 交互式 3D 可视化
python scripts/evaluation/visualize_3d_shape.py --device cuda:0
# 选择 SpatialSequence/PCSpatial checkpoint → 数据文件 → 帧范围
```

**训练结果**：

| 模型 | 编码器 | Epochs | 全部序列 CD | 测试序列 CD |
|------|--------|--------|------------|------------|
| SpatialSequence | FractionalMemory | 44 | 0.001184 | 0.001416 |
| PCSpatial (pred) | FractionalMemory | 48 | 0.001114 | 0.001154 |

### 5.3.3 状态转移模型（GTObservedTransition 主线 / StateTransition 未来扩展）

**核心思路**：从 SpatialSequence 的前馈稳态推断，升级为闭环状态转移 `s_t = F(s_{t-1}, a_t, z_{t-1})`——前一步真实状态 + 可学习迟滞潜变量 z 作为显式输入。**主线是 GTObservedTransition**（前一状态恒真实），StateTransition 的自回归 rollout 退为未来扩展。

**第一步：采集带 3D 标注的数据**

```bash
# 与空间序列模型相同的 3D 数据（positions 逐帧连续，prev_skeleton 无需重采）
python scripts/data_collection/collect.py --3d
# 当前默认数据目录 data/seq_rz_c2_sk
```

**第二步：训练 GTObservedTransition（主线）**

```bash
# 默认（cuda1，episode_len=40 对齐 action_window，dense supervision）
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py

# 短 epoch 冒烟测试（验证管线，如 cuda3）
CUDA_VISIBLE_DEVICES=3 python scripts/training/train_gt_transition.py --n_epochs 5 --episode_len 12

# 递增权重 dense supervision（窗口最后几步权重大）+ 调 z 维度
CUDA_VISIBLE_DEVICES=3 python scripts/training/train_gt_transition.py \
    --dense_step_weight linear --z_dim 32 --episode_len 40
```

**第三步：观测驱动评估（主线）**

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/eval_gt_transition.py \
    --checkpoint train_log/gt_transition/<exp>/phase_gt_transition/model/best_model.pt \
    --data_dir data/seq_rz_c2_sk --seq_idx 0 --max_steps 40
```
输出：最后一步部署 MSE（ŝ_t 精度）+ per-step 诊断 MSE + ‖z‖ 范数轨迹 + z drift ratio。理想情况 z drift ratio ≈ 1（z 稳定有界，无累积漂移）。

**（可选）训练 StateTransition 自回归版（未来扩展）**

```bash
# 状态转移：开环训练（热启动 gt_transition + 纯闭环；Stage0/Stage1 脚本已归档）
CUDA_VISIBLE_DEVICES=3 python scripts/training/train_open_loop_transition.py --n_epochs 5

# 评估纯自回归 rollout（s 和 z 都喂预测，监测漂移）
CUDA_VISIBLE_DEVICES=3 python scripts/evaluation/eval_rollout.py \
    --checkpoint train_log/state_transition/<exp>/phase_state_transition/model/best_model.pt \
    --data_dir data/seq_rz_c2_sk --seq_idx 0 --max_steps 60
```

**部署说明**（GTObservedTransition）：
- 每步采集图像 → 骨架化得真实 s_{t-1} → 喂模型得 ŝ_t（z 由模型内部跨步演化维护）
- 无需 GT、不算 loss，直接用 ŝ_t 作为当前形态预测
- Stage 2（未来）：接入图像骨架化感知前端，从 2D 图像获取真实 s_{t-1}（当前 Stage 0/1 用仿真 GT）

### 5.4 深度增强模型（Depth-CMSTNF / RGB-D Neural Field）

**第一步：采集含深度图的数据**

```bash
# canonical 数据（Phase 1 用，Depth-CMSTNF 需要）
python scripts/data_collection/collect.py --action-x zero --action-y zero --depth

# 时序数据 + 深度图（Phase 2 / RGB-D 训练用）
python scripts/data_collection/collect.py --depth
```

**第二步：训练**

```bash
# Depth-supervised CMSTNF（深度作为额外监督损失，不改变模型架构）
CUDA_VISIBLE_DEVICES=2 python scripts/training/train_depth_cmstnf.py

# 调整深度损失权重
CUDA_VISIBLE_DEVICES=2 python scripts/training/train_depth_cmstnf.py --depth_weight 0.5

# 关闭深度引导采样（只用深度损失，不用 coarse-to-fine）
CUDA_VISIBLE_DEVICES=2 python scripts/training/train_depth_cmstnf.py --no_guided_sampling
```

**核心设计**：深度图仅用于训练时的 loss 计算和采样引导，**推理时只需要驱动参数**，不需要深度图或任何传感器输入，完全符合自建模思想。

### 5.5 多视角+深度训练

**核心思想**：不改模型架构，同时从 2-3 个固定视角做 volume rendering，各视角 MSE loss 求和 + 深度 L1 loss。

**第一步：采集多视角+深度数据**

多视角数据通过 `collect.py` 的多视角模式采集（由 `collect_utils.py` 中的 `ActionSchedule` 和 `MultiCameraSystem` 支持）。

**第二步：训练**

```bash
# 推荐方式（统一入口）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
    --model cmstnf --data_dir data/exp7_multiview --multiview --depth

# 含跨视角一致性约束
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_unified.py \
    --model cmstnf --data_dir data/exp7_multiview --multiview --depth --consistency

# 或使用原有脚本（薄包装）
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_multiview.py \
    --model cmstnf --data_dir data/exp7_multiview --depth
```

**关键组件**：

| 文件 | 作用 |
|------|------|
| `src/utils/camera_system.py` | `MultiCameraSystem` — 统一管理多相机参数、射线生成、投影/反投影 |
| `src/data/dataset_multiview_depth.py` | `MultiViewDepthDataset` — 兼容新旧 npz 格式 |
| `src/rendering/view_strategy.py` | `MultiViewStrategy` — 多视角 rendering + 深度 + 一致性约束 |
| `scripts/training/train_unified.py` | 统一训练入口 |

**Loss 组成**：
```
L = Σ_v (w_recon × MSE(render_v, gt_v)) / V          # 每视角重建
  + w_depth × Σ_v L1(depth_v, depth_gt_v) / V         # 每视角深度
  + w_reproj × MSE(render_B, gt_B)                     # 重投影：视角 A depth → 3D → 视角 B → 对比 GT
  + w_consist × MSE(render_A, render_B)                # 一致性：同一 3D 点两视角渲染自洽
  + w_smooth × MSE(state_t, state_{t+1})               # 时序平滑

跨视角 loss 的 warmup 课程:
  前 warmup_epochs 个 epoch，reproj/consist 权重从 0 线性增长到设定值，
  避免训练初期跨视角梯度干扰密度场学习
```

### 5.6 评估与可视化

```bash
# 3D 几何评估（定量指标）
python scripts/evaluation/evaluate_3d.py \
    --model_type <model_type> \
    --checkpoint <path_to_checkpoint> \
    --data_dir <data_dir>

# 预测对比可视化
python scripts/evaluation/visualize_predictions.py compare

# 预测动画
python scripts/evaluation/visualize_predictions.py animate

# 3D SDF/Mesh 可视化
python scripts/evaluation/visualize_3d_shape.py \
    --model_type <model_type> \
    --checkpoint <path_to_checkpoint>
```

---

## 6. 数据依赖关系图

```
                          elastica_env.py
                 ┌──────────────┼──────────────────────┐
                 │              │                      │
          get_observation  get_observation_3d   get_observation_multiview_with_depth
                 │              │                      │
                 ▼              ▼                      ▼
           collect.py      collect.py              collect.py
           (default)       --3d                --multiview --depth
                 │              │                      │
                 ▼              ▼                      ▼
           data/seq_rr/   data/seq_rr_3d/     data/exp7_multiview/
           (img+act+cam)  (+ pos+radii)      (多视角 img+depth+cam)
                 │              │                      │
          ┌─────┴─────┐   ┌────┴──────────┐    ┌─────┴─────┐
          │           │   │               │    │           │
          ▼           ▼   ▼               ▼    ▼           ▼
        MSTNF    C-MSTNF   Phase 1   Phase 2     多视角训练
        (单阶段)  系列      骨架回归  ┌─────────┐  (cmstnf+multiview)
          │           │     │       │         │   │
          └─────┬─────┘     │       ▼         ▼   │
                │           │    MS-SCNF    SkeletonSDF
                ▼           │    (3D+2D)    (3D SDF)
          2D 渲染可视化     │       │         │
          (人眼评估)        │       ▼         ▼
                            └───────┴─────────┘
                                    │
                                    ▼
                            evaluate_3d.py
                            (定量 3D 指标)
```

---

## 7. 常见注意点

- **统一采集**：所有数据采集使用 `collect.py`，通过 `--action-x/--action-y` 独立控制每个维度。每个维度可选 zero/random/hold/file 模式。
- **数据自描述**：新采集的数据始终包含相机参数（camera_eye/center/up/focal/H/W），训练器优先使用数据自带参数，旧数据自动回退到 `camera.json`。
- **参数来源**：`collect.py` 的默认值从 `simulation.json` + `camera.json` 读取，CLI 参数可覆盖。
- **文件命名**：输出文件名包含模式标签（如 `zz`、`rr`、`rz`）和 3D 标记，一目了然。
- **保存目录**：自动推断为 `data/seq_{模式标签}[_3d]/`，可用 `--save-dir` 覆盖。
- **GPU 选择**：通过环境变量指定，如 `CUDA_VISIBLE_DEVICES=2 python scripts/training/train_mstnf.py`。脚本默认 GPU 0。
- **动作归一化**：训练时自动计算归一化因子并保存到 `action_norm_factor.txt`，推理时需加载。
- **多视角数据格式**：新版采集保存 `(N,V,H,W)` 数组格式（`images`, `depths`, `camera_params`），同时保留旧格式字段（`images_front` 等）兼容旧代码。
- **多视角训练器**：多视角训练通过 `ViewStrategy` 的 `MultiViewStrategy` 实现（`src/rendering/view_strategy.py`），不再需要独立的 Trainer。统一入口 `train_unified.py --multiview`。
- **训练架构**：所有模型通过 `training_spec` 声明式配置训练需求，`UnifiedTrainer` 统一解释执行。新增模型只需添加 `training_spec` 类属性和 `compute_losses()` 方法，无需写新 Trainer。
- **旧 Trainer 归档**：旧 Trainer 文件归档在 `docs/archived/trainers/`（分 hook_based / standalone / multiview 三类）。各训练脚本已改为 UnifiedTrainer 薄包装。
- **ODE/Smooth 归档**：ODE-CMSTNF 和 Smooth-CMSTNF 的模型和训练脚本已归档到 `docs/archived/`。
- **SDF 可视化**：`visualize_3d_shape.py` 支持 mesh（marching cubes 面片）和 pointcloud（SDF<=0 的点云）两种模式。
- **根目录旧文件**：`env.py`、`func.py`、`train.py`、`predefined.py` 来自原始 FBV-SM 论文，与当前 PyElastica 管线无关，仅供参考。
- **骨架模块复用**：`skeleton_heads.py` 从 `model_ms_scnf.py` 提取为独立模块（`src/heads/`），供 MS-SCNF 和 SkeletonSDF 共享。包含 4 种骨架参数化（point/fourier/bspline/catmullrom）及辅助函数。
- **分层重构**：共享组件已从 `src/models/` 提取到 `src/encoders/`、`src/fields/`、`src/heads/`、`src/rendering/`、`src/evaluation/`，降低耦合。
- **配置管理**：训练配置在 `config/training.json`，CLI 参数在 `src/config/args.py` 定义，运行时可覆盖 JSON 默认值。
- **骨架局部柱坐标**：`SkeletonConditionedDensity` 使用 `(dist, t_axial)` 而非 3D 绝对坐标，环向角度 `theta` 已计算但当前不使用。旧的 `point_to_segment_distance()` 保留供 SkeletonSDF 等模型使用。
- **跨视角 loss 设计**：reproj 和 consist 均不做 alpha 硬门控，全部采样射线参与。consist 对比同一 3D 点两视角的渲染结果（模型自洽性），reproj 对比视角 B 渲染与 GT（监督信号）。训练初期通过 warmup 课程逐步引入跨视角约束。
- **超参数搜索**：`train_search.py` 子进程调用 `train_unified.py`，支持网格搜索、dry_run、resume、summarize，中断后可手动修改参数继续。
- **分数阶记忆编码器**：FractionalMemory（`src/encoders/fractional_memory.py`）是 MultiScaleEMA 的物理驱动替代品，接口完全兼容。训练脚本通过 `--encoder fractional` 切换。GL 权重的 α 参数可学习，训练后可从 checkpoint 的 `temporal.raw_alphas` 读取。
- **空间序列模型**：SpatialSequence 和 PCSpatial 直接预测中心线节点坐标（31 × 3D），不走体渲染或 SDF 管线。GRU 沿 Z 轴空间传播保证拓扑连通。使用 `dataset_spatial.py` 的 `SpatialSequenceDataset`。
- **预测-修正架构**：PCSpatial 的 Phase 1 纯预测（等价 SpatialSequence），Phase 2 加入 CNN 图像修正分支（残差学习），设计目标为 sim-to-real 迁移。Phase 2 需要 `--encoder fractional` 以外的图像数据。
- **可视化支持**：`visualize_3d_shape.py` 支持所有 8 种模型类型（density/SDF/pointcloud/skeleton），骨架模型使用固定坐标轴 + 等比例显示。
- **闭环状态转移模型**：StateTransition（`model_state_transition.py`）把前一步真实状态 + 可学习迟滞潜变量 z 作为输入，学转移 `s_t=F(s_{t-1},a_t,z_{t-1})` 不学状态。Δ 预测 + delta_scale·tanh 收缩。z 无 GT，端到端从 skeleton loss 学。forward 的 prev_* 默认 None → 冷启动回退前馈，旧单参调用兼容。
- **全 GT 驱动窗口框架（当前主线）**：GTObservedTransition（`model_gt_transition.py`）继承 StateTransition，固化"前一状态恒真实"（TF=1.0）+ z 在状态窗口 K 步演化（K=episode_len=40）+ dense supervision（每步预测都算 loss，给无 GT 的 z 直接梯度）。样本自包含可打乱（z 不跨样本）。这是当前部署主线；StateTransition 的纯自回归 rollout 退为未来扩展（无法每步观测时）。
- **dense supervision 无泄漏**：状态窗口的 s 既作输入又作 GT label 是标准 teacher forcing——预测 ŝ_{j+1} 只用 ≤s_j 的历史，GT s_{j+1} 不在预测路径。部署时无 GT 不算 loss，直接用最后一步预测 ŝ_t。
- **GPU 选择约定**：测试/冒烟实验用 cuda1 或 cuda3（如 `CUDA_VISIBLE_DEVICES=3`），避免占用 cuda0。

---

## 8. 实物数据（免标定 2D）工作流

> 这是项目的**第二条数据路线**（与 §1–7 的 sim PyElastica 多视角标定路线并列）。完整细节见
> [`docs/research/2026-07-10-real-data-2d-workflow.md`](research/2026-07-10-real-data-2d-workflow.md)。

### 8.1 定位与与 sim 路线的区别

实物平台：1-DOF 双段软体臂，单相机，**免相机标定**（无棋盘格、无三角化）。

| 维度 | sim 路线（§1–7，仍有效） | 实物路线（本节，新） |
|------|------------------------|--------------------|
| 仿真/采集 | PyElastica 物理仿真 + PyVista 渲染 | 真实相机采集（RealSense D400） |
| 标定 | 多视角相机标定 + 度量 3D 内参 | **免标定**，无度量 3D / 内参 |
| state 表示 | 3D 节点坐标 `positions (T,3,31)` | **2D 图像骨架 `[col,row,0]`**（像素，z≈0 平面假设） |
| 监督 | 体渲染 / 3D SDF 直接监督 | 2D 骨架回归（无体渲染） |
| 度量验证 | 仿真 GT 直接对比 | **NDI 6DOF tracker** 独立度量（末端 mm） |
| 相机矩阵投影 | 用（有内参 + 度量 3D） | **不用**（免标定管线无度量 3D；表示本身在图像平面） |

学一个状态转移模型 `ŝ_t = F(s_{t-1}, a_t)`，其中 **state = 2D 图像骨架**（像素）、**action = 归一化气压** ∈[0,1]。
末端毫米精度用独立采集的 **NDI** 验证。GT-observed（每步观测）与 open-loop（开环 rollout）在同一网络上对比。

### 8.2 流程命令（端到端）

```bash
# ① mask → 免标定 2D npz（2D 骨架 [col,row,0] + tip_fix + 离群插值 + action 归一[0,1]）
python scripts/real/masks_to_transition_npz.py --seq real_capture/data/raw/<seq>
#   可降节点：--n-points 21（默认 31，全流水线按 N 分数自适应）

# ② 静态段共识清洗（动作段保留真实弯曲，静态段跨帧中位共识）
python scripts/real/clean_transition_npz.py --seq <seq>           # → data/real_seq/<seq>_clean/

# ③ 训练（统一入口，--mode gt|open_loop；训练吃 ..._clean/train）
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode gt --data_dir data/real_seq/<seq>_clean/train
# open_loop（热启动自最新 gt）：--mode open_loop

# ④ 定量评估（末端 NDI mm + 形态 px + open_loop drift_by_k）
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
    --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
    --data_dir data/real_seq/<seq>_clean/train

# ⑤ 可视化（模型预测叠真实照片：原图+mask+GT 骨架+预测骨架同框）
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/visualize_real_overlay.py \
    --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
    --data_dir data/real_seq/<seq>_clean/train
```

### 8.3 关键脚本（一句话说明）

| 脚本 | 作用 |
|------|------|
| `scripts/real/masks_to_transition_npz.py` | mask → 免标定 2D npz（逐行质心 + 弧长重采样 + `tip_fix` 垂直尖端切片修 corner + 离群插值 + action 归一） |
| `scripts/real/clean_transition_npz.py` | 静态段共识清洗（绝对位置锚定关节，修分割抖动 / 关节偏移 / mask 缺块；动作段保留） |
| `scripts/real/repair_masks.py` | mask 级修复（独立轨道，逐行宽共识，产 `masks_repaired/`，不重骨架化） |
| `scripts/real/composite_frames.py` | 批量 原图+mask+骨架 叠图（10214 帧 + montage） |
| `scripts/real/compare_skeleton_methods.py` | 7 法末端 corner 对比（独立真值 + bend 分层） |
| `scripts/real/skeleton_to_shape.py` | node→形态 半径偏移基线（骨架 + 常数半径 r → IoU 对比 mask） |
| `scripts/training/train_transition.py` | 统一训练入口（`--mode gt\|open_loop`，取代旧 `train_gt/open_loop_transition`） |
| `scripts/evaluation/eval_real_quant.py` | 定量评估（① 末端 NDI mm ② 像素部署 tip/node/chamfer/hausdorff/procrustes ③ 分段+按 action 分箱 ④ drift_by_k） |
| `scripts/evaluation/visualize_real_overlay.py` | 模型预测叠真实照片（`(col,row)` 直接画像素、丢 z） |
| `scripts/evaluation/inspect_real_data.py` | 骨架数据网格诊断（9 帧 2D + 3D 预览） |

底层支撑模块：`src/utils/skeleton_2d.py`（2D 骨架 + `_perpendicular_tip_fix`）、`src/evaluation/transition_metrics.py`（窗口开环 rollout + drift_by_k）、`src/evaluation/shape_metrics.py`（chamfer/hausdorff/f-score）。

### 8.4 坐标空间澄清（关键）

| 量 | 空间 | 来源 |
|---|---|---|
| 骨架 GT / 模型预测 / mask | **像素 `[col,row]`, z≈0** | 图像（免标定管线） |
| NDI 末端 | **毫米 `[x,y,z]`**（tracker 帧） | NDI 传感器（独立度量） |
| drift_by_k | 无量纲（归一化空间） | rollout/onestep MSE 比 |

- **预测是 px，不是 mm。** 模型 forward 在归一化空间运算，反归一化回 **像素** `[col,row,z]`；z 通道 `pc_scale≈eps` 使其恒≈0（平面 1-DOF 假设）。
- **整体形态误差只能算 px**（31 节点只有图像 GT，无度量 GT）；**末端误差 px + mm 都能算**（末端有 NDI 度量 GT）。
- **px↔mm 对应（免相机标定）**：NDI 末端 `(x,y,z mm)` 与图像骨架 `node0 (col,row px)` 是同一物理点逐帧配对，用全部帧 **(GT node0 px ↔ NDI x,y mm)** 最小二乘拟合 **2D 仿射** `A: (col,row,1)→(x,y)`。拟合残差 RMS = 标定噪声底；模型末端像素经同一 `A`→mm 与 NDI 比 → 末端毫米误差。
- **不用相机矩阵投影**：免标定管线无度量 3D / 内参，表示本身活在图像平面；相机矩阵是给 sim（度量 3D + 内参）用的，对实物是二次变换、会扭曲。

**实测**（GT 模型 `exp_20260709_5`，2500 帧）：NDI 仿射标定底 **0.74 mm**；GT 模型末端 mean **0.77 mm** / median 0.57 / p90 1.4 mm → 底亚毫米、模型已到噪声底。

### 8.5 模块化（A/B 快速对比）

| 想对比 | 改什么 | 影响范围 |
|---|---|---|
| 骨架节点数 | `masks_to_transition_npz --n-points N` | 关节检测/静态共识/末端修复/训练/评估全部按 N 分数自适应（已验证 N=31/21/15 同一物理关节与末端） |
| 末端修复 | `--tip-fix` / `--no-tip-fix` | 骨架提取是否修 corner（npz、composite） |
| 骨架化方法 | `scripts/real/compare_skeleton_methods.py` | 7 法末端 corner 对比 |
| gt vs open_loop | `train_transition --mode` + `eval_real_quant --mode` | 部署语义 + drift |

> 节点索引全部按 N 的分数（关节搜索 ~0.25–0.85·N、静态共识 ~0.4·N、动作段 ~0.6·N、末端修复 body 节点 ~0.10/0.25·N），故降节点**不需手调任何魔法数**。

### 8.6 实物硬件与数据布局

**硬件**（纠正旧 doc 的"电磁阀"假设）：驱动 = TwinCAT PLC（pyads，`192.168.50.56.1.1:851`）+ 电机推注射器（**电机位置 mm 才是真实控制量**）；气压 = Arduino 读 I2C 传感器（COM4@9600）；相机 = Intel RealSense D400；末端度量 = NDI 6DOF tracker（`ndi.csv`: `t_sec,x,y,z mm` + 姿态）。采集程序在 `docs/ref/Main UI-plc/`。

```
real_capture/data/raw/<seq>/{cam0/<NNNNN>.png, actions6.csv, ndi.csv, frame_times.txt, meta.json}
  + derived/<seq>/{masks, masks_repaired, overlay}
  + data/real_seq/<seq>[_clean]/{train,val}/*.npz   positions:(T,3,N) actions:(T,1) ∈[0,1]
```

