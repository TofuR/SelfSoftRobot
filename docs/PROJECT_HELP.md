# SelfSoftRobot 项目帮助文档

本文档说明每个文件的作用、如何使用，以及每个模型从数据采集到训练评估的完整运行流程。

---

## 1. 项目整体逻辑

```
PyElastica 物理仿真 → PyVista 渲染图像 → 数据采集 (.npz) → 模型训练 → 形态预测 + 3D 评估
```

项目目标：**仅用驱动参数（扭矩）预测软体机器人的完整 3D 形态**。

当前有五套模型管线，全部通过 Spec 声明式训练架构统一管理：

| 管线 | 模型 | 核心创新 | 监督模式 | 数据需求 | 输出 |
|------|------|---------|---------|---------|------|
| C-MSTNF 系列 | MSTNF / C-MSTNF | 多尺度时序编码 + D-NeRF 变形场 | rendering | 2D 图像 + 动作 | 2D 渲染图 |
| MS-SCNF | MS-SCNF | 显式骨架回归 + 骨架条件密度场 | skeleton→rendering | 2D 图像 + 动作 + 3D 节点坐标 | 3D 骨架 + 2D 渲染图 |
| SDF 3D | TemporalSDF | SIREN 坐标编码 + 3D SDF 直接监督 | direct_3d | 3D 节点坐标 + 动作（无需图像） | 3D SDF 场 |
| SkeletonSDF | SkeletonSDF | 参数化骨架 + 管状 SDF 先验 + SIREN 残差 | direct_3d | 3D 节点坐标 + 动作（无需图像） | 3D SDF 场 + 3D 骨架 |
| 多视角+深度 | MSTNF / C-MSTNF (多视角训练) | 多视角 rendering + 深度监督融合 | rendering (MultiView) | 多视角 2D 图像 + 深度图 + 相机参数 | 多视角 2D 渲染图 |

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
│   │   └── multi_scale_ema.py       #   MultiScaleEMA（多尺度指数移动平均）
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
│   │   └── model_sdf.py             #   TemporalSDF（SIREN + EMA 时序 SDF）
│   ├── data/
│   │   ├── dataset.py               #   SoftSequenceDataset（支持 2D/3D/深度）
│   │   ├── dataset_sdf.py           #   SDFDataset（3D SDF 监督采样）
│   │   ├── dataset_skeleton_sdf.py  #   SkeletonSDFDataset（骨架 + SDF 采样）
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
│   │   └── train_multiview_consistency.py  # 多视角一致性薄包装
│   ├── evaluation/
│   │   ├── evaluate_3d.py           #   3D 几何评估脚本
│   │   ├── visualize_3d_shape.py    #   3D SDF/mesh 可视化
│   │   └── visualize_predictions.py #   预测对比/动画可视化
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

CLI 参数在 `src/config/args.py` 定义，运行时可覆盖 `training.json` 默认值。

### 3.8 3D 评估指标：`src/training/metrics_3d.py`

| 函数 | 含义 |
|------|------|
| `mean_node_error(pred, gt)` | 所有节点平均 L2 误差 |
| `endpoint_error(pred, gt)` | 末端节点 L2 误差（最关心的指标） |
| `chamfer_distance(pred, gt)` | 双向最近邻点云距离 |
| `curve_smoothness(skeleton)` | 二阶差分 L2 范数（越小越平滑） |

---

## 4. 数据格式说明

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
