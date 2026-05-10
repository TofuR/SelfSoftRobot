# SelfSoftRobot 项目帮助文档

本文档说明每个文件的作用、如何使用，以及每个模型从数据采集到训练评估的完整运行流程。

---

## 1. 项目整体逻辑

```
PyElastica 物理仿真 → PyVista 渲染图像 → 数据采集 (.npz) → 模型训练 → 形态预测 + 3D 评估
```

项目目标：**仅用驱动参数（扭矩）预测软体机器人的完整 3D 形态**。

当前有四套模型管线：

| 管线 | 模型 | 核心创新 | 数据需求 | 输出 |
|------|------|---------|---------|------|
| C-MSTNF 系列 | MSTNF / C-MSTNF / ODE-CMSTNF / Smooth-CMSTNF | 多尺度时序编码 + D-NeRF 变形场 | 2D 图像 + 动作 | 2D 渲染图 |
| MS-SCNF | MS-SCNF | 显式骨架回归 + 骨架条件密度场 | 2D 图像 + 动作 + 3D 节点坐标 | 3D 骨架 + 2D 渲染图 |
| **深度增强** | **Depth-CMSTNF** | 深度图作为额外监督信号 | **2D 图像 + 动作 + 深度图（仅训练监督，部署时不需要）** | **深度感知 2D 渲染图** |
| **SDF 3D** | **TemporalSDF** | SIREN 坐标编码 + 3D SDF 直接监督 | **3D 节点坐标 + 动作（无需 2D 图像）** | **3D SDF 场** |

---

## 2. 目录结构

```
SelfSoftRobot/
├── elastica_env.py                  # PyElastica 仿真环境 + 渲染
│
├── src/
│   ├── models/                      # 模型定义
│   │   ├── layers.py                #   通用层（PositionalEncoder, MLPDecoder 等）
│   │   ├── model.py                 #   FBV_SM 基线（原始论文方法）
│   │   ├── model_mstnf.py           #   MSTNF（MultiScaleEMA 时序编码）
│   │   ├── model_cmstnf.py          #   C-MSTNF（Canonical + Deformation）
│   │   ├── model_ode_cmstnf.py      #   ODE-CMSTNF（Neural ODE 替代 EMA）
│   │   ├── model_smooth_cmstnf.py   #   Smooth-CMSTNF（正则化变形场）
│   │   ├── model_ms_scnf.py         #   MS-SCNF（骨架条件神经场）
│   │   ├── model_sdf.py             #   TemporalSDF（SIREN + EMA 时序 SDF）
│   ├── data/
│   │   └── dataset.py               # SoftSequenceDataset（支持 2D/3D 数据）
│   │   └── dataset_sdf.py           # SDFDataset（3D SDF 监督采样）
│   ├── training/
│   │   ├── base.py                  #   BaseTrainer（渲染、射线采样等共享工具）
│   │   ├── two_phase_trainer.py     #   TwoPhaseTrainer（Phase1+Phase2 基类）
│   │   ├── trainer_mstnf.py         #   MSTNF 训练器
│   │   ├── trainer_cmstnf.py        #   C-MSTNF 训练器
│   │   ├── trainer_ode_cmstnf.py    #   ODE-CMSTNF 训练器
│   │   ├── trainer_smooth_cmstnf.py #   Smooth-CMSTNF 训练器
│   │   ├── trainer_ms_scnf.py       #   MS-SCNF 训练器
│   │   ├── trainer_depth_cmstnf.py  #   Depth-CMSTNF 训练器（深度监督）
│   │   ├── trainer_sdf.py           #   TemporalSDF 训练器（3D SDF 监督）
│   │   ├── metrics_3d.py            #   3D 评估指标
│   │   └── rendering.py             #   旧版渲染工具
│   ├── config/
│   │   ├── training.json            #   训练超参数（所有模型共享）
│   │   ├── camera.json              #   相机参数
│   │   └── simulation.json          #   仿真参数
│   └── utils/
│       ├── camera.py                #   get_rays（射线生成）
│       ├── rendering.py             #   OM_rendering, sample_stratified, OM_rendering_with_depth, sample_depth_guided
│       ├── experiment.py            #   实验目录管理 + GIF 保存
│       ├── config_utils.py          #   CLI 参数覆盖 + 配置合并工具
│       └── visualization.py         #   可视化工具
│
├── scripts/
│   ├── data_collection/             # 数据采集
│   │   ├── collect.py               #   统一采集入口（per-dim 动作控制 + --3d + --depth）
│   │   ├── collect_multiview.py #   多视角采集（+ --depth）
│   │   └── collect_utils.py         #   动作策略、保存、命名工具函数
│   ├── training/                    # 训练入口脚本
│   │   ├── train_mstnf.py           #   MSTNF
│   │   ├── train_cmstnf.py          #   C-MSTNF
│   │   ├── train_ode_cmstnf.py      #   ODE-CMSTNF
│   │   ├── train_smooth_cmstnf.py   #   Smooth-CMSTNF
│   │   ├── train_ms_scnf.py         #   MS-SCNF
│   │   ├── train_depth_cmstnf.py  #   Depth-CMSTNF（深度监督，新）
│   │   ├── train_sdf.py             #   TemporalSDF（3D SDF 监督）
│   ├── evaluation/
│   │   └── evaluate_3d.py           #   3D 几何评估脚本（新）
│   └── visualization/               # 可视化工具
│
├── notebooks/                       # Jupyter 验证笔记本
│   ├── 06_linear_deform_test.ipynb  #   线性变形层验证
│   └── 07_coarse_to_fine_freq.ipynb #   课程式频率学习验证
│
├── data/                            # 数据目录（gitignore）
│   ├── seq_zz/                      #   canonical（两维 zero）
│   ├── seq_zz_3d/                   #   canonical + 3D
│   ├── seq_rr/                      #   时序（两维 random）
│   ├── seq_rr_3d/                   #   时序 + 3D
│   ├── seq_rz/                      #   x random, y zero
│   ├── seq_hh/                      #   batch（两维 hold）
│   └── ...                          #   其他组合
│
├── train_log/                       # 训练日志与模型权重
└── docs/                            # 文档
    ├── PROJECT_HELP.md              #   文件说明与运行流程（本文件）
    ├── project_status_report.md     #   项目状态报告
    ├── soft_robot_pipeline.md       #   技术管线与模型演进
    ├── literature_innovations.md    #   文献创新点总结
    ├── papers/                      #   论文 PDF 及分析
    │   ├── paper_understanding.md   #     FBV-SM 论文详解
    │   ├── chen2022_paper_understanding.md  # Chen 2022 论文分析
    │   └── shan2024_paper_understanding.md  # SoftNeRF 论文分析
    ├── directions/                  #   研究方向文档
    │   ├── directions_overview.md   #     方向总览
    │   ├── direction_1~5_*.md       #     各方向详细文档
    │   └── ...
    └── experiments/                 #   实验分析与改进
        ├── experiment_analysis.md   #     实验结果分析
        ├── results_evaluation.md    #     评估结果
        └── improvement_proposals.md #     改进方案与实施状态
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

#### `model_ode_cmstnf.py` — ODE-CMSTNF（ODE-based Canonical MSTNF）

**名字来源**：用 **ODE**（常微分方程）替代 EMA 做时序编码的 C-MSTNF。

**核心思想**：EMA 是离散的时序编码，ODE-CMSTNF 用 Neural ODE（具体为阻尼弹簧模型）替代。ODE 积分天然保证状态轨迹在时间上连续，理论上能捕捉软体臂的阻尼振荡动力学。

**与 C-MSTNF 的唯一区别**——时序编码器：
```
action_window: (B, K, D)
  │
  ↓ ODETemporalEncoder (RK4 积分)
  │   状态: [position(hidden/2), velocity(hidden/2)]
  │   动力学: ds/dt = f(s, action)
  │   具体: ds_pos/dt = s_vel
  │         ds_vel/dt = -k·s_pos - c·s_vel + B·action
  │   (阻尼弹簧: k=刚度, c=阻尼, B=外力增益)
  ↓
physics_state: (B, 128)
```

其余架构（规范场 + 变形场）与 C-MSTNF 完全相同。

---

#### `model_smooth_cmstnf.py` — Smooth-CMSTNF

**名字来源**：对变形场施加 **Smooth**ness（平滑性）约束的 C-MSTNF。

**核心思想**：C-MSTNF 的变形场可能产生高频跳变（微小的动作变化导致巨大的形状变化）。Smooth-CMSTNF 通过两种手段约束变形场的 Lipschitz 常数：
1. **Spectral Normalization**：限制变形 MLP 每层权重矩阵的谱范数
2. **Jacobian / Gradient Penalty**：显式惩罚变形场对空间坐标和时间变化的剧烈梯度

**与 C-MSTNF 的区别**：
```
额外 loss:
  - Jacobian penalty: mean((∂displacement/∂x)²) — 空间平滑性
  - Temporal gradient penalty: ||D(x,a_t) - D(x,a_{t+1})||² / ||a_t - a_{t+1}||²
```

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
  ↓ SkeletonConditionedDensity
  │   1. 计算查询点到骨架线段的距离:
  │      skeleton 相邻节点构成 30 条线段
  │      对每个查询点，找最近线段 → dist: (B*N, N_samples)
  │
  │   2. 双路位置编码:
  │      dist  → PE(d_input=1, n_freqs=6) → dist_enc:  (B*N, N_samples, 13)
  │      pts   → PE(d_input=3, n_freqs=4) → pos_enc:   (B*N, N_samples, 27)
  │
  │   3. 拼接 + 解码:
  │      [dist_enc, pos_enc] → MLPDecoder → (B*N, N_samples, 2) [vis, density]
  ↓
  体渲染 → 2D 图像
```

**独特优势**：
- `model.predict_skeleton(action_window)` 可直接输出 31 个 3D 节点坐标，无需渲染
- 多尺度预测：三个并行线性头共享 trunk，coarse-to-fine 课程式学习
- 骨架距离编码：查询点到最近骨架线段的距离作为密度先验（距骨架近 → 密度高）
- `SkeletonConditionedDensity` 不直接用 `physics_state` 或 `action`——所有动作信息都通过骨架间接传递

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

#### Depth-CMSTNF（无独立模型文件）

**名字来源**：**Depth**-supervised **C**anonical **M**STNF。用深度图增强监督的 C-MSTNF。

**核心思想**：Depth-CMSTNF **没有独立的模型文件**，它直接复用 `model_cmstnf.py` 的 `CMSTNFModel`，模型架构与 C-MSTNF 完全相同。创新点在训练策略层面——利用深度图提供额外的三维监督信号。

**与 C-MSTNF 的区别**（仅在 `trainer_depth_cmstnf.py` 中）：
```
额外训练策略:
  1. 深度 L1 loss:  |E[depth] - depth_gt|（渲染期望深度与 GT 深度图的差异）
  2. 深度引导采样:  用 GT 深度图集中采样点在物体表面附近（coarse-to-fine）
```

**关键设计**：深度图仅用于训练时的 loss 和采样引导，**推理时只需要驱动参数**，不需要深度图或任何传感器输入。

---

#### `model.py` — FBV_SM（遗留基线）

来自 FBV-SM 原始论文，输入 (xyz + action) 拼接后直接 MLP，无时序建模。仅作参考对比。

---

#### 共享层：`layers.py`

| 组件 | 作用 |
|------|------|
| `PositionalEncoder` | 正弦余弦位置编码：`x → [x, sin(2^0·x), cos(2^0·x), ..., sin(2^{L-1}·x), cos(2^{L-1}·x)]` |
| `ActuatorMLPEncoder` | 动作参数 MLP 编码器，将低维动作映射到高维特征空间 |
| `MLPDecoder` | 通用解码 MLP：`input → 2d → 2d → d → d/2 → output`，density 用 softplus 激活 |
| `MultiScaleEMA` | 多尺度指数移动平均：用 N 个可学习衰减率分别做 EMA，再加权拼接 |
| `TemporalLSTMEncoder` | LSTM 时序编码器（旧版，已被 EMA 替代） |
| `SirenLayer`（model_sdf.py） | SIREN 层：`sin(w0 · (Wx + b))`，周期性激活天然适合光滑场学习 |


### 3.5 训练器文件

所有训练器继承 `BaseTrainer`（渲染、射线采样工具）。`trainer_sdf.py` 是唯一不继承 BaseTrainer 的独立训练器，因为它不需要体渲染。

| 文件 | 模型 | Phase 1 内容 | Phase 2 内容 |
|------|------|-------------|-------------|
| `trainer_mstnf.py` | MSTNF | — | 单阶段训练 |
| `trainer_cmstnf.py` | C-MSTNF | Canonical 场 (2D loss) | Deformation 场 (2D loss) |
| `trainer_ode_cmstnf.py` | ODE-CMSTNF | 同 C-MSTNF | 同 C-MSTNF + ODE 编码 |
| `trainer_smooth_cmstnf.py` | Smooth-CMSTNF | 同 C-MSTNF | 同 C-MSTNF + 正则化 |
| `trainer_ms_scnf.py` | MS-SCNF | 骨架回归 (3D loss) | 联合训练 (3D + 2D loss) |
| `trainer_depth_cmstnf.py` | **Depth-CMSTNF** | 同 C-MSTNF | **同 C-MSTNF + 深度 L1 loss + 深度引导采样** |
| `trainer_sdf.py` | **TemporalSDF** | — | **单阶段，3D SDF 监督（SDF + 法向量 + Eikonal loss）** |

### 3.6 配置文件：`src/config/training.json`

所有模型共享，关键参数：

| 节 | 用途 |
|----|------|
| `optimization` | 学习率、batch_size、epoch 数 |
| `temporal` | 时序窗口大小 (20)、EMA 尺度数 (4)、隐层维度 (128) |
| `loss_weights` | 重建/预测/平滑权重 |
| `model` | 位置编码频率数 (10)、隐层维度 (128) |
| `canonical` | Phase 1/2 epoch 数、变形学习率 |
| `ms_scnf` | **MS-SCNF 专用**：多尺度节点数 (4/10/31)、骨架 loss 权重 |

### 3.7 3D 评估指标：`src/training/metrics_3d.py`

| 函数 | 含义 |
|------|------|
| `mean_node_error(pred, gt)` | 所有人节点平均 L2 误差 |
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

# 多视角深度数据（collect_multiview.py --depth）
depth_maps_front: (T, H, W) 正面深度图
depth_maps_side:  (T, H, W) 侧面深度图
```

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

# 带 3D 标注的时序数据 — 用于 MS-SCNF 训练
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

### 5.2 C-MSTNF 系列（MSTNF / C-MSTNF / ODE-CMSTNF / Smooth-CMSTNF）

**第一步：采集数据**

```bash
# 零动作数据（Phase 1 用，仅 C-MSTNF 系列需要）
python scripts/data_collection/collect.py --action-x zero --action-y zero

# 时序动作数据（Phase 2 / MSTNF 训练用）
python scripts/data_collection/collect.py
```

**第二步：训练**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_mstnf.py      # MSTNF
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_cmstnf.py     # C-MSTNF
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_ode_cmstnf.py # ODE-CMSTNF
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_smooth_cmstnf.py
```

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
# 完整训练
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

### 5.4 深度增强模型（Depth-CMSTNF / RGB-D Neural Field）

**第一步：采集含深度图的数据**

```bash
# canonical 数据（Phase 1 用，Depth-CMSTNF 需要）
python scripts/data_collection/collect.py --action-x zero --action-y zero --depth

# 时序数据 + 深度图（Phase 2 / RGB-D 训练用）
python scripts/data_collection/collect.py --depth

# 多视角 + 深度图
python scripts/data_collection/collect_multiview.py --depth
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

### 5.5 快速验证 Notebook

```bash
jupyter notebook notebooks/06_linear_deform_test.ipynb
jupyter notebook notebooks/07_coarse_to_fine_freq.ipynb
```

---

## 6. 数据依赖关系图

```
                          elastica_env.py
                 ┌──────────────┼──────────────┐
                 │              │              │
          get_observation  get_observation_3d  get_observation_with_depth
                 │              │              │
                 ▼              ▼              ▼
           collect.py      collect.py      collect.py
           (default)       --3d            --depth
                 │              │              │
                 ▼              ▼              ▼
           data/seq_rr/   data/seq_rr_3d/ data/seq_rr/  (+ depth_maps)
           (img+act+cam)  (+ pos+radii)   (img+act+cam+depth)
                 │              │              │
          ┌─────┴─────┐   ┌────┴────┐   ┌─────┴─────┐
          │           │   │         │   │           │
          ▼           ▼   ▼         ▼   ▼           ▼
        MSTNF    C-MSTNF   Phase 1  Phase 2   Depth-CMSTNF
        (单阶段)  系列      骨架回归  联合训练   (深度监督)
          │           │     │         │         │           │
          └─────┬─────┘     └────┬────┘         └─────┬─────┘
                │                │                     │
                ▼                ▼                     ▼
          2D 渲染可视化    evaluate_3d.py          深度误差评估
          (人眼评估)      (定量 3D 指标)
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
- **根目录旧文件**：`env.py`、`func.py`、`train.py`、`predefined.py` 来自原始 FBV-SM 论文，与当前 PyElastica 管线无关，仅供参考。
