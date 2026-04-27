# SelfSoftRobot 项目帮助文档

本文档说明每个文件的作用、如何使用，以及每个模型从数据采集到训练评估的完整运行流程。

---

## 1. 项目整体逻辑

```
PyElastica 物理仿真 → PyVista 渲染图像 → 数据采集 (.npz) → 模型训练 → 形态预测 + 3D 评估
```

项目目标：**仅用驱动参数（扭矩）预测软体机器人的完整 3D 形态**。

当前有两套模型管线：

| 管线 | 模型 | 数据需求 | 输出 |
|------|------|---------|------|
| C-MSTNF 系列 | MSTNF / C-MSTNF / ODE-CMSTNF / Smooth-CMSTNF | 2D 图像 + 动作 | 2D 渲染图 |
| **MS-SCNF（新）** | MS-SCNF | **2D 图像 + 动作 + 3D 节点坐标** | **3D 骨架 + 2D 渲染图** |

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
│   │   └── model_ms_scnf.py         #   MS-SCNF（骨架条件神经场，新方法）
│   ├── data/
│   │   └── dataset.py               # SoftSequenceDataset（支持 2D/3D 数据）
│   ├── training/
│   │   ├── base.py                  #   BaseTrainer（渲染、射线采样等共享工具）
│   │   ├── two_phase_trainer.py     #   TwoPhaseTrainer（Phase1+Phase2 基类）
│   │   ├── trainer_mstnf.py         #   MSTNF 训练器
│   │   ├── trainer_cmstnf.py        #   C-MSTNF 训练器
│   │   ├── trainer_ode_cmstnf.py    #   ODE-CMSTNF 训练器
│   │   ├── trainer_smooth_cmstnf.py #   Smooth-CMSTNF 训练器
│   │   ├── trainer_ms_scnf.py       #   MS-SCNF 训练器（新）
│   │   ├── metrics_3d.py            #   3D 评估指标（新）
│   │   └── rendering.py             #   旧版渲染工具
│   ├── config/
│   │   ├── training.json            #   训练超参数（所有模型共享）
│   │   ├── camera.json              #   相机参数
│   │   └── simulation.json          #   仿真参数
│   └── utils/
│       ├── camera.py                #   get_rays（射线生成）
│       ├── rendering.py             #   OM_rendering, sample_stratified
│       ├── experiment.py            #   实验目录管理 + GIF 保存
│       └── visualization.py         #   可视化工具
│
├── scripts/
│   ├── data_collection/             # 数据采集
│   │   ├── collect.py               #   统一采集入口（per-dim 动作控制 + --3d）
│   │   └── collect_utils.py         #   动作策略、保存、命名工具函数
│   ├── training/                    # 训练入口脚本
│   │   ├── train_mstnf.py           #   MSTNF
│   │   ├── train_cmstnf.py          #   C-MSTNF
│   │   ├── train_ode_cmstnf.py      #   ODE-CMSTNF
│   │   ├── train_smooth_cmstnf.py   #   Smooth-CMSTNF
│   │   └── train_ms_scnf.py         #   MS-SCNF（新）
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
    ├── soft_robot_pipeline.md       #   技术管线与模型演进
    ├── improvement_proposals.md     #   改进方案与实施状态
    ├── literature_innovations.md    #   文献创新点总结
    └── paper_understanding.md       #   FBV-SM 论文详解
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

暴露的属性：`H, W, focal, action_dim`，以及 `get_camera_params()`（优先返回数据自带的相机参数，无则返回 None）。

**返回格式**（随参数不同）：

```
默认:               (seq, img)
return_pairs:       (seq_t, seq_t1, img_t, img_t1)
return_pairs+3d:    (seq_t, seq_t1, img_t, img_t1, pos_t, pos_t1)
return_3d:          (seq, img, positions)
```

其中 `positions` 形状为 `(3, 31)`，即 31 个节点的 xyz 坐标。

### 3.3 数据采集工具：`scripts/data_collection/collect_utils.py`

动作策略、数据保存、文件命名的工具函数，被 `collect.py` 调用。

| 类/函数 | 作用 |
|---------|------|
| `ActionSchedule` | 每个维度独立生成动作序列（zero/random/hold/file） |
| `save_collection()` | 保存 npz，始终嵌入相机参数 |
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

#### `model_mstnf.py` — MSTNF（基线时序模型）

- 核心：`MultiScaleEMA` 时序编码器，用可学习衰减率的 EMA 替代 LSTM
- 输入：动作窗口 (B, 20, 2) → 物理状态 (B, 128)
- 空间查询：位置编码 + 物理状态 + 当前动作 → [vis, density]
- **单阶段训练**，不需要 canonical 数据

#### `model_cmstnf.py` — C-MSTNF（Canonical + Deformation）

- D-NeRF 范式：`CanonicalField`（零动作静止态）+ `DeformationField`（动作条件变形）
- 查询流程：世界点 → 变形 MLP → canonical 坐标 → canonical 场 → [vis, density]
- **两阶段训练**：Phase 1 用零动作数据训练 canonical，Phase 2 冻结 canonical 训练变形

#### `model_ode_cmstnf.py` — ODE-CMSTNF

- 与 C-MSTNF 相同架构，仅将 MultiScaleEMA 替换为 Neural ODE
- ODE 积分保证状态轨迹连续，理论上可捕捉阻尼振荡

#### `model_smooth_cmstnf.py` — Smooth-CMSTNF

- 与 C-MSTNF 相同架构，变形 MLP 增加 spectral norm + Jacobian 惩罚
- 目的：限制变形场的 Lipschitz 常数，抑制高频跳变

#### `model_ms_scnf.py` — MS-SCNF（新方法）

- **骨架回归**：`SkeletonHead` 预测多尺度 3D 骨架（coarse 4节点 / medium 10节点 / fine 31节点）
- **骨架条件密度场**：`SkeletonConditionedDensity` 根据查询点到骨架曲线的距离计算密度
- 部署时 `model.predict_skeleton(action_window)` 直接输出 31 个 3D 坐标
- **两阶段训练**：Phase 1 骨架回归（3D loss），Phase 2 联合训练（3D + 2D loss）

### 3.5 训练器文件

所有训练器继承 `BaseTrainer`（渲染、射线采样工具）。

| 文件 | 模型 | Phase 1 内容 | Phase 2 内容 |
|------|------|-------------|-------------|
| `trainer_mstnf.py` | MSTNF | — | 单阶段训练 |
| `trainer_cmstnf.py` | C-MSTNF | Canonical 场 (2D loss) | Deformation 场 (2D loss) |
| `trainer_ode_cmstnf.py` | ODE-CMSTNF | 同 C-MSTNF | 同 C-MSTNF + ODE 编码 |
| `trainer_smooth_cmstnf.py` | Smooth-CMSTNF | 同 C-MSTNF | 同 C-MSTNF + 正则化 |
| `trainer_ms_scnf.py` | MS-SCNF | **骨架回归 (3D loss)** | **联合训练 (3D + 2D loss)** |

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

### 5.4 快速验证 Notebook

```bash
jupyter notebook notebooks/06_linear_deform_test.ipynb
jupyter notebook notebooks/07_coarse_to_fine_freq.ipynb
```

---

## 6. 数据依赖关系图

```
                     elastica_env.py
                      ┌───────────────┐
             ┌────────┤ get_observation├────────┐
             │        └───────────────┘        │
             │            2D                    │  get_observation_3d()
             │         (default)                │  (with --3d)
             ▼                                  ▼
  collect.py --action-x/y ...           collect.py --action-x/y ... --3d
             │                                  │
             ▼                                  ▼
  data/seq_zz/                        data/seq_zz_3d/
  data/seq_rr/                        data/seq_rr_3d/
  data/seq_rz/                        data/seq_rz_3d/
  (images + actions + camera)         (+ positions + radii)
       │                                      │
  ┌────┴─────┐                         ┌──────┴──────┐
  │          │                         │             │
  ▼          ▼                         ▼             ▼
MSTNF   C-MSTNF系列               Phase 1       Phase 2
(单阶段) (需 zz 数据)             骨架回归      联合训练
  │          │                         │             │
  └────┬─────┘                         └──────┬──────┘
       │                                      │
       ▼                                      ▼
  2D 渲染可视化                          evaluate_3d.py
  (人眼评估)                            (定量 3D 指标)
```

---

## 7. 常见注意点

- **统一采集**：所有数据采集使用 `collect.py`，通过 `--action-x/--action-y` 独立控制每个维度。每个维度可选 zero/random/hold/file 模式。
- **数据自描述**：新采集的数据始终包含相机参数（camera_eye/center/up/focal/H/W），训练器优先使用数据自带参数，旧数据自动回退到 `camera.json`。
- **参数来源**：`collect.py` 的默认值从 `simulation.json` + `camera.json` 读取，CLI 参数可覆盖。
- **文件命名**：输出文件名包含模式标签（如 `zz`、`rr`、`rz`）和 3D 标记，一目了然。
- **保存目录**：自动推断为 `data/seq_{模式标签}[_3d]/`，可用 `--save-dir` 覆盖。
- **GPU 选择**：各训练脚本顶部都有 `CUDA_DEVICE` 变量，修改为可用 GPU 编号。
- **动作归一化**：训练时自动计算归一化因子并保存到 `action_norm_factor.txt`，推理时需加载。
- **根目录旧文件**：`env.py`、`func.py`、`train.py`、`predefined.py` 来自原始 FBV-SM 论文，与当前 PyElastica 管线无关，仅供参考。
