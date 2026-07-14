# SelfSoftRobot 项目帮助（精简版）

本文档是 [`docs/PROJECT_HELP.md`](../archived/PROJECT_HELP.md)（1710 行全量版）的**核心参考索引**，目的：**快速找命令、快速理解架构**。不是教程，逐步操作与变更历史请看全量版。

## 本文档涵盖什么

- 项目概述与两条数据路线（仿真 A + 实物免标定 B）
- 环境安装与关键运行入口（采集 / 训练 / 评估 / 可视化）
- 源码布局（`src/` 各目录职责）
- 模型架构表（仿真神经场族 + 实物状态转移族）
- 训练架构（Spec 声明式 + UnifiedTrainer）与关键约定

## 相关文档

| 主题 | 路径 |
|------|------|
| 全量文件说明与运行流程 | [`docs/PROJECT_HELP.md`](../archived/PROJECT_HELP.md) |
| 项目概览（给 AI 助手） | [`CLAUDE.md`](../../CLAUDE.md) |
| 实物数据采集硬件细节 | `docs/real_data/capture_setup.md`（硬件构成见下文一句话） |
| 实物数据流水线完整流程 | `docs/real_data/workflow.md`（或 [`docs/archived/research/2026-07-10-real-data-2d-workflow.md`](../archived/research/2026-07-10-real-data-2d-workflow.md)） |
| 技术管线与模型演进 | [`docs/overview/pipeline.md`](pipeline.md) |
| 研究方向文档 | [`docs/directions/`](../directions/) |

---

## 1. 项目概述

**目标**：仅用驱动参数（扭矩 / 气压）预测软体机器人的完整形态（3D 或 2D 骨架），实现**神经场自建模**。基于 FBV-SM（Hu et al. 2025）代码库，从刚性臂扩展到 PyElastica 软体连续臂。

### 两条数据路线

| 路线 | 仿真/采集 | 标定 | state 表示 | 监督 | 度量验证 |
|------|----------|------|-----------|------|---------|
| **(A) 仿真**（原始主线，仍有效） | PyElastica 物理 + PyVista 渲染 | 多视角相机标定 + 度量 3D 内参 | 3D 节点 `positions (T,3,31)` | 体渲染 / 3D SDF 直接监督 | 仿真 GT 直接对比 |
| **(B) 实物免标定**（当前主线） | 真实相机（RealSense D400） | **免标定**，无内参 / 无度量 3D | **2D 图像骨架 `[col,row,0]`**（像素，z≈0） | 2D 骨架回归（无体渲染） | **NDI 6DOF tracker** 末端 mm |

实物硬件一句话：1-DOF 双段硅胶臂，TwinCAT PLC+电机推注射器气动（真实控制量=电机位置 mm），单 RealSense 相机，NDI 末端追踪。详见 `docs/real_data/capture_setup.md`。

### 核心思路（所有模型共性）

```
驱动参数 ──→ 时序编码器 ──→ 物理状态向量
                              │
3D 查询点 ──→ 位置编码 ──────→ 空间解码器 ──→ [vis, density] 或 SDF
                              │              │
                         当前动作 ──────────→│
                                           ↓
                                    体渲染 → 2D 图像（训练监督，sim 路线）
```

> **核心约定**：模型输入**只有驱动参数 + 查询点**。图像 / 深度仅作监督信号，不直接输入模型。

---

## 2. 环境安装与运行入口

### 环境

```bash
pip install -r requirements.txt
```

关键依赖：PyTorch 2.6、PyElastica（`elastica`）、PyVista、OpenCV。**训练需要 CUDA GPU**。GPU 通过环境变量指定：`CUDA_VISIBLE_DEVICES=N python ...`。

无正式测试套件，验证通过 notebook（`notebooks/`）与评估脚本完成。

### 2.1 仿真数据采集（路线 A）

统一入口 `scripts/data_collection/collect.py`，每个动作维度独立控制（`zero` / `random` / `hold` / `file`）：

```bash
python scripts/data_collection/collect.py                              # 默认：两维 random → data/seq_rr/
python scripts/data_collection/collect.py --action-x zero --action-y zero   # canonical → data/seq_zz/
python scripts/data_collection/collect.py --3d                         # +3D 节点坐标 → data/seq_rr_3d/
python scripts/data_collection/collect.py --depth                      # +深度图
python scripts/data_collection/collect.py --3d --depth                 # +3D + 深度
python scripts/data_collection/collect.py --multiview --depth          # 多视角 + 深度 → data/exp7_multiview/
```

保存目录自动推断为 `data/seq_{模式标签}[_3d]/`，可用 `--save-dir` 覆盖。模式标签：每维度取首字母（z=zero, r=random, h=hold, f=file）。

### 2.2 仿真训练（统一入口，路线 A）

`scripts/training/train_unified.py` 支持全部神经场模型：

```bash
# MSTNF（单阶段，rendering）
python scripts/training/train_unified.py --model mstnf --data_dir data/sequence_data

# C-MSTNF（两阶段：canonical→deformation）
python scripts/training/train_unified.py --model cmstnf --data_dir data/sequence_data \
    --canonical_data_dir data/canonical_data

# MS-SCNF（两阶段：skeleton→rendering）
python scripts/training/train_unified.py --model ms_scnf --data_dir data/seq_rr_3d

# TemporalSDF（单阶段，direct_3d，无需图像）
python scripts/training/train_unified.py --model sdf --data_dir data/seq_rr_3d

# SkeletonSDF（两阶段，direct_3d，无需图像）
python scripts/training/train_unified.py --model skeleton_sdf --data_dir data/seq_rr_3d

# 多视角 + 深度 + 一致性
python scripts/training/train_unified.py --model cmstnf --data_dir data/exp7_multiview \
    --multiview --depth --consistency
```

各模型原有脚本（`train_mstnf.py` 等）是 UnifiedTrainer 的薄包装，CLI 不变。

**超参数搜索**：`train_search.py` 子进程调用 `train_unified.py`，支持 `--search lr=1e-4,1e-3` 网格、`--dry_run`、`--resume`、`--summarize`。

### 2.3 仿真评估与可视化

```bash
python scripts/evaluation/evaluate_3d.py --model_type <m> --checkpoint <pt> --data_dir <d>  # 3D 定量指标
python scripts/evaluation/visualize_predictions.py compare    # 预测对比
python scripts/evaluation/visualize_predictions.py animate    # GIF 动画
python scripts/evaluation/visualize_3d_shape.py --model_type <m> --checkpoint <pt>  # 3D SDF/mesh
```

### 2.4 实物管线（路线 B，免标定 2D）

完整流程一句话：照片 → 分割（white_on_blue / SAM2） → mask → 修复（`repair_masks` 三步 / SAM2） → 骨架化（逐行质心 + 弧长重采样 + tip_fix） → npz → clean（outlier + stabilize_static） → 训练。详见 `docs/real_data/workflow.md`。

```bash
# --- scripts/real/ : mask + 骨架准备 ---
python scripts/real/masks_to_transition_npz.py --seq real_capture/data/raw/<seq>   # mask → 2D 骨架 npz + tip_fix + action 归一[0,1]
python scripts/real/clean_transition_npz.py --seq <seq>          # 静态段共识清洗 → data/real_seq/<seq>_clean/
python scripts/real/repair_masks.py                              # mask 级修复（独立轨道）→ masks_repaired/
python scripts/real/composite_frames.py                          # 原图 + mask + 骨架叠图
python scripts/real/compare_skeleton_methods.py                  # 7 法末端 corner 对比
python scripts/real/skeleton_to_shape.py                         # node→形态 半径偏移基线

# --- 训练（统一入口，--mode gt|open_loop）---
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt \
    --data_dir data/real_seq/<seq>_clean/train
# open_loop（热启动自最新 gt）：--mode open_loop

# --- 评估与可视化 ---
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
    --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
    --data_dir data/real_seq/<seq>_clean/train      # 末端 NDI mm + 形态 px + drift_by_k
python scripts/evaluation/visualize_real_overlay.py ...  # 模型预测叠真实照片
python scripts/evaluation/inspect_real_data.py ...       # 骨架网格诊断
```

**坐标空间**（关键）：骨架 GT / 模型预测 / mask 都在**像素 `[col,row]`, z≈0**；NDI 末端在**毫米**。免标定 → 不用相机矩阵投影。整体形态误差只能算 px；末端误差 px + mm 都能算（mm 通过 NDI↔GT node0 px 最小二乘拟合 2D 仿射 `A: (col,row,1)→(x,y)`）。

当前默认训练数据：`data/real_seq/seq_20260627_163921_n15_rep_clean`（15 节点，repaired mask + clean）。

---

## 3. 源码布局（`src/`）

```
SelfSoftRobot/
├── elastica_env.py                  # PyElastica 仿真环境 + PyVista 渲染（路线 A 数据源）
├── config/                          # 配置文件（根目录）
│   ├── training.json                #   训练超参数（所有模型共享）
│   ├── camera.json / simulation.json #  相机 / 仿真参数
│   └── params.py                    #   YAML 配置加载
└── src/
    ├── encoders/          # 时序编码器
    │   ├── multi_scale_ema.py     #   MultiScaleEMA（多尺度指数移动平均）
    │   ├── fractional_memory.py   #   FractionalMemory（分数阶幂律记忆核，物理驱动 EMA 替代品）
    │   ├── gamma_laguerre.py      #   GammaLaguerreMemory（Gamma 延迟峰记忆核）
    │   ├── temporal_gru.py        #   TemporalGRU
    │   ├── temporal_transformer.py #  TemporalTransformer（CLS 自注意力）
    │   └── temporal_tcn.py        #   TemporalTCN（因果膨胀卷积）
    ├── fields/            # 神经场模块
    │   ├── canonical.py           #   CanonicalField（规范场）
    │   ├── deformation.py         #   DeformationField（变形场）
    │   └── skeleton_density.py    #   SkeletonConditionedDensity（骨架局部柱坐标条件密度场）
    ├── heads/             # 回归头
    │   └── skeleton_heads.py      #   4 种骨架参数化（point/fourier/bspline/catmullrom）+ 工厂函数
    ├── rendering/         # 渲染策略
    │   └── view_strategy.py       #   SingleViewStrategy / MultiViewStrategy
    ├── evaluation/        # 评估工具
    │   ├── query.py               #   模型查询（预测骨架/SDF/密度）
    │   └── render.py              #   可视化渲染（mesh/pointcloud/animation）
    ├── models/            # 模型定义（见 §4）
    ├── training/          # 训练基础设施（见 §5）
    ├── data/              # 数据集类（见下）
    ├── config/args.py     # CLI 参数定义
    └── utils/             # 共享工具
        ├── rendering.py           #   体渲染（OM rendering, ray sampling, depth-guided）
        ├── camera.py / camera_system.py  #  get_rays / MultiCameraSystem（多相机投影反投影）
        ├── sdf_utils.py           #   GT SDF 解析计算（dist_to_skeleton - radius）
        ├── model_loader.py        #   自动检测模型类型 + 加载 checkpoint
        ├── skeleton_2d.py         #   2D 骨架提取 + _perpendicular_tip_fix（路线 B）
        ├── skeleton_viz.py        #   3D 骨架可视化
        ├── experiment.py          #   实验目录管理 + GIF
        └── config_utils.py        #   CLI 覆盖 + 配置合并
```

### 数据集类（`src/data/`）

| 文件 | 类 | 用途 |
|------|----|------|
| `dataset.py` | `SoftSequenceDataset` | 仿真序列，支持 2D/3D/深度/pairs |
| `dataset_sdf.py` | `SDFDataset` | 3D SDF 监督采样（表面/近表面/离表面） |
| `dataset_skeleton_sdf.py` | `SkeletonSDFDataset` | 骨架 + SDF 联合采样 |
| `dataset_spatial.py` | `SpatialSequenceDataset` | 中心线回归（31 节点） |
| `dataset_multiview.py` | `MultiViewDataset` | 旧版双视角 |
| `dataset_multiview_depth.py` | `MultiViewDepthDataset` | 新版多视角 + 深度（兼容新旧 npz） |

### scripts 目录

| 子目录 | 关键脚本 |
|--------|---------|
| `scripts/data_collection/` | `collect.py`（统一采集） |
| `scripts/training/` | `train_unified.py`（仿真神经场，5 模型）、`train_transition.py`（实物统一入口）、`train_search.py`（网格搜索）、各 `train_*.py` 薄包装 |
| `scripts/evaluation/` | `evaluate_3d.py`、`eval_real_quant.py`、`visualize_*.py`、`inspect_real_data.py` |
| `scripts/real/` | `masks_to_transition_npz.py`、`clean_transition_npz.py`、`repair_masks.py`、`composite_frames.py` 等（路线 B 全流水线） |
| `scripts/experiments/` | `exp1`–`exp7` 实验脚本 |

### 数据布局

```
data/                     # 仿真（gitignore）
  seq_zz/ seq_zz_3d/      # canonical（两维 zero）
  seq_rr/ seq_rr_3d/      # 时序（两维 random）
  seq_rz/ seq_hh/         # 单维随机 / hold batch
  seq_rz_c2_sk/           # 空间序列模型训练用
  exp7_multiview/         # 多视角实验

real_capture/data/        # 实物原始采集
  raw/<seq>/{cam0/*.png, actions6.csv, ndi.csv, frame_times.txt, meta.json}
  derived/<seq>/{masks, masks_repaired, overlay}
data/real_seq/<seq>/      # 实物 transition npz（train/val）：positions(T,3,N), actions(T,1)
data/real_seq/<seq>_clean/  # 清洗后
```

每个 transition npz 携带元数据（含 `n_points`、`tip_fix`）于 `data_prep`；训练实验保存 `config.json`（n_nodes/z_dim/episode_len + data_prep）和一行 `model_card.txt`。

---

## 4. 模型架构（`src/models/`）

### 4.1 仿真神经场族（路线 A）

| 模型 | 文件 | 核心创新 | 监督模式 | 阶段 |
|------|------|---------|---------|------|
| **FBV_SM** | `model.py` | 原始论文基线（xyz+action 直接 MLP） | rendering | 1 |
| **MSTNF** | `model_mstnf.py` | MultiScaleEMA 多尺度时序编码 | rendering | 1 |
| **C-MSTNF** | `model_cmstnf.py` | D-NeRF 范式：Canonical + Deformation 双场 | rendering | 2（canonical→deformation） |
| **MS-SCNF** | `model_ms_scnf.py` | 显式骨架回归 + 骨架条件密度场（局部柱坐标 `dist, t_axial`） | skeleton→rendering | 2（skeleton→joint） |
| **TemporalSDF** | `model_sdf.py` | SIREN 坐标编码 + 3D SDF 直接监督（无需图像） | direct_3d | 1 |
| **SkeletonSDF** | `model_skeleton_sdf.py` | 参数化骨架 + 管状 SDF 先验 + SIREN 残差 | direct_3d | 2（skeleton→joint） |

**变体（无独立模型文件，复用 C-MSTNF）**：
- **Depth-CMSTNF**：深度 L1 loss + 深度引导采样（coarse-to-fine），推理只需驱动参数。
- **多视角+深度训练**：`MultiViewStrategy` 多视角 rendering 求和 + 深度 + reproj + consistency loss。

**已归档**：ODE-CMSTNF、Smooth-CMSTNF 在 `docs/archived/`。

### 4.2 实物状态转移族（路线 B，当前主线）

学**状态转移** `ŝ_t = F(s_{t-1}, a_t, z_{t-1})` 而非前馈稳态推断。FractionalMemory 编码动作历史（匹配粘弹性迟滞）；**可学习迟滞潜变量 z**（GRUCell 跨帧演化，无 GT 端到端学）；沿臂空间 GRU 逐节点传播；Δ 预测 `s_t = s_{t-1} + delta_scale·tanh(Δ)`。

| 模型 | 文件 | 前一状态来源 | z | s 误差累积 | 定位 |
|------|------|------------|---|-----------|------|
| **SpatialSequence** | `model_spatial_sequence.py` | 无（纯前馈稳态） | 无 | — | 基线（稳态假设） |
| **PCSpatial** | `model_pc_spatial.py` | 无（预测 + 图像残差修正） | 无 | — | sim-to-real 两阶段（预测→修正） |
| StateTransition | `model_state_transition.py` | 自身预测（rollout） | 跨步演化 | 严重（漂移 1000×） | 未来扩展（无法每步观测） |
| **GTObservedTransition** | `model_gt_transition.py` | **真实观测** | **窗口内演化** | **无（s 每步真实）** | **当前主线** |

**GTObservedTransition（主线）**：继承 StateTransition，复用全部 forward / z_module，仅固化 `training_spec`（`teacher_forcing_ratio=1.0` + `episode_len=40` 窗口 + dense supervision，每步预测都算 loss 给无 GT 的 z 直接梯度）。样本自包含（z 不跨样本）→ **样本间可打乱**。无数据泄漏：预测 `ŝ_{j+1}` 只用 ≤s_j 历史，GT s_{j+1} 不在预测路径。

### 4.3 共享层（`layers.py`）

| 组件 | 作用 |
|------|------|
| `PositionalEncoder` | 正余弦位置编码 |
| `ActuatorMLPEncoder` | 动作参数 MLP 编码器 |
| `MLPDecoder` | 通用解码 MLP（density 用 softplus） |
| `MultiScaleEMA` / `FractionalMemory` / `GammaLaguerreMemory` | 三种时序记忆核（接口兼容，`--encoder fractional` 切换） |

### 4.4 骨架回归头（`skeleton_heads.py`）

| 类 | 参数化 | 特点 |
|----|--------|------|
| `SkeletonHead` | point | 独立预测每节点，最灵活 |
| `FourierSkeletonHead` | fourier | 截断级数，带限光滑 |
| `BSplineSkeletonHead` | bspline | 三次 B-spline，局部控制 + C² 连续 |
| `CatmullRomSkeletonHead` | catmullrom | 插值型，通过控制点 |

统一接口 `forward(physics_state) → {'coarse','medium','fine'}`，工厂函数 `create_skeleton_head()`。

---

## 5. 训练架构：Spec 声明式系统

所有模型通过 `training_spec` 类属性声明训练需求，`UnifiedTrainer` 统一解释执行。**无需为每模型写 Trainer 子类**。

### 三个正交维度

| 维度 | 机制 | 负责 |
|------|------|------|
| **Phase 策略** | `PhaseSpec` + `PhaseStrategy` | 阶段数、冻结、forward、epochs |
| **监督模式** | `supervision_mode` | `"rendering"` / `"direct_3d"` / `"skeleton"` / `"spatial_sequence"` |
| **视角策略** | `ViewStrategy` | 单视角 / 多视角 / 跨视角约束 |

### 三种监督模式

| 模式 | 前向流程 | 适用模型 |
|------|---------|---------|
| `"rendering"` | rays → 3D points → model(pts, action) → 体渲染 → 像素对比 | MSTNF, C-MSTNF, MS-SCNF Phase 2 |
| `"direct_3d"` | coords → model(coords, action) → 值对比（SDF/法向量） | TemporalSDF, SkeletonSDF |
| `"skeleton"` / `"spatial_sequence"` | action → predict_skeleton → 骨架对比 | MS-SCNF Phase 1, 空间序列族 |

### 核心文件（`src/training/`）

| 文件 | 作用 |
|------|------|
| `spec.py` | `PhaseSpec` / `TrainingSpec` 数据类 |
| `phase_strategy.py` | 解析 spec，管理冻结/解冻/forward |
| `dataset_factory.py` | 按 `dataset_type` 创建数据集 + collate → dict batch |
| `trainer_unified.py` | **UnifiedTrainer**，组合 PhaseStrategy + ViewStrategy |
| `base.py` | `BaseTrainer`（渲染、射线采样等共享工具，legacy） |
| `metrics_3d.py` | `mean_node_error` / `endpoint_error` / `chamfer_distance` / `curve_smoothness` |

### 各模型 training_spec 速查

| 模型 | 阶段 | 监督模式 | 活跃 Loss |
|------|------|---------|----------|
| MSTNF | 1 | rendering | recon, smooth |
| C-MSTNF | 2（canonical→deformation） | rendering→rendering | [recon] → [recon, smooth] |
| MS-SCNF | 2（skeleton→joint） | skeleton→rendering | [skeleton] → [skeleton, recon, smooth] |
| TemporalSDF | 1 | direct_3d | sdf, normal, eikonal |
| SkeletonSDF | 2（skeleton→joint） | direct_3d→direct_3d | [skeleton] → [skeleton, sdf, normal, eikonal] |
| SpatialSequence | 1 | spatial_sequence | skeleton, spatial_smooth, smooth |
| StateTransition | 1（单帧） | spatial_sequence | skeleton, spatial_smooth, smooth |
| GTObservedTransition | 1（窗口 TF=1.0） | spatial_sequence（窗口） | skeleton(dense), spatial_smooth, smooth |

---

## 6. 配置与关键约定

### 配置文件

| 文件 | 内容 |
|------|------|
| `config/training.json` | 共享超参：optimization / normalization / temporal / loss_weights / model / canonical / ms_scnf / multiview / sdf / evaluation |
| `config/camera.json` | 相机参数（旧数据无自带参数时回退） |
| `config/simulation.json` | 仿真参数（dt=0.0001s, steps_per_action=500, record_interval=50） |
| `src/config/args.py` | CLI 参数定义，运行时覆盖 JSON 默认值 |

### 数据字段（.npz，仿真）

```
images: (T,H,W) | (T,V,H,W)        # 二值图（或多视角数组，新格式）
actions: (T,2)                      # 扭矩 [torque_x, torque_y]
dt, focal, H, W, camera_eye/center/up   # 始终嵌入，数据自描述
positions: (T,3,31)   radii: (T,31) # --3d 模式
depth_maps: (T,H,W)                 # --depth 模式（float32 米）
```

**动作-时序对应**：1 动作值 = 500 步 × dt = 50ms → 10 帧记录（每 50 步一帧）。`window_size=40`（默认）≈ 200ms 历史 ≈ 4 个不同动作值。动作值在 hold 期间重复（同值产生 10 帧，但物理状态在演化）。

### 关键约定（务必遵守）

- **模型输入**：只有驱动参数 + 查询点。图像 / 深度仅作监督信号。
- **实验日志**：`train_log/<model_name>/exp_<date>_<n>/`，含 images、best model、loss log。GPU 通过 `CUDA_VISIBLE_DEVICES` 指定。
- **统一训练**：所有模型用 `UnifiedTrainer` + `training_spec` 类属性 + `compute_losses()` 方法；新增模型无需写 Trainer。
- **模型加载**：用 `src/utils/model_loader.py`（自动检测类型）。
- **数据自描述**：新数据始终含相机参数，训练器优先用数据自带参数，旧数据回退 `camera.json`。
- **动作归一化**：训练时自动算 `action_norm_factor.txt`，推理需加载。
- **两阶段训练**：C-MSTNF / MS-SCNF / SkeletonSDF 先训 canonical/skeleton，再联合训 deformation/SDF。
- **实物免标定**：state 在图像像素 `[col,row,0]`，z≈0（平面假设）；不用相机矩阵投影。整体形态误差算 px，末端误差 px + mm。
- **代码语言**：注释 / 文档中英混排，`docs/` 主要中文。

---

## 7. 常见问题速查

| 需求 | 命令 / 文件 |
|------|------------|
| 找某个模型怎么训练 | §2.2（仿真）或 §2.4（实物） |
| 找某个文件做什么 | §3 源码布局 |
| 理解模型架构 | §4 模型表 |
| 理解训练阶段 / loss | §5 training_spec 速查 |
| 修改超参 | `config/training.json` 或 CLI 覆盖（`src/config/args.py`） |
| 加载已有 checkpoint | `src/utils/model_loader.py` |
| 多视角训练 | `train_unified.py --multiview --depth [--consistency]` |
| 实物降节点数 | `masks_to_transition_npz --n-points N`（全流水线按 N 分数自适应） |
| gt vs open_loop 对比 | `train_transition --mode` + `eval_real_quant --mode` |
| 实物末端 mm 验证 | `eval_real_quant.py`（NDI 仿射自标定） |
| 旧 Trainer 在哪 | `docs/archived/trainers/`（hook_based / standalone / multiview） |
| 根目录 `env.py`/`train.py` 等 | FBV-SM 原始论文遗留，与当前管线无关，仅供参考 |
