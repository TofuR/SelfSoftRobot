# 深度信息集成设计

**日期：** 2026-05-07
**状态：** 设计已批准，待实施

## 概述

当前项目的 neural field 训练仅使用 2D 二值图像作为监督信号，导致相机深度方向（Z轴）的重建精度很差。本设计引入深度信息，从数据收集到模型训练全面重构，以提升 3D self-modeling 的深度估计精度。

两条并行路线：
- **方案 A**：Depth-supervised CMSTNF（在现有架构上添加深度监督）
- **方案 C**：全新 RGB-D Neural Field（从头设计深度原生架构）

## 目标

1. 仿真环境中验证深度信息对 3D 重建精度的提升
2. 设计兼容真实深度相机（如 RealSense D435）的数据格式和接口
3. 两条路线独立开发，各自独立分支

## 共享基础设施

### 深度数据收集

**改动位置：** `elastica_env.py` 的 PyVista 渲染部分

当前渲染只输出 binary image。PyVista 支持 z-buffer 读取，需提取逐像素深度。

**深度图定义：** 从相机光心沿射线方向到最近物体表面的距离（z-buffer depth），与 RealSense D435 输出格式一致。

**保存格式：** 现有 `.npz` 文件新增 `depth_maps` 数组（shape: (T, H, W), dtype: float32，单位：米）。

**数据收集脚本改动：** `collect_multiview.py` 和 `collect.py` 均需增加深度图提取和保存。

### DepthProvider 接口

抽象深度数据来源，方便 sim-to-real 迁移：

```python
class DepthProvider(Protocol):
    def get_depth_map(self) -> np.ndarray: ...  # (H, W) float32
    def get_rgb_image(self) -> np.ndarray: ...  # (H, W) uint8

class SimulationDepthProvider:  # 从 PyVista z-buffer
class RealSenseDepthProvider:   # 从 RealSense SDK（未来）
```

### 深度可视化工具

- 深度图 heatmap 可视化
- 深度误差图（预测 vs GT）
- 深度一致性检查工具

---

## 方案 A：Depth-supervised CMSTNF

**分支：** `feat/depth-supervised-cmstnf`

### 架构

保持 CMSTNF 原有架构不变：
- Canonical field MLP
- Deformation field MLP
- Actuator encoder
- Temporal encoder (EMA)

### Volume Rendering 扩展

在现有 OM rendering 基础上增加深度渲染分支：

```
E[d] = Σ_i w_i × z_i
w_i = α_i × T_i
```

与 RGB 渲染共享相同的权重 w_i，不增加网络参数。

### 深度引导采样（两阶段）

1. 第一轮均匀采样 32 点
2. 根据 coarse density 估计物体大致深度
3. 第二轮在深度附近 ±0.1m 精细采样 32 点
4. 总采样数 64 不变，但集中在表面附近

### Loss 设计

```
L_total = L_img + λ_d × L_depth + L_skeleton

L_depth = |E[d] - d_gt|  (L1 loss)
λ_d = 0.1 (初始值，可调)
```

### 训练流程

- Phase 1 canonical field：不变（3D position loss）
- Phase 2 deformation field：额外加入 depth loss

### 改动文件清单

| 文件 | 改动 |
|------|------|
| `elastica_env.py` | 添加深度图渲染 |
| `src/data/dataset.py` | 加载 depth_maps |
| `src/utils/rendering.py` | 添加深度渲染函数 |
| `src/models/model_cmstnf.py` | 无改动 |
| `src/training/CMSTNFTrainer` | 添加 depth loss 和深度引导采样 |
| `scripts/data_collection/collect.py` | 深度图保存 |
| `scripts/data_collection/collect_multiview.py` | 深度图保存 |

---

## 方案 C：RGB-D Neural Field

**分支：** `feat/rgbd-neural-field`

### 架构

全新设计，原生处理 RGB-D 输入：

```
深度图 → Depth Encoder (4-Conv CNN + FC) → f_depth
驱动输入 → Actuator Encoder → f_act
f_condition = concat(f_depth, f_act)

f_condition → Canonical Field MLP → [density, color]
Temporal Encoder → Deformation Field MLP → offset
```

### Depth Encoder

```python
DepthEncoder(
    Conv2d(1, 16, 3, stride=2), ReLU,   # H/2
    Conv2d(16, 32, 3, stride=2), ReLU,   # H/4
    Conv2d(32, 64, 3, stride=2), ReLU,   # H/8
    Conv2d(64, 128, 3, stride=2), ReLU,  # H/16
    AdaptiveAvgPool2d(1),
    Linear(128, 64),                     # f_depth
)
```

### Depth-aware Volume Rendering

基于深度图预测采样分布，替代均匀采样：
1. 深度图 → depth distribution（在每个像素的射线上预测高斯分布参数 μ, σ）
2. 在 μ ± 2σ 范围内 importance sampling
3. 结合 uniform sampling 保证覆盖

### 多任务 Loss

```
L_total = L_img + λ_d × L_depth + λ_n × L_normal + λ_s × L_smooth

L_depth  = |E[d] - d_gt|              (深度监督)
L_normal = 1 - cos(n_pred, n_gt)      (法向量一致性)
L_smooth = |∇² d_pred|                (深度平滑正则化)

λ_d = 1.0, λ_n = 0.1, λ_s = 0.01
```

**Surface Normal 计算：**
- `n_gt`：从 GT 深度图计算（中心差分 → 归一化）
- `n_pred`：从 density field 梯度计算（∂σ/∂x, ∂σ/∂y, ∂σ/∂z）

### 训练策略

单阶段端到端训练，不需要两阶段。深度信息从第一轮就参与。

### 改动文件清单

| 文件 | 改动 |
|------|------|
| `elastica_env.py` | 添加深度图渲染（与方案 A 共享） |
| `src/models/model_rgbd.py` | 新建：RGB-D Neural Field 模型 |
| `src/models/layers.py` | 添加 DepthEncoder |
| `src/utils/rendering.py` | 深度渲染 + depth-aware sampling |
| `src/data/dataset.py` | 加载 depth_maps |
| `src/training/RGBDTrainer.py` | 新建：RGB-D 训练器 |
| `scripts/training/train_rgbd.py` | 新建：训练入口脚本 |

---

## 分支策略

```
master
├── feat/depth-supervised-cmstnf   # 方案 A
│   └── 共享基础设施 + CMSTNF 深度监督
└── feat/rgbd-neural-field         # 方案 C
    └── 共享基础设施 + 全新 RGB-D 架构
```

先在 master 完成共享基础设施提交，再分别开分支。

## 成功指标

1. 深度方向重建误差降低 30%+（相比当前纯 2D 方案）
2. 3D skeleton 位置误差在 Z 轴显著改善
3. 方案 A 和方案 C 的对比实验数据
4. 深度可视化结果清晰可解释
