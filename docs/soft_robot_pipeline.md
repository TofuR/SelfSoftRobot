# 软体机器人自建模 — 项目技术文档

> 本文档介绍项目的技术管线、模型设计思路、以及从基线方法到 MS-SCNF 的演进过程。
> 具体文件说明和运行命令见 [PROJECT_HELP.md](PROJECT_HELP.md)。

---

## 1. 核心问题

**目标**：给定驱动参数（扭矩）的时序历史，预测软体机器人的完整 3D 形态。

软体机器人的特殊性在于：
- **连续体**：不是刚性关节的串联，而是连续变形的弹性体
- **时序依赖**：当前形态不仅取决于当前扭矩，还取决于历史扭矩（惯性与阻尼）
- **3D 形状**：最终目标是获取 3D 空间中的形态，而非仅仅 2D 投影图像

---

## 2. 技术管线

```
PyElastica 物理仿真 (CosseratRod, 30 单元, 0.5m)
        ↓
PyVista 渲染 (单相机, 100×100, FOV=42°)
        ↓
数据采集 (collect.py, per-dim 动作控制, npz 含相机参数)
        ↓
模型训练 (MSTNF / C-MSTNF 系列 / MS-SCNF)
        ↓
形态预测 + 3D 评估 (MNE, EPE, Chamfer, Smoothness)
```

### 2.1 仿真环境

| 参数 | 值 |
|------|-----|
| 杆体模型 | CosseratRod (PyElastica) |
| 单元数 | 30（31 个节点） |
| 长度 / 半径 | 0.5 m / 0.015 m |
| 驱动方式 | 2D 分布扭矩 (torque_x, torque_y) |
| 阻尼 | 0.1, 渐升时间 0.5s |
| 底端约束 | OneEndFixedBC |

### 2.2 数据格式

所有数据自描述（相机参数嵌入 npz）：

```
基础: images(T,H,W), actions(T,2), dt, focal, H, W, camera_eye/center/up
3D:   + positions(T,3,31), radii(T,31)
```

---

## 3. 模型架构与演进

### 3.1 演进路线

```
FBV-SM (论文基线)
  │  问题：静态模型，无时序建模
  ▼
MSTNF (MultiScaleEMA 时序编码)
  │  改进：EMA 替代 LSTM 编码动作窗口
  │  问题：单阶段，无 canonical 结构
  ▼
C-MSTNF (Canonical + Deformation)
  │  改进：D-NeRF 范式，形状与变形解耦
  │  问题：变形 MLP 产生高频跳变
  ├── ODE-CMSTNF：Neural ODE 替代 EMA → 无显著提升
  ├── Smooth-CMSTNF：Spectral Norm 正则化 → 尖端发散
  ▼
MS-SCNF (多尺度骨架条件神经场)
     改进：显式骨架回归 + 骨架条件密度场 + 3D GT 监督
     优势：直接输出 3D 骨架，物理约束保证平滑性
```

### 3.2 各模型核心设计

#### FBV-SM（论文基线）

原始论文方法。将 NeRF 的视角条件替换为关节角度条件。

```
输入: (3D 坐标, 关节角度) → PosEncoder + CmdEncoder → FeedForward → [vis, density]
```

- 适用对象：刚性臂（4-DOF，前 2 个角度用旋转矩阵处理）
- 训练：单阶段，仅 2D loss
- 局限：静态模型，无法处理时序依赖和软体变形

#### MSTNF

引入 MultiScaleEMA 时序编码器，替代 LSTM 编码动作历史。

```
action_window(B, 20, 2) → MultiScaleEMA(4 scales, learnable decay) → physics_state(B, 128)
3D 点 + physics_state + current_action → decode_spatial → [vis, density] → 体渲染
```

- 关键创新：可学习衰减率的多尺度 EMA，比 LSTM 更高效
- 训练：单阶段，不需要 canonical 数据
- 文件：[model_mstnf.py](src/models/model_mstnf.py)

#### C-MSTNF 系列（Canonical + Deformation）

采用 D-NeRF 范式：CanonicalField（零动作静止态）+ DeformationField（动作条件变形）。

```
查询流程: 世界点 → DeformationField(pos_enc + state + action → Δx,Δy,Δz) → canonical 坐标 → CanonicalField → [vis, density]
```

训练流程：两阶段
1. Phase 1：用零动作数据训练 CanonicalField
2. Phase 2：冻结 canonical，训练 DeformationField

| 变体 | 改动 | 效果 |
|------|------|------|
| C-MSTNF | 基础版 | 变形 MLP 产生高频跳变 |
| ODE-CMSTNF | Neural ODE 替代 EMA | 无显著提升（瓶颈在空间不在时序） |
| Smooth-CMSTNF | Spectral Norm + Jacobian 惩罚 | 尖端发散（正则化与精度矛盾） |

三者共同的瓶颈：5 层 ReLU 变形 MLP 本身是高频函数逼近器，位置编码的高频分量被放大。

文件：
- [model_cmstnf.py](src/models/model_cmstnf.py)
- [model_ode_cmstnf.py](src/models/model_ode_cmstnf.py)
- [model_smooth_cmstnf.py](src/models/model_smooth_cmstnf.py)

#### MS-SCNF（多尺度骨架条件神经场）

**核心思路转变**：不再用隐式变形场，而是显式回归 3D 骨架 + 骨架条件密度场。

```
action_window → MultiScaleEMA → physics_state
                                    ↓
                            SkeletonHead
                          ┌────────┼────────┐
                        coarse   medium     fine
                        (4×3)    (10×3)    (31×3)
                                    ↓
              SkeletonConditionedDensity: 查询点到骨架距离 → [vis, density]
```

关键组件：

1. **SkeletonHead**：从 physics_state 预测多尺度骨架坐标
   - coarse (4 个节点) → 整体趋势
   - medium (10 个节点) → 过渡
   - fine (31 个节点) → 完整形态

2. **SkeletonConditionedDensity**：密度取决于查询点到骨架曲线的距离
   - 距离近 → 密度高，距离远 → 密度低
   - 物理合理且自动稀疏

3. **3D GT 监督**：利用 PyElastica 的 position_collection 直接监督骨架预测

训练流程：
- Phase 1：骨架回归（3D loss only）
- Phase 2：联合训练（3D + 2D 渲染 loss）

**相比 C-MSTNF 系列的优势**：
- 骨架曲线天然连续，不需要正则化约束平滑性
- 直接输出 3D 坐标，可用于运动规划
- 3D GT 监督比间接的 2D 渲染 loss 更有效
- 端点精度由骨架端点直接决定，不会发散

文件：[model_ms_scnf.py](src/models/model_ms_scnf.py)

---

## 4. 训练策略对比

| | MSTNF | C-MSTNF 系列 | MS-SCNF |
|---|---|---|---|
| 阶段数 | 1 | 2 | 2 |
| Phase 1 | — | Canonical (2D loss) | 骨架回归 (3D loss) |
| Phase 2 | 重建 + 平滑 | 重建 + 平滑 | 3D + 2D 联合 |
| 数据需求 | sequence | canonical + sequence | sequence_3d |
| 3D 监督 | 无 | 无 | 有 |
| 部署输出 | 2D 渲染图 | 2D 渲染图 | **3D 骨架 + 2D 渲染图** |

---

## 5. 评估体系

### 5.1 2D 指标（所有模型）

- MSE / PSNR：渲染图与 GT 的像素级误差
- 时序连续性：相邻帧渲染结果之间的 MSE

### 5.2 3D 指标（MS-SCNF）

| 指标 | 含义 | 关注点 |
|------|------|--------|
| Mean Node Error | 所有人节点平均 L2 误差 | 整体形状精度 |
| Endpoint Error | 末端节点 L2 误差 | 端点定位精度 |
| Chamfer Distance | 双向最近邻点云距离 | 全局形状匹配 |
| Curve Smoothness | 二阶差分 L2 范数 | 预测骨架的平滑度 |

---

## 6. 关键设计决策

### 6.1 为什么用 MultiScaleEMA 而不是 LSTM？

- 可学习的多尺度衰减率，物理上对应不同时间尺度的惯性
- 参数更少、训练更稳定
- 无需序列展开，支持并行计算

### 6.2 为什么 C-MSTNF 的变形场会产生高频跳变？

变形场使用 5 层 ReLU MLP + 位置编码（n_freqs=6），将高维频率特征映射到 3D 位移：
- ReLU 在分段边界不可导 → 产生尖点
- 高频位置编码被 MLP 放大 → 空间高频振荡
- 尖端区域数据稀疏 → 过拟合

验证：notebook 06（线性变形对比）、notebook 07（课程频率对比）

### 6.3 为什么 MS-SCNF 能避免这些问题？

- **骨架曲线**：由 SkeletonHead 直接输出 31 个 3D 坐标，物理上保证连续
- **距离函数**：密度由到骨架的距离决定，天然光滑
- **3D 监督**：GT 节点坐标直接约束骨架形状，比间接的 2D loss 更有效

### 6.4 为什么数据中嵌入相机参数？

- 数据自描述：npz 文件包含完整的相机位姿，不依赖外部配置
- 向后兼容：旧数据不含相机参数时，训练器自动回退到 camera.json
- 可复现：每个数据文件独立可用，不会因配置文件修改而失效

---

## 7. 与论文原始方法的差异

| 方面 | FBV-SM (论文) | 本项目 |
|------|--------------|--------|
| 仿真器 | PyBullet (刚体) | PyElastica (Cosserat 杆) |
| 机器人类型 | 刚性臂 (4-DOF) | 软体连续体 (2D 扭矩) |
| 时序建模 | 无 (单帧独立) | MultiScaleEMA (20 步窗口) |
| 3D 表示 | 隐式密度场 | 显式骨架 + 条件密度场 |
| 3D 监督 | 无 | GT 节点坐标 |
| 渲染方式 | PyBullet 内置相机 | PyVista 离屏渲染 |
| 部署输出 | 渲染图像 | **3D 骨架坐标** |
| 运动规划 | 梯度优化 + A* + RRT | 待实现 |
