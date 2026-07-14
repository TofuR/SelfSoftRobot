# 柔性机器人自建模文献调研与改进方案

> 调研日期: 2026-05-16
> 目标: 解决当前 SDF 训练模型"预测出分段点云"、"预测结果突变"等问题，实现对连续柔性机器人全身形状的准确估计

---

## 一、当前模型问题诊断

### 1.1 观察到的问题

| 问题 | 表现 | 可能根因 |
|------|------|----------|
| **空间不连续** | 预测出几段分割开的点云而非整体 | Deformation MLP 高频振荡；密度场在远离骨架处衰减过快 |
| **时间不连续** | 相邻帧预测结果突然改变 | 时序编码器(MultiScaleEMA)对高维形变空间建模不足 |
| **深度模糊** | Z轴误差占72-85% | 单视角无法约束深度；体渲染对细长结构的深度分辨率不够 |

### 1.2 架构层面根因分析

1. **隐式场 vs 显式形状**: 当前使用 SDF/密度隐式场 + 体渲染，但没有强制全局拓扑一致性。柔性臂是一个连通的一维拓扑结构，但隐式场没有拓扑约束，因此会产生断裂的等值面。

2. **变形场表示**: CMSTNF 系列使用 MLP 学习位置编码->位移的映射，高频位置编码 + ReLU MLP 容易产生高频振荡，导致空间不连续。

3. **骨架条件密度**: MS-SCNF 用距离场定义密度，物理上合理但依赖骨架预测精度，骨架预测误差直接导致密度场错误。

4. **训练信号**: 2D 渲染损失是弱监督——多个 3D 形状可能投影出相同的 2D 图像（深度模糊），模型没有足够的信号来区分。

---

## 二、核心文献调研

### 2.1 Action-Conditioned Flow Matching (最相关，直接对标)

**论文**: *Continuum Robot Modeling with Action Conditioned Flow Matching* (arXiv 2605.09216, 2025)

**核心思想**:
- 将连续体机器人形状预测重新定义为**条件点云生成问题**
- 不再回归坐标->SDF/密度，而是学习一个**条件速度场**，将高斯噪声"流"到目标点云
- 使用 FiLM 层注入驱动条件

**关键方法**:
```
训练: 采样 X0 ~ N(0,sigma^2*I), 插值 Xt = (1-t)X0 + tX1, 学习速度场 u_theta(Xt,t|c)
推理: 从高斯噪声出发, 积分 ODE 得到预测点云 X_hat*
损失: L_FM = E[||u_theta(Xt,t|c) - (X1-X0)||_F^2]
```

**架构**: 两种速度网络
- MLP: 每个点独立处理 + FiLM 调制, O(N) 复杂度
- Hybrid: PVConv 局部几何上下文 + 全局条件, 时间门控融合

**为什么能解决我们的问题**:
- **空间连续性**: Flow matching 的速度场天然平滑（Lipschitz 连续），生成的点云不会断裂
- **时间稳定性**: 相邻驱动条件产生相邻的速度场 -> 相邻的点云预测
- **无需体渲染**: 直接在点云空间操作，避免深度模糊问题
- **可扩展**: 可加入负载等额外条件

**实验结果**: 在模拟和真实 TDCR 上，CD 相比最强 baseline 降低 64-96%，EMD 降低 50-83%

**局限**:
- 需要 RGB-D 数据（多视角点云融合）
- 是准静态方法，不建模瞬态动力学
- 生成式方法，单次推理需要多步 ODE 积分（约100步）

**来源**: [arxiv.org/html/2605.09216v1](https://arxiv.org/html/2605.09216v1)

---

### 2.2 Shape-Interpretable Visual Self-Modeling (几何可解释性最强)

**论文**: *Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control* (arXiv 2603.01751, 2025, 中山大学)

**核心思想**:
- 用**分段二次 Bezier 曲线**参数化机器人形状（而非隐式场）
- 多视角 Bezier 控制点联合编码唯一确定 3D 形状
- Neural ODE 建模形状动力学

**关键方法**:
```
1. 图像 -> 二值化 -> 骨架提取 -> Bezier 拟合 -> 控制点
2. 多视角控制点拼接 -> 3D 形状向量 x_s
3. NODE: dx_s/dt = f_NN(x_s, u, t)
4. Jacobian 控制: u_dot = J_dagger(x_dot_d + lambda*(x_d - x))
```

**为什么相关**:
- **拓扑保证**: Bezier 曲线天然是连续的、拓扑正确的一维结构，不存在"断裂"问题
- **低维紧凑**: 7个控制点（3段Bezier）就能描述3D形状，避免高维隐式场的问题
- **几何感知**: 控制点有明确几何含义，可直接用于避障、自运动等

**实验结果**: 形状误差 < 1.56% 图像分辨率，末端误差 < 2% 机器人长度

**局限**:
- 假设形状可用曲线参数化（仅适用于细长结构）
- 需要 2D 骨架提取（受图像质量影响）
- 未考虑截面信息

**来源**: [arxiv.org/html/2603.01751v1](https://arxiv.org/html/2603.01751v1)

---

### 2.3 INR-DOM: Hypernetwork + SDF (SDF 改进方向)

**论文**: *Implicit Neural-Representation Learning for Elastic Deformable-Object Manipulations* (RSS 2025, KAIST)

**核心思想**:
- 用 **Hypernetwork + SIREN** 的组合实现条件 SDF: 输入部分点云->编码器->Hypernetwork 生成 SDF 网络权重->查询任意点的 SDF 值
- 两阶段训练: (1) 预训练重建 + (2) 对比学习微调

**关键创新**:
```
编码器 Phi: 部分点云 -> 潜变量 z (64D)
Hypernetwork Psi: z -> SDF网络权重 theta
SDF网络 Omega_theta: (x, z) -> SDF值 d

损失:
- L_SDF: Eikonal约束 + 表面约束 + 法线对齐 + 离面正则化
- L_skel: 中轴约束（解决遮挡区域重建）
- L_cns: 部分点云编码 约等于 完整点云编码（一致性）
- L_infoNCE: 对比学习微调（区分复杂形变状态）
```

**为什么相关**:
- **解决遮挡/不完整**: 一致性损失让部分观测和完整观测映射到同一潜空间区域
- **中轴约束**: SDF 在中轴线处的 Laplacian 应趋向无穷，这是对柔性臂中心线的物理约束
- **Hypernetwork 思路**: 不学一个全局 SDF，而是根据驱动状态动态生成 SDF 网络参数

**实验结果**: 重建 CD/EMD 均优于 PCN、PointTr、Point2Vec；DOM 任务成功率比次优方法高 40.3%

**局限**:
- 针对弹性带/环，不是直接面向机器人
- 需要 GT 完整点云做预训练

**来源**: [arxiv.org/html/2505.00500v1](https://arxiv.org/html/2505.00500v1)

---

### 2.4 Egocentric Visual Self-Modeling (Nature npj Robotics 2025)

**论文**: *Egocentric Visual Self-Modeling for Autonomous Robot Dynamics Prediction and Adaptation* (npj Robotics 2025, Lipson 组)

**核心思想**:
- 仅用自我中心（第一人称）视觉观测建模机器人动力学
- 无需本体感知传感器（如 IMU）
- 腿式机器人成功实现运动任务

**与我们的关系**: 自建模范式的参考，但面向刚性腿式机器人。核心启发是"仅用视觉做自建模"的理念验证。

**来源**: [nature.com/articles/s44182-025-00031-6](https://www.nature.com/articles/s44182-025-00031-6)

---

### 2.5 其他相关工作

| 论文 | 核心贡献 | 与我们的关系 |
|------|----------|-------------|
| **Robot-NO** (Adv. Eng. Informatics 2025) | 神经算子直接映射几何+载荷->全场变形，比FEM快6000倍 | Neural Operator 思路可用于加速物理预测 |
| **Jacobian Fields** (Nature 2025) | 视频流->Jacobian场->控制多种机器人形态（含软体） | 提供 Jacobian 场学习的范式 |
| **4DRecons** (arXiv 2024) | 4D隐式场重建可变形物体 | 4D时空场的思路可借鉴 |
| **Disney INR for Soft Bodies** (2022) | 隐式神经表示物理驱动的软体 | 材料空间->世界空间的隐式映射 |

---

## 三、改进方案

基于文献调研，针对当前模型的问题，提出以下改进方向（按推荐优先级排序）:

### 方案 A: Flow Matching + 条件点云生成 (推荐首选)

**思路**: 放弃隐式场+体渲染管线，改用 action-conditioned flow matching 直接生成点云。

**具体改动**:
1. **数据端**: 使用 PyElastica 的 3D 位置数据，直接构建点云（沿 Cosserat rod 采样 + 截面圆采样）
2. **模型**:
   - 驱动条件 c = action_window -> MultiScaleEMA -> physics_state
   - Flow Matching 速度场: u_theta(X_t, t | c)，FiLM 注入条件
   - 从 N(0, sigma^2*I) 积分到目标点云
3. **训练损失**: Chamfer Distance + EMD（直接在点云空间）
4. **多视角**: 如果有 RGB-D，可以加渲染一致性损失

**预期效果**:
- 点云天然连续，不存在断裂
- Flow 的平滑性保证时间连续性
- 无深度模糊问题（直接在3D空间操作）

**风险**: 生成式方法推理较慢（需要ODE积分）；需要足够多的训练数据。

---

### 方案 B: 混合 Bezier 骨架 + SDF 截面 (几何先验最强)

**思路**: 用参数化曲线保证拓扑正确性，用条件 SDF 建模截面。

**具体改动**:
1. **骨架**: 预测 N 个控制点 -> Bezier/B-spline 曲线 -> 保证拓扑连通
2. **截面**: 沿骨架法平面查询条件 SDF -> 获得截面形状
3. **训练**:
   - 骨架损失: 控制点 MSE + 曲线平滑正则
   - 截面损失: SDF 监督 + Eikonal 约束
   - 渲染损失: 体渲染一致性（可选）

**预期效果**:
- Bezier 曲线保证拓扑连通，彻底消除"断裂点云"
- 截面 SDF 保留完整的形状信息
- 控制点可解释、可直接用于控制

**风险**: 骨架+截面的联合建模复杂度高；需要仔细设计两者的耦合方式。

---

### 方案 C: Hypernetwork 条件 SDF (当前 SDF 管线的直接升级)

**思路**: 借鉴 INR-DOM 的 Hypernetwork 思路，不学一个全局 SDF，而是根据驱动状态动态生成 SDF 网络参数。

**具体改动**:
1. **架构**:
   ```
   action_window -> encoder -> z (latent, 64-128D)
   Hypernetwork(z) -> theta (SDF network weights)
   SDF_theta(x, z) -> SDF value at query point x
   ```
2. **损失**:
   - SDF 监督 (L1 on SDF values)
   - Eikonal 约束 (||grad SDF|| = 1)
   - 中轴约束 (Laplacian -> inf at skeleton)
   - 一致性正则化 (相邻 action -> 相邻 z)
3. **推理**: 给定 action -> 生成 SDF 网络 -> Marching Cubes 提取表面

**预期效果**:
- 每个驱动状态有专门的 SDF 网络，避免全局拟合困难
- 中轴约束提供拓扑引导
- 保留当前 SDF 管线的大部分代码

**风险**: Hypernetwork 训练可能不稳定；推理需要生成大量参数。

---

### 方案 D: 拓扑感知正则化 (最小改动方案)

**思路**: 在当前架构基础上加入拓扑约束，不改核心管线。

**具体改动**:
1. **连通性损失**:
   - 对预测的 3D 点云/密度场提取骨架（细化算法）
   - 约束骨架为一条连通曲线
   - L_conn = sum max(0, d_i - epsilon)^2，其中 d_i 是骨架相邻点间距
2. **平滑正则**:
   - 沿骨架方向的二阶差分正则化
   - L_smooth = sum ||x_{i+1} - 2x_i + x_{i-1}||^2
3. **时序一致性**:
   - 相邻帧预测的 Chamfer Distance 作为正则项
   - L_temp = CD(pred_t, pred_{t+1})

**预期效果**: 在不大改架构的情况下缓解断裂和突变问题。

**风险**: 治标不治本，正则化权重调节困难。

---

## 四、建议的实施路线

```
Phase 1 (1-2周): 方案 D -- 拓扑正则化
  |-- 最小改动，快速验证
  |-- 验证连通性损失是否有效
  +-- 确认问题的根因是否确实是拓扑约束缺失

Phase 2 (2-4周): 方案 A -- Flow Matching
  |-- 如果 Phase 1 效果有限，实施此方案
  |-- 实现条件 Flow Matching 点云生成
  |-- 对比 CMSTNF/MS-SCNF baselines
  +-- 这是文献中验证最充分的路线

Phase 3 (可选): 方案 B -- Bezier + SDF 混合
  +-- 如果需要可解释性 + 控制能力
```

---

## 五、参考文献

1. *Continuum Robot Modeling with Action Conditioned Flow Matching* -- arXiv 2605.09216, 2025
2. *Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control* -- arXiv 2603.01751, 2025
3. *Implicit Neural-Representation Learning for Elastic Deformable-Object Manipulations* -- RSS 2025
4. *Egocentric Visual Self-Modeling for Autonomous Robot Dynamics* -- Nature npj Robotics, 2025
5. *A Generalizable Neural Operator for Full-Field Deformation Prediction* -- Adv. Eng. Informatics, 2025
6. *Controlling Diverse Robots by Inferring Jacobian Fields* -- Nature, 2025
7. *4DRecons: 4D Neural Implicit Deformable Objects Reconstruction* -- arXiv 2406.10167, 2024
8. *Implicit Neural Representation for Physics-driven Actuated Soft Bodies* -- Disney Research, 2022
