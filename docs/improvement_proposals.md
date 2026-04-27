# 提升软体机器人自建模预测准确性的改进方案

> 基于实验分析提出的改进方案，记录问题根因、方案设计、实施状态和结论。

---

## 1. 问题根因分析

### 1.1 C-MSTNF 系列的共同瓶颈

四个模型（MSTNF / C-MSTNF / ODE-CMSTNF / Smooth-CMSTNF）的差异集中在**时序编码器**（EMA vs ODE），但共享同一个**空间映射瓶颈**：

```
时序编码器 → physics_state (128d) → 变形 MLP (5层 ReLU) → 逐点 3D 位移
                                         ↑
                                     问题所在
```

5 层 ReLU MLP + 位置编码（n_freqs=6）将高维频率特征映射到 3D 位移时：
- ReLU 在分段边界不可导 → 产生尖点
- 高频位置编码被 MLP 放大 → 空间高频振荡
- 尖端区域位移最大但数据最稀疏 → 正则化与精度矛盾

### 1.2 各模型失败的具体原因

| 模型 | 症状 | 根因 |
|------|------|------|
| C-MSTNF | 高频跳变 | 变形 MLP + 高频位置编码 |
| Smooth-CMSTNF | 尖端发散 | Spectral Norm 全局限制 Lipschitz，尖端需要大变形 → 矛盾 |
| ODE-CMSTNF | 无显著提升 | ODE 只改进时序编码，空间映射 MLP 未变 |

---

## 2. 改进方案与实施状态

### 方案 A：低频变形基（Deformation Basis）

**状态**：未实施

**核心思想**：学习一组光滑基函数，时序编码器只预测基函数系数。

```
action_window → 时序编码器 → coefficients (K 个)
空间点 x → 预定义基函数 Φ_1(x), ..., Φ_K(x)  [RBF / 低阶多项式]
deformation(x) = Σ_k coeff_k * Φ_k(x)
```

优势：基函数本身光滑 → 变形天然光滑，物理可解释。
未实施原因：MS-SCNF（方案 D+F）已从根本上解决了问题。

---

### ~~方案 B：线性变形层~~ — 已验证，见 notebook 06

**状态**：已验证（[06_linear_deform_test.ipynb](notebooks/06_linear_deform_test.ipynb)）

**核心思想**：用单层线性映射替代 5 层 MLP 变形场。

```
当前:  temporal_state + pos_enc → [MLP 5层] → displacement
改进:  temporal_state + pos_enc → [线性层]  → displacement
```

**实现**：`LinearDeformModel`（notebook 06 中定义），参数从 168,839 降至 474。

**结论**：
- 线性变形场确实更光滑（空间梯度更小）
- 但渲染质量明显低于 MLP → 非线性变形对软体机器人是必要的
- 排除了"MLP 非线性完全无用"的假设，问题在于如何约束而非去除

---

### 方案 C：多视角几何约束（Multi-view Consistency）

**状态**：未实施

**核心思想**：单视角 3D 约束不足（深度歧义），增加多视角从几何层面约束模型。

实施需要：
1. 修改仿真环境，每个时间步从多个视角渲染
2. 数据采集量增大 3-4 倍
3. 训练对每个视角分别计算重建 loss

优势：从数据层面根本解决 3D 约束不足问题。
优先级较低：MS-SCNF 的 3D GT 监督已提供了更强的 3D 约束。

---

### ~~方案 D：骨架先验 + 半径场~~ — 已实施为 MS-SCNF

**状态**：**已实施**（[model_ms_scnf.py](src/models/model_ms_scnf.py)）

**核心思想**：利用软体机械臂的强几何先验——它是一根连续细长杆。

**MS-SCNF 实现**：

```
时序编码器 → SkeletonHead → 多尺度 3D 骨架 (coarse 4 / medium 10 / fine 31)
                                    ↓
SkeletonConditionedDensity: 查询点到骨架距离 → [vis, density]
```

与原始方案的区别：
- 用多尺度骨架回归代替单尺度控制点
- 添加了 3D GT 监督（方案 F）而非仅依赖 2D 渲染 loss
- 骨架曲线天然保证连续性，无需额外正则化

---

### ~~方案 E：课程式频率学习~~ — 已验证，见 notebook 07

**状态**：已验证（[07_coarse_to_fine_freq.ipynb](notebooks/07_coarse_to_fine_freq.ipynb)）

**核心思想**：变形场位置编码频率从低到高逐步增加，避免高频分量初期干扰全局优化。

**实现**：`CoarseToFineCMSTNF`（notebook 07 中定义），频率 schedule: 2 → 6。

**结论**：
- 课程频率确实改善了训练稳定性
- 最终渲染质量与固定频率接近，但变形场空间梯度更小
- 延迟而非消除高频问题 → 不是根本解决方案
- 这个思路被吸收到 MS-SCNF 的多尺度骨架训练中（coarse → fine）

---

### ~~方案 F：显式 3D 监督~~ — 已实施

**状态**：**已实施**

**实施内容**：
1. **数据采集**：`collect.py --3d` 保存 3D 节点坐标 (`positions: (T, 3, 31)`)
2. **MS-SCNF Phase 1**：3D 骨架回归 loss（MSE on 节点坐标）
3. **MS-SCNF Phase 2**：3D loss + 2D 渲染 loss 联合训练
4. **评估**：[evaluate_3d.py](scripts/evaluation/evaluate_3d.py) 计算 4 个 3D 定量指标

3D 监督信号来源：PyElastica 的 `rod.position_collection`（31 个节点的 xyz 坐标）。

---

## 3. 方案总结

| 方案 | 名称 | 状态 | 关键结论 |
|------|------|------|---------|
| A | 低频变形基 | 未实施 | MS-SCNF 已从根本上解决 |
| B | 线性变形层 | **已验证** | 更光滑但精度不足，非线性必要 |
| C | 多视角约束 | 未实施 | MS-SCNF 的 3D GT 已提供强约束 |
| D | 骨架先验 | **已实施 (MS-SCNF)** | 核心改进，直接输出 3D 骨架 |
| E | 课程频率 | **已验证** | 改善训练稳定性，但非根本方案 |
| F | 3D 监督 | **已实施** | 比间接 2D loss 更有效 |

**最终采纳**：方案 D + F → MS-SCNF 模型
**设计吸收**：方案 E 的 coarse-to-fine 思想融入 MS-SCNF 的多尺度骨架训练

---

## 4. 理论分析

### 4.1 为什么 MLP 变形场产生高频

位置编码将坐标映射到高维频率空间，最高频率 2⁹ = 512。5 层 ReLU MLP 在如此高频的输入上：
- 分段线性插值在非常细的空间尺度响应 → 棋盘格伪影
- 尖端区域数据稀疏 → 过拟合
- ReLU 不可导性在分段边界产生尖点

### 4.2 为什么 Spectral Norm 在尖端失效

Spectral Norm 限制全局 Lipschitz 常数，但尖端恰好需要最大的变形敏感度（从 0 到最大位移）。全局正则化在尖端与需求矛盾。

### 4.3 为什么 MS-SCNF 能避免这些问题

1. **骨架曲线天然连续**：SkeletonHead 输出的是 3D 坐标序列，连续性由网络结构保证而非事后正则化
2. **距离函数天然光滑**：密度由 `exp(-dist²/2σ²)` 决定，无穷阶可导
3. **3D GT 直接约束**：节点坐标的 MSE loss 比 2D 渲染 loss 更直接有效
4. **多尺度训练**：coarse → fine 的课程式学习避免高频干扰
