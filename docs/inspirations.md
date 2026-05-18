# 灵感记录

> 记录研究中产生的每个灵感，包括已实现和未实现的。
> 归档的模型代码在 `docs/archived/` 中保留。

---

## 1. Neural ODE 时序编码（ODE-CMSTNF）

**日期**: 2026-04
**状态**: 已实现，归档

**灵感来源**: 软体臂的弯曲响应物理上近似为阻尼谐振子，EMA 是一阶系统只能指数衰减，无法捕捉振荡。

**核心想法**:
用 Neural ODE 替代 MultiScaleEMA 做时序编码。状态分为 [position, velocity]，动力学为阻尼弹簧模型：
```
ds_pos/dt = s_vel
ds_vel/dt = -k * s_pos - c * s_vel + B * action
```
k（刚度）、c（阻尼）、B（力矩阵）可学习。

**优势**:
- ODE 积分保证状态轨迹连续——微小输入变化不会导致输出跳变
- 二阶动力学可捕捉阻尼振荡
- 可嵌入物理先验

**归档原因**: 训练不稳定（RK4 积分梯度爆炸），效果不如 EMA。思路本身有价值，后续可用 adjoint method 或更稳定的 ODE solver 重新尝试。

**归档文件**: `docs/archived/ode_cmstnf/`

---

## 2. 光谱正则化变形场（Smooth-CMSTNF）

**日期**: 2026-04
**状态**: 已实现，归档

**灵感来源**: C-MSTNF 的变形 MLP 可能产生高频跳变——微小动作变化导致巨大形状变化。

**核心想法**:
在 C-MSTNF 变形场上施加三种正则化：
1. **Spectral Normalization**: 限制变形 MLP 每层权重矩阵的谱范数，控制 Lipschitz 常数
2. **Jacobian Penalty**: `mean((∂displacement/∂x)²)` — 惩罚变形场对空间坐标的剧烈梯度
3. **Temporal Gradient Penalty**: `||D(x,a_t) - D(x,a_{t+1})||² / ||a_t - a_{t+1}||²` — 变形随时间的变化率应正比于动作变化率

**归档原因**: 正则化权重调节困难，过于保守则变形能力不足，过于宽松则无效果。当前骨架+SDF 方案（方案 B）从架构层面保证了连续性，不需要额外正则化。

**归档文件**: `docs/archived/smooth_cmstnf/`

---

## 3. 参数化骨架 + 管状 SDF 先验（SkeletonSDF / 方案 B）

**日期**: 2026-05
**状态**: 实现中

**灵感来源**:
- Shape-Interpretable Visual Self-Modeling（中山大学 2025）— Bezier 曲线参数化
- 管状结构 SDF 解析公式: `SDF(x) = dist_to_skeleton(x) - radius`

**核心想法**:
1. 参数化曲线（B-spline/Fourier）预测骨架 → 数学保证拓扑连通
2. `dist - radius` 提供管状 SDF 先验 → 强几何先验加速收敛
3. SIREN 残差修正截面 → 学习非均匀截面形状

---

## 4. Flow Matching 条件点云生成（方案 A）

**日期**: 2026-05
**状态**: 调研中，未实现

**灵感来源**: Action-Conditioned Flow Matching（arXiv 2605.09216, 2025）

**核心想法**:
放弃隐式场+体渲染，改用 action-conditioned flow matching 直接生成点云。学习条件速度场 `u_theta(X_t, t | c)`，从高斯噪声积分到目标点云。

**预期优势**: 点云天然连续不存在断裂；Flow 的平滑性保证时间连续性；无需体渲染。
