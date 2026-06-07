# 方向灵感：预测-修正框架（Predictive-Corrective State Estimation）

> **状态：已实现 ✅**
> 来源：从"软体机器人状态估计"这一根本问题出发的思考
> 核心思想：时序模型做先验预测，视觉观测做在线修正
>
> ## 实现位置
> - 模型：`src/models/model_pc_spatial.py` — `PCSpatialSequenceModel` 类（两阶段：predictive + corrective）
> - 训练脚本：`scripts/training/train_pc_spatial.py`
> - 数据集：`src/data/dataset_spatial.py`（同 SpatialSequence，Phase 2 额外需要 images）
> - 评估：支持 `predict_skeleton(action_window, images=...)`
> - PROJECT_HELP：已记录在管线表中（"预测-修正"行）

---

## 一、出发点：我们的研究定位

### 从"重建"到"状态估计"的视角转换

| 视角 | 问题定义 | 输入 | 输出 |
|------|---------|------|------|
| 重建 | 从图像恢复 3D 形状 | 单帧图像 | 静态 3D 点云 |
| 状态估计 | 从历史驱动+视觉观测推断当前状态 | (action_history, image_t) | 带时序依赖的 3D 状态 |

**关键差异**：重建不关心历史，状态估计必须考虑历史。迟滞的本质就是"同一个 action 在不同历史下产生不同状态"。

### 现有工作的定位

| 工作 | 定位 | 时序？ | 3D？ | 视觉？ |
|------|------|--------|------|--------|
| Tang 2020 | 图像→控制策略（端到端） | 否 | 否 | 是 |
| Yu 2026 | actuation→2D形状（feedforward） | 否 | 否 | 是 |
| Chen 2025 | actuation→3D形状（迟滞感知） | 部分（方向编码） | 是 | 否（运动捕捉） |
| **我们** | **(history, vision)→3D状态（predictive-corrective）** | **是** | **是** | **是** |

---

## 二、核心思想

### 框架定义

```
预测分支（Predictive）：
    ŝ_t = f_θ(action_history)          # 基于驱动历史的先验预测

修正分支（Corrective）：
    s_t = ŝ_t + g_φ(image_t, ŝ_t)     # 视觉观测修正预测误差

完整状态估计：
    s_t = f_θ(history) + g_φ(observation, prediction)
```

### 为什么需要两个分支？

**只用预测分支（纯 feedforward）的问题**：
- 模型误差随时间累积
- 无法处理训练分布外的扰动（外力、负载变化）
- Chen 2025 有 OptiTrack 做 GT 所以不需要修正

**只用修正分支（纯视觉）的问题**：
- 2D 视觉有投影歧义，无法唯一确定 3D 形状
- 迟滞状态在 2D 投影中可能不可区分
- Yu 2026 就受限于这个——没有时序先验

**两者结合的优势**：
- 预测分支提供强先验（基于物理仿真的知识）
- 修正分支弥补模型误差（基于真实视觉观测）
- 类似 Kalman 滤波的思想：模型预测 + 观测更新

---

## 三、与 Kalman 滤波的类比

| Kalman 滤波 | 我们的框架 |
|------------|-----------|
| 状态转移方程 $x_{t|t-1} = Ax_{t-1}$ | 时序模型 $ŝ_t = f(history)$ |
| 观测方程 $z_t = Hx_t$ | 渲染方程 $image = \pi(s)$ |
| 卡尔曼增益 $K$ | 修正网络 $g_φ$ |
| 协方差预测 $P_{t|t-1}$ | 预测不确定性（隐式学习） |

区别：
- Kalman 滤波是线性的，我们的模型是非线性的
- Kalman 滤波需要已知系统矩阵，我们从数据学习
- 但**核心思想相同**：模型预测 + 观测修正

### 可能的深入学习框架

**方案 A：隐式修正（Neural Rendering）**

用 NeRF/体渲染作为观测方程，通过可微渲染端到端训练修正：

```python
# 预测
s_pred = temporal_model(action_history)           # (B, N, 3) 预测点云/形状参数

# 渲染
rendered_image = volume_render(s_pred, camera)    # 可微渲染

# 修正信号
loss_render = MSE(rendered_image, observed_image) # 视觉一致性
loss_temporal = MSE(s_pred, gt_3d)                # 3D 监督（训练时）

# 训练完成后，s_pred 已经被渲染损失修正过
```

**方案 B：显式修正（Residual Correction）**

```python
class PredictiveCorrectiveModel(nn.Module):
    def __init__(self):
        # 预测分支
        self.temporal_encoder = FractionalMemory(...)
        self.shape_predictor = SpatialGRU(...)

        # 修正分支
        self.image_encoder = CNNEncoder(...)
        self.correction_head = nn.Linear(hidden_dim, n_params)

    def forward(self, action_history, image=None):
        # 预测
        s_pred = self.shape_predictor(self.temporal_encoder(action_history))

        # 修正（推理时使用）
        if image is not None:
            img_feat = self.image_encoder(image)
            delta_s = self.correction_head(img_feat)
            s_corrected = s_pred + delta_s
            return s_corrected

        return s_pred
```

**方案 C：概率框架（Uncertainty-Aware）**

```python
class ProbabilisticStateEstimator(nn.Module):
    def __init__(self):
        # 预测分布
        self.temporal_model = ...  # 输出 μ_pred, σ_pred
        # 观测似然
        self.vision_model = ...    # 输出 μ_obs, σ_obs
        # 融合: μ_fused = (μ_pred/σ_pred² + μ_obs/σ_obs²) / (1/σ_pred² + 1/σ_obs²)
```

---

## 四、解决"2D 视觉学习 3D 迟滞"的本质困难

### 困难的三层结构

| 层次 | 困难 | 预测-修正如何解决 |
|------|------|-----------------|
| **投影歧义** | 多个 3D 形状投影到同一 2D | 预测分支提供先验，缩小候选空间 |
| **迟滞不可观测** | 同 action 不同状态，2D 看不出 | 预测分支编码历史，修正分支只需微调 |
| **视觉修正** | 如何从 2D 修正 3D | 渲染一致性约束，可微渲染端到端 |

### 核心论证

纯 2D 视觉学习 3D 迟滞是**欠定**的——同一个 2D 观测可能对应多个（带不同历史的）3D 状态。但如果：

1. 预测分支已经给出了一个合理的 3D 先验（基于物理仿真训练）
2. 修正分支只需要在这个先验附近做小幅调整

那么问题就从"从零推断 3D"变成了"修正已有预测"——这是一个**良定**的问题，因为修正幅度小，投影歧义的影响也小。

### 数学表述

```
纯视觉：   p(s_t | image_t)                # 欠定，多个峰值
预测+修正：p(s_t | image_t, ŝ_t)           # 良定，峰值集中在 ŝ_t 附近

其中 ŝ_t = f(action_history) 是一个强先验
```

---

## 五、训练策略

### 阶段 1：仿真预训练（预测分支）

利用 PyElastica 的 3D GT 数据：

```python
# 只训练预测分支
s_pred = temporal_model(action_history)
loss = MSE(s_pred, gt_3d_positions)  # 直接 3D 监督
```

这一步让模型学会 action_history → 3D_shape 的映射，包括迟滞效应。

### 阶段 2：渲染监督（加入修正分支）

```python
# 预测 + 渲染
s_pred = temporal_model(action_history)
rendered = volume_render(s_pred, camera_params)
loss_render = MSE(rendered, observed_image)

# 修正
s_corrected = s_pred + correction_head(image_features)
loss_3d = MSE(s_corrected, gt_3d)

total_loss = loss_render + λ * loss_3d
```

### 阶段 3：Sim-to-Real 迁移

```python
# 只有视觉观测，无 3D GT
s_pred = temporal_model(action_history)
rendered = volume_render(s_pred, camera_params)
loss = MSE(rendered, real_image)  # 唯一的监督信号
```

此时预测分支已经很强（仿真训练的），修正分支只需要微调。

---

## 六、与其他方向的组合关系

```
预测-修正框架 (本文件)
├── 预测分支: 分数阶记忆核 (fractional_order_memory) + 空间序列生成 (spatial_sequence_generation)
├── 修正分支: CNN 编码器 + 残差修正
└── 训练信号: 可微渲染 + 拓扑引导残差流 (topology_guided_residual_flow)
```

**最强组合**：
1. 分数阶记忆核编码驱动历史 → 捕获迟滞
2. 空间序列生成预测 3D 截面参数 → 结构化状态表示
3. 视觉修正网络在线修正 → 处理模型误差
4. 可微渲染提供训练信号 → 从 2D 图像学习 3D

---

## 七、与现有代码的关系

| 已有模块 | 在框架中的角色 |
|---------|-------------|
| `MultiScaleEMA` / 新 `FractionalMemory` | 预测分支的时序编码器 |
| `velocity_net` / 新 `SpatialGRU` | 预测分支的形状生成器 |
| `rendering.py` (volume rendering) | 修正分支的训练信号 |
| `camera_system.py` | 多视角渲染 |
| `SoftSequenceDataset` | 数据加载（已有 action_history + images） |

新增需要实现的：
- 修正分支的 `CorrectionHead`
- 预测-修正的联合训练循环
- Sim-to-Real 迁移的 domain adaptation

---

## 八、关键科学问题

### Q1: 修正分支的容量需要多大？

如果预测分支足够强（仿真预训练得好），修正分支可能只需要很小的网络（几个残差连接）。
反之，如果预测分支有系统性偏差（仿真与真实不一致），修正分支需要更大。

**实验**：先在仿真中测试，用预测误差的统计量指导修正分支的设计。

### Q2: 修正分支是否需要时序信息？

当前设计中修正分支只看当前帧图像。但如果物体被遮挡，可能需要多帧观测。

**变体**：
- 单帧修正：$s_t = ŝ_t + g(image_t, ŝ_t)$
- 多帧修正：$s_t = ŝ_t + g(image_{t-k:t}, ŝ_t)$
- 递归修正：$s_t = ŝ_t + g(image_t, s_{t-1})$

### Q3: 预测和修正的信任度如何平衡？

类似 Kalman 增益的问题。如果预测分支很确定（history 很稳定），应该少修正。
如果观测很确定（图像清晰无遮挡），应该多修正。

**方案**：让网络学习一个置信度/门控机制。

---

## 九、实验验证路线图

1. **仿真验证**：在 PyElastica 中测试预测分支是否准确
2. **消融实验**：有/无修正分支的对比
3. **迟滞验证**：同 action 不同历史路径的区分能力
4. **Sim-to-Real**：真实软臂 + 相机部署
