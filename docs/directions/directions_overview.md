# 研究方向总览 (Research Directions Overview)

> 基于文献调研和深度讨论，从"出发点（真问题）"推导出的研究方向体系
> 核心出发点：软体机器人自建模 = 带迟滞的 3D 全身状态估计
> 最后更新：2026-06-09

---

## 出发点

> **软体机器人的当前状态不仅取决于当前驱动，还取决于驱动历史。但现有的 2D 视觉方法无法从历史观测中恢复带时序依赖的 3D 全身状态，而 3D 运动捕捉方法成本过高不实用。**

三个子问题：
1. **时序依赖（迟滞）**：同一个 action 在不同历史下产生不同形状
2. **3D 全身状态**：不只是末端坐标，也不只是 2D 图像上的曲线
3. **低成本感知**：只用相机，不用运动捕捉

---

## 方向关系图（全景）

```
出发点: 带迟滞的 3D 全身状态估计
    │
    ├────────────────── 已实现 ───────────────────┐
    │                                              │
    │  ✅ 分数阶记忆核 ───→ ✅ 空间序列生成         │
    │       (编码器)           (GRU 沿 Z 轴)        │
    │                          │                   │
    │                     ✅ 预测-修正框架           │
    │                    (PC-Spatial 两阶段)        │
    │                                              │
    │  ✅ Gamma/Laguerre 延迟峰值权重（编码器已实现） │
    │  ✅ 顺序敏感编码器 GRU/Transformer/TCN         │
    │                                              │
    │  已归档至 docs/archived/directions/           │
    └──────────────────────────────────────────────┘
    │
    ├─────── 核心技术层（解决超前预测）──────┐
    │                                       │
    │  ┌────────────────┐                   │
    │  │自回归状态动力学 │ ← 待实现           │
    │  └────────────────┘                   │
    │  ┌────────────────┐                   │
    │  │Action偏置分析   │ ← 待消融验证       │
    │  └────────────────┘                   │
    │  ┌────────────────┐                   │
    │  │Action窗口降采样 │ ← 待实现           │
    │  │(10x冗余消除)    │                    │
    │  └────────────────┘                   │
    │  ┌────────────────┐                   │
    │  │迟滞信息容量分析 │ ← 待理论验证        │
    │  └────────────────┘                   │
    └───────────────────────────────────────┘
    │
    ├─────── 形状表达层 ──────┐
    │                        │
    │  ┌──────────────────┐  │
    │  │骨架→形状转换      │  │  固定半径→可变半径/学习截面
    │  │(skeleton_to_shape)│  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │拓扑引导残差流     │  │  物理粗变形 + FM 残差
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │从轮廓恢复形状     │  │  Visual Hull + 骨架条件
    │  │(shape_from_       │  │
    │  │ silhouette)       │  │
    │  └──────────────────┘  │
    └────────────────────────┘
    │
    ├─────── 感知与部署层 ────┐
    │                        │
    │  ┌──────────────────┐  │
    │  │多视角 2D→3D 骨架  │  │  双视角三角化/可微渲染
    │  │(multi_view_2d_    │  │
    │  │ to_3d_skeleton)   │  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │视觉辅助部署       │  │  在线适应 + 残差修正
    │  │(vision_corrected) │  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │Sim-to-Real 迁移   │  │  残差物理/域随机化
    │  │(sim_to_real)      │  │
    │  └──────────────────┘  │
    └────────────────────────┘
    │
    ├─────── 建模方法论层 ────┐
    │                        │
    │  ┌──────────────────┐  │
    │  │单DOF分解与组合    │  │  独立训练+叠加/模态分解
    │  │(per_dof_          │  │
    │  │ decomposition)    │  │
    │  └──────────────────┘  │
    └────────────────────────┘
```

---

## 第一层：核心技术（解决超前预测）

### 问题

训练 SpatialSequence 和 PC-Spatial 后发现，模型预测的中心线**系统性超前**于 GT。

### 方向列表

| 方向 | 文档 | 解决的子问题 | 核心思想 | 状态 |
|------|------|------------|---------|------|
| **自回归状态动力学** | [01_autoregressive_state_dynamics.md](01_autoregressive_state_dynamics.md) | "当前在哪" | 前一步物理状态作为输入 | 待实现 |
| **迟滞信息容量分析** | [02_hysteresis_information_capacity.md](02_hysteresis_information_capacity.md) | "能学多少" | 编码器容量 vs 迟滞复杂度 | 待理论验证 |
| **Action 窗口降采样** | [03_action_window_downsampling.md](03_action_window_downsampling.md) | "输入冗余" | 10x 重复 action 用 stride 消除 | 待实现 |
| **Action 偏置分析** | [04_temporal_encoding_bias_analysis.md](04_temporal_encoding_bias_analysis.md) | "短路问题" | current_action 拼接绕过迟滞 | 待消融 |

```
问题：预测超前于真实响应
    │
    ├─ 原因1：输入冗余（10x重复） → Action 窗口降采样
    ├─ 原因2：current_action 短路 → Action 偏置消融
    └─ 原因3：缺少物理状态反馈 → 自回归动力学
```

建议优先级：
1. **Action 窗口降采样**（零成本验证，提升所有编码器效率）
2. **Action 偏置消融**（零成本验证）
3. **自回归状态动力学**（改动较大，但最根本）

---

## 第二层：形状表达（骨架→完整形状）

当前所有模型退化为骨架预测，丢失了表面/截面信息。

| 方向 | 文档 | 核心思想 | 优先级 |
|------|------|---------|--------|
| **骨架→形状转换** | [05_skeleton_to_shape_conversion.md](05_skeleton_to_shape_conversion.md) | 可变半径 / 学习截面 / 3DGS | ★★★ |
| **拓扑引导残差流** | [09_topology_guided_residual_flow.md](09_topology_guided_residual_flow.md) | 物理粗变形 + FM 残差 | ★★☆ |
| **从轮廓恢复形状** | [07_shape_from_silhouette.md](07_shape_from_silhouette.md) | 骨架条件 Visual Hull | ★★☆ |

---

## 第三层：感知与部署

从仿真走向实际应用的路径。

| 方向 | 文档 | 核心思想 | 优先级 |
|------|------|---------|--------|
| **多视角 2D→3D 骨架** | [06_multi_view_2d_to_3d_skeleton.md](06_multi_view_2d_to_3d_skeleton.md) | 双视角三角化 / 可微渲染 | ★★☆ |
| **视觉辅助部署** | [10_vision_corrected_deployment.md](10_vision_corrected_deployment.md) | 在线适应 + 残差修正 | ★★☆ |
| **Sim-to-Real 迁移** | [11_sim_to_real_transfer.md](11_sim_to_real_transfer.md) | 残差物理 / 域随机化 | ★☆☆ |

---

## 第四层：建模方法论

| 方向 | 文档 | 核心思想 | 优先级 |
|------|------|---------|--------|
| **单 DOF 分解与组合** | [08_per_dof_decomposition.md](08_per_dof_decomposition.md) | 独立训练 + 模态叠加 | ★☆☆ |

---

## 已归档方向

以下方向已实现，归档至 `docs/archived/directions/`：

| 方向 | 原文档 | 实现位置 |
|------|--------|---------|
| **分数阶记忆核** | [fractional_order_memory.md](../archived/directions/fractional_order_memory.md) | `src/encoders/fractional_memory.py` |
| **空间序列生成** | [spatial_sequence_generation.md](../archived/directions/spatial_sequence_generation.md) | `src/models/model_spatial_sequence.py` |
| **预测-修正框架** | [predictive_corrective_state_estimation.md](../archived/directions/predictive_corrective_state_estimation.md) | `src/models/model_pc_spatial.py` |
| **Gamma/Laguerre 延迟编码** | [gamma_laguerre_temporal_encoding.md](../archived/directions/gamma_laguerre_temporal_encoding.md) | `src/encoders/gamma_laguerre.py` |
| **顺序敏感编码器 (GRU/Transformer/TCN)** | — (直接实现) | `src/encoders/temporal_gru.py`, `temporal_transformer.py`, `temporal_tcn.py` |

---

## 文献调研

完整文献综述见 [docs/papers/literature_review_shape_reconstruction.md](../papers/literature_review_shape_reconstruction.md)。

核心论文笔记：
- [Tang 2026 — 全身形状控制](../papers/notes_tang2026_whole_body_shape.md)
- [Yu 2026 — 可解释形状自建模](../papers/notes_yu2026_shape_interpretable.md)

---

## 与现有工作的差异化

| 维度 | Yu 2026 (arXiv) | Tang 2026 (ICRA) | SoftNeRF (IROS) | **我们** |
|------|---------|-----------|-----------|---------|
| 核心问题 | 形状+控制 | 负载适应 | NeRF 自建模 | **迟滞 + 3D 状态估计** |
| 时序建模 | 无 | 无 | 无 | **Gamma/Laguerre 延迟核** |
| 形状表示 | Bézier 控制点 | 图像隐式 | NeRF 密度场 | **3D 骨架 + 可变表面** |
| 推理方式 | feedforward | 在线优化 | 渲染优化 | **predictive-corrective** |
| 感知 | 2 相机 | 图像反馈 | 多视角渲染 | **单/双相机** |
| 验证平台 | 真实机器人 | 真实机器人 | 真实/仿真 | **PyElastica 仿真** |
