# 研究方向总览 (Research Directions Overview)

> 基于文献调研和深度讨论，从"出发点（真问题）"推导出的研究方向体系
> 核心出发点：软体机器人自建模 = 带迟滞的 3D 全身状态估计

---

## 出发点

> **软体机器人的当前状态不仅取决于当前驱动，还取决于驱动历史。但现有的 2D 视觉方法无法从历史观测中恢复带时序依赖的 3D 全身状态，而 3D 运动捕捉方法成本过高不实用。**

三个子问题：
1. **时序依赖（迟滞）**：同一个 action 在不同历史下产生不同形状
2. **3D 全身状态**：不只是末端坐标，也不只是 2D 图像上的曲线
3. **低成本感知**：只用相机，不用运动捕捉

---

## 方向关系图

```
出发点: 带迟滞的 3D 全身状态估计
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
┌────────────────┐                         ┌──────────────┐
│  迟滞如何建模？  │                         │ 状态如何表示？ │
│                │                         │              │
│ ┌────────────┐ │                         │ ┌──────────┐ │
│ │ 分数阶记忆核 │ │                         │ │ 空间序列  │ │
│ │ (替代 EMA) │ │                         │ │ (z-slice)│ │
│ └────────────┘ │                         │ └──────────┘ │
└────────┬───────┘                         └──────┬───────┘
         │                                        │
         │    预测分支                              │ 预测分支
         │        │                                │   │
         └────────┼────────────────────────────────┘   │
                  ▼                                        │
         ┌────────────────┐                               │
         │ 预测-修正框架   │◄──────────────────────────────┘
         │                │
         │  预测: history  │
         │  修正: vision   │
         └────────┬───────┘
                  │
                  ▼
         ┌────────────────┐
         │ 拓扑引导残差流  │  ← 可选的精细化模块
         │ (物理先验+残差) │
         └────────────────┘
```

---

## 各方向文档

| 方向 | 文档 | 解决的子问题 | 核心思想 |
|------|------|------------|---------|
| **分数阶记忆核** | [fractional_order_memory.md](fractional_order_memory.md) | 时序依赖 | 用分数阶微积分替代 EMA，物理上有根据 |
| **预测-修正框架** | [predictive_corrective_state_estimation.md](predictive_corrective_state_estimation.md) | 2D→3D 信息瓶颈 | 时序先验 + 视觉修正，类似 Kalman 滤波 |
| **空间序列生成** | [spatial_sequence_generation.md](spatial_sequence_generation.md) | 3D 状态表示 | z-slice 截面参数序列替代原始点云 |
| **拓扑引导残差流** | [topology_guided_residual_flow.md](topology_guided_residual_flow.md) | 精细化 | 物理粗变形 + Flow Matching 学习残差 |

---

## 三个子问题 → 四个技术方案

| 子问题 | 对应技术 | 详细文档 |
|--------|---------|---------|
| 1. 如何编码迟滞历史？ | **分数阶记忆核** | [fractional_order_memory.md](fractional_order_memory.md) |
| 2. 如何表示 3D 全身状态？ | **空间序列生成** | [spatial_sequence_generation.md](spatial_sequence_generation.md) |
| 3. 如何从 2D 视觉推断 3D？ | **预测-修正框架** | [predictive_corrective_state_estimation.md](predictive_corrective_state_estimation.md) |
| (可选) 如何精细化？ | **拓扑引导残差流** | [topology_guided_residual_flow.md](topology_guided_residual_flow.md) |

每个技术都是为解决出发点中的具体子问题而自然出现的，不是模块叠加。

---

## 与现有工作的差异化

| 维度 | Yu 2026 | Chen 2025 | **我们** |
|------|---------|-----------|---------|
| 核心问题 | 负载适应 | 迟滞建模 | **视觉 + 迟滞 + 3D 状态估计** |
| 时序建模 | 无 | 方向 sign | **分数阶记忆核** |
| 推理方式 | feedforward | feedforward | **predictive-corrective** |
| 感知 | 2 相机 | OptiTrack 8 相机 | **单/双相机** |

---

## 建议实施路线

```
Phase 1: 验证核心假设
  ├── 量化仿真器中的迟滞效应（数据说话）
  ├── 实现 FractionalMemory 替代 EMA
  └── 对比实验：EMA vs 分数阶 vs 无时序

Phase 2: 构建预测分支
  ├── 实现 SpatialGRU 空间序列生成
  ├── 用 PyElastica 3D GT 直接监督
  └── 验证截面参数表示的充分性

Phase 3: 加入修正分支
  ├── 实现视觉修正网络
  ├── 可微渲染提供训练信号
  └── 消融：有/无修正的对比

Phase 4: 整合与迁移
  ├── 组合所有模块
  ├── Sim-to-Real 域适应
  └── 真实软臂部署验证
```

---

## 历史版本

旧版 5 方向体系（已归档）：
- direction_1: 形态发现
- direction_2: 纯 2D 自建模
- direction_3: 多相机系统
- direction_4: Sim-to-Real 迁移
- direction_5: 时序迟滞建模

这些方向的想法已被整合到新体系中。详见 [brainstorm_research_directions.md](../papers/brainstorm_research_directions.md)。
