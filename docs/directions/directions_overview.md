# 研究方向总览 (Research Directions Overview)

> 基于文献调研和深度讨论，从"出发点（真问题）"推导出的研究方向体系
> 核心出发点：软体机器人自建模 = 带迟滞的 3D 全身状态估计
> 最后更新：2026-07-28

---

## ⚠️ 2026-07-28 三条决策(改变了下面若干方向的优先级)

1. **方法收敛到状态转移族**:只留 `open_loop_transition`(部署主线)与 `gt_transition`(论文消融)。C-MSTNF / MS-SCNF / SDF / SkeletonSDF / FlowMatch / SpatialSequence 的实验日志已归档到 `train_log/_archive/`(清单:[`docs/archived/2026-07-28_archive_manifest.md`](../archived/2026-07-28_archive_manifest.md));**代码暂不动** —— 理由见清单 §4:4 处按 model type 分派,且 6 个时序编码器与 `dataset_pointcloud.py` 都是主线硬依赖。
2. **3D 多自由度**:后续 6 个通道全部施加气压驱动。理由是 1-DOF 平面运动下"到达目标 + 避障"命题本身不成立 —— **只有 3D 冗余自由度才有零空间**。状态来源定为**双/多相机标定 + 三角化**。
   → **这反转了"免标定"不变量**:`HANDOFF §7.2 #7` 原写"`calibrate_cameras.py` / `capture_to_npz.py` 是遗留标定路线,别用于路线 B",现在它们变成主线基础设施。
   → **方向 06 与 08 从低优先级升为 ★★★ 主线**。
3. **mask 源**:训练继续用 SAM2;**在线改跑 SAM2 前向流式**。已知缺口:前向流式与训练用的双向分块仍有差异(锚帧现在来自启发式修复的干净帧,在线无此来源);GPU 单帧延迟未实测(采集节拍 0.2 s 是预算)。

---

## 出发点

> **软体机器人的当前状态不仅取决于当前驱动，还取决于驱动历史。但现有的 2D 视觉方法无法从历史观测中恢复带时序依赖的 3D 全身状态，而 3D 运动捕捉方法成本过高不实用。**

三个子问题：
1. **时序依赖（迟滞）**：同一个 action 在不同历史下产生不同形状
2. **3D 全身状态**：不只是末端坐标，也不只是 2D 图像上的曲线
3. **低成本感知**：只用相机，不用运动捕捉

---

## 状态转移主线（当前工作焦点）

> **当前主线 = 方向 15（稀疏观测窗口 OpenLoop）**。最终场景无法每步获得真实形态，模型每个窗口只接收一次状态锚点，随后根据动作和自身预测自由运行。方向 14 保留为局部转移误差与累计误差的诊断上界。状态转移模型族按“前一状态 s_{t-1} 从哪来”分为三姊妹方向：

| 方向 | 模型 | s_{t-1} 来源 | 状态 |
|------|------|------------|------|
| [13 闭环状态转移](13_closed_loop_state_transition.md) | `StateTransitionSpatialModel` | 预测（**无界** rollout） | **已被 14/15 取代**(实测漂移 1170×,不可用);**基类本身保留** —— gt/open_loop 都继承它。日志已归档 |
| [14 全 GT 驱动](14_gt_observed_transition.md) | `GTObservedTransitionModel` | **每步真实观测** | ✅ **论文消融 + 精度上界**(2026-07-28 明确其论文角色);不作为部署 |
| [**15 窗口开环(主线)**](15_open_loop_windowed_transition.md) | `OpenLoopTransitionModel` | 预测，**每 K 步重观测** | ✅ **部署主线**;唯一保留的活模型 |

三者共享可学习迟滞潜变量 z（无 GT，端到端学）与 Δ 预测 + 收缩约束设计（理论基础见 [13 §一](13_closed_loop_state_transition.md)）。**工作模型族**：`SpatialSequenceModel`（前馈基线）、`StateTransitionSpatialModel`、`GTObservedTransitionModel`（诊断）、`OpenLoopTransitionModel`（主线）、`FlowMatchPointCloudModel`。

> **下一阶段（形态 / 控制接入主线）**：当前 OpenLoop 主线只输出**骨架**。后续两步构成完整闭环：
> - **全身形态**：[05](05_skeleton_to_shape_conversion.md) 把骨架升级为全身形态（Phase 0 接口 → 1 半径场 → 2 theta 截面 → 3 时序一致）；
> - **约束导向控制**：[16](16_constraint_oriented_control.md) 用前向模型作可微黑箱，给定避障点/任务约束反求动作序列（**迟滞感知**，区别于 hu2025 静态关节求逆）。
>
> 即「**稀疏观测 OpenLoop 骨架预测 → 全身形态 → 路径依赖 IK / 约束导向控制 → 组合应用**」。机制—任务—应用三层验证方案见 [实验主方案](../experiments/openloop_sparse_observation_validation_plan.md)。实物数据采集已打通（[11 §最小验证平台](11_sim_to_real_transfer.md) + `real_capture/`）。
>
> **实物免标定 2D 工作流已落地并训练验证**：1-DOF 双段臂 + 单相机 + 免标定，2D 图像骨架 `[col,row,0]` 作 state，NDI 末端作度量验证。GT 模型（`gt_transition exp_20260709_5`）末端误差 mean 0.77mm / median 0.57mm，已到 NDI 仿射标定底（0.74mm）；骨架+常数半径 r=14 即得形态 IoU 0.91（vs repaired mask），说明形态≈骨架+管、NN 空间小。详见 [docs/research/2026-07-10-real-data-2d-workflow.md](../research/2026-07-10-real-data-2d-workflow.md)。

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
    │  │自回归状态动力学 │ ← 已实现(13/14/15) │
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
    │  │骨架→形状转换      │  │  Phase 0-3 迁移（接口/半径/截面/时序）
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
    │  │多视角 2D→3D 骨架  │  │  双视角三角化/可微渲染（实物已落地）
    │  │(multi_view_2d_    │  │
    │  │ to_3d_skeleton)   │  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │视觉辅助部署       │  │  在线适应 + 残差修正
    │  │(vision_corrected) │  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │Sim-to-Real 迁移   │  │  残差物理/域随机化（实物采集已打通）
    │  │(sim_to_real)      │  │
    │  └──────────────────┘  │
    │  ┌──────────────────┐  │
    │  │约束导向控制       │  │  前向黑箱求逆（迟滞感知）
    │  │(constraint_ctrl)  │  │
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
| **自回归状态动力学** | [01（已归档）](../archived/directions/01_autoregressive_state_dynamics.md) | "当前在哪" | 前一步物理状态作为输入 | 已实现(见13/14/15) |
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
| **骨架→形状转换** | [05_skeleton_to_shape_conversion.md](05_skeleton_to_shape_conversion.md) | Phase 0-3 迁移主线（接口→半径场→theta 截面→时序一致）；已有半径偏移基线 `skeleton_to_shape.py`（骨架+常数半径 r=14 → IoU 0.91）作骨架引导形态预测的 v0 | ★★★ |
| **拓扑引导残差流** | [09_topology_guided_residual_flow.md](09_topology_guided_residual_flow.md) | 物理粗变形 + FM 残差 | **搁置**(仿真路线遗留) |
| **从轮廓恢复形状** | [07_shape_from_silhouette.md](07_shape_from_silhouette.md) | 骨架条件 Visual Hull | **搁置**(与 05 竞争同一目标,05 已有 IoU 0.91 基线) |

---

## 第三层：感知与部署

从仿真走向实际应用的路径。

| 方向 | 文档 | 核心思想 | 优先级 |
|------|------|---------|--------|
| **多视角 2D→3D 骨架** | [06_multi_view_2d_to_3d_skeleton.md](06_multi_view_2d_to_3d_skeleton.md) | 双视角三角化 / 可微渲染 | **★★★ 主线**(2026-07-28 升级:3D 多自由度的状态来源就是它) |
| **视觉辅助部署** | [10_vision_corrected_deployment.md](10_vision_corrected_deployment.md) | 在线适应 + 残差修正 | ★★☆ |
| **Sim-to-Real 迁移** | [11_sim_to_real_transfer.md](11_sim_to_real_transfer.md) | 残差物理 / 域随机化 | **搁置**(实物已直接采数训练,不再需要迁移) |
| **约束导向控制** | [16_constraint_oriented_control.md](16_constraint_oriented_control.md) | 前向模型作可微黑箱，给定约束反求动作序列（迟滞感知） | ★★☆ |
| **路径依赖 IK（论文方向）** | [17_path_dependent_ik.md](17_path_dependent_ik.md) | 软体臂 IK 是泛函非函数：准静态方法在动态/循环加载下物理性失败 + history-aware 修复；实物 hysteresis loop 已验证（路径依赖 1.5–2mm） | ★★★ |

> **下一篇论文的骨架**：[17](17_path_dependent_ik.md) 把 [16](16_constraint_oriented_control.md) 的控制工程升级为科学问题——"软体臂 IK 路径依赖"。三问检验①（迟滞真实）已用原始数据验证；②（无记忆方法失败）待训 window=1 对照。

---

## 第四层：建模方法论

| 方向 | 文档 | 核心思想 | 优先级 |
|------|------|---------|--------|
| **单 DOF 分解与组合** | [08_per_dof_decomposition.md](08_per_dof_decomposition.md) | 独立训练 + 模态叠加 | **★★★ 主线**(2026-07-28 升级:6 通道驱动直接需要同段腔道竞争 / 跨段耦合 / 组合泛化) |

---

## 第五层：科学问题（超越工程优化）

> 不是"换一个模型"或"加一个模块"，而是发现关于软体机器人物理世界的新规律。

| 方向 | 文档 | 核心科学问题 | 类型 |
|------|------|------------|------|
| **形状即记忆** | [12_scientific_problems.md](12_scientific_problems_soft_robot_self_modeling.md) §A | 当前形状能解码多少加载历史？ | 信息论 |
| **IK 可逆性** | [12_scientific_problems.md](12_scientific_problems_soft_robot_self_modeling.md) §B | 迟滞条件下逆运动学何时有唯一解？ | 控制论 |
| **视觉材料发现** | [12_scientific_problems.md](12_scientific_problems_soft_robot_self_modeling.md) §C | 机器人能否从视觉观测推断自身材料属性？ | 逆问题 |

```
物理记忆
   ╱        ╲
读取（A）   写入/消除歧义（B）
   ╲        ╱
  粘弹性记忆信道
      ↑
材料属性决定信道（C）
```

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
| **自回归状态动力学** | [01_autoregressive_state_dynamics.md](../archived/directions/01_autoregressive_state_dynamics.md) | `src/models/model_state_transition.py` + `model_gt_transition.py` + `model_open_loop_transition.py`（方向 13/14/15） |

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
