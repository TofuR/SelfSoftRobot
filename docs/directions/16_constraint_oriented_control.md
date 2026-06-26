# 方向：约束导向控制（Constraint-Oriented Control）

> 状态：规划中（设计已定，工程待实现）
> 优先级：高（把科学问题 12 §B 接上工程控制闭环，项目独有卖点）
> 前置：前向状态转移骨架模型（[14](14_gt_observed_transition.md) / [15](15_open_loop_windowed_transition.md)）已可用且可微
> 最后更新：2026-06-19

---

## 问题：为什么"精确目标形状"不是合法的控制原语

软臂控制要回答"机器人应成为什么形状"。直觉做法是给定一个**精确 3D 目标形状**再去求动作——但这在三重意义上不诚实：

1. **物理不可达**：软臂 action 空间是低维（当前 2D），骨架空间是 93D（31 节点×3），可达集是低维流形上的薄切片。绝大多数任意 3D 形状**天然不可达**。要求"精确达到"是在问一个无解问题。
2. **迟滞非单射**（[12 §B](12_scientific_problems_soft_robot_self_modeling.md) 核心）：前向映射 F 非单射——同一形状可由**多条动作历史**达到。给"一个目标形状"会让求解器返回任意一条路径，**欠定**。
3. **目标形状本身就难生成**：用户不知道机器人能否满足，只能尽量优化（tang2026 用目标形状做动作预测+校准，但目标形状的生成本身就是难题）。

**因此本项目对"目标形状"问题的官方立场**：精确目标形状不是合法原语，必须用 **约束/任务空间规范 + 可达性校准的反演** 取代。目标本质是一个**全身条件（whole-body CONDITION）**，不是**全身形状（whole-body SHAPE）**。

---

## 核心立场：三模式目标规范（不互斥，可组合）

### (A) 任务空间约束（hu2025 模式，泛化到全身 + 迟滞感知）

不指定形状，只喂一组约束：
- **空间约束**：节点 k 经过点 p、末端进入区域 R；
- **避障约束**：骨架避开障碍 AABB（复用 `compute_gt_sdf` 的穿透惩罚）；
- **工效约束**：动作平滑（‖a_t − a_{t-1}‖）、序列短、效率高。

求解器找一条**动作序列**使其前向 rollout 满足约束；中间/全身形状**从不指定——emergent**。这是唯一诚实的原语："满足这些"，而非"成为这个"。

### (B) 形状描述符 / 视觉目标（tang2026 模式）

用**图像或学习到的描述符**作为目标：最小化预测骨架的渲染 silhouette / 投影骨架与目标的描述符距离。全身 + 仅相机（符合低成本理念）。精确复现从不要求——只最小化描述符距离，残留 > margin 时**诚实报"不可达"**。

### (C) 可达性校准层（必须先做）

任何朴素目标先经前向模型**投影到可达流形**（找最近可达形状），再反演，并**报告残差间隙**。直接回答用户痛点（"机器人可能满足不了，只能优化"）：把不可能的精确目标转成最近可达目标 + 报残差。

> ⚠️ **防假成功**：若不做 (C)，求解器会收敛到差的局部最优却看似成功（2D action → 93D skeleton，流形薄，多数目标不可达）。**可达性残差报告是整个控制层最不诚实的失败模式，必须最先建好。**

---

## 可微前向黑箱（已验证可微，零修改复用）

反演的基底是前向状态转移模型，**全程 torch 算子、对 action 与 skeleton 都可微**：

- `StateTransitionSpatialModel.forward(action_window, prev_skeleton, prev_prev_skeleton, prev_z) → {skeleton:(B,31,3), latent_z:(B,z_dim)}`
- `src/utils/sdf_utils.compute_gt_sdf(query_points, skeleton, radius) = min_seg(point_to_segment_distance) − radius`，对 skeleton 可微（已验证 grad norm ≈ 11.5）。`<0` 管内 / `>0` 管外，做穿透惩罚天然可微。

收缩性：`s_t = s_{t-1} + delta_scale·tanh(delta_head(...))`，`delta_scale` 收缩保证 action→skeleton 映射 Lipschitz 有界 → 利于 K 步反传收敛。

---

## 三类规划器（作为 mode A 的工程实现）

### (A1) 端点到达
`action_window` 声明 `requires_grad`，`loss = ‖skeleton(â)[:, −1] − target‖²`，Adam 在动作上求梯度。形状/中间构型从不指定。

### (A2) 无碰撞轨迹优化
优化整条动作序列 `a_{1:T}`：
```
loss = 到达损失 + Σ hinge(margin − sdf)穿透惩罚 + 动作平滑 + 效率
```
每步用 forward 滚动 `(s, z)`，**open_loop 语义**（1 帧 GT 种子 + K 步自回归，与训练分布一致，复用 `scripts/evaluation/eval_rollout.py` 的 `rollout_windowed_one_sequence` 结构）。

### (A3) Jacobian 控制（yu2026 模式）
`J_p = ∂ee/∂a`、`J_s = ∂(31 节点)/∂a` 用 `torch.autograd.functional.jacobian` **直接从 forward 求数值雅可比**（无需 Neural ODE、无需手推物理）：
```
u = J_p†·(K_p·e_p) + J_s†·(K_s·e_s + v_obs)
```
形状状态 = 直接预测的 31 节点 3D 骨架（显式、稠密、几何语义强，区别于 yu2026 的 Bézier 控制点）；SDF = 解析可微管状 SDF（区别于额外学一个隐式场）。避障逃逸速度 `v_obs` 用最近点 SDF 梯度经 `J_s†` 映射成全身形状调整（自运动：避障同时保持末端）。

---

## 序列级 + 迟滞感知（与 hu2025 的关键区别）

**hu2025 反演的是静态关节向量（忽略迟滞——恰是我们的状态转移族要修复的失效模式）。本项目必须反演动作序列**，以捕获路径依赖 / z：

- **反演动作序列** `a_{1:K}`，而非单个 action → 同 action 不同历史产生不同形状被正确建模。
- **z_0 处理**：作为自由变量联合优化；或从单帧视觉快照经冻结 forward-pass / encoder 估计（与 [12 §A 形状即记忆](12_scientific_problems_soft_robot_self_modeling.md) 紧耦合——z 即当前内部应力状态）。
- **K（horizon）取自 [12 §B](12_scientific_problems_soft_robot_self_modeling.md) 的临界记忆长度 T\***（消歧 IK 所需历史），使反演窗口匹配物理消歧视界；且 `K ≤ 40`（[15](15_open_loop_windowed_transition.md) 的开环漂移约束）。

> 这是把 [12 §B 的记忆信道理论] 接上 [工程控制闭环] 的桥——项目独有的论文卖点。

---

## 归一化契约（易错点，必须严守）

- actions `/ norm_factor` 喂 forward；
- 骨架在 `pc_center / pc_scale` 归一化空间算 loss；
- **目标点 / 障碍点必须全部转到同一空间**（建议全转归一化空间算 loss，最后只反归一化可视化），用 `model.pc_center / pc_scale` 统一转换。

---

## 在线校正（tang2026 思想）

每步用观测（仿真 GT `positions[t-1]` 或实物骨架化）作 `s_{t-1}` 种子**重锚位姿漂移**（[GTObservedTransitionModel](14_gt_observed_transition.md) 语义天然支持）。把未知载荷 / 磨损的形状误差 `e_s = skel_observed − skel_predicted` 反馈进 `J_s` 残差通道 → 同一前向模型即泛化到训练分布外载荷。

---

## 与三篇文献的关系（借什么 / 不借什么）

| 文献 | 借鉴 | 不借鉴（理由） |
|------|------|----------------|
| **hu2025**（FBV-SM） | query_model(a)→形状作可微黑箱、Adam-over-control、A*/RRT、碰撞用占据；"只喂约束让模型自寻路径"的范式骨干 | 静态关节反演（忽略迟滞——我们要修复的失效模式） |
| **yu2026**（shape-interpretable） | `J_p/J_s` 混合控制器结构、双视角"至少一视角不撞则 3D 不撞"不变性、最近点排斥逃逸速度→`J_s†` 映射 | Bézier 控制点作形状状态（我们用更稠密的 31 节点骨架）；Neural ODE 求 Jacobian（我们用 autograd） |
| **tang2026**（whole-body control） | 视觉形状描述符作目标（mode B）、在线模型校正（每步观测重锚） | 在线 CNN 策略再调参（超范围，与"先建模 forward map"的项目身份冲突） |

---

## 模块计划（薄包装，不动现有 forward 模型）

```
src/inverse/                         # 新目录（additive，无 breaking change）
  __init__.py
  planner.py                         # SkeletonConstraintPlanner：可微 rollout + 约束 loss + Adam over (a_seq, z_0)
  constraints.py                     # 点过 / 区域 / 碰撞 AABB / 描述符匹配（via 2D 骨架投影，用 camera.py）
  reachability.py                    # (C) 可达性投影 + 残差报告
scripts/planning/
  solve_target.py                    # CLI 入口（端点到达 / 无碰撞轨迹 / Jacobian 控制）
```

rollout 复用 `eval_rollout.rollout_windowed_one_sequence`，把"喂 GT 动作序列"换成"喂优化出的动作序列"。前向模型零修改（本就在训练 BPTT 中被反传调用）。

---

## 风险

| 级别 | 风险 | 缓解 |
|------|------|------|
| 🔴 TOP | **常数半径假设下的碰撞保守性**：`compute_gt_sdf` 第二参数固定 radius；实物软臂弯曲内侧压缩/外侧拉伸（截面真变化），碰撞查询系统性偏保守/偏激进 → 控制器物理不安全 | 碰撞保守取 `max(预测半径)` 包络；文档明确"常数半径近似下的保守余量"；待 [05 Phase 1](05_skeleton_to_shape_conversion.md) 变半径落地后细化 |
| 🔴 | **hard-min 梯度跳变**：`compute_gt_sdf`/`point_to_segment_distance` 用 `.min()` + `argmin`+`gather`，梯度只流回最近段，碰撞接近多段汇合处时梯度跳变 → 反演/控制卡死 | soft-min（weighted sum over segments, temperature 退火）替代 hard min。修一次，形态层（[05 Phase 1](05_skeleton_to_shape_conversion.md) 逐节点半径 gather）与控制层共享受益 |
| 🟠 | **action 雅可比范数小**：`delta_scale·tanh` 收缩使 action→skeleton 雅可比偏小（实测 action grad norm ≈ 0.11 vs prev_skeleton 9.6），Adam 反演收敛慢/陷平坦区 | 较大学习率 + 多起点重启 + 动作 normalize/clip；horizon `K ≤ 40` |
| 🟠 | **迟滞非唯一性**：F 非单射，求解器返回任意一条可达路径 | 加路径偏好 / 最小动作正则；报告 ambiguity-set 大小 |
| 🟡 | **z_0 不可观测**：无 z 测量手段时反演条件化在猜测的内部状态上 | z_0 作自由变量联合优化，或从单帧视觉快照估计（接 [12 §A](12_scientific_problems_soft_robot_self_modeling.md)） |
| 🟡 | **长 horizon 计算图**：K 步 rollout 显存/时间随 K 线性增长 | 截断/展开 BPTT + checkpointing；或 [15](15_open_loop_windowed_transition.md) 窗口重播种限 K |

---

## 落地优先级（按 ROI × 风险加权）

1. **立即（1 周内，零风险）**：mode (C) 可达性残差报告 + mode (A1) 端点到达原型。forward 模型已可微、per-frame MSE~1e-8 已达，反演立即可跑——这是把"科学问题 B（T* 可逆性）"接上"工程控制闭环"的最快路径。
2. **短期（1–2 周）**：mode (A2) 无碰撞轨迹优化（接 [05 Phase 1](05_skeleton_to_shape_conversion.md) 变半径后的碰撞查询）。
3. **中期（1 月）**：mode (A3) Jacobian 控制 + soft-min 几何修复。
4. **论文级**：mode (B) 视觉描述符目标 + 在线校正（实物）+ 与 [12 §A/B](12_scientific_problems_soft_robot_self_modeling.md) 记忆信道的联合实验。

---

## 交叉引用

- 前向模型：[13](13_closed_loop_state_transition.md) / [14](14_gt_observed_transition.md) / [15](15_open_loop_windowed_transition.md)
- 形态表达（碰撞用的全身 SDF）：[05 骨架→形状](05_skeleton_to_shape_conversion.md)
- 科学基础（T* 选 horizon、可达性诊断）：[12 §A/B](12_scientific_problems_soft_robot_self_modeling.md)
- 实物部署（在线校正载体）：[10 视觉辅助部署](10_vision_corrected_deployment.md) / [11 sim-to-real](11_sim_to_real_transfer.md)
- 文献笔记：[hu2025](../papers/hu2025_paper_understanding.md) / [yu2026](../papers/notes_yu2026_shape_interpretable.md) / [tang2026](../papers/notes_tang2026_whole_body_shape.md)
