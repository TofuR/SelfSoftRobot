# 方向：约束导向控制（Constraint-Oriented Control）

> 状态：**部分实现**（方向1 视野认证 ✓ + 方向2 可微逆规划 ✓ 已验证；设计详见正文，工程落地进行中）
> 优先级：高（把科学问题 12 §B 接上工程控制闭环，项目独有卖点）
> 前置：前向状态转移骨架模型（[14](14_gt_observed_transition.md) / [15](15_open_loop_windowed_transition.md)）已可用且可微
> 部署模型：**open_loop**（[15](15_open_loop_windowed_transition.md)，形状控制部署目标；gt 仅训练基础/精度上界，见下方"实现进展"）
> 脚本：`scripts/evaluation/eval_horizon.py`（方向1）· `scripts/control/inverse_plan.py`（方向2）
> 最后更新：2026-07-14

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

## 实现进展（2026-07-14）：视野认证 + 逆规划已验证

> 部署模型确定为 **open_loop**（[15](15_open_loop_windowed_transition.md)），非 gt——形状控制需"给定动作 → 预测形状"的规划能力，gt 每步观测只适合作训练基础/精度上界（详见 [[open-loop-deployment-target]] 记忆）。下用 open_loop checkpoint（SAM2-clean 数据，action_dim=1，N=15，episode_len=40）实测。

### 方向1：纯自回归视野认证（`scripts/evaluation/eval_horizon.py`）

逆规划把候选动作序列喂前向模型 rollout 评估——若模型漂移，规划动作无法迁移到真机。故先认证前向模型能否当"规划级仿真器"。8 种子纯自回归 rollout（1 帧 GT 种子，之后不再观测，300 步）：

| 模型 | drift@300步 | K_max@5px | K_max@10px | K_max@drift3× | z_norm 轨迹 |
|---|---|---|---|---|---|
| **open_loop** | **1.7×** ✓ | 51 步 | **124 步** | 135 步 | 0.00→0.00（惰性） |
| gt（对照） | 272× ✗ | 53 | 91 | 9 | 0.00→0.66 |

**结论**：open_loop 是可用规划仿真器（drift 仅 1.7×）；gt 不可用（272× 爆炸）——直接证实部署选 open_loop。可信视野 **K_max ≈ 50（紧，~5px 均值节点）~120（松，~10px）**，远超训练 episode_len=40（泛化良好）。**诚实警告**：open_loop 的 z 惰性（≈0），稳定性部分源于 z 坍缩——对规划良性（要的是稳定仿真器），但对"z 建模迟滞"的故事是警告。图：`output/horizon/horizon_comparison.png`。

### 方向2：可微逆规划（`scripts/control/inverse_plan.py`，shooting 法）

给定 s_init + s_target，动作序列 `requires_grad`，K 步 rollout（带梯度）后 backprop 进动作，Adam + 投影到真实动作范围 + 多起点。reach 任务 t500→t540（目标 = t_init+K，GT 可验证）：

| 方案 | 末态 vs s_target（均值节点 px / 末端 px） |
|---|---|
| 初始差距 s_init→s_target | 2.29 / 6.48 |
| do-nothing（重复末动作） | 8.16 / 22.27（漂移远离目标） |
| GT-actions（真实动作 rollout） | 2.69 / 0.81（模型保真上界） |
| **planner（优化动作）** | **3.07 / 5.09** |

**结论**：planner 到达目标 3.07px（均值节点），**0.38× of do-nothing（2.6× 更优）**，接近 GT-actions 保真上界（3.07 vs 2.69，+14%）——证明在 open_loop 仿真器上逆规划有效。图：`output/inverse_plan/plan_trajectory.png`。末端误差（5.09px）高于 GT-actions（0.81px）因 loss 等权所有节点、末端未加权（tip-weighted loss 可改善）。

### 关键关系：方向1 认证仿真器，方向2 在其上规划

方向2 的可靠性**完全依赖**方向1：规划 = 在仿真器里优化动作序列，若仿真器漂移则动作无法迁移真机。**K_max 是方向2 规划视野的硬上限**（K ≤ K_max 才可信，故本方向 §序列级里原写的 `K ≤ 40` 偏保守，实测可放到 ~120）。z 的长程稳定性是核心（规划时 z 无 GT 完全自演化）。这就是"方向1 是方向2 基础"的精确定义，也是把"步数惩罚"接进 loss 的依据（惩罚 K > K_max 的解）。



## 方法与验证详解（问答 + 可视化）

> 回答："认证怎么做 / gt 为何失败 / 没连机器人怎么验证 / 一次能推多久 / 方法到底如何"。

### Q1·纯自回归视野认证具体怎么做？

模拟的正是部署场景"观测一次真实姿态，之后只靠动作往前推"：

1. 取 1 帧 GT 骨架作种子 ŝ_0 = positions[t0]（这一帧"看了一眼"真实图像）；
2. 之后 k=1..K 步：每步只喂【动作窗口 + 上一步模型自己的预测 ŝ_{k-1} + 演化的 z】，**不再看任何真实图像**；
3. 记录每步 ŝ_k 与真实 positions[t0+k] 的误差 → error-by-k 曲线；
4. **K_max** = 误差首次越过容差的步数 = "模型能可信地往前推多久"；
5. 多种子（8 个不同 t0）聚合，统计稳健。

**可视化** `output/viz/horizon_rollout_grid.png`：预测臂（色）叠在真实臂（灰虚线）上，k=1→300（0.2s→61s）。open_loop 行预测臂始终贴合真实臂；gt 行 k>40 后预测臂飞走。`horizon_rollout_{open_loop,gt}.gif` 是动画版。脚本 `scripts/evaluation/viz_control.py`。

### Q2·为什么 gt 不能用、open_loop 能用？（你的直觉对：训练信息泄漏）

**核心是 train/inference gap（teacher forcing 的代价）**：

- **gt 训练 TF=1.0**：每一步的 s_{t-1} **永远喂真实值**。模型从没见过"自己的预测当输入"，没机会学"从带误差的状态自我修正"。
- **gt 开环推理**：必须喂自己的预测 → 输入落到训练分布外（带误差的 s）→ 小误差被放大 → 300 步漂移 **272×**。
- **open_loop 训练 TF=0**（退火到 0）：窗口内**故意喂模型自己的预测** → 模型显式学习了"在自身预测分布下保持稳定" → 300 步漂移仅 **1.7×**。

**你的判断完全正确**：gt 训练时"给的信息太多"，反而没学到开环所需的能力。这就是 open_loop 是**部署目标**、gt 退为**训练基础**的根本原因——要在自身预测上跑的，必须在自身预测上训。

### Q3·没连接机器人，逆规划怎么验证？（关键诚实点）

**前向模型 = 从真实数据学出来的"机器人仿真器"**（10214 帧真实物理，学到了 action+形状 → 下一帧形状 的映射），故可用它代替真机做规划。验证分三层：

| 层 | 做什么 | 结果 | 说明 |
|---|---|---|---|
| ① 模型保真（GT-actions 基线） | 把**真实录制时的实际动作**喂模型 rollout，比真实目标形状 | 末端 **0.81px** ≈ NDI 噪声底 | **模型对真实物理保真** → 仿真器可信 |
| ② 规划器 | 在可信仿真器里优化动作序列 | 末端 **3.07px** | 找到一条模型认为能到目标的动作 |
| ③ 对照（do-nothing） | 不规划，重复末动作 | 8.16px（更差） | 证明规划器确实在做事 |

**但这是"模型内验证"——规划与评估用同一个模型**。① 部分打破循环（GT-actions 来自真实物理，模型能复现 → 证明模型可信），但 **planner 自己优化的动作还没上真机验证过**。真机验证 = 把规划动作发到 PLC → 真机执行 → RealSense+NDI 测真实形状 → 比 target。**这是部署阶段，需硬件闭环，当前未做**。我们在 **val 集**（模型没训练过的真实轨迹）上跑，提供一层泛化保证（非纯过拟合）。

**可视化** `output/viz/plan_reach_compare.png`：三面板 planner / GT-actions / do-nothing 的 s_init→轨迹→s_target。`plan_reach.gif`：planner 逐步驱动 init→target 的动画。

### Q4·时间换算（0.2s/帧）

实测 `real_capture/.../frame_times.txt`：dt = **0.203s**（≈5fps），总录 10214 帧 ≈ 34.6 分钟。

| 视野 | 步数 | 秒 |
|---|---|---|
| 训练 episode_len | 40 | 8.1s |
| K_max @ 紧（5px 均值节点） | 51 | 10s |
| K_max @ 松（10px） | 124 | 25s |
| 认证最长 | 300 | 61s |

→ **单次开环规划可信 ~10–25s**。更长机动需 **receding horizon（滚动重规划）**：每执行 N < K_max 步后重新观测 + 重规划。

### 这些方法究竟怎么样？（诚实评估）

- **方向1（视野认证）**：open_loop 作仿真器，**25s 内可信**，单次规划够用。隐患：z 惰性（≈0），稳定性部分来自 z 坍缩而非真迟滞记忆——对规划良性，但削弱了"z 建模迟滞"的论文卖点。
- **方向2（逆规划）**：reach 3.07px（均值）/ 5.09px（末端），接近模型保真上界（GT-actions 2.69px），**证明"学习仿真器上做逆规划"可行**。短板：末端精度需 tip-weighted loss；速度慢（40 步 BPTT ~1s/iter，实时需 CMA-ES / 并行 / L-BFGS）；**未上真机**。
- **下一步优先级**：① 真机闭环验证（最重要，证明迁移）→ ② receding horizon（长机动）→ ③ 可达性校验（mode C，本方向原设计）→ ④ tip 加权 + 速度优化。

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

1. **立即（1 周内，零风险）**：✅ mode (A1) 端点到达原型**已实现**（`scripts/control/inverse_plan.py`，shooting 法，reach 3.07px / 0.38× do-nothing，见上方"实现进展"）。✅ 视野认证已实现（`scripts/evaluation/eval_horizon.py`，K_max ~50-120）。⏳ mode (C) 可达性残差报告待做——这是把"科学问题 B（T* 可逆性）"接上"工程控制闭环"的下一步。
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
