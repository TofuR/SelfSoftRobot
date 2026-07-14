# 相关工作与灵感综述 (Related Work & Inspirations)

> 本文档合并三份调研材料：
> - `docs/archived/inspirations.md`（灵感记录与方向探索）
> - `docs/archived/literature_innovations.md`（文献创新点与思路总结）
> - `docs/archived/research/2026-05-16-self-modeling-literature-review.md`（自建模文献调研与改进方案）
>
> 已去重：同一篇论文/同一个想法只保留最完整表述。论文深度阅读笔记仍在 `docs/papers/`，本文为综述层。
>
> 术语统一：canonical field = 标准场；deformation field = 变形场；continuum robot = 连续体机器人；self-modeling = 自建模；hysteresis = 迟滞。

---

## 一、领域背景 (Domain Background)

本项目处于三个研究方向的交叉点：**软体机器人自建模**、**神经场用于可变形体**、**粘弹性迟滞建模**。

### 1.1 软体机器人自建模 (Soft Robot Self-Modeling)

传统机器人依赖 CAD 模型或手工运动学标定。**自建模 (self-modeling)** 主张让机器人通过自我观察（视觉、本体感知）自动构建自身的几何/动力学模型，无需人工标定。代表性脉络：
- Bongard/Lipson 的形态发现 (morphological discovery)：机器人通过 babbling 探索自身形状。
- Chen et al. 2022 提出"全身视觉自建模"，从不完美的局部自我感知推断完整 3D 形态。
- FBV-SM (Hu et al. 2025) 将 NeRF 引入自建模，让刚性臂学会"输入关节指令→预测自己外观"的自仿真模型。
- 近期 (2025)：Tang et al. (ICRA) 与 Yu et al. (arXiv) 在真实硅胶/线驱动臂上实现了全身形状自建模与几何感知控制。

**核心挑战**：从不完整观测（遮挡、单视角）推断完整形态；未知/时变的材料响应；从"形状预测"过渡到"形状控制"。

### 1.2 神经场 / NeRF 用于可变形体 (Neural Fields for Deformable Objects)

NeRF (Mildenhall et al. 2020) 用 MLP 将 5D 坐标映射为颜色+密度，通过可微体渲染从多视角 2D 图像隐式重建 3D 场景。其后续向**动态/可变形场景**扩展：
- **D-NeRF** (Park et al. 2021)：canonical 场（静态形状）+ deformation 场（时刻 t 的位移），解耦形状与运动。
- **Nerfies** (Park et al. 2021)：变形基函数保证变形光滑，处理人脸/宠物等非刚性场景。
- **HyperNeRF** (Park et al. 2022)：超维切片处理拓扑变化（出现/消失、折叠）。
- **BANMo** (Yang et al. 2022)：骨架驱动变形，从随意视频构建可动画 3D 动物模型。
- **BARF** (Wang et al. 2021)：课程式位置编码，联合优化 NeRF 与相机位姿。

将神经场迁移到机器人领域的关键改造：用**关节角度/驱动参数**替代 NeRF 的"视角"作为条件（FBV-SM），进而用**动作时序窗口**替代（本项目的软体机器人场景，因为软体机器人有惯性/迟滞，非马尔可夫）。

### 1.3 粘弹性迟滞建模 (Viscoelastic Hysteresis Modeling)

软体机器人的硅胶/气动材料是**粘弹性体**：当前形状不仅取决于当前驱动，还取决于**加载历史**（速率、顺序、残余变形）。这是软体机器人区别于刚性机器人的本质物理特征：

- 刚性机器人：`当前状态 = f(当前动作)` → 无记忆，马尔可夫。
- 软体机器人：`当前状态 = f(动作历史)` → 有记忆，非马尔可夫。

经典迟滞建模工具：Preisach 模型、Bouc-Wen、Prandtl-Ishlinskii、分数阶导数（记忆核为幂律衰减）。**分数阶导数**特别契合：它的记忆核 `t^{-α}` 是幂律而非指数衰减，能同时建模短期快速衰减与长期慢衰减。本项目的 `FractionalMemory` 编码器即采用分数阶幂律记忆核。

**关键观察**：现有几乎所有软体机器人自建模工作（Tang、Yu、SoftNeRF、3DGS）都隐式假设了马尔可夫性——它们在"准静态"条件下采集数据（等待稳定再拍照），从而排除了瞬态与历史效应。一旦实验条件包含动态加载/循环加载/变速加载，这些方法会系统性失败。

---

## 二、关键相关工作 (Key Related Work)

按四个分组组织：自建模、神经场 shape、迟滞与记忆、视觉闭环控制。

### 2.1 自建模 (Self-Modeling)

#### FBV-SM (Hu et al. 2025) — Teaching Robots to Build Simulations of Themselves
让机器人通过单相机自我观察学会"自仿真模型"：输入关节指令即可预测自身外观，并用于运动规划（梯度优化、A*/RRT 碰撞检测、无碰撞轨迹规划）。
- **核心创新**：将 NeRF 的"从视角合成图像"变为"从关节角度合成机器人外观"；输出从 RGB 简化为 `(visibility, density)`（机器人是单色的）。
- **几何先验与网络学习的分工**：前两个关节（偏航/俯仰）用旋转矩阵确定性处理，剩余关节交给网络学习，大幅降低学习难度。
- **与本项目的关联**：直接复用其体渲染管线（OM_rendering、sample_stratified、get_rays）；将"关节角度"扩展为"动作时序窗口"；几何先验分工思想演化为 MS-SCNF 中骨架回归（确定性几何）与密度场（学习部分）的分离。

#### Chen et al. 2022 — Full Body Visual Self-Modeling of Robot Morphologies
全身视觉自建模：从不完美自我感知（遮挡、视角限制）中学习完整 3D 身体模型，在无 CAD 模型时通过自观察发现几何结构。强调"从不完整观测推断完整形态"的挑战，与本项目单视角设置一脉相承。

#### Egocentric Visual Self-Modeling (Lipson 组, Nature npj Robotics 2025)
仅用自我中心（第一人称）视觉观测建模机器人动力学，无需本体感知传感器，在腿式机器人上完成运动任务。验证了"纯视觉自建模"的可行性，但面向刚性腿式机器人。

#### Tang et al. 2026 (ICRA) — 全身形状控制
CNN + 在线优化，在真实硅胶臂上实现全身形状控制。贡献核心是**问题重新定义**（"全身形状控制"是新问题），而非算法复杂度。

#### Yu et al. 2026 (arXiv) — 可解释形状
Bézier 曲线 + Neural ODE，在真实线驱动臂上实现几何可解释的形状建模与 Jacobian 控制。贡献核心是**表示方法创新**（参数化曲线带来可解释性）。

### 2.2 神经场 / Shape 表示 (Neural Field Shape Representations)

#### NeRF (Mildenhall et al. 2020)
隐式神经表示的开山之作：MLP 将 5D 坐标映射为颜色+密度，位置编码 `PE(x) = [x, sin(2^k x), cos(2^k x)]` 使 MLP 能拟合高频细节，体渲染 `C(r) = Σ T_i α_i c_i` 可微端到端训练。本项目整个管线建立在此范式之上。

#### D-NeRF (Park et al. 2021)
canonical + deformation 范式：`观测点 x_t → Deformation(x_t,t) → canonical 坐标 x_c → Canonical(x_c) → (color,density)`，解耦形状与运动。**C-MSTNF 直接采用此范式**（CanonicalField = 零动作静止形态，DeformationField = 动作引起的变形），但用 MultiScaleEMA 替代时间 t。问题：变形 MLP 无约束，对软体机器人大幅变形易产生高频跳变。

#### Nerfies (Park et al. 2021)
变形基函数（deformation basis）：变形 = 一组光滑基函数的加权组合，保证变形天然光滑。启发了"约束结构保证光滑"的思路（→ MS-SCNF 的骨架结构约束）以及 coarse-to-fine 位置编码策略。

#### HyperNeRF (Park et al. 2022)
超维切片处理拓扑变化。本项目软体臂始终是连续杆、无拓扑变化，故 D-NeRF 标准范式已够用；但其"高维 canonical 空间 → 低维切片"概念可作为 MultiScaleEMA 的物理类比。

#### SoftNeRF (Shan et al. 2024) — 最直接的相关工作
首个将条件化 NeRF 用于软体机器人自建模。**Kinematic-Aware SDF Hash Grids**：Instant-NGP 风格的多分辨率哈希编码（16 级，2⁵~2¹¹），自适应分配容量——粗大结构（整体弯曲）用低分辨率，精细结构（局部皱褶）用高分辨率。以 4 根缆绳位移为条件，通过多视角 RGB 体渲染监督。
- **与本项目的关联**：多尺度精度编码启发了多尺度骨架设计（SkeletonHead 在 coarse(4)/medium(10)/fine(31) 三尺度回归）；coarse-to-fine 训练策略借鉴于此。
- **关键区别**：SoftNeRF 仍用隐式密度场，本项目用显式骨架 + 骨架条件密度场。

#### BARF (Wang et al. 2021)
课程式位置编码：训练初期只用低频（学全局结构），后期逐步增加高频（学局部细节），频率 mask `sigmoid(α(f-β))`。直接启发了课程式频率学习方案，并已在本项目变形场中使用（`deform_n_freqs=6`，建议从 2 逐步增到 6）。核心洞察：**频率控制 = 优化难度的控制**。

#### BANMo (Yang et al. 2022)
骨架驱动变形：先预测 3D 骨架（关节位置），再用骨架 + Linear Blend Skinning 驱动表面变形，从随意视频构建可动画 3D 模型。**骨架驱动思想直接启发了 MS-SCNF**：骨架 → 密度场（距离到骨架的距离）。区别：BANMo 需学习蒙皮权重，本项目用距离函数隐式处理。

#### Action-Conditioned Flow Matching (arXiv 2605.09216, 2025) — 直接对标
将连续体机器人形状预测重新定义为**条件点云生成**：学习条件速度场 `u_θ(X_t,t|c)`，从高斯噪声积分到目标点云，用 FiLM 层注入驱动条件。
- **优势**：速度场天然 Lipschitz 连续（点云不会断裂）；相邻驱动→相邻速度场→相邻点云（时间稳定）；直接在 3D 点云空间操作（无深度模糊）；CD 相比最强 baseline 降低 64-96%。
- **局限**：需要 RGB-D 多视角点云融合；准静态方法，不建模瞬态动力学；生成式推理需约 100 步 ODE 积分。

#### Shape-Interpretable Visual Self-Modeling (中山大学 2025, arXiv 2603.01751)
用**分段二次 Bézier 曲线**参数化机器人形状（7 个控制点描述 3D 形状），多视角控制点联合编码唯一确定 3D 形状，Neural ODE 建模形状动力学，Jacobian 伪逆做几何感知控制。形状误差 < 1.56% 图像分辨率，末端误差 < 2% 机器人长度。
- **优势**：Bézier 曲线拓扑天然正确（无断裂）、低维紧凑、控制点几何含义明确（可避障）。
- **局限**：假设形状可用曲线参数化（仅细长结构）；依赖 2D 骨架提取质量；未考虑截面信息。
- **启发**：SkeletonSDF 的参数化骨架 + 管状 SDF 先验（`SDF(x) = dist_to_skeleton(x) - radius`）由此而来。

#### INR-DOM (RSS 2025, KAIST)
Hypernetwork + SIREN 实现条件 SDF：部分点云 → 编码器 → 潜变量 z → Hypernetwork 生成 SDF 网络权重 → 查询任意点 SDF。两阶段训练（预训练重建 + 对比学习微调）。创新点包括**中轴约束**（SDF 在中轴线处 Laplacian 趋向无穷，对柔性臂中心线的物理约束）与**一致性损失**（部分观测 ↔ 完整观测映射到同一潜空间区域，解决遮挡）。重建 CD/EMD 优于 PCN/PointTr/Point2Vec。

#### 其他相关工作
| 论文 | 核心贡献 | 关联 |
|------|----------|------|
| Robot-NO (Adv.Eng.Informatics 2025) | 神经算子几何+载荷→全场变形，比 FEM 快 6000× | Neural Operator 思路加速物理预测 |
| Jacobian Fields (Nature 2025) | 视频流→Jacobian 场→控制多种机器人形态（含软体） | Jacobian 场学习范式 |
| 4DRecons (arXiv 2024) | 4D 隐式场重建可变形物体 | 4D 时空场思路 |
| Disney INR for Soft Bodies (2022) | 隐式表示物理驱动软体 | 材料空间→世界空间隐式映射 |

### 2.3 迟滞与记忆建模 (Hysteresis & Memory)

现有软体机器人自建模工作普遍**忽略迟滞**（准静态采集排除瞬态）。本项目的独特优势正在于此，相关工具：

- **Preisach / Bouc-Wen / Prandtl-Ishlinskii**：经典迟滞唯象模型，参数多、物理可解释性弱。
- **分数阶导数记忆核**：`t^{-α}` 幂律衰减，本项目 `FractionalMemory` 采用——匹配粘弹性材料记忆的物理本质（短期快速衰减 + 长期慢衰减）。
- **GammaLaguerre 记忆核**：Gamma 分布核带延迟峰值，`MultiScaleEMA` / `GammaLaguerreMemory` 编码器实现，可建模加载历史的延迟响应。
- **Neural ODE 时序编码**（已归档的 ODE-CMSTNF）：阻尼谐振子模型 `ds_vel/dt = -k·s_pos - c·s_vel + B·action`，二阶动力学可捕捉阻尼振荡。归档原因：RK4 积分梯度爆炸，效果不如 EMA。思路有价值，后续可用 adjoint method 或更稳定 solver 重试。

**判断标准（来自项目反思）**：迟滞建模若要成为科学贡献，必须回答三点——(1) 迟滞真的存在且可观测吗？(2) 不考虑迟滞的方法在什么条件下失败？(3) 考虑迟滞的方法能做什么不考虑就不能做的？

### 2.4 视觉闭环控制与形状控制 (Visual Closed-Loop Control)

- **FBV-SM 自仿真规划**：学到的模型不仅可视化，还能做梯度优化（关节角→端点到达目标）、A*/RRT 路径规划（3D 占据点云碰撞检测）、无碰撞轨迹规划（距离+碰撞+平滑损失联合优化）。
- **Yu 2026 Jacobian 控制**：`u_dot = J†(x_dot_d + λ(x_d - x))`，Bézier 控制点的 Jacobian 用于几何感知控制。
- **Shape-Interpretable 控制**：低维控制点直接可解释，用于避障与自运动。
- **本项目状态转移闭环**：`StateTransitionSpatialModel` 学 `s_t = F(s_{t-1}, a_t, z_{t-1})`，gt 模式（teacher forcing=1，每步真实状态）为主线，open_loop 模式（TF=0，窗口开环 rollout）测试长程预测。可学习迟滞潜变量 z（GRUCell 跨帧演化，无 GT 端到端学）。

---

## 三、创新点 / 灵感 (Innovations & Inspirations)

汇总本项目已采用或可借鉴的想法。

### 3.1 已采用的架构创新

| 创新 | 来源 | 本项目运用 |
|------|------|-----------|
| **位置编码 + 体渲染 + 隐式表示** | NeRF | 整个管线基础（仿真路线 A） |
| **几何先验与网络学习分工** | FBV-SM | 骨架回归（确定性几何）vs 密度场（学习） |
| **Canonical + Deformation 解耦** | D-NeRF | C-MSTNF 架构基础 |
| **约束结构保证光滑** | Nerfies 变形基函数 | 骨架结构天然光滑（拓扑连通） |
| **多尺度精度编码** | SoftNeRF 哈希网格 | 多尺度骨架 coarse(4)→medium(10)→fine(31) |
| **课程式频率学习** | BARF | 频率 schedule + coarse-to-fine 训练 |
| **骨架驱动表面变形** | BANMo | 骨架条件密度场（密度 = f(到骨架距离)） |
| **参数化曲线 + 管状 SDF 先验** | Shape-Interpretable | SkeletonSDF：`SDF = dist_to_skeleton - radius` + SIREN 残差修截面 |

**MS-SCNF 核心创新小结**：
1. 显式骨架回归替代隐式变形场——物理约束自然保证平滑性与端点精度。
2. 多尺度骨架监督——粗尺度先学整体趋势，细尺度再修局部。
3. 骨架条件密度场——密度取决于查询点到骨架曲线的距离，物理合理且自动稀疏。
4. 仿真器 3D GT 监督——利用 PyElastica position_collection 做定量 3D 评估。
5. 部署直接输出 3D 形状——一次前向推理输出完整 3D 骨架，无需体渲染。

### 3.2 已采用的状态转移与记忆创新

- **FractionalMemory 分数阶记忆核**：分数阶幂律记忆核匹配粘弹性迟滞，这是现有所有软体机器人自建模工作都忽略的维度。模型不再假设马尔可夫性。
- **可学习迟滞潜变量 z**：GRUCell 跨帧演化，无 GT 端到端学，捕获不可观测的内部材料状态。
- **沿臂空间 GRU**：节点间空间递推，建模变形沿臂传播。
- **预测增量 `s_t = s_{t-1} + delta_scale·tanh(Δ)`**：用 tanh 限幅增量，保证轨迹连续、避免突变。
- **gt 模式 vs open_loop 模式**：teacher forcing 切换，分别验证单步精度与长程 rollout 稳定性。

### 3.3 免标定实物管线创新（路线 B）

- **免相机标定**：2D 图像骨架 `[col,row,0]`（z=0）直接作为 state，预测输出反归一化回像素。无相机内参/外参，不做度量 3D 投影。
- **NDI 末端 mm 验证**：NDI 6DOF 追踪器提供末端 ground truth（mm），通过仿射自标定（对 GT `node0` 像素）将像素误差转换为 mm，仅作验证而非训练信号。
- **逐行质心 + 弧长重采样骨架化**：从 white_on_blue mask 提取 2D 骨架，比传统细化算法更稳健。
- **tip_fix 末端尖端修正**：水平切片在 mask 尖角处倾斜导致末端 node0 落点偏移（34% 帧受影响），M6 垂直尖端切片修正使 corner 帧误差降低 71%。
- **共识清洗**：静态段（关节 node11+ 近端）共识稳定 + 手干扰帧 npz 插值，提升实物数据质量。

### 3.4 已归档/未实现的灵感

- **Neural ODE 时序编码 (ODE-CMSTNF)**：阻尼谐振子二阶动力学，可嵌入物理先验、积分保证连续。归档原因：RK4 梯度爆炸。价值仍在，可 adjoint method 重试。
- **光谱正则化变形场 (Smooth-CMSTNF)**：谱归一化 + Jacobian penalty + temporal gradient penalty。归档原因：正则权重难调；当前骨架+SDF 方案从架构层面保证连续性，不需额外正则。
- **Flow Matching 条件点云生成（方案 A）**：放弃隐式场+体渲染，action-conditioned flow matching 直接生成点云。点云天然连续、Flow 平滑保证时间连续性、无深度模糊。未实现，调研中。

---

## 四、待探索方向 (Future Directions)

按优先级排序。核心判断标准：**物理必然性**（不是"也许能改进"，而是"物理上必然如此"）、**现有方法的根本盲区**（概念上就不可能做到）、**简单到令人信服**（实验直观，审稿人一看就懂）。

### 4.1 历史依赖逆运动学（最高优先级）
**科学问题**：软体机器人的逆运动学是"函数"还是"泛函"？
- 传统 IK：`target_shape → find action a*`（函数思维，假设相同动作永远产生相同构型）。
- 粘弹性 IK：`target_shape + current_state → find action SEQUENCE`（泛函思维，相同目标从不同初始状态出发需不同驱动轨迹；加载速度、路径曲率都影响最终形状）。
- **为什么别人没想到**：现有论文都在准静态条件采集（Tang 等待稳定、Yu 等待稳定、SoftNeRF 静态多视角），成功建立在排除粘弹性效应的实验设计上。
- **关键实验**：(1) 循环加载下 IK 失败（A→B→A→B，第二次 B ≠ 第一次）；(2) 速率依赖 IK（0.1s/1s/5s 到达同一目标，最终形状不同）；(3) 残余变形下 IK（从已变形状态出发）。
- **方法极简**：已有的有记忆正向模型 + 梯度优化求 action_sequence + 无记忆 baseline 对比。
- **目标**：RSS/ICRA（短版 6 页聚焦"循环加载下 IK 失败"）/ T-RO（长版 15 页全面分析）。

### 4.2 速率依赖的形状控制（高优先级，最易实现）
**科学问题**：加载速率能否成为形状控制的额外自由度？
- 慢速弯曲→材料松弛→接近平衡态；快速弯曲→来不及松弛→偏离平衡态→更大回弹。相同驱动量、不同速率→不同最终形状。
- 新范式：控制空间从 `{驱动量}` 扩展为 `{驱动量, 驱动速率}`，可达形状空间可能变大（某些形状只能特定速率到达）。
- **实验极简**（PyElastica 几天可出结果）：(1) 速率-形状关系量化（加载时间 vs 与平衡态偏差，预期幂律标度）；(2) 速率增强的形状可达性（B ⊃ A？）；(3) 速率-形状映射可学习性。
- **目标**：ICRA 4 页 + 补充材料 / RA-L。

### 4.3 迟滞的信息容量（高优先级，长线）
**科学问题**：软体机器人的当前形状编码了多少关于加载历史的信息？`I(S; H) = ?`
- 视角转换：迟滞不是需要克服的麻烦，而是**物理记忆装置**——加载历史被"写入"材料微观结构，表现为宏观形状（类比皮肤压痕、植物向光性、肌肉张力）。
- **信息论形式化**：测量短期记忆 `I(S; a_{t-k})` 衰减曲线（预期幂律，分数阶记忆的物理对应）、顺序可辨识性（[+1,-1,+1] vs [-1,+1,-1] 最终形状能否区分）、速率编码、极限容量。
- **为模型选择提供定量依据**：若 `I(S; a_{t-k})` 在 k>3 接近零→记忆窗口 3 步，稳态模型够用；若衰减很慢→必须用状态转移 + 长历史窗口。
- **方法极简**：采样多样加载历史→仿真得形状→训练分类器/回归器→用 MINE/binning 估互信息。
- **风险**：若互信息太小（形状几乎不编码历史），结论是负面但仍有价值。纯仿真可能不够，需真实材料参数验证。
- **目标**：Science Robotics / PRL / Nature Communications（范式转换级别）。

### 4.4 因果结构自发现（中高优先级）
**科学问题**：一个"出厂"的软体机器人，能否仅通过观察自身运动，自动发现其驱动结构的因果图？
- 现有方法隐含假设：已知驱动维度、已知驱动映射、已知响应特性。**没有人问过：如果不知道驱动结构呢？**
- 婴儿 motor babbling 类比：随机驱动 + 视觉观测→Granger 因果/条件互信息发现 DOF 数量、空间影响区域、响应延迟、自由度耦合。
- 可与方向 4.1 结合：先因果自标定→再历史依赖 IK，全程无人工干预。
- **目标**：CoRL / RSS。

### 4.5 热力学一致的自建模（中优先级，理论基础强）
**科学问题**：学习到的自模型是否满足热力学定律？
- 软体变形服从：能量守恒（输入功 = 应变能 + 耗散能）、Clausius-Duhem 不等式（耗散能 ≥ 0）、迟滞环面积 = 耗散能。
- **反问题（能量审计）**：事后审计现有学习模型——循环加载预测→反推等效应力-应变曲线→检查迟滞环面积是否非负、能量是否守恒。若违反→科学发现"端到端学习的自模型在物理上不一致"。
- **正问题（约束学习）**：在 Neural ODE 架构中硬编码能量守恒，状态 `[形状,速度]`，`dvelocity/dt` 参数化保证 `E = KE+PE+dissipated ≥ 0`。
- **实验**：(1) 现有模型热力学审计；(2) 有/无约束模型的分布内精度与分布外泛化差异；(3) 可解释的沿臂耗散率热图。
- **目标**：T-RO / PRL / ICRA。

### 4.6 内部应力场的视觉推断（中优先级，高创新性，长线）
**科学问题**：能否从外部视觉观测推断软体机器人内部应力分布？
- 所有现有方法只关注"形状预测"（表面现象），应力才是根本原因。应用：材料失效预测（应力集中点）、接触力估计（无需力传感器）、疲劳寿命估计。
- **关键转折**：单一形状下应力欠约束，但**形状的历史变化**提供动力学方程的多次观测→可反推材料参数与应力状态。又回到时序建模的独特优势。
- PyElastica 提供完美应力 GT（每节点弯矩/剪力/轴向力已知）。
- **风险**：真实机器人应力验证困难（需嵌入 FBG 等传感器）。
- **目标**：Science Robotics / T-RO。

### 4.7 方向优先级与组合策略

| # | 方向 | 核心问题 | 简单性 | 新颖性 | 影响力 | 可行性 | 优先级 |
|---|------|---------|--------|--------|--------|--------|--------|
| 4.1 | 历史依赖 IK | IK 是函数还是泛函？ | ★★★ | ★★★ | ★★★ | ★★★ | **最高** |
| 4.2 | 速率依赖控制 | 加载速度是新自由度？ | ★★★ | ★★☆ | ★★☆ | ★★★ | **高** |
| 4.3 | 迟滞信息容量 | 形状编码多少历史？ | ★★☆ | ★★★ | ★★★ | ★★☆ | 高（长线） |
| 4.4 | 因果结构自发现 | 机器人能自标定吗？ | ★★☆ | ★★★ | ★★☆ | ★★☆ | 中高 |
| 4.5 | 热力学一致性 | 模型服从物理定律吗？ | ★★☆ | ★★★ | ★★☆ | ★☆☆ | 中 |
| 4.6 | 内部应力推断 | 外观→内部状态？ | ★☆☆ | ★★★ | ★★★ | ★☆☆ | 中（长线） |

**最优论文策略**：方向 4.1 + 4.2 + 4.3 合并为一篇 T-RO 长文——"History-Aware Self-Modeling of Viscoelastic Soft Robots: From Rate-Dependent Prediction to Hysteresis-Aware Inverse Kinematics"。
贡献链：发现加载速率显著影响最终形状（4.2）→ 证明无记忆模型在循环/变速加载下根本性失败 → 提出历史依赖正向模型与 IK 框架（4.1）→ 信息论分析量化形状对历史的编码能力（4.3）。完整故事：发现问题→分析原因→提出解法→理论解释。

**执行顺序**：
- 短期（1-2 月）：4.2 速率依赖（PyElastica 几天出结果）→ 4.1 历史 IK（在 4.2 数据上设计 IK 实验，核心展示现有方法循环加载失败）。
- 中期（2-4 月）：4.3 信息容量（在 4.1 数据上做信息论分析，不需新模型）→ 4.4 因果发现（独立实验，需重设数据采集）。
- 长期（4+ 月）：4.5 热力学（需理论基础）→ 4.6 应力推断（需仿真扩展）。

---

## 五、定位反思 (Positioning Reflection)

### 应停止做的事
1. **停止优化 3D 骨架预测精度的边际改进**：从 CD 0.0014 到 0.0012 不会改变任何事情，没人会因此引用。
2. **停止尝试新编码器架构**：EMA / FractionalMemory / GammaLaguerre 的比较已足够，若核心问题是"迟滞可建模/可利用"，编码器选择是工程细节。
3. **停止追求完整 3D 表面重建**：Tang 和 Yu 的成功表明 2D + 简单方法 > 3D + 复杂方法。3D 重建是难的技术问题，不是好的科学问题。

### 应开始做的事
1. **设计让现有方法失败的实验**：不是证明我们方法好，而是证明现有方法在物理上是错的——比"CD 更低"有说服力得多。
2. **用最简单方法验证概念**：若"速率影响形状"为真，最简单的回归就能验证，不需要 Neural ODE/Flow Matching。
3. **从"预测"转向"控制"**：形状预测的终点是形状控制。展示"考虑迟滞的控制比不考虑的好 X%"就是完整故事。

### 底线
若贡献是"考虑了迟滞的软体机器人自建模"，必须回答三个问题：
1. 迟滞真的存在且可观测吗？（物理验证）
2. 不考虑迟滞的方法在什么条件下失败？（对比实验）
3. 考虑迟滞的方法能做什么不考虑就不能做的？（能力边界）

三个问题都肯定，不管方法多简单，就是一篇顶刊。

---

## 六、方法对照表 (Method Comparison)

| 特性 | NeRF | FBV-SM | D-NeRF | SoftNeRF | BARF | BANMo | Flow Matching | Shape-Interp. | **本项目** |
|------|------|--------|--------|----------|------|-------|---------------|---------------|-----------|
| 建模对象 | 静态场景 | 刚性臂 | 动态场景 | 软体机器人 | 静态场景 | 可动画动物 | 连续体机器人 | 连续体机器人 | **软体连续体** |
| 输入条件 | 视角 | 关节角度 | 时间 t | 驱动参数 | 视角(+位姿) | 骨架姿态 | 驱动条件 | 驱动+历史 | **动作时序窗口** |
| 3D 表示 | 隐式密度场 | 隐式密度场 | Canon+Deform | SDF+哈希 | 隐式密度场 | 骨架+网格 | 点云(生成式) | Bézier曲线 | **骨架+条件密度场** |
| 时序编码 | 无 | 无 | 时间 t | 单帧状态 | 无 | 骨架关节角 | FiLM(准静态) | Neural ODE | **FractionalMemory** |
| 变形约束 | N/A | 无 | 无约束MLP | 哈希+Eikonal | 课程式频率 | 骨架+LBS | Lipschitz连续 | Bézier拓扑 | **骨架物理约束** |
| 3D 监督 | 无 | 无 | 无 | 无(多视角2D) | 无 | 2D关键点 | RGB-D点云 | 2D骨架多视角 | **GT节点坐标/像素** |
| 部署输出 | 渲染图像 | 渲染图像 | 渲染图像 | 渲染图像 | 渲染图像 | 3D动画 | 3D点云 | 3D曲线+控制 | **直接3D骨架/像素** |
| 迟滞建模 | 无 | 无 | 无 | 无 | 无 | 无 | 无 | NODE(动态) | **分数阶记忆+z** |

---

## 参考文献 (References)

1. Mildenhall et al. 2020 — *NeRF: Representing Scenes as Neural Radiance Fields*. ECCV.
2. Hu et al. 2025 — *FBV-SM: Teaching Robots to Build Simulations of Themselves*. (Field-Based Vision Soft Manipulation)
3. Chen et al. 2022 — *Full Body Visual Self-Modeling of Robot Morphologies*.
4. Park et al. 2021 — *D-NeRF: Neural Radiance Fields for Dynamic Scenes*.
5. Park et al. 2021 — *Nerfies: Deformable Neural Radiance Fields*.
6. Park et al. 2022 — *HyperNeRF: A Hyperdimensional View of Deformable NeRF*.
7. Shan et al. 2024 — *SoftNeRF: A Self-Modeling Soft Robot Plugin*.
8. Wang et al. 2021 — *BARF: Bundle-Adjusting Neural Radiance Fields*.
9. Yang et al. 2022 — *BANMo: Building Animatable 3D Neural Models from Casual Videos*.
10. *Continuum Robot Modeling with Action Conditioned Flow Matching* — arXiv 2605.09216, 2025.
11. *Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control* — arXiv 2603.01751, 2025 (中山大学).
12. *Implicit Neural-Representation Learning for Elastic Deformable-Object Manipulations* — RSS 2025 (KAIST, INR-DOM).
13. *Egocentric Visual Self-Modeling for Autonomous Robot Dynamics Prediction and Adaptation* — Nature npj Robotics, 2025 (Lipson 组).
14. *A Generalizable Neural Operator for Full-Field Deformation Prediction* — Adv. Eng. Informatics, 2025 (Robot-NO).
15. *Controlling Diverse Robots by Inferring Jacobian Fields* — Nature, 2025.
16. *4DRecons: 4D Neural Implicit Deformable Objects Reconstruction* — arXiv 2406.10167, 2024.
17. *Implicit Neural Representation for Physics-driven Actuated Soft Bodies* — Disney Research, 2022.
18. Tang et al. 2026 — 全身形状控制 (ICRA).
19. Yu et al. 2026 — 可解释形状自建模 (arXiv).

> 论文深度阅读笔记：`docs/papers/`。本文为综述层，仅记录核心思想与关联。
