# ICRA 论文中文草稿(脚手架版)

> **状态**:草稿 scaffold——结构完整、方法写实、结果留 `[待填]` 占位。**最终英文正文须作者按 IEEE-RAS AI 政策自己重写**,本稿用于梳理逻辑、明确要做的实验。
> **脊柱**:P3(物理接地分数阶记忆编码器)+ P4(可信开环历史感知规划);P1/P2 作动机证据;免标定 + NDI + 速率定量评估作实证。
> **关联**:实验设计与资产见 `docs/paper/04_experiments.md`(Exp A–H);文献核实见 `01_landscape.md`;自标定设计见 `06_multiview_self_calibration.md`。
> **占位约定**:`[待填]` = 待跑实验;`[已有·需重跑]` = 旧实验有初步值但 P2 需重采重训后更新;`[锚点]` = 已有的确定事实。

---

# Learning to Model a Body That Remembers: Fractional-Order Memory for Whole-Body Shape Self-Modeling of Viscoelastic Soft Robots

*(中文标题直译:学会给一个有记忆的身体建模——面向粘弹性软体机器人全身形态自建模的分数阶记忆)*

**Index Terms** — soft robot, self-modeling, shape estimation, hysteresis, fractional-order memory, open-loop planning

---

## 摘要(Abstract)【英文成稿约 200 词;中文骨架如下】

> 软体机器人的全身形态估计是安全控制与狭窄环境操作的前提。软体材料的粘弹性使当前形态不仅取决于当前驱动指令,还取决于完整的加载历史(方向、速率、顺序),即"指令→形态"在物理上是**历史的泛函**而非函数。现有视觉自建模方法普遍在准静态下采集数据,从而在实验设计上排除了这一非马尔可夫性。
>
> 本文提出一种**免标定、物理接地**的全身形态自建模管线。感知端直接用单目相机像素骨架 `[col,row,0]` 作状态,不需要任何相机标定。模型端用一个带**可学习迟滞潜变量**的闭环状态转移网络预测全身形态,其时序编码器采用 **Grünwald–Letnikov 分数阶幂律记忆核**——与粘弹性材料的幂律弛豫谱同构,区别于通用的 EMA/GRU/LSTM/TCN 序列模型。
>
> 在实物两段硅胶软臂上(NDI 6DOF 毫米真值):[待填——全身骨架误差、末端 mm、与无记忆/通用序列模型的差距]。
>
> 关键地,这一忠实自模型解锁了两类"无记忆假设下不可能"的任务:(1)**历史感知逆规划**——同一目标形态从不同加载状态出发需不同动作序列,歧义集量化见 Exp D;(2)**可信开环动作序列规划**——观测一次、预测 K 步,并给出模型可信任的视野上限。这使软体臂能在稀疏观测下进行全身形态控制,而无需每个控制周期闭环反馈。

---

## I. 引言(Introduction)

### I.1 背景

软体机器人因其本体柔顺、对人与环境安全,正越来越多地进入狭窄空间操作与接触场景。但软体臂的**全身形态**(而非仅末端位置)在碰撞避让、缠绕式接触操作中是安全性的前提——中段同样不能碰障碍 [A2 系引用]。因此,机器人需要可靠地**估计与预测自身当前及未来的完整形态**。

### I.2 现状与矛盾

获取形态的主流路线有三条。基于嵌入式传感的方法(FBG、电磁、缆绳编码)只能得到沿臂有限个离散点,完整连续形态必须依赖物理积分重建 [A4]。模型驱动的运动学/动力学(PCC、Cosserat)强依赖精确的先验物理参数与 CAD 模型,损伤或材料漂移后失效 [A6]。近年来,数据驱动的**视觉自建模**通过让机器人观察自己,端到端学习"驱动→形态",避免了先验依赖与两阶段误差累积 [A7];神经场、3D Gaussian Splatting、隐式 SDF 等表示已被广泛用于刚性臂与软臂的自建模。

然而,一条贯穿上述所有路线的共同假设是:**"当前指令 → 当前形态"是一个函数(马尔可夫)**。这要求机器人在采集与部署时都处于**准静态**状态——等形变稳定再观测。软体材料的粘弹性(内摩擦、蠕变、应力松弛)使当前形态实际取决于完整的加载历史:同一气压沿加压与泄压路径到达,形态不同;同一路径以不同速率加载,形态也不同。我们在实物数据上测得:同一气压 load/unload 的末端位移差为 `[锚点] 1.53 mm(准静态)与 2.06 mm(动态)`,且高于 NDI 噪声底 0.74 mm。一旦进入动态、循环或变速加载,马尔可夫假设在物理上不成立,现有自建模会系统性失败 [A8]。

### I.3 空缺(Gap)

已有一些工作开始把迟滞纳入软体建模,但它们都在**闭环控制**里用通用循环网络(GRU/LSTM/TCN)拟合迟滞,并且逐控制周期喂入实测值 [Schäfke 2024; Chen 2025]。仍无人回答三个问题:

1. **表示**:物理接地的记忆核(与粘弹性幂律弛豫同构)能否作为神经形态模型的**时序编码器**,比通用序列模型在**速率外推**上更忠实?(文献核实:分数阶记忆在软体领域只出现在控制器/物理权重/优化器里,从未作为形状模型的编码层——见 `01_landscape` P3)
2. **良置性**:迟滞使逆映射"目标形态→动作"不适定(同一目标可来自多条历史分支)。这个**歧义集**有多大、需要多少历史才能消除?(无人量化过逆映射歧义集)
3. **部署**:在免标定单目视觉下,能否用一个忠实的自模型做**开环**逆动作序列规划 + 视野认证,而无需每个控制周期闭环?

### I.4 贡献

1. **(方法)** 首次把 Grünwald–Letnikov 分数阶幂律记忆核作为**神经全身形态模型的时序编码器**,物理接地于粘弹性弛豫谱;在实物软臂上给出六路时序编码器(EMA/Gamma/GRU/Transformer/TCN/分数阶)的系统消融与速率泛化实证 [Exp A–C]。
2. **(良置性)** 形式化"软臂 IK 是历史的泛函而非函数",并首次**量化迟滞下的 IK 歧义集**(逆映射前像集直径)与临界记忆长度 T* [Exp D]。
3. **(能力)** 展示忠实自模型解锁的**历史感知逆动作序列规划**与**可信开环规划**(观测一次、预测 K 步、给出信任视野),对比无记忆模型的规划系统性失败 [Exp E–F]。
4. **(实证)** 免标定单目像素骨架作状态 + NDI 6DOF 毫米真值 + 全身形态在显式速率/循环加载下的定量评估 [Exp G];多驱动 3D 升级采用无标定板的身体自我标定几何 [Exp H, 可选]。

### I.5 路线图

第 II 节综述相关工作并给出定位;第 III 节形式化问题;第 IV 节描述方法;第 V 节给出实验设置与结果;第 VI 节讨论局限与意义;第 VII 节总结。

---

## II. 相关工作(Related Work)

### II.1 嵌入式传感与模型驱动形状感知
FBG/电磁/缆绳编码等只能提供沿臂离散点,完整形态依赖物理积分重建 [Khan 2019; Shi 2017; An 2024];PCC 与 Cosserat 模型强依赖先验物理参数,动态或交互下失效 [Wang 2022; Till 2019]。

### II.2 数据驱动视觉自建模
自 Bongard–Lipson 的形态发现与 Chen 2022 的全身视觉自建模以来,神经场(NeRF/3DGS/隐式 SDF)与可微渲染已成为机器人自建模的主流范式 [Hu 2025; Shan 2024 SoftNeRF; Yang 2024 RobotSDF; Li 2025 NJF]。这些工作普遍在准静态下采集,且输出度量 3D 占用,通常需要相机内参。

### II.3 骨架/背骨的低维参数化
用 Bézier、Euler 弧样条、PH 曲线、POD/PCA 应变模式把离散骨架点压缩为连续曲线,已被证明能显著降低形状预测难度 [Yu 2026; Rao 2022; Mbakop 2024; Valadas 2024; MoSS]。**我们不做曲线参数化**(已多组占据),只做记忆表示。

### II.4 迟滞与记忆建模
软体迟滞的经典建模为 Preisach/Prandtl–Ishlinskii/Bouc–Wen 等唯象算子 [Delamorena 2025];学习侧多用 RNN/GRU/LSTM/TCN 拟合迟滞并用于闭环补偿或 MPC [Chen 2025; Schäfke 2024; Park 2024; Liu 2024]。分数阶微积分在软体领域用于**控制律**(分数阶滑模 [Shao 2025])、**物理本构**(分数阶粘超弹性 [Gao 2022])、**权重编码**(FBGNN 2026)与**优化器**(FO-Elman 2026)。**无人**把分数阶离散核用作动作历史→形状模型的时序编码层。

### II.5 规划/控制上的学习自模型
开环规划用学习动态模型已有多年 [Thuruthel 2017 单发 shooting; Bern 2020 可微模型; Du 2021 可微仿真; Krauss 2026 潜空间开环; Flow-Matching 逆动力学 2026]。它们或不做显式迟滞、或不做视野认证。**"显式迟滞 + 开环多步序列 + 认证信任视野"的组合无人占**。

### II.6 免标定与自我标定
单目 2D 像素系自建模未见(所有自模型需内参)。3D 升级可借用经典自标定(SfM/autocalibration [Faugeras; Maybank & Luong])与学习式免标定几何(DUSt3R/MUSt3R 类);**新意在于把自标定作为软体自建模管线的组成部分并与自模型耦合**,而非自标定算法本身。

### II.7 定位(差异化表)

| 已发表 | 他们 | 我们 |
|---|---|---|
| Zhang 2026(Koopman 33 构型)/ Tang 2026(突触) | 控制侧跨构型泛化/在线适应,无视觉全身形态 | 免标定视觉全身形态自建模 |
| Yu 2026(Bézier+NODE) | 曲线参数化降低预测难度 | 不争曲线;争**物理接地记忆表示** + 速率/循环定量评估 |
| Chen 2025 / Schäfke 2024(迟滞+RL/NMPC) | 闭环控制用通用循环网络 | 开环 + 分数阶核(物理接地) + 信任视野 |
| Wang 2024 / Cho 2024(无记忆失效/路径依赖) | 已量化前向失败与路径依赖 | 作动机证据;补**逆映射歧义集量化** |
| Thuruthel 2017 / Krauss 2026(开环 shooting) | 开环但无显式迟滞/无认证视野 | 显式迟滞前向模型 + **认证信任视野** |
| Shao 2025 / FBGNN 2026(分数阶+软体) | 分数阶在控制律/物理权重 | 分数阶在**编码器**(动作历史→形状的时序层) |

---

## III. 问题形式化(Problem Formulation)

### III.1 记号与免标定状态

软臂为两段三腔气动驱动(6 通道气压,归一化后 `a_t ∈ [0,1]^6`)。形态状态为全身骨架:

- **2D 基线**:`s_t ∈ ℝ^{N×3}`,由单目相机像素骨架构成,`[:, :2] = [col,row]`,第三维恒 0。**免标定**:无相机矩阵、无内参,状态直接是像素。
- **3D 升级**:多视角身体自我标定后 `s_t ∈ ℝ^{N×3}` 为毫米坐标(见 `06_multiview_self_calibration.md`)。

### III.2 非马尔可夫性(核心形式化)

物理上,当前形态是加载历史的泛函:
```
S_t = F[ a_0, a_1, ..., a_t ]        (1)
```
若 F 退化为一元函数(马尔可夫),则相同指令必有相同形态;粘弹性下不成立。我们把它表达为**带隐状态的闭环状态转移**:
```
s_t = s_{t-1} + δ_t ,   δ_t = κ · tanh( Δ( s_{t-1}, a_t, z_{t-1} ) )        (2)
z_t = Φ( z_{t-1}, a_t, s_{t-1} )                                              (3)
```
其中 z 为可学习迟滞潜变量(隐机械状态,无真值,端到端学习);Δ 由时序编码器对动作历史的记忆、当前状态与 z 共同决定;κ 为收缩系数(开环部署时 κ 有界)。

### III.3 逆规划与歧义集

给定目标形态 `S*`,逆映射 `g(S*) = { a_{t:t+K} | rollout(S*, a_{t:t+K}, z) 可达 S* }` 是**集合**(多解)。定义:
- **歧义集直径** `diam g(S*)`:`[待填——各初始状态求解所得动作序列的分散度]`;
- **临界记忆长度** `T*`:歧义随已知最近 k 步历史而缩小,k ≥ T* 时歧义消失的最小值。

第 V 节 Exp D 给出这两个量的实物测量。

---

## IV. 方法(Method)

### IV.1 系统总览

```
单/多相机 → 分割 → 2D 骨架 → [3D 自标定] → 状态 s_t
                                          ↓
       状态转移自模型 s_t = F(s_{t-1}, a_t, z_{t-1})   ← GL 分数阶记忆编码器
                                          ↓
    视野认证(K_max / 信任视野) → 历史感知逆序列规划(shooting) → 真机执行 → NDI 验证
```

### IV.2 免标定感知管线

逐帧:分割(背光/背景减/HSV/white_on_blue)→ 形态学清理 + 最大连通区 → 逐行质心骨架 + `tip_fix` 末端垂直切片修正 → 弧长均匀重采样到 N 点 → 质量门控(面积/行范围/帧间位移等判据,不合格跳过)。输出 `[col,row,0]` 像素骨架作状态。(实现:`real_validation/perception/`。训练侧 mask 用 SAM2 视频分割。)

### IV.3 状态转移自模型

沿用 `StateTransitionSpatialModel` 架构:

- **时序编码器(核心)**:`FractionalMemory` 用 Grünwald–Letnikov 离散化生成幂律权重
  ```
  w_0 = 1,  w_k = w_{k-1} · (k-1-α)/k          (4)
  cond = MLP( [Σ_i w^{(α_i)}·a_window, a_t, velocity] )      (5)
  ```
  α_i 可学习(`α_i ∈ (0,1)`,多阶次即弛豫谱)。物理上:粘弹性松弛模量 G(t) ~ t^{-α},GL 权重序列与之一致;而 EMA 是指数衰减 `e^{-t/τ}`。**这正是"物理接地"的含义**:记忆核的函数形式与材料实际弛豫谱同构。
- **迟滞潜变量 z**:`z_t = GRUCell([cond, s_{t-1}], z_{t-1})`(式 3)。
- **沿臂空间传播**:悬臂梁因果(根部→尖端),向量化 `nn.GRU` 一次核调用。
- **增量收缩**:式 2,`κ`(delta_scale)可学习,开环部署 `κ ≤ κ_max` 防 rollout 发散。

### IV.4 物理接地验证设计

学习得到的 α 应与 NDI 阶跃弛豫拟合的幂律指数一致(Exp C)。这使"物理接地"可核验而非叙事。

### IV.5 可信开环规划

- **视野认证**:对训练好的 open_loop 模型,在验证集上测"预测 K 步后的误差随 K 增长",得到给定容差(px)下的 `K_max` 表——即**信任视野**。
- **历史感知逆序列规划**:给定 `目标形态 S* + 当前状态(s_t, z_t, 历史窗口)`,用 shooting + BPTT 在可微 rollout 上优化 K 步动作序列(带压力/速率约束 + 障碍惩罚),执行时开环。
- **对比**:用**无记忆**(window=1)前向模型做同样的规划,其规划序列应在方向反转/变速段系统性失效(Exp E)。

---

## V. 实验(Experiments)

> 完整协议、命令、资产与 go/no-go 见 `docs/paper/04_experiments.md`。本节给每项实验的**目标 / 指标 / 表图骨架 / 结果占位**。

### V.0 设置

- **硬件**:两段硅胶软臂(3 腔道/段,6 通道气动),单 Intel RealSense + NDI 6DOF 追踪器(毫米真值,仅作验证)。多驱动 3D 升级:多固定相机 + 身体自标定。
- **数据**:`[锚点]` 现有 `seq_20260627_163921`(10214 帧,SAM2 mask,15 节点,1-DOF ch0);P2 重采多速率/循环/方向反转序列(见 `deployment.md §11`)。
- **指标**:全身骨架误差(px,chamfer/MSE)、末端误差(px 与 NDI mm)、`drift_by_k`、规划可达误差、最小障碍净距。
- **实现**:PyTorch 2.6;模型/规划实现见 `src/models/`、`real_validation/`。

### V.1 Exp A — 时序编码器六路消融

- **目标**:同一架构只换编码器(EMA/Gamma/GRU/Transformer/TCN/分数阶),判断 GL 是否在长视野/速率泛化上更忠实。
- **协议**:`train_transition.py --encoder_type <enc>`,open_loop 模式,同数据同超参。
- **表 I(骨架)**:各编码器的验证集骨架 MSE、tip MSE、`drift_by_k`(k=10/20/40/80)。
- **图 1**:`drift_by_k` 曲线(各编码器,误差 vs 视野)。
- **结果**:`[待填]`。预期:分数阶在中/长视野与速率泛化胜出。

### V.2 Exp B — 速率泛化

- **目标**:训一速率、测另一速率,检验物理接地的外推能力。
- **协议**:准静态(0.5s settle)vs 动态(0.2s settle)两档互训互测;P2 加 3 档。
- **图 2**:误差 vs 加载速率(GL vs GRU vs EMA)。
- **结果**:`[待填]`。预期:GL 幂律核在跨速率外推优于指数衰减 EMA 与通用 GRU。

### V.3 Exp C — 物理接地(α 匹配)

- **目标**:模型学到的分数阶 α 对应实测弛豫幂律指数。
- **协议**:NDI 阶跃弛豫拟合 `Δx(t) ~ t^{-α_meas}`;对比 `model.temporal.alphas`。
- **表 II**:α_learned vs α_measured(各阶次/主导阶);与 T* 预测互证。
- **结果**:`[待填]`。

### V.4 Exp D — IK 歧义集与临界记忆长度

- **目标**(核心科学空缺):定量回答"IK 是函数还是泛函"。
- **协议**:对录制目标形态,从不同初始历史/状态求解动作序列;测序列分散度(=歧义集直径);逐步增加已知历史 k,测歧义消失处 = T*。
- **图 3**:歧义集直径 vs 已知历史步数 k(记忆模型);叠加无记忆模型(window=1)的对照。
- **结果**:`[待填]`。

### V.5 Exp E — 记忆 vs 无记忆规划质量

- **目标**:window=1 vs 40 前向模型的逆规划,在方向反转帧的规划可达性。
- **表 III**:两模型规划序列的 rollout 终端误差(方向反转段/稳态段)。
- **结果**:`[待填]`。预期:window=1 规划在反转段失效(误差 ≈ 迟滞环宽度 `[锚点] 1.5–4.25mm`)。

### V.6 Exp F — 信任视野

- **目标**:把 `K_max` 表达为"给定容差下的信任视野",并对比各编码器支持的开环长度。
- **图 4**:信任视野 vs 容差(px 2/5/10/20),各编码器。
- **已有初步**:`[已有·需重跑]` open_loop 漂移 1.7×@300 步,K_max@10px ≈ 124 步 ≈ 25s(旧数据,需 P2 更新)。
- **结果**:`[待填]`。

### V.7 Exp G — 实机 NDI 验证(端到端)

- **目标**:相机 → 骨架 → 记忆自模型 → 开环规划 → 真机执行 → NDI,报 prediction-to-execution gap。
- **表 IV**:执行末态 vs NDI(px/mm)、vs GT-actions 上界、vs `drift_by_k` 预测;全身骨架末帧误差。
- **已有初步(需重跑)**:`[已有·需重跑]` GT 模型末端 mean 0.77mm(已到噪声底);open_loop 旧 run 末期 NaN,mm 不可信。
- **结果**:`[待填]`。

### V.8 Exp H — 3D 自标定验证(可选,多驱动升级)

- **目标**:L2 身体自标定 vs 传统标定(L1)vs L3 学习式(DUSt3R/MUSt3R)的 3D mm 精度对照;恢复 3D 全身避障 demo。
- **表 V**:自标定 mm 误差(重投影 + NDI 交叉)、L2 vs L3 一致性。
- **结果**:`[待填]`(需多相机硬件)。

---

## VI. 讨论(Discussion)

### VI.1 主要发现
`[待填——回填 Exp A–G 的结果与 Intro 问题的回答]`

### VI.2 与文献对比
- 分数阶记忆:与 Shao 2025(分数阶控制律)、FBGNN 2026(分数阶物理权重)的差异被实验结果支撑 [Exp A–C]。
- 开环规划:Thuruthel 2017 与 Krauss 2026 未做显式迟滞/视野认证;我们补上 [Exp E–F]。
- 歧义集:Cho 2024 量化了前向路径依赖;我们补了逆映射歧义集 [Exp D]。

### VI.3 局限(诚实)
1. **单构型**:1-DOF 两段臂、单数据序列。不声称跨构型泛化(那是 Koopman 的)。
2. **迟滞潜变量 z**:当前 z 范数小、per-window 重置,记忆机制主要由动作历史窗口 + 分数阶核承担。z 的"潜机械状态"解释需序列级训练或弱化声称。
3. **真机闭环未完成**:规划动作尚未在真机执行过;Exp G 的 gap 是核心开放问题。
4. **"首次"声称需谨慎**:速率定量评估的"首次"需投稿前再核近期归档。

### VI.4 意义
物理接地记忆表示给出了一个可迁移的洞见:**记忆核的函数形式应当匹配材料的弛豫谱**——这不仅对形状预测,对任何粘弹性系统的学习控制都有指导意义。免标定 + NDI 验证 + 信任视野的组合,使软体臂能在稀疏观测下可靠地做全身形态控制。

---

## VII. 结论(Conclusion)

`[待填——总结贡献,展望:3D 自标定升级、跨构型泛化、模型在环自标定]`

---

## 参考文献(References)(IEEE 数字引用,已核实优先)

> 完整核实状态见 `docs/paper/01_landscape.md` §四。以下为论文核心引用集(均已在调研中核实到摘要/页面级)。

[1] Zhang et al., "Reinforcement learning in linear embedding space unlocks generalizable control across soft robot configurations," *Nature Communications*, 2026 (arXiv:2606.08104).
[2] Z. Tang et al., "A general soft robotic controller inspired by neuronal structural and plastic synapses that adapts to diverse arms, tasks, and perturbations," *Science Advances*, 2026.
[3] P. Yu, X. Wang, N. Tan, "Shape-interpretable visual self-modeling enables geometry-aware continuum robot control," arXiv:2603.01751, 2026.
[4] Z. Chen et al., "Hysteresis-aware neural network modeling and whole-body reinforcement learning control of soft robots," *IEEE RA-L*, 2025 (arXiv:2504.13582).
[5] B. Y. Cho et al., "Accounting for hysteresis in the forward kinematics of nonlinearly-routed tendon-driven continuum robots via a learned deep decoder network," *IEEE RA-L*, 2024 (arXiv:2404.03816).
[6] Y. Wang et al., "Using neural networks to model hysteretic kinematics in tendon-actuated continuum robots," arXiv:2404.07168, 2024.
[7] X. Shao et al., "Self-attention enhanced dynamics learning and adaptive fractional-order control for continuum soft robots with system uncertainties," *IEEE T-ASE*, 2025.
[8] H. Schäfke et al., "Learning-based nonlinear model predictive control of articulated soft robots using recurrent neural networks," *IEEE RA-L*, 2024 (arXiv:2411.05616).
[9] T. G. Thuruthel et al., "Learning dynamic models for open loop predictive control of soft robotic manipulators," *Bioinspiration & Biomimetics*, 2017.
[10] H. Krauss et al., "Accurate open-loop control of a soft continuum robot through visually learned latent representations," arXiv:2603.19655, 2026.
[11] J. Bern et al., "Soft robot control with a learned differentiable model," *IEEE RoboSoft*, 2020.
[12] B. Chen et al., "Fully body visual self-modeling of robot morphologies," *Science Robotics*, 2022.
[13] S. L. Li et al., "Neural Jacobian fields: learning intrinsic mappings of arbitrary robot morphologies," *Nature*, 2025.
[14] J. Shan et al., "SoftNeRF: a self-modeling soft robot plugin for various tasks," *IEEE/RSJ IROS*, 2024.
[15] M. Kasaei et al., "A synergistic framework for learning shape estimation and shape-aware whole-body control policy for continuum robots," arXiv:2501.03859, 2025.
[16] G. Gao et al., "Fractional-order visco-hyperelastic constitutive modeling," 2022. *(待核具体出处)*
[17] Z. Zou, G.-Y. Gu, "Feedforward control of the rate-dependent viscoelastic hysteresis nonlinearity in dielectric elastomer actuators," *IEEE RA-L*, 2019.
[18] F. Liu, M. C. Yip et al., "Differentiable rendering for shape reconstruction of soft continuum robots," 2023. *(待核)*

---

## 附录:投稿前 checklist(对照 skill)

- [ ] 结果全为真实/标注 [待填],无虚构数字
- [ ] 所有引用已在 `01_landscape.md` 核实;`*(待核)*` 项补全文确认或删除
- [ ] Results 只陈述,解释移入 Discussion
- [ ] 按 ICRA 8 页含参考文献压缩;仿真/实机分节
- [ ] 最终英文正文由作者自己撰写(IEEE-RAS AI 政策)
- [ ] 视频补充(≤180s)含失败案例
