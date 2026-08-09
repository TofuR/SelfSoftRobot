# 05 · ICRA 论文大纲 + 差异化表 + 参考文献

> **定位**:ICRA(4–6 页应用论文)或 RA-L。核心卖点 = P3(物理接地分数阶记忆编码器)+ P4(可信开环历史感知规划);P1/P2 作动机/证据;P5 作 demo。
> **投稿前必做**:近期文献再核(速率定量评估"首次"声称)、真机闭环、`docs/paper/01_landscape.md` §四的待核引用逐条确认。

---

## 一、标题候选

1. **Learning to Model a Body That Remembers: Fractional-Order Memory for Whole-Body Shape Self-Modeling of Viscoelastic Soft Robots**(强调"物理接地的记忆")
2. **Non-Markovian Visual Self-Modeling of a Soft Continuum Arm via Fractional-Order Memory**(强调"非马尔可夫")
3. **The Shape of a Soft Arm Is a Functional of Its Past: Memory-Grounded Self-Modeling and Trusted Open-Loop Planning**(最强科学叙事,较长)

**推荐 1**(ICRA 风格:应用 + 方法清晰)。

---

## 二、贡献点(按可辩护性排序)

1. **(方法·P3)** 首次把 **Grünwald-Letnikov 分数阶幂律记忆核**作为**神经全身形态模型的时序编码器**(而非控制器/物理权重/优化器),物理接地于粘弹性弛豫谱;六路编码器系统消融 + 速率泛化实证。
2. **(能力·P4)** 该记忆模型使**历史感知逆规划**良置:开环逆动作序列规划 + 认证信任视野(观测一次、预测 K 步、给出可信上限),并对比"无记忆模型规划的序列在真机不可达"。
3. **(科学·P1-sharpened)** 定量化**迟滞下 IK 的歧义集**(逆映射前像集直径 / 临界记忆长度 T*),给出"IK 是函数还是泛函"的可测答案。
4. **(实证·首次级)** 免标定单相机 2D 像素骨架作 state + NDI 6DOF mm 真值 + 全身形态在**显式速率/循环加载**下的定量评估。

---

## 三、论文结构(6 页)

| 节 | 内容 | 对应资产 |
|---|---|---|
| **I. Intro** | 软臂全身形态预测是部署前提;现工作准静态回避迟滞;我们重构为"非马尔可夫系统辨识"。三句贡献。 | — |
| **II. Related** | 四脉络(嵌入式传感 / 模型驱动 / 视觉自建模 / 迟滞与规划),每条引已核实文献,末段给空白交集。 | `docs/paper/01_landscape.md`;旧稿 `docs/papers/related_work_draft.md`(需按本文 §2 重写差异化段,删"免标定单相机"作卖点之外、补 P3/P4) |
| **III. Method** | 3.1 免标定状态转移自模型 `s_t = F(s_{t-1}, a_t, history)`(含分数阶 GL 编码器 + 空间 GRU + 增量收缩);3.2 记忆核的物理接地(G L vs 实测弛豫谱);3.3 可信开环规划(视野认证 + 历史感知逆序列优化)。 | `src/models/model_state_transition.py`;`src/encoders/fractional_memory.py`;`real_validation/openloop_planner.py` |
| **IV. Experiments** | E1 六路编码器消融(px + mm);E2 速率泛化;E3 物理接地(α 匹配);E4 歧义集/T*;E5 无记忆 vs 记忆规划质量;E6 信任视野;E7 实机 NDI gap。 | `docs/paper/04_experiments.md` |
| **V. Conclusion** | 非马尔可夫自建模的物理极限;对未来"记忆核匹配材料谱"的指导。 | — |

---

## 四、差异化表(Related 里的"我们对 vs 他们")

| 已发表 | 他们 | 我们 |
|---|---|---|
| Zhang Nat Comms 2026(Koopman 33 构型) | 控制侧跨构型泛化,无视觉全身形态 | 免标定视觉**全身形态自建模**,不做"一个模型多构型" |
| Tang Sci Adv 2026(突触控制器) | 闭环在线适应外力 | 开环逆序列规划 + 信任视野(不是每步闭环补偿) |
| Yu 2026(Bézier + NODE) | 曲线参数化降低预测难度 + 双视角避障 | 不争曲线表示;争**记忆表示**(物理接地)+ 速率/循环加载定量评估 |
| Chen 2025 / Schäfke 2024(迟滞+RL/NMPC) | 闭环控制用记忆 | 开环自建模 + 信任视野;分数阶核(物理接地)而非通用 GRU |
| Wang 2024 / Cho 2024(无记忆失效/路径依赖量化) | 已量化前向映射失败与路径依赖 | 我们把这些当动机/证据,并补**逆映射歧义集量化** |
| Thuruthel 2017 / Krauss 2026(开环 shooting) | 开环规划但无显式迟滞/无认证视野 | 显式迟滞前向模型 + **认证可信视野** |
| NJF Nature 2025 / SoftNeRF IROS 2024 | 单机器人自建模,需内参/度量 3D | 免标定 2D 像素骨架 + NDI mm 真值 |
| Shao T-ASE 2025 / FBGNN 2026(分数阶+软体) | 分数阶在控制律/物理权重 | 分数阶在**编码器**(动作历史→形状的时序层) |

---

## 五、关键参考文献(区分核实状态)

> 完整地图见 `docs/paper/01_landscape.md`。这里列论文写作会用到的核心集。

### 必引(已核实)
- Zhang et al., *RL in linear embedding space unlocks generalizable control across soft robot configurations*, Nat Commun 2026 (arXiv:2606.08104)
- Tang et al., *A general soft robotic controller inspired by neuronal structural and plastic synapses...*, Sci Adv 2026
- Yu, Wang, Tan, *Shape-Interpretable Visual Self-Modeling...*, arXiv:2603.01751 (2026)
- Chen et al., *Hysteresis-Aware NN Modeling and Whole-Body RL Control of Soft Robots*, RA-L 2025 (arXiv:2504.13582)
- Cho et al., *Accounting for Hysteresis in the Forward Kinematics of Tendon-Driven Continuum Robots*, RA-L 2024 (arXiv:2404.03816)
- Wang et al., *Using Neural Networks to Model Hysteretic Kinematics in Tendon-Actuated Continuum Robots*, 2024 (arXiv:2404.07168)
- Shao et al., *Self-Attention Enhanced Dynamics Learning and Adaptive Fractional-Order Control...*, T-ASE 2025
- Schäfke et al., *Learning-Based NMPC of Articulated Soft Robots Using RNNs*, RA-L 2024 (arXiv:2411.05616)
- Thuruthel et al., *Learning dynamic models for open loop predictive control of soft robotic manipulators*, Bioinspir. Biomim. 2017
- Krauss et al., *Accurate Open-Loop Control of a Soft Continuum Robot Through Visually Learned Latent Representations*, arXiv:2603.19655 (2026)
- Bern et al., *Soft Robot Control With a Learned Differentiable Model*, RoboSoft 2020
- Du et al., *Underwater Soft Robot Modeling and Control with Differentiable Simulation*, RA-L 2021
- Chen et al., *Fully Body Visual Self-Modeling of Robot Morphologies*, Sci Robotics 2022
- Li et al., *Neural Jacobian Fields*, Nature 2025
- Shan et al., *SoftNeRF*, IROS 2024
- Kasaei et al., *Shape-NODE / Synergistic Framework...*, arXiv:2501.03859 (2025)
- Wong et al., *A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance*, arXiv:2603.19424 (2026)
- Hachen et al., *Nonlinear MPC Task-Space Controller Satisfying Shape Constraints*, RA-L 2025
- Monteiro et al., *Visuo-dynamic self-modelling of soft robotic systems*, Front. Robot. AI 2024
- DeepSoRo, RA-L 2020; DSVB, ICRA 2023
- Vid2Sid, ICLR 2026; gradSim, ICLR 2021
- Zou & Gu, *Feedforward Control of the Rate-Dependent Viscoelastic Hysteresis... DEA*, RA-L 2019
- HasMorph, Sci Adv 2025

### 待核(引用前再核全文)
- FBGNN, EAAI 2026 · FTL-GCN 2026 · Flow-Matching 逆动力学 arXiv:2604.03006 · MORPH-DSLAM · MonoPhysics arXiv:2605.30320 · PhysCon-Deform · fPLCS-DeLaN · FO-Elman 2026 · GL-LSTM 2025 · Neural Fractional Attention ODE (NeurIPS 2025)

### 不要引
- Blackwell "Self-Modeling Networks" (arXiv:2503.12767, 两次检索为空)
- "Continuum Robot Modeling with Action Conditioned Flow Matching" (arXiv:2605.09216, 未核实且易混淆)

---

## 六、后续建议(写正文前的顺序)

1. **跑 Exp A**(六路消融)—— 数据/代码现成,2 天内出 go/no-go。
2. **跑 Exp B 阶段 1 + Exp E**(手头数据)—— 确认 P3 与 P2 证据。
3. **写 Method + Related 初稿**(差异化表直接可用),同时 A3/A4/F 补图。
4. **接硬件 → Exp G**(真机闭环 + 多速率/阶跃新数据),这决定"首次/NDI/免标定"三条实证能否全部立住。
5. 投稿前:全文核对 `01_landscape.md` §四待核引用;用 `git log` 归档所有实验超参与数据版本(可复现)。

---

## 七、一句话收尾

**论文不再回答"怎么控制不同参数/外力的软臂"(已被 Nat Comms/Sci Adv 解决),而是回答"软臂的形态记忆是什么、怎么学、怎么用"——项目已有的分数阶 GL 记忆 + 免标定状态转移 + 可信开环规划,正好是这个新问题的完整答案。**
