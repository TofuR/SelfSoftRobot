# 01 · 已核实竞品地图(2020–2026)

> **调研方式**:两个并行 workflow(19 + 8 个搜索子 agent,Exa/WebSearch 逐篇核实,拒绝未验证引用)+ 项目内 `docs/papers/notes/` 63 篇深读笔记。
> **核实状态**:每条标注 已核实(publisher/arXiv 页面或全文)/ 部分核实(聚合器/摘要)/ 待核(未独立确认)。
> **日期**:2026-08-09。**结论先给**:
> 1. "单模型适配 30+ 构型"已占(控制侧)→ **不重复**;
> 2. "在线适应外力"已占(闭环侧)→ **不重复**;
> 3. "骨架→曲线参数化"已占(表示侧)→ **不重复**;
> 4. "全身避障 + 冗余臂"已占(头条)→ **不重复**;
> 5. 真正空缺集中在:**物理接地记忆(分数阶 GL 核作时序编码器)+ 全身形态速率/循环加载定量评估 + IK 歧义集量化 + 免标定 2D 骨架 state + NDI mm 真值**。

---

## 一、用户点名的 3 篇 —— 精确核实

### 1.1 "30+ 构型一个模型"(用户的判断:"原问题已被解决")

**Zhang et al., "Reinforcement learning in linear embedding space unlocks generalizable control across soft robot configurations", Nature Communications, 2026. DOI 10.1038/s41467-026-72491-9; arXiv:2606.08104.**(已核实:arXiv 摘要页 + Nature 文章页)

- **方法**:RL 策略作用于共享的 **Koopman 线性动力学嵌入空间**;单基础策略 + 在线更新迁移到 33 种构型(arXiv v1 摘要写 30,存在内部不一致);75× 迁移样本缩减。
- **能力**:高速(1.89 m/s)、1 kg 负载、多执行器故障、真实技能(钉锤、倒酒、书法、接球)。
- **本质**:动力学**控制**框架(策略在动力学嵌入里),**不是**"机器人看自己学形状"的自建模。无相机观测、无免标定、无全身形状重建、无 SDF/NeRF 自模型、无显式迟滞处理(未核实)。
- **同类邻近**(均控制侧):Tang et al., Sci Adv 2026(突触控制器,含"全身整形"= 命令形状的跟踪,44–55% 误差↓);O'Neill et al., Nat Comms 2026(接触动力学 Koopman 全局线性化);Bruder et al., IJRR 2024(Koopman 残差 + MPC);Chen et al., FASTA 2025(统一深度 Koopman)。

### 1.2 "建模+规划两模型,在线更新规划器适应外力"

- **核实结论**:已 **settled**。代表(全已核实):Tang et al., Sci Adv 2026(离线结构突触 + 在线可塑突触,误差门控 LTP/LTD,收缩度量稳定);Tang et al., IJRR 2026(元学习 + 不确定性感知最优控制);Lu et al., Soft Robotics 2024(FBG 在线形状控制,Lyapunov 收敛);Fang et al., IJRR 2026(在线模型 RL);Veronese et al., RoboSoft 2024(反馈误差学习在线前馈);Zhuang et al., T-ASE 2026(离线-在线形状控制)。
- **共性缺陷(缺口所在)**:以上**全部是每步闭环**(观测→修正→再算)或在线重训控制器。**没有**:开环逆动作序列规划(整条轨迹)+ 在线更新前向模型 + 漂移触发的再规划调度 + 规划器失效判定。这个子组合仍空缺。

### 1.3 "骨架点→曲线参数,降低预测难度"

- **核实结论**:一般论点**已多组占据**。代表:Yu et al., 2026(Bézier + NODE,形状误差≤1.56% 图像分辨率、末端<2% 臂长,双视角 + 避障 + 自运动);Rao et al., RA-L 2022(Euler Arc Splines,6 参数,末端 0.43% 臂长);Mbakop et al., T-RO 2024(PH 曲线 ROM);Caradonna et al., 2025(SoFFT 傅里叶);Valadas et al., 2024(Della Santina 组,POD/PCA 应变低维,70% 形状误差↓);MoSS(单目,0.91mm/0.36% 臂长)。
- **缺口**:① 免标定 2D 像素骨架 `[col,row,0]` 作 state 喂曲线参数输出头 —— 无人做;② 同架构"曲线参数回归 vs 原始骨架节点回归"头消融 —— 无人做(现有增益都对比物理模型/NODE,不对骨架节点回归基线);③ B-spline/NURBS、傅里叶系数作**学习预测输出** + px/mm 双指标 —— 空缺。

---

## 二、五支柱邻近威胁(写论文必须引用并区分)

> 见 [`02_scientific_problem.md`](02_scientific_problem.md) 五支柱判定。这里列每支柱的"最危险近邻"。

### P3 威胁(分数阶记忆)
| 工作 | 分数阶放哪 | 与我们的差异(必须写明) |
|---|---|---|
| **Shao et al., T-ASE 2025**(自注意力动力学学习 + 自适应**分数阶滑模控制**,连续体软机器人) | **控制器**,整数阶注意力编码器 | 我们:分数阶在**编码器**(时序层),输出**视觉全身形态**;他们:分数阶在控制律,输出关节空间动力学 |
| **fPLCS-DeLaN 2026**(分数阶深度拉格朗日网) | 物理先验结构,时序用整数阶卷积自注意力 | 同上 |
| **FBGNN, EAAI 2026**(介电弹性体致动器) | 物理权重(分数阶 backlash 微分方程参数映射)+ 整数 GRU | 单致动器位移;我们:全身形态 + 免标定视觉 |
| **FO-Elman 2026 / GL-LSTM 2025** | **优化器/激活**(分数阶梯度),非时序核,非机器人 | 我们:GL 离散幂律核作序列记忆核 |
| **FTL-GCN 2026** | GL 核作图卷积时序层,但用于微纳导航 | 结构最近的先例,须引 |

### P4 威胁(开环逆序列规划)
| 工作 | 内容 | 与我们的差异 |
|---|---|---|
| **Thuruthel et al., 2017**(Bioinspir. Biomim.) | NARX RNN 动力学 + 单发 shooting 开环 | 无状态条件、无显式迟滞、无视野认证 |
| **Krauss et al., 2026**(arXiv:2603.19655) | 视频学习潜动力学 + 潜空间单发开环最优控制 + 潜状态条件 | **最接近的近邻**:两段气动 SCR,无相机反馈。但我们有显式迟滞建模 + 视野认证;他们无 |
| **Flow-Matching 逆动力学, 2026**(arXiv:2604.03006) | Rectified Flow 开环前馈逆动力学,>50% 跟踪 RMSE↓ | 动作空间逆动力学,**非**视觉全身形状自模型 |
| **S2C2A, IEEE ToR 2025** | biLSTM 前向模型 + 多步规划 | 执行是闭环 |
| **Borvorntanajanya, RA-L 2026** | 序列基 IK 补偿迟滞/串扰,开环 | 手术致动器,非视觉全身 |

### P1/P2 威胁(非马尔可夫性/无记忆失效)—— 已被占,只能作动机
- **Chen et al., RA-L 2025**(arXiv:2504.13582):方向馈入→MSE↓84.95%,路径差达 3.4% 臂长;闭环 RL。
- **Cho et al., RA-L 2024**(arXiv:2404.03816):全身形态路径依赖量化 9±6.5% 臂长(2773 构型);历史条件前向模型。
- **Wang et al., 2024**(arXiv:2404.07168):无记忆 FNN RMSE 3.061 vs LSTM 0.649(~2–5×);循环加载。
- **导管对比 2024**:无记忆 FNN 预测迟滞环均值,±2.5°。
- **Ma et al., 2022**:开环静态 BP IK 失败,自述"迟滞导致误差"。

### P5 威胁(全身避障)—— 头条被占
- **Yu et al., 2026**; **Kasaei et al., 2025/2026**(Shape-NODE + MPPI,支气管镜 phantom); **Hachen et al., RA-L 2025**(冗余 MPC 全身安全约束,30Hz 硬件); **Wong et al., 2026**(闭式 CLF-CBF 全身避障,硬件); **Gandhi et al., 2026**(视觉伺服全身形状控制); **Veil et al., RA-L 2026**(形状空间图规划)。

---

## 三、已核实空缺清单(按可辩护性排序)

1. **免标定 2D 像素骨架 state 喂学习型全身形态场** —— 全部已有自模型(Chen 2022, SoftNeRF, RobotSDF, 3DGS, NJF)输出度量 3D 占用,需内参。**无人**在无相机矩阵的图像像素系里做全身形态自建模+规划。→ **最强硬件/数据空缺**
   > ⚠️ **2026-08-10 升级注记**(多驱动 3D):这个空缺在 3D 化时升级为"**无标定板的身体自我标定 + 学习式免标定几何**(L2/L3,见 [`06_multiview_self_calibration.md`](06_multiview_self_calibration.md))"——同样无人做。注意:**自标定(SfM/autocalibration)本身是老领域**,新意不在算法,而在"自标定作为软体自建模管线的组成部分 + 与自模型耦合(模型在环)"。写 related work 须引经典自标定(Faugeras、Maybank & Luong、Pollefeys)作背景。
2. **物理接地记忆(分数阶 GL 幂律核)作神经形态模型的时序编码器** —— P3,见上。→ **最强方法空缺**
3. **全身形态在显式速率变化/循环加载下的定量评估** —— 速率依赖只在 DEA 致动器(Zou & Gu 2019, 0.05–1.5Hz, 最大误差 6.18%)或准静态耐久里测;连续体臂的全身形态-vs-速率指标**检索未见**。→ **可写"首次报告"**
4. **NDI 6DOF mm 外部真值验证学习型软臂自模型** —— 检索未见。→ **最强经验差异点**
5. **IK 歧义集量化**(迟滞下逆映射前像集/直径)—— 前向映射路径依赖已量化(Cho/Chen),**逆映射歧义集无人量化**。→ **最深的科学空缺**
6. **时序编码器六路系统性消融**(EMA/Gamma/GRU/Transformer/TCN/GL/S4-Mamba)—— 软体领域只有 GRU-vs-LSTM(Schäfke 2024)和逆动力学里的 MLP/LSTM/Transformer。→ **项目编码器套件唯一可填**
7. **从外部单相机观测学习软体自身隐蔽粘性/塑性潜状态** —— 现有潜状态来自嵌入式相机(DeepSoRo)、驱动流(DSVB)、仿真力学(MORPH-DSLAM)。外部免标定相机 + 物理接地记忆变量的组合空缺。

---

## 四、引用诚信提醒(写论文前必看)

- ✅ **已核实可引**:Zhang Nat Comms 2026、Tang Sci Adv 2026、Yu arXiv:2603.01751、Chen RA-L 2025 (arXiv:2504.13582)、Cho RA-L 2024、Wang arXiv:2404.07168、Shao T-ASE 2025、Schäfke RA-L 2024、Thuruthel 2017、Krauss arXiv:2603.19655、Bern RoboSoft 2020、Du RA-L 2021、Monteiro Front. Robot. AI 2024、DeepSoRo RA-L 2020、DSVB ICRA 2023、Vid2Sid ICLR 2026、gradSim ICLR 2021、Zou & Gu RA-L 2019、HasMorph Sci Adv 2025。
- ⚠️ **部分核实/待核(引用前再核全文)**:FBGNN EAAI 2026(摘要/高亮级)、FTL-GCN 2026(第二手)、Flow-Matching 逆动力学 arXiv:2604.03006(preprint,聚合器)、MORPH-DSLAM/MonoPhysics/PhysCon-Deform 2026(摘要级)、fPLCS-DeLaN(学者主页)。
- ❌ **不要引**:Blackwell "Self-Modeling Networks" arXiv:2503.12767(两次独立检索为空);"Continuum Robot Modeling with Action Conditioned Flow Matching" arXiv:2605.09216(未核实且易与 2604.03006 混淆);Hu/Lin/Lipson NMI 页码未确认。
- 📌 **易混**:"Self-Modeling Robots by Photographing" 指 IJRR 2025 关节 3DGS 论文,区别于 Sci Robotics 2022 的 Chen 全身体视觉自建模;MoSS 训练用了深度/仿真数据,不要夸成"纯视觉无任何外源"。
