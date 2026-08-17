# ICRA 论文规划 · 软体机器人整体形态自建模(非马尔可夫记忆视角)

> **日期**:2026-08-09 · 分支 `feat/real-data-transition`
> **状态**:调研完成(两个并行 workflow 核实 100+ 篇文献)+ 科学问题重构完成 + 添加项与实验计划已定
> **产出**:本文档为索引;详细内容见 01–05 各章
> **关联**:[`docs/papers/related_work_draft.md`](../papers/related_work_draft.md)(旧 related work,差异化论点需按本文 §2 重写)· [`docs/directions/12_scientific_problems_soft_robot_self_modeling.md`](../directions/12_scientific_problems_soft_robot_self_modeling.md)(科学问题 A/B/C)· [`docs/directions/17_path_dependent_ik.md`](../directions/17_path_dependent_ik.md)(方向 17 的路径依赖 IK)

---

## 〇、TL;DR —— 一句话结论

**原始科学问题("不同参数/不同外力下控制难")在控制侧已被 2026 年 Nature Comms + Sci Adv 解决,不能再当头条。但"视觉全身形态自建模"是另一条能力线,项目在这里有真实、可辩护的空缺——核心是:粘弹性软臂的形态是驱动历史的泛函而非当前指令的函数(非马尔可夫),项目独有的分数阶 Grünwald-Letnikov 记忆核正好物理接地了这个"记忆",并且这种表示能解锁**良置的历史感知 IK + 可信开环动作序列规划**。论文应围绕这三点写,免标定 2D 骨架 + NDI mm 真值 + 速率变化定量评估作为硬件/数据差异化。**

---

## 一、用户点名的 3 篇工作核实结果(§01 详述)

| 用户印象 | 实际核实 | 覆盖了什么 / 没覆盖什么 |
|---|---|---|
| "30+ 构型一个模型适配" | **Zhang et al., "RL in linear embedding space unlocks generalizable control across soft robot configurations", Nature Communications 2026**(arXiv:2606.08104) | ✅ 跨构型**控制**(Koopman 线性动力学嵌入 + RL 策略迁移,33 构型,75× 样本)。❌ **不做视觉全身形态自建模**:无相机观测、无免标定、无全身形状重建、无 SDF/NeRF 自模型。是动力学控制,不是"机器人看自己学形状" |
| "建模+规划两模型,在线更新规划器适应外力" | Tang et al., Sci Adv 2026(突触控制器)+ Tang IJRR 2026(元学习)+ Lu Soft Robotics 2024(FBG 在线)+ Fang IJRR 2026,等 10+ 篇 | ✅ "在线适应外力"已**settled**(全是**每步闭环**反馈/在线重训控制器)。❌ 无**开环**逆动作序列规划 + 在线模型更新 + 再规划调度 + 规划器失效判定的组合 |
| "骨架点→曲线参数,降低预测难度,大提升" | **Yu et al. 2026**(Bézier 曲线 + Neural ODE, arXiv:2603.01751)+ Euler arc splines(RA-L 2022)+ PH 曲线(T-RO 2024)+ POD/PCA 应变模式 + SoFFT(傅里叶)+ Della Santina 低维应变模型 | ✅ "曲线参数化降低形状预测难度"**已多组占据**(MoSS 0.36%/臂长, Della Santina 70%↓, Yu 2026 形状误差≤1.56% 图像分辨率)。❌ 无"免标定 2D 像素骨架作为 state" + 无同架构 skeleton-vs-curve 头消融 |

**共同结论**:三篇解决的都在**控制侧**或**表示侧**;没有一篇做"免标定单相机 + 视觉全身形态自建模 + 物理接地记忆 + 用它做规划"的完整链路。这正是本项目的立足点。

---

## 二、科学问题重构(§02 详述)

### 旧问题(已死)
> "软体机器人在不同参数、不同外力的情况下控制难。"

### 新问题(可辩护)
> **软体机器人的整体形态自建模是"非马尔可夫系统辨识"问题:粘弹性使"指令→形态"映射不是函数而是加载历史的泛函。我们(1) 用 NDI 真值定量证明全身形态的路径依赖与速率依赖;(2) 证明无记忆学习自模型在动态/循环加载下系统性失效;(3) 提出物理接地的记忆表示——Grünwald-Letnikov 分数阶幂律核作为神经形态模型的时序编码器,匹配粘弹性弛豫谱;(4) 展示它解锁的新能力:良置的历史感知逆规划 + 可信开环动作序列规划。**

### 五支柱判定(对抗复核结论)

| 支柱 | 判定 | 空缺残余 |
|---|---|---|
| P3 GL 分数阶核作**神经形状模型的时序编码器** | **STILL-OPEN(最强支柱)** | 所有已核实"分数阶+软体+NN"组合都把分数阶放在控制器(Shao T-ASE 2025 分数阶滑模)/物理权重+整数 GRU(FBGNN 2026)/优化器(FO-Elman, GL-LSTM 2025)/图导航(FTL-GCN 2026)。**无人放 GL 离散核作动作历史→形状模型的编码层** |
| P4 迟滞感知 + 开环逆动作序列规划 + **可信视野认证** | PARTIALLY-CLAIMED(组合空缺) | 每件单独已发表(Thuruthel 2017 开环 shooting; Krauss 2026 潜空间开环; Chen 2025/Schäfke/S2C2A 迟滞前向模型; Flow-Matching 逆动力学 2604.03006)。**"显式迟滞 + 开环多步序列 + 认证可信视野"的组合无人占** |
| P1 经验非马尔可夫性(全身形态,NDI 真值,速率+路径) | PARTIALLY-CLAIMED | Chen RA-L 2025(3.4% 路径差)、Cho RA-L 2024(9±6.5% 全身多值)已量化前向映射。**空缺:全身形状 + 速率依赖 + NDI mm 统计严谨性,以及 IK 歧义集量化**(逆映射前像集直径,无人做过) |
| P2 无记忆模型在反转处失效 | **CLAIMED(不能做头条)** | Wang 2024(3.061 vs 0.649 deg)、导管 2024、Ma 2022 已量化。只能作动机/证据 |
| P5 全身避障 + 冗余 3D 臂 | **CLAIMED(头条被占)** | Yu 2026、Kasaei 2025/26、Hachen 2025、Wong 2026、Gandhi 2025 全部已做。降级为 demo |

---

## 三、要添加到项目的内容(§03 详述,映射到代码)

| 优先级 | 添加项 | 是否已实现 | 对应代码 |
|---|---|---|---|
| ★★★ | **时序编码器系统性消融**(EMA/GammaLaguerre/GRU/Transformer/TCN/GL 六选) | 代码全有,**实验未跑** | `src/encoders/*`(六种全在);`model_state_transition.py` 的 `_ENCODERS` dict |
| ★★★ | **速率泛化实验**(训一个速率,测其他速率) | 数据有(准静态 173114 / 动态 172916),**未跑** | `scripts/real/*` + `train_transition.py` |
| ★★★ | **学习 α vs 实测弛豫幂律指数**(物理接地证明) | 数据有,**未做分析** | GL 核 `fractional_memory.py` 的 `alphas` + NDI 阶跃弛豫序列 |
| ★★★ | **IK 歧义集量化**(函数 vs 泛函的形式化,真正空缺) | **未实现** | 新脚本,基于 `openloop_planner` 的 rollout |
| ★★ | 历史感知 IK 规划质量对比(window=1 vs 40 → 无记忆 vs 记忆) | 方向 17 Exp2 已规划,**未跑** | `scripts/training/train_transition.py --window_size` |
| ★★ | 修复 z 懒惰(序列级训练或放弃 z 声称) | **z 需修/需弱化** | `model_state_transition.py`;诚实边界见 §05 |
| ★ | 可信视野认证形式化(K_max/auto_k → "信任视野") | 代码已实现,需论文化表述 | `real_validation/planning/auto_k.py` + `eval_horizon.py` |
| ★ | 免标定端到端实机演示(NDI mm) | 部分(有 0.77mm GT 结果),需 P2 重采重训 | 见 `docs/real_data/deployment.md` §11 |

---

## 四、文档导航

| 文件 | 内容 |
|---|---|
| [`01_landscape.md`](01_landscape.md) | 已核实竞品地图 + 三篇点名工作详析 + 空缺清单 + 引用诚信提醒 |
| [`02_scientific_problem.md`](02_scientific_problem.md) | 科学问题重构论证 + 五支柱判定 + 与方向 12/17 的衔接 |
| [`03_additions.md`](03_additions.md) | 逐项添加内容(实验/模型修复/数据),映射到代码,含工作量 |
| [`04_experiments.md`](04_experiments.md) | 实验计划(Exp A–G):目标、资产、命令、go/no-go、时间线 |
| [`05_outline.md`](05_outline.md) | ICRA 论文大纲 + 差异化表 + 需引用的关键文献(区分已核实/待核) |
| [`06_multiview_self_calibration.md`](06_multiview_self_calibration.md) | **多视角自标定设计(L2/L3)**—— 多驱动 3D 升级如何保持"免标定":身体/场景自我标定 + 学习式免标定几何 |
| [`icra_draft.md`](icra_draft.md) | **ICRA 论文中文草稿(脚手架版)**—— 结构完整、方法写实、结果留 `[待填]`;每节标明要跑的实验(Exp A–H) |

---

## 五、给用户的一句话决策建议

1. **别再当头条**:跨构型控制、在线外力适应、骨架→曲线、全身避障 —— 全是已占的。相关工作中把它们当背景与对比,证明你**不重复**它们。
2. **论文脊柱 = P3 + P4**:物理接地的分数阶记忆编码器(学形状模型)+ 它解锁的可信开环历史感知规划。P1/P2 作动机/证据,P5 作 demo。
3. **三个"第一次"可写进贡献**:① 全身形态在速率变化/循环加载下的定量评估(检索未见);② NDI mm 真值验证的学习型软臂自模型(检索未见);③ 免标定 2D 像素骨架 state(检索未见)。它们互相叠加,构成硬件/数据差异化的护城河。**多驱动 3D 升级时,③ 升级为"身体自我标定 + 学习式免标定的多视角几何"(见 [`06_multiview_self_calibration.md`](06_multiview_self_calibration.md))—— 检索同样未见,且更硬。**
4. **最该先跑的实验是时序编码器六路消融** —— 代码现成、数据现成,是 go/no-go 的最快信号。
