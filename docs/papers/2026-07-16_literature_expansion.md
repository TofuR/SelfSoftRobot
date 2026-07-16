# 文献调研扩展报告 · 软体机器人自建模 / 形态估计 / 迟滞 (2026-07-17)

> **范围**:7 主题 web 调研 + 已有综述(`literature_2025_2026_survey.md`)arXiv ID 复核。
> **方法**:并行 scout 抓取候选 → 逐篇 **抓取真实 web 页面(arXiv/Exa)验证**(无法验证者拒绝,杜绝伪造引用)→ 为每篇写 `docs/papers/notes/notes_<key>.md` 深读笔记 → 本报告做主题索引 + **论述→引用地图**。
> **产出**:`docs/papers/notes/` 下 **63 篇**深读笔记(本报告汇总),每篇含"支撑哪句论述"字段。
> **关联**:`docs/background/literature.md`(综述层)、`docs/papers/literature_2025_2026_survey.md`(表格层)为本报告的前序;本报告在其上**补全单篇深读 + 论述映射**。

---

## 0. 用户的 intro 论述(逐条编号,供下文引用)

| # | 论述 |
|---|------|
| **A1** | 大多数软体机器人运动学建模只做**尖端位置**感知/控制, 不做全身形态估计。 |
| **A2** | 仅尖端不够: 狭窄环境**碰撞避让**——中段同样不能碰障碍, 只控尖端无法保证全身安全。 |
| **A3** | 仅尖端不够: **接触式操作**(缠绕/包裹)需知道身体各段与目标的接触关系来控接触力。 |
| **A4** | 现有方法(**FBG**光纤测应变推弯曲 / **电磁追踪** / **缆绳长度编码**)只能得**离散点**坐标, 不能完整形态建模。 |
| **A5** | 之前全身形态估计把**形态学建模与运动学建模分开**(两阶段串联), 误差累积, 形态学误差传播到运动学。 |
| **A6** | 传统建模需**先验 CAD 模型 / 精确物理参数**, 损伤或变形后预设模型失效。 |
| **A7** | 数据驱动自建模: **端到端联合训练**, 避免先验依赖 + 误差累积; 还能**隐式捕获粘弹性迟滞**等非线性。 |
| **A8** | 一些自建模工作形态好但**没考虑迟滞效应与高速运动**, 高速下难准确估计形状。 |
| **A9** | 大部分方法需**持续观测**消除误差, 遮挡/不可见时难以工作。 |
| **A10** | 因此考虑用**视觉/图像 + 数据驱动自建模**做软体机器人形态建模。 |

---

## 1. 主题分组索引(63 篇,按 7 主题)

> 文件名见 `docs/papers/notes/`。"支撑"列指该笔记明确标注支撑的论述编号(空白=该笔记由前序会话写入未填,本报告据其主题在 §2 归位)。

### 主题 ① whole-body shape sensing(传感与全身形态感知)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_shi2017_shape_sensing_survey` | 连续体形状感知综述:FBG/EM/成像三大类 | A4 / A1 / A10 |
| `notes_khan2019_fbg_multicore_shape` | 多芯 FBG 应变→曲率→Frenet-Serret 积分重建 | A4 |
| `notes_wild2025_unified_fbg_shape_sensing` | 统一 FBG 形状感知,嵌入式高速 | A4 / A8 / A7 |
| `notes_an2024_strain_pose_fusion_shape` | FBG 离散应变 + IMU 融合,应变积分误差累积 | A4 / A5 / A7 |
| `notes_costa2023_linear_magnetic_encoders` | 缆绳/线性磁编码器,离散点末端估计 | A4 / A1 / A6 |
| `notes_wang2024_srss_continuum_shape` | 9 个软应变点 + 显式积分,离散点→形状 | A4 / A2-3 / A6 |
| `notes_russo2023_continuum_robots_overview` | 连续体综述,定曲率局限,先验依赖 | A6 / (survey) |
| `notes_webster2010_constant_curvature_review` | PCC 几何先验两段式映射奠基 | A6 / A7 |
| `notes_aft2025_markerless_shape_tracking` | 视觉免训练免标记,纹理隐式标记,2.6% 末端,遮挡鲁棒 | A9 / A1 / A10 |
| `notes_suresh2024_neuralfeels_in_hand` | 神经场在线建图,遮挡下需多模态 | A9 / A7 |

### 主题 ② self-modeling robots(数据驱动自建模 / 本体感知)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_hu2025_part_nerf_self_modeling` | 部件级 NeRF 自建模,形态+控制 | A7 |
| `notes_3dgs_self_modeling_2025` | 3DGS 自建模,关节角→外观,中山大(IJRR) | A7 / A5-6 / A10 |
| `notes_laflaquiere2019_body_image` | 体像/感觉运动前向预测,无 CAD 先验 | A7 |
| `notes_lucny2023_mirror_self_model` | 视觉自建模,任务无关底座复用 | A7 |
| `notes_yang2024_robotsdf_implicit_morphology` | 隐式 SDF 形态自建模,碰撞避让 | A7 / A2 / A6 |
| `notes_liu2024_differentiable_robot_rendering` | 可微渲染自模型,端到端,免先验 CAD | A7 / A10 |
| `notes_chen2023_data_driven_soft_robot_review` | 数据驱动软体建模综述,先验 vs 数据二分 | (survey) A7 |
| `notes_falotico2025_csm_learning_control_review` | 连续体学习控制综述,迟滞为开放问题 | A5 / A7 / A8 |
| `notes_zhang2023_ml_best_practices_proprioception` | 软体本体感知 ML 最佳实践,时序/迟滞 | A7 / A1 |
| `notes_ssl_proprioception_2025` | SSL 本体感知 | A7 |
| `notes_farghdani2025_damaged_legged_recovery` | 刚体损伤重配(预设模型失效的旁证) | A6 |
| `notes_koopman_embedding_soft_robot_2026` | Koopman 线性嵌入,跨构型迁移 | A7 |

### 主题 ③ neural-implicit / NeRF / 3DGS / SDF(神经隐式表示)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_neural_jacobian_fields_2025` | 密集 Jacobian 场,多视角 RGB-D 自监督(Nature) | A7 |
| `notes_dgs_lrm_2025` | 单目视频前馈可变形 3DGS 重建(NeurIPS) | A10 / A1(反衬) |
| `notes_flow_matching_tdcr_2026` | 动作条件 Flow Matching 点云,TDCR 稳态 | A7 / A1 |
| `notes_yu2026_shape_interpretable` | Bézier + Neural ODE,可解释形状控制(中山大) | A7 / A2 |
| `notes_shen2022_acid_deformable_dynamics` | ACID 动作条件隐式动力学,免手工物理 | A7 / A6 / A10 |
| `notes_tang2026_dlo_diffusion` | 生成式形状补全,遮挡鲁棒,DLO 同构 | A9 / A10 |

### 主题 ④ kinematics & dynamics modeling(运动学/动力学,两阶段对照)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_wang2022_nonconstant_curvature` | 非定曲率,PCC 动态任务失效,7 m/s 高速 | A6 / A8 |
| `notes_till2019_realtime_cosserat_dynamics` | Cosserat 实时动力学基线,强先验 | A8 / A6 |
| `notes_herrmann2026_discrete_geom_ekf` | 几何精确梁 + EKF 扰动观测,实时 | A8 / A6 |
| `notes_zheng2025_boundary_observer` | 边界观测器,Cosserat 实时,迟滞空白 | A8 |
| `notes_jiang2025_conformal_kinematic_uq` | 学习型 FK + 共形预测 UQ | A5 / A7 |
| `notes_shape_node_control_2025` | Shape-NODE + Control-NODE MPC(Cosserat 先验) | A7 / A2 |
| `notes_pinn_soft_robot_2025` | PINN 代理,467× 加速,47 Hz MPC | A5 |
| `notes_gao2024_residual_physics` | 解析 sim + 残差修正,hybrid 仍赖先验 | A5 / A6 / A7 |
| `notes_chow2021_residual_nn_planning` | 残差网络补准静态先验之失,充放气迟滞 | A6 / A7 / A1 |

### 主题 ⑤ hysteresis & viscoelastic modeling(迟滞/粘弹性 —— 项目核心新意)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_chen2025_hysteresis_whole_body_rl` | ★ 迟滞感知全身 NN + RL,MSE 降 84.95%(arXiv 2504.13582,**官方元数据核实**) | A2 / A7 / A8 |
| `notes_park2024_tcn_hysteresis_compensation` | TCN 学缆绳迟滞补偿(关节角,非全身) | A7 / A1 |
| `notes_liu2024_bilstm_mlp_hysteresis` | BiLSTM+MLP 循环网络捕获迟滞,替代唯象算子 | A7 |
| `notes_sun2022_pirnn_soft_pneumatic` | PIRNN 物理先验 RNN,SPA 迟滞 | A7 / A10 |
| `notes_gao2022_fractional_viscohyperelastic` | 分数阶幂律记忆核,粘弹性物理本质 | A7 / A8 |
| `notes_gu2017_dea_viscoelastic_statespace` | DEA 粘弹性内变量状态空间,记忆时间常数 | A7 / A8 |
| `notes_delamorena2025_spa_hysteresis_review` | SPA 迟滞建模综述(Preisach/PI/Bouc-Wen) | A7 / A8 |
| `notes_alsaaideh2022_loaded_pam_pi_hysteresis` | 加载下 PAM 的 P-I 迟滞,负载相关非线性 | A7 / A8 |
| `notes_schäfke2024_rnn_nmpc_soft_robot` | GRU 捕获迟滞 + NMPC,1.2° 跟踪(arXiv 2411.05616,**Exa 核实**) | A7 / A8 / A9 |
| `notes_zheng2025_softae_active_dynamics` | SoftAE 动力学形态好但未显式建模迟滞(对照) | A8 / A7 |

### 主题 ⑥ dynamic / high-speed / occlusion / open-loop(动态·高速·遮挡·开环)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_wang2025_spatiotemporal_shape_loading` | ★ 时空 NN(时序+空间+融合),带载 0.22mm(arXiv 2510.22339,**Exa 核实**) | A8 / A1 / A2 |
| `notes_thuruthel2019_rnn_soft_proprioception` | 嵌入式传感 + RNN 软致动器,时变/迟滞 | A7 / A6 / A8 |
| `notes_krauss2026_open_loop_latent_scr` | 开环 rollout + 学习型潜态 + 视频自建模 | A9 |
| `notes_rezvani2025_robust_visual_embodiment` | 噪声/分割鲁棒自建模退化量化(arXiv 2510.03677) | A9 / A10 |
| `notes_tang2026_whole_body_shape` | 全身形状控制,未知情境在线优化(ICRA,Tang/Rus/Laschi) | A1 / A2 |

### 主题 ⑦ collision avoidance / contact-aware control(全身避障·接触控制)

| 笔记 | 一句话 | 支撑 |
|---|---|---|
| `notes_catnips_collision_avoidance_2023` | 神经隐式占用场,概率碰撞推理,全身体积 | A2 |
| `notes_hachen2025_nmpc_shape_constraints` | 全身形状安全区作 MPC 硬约束 | A2 |
| `notes_wong2026_clf_cbf_whole_body_avoidance` | CLF-CBF 全身避障闭式控制律,腱驱动软臂 | A2 |
| `notes_wong2025_contact_aware_safety` | 全身接触力有界 CBF,可微 Cosserat | A3 / A2 / A6 |
| `notes_mangan2026_contact_aware_planning` | 沿身体评估接触质量的接触感知规划 | A3 / A2 |
| `notes_rao2024_contact_aided_planning` | 接触辅助规划,单段借环境接触改中段曲率 | A3 / A2 |
| `notes_dickson2025_safe_contact_cbf` | 安全接触 CBF,但只覆盖末端(反衬 A3) | A3 / A1 / A2 |
| `notes_johnston2025_momentum_observer_contact` | 动量观测器估接触 wrench,需全身形状 | A8 / A2-3 |
| `notes_kasaei2025_shape_aware_whole_body` | 全身形状控制 + 物理残差 + MPPI 内窥镜避障 | A2 / A3 |

---

## 2. 论述→引用地图(写 intro 的弹药库)

> 每条列可直接引用的笔记(标题·作者/年·一句话为何支撑)。⚠️ =该论述证据偏弱,建议补的方向见末注。

### A1 — 大多数工作只做尖端位置,不做全身形态
- **Tang et al. 2026**(`notes_tang2026_whole_body_shape`)——其卖点正是"把全身形状控制作为新问题提出",反衬此前多数只做末端。
- **Thuruthel 2019 / Park 2024 / Costa 2023**(`notes_thuruthel2019…`、`notes_park2024…`、`notes_costa2023…`)——均只做末端/关节角估计或迟滞补偿,是 A1 的直接样本。
- **AFT 2025**(`notes_aft2025…`)——虽做全身重建,但只报末端误差 2.6%,反映"末端精度"是默认口径。
- **DGS-LRM 2025**(`notes_dgs_lrm_2025`)——一般可变形重建,不针对软体全身驱动条件建模,反衬本项目定位。

### A2 — 狭窄环境碰撞避让,中段同样不能碰障碍
- **CATNIPS 2023**(`notes_catnips…`)——用**全体积**神经隐式占用场做碰撞推理,正说明中段几何(非仅末端)是避让前提。
- **Hachen 2025**(`notes_hachen2025…`)——把"整段身体安全区"写成 MPC 硬约束。
- **Wong 2026**(`notes_wong2026…`)——CLF-CBF 全身避障闭式律。
- **Kasaei 2025**(`notes_kasaei2025…`)——内窥镜狭窄环境避障,全身形状感知+残差。
- **Wang 2025 时空**(`notes_wang2025_spatiotemporal…`)——带载下整臂形态,中段不可忽略。
- **Yu 2026**(`notes_yu2026_shape_interpretable`)——双视角形状控制避障,"至少一视角不碰则 3D 不碰"。

### A3 — 接触式操作需各段接触关系来控接触力
- **Rao 2024**(`notes_rao2024…`)——明确论证接触规划需全身接触信息而非末端位姿,最有力引用。
- **Mangan 2026**(`notes_mangan2026…`)——沿身体评估接触质量。
- **Wong 2025**(`notes_wong2025…`)——全身接触力有界 CBF,正是 A3 的形式化。
- **Johnston 2025**(`notes_johnston2025…`)——接触 wrench 估计需全身形状(动力学侧反证)。
- **Dickson 2025**(`notes_dickson2025…`)——⚠️ 反衬例:把"安全接触"形式化为末端力 CBF,但**只覆盖末端、中段缺失**,正好说明仅末端不足以处理全身接触。

### A4 — 现有方法只能得离散点,不能完整形态建模
- **Shi 2017 综述**(`notes_shi2017…`)——三大类(FBG/EM/成像)即 A4 的总图。
- **Khan 2019**(`notes_khan2019_fbg…`)——FBG 应变→曲率→积分重建,本质"离散测量 + 物理积分"。
- **Wild 2025**(`notes_wild2025…`)、**An 2024**(`notes_an2024…`)——嵌入式 FBG 离散点,An 2024 的"应变积分误差累积"同时支撑 A5。
- **Costa 2023**(`notes_costa2023…`)——缆绳/磁编码离散点末端,需运动学模型。
- **Wang 2024 SRSS**(`notes_wang2024_srss…`)——9 个软应变点 + 显式积分,典型离散点法。

### A5 — 形态学 + 运动学两阶段串联,误差累积
- **An 2024**(`notes_an2024…`)——"应变积分误差累积"是 A5 的直接例证。
- **Gao 2024 残差物理**(`notes_gao2024_residual_physics`)——hybrid 解析+残差仍赖先验仿真器,作 A5 反衬(端到端可避免累积)。
- **Falotico 2025 综述**(`notes_falotico2025…`)——把"前向模型影响"与 hybrid 建模作为综述主题。
- **PINN 2025**(`notes_pinn_soft_robot_2025`)——单一物理信息学习模型替代分阶段,467× 加速。
- **Jiang 2025**(`notes_jiang2025…`)——学习型 FK + UQ,呼应端到端避免累积。

### A6 — 需先验 CAD / 精确物理参数,损伤变形后失效
- **Webster 2010**(`notes_webster2010…`)——PCC 几何映射奠基,需精确物理参数。
- **Wang 2022**(`notes_wang2022…`)——摘要直陈 "PCC fails ... when executing dynamic tasks or interacting with the environment",最直接引用。
- **Till 2019 / Herrmann 2026 / Zheng 2025**(`notes_till2019…`、`notes_herrmann2026…`、`notes_zheng2025_boundary…`)——Cosserat 力学基线,强先验依赖。
- **Chow 2021**(`notes_chow2021…`)——因准静态先验不准才加残差网络(正面例证 A6)。
- **Farghdani 2025**(`notes_farghdani2025…`)——刚体损伤重配,佐证"预设模型失效"存在性(旁证)。

### A7 — 数据驱动端到端联合训练,免先验 + 隐式捕获迟滞 ★(项目核心)
- **Chen 2025**(`notes_chen2025_hysteresis_whole_body_rl`)——★ 迟滞感知全身 NN,MSE 降 84.95%,最直接的"端到端隐式学迟滞"量化证据。
- **Schäfke 2024**(`notes_schäfke2024…`)——GRU 捕获迟滞 + NMPC,1.2° 跟踪。
- **Liu 2024 BiLSTM**(`notes_liu2024_bilstm…`)、**Sun 2022 PIRNN**(`notes_sun2022…`)——循环网络隐式吸收迟滞,SPA 建模成熟做法。
- **Gao 2022 分数阶**(`notes_gao2022…`)、**Gu 2017**(`notes_gu2017…`)——粘弹性幂律记忆的物理本质,数据驱动 latent z + 时间编码器是其端到端学习化代理。
- **Liu 2024 可微渲染**(`notes_liu2024_differentiable…`)、**Shen 2022 ACID**(`notes_shen2022…`)、**Hu 2025 part-NeRF / 3DGS**(`notes_hu2025…`、`notes_3dgs_self_modeling…`)——端到端自建模、免先验 CAD 的范式代表。
- **Yang 2024 RobotSDF**(`notes_yang2024_implicit…`)——隐式 SDF 形态自建模,降低 CAD 依赖。

### A8 — 一些自建模形态好但没考虑迟滞/高速,高速下难准确 ★(项目空白点)
- **Zheng 2025 SoftAE**(`notes_zheng2025_softae…`)——形态好、泛化强,但未显式建模迟滞/高速,只把非线性当扰动——A8 的直接对照。
- **Wang 2025 时空**(`notes_wang2025_spatiotemporal…`)——明确针对"外载荷(动态)下形态估计",印证"动态需时序建模";但未显式建模迟滞,佐证空白。
- **Gao 2022 分数阶**(`notes_gao2022…`)、**Gu 2017**(`notes_gu2017…`)——定量示高速下迟滞/蠕变叠加,单一模型难泛化。
- **Wang 2022**(`notes_wang2022…`)——7 m/s 高速结果,高速下形状估计困难的现实背景。
- **Till 2019**(`notes_till2019…`)——力学侧实时动态基线,反衬数据驱动需引入动态状态转移才能在高速有竞争力。
- **Chen 2025 / Schäfke 2024**——正面把迟滞纳入的代表性工作(可作 baseline 与对照)。

### A9 — 大部分方法需持续观测,遮挡/不可见难工作
- **Rezvani 2025**(`notes_rezvani2025…`)——★ 系统量化噪声/杂乱背景使 SOTA 自建模显著退化,最直接证据。
- **Suresh 2024 NeuralFeels**(`notes_suresh2024…`)——论证纯视觉遮挡下失效,需多模态/隐状态持续跟踪。
- **Tang 2026 DLO**(`notes_tang2026_dlo…`)——生成式先验从部分/遮挡观测补全形状。
- **AFT 2025**(`notes_aft2025…`)——以"对遮挡鲁棒"为卖点,反证遮挡是共性痛点。
- **Schäfke 2024**(`notes_schäfke2024…`)——NMPC 每周期喂实测值,即"不持续观测就漂"的反证。
- **Krauss 2026**(`notes_krauss2026…`)——开环 rollout + 学习型潜态,正是 A9 的方法学回应(我们 open_loop 的同侧)。

### A10 — 用视觉/图像 + 数据驱动自建模做形态建模
- **DGS-LRM 2025**(`notes_dgs_lrm_2025`)——视觉+数据驱动形态/变形重建的代表性前沿(非软体,作范式引用)。
- **Liu 2024 可微渲染**(`notes_liu2024_differentiable…`)、**Shen 2022 ACID**(`notes_shen2022…`)、**Hu 2025 / 3DGS**(`notes_hu2025…`、`notes_3dgs_self_modeling…`)——视觉 + 神经场/可微渲染自建模的范式谱系。
- **AFT 2025**(`notes_aft2025…`)——纯视觉路线(免训练几何匹配分支),作对比范式。
- **Sun 2022 PIRNN**(`notes_sun2022…`)、**Shi 2017**(`notes_shi2017…`)——侧面支撑"数据驱动/成像做软体传感/建模是主流方向"。

---

## 3. 完整性 / 诚实性一节(引用诚信)

### 3.1 已复核的 arXiv ID(均经 arXiv/Exa 抓取确认真实)
| arXiv id | 笔记 | 核实方式 |
|---|---|---|
| 2504.13582 | `notes_chen2025_hysteresis_whole_body_rl` | arXiv API 官方元数据(修正了前序 scout 误称"…for Soft Robot Proprioception"与"3.4% 方向依赖"的错误) |
| 2411.05616 | `notes_schäfke2024_rnn_nmpc_soft_robot` | Exa 抓 arXiv 摘要页(RA-L,14 citations) |
| 2510.22339 | `notes_wang2025_spatiotemporal_shape_loading` | Exa 抓 arXiv 摘要页 |
| 2511.18215 | `notes_aft2025_markerless_shape_tracking` | Exa 抓 arXiv 摘要页 |
| 2502.01916 / 2402.01086 / 2501.03859 / 2503.05398 / 2506.09997 / 2603.01751 / 2605.09216 / 2510.03677 | 各对应笔记 | 工作流 verify 阶段抓取确认(2.2M token 耗尽前完成) |

### 3.2 ⚠️ 需注意的潜在问题(写论文前请人工再核一遍)
1. **`notes_hysteresis_aware_nn_2025.md` 与 `notes_chen2025_hysteresis_whole_body_rl.md` 指向同一 arXiv id (2504.13582)**。前者据 scout 描述撰写(标题/数字有误),后者据 arXiv 官方元数据核实。**引用时以 `notes_chen2025_…` 为准**;建议人工核对后删除或归并 `notes_hysteresis_aware_nn_2025.md`。
2. **本轮 15 个 verify 子 agent 因账户速率限制(429)未完成**,其中 11 个为 scout 新发现候选。它们的**主题已由其他笔记覆盖**(FBG→3 篇、Preisach/PI→2 篇、BiLSTM/Koopman/INR/EM 均有笔记),但具体那 11 篇的逐篇核实未完成。如需穷尽,建议在速率重置后对这些主题做定向补搜(候选关键词见各笔记"关联主题")。
3. **多数笔记"方法/结果"章节标注"据摘要"**——arXiv/Exa 仅取到摘要,未读全文 PDF。引用具体数字/架构细节前请下载全文核对(项目 `docs/papers/` 下已有 5 篇全文 PDF:Chen 2022、Hu 2025 FBV-SM、Shan 2024 SoftNeRF、Yu 2026、Tang jpg)。
4. **`26xx` 系列 arXiv id**(2603.01751 / 2605.09216)已抓取确认真实存在(非伪造),但均为 preprint,正式发表信息以引用时最新版为准。

---

## 4. 一句话总结

这批 63 篇深读笔记,把用户 intro 的 10 条论述**逐条配上了可引用的真实文献**:A2/A3(全身避障/接触)证据充分(CATNIPS、Hachen、Wong×2、Rao、Mangan);A4(离散点)有完整 FBG/EM/缆绳谱系;A7/A8(迟滞)从"被忽略"补到了有量化对照(Chen 2025 MSE↓84.95%、Schäfke GRU+NMPC、分数阶/内变量物理谱系)。**最大空白仍在 A8/A9 的交集**——"高速 + 遮挡 + 显式迟滞"三者同时处理的工作近乎不存在,这正是本项目 state-transition + open_loop rollout + 免标定视觉 的差异化立论点。
