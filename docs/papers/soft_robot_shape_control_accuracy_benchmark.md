# 软体/连续体机器人形状估计与真实控制精度基准

生成日期：2026-09-03

综述类型：范围综述（scoping review，不声称穷尽全部文献）
目的：为 SelfSoftRobot 的形状预测结果确定可比边界，并为下一阶段真实控制实验设定指标与目标

## 1. 问题与口径

本文回答两个彼此不同的问题：

1. 已有软体/连续体机器人研究的**形状感知或前向形状预测**达到什么精度？
2. 已有研究在**真实机器人闭环执行后**达到什么控制精度？

二者不能混用。当前项目的 `node mean` 和 `endpoint mean` 是记录轨迹上的 forward-model
误差；规划器内部 terminal-to-target 只是离线规划残差。只有动作在真实机器人上执行，并由执行后的
视觉骨架或独立 NDI 测量与目标比较，才能称为控制精度。

为避免伪排名，本综述保留原论文指标，不把像素、角度、Chamfer、mask overlap 和毫米误差强行换算。
长度百分比只在论文给出机器人长度或原文已经报告时列出。

## 2. 检索与筛选

### 2.1 检索范围

检索/核验日期为 2026-09-03。来源包括 IEEE/期刊论文全文、作者版全文、arXiv 全文、Crossref
和 OpenAlex 元数据。使用的概念组合为：

```text
(soft robot OR continuum robot) AND
(shape sensing OR shape estimation OR forward kinematics OR hysteresis) AND
(accuracy OR error OR RMSE)

(soft robot OR continuum robot) AND
(shape control OR trajectory tracking OR visual servoing) AND
(real robot OR experiment) AND (error OR accuracy)
```

纳入条件：给出定量形状估计或真实控制结果；对象为软体/连续体机器人；能够辨认平台、指标和实验类型。
排除条件：只有定性图、只有刚性机器人、无法区分仿真与实机、二手材料无法追溯到原文。

### 2.2 证据等级

- A：同行评审全文中的实机结果；
- B：预印本全文中的实机结果；
- C：仿真结果或只有摘要/元数据可核验的结果。

## 3. 形状感知与前向预测

| 工作 | 对象与输入 | 实验 | 原指标 | 证据 | 与本项目的可比性 |
| --- | --- | --- | --- | --- | --- |
| Shentu et al., MoSS, 2024 [1] | 两段腱驱连续体；单目 RGB 当前帧 | 实机 | mean shape error `0.91 mm`，臂长 `0.36%`；`70 fps` | A（摘要及出版元数据核验） | 是强实时感知结果，但每帧都看见机器人，不解决遮挡后的动作递推 |
| Cho et al., 2024 [2] | 200 mm 腱驱连续体；当前与前一构型 | 实机 | 50 个独立测试：Chamfer `5.7 ± 2.4 mm`，tip `7.6 ± 3.7 mm`；6 条未见轨迹：Chamfer `9 ± 3 mm`，tip `9 ± 4 mm` | B | 同为历史条件 forward shape；输出为表面点云，指标与中心线 node error 不完全相同 |
| SoftNeRF, Shan et al., 2024 [3] | 缆驱软臂；动作到神经表面模型 | 仿真+实机 | 仿真 surface accuracy `0.241 cm`、completeness `0.766 cm`、2 cm 阈值 completion ratio `95.375%`；实机 2D mask MSE `0.0139` | A/C | 实机没有 3D GT；不可把 mask MSE 写成毫米形状精度 |
| Kasaei et al., 2025 [4] | Cosserat 先验 + Shape-NODE | 仿真 | 1–2 段各轴 RMSE `<0.6 mm`；3 段 `0.981–1.169 mm`；4 段最高 `2.161 mm` | C | 数字来自仿真，不是实机视觉形状 GT |
| Chen et al., 2025 [5] | 70 mm 气动手术软臂；压力+运动方向 | 实机 | hysteresis-aware model 相对 conventional model 的 MSE 降低 `84.95%` | B | 没给可直接对照的中心线 node mm；适合证明迟滞输入有效，不适合绝对排名 |
| 本项目 ISHSM v4 | 15 点二维中心线；动作 + 一次锚点/无锚点 | 实机记录轨迹 dev | zero-init `1.7486 mm node / 3.2866 mm endpoint`；single-anchor `1.7510 / 3.2902 mm` | 开发集 | 直接面向稀疏观测 rollout；dev 只有两条且参与选模，尚非独立 test |
| 本项目 Hereditary v2 | 15 点二维中心线；仅动作连续状态 | 实机记录轨迹 dev | continuous `1.7818 mm node / 2.9004 mm endpoint`；cold-restart-40 `1.8331 / 3.0527 mm` | 开发集 | 历史条件 forward 结果；不是控制精度 |

结论：当前帧始终可见时，专用视觉形状感知可达到约 `0.9 mm`。历史条件的实机全形状前向预测中，
公开结果常处于数毫米到约 10 mm，但任务、长度和形状表示差异很大。本项目的 `1.75–1.78 mm`
node error 有竞争力，然而现有两条 dev 轨迹不足以支持 SOTA 声明。

## 4. 真实机器人控制精度

| 工作 | 控制任务与反馈 | 实机结果 | 证据 | 注意事项 |
| --- | --- | --- | --- | --- |
| Almanzor et al., 2023 [6] | 单目视觉闭环全身静态 shape control | normalized mask error 从 `0.0354` 降至 `0.0133`（`63%`）；平均约 10 s | A | 无毫米中心线或 tip GT |
| Schäfke et al., 2024 [7] | GRU-NMPC，5-DoF 气动铰接软臂 | 平均轨迹跟踪误差约 `1.2°` | A | 角度指标；每个控制周期使用实测状态 |
| Kasaei et al., 2025 [4] | Shape-NODE + Control-NODE 闭环 tip tracking | 综合 `2.653 ± 1.737 mm`，open-loop `9.265 ± 1.063 mm`；circle 各轴 `3.11/3.41/3.23 mm`，square `5.04/4.31/4.81 mm`，S `2.12/2.22/1.83 mm`，ellipse `3.16/3.82/2.18 mm` | B | 实机 tip marker；不是全身中心线控制误差 |
| Chen et al., 2025 [5] | whole-body RL，激光轨迹任务 | circle `0.250 mm`，square `0.126 mm` | B | 70 mm 专用手术平台、受限轨迹、预印本；不能视为通用领域常态 |
| Yu et al., 2026 [8] | 双目形状可解释视觉闭环 | regulation 10 s：最大 shape error `4 px`，最大 tip `3 mm`；tracking：最大 shape `<2.5 px`，tip 最大 `<6 mm` | B | 300 mm 三段机器人；shape 与 tip 使用不同指标 |
| Tang et al., 2026 [9] | 多平台自适应控制、未知负载/气流/故障 | tip load `4.8 mm`，distributed load `5.9 mm`，actuator failure `5.4 mm`；动态扰动适应后误差约束在 `5 mm`；shape overlap nominal `98.7%`，多种扰动下 `>92%` | A | tip RMSE 与图像 overlap 均不是 node-mm |
| Xu et al., 2026 [10] | 接触环境安全自适应视觉伺服 | 所有测试最终 image error `<1 pixel`；settling time 约 `3–19 s` | A（accepted author version） | 控制图像特征并约束接触力，不是全身毫米形状控制 |

综合看，跨平台可复用的实机闭环 tip tracking 常见于约 `2–6 mm`；非常小、专用且受限的平台可以报告
亚毫米结果。全身形状控制尚无统一 benchmark，多使用 pixel、mask 或 overlap。

## 5. 对本项目控制阶段的建议

第一阶段应采用保守且可证伪的目标，而不是直接追逐某篇论文的最小数字：

- 真实执行后的 terminal endpoint mean/median `<5 mm`；
- 真实执行后的 whole-body node mean `<3 mm`；
- 同时报 endpoint 与 node 的 mean、median、95th percentile、max；
- 报成功率、settling time、overshoot、动作次数和安全中止次数；
- 视觉骨架是控制/全身评价通道，NDI 只作独立 endpoint 评价，不能输入模型或规划器；
- 分成持续视觉闭环与 sparse/OpenLoop 两条协议，不能用闭环结果替代“只观察一次”的主张；
- 至少保留未参与训练和选模的新轨迹/目标作为 test，并按轨迹而非帧做统计单元。

建议的递进实验是：离线 replay 可达性检查 → mock 执行管线 → 低幅、空旷场景真实 terminal
regulation → 多目标重复试验 → sparse-observation → 障碍/全身约束。第一轮真实控制只验证定位与状态
接续，不同时引入障碍、接触和极限动作。

## 6. 证据边界

- `0.91 mm` MoSS 是“看见当前形状”的感知，不是动作到未来形状的预测。
- Kasaei 的 shape estimation 毫米值来自仿真，而实机数字是 tip tracking。
- SoftNeRF 实机只有 2D mask 指标。
- Chen、Kasaei、Yu 的相关版本为预印本，结论强度低于同行评审实机证据。
- 本项目 dev 同时用于 checkpoint 选择与结构裁决，且只有两条序列；当前结果只用于开发决策。

## 参考文献

1. C. Shentu et al., “MoSS: Monocular Shape Sensing for Continuum Robots,” *IEEE Robotics and Automation Letters*, 9(2), 1524–1531, 2024. DOI: [10.1109/LRA.2023.3346271](https://doi.org/10.1109/LRA.2023.3346271).
2. N. J. Cho et al., “Accounting for Hysteresis in Forward Kinematics of Tendon-Driven Continuum Robots Using Neural Networks,” arXiv:2404.03816, 2024. [arXiv](https://arxiv.org/abs/2404.03816).
3. J. Shan et al., “SoftNeRF: A Self-Modeling Soft Robot Plugin for Various Tasks,” *IROS 2024*. DOI: [10.1109/IROS58592.2024.10801344](https://doi.org/10.1109/IROS58592.2024.10801344).
4. M. Kasaei, F. Alambeigi, and M. Khadem, “A Synergistic Framework for Learning Shape Estimation and Shape-Aware Whole-Body Control Policy for Continuum Robots,” arXiv:2501.03859v4, 2025. [arXiv](https://arxiv.org/abs/2501.03859).
5. Z. Chen et al., “Hysteresis-Aware Neural Network Modeling and Whole-Body Reinforcement Learning Control of Soft Robots,” arXiv:2504.13582v2, 2025. [arXiv](https://arxiv.org/abs/2504.13582).
6. E. Almanzor et al., “Static Shape Control of Soft Continuum Robots Using Deep Visual Inverse Kinematic Models,” *IEEE Transactions on Robotics*, 39(4), 2973–2988, 2023. DOI: [10.1109/TRO.2023.3275375](https://doi.org/10.1109/TRO.2023.3275375).
7. H. Schäfke et al., “Learning-Based Nonlinear Model Predictive Control of Articulated Soft Robots Using Recurrent Neural Networks,” *IEEE Robotics and Automation Letters*, 2024. DOI: [10.1109/LRA.2024.3495579](https://doi.org/10.1109/LRA.2024.3495579).
8. P. Yu, X. Wang, and N. Tan, “Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control,” arXiv:2603.01751v1, 2026. [arXiv](https://arxiv.org/abs/2603.01751).
9. Z. Tang et al., “A General Soft Robotic Controller Inspired by Neuronal Structural and Plastic Synapses That Adapts to Diverse Arms, Tasks, and Perturbations,” *Science Advances*, 12(2), eaea3712, 2026. DOI: [10.1126/sciadv.aea3712](https://doi.org/10.1126/sciadv.aea3712).
10. F. Xu, X. Kang, and H. Wang, “Safety-Aware Adaptive Visual Servoing of Soft Robots through Online Shape-Force Transformation,” *IEEE Transactions on Robotics*, accepted 2026. DOI: [10.1109/TRO.2026.3706564](https://doi.org/10.1109/TRO.2026.3706564).
