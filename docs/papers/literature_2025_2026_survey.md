# 2025-2026 软体机器人自建模与形状估计文献全景

> 调研时间：2026-06-10
> 范围：软体机器人视觉自建模、形状估计、迟滞建模、学习控制
> 目的：支撑方向12（科学问题识别）的文献空白分析

---

## 一、视觉自建模（Neural Fields / NeRF / 3DGS）

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 1 | Neural Jacobian Fields: Learning Intrinsic Mappings of Arbitrary Robot Morphologies | Li et al. (MIT CSAIL) | Nature 2025 | 10.1038/s41586-025-09170-0 | 密集 Jacobian 场，多视角 RGB-D 自监督，适用刚/软/混合机器人 |
| 2 | Self-Modeling Robots by Photographing | Hu, Yu, Tan (中山大学) | IJRR 2025 | arXiv:2503.05398 | 3D Gaussian Splatting + 神经骨骼聚类，高质量纹理自建模 |
| 3 | Egocentric Visual Self-Modeling for Robot Manipulators | Hu et al. | Nature npj Robotics 2025 | 10.1038/s44182-025-00031-6 | 第一视角自建模，无需外部相机 |
| 4 | Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control | Peng Yu, Wang, Tan (中山大学) | arXiv 2026 | arXiv:2603.01751 | Bézier 曲线参数化 + Neural ODE，可解释形状控制 |
| 5 | Robust Visual Embodiment: Noise-Robust, Segmentation-Aware Robot Self-Modeling | — | arXiv 2025 | arXiv:2510.03677 | 噪声鲁棒的分割感知自建模 |
| 6 | High-DOF Dynamic Neural Fields for Robot Self-Modeling and Motion Planning | Sitzmann et al. | NeurIPS Workshop 2023/2024 | — | 高自由度动态神经密度场，单相机 2D 自建模 |
| 7 | SoftNeRF: A Self-Modeling Soft Robot Plugin for Various Tasks | Shan, Li, Feng, Wang (上海交大) | IROS 2024 | — | NeRF 作为软体机器人通用模块 |

## 二、连续体机器人形状估计

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 8 | Continuum Robot Modeling with Action Conditioned Flow Matching | — | arXiv 2026 | arXiv:2605.09216 | Flow Matching 点云生成，线驱动连续体 3D 形状预测 |
| 9 | Synergistic Shape Estimation and Control of Continuum Robots using Augmented Neural ODEs | — | arXiv 2025 | arXiv:2501.03859 | Shape-NODE (Cosserat 先验) + Control-NODE (MPC) |
| 10 | MoSS: Monocular Shape Sensing of Continuum Robots Using a Single RGB Image | Shentu et al. | IEEE RA-L 2024 | — | 单目 RGB 实时形状感知，70fps，0.91mm 误差 |
| 11 | Spatiotemporal Neural Network for Shape Estimation of Continuum Robots Under External Loading | — | arXiv 2025 | arXiv:2510.22339 | RNN 时序 + 空间编码 + 注意力融合，负载下形状估计 |
| 12 | AFT: Appearance-Based Feature Tracking for Markerless Real-Time Shape Reconstruction of Continuum Robots | — | arXiv 2025 | arXiv:2511.18215 | 无标记、无需训练，利用表面纹理做形状重建，2.6% 末端误差 |
| 13 | Learning Whole-Body Shape Control of Soft Robotic Arms in Unknown Situations | Tang, Wang, Rus, Laschi | ICRA 2026 Poster | — | CNN + 图像形状模型 + 在线策略优化 |

## 三、迟滞与动力学建模

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 14 | Hysteresis-Aware Neural Network for Soft Robot Proprioception | — | arXiv 2025 | arXiv:2504.13582 | 迟滞感知网络，方向依赖性达 3.4% 臂长 |
| 15 | Hysteresis Compensation of Soft Pneumatic Actuators using Temporal Convolutional Networks | — | arXiv 2024 | arXiv:2402.11319 | TCN 补偿气动迟滞 |
| 16 | Learning-Based Nonlinear Model Predictive Control for Soft Robots with RNNs | — | arXiv 2024 | arXiv:2411.05616 | RNN-NMPC，实时非线性模型预测控制 |
| 17 | Cycle-Consistency Dual LSTM for Soft Robot Hysteresis | — | 2026 | — | 双 LSTM 循环一致性解决迟滞一对多映射 |

## 四、学习控制与泛化

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 18 | A General Soft Robotic Controller Inspired by Neuronal Structural and Plastic Synapses | Tang, Tian, Xin, Wang, Rus, Laschi | Science Advances 2026 | 10.1126/sciadv.adr0767 | 突触启发通用控制器，44-55% 误差降低 |
| 19 | Koopman Embedding for Cross-Configuration Control of Soft Robots | Zhang et al. | Nature Communications 2026 | — | Koopman 线性嵌入，75x 迁移样本缩减，33 构型泛化 |
| 20 | SOFTMAP: Sim-to-Real Forward Modeling of Tendon-Actuated Soft Finger Manipulators | — | 2026 | — | ARAP 拓扑对齐 + 仿真预训练 + 残差修正，比 DeepSoRo 好 40.7% |
| 21 | Soft Robotic Sim2Real via Conditional Flow Matching | — | Advanced Intelligent Systems 2026 | 10.1002/aisy.202500690 | 条件 Flow Matching 实现 sim-to-real 迁移 |

## 五、物理信息与材料建模

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 22 | Physics-Informed Neural Networks as Surrogate Models for Soft Robot Control | — | arXiv 2025 | arXiv:2502.01916 | PINN 代理模型，467x 加速，47 Hz MPC |
| 23 | Learned Residual Physics for Sim-to-Real Transfer of Soft Robots | — (ETH Zurich) | arXiv 2024 | arXiv:2402.01086 | 解析模型 + 学习残差修正 |

## 六、本体感知与传感器

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 24 | Self-Supervised Learning for Soft Robot Proprioception via Capacitance Masked Autoencoder | Hu, Dong, Giorgio-Serchi, Yang | IEEE TNNLS 2025 | PMID:40982515 | CMAE 自监督，1/20 标注量达到全监督效果 |
| 25 | 3D-Printed Soft Continuum Robot with Integrated Sensing for Shape Reconstruction | Goh, Yu et al. | npj Flexible Electronics 2026 | — | 导电高分子复合传感器 + Conformer，6.3mm RMSE |
| 26 | Zero-Shot Deformation Reconstruction with Cage-Based 3D Gaussian Modeling | — | 2026 | — | 触觉传感器 + 笼形变形 + 3DGS，零样本迁移 |
| 27 | Latent Proprioception: Anchoring Morphological Representations (ProSoRos) | — | 2025 | — | 多模态 VAE 统一运动/力/形状，单内部相机 |

## 七、变形体建模与操控

| # | 论文 | 作者 | 发表 | DOI/arXiv | 一句话 |
|---|------|------|------|-----------|--------|
| 28 | SoMA: Real-to-Sim Neural Simulator for Soft-Body Manipulation | — | 2026 | — | 统一 Gaussian Splat 表示机器人+物体+环境 |
| 29 | INR-DOM: Implicit Neural Representation for Deformable Object Manipulation | — | RSS 2025 | — | SDF 预训练 + RL 微调，遮挡鲁棒的变形体操控 |
| 30 | DGS-LRM: Deformable Gaussian Reconstruction Model | — | arXiv 2026 | arXiv:2506.09997 | 可变形 3DGS 重建模型 |

---

## 文献空白总结（支撑方向12）

| 空白 | 已有工作的覆盖 | 缺失 |
|------|--------------|------|
| 粘弹性记忆容量 | #14,15,17 拟合迟滞行为 | 无人量化"形状编码多少历史"的信息论极限 |
| 迟滞下 IK 可逆性 | #9,18 假设正向模型是函数 | 无人分析什么条件使逆映射有唯一解 |
| 视觉材料发现 | #22,23 从力学数据辨识材料 | 无人从纯视觉观测推断材料属性 |
| 时序编码 + 神经场 | #6,7 神经场自建模（无时序） | 无人结合动作历史编码与神经场 3D 建模 |
| 2D→3D 信息论极限 | #10 单目形状感知 | 无人形式化分析恢复性的充分条件 |
