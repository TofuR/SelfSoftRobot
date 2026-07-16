# 阅读笔记：NeuralFeels with neural fields: Visuotactile perception for in-hand manipulation

> Sudharshan Suresh, Haozhi Qi, Tingfan Wu, Taosha Fan, Luis Pineda, Mike Lambeta, Jitendra Malik, Mrinal Kalakrishnan, Roberto Calandra, Michael Kaess, Joseph Ortiz, Mustafa Mukadam — Science Robotics, 2024
> 链接: https://www.science.org/doi/10.1126/scirobotics.adl0628
> arXiv: 2312.13469

## 一句话概括
NeuralFeels 在在线学习的神经场（neural field）中编码未知物体的几何，并同时通过位姿图优化跟踪物体位姿；融合视觉+触觉，在严重遮挡下显著优于纯视觉，是"神经场 + 状态估计"用于被遮挡场景的范例。

## 核心问题 / 动机
- 灵巧手内操作（in-hand manipulation）需要持续估计**未知物体的形状与位姿**。
- 现状（status quo）：几乎只用视觉，且仅能跟踪**预先已知**的物体；操作过程中手指遮挡物体是常态，纯视觉系统在遮挡下失效。
- 动机：视觉在可见时提供全局约束，触觉在遮挡/接触时提供局部约束；二者融合才能让感知在遮挡下持续工作。

## 方法
- **NeuralFeels**：在线学习一个神经场来表示物体几何（无 CAD 先验也能增量重建），并联合求解一个**位姿图优化（pose-graph）问题**来跟踪物体 6-DoF 位姿——即"神经场建图 + 位姿跟踪"的紧耦合 SLAM 式结构（据摘要与编辑摘要）。
- 多指手 + 基于视觉的触觉传感器（visuotactile），固定相机提供视觉，本体感知驱动的策略旋转物体收集触觉信号。
- 释放数据集 **FeelSight**（70 个实验）作为该领域的基准。

## 主要结果
- 重建 F-score 达 **81%**。
- 平均位姿漂移 **4.7 mm**；当有已知 CAD 模型时降到 **2.3 mm**。
- 在**重度视觉遮挡**下，相较纯视觉方法跟踪性能提升最高达 **94%**。
- 结论：触觉至少能精化（refine）、最多能消歧（disambiguate）视觉估计。

## 与本项目的关系
- **关联主题**: neural-field 在线建图 / 视觉遮挡下的状态估计 / 多模态融合 / 触觉-视觉 / SLAM 式跟踪
- **可借鉴 / 差异**:
  - 共同范式：用**神经场作为可更新的隐式形状/状态表示**，并叠加一个**状态估计/跟踪**模块——这正是本项目用神经场建模软臂形态 + rollout 状态转移的同构思路；NeuralFeels 的"在线增量更新 + 位姿图跟踪"可作为迟滞隐状态/latent 在线更新的参照。
  - 关键差异：NeuralFeels 建模**刚体未知物体**（形状不变、只估位姿），而本项目建模**软体自体形态**（形状本身随驱动+迟滞变化）；其在线更新针对几何重建，我们的 latent 针对粘弹性迟滞历史。NeuralFeels 用触觉补遮挡，本项目走免标定 2D 管线、依赖视觉骨架+时间记忆而非触觉。
- **支撑哪句论述**: **A9**（大部分方法需持续观测消除误差，遮挡/不可见时难工作）——NeuralFeels 正是论证"纯视觉在遮挡下失效、需额外模态/隐状态持续跟踪"的直接证据；同时为 A7（数据驱动隐式捕获迟滞/状态）提供"神经场+状态估计"可工作的方法论旁证。

## 验证状态
- 经 web 抓取确认 (https://arxiv.org/abs/2312.13469 与 https://www.science.org/doi/10.1126/scirobotics.adl0628) 于 2026-07-17；标题/作者/期刊（Science Robotics 2024）/摘要均一致。
- caveat: 仅读到摘要 + 编辑摘要 + 引言开头，未获取全文；方法/结果细节据摘要表述，未核对正文实验配置（传感器型号、具体 baseline 列表等）。
