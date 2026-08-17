# 阅读笔记：Compensating for Unmodeled Forces using Neural Networks in Soft Manipulator Planning

> Scott Chow, Gina Olson, Geoffrey A. Hollinger — IEEE ICRA 2021
> 链接: https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/icra21_home_papercept_ras.papercept.net_www_conferences_conferences_icra21_submissions_4110_fi.pdf
> arXiv: 无

## 一句话概括
把气动多段软臂的简化准静态无负载模型当作先验, 再用一个神经网络去补偿那些被忽略的力(摩擦、负载), 得到一个够快、够准、可用于采样式运动规划(RRT*)的前向模型。

## 核心问题 / 动机
- 气动软臂的完整形状随驱动而变, 多段串联时下游段对基段产生负载, 内部摩擦也不可忽略。
- 纯运动学/PCC 模型快但不准(假设常曲率, 外力下被违反); 纯物理模型(Cosserat/FEA)准但太慢(模拟一秒需数分钟), 无法支撑需要上千次前向查询的采样式规划器。
- 论文给出一个直接动机: 在充气→放气一个循环后, 由于摩擦+负载未被建模, 软臂回不到初始伸展状态(原文 Fig.1 的迟滞回线)。

## 方法
- 混合/残差学习: 取单段无负载的准静态闭式解作为基线先验, 再训练一个神经网络去学习"真实观测 − 准静态预测"的残差, 即未建模力的影响。
- 神经网络以准静态模型的输出为输入, 估计残差修正量; 整体保持快于实时, 避免成为规划器的瓶颈。
- 把该前向模型嵌入 RRT* 采样式规划器, 用残差修正后的模拟器验证采样候选, 生成可执行轨迹。
- 灵感来自计算机图形学中"线性最小二乘 + 神经网络"实时预测可形变物体受力的方法。

## 主要结果
- 相比准静态无负载模型, 神经网络模型把末端位置平均误差降低 62%。
- 预测速度仍快于实时, 满足采样式规划器的查询频率要求。
- 在硬件执行上, 用该模型生成的 RRT* 规划比用纯准静态模拟器生成的规划更可能在真实硬件上可行, 且能更可靠地到达目标并避障。

## 与本项目的关系
- **关联主题**: residual/hybrid learning, forward model for motion planning, unmodeled forces & hysteresis, soft pneumatic arm
- **可借鉴 / 差异**: 这篇是典型的"先验物理模型 + 数据驱动残差修正"两段式思路, 直接对应我们论述里的"传统建模需先验物理参数、损伤后失效"(A6), 以及"承认物理模型不完整、用数据补"。与我们项目的差异在于: 他们仍依赖一个显式的准静态先验模型(强先验), 且只关注末端位置预测, 不做全身形态估计; 我们走的是完全端到端、免标定的神经场自建模(A7), 不预设曲率/物理参数, 还能隐式捕获粘弹性迟滞。该工作也印证了软臂存在摩擦/负载引起的迟滞(Fig.1 充放气不回零), 是 A7 中"隐式捕获迟滞"动机的一个外部佐证。此外它把前向模型嵌入规划器的做法, 与本项目 open-loop 部署/形状控制的目标形成方法学呼应。
- **支撑哪句论述**: 主要支撑 A6(传统建模需先验物理参数, 损伤/变形后预设模型失效——他们正是因准静态先验不准才要加残差网络)与 A7(数据驱动自建模可避免先验依赖、隐式捕获迟滞——Fig.1 的充放气迟滞直接佐证迟滞存在)。同时其只做末端位置、不做全身形态的设定, 也间接反衬 A1(多数工作只做尖端)。

## 验证状态
- 经 web 抓取确认 (https://research.engr.oregonstate.edu/rdml/.../icra21_..._4110_fi.pdf) 于 2026-07-17; 读到完整摘要及 Background/Method 多个章节文本。注: 候选给出的标题 "Compensating for Unmodeled Forces Using Neural Networks in Soft Manipulators" 与论文真实标题略有出入, 真实标题为 "Compensating for Unmodeled Forces using Neural Networks in Soft Manipulator **Planning**"; 作者为 Scott Chow, Gina Olson, Geoffrey A. Hollinger(候选 "S. Chow, et al." 大致一致, 隶属 Oregon State / CMU)。
