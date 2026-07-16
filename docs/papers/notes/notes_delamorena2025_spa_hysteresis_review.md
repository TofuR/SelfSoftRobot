# 阅读笔记：Hysteresis Modeling of Soft Pneumatic Actuators: An Experimental Review

> Jesús de la Morena, Francisco Ramos, Andrés S. Vázquez — MDPI Actuators, 2025, 14(7), 321
> 链接: https://www.mdpi.com/2076-0825/14/7/321
> arXiv: 无（MDPI 开放获取期刊论文，DOI: 10.3390/act14070321）

## 一句话概括
针对软体气动执行器（viscoelastic 材料的粘弹性迟滞），系统比较了 Preisach、(Classic/Generalized) Prandtl–Ishlinskii、Maxwell-Slip、(Classic/Asymmetric) Bouc-Wen 等主流**现象学迟滞模型**在同一根水凝胶气动弯曲致动器实测数据上的拟合精度、参数紧凑性与计算效率，结论是 **Generalized Prandtl–Ishlinskii (GPI) 综合最优**（GOF > 0.96，10 个算子即够），而**带概率密度函数的 Preisach 最省参数**（仅 5 参数即可刻画非对称迟滞环）。

## 核心问题 / 动机
迟滞是软体粘弹性致动器中普遍存在的强非线性现象：同一压力输入沿不同加载历史到达不同弯曲位形，严重制约其在机器人中的开环精度与闭环控制。物理模型虽能揭示材料机理但过于复杂、难以实时用；现象学模型（输入-输出映射 + 实验辨识）更实用，且可解析求逆用于前馈补偿。文章要做的是：在**同一根真实致动器、同一组实测迟滞回线**上，公平对比各类现象学模型的精度/复杂度/计算成本，给出"该选哪个模型"的可操作结论。

## 方法
（据摘要 + 正文抓取到的章节）
- **被测对象**：一根水凝胶（CANESHA）+ TPU Filaflex 82A 增强外层的气动弯曲致动器，采集其压力-弯曲角的迟滞回线作为统一基准数据。
- **对比的模型族**（每族含经典与扩展变体）：
  - **Preisach (PR)**：带概率密度函数 (Gaussian / Cauchy) 形式，算子数 10–50 可调。
  - **Prandtl–Ishlinskii**：经典对称 CPI + 广义非对称 GPI（通过 envelope 函数扩展 play 算子）。
  - **Maxwell-Slip (MS)**：elasto-slide 算子。
  - **Bouc-Wen**：经典 CBW + 非对称 ABW。
- **评估维度**：拟合优度 GOF（R² 类指标）、参数个数（紧凑性）、单次前向仿真耗时（ms，用于评估实时控制可行性）、物理一致性（能否再现加载曲线与非对称环方向）。

## 主要结果
（据摘要 + 正文 results/conclusions 章节抓取）
- **CPI、CBW 因对称性假设被排除**（无法刻画本致动器的非对称迟滞环）；**Maxwell-Slip 表现最差**（elasto-slide 算子产生的环方向与实测相反）。
- **Preisach（概率密度函数形式）**：仅 5 个参数即可复现非对称环（参数最省）；但需 ≥50 个算子才能压低离散化误差使 GOF > 0.95，**唯一一个仿真耗时 > 10 ms** 的模型，可能影响实时控制；且不再现初始加载曲线。
- **Generalized Prandtl–Ishlinskii (GPI)**：**综合最优**。10 个算子即 GOF > 0.96（多项式 envelope ~0.97、tanh envelope ~0.974），仿真耗时 4–8 ms，解析可逆，适合前馈补偿；唯一小缺点是初始值估计略偏。
- **结论建议**：精度/物理一致性/复杂度/计算效率折衷下，**GPI 是软体气动致动器迟滞建模的首选**；若极致求省参数则选概率密度 Preisach。

## 与本项目的关系
- **关联主题**: hysteresis modeling; soft pneumatic actuator; phenomenological operator models (Preisach / Prandtl-Ishlinskii / Bouc-Wen); feedforward compensation; viscoelastic nonlinearity
- **可借鉴 / 差异**:
  - **可借鉴**：(1) 它权威地确立了"迟滞是软体气动致动器中**被单独、显式建模**的公认现象"这一前提，可作为 A7/A8 论述的权威支撑文献——传统范式是**为迟滞单独建立算子模型并解析求逆做前馈补偿**，而不是让形状模型隐式吸收它。(2) 其 GPI/Preisach 的精度-参数-耗时三维评估框架，可作为我们讨论"隐式 vs 显式迟滞建模"权衡时的对照基线。(3) 其使用的 pressure-bending 迟滞回线采集范式，与我们 real-data 管线中加载-卸载迟滞回线（exp5b 对照）的可视化口径一致。
  - **差异/我们如何更优**：本文**只建模单根致动器输入-输出（压力→弯曲角）的标量迟滞**，不涉及全身形态、不做形状感知、无视觉；而我们用**可学习隐变量 z 的状态转移网络**在免标定 2D 视觉骨架上**隐式捕获**同一粘弹性迟滞（加载-卸载回线），同时给出全身形态估计。即：他们显式建迟滞、隐式丢形态；我们显式建形态、隐式收迟滞——两条互补路线，可作为我们论点"端到端数据驱动自建模可隐式捕获迟滞，避免单独建迟滞算子的复杂度"的直接反衬。
- **支撑哪句论述**:
  - **A8**（自建模工作形态好但没考虑迟滞/高速）——本文证明迟滞在 SPA 文献里是一个**独立且公认**的建模子问题，主流做法是专门搭 Bouc-Wen/Preisach/PI 模型，正好反衬"许多自建模工作未把迟滞纳入"这一空白。
  - **A7**（数据驱动端到端联合训练隐式捕获粘弹性迟滞）——本文代表"显式现象学算子"路线，我们可借此说明：可学习隐变量 z 的状态转移网络能在**不显式建迟滞算子**的前提下隐式捕获同一效应，是更简洁的替代。
  - **A6**（传统建模需先验物理参数）——本文物理模型 vs 现象学模型的对比，旁证依赖先验材料参数的物理路线"复杂且不实用"，间接支持数据驱动路线。

## 验证状态
- 经 web 抓取确认 (https://www.mdpi.com/2076-0825/14/7/321 + https://doi.org/10.3390/act14070321) 于 2026-07-17；标题、作者（Jesús de la Morena, Francisco Ramos, Andrés S. Vázquez）、期刊卷期（Actuators 2025, 14(7), 321）、DOI、出版日期（2025-06-27）均由 MDPI 官方页 + Semantic Scholar + Zenodo 数据集记录 (doi.org/10.5281/zenodo.15584329) 三处独立交叉验证一致。
- 抓取范围：完整摘要 + 方法论章节 + Preisach/PI/GPI 拟合结果 + 结论章节的定量对比（GOF、参数数、ms 级耗时），但未下载完整 PDF 全文，Maxwell-Slip 与 Bouc-Wen 子节的逐项数字未逐一核验（标记为"据摘要/结论"）。
- 注：scout 描述基本准确——确为 2025 年系统对比 SPA 迟滞模型的综述/实验评论；唯标题中"Review"更准确译为"实验性评论/对比综述"而非纯文献综述（它含一手实测对比）。
