# 阅读笔记：Multi-core Optical Fibers with Bragg Gratings as Shape Sensor for Flexible Medical Instruments

> Fouzia Khan, Alper Denasi, David Barrera, Javier Madrigal, Salvador Sales, Sarthak Misra — IEEE Sensors Journal, 2019 (in press / accepted 版本)
> 链接: https://ris.utwente.nl/ws/files/105398009/khan19_sensors_inpress.pdf
> arXiv: 无
> DOI: 10.1109/JSEN.2019.2905010

> 注: 候选元数据将期刊误记为 "Sensors (MDPI)"; 经抓取 PDF 确认实际期刊为 **IEEE Sensors Journal**。标题也比候选多一个副标题 "for Flexible Medical Instruments"。

## 一句话概括
利用刻写在多芯光纤(multi-core fiber)中的光纤布拉格光栅(FBG)测量应变, 推算曲率与挠率, 再用 Frenet-Serret 方程积分重建柔性医疗器械的三维形态, 在带 4 根多芯光纤的导管上验证, 最大重建误差 1.05 mm。

## 核心问题 / 动机
医疗介入(硬膜外给药、结肠镜、活检、心脏手术)中, 柔性器械在病人体内的空间位姿至关重要。现有手段(触觉反馈主观、超声中器械不可见、荧光透视有辐射)都有缺陷。光学传感是替代方案, 但如何用有限个离散传感点还原器械的连续完整三维形态是核心难题。

## 方法
据摘要及引言:
- 将多芯光纤中的 FBG 传感器数据先转换为应变(strain)测量。
- 由应变计算光纤的曲率(curvature)与挠率(torsion)。
- 用 Frenet-Serret 方程结合曲率/挠率沿光纤积分, 重建器械在三维欧氏空间中的形态。
- 验证平台: 一根传感化的导管, 沿圆周布置 **4 根含 FBG 的多芯光纤**; 将导管置于 8 种构型, 与 ground truth 对比重建结果。

> 关键结构性事实(据摘要): 形态是"由离散光栅位置测得的应变 → 曲率/挠率 → 积分"间接得到的, 即离散点到连续曲线的积分重建。这正是本项目 A4 所指的"离散点"范式。

## 主要结果
据摘要:
- 8 种构型下, 与 ground truth 相比, 最大重建误差 **1.05 mm**。
- 结论: 用多芯光纤 FBG 做柔性医疗器械的形态传感是可行的。

## 与本项目的关系
- **关联主题**: shape sensing, FBG fiber optics, curvature integration, discrete-point reconstruction, soft/flexible instrument
- **可借鉴 / 差异**: 本文是 FBG 多芯光纤形态传感的代表性/被广泛引用工作, 正是本项目 A4 点名的"FBG 光纤测应变推弯曲"路线的典型实例。它清楚展示了该范式的两个特征: (1) 只能在光栅刻写位置得到离散应变测量, 形态靠 Frenet-Serret 积分外推; (2) 依赖物理建模(曲率-挠率-积分链), 而非端到端数据驱动。我们的神经场自建模路线与之形成对比: 不依赖离散点积分, 而是从视觉观测端到端学习连续形态场, 且可隐式捕获粘弹性迟滞。本文可作为"传统离散点形态传感"的对照基线被引用。
- **支撑哪句论述**: **A4** — 现有方法(FBG 光纤测应变推弯曲)只能得离散点, 不能完整形态建模。本文的"应变→曲率→Frenet-Serret 积分"流程恰好例证了"离散点到连续形态"的间接重建与积分依赖, 说明该路线本质上是离散测量加物理积分, 而非直接的完整形态建模。

## 验证状态
- 经 web 抓取确认 (https://ris.utwente.nl/ws/files/105398009/khan19_sensors_inpress.pdf) 于 2026-07-16; 抓取到完整摘要、引言开头及作者/DOI 元数据。
- 注: PDF 为 IEEE Sensors Journal 的 accepted/in-press 预印本(版权页标注 1558-1748 (c) 2018 IEEE, DOI 10.1109/JSEN.2019.2905010), 候选提供的期刊 "Sensors (MDPI)" 有误, 已据正文更正为 IEEE Sensors Journal。方法/结果部分基于摘要; 正文细节未全文逐字读取。
