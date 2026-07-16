# 阅读笔记：Shape Sensing Techniques for Continuum Robots in Minimally Invasive Surgery: A Survey

> Chaoyang Shi, Xiongbiao Luo, Peng Qi, Tianliang Li, Shuang Song, Zoran Najdovski, Toshio Fukuda, Hongliang Ren — IEEE Transactions on Biomedical Engineering (Vol. 64, Issue 8), 2017
> 链接: https://doi.org/10.1109/tbme.2016.2622361
> arXiv: 无

## 一句话概括
一篇综述系统性梳理了连续体机器人(以微创手术为背景)的三类 3D 形态感知技术——光纤传感器(FBG)、电磁追踪、术中成像——并明确指出在未知/动态载荷下实现精确的实时全身形态感知仍是开放难题。

## 核心问题 / 动机
连续体机器人凭借结构柔顺性可沿曲折解剖路径到达手术靶点,但其可变形设计使得 3D 术中实时形态感知极为困难。精确的形态感知是闭环控制、路径规划、人机交互与手术安全的前提。作者指出,尽管依赖运动学/力学的大量 model-based 研究已开展,在未知且动态的载荷下,精确形态感知仍是挑战——这恰恰是软体机器人建模面临的同源问题。

## 方法
据摘要(未获取全文),综述按三大类技术组织:
- **光纤传感器(FBG)**: 通过应变测量重建弯曲形态;
- **电磁追踪(EM tracking)**: 借助电磁场与传感线圈获取位姿;
- **术中成像模态(intraoperative imaging)**: 基于影像/视觉的形态重建。

并讨论各类技术的局限性与新技术的展望。摘要未给出单一方法,综述本身为分类与对比性质。

## 主要结果
据摘要:该综述汇总并比较了三类形态重建方法,404 次引用量级,是该领域的标准引用。(注:仅为综述,无单一实验定量结果;具体定量对比需查全文,本次仅获取摘要。)

## 与本项目的关系
- **关联主题**: continuum-robot shape sensing, FBG strain-to-shape, electromagnetic tracking (NDI/Aurora), imaging-based reconstruction, survey/contrast baseline
- **可借鉴 / 差异**: 与本项目(神经场自建模 + 迟滞状态转移 + 免标定 2D 视觉管线)的对比点非常清晰——综述明确把 FBG / EM / 成像这三家当作"在未知动态载荷下难以精确感知全身形态"的局限家族,而我们走的是数据驱动视觉自建模路线,不依赖先验力学参数、不做离散点插值,可整体重建形态并隐式捕获粘弹性迟滞。引用价值在于它正是"现有感知家族只能给离散点、不建完整连续形态"的权威出处,可作为我们动机论述的标准对立面引用。
- **支撑哪句论述**: 主要支撑 **A4**(现有方法 FBG 光纤测应变推弯曲 / 电磁追踪 / 缆绳长度编码只能得离散点,不能完整形态建模——综述三大类正是 FBG + EM + 成像);同时佐证 **A1**(现状以尖端/离散点跟踪为常态)与 **A10**(视觉/成像 + 数据驱动自建模是合理替代方向之一)。

## 验证状态
- 经 web 抓取确认 (https://doi.org/10.1109/tbme.2016.2622361) 于 2026-07-16;DOI 正确解析到该论文,标题与作者(含 scout note 遗漏的 Toshio Fukuda)一致,摘要已读到。仅获取摘要,方法/结果为据摘要归纳,全文未读。
