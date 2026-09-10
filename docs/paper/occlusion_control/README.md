---
title: 部分观测闭环控制论文准备入口
kind: paper-planning
status: active
updated: 2026-09-10
scope: evidence-grounded manuscript preparation, not a completed manuscript
sources:
  - ../../experiments/hereditary_real_validation_integration.md
  - ../../papers/review_partial_observation_20260910/README.md
---

# 部分观测闭环控制：论文准备工作台

主线：**在单视角图像存在遮挡时，用动作历史维持全身形态预测，用可靠可见证据校正内部状态，并在控制周期允许的时间内修订剩余动作。**

这里从代码、运行产物和原论文重新组织，不使用既有论文初稿的论证、贡献和实验结论。代码核对基准为 `96e684f`；截至 2026-09-10，真实数据预测/录制图像回放与虚拟闭环已有证据，真实机器人遮挡闭环仍待验证。**研究目标是遮挡下的闭环控制；已有 Mock 链路不是论文中的实机控制结果。**

## 按写作问题查找

| 要解决的问题 | 文档 |
|---|---|
| 工作到底是什么，哪些词能写，哪些证据还缺 | [01 关键点与主张](01_claims_and_evidence.md) |
| 引言怎样逐段展开，相关工作怎样分组 | [02 故事与引言](02_story_and_introduction.md) |
| 方法公式如何对应当前实现，哪些性质不能声称 | [03 方法与公式](03_method_and_equations.md) |
| 对比、消融、任务、指标与数据划分怎么设计 | [04 实验证据链](04_experimental_program.md) |
| 摘要、图表、结论、展望和成稿顺序 | [05 写作与交付](05_writing_and_outlook.md) |
| 最近调研过哪些文献、哪些最接近、哪些需要重核 | [文献分类入口](../../papers/review_partial_observation_20260910/README.md) |
| 原论文引言每段在干什么 | [8 篇引言逐段拆解](../../papers/review_partial_observation_20260910/introduction_atlas.md) |
| 离线搜索标题/标签、查看文档 | [可搜索 HTML 工作台](../../../workspace/reports/paper_preparation_20260910_000/index.html) |

## 最重要的三个决定

1. **问题主线优先于模型名。** 全身形状、单视角、自建模、历史依赖、遮挡和时延共同限定任务；不能每个都算一个原创贡献。
2. **以 AFT、Krauss、Yu、Chen、Schäfke 和 Zheng 为主要参照。** AFT 已有单 RGB-D、遮挡形状估计和全身闭环实验；Krauss 已有视觉学习的真实开环控制；Yu 已有可解释形状与几何控制。不能把这些能力写成空白。
3. **要证明完整因果链。** 可见证据 → 状态修正 → 后续预测改变 → 动作改变 → 真实形状误差改善。现阶段最缺最后一环的独立实机证据，以及同观测条件的强基线。

建议先读 01 → 文献核心卡片 → 02 → 04，再据 03 写方法。旧稿保留作历史存档，不是本套材料的输入。
