---
title: 论文准备与正文入口
kind: paper-index
status: active
updated: 2026-09-10
scope: canonical paper preparation and manuscript ownership
sources:
  - icra2027/README.md
  - occlusion_control/README.md
  - ../papers/review_partial_observation_20260910/README.md
---

# 论文准备与正文

当前写作主线为**遮挡下的全身形状闭环控制：部分视觉观测、历史状态校正与动作补偿**。

当前正文：[ICRA 2027中文初稿](icra2027/README.md)，提供融合相关工作的引言、方法、平台与数据采集、分层实验、总结与展望，以及Word阅读版。正式实验数据使用占位符，预期结果与已有证据分开。

从[论文准备工作台](occlusion_control/README.md)开始：关键点与证据 → 引言逻辑 → 方法公式 → 对比与消融实验 → 写作、结论和展望。原论文的分类、核对范围和逐段引言分析位于[文献证据工作台](../papers/review_partial_observation_20260910/README.md)。

本轮依据代码 `96e684f`、运行产物和原论文重新组织，不采用既有论文初稿的逻辑或结论。已有回放和虚拟设备验证不能替代真实遮挡闭环证据；主张状态见[关键点与证据](occlusion_control/01_claims_and_evidence.md)。

目录根既有 `01_*.md`–`06_*.md`、`icra_draft.md`、`manuscript_partial_observation_v6_zh.md/.docx`保留为历史阶段材料，不作为本轮写作输入。`occlusion_control/`维护研究准备，`icra2027/`维护当前初稿、方法和实验表格；文献笔记由 `docs/papers/` 维护。
