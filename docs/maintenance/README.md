---
title: SelfSoftRobot 治理历史
kind: history
status: active
updated: 2026-09-10
scope: material migrations, releases, removals, and acceptance decisions
supersedes: []
superseded_by: null
sources:
  - ../overview/status.md
---

# 治理历史

本页只索引难以仅凭普通 commit 理解的重要决定。当前健康、阻塞和删除区由
[`../overview/status.md`](../overview/status.md) 拥有；日常代码变更直接查 Git。

| 日期 | 类型 | 决定/事件 | 证据 |
|---|---|---|---|
| 2026-09-10 | supersede | 论文入口转到遮挡闭环证据准备包；旧初稿保留但不作为本轮论证输入；paper/papers所有权不变 | [论文入口](../paper/README.md)、[新文献核对](../papers/review_partial_observation_20260910/README.md) |
| 2026-09-01 | migration | 67 项历史资产原子迁入统一 workspace，旧入口在主线短跑后移除 | [`2026-09-01_legacy_asset_migration.json`](2026-09-01_legacy_asset_migration.json) |
| 2026-09-01 | validation | 公共 engine 的 GT/OpenLoop validation 与旧评价排序等价 | [`2026-09-01_transition_validation_equivalence.md`](2026-09-01_transition_validation_equivalence.md) |
| 2026-09-01 | release | 发布不可变 10 Hz reference dataset，并完成 formal frozen test/offline fixture | [`2026-09-01_reference_dataset_release.md`](2026-09-01_reference_dataset_release.md) |
| 2026-09-01 | removal | 移除误跟踪字节码缓存和有明确生成入口的旧展示产物 | [`2026-09-01_generated_asset_cleanup.md`](2026-09-01_generated_asset_cleanup.md) |
| 2026-09-01 | decision | `sam2/sam2_src` 作为上游嵌套仓库管理；`docs/ref` 保留硬件资料和参考项目，不按普通文档清理 | [`../../sam2/README.md`](../../sam2/README.md)、[`../ref/README.md`](../ref/README.md) |
| 2026-09-01 | supersede | 动态 `HANDOFF.md` 退出当前事实所有权；原 2026-07-28 内容完整归档，旧路径保留重定向 | [`../HANDOFF.md`](../HANDOFF.md)、[`../archived/HANDOFF_2026-07-28.md`](../archived/HANDOFF_2026-07-28.md) |
| 2026-09-01 | decision | `paper/` 固定为唯一活跃 manuscript；`papers/` 只作文献证据和阶段草稿输入 | [`../paper/README.md`](../paper/README.md)、[`../papers/README.md`](../papers/README.md) |

新增记录只覆盖路径/数据所有权改变、难回退决策、重要事故、正式发布和有意删除；
训练过程与指标继续归对应 run manifest 或实验记录，不复制到本页。
