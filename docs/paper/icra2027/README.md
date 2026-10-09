---
title: ICRA 2027 自建模论文与实验
kind: paper-index
status: draft
updated: 2026-09-12
scope: modeling experiment implementation and manuscript drafts
sources:
  - manuscript_zh.md
  - writing_notes.md
  - ../occlusion_control/README.md
---

# ICRA 2027 中文初稿

当前研究主线：**软体机器人全身形态自建模：历史记忆、几何表示与跨序列评估**。

建模实验的实现、方法对照和运行说明以[建模实验协议](modeling_experiments.md)为准。现有中文正文与Word保留2026-09-10初稿内容，下一轮正文修订应依据建模实验结果调整论证与表格。

| 文件 | 用途 |
|---|---|
| [5 Hz正式实验结果](../../../workspace/reports/modeling_5hz_622_20260912_000/report.html) | 70次训练与测试的主对照、消融、显著性和收敛分析；[图表及复核资料](../../../workspace/reports/modeling_5hz_622_20260912_000/README.md) |
| [本次5 Hz单次6:2:2正式实验](single_holdout_5hz_20260912.md) | 全部七条序列、验证筛选、四GPU、五种子和70次正式训练记录 |
| [建模实验与运行协议](modeling_experiments.md) | 21种模型、数据划分、训练、真实mask评价、多种子统计与论文实验层次 |
| [基准默认配置](modeling_benchmark_config.json) | 可直接交给统一训练和批量运行入口 |
| [相关方法与官方源码](modeling_baselines_sources.md) | 近期论文结构、三个官方仓库、许可与适配范围 |
| [模型来源清单](modeling_sources.json) | 固定commit、来源快照与接入记录 |
| [实现验证记录](modeling_validation.md) | 实际执行范围、测试结果与待开展的正式实验 |
| [中文正文](manuscript_zh.md) | 摘要、融合相关工作的引言、方法、平台与采集、六层实验、总结与展望、14条参考文献；唯一编辑源 |
| [Word 阅读与批注版](manuscript_zh.docx) | 由正文导出，公式为Word数学对象；适合批注，修改后应同步回Markdown |
| [论证与证据说明](writing_notes.md) | 引言段落衔接、文献借鉴边界、实验因果关系、已核参数与待补证据 |
| [本轮引用核对记录](source_audit.json) | 14条引用对应的本地快照、哈希与原核对范围；不代表本轮重新精读全部实验 |
| [Word导出脚本](export_word.py) | 保留18组公式编号、中文样式与表格排版；需要Pandoc和python-docx |

本稿是内容完整的工作初稿，尚未压缩至会议篇幅或套用ICRA 2027最终模板。实验协议包含待实现的强基线和待开展的实机验证；`【待填：…】`是数值/配置占位，“预期结果与判定”是假设，不是实验结果。已有回放和虚拟设备数值不进入正式控制结果表。

正文按“历史是否重要 → 模型能否预测 → 局部观测是否有用 → 实际控制是否改善 → 遮挡边界 → 时间条件”组织实验。先填写平台、单位与数据划分，再补各实验的独立结果，最后据实际效应修改摘要与结论。

Word可在仓库根目录重新导出：

```bash
python docs/paper/icra2027/export_word.py --overwrite
```

该命令会替换阅读副本；重新导出前先将Word内独有批注或修改另存。不带`--overwrite`时拒绝覆盖，也可用`--out <新路径.docx>`另存。脚本仅为Word转换调整公式对齐和编号，不改变Markdown源及数学含义。原始研究准备材料仍见[准备包](../occlusion_control/README.md)与[文献工作台](../../papers/review_partial_observation_20260910/README.md)。
