---
title: 文献证据与阶段草稿区
kind: literature-index
status: active
updated: 2026-09-01
scope: literature evidence, source copies, deep-reading notes, and dated draft inputs
supersedes: []
superseded_by: null
sources:
  - ../paper/README.md
  - ../background/literature.md
---

# docs/papers 文献与草稿输入

本目录不是活跃论文正文。唯一活跃 manuscript 树是 [`../paper/`](../paper/README.md)。

| 内容 | 角色 | 使用规则 |
|---|---|---|
| `notes/` | 单篇文献证据笔记 | 引用具体结果前仍需核对原文和笔记的证据等级 |
| `understanding/` | 深读与研究发散 | 作为分析输入，不直接拥有项目当前结论 |
| `literature_*.md`、dated survey/map | 阶段性综述与检索记录 | 长期综述由 [`../background/literature.md`](../background/literature.md) 汇总 |
| `*_draft.md`、`*_blueprint.md` | 旧阶段草稿 | 只有被 `docs/paper/` 明确引用的部分才进入活跃论文 |
| `*.pdf`、`*.jpg` | 本地原文副本 | 默认 ignored；版权和引用信息以原出版物为准 |

新增论文正文、方法、实验表格和 claim 修改应进入 `docs/paper/`；新增单篇阅读证据
进入 `notes/`。不要再在本目录根创建另一套“当前论文”。
