---
title: 可重建生成资产清理记录
kind: maintenance
status: active
updated: 2026-09-01
scope: Phase 6 low-risk cleanup
---

# 可重建生成资产清理记录

本轮只处理已确认可重建的生成物，不处理数据集、checkpoint、历史 run、
`.claude/worktrees`、`sam2/sam2_src` 源码或 `docs/ref` 内容。

| 类别 | 删除范围 | 删除前规模 | 恢复方式 |
|---|---|---:|---|
| Python 字节码缓存 | 工作树内、`.claude/worktrees` 外的 `__pycache__/*.pyc` 与 `*.pyo` | 47 个目录、385 个文件、2,834,492 bytes；其中 22 个被 Git 错误跟踪 | 重新导入或运行对应 Python 模块/测试 |
| 旧展示产物 | `verification_result.png`、`tests/prediction_3d_comparison.gif`、`tests/prediction_comparison.gif`、`tests/vis_seq_final.gif` | 4 个文件、4,776,510 bytes | 分别运行 `scripts/visualization/verify_simulation_3d.py`、`tests/visualize_prediction_3d.py`、`tests/visualize_prediction.py`、`tests/visualize_seq_prediction.py`，并提供脚本要求的历史输入/checkpoint |

删除前后均运行 P1 registry、正式流水线、workspace index、reference release 和
offline fixture 的 33 个聚焦测试；删除后测试使用 `PYTHONDONTWRITEBYTECODE=1`，
避免把缓存重新写回源码树。`.gitignore` 已覆盖后续生成的字节码。

`tests/prediction_comparison_v2.gif` 未找到明确生成入口，虽然它未被 Git 跟踪且无
代码/文档引用，本轮仍保留。展示文件清理不包含正式评价目录中的图表和叠图。
