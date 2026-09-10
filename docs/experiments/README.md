---
title: 实验文档索引
kind: map
status: active
updated: 2026-09-10
scope: experiment plans, records, and historical simulation notes
---

# 实验文档索引

这里按“是否仍指导当前工作”分类。训练产物保持独立目录，过程性说明合并到对应模型的总记录。

## 当前使用

- [2026-09-11 实机反馈整改](hereditary_field_followup_20260911.md)：有限末端余量、复用配准、自动初始化与图像/模型证据区分。

- [原生5/10 Hz全平台部署训练](../real_data/validation_deployment_hardware_and_training_20260910.md)：补齐同平台序列处理、独立tmux、验证选择和自动导出。

- [`hereditary_real_validation_integration.md`](hereditary_real_validation_integration.md)：Analytic B 工作台接入、模型部署包、六腔映射、画笔目标与虚拟设备验证。
- [`partial_observation_replay_validation.md`](partial_observation_replay_validation.md)：固定图像遮挡、运动选段、状态更新和全后缀优化的上实机前回放验证。

- `openloop_sparse_observation_validation_plan.md`：当前论文实验方案。
- `real_robot_validation_workbench_todo.md`：实机验证工作台规划。
- `hereditary_v2_training_validation.md`：Hereditary v2 的正式训练与验证记录。
- `ishsm_optimization_validation_record.md`：ISHSM v1--v5 的模型演进、预注册裁决、TDD 证据、
  Hereditary 公平对照和机器结果入口；这是 ISHSM 实验历史的唯一入口。
- `hereditary_v2_interpretability_assessment_20260903.md`：Hereditary v2 的公式、训练参数涌现、
  物理贡献分解与可解释性边界审计。

## 历史记录

- `experiment_analysis.md`
- `results_evaluation.md`
- `improvement_proposals.md`

以上三份主要记录 2026 年 4–5 月的仿真探索（exp1–exp7）。其中的数值和结论保留作背景，
不应当作当前模型、数据或最佳结果的入口。新的实验请使用独立 run 目录，并把正式结论链接到
具体 dataset/run manifest。
