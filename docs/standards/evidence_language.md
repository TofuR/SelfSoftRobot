---
title: 实验与控制证据表述规范
kind: standard
status: active
updated: 2026-08-31
scope: reports, experiment records, paper claims, deployment evidence
supersedes: []
superseded_by: null
sources:
  - training_and_evaluation.md
---

# 实验与控制证据表述规范

| 已有证据 | 可以说 | 不能直接说 |
|---|---|---|
| 单元/fixture 测试 | 接口与合同通过 | 真实数据质量或硬件效果已验证 |
| Mock 相机/阀/NDI ACK | 软件和通信链路打通 | 真实机器人控制成功 |
| recorded GT vs prediction | 前向模型误差 | 闭环控制误差 |
| 两个 recorded shapes | 目标形态差异/位移 | planner 已到达目标 |
| offline terminal residual | 模型内规划残差 | 实机达到目标 |
| 真实执行 + 执行后视觉观测 | 相应协议下的控制结果 | 未覆盖工况的普适安全性 |

报告结论至少绑定 `dataset_id`、`run_id`、checkpoint 选择语义、metric、单位和证据等级。Derived equations、物理/算子迁移和本项目工程选择应分别标注；文献支持不能被写成项目代码由该论文直接推导。
