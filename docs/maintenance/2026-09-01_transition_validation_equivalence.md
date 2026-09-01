---
title: 状态转移内部验证与旧评价口径等价检查
kind: maintenance-record
status: complete
updated: 2026-09-01
scope: GTObserved and OpenLoop validation checkpoint selection
---

# 状态转移验证等价检查

本记录只验证公共 engine adapter 与原 `eval_real_quant.py` / watcher 聚合口径一致，
不用于声明模型效果或真实控制性能。所有 smoke 均使用独立的新 run 目录，没有覆盖历史训练。

## 检查对象

- 数据集：`workspace/data/processed/real/seq_20260819_182253_n15_sam2_robot_mm`
- GT smoke：`workspace/runs/training/gt_transition/exp_20260901_000`
- OpenLoop smoke：`workspace/runs/training/open_loop_transition/exp_20260901_000`
- OpenLoop 外部对照：`workspace/runs/analysis/open_loop_validator_equivalence/run_20260901_000`
- 验证范围：20 个 val 帧上最多产生 19 个 OpenLoop 预测行；不使用 NDI

## 结果

GT 1 epoch 的 engine 全节点均误为 `1.176337 mm`，旧 CLI CSV 聚合为
`1.176450 mm`，差值 `0.000113 mm`，来自 CSV 保留三位小数。

OpenLoop 采用 3 epoch 线性 teacher-forcing 退火（`1.0 -> 0.5 -> 0.0`）：

| epoch | engine `node_mean_mm` | 旧 CLI CSV 聚合 | 排名 |
|---:|---:|---:|---:|
| 1 | 2.086399 | 2.086368 | 3 |
| 2 | 1.624308 | 1.624316 | 1 |
| 3 | 1.776800 | 1.776684 | 2 |

两条路径的 checkpoint 排序完全一致，均选择 epoch 2。逐 tensor 比较确认
`best_eval_model.pt` 与 `model_epoch_0002.pt` 完全相同，`final_model.pt` 与
`model_epoch_0003.pt` 完全相同。

## 结论与边界

GT 和 OpenLoop 均满足指标口径与 checkpoint 选择等价门槛，完整真实训练流水线
可以切换到内部 adapter。切换时不得让 watcher 与 engine 同时写
`best_eval_model.pt`。watcher 源文件先保留，待流水线切换后的测试与短跑通过后再单独审查。
