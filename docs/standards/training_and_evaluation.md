---
title: 训练、验证、早停与 checkpoint 规范
kind: standard
status: active
updated: 2026-09-01
scope: model training, validation selection, early stopping, final evaluation
supersedes: []
superseded_by: null
sources:
  - ../designs/2026-08-31_repository_organization_refactor.md
  - dataset_split.md
---

# 训练、验证、早停与 checkpoint 规范

共同协议统一的是“结果如何解释和比较”，不是把 GTObserved、OpenLoop、Hereditary 和渲染模型强行训练成同一种形式。

## 1. 三层配置

| 层 | 所有者 | 内容 |
|---|---|---|
| 项目协议 | 本规范与 run schema | 数据角色、选择集、不可覆盖、必备产物、测试集使用 |
| 模型声明 | `TrainingSpec/PhaseSpec` 或模型设计 | phase、监督模式、损失、默认预算、是否允许早停 |
| 具体试次 | run 内 resolved config | dataset、seed、CLI 覆盖、设备、实际阶段和命令 |

模型文档只写相对本规范的差异。一次试次的真实参数以 run 内 resolved config 和命令记录为准。

## 2. 正式训练的最低字段

- `study_id`、`run_id`、`run_kind`；
- dataset manifest URI 与 hash；
- Git commit、dirty 状态及 dirty patch；
- seed、batch、优化器、学习率/调度器、最大 epoch；
- 每个 phase 的输入、损失、训练模块和预算；
- 验证频率、选择 metric、方向（`min/max`）和 dataset role；
- early-stop 是否启用、patience、`min_delta`、warmup；
- resume 来源、完成状态和预期产物。

缺字段的运行可以作为 exploratory/smoke 使用，不能冒充可复现的 formal 试次。

## 3. checkpoint 语义

文件名不是指标。正式引用必须同时写 checkpoint URI、选择 metric、dataset role 和选择方向。

当前迁移期的已知语义：

- `best_model.pt` 通常是训练 loss 最优，不能称为“验证最优”；
- `best_eval_model.pt` 只有在验证 watcher 实际创建时才表示验证选择结果；
- `current_model.pt`/训练状态用于恢复，不用于报告最好性能；
- `final_model.pt` 表示预算结束时参数，不自动等于最佳参数。

完整真实 GT/OpenLoop 流水线优先使用实际存在的 `best_eval_model.pt`，否则回退到 `best_model.pt`，并在试次记录中保留真实选择结果。Hereditary 历史 `_001`–`_006` 的结论来自训练-loss-best checkpoint，不能追溯改称验证最优。

## 4. 验证与早停

早停不是全项目固定常数，必须逐 phase 声明：

- 有独立 `val` 且 metric 稳定时，可以 early stop；必须保留最佳 checkpoint 与停止原因。
- 无 `val` 的训练只能按预先预算或数值故障停止，不能用训练 loss 伪装泛化早停。
- 多阶段模型分别声明选择规则；后一阶段的 metric 不能倒用于选择前一阶段，除非协议预先定义联合选择。
- scheduler patience 与 early-stop patience 是两个概念，必须分别记录。
- 长训练是否提前终止应查看曲线、验证 metric 和 checkpoint 选择语义，不能只因名义 epoch 很大或训练 loss 短期平台就停止。

允许的模型差异包括：Hereditary 的固定预算/状态诊断，OpenLoop 的 teacher-forcing 退火与 rollout 指标，GTObserved 的逐步观测上界，以及渲染模型的分阶段监督。

## 5. 最终评价与证据隔离

1. 冻结模型、checkpoint 和评价脚本后才使用 `test`。
2. 调参后重复查看 test 会使其退化为 val；此时必须另留独立 test。
3. NDI 是独立 endpoint 评价，不进入模型、规划器或在线状态估计。
4. recorded-GT 对比是前向模型误差；离线 planner residual 不是实机控制成功。
5. 控制成功必须有真实执行及执行后的观测骨架到目标比较。

## 6. 当前实现与待迁移项

`UnifiedTrainer` 已把未指定目录的直接训练写入 workspace；完整真实流水线已统一试次根，并保存阶段、评价、配置和命令。

`PhaseSpec` 现可选声明 `ValidationSpec`，统一记录 val role、选择 metric/方向、验证频率、`min_delta`、warmup、early-stop patience、scheduler metric 和 best 恢复策略。公共 `SelectionState` 已实现按“验证次数”计数的选择/早停状态，并支持 OpenLoop teacher-forcing 退火门控；未声明 validation 的历史模型行为不变。

公共 engine 现支持显式 `validation_data_dirs + validation_adapters`：声明 validation 的 phase 缺少任一输入会拒绝训练；adapter 返回合同 metric 后，engine 写 `validation_metrics.jsonl`、生成 `best_eval_model.pt`、以声明 metric 驱动 scheduler、按验证次数早停、保留 `final_model.pt`，并按配置恢复 validation-best 供后续 phase 使用。未声明 validation 的历史训练仍按训练 loss 生成 `best_model.pt`。

当前各模型尚未默认绑定自己的 adapter，完整真实 GT/OpenLoop 流水线仍使用验证 watcher。必须先为主线接入 rollout validator，并对比 watcher 与 engine 的选择结果后，才能删除 watcher。
