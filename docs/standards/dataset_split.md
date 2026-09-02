---
title: 数据划分与可比较性规范
kind: standard
status: active
updated: 2026-08-31
scope: processed datasets, temporal windows, train/val/test evidence
supersedes: []
superseded_by: null
sources:
  - ../designs/2026-08-31_repository_organization_refactor.md
---

# 数据划分与可比较性规范

本规范只统一科学比较所需的最低合同，不要求不同模型使用相同的数据加载器、窗口长度或损失函数。

## 1. 强制规则

1. 先按独立采集单元划分，再构造窗口或 episode。由同一帧衍生的历史窗口不得跨 split 重用。
2. 默认分组键是 `sequence_id`；同一次连续采集不能被伪装成多个独立样本组。
3. `val` 用于超参数、早停和 checkpoint 选择；`test` 只用于方案冻结后的最终报告。
4. 标准化统计只能由 `train` 估计，再原样应用到 `val/test`。
5. dataset manifest 必须记录 split 策略、分组键、seed、embargo、证据等级，以及每个文件的 URI、帧数和 hash。
6. 修改划分、窗口、清洗或坐标合同会产生新的 `dataset_id`；不得静默替换原数据集内容。

## 2. 证据等级

| 等级 | 允许的划分 | 可以支持的结论 |
|---|---|---|
| `smoke` | 小样本、可无独立 val | 代码能运行、张量和产物合同正确 |
| `within_sequence` | 单序列按时间连续划分，必要时设 embargo | 同一采集条件下的时段泛化 |
| `cross_sequence` | train/val/test 使用独立采集序列或实验组 | 跨轨迹、跨循环或跨工况泛化 |

报告必须写明等级。`within_sequence` 不能被表述为跨序列泛化；无独立 `test` 时不能声称最终无偏性能。

## 3. 时序模型的额外约束

- `H` 是输入历史长度，`K_train` 是训练 rollout 长度，`K_eval` 是评价视野；三者分别记录，不能用一个 `window` 混称。
- OpenLoop 的一个评价窗口只能有协议允许的观测锚点，其后预测必须按声明的策略自馈。
- episode 跨越 split 边界时必须丢弃，不能从 train 取历史、在 val 计目标。
- 若使用相邻帧 embargo，至少覆盖模型所需的最大历史和目标跨度；设为 0 时必须在 manifest 中显式记录。

## 4. 模型可声明的差异

模型可以有不同的采样频率、窗口长度、阶段、损失、batch 组织和增强策略，但必须在 run 配置中声明，并引用同一个不可变 dataset manifest。为某模型改变数据内容时，应发布新数据集，而不是在训练脚本中隐藏过滤逻辑。

## 5. 当前迁移状态

新预处理数据集使用 schema v2 manifest 和 `artifact://` 文件引用。历史 `data/real_seq` 清单仍由只读 inventory 兼容；组合数据集和旧清洗脚本尚未全部迁移到本规范，使用它们时必须把实际划分写入试次配置。
