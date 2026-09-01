---
title: 10 Hz 真实状态转移参考数据集发布记录
kind: maintenance
status: active
updated: 2026-09-01
scope: P1 canonical dataset and frozen test
supersedes: []
superseded_by: null
sources:
  - ../designs/2026-08-31_repository_organization_refactor.md
  - ../standards/dataset_split.md
---

# 10 Hz 真实状态转移参考数据集发布记录

canonical dataset：`real_transition_reference_10hz_v1`。发布目录为
`workspace/data/processed/real/real_transition_reference_10hz_v1/`，目标已使用
非覆盖原子发布；原 processed NPZ 未移动、未修改。

| role | source sequence | source role | frames | purpose |
|---|---|---:|---:|---|
| train | `seq_20260819_182519` | train | 2459 | 参数拟合 |
| val | `seq_20260819_182519` | val | 614 | checkpoint 选择与早停 |
| test | `seq_20260819_182253` | train | 911 | frozen final only |

三个 split 均为 10 Hz、`robot_planar_mm_v1`、mm、15 个 base-to-tip 节点，
raw action 为 6 维，模型动作视图固定为 `[0, 1, 3, 5]`。test 序列只激活
其中前两个独立动作通道，因此它是独立序列上的动作子域测试。train/val 仍来自
同一连续采集，manifest 保守标为 `within_sequence`；不能据此宣称完整跨工况泛化。

发布时只对三份小型 NPZ 和两份轻量 raw registration manifest 计算合同要求的
SHA-256；没有扫描或重算 raw 图像逐文件 hash。raw manifest 只记录采集 metadata、
帧数和 canonical URI。

验收结果：

- dataset manifest schema v2 通过；training-ready 4/4；
- train/val/test selector 均通过 manifest hash 校验；
- shape 分别为 `(2459,3,15)`、`(614,3,15)`、`(911,3,15)`，action 均为 6 维；
- manifest 不含 `/Data5/` 本机绝对路径；
- test policy 为 `frozen_final_only`，后续训练和 selection 不得读取 test；
- workspace registry 已刷新，可按 dataset ID 查询。

## 首个正式试次验收

正式试次：
`workspace/runs/training/real_pipeline/real_transition_reference_10hz_v1/trial_20260901_000/`。
它从干净提交 `9608dc7` 启动，使用 GT 1 epoch、OpenLoop 2 epoch，并在每个
epoch 以 val 的 `validation.node_mean_mm` 选择 `best_eval_model.pt`。GT 选择
epoch 1（1.445 mm），OpenLoop 选择 epoch 2（3.354 mm）。

独立 frozen test 共 911 帧，最终结果如下；这些数值是模型预测相对数据集记录
骨架的前向模型误差，不是实机控制成功率：

| stage | rollout | node mean | tip mean | drift |
|---|---|---:|---:|---:|
| GTObserved | 每步输入记录骨架 | 1.359 mm | 1.689 mm | 不适用 |
| OpenLoop | 每窗口一个记录锚点，随后自回馈 | 3.697 mm | 6.530 mm | mean 6.14x / final-k 5.59x |

试次已落盘 `COMPLETE`、run manifest v2、两阶段 `best_eval_model.pt`、OpenLoop
deploy manifest、test 定量/叠图和 `artifacts.json`；offline fixture 使用 test
第 39 帧、40 步历史，状态为 `passed`。deploy manifest 的相机参考帧、分割统计和
`k_safe_table` 仍为空，因此本验收只证明离线加载与推理链路成立，不授权真机执行。

重建命令由 dataset manifest 的 `recipe.commands` 保存。发布目录存在时工具拒绝
覆盖；若本次发布本身需要撤销，应先确认没有 run 引用该 dataset，再删除整个新
release 和两份新增 raw sidecar，原 source NPZ 始终保留。
