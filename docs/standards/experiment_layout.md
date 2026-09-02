---
title: 实验试次与产物归档规范
kind: standard
status: active
updated: 2026-08-31
scope: training runs, validation runs, analysis runs, completion markers
supersedes: []
superseded_by: null
sources:
  - repository_layout.md
  - training_and_evaluation.md
---

# 实验试次与产物归档规范

## 1. 标准布局

```text
workspace/runs/training/<study_id>/<run_id>/
├── run_manifest.json
├── config.json
├── commands.sh
├── status.txt
├── stages/<phase>/
│   ├── logs/
│   └── checkpoints/
├── evaluations/<phase>/<selection>/
├── diagnostics/
├── artifacts.json
└── COMPLETE
```

不是每个 exploratory run 都立即具备全部文件，但 formal run 必须在完成前补齐 schema 声明的产物。`environment.txt` 和 `source.patch` 可按实验风险选用，不作为轻量 v2 的强制字段。阶段目录可以保留模型现有内部结构，run 根语义不能改变。

## 2. 创建与恢复

- 新 run 使用日期加原子递增编号；目标存在即拒绝。
- 显式 `RUN_DIR` 必须为空，或走有合同校验的 resume；不能把新训练追加进任意旧目录。
- resume 必须核对 dataset manifest、模型合同、阶段、优化器状态和关键超参数。
- 并行试验使用不同 run 目录和 tmux session；共享只读 dataset，不共享可写 checkpoint。

## 3. 状态

允许状态为 `planned`、`running`、`complete`、`failed`。只有预期训练、选择评价和必备产物均成功后才能写 `COMPLETE`；进程退出、生成 `best_model.pt` 或 mock ACK 都不等于试次完成。

## 4. 直接训练与正式流水线

- `scripts/training/train_transition.py` 适合短小的单阶段探索；未指定目录时也必须创建独立 workspace run。
- `scripts/real/train_real_transition.sh` 是 GT→验证选择→OpenLoop→最终评价的正式主线，所有阶段归入同一 trial。
- 将要联合分析的试验应使用相同 manifest 版本和评价输出格式；每个 run 仍保留自己的真实命令与参数。

新正式试次使用轻量 `run_manifest` v2：记录 dataset manifest URI、创建时 Git commit/dirty 状态、实际命令文件、resolved config、seed、逐阶段 validation 选择、预期产物和 `COMPLETE`。不重复计算 checkpoint/NPZ 的逐文件 SHA；dataset 自身的不可变文件记录由 dataset manifest 负责。旧 run manifest v1 继续只读兼容。

## 5. 历史运行

`train_log` 只读保留。registry 可以登记其路径、配置和完成标记，但不得重命名旧 run 或补写成看似原生的新结构。若要重评历史 checkpoint，评价结果写入新的 workspace analysis/validation run，并显式引用源 checkpoint。
