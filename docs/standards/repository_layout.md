---
title: SelfSoftRobot 仓库与运行工作区规范
kind: standard
status: active
updated: 2026-09-01
scope: source tree, data artifacts, experiment runs, compatibility paths
supersedes: []
superseded_by: null
sources:
  - ../designs/2026-08-31_repository_organization_refactor.md
---

# 仓库与运行工作区规范

本规范是整理期间和后续新增代码必须遵守的最小合同。完整动机、目标结构和分阶段迁移见[仓库整理与重构设计](../designs/2026-08-31_repository_organization_refactor.md)。

## 1. 两类根目录

源码仓库只保存代码、配置模板、测试和长期文档。数据、模型权重、日志、评价图片和缓存统一写入可配置的 `workspace_root`：

```text
workspace/
├── data/{raw,intermediate,processed,fixtures}/
├── runs/{training,validation,analysis}/
├── models/pretrained/
├── reports/
├── cache/
└── registry/
```

默认根为仓库内被 git 忽略的 `workspace/`。服务器可在 `config/paths.local.toml` 或环境变量中映射到仓库外的大容量磁盘。业务代码不能根据当前工作目录自行猜测数据根。

项目继续使用已有的 `config/`，不为了目录名统一再并行创建 `configs/`。`config/paths.example.toml` 是可提交模板，`config/paths.local.toml` 只保存本机覆盖且不得提交。

## 2. 数据生命周期

| 层级 | 路径 | 可变性 | 内容 |
|---|---|---|---|
| raw | `data/raw/<domain>/<sequence_id>` | 不可覆盖 | 原始帧、动作 ACK、时间戳、NDI 原始记录 |
| intermediate | `data/intermediate/<domain>/<sequence_id>/<recipe_id>` | 可重建 | crop、mask、骨架、QC |
| processed | `data/processed/<domain>/<dataset_id>` | 发布后不可覆盖 | manifest 与 train/val/test 数据 |
| fixture | `data/fixtures/<fixture_id>` | 小型、固定 | 测试或 GUI 演示数据，不充当正式数据集 |

路径不是身份。正式引用使用 `sequence_id`、`dataset_id`、`run_id` 和 manifest/hash；目录名只负责定位。

## 3. 运行产物

- 正式训练：`runs/training/<study_id>/<run_id>`；
- 真实验证：`runs/validation/<run_id>`；
- 分析与可视化：`runs/analysis/<analysis_id>/<run_id>`；
- 一次运行只写一个新目录，目标存在即拒绝，显式且合同匹配的 resume 除外；
- 配置、命令、日志、checkpoint、评价和完成标记必须留在同一 run 下；
- 不得用“最新目录”作为正式实验的隐式输入。

## 4. 历史路径状态

以下历史资产已于 2026-09-01 原子迁入 workspace，并在主线短跑通过后移除旧入口：

| 历史位置 | 目标角色 |
|---|---|
| `real_capture/data/raw` | raw real sequence |
| `real_capture/data/derived`、`sam2/masks` | intermediate |
| `data/real_seq` | processed real dataset |
| `train_log` | training run |
| `real_validation/runs` | validation run |
| `output` | analysis run |

这些路径不再是可用的数据入口。路径解析器仍保留 legacy fallback，用于外部 workspace 或尚未导入的机器本地资产；本仓库内的正式命令、配置和新产物必须使用 workspace/canonical 路径。迁移清单与回滚映射见 [`../maintenance/2026-09-01_legacy_asset_migration.json`](../maintenance/2026-09-01_legacy_asset_migration.json)。

## 5. 禁止事项

- 新增硬编码的 `real_capture/data`、`data/real_seq`、`train_log`、`real_validation/runs` 或 `output` 写入；
- 把大文件、checkpoint、mask、缓存或运行日志提交进源码树；
- 覆盖 raw、processed release 或历史 run；
- 在未校验文件数、大小、hash、引用和回滚路径前移动或删除历史资产；
- 让 `real_capture`、`real_validation` 或某个模型脚本拥有独立的正式数据副本。

提交前运行 `python scripts/maintenance/check_legacy_path_literals.py`。守门只扫描
Python 可执行字符串和 shell 非注释行；现存兼容读取使用非递增 baseline，迁移一个
consumer 就同步降低对应计数，禁止用扩大 baseline 的方式接纳新业务写入。

## 6. 第三方源码与本地参考资产

- `sam2/sam2_src/` 是带独立 `.git`、remote 和许可证的上游 SAM2 checkout；
  外层仓库不拥有、不逐文件跟踪它，项目集成边界见 [`../../sam2/README.md`](../../sam2/README.md)；
- `docs/ref/` 保存硬件实验资料和参考项目，内容默认 ignored，边界见
  [`../ref/README.md`](../ref/README.md)；
- 第三方或参考代码不能被当作当前项目实现、配置或科学结论的权威来源；
- 更新嵌套仓库要记录 upstream remote、commit、license 和本地 patch 状态；
- 不因目录体积大而删除硬件资料或参考仓库，清理需逐项确认用途与恢复方式。

## 7. 变更验收

涉及路径或产物的提交至少验证：

1. 默认 workspace 与显式外部 workspace 的解析；
2. 从仓库外 cwd 调用时路径仍正确；
3. legacy read 开关和缺失路径报错；
4. 新目标存在时拒绝覆盖；
5. manifest 不写本机绝对项目路径；
6. 相关 focused tests 与 `git diff --check`。
