---
title: SelfSoftRobot 文档地图
kind: map
status: active
updated: 2026-09-09
scope: canonical documentation navigation and ownership
supersedes: []
superseded_by: null
sources: []
---

# docs/ 导航索引

> docs/ 的唯一地图。按主题找权威文档；“当前/最好/最新”只由状态页或具体 run
> 记录拥有。文档内容需用当前代码、测试、manifest 与 Git 验证。

## 治理角色

| 角色 | 权威来源 | 只负责 |
|---|---|---|
| Constitution | [`../CLAUDE.md`](../CLAUDE.md) + [`standards/`](standards/) | 必须遵守的约束；详细规则链接到 standards |
| Map | 本页 | 结构、所有权和“去哪里找” |
| Status | [`overview/status.md`](overview/status.md) | 当前主线、阻塞、健康和删除区 |
| History | [`maintenance/README.md`](maintenance/README.md) | 重要迁移、替换、删除及验收决定 |

[`HANDOFF.md`](HANDOFF.md) 只保留兼容重定向；2026-07-28 完整快照已归档，
不再作为当前入口。

治理更新规则：结构/所有权变化改本页；当前里程碑、阻塞或删除区改 status；难回退
决定和重要迁移改 maintenance。提交这些文档前运行：

```bash
python scripts/maintenance/check_docs_governance.py
```

## 项目总览 (`overview/`)
| 文档 | 说明 |
|---|---|
| [`overview/project_help.md`](overview/project_help.md) | **核心参考**: CLI 运行入口 + 源码布局 + 模型架构表 + 关键约定(从 1710 行 PROJECT_HELP 精简) |
| [`overview/pipeline.md`](overview/pipeline.md) | 技术管线与模型演进(MSTNF→C-MSTNF→MS-SCNF→state-transition) |
| [`overview/status.md`](overview/status.md) | 项目状态快照: 现在到哪了 + 接下来做什么 |

## 实物数据 (`real_data/`)
| 文档 | 说明 |
|---|---|
| [`real_data/README.md`](real_data/README.md) | 实物文档入口：当前流程、实验记录和旧流程分类 |
| [`real_data/general_6ch_postprocess.md`](real_data/general_6ch_postprocess.md) | **当前通用数据前处理主线** |
| [`real_data/automated_real_pipeline.md`](real_data/automated_real_pipeline.md) | 当前自动前处理、训练和离线验证编排 |
| [`real_data/capture_setup.md`](real_data/capture_setup.md) | 硬件采集系统: 双段硅胶臂 + 6通道 Modbus 比例阀 + RealSense + NDI Aurora |
| [`real_data/deployment.md`](real_data/deployment.md) | **实机部署指南**: 采集→数据前处理→训练→deploy_manifest→工作台闭环;含已知坑与诚实边界 |

## 研究方向 (`directions/`)
| 文档 | 说明 |
|---|---|
| [`directions/directions_overview.md`](directions/directions_overview.md) | 18 个研究方向索引（部分为历史假设） |
| `directions/02_*.md` ~ `18_*.md` | 各方向详述(迟滞/编码/骨架/多视角/sim2real/OpenLoop/控制/路径依赖 IK 等) |

## 设计 (`designs/`)
| 文档 | 说明 |
|---|---|
| [`designs/2026-09-08_partial_observation_hereditary_control.md`](designs/2026-09-08_partial_observation_hereditary_control.md) | **遮挡控制唯一设计**：保留历史的状态校正＋剩余序列修正；已有数据离线原型见[回放验证记录](experiments/partial_observation_replay_validation.md)，实机闭环待验证 |
| [`designs/2026-08-29_hereditary_operator_model.md`](designs/2026-08-29_hereditary_operator_model.md) | **HereditaryOperatorModel v2 设计与实现边界**：PI play + 广义 Maxwell + 局部模态读出、验证协议和文献边界 |
| [`designs/2026-08-31_repository_organization_refactor.md`](designs/2026-08-31_repository_organization_refactor.md) | **仓库整理与重构提案**：统一数据/运行路径、manifest 谱系、训练验证与早停共识、文档治理及分阶段迁移验收 |

## 项目规范 (`standards/`)
| 文档 | 说明 |
|---|---|
| [`standards/repository_layout.md`](standards/repository_layout.md) | **当前生效的仓库与工作区合同**：源码/产物边界、统一 workspace、历史路径只读兼容和迁移验收 |
| [`standards/dataset_split.md`](standards/dataset_split.md) | **数据划分共识**：先划分后构窗、证据等级、时序泄漏与模型可声明差异 |
| [`standards/training_and_evaluation.md`](standards/training_and_evaluation.md) | **训练与选择共识**：验证、早停、checkpoint 语义、模型差异和测试集使用 |
| [`standards/experiment_layout.md`](standards/experiment_layout.md) | **试次归档共识**：run 布局、无覆盖、resume、完成标记和历史运行 |
| [`standards/evidence_language.md`](standards/evidence_language.md) | **证据语言共识**：模型误差、离线规划、Mock 链路和实机控制的表述边界 |

## 治理历史 (`maintenance/`)

| 文档 | 说明 |
|---|---|
| [`maintenance/README.md`](maintenance/README.md) | 重要迁移、发布、验收和有意删除的索引；普通提交仍以 Git 为准 |

## 文献与背景 (`background/`)
| 文档 | 说明 |
|---|---|
| [`background/literature.md`](background/literature.md) | 相关工作综述(NeRF系/自建模/迟滞/视觉控制) + 本项目创新点 |

## 论文与文献

| 入口 | 说明 |
|---|---|
| [`paper/README.md`](paper/README.md) | 唯一活跃 manuscript 树；方法、实验、outline 和正文草稿在这里维护 |
| [`papers/README.md`](papers/README.md) | 文献证据、66 篇单篇笔记、深读材料、原文副本和旧阶段草稿的边界 |

## 实验 (`experiments/`)
| 文档 | 说明 |
|---|---|
| [`experiments/README.md`](experiments/README.md) | 实验文档入口：当前方案、正式记录和历史仿真实验 |
| [`experiments/openloop_sparse_observation_validation_plan.md`](experiments/openloop_sparse_observation_validation_plan.md) | **当前论文实验主方案**：机制层物理记忆与 H–K 可行域，任务层路径依赖 IK/不可见轨迹，应用层不透明通道稀疏观测巡检 |
| [`experiments/real_robot_validation_workbench_todo.md`](experiments/real_robot_validation_workbench_todo.md) | **实机验证界面 TODO**：模型/场景/规划/安全执行/同步评价的通用工作台与任务插件 |
| `experiments/hereditary_v2_training_validation.md` | Hereditary v2 正式训练与验证记录 |
| `experiments/experiment_analysis.md`、`improvement_proposals.md`、`results_evaluation.md` | **历史仿真阶段记录**，仅用于追溯，不作为当前结论入口 |

## 演示 (`presentations/`)
- `presentations/Project_presentation1.md` / `Project_presentation2.md`

## 其他
| 文档 | 说明 |
|---|---|
| [`encoders.md`](encoders.md) | 时序编码器(EMA/Fractional/Gamma/GRU/Transformer/TCN) |
| `superpowers/` | 设计规格与计划(specs/ + plans/, 工具生成) |
| [`ref/README.md`](ref/README.md) | 本机硬件实验资料与外部参考项目边界；内容默认 ignored，不是当前项目事实来源 |

## 归档 (`archived/`)
被合并或取代的旧文档(完整内容在 git 历史 + 新文档里):
- `archived/PROJECT_HELP.md` — 1710 行全量版(已精简到 `overview/project_help.md`)
- `archived/project_status_report.md` — 829 行旧状态报告(已精简到 `overview/status.md`)
- `archived/inspirations.md` + `archived/literature_innovations.md` — 已合并到 `background/literature.md`
- `archived/multiview_depth_supervision_proposal.md` — 已实现的设计提案
- `archived/research/` — 旧 dated 工作文档(06-19 多视角标定路径已弃用; 07-10/07-14 已合并到 `real_data/workflow.md`; 05-16 文献已合并到 `background/literature.md`)
- `archived/directions/` `archived/ode_cmstnf/` `archived/smooth_cmstnf/` `archived/trainers/` — 早期方向/模型

---

**入口推荐**: 新读者 → `overview/status.md`(项目到哪了) → `overview/project_help.md`(怎么跑) → `real_data/workflow.md`(实物主线)。

Hereditary 工作台新入口：[GUI 实验指南](../real_validation/HEREDITARY_GUIDE.md)、[接入与虚拟设备验证](experiments/hereditary_real_validation_integration.md)。
