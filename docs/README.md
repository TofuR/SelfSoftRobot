# docs/ 导航索引

> docs/ 的地图。按主题找文档; 每条一行说明。最后更新 2026-07-20。

## 接手指南(给 AI agent)

| 文档 | 说明 |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | **新接手先读**: 5 分钟心智模型 + 当前真实状态(exp_20260714_7/8) + 别破坏的不变量 + 怎么跑最新控制/规划 + 诚实边界 + 术语表 |

## 项目总览 (`overview/`)
| 文档 | 说明 |
|---|---|
| [`overview/project_help.md`](overview/project_help.md) | **核心参考**: CLI 运行入口 + 源码布局 + 模型架构表 + 关键约定(从 1710 行 PROJECT_HELP 精简) |
| [`overview/pipeline.md`](overview/pipeline.md) | 技术管线与模型演进(MSTNF→C-MSTNF→MS-SCNF→state-transition) |
| [`overview/status.md`](overview/status.md) | 项目状态快照: 现在到哪了 + 接下来做什么 |

## 实物数据 (`real_data/`) — 当前主线
| 文档 | 说明 |
|---|---|
| [`real_data/workflow.md`](real_data/workflow.md) | **免标定 2D 骨架→状态转移→NDI 验证** 完整流程(分割/修复/SAM2/骨架化/npz/clean/训练/评估) |
| [`real_data/capture_setup.md`](real_data/capture_setup.md) | 硬件采集系统: 双段硅胶臂 + 6通道 Modbus 比例阀 + RealSense + NDI Aurora |

## 研究方向 (`directions/`)
| 文档 | 说明 |
|---|---|
| [`directions/directions_overview.md`](directions/directions_overview.md) | 17 个研究方向索引 |
| `directions/02_*.md` ~ `17_*.md` | 各方向详述(迟滞/编码/骨架/多视角/sim2real/OpenLoop/控制/路径依赖 IK 等) |

## 文献与背景 (`background/`)
| 文档 | 说明 |
|---|---|
| [`background/literature.md`](background/literature.md) | 相关工作综述(NeRF系/自建模/迟滞/视觉控制) + 本项目创新点 |

## 论文笔记 (`papers/`)
| 子目录/文件 | 说明 |
|---|---|
| `papers/notes/` | 10 篇短笔记(3DGS/flow_matching/hysteresis/koopman/jacobian/pinn/shape_node/ssl/tang/yu) |
| `papers/understanding/` | 深读: hu2025(FBV-SM)/chen2022/shan2024(SoftNeRF) + brainstorm + depth_supervision_innovation |
| `papers/*.pdf` `*.jpg` | 论文原文 |

## 实验 (`experiments/`)
| 文档 | 说明 |
|---|---|
| [`experiments/openloop_sparse_observation_validation_plan.md`](experiments/openloop_sparse_observation_validation_plan.md) | **当前论文实验主方案**：机制层物理记忆与 H–K 可行域，任务层路径依赖 IK/不可见轨迹，应用层不透明通道稀疏观测巡检 |
| [`experiments/real_robot_validation_workbench_todo.md`](experiments/real_robot_validation_workbench_todo.md) | **实机验证界面 TODO**：模型/场景/规划/安全执行/同步评价的通用工作台与任务插件 |
| `experiments/experiment_analysis.md` | 全部实验结果分析 |
| `experiments/improvement_proposals.md` | 改进方案记录 |
| `experiments/results_evaluation.md` | 评估结果 |

## 演示 (`presentations/`)
- `presentations/Project_presentation1.md` / `Project_presentation2.md`

## 其他
| 文档 | 说明 |
|---|---|
| [`encoders.md`](encoders.md) | 时序编码器(EMA/Fractional/Gamma/GRU/Transformer/TCN) |
| `superpowers/` | 设计规格与计划(specs/ + plans/, 工具生成) |
| `ref/` | 外部参考(SelfSimRobot 旧刚臂代码 / TwinCAT Project8 / Main UI-plc / visual-selfmodeling) |

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
