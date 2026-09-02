---
title: 本地硬件与参考项目资产说明
kind: reference
status: active
updated: 2026-09-01
scope: ignored hardware experiment files and external reference repositories
supersedes: []
superseded_by: null
sources: []
---

# docs/ref 本地参考资产

本目录保留硬件实验实际使用过的文件和外部参考项目，默认被主仓库 Git 忽略；只有
本说明文件进入主仓库。这里不是当前代码、配置、数据或项目状态的权威来源。

| 目录 | 角色 | 管理边界 |
|---|---|---|
| `Main UI-plc/` | 早期采集/PLC/RealSense/NDI 程序、设备配置和硬件手册 | 用于核对硬件协议与历史实验，不直接作为当前采集入口 |
| `TwinCAT Project8/` | TwinCAT PLC 工程与许可证相关本机文件 | 硬件工程参考；修改和部署需在 TwinCAT 环境单独验证 |
| `pre_14_click/` | 早期 Arduino 控制程序 | 历史硬件参考，不是当前控制主线 |
| `SelfSimRobot/` | 早期仿真/刚臂参考实现 | 只作算法来源参考 |
| `visual-selfmodeling/` | BoyuanChen 的外部参考项目 | 独立嵌套 Git 仓库，不归入 SelfSoftRobot 源码所有权 |

2026-09-01 盘点时 `visual-selfmodeling/` 工作树干净，remote 为
`https://github.com/BoyuanChen/visual-selfmodeling.git`，HEAD 为
`67b6df654f12c6ad8cc3778eabe217bbe9ef915c`。更新或引用其代码时必须保留上游
来源和许可证边界；项目文档中的研究结论不能仅以本目录内容作为当前证据。

本目录可能包含本机配置、IDE 工程和实验记录，不做批量格式化、重命名或删除。
清理前必须按具体子目录确认硬件用途、唯一副本和恢复方式。
