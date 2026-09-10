---
title: SelfSoftRobot 项目状态快照
kind: overview
status: active
updated: 2026-09-10
scope: active research line, engineering migration, evidence boundaries
supersedes: []
superseded_by: null
sources:
  - ../designs/2026-08-31_repository_organization_refactor.md
  - ../experiments/hereditary_v2_training_validation.md
---

# 项目状态快照

2026-09-10：已启动原生5/10 Hz全平台HOV2.2部署训练，tmux `hov_full_native_20260910`；数据处理已全部完成；按用户要求停止旧任务后，在新study `hov_full_native_20260910_001` 从头训练两个频率各300 epoch，关闭早停，复用旧study已核对的数据。训练/导出是否完成以study内 `status_5hz.json`、`status_10hz.json` 为准；详见[部署与训练说明](../real_data/validation_deployment_hardware_and_training_20260910.md)。[腔道接线图册](../real_data/chamber_mapping_reference_20260910.md)提供原始实拍与六路压力。

> “现在到哪了 + 接下来做什么”。本页是唯一当前状态入口；具体参数和数值以 dataset/run manifest 与实验记录为准。
> 文献详见 [`../background/literature.md`](../background/literature.md); 模型架构详见 [`project_help.md`](project_help.md); 研究方向详见 [`../directions/`](../directions/)。

## 当前研究主线：遮挡下的部分观测闭环控制

2026-09-10：围绕单RGB、平面全身形状、历史状态校正、剩余动作补偿和控制期限，建立[新论文准备包](../paper/occlusion_control/README.md)。不采用旧初稿叙事。文献已整理为[7类索引](../papers/review_partial_observation_20260910/README.md)，21篇核对指定原文章节，8篇逐段拆解引言；AFT等作为直接相关工作，不将已有能力写成空白。

当前证据支持模型/回放诊断和虚拟执行链路；正式论文仍需独立预测、同观测强状态基线及真实机器人遮挡闭环。下一步实验按[六层证据链](../paper/occlusion_control/04_experimental_program.md)推进。下列OpenLoop/早期NeRF路线保留为历史背景，不再单独定义当前论文主线。

## 0. 2026-09-09 部分观测回放与反馈加速

- 已在 `feat/partial-observation-replay` 开展固定图像遮挡的上实机前验证：运动窗口选择、原始命令/图像配对、hereditary 非推进读出、状态校正、B 全剩余后缀优化与可视化回放。
- 两条输出分别为“假设执行修订动作并注入录制图像”的流程诊断，以及始终使用原采集动作的预测对照。前者不提供新动作的真实响应证据。
- 已实现向量化图像前端、批量 B、显式导数 B 和预计算 A。80 帧窗口上默认显式 B 两次单线程完整反馈计算 P95 为 48.1/58.5 ms、最大 54.6/139.2 ms；复测有一次超过 100 ms，两次均低于 200 ms。A 更快但仅接受 16/79 次后缀修订。四线程和繁忙 GPU 探测未带来进一步收益。
- 当前实现、计时边界及数值分支敏感性见[回放验证记录](../experiments/partial_observation_replay_validation.md)。尾部抖动、相机/通信延迟、异步执行整合和实机控制验证仍待处理；离线计算结果不等于实机整周期保证。

### 工作台接入

- 2026-09-10：反馈改为独立快照、按模型周期截止整体提交或丢弃，后台任务与图像保存队列有容量上限；新增逐步时延/完整序列修订审计及初始规划参数弹窗。最终虚拟 GUI 40 步中 39 次按期提交、1 次超时丢弃；初始规划 274.2 ms，实际发令间隔 P95/最大 116.3/121.0 ms，仍非严格 100 ms。可选 NDI、全遮挡停止及独立包流程均有测试，详见接入验证记录。

- `real_validation` 默认改为设备模型 → 部署预热 → 目标规划 → 执行记录四页；独立六腔调压/限制、非零输入状态估计与质量预热、自动时域搜索、轨迹/压力预览和手动接管已接入。加载不发压力，不记录准备动作。旧 OpenLoop 仅在显式兼容入口启用，可导出含 HTML 图解的自包含 PC 包。
- 四步虚拟设备 GUI 已完成较大弯曲目标的 40 条动作 ACK、40 帧反馈、36 次后缀更新、手动接管和最后归零（redesigned_006）；虚拟发令间隔 P95/最大 126.7/139.4 ms。已修复阀端实际限速后剩余动作的接续约束。相机缓冲与旧 OpenLoop 回归均已检查；真实硬件未连接，实际负压和避障未支持。操作与证据见[接入验证记录](../experiments/hereditary_real_validation_integration.md)。

- 2026-09-10：验证 GUI 恢复可选 NDI，增加自动 SDK/UVC 相机和最多 16 个软件遮挡矩形；每次执行独立保存压力、原图/反馈图、NDI 与时间关联。已验证有/无 NDI 的 40 步虚拟运行和全幅遮挡 3 次失效后归零，真实硬件仍未连接。

## 0.1 2026-09-01 既有路线快照

- 部署研究主线是少观测/有遮挡条件下的 windowed OpenLoop；GTObserved 用于上界、诊断和验证，不作为部署主张。
- HereditaryOperatorModel v2 已实现，用显式 PI/Maxwell 状态研究迟滞份额和可解释性；它是灰盒运动学模型，不等同于材料谱或 Cosserat 材料模型。
- 仓库分阶段整理已完成历史资产搬迁：67 项 raw/intermediate/processed/training/analysis/model 资产统一位于 workspace；GT/OpenLoop 2+2 epoch 主线短跑及评价通过后，旧兼容入口与空根已移除。训练验证现由公共 engine 选择 `best_eval_model.pt`；10 Hz canonical release 的首个 formal run 已完成独立 frozen test 和 offline fixture 验收。
- 数据划分、checkpoint 语义、早停、试次归档和证据表述以 [`../standards/`](../standards/) 为权威规则。
- `real_capture` 与 `real_validation` 继续分离；NDI 只作独立 endpoint 评价，不能成为模型或 planner 输入。
- 尚未完成：15 条历史 dataset 引用审计、废弃工作树审计、文档逐份归档。

## 0.2 文档治理健康与删除区

治理角色已经固定：`CLAUDE.md` 是约束面、`docs/README.md` 是地图、本页是当前
状态、`docs/maintenance/README.md` 是重要历史。旧 `docs/HANDOFF.md` 是
2026-07-28 快照，不再拥有当前状态。`paper/` 已固定为唯一活跃 manuscript，
`papers/` 只保存文献证据和阶段草稿输入。仍待处理的是通用/单序列 workflow
分离，以及旧文档逐份 keep/merge/supersede/archive 裁决。
权威入口、standards 和 maintenance 已由
`scripts/maintenance/check_docs_governance.py` 检查 front matter、四角色映射和
关键相对链接；历史长尾文档暂不强制批量改写。

以下路径或概念属于删除区；除非 replacement 合同发生经记录的变更，否则不要重建：

| 已移除入口/内容 | 原因 | replacement |
|---|---|---|
| `real_capture/data/`、`data/real_seq/`、`train_log/`、`output/` 正式写入口 | 同一资产多处存放 | `workspace/` + registry URI |
| 外部 validation watcher | 与公共 engine 双写 selection checkpoint | `src/training/` validation adapter |
| 被 Git 跟踪的 `__pycache__/*.pyc` | 可重建缓存污染源码历史 | Python 运行时缓存，保持 ignored |
| 根目录/`tests/` 的旧生成图 | 源码与展示产物混放 | 对应 run/analysis 输出目录 |

下文保留仿真与早期实物路线的历史基线；其中带“当前默认”“最好”的旧表述若与本节或具体 run 记录冲突，以本节和带日期证据为准。

---

## 1. 项目目标

让软体机器人仅通过观察自身外部图像(2D 相机), 学会预测自身在不同驱动输入下的形态。NeRF 范式: 用体渲染把 3D 场投影到 2D, 用 2D 图像做监督; 软体特有挑战 = **迟滞(路径依赖)** + **深度歧义(单视角)**。

---

## 2. 模型演进路线

| 代 | 模型 | 核心思路 | 问题/结果 |
|---|---|---|---|
| 第一代 | MSTNF | 直接 NeRF: 动作时序→EMA→MLP→[vis,dens]→体渲染 | 搜索空间太大, 骨架不光滑 |
| 第二代 | C-MSTNF 系列 | 典范场+变形场(D-NeRF 范式) | 变形场 MLP 高频振荡, 骨架不光滑(ODE/Smooth 变体未解决, 已归档) |
| 第三代 | **MS-SCNF** | **显式骨架回归** + 骨架条件密度场(距离→密度) | 仿真主线, MNE 0.0646m(有3D GT) |
| 第四代(当前主线) | **StateTransitionSpatialModel** (gt/open_loop) | **闭环状态转移** s_t=F(s_{t-1},a_t,z_{t-1}); 分数阶记忆 + 迟滞潜变量 z + 增量预测 | 实物免标定路线, 末端 NDI 0.77mm |

演进逻辑: 隐式变形场不收敛 → 显式骨架(连续, 可直接3D监督) → 从"action→state 前馈"升级为"状态转移闭环"(解决迟滞)。

> 模型组件详见 `project_help.md`; 信号流(编码器→z→空间GRU→增量)详见 `real_data/workflow.md` §6 + 模型源码 docstring。

---

## 3. 仿真实验结果(MS-SCNF 阶段)

### 3.1 横向对比

| 方法 | MNE (m) | Tip (m) | Z轴误差占比 | 说明 |
|---|---|---|---|---|
| MS-SCNF (有3D GT) | **0.0646** | 0.1085 | 28% | 仿真主线, 第一梯队 |
| Exp2 Model A (3D GT) | 0.0611 | 0.1582 | 22% | 第一梯队 |
| **Exp1b (渐进2D+物理先验)** | **0.1535** | 0.2069 | 72% | 纯2D 最佳, 第二梯队 |
| Exp1 (纯2D 无先验) | 0.3129 | 0.5884 | 85% | 第三梯队 |
| Exp2 Model B (2D only) | 0.3028 | 0.5534 | 58% | 第三梯队 |

### 3.2 关键结论

- **深度歧义是单视角纯2D的根本瓶颈**: Z轴误差占比 85%(纯2D)→72%(+先验), 物理先验缓解但无法消除(Exp1b 改善2×, Exp3 量化深度方向误差占41%)。
- **3D GT vs 纯2D 差 4.96×**(Exp2), Z轴差 12.26×。
- **多视角是突破关键**(方向C): 双视角 RGB + 体渲染预期 MNE 0.15→0.08-0.10m。
- **迟滞建模**: HA-EMA(方向感知衰减)未显著优于 StandardEMA(Exp5, 仿真迟滞弱); 后续用**分数阶记忆** + **迟滞潜变量 z**(实物路线已采用)。
- **域随机化**: 轻量 DR 提升渲染鲁棒性(Exp4b, dr_light PSNR 最优), 需控制强度。

---

## 4. 实物免标定路线(当前主线)

从仿真转向实物, **免相机标定**: state = 图像骨架像素 `[col,row,0]`, NDI 仅作末端 mm 验证。完整流程见 [`../real_data/workflow.md`](../real_data/workflow.md)。

### 4.1 已打通的管线

```
照片 → 分割(white_on_blue / SAM2视频) → mask → 修复 → 骨架化(逐行质心+tip_fix) → npz(15节点) → clean → train_transition(gt/open_loop) → eval_real_quant(NDI mm)
```

- **数据**: 序列 `seq_20260627_163921`, 10214 帧, 1-DOF 双段硅胶臂, 单 RealSense + NDI。
- **mask 三来源**: raw(有腐败) / masks_repaired(启发式三步修复) / **SAM2 视频**(分块双向, 最干净 area std 1.7%, 修好32帧启发式仍残缺的半mask)。
- **骨架化**: 逐行质心 + tip_fix(末端垂直切片修 corner, 7法对比 0.80px 最优, medial_axis 7.50px 最差)。默认 **N=15** 节点。
- **当前默认训练数据**: `data/real_seq/seq_20260627_163921_n15_rep_clean/`; SAM2 版 `*_n15_sam2_clean/` 可选。

### 4.2 模型(StateTransitionSpatialModel)

学状态转移 `s_t = s_{t-1} + delta_scale·tanh(Δ)`, 其中 Δ 由:
- **FractionalMemory** 编码动作历史(分数阶 Grünwald-Letnikov 幂律记忆核, 匹配硅胶粘弹性迟滞, 区别于 EMA 指数衰减)。
- **迟滞潜变量 z**(GRUCell 跨帧演化, 无 GT, 端到端从 skeleton loss 学)。
- **沿臂空间 GRU**(悬臂梁因果, base→tip 传播)。

两种部署(同网络, 差 teacher forcing):
- **gt**(主线, TF=1.0): 每步喂真实 s, 不漂移, 部署=每步观测。
- **open_loop**(TF=0): 窗口开环 rollout, 部署=观测一次预测 K 步, drift_by_k 评估。

### 4.3 实测精度

| 指标 | 值 | 说明 |
|---|---|---|
| NDI 标定底(px→mm 仿射残差) | **0.74 mm** | mask 骨架化 + NDI 噪声 + 非平面 |
| GT 模型末端 mean | **0.77 mm** / median 0.57 / p90 1.4 | 已到噪声底, mm 可信 |
| SAM2 mask area std | 1.7% | 无漂移/无丢目标/无手污染泄漏 |
| clean 对 SAM2 的影响 | outlier 3+1帧, act std 3.92→3.73px | 近 no-op, 安全 |

> 实物 GT 模型近零 loss = 训练成功(56× 优于 copy 基线); 蓝点/相机/GRU 迁移都是 viz bug, 诊断时必须归一化 action。

---

## 5. 当前模型与原始设想的差距

| 原始设想 | 现状 | 原因 |
|---|---|---|
| 隐式学习3D形态 | 显式骨架坐标(半显式) | 隐式搜索空间太大不收敛 |
| 只用2D图像训练 | 仿真用3D GT; 实物用2D骨架(免标定) | 2D渲染loss梯度太弱 |
| 网络自由发现结构 | 假设链式骨架 | 软体臂是细长杆, 先验合理 |
| 单阶段端到端 | 两阶段(骨架→外观) / 状态转移 | 更稳定 |
| action→state 前馈 | **状态转移闭环**(带 z 迟滞) | 迟滞下稳态假设失效 |

**本质**: 形态先验(链式骨架)是**输出侧约束**(区别于 FBV-SM/Chen/SoftNeRF 的输入侧假设), 牺牲通用性换高精度+少量数据+可直接控制。适用于可定义骨架的软体连续体(气动/缆绳驱动的杆/臂)。

---

## 6. 后续方向

17 个研究方向在 [`../directions/`](../directions/)(`directions_overview.md` 索引)。当前焦点:

- **部署目标 = open_loop(方向15)**: 形状控制需"给定动作→预测形状"的规划能力, open_loop 是部署形态; gt(方向14)退为训练基础+精度上界(NDI 0.77mm)。[[open-loop-deployment-target]]
- **方向 16(逆运动学形状控制, 新进展)**: 方向1 视野认证(`eval_horizon.py`, open_loop drift 1.7×, K_max~50-120步) + 方向2 可微逆规划(`scripts/control/inverse_plan.py`, reach 3.07px / 0.38× do-nothing)已验证。详见 [`../directions/16_constraint_oriented_control.md`](../directions/16_constraint_oriented_control.md)。
- **方向 13(closed_loop)**: 纯自回归(误差无界, 作对照; gt 漂移 272× 不可作规划仿真器)。
- **SAM2 mask**: 已就绪, 待对比 SAM2 vs 启发式 mask 对模型精度的影响。
- 仿真侧: 多视角(方向C, 消除深度歧义)、sim2real(方向D)。

---

## 7. 论文定位

**骨架条件化软体机器人自建模 + 免标定状态转移 + 分数阶迟滞记忆**。

核心故事线:
1. 软体机器人需从 2D 图像学 3D 形态 → NeRF 范式。
2. 纯隐式不收敛 → 显式骨架 + 条件密度场(MS-SCNF)。
3. 软体迟滞(路径依赖) → 闭环状态转移 + 分数阶记忆 + 潜变量 z(实物路线)。
4. 免标定实用部署 → 2D 像素骨架作 state, NDI 独立验证 mm 精度。
5. 诚实承认形态先验限制通用性, 强调高精度+少量数据+可控制的优势。

差异化(vs FBV-SM/Chen2022/SoftNeRF): 软体连续体 + 显式骨架 + 时序迟滞建模 + 免标定单相机。详见 [`../background/literature.md`](../background/literature.md)。
