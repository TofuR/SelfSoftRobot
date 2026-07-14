# 项目状态快照

> "现在到哪了 + 接下来做什么"。精简自 `../archived/project_status_report.md`(2026.04.29, 仿真 MS-SCNF 阶段) + 实物路线后续进展。
> 文献详见 [`../background/literature.md`](../background/literature.md); 模型架构详见 [`project_help.md`](project_help.md); 研究方向详见 [`../directions/`](../directions/)。

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

16 个研究方向在 [`../directions/`](../directions/)(`directions_overview.md` 索引)。当前焦点:

- **方向 14(gt_observed)**: 实物每步观测, 当前主线, 已到 NDI 噪声底。
- **方向 15(open_loop)**: 窗口开环 rollout, 热启动自 gt, drift_by_k 评估中。
- **方向 13(closed_loop)**: 纯自回归(误差无界, 作对照)。
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
