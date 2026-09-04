# ISHSM 优化、训练与验证总记录

> 文档角色：ISHSM 实验历史与当前裁决的唯一入口
>
> 覆盖版本：v1--v5
>
> 数据日期：2026-09-02--2026-09-03
>
> 整理日期：2026-09-04
>
> 设计依据：[ISHSM 设计](../designs/2026-09-02_ISHSM.md)

## 1. 文档范围

本文合并原先按版本分散的训练计划、TDD 证据和首轮结果。它回答四个问题：

1. 每一版为什么修改；
2. 修改由什么实验裁决；
3. 哪些结论保留，哪些假设已被数据否定；
4. 训练和分析产物在哪里，如何复核。

训练目录不做物理合并或改名。每次实验继续保存独立的命令、配置、日志、checkpoint、退出码和
完成标记，避免覆盖并保持结果可追溯。本文只合并说明性文档并建立统一索引。

## 2. 研究问题与证据边界

核心问题是：

> 能否用低维、空间化、因果递推的状态表示软体机器人迟滞，并在仅输入动作或稀疏骨架观测时
> 预测完整骨架？

当前数据可以检验模型级的衰减动态、空间模态、观测校正和前向预测误差，不能直接识别腔道内
压力传播，也不能把学习到的时间常数称为纯材料参数。所有 v2--v5 主要结果来自已参与模型选择的
dev 集，属于开发集 forward prediction；它们不是最终独立 test，更不是真实控制精度。

## 3. 数据、选模与统计合同

- 原始处理数据：`workspace/data/processed/real/seq_20260819_10hz_n15_sam2_robot_mm`；
- 派生划分：`workspace/data/derived/ishsm_v1_20260902_000`；
- fit：2695 帧，只用于归一化、冻结 H0、POD 基和动力学训练；
- dev：2 条序列，保留 40 帧上下文后共 675 个计分帧；
- 旧 test：1 条序列、613 个锚点后计分帧，v1 使用后不再视为独立确认集；
- 采样周期：`dt=0.1 s`；骨架为 15 节点、两段各 7 个线段；
- 动作通道：`[0, 1, 3, 5]`；
- 正式选模：`best_eval_model.pt`，按聚合 dev `validation.node_mean_mm` 最小选择；
- 强制同时报告 endpoint mean；
- 配对区间：按序列分层的 circular moving-block bootstrap，block 20 帧，10000 次，seed 42；
- “改善”定义为 baseline error 减 candidate error，正数表示 candidate 更好。

`window_size=40` 是动作缓存长度；`single_anchor` 是在计分边界前只观察一次骨架；
`periodic_40` 是每 40 帧重锚一次状态。三者不能混为同一个协议。

## 4. 不变的模型骨架

### 4.1 可观测广义坐标

15 个节点形成 14 个线段。形状被编码为 8 个 POD 弯曲空间系数和 2 个分段对数伸长状态：

```math
z_t = [z^{\mathrm{bend}}_{1:8,t}, z^{\mathrm{len}}_{1:2,t}].
```

8 个弯曲状态是 8 个空间基的幅值，不是 8 个节点、8 条真实波或天然对应 8 个材料时间尺度。
两个伸长状态分别对应机器人两段的低维长度偏差。位置通过正段长、切向角和固定基座进行确定性
积分重建，不使用自由坐标解码网络。

### 4.2 冻结的记忆无关参考 H0

H0 只读取当前动作，在 fit 上拟合后冻结。最终采用逐通道单调铰链样条，并让拟合目标包含
广义坐标、重建节点和端点：

```math
q_{0,t}=H_0(a_t),
\qquad
\mathcal L_{H0}=\mathcal L_q
+w_n\frac{\mathcal L_{node}}{\mathcal L_{node}^{linear}}
+w_e\frac{\mathcal L_{tip}}{\mathcal L_{tip}^{linear}}.
```

H0 是“记忆无关参考形状”，在没有专门的长保持实验前不称作真实静态平衡。

### 4.3 动作驱动的稳定动态

动态状态使用动作增量激励的一阶稳定递推：

```math
z_t=\Lambda z_{t-1}+B(a_t-a_{t-1}),
\qquad
\Lambda_m=\exp(-\Delta t/\tau_m).
```

最终结构保留 8 个弯曲空间模态，但共享一个弯曲时间常数；两个伸长状态各有独立时间常数。
这将动态时间参数从 10 个减少为 3 个，同时不削减空间表达能力。所有 `tau` 均限制在
`[0.3, 2.0] s`。

### 4.4 稀疏观测更新

观测骨架先反演到相同的 8+2 状态。v4 起不再直接硬替换，而使用两个有界分组增益：

```math
z_t^+=z_t^-+K(z_t^{obs}-z_t^-),
\qquad 0<K_{bend},K_{len}<1.
```

v5 增加可选的端点一致投影。它在 modal 投影后用确定性几何 decoder 的解析端点 Jacobian 做
一次阻尼最小二乘校正：

```math
\Delta z=J_{tip}^{T}(J_{tip}J_{tip}^{T}+\lambda I)^{-1}e_{tip},
\qquad z_{obs}=z_{modal}+\Delta z.
```

该投影不增加可学习参数。最终比较表明它只适合作为稀疏重锚的候选分支，不应替代 action-only
主线。

## 5. 版本演进与因果裁决

| 版本 | 唯一主要改动 | 得到的证据 | 裁决 |
|---|---|---|---|
| v1 | 线性 H0 + 独立 8+2 一阶状态 | H1 和伸长状态有效；single≈zero；弯曲 tau 聚下界 | 保留低维状态，改进 H0；不解释多时间谱 |
| v2 | 单调 H0 + 常数持久形状偏置 | H0 改善；持久分支不改善 single；8 模态略优于 4 | 删除默认持久分支，保留 8+2 |
| v3 | geometry H0 + endpoint loss + 协议显式化 | node/tip 均明显改善；硬重锚产生 node/tip 交换 | 保留 geometry H0；重做 observer |
| v4 | 分组 innovation observer + shared bending tau | shared tau 更好；observer 未通过 endpoint guardrail | shared tau 成为主线；不宣称观测问题已解决 |
| v5 | tip-DLS 投影 + 200 epoch + Hereditary F5 | 高频重锚 tip 改善但 action-only 略退化；Hereditary 明显更准 | ISHSM 保持解释性基线；精度主线倾向 Hereditary |

### 5.1 v1：最小 8+2 动态状态

几何预检显示：固定训练平均段长但使用真实角度时，验证几何误差地板约 `1.35 mm`；按动作预测
两段长度可降至约 `0.68 mm`；使用真实两段整体伸长量约为 `0.21 mm`。动作线性参考之外的
弯曲残差前 8 个 POD 模态覆盖约 `97.2%` 能量。因此选择“8 个弯曲状态 + 2 个分段伸长状态”，
而不是自由预测 14 个节点坐标。

v1 seed 42 的主要旧 test 结果如下。该 test 在本轮使用后不再用于后续确认性结论。

| 协议 | 8+2 node / endpoint (mm) | 无动态伸长 node (mm) |
|---|---:|---:|
| H0 | 1.9433 / 4.6891 | 1.9433 |
| zero-init | 1.7939 / 3.9114 | 1.8901 |
| single-anchor | 1.7934 / 3.9092 | 1.8897 |
| periodic-1 | 1.7071 / 3.8424 | 1.8687 |
| periodic-40 | 1.7931 / 3.9263 | 1.8913 |

H1 相对 H0 改善 `0.1499 mm [0.1244, 0.1765]`；8+2 相对无动态伸长在 single 下改善
`0.0964 mm [0.0710, 0.1228]`。但 single 相对 zero 仅改善
`0.00055 mm [-0.00008, 0.00177]`，一次初始观测无稳定收益。前 8 个弯曲 tau 聚集在
`0.30--0.37 s`，两个伸长 tau 约为 `0.42 s` 和 `1.75 s`。

### 5.2 v2：单调参考与持久状态反证

v2 将 H0 改为单调样条，并尝试把观测残差拆为衰减状态 `z` 与常数持久状态 `c`。dev 结果：

| 模型 | 参数 | single | periodic-1 | periodic-40 |
|---|---:|---:|---:|---:|
| v2 main，带 persistent | 60 | 1.9129 | 1.6775 | 1.8891 |
| v2 no-persist | 50 | **1.9081** | 1.6872 | 1.9040 |
| v2 bend4 | 36 | 1.9393 | 1.7119 | 1.9173 |
| v1 8+2 | 50 | 1.9895 | **1.6593** | 1.9678 |

单调 H0 相对线性 H0 改善 `0.1255 mm [0.0782, 0.1930]`。persistent main 的 single
相对 zero 退化 `0.0048 mm [0.0035, 0.0058]`，也比 no-persist 差约 `0.0048 mm`；因此
“常数形状偏置保存初始观测”的假设被否定。8 模态相对 4 模态在 single 上改善
`0.0264 mm [0.0213, 0.0323]`，保留空间容量。

### 5.3 v3：几何对齐与端点约束

v3 让 H0 拟合与真实几何指标对齐，同时降低弯曲辅助损失、提高 endpoint 权重。dev：

| 模型 | H0 node / endpoint | single node / endpoint | periodic-1 node / endpoint |
|---|---:|---:|---:|
| v2 no-persist | 1.9712 / 4.5232 | 1.9081 / 4.1739 | 1.6872 / 4.0918 |
| v3 geometry S42 | 1.9045 / 4.4403 | **1.7572 / 3.3838** | **1.6224 / 3.7176** |
| v3 linear S42 | 2.0967 / 5.0527 | 1.9537 / 3.9131 | 1.6364 / 3.3009 |

v3 geometry 相对 v2 no-persist 的 single 改善 node
`0.1510 mm [0.1267, 0.1773]`、endpoint `0.7902 mm [0.6367, 0.9929]`。H1 相对自身
geometry H0 仍改善 node `0.1473 mm`、endpoint `1.0566 mm`，说明并非只有静态参考在工作。

但是 periodic-1 相对 single 虽改善 node `0.1348 mm`，却使 endpoint 退化
`0.3338 mm [0.2057, 0.5076]`。原因是低秩 modal 硬投影优化平均弯曲残差，小的基角误差却会
沿几何积分放大到端点。该结果推动了 v4 observer。

v3 同时发现 dev 文件含 40 帧不计分前缀，而旧 evaluator 把 single anchor 放在文件第 0 帧。
修正后锚点统一位于首个计分帧之前；结论不变，single 仍不优于 zero。此后所有报告均按修正协议。

### 5.4 v4：有界 observer 与共享时间尺度

v4 保留 8 个空间模态，只将弯曲时间常数收缩为一个共享参数，并加入分组 innovation update。
正式训练与选模状态：

| 运行 | 最优 epoch | dev node / endpoint (mm) | 结束原因 |
|---|---:|---:|---|
| innovation/shared S42 | 64 | 1.7510 / 3.2902 | epoch 80 early stop |
| innovation/shared S43 | 36 | 1.7526 / 3.2912 | epoch 52 early stop |
| hard/shared S42 | 68 | 1.7514 / 3.2738 | epoch 84 early stop |
| innovation/independent S42 | 98 | 1.7570 / 3.3512 | epoch 100 budget |
| Hereditary v2 equilibrium S42 | 100 | 1.7818 / 2.9004 | epoch 100 budget |

shared 相对 independent 在 zero、single、K=1、K=5、K=40 的 node/endpoint 均小幅改善，
因此“8 个空间模态共享一个弯曲 tau”通过预注册判据。S42/S43 弯曲 tau 均为 `0.3000 s`；
伸长 tau 分别约 `[0.4187, 1.9930] s` 与 `[0.4827, 1.9984] s`。边界仍约束解，数值只能
称为有效动态尺度。

innovation 没有通过 observer 判据：相对 single，K=1 改善 node
`0.0866 mm [0.0458, 0.1274]`，却使 endpoint 退化
`0.1645 mm [0.0620, 0.2758]`；K=5 endpoint 也退化约 `0.0945 mm`。single 相对 zero
在 node 和 endpoint 分别退化约 `0.00235`、`0.00356 mm`。因此不宣称“一次观测长期有效”。

公平 action-only 对照中，ISHSM zero-init 为 `1.7486 / 3.2866 mm`，Hereditary continuous
为 `1.7818 / 2.9004 mm`。node 差异区间跨零，Hereditary endpoint 稳定好
`0.3862 mm [0.1373, 0.6267]`。ISHSM 只有 45 个可学习参数，Hereditary 为 3187 个。

### 5.5 v5：端点一致投影与 200-epoch 裁决

v5 比较 modal 与 tip-DLS 投影，并将 Hereditary 的 0.3/0.5 residual 上限都训练到 200 epoch。
ISHSM dev 结果：

| 模型 | zero-init | single-anchor | periodic-1 | periodic-5 | periodic-40 |
|---|---:|---:|---:|---:|---:|
| modal | 1.7486 / 3.2866 | 1.7510 / 3.2902 | 1.6644 / 3.4547 | 1.7366 / 3.3847 | 1.7535 / 3.3174 |
| tip-DLS | 1.7497 / 3.3227 | 1.7500 / 3.3235 | 1.6843 / 2.9994 | 1.7388 / 3.2575 | 1.7493 / 3.3256 |

tip-DLS 相对 modal 在 K=1 的 endpoint 改善
`0.4553 mm [0.2904, 0.6444]`，K=5 改善 `0.1272 mm [0.0696, 0.1879]`，且 node 差异
区间均跨零；端点一致投影的局部假设得到支持。但它在 action-only zero 的 endpoint 稳定退化
`0.0360 mm [0.0243, 0.0491]`，single 退化 `0.0332 mm [0.0197, 0.0475]`，因此不能作为
无条件默认优势。single 仍不优于 zero。

Hereditary 200-epoch dev：

| residual 上限 | continuous node / endpoint | cold-restart-40 node / endpoint | raw / effective scale |
|---:|---:|---:|---:|
| 0.3 | 1.4344 / 2.4665 | 1.4458 / 2.4971 | 0.3060 / 0.3000 |
| 0.5 | **1.3366 / 2.3704** | **1.3467 / 2.3765** | 0.5004 / 0.5000 |

0.5 相对 0.3 在 continuous 改善 node `0.0978 mm [0.0751, 0.1241]`、endpoint
`0.0961 mm [0.0385, 0.1514]`；在 cold-restart-40 改善 node `0.0990 mm`、endpoint
`0.1206 mm`。0.5 仍顶住上限，说明 F5 确为当前模型的容量信号，但 dev 上继续放宽的泛化收益
尚未被独立 test 验证。

Hereditary residual-0.5 continuous 相对 ISHSM modal zero-init 改善 node
`0.4121 mm [0.3323, 0.4931]`、endpoint `0.9162 mm [0.7120, 1.1333]`。continuous 相对
cold-restart-40 的 node 小幅改善 `0.0102 mm [0.0013, 0.0195]`，endpoint 区间跨零；没有出现
旧版链式单调爆炸。

## 6. 当前裁决

1. **精度主线：HereditaryOperatorModel v2。** 在同 fit/dev、同选模和 200-epoch 预算下，
   residual-0.5 明显优于当前 ISHSM；论文若只保留一个精度模型，应优先使用该版本，但必须在新
   独立轨迹上冻结确认。
2. **解释性主线：ISHSM shared-tau。** 其 8+2 空间状态、冻结 H0、稳定一阶动态和确定性几何
   重建具有更直接语义，只有 45 个可学习参数，适合作为强解释性对照或结构消融。
3. **已证实机制：** 当前动作之外的低维动态状态有用；两段伸长状态有用；geometry H0 有用；
   8 个空间模态可以共享短弯曲时间尺度；tip-DLS 能修复高频重锚的部分端点误差。
4. **未证实机制：** 一次骨架观测长期有效；常数持久形状偏置；8 个独立弯曲时间尺度；压力波、
   扩散或局部材料松弛的唯一来源解释。
5. **下一步证据：** 冻结模型与选模规则后采集新的独立速度/保持/释放轨迹；同时报告
   Hereditary continuous 与 cold-restart-40，以及 ISHSM zero/single/periodic 的 node、endpoint
   和 horizon 指标。

## 7. 机器可读证据入口

| 阶段 | 入口 |
|---|---|
| v1 主模型与无伸长 | `workspace/runs/analysis/ishsm/run_20260902_000_main/summary.json`；`run_20260902_001_alen/summary.json` |
| v2 | `workspace/runs/analysis/ishsm_v2/` |
| v3 | `workspace/runs/analysis/ishsm_v3/analysis_summary.json` |
| v4 | `workspace/runs/analysis/ishsm_v4/analysis_summary.json` |
| v5 / Hereditary 200 | `workspace/runs/analysis/ishsm_v5_200/analysis_summary.json` |
| Hereditary 解释性审计 | `workspace/runs/analysis/hereditary_v2_interpretability_20260903_004/summary.json` |

训练目录的 `command.txt`、`config.json`、`session.log`、`exit_code.txt`、`TRAINING_COMPLETE` 和
`best_eval_model.pt` 是运行级事实来源。分析 JSON/CSV 是数值来源；本文是解释和裁决来源。

## 8. 实现、运行与验证入口

- 模型：`src/models/model_ishsm.py`；
- 训练入口：`scripts/training/train_transition.py --mode ishsm`；
- 非覆盖运行器：`scripts/experiments/run_ishsm_training.sh`；
- 公平 Hereditary 基线：`scripts/experiments/run_hereditary_ishsm_baseline.sh`；
- 稀疏观测评估：`scripts/evaluation/eval_ishsm.py`；
- v4/v5 聚合分析：`scripts/evaluation/analyze_ishsm_v4_results.py`、
  `scripts/evaluation/analyze_ishsm_v5_results.py`；
- 数据划分：`scripts/experiments/prepare_ishsm_split.py`。

定向验证覆盖：几何往返、稳定递推、共享 tau、分组 innovation、tip-DLS、因果重锚、锚点计分
边界、validation-best 协议、checkpoint 严格恢复、旧 checkpoint 兼容和非覆盖数据划分。v4 完整
定向回归曾运行 60 项测试并通过；整理提交时应重新运行当前测试集合，以当前结果为准。

## 9. 归档规则

- 不删除或重命名已被分析引用的 `workspace/runs/` 目录；
- 不把多个 checkpoint 或日志拼接成一个文件；
- 新实验必须使用新目录，并以完成标记区分成功、失败和中断；
- 普通过程由 Git 和 run manifest 保存，不再新建“某版本计划 + TDD + 第一轮结果”三份文档；
- 新的材料结论直接追加到本文版本演进和当前裁决，详细机器结果留在独立 analysis 目录；
- 最终论文数值必须来自冻结后的新独立 test，不能把本页 dev 数值改写成泛化或控制结论。
