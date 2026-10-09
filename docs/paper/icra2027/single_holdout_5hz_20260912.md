---
title: 5 Hz 全量数据单次6:2:2建模实验
kind: experiment-protocol
status: complete
updated: 2026-09-12
scope: full seven-sequence corpus, rapid validation screening, five training seeds
sources:
  - modeling_experiments.md
  - modeling_baselines_sources.md
---

# 5 Hz 全量数据单次6:2:2建模实验

2026年9月12日已完成70次正式训练与70次测试，并按本协议完成结果复核和配对统计。[实验报告](../../../workspace/reports/modeling_5hz_622_20260912_000/report.html) · [表格、图表与复核资料](../../../workspace/reports/modeling_5hz_622_20260912_000/README.md)。下文保留本次运行时冻结的实验与执行协议。

**后续有效性复核：部分SAM2标签存在跨序列来源污染，当前计分应作为历史记录保留，模型能力结论需修正数据后重新实验。** 问题发生于上游并行分割的共享JPEG缓存，6:2:2划分本身与本协议一致。详见[标签来源诊断](../../../workspace/reports/modeling_5hz_622_20260912_000/sequence_diagnostic/README.md)。

本次使用全部7条5 Hz序列，统一训练/验证/测试划分，正式配置跑5个随机种子。实验根目录：[`modeling_5hz_622_20260912_000`](../../../workspace/runs/training/modeling_5hz_622_20260912_000)。准确运行状态以根目录 `status.json` 为准。

## 数据和统计单位

每条原始序列按时间顺序划分60%训练、20%验证、20%测试，再按角色合并训练。以最大余数分配处理整数帧，得到精确全局比例：

| 角色 | 原始帧数 | H=20的计分窗口 |
|---|---:|---:|
| train | 10464 | 10331 |
| val | 3488 | 3355 |
| test | 3488 | 3355 |
| 合计 | 17440 | 17041 |

每个序列的每个分段拥有自己的19帧历史上下文，首个目标为该段第20帧；不跨集合或跨序列取历史。全部帧参与所属分段的上下文或目标，边界没有额外复用。清单保留原序列帧号、完整父文件哈希、三个区间、mask库存哈希和裁剪标定，测试mask按原始帧号对应。

证据等级为 `within_sequence`：训练与测试包含相同原始序列的不同时段，结论针对当前采集条件下的时段泛化。相邻帧、重叠窗口和五次初始化不被计为独立物理样本。后续配对统计先在每个原序列内平均五个种子，再对七条原序列的测试时段进行比较；置信区间与显著性解释保留同日采集、有限独立组数等条件。

## 方法与新增基础对照

比较配置10种：`chen_direction, bezier_gru, park_tcn, oscillator, koopman, pcc, mlp, window_mlp, linear, polynomial2`。

- Chen、Yu、Park、Krauss分别对应压力＋方向网络、分段二次Bézier读出、因果TCN、潜振子动态核的当前数据适配。准确保留/修改范围见[来源说明](modeling_baselines_sources.md)，名称不表示完整原论文实验等价复现。
- `koopman`：动作非线性提升、受控线性潜状态更新、线性骨架读出；状态转移矩阵的谱范数由Frobenius范数上界约束为小于1，每窗零状态初始化。这是动作预算下的Koopman动态核适配，没有使用测试观测初始化或声称完成原系统Koopman特征函数辨识。
- `pcc`：两段平面分段常曲率模型；压力MLP预测两段弯曲角和长度变化，确定性圆弧积分生成15节点。基础段长、基点取训练目标帧统计，基切向固定沿机器人坐标正Y方向。长度为正，各段共享接点。这是PCC几何与学习驱动映射的组合。

消融4种：`hov, hov_no_play, hov_no_maxwell, hov_no_memory`。前三者分别为完整模型、关闭play、关闭Maxwell；最后一个关闭全部记忆，保留同一训练集拟合的静态几何参考。所有变体独立拟合/重训练，继承完整模型选择后的学习率、算子容量及先验预算。静态模型和闭式方法的重复结果可能完全一致，如实保留。

## 快速验证选择

所有方法只使用train拟合、val选型，测试数组不参与筛选。可训练对照和完整HOV均做4个候选：学习率 `{0.001,0.003}` × 基础/扩展容量，固定筛选种子101。每个候选使用完整10331个训练窗口训练15 epoch，batch256；第1、5、10、15 epoch计算全部3355个val窗口，按七条序列等权的平均节点欧氏误差选checkpoint/候选。静态线性和二次多项式各筛选3个岭系数 `{1e-5,1e-4,1e-3}`。共42个候选任务。

| 方法 | 基础容量 | 扩展容量 |
|---|---|---|
| Chen | 4×128隐藏单元 | 4×256 |
| Yu读出／MLP／窗口MLP／PCC | 隐藏64 | 隐藏128 |
| Park | 通道4 | 通道8 |
| Krauss振子 | 潜维8、力网络32 | 潜维16、力网络64 |
| Koopman核 | 提升隐藏64、潜维8 | 提升隐藏128、潜维16 |
| HOV | 每通道2 play＋6 Maxwell | 每通道3 play＋8 Maxwell |

HOV参考拟合预算均为500步，且先验和主训练使用相同训练目标帧。共同目标为归一化骨架坐标MSE＋0.25×末节点坐标MSE。容量、学习率和验证曲线写入各候选文件夹；`frozen_configs.json`记录全部候选、获选配置及最近三次验证变化。这是赶时间下的等预算筛选，不把15 epoch自动视为已收敛。

## 正式训练、测试及后台顺序

全部候选结束后才冻结所有方法配置，然后执行14种配置×种子 `{0,1,2,3,4}`，共70次训练。共同最多300 epoch，至少100 epoch才允许按验证平台期早停；连续12次验证检查未有0.001 mm改善时停止。验证每5 epoch一次，学习率使用ReduceLROnPlateau（4次检查、乘0.5、下限1e-5），恢复验证最优checkpoint作为测试对象。闭式回归和静态参考拟合完成后即可结束对应run。

四个固定GPU工作进程从任务队列取作业，每张GPU同一时间只运行一个作业。正式队列先启动完整HOV、Chen、Yu、Park，随后分配其他对照与记忆消融；五个种子都采用冻结后的同一方法配置。全部正式训练结束后写 `TRAINING_COMPLETE`，再批量执行固定测试评价：全部3355个测试目标帧、真实mask、固定8 mm管状半径和2 px边界容差。完成后写 `TEST_SCORING_COMPLETE`，状态转为 `awaiting_user_analysis`。

显著性与论文结论汇总由用户通知后执行，准备的 `analyze` 子命令会分别生成主对照和消融的配对统计。当前后台进程只完成训练、选模和固定测试计分，不根据测试结果重新调参。

## 文件与运行状态

```text
modeling_5hz_622_20260912_000/
  study_plan.json                 # 顺序、预算、种子、源码哈希
  preflight_checks.json           # 数据与CUDA预检
  data/dataset_manifest.json      # 单一6:2:2划分
  frozen_code/                    # 实际运行的源码副本
  screening/<model>/candidate*/   # 快速train/val筛选
  frozen_configs.json            # 全部方法一次冻结
  formal/comparison/<model>/seed*/
  formal/ablation/<model>/seed*/
    best_eval_model.pt
    training_state.pt            # 有梯度训练每25 epoch保存恢复资料
    history.json
    run_manifest.json
    evaluation_test/
  jobs/                          # 每次任务的完整参数、命令、GPU分配
  logs/                          # 每个任务独立日志
  workers/gpu0.json ... gpu3.json # 当前任务、PID和进度
  status.json
  orchestrator.log
```

`status.json`和各run记录成功/失败状态。任一阶段有失败会阻止进入下一阶段；已有文件和日志保留。冻结源码与数据哈希在启动前校验，后续编辑工作目录的代码不改变本次正在运行的实现。

本次tmux会话：`modeling_5hz_622_20260912_000`。只需在需要时读取 `status.json`。正式训练已启动后无需持续轮询；用户通知训练结束后再开展比较、显著性检验与论文表格整理。

已完成预检：42项测试通过，14个运行配置的CUDA前向/反向检查通过，四张RTX3090均可用。新增实现位于 [`modeling_study.py`](../../../scripts/experiments/modeling_study.py)、[`modeling_fast_training.py`](../../../src/benchmarks/modeling_fast_training.py)、[`modeling_foundations.py`](../../../src/benchmarks/modeling_foundations.py)。

## 本次冻结与启动记录

42个筛选候选已全部完成，正式训练已进入四GPU队列，启动时未见失败。可读取[冻结配置](../../../workspace/runs/training/modeling_5hz_622_20260912_000/frozen_configs.json)、[70次正式计划](../../../workspace/runs/training/modeling_5hz_622_20260912_000/formal_plan.json)及[启动核验](../../../workspace/runs/training/modeling_5hz_622_20260912_000/formal_launch_verified.json)。

获选配置：HOV为学习率0.003、2 play＋6 Maxwell，三项记忆消融继承；Chen、Yu读出、Park、Krauss、Koopman、MLP及窗口MLP均选择0.003与扩展容量；PCC选择0.001与扩展容量；线性岭系数0.001、二次多项式岭系数0.00001。正式训练仍执行上述收敛检查和验证选模规则，短筛选成绩不作为论文比较结果。
