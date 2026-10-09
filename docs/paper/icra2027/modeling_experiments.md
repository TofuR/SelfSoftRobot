---
title: 全身形态自建模实验：实现与运行协议
kind: experiment-protocol
status: implemented
updated: 2026-09-12
scope: modeling benchmark, literature adaptations, retrained ablations, evaluation
sources:
  - modeling_baselines_sources.md
  - modeling_sources.json
  - ../../standards/dataset_split.md
  - ../../standards/training_and_evaluation.md
---

# 全身形态自建模实验：实现与运行协议

研究问题是：在相同动作历史与训练标签预算下，显式迟滞记忆和可解释几何读出，能否提高软体机器人的全身形态预测精度及跨采集序列泛化？实验先建立基础映射的误差水平，再比较不同记忆结构，最后通过重新训练的结构消融解释误差变化。骨架、末端和整体轮廓提供互补评价。

入口：[`scripts/experiments/modeling_benchmark.py`](../../../scripts/experiments/modeling_benchmark.py)。实现分为[数据](../../../src/benchmarks/modeling_data.py)、[模型](../../../src/benchmarks/modeling_models.py)、[训练与评价](../../../src/benchmarks/modeling_runner.py)、[指标与统计](../../../src/evaluation/modeling_benchmark_metrics.py)。命令均在仓库根目录执行；输出目录已存在时会拒绝覆盖。

## 1. 已实现的对照

| CLI 名称 | 结构与用途 | 来源和解释范围 |
|---|---|---|
| `mean` | 训练目标平均形状 | 测量数据变化幅度与固定形状误差 |
| `linear`, `polynomial2` | 当前四路动作的岭回归；二次项包含全部交叉项 | 基础系统辨识；截距不正则化 |
| `mlp` | 当前动作 → 2×64 Tanh → 45坐标 | 静态非线性基础方法 |
| `window_mlp` | 展平20×4动作窗 → 2×64 Tanh → 45 | 不预设记忆结构的历史基线 |
| `direction_mlp` | 当前动作＋窗口内末次非零方向 → 2×64 Tanh | 与小型 MLP 对照方向特征 |
| `chen_static`, `chen_direction` | 4个128单元 ReLU隐藏层；分别输入4/8维 | Chen 2025 的压力＋方向全身网络适配；静态同容量对照。激活、动态数据、方向保持规则和统一损失是本项目选择 |
| `gru`, `lstm` | 单层64单元门控递归＋末时刻线性骨架头 | **直接提取 SPONGE 官方模型类**，保留 MIT 许可；输入改为四路动作、每窗初始化为零、监督当前骨架 |
| `tcn` | 3个单卷积残差块，64通道，膨胀率1/2/4 | 通用容量对照，感受野15步 |
| `park_tcn` | 每块两个因果卷积，核宽3，膨胀率2^b；4通道 | Park 2024 结构适配；块数按窗口长度计算。20步用3块，感受野29步；超出窗口部分补零 |
| `bezier_gru` | 64单元GRU → 两段二次 Bézier 控制点 → 每段8个等弧长点，共15点 | Yu 2026 曲线表示思想适配。固定基点、共享接点；65点细采样后分段弧长重采样。保留位置连续性，切向连续性不受约束；不代表其完整多视角 NODE 方法 |
| `oscillator` | 8维潜振子，正对角刚度/阻尼、单位质量、动作力网络、骨架头 | Krauss VON 方程的独立简化实现，辛Euler更新；首个压力处平衡初始化。比较动作驱动的结构化动态核；视觉表示与原视觉损失属于原方法额外组件 |
| `hov` | 当前 HOV 几何模型；2 play＋6 Maxwell/通道，14局部弯曲＋2段长度，冻结单调样条参考 | 复用项目真实实现；按本基准统一目标训练，权重由训练集重新学习 |
| `hov_no_play` | 训练与推理均关闭 play 读出，冻结该分支参数 | 重新训练其余可学习模块，检验率无关记忆贡献 |
| `hov_no_maxwell` | 同理关闭 Maxwell 读出 | 检验率相关记忆贡献 |
| `hov_static` | 仅使用训练集拟合的静态几何参考 | 衡量静态参考所能解释的部分；闭式/先验拟合后评价 |
| `hov_linear_reference` | HOV 的单调样条参考换为线性参考 | 线性坐标岭拟合与样条几何拟合的整体设计对照 |
| `hov_pod8` | 用训练集8个POD弯曲模态替代14个局部弯曲自由度 | 同时改变表示基与容量，解释为压缩表示对照 |
| `hov_memory_residual` | 加入平衡态为零的有界记忆残差网络 | 检验额外非线性残差是否有收益 |

共有 **21种配置**。普通网络的 `hidden` 可通过配置改变；Chen、Park、潜振子的结构常数按上述文献风格固定。模型参数量、可训练参数量、先验拟合帧数及训练耗时写入每个 `run_manifest.json`。关闭分支的算子状态保留合同尺寸，不能据此声称节省了运行状态或推理开销。

SPONGE 提取文件是 [`vendor/sponge_rnn.py`](../../../src/benchmarks/vendor/sponge_rnn.py)，两个类与固定 commit 的原类逐字一致；相关许可见 [`SPONGE_LICENSE.txt`](../../../src/benchmarks/vendor/SPONGE_LICENSE.txt)。论文原任务含状态输入和未来动态预测，当前动作到当前形状接口属于明确的任务适配。

三个已下载官方仓库、commit、原文位置、许可和下载证据见[来源核查](modeling_baselines_sources.md)。SoftNeRF 已在本地获取、静态检查；其条件 SDF 与三维渲染训练有额外监督及依赖要求，当前比较表不把骨架MLP称为完整 SoftNeRF。VON 的实现根据论文方程独立编写。原平台结果均不作为本数据集的比较数值。

## 2. 数据合同与划分

已核对的5 Hz扩充语料包含7条可训练独立序列：`172644,181044,181548,183351,183547,183740,184036`，均为2026-08-19同平台采集。零压短校准序列不在源部署语料内。10 Hz现有两条独立序列不足以同时建立独立train/val/test，准备入口会拒绝这种划分；不能把两个频率直接拼接并继续使用同一个 `dt`。

准备程序从部署清单的 `source_files` 找回原始完整 train＋val文件，恢复部署划分曾留出的上下文帧，核对完整帧数、每帧六路物理动作、原生采样间隔、四路有效通道 `[0,1,3,5]` 及耦合关系。统一动作除以平台固定150 kPa；骨架保存为 `(T,15,3)`、机器人平面坐标毫米、`base_to_tip`。相机坐标通过继承的固定采集日标定变换到模型坐标，mask投影额外扣除ROI左上角。

每折按排序后的完整序列留一条test，紧接的一条val，其余五条train；窗口在分组后构造，跨序列上下文不复用，序列开头不足历史长度的帧不计分。每份清单包含NPZ、原始文件与mask库存哈希，逐帧mask编号，裁剪元数据、原生 `dt` 与实际时间戳抖动。训练入口只打开train/val数组；输出归一化、HOV样条参考、段长、基与尺度均由train拟合。

这些序列及历史标定此前参与过项目开发，清单标为 `cross_sequence`、`exploratory_historical_corpus`。它们支持现有采集日内的分组对比；论文最终泛化结论需要新增冻结采集及独立标定。时间更新使用名义采样周期，实际时间戳抖动作为限制报告。SAM2轮廓与从其提取的骨架是相关标签，末节点误差也来自同一骨架标签，不能当作独立 NDI 测量精度。

## 3. 公平训练与历史语义

默认共同任务为 `U[t-H+1:t] → S[t]`，5 Hz下 `H=20` 覆盖首末动作间3.8秒，包含20个动作采样。`K_train=K_eval=1`，每个窗口重新初始化；模型看不到当前图像、当前骨架和窗口外状态。HOV在首个动作处按平衡态初始化，随后每个动作只消费一次。方向网络在持压阶段沿用窗内末次非零方向，窗口起点未知方向为零。GRU/LSTM零初始化；振子在首动作处初始化平衡；这些差异公开记录。

默认共同监督为归一化坐标 MSE 加 `0.25 × 末节点坐标MSE`。HOV的冻结参考还包含训练集先验拟合阶段，使用与其他方法相同的训练窗口目标帧，其预算500步单独记录。此基准不调用部署训练器的广义坐标辅助损失。默认Adam、学习率0.001、100 epoch、batch128，不提前停止，每个epoch用val上“逐序列平均骨架欧氏误差”的宏平均选择最小值checkpoint。线性/多项式按同一训练窗口目标闭式拟合，静态参考完成先验拟合后评价。

固定窗口实验检验**有限历史预算**下的建模精度。若论文另报告跨整个序列持续携带状态的模型，应另设连续状态训练/评价协议；不能把本入口的滑动窗口预测称为全程状态rollout。窗口长度可用10/20/40做敏感性对照；各模型在一个比较表里需使用相同H、采样周期和计分帧协议。

100 epoch等默认配置是可运行的起始预算。正式对照应在val上给予各模型相当的学习率、容量与收敛检查机会，再冻结配置。HOV先验拟合及少参数的优势不能靠尚未收敛的MLP短训练来证明。

## 4. 指标与统计

- 骨架：每帧15个对应节点欧氏距离均值 `mean_node_mm`；所有节点/帧欧氏误差的RMS `node_rmse_mm`；逐帧最大节点误差的均值 `max_node_mm`；所有节点误差的95分位 `node_p95_mm`；离散节点集双向最近距离均值 `chamfer_mm`。对应节点误差为主，Chamfer不能代替节点语义。
- 末端：第15点欧氏距离均值 `endpoint_mm`、95分位 `endpoint_p95_mm`。不对预测做测试标签驱动的刚体对齐。
- 整体轮廓：共同的固定半径平面管状投影，对真实SAM2 mask计算IoU、Dice、precision、recall与边界F1。默认物理半径8 mm，来自平台16 mm名义直径；边界容差2 px。全部方法使用相同半径和投影，不逐测试帧拟合宽度。只有双方均为空时IoU/Dice等记1；单方为空记0，失败预测不能丢弃。
- `label_tube_consistency`：把**标注骨架**通过同一管状适配器投影后与真实mask比较，帮助识别骨架标签和固定宽度表示的误差。它是标注/表示诊断，不是模型成绩，也不是严格数学上界。

mask结果衡量“骨架建模＋共同轮廓读出”。这些模型没有直接学习全身宽度或三维表面，论文应按此解释。输出逐帧预测、标签、计分帧号和指标到NPZ；逐序列记录到JSON，汇总输出 `per_sequence.csv`、`summary.csv` 与 `statistics.json`。

验证集用于选模，其统计汇总标为开发诊断，`significant` 强制为false。冻结测试评价才允许根据统计结果作显著性判断。统计先按同一测试序列对不同随机种子求平均，再对各模型与 `hov` 的序列均值作配对；全程等序列权重。不同模型缺少某个序列/种子、重复记录、采用不同划分或评价合同都会被拒绝。报告模型减参考的均值差、配对序列bootstrap 95% CI、双侧符号翻转置换p值、Holm校正p值，以及差值尺度上的效应。CI是逐对区间，没有进行同时区间校正。`summary.csv` 额外给出种子宏平均的样本标准差及序列差异，种子标准差不代表独立采集的不确定性。

少于6组时标为 `inconclusive`。即使使用全部7条序列，双侧精确检验最小p值也是 `2/2^7=0.015625`；20个模型对照一起Holm校正时，最小可能首项校正值为0.3125，无法据此获得0.05水平的显著差异。更多重复种子不会改变这一限制。应预先指定主指标/主比较，并增加独立采集序列；多指标检验将进一步扩大比较族。LOSO训练集之间重叠、所有序列来自同一天也限制了置换/重采样的独立性解释，最终需用新增固定训练集与独立测试序列验证。

## 5. 运行命令

依赖沿用现有环境：Python、NumPy、PyTorch、SciPy、OpenCV；官方GRU/LSTM只依赖PyTorch。本次验证使用CPU，CUDA运行需要在相应环境另核对确定性算子支持。

### 5.1 准备一折

```bash
python scripts/experiments/modeling_benchmark.py prepare \
  --source-manifest workspace/runs/training/hov_full_native_20260910_000/dataset_5hz/dataset_manifest.json \
  --out workspace/data/derived/modeling_5hz_fold0_v1 --fold 0
```

`--fold`可取0至6；创建不同目标目录即可得到其余折。每条序列作为test恰好一次，汇总时选取每折的test评价，不混入val评价。

### 5.2 训练、验证与测试

```bash
python scripts/experiments/modeling_benchmark.py train \
  --manifest workspace/data/derived/modeling_5hz_fold0_v1/dataset_manifest.json \
  --config docs/paper/icra2027/modeling_benchmark_config.json \
  --model chen_direction --seed 0 \
  --out workspace/runs/modeling/chen_direction_fold0_seed0_v1

python scripts/experiments/modeling_benchmark.py evaluate \
  --run workspace/runs/modeling/chen_direction_fold0_seed0_v1 \
  --role val --out workspace/reports/modeling/chen_direction_fold0_seed0_val_v1
```

训练完成后再冻结配置，独立执行 `evaluate --role test`，输出到新的test目录。测试阶段校验选中的checkpoint及数据清单哈希。`--skeleton-only`可只做骨架评价；完整论文轮廓表使用默认mask开启。`--mask-stride`默认1；若为性能原因固定稀疏采样，所有模型必须一致且报告实际mask帧数。所有序列启动时不足H帧的部分统一排除。

### 5.3 多模型、多随机种子与多折

```bash
python scripts/experiments/modeling_benchmark.py sweep \
  --manifests workspace/data/derived/modeling_5hz_fold0_v1/dataset_manifest.json \
  --config docs/paper/icra2027/modeling_benchmark_config.json \
  --models linear polynomial2 mlp window_mlp chen_static chen_direction park_tcn gru lstm bezier_gru oscillator hov \
  --seeds 0 1 2 3 4 --out workspace/runs/modeling/main_comparison_v1
```

此命令写出可审阅的 `sweep_plan.json`。在首次运行时加 `--execute` 即按计划训练并评价；现有计划目录不用于覆盖执行，应换新目录。`--manifests`后可列出全部7折；结构消融用同样命令选择 `hov hov_no_play hov_no_maxwell hov_static hov_linear_reference hov_pod8 hov_memory_residual`。默认评价val，方案冻结后的计划可显式指定 `--evaluate-role test`。每个run保存准确配置、Git状态、源码快照、先验、标准化、验证曲线和selected checkpoint。

```bash
python scripts/experiments/modeling_benchmark.py aggregate \
  --evaluations workspace/runs/modeling/main_comparison_v1/*/evaluation_val \
  --reference hov --metrics mean_node_mm endpoint_mm mask_iou mask_dice \
  --out workspace/reports/modeling/main_comparison_val_v1
```

此示例汇总开发集，标题和评价角色会保留。正式表改用冻结的 `evaluation_test` 输出。全指标一次校正与预先指定主指标的比较属于不同统计方案，不能看完p值后挑选显著的方案。

### 5.4 快速接线检查

```bash
python scripts/experiments/modeling_benchmark.py sweep \
  --manifests workspace/data/derived/modeling_5hz_fold0_v1/dataset_manifest.json \
  --models mlp chen_direction park_tcn gru bezier_gru oscillator hov hov_no_play hov_no_maxwell \
  --seeds 0 1 --run-kind smoke --epochs 2 --prior-steps 2 \
  --max-train-windows 32 --max-eval-windows 8 --batch-size 16 \
  --out workspace/runs/modeling/smoke_v1 --execute

python -m unittest tests.test_modeling_benchmark tests.test_modeling_benchmark_metrics -v
```

窗口截断仅允许明确的 `smoke` 运行。短检查中的先验与主训练均使用选中的训练目标帧；少量优化步数仅验证接口，不能比较模型优劣。统计汇总smoke需 `--allow-smoke`，报告保留诊断标签。

## 6. 论文实验的递进关系

**E1：全身形态建模主表。** 固定划分与预算，对基础映射、方向特征、通用历史编码器、近期结构适配和HOV进行比较。填写骨架均值/RMS、末端均值/P95、mask IoU/Dice/边界F1、参数量。结果为 `【待填：各方法跨序列均值、种子变异、差值及CI】`。待检验假设是历史信息能降低方向条件静态映射无法解释的形态误差；若Chen方向网络已充分解释数据，应承认当前工况对更长记忆的区分度有限。

**E2：显式记忆的结构消融。** 重训练play关闭、Maxwell关闭、静态参考等变体，其余协议固定。通过同一指标与配对统计量检验各项贡献。`【待填：迟滞分支、时间相关分支与完整模型的差值】`。将差异与压力循环、持压时间等工况联系起来；单纯全局误差下降不能证明具体物理机制已被辨识。

**E3：几何表示和剩余误差。** 比较局部14维、POD8、记忆残差、GRU直接节点头与Bézier头，结合 `label_tube_consistency` 检查轮廓误差是否受固定宽度影响。`【待填：表示压缩与几何读出的精度/参数折中】`。POD8包含容量变化，Bézier头包含固定基点及曲线先验，应把这些差异写清。线性参考按广义坐标做岭拟合，样条参考按骨架及末端几何损失进一步优化，因此 `hov_linear_reference` 同时改变函数族与拟合目标，属于参考设计对照。

**E4：历史长度敏感性与重复实验。** 用H=10/20/40重新训练同一主对照集合，报告同一评价范围内的精度变化以及不同种子变异。当前窗口起始计分点随H改变；若跨H直接比较数值，应先从逐帧NPZ中取共同帧号，或统一选择相同计分起点。比较较短/较长窗口是否有稳定收益，`【待填：窗口长度—误差曲线及CI】`。通用 `tcn` 的有效感受野固定为15帧，H超过15不增加其有效信息；历史长度主对照采用随H扩展感受野的 `park_tcn`、窗口MLP、GRU和HOV。长期持续状态的评价应另设协议。

各表先报告实测结果，再根据误差变化修改方法贡献与总结。以上是待检验假设和可运行实验设计，代码短检查不填入论文性能表。
