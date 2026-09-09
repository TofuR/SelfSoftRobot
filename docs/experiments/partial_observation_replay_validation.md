---
title: 固定图像遮挡下的 hereditary 串行回放验证
kind: experiment
status: active
updated: 2026-09-09
scope: development-sequence image injection, same-input prediction, and pre-hardware suffix optimization
sources:
  - ../designs/2026-09-08_partial_observation_hereditary_control.md
  - ../../scripts/evaluation/replay_partial_observation.py
---

# 固定图像遮挡下的 hereditary 串行回放验证

2026-09-08 用户授权：先提交既有修改，再建分支；在已有序列上按“初始规划 → 假设执行 → 注入原采集图像 → 更新状态 → 优化剩余压力”验证流程。反馈图像中机器人中下部使用固定像素位置的小纯色方块覆盖，选择运动变化较大的窗口。

既有修改保存于 `7b00b55`；开发分支为 `feat/partial-observation-replay`。本记录对应离线原型；硬件与采集程序不由此脚本调用。

## 1. 本轮实现与输入约定

- `HereditaryGeometryModel.step_state(action,state)` 消费一个模型时间步；`observe_state(action,state)` 只读已推进状态。原 forward/训练接口和 checkpoint 参数键兼容。
- `src/control/hereditary_feedback.py`：固定对角先验的鲁棒状态校正，以及 B 方案的一次当前 rollout/Jacobian 和受约束二次优化。状态必要边界、有限修正、固定关联下非线性接受检查均保留。
- `src/control/partial_image.py`：在预测臂侧搜索白色臂身到较暗背景的边缘过渡；采用局部法向、外观、方向及歧义拒绝。固定半径距离残差对状态可导。自动模式不读取合成遮挡 mask 或参考骨架。
- `src/evaluation/partial_replay_data.py`：开发 NPZ、split manifest、原始动作、command id、样本时间和原图路径逐行核对；不读取 reserved test NPZ。
- `scripts/evaluation/replay_partial_observation.py`：选段、固定遮挡、初始规划、两条串行回放、保存状态/后缀/时延及交互 HTML。

动作从 NPZ 的逐通道尺度恢复 kPa，与原始记录核对后除以 checkpoint 的 `action_norm_factor` 进入模型。不能遗漏后一个归一化因子。范围及 rise/fall 来自原采集 meta；模型只处理四个有效通道，报告展开为六通道。

初始状态：按 checkpoint 的 equilibrium 约定初始化，再重放所选窗口之前、dev NPZ 已存储的动作上下文。本轮不声称重建了采集开始前的真实历史，也未增加初始视觉拟合。

## 2. 选段和固定遮挡

固定窗口长度 80 帧。选段指标为：对窗口内每个非基座节点计算二维坐标的时间标准差范数，再对节点取均值；选最大指标窗口。指标只使用录制的参考形状，在反馈运行前确定，不按校正改善量选择。

开发数据预检查：

| 序列 | 最强 80 帧窗口指标 | 同序列候选窗口中位数 |
|---|---:|---:|
| `seq_20260819_182253` | 2.746 mm | 2.121 mm |
| `seq_20260819_182519` | 6.179 mm | 4.686 mm |

本轮使用 `182519`：dev 局部 `[271,351)`，对应原图 **2198–2277**。指标是形状变化量，不是模型误差或运动速度。

方块以窗口首帧参考骨架约 65% 臂长处为中心定位一次：原始 640×480 图像中的 `xywh=[350,222,56,56]`，BGR `[35,35,35]`。每帧直接覆盖原像素；位置、大小、颜色保持不变，不随臂移动。使用首帧参考仅用于离线布置遮挡；自动边缘校正器不接收它。相同矩形还用于独立评分中的隐藏节点划分，基座不计分。

## 3. 两条回放的区别

### 假设执行与图像注入：检查优化顺序

1. 把记录中的未来骨架显式指定为任务参考轨迹，记录动作作为初始优化种子。
2. 从当前历史状态优化完整 80 步压力；输入范围和变化率约束先投影到可行种子。
3. 假设执行当前工作序列首条压力，并用该条压力推进模型一次。
4. 读取对应原图，覆盖固定方块，提取可见边缘并更新 p、h。
5. 使用同一个旧后缀分别保存状态校正前、后的未来形状预测。
6. 从校正后的状态出发，用 B 修订整个剩余后缀；下一帧执行新后缀首条。

每次最多 8 个动作块覆盖全部剩余动作；四通道对应至多 32 个优化变量。动作块内采用相同增量，原始序列形状仍可逐步变化。二次问题使用 SciPy SLSQP，固定 Hessian 和线性约束；最多 80 次内部迭代，不是每步多次非线性重规划。初始规划允许最多 8 次局部迭代。

新后缀以绝对值替换，只保留未执行部分。保存 `future_before_state`、`future_after_state`、`future_after_control`，可检查第一重怎样影响第二重。无有效测量时只推进有效计划；候选失败则保留旧后缀。

原图由旧采集动作产生，新压力没有相应的真实响应图像。因此此流中的“预测与原图差异”只叫注入失配，不能称新控制器的真实误差。未来参考来自已录制骨架，也属于预先给定的离线任务，不是未见目标泛化。

### 原采集动作对照：检查状态更新的预测价值

独立维护持续不校正、逐帧校正两份状态，始终消费原采集动作。统计当前全身、矩形内隐藏节点误差。每次校正前后各分出副本，使用随后相同的真实动作，在无新增测量下预测 1/5/10 步；主估计器继续正常接受下一帧图像。

比较 `edges` 与 `nodes`：前者使用遮挡原图自动边缘；后者直接选取矩形外具有身份的参考节点，属于理想测量对照。`--oracle-visibility` 可对边缘组明确提供遮挡真值，只作为另一个上界，默认关闭。nodes 模式报告中的青色边缘仅是图像诊断，实际状态更新使用已知可见节点。

## 4. 配对审计与解释范围

- dev NPZ 包含原始帧 `[1927,2459)`；逐行恢复压力与 actions6.csv 最大偏差约 `1.14e-5 kPa`，command id 与 ACK 内容一致。
- 模型毫米坐标转回原图与保存的 `positions_camera_px` 最大偏差约 `7.63e-6 px`。
- 审计段无跨命令时间配对；原图主机接收估计为 `t_grab-frame_age0`，不是曝光时间。
- 实际 command 间隔中位数约 `0.109 s`，P95 `0.125 s`，最大 `0.203 s`；checkpoint 步长 `0.1 s`。本原型每条记录推进一个固定模型步，不把这称为已解决时序失配。
- 坐标变换来自历史序列统计，固定用于已有数据回放，不是任务前独立标定的实机因果证据。
- 评分来自离线 SAM2 中心线伪参考；这是一个物理序列的开发窗口，帧数不等于独立实验重复数。
- 固定半径、白色侧边前端、局部搜索尚不覆盖任意外观、长时丢失后的全局重定位、接触和宽度明显变化。
- 本轮采用串行模型时间，没有模拟计算期间继续下发动作；计算时延如实记录，后续还需延迟重放、冻结前缀、候选版本及真实截止时间测试。

## 5. 复现命令与产物

在项目环境和仓库根目录执行，每次使用一个不存在的新 `--out`：

```bash
MPLCONFIGDIR=/tmp/ssr-partial-replay-mpl python scripts/evaluation/replay_partial_observation.py \
  --checkpoint workspace/runs/training/hov22_formal/hov22_local14_unit_balanced_s42_20260905_000/phase_hereditary_geometry/model/best_eval_model.pt \
  --data workspace/data/derived/ishsm_v1_20260902_000/dev/seq_20260819_182519_train.npz \
  --raw workspace/data/raw/real/seq_20260819_182519 \
  --frames 80 --occluder-size 56 \
  --out workspace/runs/analysis/partial_observation_replay/edges_YYYYMMDD_NNN
```

同条件节点对照增加 `--measurement nodes`，使用另一新目录。`--start` 是 dev NPZ 局部索引，不是原图编号；默认自动选最大变化窗口。

| 产物 | 用途 |
|---|---|
| `report.html` | 自包含滑块回放：原图/遮挡图、本帧状态、整个剩余压力、终点预测变化 |
| `overview.png` | 原采集动作对照误差、隐藏误差、压力差异和计算时延 |
| `pairing.csv`、`data_audit.json` | 原图、命令、时间、坐标、数据谱系与近似说明 |
| `window_candidates.csv` | 全部候选窗口的运动指标 |
| `initial_planning.json` | 初始规划的每次局部求解、接受与残差 |
| `steps.json`、`metrics.csv` | 每帧测量、状态修正、后缀接受、求解信息、耗时 |
| `traces.npz` | 状态前后、三类未来预测、每次完整旧/新后缀、初始/录制/假设执行动作 |
| `pressures.csv` | 三类压力各自展开后的六通道值，用于对照，不直接作为硬件命令 |
| `config.json`、`commands.sh`、`run_manifest.json` | 可复现参数、模型选择及代码状态 |
| `COMPLETE` | 全部预期计算与报告成功完成，含义是回放完成 |

## 6. 2026-09-08 原型结果

两组 80 帧回放均已产生 `COMPLETE`：

- 自动图像边缘：[交互报告](../../workspace/runs/analysis/partial_observation_replay/edges_20260908_001/report.html)、[机器结果](../../workspace/runs/analysis/partial_observation_replay/edges_20260908_001/summary.json)。
- 已知可见节点：[交互报告](../../workspace/runs/analysis/partial_observation_replay/nodes_20260908_000/report.html)、[机器结果](../../workspace/runs/analysis/partial_observation_replay/nodes_20260908_000/summary.json)。

沿原采集动作的同输入预测结果，单位 mm；“前→后”指本帧校正前后，之前帧已经接受的校正保持不变：

| 指标 | 自动图像边缘 | 已知可见节点 |
|---|---:|---:|
| 持续不校正，全身平均误差 | 1.382 | 1.382 |
| 本帧全身误差，校正前→后 | 1.291 → 1.044 | 1.168 → 0.700 |
| 当前隐藏节点误差，校正前→后 | 1.566 → 1.259 | 1.469 → 0.923 |
| 随后 1 步，同输入无新增观测预测 | 1.387 → 1.295 | 1.340 → 1.170 |
| 随后 5 步，同输入无新增观测预测 | 1.340 → 1.325 | 1.390 → 1.366 |
| 随后 10 步，同输入无新增观测预测 | 1.352 → 1.356 | 1.414 → 1.422 |

未来 1/5/10 步分别有 79/75/70 个起点。当前及短期预测改善，10 步没有改善；这不支持长期状态充分性或长时无视觉收益。节点上界比边缘组的当前误差更低，说明当前图像测量及其模型仍留有改进空间；两组的长时改善都弱，后续不能仅通过加强视觉拟合来宣布解决了未来预测。

假设执行/图像注入流程：

| 指标 | 自动图像边缘 | 已知可见节点 |
|---|---:|---:|
| 接受的状态更新 | 80/80 | 76/80 |
| 接受的剩余后缀更新 | 79/79 | 76/79 |
| 假设下发与原采集压力的逐步最大通道差，时间均值 | 20.54 kPa | 23.39 kPa |
| 该通道差的全段最大值 | 44.00 kPa | 57.23 kPa |
| B 后缀计算 P50 / P95 | 802.7 / 1537.9 ms | 813.5 / 1511.9 ms |
| 假设下发序列的范围/速率/接续验证 | 通过 | 通过 |

最后一帧没有剩余动作，所以后缀最多更新 79 次。两组使用同一初始规划，8 次局部更新均被接受，模型内坐标 MSE 从可行种子的 1.550 降到 0.938 mm²；这只是规划残差。两条运行以 CPU 单线程各自执行，运行期间有并行回放进程，时延是本次离线记录，不是隔离部署基准。即使如此，当前实现显然尚不能作为已通过 100 ms 时限的实时控制器。

新动作与采集动作逐渐相差数十 kPa，是图像注入实验需要显式展示的失配：压力修订没有改变随后送来的图像。不能将原录制轨迹当成新动作的真实闭环结果，也不能以该流压力变化多就证明控制有效。

首个调试目录 `edges_20260908_000` 暴露了饱和限速处的 QP 数值问题：动作块内部的速率导数为零，float32 存储造成约 `1e-8` 误差后，无作用约束变成不可行。修复在已有可行性容差内核对并移除这些恒定行；所有能受更新影响的范围/速率约束继续保留。增加对应回归测试。该目录保留为故障诊断，不作为修复后 A/B 或控制收益依据。

## 7. 检查和后续边界

本轮 55 项聚焦测试通过；文档治理与 `git diff --check` 通过。两组完整报告均检查了 JSON 有限性、JavaScript 语法及逐步后缀消费关系；读取报告图确认固定方块确实覆盖原像素而不跟随机器人。

聚焦测试：

```bash
python -m unittest tests.test_partial_observation_replay tests.test_hereditary_geometry_model tests.test_hereditary_model tests.test_hereditary_operators -v
python scripts/maintenance/check_docs_governance.py
git diff --check
```

覆盖非推进读出、训练 forward 等价、Jacobian 数值差分、空测量、状态边界、候选不污染输入、全后缀覆盖、限速接续与饱和浮点误差、合成截断边缘、选段边界、动作/坐标/时间配对。

截至 2026-09-08 已实施 B 和离线状态校正；A 的缓存响应当时待实施，2026-09-09 进展见下节。输出偏置/合理重初始化等完整消融、一般自动重定位、异步执行器后缀替换及实机试验仍待后续开发。固定先验校正未维护校准协方差，不能称完整 EKF。

## 8. 2026-09-09 反馈计算加速

用户要求完整反馈计算至少低于 200 ms、优先低于 100 ms，并实际比较方案 A、批处理、多线程和 GPU。本轮保持 checkpoint、80 帧高变化窗口、固定方块、初始规划及任务参考一致。主回放现在默认 `--controller fast_b --detector vectorized`，也可显式选 `torch_b`、`batched_b`、`cached_a`；原型历史命令若需重现旧实现应指定 `--controller torch_b --detector scalar`。

### 8.1 遮挡图像怎样提供误差

目前提取的是可信的局部侧边测量。输入只有遮挡后的 RGB 和当前模型预测中心线，默认不把遮挡矩形、原图骨架或 SAM2 结果交给检测器。

1. 图像灰度化、3×3 模糊、Sobel 梯度，并计算 HSV 外观。
2. 在预测中心线每段的 1/4、3/4 位置沿两侧法向搜索。固定臂半径约 10 px，侧边位置附近搜索 ±12 px；排除基座和末端帽。
3. 要求臂内偏白且亮、臂外较暗，梯度方向与侧面法向一致；多个近似同强候选时拒绝关联。方块内部通常不产生有效侧边；横向截断边缘通常被方向门限拒绝。当前方法不保证识别任意遮挡物。
4. 接受边缘位置后，在本次状态求解期间固定段关联。残差为 `r=(边缘点到预测中心线段的距离−半径)/2 px`，沿该残差对 `p,h` 求导，结合稳健权重和状态先验更新，再检查真实非线性残差是否下降。遮住部分由模型状态推断，不作为观察到的完整骨架。

向量化版本批量读取候选像素与梯度，保持原筛选/去重顺序。核对同一归档预测下全部 80 张遮挡图，接受位置、段身份、得分与标量版本完全一致。开发时发现 NumPy 整数偏移会把 float32 坐标升级到 float64，改变半像素舍入；现已保持坐标 dtype，并加入回归测试。

### 8.2 计算方法和测量结果

主要瓶颈是原 B 在每个未来时间步重复几何读出及自动微分。实现了三种改进：

- `batched_b`：PI/Maxwell 按时间递推，把整个 horizon 的几何读出批量计算，仍使用 PyTorch 自动微分。
- `fast_b`：把冻结 HOV2.2 参数导出到 NumPy，显式传播当前分段光滑分支上的状态/动作导数，批量计算几何和各导数方向；合并 QP 中重复约束行并取上下界交集。保留全后缀、至多 8 个动作块、原正则/信赖范围、压力范围/变化率和非线性下降检查。本适配器只支持 `residual_mode=none`，遇到神经残差明确拒绝。
- `cached_a`：在初始计划各边界预计算状态响应、全动作响应及正则化逆映射。在线同时计入 `当前状态−名义状态` 和 `上次修订后缀−名义后缀`，避免重复补偿；直接映射后投影压力范围/变化率，再用真实非线性 rollout 检查。缓存模型失配 RMS 超过 0.5 mm 时保留旧计划；没有把 B 回退或缓存刷新隐藏在 A 的计时外。

CPU 为 Intel Xeon Platinum 8336C @ 2.30 GHz；每方法串行独立执行，OpenCV、Torch、BLAS 单线程，完整流程预热两次。统计 79 个具有剩余动作的反馈周期，终帧不参与延迟统计。

| 方法 | 完整计算 P50 | P95 | P99 | 最大 | ≥100 / ≥200 ms 次数 | 接受后缀修订 |
|---|---:|---:|---:|---:|---:|---:|
| 原 B，统一使用向量化图像前端 | 766.5 | 1394.5 | 1469.5 | 1533.4 | 75 / 70 | 79/79 |
| 批量读出 B | 121.4 | 222.0 | 289.9 | 298.2 | 49 / 7 | 79/79 |
| **显式导数 B，默认** | **39.8** | **48.1** | **50.9** | **54.6** | **0 / 0** | **78/79** |
| 预计算 A | 22.1 | 27.7 | 30.7 | 33.8 | 0 / 0 | 16/79 |

单位均为 ms。计时是一个连续墙钟区间：内存原图→覆盖方块→模型预测→图像提取→状态校正→剩余动作优化→压力约束验证。文件读取、评分绘图、初始规划、模型加载及 A 预计算不在在线区间；相机曝光/图像年龄、通信和阀门 ACK 未测量。不能把第 6 节包含诊断工作的整步时延与此表直接比较，也不能把 79 帧观测最大值视为硬实时上界。

显式 B 各阶段 P50 为：预测 2.23、图像 8.85、状态校正 7.91、控制及校验 20.55 ms。端到端 P95 直接从完整周期计算，不能把阶段 P95 相加替代它。CPU 表来自 `cpu_20260909_001`；`cpu_20260909_000` 为像素舍入修复前的开发测量，保留作诊断。

**独立启动复测的抖动也必须计入结论**：最后补齐源配置中动作块数量的传递后，以相同单线程参数单独运行 B/A（`cpu_dispatch_check_20260909_000`）。动作及状态与上表逐值一致；B 的 P50/P95/P99/最大为 **50.9/58.5/80.4/139.2 ms**，1/79 超过 100 ms，0/79 超过 200 ms。最慢是 k=7，控制器 117.8 ms，其中 QP 求解 96.8 ms；本次未进一步区分求解内部工作、首次警告输出和主机调度所占比例。A 的 P95/最大为 26.5/28.3 ms。不能只选择前一轮最大 54.6 ms 宣称每帧稳定小于 100 ms：当前证据支持计算 P95 小于 60 ms、这些试次均小于 200 ms，但仍有超过 100 ms 的峰值。

A 的全边界预计算耗时 1415.7 ms、缓存约 206.2 MB；在线虽然快，但 63 次未接受，其中 7 次缓存失效，其余候选在投影及非线性校验后没有下降。当前 A 的失败机制包括受约束投影改变补偿方向，以及工作轨迹离开名义轨迹；低延迟不能抵消这些限制。因此目前推荐显式 B，A 保留为对照，后续可尝试保持可行方向的缓存 QP 或后台局部刷新，并重新计入刷新/切换开销。

### 8.3 多线程、GPU 与数值差异

- **四线程 CPU**：相同显式 B 全流程 P50/P95/最大为 51.0/58.5/125.8 ms，1/79 超过 100 ms；相同输出，未优于单线程。对于本例的小矩阵和有时间依赖的递推，增加线程没有带来收益；已使用的批量几何/导数计算仍然保留。
- **GPU**：在 GPU 3 RTX 3090 上测试批量 B，同步后计时并包含 CPU/GPU 传输、CPU QP 与候选校验。horizon=79、39、9 的两次预热后控制器耗时分别为 370.1–376.3、216.5–220.0、87.8–100.6 ms；首次 CUDA 预热 2530 ms。所有卡当时接近满载，测试卡前后 99–100%，这是共享繁忙 GPU 的有限探测，不能据此判断空闲 GPU 性能；也不能把仅控制器的数值当作完整反馈延迟。
- **相同快照核对**：在 k=0/1/10/14/15/30/50/70，用同一状态及后缀比较原自动微分与显式导数。形状最大差约 `2.72e-5 mm`，Jacobian 相对 L2 差不超过 `1.91e-7`，候选压力最大差 `5.63e-5 kPa`。k=70 两者均拒绝没有非线性下降的候选。
- **连续轨迹不完全等价**：批量 B 相对原 B 的已执行压力最大差 0.00574 kPa；显式 B 的平均绝对差 2.20 kPa、最大 18.68 kPa。首次明显的后缀分歧在 k=14：先前约 `1e-6` 量级状态/后缀差触发了 1 个 play 分支不同，两份工作快照的 Jacobian 相对差达到 0.549；分别固定任一快照时，两种求导/求解实现又一致。已定位到非光滑分支对微小数值扰动的敏感性，不能声称整段动作逐位等价或所有部署条件下性能不变。

所有方法的压力约束检查通过。动作注入失配均值原 B 0.714、显式 B 0.706、A 0.747 mm，只记录作流程诊断，不作为新压力实际控制精度或优越性证据。主脚本完整重跑 `fast_b_20260909_000` 后，同原采集动作的事实预测评分保持第 6 节数值：1.382→1.044 mm；隐藏区域本帧 1.566→1.259 mm，10 步无改善。

### 8.4 复现、产物与下一步

```bash
MPLCONFIGDIR=/tmp/ssr-partial-replay-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/evaluation/benchmark_partial_feedback.py \
  --source-run workspace/runs/analysis/partial_observation_replay/edges_20260908_001 \
  --out workspace/runs/analysis/partial_observation_latency/cpu_YYYYMMDD_NNN

# 只比较线程数，使用另一新目录
MPLCONFIGDIR=/tmp/ssr-partial-replay-mpl python scripts/evaluation/benchmark_partial_feedback.py \
  --source-run workspace/runs/analysis/partial_observation_replay/edges_20260908_001 \
  --methods fast_b --threads 4 \
  --out workspace/runs/analysis/partial_observation_latency/threads4_YYYYMMDD_NNN

MPLCONFIGDIR=/tmp/ssr-partial-replay-mpl python scripts/evaluation/summarize_partial_feedback.py \
  --run workspace/runs/analysis/partial_observation_latency/cpu_20260909_001 \
  --out workspace/runs/analysis/partial_observation_latency/report_YYYYMMDD_NNN

python -m unittest tests.test_hereditary_fast_feedback tests.test_partial_observation_replay \
  tests.test_hereditary_geometry_model tests.test_hereditary_model tests.test_hereditary_operators -v
```

- [四方法耗时图表与自包含报告](../../workspace/runs/analysis/partial_observation_latency/report_20260909_000/report.html)、[CPU 机器结果](../../workspace/runs/analysis/partial_observation_latency/cpu_20260909_001/summary.json)、[独立启动复测报告](../../workspace/runs/analysis/partial_observation_latency/report_restart_20260909_000/report.html)。
- [四线程结果](../../workspace/runs/analysis/partial_observation_latency/cpu_threads4_20260909_000/summary.json)、[GPU 结果](../../workspace/runs/analysis/partial_observation_latency/gpu_20260909_000/summary.json)。
- [显式 B 主流程交互回放](../../workspace/runs/analysis/partial_observation_replay/fast_b_20260909_000/report.html)。
- 每方法保存逐帧阶段计时、接受理由、状态、完整压力后缀；CPU 根目录另存 `same_snapshot_audit.json` 和 `first_divergence_audit.json`。

63 项聚焦测试通过，涵盖既有模型/算子、非推进观察接口、边缘一致性、显式形状/导数、动作约束、A 重复补偿与失效拒绝。主回放和专用基准均完成。本轮开发服务器上的 B 反馈计算均低于 200 ms，P95 低于 60 ms；保证每帧低于 100 ms 仍需收紧 QP/调度抖动。上实机还需把曝光/帧龄、时序重放、通信、冻结前缀和过期候选拒绝纳入整周期测量。下一步优先补齐这些执行时序和尾部延迟，而非单凭换用 GPU 推断实时性。


### 8.5 Cached A 与 Analytic B 的终帧差异核对（2026-09-09）

直接读取 `cpu_20260909_001/{cached_a,fast_b}/traces.npz` 的 `shapes`，与源回放 `edges_20260908_001/traces.npz` 的 `reference` 比较。使用与原逐帧评分相同的口径：图像校正后的模型形状、排除固定 base、节点欧氏距离取均值。

| 诊断量（mm） | A | Analytic B |
|---|---:|---:|
| 整段平均 | 0.747380 | 0.706442 |
| 最后一帧平均节点距离 | 0.572350 | 0.366819 |
| 最后一帧 tip 距离 | 1.069636 | 0.171206 |

在这个窗口 B 的差异更小。16/79 的接受次数不意味着 A 失去全部反馈：A/B 都继续执行图像状态校正，A 拒绝动作候选时保留旧合法后缀。原图不响应新压力，此表不是实机闭环到位精度，亦不能仅凭一个开发窗口概括方法优越性。

当前工作台已按四步顺序重构，操作与虚拟设备证据见 [接入记录](hereditary_real_validation_integration.md)。

[终帧比较机器结果](../../workspace/runs/analysis/partial_observation_latency/accuracy_20260909_000/metrics.json) 保存上述口径与来源。
