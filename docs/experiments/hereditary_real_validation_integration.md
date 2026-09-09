---
title: Hereditary Analytic B 工作台接入与虚拟设备验证
kind: experiment
status: active
updated: 2026-09-10
scope: portable GUI, pressure mapping, alignment, timed state history, and virtual-device feedback
sources:
  - ../../real_validation/HEREDITARY_GUIDE.md
  - partial_observation_replay_validation.md
---

# Hereditary Analytic B 工作台接入与虚拟设备验证

用户要求将当前反馈方法接入 `real_validation`，提供模型加载、六腔到四输入映射、部署对齐与预热、部署后的动作历史、画笔全形状目标、规划和执行。随后确认所指 `-max` 是实际负压；当前服务器没有设备，也没有访问实机电脑的远程入口，因此本轮按用户授权使用虚拟设备，保留真实相机/阀控接口。

详细操作和解释以 [工作台实验指南](../../real_validation/HEREDITARY_GUIDE.md) 为准。当前分支 `feat/partial-observation-replay`，前序快照为 `7b00b55`；本轮不启动训练、不发送真实硬件指令。

## 2026-09-10：按控制周期丢弃迟到反馈、逐步审计与规划参数

当前 [带参数弹窗的图解](../../workspace/runs/analysis/hereditary_deployment/deadline_20260910_002/guide.html)、[机器结果](../../workspace/runs/analysis/hereditary_deployment/deadline_20260910_002/summary.json)、[独立部署包](../../workspace/runs/analysis/hereditary_deployment/SelfSoftRobot_Hereditary_workbench_20260910_003.zip)。

原固定 200 ms 返回后检查不能撤销已经修改的 p/h。现在反馈线程只操作独立快照；截止时间为实际发令时间 + 模型 dt − 3 ms，按时且版本未变化才整体提交状态、历史和后缀。迟到任务永远不能写实时运行状态，最多一个任务，忙时跳过，不排队。默认连续 10 次跳过后归零（第四页可调）；无图/无可信边缘仍默认连续 3 次停止。新帧必须晚于 ACK，额外等图默认 0 ms，可配置但必须小于 dt。

第一轮 `deadline_20260910_000` 保留了固定 50 ms 等图，常只剩 10–30 ms 计算预算，触发连续超时停止；该失败记录保留。取消固定等待之后，`001` 为 40 步、38 次按期提交、2 次超时；最终 `002` 为 40 步、39 次按期提交、1 次超时，其中 37 次实际采用 B 后缀更新（其余提交可以只更新状态，末步无后缀）。未放宽 100 ms 模型周期或反馈截止条件。全遮挡另测 3 次后归零，NDI 正常记录 209 条。

| 最终 GUI 虚拟验证指标 | 结果 |
|---|---:|
| 大变化目标初始规划 | 274.2 ms |
| 初始尝试 10 / 20 / 40 步 | 52.4 / 60.0 / 161.4 ms |
| 选中计划平均 / 最大节点预测偏差 | 1.440 / 3.100 mm |
| 完整反馈任务计算 P50 / P95 / max（包含迟到任务） | 40.2 / 51.2 / 57.2 ms |
| 实际发令间隔 P50 / P95 / max | 107.7 / 116.3 / 121.0 ms |

### 初始规划 272–274 ms 的含义与旧优化路线

`deadline_20260910_001` 的初始规划总墙钟时间为 272.386 ms，`002` 为 274.224 ms。计时覆盖 `plan_to_tolerance` 内的状态快照、所有尝试长度的 B 优化、终态压力搜索及最终误差判断；不包含模型加载、图像提取/对齐、部署预热、GUI 排队/预览绘制与计划文件保存。该目标由同一冻结模型的 40 步动作生成，再经过 GUI 描画，属于较大弯曲且较容易满足模型可达性的虚拟任务，不代表任意手绘目标都能在相同时间内求解。

当前仍优化形状误差：冻结 NumPy Hereditary 模型按动作推进 p/h，批量读出 15 个平面节点，并用显式链式导数计算动作影响。每次局部修订将动作增量分成最多 8 块 × 4 通道，即最多 32 个变量；SLSQP 求解受压力/速率/信赖范围约束的二次近似，再通过完整非线性 rollout 与回溯检查下降。必要时再以 4 个终态压力为变量做 `least_squares`，投影成限速趋近并保持的序列。没有用 A 缓存，也没有使用 GPU。该轮依次尝试 10、20、40 步，B 分别迭代 5、3、6 次；平均节点偏差 ≤2 mm 且最大节点偏差 ≤4 mm 就停止，未求全局最优或最短轨迹，也未包含避障。

仓库中的旧路线有两类，不能混作同一基准：旧 OpenLoop `ShootingConfig` 默认 4 次重启 × 400 次 Adam，每次对完整神经网络时序 rollout 反向传播；早期 Hereditary 回放使用 PyTorch 自动 Jacobian + 局部求解，`edges_20260908_001/initial_planning.json` 中 8 次初始优化累计 13.432 s，`fast_b_20260909_000` 中仍沿用该初始规划、累计 12.699 s。后者的 fast_b 名称指在线反馈，不意味着初始规划也已换为 NumPy 解析实现。尚未定位用户所述“十几分钟一次”的具体运行记录；不能宣称该次任务获得了确定倍数的等精度加速。

这些是服务器虚拟相机与 Qt 阀桥接下的软期限调度结果，不是硬实时 100 ms 或实机控制精度。所有完整计算时间来自 `feedback_jobs`；实时等待时间另列，避免把超时处截断的等待当成完整求解耗时。

第三页非模态「初始规划参数…」提供平均/最大容限、最大长度、总预算、B 迭代上限、终态求解评估上限和长度搜索步长。修改撤销旧计划；预算在迭代间检查而非硬中断。每个长度的 B 与终态搜索耗时、总耗时和退出原因均保存。

每次执行新增 `initial_plan.npz`、`timings.csv`、`steps.jsonl` 和独立 `feedback_jobs/*.json`，可还原修订前/ACK 后合法后缀/最终采用后缀及未采用提案。写图改为有界后台队列，复制原始像素，积压或保存失败报错；`samples.csv` 保存实际写图时间。任务关闭时未完成的提案明确标记，不能伪造耗时。循环计时不包含本行审计写入本身，实际发令间隔包含这一开销。

239 项综合回归中 238 通过、1 跳过；随后增加后台图像所有权与队列上限测试，最终 39 项部署/记录聚焦检查全部通过。测试注入阻塞反馈，验证命令不等待慢任务、无任务积压、迟到状态和计划不泄漏、版本变化拒绝提交、连续跳过停机、保存每步修订和完整迟到计算时间。提交前完整复核运行 240 项测试，239 通过、1 跳过；文档治理和 diff 空白检查通过。真实 USB/串口仍未连接。新版独立包在无 `src` 的临时目录完成相同流程，结果见 [隔离验证](../../workspace/runs/analysis/hereditary_deployment/portable_deadline_20260910_001/summary.json)。

## 2026-09-10：可选 NDI、SDK/UVC、多区域遮挡与分析记录

新版 [图解手册](../../workspace/runs/analysis/hereditary_deployment/devices_20260910_002/guide.html) 与 [独立包](../../workspace/runs/analysis/hereditary_deployment/SelfSoftRobot_Hereditary_workbench_20260910_001.zip)。第一页恢复可选 NDI（模式、端口、探头数）；不连接不影响模型部署与执行，NDI 不输入模型。相机支持自动优先 RealSense SDK、显式 SDK 或 UVC/OpenCV 索引；指定 SDK/serial 失败不自动换设备，运行中不自动换驱动。仅采集彩色流。

第四页软件遮挡同时适用于真实与虚拟相机，最多 16 个矩形，以图像百分比设置位置/宽高及灰度，参数在执行前应用。算法不接收矩形位置，原图不修改。连续无新图像或无边缘的门限默认 3 次，可设 1–10 次；达限停止归零并撤销部署就绪，恢复需重新部署/规划，不自动搜索或恢复旧后缀。短暂失去视觉期间只有模型预测，不能保证全遮挡下稳定控制；有纹理遮挡物仍可能导致误匹配。

每次执行独立保存 `commands.csv`、`raw/camN`、观察器实际输入 `frames`、`samples.csv`、连续 NDI 的 `ndi.csv`、时间关联 `frame_ndi.csv` 与 `metadata.json`。失败/停止也关闭并写完记录，最后一张无边缘图像先保存再触发停止；故障归零单列命令。辅助相机新鲜性和 NDI 年龄明确记录，时间是主机接收时间，不是硬件同步。NDI 失锁保留 NaN，未连接只生成空数据表与未连接标记。

[带 NDI 虚拟验证](../../workspace/runs/analysis/hereditary_deployment/devices_20260910_002/summary.json)：40 ACK/40 反馈、238 条 NDI 样本、原图与软件遮挡图逐帧分离，另一次全幅遮挡在 3 条命令后归零且保留 3 张失败原图。该轮反馈计算 P95/最大 37.5/42.4 ms，新增原图/反馈图保存后的实际命令间隔 P95/最大 148.0/155.5 ms。

[不连接 NDI 验证](../../workspace/runs/analysis/hereditary_deployment/devices_no_ndi_20260910_000/summary.json) 同样完成 40 步与手动接管，NDI 样本为 0。236 项综合回归，235 通过、1 跳过，涵盖 SDK/UVC 选择、UVC 驱动替身、多个遮挡矩形、原图保真、NDI 可选/失锁/时间关联与断连旧回调隔离。当前服务器无实际 USB/串口，以上不证明真实相机兼容性或实机控制精度。

最终 ZIP 在无 `src` 的隔离临时目录完成同一 40 步流程，见 [独立包验证结果](../../workspace/runs/analysis/hereditary_deployment/portable_devices_20260910_001/summary.json)。

## 2026-09-09 实现（历史版本）


- 默认四页为 **设备模型 → 部署预热 → 目标规划 → 执行记录**。第 1 页仅加载与连接，不发压力、不记录准备期动作；六腔映射两行三列，阀组独立连接/断开，连接内部校验配置。
- 全局非模态六腔工具：共享目标联动，每腔 min/max/rise/fall 独立，与模型取交集；独立手动时钟，运行时应用配置，关闭窗口停止调压并保持。参数整套校验，拒绝范围排除当前压力；命令真正下发前再用最新限制投影，处理编辑与已排队命令之间的竞争。
- 只保留“提取完整形状 → 编辑检查 → 确认形状并部署预热”。提取先确认当前持压 ACK；可从非零输入采用训练一致平衡先验，已有连续状态则延续。固定相似变换后用完整形状有约束地拟合 p/h，持续新帧观测和稳定性检查，达标才就绪并保存部署快照。
- 规划从不可变状态副本自动尝试更长时域，复用 B 候选，并补充受相同约束的终态压力搜索。初始规划预算 15 s；成功候选用完整形状轨迹跟踪，在线仍为 B。默认均值 2 mm、最大节点 4 mm，排除 base；未达标不执行，不宣称不可达或全局最短。
- 紫色轨迹播放/时间轴、六腔压力曲线及每拍数值用于预览；青色当前估计、红色目标与黄色草稿分开。执行前重验原预览是否仍满足容限及漂移门限，失败则要求重新规划。
- 正常完成/停止保持当前压力并继续观测，紧急归零独立。手动接管等待自动发送结束和在途 ACK，再开始手动；旧计划作废，连续记忆和相机对齐保留。
- 结果区分指令完成、模型全形状误差、可信可见边缘到目标管身的距离；缺失证据不作为真实隐藏形状。模型驱动 Mock 与真实 Modbus/RealSense 保持同一入口，服务器没有真实硬件。

[当前图解](../../workspace/runs/analysis/hereditary_deployment/redesigned_20260909_006/guide.html) 含九张实际 GUI 截图与编号圈注。[机器结果](../../workspace/runs/analysis/hereditary_deployment/redesigned_20260909_006/summary.json) 验证了部分组连接/跨组共享禁用、实时修改目标、无效限制原子拒绝、5 kPa 非零初始化、像素提取与手动节点修正、部署预热、自动选择 40 步、大弯曲目标和固定遮挡、40 ACK/40 反馈、自动到手动接管、关闭工具保持、最后归零。接管后模型历史 epoch 与几何变换未重置。

该轮反馈计算 P50/P95/最大 **28.1/39.9/40.2 ms**，后缀接受 36/39。计算时延不含等图、串口及写图。另测实际发令间隔 P50/P95/最大 **110.0/126.7/139.4 ms**：已去掉稍微超时后额外等待一个整时隙的调度浪费；保留最小 dt 和禁止积压突发发送。前一版 redesigned_005 的间隔约 199.8/206.3/208.2 ms。当前只验证虚拟链路在本轮低于 200 ms，未达到全周期严格小于 100 ms，也未验证真实设备。

初始 `redesigned_000/001` 保留了仅靠线性接近参考难以达标的诊断；`002` 加入终态搜索后，严格 1 mm 平均、3 mm 最大容限依然未达标（80 步约 1.293/3.133 mm），正确拒绝执行。没有把放宽至当前 2/4 mm 门限的后续结果称为 1 mm 成功。单姿态几何/记忆对齐、像素草稿与目标描画存在误差，门限还需实机标定。

[独立工作台 ZIP](../../workspace/runs/analysis/hereditary_deployment/SelfSoftRobot_Hereditary_workbench_20260909_004.zip) 包含模型与本版 HTML。[无 src 隔离验证](../../workspace/runs/analysis/hereditary_deployment/portable_redesigned_20260909_001/summary.json) 使用相同 GUI 自动化。

旧 `workflow_20260909_004` 的按钮组织和 27.0/32.3/41.2 ms 结果属于前一版，不作为当前操作说明。原先实际限速导致后缀不合法的修复继续保留：每个 ACK 后用实际 applied6 修复剩余动作。

## 最终验证

综合回归 230 项，229 项通过、1 项跳过。覆盖非零初始化、完整状态拟合、预热重复帧/质量/超时、限制交集与原子拒绝、自动搜索失败门禁、预览失效禁止发令、普通停止保持、调度超时不多等整拍、设备配置生命周期和下发前最新限制校验。`git diff --check` 与文档治理检查通过。

最终独立 ZIP 在临时目录以 Python isolated 模式运行，确认没有导入 `src`，同样完成 40 步遮挡运行与手动接管；该轮发令间隔 P95/最大 130.0/136.5 ms，反馈计算 P95/最大 40.6/49.6 ms。原图、指令与 HTML 在对应输出目录，均为虚拟设备证据。

## 前一版接入产物和结果

冻结部署包：[hereditary.npz](../../workspace/runs/analysis/hereditary_deployment/package_20260909_000/hereditary.npz)、[模型与部署限制](../../workspace/runs/analysis/hereditary_deployment/package_20260909_000/hereditary.json)。原 checkpoint 选择和开发数据谱系见前序回放记录。

[自包含部署 ZIP](../../workspace/runs/analysis/hereditary_deployment/SelfSoftRobot_Hereditary_workbench_20260909_001.zip) 已在隔离临时目录中解压，以 Python isolated 模式运行同一套 GUI 流程，并断言没有导入任何 `src` 模块。[隔离部署验证结果](../../workspace/runs/analysis/hereditary_deployment/portable_smoke_20260909_000/summary.json) 同样完成 20 条命令、20 帧反馈、19 次后缀更新和归零 ACK；反馈计算 P95 26.2 ms、最大 27.0 ms，仅为短 horizon 虚拟设备计算指标。

最终缓冲版本的 [GUI 机器结果](../../workspace/runs/analysis/hereditary_deployment/gui_smoke_20260909_005/summary.json)、[规划截图](../../workspace/runs/analysis/hereditary_deployment/gui_smoke_20260909_005/gui_plan.png)、[完成截图](../../workspace/runs/analysis/hereditary_deployment/gui_smoke_20260909_005/gui_complete.png)：

| 指标 | 虚拟设备 GUI 测试 |
|---|---:|
| 正常执行命令 / 反馈帧 | 20 / 20 |
| 后缀接受更新 | 19/19（末帧无剩余动作） |
| 图像反馈计算 P50 / P95 / 最大 | 18.3 / 24.7 / 31.2 ms |
| 输入帧主机年龄 P50 / P95 / 最大 | 2.2 / 5.1 / 5.3 ms |
| 当前形状描画对齐 RMS | 0.349 px |
| 最后归零 ACK | 通过 |
| 真实硬件通信/真实控制精度 | 未验证 |

`gui_smoke_20260909_000/001` 是自动化测试脚本未正确应用 Mock profile 的失败诊断；`002` 保留零压驻点诊断；`003/004` 是修复后的完整流程，`005` 还包括采集线程缓冲。不得只用早期能发送零压的 20 条 ACK 作为有效动作规划验证。日志和原图均使用独立目录，没有覆盖原始数据。

此前 80 帧实拍回放 B 的单线程计算最大曾达到 139.2 ms；本次虚拟任务更短、图像更简单，表中反馈计算不包括发指令后等新帧、写图和完整阀控周期，不能据此宣称实机周期已经稳定小于 100 ms。串行等待和跳过时隙会使实际命令间隔变长，必须看日志中的命令时间与 slot。

## 前一版验证范围与后续

前一版 213 项聚焦检查运行完成，212 项通过、1 项跳过。测试覆盖冻结数学/导数、原模型/算子、GUI、硬件 profile/桥接、状态时间、迟到图像、重复 ACK、六腔范围交集、相机缓冲所有权、指令失败/断流/全遮挡/中止归零。原 OpenLoop 路线回归继续通过。

```bash
QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m real_validation.tools.hereditary_smoke \
  --bundle workspace/runs/analysis/hereditary_deployment/package_20260909_000/hereditary.npz \
  --out workspace/runs/analysis/hereditary_deployment/gui_NEW

python -m real_validation.tools.package_hereditary \
  --bundle workspace/runs/analysis/hereditary_deployment/package_20260909_000/hereditary.npz \
  --out workspace/runs/analysis/hereditary_deployment/workbench_NEW.zip
```

当前实际负压不受模型/硬件合同支持；长期零压不能保证擦除 play 记忆。相似变换不能纠正任意透视/出平面变化，绘制目标仍属于中心线全形状而非完整轮廓图像拟合。本轮未接入避障。实机需要继续验证曝光/主机时钟、持压/相机延迟、实际光照边缘质量、压力端口映射以及真实终态；此轮没有把这些未来工作写成已完成证据。
