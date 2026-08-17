# 双段六腔机器人的平面约束中间阶段

> 日期：2026-08-17  
> 分支：`feat/planar-constrained-control`（从 `master` 创建）  
> 状态：P1–P4 软件实施完成；等待真机 mapping、阈值、数据与任务验收
> 定位：先验证单相机可可靠监督的双段形态控制与避障，再扩展到真实三维

## 1. 结论

这一步值得做，而且比直接把采集、GT、模型、Planner 和 GUI 同时升级到三维更适合当前项目。

建议第一阶段保留六个物理阀，但把每段三个腔道限制在一个对称动作流形上：选择一个
“单腔”，另外两个腔道始终接受相同命令。两段分别这样约束后：

- 原始硬件仍是 6 通道；
- 每段命令由“单腔压力 + 等压腔对压力”两个数描述，两段共 4 维；
- 每段主要只有一个弯曲方向，两段装配方向一致时整臂应近似留在同一平面；
- 单台固定相机继续提供有序的 15 节点二维骨架 GT；
- 两段各自的曲率可调，能够先研究全身形态控制、末端到达和二维无接触避障。

结合现有 `real_capture` 和 `real_validation` 接口，第一版采用更小的工程改动：模型和 NPZ
仍保留 6 维动作，只是在 GUI/控制层强制两个通道对相等。数据实际位于一个 4 维动作流形上；
Planner 用 4 个独立变量构造满足约束的 6 维模型输入。暂不把训练动作压缩成 4 维。

它与 `feat/real-3d-pipeline` 不冲突。三维分支保留 RGBD、多视角和三角化能力；本分支
回答更早、更基础的问题：在可靠 GT 下，动作历史模型能否支持双段整形和避障。

## 2. 必须澄清的自由度

假设一段三个腔道位于等边三角形顶点，截面角度为
`0°/120°/240°`。一阶近似下，弯曲向量与三个腔压的差分分量有关：

```text
b ∝ p0 e0 + p1 e1 + p2 e2,       e0 + e1 + e2 = 0
```

若约束 `p1 = p2 = q`，则：

```text
b ∝ (p0 - q) e0
```

因此需要区分：

| 概念 | 两段机器人中的数量 | 含义 |
|---|---:|---|
| 物理阀/腔道 | 6 | 实际下发和记录的 `c0..c5` |
| 独立压力变量 | 4 | 每段一个单腔压力和一个等压腔对压力 |
| 主要弯曲自由度 | 2 | 每段一个有符号差压，决定该段平面曲率 |
| 第一版模型动作维度 | 6 | 两列成对重复，保持现有训练/部署接口兼容 |

每段的差压 `d = p_single - p_pair` 主要控制曲率；共同压力
`c = (p_single + 2 p_pair)/3` 还可能改变轴向伸长、刚度和动态响应。因此第一版不应直接
声称系统只有 2 个动作输入。虽然模型文件仍接收 6 列，但必须明确其有效动作流形只有 4 维。

后续可比较压缩 4 维模型，以及每段固定共同压力的 2 维消融；它们不是第一版前置条件。

## 3. 这一步能证明什么，不能证明什么

可以形成的结论：

- 结构化 6 腔驱动流形上的双段平面动力学可以由有限动作历史学习；
- 单次真实观测锚定后的窗口化 OpenLoop 能预测一段可信视野；
- 两段曲率提供了比原单腔实验更强的全身形态调节能力；
- 在无接触前提下，模型规划可以完成平面目标到达和整条骨架避障。

不能据此声称：

- 六个独立几何自由度已经得到验证；
- 已经完成任意三维形态控制或三维避障；
- “两个等压命令”必然产生完全相等的真实腔压；
- 末端不离面就代表整条机器人都不离面。

论文中应称为“structured-actuation planar whole-body control/avoidance”，把它作为三维扩展前
的能力阶段，而不是用二维结果替代最终三维目标。

## 4. 单相机 GT 与平面性验证

### 4.1 主训练观测

主相机固定并尽量正对运动平面。继续使用现有流程：

```text
cam0 RGB → 分割/修复 → tip-to-base 15节点骨架 → [col,row,0]
```

第一版可继续在像素坐标训练和规划；目标、障碍和机器人半径必须来自同一相机坐标系。
若需要毫米指标，再使用平面标定板求 homography，把像素映射到机器人运动平面。

### 4.2 平面性不能只靠理论假设

单相机看不到离面弯曲，因此每个序列必须额外保存一个独立的平面性质控结果：

1. 首选使用现有 NDI 的末端三维坐标，统计相对标定平面的离面漂移；
2. 可选增加一台侧视普通 RGB 相机，只做离面轮廓/中心线质控和序列拒收；
3. 侧视图不参与节点对应、三角化或训练 GT，因此不会重新引入多视角骨架对应问题；
4. 阈值应由零驱动噪声和重复动作实验估计，不在代码中写死一个毫米数。

建议的 QC 字段：

```text
planarity_tip_abs_mm_p50/p95/max
planarity_side_residual_px_p95       # 如果存在侧视相机
planarity_pass
planarity_threshold_source
```

只有 `planarity_pass=true` 的序列进入二维训练主集。失败序列保留用于诊断，不静默删除。

## 5. 最小的 GUI 等值约束与数据合同

### 5.1 参数化不能硬编码通道编号

GUI 增加两行可选的等值关系，例如“`ch2 跟随 ch1`”“`ch5 跟随 ch4`”。底层仍需要
一个很小的可配置约束对象，避免只在界面显示值相等而实际命令不同。默认假设只是示例：

```json
{
  "segments": [
    {"channels": [0, 1, 2], "single": 0, "equal_pair": [1, 2]},
    {"channels": [3, 4, 5], "single": 3, "equal_pair": [4, 5]}
  ]
}
```

真实 mapping 必须通过接管和小幅单腔试验确认；还要确认两段绕主轴的装配角一致。若第二段
相对第一段发生轴向旋转，即使各段局部平面弯曲，也不一定处于同一全局平面。

四个独立压力变量与六维动作的关系为：

```text
u4 = [s0, q0, s1, q1]
a6 = [s0, q0, q0, s1, q1, q1]      # 仅为默认 mapping 示例
```

约束启用后应统一作用于 Manual、Random、Sweep 和 Replay，不能只修改 Random。follower
通道的 target/min/max/rise/fall 控件应镜像 leader 并禁用编辑。开始采集前要求等值组当前命令
已经相等（最简单是先归零）；下发后再次检查 `applied6`，不相等则停止而不是继续采坏数据。

当前自动动作的真实逻辑是：

- Random：每通道在自己的 `[min,max]` 内做有界随机游走；
- Sweep：每通道在自己的范围内独立往返；
- seed：控制 Random 的可复现随机序列；
- rise/fall：在 controller 下发前按真实命令时间间隔逐通道限速；
- “预生成步数”：只缓存前 N 个动作，耗尽后继续在线生成，不是采集自动停止步数。

因此无需新增“先生成 CSV 再 Replay”的必经流程。若以后确实需要严格固定实验长度，可单独增加
“最大命令步数”并自动停止；不要改变现有“预生成步数”的语义。

### 5.2 原始数据仍保存六通道

`real_capture` 第一版不需要改变采集格式。Replay 仍下发六列，原始目录继续保存：

```text
actions6.csv       # 本拍对应的六维动作
commands.csv       # requested6 / applied6 / 通信状态
cam0/              # 主训练视角
ndi.csv            # 独立末端三维质控
```

第一版离线处理继续直接使用六维动作，只额外验证并保存约束元数据：

```text
actions:             (T,6)       # 原始训练文件，等值列保留
model_action_channels: [0,1,3,4] # Dataset 投影合同
action_expansion6:   [0,1,1,2,3,3]
channel_equalities:  JSON/string
pair_residual:       (T,2)
planarity_qc:        metadata
```

若任一等压对在 `applied6` 命令历史中超过记录的命令容差（默认 0.5 kPa），转换必须
fail-closed；量化或浮点造成的小残差允许存在。需要注意：当前 ACK 证明的是
命令被驱动层接受，不是六个腔体的真实压力完全相等；真正的物理偏差由压力传感器（若有）、
NDI 和侧视 QC 揭示。

### 5.3 Planner 必须搜索同一个动作流形

训练只覆盖 `a6 = expand(u4)`，Dataset 因此选择 `u4=[ch0,ch1,ch3,ch4]`，模型直接使用
`action_dim=4`。Planner 优化同一 `u4`，到计划/硬件边界才展开成六维。不能训练时约束等压、
规划时又让六个通道独立搜索，否则会立即产生
动作 OOD。

部署合同使用 `channel_map=(0,1,3,4)`、`channel_equalities=((1,2),(4,5))`：

```text
optimizer/model/history u4 → EqualityConstraint.expand → controller actions6
applied6 → channel_map.project → model history4
```

压力上下界和 rise/fall 限制仍按六维检查。等值通道使用相同范围和速率，并从相等初值开始，
逐通道 limiter 才会继续给出近似相等结果。模型历史从验证后的六维 `applied6` 选择独立四列。

## 6. 最小实施顺序

### P0：确认硬件几何，不改模型

- 记录每段三个物理腔体到 `ch0..ch5` 的映射；
- 标记每段截面方向以及两段之间是否绕轴旋转；
- 每次只施加小幅单腔压力，确认弯曲方向和安全范围；
- 选择使两段处于同一全局图像平面的单腔/等压对组合；
- 用 NDI/侧视图测零驱动噪声和重复动作离面漂移。

停止条件：找不到一组在安全范围内稳定保持平面的 mapping，就不要开始二维训练，转回三维分支。

### P1：在现有 GUI 增加等值约束

- 增加最多两组 `follower = leader` 选择，禁止 self-link、环和重复 follower；
- follower 的 target/min/max/rise/fall 镜像 leader；
- 约束统一投影 Manual、Random、Sweep 和 Replay 的每一拍；
- 配置写入 `meta.json` 和 GUI 持久化文件；
- 启动时检查两组 Modbus 均连接、链接通道初值相等；
- 保存前验证最终 `applied6` 的 pair residual 不超过显式容差，默认 0.5 kPa。

Random、Sweep、seed、预生成和每通道安全范围继续复用现有实现，不新增前置动作文件。
采集层仍只负责原始动作、图像和 NDI，不在其中做骨架或平面判断。

### P2：离线验证六维受约束数据

- 扩展 `masks_to_transition_npz.py`，读取 `channel_equalities`；
- 从 `applied6`/`actions6` 验证等值关系，不压缩动作列；
- 默认 15 节点，保留现有二维分割、tip 修复和清洗流程；
- 写入平面 QC、六维动作和 equality metadata；
- 按完整轨迹/激励段切分 train/val/test，不能随机打散相邻帧。

### P3：训练与可信视野

- 先训练 GT-observed，用于检查数据和单步模型；
- 再训练窗口化 OpenLoop，作为部署主线；
- 原始 NPZ 为六维，Dataset 投影后模型 `action_dim=4`；状态仍为 `(15,3)`，第三维为零；
- 对每段动作历史分别覆盖加载、卸载和 hold；
- 在从未出现过的联合轨迹上评估单步误差、rollout 漂移和 `K_safe`。

保留六维模型可作为消融；部署主线使用没有冗余等值列的四维模型。

### P4：受约束规划和执行

- 在部署 manifest 保存 model action channels、channel equalities、展开关系和动作尺度；
- `OpenLoopShootingPlanner` 直接优化四维模型动作，生成计划时展开到六维硬件动作；
- preflight 在 6 维展开后检查压力、速率、通信组和 pair equality；
- 执行记录保留真实 `applied6`，模型历史按 channel map 投影为四维；明显违反 pair 容差时归零；
- NDI 继续只做评价和离面安全监视，不进入模型或 Planner。

### P5：实机任务递进

1. 单段与双段重复轨迹：先证明动作流形和平面性；
2. 双段目标骨架跟踪：验证整形，不放障碍；
3. 末端目标区域 + 一个贯穿运动平面厚度的圆柱障碍；
4. 同一末端目标、不同障碍位置：检验是否真的利用中间形态绕障；
5. 多个起始形态和加载历史：检验 OpenLoop 历史模型的必要性。

主避障实验建议使用“末端目标 + 全身障碍约束”，而不是一开始就指定完整目标骨架。完整骨架
几乎已经规定了最终形状，不利于展示两段冗余如何在保持目标时主动改变中间形态。

## 7. 障碍物和评价定义

二维碰撞只有在障碍物沿离面方向贯穿机器人可能运动的厚度时才具有明确物理意义。建议使用
圆柱柱体或板状障碍，并在主相机平面中标定其投影和安全膨胀半径。

Planner 的碰撞代价要覆盖节点之间的线段/胶囊，而不只检查 15 个离散节点。

核心指标：

| 类别 | 指标 |
|---|---|
| 平面性 | NDI 离面漂移 p95/max；侧视残差；拒收率 |
| 预测 | 15节点 RMSE、末端误差、rollout error-vs-horizon、`K_safe` |
| 形态 | 有序节点误差、最大节点误差、目标骨架成功率 |
| 避障 | 实机无碰撞成功率、最小胶囊净距、末端目标成功率 |
| 执行 | plan-to-execution gap、压力/速率违规、重规划次数 |

至少比较：

- 无历史或仅当前动作；
- GT-observed；
- 窗口化 OpenLoop；
- 只优化末端、不约束全身障碍的 Planner。

这些比较分别回答历史是否必要、OpenLoop 漂移多大、全身约束是否真的避免“末端到了但身体撞了”。

## 8. 后续预计修改的文件

正式实施时优先新增小而独立的模块，不把约束散落在 GUI 中：

```text
real_capture/main_capture.py                         # 两组 GUI follower=leader
real_capture/valve_control.py                        # 小型 equality 投影/校验
real_capture/recorder.py                             # meta 和每拍 applied6 invariant
scripts/real/masks_to_transition_npz.py              # 六维 pair 校验与 QC
real_validation/contracts/deploy_manifest.py         # equality metadata
real_validation/planning/openloop_planner.py         # 4变量展开为6维模型输入
real_validation/execution/preflight.py               # 六维安全与 equality 检查
scripts/evaluation/eval_planarity.py                 # NDI/侧视平面性报告
tests/test_planar_actuation.py
```

相机线程和原始目录结构不需要改变；`recorder.py` 只增加 equality 元数据和 fail-closed 校验。
不能把动作压缩、平面判断或骨架处理塞进采集线程。

## 9. 第一版验收门

只有以下条件全部满足，才进入真实避障：

- 两段通道 mapping 和装配方向已经记录；
- Manual/Random/Sweep/Replay 的每一拍都满足等压对 invariant；
- 单段和双段联合动作的平面 QC 均通过；
- 训练数据、Planner 和执行都满足同一个六维 equality invariant；
- OpenLoop 的真实验证误差在认证的 `K_safe` 内；
- 碰撞检查覆盖整条节点间胶囊；
- 障碍物几何确实代表二维平面中的不可穿越区域；
- preflight 能拒绝 mapping、压力范围、历史或平面 QC 缺失的实验。

若平面性反复失败，最可能的问题依次是：通道 mapping 错误、两段绕轴装配不一致、阀/腔体
响应不匹配、材料扭转耦合。此时保留本分支作为二维基线，继续使用
`feat/real-3d-pipeline` 获取真实三维监督，而不是用单相机二维标签掩盖离面运动。

## 10. 2026-08-17 实施记录

已按层次独立提交：

- `ba6252b`：采集 GUI、真/Mock Controller、Recorder 的等值投影、镜像、元数据和残差；
- `ca9ebe7`：二维前处理的六维动作验证、NPZ equality/QC 元数据；
- `13bdff8`：deploy manifest、四变量 Planner、preflight、warmup 和 ACK 执行守卫；
- `6091457`：NDI 末端离面 p50/p95/max 质控工具。

运行操作见 `docs/real_data/planar_constrained_6ch_workflow.md`。尚未完成的 P0/P5 都依赖真实硬件：
通道 mapping、两段装配平面、QC 阈值、采集/训练、`K_safe` 认证和真实避障结果。软件通过不等于
这些物理条件已经成立。
