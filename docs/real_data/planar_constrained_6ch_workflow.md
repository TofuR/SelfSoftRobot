# 六通道来源约束平面实验：实际运行流程

> 实施分支：`feat/planar-constrained-control`  
> 目标：保留六维硬件与原始数据，用通用 `channel_source6` 描述任意同源通道组；
> 当前双段实验配置恰好形成四维模型动作，先完成单相机 15 节点二维整形与避障，再扩展真实三维。

## 1. 不变与变化的边界

- `real_capture` 仍只采原始 RGB、六维动作、NDI 和时间戳，不做骨架或平面判断。
- `actions6.csv`、训练 NPZ 和执行命令始终是六维；Dataset/模型/checkpoint 只使用来源根通道。
- 权威合同 `channel_source6[i]` 表示硬件 `chi` 从哪个根通道取值；自身表示独立。
- 当前示例为 `[0,1,1,3,4,4]`，因此根通道为 `[0,1,3,4]`、`action_dim=4`。
- identity `[0,1,2,3,4,5]` 是 6D；一组 follower 是 5D；其他合法来源图可自然得到 1–6D。
- Random、Sweep、Manual、Replay 统一在下发前投影；不能只约束某一种采集模式。
- Dataset 自动读取 NPZ 动作视图合同；Planner 优化同一组根变量，在 ActionPlan 边界展开为 6 列。
- NDI 只做末端离面 QC/评价，不作为骨架 GT、模型输入或 Planner 输入。

## 2. 真机前先确认物理 mapping

GUI 的来源选择必须由真机辨识决定，不能把当前 `[0,1,1,3,4,4]` 示例当作物理事实。先在安全低压范围内逐腔测试并记录：

1. 两段各三个物理腔体到 `ch0..ch5` 的对应关系；
2. 各腔单独增压时的弯曲方向；
3. 两段绕轴装配角是否使局部弯曲平面共面；
4. 选定每段的 single 腔和 equal pair；
5. 零驱动 NDI 噪声、重复动作离面漂移，据此确定 QC 阈值。

如果找不到稳定共面的 mapping，应停止二维主实验并回到三维分支，不能用单相机投影掩盖离面运动。

## 3. 采集

先用全 Mock 检查 GUI 和目录合同：

```bash
python real_capture/main_capture.py --mock
```

真机启动示例：

```bash
python real_capture/main_capture.py \
  --group1 COM3 --group2 COM46 --ndi COM9
```

操作顺序：

1. 连接两组 Modbus、NDI，确认主相机预览正常；
2. 主通道选择 `all`；
3. 在“通道来源”中逐路选择经真机确认的根通道（自身=独立）；
4. 确认界面显示的规范化来源关系正确；链式选择应压平，循环应自动回退；
5. 修改根通道的 target/min/max/rise/fall，确认所有 follower 自动镜像且不可编辑；
6. 先“全部归零”，再开始 Manual 小幅测试；
7. 确认方向与平面性后再运行 Random/Sweep；需要复现实验时使用同一 seed；
8. Replay 仍读取普通六列 `actions6.csv`，但每拍也会经过同一来源投影；
9. 停止后检查 `meta.json.channel_source6`，并对比 `commands.csv` 的 proposed/requested/applied。

约束启用时只允许 `all`，且 linked 通道的范围、rise/fall 和当前命令必须相等。控制器会在逐通道
限速后再次检查 `applied6`；残差超过记录容差（默认 0.5 kPa）才停止采集，正常量化小误差不会
中止。ACK 只证明驱动层接受了近似相等命令，不证明真实腔压
完全相同，真实离面偏差仍由 NDI/侧视质控判断。

## 4. 离面 QC

先标定运动平面的 NDI 法向量。若不传 `--plane-point`，脚本用开头零驱动有效样本 XYZ 的中位数
作为面上一点：

```bash
python scripts/evaluation/eval_planarity.py \
  --seq real_capture/data/raw/<seq> \
  --plane-normal nx,ny,nz \
  --baseline-samples 30 \
  --threshold-mm <实验测得阈值> \
  --threshold-source "zero-pressure-repeat-2026xxxx" \
  --pass-stat p95
```

输出 `<seq>/planarity_qc.json`，包含有效样本数、离面绝对距离 p50/p95/max 和
`planarity_pass`。默认用 p95 判定，也可用更保守的 `--pass-stat max`。明确失败的序列会被后续
NPZ 转换拒绝，但原始目录不会删除。

限制：单个 NDI 末端只能证明末端近似留在平面内，不能证明整条软臂都未扭出平面。论文主实验最好
增加一台固定侧视 RGB 相机做全身离面轮廓 QC；它不参与训练节点对应或三角化。

## 5. 15 节点二维骨架与六维 NPZ

分割仍在采集后独立完成。准备好 mask 后运行：

```bash
python scripts/real/masks_to_transition_npz.py \
  --seq real_capture/data/raw/<seq> \
  --masks-dir real_capture/data/derived/<seq>/masks_sam2 \
  --action-channels auto \
  --n-points 15
```

只要 `meta.json` 声明了 `channel_source6`（旧 `channel_equalities` 会自动迁移），工具就会：

- 从来源根自动确定 `model_action_channels`；当前示例得到 `[0,1,3,4]`；
- 在归一化前逐帧验证原始 kPa 等值残差；
- 要求 linked 通道的原始归一化上限相同；
- 保留 `actions:(T,6)`，不提前压缩；
- 在 NPZ 保存 `channel_source6`、`model_action_channels`、`action_expansion6`、兼容 `channel_equalities`、
  `pair_residual_max` 和可选 `planarity_qc`。

没有等值元数据的旧单通道序列继续兼容原来的 `--action-channels 0`。

## 6. 训练、视野认证与部署包

训练命令不变，先 GT-observed 检查数据，再训练部署主线 OpenLoop：

```bash
DATA=data/real_seq/<seq>/train
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_transition.py --mode gt --data_dir "$DATA"
CUDA_VISIBLE_DEVICES=0 python scripts/training/train_transition.py --mode open_loop --data_dir "$DATA"
```

确认训练 `config.json` 中 `action_view.raw_action_dim=6`、`channel_source6`、
`model_action_channels` 与来源根一致，且 `action_dim=len(model_action_channels)`；当前示例才是
`action_dim=4`、`model_action_channels=[0,1,3,4]`。完成独立序列 rollout 和 `K_safe`
认证后生成 manifest：

```bash
python scripts/utils/build_deploy_manifest.py \
  --exp-dir train_log/open_loop_transition/<exp> \
  --raw-seq real_capture/data/raw/<seq> \
  --channels 0,1,3,4 \
  --horizon-summary <exp>/eval_horizon/horizon_summary.json
```

构建器会交叉验证 raw `meta.json` 与训练 `action_view` 的 `channel_source6`。部署包必须满足：

- `action_dim` 等于来源根数量；
- `channel_map` 等于按硬件顺序排列的来源根；
- `action_expansion6` 完全由 `channel_source6 + channel_map` 推导；
- 当前示例分别为 `4`、`[0,1,3,4]`、`[0,1,1,2,3,3]`；
- raw `actions6.csv` 全序列残差在采集 tolerance 内。

## 7. real_validation 规划与执行

加载带 manifest 的 checkpoint 后，工作台沿用现有流程：相机锚定 → 场景 → Plan → Preflight →
Arm → Execute。约束在后台合同中自动生效：

```text
optimizer uD（D=来源根数量）
  → D 维 OpenLoop 模型
  → 按 action_expansion6 展开 a6
  → 当前示例 a6=[u0,u1,u1,u2,u3,u3]
  → 六维 kPa ActionPlan
  → preflight
  → transport
  → ACK applied6 等值复核
```

Preflight 会拒绝：plan/model equality 不一致、历史动作维度错误、linked 通道压力范围/速率/初值
不同、动作越界或超速。执行器仅把通过 ACK 且等值残差合格的 `applied6` 写入历史；失配时归零并
中止，必须重新锚定和规划。

Mock Warmup 直接生成 D 维模型历史；真机下发时才展开六维。Real 模式仍允许操作者显式选择零历史起步，但初始窗口
属于 OOD，界面会告警；后续用合格的真实 `applied6` 逐步替换。

## 8. 第一轮实机验收顺序

1. 单段、双段重复动作：确认命令 residual 为零且 NDI p95 通过；
2. 联合随机激励：检查 15 节点分割和 OpenLoop rollout 误差；
3. 无障碍末端点目标；
4. 无障碍完整目标骨架；
5. 一个贯穿运动厚度的圆柱/板障碍，使用末端目标 + 全身胶囊碰撞约束；
6. 固定目标、改变障碍位置和初始历史，验证中间形态确实随障碍改变。

在 mapping、平面 QC、六维 invariant、`K_safe` 和全身碰撞检查全部通过前，不进入真实避障。
