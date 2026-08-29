# 真实序列自动前处理、训练与前向模型验证流程

模型导出后的相机 ROI、在线 Anchor、毫米场景、规划、执行和再观测流程见
[`real_validation_online_workflow.md`](real_validation_online_workflow.md)。
单帧在线分割、SAM2因果前向和离线双向标签的实测比较见
[`online_segmentation_benchmark_20260824.md`](online_segmentation_benchmark_20260824.md)。

## 1. 流水线边界

本流程把一次 `real_capture` 序列转换为可训练的状态转移数据，并完成 GTObserved 与
windowed OpenLoop 两阶段训练和前向模型评价。执行程序由固定算法、显式配置和机器检查
驱动。每个序列只需要提供成像几何与采集条件相关的配置：原始序列、相机、ROI、可选
基座锚点和 GPU。光照或背景发生明显变化时，在同一配置中调整候选分割参数。

SAM2 mask 与由其提取的中心线定义为二维视觉监督标签。独立测量流程负责定义三维物理
真值。NDI 作为隔离的末端诊断通道，与分割、骨架、Dataset、模型输入和 Planner 分层。

自动流水线保存抽样 QC 图。训练准入由可确定的文件、时序、形状、数值和坐标合同检查
决定；QC 图用于定位异常阶段。

每次完整前处理还会从同一个真实 frame 生成逐阶段示例。程序优先选择质量最高的已选
SAM2 锚帧，并在 `real_capture/data/derived/<seq>/qc_pipeline_example/` 保存：

```text
00_pipeline_overview.png     # 同一帧的全流程总览
01_...png ～ 17_...png       # 每个阶段的独立图片
stage_manifest.json          # frame、chunk、锚帧、传播方向和参数
README.md                    # 阶段中文索引
```

运行阶段的决策源是 JSON 配置、固定图像算法和确定性检查。LLM 的角色集中在算法开发、
异常根因分析和论文表述；后续批次由同一程序与数据合同重复处理。

## 2. 两条主命令

复制模板并为新序列填写一次配置：

```bash
cp config/real_preprocess.example.json config/real_preprocess.seq_NEW.json
```

配置中的 `seq`、`roi` 和 `base_anchor` 使用当前新序列的值。随后运行完整前处理：

```bash
MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
python scripts/real/preprocess_capture.py \
  --config config/real_preprocess.seq_NEW.json
```

显式 CLI 参数会覆盖 JSON 中的同名值。例如临时改用两张 GPU 运行 SAM2：

```bash
MPLCONFIGDIR=/tmp/selfsoftrobot-mpl \
python scripts/real/preprocess_capture.py \
  --config config/real_preprocess.seq_NEW.json --gpus 0,1
```

前处理成功后，数据目录为：

```text
data/real_seq/<seq>_n15_sam2_robot_mm/
├── dataset_manifest.json
├── train/<seq>_train.npz
├── val/<seq>_val.npz
└── qc_skeleton/
```

在 tmux 中启动两阶段训练、周期评价和最终评价：

```bash
GPU_ID=2 EVAL_GPU_ID=2 \
bash scripts/real/start_training_tmux.sh \
  real_<seq> \
  data/real_seq/<seq>_n15_sam2_robot_mm/train \
  data/real_seq/<seq>_n15_sam2_robot_mm/val
```

训练与评价使用同一张 GPU 时，评价 watcher 作为独立进程读取 checkpoint 里程碑；训练
进程继续运行。可用以下命令查看会话：

```bash
tmux attach -t real_<seq>
tmux capture-pane -pt real_<seq>:pipeline.0 -S -100
```

tmux 设置了 `remain-on-exit`，流水线完成后会话和终端输出仍然保留。每个试次保存在：

```text
train_log/real_pipeline/<dataset_tag>/trial_YYYYMMDD_NNN/
```

常用训练覆盖参数：

```bash
GPU_ID=2 EVAL_GPU_ID=2 \
GT_EPOCHS=60 OPEN_LOOP_EPOCHS=240 BATCH_SIZE=128 NUM_WORKERS=4 \
SAVE_INTERVAL=5 PERIODIC_EVAL_INTERVAL=10 \
bash scripts/real/start_training_tmux.sh \
  real_<seq> \
  data/real_seq/<seq>_n15_sam2_robot_mm/train \
  data/real_seq/<seq>_n15_sam2_robot_mm/val
```

## 3. 前处理配置

模板文件为 `config/real_preprocess.example.json`。主要字段如下：

| 字段 | 含义 |
|---|---|
| `seq` | `real_capture/data/raw/` 下的序列名或完整路径 |
| `camera` | 当前处理的 RGB 视角，默认 `cam0` |
| `roi` | 源相机坐标中的固定 `x,y,w,h` |
| `gpus` | SAM2 shard 使用的物理 GPU 编号 |
| `chunk_size` | SAM2 独立传播块长度，默认 200 帧 |
| `n_points` | 中心线节点数，当前实验为 15 |
| `segment_lengths` | base 到 tip 各物理段的相对长度，当前为 `[1,1]` |
| `base_anchor` | 可选源相机基座坐标 `[x,y]` |
| `mask_close_k` | 中心线提取前闭运算椭圆核，默认 11 |
| `max_interpolated_fraction` | 自动插值骨架帧比例上限，默认 0.05 |
| `anchor` | 候选 mask 和 SAM2 锚点生成参数 |

同一相机安装和光照条件可复用 `anchor` 参数。相机位置改变时，各序列可以使用不同 ROI；
骨架会恢复到源相机像素，再进入机器人毫米坐标；模型状态由机器人坐标定义。已有 mask
可以用下列工具按机器人直径和运动包络推荐下一次固定方形 ROI：

```bash
python scripts/real/recommend_robot_roi.py \
  --masks-dir sam2/masks/<seq>_full \
  --crop-meta real_capture/data/derived/<seq>/crop/crop_meta.json \
  --padding-diameters 3 --out roi_recommendation.json
```

## 4. 前处理算法

完整入口依次执行：

```text
采集审计
  → 固定 ROI 裁剪
  → 图像候选 mask 与 chunk 锚点
  → SAM2 分块双向传播
  → mask 中心线与端帽修正
  → 15 节点分段重采样
  → 机器人毫米坐标
  → 六维动作合同与模型动作视图
  → 连续时序切分
  → 自动检查与 dataset_manifest
```

### 4.1 采集审计

`scripts/real/audit_capture.py` 读取：

```text
cam0/*.png
frame_times.txt
actions6.csv
commands.csv
samples.csv
meta.json
ndi.csv（存在时）
```

一条训练记录的粒度是一个 `frame_idx` 对应一张图像、一个帧时间和一组最终 applied
六通道动作。程序检查：

1. 图像、帧时间、动作和 sample 数量一致；
2. 图像帧号与 sample 帧号连续；
3. 各时间列有限且严格递增；
4. `samples.command_id` 能连接到命令日志；
5. sampled command 的 applied6 与 `actions6.csv` 一致；
6. `channel_source6` 指定的等值通道残差位于 `meta.json` 的容差内；
7. ACK 状态、图像年龄和 NDI 年龄形成统计与诊断图。

计数、帧号、时间顺序、命令外键和动作合同属于训练准入条件。通信状态和陈旧帧以问题
统计保存。NDI 文件存在时审计其质量和有限值比例；NDI 文件作为可选评价输入。

输出：

```text
real_capture/data/derived/<seq>/qc_capture/
├── capture_audit.json
├── timing_quality.png
└── action_coverage.png
```

### 4.2 固定 ROI

`scripts/real/crop_capture.py` 在全部帧上使用同一个源图 ROI，并保持原始采集目录只读。
该阶段执行裁剪并保留原像素分辨率。`crop_meta.json` 保存源图尺寸、`crop_xywh`、处理后尺寸、
帧数和完成状态。

SAM2 与候选分割使用 ROI 局部像素。中心线阶段把 `crop_xywh[:2]` 加回每个节点，恢复到
源相机像素坐标：

\[
p_{camera}=p_{crop}+[x_{roi},y_{roi}]^T.
\]

输出：

```text
real_capture/data/derived/<seq>/crop/
├── cam0/*.png
├── crop_meta.json
└── qc/{roi_reference.png,crop_overview.png}
```

### 4.3 自动候选 mask 与 SAM2 锚点

`scripts/real/prepare_sam2_anchors.py` 生成 SAM2 的二值 mask 提示。候选 mask 是提示源，
最终训练标签来自 SAM2 传播 mask 的中心线。

背景图 `B` 有两种来源：显式提供的静态背景，或从全序列均匀采样最多 `n_bg` 帧并逐像素
取灰度中位数。默认候选分割为：

\[
W=\mathbb{1}[S<\tau_S\land V>\tau_V],
\]

\[
M=\operatorname{Dilate}(\mathbb{1}[|I_{gray}-B|>\tau_D],k_D),
\]

\[
C_0=W\land M.
\]

其中 `W` 提取低饱和、高亮度的白色半透明主体，`M` 提取相对背景发生变化的区域。
之后依次执行形态学开运算、闭运算、孔洞填充，再保留同时满足最小面积与最小高度的最大
连通域。开运算抑制细线，闭运算和孔洞填充恢复主体内部。

ROI 包含顶部支架时，程序从指定 `base_side` 沿截面检查宽度。它以占用截面宽度中位数
估计主体宽度，并删除进入连续 `stable_span` 个主体宽度截面之前的宽附件，避免支架横梁
成为中心线最长路径的一部分。

每帧保存面积、包围框、跨度、中心和截面宽度。面积与宽度使用序列中位数和
`max(1.4826*MAD, 0.05*median, 1)` 做稳健尺度；几何偏离、主体跨度和基座连续性组合为
代价 `z`，锚点质量定义为 `q=1/(1+z)`。每个实际帧号 chunk 选择质量最高的一帧作为
SAM2 提示，选择过程与动作通道无关。

输出包括 `masks_candidate/`、`anchor_manifest.csv`、`candidate_metrics.csv`、
`candidate_summary.json`、`selected_anchors.png` 和逐阶段 `stage_pipeline.png`。

### 4.4 SAM2 分块双向传播

`sam2/segment_video_full.py` 使用 SAM2.1 Hiera Tiny video predictor。每个 chunk 独立执行：

1. 从 `anchor_manifest.csv` 读取该 chunk 已选锚帧；
2. 把候选二值 mask 作为对象 `obj_id=1` 的 mask prompt；
3. 为该 chunk 建立独立 video state；
4. 从锚帧向 chunk 尾部正向传播；
5. 从锚帧向 chunk 首部反向传播；
6. 每帧写出一个 0/255 PNG mask。

chunk 之间的 state 相互隔离，各 chunk 独立承担传播误差。多 GPU 模式按
`chunk_index % shard_count` 分配 chunk，各 shard 写入互不重叠的帧号。完整 chunk 已存在
时直接跳过，因此相同空间合同下支持断点续算。

分块同时约束 video state 的显存/内存规模和单个锚点的传播距离，并为长序列提供并行与
断点恢复边界。`chunk_size=200` 表示每 200 帧重新选择一次高质量提示，它是离线标签生成
参数，不是控制器的规划窗口。

双向传播中，锚帧之后使用正向传播，锚帧之前使用反向传播。反向部分读取未来帧，因此
属于离线、非因果处理。严格实时控制只能在每张新图到达后执行单帧分割，或维护 SAM2
前向流式状态；带固定回看窗口的反向修正会引入相应视频时长的控制延迟。当前
`real_validation` 使用已确认 ROI 内的 `white_on_blue`/背光单帧分割和同一套15节点骨架
提取，满足因果在线执行；SAM2 双向结果用于训练监督标签和离线质量检查。

SAM2 阶段结束后，入口程序自动检查裁剪图与 mask 的帧号集合完全相同、mask 全部可读、
每个 `failures_shardK.txt` 为空，并汇总 mask 面积的 min、p05、p50、p95 和 max。

### 4.5 mask 到中心线

`scripts/real/masks_to_transition_npz.py` 首先对每个 SAM2 mask 使用 `mask_close_k` 的椭圆
核做闭运算，用于连接被细线遮挡切开的窄缝。随后执行：

1. `skimage.morphology.skeletonize` 把主体细化为单像素中轴；
2. 把中轴像素构造成 8 邻接无向图；
3. 自动定向模式在各连通分量中取图直径最长的路径；
4. 显式基座模式取离锚点最近的中轴像素为 base，并取图中最远点为 tip；
5. 按 `node0=base → nodeN-1=tip` 定向路径。

细化中心线通常停在长条 mask 内部，也可能在倾斜端帽处分向角点。默认双端端帽修正使用
距离变换估计局部主体宽度，在端部内侧约 0.65–1.65 个管径的区间估计局部切向，然后在
局部端帽区域寻找接近完整管径的横截面，把端点移到该横截面的宽边中心。显式
`base_anchor` 直接作为固定端合同，base 端连接到该锚点；tip 仍执行端帽修正。

修正后的有序路径按弧长重采样。对于两段等长机器人和 15 个节点，总共 14 个节点区间
分配为 `(7,7)`，两段共享 `node7`：

```text
第一段：node0 ... node7
第二段：node7 ... node14
```

这一分段合同用于节点位置和评价分段。当前状态转移模型采用一条 15 节点完整中心线，
多段结构信息体现在共享关节节点与分段评价中。

### 4.6 时间 QC 与确定性修复

每帧记录提取成功状态、mask 面积、路径长度、主体像素宽度、端点修正位移与端点修正原因。
时间 QC 计算：

- 重采样中心线总弧长；
- 当前帧相对前后两帧中点的最大节点残差；
- 序列中位数与 MAD 下的稳健离群标记。

提取失败、非有限骨架或弧长小于有效帧中位数 50% 的帧标记为 `hard_invalid`，并用最近
前后有效帧线性插值。高于 8 倍稳健尺度的长度或时间残差标记为 `suspicious`，保存在
`skeleton_metrics.csv` 和 QC 图中。`repair_frames` 可记录明确选定的真实 frame ID，并使用
同一线性插值规则重建。最终 NPZ 必须包含与图像数相同的有限骨架。

### 4.7 机器人平面毫米坐标

程序从中心线主体中段的 mask 距离变换估计机器人像素直径 `d_px`，物理直径固定为
`d_mm=16 mm`。整段序列只建立一个固定相似变换：

```text
origin = 所有 base 节点的坐标中位数
axial  = base 指向 base 邻近节点的单位方向中位数
lateral = axial 顺时针旋转 90°
pixels_per_mm = d_px / 16
```

源相机像素点 `p` 转为机器人状态：

\[
[x_{lat},y_{axial}]^T=
\frac{[lateral,axial]^T(p-origin)}{pixels\_per\_mm}.
\]

`positions` 保存 `robot_planar_mm_v1/mm`，`positions_camera_px` 保存源相机像素骨架，
`skeleton_frame_transform` 保存正反矩阵。这个固定变换吸收相机平面内的平移、旋转与统一
尺度变化，同时保留机器人自身的运动和弯曲。

### 4.8 动作合同

`actions6.csv` 始终作为六维原始记录读取。`meta.json` 的 `channel_source6[j]=i` 表示硬件
通道 `j` 由来源根通道 `i` 决定。程序递归解析到根通道并检查映射无环。例如：

```text
channel_source6 = [0,1,1,3,3,5]
model_action_channels = [0,1,3,5]
action_expansion6 = [0,1,1,2,2,3]
```

六通道 applied 动作按 `meta.json` 的 `hi6` 操作上限归一化：

\[
u_j=p_j/h_j\in[0,1].
\]

NPZ 保留归一化后的六维 `actions`，同时保存 `raw_action_scale6_kpa`、模型根通道的
`action_scale_kpa`、`model_action_channels` 和 `action_expansion6`。Dataset 根据合同把六维
数组投影到模型动作维度。当前示例为 4 维；六路独立采集时同一代码自动得到 6 维。

Dataset 还用训练目录中模型动作的全局最大绝对值作为 `norm_factor`，把动作送入网络前
再除以该值。`norm_factor`、动作视图和状态坐标合同保存在 checkpoint 的配置中。

### 4.9 连续时序切分

单序列按原始时间顺序切分：前 80% 写入 `train/`，后 20% 写入 `val/`。连续切分保持帧序，
并隔离相邻图像的 train/val 边界。训练 Dataset 在每个 NPZ 内构造动作历史窗口和
长度 `K=40` 的连续 episode。

### 4.10 自动训练准入清单

前处理入口在数据根目录写 `dataset_manifest.json`。以下项目全部通过后
`training_ready=true`：

1. 采集关键合同通过；
2. crop 帧数与原图一致；
3. candidate 覆盖全部裁剪帧；
4. SAM2 mask 与裁剪图帧号一一对应；
5. train 和 val 各有一个 NPZ，帧数之和等于图像数；
6. 15 节点骨架数量正确，自动插值比例位于配置上限内，模型坐标和源像素骨架均为有限值；
7. train/val 状态坐标和单位一致；
8. 六维动作、模型动作视图、等值来源和展开合同一致，动作位于 `[0,1]`。

训练脚本启动前会再次验证该清单。抽样图位于 crop、candidate、SAM2 和 skeleton 的 QC
目录，适合在分割环境变化或指标异常时查看。

## 5. 训练算法

### 5.1 试次布局与 checkpoint

每个 trial 保存根配置、实际命令、状态、两个阶段的日志和权重、周期评价、最终评价及
照片叠图。根配置同时记录 NDI 评价输入的可用状态与路径：

```text
trial_YYYYMMDD_NNN/
├── config.json
├── commands.sh
├── status.txt
├── stages/{gt,open_loop}/
├── evaluations/{gt,open_loop}/{periodic,best}/
├── diagnostics/
└── artifacts.json
```

默认每 5 epoch 保存 checkpoint，每 10 epoch 对该 epoch 快照做定量评价与照片叠图。
周期评价以验证集全节点均误选出 `best_eval_model.pt`，阶段结束后的定量评价、
叠图和部署清单均使用该权重。`best_model.pt` 保留为训练 loss 最低点记录。
默认 epoch 为 GT 60、OpenLoop 240。

### 5.2 GTObserved 阶段

模型学习归一化状态转移：

\[
\hat{s}_t=F(s_{t-1},s_{t-2},a_{t-W+1:t},z_{t-1}),
\]

其中 `W=40` 是动作历史窗口，`z` 是在 episode 内演化的迟滞潜变量。GTObserved 训练在
每一步使用记录骨架 `s_{t-1}`，teacher forcing ratio 固定为 1。长度 40 的 episode 内
`z` 连续演化并通过 BPTT 训练，每一步都有监督。

损失由两项组成：

\[
L_{skeleton}=\operatorname{mean}\|\hat{s}_t-s_t\|_2^2,
\]

\[
L_{spatial}=\operatorname{mean}\|
(\hat{s}_{t,i+1}-\hat{s}_{t,i})-(s_{t,i+1}-s_{t,i})\|_2^2.
\]

第一项监督全部节点坐标，第二项监督相邻节点的局部中心线增量。默认 40 步等权。该阶段
得到观测驱动单步前向模型的 best checkpoint。

### 5.3 windowed OpenLoop 阶段

OpenLoop 使用相同网络结构，并从 GTObserved best checkpoint 初始化。每个长度 40 的窗口
使用一帧记录骨架作为初始锚点；窗口内逐步把预测骨架反馈为下一步输入，同时让潜变量
`z` 演化。下一窗口重新使用记录骨架锚定。

默认训练设置为 `tf_ratio=1`、`tf_anneal_epochs=40`、staircase：前 20 个退火 epoch 使用
记录前态，随后切换为预测前态；剩余 OpenLoop epoch 保持自反馈。损失仍在窗口内每一步
计算 `L_skeleton + L_spatial`。OpenLoop 模型的增量尺度被约束在稳定范围内，以控制 40 步
自回归中的数值发散。

## 6. 前向模型验证与输出

### 6.1 GTObserved 前向验证

验证时每一步把记录的前一骨架作为输入，比较预测 `ŝ_t` 与同一记录中的视觉监督骨架
`s_t`。这些指标度量观测驱动前向模型的单步预测能力。

### 6.2 OpenLoop 前向验证

验证按 `K=40` 分窗。每个窗口用一帧记录骨架作为种子，之后 40 步使用模型自身预测；
下一窗口重新锚定。评价同时计算纯单步输入下的预测，用于分离单步建模误差与窗口内
自反馈漂移。

### 6.3 定量产物

`eval_real_quant.py` 输出：

- 每帧、每节点欧氏误差；
- tip 误差和全节点平均误差；
- Chamfer、Hausdorff 与 Procrustes shape RMS；
- tip/mid/base、每个物理段和共享关节节点误差；
- 按动作通道幅值分箱的误差；
- OpenLoop 的 `k=1...40` 误差曲线与 rollout/one-step 漂移比；
- `per_frame.csv`、`summary.txt` 和相应曲线图。

当前 `robot_planar_mm_v1` 数据的欧氏误差单位直接为毫米。像素基线数据使用机器人
16 mm 直径估计 `mm/px`，对应字段标记为估计毫米值。

NDI 与帧时间存在时，GTObserved best 评价拟合当前序列状态末端到 NDI 平面的仿射映射，
并把该映射复用于 OpenLoop 评价。该结果是同序列末端诊断；严格跨序列物理评价需要用
独立标定序列建立映射。骨架前向指标和叠图适用于全部序列，NDI 指标在对应输入可用时生成。

`visualize_real_overlay.py` 使用 NPZ 中的逆变换，把毫米状态映射回源相机图像，输出：

```text
真实照片 + SAM2 mask + 记录骨架 + 预测骨架
```

OpenLoop 叠图还可显示同一模型的 one-step 预测，便于观察漂移从窗口的哪一步开始增长。
窗口第 0 帧是真实观测 Anchor，叠图标记为 `observed anchor`；误差曲线和汇总从
第 1 帧模型预测开始统计。

上述结果属于前向模型验证。Planner 预测终态到目标的距离属于离线规划残差。机器人执行
规划动作后，由新相机观测骨架与目标形态计算的差值才属于真实控制误差。

## 7. 阶段产物索引

| 阶段 | 关键产物 |
|---|---|
| 采集审计 | `qc_capture/capture_audit.json`、时序与动作图 |
| ROI | `crop/cam0/*.png`、`crop_meta.json`、ROI 抽样图 |
| 候选分割 | `masks_candidate/`、`anchor_manifest.csv`、逐阶段 QC |
| SAM2 | `sam2/masks/<seq>_full/*.png`、面积曲线、传播对比图 |
| 中心线 | `qc_skeleton/skeleton_metrics.csv`、中心线叠图 |
| 单帧全流程 | `qc_pipeline_example/00_pipeline_overview.png`、各阶段独立图与清单 |
| Dataset | train/val NPZ、`dataset_manifest.json` |
| GTObserved | best/final/checkpoint、loss log、周期和最终评价 |
| OpenLoop | best/final/checkpoint、loss log、drift 与照片叠图 |
| Trial | `config.json`、`commands.sh`、`artifacts.json` |

## 8. 分阶段复跑

完整入口默认断点复用正确尺寸的 crop 和完整 SAM2 chunk。只重跑中心线与 NPZ：

```bash
python scripts/real/preprocess_capture.py \
  --config config/real_preprocess.seq_NEW.json --stages skeleton
```

ROI 合同改变时，裁剪图和对应 SAM2 mask 需要共同重建：

```bash
python scripts/real/preprocess_capture.py \
  --config config/real_preprocess.seq_NEW.json \
  --overwrite-crop --reset-sam2
```

每次前处理的实际子命令保存到：

```text
real_capture/data/derived/<seq>/preprocess_manifest.json
real_capture/data/derived/<seq>/PREPROCESS_REPORT.md
```

每次训练的实际命令保存到 trial 根目录的 `commands.sh`。

完成试次的可视化可直接复现：

```bash
# 前向定量评价与照片叠图
bash <trial>/commands.sh

# 已保存离线规划工件的形态、动作、残差动图
python scripts/evaluation/export_offline_control_demo.py \
  --plan-dir <trial>/evaluations/open_loop/<plan_name> \
  --out output/real_control_demo/<demo_name> --target-frame <val_index>
```

`commands.sh` 会重新执行训练。只复现已有权重的评价时，从文件中复制
`eval_real_quant.py`和`visualize_real_overlay.py`两条命令执行。
