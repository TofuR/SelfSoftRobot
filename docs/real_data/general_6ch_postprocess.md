# 六通道通用真实数据后处理主线

## 1. 目标与边界

本流程面向任意 `channel_source6` 和 1–6 个独立模型动作维度。图像处理只读取 RGB，
不按某个动作通道挑帧，也不假设任一机器人段静止。原始动作始终保存为六维
`applied actions6`，Dataset 再按 NPZ 中的动作合同投影为模型输入。

主线：

```text
raw RGB
  -> 自动候选分割和锚点评分
  -> SAM2 分块双向传播
  -> 通用中轴主路径
  -> 分段弧长重采样（默认15节点、两段共享node7）
  -> 无静态段假设的时间QC
  -> 6D NPZ + 动作视图合同 + segment_layout
  -> GTObserved/OpenLoop训练
  -> 图像骨架、分段/关节、NDI末端和OpenLoop漂移评价
```

NDI始终是独立评价流，不进入分割、模型、规划器或控制观测。

## 1.1 大图像先做固定ROI裁剪

相机画面中只有局部实验区域有用、且机器人安装位置固定时，先在**源相机坐标**中选择一个
覆盖完整运动包络的固定ROI。建议使用方形、边长可被常见网络步长整除的尺寸，例如
`320×320`；同一序列和同一相机合同下所有帧必须使用完全相同的ROI，不能逐帧跟随机器人。

```bash
python scripts/real/crop_capture.py \
  --seq real_capture/data/raw/<seq> \
  --camera cam0 \
  --roi x,y,w,h
```

原始图像保持不变，输出到`real_capture/data/derived/<seq>/crop/cam0/`。脚本同时保存
`crop_meta.json`和跨全时段的`qc/roi_reference.png`、`qc/crop_overview.png`。若已有裁剪合同与
新请求不同，脚本默认拒绝静默混用；只有确认重建时才显式加`--overwrite`。

裁剪只是SAM2的处理视图。当前状态转移模型输入是15节点骨架而不是RGB；构建NPZ时通过
`--crop-meta`把局部骨架加回ROI的`(x,y)`偏移，模型状态仍统一为原相机像素坐标。未来若训练
图像模型，可在这个固定方形ROI上统一resize，但必须另行记录resize尺度。

## 2. 阶段A：自动候选mask和SAM2锚点

运行：

```bash
python scripts/real/prepare_sam2_anchors.py \
  --seq real_capture/data/derived/<seq>/crop \
  --camera cam0 \
  --out-root real_capture/data/derived/<seq> \
  --chunk-size 200 \
  --base-side top
```

如果采集前能拍一张无机器人背景，建议加`--background-image <background.png>`；否则默认从
全序列自动构建中值背景。两种来源都会写入`candidate_summary.json`，避免后续混淆。

算法逐帧执行五个可解释阶段；完整序列保存`final`，中间阶段自动抽样到QC图，避免把
万帧数据膨胀成五套mask：

1. `white`：HSV白色区域；
2. `moved`：与全序列中值背景不同的区域；
3. `gated`：白色与运动区域交集；
4. `morph`：去细气管、闭合和填洞；
5. `pretrim`：满足面积/跨度条件的最大主体；
6. `final`：从base侧按主体管径删除偶发粘连的宽支架短分支，作为SAM2候选锚点。

`base attachment trim`只检查当前mask截面宽度，不使用动作，也不把任何一段视为静态。
它允许方形RGB裁剪保留基座/支架上下文，同时防止最长骨架沿横梁偏移。可用
`--base-trim-width-ratio`调节，或传小于等于0显式禁用。

对全部候选按面积、主体宽度、图像跨度和基座侧连续性进行稳健打分，每个SAM2 chunk
自动选一个高置信帧。该过程不查看 `c0..c5`。输出：

```text
real_capture/data/derived/<seq>/
  bg_median.png
  masks_candidate/*.png
  anchors/*.png
  candidate_metrics.csv
  anchor_manifest.csv
  candidate_summary.json
  qc_candidate/
    stage_pipeline.png
    candidate_metrics.png
    selected_anchors.png
```

正式处理必须使用 `--frame-step 1`（默认）。`--frame-step 100`只用于快速跨全序列诊断，
其输出不完整，不能直接喂SAM2。

当前旧真实数据是宽白色机器人、稳定蓝色背景，主要干扰为同色细气管和偶发手部：形态学
负责抑制细管，面积/宽度稳健分数避免选中手部粘连帧，SAM2负责从干净锚点传播主体。
如果以后改变背景/照明，应保留同样的阶段合同，但可替换候选分割器；无需修改动作、骨架
或训练代码。

## 3. 阶段B：SAM2传播与自动对比

```bash
CUDA_VISIBLE_DEVICES=0 python sam2/segment_video_full.py \
  --seq real_capture/data/derived/<seq>/crop \
  --anchor-mask-dir real_capture/data/derived/<seq>/masks_candidate \
  --anchor-manifest real_capture/data/derived/<seq>/anchor_manifest.csv \
  --out sam2/masks/<seq>_full \
  --chunk-size 200 \
  --device cuda:0
```

默认优先读取：

```text
derived/<seq>/masks_candidate
derived/<seq>/anchor_manifest.csv
```

旧序列没有新产物时才回退到 `masks_repaired`，仅用于历史复现。SAM2阶段输出：

```text
sam2/masks/<seq>_full/
  *.png
  area_curve.txt
  failures.txt
  run_meta_shardN.json
  qc/
    compare_candidate_sam2_shardN.png
    area_candidate_sam2_shardN.png
```

训练转换前必须满足：原始帧、动作行和SAM2 mask数量一致；`failures.txt`为空；面积曲线没有
未解释的突变。SAM2 mask仍是伪GT，自动QC用于发现问题，不等于独立人工真值。

## 4. 阶段C：通用中心线和两段语义

```bash
python scripts/real/masks_to_transition_npz.py \
  --seq real_capture/data/raw/<seq> \
  --masks-dir sam2/masks/<seq>_full \
  --crop-meta real_capture/data/derived/<seq>/crop/crop_meta.json \
  --skeleton-method skeletonize \
  --endpoint-fix \
  --n-points 15 \
  --segment-lengths 1,1 \
  --qc-frames 1000,5000 \
  --action-channels auto \
  --out-root data/real_seq/<seq>
```

默认 `skeletonize` 从完整mask提取快速细化图，取图的最长主路径以剪除短分支，再按弧长
重采样。它不依赖“每一行只有一个机器人截面”，可处理局部水平和S形中心线；需要比较严格
中轴时可显式使用较慢的 `--skeleton-method medial_axis`。真正的二维
自重叠仍是单目不可辨识情况，应通过实验动作范围或额外标记避免。

`skeletonize/medial_axis` 的数学中轴通常会在端帽处向内收缩；旋转后的平端帽还可能产生
通向角点的短分支。因此现代流程默认开启 `--endpoint-fix`，分别对 tip/base：

1. 从端部内侧约一个管径的路径估计局部切向，避免用角点分支估方向；
2. 用距离变换估计主体管径，并沿切向扫描横截面；
3. 找到最靠外且宽度达到主体管径约 85% 的截面，以它确定法向中心；
4. 只在该中心附近向外追到 mask 边界，得到平端、斜端或圆端的中心交点；
5. 删除原端帽角点分支，将两个修正端点接回主路径，再统一按弧长采样 15 节点。

该修正与动作维度、通道等值关系、机器人段数和图像绝对方向无关。需要做消融时可用
`--no-endpoint-fix` 保留原始细化端点。旧 `row_centroid` 的 `--tip-fix` 仍单独保留，
只修 node0，不参与现代双端修正。

节点顺序固定为 `node0=base -> nodeN-1=tip`。两段等长、15节点时：

```text
proximal: node0..node7   （7个区间）
distal:   node7..node14  （7个区间）
joint:    node7          （两段共享）
```

不等长机器人用实际长度比，例如 `--segment-lengths 80,120`。可用
`--base-anchor x,y`显式确定主路径哪一端是base；未提供时使用当前相机中较上方端点。

NPZ除 `positions` 和原始六维 `actions` 外，还保存：

```text
node_order=base_to_tip
skeleton_method
endpoint_fix
segment_lengths
segment_intervals
joint_node_indices
channel_source6
model_action_channels
model_action_dim
action_expansion6
robot_diameter_mm=16.0
robot_diameter_px
mm_per_px
```

### 4.1 由16 mm外径建立图像尺度

软体机器人外径统一按`16 mm`记录。中心线提取后，在弧长20%–80%的主体区域读取距离变换
半径，以逐帧主体宽度中位数估计`robot_diameter_px`，并计算：

```text
mm_per_px = 16 mm / robot_diameter_px
estimated_error_mm = image_error_px * mm_per_px
```

新数据把三个尺度字段写入train/val NPZ；评价脚本依次读取CLI覆盖值、NPZ合同和
`qc_skeleton/skeleton_metrics.csv`。该尺度适合相机视场内的平面形态误差、规划视野和
障碍间距解释。NDI仍提供独立的末端三维毫米评价，用于检验图像尺度估计及离平面运动。

阶段产物：

```text
<out-root>/qc_skeleton/
  skeleton_metrics.csv
  skeleton_metrics.png
  skeleton_overlays.png
  README.txt
```

叠加图中红色是输入mask，彩色线是各物理段，黄色节点是关节；端部青色叉是原始细化端点，
黄色圆是端帽修正点，二者之间的短线表示修正位移。若发生时间自动插值，青色线显示插值前
中心线。`skeleton_metrics.csv` 同时保存两端原始/修正坐标、修正原因、横向宽度与轴向延伸量。
时间QC默认只自动插值空mask、提取失败或非有限值。弧长/时间残差异常只标记
`suspicious`，不自动删除，因为六通道可能产生远离全序列中位但完全合法的形态。经过查看后
确认确实是坏帧，才使用 `--repair-suspicious`重新转换。

### 当前“两段”信息进入了哪里

当前 `segment_lengths/segment_intervals/joint_node_indices` 不是只为着色：它们把物理段边界
固定到稳定的节点编号，并用于NPZ数据合同、QC和分段/关节误差统计。但是训练数据集只向模型
提供整条 `(N,3)` 骨架和投影后的D维动作；状态转移模型只使用均匀节点位置嵌入，并沿整条
base→tip序列用同一个GRU传播。当前没有 segment embedding、关节标志、每段动作编码器、
每段刚度参数或关节连续性专用loss。因此模型会从数据中隐式学习两段耦合，但尚未显式使用
“这是两段机器人”的结构先验。

## 5. 训练与试次归档

训练入口不变：

```bash
python scripts/training/train_transition.py \
  --mode gt \
  --data_dir data/real_seq/<seq>/train

python scripts/training/train_transition.py \
  --mode open_loop \
  --data_dir data/real_seq/<seq>/train
```

长时间训练建议显式传`--save_interval 5`。训练器会持续维护`best_model.pt`（当前实现按
epoch训练loss创新低更新），每5轮覆盖`current_model.pt`、归档`model_epoch_XXXX.pt`，并保存
包含optimizer/scheduler/epoch的`training_state_latest.pt`；正常结束另有`final_model.pt`。

Dataset读取六维原始动作后，根据 `model_action_channels`投影为D维。因此同一套代码兼容：

- 六路完全独立：`action_dim=6`；
- 两对通道联动：可能为`action_dim=4`；
- 旧单通道序列：`action_dim=1`。

第一版模型仍把整条15节点机器人作为一个耦合对象，不拆成两个独立模型。分段合同首先用于
稳定节点对应和评价；只有关节/分段误差显示确有必要时，才增加segment embedding或关节loss。

完整真实训练使用统一试次入口：

```bash
GPU_ID=2 EVAL_GPU_ID=2 \
bash scripts/real/train_real_transition.sh \
  data/real_seq/<sequence_tag>/train \
  data/real_seq/<sequence_tag>/val
```

每次启动自动创建
`train_log/real_pipeline/<sequence_tag>/trial_YYYYMMDD_NNN/`。根`config.json`记录
数据、资源、epoch、batch、seed、窗口和teacher forcing合同；`commands.sh`记录实际命令；
`artifacts.json`在完成时给出所有关键产物的相对路径。GT与OpenLoop分别保存到`stages/gt/`
和`stages/open_loop/`，周期评价位于`evaluations/<mode>/periodic/`，最终best定量结果与照片叠图
位于`evaluations/<mode>/best/{quantitative,overlay}/`。每个试次因此可以整体复制、比较和归档。

正式实验不要只把同一条序列前80%/后20%作为论文训练/测试。应按独立采集run、seed或轨迹
拆分train/validation/test。

## 5.1 新序列的一键训练前处理

以下入口串联采集审计、固定裁剪、候选锚点、SAM2多GPU传播、完整性检查、15节点骨架和
全部QC，但**不会启动训练**：

```bash
python scripts/real/preprocess_capture.py \
  --seq real_capture/data/raw/<seq> \
  --roi x,y,w,h \
  --gpus 1,3 \
  --n-points 15 \
  --segment-lengths 1,1 \
  --qc-frames 1000,5000
```

每个物理GPU只暴露为其子进程的`cuda:0`，避免`CUDA_VISIBLE_DEVICES`重映射后出现
`invalid device ordinal`。脚本支持断点续跑；改变ROI时必须显式传`--overwrite-crop`，若已有
SAM2产物则还必须传`--reset-sam2`，防止旧mask与新空间合同混用。结束后自动写
`preprocess_manifest.json`和`PREPROCESS_REPORT.md`，其中`training_started=false`；人工检查
candidate/SAM2/skeleton QC后再单独启动训练。

## 6. 定量评价

```bash
python scripts/evaluation/eval_real_quant.py \
  --checkpoint <best_model.pt> \
  --data_dir data/real_seq/<test-seq>/val \
  --ndi real_capture/data/raw/<test-seq>/ndi.csv \
  --frame-times real_capture/data/raw/<test-seq>/frame_times.txt \
  --ndi-index 0 \
  --calibration-file calibration/px_to_ndi_plane.npz
```

评估会使用与训练相同的动作合同，把NPZ六维动作投影到checkpoint的D维输入，并输出：

- tip、全节点、每节点像素误差；
- Chamfer、Hausdorff和Procrustes形态误差；
- 每物理段误差、共享关节误差；
- 每个独立动作根通道的分箱误差；
- OpenLoop误差随离真实观测步数`k`的漂移；
- 由16 mm主体直径换算的tip、全节点、分段及视野误差毫米估计；
- NDI末端毫米误差；
- NDI z相对中位平面的p50/p95/max。

新GUI格式 `ndi0_x/ndi0_y/ndi0_z/.../ndi0_quality`和旧`x/y/z`格式均可读取。严格毫米
评价必须使用独立calibration序列保存的仿射文件；不提供`--calibration-file`时，脚本会明确
标记为`same_split_diagnostic`，其结果只能诊断，不能作为held-out论文结果。

视野认证也使用同一尺度：

```bash
python scripts/evaluation/eval_horizon.py \
  --checkpoints <OPEN_LOOP_BEST_MODEL_PT> \
  --data_dir data/real_seq/<test-seq>/val \
  --max_steps 40 --n_seeds 8 \
  --out output/horizon
```

`horizon_summary.json`同时保存`roll_px`、`roll_mm`、`Kmax_px_*`、`Kmax_mm_*`和尺度来源。
建立部署合同时，还会从训练骨架计算每个K下“起点到K步后的最大节点位移p95”，
写入`planning_displacement_px_p95`。`auto_k`根据当前目标的最大节点差距，选择位移表覆盖该距离的
最小K；超出表容量时使用`K_safe`与`k_max`共同限定的上限。

## 7. 目标形态逆规划与执行

完整形态目标使用与模型一致的15个节点，顺序固定为`base -> tip`。工作台将当前相机骨架和
最近`H`步`applied6`冻结为anchor，shooting planner直接优化`K × action_dim`个独立kPa动作，
再按`channel_source6`展开为六路压力。本阶段的两对联动关系为：

```text
模型动作: [ch0, ch1, ch3, ch5]
硬件展开: [ch0, ch1, ch1, ch3, ch3, ch5]
```

规划loss由终态全骨架距离、路径距离、单调接近、动作平滑和障碍代价组成；压力上下界、
rise/fall、训练动作周期和`K_safe`在每次规划后由Preflight检查。通过后，GUI中的
`Arm / Confirm`放行`PlanExecutor`按训练周期逐拍下发，并把每拍ACK中的`applied6`写入
`execution.csv`和下一次anchor历史。

离线读取已记录目标骨架并检查规划与Mock命令链：

```bash
python scripts/control/run_avoidance.py \
  --checkpoint <OPEN_LOOP_BEST_MODEL_PT> \
  --data-dir data/real_seq/<seq>/val \
  --t-init 120 --target-frame 160 \
  --target-tolerance 2 --auto-k --k-min 4 --k-max 40 \
  --n-iter 400 --n-restarts 4 \
  --rise 50,50,50,50,50,50 \
  --fall 50,50,50,50,50,50 \
  --mock-execute --out output/shape_control
```

输出目录包含`anchor.json`、`scene.json`、`safety.json`、`plan.json`、
`planned_actions6.csv`、`predicted_states.npz`和`execution.csv`。真机实验在GUI中加载同一
checkpoint与部署manifest，实时锚定后定义目标骨架、规划、预览、Preflight、Arm并执行。
长任务采用“执行不超过`K_safe`步→重新观测→重规划”的滚动方式。

这里需要区分三种数据：前向评价误差来自预测骨架与数据集记录骨架GT；离线规划结果中的
终态目标残差来自学到的前向模型计算；实物控制误差来自真机执行后重新观测的骨架与目标。
Mock ACK只检查命令、安全和执行生命周期。

OpenLoop定量表使用`is_prediction`标识预测行。窗口起点的真实骨架是rollout锚点，其后的
自回归结果才进入前向预测误差、动作分箱、直径尺度换算和节点误差曲线。

## 8. Legacy边界

以下算法只为复现旧“近端段静止、单段驱动”实验保留，不属于六通道主线：

- `scripts/real/repair_masks.py`：把静态近端段替换为跨帧宽度共识；
- `scripts/real/clean_transition_npz.py`：把静态近端段替换为跨帧节点共识；
- `masks_to_transition_npz.py --legacy-global-outlier`：按全序列中位删除大形变。

前两个脚本必须显式传`--allow-legacy-static-proximal`才允许运行，以防误用于两段都运动的
新数据。旧逐行质心仍可通过`--skeleton-method row_centroid`复现，但新数据默认不用。
