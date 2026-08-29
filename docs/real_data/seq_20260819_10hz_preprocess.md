# 2026-08-19 约 10 Hz 真实数据处理与训练手册

> 2026-08-26 节点合同升级：本文记录的旧派生 NPZ 与 checkpoint 已归档。
> 从原始序列重跑自动前处理后，节点统一为 `node0=base -> node14=tip`。

## 1. 数据范围与时间合同

本轮使用两个同步采集序列：

| 序列 | 帧数 | 中位采样间隔 | 平均采样间隔 | 中位频率 | 动作覆盖 |
|---|---:|---:|---:|---:|---|
| `seq_20260819_182253` | 1138 | 0.1090 s | 0.11013 s | 9.174 Hz | ch0、ch1 激励；ch2 跟随 ch1 |
| `seq_20260819_182519` | 3073 | 0.1090 s | 0.11045 s | 9.174 Hz | 四个独立根通道均激励 |

两段序列使用相同动作合同：

```text
raw_action_dim = 6
channel_source6 = [0, 1, 1, 3, 3, 5]
model_action_channels = [0, 1, 3, 5]
model_action_dim = 4
```

模型窗口保持 40 步，对应约 `40 × 0.109 = 4.36 s`。部署时应使用本轮实测
`train_dt≈0.110 s`，不能沿用约 5 Hz 模型的时间步长。完整合同保存在：

```text
data/real_seq/seq_20260819_10hz_n15_sam2/dataset_manifest.json
```

## 2. 原始数据审计

两段数据的图像、动作、时间戳和 NDI 行数均与帧数一致，训练动作读取 `actions6.csv` 中的
最终 applied6。审计产物位于：

```text
real_capture/data/derived/seq_20260819_182253/audit/
real_capture/data/derived/seq_20260819_182519/audit/
```

复现命令：

```bash
MPLCONFIGDIR=/tmp/selfsoftrobot-mpl python scripts/real/audit_capture.py \
  --seq real_capture/data/raw/seq_20260819_182253

MPLCONFIGDIR=/tmp/selfsoftrobot-mpl python scripts/real/audit_capture.py \
  --seq real_capture/data/raw/seq_20260819_182519
```

## 3. 图像、SAM2 与 15 节点骨架

相机位置与 `seq_20260819_172644` 一致，固定方形 ROI 为：

```text
x=220, y=68, w=300, h=300
```

该 ROI 覆盖 base 连接处、完整两段机器人和末端，同时保持机器人位于画面中央。两段数据均
使用 SAM2 mask、最长中轴主路径、双端端帽中心修正和 15 节点弧长重采样。节点顺序为
`node0=base → node14=tip`，两段等长时 `node7` 为共享节点。

一键复现命令：

```bash
python scripts/real/preprocess_capture.py \
  --seq real_capture/data/raw/seq_20260819_182253 \
  --roi 220,68,300,300 --gpus 0 \
  --n-points 15 --segment-lengths 1,1 \
  --mask-close-k 11 --base-anchor 370,110 \
  --qc-frames 0,569,1137

python scripts/real/preprocess_capture.py \
  --seq real_capture/data/raw/seq_20260819_182519 \
  --roi 220,68,300,300 --gpus 1 \
  --n-points 15 --segment-lengths 1,1 \
  --mask-close-k 11 --base-anchor 370,110 \
  --qc-frames 0,1536,3072
```

本次实际结果：

| 序列 | SAM2 mask | 有效骨架 | 自动插值 | 时间 QC 可疑帧 | 直径尺度 |
|---|---:|---:|---:|---:|---:|
| `182253` | 1138/1138 | 1138/1138 | 0 | 0 | 16 mm / 20 px = 0.8 mm/px |
| `182519` | 3073/3073 | 3073/3073 | 0 | 0 | 16 mm / 20 px = 0.8 mm/px |

关键检查文件：

```text
real_capture/data/derived/<seq>/crop/qc/crop_overview.png
real_capture/data/derived/<seq>/qc_candidate/selected_anchors.png
sam2/masks/<seq>_full/qc/compare_candidate_sam2_shard0.png
data/real_seq/<seq>_n15_sam2/qc_skeleton/skeleton_overlays.png
data/real_seq/<seq>_n15_sam2/qc_skeleton/skeleton_metrics.png
```

本轮抽查确认 mask 聚焦机器人主体，细线没有改变主中心线；tip/base 均落在端部宽边中心，
node7 沿两段连接位置稳定。

## 4. 多 run 训练数据

训练目录内保留两个独立 NPZ。Dataset 将其识别为两个 run，动作历史和 OpenLoop episode 均
在各自序列内构造：

```text
data/real_seq/seq_20260819_10hz_n15_sam2/
  dataset_manifest.json
  train/
    seq_20260819_182253_train.npz   # 911 帧
    seq_20260819_182519_train.npz   # 2459 帧
  val/
    seq_20260819_182519_val.npz     # 614 帧，主验证
```

`seq_20260819_182253` 的末尾 227 帧保存在其原始处理目录，作为低维动作子空间的附加验证：

```text
data/real_seq/seq_20260819_182253_n15_sam2/val/
```

多 run 评价会根据所选 NPZ 文件名查找同名训练 NPZ，从而得到正确的原始帧偏移，并关联对应
原图、mask、frame_times 和 NDI。

## 5. 训练配置选择

正式配置沿用上一轮已经验证的预算：

```text
GT:                  60 epoch
OpenLoop:           240 epoch
batch_size:         128
window_size:         40
episode_len:         40
tf_anneal_epochs:    40
checkpoint interval: 5 epoch
evaluation interval: 10 epoch
GPU:                  0（训练与周期评价）
seed:                 20260823
```

上一轮收敛证据：GT 验证全节点误差在 epoch 40–50 进入约 `1.04 px` 平台；OpenLoop 训练
loss 在 epoch 19 达到最低，epoch 60 后稳定。60/240 为本轮提供了完整 GT 收敛区间和
OpenLoop 调度后的训练余量。本轮首个 GT epoch 约 22 秒，完整流水线预计约 2 小时。

正式试次：

```text
train_log/real_pipeline/seq_20260819_10hz_n15_sam2/trial_20260823_001/
```

该目录统一保存配置、命令、GT/OpenLoop checkpoint、周期评价、最终定量结果和照片叠图。
`trial_20260823_000` 保存了 5 个 GT epoch 的启动检查和退出状态，正式结果以 `_001` 为准。

## 6. tmux 启动与查看

推荐使用持久启动脚本：

```bash
GPU_ID=0 EVAL_GPU_ID=0 \
GT_EPOCHS=60 OPEN_LOOP_EPOCHS=240 \
BATCH_SIZE=128 NUM_WORKERS=4 \
SAVE_INTERVAL=5 PERIODIC_EVAL_INTERVAL=10 PERIODIC_MAX_STEPS=500 \
SEED=20260823 WINDOW_SIZE=40 EPISODE_LEN=40 TF_ANNEAL_EPOCHS=40 \
bash scripts/real/start_training_tmux.sh real10hz_train \
  data/real_seq/seq_20260819_10hz_n15_sam2/train \
  data/real_seq/seq_20260819_10hz_n15_sam2/val
```

当前正式训练已经在会话 `real10hz_train` 中运行：

```bash
# 查看所有会话
tmux list-sessions

# 进入训练会话；按 Ctrl-b，再按 d，可退出界面并保持训练
tmux attach -t real10hz_train

# 直接查看最近输出
tmux capture-pane -pt real10hz_train:pipeline.0 -S -100

# 查看结构化状态和 loss
cat train_log/real_pipeline/seq_20260819_10hz_n15_sam2/trial_20260823_001/status.txt
tail -n 20 train_log/real_pipeline/seq_20260819_10hz_n15_sam2/trial_20260823_001/stages/gt/phase_gt_transition/loss_log.csv

# 需要结束训练时向 pane 发送 Ctrl-C，试次会记录退出码
tmux send-keys -t real10hz_train:pipeline.0 C-c
```

tmux 会话是否保留取决于窗口进程的生命周期。直接用
`tmux new-session -d -s <name> "<training-command>"` 启动时，训练命令就是窗口的唯一进程；
命令完成后窗口关闭，最后一个窗口关闭后会话随之结束。本项目的启动脚本先创建交互 shell，
再发送训练命令，并设置 `remain-on-exit on`，所以训练完成后仍可进入会话查看最终退出码。

完成检查后可主动关闭会话：

```bash
tmux kill-session -t real10hz_train
```

## 7. 训练完成后的产物

每 5 epoch 保存：

```text
stages/<stage>/phase_<stage>_transition/model/current_model.pt
stages/<stage>/phase_<stage>_transition/model/best_model.pt
stages/<stage>/phase_<stage>_transition/checkpoints/model_epoch_XXXX.pt
stages/<stage>/phase_<stage>_transition/checkpoints/training_state_latest.pt
```

每 10 epoch 的定量评价与照片叠图保存在：

```text
evaluations/gt/periodic/epoch_XXXX/
evaluations/open_loop/periodic/epoch_XXXX/
```

完整流水线结束后，最终 best 模型结果位于：

```text
evaluations/gt/best/quantitative/
evaluations/gt/best/overlay/
evaluations/open_loop/best/quantitative/
evaluations/open_loop/best/overlay/
```

根目录中的 `config.json` 记录两个训练 NPZ、主验证 NPZ、数据 manifest、GPU 和全部训练参数；
`commands.sh` 记录实际执行命令；`artifacts.json` 在完整结束时生成。

## 方向合同重建后的训练结果（2026-08-29）

当前完整试次为：

```text
train_log/real_pipeline/seq_20260819_10hz_n15_sam2_robot_mm/trial_20260828_000/
```

OpenLoop 验证集全节点均误从 epoch 20 的 `1.825 mm` 降到 epoch 40 的 `1.481 mm`，
epoch 80–240 稳定在约 `1.474 mm`；epoch 240 全量验证为全节点 `1.476 mm`、tip
`3.161 mm`。因此该试次的 `best_eval_model.pt` 选择 epoch 240。这个结论来自模型预测骨架与
记录骨架GT的前向误差，不表示真机控制精度。

最终叠图位于：

```text
train_log/real_pipeline/seq_20260819_10hz_n15_sam2_robot_mm/trial_20260828_000/
  evaluations/open_loop/best/overlay/
```
