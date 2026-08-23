# 相机位置变化下的 ROI、训练状态与控制坐标

## 结论

实物平面控制采用 `robot_planar_mm_v1` 作为统一状态坐标：基座为原点，纵轴从基座
指向机器人末端，横轴与纵轴正交，长度单位为毫米。每段采集序列和每次在线实验分别
建立一个固定的相机像素正反变换。

ROI 属于分割层，可以随相机和序列调整。SAM2 mask 和骨架先恢复到源相机像素，再转换
到机器人坐标。模型、目标、障碍和 Planner 始终使用机器人坐标；GUI 通过逆变换叠回
当前相机画面。

这一合同覆盖相机成像平面内的平移、旋转和统一尺度变化。相机与机器人运动平面的夹角
保持接近采集条件时，相似变换能够稳定工作。明显的透视变化使用固定相机支架，或在基座
附近布置四个刚性标记并先求平面单应性；registration 继续用于检测图像配准质量。

## 数据流

```text
源相机帧
  └─ 序列专属方形 ROI
      └─ SAM2 mask
          └─ 15 节点像素骨架 + crop offset
              ├─ positions_camera_px             用于 QC 和照片叠图
              └─ 固定 SkeletonFrameTransform
                  └─ positions                    robot_planar_mm_v1
                      └─ Dataset 归一化
                          └─ checkpoint / Planner
                              └─ inverse transform → 当前相机叠图
```

## 坐标定义

节点仍按 `node0=tip → node14=base` 排列。对源相机像素点 `p`：

```text
origin = 序列中所有 base 节点的中位位置
axial  = 序列中 base 附近切向的中位单位向量（base → tip）
lateral = axial 顺时针旋转 90°
pixels_per_mm = 机器人 mask 主体直径像素 / 16 mm

[lateral_mm, axial_mm] = ([lateral, axial]^T · (p - origin)) / pixels_per_mm
```

离线变换使用整段序列的稳健中位数，在线变换由第一次相机 Anchor 的骨架和当前 mask
直径建立。在线变换随后固定在该实验中，重锚定继续复用，因此机器人自身运动产生的形态
变化完整保留。相机重新摆放后新建实验并重新 Anchor。

## ROI 规则

ROI 的目标是稳定分割速度和上下文。各序列使用自己的 ROI，并遵循：

1. 方形 ROI 覆盖完整 base、两段机器人、末端连接物在运动中的包络。
2. 四周按当前相机下的机器人直径留出 3 个直径的上下文。
3. 同一序列全部帧固定使用一个 ROI，并保存 `crop_meta.json`。
4. 模型输入尺寸由后续 resize/网络配置决定；ROI 像素位置和边长不进入模型状态。

已有 mask 可自动推荐下一轮固定 ROI：

输入 masks 对应的初始 ROI 需要覆盖完整机器人运动包络；推荐工具负责按物理直径统一
上下文并收紧边界。

```bash
python scripts/real/recommend_robot_roi.py \
  --masks-dir sam2/masks/seq_20260819_182519_full \
  --crop-meta real_capture/data/derived/seq_20260819_182519/crop/crop_meta.json \
  --diameter-px 20 --padding-diameters 3 \
  --out data/real_seq/seq_20260819_182519_n15_sam2_robot_mm/roi_recommendation.json
```

当前序列得到 `--roi 189,40,368,368`。该范围比原 `300×300` ROI 提供更完整的物理
上下文，适合相机重新安装后的下一轮处理。

## NPZ 合同

`masks_to_transition_npz.py` 默认输出机器人平面毫米状态：

| 字段 | 含义 |
|---|---|
| `positions` | `(T,3,N)`，模型使用的 `robot_planar_mm_v1` 骨架 |
| `positions_camera_px` | `(T,3,N)`，源相机像素骨架 |
| `state_coordinate_frame` | `robot_planar_mm_v1` |
| `state_length_unit` | `mm` |
| `skeleton_frame_transform` | 原点、纵轴、px/mm、3×3 正反矩阵和来源 |
| `image_crop_xywh` | 当前序列 ROI，用于 mask 恢复和 QC |
| `robot_diameter_mm/px` | 16 mm 与当前序列的像素直径 |

Dataset 会检查目录内全部 NPZ 的状态坐标合同。像素数据与毫米数据分别放在不同目录；
checkpoint 的 `config.json` 和 `deploy_manifest.json` 会记录状态坐标与单位，在线 Anchor
按该合同构建输入。

## 一键前处理

新序列沿用现有命令，默认状态坐标已经是机器人毫米坐标：

```bash
python scripts/real/preprocess_capture.py \
  --seq real_capture/data/raw/<seq> \
  --roi x,y,w,h --gpus 0 \
  --n-points 15 --segment-lengths 1,1 \
  --mask-close-k 11 --base-anchor base_x,base_y
```

已有 SAM2 masks 可以直接重建坐标数据：

```bash
python scripts/real/masks_to_transition_npz.py \
  --seq real_capture/data/raw/<seq> \
  --masks-dir sam2/masks/<seq>_full \
  --crop-meta real_capture/data/derived/<seq>/crop/crop_meta.json \
  --skeleton-method skeletonize --endpoint-fix --mask-close-k 11 \
  --n-points 15 --segment-lengths 1,1 --action-channels auto \
  --state-frame robot_planar_mm \
  --out-root data/real_seq/<seq>_n15_sam2_robot_mm
```

训练使用新的毫米坐标目录。现有像素 checkpoint 保留为基线，其状态合同为
`camera_pixel_v1/px`。

## 现有 10 Hz 数据验证

已从现有 SAM2 masks 生成：

```text
data/real_seq/seq_20260819_182253_n15_sam2_robot_mm/
data/real_seq/seq_20260819_182519_n15_sam2_robot_mm/
data/real_seq/seq_20260819_10hz_n15_sam2_robot_mm/
```

结果：

| 序列 | 帧数 | base 中位坐标 | px/mm | 纵轴 |
|---|---:|---:|---:|---:|
| `182253` | 1138 | `(0,0) mm` | 1.25 | `(0.0216,0.9998)` |
| `182519` | 3073 | `(0,0) mm` | 1.25 | `(0,1)` |

两段序列均为 15 节点、4 维模型动作视图，组合训练目录能够构建 3214 个长度 40 的
OpenLoop episode。坐标范围约为横向 `-39..40 mm`、纵向 `0..187 mm`，两段数据已经
进入同一机器人坐标分布。

## 验证检查

每次相机安装后检查：

- ROI 推荐图覆盖完整运动包络；
- skeleton QC 中 base/tip 和两段连接点正确；
- `skeleton_frame_transform` 的 px/mm 与当前直径一致；
- 在线 Anchor 画出的目标、障碍和预测轨迹能通过逆变换贴合当前画面；
- deploy manifest 的 `state_coordinate_frame`、`state_length_unit` 与 checkpoint 配套；
- 相机平面夹角变化明显时完成刚性标记单应性或恢复固定安装姿态。
