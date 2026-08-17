# 实物三维管线第一版实施计划

> 日期：2026-08-17
>
> 分支：`feat/real-3d-pipeline`
>
> 目标：先形成一条能用 Mock/合成数据完整运行、随后可接真实双相机的最小闭环；不在第一版解决复杂自遮挡学习或三维 GUI。

## 1. 本版验收结果

```text
RealSense RGB+对齐Depth / 普通USB RGB / Mock
→ real_capture只保存原始模态、时间戳和设备元数据
→ capture_to_npz离线提取每视角15节点骨架
→ 标定后DLT三角化并输出重投影误差/可见性/置信度
→ positions(T,3,15)训练文件
→ 现有GT/OpenLoop transition + masked 3D loss + skeleton reprojection loss
```

第一版成功标准：

1. 原采集系统的单/多 RealSense RGB 行为保持兼容；
2. RealSense 可选保存与 color 对齐的原始 `uint16` depth；
3. 普通相机通过 OpenCV `VideoCapture` 接入，与 RealSense 一起进入相同 RGB freshness 合同；
4. 采集目录只含原始观测和日志，不执行分割、骨架化或三角化；
5. 离线转换默认 `n_nodes=15`，保留二维证据和三维重建质量；
6. transition loader 能消费置信度和多视角二维投影，训练器能计算带 mask 的 3D loss 与重投影 loss；
7. 合成双视角数据能跑通采集/重建核心测试和至少一个训练 step。

## 2. 层次边界

### A. `real_capture/`：只做原始数据采集

允许：

- 枚举/打开相机；
- RGB、depth 和硬件/单调时间戳缓存；
- 动作门控后把同一拍的所有 RGB 与可用 depth 保存；
- 保存 source、serial、分辨率、depth scale 和 frame age。

禁止：

- 图像分割；
- 2D/3D 骨架；
- 相机标定；
- 三角化；
- 训练数据清洗。

### B. `src/data/real/` 与 `scripts/real/`：离线观测处理

- 读取 `camN/` 和可选 `depthN/`；
- 去畸变、分割和15节点中心线；
- 用棋盘格标定结果构造投影矩阵；
- DLT、正深度和重投影质控；
- 输出节点级 visibility、confidence、source mask；
- 为兼容训练生成 `positions(T,3,15)`。

### C. `src/data/` 与 `src/training/`：模型监督

- 模型主体继续使用现有 `(N,3)` state transition；
- loader 携带 3D confidence、2D observation、visibility 和投影矩阵；
- 高置信三角化节点进入 3D loss；
- 所有可见二维节点进入重投影 loss；
- 插值/模型补全节点不能与实测三角化节点等权。

## 3. 原始采集格式 v2

```text
seq_xxx/
├── cam0/00000.png
├── cam1/00000.png
├── depth0/00000.png          # 仅启用depth的源存在，uint16原始单位
├── frame_times.txt
├── actions6.csv
├── samples.csv
├── camera_times.csv          # 每拍各RGB/depth时间与age
└── meta.json
```

`meta.json` 新增：

```json
{
  "schema_version": 2,
  "camera_sources": [
    {"kind": "realsense", "serial": "...", "has_depth": true, "depth_scale_m": 0.001},
    {"kind": "opencv", "device": "0", "has_depth": false}
  ]
}
```

RGB 是每个视角的必需模态。配置为 depth 的源必须提供与该 RGB frameset 对齐且不过期的 depth，否则整拍拒绝；普通 RGB 相机没有 depth 不构成失败。

当前 freshness 是软件采样对齐，不声称硬件同步。`camera_times.csv` 必须让后处理能够计算跨相机曝光时差；真实动态三角化能否使用由该时差门控。

## 4. 相机源实现

### 4.1 RealSense

- 保留 `frame_ready(BGR, monotonic)` 兼容信号；
- 可选同时启用 depth stream；
- 用 `rs.align(rs.stream.color)` 得到与 RGB 同尺寸 depth；
- 发 `depth_ready(depth_uint16, monotonic, depth_scale_m)`；
- Mock depth 使用确定性的合成深度，便于测试。

### 4.2 普通相机

新增 `OpenCVCam(QThread)`：

- `cv2.VideoCapture(int index | path)`；
- 输出与 RealSense 相同的 `frame_ready`；
- 不声称有 depth；
- 打开失败明确发 error，不回退其他设备。

相机描述串第一版使用：

```text
realsense:SERIAL
realsense-depth:SERIAL
opencv:0
opencv:/dev/video2
mock
mock-depth
```

GUI/CLI 把描述串交给 factory，recorder 不关心厂商类型。

## 5. 离线15节点重建

`capture_to_npz.py` 新增/调整：

- `--n-nodes 15` 为默认；
- `--max-reprojection-error-px`；
- 输出 `positions_2d(T,V,N,2)`；
- 输出 `visibility(T,V,N)`；
- 输出 `reprojection_error(T,V,N)`；
- 输出 `position_confidence(T,N)`；
- 输出 `source_mask(T,N)`；
- 图像和 mask 明确从 `(V,T,...)` 转成训练 schema `(T,V,...)`，避免视角/时间轴混淆。

第一版仍假设每视角的 `tip→base` 等弧长点能够对应，适合无遮挡/轻遮挡验证。重投影异常节点会被 mask，而不是静默作为 GT。复杂自遮挡的数据关联留到第二版。

## 6. 训练改动

### 6.1 Loader

`StateTransitionDataset` 在字段存在时额外返回：

```text
position_confidence: (T,N)
positions_2d:        (T,V,N,2)
visibility:          (T,V,N)
projection_matrices: (V,3,4)
image_size:          (2,) = [H,W]
```

旧 NPZ 没有这些字段时行为不变。

### 6.2 Loss

三维节点项：

```text
sum(confidence * ||pred_3d - gt_3d||²) / sum(confidence)
```

重投影项：

```text
pred_norm → 反归一化3D → P_v投影 → 与positions_2d比较
只计算visibility=True节点，并按图像对角线归一化
```

普通 `train_transition.py` 默认不启用重投影，保持旧实验完全兼容。新入口
`train_real_3d_transition.py` 默认启用重投影并先执行数据合同检查。

## 7. 测试顺序

1. RealSense Mock RGBD 能输出同 timestamp 的 BGR/depth；
2. 普通 OpenCV 视频源能输出 RGB，打开失败明确报错；
3. recorder 混合 `mock-depth + mock RGB` 写出等数量 cam/depth 和 schema v2 meta；
4. 已知双相机/已知3D曲线的三角化误差和重投影误差；
5. `capture_to_npz` 输出 `positions.shape == (T,3,15)`，图像轴为 `(T,V,...)`；
6. 旧二维/三维 NPZ loader 回归；
7. 带 confidence/reprojection 的 transition loss 前向反向各一步；
8. `git diff --check` 和相关 unittest。

## 8. 第一版明确不做

- 不把 NDI 作为模型或 Planner 输入；
- 不在采集 GUI 中执行棋盘格标定；
- 不在采集线程中骨架化；
- 不训练遮挡补全网络；
- 不实现三维场景编辑与三维避障 Planner；
- 不把软件 freshness 写成硬件同步；
- 不把插值/先验补全节点命名为真实 GT。

完成本版并看真实双相机的有效节点率、重投影误差和 NDI tip 误差后，再决定第二版优先增加第三视角、稀疏标记还是时序遮挡关联。

## 9. 2026-08-17 实际运行记录

已在无硬件环境执行正式离线入口，而不是只调用内部函数：

```bash
python scripts/real/smoke_3d_pipeline.py --frames 12 \
  --out-dir /tmp/selfsoftrobot_real3d_final_smoke_20260817
```

结果：双视角 12 帧全部经过背光分割、15 节点骨架化、DLT 和重投影门控；
`positions=(12,3,15)`、`positions_2d=(12,2,15,2)`、
`images=(12,2,480,640)`，实测三角化节点比例为 100%。

随后用该 NPZ 在 CPU 完成一轮最小 GT-observed 训练 smoke：动作维数 6、节点数 15，
同时启用 confidence-masked skeleton、masked spatial smooth 和 skeleton reprojection。
GT-observed 只用于确认训练合同和热启动；论文部署主线仍是之后用同一数据入口训练
`--mode open_loop` 的窗口化 OpenLoop 模型。

回归验证：`python -m unittest discover -s tests` 共 171 项通过。当前结果尚未包含
真实相机标定、真实硬件同步误差、自遮挡数据关联或真实三维 Planner/GUI，因此不能把
合成数据的 100% 有效率当作真实实验结论。
