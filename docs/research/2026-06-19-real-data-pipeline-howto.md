# 实物数据采集→训练 跑通手册（How-to Runbook）

> 从"采集到真实图像/气压"一路跑到"训练出模型"的可执行步骤手册。
> 架构与"为什么"见 [`docs/directions/11_sim_to_real_transfer.md`](../directions/11_sim_to_real_transfer.md)（实物数据采集平台）。
> 本手册只讲"按什么命令一步步跑出结果"。更新：2026-06-19。

---

## 0. 我需要几个相机？

**核心原则**：3D 骨架 GT 要么靠**多视角三角化**（≥2 相机），要么靠 **NDI**。**1 个相机只能给 2D 骨架（没有深度）**，单独无法训练 3D 模型。

| 你的情况 | 相机数 | 能产出 | 能否训练 3D 模型 |
|----------|--------|--------|------------------|
| 单平面弯曲（当前单腔道驱动，1-DOF） | **2** | 三角化 3D 骨架 | ✅ 推荐 |
| 全 2-DOF（三腔道，任意方向弯） | **3**（推荐）/ 2 | 稠密 3D 骨架 | ✅ |
| 只有 1 相机 + **1-DOF 平面弯曲** | 1 | 2D 骨架 + `--planar-lift` 升 3D | ✅（仅平面弯曲，见下节） |
| 只有 1 相机 + 2-DOF | 1 | 仅 2D 骨架 | ❌（离面分量丢失） |
| 1 相机 + NDI 末端锚点 | 1 + NDI | 稀疏 3D 锚点 | ⚠️ 可，但 GT 稀疏 |

**建议**：
- **单平面弯曲起步 → 2 相机**：1 台在弯曲平面内（看清弯），1 台垂直平面外（给深度，三角化出 z）。距离 ~30–50 cm（10 cm 臂）。
- **2-DOF → 3 相机**，绕臂 ~120° 分布，避免两视角共面导致三角化退化。
- **相机就是部署时的传感器**——用相机做 GT 训练出的模型，部署时用同样的相机推理，没有跨模态 gap。这也是为什么"2–3 相机三角化"优于"纯 NDI"。
- **1 相机能先做什么**：先打通"分割 + 2D 骨架"链路（用 Step 2 的 `inspect_capture.py` 调阈值），等加到 ≥2 相机再产出 3D GT 训练。**或**——若是 1-DOF 平面弯曲，单相机直接用 `--planar-lift` 出 3D（见下节）。

---

## 单相机平面模式（1-DOF 原型最简路径，可选）

> 只有一个相机、且是**单腔道单方向弯曲**时，用 `--planar-lift` 把 2D 骨架升成 3D，
> 跳过多视角三角化。**几何上合法**：单方向弯曲的中心线落在一个平面内，相机正对该平面时
> 深度恒定，2D↔3D 是已知映射（射线-平面相交）。

**前提**：
- 弯曲是**平面**的（单腔道 1-DOF，臂在一个平面内弯）。
- 相机**正对**弯曲平面安装（平面平行于像面）→ 此时平面法向 ≈ 相机朝向（脚本默认值）。
- 已标定这 1 个相机（`calibrate_cameras.py` 只给 1 个视角即可）。

**命令**（替代多视角的 Step 3，单相机）：
```bash
python scripts/real/capture_to_npz.py \
  --view-dirs raw/seq1/cam0 \
  --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 --dt 0.0333 \
  --actions raw/seq1/pressure.npz --actions-has-timestamps --fps 30 \
  --planar-lift --clean-nan \
  --out data/real_seq/seq1.npz
```
- `--planar-lift`：启用平面升维（自动用**相机朝向**作平面法向 = 正对安装的默认）。
- `--plane-point X Y Z`：弯曲平面上一点（世界系米，默认基座原点 `[0,0,0]`；基座不在原点则改）。
- `--plane-normal NX NY NZ`：平面法向（**默认=相机朝向**；相机有俯仰/偏航、不正对时填真实法向）。
- 产出与多视角模式**完全同 schema** 的 `.npz` → `train_gt_transition` 直接吃，无需改训练命令。

**何时失效**：
- 弯曲**离面**（2-DOF、三腔道）→ 离面分量被投影掉，单相机无法恢复，必须 ≥2 相机三角化。
- 相机**不正对**平面、且未给正确 `--plane-normal` → 升维偏。

**精度**：取决于标定 + 分割 + 2D 骨架质量（与多视角三角化同源误差）；数值自检往返误差 <1e-12。

---

## 1. 采集前要准备好的原始数据

| 用途 | 内容 | 说明 |
|------|------|------|
| **标定（一次性）** | 每相机：多张不同姿态的**棋盘格图**（内参）+ 一张贴在**机器人基座处**的棋盘格图（外参/世界系） | 方格边长用尺子量一次（米） |
| **每段序列** | 每视角图像目录（或视频）+ **气压日志**（带时间戳！）+ 可选 NDI 末端锚点 | 气压与相机须**共享同一时钟原点**（见 §同步） |

> 棋盘格靶贴在机器人基座 → 靶自身坐标系 = 世界系 = robot-base 系。这是唯一需要"量"的几何。

---

## 2. 完整步骤（每步都有脚本）

```
Step 1  calibrate_cameras.py    标定 → camera_params.npz          （一次性）
Step 2  inspect_capture.py      QA：核对分割/2D骨架/三角化，调阈值  （先调好再批量）
Step 3  capture_to_npz.py       每段序列 → data/real_seq/*.npz      （批量，含清NaN+同步）
Step 4  (可选) 复查产出的 npz
Step 5  train_gt_transition.py  训练单步状态转移（主线）
Step 6  train_open_loop_transition.py  开环 rollout（热启动自 Step 5）
Step 7  看 eval_metrics.csv / visualize_3d_shape.py
```

### Step 1 — 标定（一次性）

```bash
python scripts/real/calibrate_cameras.py \
  --intrinsic-dirs calib/cam0 calib/cam1 \
  --extrinsic-imgs calib/cam0_world.jpg calib/cam1_world.jpg \
  --pattern 9 6 --square 0.005 \
  --H 480 --W 640 \
  --out config/real_camera_params.npz
```
- `--square`=方格边长（米）；`--pattern`=内角点 列 行。
- 产出 `camera_params(V,10)` + 内参 K/dist。**内参重投影误差应 <0.5 px**，否则重拍棋盘格。

### Step 2 — QA：先核对再批量（关键）

用几张图确认分割阈值/标定没问题，**避免没调好就跑全量**：

```bash
python scripts/real/inspect_capture.py \
  --view-dirs raw/seq1/cam0 raw/seq1/cam1 \
  --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 \
  --n-frames 3 --out inspect_seq1.png
```
- 打开 `inspect_seq1.png`：**红=分割掩码**应贴合整条臂、**青=2D 骨架**应在臂中心线上、**右列=3D 三角化**应是一条合理的空间曲线、相机位置（红▲）在周围。
- 反复调 `--method`（backlight/bg_subtract/color）和阈值（`--gray-thresh`/`--bg-thresh`），直到掩码干净。硅胶半透明→**背光剪影法最稳**。

### Step 3 — 批量产出 .npz（每段序列一次）

```bash
python scripts/real/capture_to_npz.py \
  --view-dirs raw/seq1/cam0 raw/seq1/cam1 \
  --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 \
  --dt 0.0333 \
  --actions raw/seq1/pressure.npz --actions-has-timestamps --fps 30 \
  --clean-nan \
  --out data/real_seq/seq1.npz
```
- `--clean-nan`：自动沿节点轴插值清三角化 NaN（**必加**，否则训练遇 NaN 会崩）。
- `--actions-has-timestamps --fps 30`：把高频气压按帧时刻插值对齐（见 §同步）。
- 每段序列重复，所有 `.npz` 放进 `data/real_seq/`。

### Step 4 —（可选）复查产出的 npz

```bash
python -c "
import numpy as np
d = np.load('data/real_seq/seq1.npz', allow_pickle=True)
print({k: d[k].shape for k in d.files})
print('positions NaN:', np.isnan(d['positions']).any(), '| actions[:3]:', d['actions'][:3])
"
```
- 确认 `positions` 无 NaN、`actions` 量级合理（kPa）。

### Step 5 — 训练：gt_transition（主线，单步）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_gt_transition.py \
  --data_dir data/real_seq --n_epochs 200
```
- 自动探测 action_dim（=2）、n_nodes（=31）；episode 模式，潜变量 z 跨帧演化→学迟滞。
- 输入=动作历史，GT=三角化骨架。日志→`train_log/gt_transition/exp_<日期>/`；每 `eval_interval=50` epoch 自动写 `eval_metrics.csv` + `transition_metrics.json`。

### Step 6 —（可选）开环 rollout：open_loop（热启动自 Step 5）

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_open_loop_transition.py \
  --data_dir data/real_seq \
  --init_from train_log/gt_transition/exp_XXXX/phase_gt_transition/model/best_model.pt
```
- 默认自动找最新 `gt_transition` 的 `best_model.pt` 热启动；tf=0 纯闭环。

### Step 7 — 看结果

- `train_log/gt_transition/exp_*/eval_metrics.csv`：看 `mean_node_mm`（部署精度，mm）、`mean_drift`（漂移×）、`model_vs_copy`（是否优于"不动"基线，<1 为优）。
- `python scripts/evaluation/visualize_3d_shape.py`（已支持开环 rollout 模式 + 指标嵌 HTML）。

---

## 3. 气压–相机帧同步（重点）

**问题**：气压日志通常是**高频带时间戳**（如 1000 Hz，`(M, 1+A)`，第 0 列秒），相机只有 30 Hz。两者不是 1:1，不能直接按行配对。

**为什么按帧插值就够**：软臂机械时间常数 τ≈0.5–3 s ≫ 帧间隔（1/30 s≈33 ms）。臂对"一帧之内"的气压高频波动**无响应**——所以取**每帧时刻**的气压值，物理上不丢信息（这是正确的降采样，不是有损）。高频记录只是过采样。

`capture_to_npz.py` 按你手上有什么自动选模式：

| 你手上的气压日志 | 用法 |
|------------------|------|
| `(M,1+A)` 第 0 列=秒，相机 30fps、与气压同 t=0 | `--actions-has-timestamps --fps 30` |
| `(M,A)` 无时间列，但已知采样率 1000 Hz | `--actions-rate 1000 --fps 30` |
| `(N,A)` 已按帧对齐（你上游已下采样） | 只传 `--actions`（默认截断/补零） |
| 每帧有独立时间戳文件（硬件触发，最佳） | `--actions-has-timestamps --frame-times raw/seq1/frame_times.txt` |

**时间原点要求**：`--fps` 模式假设相机第 i 帧在 `t = i/fps` 秒，且与气压日志**共享 t=0**。所以：
- **同时按下**气压记录与相机录制（同一触发/同一按键），或
- 记一个**同步事件**（如相机视野里闪一下 LED，同时在气压日志打标记），事后从两边各减去该事件时刻。

**示例（带时间戳的高频气压）**：
```bash
python scripts/real/capture_to_npz.py \
  --view-dirs raw/seq1/cam0 raw/seq1/cam1 \
  --camera-params config/real_camera_params.npz \
  --method backlight --gray-thresh 60 --dt 0.0333 \
  --actions raw/seq1/pressure.npz --actions-has-timestamps --fps 30 \
  --clean-nan --out data/real_seq/seq1.npz
```
> 若气压**完全没有时间信息**也无法确定采样率 → 无法对齐，必须先在采集端给气压打时间戳（最简单：记录时和相机共用一个起始触发）。

---

## 4. NaN 处理（现已自动）

三角化对遮挡/分割失败会产生 NaN，而 `StateTransitionDataset`/训练**不处理 NaN** → loss 变 NaN 崩溃。

`--clean-nan` 自动做：逐帧沿节点轴线性插值补全 NaN（相邻节点空间接近）；整帧全失败→置零。
- 若某帧"整帧失败"（置零）较多，训练时那帧是错的——**最好直接丢弃该帧**（或提高分割质量）。可用 Step 4 检查有多少帧被置零。

---

## 5. 常见坑

| 坑 | 对策 |
|----|------|
| 硅胶半透明 → 分割碎裂 | **背光剪影法**（臂成暗剪影）最稳；或消光涂层/染色 |
| 10 cm 小臂 → 像素不够 | 近距离（30–50 cm）+ 高分辨率/手机 4K |
| 标定漂移 → 三角化系统性偏 | 重投影误差 <0.5 px；定期复标；相机刚性固定 |
| 气压 offset/scale 不对 | 记录前清零（消除大气压偏置）；统一 kPa 单位 |
| `arm_length` 用错（默认 0.5 仿真值） | 实物 ~0.1 m → `%arm` 类相对指标需改 `transition_metrics.ARM_LENGTH`（绝对 mm 指标不受影响） |
| GPU 显存 | 测试实验用 `CUDA_VISIBLE_DEVICES=1`（脚本默认） |

---

## 6. 脚本清单（谁做什么）

| 脚本 | 作用 |
|------|------|
| `scripts/real/calibrate_cameras.py` | 棋盘格图 → `camera_params(V,10)` + 内参 |
| `scripts/real/inspect_capture.py` | QA：几张图的分割/2D骨架/3D三角化核对图 |
| `scripts/real/capture_to_npz.py` | 图像+气压 → 仿真 schema `.npz`（去畸变+分割+2D骨架+三角化+清NaN+同步+组装） |
| `scripts/real/_smoke_triangulation.py` | 自检：标定↔camera_params↔三角化 数值往返（<1e-12） |
| `scripts/training/train_gt_transition.py` | 训练单步状态转移（主线） |
| `scripts/training/train_open_loop_transition.py` | 训练开环 rollout（热启动） |

底层模块：`src/calibration/`（标定+格式桥）、`src/data/real/`（io_video/segmentation/triangulation/assemble_npz/preprocess），2D 骨架复用 `src/utils/skeleton_2d.py`。

---

**一句话流程**：`calibrate_cameras`（一次）→ `inspect_capture`（调阈值）→ `capture_to_npz --clean-nan --actions-has-timestamps --fps 30`（每序列）→ `train_gt_transition --data_dir data/real_seq` → 看 `eval_metrics.csv`。
