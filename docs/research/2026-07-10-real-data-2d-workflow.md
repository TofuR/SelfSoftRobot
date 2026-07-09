# 实物数据工作流：免标定 2D 骨架 → 状态转移 → NDI 度量验证

> 本文记录**当前在用**的实物 1-DOF 双段软体臂工作流（单相机、**免标定**、2D 图像骨架作 state、
> NDI 末端作独立度量验证）。与 [`2026-06-19-real-data-pipeline-howto.md`](2026-06-19-real-data-pipeline-howto.md)
> 的区别：那篇是**多视角标定→3D 三角化**路径；本篇是**单相机免标定→2D 像素骨架**路径
> （无需棋盘格标定、无需三角化，state 直接是图像骨架 [col,row,0]）。
> 更新：2026-07-10。

---

## 0. 一句话定位

学一个状态转移模型 `ŝ_t = F(s_{t-1}, a_t)`，其中 **state = 2D 图像骨架** `[col,row,0]`（像素），
**action = 归一化气压** ∈[0,1]。整套流程**免相机标定**；末端**毫米精度**用独立采集的 **NDI** 验证。
GT-observed（每步观测）vs open-loop（开环 rollout）两种部署在同一网络上对比。

---

## 1. ★ 坐标空间：预测到底是 px 还是 mm？怎么和 NDI 对应？

这是实物评估最容易混的点，先讲清楚：

| 量 | 空间 | 来源 |
|---|---|---|
| 骨架 GT / 模型预测 / mask | **像素 [col,row], z≈0** | 图像（免标定管线） |
| NDI 末端 | **毫米 [x,y,z]**（tracker 帧） | NDI 传感器（独立度量） |
| 漂移比 drift_by_k | 无量纲（归一化空间） | rollout/onestep MSE 比 |

**预测是 px，不是 mm。** 模型 forward 在归一化空间运算，反归一化 `world = norm·pc_scale + pc_center`
回到 **像素** [col,row,z]。z 通道 `pc_scale≈eps` 使其恒≈0（平面 1-DOF 假设）。所以：
- **整体形态误差只能算 px**——31 个节点只有图像 GT，没有度量 GT。
- **末端误差 px + mm 都能算**——末端有 NDI 度量 GT。

**px↔mm 怎么对应（免相机标定）：** NDI 末端 (x,y,z mm) 与图像骨架 node0 (col,row px) 是同一物理点、
逐帧配对。末端在平面内做 ~1-DOF 弯曲（NDI 实测 x 扫 ~24mm、y 扫 ~9mm，2D 铺开）。用全部帧
**(GT node0 px ↔ NDI x,y mm)** 最小二乘拟合 **2D 仿射** `A: (col,row,1)→(x,y)`：
- **拟合残差 RMS = 标定噪声底**（mask 骨架化 + NDI 噪声 + 非平面）；
- 模型末端像素经同一 `A`→mm，与 NDI 比 → **末端毫米误差**。

**实测**（GT 模型 `exp_20260709_5`，2500 帧）：标定底 **0.74 mm**；模型末端 mean **0.77 mm** /
median 0.57 / p90 1.4 mm → 底亚毫米，**mm 可信**，模型已到噪声底。

> 为什么不用相机矩阵投影？免标定管线**没有度量 3D、没有标定内参**；末端表示本身就活在图像平面。
> 可视化叠图也同理：模型 (col,row) 直接是图像像素 → 直接画 `(x=col, y=row)`、丢 z（详见
> `visualize_real_overlay.py` 顶部说明）。相机矩阵是给 sim（度量 3D+内参）用的，对实物是二次变换、会扭曲。

---

## 2. 数据布局

```
real_capture/data/raw/<seq>/
    cam0/<NNNNN>.png        原图 480×640 BGR（5 位补零，10214 帧）
    actions6.csv            t_sec,c0..c5（气压 kPa；本序列只驱动 ch0）
    ndi.csv                 t_sec,x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality（末端 mm + 姿态）
    frame_times.txt         每帧 t_sec（相机+气压+NDI 同 t0 对齐）
    pressure.csv / meta.json
real_capture/data/derived/<seq>/masks/<NNNNN>.png   二值 mask 0/255（white_on_blue 分割产物）
data/real_seq/<seq>/{train,val}/*.npz               基础 npz（tip 修复后）
data/real_seq/<seq>_clean/{train,val}/*.npz         清洗后 npz（静态段共识）← 训练吃这个
    positions:(T,3,N) float32 [col,row,0]   actions:(T,1) float32 ∈[0,1]
    train=首 80% / val=末 20%（时序连续切分）
```

---

## 3. 流水线（每步：脚本 / 功能 / 算法 / 关键参数）

### 3.1 分割（mask）— 已完成
`white_on_blue` 分割（白半透明臂 / 蓝背景 / 白气管）。R&D + 批量脚本已在 `scripts/real/`（segment_batch）。
产物在 `derived/<seq>/masks/`。**本工作流从这里起。**

### 3.2 骨架提取 + npz 合成 — `scripts/real/masks_to_transition_npz.py`
- **功能**：mask → 2D 骨架（逐行质心 + 弧长重采样到 N 点）+ `tip_fix`（垂直尖端切片修 corner）
  → 离群骨架（手/管茬）时间插值 → actions 归一 [0,1]（操作上限 `hi6`）→ 时序切分 train/val。
- **算法**：
  - 逐行质心：每行白像素列均值，底→顶，弧长重采样。
  - **末端 corner 修复**（`tip_fix`）：弯管 cap 倾斜时逐行质心把 node0 落到角落（34% 帧）；
    改为"垂直于局部轴的尖端切片质心"=cap 中点（corner 帧 −71%，详见 `compare_skeleton_methods.py`）。
- **关键参数**：`--n-points N`（默认 31，**可改 21/15，全流水线自适应**）、`--tip-fix`（默认开）。
- **CLI**：
  ```bash
  python scripts/real/masks_to_transition_npz.py --seq real_capture/data/raw/seq_20260627_163921
  # 降节点: --n-points 21
  ```

### 3.3 静态段共识清洗 — `scripts/real/clean_transition_npz.py`
- **功能**：双段臂只驱动末端 → 动作段（node0..关节）保留真实弯曲；静态段（关节..base）用
  跨帧中位**共识**替换（修分割抖动 / 关节偏移 / 上方 mask 缺块）。动作气压全保留。
- **算法**：
  - `detect_joint_xy`：robust 估关节绝对位置（取跨帧局部 col 突偏峰值 node 的中位；搜索范围按 N 分数）。
  - `stabilize_static_region`：每帧关节 node = 离绝对位置最近（handles node-id 漂移）；静态段弧长重采样→跨帧中位共识→按每帧弧长映射回。
- **关键参数**：`--act-dev-thresh`（残余离群阈值，默认 60px）、`act_nodes`（自适应 0.6·N）。
- **CLI**：
  ```bash
  python scripts/real/clean_transition_npz.py --seq seq_20260627_163921
  ```

### 3.4 训练 — `scripts/training/train_transition.py --mode gt|open_loop`
- **功能**：同一网络（StateTransitionSpatialModel 派生），差别在 teacher forcing：
  - `gt`（主线）：每步喂真实 s_{t-1}（tf=1.0）→ s 不漂移，部署=每步观测。
  - `open_loop`：窗口开环（tf 退火到 0，喂自身预测）→ 漂移随 k 累积，部署=观测一次预测 K 步。
- action_dim / n_nodes 自动探测。
- **CLI**：
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
      --mode gt --data_dir data/real_seq/seq_20260627_163921_clean/train
  # open_loop（热启动自最新 gt）:
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
      --mode open_loop --data_dir data/real_seq/seq_20260627_163921_clean/train
  ```

### 3.5 定量评估 — `scripts/evaluation/eval_real_quant.py`
- **功能**：四块指标 ① 末端 NDI mm（仿射自标定）② 像素部署（tip/node/chamfer/hausdorff/procrustes px）
  ③ 分段 tip/mid/base + 按 action 分箱 ④ open_loop drift_by_k。聚合：每帧 csv + 整体 + 分箱 + drift。
- **CLI**：
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
      --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_clean/train
  ```
- 输出 `output/real_quant/<exp>/`：`summary.txt` + `per_frame.csv` + 图（err_vs_action、drift_by_k、per_node_profile、tip_trajectory_mm）。

### 3.6 可视化 — 三个脚本
- `visualize_real_overlay.py`：**模型预测叠在真实照片上**（原图+mask+GT 骨架+预测骨架同框）。
  模型 (col,row) 直接画像素、丢 z（见 §1）。
- `inspect_real_data.py`：骨架网格（9 帧 2D + 3D 预览），看数据本身。
- `composite_frames.py`：批量 原图+mask+骨架 叠图（10214 帧 + montage）。
- **CLI**（overlay）：
  ```bash
  CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/visualize_real_overlay.py \
      --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_clean/train
  ```

---

## 4. 模块化 / 快速对比（A/B）

为快速对比不同参数/方法，关键开关都做成一处改、全流水线跟着走：

| 想对比 | 改什么 | 影响范围 |
|---|---|---|
| **骨架节点数** | `masks_to_transition_npz --n-points N` | 关节检测/静态共识/末端修复/训练/评估全部按 N 分数自适应（已验证 N=31/21/15 给同一物理关节与末端） |
| **末端修复** | `--tip-fix` / `--no-tip-fix`（npz、composite） | 骨架提取是否修 corner |
| **骨架化方法** | `scripts/real/compare_skeleton_methods.py` | 7 法末端 corner 对比（独立真值+bend 分层） |
| **gt vs open_loop** | `train_transition --mode` + `eval_real_quant --mode` | 部署语义 + drift |

> 节点索引已全部改成 N 的分数（关节搜索 ~0.25–0.85·N、静态共识 ~0.4·N、动作段 ~0.6·N、
> 末端修复 body 节点 ~0.10/0.25·N），故降节点**不需手调任何魔法数**。

---

## 5. 关键算法索引

- **末端 corner 修复**（perpendicular tip fix）：`src/utils/skeleton_2d.py::_perpendicular_tip_fix`；
  根因+7 法对比见 `scripts/real/compare_skeleton_methods.py` 与 memory `skeleton-tip-corner-fix`。
- **静态段共识**（绝对位置锚定）：`scripts/real/masks_to_transition_npz.py::{stabilize_static_region, detect_joint_xy}`。
- **NDI 仿射自标定**（px→mm）：`scripts/evaluation/eval_real_quant.py::{load_ndi_tip, fit_affine_px_to_mm}`。
- **rollout/漂移**：`src/evaluation/transition_metrics.py`（窗口开环 + drift_by_k）。

---

## 6. 脚本清单

| 脚本 | 作用 |
|---|---|
| `scripts/real/masks_to_transition_npz.py` | mask+actions → 免标定 2D npz（tip_fix+离群插值+归一） |
| `scripts/real/clean_transition_npz.py` | 静态段共识清洗（绝对位置锚定） |
| `scripts/real/composite_frames.py` | 批量 原图+mask+骨架 叠图 |
| `scripts/real/compare_skeleton_methods.py` | 7 法末端 corner 对比 |
| `scripts/training/train_transition.py` | 统一训练入口（`--mode gt\|open_loop`） |
| `scripts/evaluation/eval_real_quant.py` | 定量评估（NDI mm + 形态 px + drift） |
| `scripts/evaluation/visualize_real_overlay.py` | 预测叠真实照片 |
| `scripts/evaluation/inspect_real_data.py` | 骨架数据网格诊断 |

底层：`src/utils/skeleton_2d.py`（2D 骨架+tip_fix）、`src/evaluation/transition_metrics.py`（rollout 指标）、
`src/evaluation/shape_metrics.py`（chamfer/hausdorff/f-score）。

---

**一句话流程**：`masks_to_transition_npz`（mask→2D npz, tip_fix）→ `clean_transition_npz`（静态共识）
→ `train_transition --mode gt/open_loop --data_dir ..._clean/train` → `eval_real_quant`（末端 mm via NDI + 形态 px）
→ `visualize_real_overlay`（预测叠照片）。
