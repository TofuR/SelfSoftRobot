# 实物数据完整处理流程(免标定 2D)

> 实物 1-DOF 双段软体臂的端到端数据处理与训练流程:从原始照片到状态转移模型再到 NDI 度量验证。
> 本文档是稳定参考(非变更日志),合并并取代以下三份早期文档:
> - `docs/archived/research/2026-06-19-real-data-pipeline-howto.md`(早期多视角标定→3D 三角化路径,现归档为替代路线)
> - `docs/archived/research/2026-07-10-real-data-2d-workflow.md`(免标定 2D 主线)
> - `docs/archived/research/2026-07-14-mask-node-pipeline.md`(mask→node 每步算法 + QC 列含义)
>
> 相关文档:硬件与采集协议见 [`capture_setup.md`](capture_setup.md);模型与训练架构见 `CLAUDE.md`。
>
> 本文保留旧单通道序列的稳定处理说明。双段六腔但按等值对约束在平面内的当前实验，使用
> [`planar_constrained_6ch_workflow.md`](planar_constrained_6ch_workflow.md)：骨架仍是单相机
> 15 节点二维 GT，但动作、NPZ、模型和部署合同均保持六维。

---

## 1. 概述:免标定思想

学一个状态转移模型 `ŝ_t = F(s_{t-1}, a_t)`,其中:

- **state = 2D 图像骨架** `[col, row, 0]`,单位是像素,平面假设(z 恒≈0)。
- **action = 归一化驱动量** ∈[0,1](本序列驱动 ch0,操作上限 `hi6`)。

整套流程**免相机标定**:不拍棋盘格、不算内参/外参、不做三角化。直接把图像骨架当 state 训练,部署时也用同一相机推理,没有跨模态 gap。

### 为什么不标定?

- 1-DOF 单平面弯曲时,臂的中心线落在一个平面内。相机正对该平面拍摄,2D 图像骨架已经编码了弯曲的全部信息(深度恒定)。
- 标定/三角化是额外误差源(重投影、畸变、多视角配准),对平面 1-DOF 反而是噪声。
- 相机矩阵投影是给仿真(度量 3D + 内参)用的;对实物是二次变换、会扭曲。

### 那么"毫米精度"从哪来?

NDI 6DOF tracker **独立采集**末端三维位置(mm),作度量验证而非训练输入。模型末端像素经仿射自标定(见 §7)换算成 mm,与 NDI 比 → **末端毫米误差**。

> 注意:**整体形态误差只能算 px**(只有图像 GT,没有度量 GT);**末端误差 px + mm 都能算**(末端有 NDI 度量 GT)。

### 两条部署语义(同一网络)

- **gt 模式**(主线):每步喂真实 s_{t-1}(teacher forcing=1.0) → s 不漂移,部署=每步观测。
- **open_loop 模式**:窗口开环 rollout(tf 退火到 0,喂自身预测) → 漂移随 k 累积,部署=观测一次预测 K 步。

---

## 2. 数据采集(raw)

硬件简述(详见 [`capture_setup.md`](capture_setup.md)):硅胶双段臂 + 单腔道驱动;TwinCAT PLC(pyads)控电机推注射器加压,电机位置=控制量;Arduino 只读气压;单 Intel RealSense 相机(当 RGB 用);NDI 6DOF tracker 追踪末端。

采集程序已实现在 `docs/ref/Main UI-plc/`(`main_capture.py` GUI / `dataset_recorder.py` headless / `realsense_cam.py`),时间同步 相机+气压+电机位置,直接产出 raw 目录。

每段序列产出如下结构:

```
real_capture/data/raw/<seq>/
    cam0/<NNNNN>.png        原图 480×640 BGR(5 位补零)
    actions6.csv            t_sec,c0..c5(气压 kPa;本序列只驱动 ch0)
    ndi.csv                 t_sec,x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality(末端 mm + 姿态)
    frame_times.txt         每帧 t_sec(相机+气压+NDI 共享同一 t0)
    meta.json               序列元信息
```

**时间同步关键**:气压日志高频带时间戳,相机 30 Hz。软臂机械时间常数 τ≈0.5–3 s ≫ 帧间隔(33 ms),臂对一帧之内的气压高频波动无响应——所以取每帧时刻的气压值是物理上无损降采样。要求气压记录与相机录制共享同一 t=0(同一触发或同步事件标记)。

---

## 3. mask 处理

### 3.1 分割(white_on_blue)

白半透明硅胶臂 / 蓝背景 / 白气管 → 阈值出二值 mask(0/255)。脚本:`scripts/real/segment_rd.py`(R&D 单帧)/ `scripts/real/segment_batch.py`(批量)。产物:

```
real_capture/data/derived/<seq>/masks/<NNNNN>.png   二值 mask(含分割腐败,见下)
```

**分割会出三类腐败**(下文修复):

| 腐败类型 | 现象 | 典型帧 |
|---|---|---|
| 静态段截断 | 顶部宽 17(应 31) | f4080 |
| 动作段半 mask | 中段只剩右半(宽 19) | f4902 |
| 整帧手污染/臂缺失 | area 异常或臂没到顶部行 | f2330, f2316 |

### 3.2 mask 级修复 — `scripts/real/repair_masks.py`(三步,顺序执行)

修复 mask 本身的分割误差(不重骨架化)。产物:`derived/<seq>/masks_repaired/`。这是**形态预测的 GT**(shape GT),但**不进 npz 训练链**。

| 步骤 | 函数 | 修什么 | 怎么修 |
|---|---|---|---|
| ① 手帧 | `repair_hand_frames` | 整帧手污染/管茬/臂缺失(area>1.5×中位 或 臂没到顶部行>20) | 找最近 clean 邻帧(顶部行≤20 且 0.7~1.3×中位 完整臂),逐行 [min,max] col 按 α 线性插值整帧替换(跟随臂运动,手被剔除) |
| ② 静态段 | `repair_static_segment` | 静态段(关节以上)顶部截断/抖动 | 逐行跨帧 [min,max] col 中位共识,每帧静态行替换为共识宽 |
| ③ 动作段 | `repair_actuated` | 动作段(关节以下)半 mask/缺块 | 时间插值(主):那块在单帧不可见(半透明),从邻帧补——边重合配准(不用腐败质心);宽度补全(辅,无健康邻帧时单边扩展) |

**关节行定位**:用**宽度凸起**(管-臂合并处最宽 ~36 vs 常态 ~31)而非质心 std——顶部质心 std 受缺失噪声干扰反而偏大,宽度凸起是结构性、跨帧稳定的。实测关节 row~96,与 node 层检测一致。

```bash
# 默认三步全开
python scripts/real/repair_masks.py --seq seq_20260627_163921
# 手动指定关节行
python scripts/real/repair_masks.py --seq seq_20260627_163921 --joint-row 95
# 仅预览前 N 帧
python scripts/real/repair_masks.py --seq seq_20260627_163921 --limit 50
```

**实测效果**:f4080 静态段顶部被截成 w=17 → 共识修复回 w=31;f4902 area 6050→8669;跨帧逐行质心 col std 4.57 → 0.75 px(静态段);动作段弯曲保留。

### 3.3 SAM2 视频分割(可选替代)

作为 white_on_blue + repair_masks 的替代轨道:用 SAM2 视频分块双向分割,产物 `masks_sam2/`。SAM2 分割本身较干净,通常跳过 repair_masks 直接送骨架化。当前默认数据有两条来源:

- `*_n15_rep_clean`(repaired mask + 15 节点 + clean)
- `*_n15_sam2_clean`(SAM2 mask + 15 节点 + clean)

---

## 4. 骨架化

从 mask 提 2D 骨架:核心在 `src/utils/skeleton_2d.py::extract_skeleton_2d`,由 `scripts/real/masks_to_transition_npz.py` 调用。

### 4.1 算法(三步)

1. **逐行质心**:每行白像素列均值,底→顶排列成有序点列。
2. **弧长重采样**:沿点列弧长均匀重采样到 N 个节点(默认 N=15)。
3. **末端 corner 修复**(`_perpendicular_tip_fix`):弯管 cap 倾斜时,逐行质心会把末端 node0 落到 cap 尖角根(约 34% 帧);改为"垂直于局部轴的尖端切片质心"=cap 中点。

### 4.2 为什么末端要单独修

弯管的半透明 cap 在水平切片(逐行)里是倾斜的,水平质心落在 cap 角落而非中心。`tip_fix` 改用垂直于局部骨架轴的切片取质心,把 node0 拉回 cap 中点。

**7 法对比**(独立真值 + bend 分层,`scripts/real/compare_skeleton_methods.py`)证明 tip_fix 最优:corner 帧 node0 偏移 −71%。

### 4.3 关键参数(模块化,一处改全链路)

- `--n-points N`(默认 15,可改 21/31)。节点索引全部按 **N 的分数**自适应:关节搜索 ~0.25–0.85·N、静态共识 ~0.4·N、动作段 ~0.6·N、末端修复 body 节点 ~0.10/0.25·N。**降节点不需手调任何魔法数**。
- 已验证 N=31/21/15 给**同一物理关节与末端**(节点索引分数化的直接收益)。

---

## 5. npz 构建 + 清洗

### 5.1 npz 构建 — `scripts/real/masks_to_transition_npz.py`

- **功能**:mask → 2D 骨架(§4 算法 + `tip_fix`) → 离群骨架(手/管茬)时间插值 → actions 归一 [0,1] → 时序切分 train/val。
- **切分**:train=首 80% / val=末 20%(时序连续)。

```bash
python scripts/real/masks_to_transition_npz.py --seq seq_20260627_163921
# 自定义 mask 源(repaired / sam2)
python scripts/real/masks_to_transition_npz.py --seq seq_20260627_163921 \
    --masks-dir real_capture/data/derived/seq_20260627_163921/masks_repaired
# 降节点
python scripts/real/masks_to_transition_npz.py --seq seq_20260627_163921 --n-points 21
```

产物 npz schema:

```
positions:(T, 3, N) float32   # [col, row, 0] 像素
actions: (T, 1)    float32   # 归一 [0,1]
# 元数据(data_prep 子 dict):n_points、tip_fix 等
```

> npz 元数据(`data_prep`)会透传进训练实验的 `config.json` 和 `model_card.txt`,便于辨识"这个模型用的什么骨架节点数 / 是否开了末端修复"。

### 5.2 node 级清洗 — `scripts/real/clean_transition_npz.py`(两步)

双段臂只驱动末端 → 动作段(关节以下)保留真实弯曲;静态段(关节以上)用跨帧中位共识替换(修分割抖动 / 关节偏移 / 上方 mask 缺块)。动作气压全保留。

| 步骤 | 函数 | 修什么 | 怎么修 |
|---|---|---|---|
| ① 整帧离群 | `clean_outlier_skeletons` | 整帧骨架偏离时间中位 >80px(管茬等) | 时间插值整帧骨架 |
| ② 静态段共识 | `stabilize_static_region` | 静态段节点抖动 | `detect_joint_xy` robust 估关节绝对位置(跨帧局部 col 突偏峰值 node 中位);每帧关节 node = 离绝对位置最近(handles node-id 漂移);静态段弧长重采样 → 跨帧中位共识 → 按每帧弧长映射回 |

```bash
python scripts/real/clean_transition_npz.py --seq seq_20260627_163921
# 调残余离群阈值(默认 60px)
python scripts/real/clean_transition_npz.py --seq seq_20260627_163921 --act-dev-thresh 60
```

产物(训练吃这个):

```
data/real_seq/<seq>_clean/{train,val}/*.npz   # 或 *_n15_rep_clean / *_n15_sam2_clean
```

### 5.3 两条修复轨道(独立、不累计)

至此有两条独立的修复轨道,各自从原始 `masks/` 出发,互不喂给对方:

| 轨道 | 脚本 | 修什么 | 产物 | 用途 |
|---|---|---|---|---|
| **node 轨道** | `masks_to_transition_npz` + `clean_transition_npz` | 骨架化后的节点偏移/抖动 | `data/real_seq/<seq>_clean/*.npz` | **训练 state**(本工作流吃这条) |
| **mask 轨道** | `repair_masks` | mask 本身的分割误差(不重骨架化) | `derived/<seq>/masks_repaired/` | **形态 GT**(IoU/形态误差时单独引用) |

> node 层用共识把点拉回了,但 mask 仍可能错,不能当形态 GT。所以两条轨道并存,不累计(`masks_repaired/` 不进 npz 训练链)。

### 5.4 QC 可视化

`viz_qc.py dataset` 产全链路图 `full_chain_*.png`,每帧 4 列(左→右):

| 列 | 标注 | 内容 | 是"最终"吗 |
|---|---|---|---|
| 列1 | RAW mask | 原始分割 mask(含腐败) | 否,仅对比 |
| 列2 | `{src} mask` | repair_masks 后的 mask | 是,**形态 GT** |
| 列3 | `skeleton from {src}` | 从 repaired mask 提的骨架(tip_fix,清洗前) | 中间 |
| 列4 | `CLEANED skeleton` | clean 后的骨架(npz 里存的) | 是,**训练 state** |

列1→列2 = mask 修复效果;列3→列4 = 骨架清洗效果(通常差别很小,主要静态段共识);列2 vs 列1 才是 mask 修复主战场。

---

## 6. 训练

统一入口 `scripts/training/train_transition.py --mode gt|open_loop`。同一网络(StateTransitionSpatialModel 派生),差别仅在 teacher forcing:

- `gt`(主线):tf=1.0,每步喂真实 s_{t-1}。
- `open_loop`:tf 退火到 0(喂自身预测),窗口开环 rollout,通常热启动自最新 gt。

action_dim / n_nodes 自动从 npz 探测。

```bash
# GT(主线)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode gt --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/train

# Open-loop(热启动自最新 gt,自动找 best_model.pt)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode open_loop --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/train \
    --init_from train_log/gt_transition/exp_XXXX/phase_gt_transition/model/best_model.pt

# SAM2 数据
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode gt --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train
```

**模型要点**(详见 CLAUDE.md):

- 学状态转移 `s_t = F(s_{t-1}, a_t, z_{t-1})`。
- `FractionalMemory` 编码动作历史(分数阶幂律记忆核,匹配粘弹性迟滞)。
- 可学习迟滞潜变量 z(GRUCell 跨帧演化,无 GT 端到端学)。
- 沿臂空间 GRU。
- 预测增量 `s_t = s_{t-1} + delta_scale·tanh(Δ)`。

**实验辨识**:训练产出在 `train_log/<mode>_transition/exp_<date>/`,根目录两处落盘:

- `config.json`(机器可读):`n_nodes`、`z_dim`、`episode_len`、`encoder_type`、`action_dim`、`window_size`、`hidden_dim` + `data_prep` 子 dict(从 npz 透传 `n_points`/`tip_fix`)。
- `model_card.txt`(一行人类可读):扫目录时一眼辨识 `<model_tag> | <Model> | n_nodes=.. action_dim=.. ... | data: ..._clean | data_prep: n_points=.. tip_fix=..`。

---

## 7. 评估与可视化

### 7.1 定量评估 — `scripts/evaluation/eval_real_quant.py`

四块指标:① 末端 NDI mm(仿射自标定) ② 像素部署(tip/node/chamfer/hausdorff/procrustes px) ③ 分段 tip/mid/base + 按 action 分箱 ④ open_loop drift_by_k。聚合:每帧 csv + 整体 + 分箱 + drift。

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
    --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
    --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/train

# Open-loop 窗口评估
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_real_quant.py \
    --checkpoint train_log/open_loop_transition/exp_*/phase_open_loop_transition/model/best_model.pt \
    --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/val \
    --mode open_loop --window-len 40
```

输出 `output/real_quant/<exp>/`:`summary.txt` + `per_frame.csv` + 图(err_vs_action、drift_by_k、per_node_profile、tip_trajectory_mm)。

### 7.2 NDI 仿射自标定(px → mm)

这是"免标定但能量出 mm"的核心。模型 forward 在归一化空间运算,反归一化 `world = norm·pc_scale + pc_center` 回到像素 [col, row];z 通道 `pc_scale≈eps` 使其恒≈0。

NDI 末端 (x,y,z mm) 与图像骨架 node0 (col,row px) 是同一物理点、逐帧配对。末端在平面内做 ~1-DOF 弯曲(NDI 实测 x 扫 ~24 mm、y 扫 ~9 mm)。用全部帧 **(GT node0 px ↔ NDI x,y mm)** 最小二乘拟合 2D 仿射 `A: (col,row,1)→(x,y)`:

- **拟合残差 RMS = 标定噪声底**(mask 骨架化 + NDI 噪声 + 非平面)。
- 模型末端像素经同一 `A` → mm,与 NDI 比 → **末端毫米误差**。

> 为什么不用相机矩阵投影?免标定管线没有度量 3D、没有标定内参;末端表示本身就活在图像平面。相机矩阵是给 sim(度量 3D + 内参)用的,对实物是二次变换、会扭曲。

**实测参考**(GT 模型 2500 帧):标定底 0.74 mm;模型末端 mean 0.77 mm / median 0.57 / p90 1.4 mm → 底亚毫米,mm 可信,模型已到噪声底。

### 7.3 可视化(三脚本)

```bash
# 模型预测叠真实照片(原图 + mask + GT 骨架 + 预测骨架同框)
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/visualize_real_overlay.py \
    --checkpoint train_log/gt_transition/exp_*/phase_gt_transition/model/best_model.pt \
    --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/train

# 骨架数据网格诊断(9 帧 2D + 3D 预览,看数据本身)
python scripts/evaluation/inspect_real_data.py \
    --data_dir data/real_seq/seq_20260627_163921_n15_rep_clean/train

# 批量 原图+mask+骨架 叠图
python scripts/real/composite_frames.py --seq seq_20260627_163921
```

> overlay 可视化里模型 (col,row) 直接画像素、丢 z(见 §1 坐标空间)。

---

## 8. 形态预测基线(可选,量化 NN 空间)

`scripts/real/skeleton_to_shape.py`:复用 SDF 思路(SDF = dist_to_skeleton − radius),骨架画成厚度 2r 的粗折线 + 节点圆 = 半径 r 的管。量化"形态被骨架+常数半径解释了多少",决定 NN decoder / action 是否必要。

- **uniform**:全臂常数半径 r(自动拟合 max-IoU vs GT mask)。
- **variable**:per-node 半径(从 GT mask 估局部半宽 = `distance_transform_edt`),处理 taper/末端;这是 offset 法的上界(预测时无 GT,需 NN 预测半径)。

```bash
python scripts/real/skeleton_to_shape.py \
    --npz data/real_seq/seq_20260627_163921_n15_rep_clean/train/*.npz \
    --masks-dir real_capture/data/derived/seq_20260627_163921/masks_repaired
# per-node 半径上界
python scripts/real/skeleton_to_shape.py \
    --npz ... --masks-dir .../masks_repaired --mode variable
```

**实测结论**:uniform r=14 → IoU 0.91(vs repaired mask)。即形态 ≈ 骨架 + 常数管,**NN 空间小(~9%)**;残差(压力依赖宽度变化、末端 cap 形、taper)才是 NN 该学。action 窗口仅在"形态有骨架未捕捉的压力依赖形变"时才需要(骨架已编码弯曲,瞬时形态 ≈ f(骨架))。

---

## 9. 模块化 / 快速对比(A/B)

关键开关都做成一处改、全流水线跟着走:

| 想对比 | 改什么 | 影响范围 |
|---|---|---|
| 骨架节点数 | `masks_to_transition_npz --n-points N` | 关节检测/静态共识/末端修复/训练/评估全部按 N 分数自适应 |
| 末端修复 | `--tip-fix` / `--no-tip-fix`(npz、composite) | 骨架提取是否修 corner |
| mask 来源 | `--masks-dir`(repaired / sam2) | 骨架化输入的 mask 质量 |
| 骨架化方法 | `scripts/real/compare_skeleton_methods.py` | 7 法末端 corner 对比(独立真值 + bend 分层) |
| gt vs open_loop | `train_transition --mode` + `eval_real_quant --mode` | 部署语义 + drift |

---

## 10. 脚本清单

### 数据处理(`scripts/real/`)

| 脚本 | 作用 |
|---|---|
| `segment_rd.py` / `segment_batch.py` | white_on_blue 分割(R&D / 批量) |
| `repair_masks.py` | mask 级三步修复(手帧/静态/动作)→ `masks_repaired/` |
| `masks_to_transition_npz.py` | mask+actions → 免标定 2D npz(tip_fix+离群插值+归一) |
| `clean_transition_npz.py` | node 级清洗(整帧离群+静态段共识)→ `*_clean/` |
| `compare_skeleton_methods.py` | 7 法末端 corner 对比 |
| `composite_frames.py` | 批量 原图+mask+骨架 叠图 |
| `skeleton_to_shape.py` | 骨架+半径形态基线(uniform/variable) |
| `viz_qc.py` | 全链路 QC 图(`full_chain_*.png` 4 列) |

### 训练(`scripts/training/`)

| 脚本 | 作用 |
|---|---|
| `train_transition.py` | 统一训练入口(`--mode gt\|open_loop`) |
| `train_gt_transition.py` / `train_open_loop_transition.py` | 薄封装(等价于 `--mode`) |

### 评估与可视化(`scripts/evaluation/`)

| 脚本 | 作用 |
|---|---|
| `eval_real_quant.py` | 定量评估(NDI mm + 形态 px + drift) |
| `visualize_real_overlay.py` | 预测叠真实照片 |
| `inspect_real_data.py` | 骨架数据网格诊断 |

### 底层模块(`src/`)

| 模块 | 作用 |
|---|---|
| `src/utils/skeleton_2d.py` | 2D 骨架提取 + `_perpendicular_tip_fix` |
| `src/evaluation/transition_metrics.py` | rollout + `drift_by_k` |
| `src/evaluation/shape_metrics.py` | chamfer / hausdorff / f-score |
| `src/training/trainer_unified.py` | `UnifiedTrainer` + `config.json`/`model_card.txt` 落盘 |

---

## 11. 一句话流程

```
masks_to_transition_npz (mask→2D npz, tip_fix)
    → clean_transition_npz (静态段共识)
    → train_transition --mode gt --data_dir ..._clean/train
    → eval_real_quant (末端 mm via NDI + 形态 px)
    → visualize_real_overlay (预测叠照片)
```

> 形态 GT 走另一条 mask 修复轨道:`repair_masks`(静态段宽共识 → `masks_repaired/`)→ `skeleton_to_shape`(骨架+半径基线,量化 NN 空间)。两条修复轨道独立、不累计(详见 §5.3)。
