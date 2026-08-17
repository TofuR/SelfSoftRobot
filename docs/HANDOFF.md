# 接手指南 · 给下一个 AI agent

> **你是新接手 SelfSoftRobot 的 AI agent?先读这一篇。**
> 目标:5 分钟建立正确心智模型 + 知道"当前在哪/别破坏什么/怎么跑最新东西"。
> 这份文档面向 AI agent,**与现有人向文档互补、不重复**——人向深度参考见文末导航索引。
> 最后更新:2026-07-28 · 分支 `feat/real-data-transition`

> **⚠️ 2026-07-28 三条决策(读本文其余部分前先看这里,因为它们改了两条前提)**
> 1. **方法收敛**:只留 `open_loop_transition`(部署主线)+ `gt_transition`(论文消融)。C-MSTNF / MS-SCNF / SDF / SkeletonSDF / FlowMatch / SpatialSequence 的实验日志已归档到 `train_log/_archive/`(1.27 G);**代码暂不动**。清单:[`docs/archived/2026-07-28_archive_manifest.md`](archived/2026-07-28_archive_manifest.md)。
> 2. **3D 多自由度**:后续 6 通道全部气压驱动,状态来源改为**双/多相机标定 + 三角化** → **§7.2 不变量 #7"免标定"已反转**(详见该处)。方向 06 与 08 升为 ★★★ 主线。
> 3. **mask 源**:训练继续用 SAM2,**在线改跑 SAM2 前向流式**。已知缺口:前向流式 vs 训练用的双向分块仍有差异(在线没有"启发式修复的干净锚帧"这个来源);GPU 单帧延迟未实测。
>
> 本文其余部分描述的 2D 免标定路线**仍然有效**(它是 3D 之前的基线),但不再是终点。

---

## 0. 一页 TL;DR(读完这节你就上手)

1. **项目本质**:软体机器人**神经场自建模**——只用驱动参数(扭矩/气压)+ (仿真才有)查询点,预测软臂形态。基于 FBV-SM(Hu et al. 2025),从刚体臂扩展到 PyElastica 软体连续臂。
2. **两条路线,只有一条是活的**:
   - **(A) 仿真**:PyElastica + PyVista + 多视角标定 + 体渲染 + 度量 3D。**已稳定/归档,非当前焦点**。
   - **(B) 实物免标定(当前主线)**:真实硅胶臂 + 单 RealSense,**不用相机标定**,state = 图像像素骨架 `[col,row,0]`,NDI 仅作末端 mm 验证。
3. **当前焦点 = 实物状态转移模型族 + 其上的控制/规划**:`s_t = F(s_{t-1}, a_t, z_{t-1})`,带可学习迟滞潜变量 z。`gt`(TF=1.0,每步观测,精度上界)→ `open_loop`(TF=0,开环 rollout,**部署目标**)→ 在 open_loop 上做视野认证 + 可微逆规划 + 避障。
4. **当前最好的实验**:`gt_transition/exp_20260714_7`(精度上界,best_loss 0.00077)、`open_loop_transition/exp_20260714_8`(部署模型,best_loss 0.080,带全套 eval)。**两者都训在 `data/real_seq/seq_20260627_163921_n15_sam2_clean`**(15 节点,SAM2 mask)。
5. **最该记住的不变量**:模型输入**只有驱动参数**(图像/深度仅监督);**免标定**(无相机矩阵);**gt≠部署目标,open_loop 才是**;**gt/open_loop 共享同一份 state_dict**,靠 buffer + config.json 区分。
6. **诚实边界**:所有定量结论目前是**模型内验证**(val 集 + GT-actions 基线证明模型保真),**planner 优化出的动作还没在真机上执行过**。下一步里程碑是真机闭环。

> ⚠️ **当前工作树非干净**(handoff 时):`docs/reports/2026-07-15_control_shape_planning.html`(已改)、`scripts/utils/build_control_report.py`(已改)、`docs/presentations/2026-07-15_shape_control_planning.md`(新增未跟踪)、`data/readme.md`(已删)。接手时先 `git status` 确认。

---

## 1. 项目是什么

让软体机器人仅通过观察自身外部图像(2D 相机),学会预测自身在不同驱动下的形态。NeRF 范式:用体渲染把 3D 场投影到 2D,用 2D 图像做监督。软体特有挑战 = **迟滞(路径依赖)** + **深度歧义(单视角)**。

模型演进(详见 `docs/overview/status.md`):
```
MSTNF(直接 NeRF) → C-MSTNF(典范+变形) → MS-SCNF(显式骨架+条件密度,仿真主线)
                                              ↓ 从"action→state 前馈"升级为"状态转移闭环"(解决迟滞)
                                   StateTransition 族(gt / open_loop)—— 实物当前主线
                                              ↓ 把前向模型当可微仿真器
                                   控制/规划(视野认证 + 逆规划 + 避障)—— 最新层
```

---

## 2. 两条路线

| 路线 | 采集 | 标定 | state 表示 | 监督 | 度量验证 | 状态 |
|------|------|------|-----------|------|---------|------|
| **(A) 仿真** | PyElastica + PyVista | 多视角标定 + 内参 | 3D 节点 `positions(T,3,31)` | 体渲染 / 3D SDF | 仿真 GT 直接对比 | 稳定/归档 |
| **(B) 实物免标定** | 真 RealSense D400 | **免标定,无内参** | **2D 像素骨架 `[col,row,0]`** | 2D 骨架回归 | **NDI 6DOF tracker** 末端 mm | **当前主线** |

**实物硬件一句话**:1-DOF 双段硅胶臂,TwinCAT PLC+电机推注射器气动(真实控制量=电机位置 mm),单 RealSense,NDI 末端追踪。详见 `docs/real_data/capture_setup.md`。

> **核心约定(所有模型)**:模型输入**只有驱动参数 + 查询点**。图像/深度仅作监督信号,**绝不直接输入模型**(唯一例外:`PCSpatialSequenceModel` 的修正相,用图像做残差修正)。

---

## 3. 当前焦点的心智模型(实物状态转移 + 控制)

### 3.1 前向模型:状态转移族

学**状态转移** `ŝ_t = s_{t-1} + clamp(delta_scale, max=delta_scale_max)·tanh(Δ)`,其中 Δ 由三部分决定:
- **FractionalMemory**(默认编码器):分数阶 Grünwald-Letnikov 幂律记忆核,匹配硅胶粘弹性迟滞(区别于 EMA 指数衰减)。
- **可学习迟滞潜变量 z**(`nn.GRUCell`,跨帧演化,**无 GT**,端到端从 skeleton loss 学)。
- **沿臂空间 GRU**(悬臂梁因果,base→tip 传播;向量化 `nn.GRU`,单次 cuDNN 调用)。

三个姊妹模型**共享同一基类 `StateTransitionSpatialModel` + 同一份 state_dict**,仅 `training_spec` 不同:

| 模型 | 文件 | TF | delta_scale_max | s_{t-1} 来源 | 定位 |
|------|------|----|-----------------|------------|------|
| StateTransitionSpatialModel | `model_state_transition.py` | 可调 | inf | 预测(无界 rollout) | 方向13,纯自回归对照(实测漂移 1170×,不可用) |
| **GTObservedTransitionModel** | `model_gt_transition.py` | **1.0** | inf | **每步真实观测** | **训练基础 + 精度上界**(部署=每步观测) |
| **OpenLoopTransitionModel** | `model_open_loop_transition.py` | **0.0** | **1.0** | 预测,**每 K 步重观测** | **部署目标**(观测一次预测 K 步) |

> **关键认知**:`gt` 与 `open_loop` 是**同一个网络、同一份权重键**,区别只在 `teacher_forcing_ratio` + 一个 marker buffer(`gt_observed_mode`/`open_loop_mode`)+ `delta_scale_max`。gt 每步喂真实 s 故零漂移;open_loop 纯开环故会漂移但能离线规划。**任何部署/控制/规划讨论必须强调 open_loop 是目标,gt 是上界。**

### 3.2 控制层(最新、文档最少)

把已训练的 `open_loop` 前向模型当作**可微规划仿真器**,在其上做:

```
方向1 视野认证(eval_horizon.py)        方向2 可微逆规划(inverse_plan.py)
  纯自回归 rollout K 步,                给定 s_init / s_target,
  找模型可信视野上限 K_max。              优化 K 步动作序列使 rollout 到达目标。
        │                                     │
        └──── K_max = 规划视野硬上界 = 滚动重观测频率 ────┐
                                                          ↓
                        变长 K(--auto_k):K=clamp(ceil(gap_tip_px/4.0),4,40)
                        避障(--obstacle):逐步 keep-out 圆惩罚
                                                          ↓
                        可视化(viz_control.py) → 汇报(build_control_report.py)
```

**关键机制定量结论(2026-07-15)**:
- 视野:`open_loop` 漂移 **1.7× @300步**(可信 ~25s / 124步 @10px);`gt` **272× 爆炸**(TF=1.0 训练泄漏,推理时喂自己预测会 OOD)。
- 逆规划:planner **3.07px = 0.38× do-nothing**,接近 GT-actions 保真上界 2.69px。
- 避障:无避障末端**穿透障碍**(到圆心 10.2px < r=12);加避障**绕开**(12.7px > r=12),reach 4.3→8.7px(诚实 tradeoff)。
- 帧间隔 `FRAME_DT=0.203s`(~5fps):K=40→8.1s,K_max@10px=124→25s。

---

## 4. 当前真实状态(2026-07-15,别信文档信这里)

### 4.1 最好的两个实验

| 实验 | 路径 | 模型 | best_loss | eval 产物 | 备注 |
|------|------|------|-----------|----------|------|
| GT 上界 | `train_log/gt_transition/exp_20260714_7/` | GTObserved | **0.00077** | 仅 `eval_metrics.csv` + `transition_metrics.json`(`rollout_mse=0.26`,**无 NaN**) | 无控制 viz;作精度天花板 |
| OpenLoop 部署 | `train_log/open_loop_transition/exp_20260714_8/` | OpenLoop | **0.080** | **全套**:`eval_horizon/`、`eval_plan/`、`eval_plan_obs/`、`eval_viz/` | ⚠️ 末期训练 NaN(见 §10.2) |

两者 checkpoint 都确认存在:`phase_<phase>/model/{best_model.pt, final_model.pt}`。

### 4.2 数据

- **实际默认训练数据 = `data/real_seq/seq_20260627_163921_n15_sam2_clean/train`**(15 节点,SAM2 mask,全清洗)。**注意**:`data/real_seq/README.md` 声称默认是 `_rep_clean`,**与 config.json 矛盾**——以 **config.json 的 `data_dirs.sequence` 为准**(见 §10.1)。
- npz 格式:`positions(T,3,N)` float32(`[:,0]=col, [:,1]=row, [:,2]=0`),`actions(T,A)` float32 **已归一化 [0,1]**,metadata `n_points=15`、`tip_fix=True`。
- 共 10 个变体(raw/rep/sam2 × 15/31 节点 × clean/未clean);**31 节点版已废弃**。

### 4.3 什么验证了 / 什么没验证

✅ **已验证(模型内)**:视野认证、逆规划 reach、避障绕开、SAM2 mask 干净(area std 1.7%)、tip_fix 最优(0.80px vs medial_axis 7.50px)、gt 末端 NDI **0.77mm**(到标定底 0.74mm)。
⚠️ **未验证(诚实边界)**:**planner 优化出的动作从未在真机执行**;`open_loop` 多步 rollout 末期 NaN(§10.2);`z` 实测"懒惰"≈0.00(§10.4)。

---

## 5. 5 分钟跑通(从已训模型到控制演示)

> 所有命令假设 `cd /Data5/ddf/projects/SelfSoftRobot`。GPU 用 `CUDA_VISIBLE_DEVICES` 指定(`train_transition.py` 默认用 1 号卡)。

```bash
CKPT_OL=train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt
CKPT_GT=train_log/gt_transition/exp_20260714_7/phase_gt_transition/model/best_model.pt
DATA=data/real_seq/seq_20260627_163921_n15_sam2_clean/val

# 1) 视野认证:open_loop vs gt 谁能当规划仿真器
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/eval_horizon.py \
    --checkpoints $CKPT_OL $CKPT_GT --data_dir $DATA --max_steps 300 --n_seeds 8 \
    --out train_log/open_loop_transition/exp_20260714_8/eval_horizon

# 2a) 变长 K 逆规划(推荐:auto_k 按首末差距选步数)
CUDA_VISIBLE_DEVICES=0 python scripts/control/inverse_plan.py \
    --checkpoint $CKPT_OL --data_dir $DATA --t_init 500 --t_target 900 \
    --auto_k --step_budget_px 4 --k_min 4 --k_max 40 --n_iter 400 \
    --out train_log/open_loop_transition/exp_20260714_8/eval_plan

# 2b) 避障逆规划(给定 keep-out 圆 cx,cy,r_px)
CUDA_VISIBLE_DEVICES=0 python scripts/control/inverse_plan.py \
    --checkpoint $CKPT_OL --data_dir $DATA --t_init 1772 --t_target 1812 \
    --auto_k --obstacle '322,268,12' --w_obs 1.0 \
    --out train_log/open_loop_transition/exp_20260714_8/eval_plan_obs

# 3) 可视化(视野网格/gif + 规划对比/gif + 真实照片叠加)
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/viz_control.py \
    --open_loop $CKPT_OL --gt $CKPT_GT --data_dir $DATA \
    --plan_json train_log/open_loop_transition/exp_20260714_8/eval_plan/plan_result.json \
    --t0 500 --max_steps 300 --t_init 500 --t_target 540 --K 40 --overlay \
    --out train_log/open_loop_transition/exp_20260714_8/eval_viz

# 4) 组装自包含 HTML 汇报(base64 内嵌所有图)
python scripts/utils/build_control_report.py \
    --horizon_json train_log/open_loop_transition/exp_20260714_8/eval_horizon/horizon_summary.json \
    --plan_json   train_log/open_loop_transition/exp_20260714_8/eval_plan/plan_result.json \
    --fig_dir     train_log/open_loop_transition/exp_20260714_8/eval_viz \
    --horizon_curve train_log/open_loop_transition/exp_20260714_8/eval_horizon/horizon_comparison.png \
    --out docs/reports/2026-07-15_control_shape_planning.html
```

**加载任意 checkpoint 做推理**(自动检测类型+phase,读同目录 config.json):
```bash
python -c "from src.utils.model_loader import load_model; \
  i=load_model('train_log/gt_transition/exp_20260714_7/phase_gt_transition/model/best_model.pt', device='cuda'); \
  print(i['model_type'], i['action_dim'], i['norm_factor'])"
```

---

## 6. 完整流程:照片 → 控制(实物路线 B)

```bash
SEQ=seq_20260627_163921

# A. 数据准备(免标定)
python scripts/real/segment_batch.py        --seq real_capture/data/raw/$SEQ        # 照片→mask(white_on_blue)
python scripts/real/repair_masks.py         --seq $SEQ                              # (可选)mask 级三步修复→masks_repaired/
python scripts/real/masks_to_transition_npz.py --seq real_capture/data/raw/$SEQ \
    [--masks-dir real_capture/data/derived/$SEQ/masks_repaired] --n-points 15       # mask→2D骨架 npz + tip_fix + action归一[0,1]
python scripts/real/clean_transition_npz.py --seq ${SEQ}_n15                        # 静态段共识清洗→_clean/

# B. 训练(gt 先,open_loop 从 gt 热启动)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt \
    --data_dir data/real_seq/${SEQ}_n15_sam2_clean/train
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode open_loop \
    --data_dir data/real_seq/${SEQ}_n15_sam2_clean/train       # 自动找最新 action_dim 匹配的 gt 热启动

# C. 评估
python scripts/evaluation/eval_real_quant.py \
    --checkpoint train_log/.../best_model.pt --data_dir data/real_seq/${SEQ}_n15_sam2_clean/train  # 末端 NDI mm + 形态 px + drift_by_k

# D. 控制(见 §5)
```

> **mask 三来源**:raw(有腐败)/ masks_repaired(启发式三步修复)/ **SAM2 视频**(分块双向,最干净)。当前最好实验用 SAM2 版。
> **npz 命名后缀**:`_n15`(节点数)/ `_rep`(repaired mask)/ `_sam2`(SAM2 mask)/ `_clean`(清洗后)。

---

## 7. 千万别破坏的不变量(违反会**静默**出错)

这是全文最重要的一节。下面每条都是"看起来改了没事、实则悄悄崩"的陷阱。

### 7.1 模型/加载层

| # | 不变量 | 后果 |
|---|--------|------|
| 1 | **GRU key 迁移是 load-bearing**:`_migrate_gru_keys` 把旧 `GRUCell` 键(`gru.weight_ih`)重命名成 `nn.GRU` 键(`gru.weight_ih_l0`)。仅在 `state_transition` 族迁移(`spatial_sequence` 仍用 GRUCell,**不能**迁移)。在 `model_loader` 和 `train_transition._warm_start_open_loop` 两处都做。 | 跳过 → `load_state_dict(strict=False)` 静默丢掉整个 GRU 层 → 随机输出(即历史上的"蓝点"viz bug)。 |
| 2 | **gt/open_loop/base 三类共享同一 state_dict**。靠 marker buffer(`gt_observed_mode`/`open_loop_mode`)+ config.json 的 `model` 类名字符串区分。 | 删 config.json → model_loader 回退到基类 → 推理语义错乱。 |
| 3 | **delta_scale_max 是 gt→open_loop 的安全开关**:gt 留 `inf`(delta_scale 漂到 ~4 也 OK);open_loop 硬设 `1.0`。**热启动会 reset delta_scale=0.1**。 | 把 gt 权重(delta_scale~4)直接喂 tf=0 的开环 → 40 步 rollout 发散 → BPTT 梯度 NaN。**别删这个 reset。** |
| 4 | **action_dim 不匹配是真失败模式**:仿真 ad=2,实物 ad=1。热启动按 action_dim 过滤候选 gt checkpoint。 | 用仿真 ad=2 热启动实物 ad=1 → state_mlp 尺寸不匹配崩溃。 |
| 5 | **action_norm_factor 推理时读 checkpoint 的 buffer**,不是文件。eval 代码(`transition_metrics.py`)直接读 `model.action_norm_factor`。 | 没带 buffer 的模型静默用 1.0 → 动作归一错。 |
| 6 | **inverse_plan 必须 `model.train()`**(不是 eval):规划要 backprop 进空间 `nn.GRU`,cuDNN RNN 反向只在 train 模式可用。模型无 Dropout/BN 故 train==eval 前向一致。 | 用 eval() → cuDNN 报错或梯度缺失。 |

### 7.2 数据/坐标系

| # | 不变量 | 后果 |
|---|--------|------|
| 7 | ~~**免标定**:state = 图像像素骨架 `[col,row,0]`,z 恒 0。无相机矩阵/内参/三角化。`capture_to_npz.py`/`inspect_capture.py`/`calibrate_cameras.py` 是**遗留标定路线**,别用于路线 B。~~ **⚠️ 2026-07-28 已反转,见下方** | ~~引入投影 → 破坏整条 calibration-free 论点。~~ |

> **⚠️ 不变量 #7 已于 2026-07-28 反转。** 决定后续 6 个通道全部气压驱动进入 3D(理由:1-DOF 平面运动下"到达目标 + 避障"命题本身不成立,**只有 3D 冗余自由度才有零空间**),而**单目免标定不可能恢复深度**,故状态来源改为**双/多相机标定 + 三角化**。
>
> 因此 `scripts/real/calibrate_cameras.py`、`scripts/real/capture_to_npz.py --view-dirs`、`src/utils/camera_system.py`、`src/data/real/triangulation.py`、`src/data/dataset_multiview*.py` 从"遗留"变为**主线基础设施**。
>
> 连带失效:`pc_scale[2] = 1e-6` 的退化保护(真 3D 数据下 z 有真实量程 —— **旧 checkpoint 的 buffer 是 1e-6,不能与 3D 数据混用**);"障碍是平面近似"的诚实边界可升级为真 3D;`docs/papers/related_work_draft.md` §2.7 的"免标定"差异化论点要重写。
>
> **过渡期说明**:现有 `exp_20260714_7/8` 与本文件其余部分描述的 2D 免标定路线**仍然有效**(它是 3D 之前的基线),只是不再是终点。完整决策记录见 [`docs/archived/2026-07-28_archive_manifest.md`](archived/2026-07-28_archive_manifest.md) §3。
| 8 | **action 归一到 [0,1] 不是 [-1,1]**(气动单向,ch0 只充气 0→150kPa;映射到 [-1,1] 会把"静止"和"全反向"塌成同一点)。上界优先用 `meta.json hi6[ch]`(操作极限)而非数据 max。 | 用 [-1,1] → OOD 负值预测。 |
| 9 | **节点误差只用 `[:2]`(col,row 平面)**;反归一 `px = norm*pc_scale + pc_center`。整体形态误差只能算 px;末端误差 px + mm(mm 经 NDI↔GT node0 px 最小二乘 2D 仿射)。 | 混用 px/mm → 量纲错。 |
| 10 | **tip_fix 默认开且必须开**:修末端 node0 落 mask 尖角(根因是逐行质心对倾斜 cap 做**水平**切片→落在角不是中点;修法是沿局部轴**垂直**切片)。 | 关掉 → 末端偏 ~6px,34% 帧受影响。别换通用骨架化(medial_axis 7.50px vs tip_fix 0.80px)。 |
| 11 | **关节节点 id 漂移**(实测 19-27,中位 20,仅 64% 帧在 node20)——arc-length 重采样到 N 点时关节落点会变。 | 用固定 node id 做静态段共识 → 失败;`clean_transition_npz` 用关节**绝对位置**锚定。 |
| 12 | **n_points 默认 15 不是 31**(老 docstring 仍写 31)。`--n_nodes` 默认 None=从 npz 自检。 | 假设 31 → 维度错。 |

### 7.3 训练

| # | 不变量 | 后果 |
|---|--------|------|
| 13 | **open_loop 默认是纯闭环(TF=0)**,不是渐进退火。要退火须显式 `--tf_ratio 1.0 --tf_anneal_epochs N --tf_schedule staircase`。推荐 staircase(前半 hold nominal,后半跳到 tf_min),**避开 0<tf<1 的混合区**(那里速度输入 v=prev-prev_prev 混了 GT 和预测帧)。 | 线性退火进入混合区 → 不稳定。 |
| 14 | **open_loop 的 z 是 per-window 记忆**,每个窗口重置(`z_0=z_init(cond)`),**故意不跨窗口**(否则退化成有界的方向13)。 | "修"成跨窗口 → 退化。 |
| 15 | **supervision_mode 分派两处一致**:`_compute_losses` 仅在 `supervision_mode=='rendering'` 且 `views is not None` 时调 ViewStrategy → `direct_3d/skeleton/spatial_sequence` 模型**必须** `view_strategy=None`;`use_episode_mode` 走 `_compute_sequence_losses` 完全绕过 ViewStrategy。 | 传错 view_strategy → 渲染路径对非渲染模型报错。 |

---

## 8. 怎么扩展(常见任务)

| 我想… | 做什么 |
|-------|--------|
| **加一个新模型** | 在 `src/models/` 写类,**继承合适的基类**,设 `training_spec` 类属性(`PhaseSpec`/`TrainingSpec`)。`UnifiedTrainer` 自动解释——**无需写 Trainer 子类**。三维度:Phase 策略(`PhaseSpec`)× 监督模式(`rendering`/`direct_3d`/`skeleton`/`spatial_sequence`/`pointcloud`)× 视角策略。 |
| **改超参** | `config/training.json`(optimization/temporal/loss_weights/canonical/window_size=40/n_scales=4/hidden=128/grad_clip=1.0),或 CLI(`src/config/args.py`)覆盖。 |
| **换时序编码器** | `--encoder {ema,fractional,gamma,gru,transformer,tcn}`。编码器接口兼容,在 `src/encoders/`。 |
| **降/升节点数** | `masks_to_transition_npz --n-points N`,全流水线按 N 自适应(n_static=max(4,int(0.4N)),act_nodes=max(5,int(0.6N)))。 |
| **新数据序列** | 跑 §6 的 A 段;产物在 `data/real_seq/<seq>[_n15][_rep|_sam2][_clean]/`。 |
| **改 planner loss 权重** | `inverse_plan.py`:`w_reach=1.0`/`w_mono=1.0` **硬编码在签名(非 CLI)**;`--w_smooth`(0.01)/`--w_path`(0.5)/`--w_obs`(1.0) 是 CLI。变长 K 的 `step_budget_px=4.0` 绑定 `delta_scale_max(1.0)×pc_scale`,改模型归一就重新调它。 |

> **新模块必须向后兼容**(用户硬性要求):别破坏已有函数签名/调用。改前确认在正确分支 `feat/real-data-transition`。

---

## 9. 在哪找 X(导航索引)

| 需求 | 去哪 |
|------|------|
| **项目总览/怎么跑(人向)** | `docs/overview/project_help.md`(CLI + 源码布局 + 模型表 + 约定) |
| **现在到哪了/接下来** | `docs/overview/status.md` |
| **技术管线/模型演进** | `docs/overview/pipeline.md` |
| **实物免标定完整流程** | `docs/real_data/workflow.md` |
| **硬件采集系统** | `docs/real_data/capture_setup.md`(程序在 `docs/ref/Main UI-plc/`) |
| **16 个研究方向** | `docs/directions/directions_overview.md`(当前焦点:14 gt / 15 open_loop / 16 约束导向控制) |
| **控制方向详述** | `docs/directions/16_constraint_oriented_control.md` |
| **文献综述/差异化** | `docs/background/literature.md` |
| **某文件做什么** | `docs/overview/project_help.md` §3 源码布局 |
| **某模型训练阶段/loss** | `project_help.md` §5 training_spec 速查 |
| **加载 checkpoint** | `src/utils/model_loader.py`(自动检测) |
| **docs 全索引** | `docs/README.md` |

**源码布局速记**(`src/`):
```
encoders/    时序编码器(EMA/Fractional/Gamma/GRU/Transformer/TCN)
fields/      神经场(Canonical/Deformation/SkeletonDensity)—— 仿真路线
heads/       骨架回归头(point/fourier/bspline/catmullrom)
rendering/   渲染策略(SingleView/MultiView)—— 仿真路线
models/      模型定义(神经场族 + 状态转移族,见 §3.1)
training/    spec.py(声明)/trainer_unified.py(UnifiedTrainer)/dataset_factory.py/phase_strategy.py
data/        数据集类(SoftSequence/SDF/SkeletonSDF/SpatialSequence/MultiView...)
evaluation/  query.py / render.py(仿真);transition_metrics.py / shape_metrics.py(实物)
utils/       model_loader/skeleton_2d(+tip_fix)/sdf_utils/camera/rendering/experiment/config_utils
config/args.py  CLI 参数
```

---

## 10. 已知坑 & 诚实边界

### 10.1 数据默认不一致(坑)
`data/real_seq/README.md` 说默认是 `_rep_clean`,但**两个最好实验实际训在 `_sam2_clean`**。**以 config.json `data_dirs.sequence` 为真相源**。被问"默认数据"时主动澄清。

### 10.2 open_loop 末期训练 NaN(坑)
`exp_20260714_8` 的 `loss_log.csv` 在 epoch 48-50 变 NaN,`eval_metrics.csv` 的 rollout/drift/mm 全 NaN(只有 `copy_mse=1.358` 有限)。**best_model.pt 是更早保存的(epoch~2 区,best_loss=0.080),checkpoint 仍可用**;但这个 run 的**多步 rollout mm 数不可信**——用 `eval_plan`/`eval_horizon` 里的 JSON+PNG,别引 mm CSV。gt 的 `exp_20260714_7` 无 NaN(rollout_mse=0.26)。

### 10.3 汇报模板里有硬编码数字(坑)
`build_control_report.py` 的 HTML 模板把若干叙述数字(如 `__OVERLAY_ERR__`→字面 `'4.2'`,1.7×/272×/3.07px/2.69px)写死在模板文本里,**不从 JSON 读**。只有 K/gap_tip/step_budget 和 init/do/gt/planner px 块是真正替换的。**用不同结果重跑,模板文本会撒谎**——改结果时同步改模板。

### 10.4 z 实测"懒惰"(边界)
z 在当前训练下范数 ≈0.00 全程,稳定部分来自 z 坍缩——对规划是良性的,但**削弱了"z 建模迟滞"的卖点**。且 per-frame/Stage-0 训练里 z 每步从 `z_init(cond)` 重置→退化成 cond 的函数,**不是真记忆**;真迟滞 z 需 episode 序列训练让 z 跨帧演化。

### 10.5 reachability(核心开放问题)
planner 优化出的动作**可能是真机不可达的**。三层分解:
1. **动作可执行性**(注射器电机能否产生)——✅ 已解决:动作 clamp 到训练时真机真实执行范围 `[a_lo,a_hi]`,平滑约束保证可跟踪。
2. **目标形状可达**——✅ 目标取自真实录制数据(真机确实做到过),故一定可达。
3. **模型-现实鸿沟**(优化出的动作真机能否复现预测形状)——⚠️ **未完全解决,核心开放**。GT-actions 基线只证明模型对**真实动作**保真;planner 是**优化**出的动作,可能在模型盲区钻空子。

**解决路径(诚实排序)**:① 留分布内(惩罚规划进低训练密度区);② 诚实裁决(规划完再 rollout 一次,残差超容差就报"不可达",不硬称成功);③ **滚动视野 MPC(部署级解法)**:执行前 N<K_max 步→RealSense 重观测→重规划(**K_max≈25s 正是重观测频率**);④ 真机闭环(动作发 PLC→真机→RealSense+NDI 测真值→比 target)。

> **最尖锐的一点**:学习模型**永远不能单方面保证真机可达——这是范畴错误**。它的价值是"告诉你能信多久、多久该重新看一眼"。

### 10.6 其它已知项
- `output/` 里 `real_quant/exp_20260709_5`、`real_overlay/exp_20260709_5` 指向**旧的** gt 实验,不是当前最好——权威 per-experiment eval 在 `train_log/<model>/exp_<n>/eval_*`。
- 仿真路线(A)的 trainer/model 仍可用,但 `view_strategy` 仅渲染模式需要。
- 无正式测试套件;验证靠 `notebooks/` + eval 脚本 + `loss_log.csv`。

---

## 11. 术语表

| 术语 | 含义 |
|------|------|
| **路线 A / B** | 仿真(PyElastica+标定+体渲染) / 实物免标定(2D 像素骨架) |
| **gt / open_loop** | teacher_forcing=1.0(每步观测,精度上界) / =0(开环 rollout,部署目标)。同网络不同 TF。 |
| **state / skeleton** | 形态表示。实物=图像像素 `[col,row,0]`;仿真=3D 节点 `(T,3,N)` |
| **z (latent)** | 可学习迟滞潜变量(GRUCell 跨帧演化,无 GT,从 skeleton loss 学) |
| **delta_scale / delta_scale_max** | 增量预测缩放系数(可学习,init 0.1)/ 其上界(gt=inf,open_loop=1.0) |
| **teacher_forcing (TF)** | 训练时喂真实 s(tf=1)还是模型预测(tf=0)作 s_{t-1} |
| **tip_fix** | 末端 node0 垂直尖端切片修正(修弯管 cap 角落偏移) |
| **K / K_max** | 规划步数 / 模型可信视野上限(=滚动重观测频率) |
| **step_budget_px** | 变长 K 的单步位移预算(默认 4px ≈ delta_scale_max×pc_scale) |
| **drift_by_k** | rollout 误差随步数的累积比(rollout/onestep) |
| **NDI** | NDI Aurora 6DOF 磁追踪,末端真值(mm),仅验证不训练 |
| **pc_center / pc_scale** | 骨架归一化的平移/缩放(3 向量 buffer) |
| **SAM2** | SAM2 视频分割(分块双向),最干净的 mask 来源 |
| **FBV-SM** | Hu et al. 2025 基线代码库(本项目前身) |
| **UnifiedTrainer / training_spec** | 声明式训练:模型用类属性声明需求,trainer 统一解释,无需 Trainer 子类 |

---

## 12. 仓库工作约定(接手必读)

- **提交信息**:Conventional Commits(`feat:`/`fix:`/`refactor:`/`docs:`/...),中英混排可。**禁止加 `Co-Authored-By`**(用户全局 `~/.claude/settings.json` 已禁用署名)。
- **提交前必须问用户**(用户硬性偏好:每次提交前询问,小修改不自动提交)。
- **改文件前确认在正确分支**(`feat/real-data-transition`)。
- **新模块向后兼容**(不破坏已有调用)。
- **实验日志**:`train_log/<model_tag>/exp_<date>_<n>/`,含 `config.json` + `model_card.txt`(exp 根)+ `phase_*/{loss_log.csv, model/best_model.pt+final_model.pt}` + 按需 `eval_*` 子目录。GPU 用 `CUDA_VISIBLE_DEVICES`。
- **GateGuard hook**:每个 context window 首次 Bash/Write/Edit 某"文件/命令"时会要求先陈述事实再重试——正常现象,陈述 4 条事实(用户请求/该命令产出什么/影响哪些公开接口/数据结构)后重试即可。若阻塞修复工作,可用 `ECC_GATEGUARD=off` 或把对应 hook 加进 `ECC_DISABLED_HOOKS`。
- **大量产物被 gitignore**:`data/`、`train_log/`、`output/`、`*.pt/*.npz/*.gif/*.log`、`sam2/` 源码+权重、`docs/ref/`、`docs/papers/*.pdf`。**仓库只含代码 + 文档 + 小 config JSON**。真机原始采集在 `real_capture/data/`(也不提交),重建 npz 需要它。
- **代码语言**:注释/文档/变量名中英混排;`docs/` 主要中文。

---

### 一句话总结给接手者
> 你接手的是一个"**软体臂免标定 2D 自建模 → 状态转移(带迟滞)→ 可微逆规划+避障**"的研究项目。当前活的线是**实物 open_loop 模型上的控制/规划**,最好的模型在 `train_log/open_loop_transition/exp_20260714_8`。先跑通 §5 的 5 分钟流程,记住 §7 的不变量(尤其 GRU 迁移、gt/open_loop 共享权重、delta_scale_max),诚实对待 §10 的边界——planner 还没上真机,这是下一个里程碑。
