# 实机部署指南 · 从采集到工作台闭环

> 面向"把软臂从数据采集一路部署到实机验证工作台"的完整操作手册。
> 当前状态:感知层(P1a)与契约/单位层(P1b)已完成并通过 95 个测试;数据前处理管线全部打通。
> 最后更新:2026-08-09 · 分支 `feat/real-data-transition`
> 关联:[`workflow.md`](workflow.md)(数据管线详细版)·[`capture_setup.md`](capture_setup.md)(硬件)·[任务层 IK 设计 spec](../superpowers/specs/2026-07-28-real-validation-task-layer-ik-design.md)

---

## 0. 一页总览

```
相机帧 ──► mask ──► 15点骨架 ──► npz(positions + actions) ──► 训练(gt → open_loop)
                                                                      │
       部署契约 deploy_manifest.json ◄── build_deploy_manifest.py ── checkpoint
                                                                      │
       实机工作台 real_validation/ ◄── 拷贝到 PC(checkpoint+config+manifest)
              │
       实时感知 → 质量门控 → 规划(OpenLoop shooting)→ preflight → 真阀执行 → 记录/评价
```

**两条路线**:
- **(A) 仿真**(PyElastica + 体渲染)—— 已归档,非当前焦点
- **(B) 实物免标定(当前主线)** —— 1-DOF 双段硅胶臂 + 单 RealSense,state = 图像像素骨架 `[col,row,0]`,NDI 仅作末端 mm 验证

**核心约定(所有模型)**:模型输入**只有驱动参数 + 查询点**;图像/深度仅作监督信号。gt(每步观测,精度上界)与 open_loop(开环 rollout,部署目标)共享同一份 state_dict,靠 `config.json` 的 `model` 类名区分。

---

## 1. 里程碑现状(P1a + P1b 已完成)

| 阶段 | 交付 | 提交数 | 测试 |
|---|---|---|---|
| **P1a(感知迁移)** | 骨架/分割迁为唯一实现 `real_validation/perception/`,`src/` 改薄壳;tip_fix 可观测;相机位姿注册(只检测不 warp);在线质量门控(8 判据);命令行感知探针 | 8 | 69 |
| **P1b(契约与单位)** | `units.py`(单位唯一换算)、`obstacles.py`(障碍唯一实现)、`deploy_manifest.json`(部署契约)、planner 单位收口 + 全身目标 + AABB + auto_k + GL 缓存、preflight 新门、session/GUI 守卫 | 10 | 95 |

**P1a/P1b 之后**:数据前处理脚本(`scripts/real/*`)**行为零变化**(薄壳 re-export,parity 测试锁死逐点 px 差==0),但底层实现已切换为可移植的唯一实现。

---

## 2. 数据前处理(每步详解)

### 2.1 原始采集(硬件侧,`real_capture/`)

```
real_capture/data/raw/seq_YYYYMMDD_HHMMSS/
  ├─ cam0/NNNNN.png            # RGB 640×480,与动作同步
  ├─ actions6.csv              # t_sec, c0..c5(ch0 是 0-150 kPa 气压)
  ├─ frame_times.txt / ndi.csv / meta.json
```

meta.json 关键字段:`hi6=[150,0,...]`(ch0 上界)、`active_channel=0`、`action_interval_s=0.2`、`settle_s=0.19`。

> ⚠️ **实测 Δt 是 0.2031s 不是 0.2s**(frame_times 逐帧 0.187–0.219 量化,均值 0.2031)。preflight 有 `dt_mismatch` 门(偏差 ≥5% 阻断)。

### 2.2 mask 产出(三条轨道)

**① white_on_blue 分割**(逐帧,唯一实现 `real_validation/perception/segmentation.py`):

```bash
python scripts/real/segment_batch.py --seq real_capture/data/raw/seq_20260627_163921 --val 100
```

> ⚠️ **必须显式 `--val 100`**:CLI 默认 120,但实际产物用 100(见 `derived/<seq>/segment_meta.json`)。

实际生效参数:`sat=100, val=100, diff=25, dil=35, open_k=5, close_k=15, min_area_frac=0.003, min_h_frac=0.15, n_bg=500`。
管线:`HSV白 ∩ dilate(背景差) → OPEN去细管 → CLOSE填体 → fill_holes → 最大连通区`。输出 `derived/<seq>/masks/` + `bg_median.png` + `segment_meta.json`。

**② 启发式修复**(跨帧,离线):

```bash
python scripts/real/repair_masks.py --seq seq_20260627_163921
```

三步:手干扰帧插值(默认开)、静态段宽度共识(无条件)、动作段插值(默认开)。输出 `derived/<seq>/masks_repaired/`。

**③ SAM2 视频分割**(★ **当前训练数据的来源**,跨帧双向,离线):

```bash
CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq seq_20260627_163921
```

分块 200 帧,锚帧取"顶部行≤20 且面积∈[0.7,1.3]×中位"、前/反向各传播 100 帧。输出 `sam2/masks/<seq>_full/`(area std 1.7%,无漂移)。

> **在线 vs 离线**:①逐帧可在线复现;②③跨帧离线不可逐帧。这就是 3D 决策里"训练 SAM2、在线跑 SAM2 前向流式"要量化两者差异的原因。

### 2.3 骨架 + npz(`masks_to_transition_npz.py`)

```bash
python scripts/real/masks_to_transition_npz.py \
    --seq real_capture/data/raw/seq_20260627_163921 \
    --masks-dir sam2/masks/seq_20260627_163921_full \
    --out-root data/real_seq/seq_20260627_163921_n15_sam2_clean --n-points 15
```

骨架 = 逐行质心 + 弧长均匀重采样到 15 点 + **tip_fix**(末端垂直切片修正,实物 34% 帧受益、末端误差 -71%)。唯一实现 `real_validation/perception/skeleton.py`。

产出 npz(`np.savez_compressed`):

| 字段 | 值 |
|---|---|
| `positions` | `(T,3,15)` float32,`[:,0]=col, [:,1]=row, 第3维=0`(免标定 2D state) |
| `actions` | `(T,1)` **已归一化 [0,1]** = kPa / `hi6[0]=150` |
| `n_points` / `tip_fix` | 15 / True |
| 切分 | 首 80% train / 末 20% val,时间连续不 shuffle(防乱序泄漏) |

### 2.4 清洗(`clean_transition_npz.py`)

```bash
python scripts/real/clean_transition_npz.py --seq seq_20260627_163921_n15_sam2
```

静态段跨帧中位共识(关节绝对位置锚定,node 漂移鲁棒)+ 动作段离群插值(`--act-dev-thresh 60`)。**完全离线**(需全序列/未来帧)。对 SAM2 数据近 no-op(离群 3+1 帧)。

### 2.5 归一化(训练时,不进 npz)

- 动作已在 npz 归到 [0,1](/150);训练时 `norm_factor = max|actions| ≈ 0.99999`,再除近似 no-op
- **`pc_center` / `pc_scale`** 训练时由 `dataset_spatial.py` 每序列抽 ~5 帧算 min/max,`set_normalization` 写进 **checkpoint buffer**(不是 npz)

---

## 3. 训练

```bash
# gt 先(精度上界)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt \
    --data_dir data/real_seq/<seq>_n15_sam2_clean/train
# open_loop 从 gt 热启动(部署目标)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode open_loop \
    --data_dir data/real_seq/<seq>_n15_sam2_clean/train
```

关键超参:`window_size=40, episode_len=40, z_dim=16, action_dim=1`。产物 `train_log/<tag>/exp_<date>_<n>/`:`config.json` + `model_card.txt` + `phase_*/model/{best_model, final_model}.pt` + `eval_*`。

当前最好的旧实验(供参考,非部署主线):`gt_transition/exp_20260714_7`(best_loss 0.00077)、`open_loop_transition/exp_20260714_8`(best_loss 0.080)。

> ⚠️ `exp_20260714_8` 末期训练 NaN(best_model 是发散前快照,可用;但该 run 的多步 rollout mm 数不可信)。P2 重采重训后应彻底避开。

---

## 4. 部署契约(`deploy_manifest.json`)

把部署所需的隐式知识显式化,与 checkpoint 同目录。**缺失或 `action_scale_kpa` 缺失 → fail-closed 阻断规划**(单位 bug 是活的,kPa 0-150 直接除 ≈1.0 的 norm_factor 喂进 [0,1] 训练域)。

```bash
python scripts/utils/build_deploy_manifest.py \
    --exp-dir train_log/open_loop_transition/<exp> \
    --raw-seq real_capture/data/raw/<seq> \
    --horizon-summary <exp>/eval_horizon/horizon_summary.json \
    --out <exp>/deploy_manifest.json
```

关键字段(3 源 join:checkpoint+config / raw meta+frame_times / horizon_summary):

| 字段 | 来源 | 说明 |
|---|---|---|
| `action_scale_kpa` | `meta.json hi6` 经 `action_max_per_channel()` | kPa 上界(150);**复用该函数,不自己读 hi6** |
| `channel_source6` | raw `meta.json` + 训练 `action_view` | 权威六维来源图；`source[i]` 是硬件 `chi` 的根通道 |
| `channel_equalities` | 由 `channel_source6` 派生 | 旧工具兼容字段，不再单独配置 |
| `channel_map` | `channel_source6` 的根通道 | 模型动作列的硬件语义，长度即 `action_dim` |
| `action_expansion6` | `channel_source6 + channel_map` 派生 | 例如 `[0,1,1,2,3,3]`；支持任意 1–6D 来源合同 |
| `train_dt_measured_s` | `frame_times.txt` 现算 `np.diff` | 实测 Δt(**禁止硬写 0.203125**) |
| `mask_source` | `config.json data_dirs.sequence` 路径后缀 | 判断**完整路径**而非 basename(`.../train` 会丢后缀) |
| `segment_params` | `segment_meta.json`(仅 white_on_blue) | SAM2/修复 mask 时 null |
| `k_safe_table_px` | `horizon_summary.json` `Kmax_px_{5,10,20}` | **键是 px 不是 mm** |

> ⚠️ 旧实验 `exp_20260714_8` 训在 SAM2 上(`segment_params=None`),作部署主线 manifest 不完整 —— **P2 重采重训后重新生成**。

---

## 5. 实机工作台部署(PC 侧)

把 `real_validation/` 与 `real_capture/` **并排**拷到 PC(两个"整目录拷走"契约都不破;`real_validation` 不 import `src/`)。

```bash
pip install -r real_validation/requirements.txt \
             -r real_validation/requirements-perception.txt
# 有硬件再加: -r real_validation/requirements-hardware.txt
```

`checkpoints/current/` 需放三件套:**`best_model.pt` + `config.json` + `deploy_manifest.json`**。

自检(PC 上无 `tests/`,只做运行时自检):

```bash
python -c "import real_validation; print('contracts ok')"
python real_validation/perception_probe.py --source dir --frames-dir <帧目录> \
    --background <bg.png> --segment-params <segment_meta.json> \
    --reference <无臂静态背景.png> --n-points 15 --frames 12 --out <out>
```

完整测试在仓库根：`python -m unittest discover -s tests`。

---

## 6. 实机执行前的检查(工作台 preflight 门)

工作台 `preflight.validate_plan` 在 Arm 前拦截,任一项失败不放行:

| 门 | 含义 |
|---|---|
| `stale_model/scene/anchor/safety` | 四重 hash 绑定,任一变更计划失效 |
| `action_scale_missing` | deploy_manifest 缺失/残缺 → 阻断 |
| `k_safe_uncertified` | 无 K_safe 且无认证表 → 阻断任意 horizon |
| `dt_mismatch` | 动作周期与训练 Δt 偏差 ≥5% |
| `unsupported_obstacle` | scene 含 planner 未支持的障碍类型 |
| `predicted_collision` | 预测轨迹侵入障碍(最小净距 < 0) |
| `slew_rate` / `pressure_bound` | 压力越界 / 速率超限 |
| `channel_source_contract` | plan 与模型的权威 `channel_source6` 不一致 |
| `channel_equality_contract` | 兼容派生等值关系不一致 |
| `history_dim` / `safety_equality` | D 维模型历史宽度错误，或同源硬件通道范围/速率/初值不一致 |

**执行态守卫**:EXECUTING 中锁页 1/2/3(防执行中改 scene 致执行记录与计划脱钩);`invalidate_model` 在模型加载失败时清旧 runtime。

---

## 7. 后续路线

| 阶段 | 干什么 | 谁做 |
|---|---|---|
| **P2 = M1+M2** | 按采集协议重采 ≥3-5 条序列、重训 gt + open_loop、跑视野认证、生成新 manifest | **需要你**:配置硬件;我提供脚本 |
| **P3 = M4** | GUI 感知/场景/锚定:`camera_view`(图像+点击)、`scene_editor`、实时锚定、warmup | 我(代码) |
| **P4 = M5+M6** | 真机安全执行 + 全身目标/避障 + 滚动重锚定 | 我(代码)+ 你在真机跑 |

**P2 采集协议要点**:
1. 固定相机位姿,**单独拍一张无臂静态背景**做配准参考帧(P1a 教训:中值背景混入臂运动假位移 ~3.4px)
2. 先用单通道验证链路，再进入六通道等值约束的双段平面阶段；六通道独立三维运动仍留到
   多视角/深度 GT 打通之后。平面阶段见 `planar_constrained_6ch_workflow.md`。
3. 训练用 SAM2 mask;在线用 SAM2 前向流式(量化两者差异是开放项)
4. 按序列划分 train/val/test,覆盖 loading/unloading/hold/反转/变速
5. 目标形态取自录制帧(保证可达)

---

## 8. 已知坑与诚实边界

| # | 项 | 说明 |
|---|---|---|
| 1 | **配准参考帧须取无臂静态背景** | 用 `bg_median.png`(含运动臂)当参考会混入臂运动假位移 ~3.4px → 误判"相机动了" |
| 2 | **raw(white_on_blue)mask 质量参差** | 探针扫描大量帧段被门控拒;这是选 SAM2 的动机。在线阈值分割时门控必须兜底 |
| 3 | **在线 SAM2 前向 vs 训练双向差异未量化** | 训练锚帧来自 masks_repaired,在线无此来源;GPU 单帧延迟未实测(0.2s 是预算) |
| 4 | **动作单位链** | kPa → /action_scale_kpa → [0,1] → /norm_factor → 模型。换算只允许出现在 valve/planner 两处 |
| 5 | **冷启动需 40 步真实动作(≈8.1s)** | 模型从没见过零填充窗口,且分数阶 GL 核把最大权重压在窗口最旧格 |
| 6 | **z 无 GT,接管时重置误差不可消除** | 锚定时提示"迟滞潜变量已重置,首窗口精度略降" |
| 7 | **障碍是平面近似** | NDI 实测平面外跨度 4.35mm vs 平面内 24.2mm;3D 路线落地前 UI 必须标 `planar approx` |
| 8 | **planner 动作从未真机执行** | prediction-to-execution gap 是核心开放问题;执行后必须报 gap,超容差报"不可达"不硬称成功 |
| 9 | **README 默认数据标注过期** | `data/real_seq/README.md` 写 `_n15_rep_clean`,实际训练用 `_n15_sam2_clean`。以 config.json `data_dirs.sequence` 为真相 |
| 10 | **Δt 是 0.2031 不是 0.2** | 工作台 plan_dt 默认取 manifest 的实测值;preflight 有 dt_mismatch 门 |

---

## 9. 术语表

| 术语 | 含义 |
|---|---|
| **gt / open_loop** | teacher_forcing=1.0(精度上界)/ =0(开环 rollout,部署目标)。同网络不同 TF |
| **state / skeleton** | 形态表示。实物 = 图像像素 `[col,row,0]` |
| **tip_fix** | 末端 node0 垂直尖端切片修正(修弯管 cap 角落偏移) |
| **deploy_manifest** | 部署契约文件(action_scale_kpa / train_dt / mask_source / k_safe_table_px 等) |
| **K / K_safe** | 规划步数 / 模型可信视野上限(=滚动重观测频率) |
| **pc_center / pc_scale** | 骨架归一化的平移/缩放(3 向量 buffer,随 checkpoint) |
| **SAM2** | 视频分割(分块双向),最干净的 mask 来源,训练用 |
| **preflight** | 计划执行前的安全检查(见 §6) |

---

## 10. 诊断速查

| 症状 | 查哪 |
|---|---|
| 规划出垃圾动作但 preflight 全绿 | ① `deploy_manifest.json` 的 `action_scale_kpa` 是否在(checkpoint 旁);② manifest 是否与 checkpoint 匹配(`checkpoint_sha256`) |
| 相机一动结果全错 | 跑 `perception_probe.py --reference <无臂静态背景>` 看 `displacement_px`;>2px 阻断 |
| 在线骨架跳变/大量 reject | `quality.jsonl` 的 `reasons`:mask 缺行(area_ratio_low)、手入画(top_row_high)、帧间位移大(node_step_high) |
| 测试失败 | `python -m unittest discover -s tests -v`(102 个);parity 失败 = 感知实现被改 |
| 规划慢 | metadata `duration_s`;GL 缓存是否命中(planner 用 `getattr` 守卫) |

---

## 11. 接实机后的操作步骤(端到端)

> 代码侧已就绪(P1a/P1b + P3/P4 可离线部分)。以下按顺序操作,从接设备到跑通避障实验。

### 11.1 硬件连接与自检

1. 接上相机(RealSense)、两组阀(USB-RS485 dongle)、NDI(Aurora 串口)。`real_capture` 与 `real_validation/` **并排**在工作机(或服务器)上。
2. **固定相机位姿,单独拍一张无臂静态背景图** —— 这是配准参考帧(`deployment.md §8 #1` 的教训:中值背景混入臂运动假位移)。
3. 跑一次感知探针确认链路:

```bash
python real_validation/perception_probe.py --source live \
    --background <无臂静态背景.png> --reference <无臂静态背景.png> \
    --n-points 15 --frames 12 --out output/probe_selfcheck
```
期望:`total p90 < 200ms`、配准 `displacement < 2px`、判决以 ok 为主。

### 11.2 采集三腔道数据(两段 × 3 腔 = 6 通道)

用激励协议生成器生成 actions6.csv(覆盖单腔/成对/三腔协同),走 `real_capture` 的 Replay 模式采集:

```bash
# 生成激励(3 腔道一段 × 2 段 = 6 通道;每腔上限 hi6)
python scripts/real/gen_3chamber_excitation.py \
    --channels 0,1,2,3,4,5 --hi6 150,150,150,150,150,150 \
    --ramp 15 --hold 10 --out <seq>/excitation_6ch.csv
```
> 若先做单段(3 通道),用 `--channels 0,1,2`。`gen_3chamber_excitation.py` 对任意通道集通用。

在 `real_capture/main_capture.py`:选 Replay 模式 → 选 `excitation_6ch.csv` → 开始采集(相机/NDI 同步录)。产物 `real_capture/data/raw/seq_YYYYMMDD_HHMMSS/`。

### 11.3 数据前处理

```bash
SEQ=real_capture/data/raw/seq_YYYYMMDD_HHMMSS
# 1) 阈值分割(须显式 --val 100)
python scripts/real/segment_batch.py --seq $SEQ --val 100
# 2) SAM2(训练数据来源;可选但推荐)
CUDA_VISIBLE_DEVICES=3 python sam2/segment_video_full.py --seq $(basename $SEQ)
# 3) 骨架 + npz(--action-channels 指定驱动通道)
python scripts/real/masks_to_transition_npz.py --seq $SEQ \
    --masks-dir sam2/masks/$(basename $SEQ)_full \
    --out-root data/real_seq/$(basename $SEQ)_n15_sam2_clean \
    --n-points 15 --action-channels 0,1,2,3,4,5
# 4) 清洗
python scripts/real/clean_transition_npz.py --seq $(basename $SEQ)_n15_sam2
```

> **验证数据正确**:npz `actions` shape = (T, action_dim) ∈ [0,1];`positions` = (T,3,15);train/val 连续切分(首 80%/末 20%)。**3 腔道若使臂离平面,单相机 2D 骨架假设失效** —— 需先确认运动平面性,必要时升级多视角三角化。

### 11.4 训练(gt → open_loop)

```bash
DATA=data/real_seq/$(basename $SEQ)_n15_sam2_clean/train
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt --data_dir $DATA
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode open_loop --data_dir $DATA
```
> `train_transition.py` 自动按 `action_dim` 过滤 gt checkpoint 热启动;action_dim=3/6 无需改代码。确认 `config.json` 的 `action_dim` 与 `data_dirs.sequence`。

### 11.5 视野认证 + 部署契约

```bash
EXP=train_log/open_loop_transition/<最新 exp>
CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/eval_horizon.py \
    --checkpoint $EXP/phase_open_loop_transition/model/best_model.pt \
    --data_dir data/real_seq/<seq>_n15_sam2_clean/val \
    --max_steps 300 --n_seeds 8 --out $EXP/eval_horizon
python scripts/utils/build_deploy_manifest.py \
    --exp-dir $EXP --raw-seq $SEQ \
    --horizon-summary $EXP/eval_horizon/horizon_summary.json
```
> manifest 落 `$EXP/deploy_manifest.json`;确认 `action_scale_kpa` 长度 = action_dim、`k_safe_table_px` 非空。

### 11.6 工作台(实时锚定 → 规划 → 执行)

1. 拷贝 `real_validation/` + `real_capture/` 到工作机;`checkpoints/current/` 放 `best_model.pt + config.json + deploy_manifest.json`。
2. 启动 `python real_validation/main_validation.py`:
   - Setup:加载模型(自动读 manifest → K_safe/plan_dt 自动回填);安全表按通道设 min/max/rise/fall。
   - Observe:Start Camera(真机改 `_make_transport`/相机源)→ Warmup(填动作历史)→ 从相机取流锚定(真机用无臂静态背景 + manifest 的 area_median)。
   - Scene:点上点加目标/障碍(工具按钮)。
   - Plan:Run Planner(变长 K 或固定)→ Preflight(全绿才 Arm)。
   - Execute:Arm → Execute(**真机时 `_make_transport` 返回 QtValveTransport**);执行中锁页;Results 显示命令安全 + jitter。
3. **避障实验 CLI 入口**(任意目标点 + 圆障碍,不依赖 GUI):

```bash
python scripts/control/run_avoidance.py \
    --checkpoint $EXP/phase_open_loop_transition/model/best_model.pt \
    --data-dir data/real_seq/<seq>_n15_sam2_clean/val --t-init <起始帧> \
    --target-x <px> --target-y <px> --target-radius 5 \
    --obstacle '<cx>,<cy>,<r>' --auto-k \
    --safety-max 150,150,150,150,150,150 --out $EXP/eval_avoid
```
> 输出 `plan.json`(含 kPa 动作 + predicted_states)+ 规划耗时 + 最小净距。把 `plan.json` 的 `actions6` 喂实机执行,再用 NDI/相机对比 `predicted_states.npz` 得 **prediction-to-execution gap**。

### 11.7 待真机确认/需改动点(诚实边界)

| 项 | 状态 |
|---|---|
| GUI 相机源(现在 Mock 合成帧)→ RealSenseCam | 接设备后改 |
| `_make_transport`(现在 Mock)→ QtValveTransport(需活 ValveController) | 接设备后改 |
| 3 腔道运动平面性(2D 骨架假设) | 需实机验证;离平面则升级多视角三角化 |
| 每腔道上限 hi6/rise/fall 实际值 | 采集时确认,写进 manifest |
| 骨架离群阈值 80px / 关节定位对 3D 弯曲 | 需重训验证 |
| `run_avoidance` 的 target 是图像像素(免标定) | 目标需在训练像素系内 |
