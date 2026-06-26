# 方向：窗口开环状态转移——"观测一次，开环预测 K 步"

> 状态：已实现 + 冒烟通过（warm-start 零缺失键、tf=0 前向无 NaN、窗口评估基线已测，见 §4.1）；完整开环训练待跑
> 模型：`src/models/model_open_loop_transition.py` 的 `OpenLoopTransitionModel`
> 训练：`scripts/training/train_open_loop_transition.py`（默认热启动 gt_transition + 纯闭环 tf=0）
> 评估：`scripts/evaluation/eval_rollout.py --windowed`
> 姊妹方向：[13_closed_loop_state_transition.md](13_closed_loop_state_transition.md)（无界自回归 rollout）、[14_gt_observed_transition.md](14_gt_observed_transition.md)（每步真实观测，当前主线）
> 核心思想：**每个窗口仅以 1 帧 GT 观测作种子，窗口内 K 步把模型自身预测喂回（s 与 z 自演化）**，窗口结束重新观测 → 把 rollout 漂移约束在 K 步内
> 创建：2026-06-17
> 🔁 **2026-06-24 补充**：本方向的「窗口开环 rollout（给定动作序列预测 K 步形状）」正是 [16 约束导向控制](16_constraint_oriented_control.md) 里「前向可微黑箱」的直接实现——控制求逆时复用本窗口前向。实物强迟滞验证已解锁（采集程序 `docs/ref/Main UI-plc/` + [11 §最小验证平台](11_sim_to_real_transfer.md)）。

---

## 〇、定位：s_{t-1} 来源轴上的第三个象限

13 / 14 / 15 三个方向本质是按"前一状态 s_{t-1} 从哪来"区分的：

| 维度 | 13 无界 rollout | **15 窗口开环（本方向）** | 14 全 GT 驱动 |
|------|----------------|--------------------------|--------------|
| s_{t-1} 来源 | 模型预测（整序列一路喂回） | **模型预测，但每 K 步用 GT 重新种子** | **每步真实观测** |
| s 自由运行步数 | 整序列 T（无界） | **≤ K-1 步（有界）** | 0（从不） |
| z 演化喂什么 | 预测 s（train/inference gap） | 预测 s（窗口内） | 真实 s |
| 误差累积 | 无界（实测漂移比 1170×，但分母曾被污染见 §五） | **约束在 K 步内（锯齿、有界）** | 无（每步重置真实） |
| 部署语义 | 无法观测、多步前瞻 | **"观测一次 → 预测 K 步"** | 每步都观测 |
| 适用场景 | 未来扩展（预测未来 N 步） | **每 K 步能观测一次的开环预测** | 常规部署（每步观测） |

**一行区分**：13=(s,z) 从 1 个种子自由跑到序列尾；14=s 永不自由跑；**15=(s,z) 至多自由跑 K-1 步后用 GT 重新锚定**。K 这个界使 train/inference gap 有界、可部署。

### 为什么单独成方向（不并入 13）

13 的纯自回归 rollout 漂移比 1170×（且该数字来自 `eval_rollout.py` 旧版——见 §五 bug 1，分母被污染，实际未必如此极端，但无界累积的性质不变）。15 用"每 K 步重观测"把累积截断，是部署可用的开环预测——这需要**不同的训练数据组织**（窗口需连续的 GT 锚定段，非 13 的整 episode 也非 14 的可打乱单帧）和**不同的评估**（窗口内逐位误差曲线）。并入 13 会污染其"无界未来扩展"的定位，并入 14 会污染其"tf=1.0 恒定"的身份。

---

## 一、核心设计

### 1. 1 帧 GT 种子 + K 步自回归 rollout

```
窗口 [t0, t0+K)：
  种子：s_seed = GT positions[t0-1]   （唯一 GT，锚定绝对位姿）
        z_0     = z_init(encode(aw[t0]))   （cond-only 初始化）
  for k = 0 .. K-1:                          ── 窗口内自回归 rollout ──
    ŝ_{t0+k} = F(s_prev, z_{k-1}, aw[t0+k])   s_prev / z 都喂"自身上一步预测"
  窗口结束 → 下一窗口重新用 GT 种子
```

**性质**：rollout 漂移被约束在 K 步内；每窗口重置基准。部署时"拍一帧图像 → 骨架化得 s_seed → 开环预测 K 步"。

### 2. 假设的正确表述（重要——避免过度声称）

用户最初表述"假设只有最近几十步影响当前状态"在字面上暗示**纯 action-history-only**（仅靠 a_{t-K..t} 推状态，即 `SpatialSequenceModel` 的前馈稳态假设）。但本框架**用 1 帧 GT 种子锚定绝对位姿**，并非纯 action-history。

**正确表述**：**"每 K 步一个绝对锚点约束位姿漂移；K 步潜轨迹 z 编码路径依赖（迟滞）。"**

为何保留 1 帧种子而非纯冷启动（s_0=0）：纯冷启动要求模型仅从动作历史推断绝对位姿，这正是 [13 §〇](13_closed_loop_state_transition.md) 论证的"迟滞下稳态假设失效、欠定"的情形。1 帧绝对锚点 + K 步累积潜轨迹才是自洽设计：位姿由锚点定，迟滞由 z 轨迹定。

### 3. z 是窗口内记忆（不跨窗口携带）

z 在**每个窗口**重新初始化（z_0 = z_init(cond)），是窗口内记忆，**不跨窗口携带**。
- 跨窗口携带 z 会退化成"有界累积的方向 13"，违背"只看一个 window"的定位。
- 这与训练数据组织一致：每个 episode 样本（窗口）自包含，样本间可打乱。

> K=40 时 z_0 影响衰减 ≈ 0.9^40 ≈ 2%（继承 [14 §一.5](14_gt_observed_transition.md) 的论证），故 cond-only 初始化足够。

### 4. dense supervision（每步 loss）

z 无 GT，靠端到端学。窗口内**每步都预测 ŝ_j、每步都算 loss**（dense），给无 GT 的 z 每步直接梯度（sparse 单点 loss 的梯度穿 K 层 GRUCell 衰减到不了 z_init/早期 Φ_z）。部署/评估只看每窗口最后一步或整窗口轨迹。

---

## 二、实现（全部复用 13/14 基础设施）

`OpenLoopTransitionModel` 继承 `StateTransitionSpatialModel`，**复用全部 forward / z_module（零参数增量，三类模型 state_dict key 完全相同）**，仅固化 training_spec 为"窗口开环"身份 + `open_loop_mode` buffer（供 model_loader 从 config.json 区分）。

| 组件 | 来源 | 说明 |
|------|------|------|
| forward / z_module | 继承父类 | 零改动复用（`tf_ratio=0` 走纯闭环，s/z 都喂预测） |
| training_spec | **固化** | episode 模式 + `teacher_forcing_ratio=0.0`（默认纯闭环）+ `episode_len=K` |
| open_loop_mode buffer | **新增** | 标识本模型，供 model_loader 区分 |
| tf 退火 | **新增**（PhaseSpec + trainer） | `tf_anneal_epochs`/`tf_min`/`tf_schedule`，可选退火 1.0→0.0 |

### 文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `src/models/model_open_loop_transition.py` | 新建 | `OpenLoopTransitionModel`（继承，固化 spec + buffer） |
| `src/models/__init__.py` | 编辑 | lazy export |
| `src/utils/model_loader.py` | 编辑 | state_transition 分支按 config.json `model` 字段加 `OpenLoopTransitionModel` 分支 |
| `src/training/spec.py` | 编辑 | PhaseSpec 显式声明 `tf_anneal_epochs`/`tf_min`/`tf_schedule`/`dense_step_weight`（向后兼容默认） |
| `src/training/trainer_unified.py` | 编辑 | `_effective_tf_ratio`（epoch 退火）+ `_compute_sequence_losses` 加 z_norm/tf_ratio 监控 |
| `scripts/training/train_open_loop_transition.py` | 新建 | 训练入口（热启动 gt_transition + tf 退火 CLI） |
| `scripts/evaluation/eval_rollout.py` | 编辑 | 修 onestep z 污染 + 新增 `--windowed` 窗口开环评估 |

### 关键正确性（经对抗验证工作流确认）

- **闭环数据流干净**：`tf_ratio≤0` 下，step-0 的 `init_skeleton` 是唯一进入预测路径的 GT；step 1+ 的 s-path（state_encoder）和 z-path（z_cell 输入 `[cond, flatten(prev_s)]`）都喂模型自身预测；监督目标恒为 GT（无预测泄漏）。
- **热启动兼容**：`GTObservedTransitionModel` 与 `StateTransitionSpatialModel` 参数数完全相同（256,620），state_dict key 仅多一个非参数 buffer。⚠️ gt_transition checkpoint 用旧 GRUCell 键名（`gru.weight_ih`），**必须经 `_migrate_gru_keys`** 迁移到 `gru.weight_ih_l0`，否则 `strict=False` 静默丢弃整层训练好的空间 GRU → 蓝点。训练入口已内置迁移 + missing-key 断言。

---

## 三、训练配方（验证工作流建议）

**首试（成本最低）**：从 gt_transition checkpoint 热启动，**直接纯闭环 tf=0**（默认），~15-20 epoch，dense uniform 加权，报告窗口 rollout 漂移比。
- 理由：gt_transition 已学好单步动力学（per-frame rollout MSE ~1.4e-8，56× 优于 copy）；per-frame 运动极小（~0.27mm/帧），K=40 内误差累积缓慢，退火**未必必要**。
- 13 的"1170×"是 Stage-0 per-frame 模型测的（且分母被污染，见 §五），不是热启动 episode 模型的预测。

**升级（仅当首试漂移 > ~50×）**：staircase 退火 `--tf_ratio 1.0 --tf_anneal_epochs 15 --tf_schedule staircase`（前 7 epoch 纯 GT、后 8 epoch 纯闭环）。**优先 staircase 而非 linear**：linear 的中段 0<tf<1 下速度输入 v=prev-prev_prev 会混入 GT/预测帧（标准 scheduled sampling 性质，非 bug 但噪声大），staircase 二值切换规避之。

**z 监控（早期预警）**：trainer 已输出 `z_norm_monitor`（每 epoch 写 loss_log.csv）。z 无 GT、闭环下 z_cell 输入分布漂移，**z 失稳先于 skeleton loss 失稳**——若 z_norm 持续增长而 skeleton loss 低，是潜变量不稳定信号（长程会爆发）。

**不要过度声称正则**：仿真迟滞弱（线性阻尼），z 应只携带少量真实信号，skeleton loss 主要由 cond+s_{t-1} 驱动。`delta_scale·tanh` + GRUCell 有界激活足够；除非 z_norm 监控显示漂移，否则不加谱正则。

---

## 四、评估：窗口开环核心指标

```bash
# 窗口开环评估（每 K=40 步用 GT 重新种子）
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_rollout.py \
    --checkpoint train_log/open_loop_transition/exp_xxx/phase_open_loop_transition/model/best_model.pt \
    --data_dir data/seq_rz_c2_sk --windowed --window_len 40

# 对比：整序列单种子 rollout（方向 13 风格）
CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_rollout.py \
    --checkpoint <同上> --data_dir data/seq_rz_c2_sk
```

**核心指标**（窗口内位置 k=0..K-1 的误差曲线，聚合所有窗口）：

| 指标 | 含义 | 健康判据 |
|------|------|----------|
| `drift_by_k` = rollout_MSE[k] / onestep_MSE[k] | 距上次观测 k 步的漂移比 | k=0≈1，随 k 缓增；K-1 处 <~30× 为健康（sqrt(K)≈6.3 随机游走地板） |
| `onestep_err_by_k` | 干净 teacher-forced 参考（独立 z_tf 轨迹） | ~1e-8 量级（与 gt_transition 一致） |
| `z_norm_by_k` | 窗口内 ‖z_t‖ | 窗口内有界（start/mid/end 接近） |

**区分性结果**：窗口开环 = 有界锯齿（每窗口 k=0 回落）；13 无界 = 单调发散；14 = 平坦。

> 期望健康模型落在 5×–30× 漂移比。>100× 说明热启动没迁移好、需退火。

### 4.1 预训练基线（gt_transition 模型，未做开环训练）

`eval_rollout.py --windowed --window_len 40` 跑在 gt_transition checkpoint（TF=1.0 训练，从未见过自身预测）上，给出开环训练的**起点**：

| k（距上次观测） | rollout_MSE | onestep_MSE | drift_ratio | z_norm |
|---|---|---|---|---|
| 0 | 7.7e-8 | 7.7e-8 | **1.0×** | 2.36 |
| 10 | 5.0e-6 | 1.3e-8 | 367× | 2.47 |
| 20 | 1.5e-5 | 1.0e-8 | 1450× | 2.49 |
| 30 | 2.6e-5 | 8.3e-9 | 2599× | 2.53 |
| 39 | 3.7e-5 | 8.7e-9 | **3681×** | 2.57 |

**读数**：
- k=0 drift=1×（种子是 GT，模型单步预测准，与训练分布一致）。
- drift 随 k 近二次增长（1→3681×）→ **纯 s 误差累积**（train/inference gap）。
- `onestep_MSE` 恒 ~1e-8（干净的 teacher-forced 参考，z_tf 修复生效；旧污染版会随之漂移）。
- **z_norm 稳定（2.36→2.57）→ z 没爆**。故基线漂移来自 s 累积，不是 z 失稳——开环训练（让模型适应自身预测）是对症下药，而非 z 收缩正则。

> 这就是开环训练要消除的 gap：目标把 k=39 drift 从 3681× 压到 <~30×。z 已稳定，故首选 tf=0 直接训练（成本低），退火仅兜底。

---

## 五、修复的两个 bug（实现时发现）

### Bug 1：`eval_rollout.py` onestep 参考被 z 污染

旧版 `rollout_one_sequence` 用**单个 `z_t`** 同时供 onestep 参考分支和 rollout 分支，且 `z_t` 仅被 rollout 路径更新（喂预测 s）→ onestep 参考的 z 是 rollout 演化的，**不是干净的 teacher-forced 参考**。漂移比 `rollout/onestep` 分母被污染。

**影响**：13/14 文档引用的"1170× 漂移"来自此污染分母，**不是干净的 rollout-vs-teacher-forced 比**（实际未必如此极端，但无界累积性质不变）。

**修复**：维护两条独立 z 轨迹——`z_t`（rollout，喂预测 s）+ `z_tf`（onestep 参考，喂 GT s）。窗口评估函数同样分离。修复后 onestep 是真正的单步 teacher-forcing 上界。

### Bug 2（性质，非代码 bug）：中段 tf 速度混入

`_compute_sequence_losses` 在 0<tf<1 下，每步独立 scheduled sampling 使速度输入 v=prev-prev_prev 混入 GT/预测帧。**端点 tf∈{0,1} 无此问题**（本方向默认 tf=0；退火用 staircase 规避中段）。已在 trainer 注释说明，无需改 scheduled-sampling 推进逻辑（那是标准 Bengio 语义，改了反而错）。

---

## 六、与现有方向体系的整合

- 本方向补全 s_{t-1}-来源轴的第三点：[13](13_closed_loop_state_transition.md)（无界）/ [14](14_gt_observed_transition.md)（每步）/ **15（每 K 步）**。
- z 复用 13/14 的可学习潜变量设计（方案 A），无 GT，端到端学。
- 部署层（[10_vision_corrected_deployment](10_vision_corrected_deployment.md)）契合：图像骨架化提供每 K 步的绝对锚点，开环模型预测窗口内轨迹。

---

## 七、未来扩展（记录，非当前）

- **缩短 K**：若部署观测频率高，K 可减小；此时 z_0 影响衰减论证（~0.9^K）需重算，z_0 初始化方式可能需要消融。
- **跨窗口携带 z 的有界累积变体**：若要更长的有效记忆又不重观测，可让 z 跨窗口携带（退化为有界版 13）——需配 z 收缩正则。
- **实物强迟滞验证**：仿真迟滞弱，z 的迟滞有效性留实物数据验证（与 13/14 一致）。
