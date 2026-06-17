# 方向：全 GT 驱动单步状态转移框架（独立于自回归 rollout）

> 状态：已实现（Stage 0），冒烟测试通过
> 模型：`src/models/model_gt_transition.py` 的 `GTObservedTransitionModel`
> 姊妹方向：[13_closed_loop_state_transition.md](13_closed_loop_state_transition.md)（自回归闭环，未来扩展）
> 核心思想：**前一状态 s_{t-1} 永远来自真实观测**（仿真 GT / 实物图像骨架化），模型做单步转移，z 跨帧演化
> 公式：见 §六（数据流与训练公式精确定义）；速度与改进：见 §七（性能瓶颈与改进路线）
> 创建：2026-06-15

> 📝 **2026-06-17 勘误**：本文对比方向 13 时引用的"1170× 漂移"（§一对比表 / §三冒烟对比）来自 `eval_rollout.py` 旧版的**污染分母**——onestep 参考与 rollout 共用单条 `z_t`，被 rollout 演化的 z 污染。该 bug 已修（独立 `z_t`/`z_tf` 轨迹），见 [15 §五 Bug 1](15_open_loop_windowed_transition.md)。故 1170× 非干净 rollout-vs-teacher-forced 比；**本文核心结论（GT 驱动下 s 每步重置真实 → 不累积漂移）不受影响**。另：姊妹方向 [15 窗口开环](15_open_loop_windowed_transition.md) 已实现，补全 s_{t-1}-来源轴第三点（每 K 步观测）。

---

## 〇、为什么单独立一个框架

[方向 13](13_closed_loop_state_transition.md) 的 `StateTransitionSpatialModel` 是"自回归闭环"框架——推理时把模型自己的预测喂回（rollout），适用于"无法每步观测真实状态、需一路推下去"的场景。

但**实际部署并非如此**。软体机器人的真实部署是：

```
每一步：
  采集图像 → 骨架化 → 得到真实 s_{t-1}
  输入 (真实 s_{t-1}, action_t) → 模型 → 预测 s_t
```

**前一状态永远是真实观测**（仿真里是 GT `positions[t-1]`，实物里是图像骨架化结果），不是模型自己的预测。模型本质是**单步状态转移**——给定真实的前一状态和当前动作，预测这一步的状态。这像一个**观测驱动的滤波器/预测器**，而非一路推到底的世界模型。

因此纯自回归 rollout（方向 13 的核心验证方式）**不是当前场景**，而是"跑通当前管线后"的未来扩展方向。本框架（14）专注当前真实场景。

### 与方向 13 的对比

| 维度 | 方向 13（自回归闭环） | **方向 14（全 GT 驱动）** |
|------|---------------------|--------------------------|
| 前一状态 s_{t-1} | 模型自己的预测（rollout 喂回） | **真实观测**（GT / 图像骨架化） |
| 推理方式 | 自回归一路推 T 步 | **单步转移，每步喂真实前一状态** |
| train/inference | 不一致（train 喂 GT，inference 喂预测）→ 需 scheduled sampling | **完全一致**（都喂真实 s_{t-1}） |
| s 误差累积 | 有（rollout 漂移，实测 1170×） | **无**（每步重置为真实观测） |
| 误差风险源 | s 和 z 都可能漂移 | **仅 z**（z 无 GT，跨帧演化） |
| 适用场景 | 无法每步观测（如预测未来 N 步） | **常规部署**（每步都能采集图像） |
| 定位 | 未来扩展 | **当前主线** |

---

## 一、设计要点

### 1. s_{t-1} 永远真实

```
ŝ_t = F(真实 s_{t-1}, z_{t-1}, action_t)
```

- 仿真：`prev_skeleton = positions[t-1]`（GT，已在 `.npz`，无需重采）
- 实物：`prev_skeleton = 图像骨架化(s_{t-1})`（Stage 2 接入感知前端）

**性质**：s 每步重置为真实观测 → s 不累积漂移。train 与 inference 完全一致（都喂真实前一状态），**无需 scheduled sampling**（`teacher_forcing_ratio = 1.0`）。

### 2. z 跨帧演化（保留，用户决策）

z 是可学习迟滞潜变量，跨帧演化 `z_t = Φ_z(z_{t-1}, a_t, s_{t-1})`，无 GT，端到端从 skeleton loss 学。在"每步真实 s"下，z 是**唯一跨帧、唯一无 GT 的状态**。

**为什么保留 z**：s_{t-1}（位置）和 action 虽是真实输入，但 z 可编码**位置+动作之外的深度历史**——例如内部应力方向（充气中 vs 放气中），这些信息位置本身不直接体现。z 是这层历史的低维潜表示。

**风险**：z 无 GT、跨帧演化，可能漂移。但在全 GT 驱动下，z 漂移的影响是**逐步纠正的**（每步用真实 s 重置位置基准），不致失控。冒烟测试证实 z 收敛有界（见 §三）。

### 3. 数据集：单帧 vs episode（窗口模式）

- **z 训练需要序列**：z 跨帧演化必须用 episode/窗口模式（z 在窗口内 BPTT 学习）。
- **s 学习不需要顺序**：单帧模式每样本独立即可（prev 来自 GT，样本间可 shuffle）。
- 本框架用 **窗口模式**（episode 模式，K=episode_len）：z 在窗口内 K 步演化，**样本自包含、样本间可打乱**（z 不跨样本携带）。

### 4. z 演化范围 = 状态窗口（K 步，可打乱）

z **不跨样本叠加**，只在每个样本的状态窗口 `[s_{t-K}, ..., s_{t-1}]` 内演化 K 步，每步喂**真实状态**：

```
z_0 = z_init(cond)                          # cond-only 初始化（见 §5）
z_1 = Φ_z(z_0,  a_{t-K+1}, s_{t-K})         # 真实 s_{t-K}
...
z_{K-1} = Φ_z(z_{K-2}, a_{t-1}, s_{t-2})    # 真实 s
ŝ_t = F(真实 s_{t-1}, z_{K-1}, a_t)          # 唯一的部署预测
```

- K 默认 = action_window 长度（40），可调（`--episode_len`）。
- 样本自包含 → **可打乱**（解决"必须按顺序"的顾虑）。
- z 演化**恒喂真实 s** → train/inference 一致，s 不漂移。

### 5. z_0 初始化：cond-only（先简后消融）

z_0 用 cond（动作编码）初始化，**暂不用动作+状态联合**。理由（经设计验证工作流确认）：K=40 步演化下，GRUCell 门控使 z_0 的留存 ≈ 0.9^40 ≈ 2%，且每步都注入真实 s，z_{K-1} 编码的是 40 步历史而非初始先验 → z_0 初始化方式影响很小。

实践步骤：先 cond-only 跑通 → 加 zero-init baseline 对比 → 仅在 z 不稳定时才试动作+状态联合。训练时跟踪 `‖z_0‖` vs `‖z_{K-1}‖`（eval 已有 z_norm）判断 z_0 是否冗余。

### 6. dense supervision（每步预测 + 每步 loss）——关键

z 无 GT，只能端到端从 skeleton loss 学。监督密度是 z 能否学动的**决定因素**：

- **sparse（只预测窗口最后一步 ŝ_t）**：单点 loss 要 BPTT 穿过 K=40 层 GRUCell，梯度到不了 z_init 和早期 Φ_z → **z 学不动**。
- **dense（窗口内每步都预测 ŝ_j，每步都算 loss）**：K 个监督点，每个演化步拿直接梯度，相当于"40 条 1 步短路径" → **z 学得动**。且**几乎免费**（z 演化本就要 K 次 forward，多算 MSE 可忽略）。

**无数据泄漏**（经对抗验证）：预测 `ŝ_{j+1}=F(s_j, z_j, a_{j+1})` 的 GT 是 s_{j+1}，而 s_{j+1} 从未出现在预测路径（z_j 只依赖 ≤s_j 的历史）。状态窗口的双重使用（既作输入又作 GT label）是标准 teacher forcing，合法。

**部署/评估只看最后一步 ŝ_t**：dense 是训练手段（帮 z 学），部署时无 GT 不算 loss，直接用最后一步预测。可选递增权重（`--dense_step_weight linear`）让最后几步贡献更大，缓解早期窗口噪声。


---

## 二、实现（全部复用方向 13 基础设施）

`GTObservedTransitionModel` 继承 `StateTransitionSpatialModel`，**复用全部 forward / z_module**，仅固化训练 spec 为"全 GT 驱动"身份：

| 组件 | 来源 | 说明 |
|------|------|------|
| forward / z_module | 继承父类 | 零改动复用 |
| training_spec | **固化** | episode 模式 + `teacher_forcing_ratio=1.0` + episode_len |
| gt_observed_mode buffer | **新增** | 标识本模型，供 model_loader 从 config.json 区分 |

### 文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `src/models/model_gt_transition.py` | 新建 | `GTObservedTransitionModel`（继承，固化 spec） |
| `src/models/__init__.py` | 编辑 | lazy export |
| `src/utils/model_loader.py` | 编辑 | state_transition 分支按 config.json `model` 字段区分子类 |
| `scripts/training/train_gt_transition.py` | 新建 | 训练入口（episode, cuda1, 短 epoch） |
| `scripts/evaluation/eval_gt_transition.py` | 新建 | **观测驱动评估**（s 每步真实 + z 演化，监测 z 漂移） |

> model_loader 用 config.json 的 `model` 字段（而非 state_dict key）区分本模型与方向 13 模型——二者继承关系导致 state_dict key 完全相同，无法靠 key 区分。

---

## 三、冒烟测试结果（cuda1, 5 mini-batch + 15 步评估）

训练 loss 持续下降（4.3e-5 → 7.8e-6），观测驱动评估：

| 指标 | 值 | 含义 |
|------|-----|------|
| per-step MSE | ~6e-6，15 步**几乎不增长** | s 每步真实 → **部署精度稳定，无 s 累积漂移** |
| z norm | 0.64 → 1.33，**收敛**（13 步后稳定） | z 跨帧演化但**有界收敛** |
| z drift ratio | 2.08× | z 启动后稳定，非单调发散 |

**对比方向 13 自回归 rollout**（漂移比 1170×）：全 GT 驱动下 s 漂移消失，仅 z 有微小收敛性演化——**这正是当前部署场景的预期**。

---

## 四、与现有方向体系的整合

- 本框架是 [13_closed_loop_state_transition.md](13_closed_loop_state_transition.md) 的**当前主线**，13 退为未来扩展（无法每步观测时的自回归预测）。
- z 复用方向 13 的可学习潜变量设计（方案 A），无 GT，端到端学。
- 部署层（[10_vision_corrected_deployment](10_vision_corrected_deployment.md)）天然契合：图像骨架化提供真实 s_{t-1}，闭环模型做单步转移。

---

## 五、未来扩展（记录，非当前）

- **2D→3D 状态获取**（Stage 2）：实物上从图像骨架化得到真实 s_{t-1}（替代仿真 GT）。需感知前端。
- **纯自回归 rollout**（方向 13）：当无法每步观测时（预测未来、快速控制跳过观测），把模型预测喂回。这是已实现但当前不主用的扩展。
- **z 收缩正则**：若长序列上 z 漂移失控，对 Φ_z 加谱约束（与方向 13 Stage 1 共享）。

---

## 六、数据流与训练公式（精确定义）

> 本节把方法的数据流动与训练过程用公式写清楚。符号与 `model_state_transition.py::forward` 一一对应。

### 6.1 符号

| 符号 | 维度 | 含义 |
|------|------|------|
| $a_t$ | $\mathbb{R}^D$ | 第 $t$ 步驱动动作（$D=2$） |
| $\mathcal{W}_t=[a_{t-K+1},\dots,a_t]$ | $\mathbb{R}^{K\times D}$ | 以 $t$ 结尾的动作窗口（$K=$ `window_size`=40） |
| $s_t$ | $\mathbb{R}^{N\times 3}$ | 中心线骨架（$N=$ `n_nodes`=31，归一化空间） |
| $z_t$ | $\mathbb{R}^{z_d}$ | 可学习迟滞潜变量（$z_d=$ `z_dim`=16，**无 GT**） |
| $c_t$ | $\mathbb{R}^{H}$ | 动作编码 cond（$H=$ `hidden_dim`=128，每步重算、无记忆） |
| $h^{(i)}_t$ | $\mathbb{R}^{H}$ | 第 $i$ 节点、第 $t$ 步的空间 GRU 隐状态 |
| $\bar{z}_t$ | $\mathbb{R}^{H}$ | z 投影（加性注入每个节点） |
| $\zeta_i\in[-1,1]$ | 标量 | 第 $i$ 节点沿 Z 轴的归一化位置（`linspace(-1,1,N)`） |

### 6.2 单步前向 $\hat{s}_t = F(s_{t-1},\,z_{t-1},\,\mathcal{W}_t)$

**(1) 动作编码**（无记忆，每步重算）：
$$c_t = \mathrm{Encoder}(\mathcal{W}_t)\in\mathbb{R}^H$$

**(2) 潜变量 z 演化**（唯一的跨步、无 GT 状态）：
- 冷启动首步：$z_0 = \mathrm{z\_init}(c_0)$（`Linear-SiLU-Linear`）
- 后续步骤（拼接 cond 与上一步骨架）：
$$\tilde{c}_t = [\,c_t\;;\;\mathrm{flatten}(s_{t-1})\,]\in\mathbb{R}^{H+3N}$$
$$z_t = \mathrm{GRUCell}_z(\tilde{c}_t,\;z_{t-1})\in\mathbb{R}^{z_d}$$
- 投影注入：$\bar{z}_t = W_z\,z_t\in\mathbb{R}^H$（`z_proj`）

**(3) 空间 GRU 种子**（warm start，编码位置 + 速度）：
$$v_{t-1}=s_{t-1}-s_{t-2},\qquad h^{(0)}_t=\mathrm{StateEnc}([\,\mathrm{flatten}(s_{t-1})\;;\;\mathrm{flatten}(v_{t-1})\,])\in\mathbb{R}^H$$

**(4) 沿 Z 轴逐节点因果传播**（悬臂梁因果性；节点位置嵌入 $e^{(i)}=\mathrm{ZEmb}(\zeta_i)$）：
$$g^{(i)}_t=c_t+e^{(i)}+\bar{z}_t,\qquad h^{(i)}_t=\mathrm{GRUCell}_{sp}(g^{(i)}_t,\;h^{(i-1)}_t),\quad i=1\dots N$$

**(5) 增量预测**（预测增量而非绝对坐标；`tanh` 收缩约束）：
$$\Delta^{(i)}_t=\alpha\cdot\tanh(\mathrm{DeltaHead}(h^{(i)}_t))\in\mathbb{R}^3,\quad \alpha=\text{delta\_scale（可学习标量）}$$
$$\hat{s}^{(i)}_t=s^{(i)}_{t-1}+\Delta^{(i)}_t$$

> 冷启动首帧（`prev_skeleton=None`）：$s_{t-1}\equiv 0$、$v\equiv 0$、$h^{(0)}=\mathrm{init\_hidden}(c_t)$，$\hat{s}^{(i)}_t=\Delta^{(i)}_t$。本框架训练/评估恒为 warm start（$s_{t-1}$ 恒真实）。

### 6.3 窗口模式 rollout（训练 episode）

一个 episode 是同一序列内连续 $T=$ `episode_len`(=40) 步。给定起始步 $t_0$ 与 GT 初始骨架 $s_{t_0-1}$：

- **z 初始化**（cond-only）：$z_0=\mathrm{z\_init}(c_{t_0})$
- **逐步单步转移**（$j=0\dots T-1$）：
$$\hat{s}_{t_0+j}=F\big(\underbrace{s_{t_0+j-1}}_{\text{恒为真实 GT}},\;z_{j-1},\;\mathcal{W}_{t_0+j}\big),\qquad z_j\text{ 在 }F\text{ 内演化}$$

> 关键性质（`teacher_forcing_ratio=1.0`）：每步喂**真实** $s_{t-1}$，不喂模型自身预测 → **s 不累积漂移**；跨步携带的**只有 z**。这正是「全 GT 驱动」与方向 13 自回归 rollout 的本质区别。

### 6.4 训练损失（dense supervision）

一个 episode 的 GT 为 $\{s_{t_0},\dots,s_{t_0+T-1}\}$，预测为 $\{\hat{s}_{t_0},\dots\}$。**dense**：每步都算 loss（给无 GT 的 z 每步直接梯度，否则单点 loss 的梯度经 $T$ 层 GRUCell 衰减到不了 z）。

**骨架回归**（逐步加权 MSE）：
$$\mathcal{L}_{skel}=\frac{1}{BT}\sum_{b=1}^{B}\sum_{j=0}^{T-1}w_j\cdot\mathrm{MSE}(\hat{s}^{(b)}_{t_0+j},\;s^{(b)}_{t_0+j})$$
- 权重 $w_j$：`uniform`（$w_j\equiv 1$）或 `linear`（$w_j=(j+1)/T$，最后步权重大）

**空间平滑**（相邻节点位移一致性）：
$$\mathcal{L}_{spatial}=\mathrm{MSE}(\Delta_{sp}\hat{s},\;\Delta_{sp}s),\qquad (\Delta_{sp}x)_i=x_{i+1}-x_i$$

**总损失**（各项乘 config 权重 $\lambda$，$\lambda_{skel}=1.0,\;\lambda_{spatial}=0.5$）：
$$\mathcal{L}=\lambda_{skel}\,\mathcal{L}_{skel}+\lambda_{spatial}\,\mathcal{L}_{spatial}$$

> ✅ **Q1 已修**：序列路径 `_compute_sequence_losses` 现对各项乘 `_get_loss_weight`（`spatial_smooth` 按 config ×0.5），与逐帧路径一致。此前是直接相加、漏乘权重。

z 无 GT → 无直接 loss，靠 BPTT 穿过 $T$ 步从 $\mathcal{L}_{skel}$ 端到端学。

### 6.5 一个 episode 的完整数据流

```
batch(episode) = { action_windows:(B,T,K,D),  gt_skeletons:(B,T,N,3),  init_skeleton:(B,N,3) }
   │
   │  z_0 = z_init( encode(aw[:,0]) )                       # cond-only 初始化
   ▼
 for j = 0 .. T-1:                                          # ── 逐步单步转移 ──
   aw[:,j] ──Encoder──► c_j                                 # 动作编码（无记忆）
   [ c_j ; s_{t-1}(GT) ] , z_{j-1} ──GRUCell_z──► z_j       # z 演化（唯一无 GT）
   z_j ──z_proj──► z̄_j
   [ s_{t-1}(GT) ; v=s_{t-1}-s_{t-2} ] ──StateEnc──► h⁰      # 空间 GRU 种子
   for i = 1 .. N:                                          # 沿 Z 轴因果传播
     c_j + e_i + z̄_j ──GRUCell_sp──► hⁱ
     hⁱ ──DeltaHead──► Δ_i = α·tanh(·)
     ŝⁱ_j = sⁱ_{t-1}(GT) + Δ_i
   stack i → ŝ_j                                            # (B,N,3)
 stack j → ŝ_{0..T-1}                                       # (B,T,N,3)
   │
   ▼  dense MSE vs gt_skeletons（+ 空间平滑）→ 反向传播（BPTT 穿过 T×N 个 GRU 步）
```

---

## 七、性能瓶颈与改进路线（1055 episodes / ~33 min）

> 现状：默认配置 `batch_size=4, episode_len=40, n_nodes=31, hidden=128`，一次完整 epoch（~1055 episodes）约 33 分钟。下面是瓶颈定位与按收益排序的改进项。

> ✅ **已实现（2026-06-16）**：
> - **S1**：节点级 `GRUCell` Python 循环 → 单次 `nn.GRU`（`batch_first`），cuDNN 融合 N 步递归。**逐位等价已核对**：`max|GRUCell-loop − nn.GRU| = 0`。
> - **S2**：节点位置嵌入 `z_embed(ζ)` 由 N 次逐节点调用合并为每 forward 1 次（`z_embed` 仍可训练，故**不**跨 forward 缓存）。
> - **Q1**：序列路径补 `_get_loss_weight`（`spatial_smooth` ×0.5），与逐帧路径一致。
> - ⚠️ **兼容性**：S1 改变了 `self.gru` 的 state_dict 键（`weight_ih/weight_hh/bias_ih/bias_hh` → `*_l0`），旧（GRUCell）checkpoint 不直接兼容，需迁移或重训。

### 7.1 瓶颈定位

- **每个 episode 的前向 = $T=40$ 步 × $N=31$ 节点 = 1240 次串行 `GRUCell` 小核启动**（外加每节点 `z_embed`/`delta_head` 各约 1240 次）。反向约再翻倍。
- **一个 epoch ≈ 1055/4 ≈ 264 batch × 1240 ≈ 3.3×10⁵ 次串行小核**（仅节点循环）。
- 模型极小、`batch=4` 极小 → **GPU 被 Python 循环 + 核启动开销 bound，算力严重闲置**。这是 33 min 的根因（非算力不足，而是小算子串行调度开销）。

### 7.2 速度改进（按收益排序）

| # | 改进 | 收益 | 代价/风险 |
|---|------|------|-----------|
| **S1** ✅ | **节点级 `GRUCell` Python 循环 → 单次 `nn.GRU`**：把 `for i in range(N): h=GRUCell(g_i,h)` 换成一次 `nn.GRU(input (B,N,H), h0)`，cuDNN 把 N 步递归融合成一个核。**已实现、逐位等价**（`max\|loop−GRU\|=0`）。消除 ~31×T 次核启动/episode | ✅ 等价已核对；state_dict 键变（`weight_ih`→`weight_ih_l0`），旧 ckpt 需迁移 |
| **S2** ✅ | **节点位置嵌入合并调用**：`z_embed(ζ)` 仅依赖节点（ζ 固定）→ 每 forward 由 N 次逐节点调用合并为 1 次（`z_embed` 仍可训练，不跨 forward 缓存） | 中 | 零风险 |
| **S3** | **`batch_size` 4 → 16/32**：模型小、GPU 闲置，加大 batch 摊薄核启动开销；显存几乎无压力 | 中-高（取决于显存） | 需调 lr（线性缩放） |
| **S4** | **`torch.compile(model.forward)`**：融合剩余小算子（`delta_head`+`tanh` 等） | 中 | 编译首次开销；动态 shape 需固定 |
| **S5** | **AMP 混合精度**（`autocast` + `GradScaler`）：tensor-core GPU ~1.5–2× | 中 | z/梯度数值稳定性需观测 |
| **S6** | **截断 BPTT**：$T=40$ 全程反传过 $40\times31$ 个 GRU 步很重；按 ~10 步分段 `detach`，有界 BPTT 代价 | 中（主要省显存/反传） | z 长程梯度被截断 |

> 建议先做 **S1 + S2**（行为保持、风险最低、收益最大），再用 `torch.profiler` 量化验证，再决定 S3–S6。

### 7.3 质量 / 建模改进

| # | 改进 | 说明 |
|---|------|------|
| **Q1** ✅ | **修 loss_weights 不一致**：`_compute_sequence_losses` 已补 `_get_loss_weight`（`spatial_smooth` ×0.5），与逐帧路径一致。 |
| **Q2** | **episode 时序平滑 loss**：实现 $\mathcal{L}_{temporal}=\mathrm{MSE}(\hat{s}_t-\hat{s}_{t-1},\;s_t-s_{t-1})$。序列路径本就有 $\hat{s}$ 序列，可正则时序连续性（§一去掉了未实现的 `smooth`，此处补一个真正可用的时序项）。 |
| **Q3** | **z 收缩正则**：若长序列 z 漂移，对 $\Phi_z$ 加谱范数约束（与方向 13 共享）。eval 已监测 `‖z‖` 漂移比。 |
| **Q4** | **`delta_scale` 灵活化**：当前全局可学习标量 $\alpha$ → 改 per-axis 或 per-node，避免各方向/各节点增量幅度被同一标量限制；或评估 `tanh` 界是否过紧。 |
| **Q5** | **z_0 初始化消融**：cond-only vs（动作+状态）联合（§五已记）。 |
| **Q6** | **验证集划分 + 早停**：默认 `n_epochs=500` 偏多，易过拟合；划出 val split 监控、早停。 |

### 7.4 工程改进

- **`torch.profiler` 定量**：改动前后各跑一次 profile，确认瓶颈是否从「核启动」转移到别处（避免靠猜）。
- **DataLoader**：`num_workers=4` 已开；上 GPU 时加 `pin_memory=True`（若未加）。
- **可复现基准**：固定一个 epoch 的 wall-time 作为回归指标，每次改动对比。
