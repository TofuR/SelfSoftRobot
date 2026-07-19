# 方向：闭环状态转移模型——从"稳态推断"到"一步状态转移"

> 状态：Stage 0 已实现；当前论文主线已转为 [15 稀疏观测窗口 OpenLoop](15_open_loop_windowed_transition.md)
> 基础模型：`model_spatial_sequence.py` 的 `SpatialSequenceModel`
> 相对方向：比 [01 自回归状态动力学（已归档）](../archived/directions/01_autoregressive_state_dynamics.md) 更进一步
> 核心思想：`s_t = F(s_{t-1}, a_t, z_{t-1})` —— 把前一步物理状态 + 可学习迟滞潜变量作为显式输入，让模型学习状态转移而非状态推断
> 创建：2026-06-15

> 🔁 **2026-07-20 主线更新**：本文“方向 14 为当前主线”的表述是历史决策记录。最终场景无法每步观察，当前主线为 [15](15_open_loop_windowed_transition.md) 的“观察一次、开环预测 K 步”；方向 14 只保留为局部转移误差与累计误差诊断。本文关于无界 rollout、潜变量 z 和误差收缩的理论分析继续有效。

> 📝 **2026-06-17 更新（勘误 + 定位收窄）**：
> 1. **1170× 勘误**：本文引用的"纯自回归 rollout 漂移比 1170×"（§〇·五冒烟结论）与表格"1000×"来自 `eval_rollout.py` 旧版——onestep 参考与 rollout 共用单条 `z_t`，分母被 rollout 演化的 z 污染。bug 已修（现维护独立 `z_t` rollout + `z_tf` 干净 teacher-forced 参考，见 [15 §五 Bug 1](15_open_loop_windowed_transition.md)）。故 1170× 非干净比值，定量待重测；**无界累积的定性结论不变**。
> 2. **定位收窄**：姊妹方向 [15 窗口开环](15_open_loop_windowed_transition.md) 已实现"每 K 步重观测的有界开环"。故本方向 13 现仅覆盖 s_{t-1}-来源轴上**唯一不被 14/15 覆盖**的点：**无界/多步前瞻 rollout**（运行中完全不重观测）。三方向分类见 [15 §〇](15_open_loop_windowed_transition.md)。
> 3. **脚本更替**：Stage 0/1 文件计划表中的 `train_state_transition.py`（逐帧）/`train_state_transition_s1.py`（固定 tf）已被 `train_gt_transition.py` + `train_open_loop_transition.py` 取代（`_s1` 尚有热启动 GRU 键迁移隐患：`load_state_dict(strict=False)` 未走 `_migrate_gru_keys`）。本文分析章节（§〇 稳态失效、§一 迟滞 z 设计、scheduled sampling/收缩映射误差界）仍是 14/15 共享的理论基础，保留。


## 设计锁定（本会话确认的关键约束）

1. **闭环状态转移** `s_t = F(s_{t-1}, a_t, z_{t-1})`，学转移不学状态。
2. **迟滞潜变量 z = 可学习（方案 A）**。判据：**实物上无法采集真实 z**——z 是 Bouc-Wen 模型的抽象内变量，无物理传感器可读，永远只能拟合/学习，不存在真值。故 z 由模型自演化 `z_t = Φ_z(z_{t-1}, a_t, s_{t-1})`，无 GT，端到端从 skeleton loss 学。
3. **仿真保持线性阻尼（PyElastica）**。**仿真只为跑通管线**（dataset→model→train→eval 全流程 + rollout 不发散），不强求仿真迟滞精度。z 是否真正建模了迟滞，留给**实物强迟滞数据**验证。
4. **3D 纯监督先行**，正向转移模型优先（非逆向/联合）。
5. `prev_skeleton = positions[t-1]` 已在 `.npz`（连续逐帧），Stage 0 **无需重采数据**。
6. **向后兼容**：`SpatialSequenceModel` / `PCSpatial` / 现有 dataset / trainer 调用一字不改。

---

## 〇·五、主线确定：GT 驱动窗口框架（当前主线）vs 自回归 rollout（未来扩展）

经过多轮讨论，确认实际部署模型是**单步状态转移 + 前一状态永远真实**，而非"一路自回归推下去"：

```
每一步：输入 (真实的 s_{t-1}, action_t) → 模型 → 预测 ŝ_t
         ↑ s_{t-1} 总是真实：仿真=positions[t-1]（GT），实物=图像骨架化
```

由此分流出两条路径，**当前主线是 GT 驱动窗口框架**：

| 维度 | GT 驱动窗口框架（方向 14，**主线**） | 纯自回归 rollout（本方向 13，**未来扩展**） |
|------|--------------------------------------|----------------------------------------------|
| 前一状态 s_{t-1} 来源 | **真实观测**（GT/图像骨架化） | 模型自己的上一步预测 |
| 误差累积 | s 不漂移（每步重置真实） | s 误差累积（漂移比可达 1000×） |
| z 演化喂什么 | 恒喂真实 s | 喂预测（train/inference gap） |
| 适用场景 | **当前部署**（每步都能观测） | 无法每步观测、需多步前瞻预测 |
| 训练/推理一致性 | 完全一致（TF=1.0） | 需 scheduled sampling 弥合 gap |

**冒烟验证结论**（cuda3，episode_len=12，3 epoch）：GT 窗口框架 z drift ratio=2.06x（z 温和演化、收敛有界），对比纯自回归 rollout 漂移比 1170× → **GT 框架是当前部署的唯一合理选择**，rollout 留待"去除部署时状态采集"再启用。

### 窗口模式关键设计（主线采用，详见 [14](14_gt_observed_transition.md) §4–6）

1. **z 演化范围 = 状态窗口 K 步，可打乱**：z 不跨样本叠加，只在每个样本的窗口 `[s_{t-K}...s_{t-1}]` 内演化 K 步（每步喂真实 s）。样本自包含 → 样本间可 shuffle。K 默认 = action_window（40），可调。
2. **z_0 = cond-only 初始化**（暂不用动作+状态联合）：K=40 演化下 z_0 留存 ≈ 0.9^40 ≈ 2%，初始化方式影响小；先简后消融（zero-init baseline 对比）。
3. **dense supervision（每步预测 + 每步 loss）——z 学习的关键**：z 无 GT，sparse（只预测窗口最后一步）会让 BPTT 穿 40 层、梯度到不了早期 Φ_z；dense 给每个演化步直接梯度，且几乎免费（z 演化本就 K 次 forward）。**无数据泄漏**（预测 ŝ_{j+1} 的 GT 是 s_{j+1}，而 s_{j+1} 从未出现在预测路径，标准 teacher forcing）。
4. **部署/评估只看最后一步 ŝ_t**：dense 是训练手段（帮 z 学）；部署无 GT 不算 loss，直接用最后一步预测。可选递增权重（`--dense_step_weight linear`）缓解早期窗口噪声。



---

## 〇、为什么必须放弃稳态假设

### 当前模型的信息流（纯前馈）

```
action_window [a_{t-K}, ..., a_t] → TemporalEncoder → cond → GRU(z₀→z_K) → pred_skeleton_t
```

`SpatialSequenceModel.forward()`（`model_spatial_sequence.py:138`）只接收 `action_window`，**没有任何前一步状态**。它隐含地假设：**给定近端动作历史，机器人已到达由该历史决定的稳态形状**。

这正是方向 01 §〇的"假设一：稳态假设"。其数学前提是：粘弹性材料在 `k→∞` 时 `A^k → 0`，初始条件被遗忘，稳态 `x* = (I-A)^{-1} B u*` 只取决于输入 `u*`。

### 为什么这个假设对软体机器人失效

软体机器人文献一致表明迟滞显著（参见 [literature_review_shape_reconstruction.md](../papers/literature_review_shape_reconstruction.md)）：

- 同一 `action` 值，充气路径 vs 放气路径 → 不同形状（迟滞回线）
- 加载速率依赖：同一目标值、不同速率 → 不同形状
- 多平衡态：非线性迟滞材料（Bouc-Wen）下，即使 `t→∞` 稳态也依赖加载路径

只要迟滞不可忽略，**"动作历史 → 唯一稳态形状"的映射就不存在**，前馈模型从根本上是欠定的。这解释了我们在 SpatialSequence / PC-Spatial 上看到的预测**系统性超前**于 GT——模型无法区分"正在弯曲中"和"正在回弹中"两种物理状态。

### 与方向 01（自回归）的关系

| 维度 | 方向 01（自回归） | **本方向（闭环状态转移）** |
|------|------------------|--------------------------|
| 输入 | action_window + (可选 prev_skeleton) | **prev_state + current action**（prev_state 为主输入） |
| 学什么 | 仍是 action → 完整状态（prev 仅作 GRU 初始化） | **状态转移函数 F(s, a)→s'** |
| 物理对应 | 带状态反馈的状态推断 | **状态空间模型 x(k+1)=f(x(k),u(k))** |
| 推理 | 单帧预测 | **闭环 rollout（自身预测喂回）** |
| 训练目标 | 逐帧 MSE | **单步转移 MSE（+ 多步 rollout MSE）** |

**关键区别**：方向 01 把 `prev_skeleton` 当作 GRU 的 hidden state 种子，预测的仍是"绝对状态"；本方向把 `prev_state` 作为**一等公民输入**，模型学习的是"状态如何变化"。这才是控制论意义上的闭环。

---

## 一、核心设计决策（回答你的四个问题）

### Q1：学正向转移 / 逆模型 / 两者？

| 模型 | 形式 | 用途 | 建议 |
|------|------|------|------|
| **正向转移**（forward） | `s_t = F(s_{t-1}, a_t)` | 自建模（"我现在长什么样"）、前向仿真、世界模型 | **优先实现** ★★★ |
| 逆模型（inverse） | `a_t = G(s_{t-1}, s_t)` | 控制（"我要从 s_{t-1} 到 s_t，该施加什么动作"）、逆运动学 | Stage 3 选做 |
| 联合 | 两者共享 encoder，双头输出 | 任务自适应 | Stage 3 选做 |

**推荐：先做正向转移。** 理由：

1. 本项目的核心科学问题是**自建模**（从感知恢复自身状态），正向转移直接服务于此。
2. 正向转移误差可验证（rollout 对比 GT）；逆模型在迟滞条件下可能多解（逆运动学可逆性见 [12_scientific_problems.md](12_scientific_problems_soft_robot_self_modeling.md) §B），优先级低。
3. 正向转移学到的 `F` 可直接作为逆模型的 backbone（共享 state encoder）。

### Q2：先 3D 后 2D

**Stage 0–1：纯 3D 监督。** `prev_state` 直接取 GT 的 `positions[t-1]`（归一化后的中心线），不引入图像。

**为什么 3D 先行：**

- `positions` 数组在 `.npz` 中是**连续逐帧存储**的（`dataset_spatial.py:60`，`(T, 3, N)`），`prev_skeleton = positions[t-1]` **无需重新采集数据**，只改 `__getitem__`。
- 把"状态转移"这个新范式先在最干净的 3D 信号上跑通，隔离"建模能力"与"2D→3D 状态获取"两个问题。
- 部署阶段（Stage 2）才需要从 2D 图像恢复 3D 状态——那时 `F` 已训练好，只需一个感知前端把图像映射到 `s`（这正是 [05_skeleton_to_shape](05_skeleton_to_shape_conversion.md) / 多视角方向的工作）。

### Q3：闭环下如何处理迟滞？——可学习潜变量 z

**核心洞察：迟滞的物理本质是"路径依赖"。路径依赖的关键信息是运动的趋势与历史，而速度（`v = s_{t-1} - s_{t-2}`）只是它的一阶近似。**

方向 01 §一已论证：仅凭 `positions[t-1} + action[t]` 无法区分充气/放气（两者位置相同、动作相同、方向相反）。速度 `v` 能区分方向，但真实迟滞是**高阶、非线性**的（Bouc-Wen 的 `dz/dt` 含 `|z|ⁿ` 项），单凭速度仍不足以完整建模。

**因此闭环模型引入一个可学习的迟滞潜变量 `z`。**

#### 为什么 z 是"可学习"而非"物理真值"

**判据：实物上无法采集到真实的 z。**

Bouc-Wen 内变量 `z` 是**模型抽象的迟滞状态**，不是物理可测量：

```
dz/dt = (dx/dt)·[A − |z|ⁿ·(β + γ·sgn(dx/dt·z))]
```

实物上传感器只能读到**可观测量**：形状（3D/图像）、动作（气压/电压）、时间。`z` 本身没有传感器，永远只能从可观测量**反推/拟合/学习**。所以"真 z"不存在，z 必然是一个学出来的近似。

→ **方案 A（可学习潜变量）是唯一自洽的选择**，且不依赖仿真重采：

```
z_0 = z_init(action_cond)                        # 冷启动从动作编码初始化
z_t = Φ_z(z_{t-1}, action_cond, s_{t-1})         # 自演化（无 GT，端到端学）
```

#### z 的三种实现方案对比（最终选定 A）

| 方案 | z 来源 | 需要 GT？ | 要改仿真/重采？ | 与"实物部署"一致性 | 选定 |
|------|--------|----------|----------------|------------------|------|
| **A. 可学习潜变量** | 模型自演化 `Φ_z` | 否 | 否 | ✓（实物也无 z 传感器，部署完全一致） | **✓ Stage 0** |
| B. 后处理解析 z | 离线 Bouc-Wen 拟合反解 | 是 | 是（重写数据） | ✗（实物拟合误差大，且不一致） | 否 |
| C. 仿真加 Bouc-Wen 本构 | 仿真直接产出 | 是 | 是（改物理） | ✗（实物无此物理量） | 否 |

方案 B/C 在仿真上有 z 的 GT，看似更"干净"，但**它们在实物上不可复现**（实物没有 Bouc-Wen 本构，也拟合不准）。方案 A 在仿真和实物上行为完全一致——这是部署一致性的硬要求。

#### 防止 z 坍缩 / 退化的设计要点

z 无 GT、从 skeleton loss 端到端学，最大风险是**退化为平凡解**（比如坍缩成 ≈ velocity，或常数）。缓解：

1. **z 用递归记忆单元实现（GRUCell/LSTMCell）而非自由 MLP**——rate-dependent 迟滞本质是高阶动力学，记忆单元提供天然的"内变量"演化结构，比自由向量更难坍缩。
2. **`z_init` 从 action_cond 初始化，但 `Φ_z` 转移与 cond 解耦**——z 是"演化中的迟滞状态"，cond 是"当前动作编码"，二者职责不同（见 §二 z 与 TemporalEncoder 的区别）。
3. **`latent_dim` 取 16–32**（远小于 skeleton 的 93 维），强制 z 是低维潜变量。
4. **训练后期（Stage 1）做可解释性检查**：观察 z 是否对"加载 vs 卸载"路径有选择性响应——若 z 对充放气路径无区分，说明坍缩了，需调整 `Φ_z` 结构。

#### 诚实警告：仿真阶段验证不了 z 的有效性

PyElastica 线性阻尼迟滞弱（[exp5](../project_status_report.md)）。这意味着：

- **仿真阶段能验证**：① 管线跑通（dataset→model→train→eval）；② rollout 不发散（误差累积可控）。
- **仿真阶段验证不了**：③ z 是否真正建模了迟滞（仿真迟滞太弱，z 学不到强迟滞行为是预期的，不是 bug）。
- z 的有效性（③）**留给实物强迟滞数据验证**——这是用户明确的目标定位（仿真跑通，实物见真章）。

> 这一条是关键预期管理：不要因为"仿真上 z 没提升"就否定方案。仿真本就不该提升。

### Q4：误差累积怎么办？

闭环模型的最大风险。两层来源：

- **s 的累积**：训练 `prev_s = GT`（teacher forcing），推理 `prev_s = 自身预测` → s 漂移。
- **z 的累积（z 无 GT，更棘手）**：z 没有 teacher forcing 的可能（无 GT），rollout 时 z 完全靠模型自演化，若 `Φ_z` 不稳定则 z 无界漂移，连带 s 失稳。

**三层缓解（缺一不可）：**

1. **学习 Δ 而非绝对状态**（架构层，针对 s）：
   ```
   s_t = s_{t-1} + Δ(s_{t-1}, z_{t-1}, action_cond)
   ```
   预测增量而非绝对坐标。输出范围小、天然连续、物理合理。

2. **Scheduled Sampling**（训练策略层，针对 s）：
   ```
   p = min(1.0, epoch / warmup_epochs)
   if random() < p: prev_s = predicted_s[t-1]   # 喂模型自己的预测
   else:            prev_s = gt_s[t-1]            # teacher forcing
   ```
   warmup 期纯 teacher forcing，逐步过渡。**注意：这只对 s 有效；z 无 GT 无法 scheduled sampling**，z 的稳定性完全依赖下一条。

3. **收缩约束（针对 s 和 z 的转移）**：
   物理上阻尼系统谱半径 `ρ < 1`（`A^k → 0`）。若 `F` 和 `Φ_z` 都是收缩映射，则 T 步 rollout 误差有界：
   ```
   ‖s 误差_T‖ ≤ ε_s / (1 - ρ_F)
   ```
   实现上：
   - s：对 `Δ` 输出加 `tanh` 缩放（`delta_scale`）。
   - z：对 `Φ_z` 的 GRU/LSTM 单元天然有 `tanh`/有界激活，谱半径可控；可选雅可比谱正则 `max(0, ρ̂ - 0.99)²`。

**必须同时报告三种验证指标**（否则看不出 s 和 z 各自的漂移）：
- **s 单步误差**（GT 喂 prev_s）：`MSE(F(s_{t-1}^{GT}, z_{t-1}, a_t), s_t^{GT})`
- **s rollout 误差**（自身预测 s 喂回，T 步）：从 `s_0` 自由滚动
- **z 漂移监测**：rollout 中 `‖z_t‖` 的范数轨迹（发散 = z 漂移失控）

---

## 二、架构设计

### 数据流（Stage 0–1，3D 纯监督）

```
  action_window [a_{t-K..t}] ── TemporalEncoder ──→ cond (B,128)
                                                        │
  z_{t-1} (B, latent_dim) ──┐                          │
                            ├─→ z_cell (GRUCell) ──→ z_t (B, latent_dim) ──┐ (返回，供 rollout)
                            │   输入 = (z_{t-1}, cond, s_feat)              │
  s_{t-1} (B,N,3) ──┐      │                                                │
  v=s_{t-1}-s_{t-2}─┤─→ StateEncoder ──→ state_seed (B,128) ──┐             │
                   │      │                                    │            │
                   │      │   cond + state_seed + z_t 注入       ↓            │
                   │      │   z 位置嵌入 ──→ GRU(z₀→z_K) ──→ 每节点 Δxyz     │
                   │      │                                                │
                   └──────┴────────────────────────  s_{t-1} + delta_scale·tanh(Δ) ──→ s_t 预测
                                                                     │                    │
                                                          返回 (s_t, z_t)        与 GT s_t 对比 (MSE + smooth)
```

### z 与现有 TemporalEncoder 的区别（重要，避免重复）

| | TemporalEncoder → cond | z_module → z_t |
|--|----------------------|----------------|
| 来源 | action_window（当前动作历史） | 自演化（前一步 z + cond + s） |
| 角色 | "现在施加什么力"（输入编码） | "材料当前处于什么迟滞状态"（潜状态） |
| 训练 | 有 action 监督（输入） | 无 GT，从 skeleton loss 学 |
| rollout | 每步重算（取决于 action_window） | 逐步演化（带历史记忆） |

**二者不可合并**：cond 每步从 action 重算，无记忆；z 是带记忆的演化潜变量。

### 模块清单（相对 SpatialSequenceModel 的增量）

> 命名约定：所有 z 相关子模块统一挂在 `z_module` 名义下，以便 `model_loader` 用 `'z_module' in keys` 一键检测（见 §三）。

| 模块 | 来源 | 改动 |
|------|------|------|
| `TemporalEncoder` | 复用现有 | 不变，action_window → cond |
| `StateEncoder` | **新增** | MLP：`(s_{t-1} ‖ v_{t-1})` (6N) → (B,128)，warm start 时**替代** `init_hidden(cond)` 作为 GRU 种子 |
| `z_module` | **新增**（含 `z_init`/`z_cell`/`z_proj`） | `z_init`（cond → z_0，冷启动）+ `z_cell`（GRUCell，输入 `[cond, flatten(s_{t-1})]` → z_t）+ `z_proj`（z_t → 注入 GRU 的特征）。`latent_dim`=16–32 |
| `z_embed`, `gru` | 复用 | 不变（z_embed 是 Z 轴位置嵌入，勿与 latent z 混淆） |
| `delta_head` | **新增**（替代 `slice_head`） | MLP：hidden → 3（输出 `Δ_raw`） |
| `delta_scale` | **新增** | 可学习标量（init 0.1），`Δ = delta_scale · tanh(Δ_raw/10)`，保证收缩、防 NaN |
| `init_hidden` | 保留 | 仅 `prev_s=None`（冷启动）时回退 |

> **z_cell 用 GRUCell 而非自由 MLP**：GRUCell 的门控提供选择性记忆更新（迟滞需要），且天然 `tanh` 有界利于收缩。z 通过 `z_proj` **加性注入** GRU 输入（`gru_input = cond + z_emb + z_proj`），与 cond/z_emb 职责分离，不重复。

### forward 签名（向后兼容关键）

```python
def forward(self, batch_or_action_window, prev_skeleton=None,
            prev_prev_skeleton=None, prev_z=None):
    """
    Args:
        batch_or_action_window: dict batch（训练，含 action_window/prev_gt_skeleton）
                                或 action_window 张量（推理/旧调用）
        prev_skeleton: (B,N,3) 前一步骨架。None → 回退 init_hidden(cond)（旧行为）
        prev_prev_skeleton: (B,N,3) 前两步骨架（速度）。None → v=0
        prev_z: (B, latent_dim) 前一步 z。None → z_init(cond)（冷启动）
    Returns:
        (s_pred, z_t) —— s_pred (B,N,3)；z_t (B, latent_dim) 供 rollout 喂回
        注：旧调用者只取返回值第一个（s_pred）即可，向后兼容。
    """
```

**冷启动（t=0）**：`prev_s=None` → 回退 `init_hidden(cond)`；`prev_z=None` → `z_init(cond)`。第一步退化为带 z 初始化的前馈，合理。

### Stage 0 的 z 退化警告（per-frame 训练的根本限制）——已由代码集成工作流确认

`UnifiedTrainer` 训练循环（`trainer_unified.py:303-337`）是**逐帧独立**（一个 sample = 一个 timestep，shuffle=True）。这意味着：

- **Stage 0 的 per-frame 训练下，z 没有跨帧记忆**——每个 sample 的 z 都从 `z_init(cond)` 重新初始化，无法演化。
- shuffle 还会打断 episode 连续性（batch 内相邻样本来自不同 episode），即使想"携带 z"也无从携带。
- **结果**：Stage 0 的 z 退化为 `cond` 的确定性函数，**不是真正的迟滞潜变量**。这是 per-frame trainer 的固有限制，不是 bug，也不是设计错误。

**Stage 0 的真实目标**（重新校准）：
1. ✅ 闭环**架构**正确（`prev_skeleton → model → s_t`，Δ 预测，z 结构能前向传播不 NaN）
2. ✅ 监督流跑通（dataset→model→train→eval）
3. ✅ s 的单步转移误差 ≤ 当前前馈模型

**Stage 0 验证不了**：z 是否真正建模迟滞——这需要 **Stage 1 的序列级训练**（episode 内逐步 rollout，z 跨帧演化 + scheduled sampling）。Stage 0 不应因"z 没提升"而否定整个设计。

**Stage 0 的 trainer 改动：零**（工作流确认）。模型 `compute_losses` 内部每次 forward 重新 `z_0 = z_init(cond)`，接受退化。trainer 改动全部推迟到 Stage 1（约 100 行新增序列级路径，非重写）。

---

## 三、分阶段修改计划（确认后开新 branch）

### Stage -1：前置实验（**已跳过**）

用户决策：跳过信息容量实验，直接 Stage 0。理由：目标定位是**实物部署**，仿真只是跑通管线，仿真迟滞强弱的定量分析对最终目标无决定性影响。保留此处仅为记录决策依据。

### Stage 0：3D 纯监督正向转移 + Teacher Forcing（核心）

**目标**：把闭环范式跑通——dataset→model→train→eval 全流程通；s 的单步转移误差 ≤ 当前前馈模型逐帧误差。

> **z 的预期**：per-frame 训练下 z 会退化为 cond 的函数（非真正迟滞潜变量，见 §二警告）。Stage 0 不指望 z 发挥作用，只验证它在结构上能前向传播、不 NaN。

| 文件 | 类型 | 改动 |
|------|------|------|
| `src/data/dataset_state_transition.py` | **新增**（推荐） | `StateTransitionDataset`（继承 `SpatialSequenceDataset`，重写 `__getitem__` 返回 `prev_gt_skeleton`=`positions[t-1]`、`prev_prev_gt_skeleton`=`positions[t-2]`，归一化同 `pc_center`/`pc_scale`）。保持原 dataset 不动。 |
| `src/data/dataset_spatial.py` | 不动 | 仅作为父类被继承，零改动。 |
| `src/training/dataset_factory.py` | 编辑 | `create_dataset` + `get_collate_fn` 各加一条 `"state_transition"` 分支。`spatial_collate_fn` 通用合并，加 key 无需改 collate。 |
| `src/models/model_state_transition.py` | **新增** | `StateTransitionSpatialModel`：复用 TemporalEncoder/z_embed/gru/slice_head；新增 `StateEncoder`、`z_module`（z_init+z_cell）、`z_to_hidden`、`delta_scale`；forward 读 batch 的 prev 键，返回 `(s_pred, z_t)`。`training_spec` 单 PhaseSpec，`dataset_type="state_transition"`，`supervision_mode="spatial_sequence"`，`active_losses=["skeleton","spatial_smooth","smooth"]`。 |
| `src/models/__init__.py` | 编辑 | lazy `__getattr__` 加 `"StateTransitionSpatialModel"` 分支。 |
| `src/utils/model_loader.py` | 编辑 | `_detect_model_type` 加检测（state_dict 含 `z_module` key → `'state_transition'`）；`load_model` 加分支。 |
| `scripts/training/train_state_transition.py` | **新增** | 仿 `train_spatial_sequence.py`，实例化新模型，注册归一化，`UnifiedTrainer` 训练。 |
| `scripts/evaluation/eval_rollout.py` | **新增**（Stage 0 也需要，验证 rollout） | 从 `s_0` 自由滚动 T 步（喂自身 s 和 z），对比 GT，输出单步/rollout 误差 + z 范数轨迹。 |

**向后兼容保证**：
- `SpatialSequenceModel` / `PCSpatialSequenceModel` 一字不改。
- 新 dataset_type 是新增分支，不影响现有 `"spatial_sequence"`。
- `model_loader` 对旧 checkpoint 检测逻辑不变（新 key `z_module` 不出现在旧 checkpoint）。
- 新模型 `forward` 的 `prev_skeleton`/`prev_z` 默认 `None`，旧调用方式（只传 action_window）仍能跑（退化为带 z_init 的前馈）。

**工作量**：2–3 天。

### Stage 1：Scheduled Sampling + z 跨帧演化 + Rollout 训练

**目标**：让 z 成为真正的迟滞潜变量；证明闭环 rollout 在 T 步内不发散。

> 工作流确认：Stage 1 不重写 trainer，而是**新增 ~100 行条件路径** `_compute_sequence_losses`，per-frame 路径完全不动。

| 文件 | 改动 |
|------|------|
| `src/training/spec.py` | PhaseSpec 加 `use_episode_mode`（bool，默认 False）+ `teacher_forcing_ratio`（float，默认 0.5）字段，向后兼容 |
| `src/training/trainer_unified.py` | 训练循环加条件分支：`use_episode_mode=True` 走新的 `_compute_sequence_losses`（episode 内逐步 rollout，scheduled sampling，z 跨帧携带 + BPTT），否则走现有 per-frame 路径 |
| `src/data/dataset_state_transition.py` | 加 episode 模式：`__getitem__` 返回连续 `(B, T, ...)` 序列而非单帧 |
| `src/models/model_state_transition.py` | 加 `init_z_from_action`（rollout/序列训练用）；可选雅可比谱正则 loss |
| `scripts/evaluation/eval_rollout.py` | **Stage 0 已建**，Stage 1 强化：从 `s_0` 自由滚动 T 步，喂自身 s 和 z，输出单步/rollout 误差 + `‖z_t‖` 范数轨迹 + 发散检测 |

**工作量**：2–3 天。

### Stage 2：2D→3D 状态获取（部署感知，延后）

**目标**：部署时无法拿到 GT `s_{t-1}`，需从图像恢复。

- 复用 `PCSpatialSequenceModel` 的 image encoder 思路：用 `image[t-1]` 预测 `s_{t-1}`（或直接把 `image[t-1]` 编码喂入 StateEncoder）。
- 或多视角三角化（[06_multi_view_2d_to_3d_skeleton.md](06_multi_view_2d_to_3d_skeleton.md)）。
- **此阶段才真正引入图像作为模型输入**——这是对 CLAUDE.md "图像仅作监督信号"约定的有意放宽，设计文档需明确说明。
- 工作量：4–5 天（仅 sketch，待 Stage 0/1 验证后细化）。

### Stage 3（可选）：逆模型 / 联合模型

- 共享 StateEncoder，加逆头 `G(s_{t-1}, s_t) → a_t`。
- 服务控制任务。工作量待评估。

---

## 四、风险与未决问题

| 风险 | 影响 | 缓解 |
|------|------|------|
| **z 在 Stage 0 退化**（per-frame 训练无跨帧记忆，z 退化为 cond 的函数） | Stage 0 的 z 不发挥迟滞作用 | **接受**——Stage 0 只验架构+管线+s 单步转移；z 成长留给 Stage 1 序列级训练。不要因 z 没提升否定设计 |
| **仿真迟滞弱**（exp5） | 仿真上 s 的闭环提升可能也不显著 | 仿真只验管线 + rollout 不发散；z/s 的迟滞有效性留**实物**验证（用户目标） |
| 误差累积发散（rollout 时 s 漂移、z 无 GT 无界漂移） | rollout 不可用 | Δ 预测 + `delta_scale·tanh` 收缩；z 用 GRUCell 有界激活；Stage 1 加 scheduled sampling；报告 rollout 误差 + `‖z_t‖` 范数轨迹 |
| Δ 输出爆炸 → s_t NaN | 训练崩溃 | `delta_scale`（init 0.1）+ `tanh` 裁剪；冷启动 `s_{t-1}=0` |
| episode 边界（t=0/1）prev 无定义 | NaN/跳变 | **工作流确认：无需 zero-pad**——样本循环 `t ∈ [seq_len-1, T-2]`，`positions[t-1]`/`[t-2]` 恒有效。冷启动仅发生在推理 rollout 的首帧，用 `init_hidden(cond)` + `z_init(cond)` |
| z 与 TemporalEncoder 重复（都从 action 来） | z 沦为冗余 | `z_proj` 加性注入，职责分离（cond=输入编码，z=演化潜状态）；Stage 1 后验检查 z 对加载路径的选择性响应 |
| 序列级训练（Stage 1）改动影响 trainer 通用性 | 破坏其他模型 | 工作流确认：~100 行**新增** `_compute_sequence_losses` 条件路径，per-frame 路径不动；PhaseSpec 加 `use_episode_mode` 标志默认 False |
| 2D→3D 状态获取精度差 | Stage 2 部署误差 | 仅 Stage 2，3D 先行隔离 |

---

## 五、建议的实验验证顺序

1. **Stage 0**（3D + teacher forcing，**零 trainer 改动**）→ 验证闭环架构正确、管线跑通、s 单步转移误差 ≤ 当前前馈模型。**同时写 `eval_rollout.py`**（项目当前**没有任何** rollout eval，必须新写）——先于训练写，及早抓集成 bug。
2. **Stage 1**（scheduled sampling + z 跨帧演化 + 收缩正则 + rollout 训练）→ 验证 z 成为真正迟滞潜变量、T 步 rollout 不发散。
3. **Stage 2**（2D→3D）→ 部署感知，仅在前两步成功后启动。
4. **Stage 3**（逆模型）→ 按需。

每个 Stage 产出实验报告（loss 曲线 + 单步/rollout 误差 + `‖z_t‖` 轨迹 + 可视化 GIF），归档到 `train_log/state_transition/`。

---

## 六、与现有方向体系的整合

- 本方向补全 [directions_overview.md](directions_overview.md) 第一层"核心技术"中"自回归状态动力学"的升级版。
- 与 [02_hysteresis_information_capacity.md](02_hysteresis_information_capacity.md) 互为前提：02 量化迟滞，本方向建模迟滞下的状态转移。
- 与 [05_skeleton_to_shape_conversion.md](05_skeleton_to_shape_conversion.md) 正交：本方向学骨架的时间转移，05 学骨架→表面的空间转换，可叠加。
- 部署层（[10_vision_corrected_deployment](10_vision_corrected_deployment.md)）的"在线适应 + 残差修正"天然契合闭环 rollout——闭环模型本身就是一个可在线修正的世界模型。

确认后，我会在新 branch（建议名 `feat/state-transition-model`）上从 Stage 0 开始实现。
