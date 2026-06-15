# 方向：自回归状态动力学——从"猜状态"到"推状态"

> 状态：概念阶段
> 解决问题：模型只有 action 输入，没有物理状态反馈，导致预测超前
> 核心思想：将前一步的物理状态（或预测状态）作为输入，让模型学习状态转移而非状态推断

---

## 〇、前提问题：两种关于软体机器人状态的基本假设

我们在实验中隐含地采用了一种假设，但存在另一种同样合理的假设。理清这一点是选择建模方法的基础。

### 假设一：稳态假设——"动作序列 → 最终形态"

**内容**：给定一段动作参数序列，软体机器人最终会到达一个确定的整体形态。

**物理前提**：每个动作维持足够长的时间，使得粘弹性材料充分松弛，系统达到力平衡。在这种条件下，每个动作值唯一对应一个平衡状态。

**我们一直在使用这个假设**：模型接收 action_history，直接输出当前 shape。隐含的条件是动作间隔足够长，机器人已经到达稳态。

**有效性条件**：
- 动作变化频率 << 材料松弛频率（准静态）
- 数据采集时每个动作保持足够长时间
- 不关心瞬态过程，只关心最终平衡

**失效场景**：
- 连续快速变化的动作序列（机器人来不及松弛）
- 需要预测中间过程（非平衡态的形状）
- 同一动作值，但到达路径不同（迟滞效应导致不同最终状态）

### 假设二：状态转移假设——"前一状态 + 当前动作 → 当前状态"

**内容**：当前状态由前一时刻的状态和当前施加的动作共同决定。状态是逐步演化的，不是从动作直接推断的。

**物理前提**：软体机器人的运动是在当前构型基础上叠加新的力。由于迟滞效应，同样的动作施加在不同初始状态上，产生不同的响应。

**有效性条件**：
- 动作变化频率 ≈ 材料松弛频率（非准静态）
- 需要连续追踪形状变化
- 存在显著的路径依赖（迟滞回线不可忽略）

**失效场景**：
- 材料为线性粘弹性（单一平衡态），且动作间隔远大于松弛时间（系统充分松弛后，初始条件的影响被遗忘）
- 没有前一状态可用（冷启动）

> 注意：只有当系统对每个恒定输入有**唯一平衡态**时（如线性粘弹性），假设二才会在充分松弛后退化为假设一。如果材料存在非线性迟滞（多平衡态），即使 t→∞ 稳态也依赖于加载路径，假设二不会退化。

### 两种假设的关系

两种假设不是矛盾的，而是对应不同的物理条件：

- **线性粘弹性材料**（单一平衡态）：充分松弛后初始条件被遗忘 → 假设二退化为假设一
- **非线性迟滞材料**（多平衡态）：即使充分松弛，稳态也依赖加载路径 → 假设二不退化

简言之，退化的前提是"每个恒定输入对应唯一平衡态"，而非 t→∞ 本身。

### 控制论基础：稳态与状态转移的数学定义

#### 状态空间模型的一般形式

离散时间状态空间模型：

```
x(k+1) = f(x(k), u(k))       状态转移方程
y(k)   = g(x(k), u(k))       输出方程
```

其中 x 是状态向量，u 是输入（动作），y 是输出（形状）。

#### 稳态 (Steady State)

稳态是指系统在**恒定输入** u* 下，所有瞬态衰减完毕，输出不再随时间变化：

```
x* = f(x*, u*)                平衡条件：x(k+1) = x(k) = x*
y* = g(x*, u*)                稳态输出
```

稳态存在的条件：系统渐近稳定（对线性系统即 A 的所有特征值 |λ_i| < 1）。

#### 线性系统的状态转移与稳态退化

对线性时不变系统 x(k+1) = Ax(k) + Bu(k)，在恒定输入 u* 下展开：

```
x(k) = A^k · x(0) + (I + A + A² + ... + A^(k-1)) · B · u*
     = A^k · x(0) + (I - A^k)(I - A)^(-1) · B · u*
```

当 k→∞ 且 |λ_i| < 1 时：

```
A^k → 0  （初始条件的影响消失）

x* = (I - A)^(-1) · B · u*   （稳态只取决于 u*，与 x(0) 无关）
```

**这就是"假设二退化为假设一"的数学基础**：A^k → 0 使得初始条件被遗忘。

#### 非线性迟滞系统：为什么退化不成立

以 Bouc-Wen 迟滞模型为例：

```
总恢复力: F = α·k·x + (1-α)·k·z

内变量 z 的演化方程:
  dz/dt = dx/dt · [A - |z|^n · (β + γ·sgn(dx/dt · z))]
```

其中 z 是迟滞内变量，β、γ 控制迟滞回线的形状：

- β > 0, γ > 0 时产生经典迟滞环
- 不同加载路径 → z 收敛到不同值 → 多个平衡态共存
- 同一个恒定输入 u*，从不同初始条件出发到达不同 x*

此时稳态输出 y* = g(x*(path), u*) **依赖于加载路径**，假设二不会退化。

#### 粘弹性材料的具体分析

**Kelvin-Voigt 模型**（线性粘弹性）：

```
σ = E·ε + η·dε/dt

恒定应力 σ* 下的应变演化:
  ε(t) = (σ*/E) · (1 - e^(-E·t/η))

稳态: ε* = σ*/E   （唯一，与加载路径无关）
松弛时间: τ = η/E

→ 单一平衡态，假设二退化为假设一 ✓
```

**标准线性固体模型**（Zener 模型）：

```
σ + τ_σ·dσ/dt = E_R·(ε + τ_ε·dε/dt)

稳态: ε* = σ*/E_R   （仍唯一）

→ 单一平衡态，假设二退化为假设一 ✓
```

**含塑性变形的粘弹性**：

```
ε_total = ε_elastic + ε_viscoelastic + ε_plastic

ε_plastic 在卸载后不恢复，永久改变参考构型
→ 不同加载路径产生不同 ε_plastic → 不同平衡态

→ 多平衡态，假设二不退化 ✗
```

#### 对 PyElastica 仿真的适用性

PyElastica 的 Cosserat rod 使用**线性阻尼**（类似 Kelvin-Voigt）：

```
内部力 + 线性阻尼力 + 外部力 = 0
```

理论上每个恒定输入对应唯一平衡态，假设二应退化。但需要注意：

1. **几何非线性**：大变形下线性本构也会产生路径相关的行为
2. **接触/摩擦**：如果仿真包含自接触，会引入多平衡态
3. **实际时间尺度**：即使理论上有唯一平衡态，如果动作变化频率 ≈ 松弛频率，系统始终处于瞬态，退化条件不满足

| 条件 | 是否退化 | 实际含义 |
|------|---------|---------|
| 线性粘弹性 + 充分松弛 | 是 | 准静态假设成立，静态映射足够 |
| 线性粘弹性 + 快速动作变化 | 不适用（未达稳态） | 需要状态转移建模 |
| 非线性迟滞 + 任意时间尺度 | 否 | 始终需要状态转移建模 |

**关键问题：我们的材料和实验条件，属于哪种情况？**

如果 PyElastica 的线性阻尼 Cosserat rod 对每个恒定输入有唯一平衡态，且动作间隔远大于松弛时间，则假设一成立。但如果：
- 数据采集时动作变化较快
- 实际控制中需要连续、快速的动作调整
- 材料松弛时间 > 动作间隔

那么假设二更合理，自回归/状态转移的建模方式更符合物理实际。

**从实验设计角度**：可以设计一个验证实验——对同一动作值，用不同的历史路径到达（递增到达 vs 递减到达），看最终形状是否相同。如果不同，则假设一失效，必须采用假设二。

---

## 一、问题的本质：欠定的状态推断

### 当前模型的信息流

```
action_history [a_{t-K}, ..., a_t] → FractionalMemory → cond → GRU(z) → pred_state_t
```

模型只有 action（输入），没有 state（输出反馈）。它试图从 action 历史推断当前状态。

### 为什么这是欠定的？

考虑两个物理场景：

| 场景 | 历史动作 | 当前 action | 当前物理状态 | 下一步状态 |
|------|---------|------------|------------|----------|
| A（充气） | [0.1, 0.2, 0.3, 0.4] | 0.4 | 正在弯曲中 | 继续弯曲 |
| B（放气） | [0.7, 0.6, 0.5, 0.4] | 0.4 | 正在回弹中 | 开始恢复 |

**同一个 action=0.4，但因为历史不同（充气 vs 放气），物理状态完全不同，下一步也完全不同。**

当前的 FractionalMemory 通过加权求和来区分这两种情况：
- 场景 A 的加权和 ≈ 0.25（偏小，因为历史值小）
- 场景 B 的加权和 ≈ 0.55（偏大，因为历史值大）

这提供了一定的区分能力，但加权求和本质上是一个**有损压缩**——把整个历史序列压成一个数，丢失了：
- 序列的**顺序信息**（先增后减 vs 先减后增）
- **变化的速率**（快速变化 vs 缓慢变化）
- **内部应力分布**（分布在不同 z 位置，无法用单一向量表示）

### positions[t-1] 能解决问题吗？

你提出了一个关键问题：仅用 `positions[t-1]` 能否准确表示迟滞状态？

**答案：不够充分，但是必要条件。**

考虑充放气的区分问题。核心难点在于：**仅凭一个位置值，无法判断机器人正在充气还是放气。**

**情况一：positions 不同时可以区分**

| 场景 | positions[t-1] | action[t] | positions[t] |
|------|---------------|-----------|-------------|
| A（充气到 0.4） | 弯曲度 30% | 0.4 | 弯曲度 35%（继续弯） |
| B（放气到 0.4） | 弯曲度 60% | 0.4 | 弯曲度 55%（开始恢复） |

此时 positions[t-1] 不同（30% vs 60%），可以隐式推断方向。但这只是因为两个路径还没到达同一位置。

**情况二：positions 相同时无法区分——这才是关键问题**

| 场景 | positions[t-1] | action[t] | 真实方向 | positions[t] |
|------|---------------|-----------|---------|-------------|
| C（经过 30%→40% 的充气中） | 弯曲度 40% | 0.4 | 正在充气 ↑ | 45%（继续弯） |
| D（经过 50%→40% 的放气中） | 弯曲度 40% | 0.4 | 正在放气 ↓ | 38%（继续恢复） |

同样的 positions[t-1]=40% 和 action[t]=0.4，但一个在充气（位置递增），一个在放气（位置递减）。**仅凭当前位置和当前动作，完全无法区分这两种情况。**

区分充放气需要知道**运动的趋势**（速度/动量的方向），而这只能来自历史信息：
- `positions[t-1] - positions[t-2]`（速度方向）
- 或完整的 action 历史序列（推断加载路径）

**因此，positions[t-1] + action[t] 的组合在一般情况下不足以区分充放气。** 只有在 positions 恰好不同时才能隐式区分，但这不是可靠的方法。

更完整的信息需求：

| 场景 | positions[t-1] | velocity | action[t] | positions[t] |
|------|---------------|----------|-----------|-------------|
| C（慢充气） | 40% | +1%/step | 0.4 | 41%（缓慢增长） |
| D（快充气） | 40% | +5%/step | 0.4 | 44%（快速增长，可能过冲） |
| E（放气经过 40%） | 40% | -2%/step | 0.4 | 39%（继续恢复） |

三者的 positions[t-1] 和 action[t] 完全相同，但因为 velocity（即历史运动趋势）不同，下一步状态完全不同。velocity 的符号区分充放气方向，velocity 的大小区分快慢。

### 结论：区分充放气需要历史运动信息

```
区分充气/放气至少需要：
  1. positions[t-1]     — 当前在哪（必要，但不足）
  2. velocity 或 action_history — 运动趋势/从哪来（必要，用于区分方向）
  3. action[t]          — 当前施力（必要）

仅 positions[t-1] + action[t] 不能可靠地区分充放气。
速度/动量信息（来自历史）是不可或缺的。
```

---

## 二、方案设计

### 方案 A：最小自回归（positions 反馈）

```python
class AutoregressiveSpatialModel(nn.Module):
    def forward(self, action_window, prev_skeleton=None):
        cond = self.temporal(action_window)    # action 历史 → 条件向量

        # 如果有前一步状态，用它初始化 GRU hidden state
        if prev_skeleton is not None:
            h = self.state_encoder(prev_skeleton)  # (B, N, 3) → (B, hidden)
        else:
            h = self.init_hidden(cond)              # 无状态时用 action 初始化

        # GRU 沿 Z 轴生成
        z_positions = self._get_z_positions(device)
        skeleton = []
        for i in range(self.n_nodes):
            z_emb = self.z_embed(z_positions[i])
            h = self.gru(cond + z_emb, h)
            skeleton.append(self.slice_head(h))

        return torch.stack(skeleton, dim=1)
```

**训练**（teacher forcing）：
```python
pred_t = model(action_window_t, prev_skeleton=gt_skeleton[t-1])
loss = MSE(pred_t, gt_skeleton[t])
```

**推理**（autoregressive）：
```python
pred_0 = model(action_window_0)          # 第一步无前驱
pred_1 = model(action_window_1, pred_0)   # 用上一步预测
pred_2 = model(action_window_2, pred_1)   # ...
```

### 方案 B：Delta 预测（预测变化量）

不直接预测绝对位置，而是预测**位移量**：

```python
class DeltaSpatialModel(nn.Module):
    def forward(self, action_window, prev_skeleton):
        cond = self.temporal(action_window)
        h = self.state_encoder(prev_skeleton)
        # ... GRU 沿 Z 轴 ...
        delta = self.slice_head(h)  # 预测位移量
        return prev_skeleton + delta
```

**优势**：
- 输出范围更小（位移 vs 绝对位置），更容易学习
- 天然保持连续性（不会跳变）
- 物理上更合理（状态 = 上一步 + 变化量）

**风险**：
- 误差累积：每步的 delta 误差会叠加
- 需要 Scheduled Sampling 避免训练/推理不一致

### 方案 C：混合方案（action 历史 + state 反馈 + velocity）

```python
class HybridSpatialModel(nn.Module):
    def forward(self, action_window, prev_skeleton=None, prev_prev_skeleton=None):
        cond = self.temporal(action_window)    # 时间上下文（迟滞信息）

        if prev_skeleton is not None:
            state_feat = self.state_encoder(prev_skeleton)

            if prev_prev_skeleton is not None:
                velocity = prev_skeleton - prev_prev_skeleton
                vel_feat = self.vel_encoder(velocity)
                h = state_feat + vel_feat
            else:
                h = state_feat
        else:
            h = self.init_hidden(cond)

        h = h + cond  # 条件注入
        # GRU 沿 Z 轴 ...
```

这是最完整的方案：action_history 提供迟滞上下文，prev_state 提供当前位置，velocity 提供动量。

---

## 三、迟滞建模的深入分析

### 迟滞回线的本质

```
       位置
        |
    B ←─┐     ┌─→ C（充气路径）
        |     |
    A ←─┘     └─→ D（放气路径）
        |
        └──────── 动作
```

同一条 action 值对应两个不同的位置（上支和下支）。

### 三种信息的迟滞区分能力

| 信息 | 能否区分充放气？ | 原因 |
|------|---------------|------|
| action_window 加权求和 | 部分 | 加权和可能不同，但有损压缩，不可靠 |
| positions[t-1] + action[t] | 不可靠 | positions 恰好不同时可以，相同时无法区分 |
| positions[t-1] + velocity | 可靠 | velocity 符号直接指示运动方向 |
| action_window 完整序列 | 最强 | 完整历史信息，可推断加载路径 |
| action_window（非线性编码如 GRU） | 强 | 保留序列顺序，隐式提取速度/方向信息 |

### 关于"加权求和 = 平均值"的担忧

你说得对——对整个窗口做加权求和本质上是在算一个**加权平均**，它无法区分：

```
序列 [0.1, 0.2, 0.3, 0.4] 和 [0.4, 0.3, 0.2, 0.1]
加权平均可能相同，但物理含义完全相反
```

**根本原因**：加权求和是**线性**操作，而迟滞是**路径依赖**的（非线性）。

解决方案有两个方向：
1. **非线性时间编码**：用 GRU/LSTM 代替加权求和（保留序列顺序）
2. **增加状态反馈**：用 positions[t-1] 提供位置信息（非线性的）

---

## 四、训练策略

### Teacher Forcing vs Scheduled Sampling

**Teacher Forcing**（简单版）：
- 训练时始终用 GT state_{t-1}
- 问题：训练时模型从未见过自己的预测误差，推理时误差累积

**Scheduled Sampling**（推荐）：
```python
p = min(1.0, epoch / warmup_epochs)
if random() < p:
    prev_state = predicted_state[t-1]
else:
    prev_state = gt_state[t-1]
```

### 序列级训练

当前训练是**逐帧独立**的。自回归需要**序列级训练**：
```python
for t in range(T):
    pred_t = model(action_window_t, prev_state)
    loss += MSE(pred_t, gt_skeleton_t)
    prev_state = pred_t.detach()  # 或 gt_skeleton[t]
```

---

## 五、实现路线

### 改动清单

1. **Dataset** (`src/data/dataset_spatial.py`)
   - `__getitem__` 额外返回 `positions[t-1]`（归一化后的 prev_skeleton）
   - 注意 t=0 时 prev_skeleton 为 zero-pad

2. **Model** (`src/models/model_spatial_sequence.py`)
   - `forward` 增加 `prev_skeleton` 参数
   - 新增 `state_encoder`: (B, N, 3) → (B, hidden)
   - 用 prev_skeleton 编码替代 `init_hidden(cond)`

3. **Trainer** (`src/training/trainer_unified.py`)
   - 添加序列级训练模式
   - 支持 Scheduled Sampling

4. **推理**
   - 逐步推理，每步把上一步预测结果喂回
   - 第一步无前驱时回退到 `init_hidden(cond)`

---

## 六、与 Gamma/Laguerre 的关系

两者解决不同层面的问题，可以组合：

| 问题层面 | Gamma/Laguerre | 自回归 |
|---------|---------------|--------|
| "何时响应" | 解决（延迟峰值权重） | 不直接解决 |
| "当前在哪" | 隐式（从 action 推断） | 显式（state 反馈） |
| "往哪走" | 隐式（action + 加权历史） | 显式（state + action） |

**最佳组合**：Gamma/Laguerre 编码器 + 自回归 state 反馈
- Gamma 核提供延迟建模的物理先验
- State 反馈提供当前位置的准确信息
- Action history 通过 Gamma 核提供迟滞方向上下文

---

## 七、先验实验：用互信息量化迟滞强度

> 在设计模型之前，先回答一个更基本的问题：我们的数据中到底有多少迟滞？

### 为什么需要这个实验

第〇节讨论了两种假设，但没有回答"到底该用哪种"。如果迟滞很弱（材料松弛快），假设一就够了，自回归是多余的。如果迟滞很强，假设二才合理。

**用互信息 I(S; H) 可以定量回答这个问题，不需要先设计模型。**

### I(S; H) 的物理含义

I(S; H) = 当前形状 S 包含多少关于加载历史 H 的信息。

- **无迟滞的理想弹性体**：形状只由当前动作决定 → I(S; H) ≈ 0
- **强迟滞的粘弹性体**：形状编码了过去的加载路径 → I(S; H) 较大

因此 I(S; H) 本质上量化了"材料有多少迟滞"。

### 具体实验设计

**实验 A：记忆窗口 — I(形状; k 步前的动作)**

```python
for k in range(1, 50):
    # 固定其他条件，只看第 k 步前的动作能从形状中解码多少
    I_k = mutual_information(shapes, actions[:, t-k])
# 画 I_k vs k 的衰减曲线
```

- 如果 I_k 在 k=3 后就 ≈ 0 → 记忆只有 3 步，模型只需 3 步历史
- 如果 I_k 缓慢衰减 → 需要长历史窗口

**实验 B：顺序可辨识性 — 反序加载能否区分**

```python
# 序列 A: [0.1, 0.2, 0.3, 0.4]  和 C: [0.4, 0.3, 0.2, 0.1]
# 最终驱动值相同（0.4），但到达路径相反
shape_A = simulate([0.1, 0.2, 0.3, 0.4])
shape_C = simulate([0.4, 0.3, 0.2, 0.1])
# 能区分 shape_A 和 shape_C 吗？
```

- 如果形状有显著差异 → 假设一失效，必须用假设二
- 如果形状几乎相同 → 假设一成立（至少对这个时间尺度）

**实验 C：速率依赖性 — 同一驱动值、不同加载速率**

```python
for rate in [0.01, 0.05, 0.1, 0.5]:
    shape = simulate(ramp_to(target=0.4, rate=rate))
# 不同速率得到的形状一样吗？
```

### 如何指导模型设计

| 实验结果 | 含义 | 对模型设计的指导 |
|---------|------|----------------|
| 记忆窗口 < 5 步 | 迟滞弱，短期记忆 | 短窗口 EMA 即可，无需自回归 |
| 记忆窗口 > 20 步 | 迟滞强，长期记忆 | 需要 Gamma/Laguerre 长核 + 自回归 |
| 顺序可区分 | 路径依赖显著 | 必须用非线性编码（GRU/LSTM），不能线性加权 |
| 速率依赖 | 加载速率影响形状 | 训练数据必须包含变速率，模型需编码速率信息 |
| 以上都不显著 | 迟滞可忽略 | 假设一成立，当前方法没问题 |

### 在 PyElastica 中的实现

这个实验在 PyElastica 中非常容易做：
1. 修改 `collect.py` 生成反序/变速率加载序列
2. 仿真得到形状数据
3. 用简单的 sklearn 分类器/回归器做解码
4. 用 binning 方法或 MINE 估计互信息

**工作量估计**：1-2 天即可完成所有实验，获得定量结论。
