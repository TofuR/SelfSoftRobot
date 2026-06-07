# 方向：自回归状态动力学——从"猜状态"到"推状态"

> 状态：概念阶段
> 解决问题：模型只有 action 输入，没有物理状态反馈，导致预测超前
> 核心思想：将前一步的物理状态（或预测状态）作为输入，让模型学习状态转移而非状态推断

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

考虑上面的例子：

| 场景 | positions[t-1] | action[t] | positions[t] |
|------|---------------|-----------|-------------|
| A（充气到 0.4） | 弯曲度 30% | 0.4 | 弯曲度 35%（继续弯） |
| B（放气到 0.4） | 弯曲度 60% | 0.4 | 弯曲度 55%（开始恢复） |

同样的 action=0.4：
- 场景 A：positions[t-1]=30% → positions[t]=35%（增加）
- 场景 B：positions[t-1]=60% → positions[t]=55%（减少）

**positions[t-1] 提供了"当前在哪"的信息，结合 action[t] 提供了"往哪走"的信息。两者的组合足以区分充气/放气——因为即使 action 相同，positions 不同。**

但是，positions[t-1] 仍然丢失了**速度/动量**信息。考虑：

| 场景 | positions[t-1] | positions[t-2] | action[t] |
|------|---------------|---------------|-----------|
| C（慢充气） | 40% | 39% | 0.4 |
| D（快充气） | 40% | 35% | 0.4 |

同样的 positions[t-1]=40%，但 C 在缓慢接近，D 在快速接近。它们的**下一步**可能不同——D 可能因为惯性"过冲"。

### 结论：三层信息缺一不可

```
完整状态推断需要：
  1. positions[t-1]  — 当前在哪（必要）
  2. action_history  — 从哪来（提供方向/历史上下文）
  3. velocity/加速度 — 多快在变化（可选但有益）
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
| action_window 加权求和 | 部分 | 加权和不同，但有损压缩 |
| positions[t-1] + action[t] | 能 | 不同路径到达不同位置 |
| positions[t-1] + velocity | 更好 | velocity 符号直接指示方向 |
| action_window 完整序列 | 最强 | 完整历史信息 |

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
