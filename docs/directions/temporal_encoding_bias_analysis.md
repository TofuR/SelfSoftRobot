# 方向：当前 Action 偏置分析与修正

> 状态：概念阶段
> 解决问题：FractionalMemory 拼接 current_action + velocity 可能导致模型对当前动作过度响应
> 核心思想：分析并修正模型输入中对"当前 action"的偏置

---

## 一、问题发现

### 当前 FractionalMemory 的特征拼接

`src/encoders/fractional_memory.py:121-128`：

```python
frac_flat = torch.cat(frac_features, dim=-1)          # (B, n_orders * D)
current_action = action_window[:, -1, :]               # (B, D)
velocity = action_window[:, -1, :] - action_window[:, -2, :]  # (B, D)
features = torch.cat([frac_flat, current_action, velocity], dim=-1)
return self.state_mlp(features)
```

模型最终看到的特征 = `[GL加权特征, 当前action, 速度]`。

### 为什么 current_action 会造成偏置？

**信息冗余**：`current_action` 已经包含在 `frac_flat` 中了（GL 权重的 w_0 最大）。额外拼接等于**双倍强调**当前动作。

**MLP 的短路行为**：`state_mlp` 是一个两层 MLP。如果 `current_action` 的几个维度直接决定了大部分输出，MLP 会倾向于走这条最短路径：

```
current_action ──→ MLP ──→ output（直接映射 action → 稳态位置）
         ↑
      这条路径太短了！GL加权特征几乎被忽略
```

**结果**：模型学到 `action_now → 稳态位置` 的直接映射，跳过了迟滞/延迟的建模。

### 验证方法

```python
# 消融实验：去掉 current_action 和 velocity
# 只保留 frac_flat，看性能变化
features = frac_flat  # 不拼接 current_action 和 velocity
```

如果去掉后性能大幅下降 → 说明模型严重依赖 current_action，GL 加权特征没起作用。
如果去掉后性能不变或更好 → 说明 GL 加权特征已经足够，current_action 是多余的。

---

## 二、问题本质：时序编码器的"绕过"现象

### 期望 vs 实际

```
期望的信息流：
  action_history → GL加权 → 捕获迟滞/延迟 → 条件向量 → 预测

实际可能的信息流：
  action_history → GL加权 ──→ 被忽略
  current_action ──────────→ MLP 直接映射 → 预测（绕过迟滞建模）
```

这类似于 ResNet 中残差连接太强导致主干网络不被训练的问题。

### 从 loss 角度理解

MSE loss 会选择**最容易优化的路径**。如果 `current_action → 稳态位置` 能覆盖 80% 的方差（因为大部分时间系统接近稳态），模型会优先学这个映射。剩下 20% 的瞬态/迟滞效应被忽略——但正是这 20% 造成了你看到的"超前预测"。

---

## 三、修正方案

### 方案 A：完全移除 current_action（激进）

```python
def forward(self, action_window):
    B, K, D = action_window.shape
    # ... GL 加权计算 ...
    frac_flat = torch.cat(frac_features, dim=-1)
    return self.state_mlp(frac_flat)  # 只用 GL 特征
```

**优点**：强迫模型从 GL 加权特征中提取所有信息。
**缺点**：可能丢失精度（current_action 确实有信息量）。

### 方案 B：降低 current_action 的权重（温和）

```python
# 在拼接时乘以一个可学习的缩放因子
self.action_gate = nn.Parameter(torch.tensor(0.1))  # 初始小值

def forward(self, action_window):
    # ...
    current_action = action_window[:, -1, :] * torch.sigmoid(self.action_gate)
    velocity = ... * torch.sigmoid(self.vel_gate)
    features = torch.cat([frac_flat, current_action, velocity], dim=-1)
    return self.state_mlp(features)
```

让网络自己学习需要多少 current_action 信息。

### 方案 C：GL 权重化后提取 current（推荐）

不直接拼 raw action，而是让 GL 加权特征**承担所有工作**。如果需要 current action 信息，调整 GL 权重使其更强调最近帧（比如 α→0 时 GL 退化为只看最后一帧）：

```python
# 不拼接 current_action，完全依赖 GL 加权
# 但 GL 的 α 参数会学习是否需要强调最近帧
def forward(self, action_window):
    # ... GL 加权 ...
    frac_flat = torch.cat(frac_features, dim=-1)
    # 只拼接 velocity（不拼接 current_action）
    velocity = action_window[:, -1, :] - action_window[:, -2, :] if K >= 2 \
               else torch.zeros(B, D, device=action_window.device)
    features = torch.cat([frac_flat, velocity], dim=-1)
    return self.state_mlp(features)
```

**理由**：
- velocity 是**一阶差分**，包含方向信息（充/放气），这是 GL 加权无法直接表达的
- current_action 已经隐含在 GL 加权中（w_0 权重）
- 保留 velocity 但去掉 current_action，避免信息冗余

### 方案 D：多尺度 EMA 中的类似问题检查

`src/encoders/multi_scale_ema.py` 也有同样的拼接模式。如果确认这是偏置来源，两种编码器都需要修正。

---

## 四、更深层思考：current_action 的物理角色

### current_action 代表什么？

在仿真数据中，`action = torque`（扭矩）。扭矩是**输入**，不是**状态**。

在经典力学中：
- 状态 = (位置, 速度)
- 输入 = 外力/扭矩
- 动力学 = state_{t+1} = f(state_t, input_t)

模型应该从 **input_history** 推断 **state_t**。直接把 current_action 拼进特征，等价于让模型部分跳过了"从输入推断状态"这个过程。

### 但是 current_action 也有用

current_action 的作用是告诉模型"**当前的驱动目标是什么**"。GL 加权特征给出的是"**过去的历史上下文**"。两者组合才能回答"**在给定历史下，当前驱动的效果**"。

关键是如何**平衡**两者的贡献，而不是让 current_action 主导。

---

## 五、实验验证

### 消融矩阵

| 变体 | frac_flat | current_action | velocity | 预期 |
|------|-----------|---------------|----------|------|
| 当前 | ✓ | ✓ | ✓ | 基线（有超前） |
| 无 current | ✓ | ✗ | ✓ | 减少超前，但可能欠拟合 |
| 无 current 无 vel | ✓ | ✗ | ✗ | 最强约束，看 GL 能否独立工作 |
| 仅 current | ✗ | ✓ | ✓ | 无迟滞建模，最大超前 |
| current 乘 0.1 | ✓ | ✓ (×0.1) | ✓ | 温和修正 |

### 关键指标

1. **时间相关系数**：预测和 GT 的帧级别相关系数（消除平移后）
2. **超前帧数**：用互相关函数测量预测相对 GT 的时移
3. **α 学习值**：去掉 current_action 后，GL 参数是否变得更合理（更大 → 更强调远期记忆）

---

## 六、总结

| 方案 | 改动量 | 风险 | 推荐优先级 |
|------|--------|------|-----------|
| 消融验证 | 无改动，只跑实验 | 无 | **最高**（先验证假设） |
| 去掉 current_action | 改 1 行 | 可能欠拟合 | 高 |
| 可学习缩放 | 改 3 行 | 低 | 中 |
| 只保留 velocity | 改 2 行 | 低 | 中 |

**建议**：先做消融实验确认 current_action 是否真的是偏置来源，再决定用哪个修正方案。这个改动成本极低（改几行代码），如果确认有效，可以和 Gamma/Laguerre 或自回归方案叠加。
