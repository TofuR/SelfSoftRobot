# 方向：动作窗口降采样——消除冗余重复值

> 状态：待实现
> 优先级：中
> 关联：所有时序编码器、PROJECT_HELP.md 4.0 节

---

## 一、问题

### 当前数据中的 action 重复

PyElastica 采集参数：
- `steps_per_action = 500`（每个动作值保持 500 步 = 50ms）
- `record_interval = 50`（每 50 步记录一帧 = 5ms）
- **每个动作值产生 10 帧记录**

因此 action_window 中大量重复：

```
window_size=40 时的实际内容:
[a1, a1, a1, a1, a1, a1, a1, a1, a1, a1,   ← 10 帧同一个值
 a2, a2, a2, a2, a2, a2, a2, a2, a2, a2,
 a3, a3, a3, a3, a3, a3, a3, a3, a3, a3,
 a4, a4, a4, a4, a4, a4, a4, a4, a4, a4]
                     ↑
          仅 4 个不同值，75% 冗余
```

验证：`data/seq_rz_c2_sk/` 中 500 帧 = 50 个唯一动作对，10x 冗余。

### 对时序编码器的影响

**对所有编码器类型**（EMA、Gamma、GRU、Transformer、TCN）都有影响：

| 编码器 | 影响 |
|--------|------|
| EMA/Gamma | 10 个相同值的加权和 = 1 个值 × 等效权重，计算浪费 |
| GRU | 40 步展开中 36 步是"空转"，门控在重复值上无意义 |
| Transformer | 注意力矩阵 90% 冗余，自注意力在重复 token 上无信息增益 |
| TCN | 卷积核在重复值上空卷，感受野覆盖的有效信息只有 4 个值 |

**核心问题**：编码器的有效输入序列长度 = `window_size / 重复倍数` = 40/10 = **4 个值**，而不是 40。

---

## 二、解决思路

### 方案：动作降采样

在 dataset 的 `_get_action_window` 中，对 action 序列按 `stride = steps_per_action / record_interval = 10` 降采样。

**降采样前**（window_size=40, 4 个不同值）：
```
[a1×10, a2×10, a3×10, a4×10]  → 40 步，4 个有效值
```

**降采样后**（stride=10, 同样 40 步，但覆盖 40 个不同值）：
```
[a_{t-39}, a_{t-38}, ..., a_{t-1}, a_t]  → 40 步，40 个有效值
```

或者用更小的 window_size 覆盖同样的历史范围：
- 降采样后 window_size=4 就覆盖了和之前 window_size=40 相同的历史
- window_size=40 则覆盖了 10 倍长的历史

### 所有编码器都适用

降采样对所有编码器类型都有益，无需区分处理：

| 编码器 | 降采样后的改善 |
|--------|--------------|
| EMA/Gamma | 相同：加权求和对唯一值和重复值的等价加权结果一致 |
| GRU | 有效序列长度 ×10，门控在每步都有新信息 |
| Transformer | 注意力矩阵全部有效，无冗余 |
| TCN | 卷积感受野覆盖有效信息 |

### 实现方式

在 dataset 中添加 stride 参数：

```python
def _get_action_window(self, data, t):
    start = t - self.seq_len * self.stride + 1
    end = t + 1
    raw = data['actions'][max(0, start):end]
    # 降采样：从末尾取 seq_len 个点（保留最近的信息）
    indices = np.arange(len(raw) - 1, -1, -self.stride)[:self.seq_len][::-1]
    sampled = raw[indices]

    if len(sampled) < self.seq_len:
        pad = np.zeros((self.seq_len - len(sampled), self.action_dim))
        sampled = np.concatenate([pad, sampled])
    return sampled
```

CLI 参数：
```bash
python scripts/training/train_spatial_sequence.py --encoder gamma --action_stride 10
```

### 需要注意的问题

1. **stride 的计算**：`stride = steps_per_action / record_interval`，需要和数据采集参数匹配
2. **边界处理**：靠近序列开头时，可用帧数不够 window_size × stride，需要正确 zero-pad
3. **smooth loss**：`compute_smoothness` 需要相邻帧的 action_window，降采样后两帧的 action_window 变化更大，smooth loss 会自然变大（可能需要调整权重）
4. **已有模型兼容**：stride=1 时退化为当前行为，不影响已有 checkpoint

---

## 三、预期效果

| 指标 | 降采样前 | 降采样后 |
|------|---------|---------|
| 有效序列长度 | 4 | 40（或相同历史下 =4 但无冗余） |
| 编码器计算量 | 高（冗余计算） | 低（每步都有信息） |
| 历史覆盖范围 | 4 个动作值 | 40 个动作值（同 window_size） |
| 速率编码 | 隐式（重复次数间接表示） | 显式（每步不同的值直接表示） |

---

## 四、与其他方向的关系

- **迟滞信息容量实验** (`hysteresis_information_capacity.md`)：降采样后有效序列更长，互信息实验结果可能变化
- **状态转移主线** ([13](13_closed_loop_state_transition.md)/[14](14_gt_observed_transition.md)/[15](15_open_loop_windowed_transition.md)，原 01 已归档)：更长的有效历史窗口 → 更好的迟滞建模
