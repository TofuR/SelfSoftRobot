# 时序编码器对比

## 接口规范

所有编码器遵循统一接口：

```python
class TemporalEncoder(nn.Module):
    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128)
    def forward(self, action_window: (B, K, D)) -> (B, hidden_dim)
    @property decays -> (n_scales,) tensor  # TemporalMixin logging
    def compute_smoothness(self, aw_t, aw_t1) -> scalar
```

## 编码器列表

### 加权求和类（顺序不敏感）

| 编码器 | CLI 名称 | 核心机制 | 物理类比 | n_scales 含义 |
|--------|---------|---------|---------|-------------|
| MultiScaleEMA | `ema` | 多尺度指数衰减 | 多松弛时间常数 | 衰减率个数 |
| FractionalMemory | `fractional` | 分数阶 Grünwald-Letnikov 权重 | 分数阶微积分记忆核 | 阶次个数 |
| GammaLaguerre | `gamma` | Gamma 分布核（有延迟峰值） | 粘弹性延迟响应 | 核个数 |

**共同局限**：都是加权求和（线性操作），丢失序列顺序信息。`[0.1→0.4]` 和 `[0.4→0.1]` 可能产生相同输出。

### 顺序敏感类

| 编码器 | CLI 名称 | 核心机制 | 物理类比 | n_scales 含义 | 复杂度 |
|--------|---------|---------|---------|-------------|--------|
| TemporalGRU | `gru` | GRU 沿时间轴扫描 | 内部应力状态逐步演化 | 隐状态维度=n_scales×D | O(K) |
| TemporalTransformer | `transformer` | 自注意力 + CLS 聚合 | 全历史全局关联 | model_dim=n_scales×D | O(K²) |
| TemporalTCN | `tcn` | 因果膨胀 1D 卷积 | 多尺度局部时间模式 | 通道数=n_scales×D | O(K) |

## 用法

```bash
# EMA（最简单，适合迟滞弱的数据）
python scripts/training/train_spatial_sequence.py --encoder ema

# Gamma（有延迟峰值，物理先验更强）
python scripts/training/train_spatial_sequence.py --encoder gamma --n_scales 6

# GRU（顺序敏感，适中复杂度）
python scripts/training/train_spatial_sequence.py --encoder gru --n_scales 8

# Transformer（最强顺序保持，适合长窗口）
python scripts/training/train_spatial_sequence.py --encoder transformer --n_scales 16

# TCN（顺序敏感 + 并行 + 多尺度感受野）
python scripts/training/train_spatial_sequence.py --encoder tcn --n_scales 8
```

`--n_scales` 控制编码器容量。对于 Transformer/TCN/GRU，建议用较大的值（8-16），因为它们的特征维度 = n_scales × action_dim。

## 内部结构对比

```
加权求和类：
  action_window → 多尺度加权求和 → features → cat(features, action, velocity) → MLP → physics_state

TemporalGRU：
  action_window → GRU(K步) → 最后隐状态 → cat(h_K, action, velocity) → MLP → physics_state

TemporalTransformer：
  action_window → 线性投影 → [CLS]+tokens+位置编码 → 2层自注意力 → CLS输出 → cat(cls, action, velocity) → MLP → physics_state

TemporalTCN：
  action_window → 线性投影 → 因果膨胀卷积(dilation=[1,2,4,...]) → 最后时间步 → cat(feat, action, velocity) → MLP → physics_state
```

## 选择建议

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速基线实验 | `ema` | 最简单，参数少 |
| 已知有粘弹性延迟 | `gamma` | 延迟峰值是物理先验 |
| 怀疑顺序信息重要 | `gru` | 顺序敏感，适中复杂度 |
| 需要最强的顺序建模 | `transformer` | 全局注意力，上限基线 |
| 长窗口 + 并行训练 | `tcn` | O(K) 且可并行 |
