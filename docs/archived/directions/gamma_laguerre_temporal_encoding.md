# 方向：Gamma/Laguerre 时序编码——带延迟峰值的记忆核

> 状态：概念阶段
> 解决问题：模型预测超前于真实响应（迟滞延迟）
> 核心思想：用 Gamma 分布权重替代指数/幂律权重，让记忆核具有"先升后降"的延迟峰值特性

---

## 一、问题：当前权重为什么造成"超前预测"？

### 当前的 FractionalMemory 权重形态

GL 权重（分数阶）和 EMA 权重（指数衰减）都是**单调递减**的：

```
EMA (β=0.1):       [1.00, 0.90, 0.81, 0.73, 0.66, ...]   ← 立即最大，逐步衰减
GL (α=0.37):       [1.00, -0.37, 0.06, -0.02, ...]        ← 第0步最强调
```

**问题**：当前时刻的 action 获得最大权重。这意味着模型倾向于认为"当前 action 立刻决定当前状态"。

### 软体材料的真实响应

粘弹性材料对一个阶跃输入的响应是 **S 形**的：

```
力矩阶跃 ↑
  |
  |        ___________  ← 稳态
  |      /
  |    /    ← 逐渐上升（延迟）
  |  /
  |/
  |___________________ 时间
  0  2  4  6  8  10 (帧)
```

即：施加 action 后，**不是第0帧响应最大**，而是几帧后才达到峰值响应。

### "单调递减权重" vs "S 形响应"的矛盾

```
模型假设：  w_0 最大 → 当前 action 立即生效
物理真实：  t=0 几乎无响应 → t=3-5 才是峰值响应
```

这个不匹配导致模型**系统性地超前预测**——它认为 action 已经完全生效，但实际物理响应还在爬坡。

---

## 二、方案：Gamma 分布权重

### 定义

$$w_t = \frac{t^{k-1}}{(k-1)!} \lambda^t, \quad t = 0, 1, 2, \ldots$$

参数：
- $k$（阶次，正实数）：控制峰值出现的时间。$k$ 越大，峰值越晚
- $\lambda$（衰减率，0 < λ < 1）：控制记忆持续长度

### 不同 k 值的权重形态

```
k=1: [1, λ, λ², λ³, ...]           ← 指数衰减（退化为 EMA）
k=2: [0, λ, 2λ², 3λ³, ...]         ← 从0升起，先升后降
k=3: [0, 0, λ², 3λ³, 6λ⁴, ...]     ← 更晚的峰值
k=4: [0, 0, 0, λ³, ...]             ← 更更晚
```

注意：$t=0$ 时 $w_0 = 0$（当 $k \geq 2$），这正是我们想要的——当前 action 不应该有最大影响。

### 物理对应

| k 值 | 物理含义 |
|------|---------|
| k=1 | 标准线性固体（指数松弛）= 现有 EMA |
| k=2 | Kelvin-Voigt + 惯性延迟 |
| k=3-4 | 高阶粘弹性链（多级传播延迟） |
| k 组合 | 多阶粘弹性模型的叠加 |

### 递推计算（数值稳定版）

使用 log 空间计算避免溢出：

```python
def compute_gamma_weights(k, lam, length):
    t = torch.arange(length, dtype=torch.float32)
    log_w = (k - 1) * torch.log(t.clamp(min=1e-10)) \
            - torch.lgamma(k) \
            + t * torch.log(lam)
    if k > 1.5:
        log_w[0] = -100.0
    w = torch.exp(log_w - log_w.max())
    return w / (w.abs().sum() + 1e-8)
```

---

## 三、多尺度 Gamma 记忆核

类比 MultiScaleEMA 用多个 β，用多组 (k, λ) 捕获不同延迟特征：

```python
class GammaLaguerreMemory(nn.Module):
    """Gamma/Laguerre 时序编码器。"""
    def __init__(self, action_dim, n_kernels=6, window_size=40, hidden_dim=128):
        super().__init__()
        self.n_kernels = n_kernels

        # 可学习参数
        init_ks = torch.linspace(1.0, 6.0, n_kernels)
        self.k_logits = nn.Parameter(init_ks)          # softplus → k ≥ 1

        init_lambdas = torch.linspace(0.95, 0.7, n_kernels)
        self.log_lambdas = nn.Parameter(torch.log(init_lambdas))  # sigmoid → (0, 1)

        self.kernel_weights = nn.Parameter(torch.ones(n_kernels))

        mlp_input = n_kernels * action_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _compute_weights(self, k, lam, length):
        t = torch.arange(length, dtype=torch.float32)
        log_w = (k - 1) * torch.log(t.clamp(min=1e-10)) \
                - torch.lgamma(k) + t * torch.log(lam)
        if k > 1.5:
            log_w[0] = -100.0
        w = torch.exp(log_w - log_w.max())
        return w / (w.abs().sum() + 1e-8)

    def forward(self, action_window):
        B, K, D = action_window.shape
        ks = F.softplus(self.k_logits)
        lambdas = torch.sigmoid(self.log_lambdas) * 0.9 + 0.05

        features = []
        for i in range(self.n_kernels):
            w = self._compute_weights(ks[i], lambdas[i], K)
            feat = torch.einsum('k,bkd->bd', w, action_window)
            features.append(feat * self.kernel_weights[i])

        gamma_flat = torch.cat(features, dim=-1)
        current_action = action_window[:, -1, :]
        velocity = action_window[:, -1, :] - action_window[:, -2, :] if K >= 2 \
                   else torch.zeros_like(current_action)
        return self.state_mlp(torch.cat([gamma_flat, current_action, velocity], dim=-1))
```

---

## 四、与 EMA / FractionalMemory 的对比

| 维度 | EMA | FractionalMemory (GL) | Gamma/Laguerre |
|------|-----|----------------------|----------------|
| 权重形态 | 单调衰减 | 单调衰减（有负项） | **先升后降** |
| 峰值位置 | t=0 | t=0 | **可学习延迟** |
| 物理对应 | 标准线性固体 | 分数阶粘弹性 | **多阶延迟系统** |
| 延迟建模 | 无 | 隐式（通过负项） | **显式（k 控制峰值延迟）** |
| 超前预测倾向 | 强 | 中 | **弱** |
| 可解释性 | 时间常数 τ | 分数阶 α | 阶次 k + 衰减率 λ |

### 关键优势

Gamma 核天然匹配软体材料的延迟响应特性：
- k=1：瞬时响应（弹性分量）
- k=2-3：中等延迟（粘弹性主体）
- k=4-6：长延迟（深层次结构响应）

模型可以学习哪些延迟尺度是重要的，而不需要从指数衰减去"拼凑"延迟效果。

---

## 五、与自回归方案的对比

| 维度 | Gamma/Laguerre | 自回归 (state feedback) |
|------|---------------|----------------------|
| 是否需要逐步推理 | 否（单次前向） | 是（序列依赖） |
| 误差累积 | 无 | 有（推理时用预测值） |
| 物理白盒性 | 高（权重可解释） | 低（黑盒修正） |
| 迟滞建模能力 | 通过窗口加权隐式捕获 | 通过 state 显式捕获 |
| 实现复杂度 | 低（只改编码器） | 中（改数据+模型+推理） |
| 充放气区分 | 依赖窗口加权 | 依赖 state+action |

**推荐**：两者不冲突，可以组合。先用 Gamma 替换 EMA/GL 解决"延迟峰值"问题，再视效果决定是否加入自回归。

---

## 六、实验验证计划

### 阶段 1：可视化验证

训练后查看学到的 (k, λ) 值和对应权重曲线。期望至少 2-3 个核的峰值不在 t=0。

### 阶段 2：替换 FractionalMemory

在 `model_spatial_sequence.py` 和 `model_pc_spatial.py` 中替换编码器。

对比指标：
- 预测 vs GT 的时间相关系数
- 不同 action 转换处的瞬时误差
- 学到的 (k, λ) 值的物理合理性

### 阶段 3：消融实验

| 变体 | 说明 |
|------|------|
| EMA | 基线（当前） |
| FractionalMemory (GL) | 幂律衰减 |
| Gamma/Laguerre | 延迟峰值 |
| Gamma + 自回归 | 组合方案 |

---

## 七、风险与备选

| 风险 | 概率 | 应对 |
|------|------|------|
| k 学到接近 1（退化为 EMA） | 中 | 正则化鼓励 k > 1.5 |
| λ 学到不合理值 | 低 | 约束 λ ∈ (0.3, 0.95) |
| 不比 FractionalMemory 好 | 中 | 保留 FractionalMemory 作为备选 |
| 数值稳定性（大 k 时权重溢出） | 中 | log 空间计算 + 归一化 |

---

## 八、关键参考文献

1. **Wahlberg 1991** — "System Identification Using Laguerre Models" — Laguerre 基础网络在系统辨识中的经典论文
2. **Campello et al. 2004** — "Laguerre Filters in System Identification" — 工程控制论中的 Laguerre 滤波器
3. **Mainardi 2010** — "Fractional Calculus and Waves in Linear Viscoelasticity" — 分数阶粘弹性框架
4. **Tschoegl 1989** — "The Phenomenological Theory of Linear Viscoelastic Behavior" — 粘弹性本构中的 Gamma/Erlang 分布
