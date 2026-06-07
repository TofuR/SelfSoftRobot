# 方向灵感：分数阶记忆核替代 EMA 建模迟滞

> **状态：已实现 ✅**
> 来源：从软体材料物理本质出发推导出的迟滞建模方法
> 与 spatial_sequence_generation、topology_guided_residual_flow 可组合
>
> ## 实现位置
> - 编码器：`src/encoders/fractional_memory.py` — `FractionalMemory` 类
> - 使用模型：`src/models/model_spatial_sequence.py`、`src/models/model_pc_spatial.py`
> - 训练脚本：`scripts/training/train_spatial_sequence.py`、`scripts/training/train_pc_spatial.py`
> - 数据集：`src/data/dataset_spatial.py`
> - PROJECT_HELP：已记录在管线表中（"分数阶记忆"行）

---

## 一、动机：EMA 为什么不是最优选择？

### 当前方案：多尺度 EMA

```python
y_t = Σ_i w_i × (Σ_k (1-β_i)^k × x_{t-k})
```

多个指数衰减核叠加，用不同时间尺度捕获不同频率的历史信息。

### EMA 的问题

1. **指数衰减核不是软体材料真正的记忆核**：聚合物/硅胶的粘弹性实验大量表明记忆核是**幂律衰减** $G(t) \propto t^{-\alpha}$，而非指数衰减 $G(t) \propto e^{-t/\tau}$
2. **需要很多尺度才能拟合幂律行为**：用指数函数逼近幂律函数效率低，导致参数多、拟合差
3. **缺乏物理根基**：EMA 是信号处理工具，不是材料力学模型

### 软体材料的粘弹性物理

材料科学中，粘弹性本构关系：

$$\sigma(t) = \int_0^t G(t-\tau) \dot{\varepsilon}(\tau) d\tau$$

其中 $G(t-\tau)$ 是记忆核（松弛模量）。实验测得软材料的松弛模量：

| 材料 | 记忆核形状 | 参考文献 |
|------|----------|---------|
| PDMS 硅胶 | 幂律 $t^{-0.35}$ | Sorichetti et al. 2021 |
| 水凝胶 | 幂律 $t^{-0.5}$ | Cai et al. 2020 |
| 橡胶 | 幂律 + 指数尾巴 | STL 模型 |
| Cable-driven 机构 | 摩擦 + 粘弹性耦合 | Renda et al. 2018 |

**关键发现**：大多数软材料的记忆核是**幂律衰减**的，不是指数衰减的。

---

## 二、分数阶微积分：幂律记忆的数学框架

### 分数阶导数定义（Caputo）

$$D^\alpha f(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{f'(\tau)}{(t-\tau)^\alpha} d\tau, \quad 0 < \alpha < 1$$

- $\alpha = 0$：无记忆（纯弹性）
- $\alpha = 1$：完全记忆（纯粘性）
- $0 < \alpha < 1$：中间态（粘弹性，即软体材料的状态）

### Grünwald-Letnikov 离散化

将分数阶导数离散化为加权求和：

$$D^\alpha f(t_n) \approx \frac{1}{h^\alpha} \sum_{k=0}^{n} w_k f(t_{n-k})$$

权重递推公式：

```
w_0 = 1
w_k = w_{k-1} × (k - 1 - α) / k    for k ≥ 1
```

展开前几项：
```
w_0 = 1
w_1 = -α
w_2 = α(α-1) / 2       ≈ 0.06 (α=0.37)
w_3 = -α(α-1)(α-2) / 6 ≈ -0.025
...
```

### 与 EMA 权重的对比

```
EMA (β=0.1):            [1.00, 0.90, 0.81, 0.73, 0.66, 0.59, ...]  # 指数衰减
Fractional (α=0.37):    [1.00, -0.37, 0.06, -0.02, ...]              # 幂律衰减，有负项
```

**关键区别**：分数阶权重有**负项**，这意味着它不只是"衰减记忆"，还有**回弹效应**——这在物理上对应材料的恢复行为。

### 长程记忆对比

```
         记忆权重（归一化）
k=0      k=5      k=10     k=20     k=50
EMA β=0.1:      1.00     0.59     0.35     0.12     0.005   ← 指数衰减，快速遗忘
EMA β=0.01:     1.00     0.95     0.90     0.82     0.61    ← 需要很小的 β 才能记住远期
分数阶 α=0.37:  1.00     0.18     0.11     0.07     0.04    ← 幂律衰减，缓慢遗忘
```

分数阶在 k=50 时仍有显著权重 0.04，而 EMA β=0.1 已经是 0.005。

---

## 三、具体实现方案

### 方案 A：可学习分数阶记忆核（推荐先试）

```python
class FractionalMemory(nn.Module):
    """
    分数阶记忆模块：用幂律核替代指数核

    物理对应：软体材料的分数阶粘弹性
    参数：α ∈ (0,1) 控制记忆强度，projection 控制维度映射
    """
    def __init__(self, input_dim, hidden_dim, memory_length=50):
        super().__init__()
        self.memory_length = memory_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 唯一的记忆参数：α 控制记忆衰减速率
        self.log_alpha = nn.Parameter(torch.tensor(-0.5))  # sigmoid → α ≈ 0.37

        # 维度映射
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _compute_weights(self):
        """Grünwald-Letnikov 权重递推计算"""
        alpha = torch.sigmoid(self.log_alpha)  # 保证 0 < α < 1
        weights = torch.ones(self.memory_length, device=alpha.device)
        for k in range(1, self.memory_length):
            weights[k] = weights[k - 1] * (k - 1 - alpha) / k
        return weights

    def forward(self, action_history):
        """
        Args:
            action_history: (B, T, D) 最近 T 步的驱动历史
        Returns:
            (B, hidden_dim) 记忆编码
        """
        w = self._compute_weights()                          # (T,)
        weighted = action_history * w.view(1, -1, 1)         # (B, T, D)
        memory = weighted.sum(dim=1)                          # (B, D)
        return self.projection(memory)
```

### 方案 B：多分数阶叠加（类比多尺度 EMA）

如果单一 α 不够，可以用多个不同 α 的分数阶核叠加：

```python
class MultiFractionalMemory(nn.Module):
    """多分数阶记忆：不同 α 捕获不同时间尺度的记忆"""
    def __init__(self, input_dim, hidden_dim, n_orders=4, memory_length=50):
        super().__init__()
        self.n_orders = n_orders
        self.memory_length = memory_length

        # 多个 α 参数，初始化为不同值
        self.log_alphas = nn.Parameter(torch.tensor([-0.8, -0.5, -0.3, -0.1]))

        # 每个阶次一个权重
        self.order_weights = nn.Parameter(torch.ones(n_orders))

        self.projection = nn.Sequential(
            nn.Linear(input_dim * n_orders, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def _gl_weights(self, alpha, length):
        weights = torch.ones(length, device=alpha.device)
        for k in range(1, length):
            weights[k] = weights[k - 1] * (k - 1 - alpha) / k
        return weights

    def forward(self, action_history):
        outputs = []
        for i in range(self.n_orders):
            alpha = torch.sigmoid(self.log_alphas[i])
            w = self._gl_weights(alpha, self.memory_length)
            weighted = (action_history * w.view(1, -1, 1)).sum(dim=1)
            outputs.append(weighted * self.order_weights[i])
        return self.projection(torch.cat(outputs, dim=-1))
```

### 方案 C：与 EMA 的混合核

```python
class HybridMemory(nn.Module):
    """混合记忆核：EMA 捕获短程 + 分数阶捕获长程"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ema = MultiScaleEMA(input_dim, hidden_dim)  # 现有模块
        self.fractional = FractionalMemory(input_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)  # 学习何时用哪个

    def forward(self, action_history):
        ema_out = self.ema(action_history)
        frac_out = self.fractional(action_history)
        gate = torch.sigmoid(self.gate(torch.cat([ema_out, frac_out], dim=-1)))
        return gate * ema_out + (1 - gate) * frac_out
```

---

## 四、理论优势

### 4.1 物理可解释性

| EMA | 分数阶 |
|-----|--------|
| 信号处理工具 | 材料力学模型 |
| 多个时间常数 τ₁, τ₂, ... | 单一材料参数 α |
| 无负权重（单调衰减） | 有负权重（回弹效应） |
| 拟合幂律需很多项 | 天然幂律衰减 |

### 4.2 参数效率

| 方法 | 参数量 | 拟合能力 |
|------|--------|---------|
| 单尺度 EMA | 1 (β) | 差 |
| 多尺度 EMA (4 scales) | 8 (w₁-₄, β₁-₄) | 中 |
| 单分数阶 | 1 (α) | 好（长程记忆） |
| 多分数阶 (4 orders) | 8 (α₁-₄, w₁-₄) | 最强 |

### 4.3 与经典迟滞模型的关系

| 经典模型 | 数学形式 | 对应 |
|---------|---------|------|
| 弹性（无迟滞） | $\sigma = E\varepsilon$ | α = 0 |
| 粘性（完全记忆） | $\sigma = \eta\dot\varepsilon$ | α = 1 |
| 标准线性固体 | 指数松弛 | → EMA |
| 广义 Maxwell | 多指数 | → 多尺度 EMA |
| **分数阶 Kelvin-Voigt** | **$D^\alpha \varepsilon$** | **α ∈ (0,1)** |

分数阶 Kelvin-Voigt 模型是标准 Kelvin-Voigt 模型的推广，已在材料科学中广泛使用。

---

## 五、实验验证计划

### 阶段 1：对比实验（在现有 Flow Matching 框架内）

1. 保持 FlowMatch 架构不变，仅替换 MultiScaleEMA → FractionalMemory
2. 对比指标：
   - FM loss 收敛速度
   - 不同 action 下的形状区分度（cosine similarity）
   - 迟滞回线的重建精度

```bash
# 替换编码器后训练
python scripts/training/train_flowmatch.py --encoder fractional --memory_length 50
```

### 阶段 2：迟滞专项实验

1. 收集 "递增-递减" 循环数据（同一 action，不同历史路径）
2. 测量模型是否能区分同 action 不同历史的状态
3. 可视化学到的 α 值，看是否与材料物理一致

### 阶段 3：与预测-修正框架结合

参见 predictive_corrective_state_estimation。

---

## 六、风险与备选

| 风险 | 概率 | 应对 |
|------|------|------|
| α 学到的值不合理 | 中 | 添加正则化约束 α ∈ [0.2, 0.8] |
| 权重计算效率低 | 低 | 预计算 + cache，α 变化时才重算 |
| 不比 EMA 好 | 中 | 混合方案 C 兜底 |
| 实现复杂度 | 低 | 核心 < 20 行代码 |

---

## 七、关键参考文献

1. **Sorichetti et al. 2021** — "Viscoelasticity of PDMS: Fractional model" — PDMS 硅胶的分数阶粘弹性测量
2. **Mainardi 2010** — "Fractional Calculus and Waves in Linear Viscoelasticity" — 分数阶微积分在粘弹性中的理论框架
3. **Caputo 1967** — "Linear models of dissipation whose Q is almost frequency independent" — 分数阶本构的原始论文
4. **Podlubny 1999** — "Fractional Differential Equations" — 分数阶微积分的标准教材
5. **Chen 2025 (清华)** — "Hysteresis-Aware Neural Network Modeling" — 对比基准，他们用方向编码
