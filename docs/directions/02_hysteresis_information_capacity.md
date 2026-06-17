# 方向：粘弹性迟滞的信息容量——定量度量"身体记忆"

> 状态：待实验验证
> 优先级：高（作为所有建模方向的先验实验）
> 关联：自回归状态动力学（第〇节假设选择）、inspirations.md 方向二
> 科学问题：软体机器人的当前形状编码了多少关于加载历史的信息？
> 📝 **2026-06-17 更新**：本文原为"假设一(稳态) vs 假设二(状态转移)"的**工程选型先验实验**。该选型已务实解决——团队已确定构建状态转移模型（[13](13_closed_loop_state_transition.md)/[14](14_gt_observed_transition.md)/[15](15_open_loop_windowed_transition.md)），方向 13 设计锁定选 z 可学习、仿真保持线性阻尼、迟滞有效性留实物验证。故本文现为**纯科学/信息论实验**（解码容量、记忆视界），不再 gate 模型选型；实验设计（SVM/Ridge/MINE、1D-first）仍 sound 且未实现。原引用 autoregressive_state_dynamics.md（即将归档）改为指向 [13](13_closed_loop_state_transition.md)。

---

## 一、动机

### 从"迟滞是麻烦"到"迟滞是信息"

粘弹性迟滞通常被视为需要克服的工程问题。但换一个视角：

**软体机器人的身体就是一个物理记忆装置。**

加载历史被"写入"材料的微观结构中，表现为当前的宏观形状。这就像一个模拟存储器——不是数字的 0/1，而是连续的空间变形场。

问题：**这个记忆的容量有多大？能否被读取？**

### 为什么这是先验实验

在 [13_closed_loop_state_transition.md](13_closed_loop_state_transition.md) §〇（继承自已归档的 01）中，我们讨论了两种建模假设：

| 层次 | 问题 | 性质 |
|------|------|------|
| 假设一（稳态） vs 假设二（状态转移） | 迟滞**怎么建模**？ | 工程问题：选哪个模型 |
| 本方向 | 迟滞**有多少**？ | 科学问题：量化一个物理性质 |

本方向的实验结果直接指导假设选择：
- 如果迟滞可忽略 → 假设一成立，当前方法没问题
- 如果迟滞显著 → 假设二必须用，且实验数据指导历史窗口长度

---

## 二、关键问题与讨论

### Q1：只看形状真的能推断历史吗？

**直觉上的担忧**：动作空间是 2D 的（torque_x, torque_y），不同的动作序列可能有无数条路径到达相近的位置，映射空间太大了。

**关键观察**：这不影响实验可行性。原因：

1. **我们不需要解码完整序列**。不需要从形状反推出整个加载历史，只需要回答"形状能区分这两类序列吗？"——这是一个分类问题，不是逆问题。
2. **物理约束大幅缩减搜索空间**。软体臂的变形是连续、平滑的，不是任意的。到达同一位置的不同路径，产生的中间变形历史不同，残留的内部应力分布也不同。
3. **先做 1D 实验**。固定一个方向（如 torque_y=0），只在 torque_x 上变化。这样搜索空间从 2D 降到 1D，更容易验证可行性。

**建议实验顺序**：
- 先做 1D（单方向驱动）→ 如果 1D 下都无法区分 → 2D 更不可能 → 假设一成立
- 1D 可区分后再扩展到 2D

### Q2：多条路径到达同一位置怎么办？

具体场景：
- 路径 A：10 步从 0.0 线性增加到 0.5
- 路径 B：20 步从 0.0 线性增加到 0.5
- 路径 C：先到 0.8 再降回 0.5（过冲）
- 路径 D：随机游走恰好到达 0.5

**处理方式**：不要试图解码完整路径，而是解码路径的**统计特征**：

| 解码目标 | 分类/回归类型 | 物理意义 |
|---------|-------------|---------|
| 序列方向（升 vs 降） | 二分类 | 迟滞方向 |
| 序列均值 | 回归 | 平均载荷 |
| 序列变化速率 | 回归/分类 | 加载速率 |
| 是否有过冲 | 二分类 | 路径类型 |
| 序列方差 | 回归 | 载荷波动 |

**从简单到复杂**：先做二分类（升/降），再做属性回归，最后才考虑完整序列解码。

### Q3：状态表示只用图像或骨架够吗？

**仅用骨架坐标可能区分度不够**。当前的状态表示：

| 表示 | 维度 | 来源 | 问题 |
|------|------|------|------|
| 骨架节点坐标 | 31×3 = 93 | 3D GT | 高维，但很多维度对迟滞不敏感 |
| 渲染图像 | H×W | PyVista | 极高维，冗余信息多 |
| 2D 骨架 | N×2 | 图像提取 | 有损，精度有限 |

**建议增加的物理/统计特征，提高区分度**：

| 特征 | 计算方式 | 维度 | 物理意义 | 对迟滞的敏感度 |
|------|---------|------|---------|-------------|
| **节点曲率** | 相邻节点角度差的累计 | 30 | 每段的弯曲程度 | 高——迟滞直接体现在弯曲响应上 |
| **曲率变化率** | Δκ/Δt | 30 | 弯曲速度 | 高——反映动态过程 |
| **节点速度** | positions[t] - positions[t-1] | 31×3 = 93 | 运动方向和速率 | 高——直接编码运动趋势 |
| **应变能** | ∫κ²ds | 1 | 储存的弹性势能 | 中——整体能量状态 |
| **傅里叶描述子** | 骨架 FFT 的前 k 个系数 | 2k | 全局形状特征 | 中——捕捉空间频率差异 |
| **最大曲率位置** | argmax(κ) | 1 | 弯曲最严重的位置 | 中——不同加载路径可能在不同位置产生最大弯曲 |

**推荐特征提取代码**（用于信息容量实验）：

```python
def extract_features(positions, prev_positions=None):
    """从骨架位置提取多尺度特征"""
    # positions: (31, 3)

    # 1. 原始坐标（基线）
    coords = positions.flatten()  # (93,)

    # 2. 曲率（对迟滞敏感）
    segments = np.diff(positions, axis=0)          # (30, 3)
    tangents = segments / (np.linalg.norm(segments, axis=-1, keepdims=True) + 1e-8)
    kappa = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=-1), -1, 1))  # (29,)

    # 3. 累积曲率（整体弯曲）
    cumulative_kappa = np.cumsum(kappa)  # (29,)

    # 4. 应变能（标量）
    strain_energy = np.sum(kappa**2)  # scalar

    # 5. 速度（如果有时序数据）
    if prev_positions is not None:
        velocity = (positions - prev_positions).flatten()  # (93,)
        speed = np.linalg.norm(positions - prev_positions, axis=-1)  # (31,)
    else:
        velocity = np.zeros_like(coords)
        speed = np.zeros(31)

    return np.concatenate([
        coords,           # 93
        kappa,            # 29
        cumulative_kappa, # 29
        [strain_energy],  # 1
        velocity,         # 93
        speed,            # 31
    ])  # 总计 276 维
```

**实验策略**：先用骨架坐标做基线，然后逐步加入曲率、速度等特征，看哪个特征提升了分类/回归精度——提升最大的特征就是"迟滞信息的主要载体"。

### Q4：速率在当前系统中的含义

当前数据采集中，"速率"受两个参数控制：

- `steps_per_action = 500`：每个动作值维持 500 步 = 50ms
- `record_interval = 50`：每 50 步记录一帧 = 5ms
- 每个动作值产生 10 帧记录

**在当前数据中**：动作是"阶跃式"变化的（hold 500 步后突变），不是连续渐变的。如果要研究速率效应，需要修改采集策略：

| 方式 | 做法 | 速率范围 |
|------|------|---------|
| 改变 steps_per_action | 50/100/500/1000 步 | 5ms~100ms 每步 |
| 使用渐变 ramp | linspace(a, b, N) 平滑过渡 | 连续变速 |
| 使用 ContinuousSoftArmEnv | 逐帧设定动作 | 完全自由 |

详见 PROJECT_HELP.md 第 4.0 节的时序对应关系说明。

---

## 三、I(S; H) 的物理含义

### 定义

I(S; H) = 互信息，其中：
- **H** = 加载历史 = 过去的动作序列 {a₁, a₂, ..., aₜ}
- **S** = 当前形状

### 直白理解

**"如果你只看机器人当前的形状，你能推断出多少关于它过去被怎么驱动的信息？"**

两种极端情况：

| 材料 | 物理特性 | I(S; H) | 含义 |
|------|---------|---------|------|
| 理想弹性体 | 无迟滞，即时恢复 | ≈ 0 | 看形状猜不出历史 |
| 完美记忆体 | 迟滞无穷大 | 很大 | 看形状能反推出完整加载路径 |

**I(S; H) 本质上量化"这个材料有多少迟滞"。**

### 具体子问题

| 子问题 | 互信息形式化 | 物理含义 |
|--------|------------|---------|
| 短期记忆 | I(S; a_{t-k}) vs k | k 步前的动作对当前形状还有多少影响？ |
| 顺序记忆 | I(S; order(H)) | 形状能否区分"先升后降"和"先降后升"？ |
| 速率记忆 | I(S; rate(H)) | 形状能否区分"快充"和"慢充"？ |
| 极限容量 | I(S; H) 的上界 | 形状最多能编码多少位历史信息？ |

---

## 三、计算方法

### 核心难点

形状 S 是高维连续的（骨架 N×3 或图像特征），动作序列 H 也是高维连续的。直接计算 I(S; H) 在连续空间中很困难。

### 解决思路：用解码精度代替直接估计

互信息有一个关键性质：**如果从形状能解码出历史的某个属性，解码精度越高，互信息越大。**

所以我们不需要算完整的 I(S; H)，只需要回答几个更简单的子问题。每个子问题的回答都给出 I(S; H) 的一个**下界**——足够指导决策。

### 方法一：二分类法（最简单，1 小时出结果）

**回答"假设一还是假设二"**

```python
# 问题：相反顺序的加载，最终形状能区分吗？
# A: ramp_up   = [0.1, 0.2, 0.3, 0.4]
# B: ramp_down = [0.4, 0.3, 0.2, 0.1]
# 最终驱动值相同（0.4），但到达路径相反

shapes_A = [simulate(ramp_up) for _ in range(500)]
shapes_B = [simulate(ramp_down) for _ in range(500)]

X = np.concatenate([shapes_A, shapes_B])
y = np.array([0]*500 + [1]*500)

from sklearn.svm import SVC
acc = cross_val_score(SVC(), X, y, cv=5).mean()

# acc ≈ 50% → 形状无法区分，假设一成立（迟滞可忽略）
# acc ≈ 100% → 形状完全不同，假设一失效（迟滞显著）
```

**解读**：
- 50% = 瞎猜 → 迟滞不影响形状 → 假设一 OK
- 接近 100% → 迟滞完全可观测 → 必须用假设二

### 方法二：回归衰减法（半天，定量给出记忆窗口）

**回答"模型需要多长的历史"**

```python
from sklearn.linear_model import Ridge

results = {}
for k in range(1, 50):
    # X: 当前形状特征（骨架坐标或图像特征）
    # y: k 步前的动作值
    X = shapes_at_time_t           # (N, shape_dim)
    y = actions_at_time_t_minus_k   # (N, action_dim)

    model = Ridge().fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    results[k] = r2

# 画 R² vs k 的衰减曲线
# R² 衰减到 0 的那个 k 就是记忆窗口长度
```

**解读**：
- R² 在 k=3 时降到 ≈0 → 记忆 3 步，短窗口 EMA 够用
- R² 在 k=20 时还有 0.3 → 需要长历史窗口（Gamma/Laguerre + 自回归）

### 方法三：MINE（精确互信息估计，用于论文）

**如果前两个实验结果有意思，用 MINE 出精确曲线**

MINE (Mutual Information Neural Estimation) 利用 Donsker-Varadhan 表示：

```
I(X; Y) ≥ E[T(x,y)] - log(E[exp(T(x,y'))])
```

其中 T 是一个可学习的神经网络，y' 是 y 的随机置换。

```python
import torch
import torch.nn as nn

class MINENetwork(nn.Module):
    """互信息估计网络"""
    def __init__(self, shape_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(shape_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, shape, action):
        return self.net(torch.cat([shape, action], dim=-1))


def mine_estimate(T, shapes, actions, n_iter=1000, lr=1e-3):
    """训练 MINE 网络估计 I(shapes; actions)"""
    optimizer = torch.optim.Adam(T.parameters(), lr=lr)

    estimates = []
    for _ in range(n_iter):
        # 正样本：(shape, 对应的 action)
        joint = T(shapes, actions)
        # 负样本：(shape, 随机打乱的 action)
        perm = torch.randperm(len(actions))
        marginal = T(shapes, actions[perm])

        # MINE 目标（最大化 I 的下界）
        loss = -(joint.mean() - torch.log(torch.exp(marginal).mean() + 1e-8))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        estimates.append(-loss.item())

    return estimates[-1]  # 最终估计值
```

**使用场景**：画 I(S; a_{t-k}) vs k 的精确衰减曲线，放到论文里。

---

## 四、实验设计

### 实验路线（推荐顺序）

```
步骤 1：二分类实验（1小时）
  ↓
  acc ≈ 50% → 停。假设一成立，当前方法 OK。
  acc >> 50% → 继续。
  ↓
步骤 2：回归衰减实验（半天）
  ↓
  得到记忆窗口长度 K。
  ↓
步骤 3：速率依赖实验（半天）
  ↓
  快/慢加载能否区分？
  ↓
步骤 4（可选）：MINE 精确曲线（1天）
  ↓
  论文级别的定量结果。
```

### 实验 1：顺序可辨识性（二分类）

- 生成加载序列 A = ramp_up 和 B = ramp_down，最终值相同
- 多组不同目标值，每组 500 次重复
- 分类器：SVM（线性核就够了）
- **判断标准**：准确率 > 70% → 迟滞显著

### 实验 2：记忆窗口（回归衰减）

- 对 k = 1, 2, ..., 50，训练 Ridge/MLP 回归 shape → action_{t-k}
- **判断标准**：R² 降到 0.05 以下的 k 值 = 记忆窗口

### 实验 3：速率依赖性（二分类或回归）

- 同一目标驱动值，不同加载速率 [0.01, 0.05, 0.1, 0.5]
- 分类器区分不同速率
- **判断标准**：准确率 > 70% → 速率信息被编码在形状中

### 实验 4：记忆保持时间

- 施加加载序列 → 等待 τ 步 → 观测形状 → 尝试解码历史
- τ = [0, 5, 10, 20, 50, 100]
- **目的**：迟滞信息能保持多久？

### 数据生成（PyElastica）

```python
# 修改 collect.py，生成特殊的加载序列
def generate_probing_sequences(n_repeat=500):
    """生成用于信息容量估计的特殊加载序列"""
    sequences = []

    # 1. 反序序列对
    for target in np.linspace(0.1, 0.9, 10):
        ramp_up = np.linspace(0.0, target, 10)
        ramp_down = np.linspace(0.9, target, 10)
        for _ in range(n_repeat // 10):
            sequences.append(('ramp_up', ramp_up, target))
            sequences.append(('ramp_down', ramp_down, target))

    # 2. 变速率序列
    for target in [0.4, 0.6, 0.8]:
        for rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            seq = np.linspace(0, target, max(2, int(target / rate)))
            for _ in range(100):
                sequences.append(('rate', seq, rate))

    return sequences
```

---

## 五、实验结果对建模的指导

| 实验结果 | 含义 | 对模型设计的指导 |
|---------|------|----------------|
| 顺序分类 acc ≈ 50% | 迟滞可忽略 | 假设一成立，当前方法够用 |
| 顺序分类 acc >> 50% | 迟滞显著 | 必须用假设二，自回归/状态转移 |
| 记忆窗口 < 5 步 | 短期记忆 | 短窗口 EMA 即可 |
| 记忆窗口 > 20 步 | 长期记忆 | Gamma/Laguerre 长核 + 自回归 |
| 速率可区分 | 加载速率影响形状 | 训练数据必须包含变速率 |
| 速率不可区分 | 速率无关 | 当前数据采集策略 OK |

---

## 六、科学贡献（如果结果有意思）

1. **首次量化**粘弹性软体机器人形状的信息编码能力
2. **建立**迟滞的信息论框架——从"工程麻烦"升格为"物理通道"
3. **连接**软体机器人学与信息论、非平衡态统计力学
4. **启发**新型传感器设计——利用身体本身作为分布式传感器

### 与现有工作的区别

| 工作 | 视角 | 我们 |
|------|------|------|
| Tang 2026, Yu 2026 | 假设迟滞不存在，直接建模 | 首先量化迟滞有多强 |
| Chen 2025 (迟滞建模) | 建模迟滞以补偿它 | 把迟滞看作信息通道 |
| 材料科学 (DMA 测试) | 测量应力-应变迟滞回线 | 用信息论量化形状对历史的编码 |

### 发表策略

- **如果迟滞显著**：结果本身就是一个发现（"软体机器人形状编码了加载历史"），可以作为独立短文投 ICRA/RSS
- **如果迟滞不显著**：虽然不够独立发表，但为后续方向（自回归等）提供了定量依据，避免走弯路
- **作为其他论文的分析章节**：无论结果如何，这些数据都是有价值的，可以加到任何相关论文的实验部分

---

## 七、工作量估计

| 步骤 | 时间 | 产出 |
|------|------|------|
| 修改 collect.py 生成探测序列 | 2 小时 | 特殊数据集 |
| 运行 PyElastica 仿真 | 2 小时 | 形状数据 |
| 二分类实验 | 1 小时 | 顺序可辨识性结论 |
| 回归衰减实验 | 2 小时 | 记忆窗口曲线 |
| 速率依赖实验 | 2 小时 | 速率可辨识性结论 |
| MINE 精确估计（可选） | 4 小时 | 论文级互信息曲线 |
| **总计** | **1-2 天** | **定量的迟滞特性报告** |
