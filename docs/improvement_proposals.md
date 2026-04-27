# 提升软体机器人自建模预测准确性的改进方案

> 基于 C-MSTNF 高频跳变、Smooth-CMSTNF 尖端发散、ODE-CMSTNF 无显著提升的实验现状，
> 分析根因并提出多个方向性改进方案。

---

## 1. 问题根因分析

### 1.1 当前架构的统一瓶颈

四个模型（MSTNF / C-MSTNF / ODE-CMSTNF / Smooth-CMSTNF）的差异集中在**时序编码器**（EMA vs ODE），但它们共享同一个**空间映射瓶颈**：

```
时序编码器 → physics_state (128d) → 变形 MLP → 逐点 3D 位移
                                      ↑
                                  这里是问题所在
```

无论时序编码多光滑，最终的 `deform_mlp`（5 层 ReLU MLP）将 `(pos_enc, state, action)` 映射到 3D 位移时，MLP 本身就是一个高频函数逼近器。这是所有方法都表现不理想的共同原因。

### 1.2 各模型失败的具体原因

| 模型 | 症状 | 根因 |
|------|------|------|
| C-MSTNF | 高频跳变 | 变形 MLP 的 ReLU 激活在边界处不可导；位置编码引入高频分量被 MLP 放大 |
| Smooth-CMSTNF | 尖端发散 | Spectral norm 全局限制了 Lipschitz 常数，但尖端恰好是变形最大的区域——正则化与精度矛盾 |
| ODE-CMSTNF | 无显著提升 | ODE 只改进了时序编码，空间映射 MLP 未变——瓶颈不在时序而在空间 |

### 1.3 对比 FBV_SM（刚性臂基线）

FBV_SM（`model.py`）中，命令参数经 `cmd_encoder`（2 层 MLP）后直接与位置特征拼接，再经一个浅层 `feed_forward`（2 层）输出。关键区别：

- **没有独立的变形 MLP**：驱动参数不经过额外的非线性映射，连续性有保证
- **浅层网络**：2 层 vs 5 层，减少了对高频分量的拟合能力（也意味着不会产生高频跳变）

---

## 2. 改进方案

### 方案 A：低频变形基（Deformation Basis）

**核心思想**：不用 MLP 直接输出逐点 3D 位移，而是学习一组光滑的基函数，让时序编码器只预测基函数的系数。

**架构**：

```
action_window → 时序编码器 → coefficients (K 个系数)
空间点 x → 预定义光滑基函数 Φ_1(x), ..., Φ_K(x)  [RBF / 低阶多项式 / 低频 Fourier]
deformation(x) = Σ_k  coeff_k * Φ_k(x)
```

**具体实现选择**：

1. **RBF 基函数**：沿机械臂轴线均匀放置 M 个中心点 c_i，定义
   ```
   Φ_i(x) = exp(-||x - c_i||² / (2σ²))
   displacement(x) = Σ_i  W_i * Φ_i(x)
   ```
   其中 W_i 是 (3,) 的位移向量，由时序编码器预测。RBF 天然光滑且局部支撑。

2. **低阶多项式基**：用沿轴线位置 s 的低阶多项式
   ```
   Δx(s) = a_0 + a_1*s + a_2*s² + a_3*s³
   Δy(s) = b_0 + b_1*s + b_2*s² + b_3*s³
   Δz(s) = c_0 + c_1*s + c_2*s² + c_3*s³
   ```
   系数 [a_0..a_3, b_0..b_3, c_0..c_3] 由时序编码器直接输出。这直接对应 Cosserat 杆模型的分段多项式变形。

**优势**：
- 基函数本身光滑 → 变形天然光滑，无需正则化
- 时序编码器只预测低维系数（如 12 维）→ 参数少、训练快
- 物理可解释：系数直接对应弯曲曲率等物理量

**风险**：
- 基函数的表达能力受预设数量限制
- 需要合理选择基函数类型和数量

---

### 方案 B：去除变形 MLP（Linear Deformation）

**核心思想**：参照 FBV_SM 的设计，时序编码器输出直接作为"驱动参数"，不经额外 MLP。

**架构对比**：

```
当前:  temporal_state → [MLP 5层] → displacement  (非线性、高频)
改进:  temporal_state → [线性层]  → displacement   (线性、光滑)
```

**具体做法**：

```python
# 替代 DeformationField 中的 deform_mlp
self.deform_linear = nn.Linear(hidden_dim + action_dim, 3)
# 或带位置依赖的版本:
self.deform_linear = nn.Linear(pos_enc_dim + hidden_dim + action_dim, 3)
```

**与方案 A 的关系**：方案 A 是用基函数展开代替 MLP；方案 B 是直接去掉非线性。方案 B 更简单但表达能力更弱，可以作为方案 A 的快速验证基准。

**优势**：
- 最小改动，只需修改 `deform_mlp` 为线性层
- 线性映射保证 Lipschitz 连续
- 训练更快、更稳定

**风险**：
- 线性映射可能表达能力不足
- 如果变形本身是非线性的（软体机器人的大变形），线性假设可能不够

---

### 方案 C：多视角几何约束（Multi-view Consistency）

**核心思想**：单视角图片提供的 3D 约束严重不足（深度歧义），增加视角可以从几何层面约束模型。

**架构改动**：

```
                    ┌─ 视角 1 → 渲染 → loss_1 ─┐
3D 场查询网络 → 体渲染 ─├─ 视角 2 → 渲染 → loss_2 ─┼─ total_loss
                    ├─ 视角 3 → 渲染 → loss_3 ─┤
                    └─ ...                      ┘
```

**多视角的额外好处**：

1. **几何一致性 loss**：同一 3D 点在不同视角下的投影应该一致，这个约束比单视角的重建 loss 更强
2. **遮挡推理**：单视角无法区分遮挡与不存在，多视角可以帮助
3. **尖端约束增强**：尖端在多个视角都可见 → 多个 loss 同时约束尖端位置

**实施步骤**：

1. **数据采集**：修改仿真环境，每个时间步从 3-4 个视角渲染图像
2. **训练 loss**：对每个视角分别计算重建 loss，求和
3. **可选：多视角深度监督**：利用仿真器的深度图作为额外监督信号

**数据采集改动示例**：

```python
# 在 elastica_env.py 中添加多个相机位置
MULTI_VIEW_CAMERAS = [
    {"eye": (1.5, 0.0, 0.5),  "center": (0, 0, 0.25)},  # 当前视角
    {"eye": (0.0, 1.5, 0.5),  "center": (0, 0, 0.25)},  # 侧面 90°
    {"eye": (-1.0, -1.0, 0.5),"center": (0, 0, 0.25)},  # 45° 对角
]
```

**优势**：
- 从根本上解决 3D 约束不足的问题
- 与任何模型架构兼容（不改变网络结构）
- 在真实机器人部署时，多摄像头也是可行的

**风险**：
- 数据采集量增大 3-4 倍
- 训练时间相应增加
- 需要精确的多相机标定（仿真中可忽略，真实部署需考虑）

---

### 方案 D：骨架先验 + 半径场（Skeleton + Radius Prior）

**核心思想**：利用软体机械臂的强几何先验——它是一根连续的细长杆，不是任意的 3D 密度场。

**架构**：

```
时序编码器 → 骨架曲线参数 (控制点坐标)
                                    ↓
给定骨架曲线 C(s)，任意 3D 点 x 的密度为:
  d(x) = f( distance_to_curve(x) )
  其中 f 是一个 1D 函数（如高斯: f(r) = exp(-r²/(2σ²))）
```

**具体实现**：

```python
class SkeletonModel(nn.Module):
    def __init__(self, action_dim, n_control_points=8):
        # 时序编码器 → n_control_points 个 3D 控制点
        self.temporal = MultiScaleEMA(action_dim=action_dim, ...)
        self.skeleton_head = nn.Linear(hidden_dim, n_control_points * 3)

        # 半径参数（可学习但动作无关，或也可动作条件化）
        self.radius = nn.Parameter(torch.tensor(0.015))

    def forward(self, points, action_window):
        # 1. 预测骨架控制点
        ctrl_pts = self.skeleton_head(self.temporal(action_window))
        ctrl_pts = ctrl_pts.reshape(-1, self.n_ctrl, 3)
        curve = cubic_spline(ctrl_pts)  # 光滑曲线

        # 2. 计算 points 到曲线的最短距离
        dist = point_to_curve_distance(points, curve)

        # 3. 密度 = 高斯距离函数
        density = torch.exp(-dist**2 / (2 * self.radius**2))
        visibility = ...
        return visibility, density
```

**与当前方法的本质区别**：

| | 当前方法 | 骨架先验 |
|---|---|---|
| 表示 | 隐式场 (MLP) | 显式骨架 + 隐式距离场 |
| 几何先验 | 无 | 杆状拓扑 |
| 连续性保证 | 靠正则化 | 靠骨架曲线的光滑性 |
| 参数量 | 多（整个 MLP） | 少（只需控制点坐标） |

**优势**：
- 利用强几何先验大幅减少自由度
- 骨架曲线天然连续（样条插值保证）
- 与软体机器人的物理结构完全对应
- 尖端位置直接由骨架端点决定，不会发散

**风险**：
- 需要实现可微的点到曲线距离计算
- 对非杆状机器人泛化性差
- 渲染管线需要适配

---

### 方案 E：课程式频率学习（Coarse-to-Fine Frequency）

**核心思想**：不让网络一次性使用全部频率的位置编码，而是从低频开始逐渐增加，让网络先学大变形再学细节。

**实现方式**：

```python
class CoarseToFinePositionalEncoder(PositionalEncoder):
    def __init__(self, d_input, n_freqs, log_space=True):
        super().__init__(d_input, n_freqs, log_space)
        # 可学习的频率 mask（初始全零 = 只用最低频）
        self.freq_mask = nn.Parameter(torch.zeros(n_freqs))

    def forward(self, x):
        encoded = super().forward(x)  # 获取所有频率编码
        # 通过 sigmoid mask 控制每个频率的贡献
        mask = torch.sigmoid(self.freq_mask)  # (n_freqs,)
        # 对 sin/cos 频率分量加权
        ...
```

训练策略：
1. **Epoch 0-50%**：`freq_mask` 冻结为 0，只使用最低 2-3 个频率 → 学习大尺度变形
2. **Epoch 50-100%**：`freq_mask` 解冻，让网络学习是否需要更高频率
3. 或更简单：手动 schedule，逐步解锁频率

**对变形场的应用**：变形场当前使用 `deform_n_freqs=6`，可以：
- 初始阶段只用 `deform_n_freqs=2`
- 每 N 个 epoch 增加 1 个频率

**优势**：
- 改动极小（只改编码器 + 训练 schedule）
- 先低频保证全局形状正确，再高频加细节
- 可以和任何其他方案组合

**风险**：
- 需要仔细调整频率增加的 schedule
- 可能最终仍会学到高频分量（只是延迟了问题）

---

### 方案 F：显式 3D 监督（3D Supervision from Simulator）

**核心思想**：仿真器有完整的 3D 信息（杆体节点坐标），但当前只用了 2D 渲染图像作为监督。直接用 3D 信息辅助训练。

**3D 监督信号来源**：

```python
# PyElastica 仿真器在每个时间步都能提供:
rod.position_collection   # (3, N+1) 节点坐标
rod.radius                # (N,) 每段半径
rod.director_collection   # (3, 3, N) 截面朝向
```

**可添加的 3D loss**：

1. **骨架匹配 loss**：模型预测的密度场中心线应与仿真器的节点坐标一致
2. **变形场位移监督**：deformation(x_canonical) 的输出应接近 (x_simulated - x_canonical)
3. **密度场监督**：在已知的杆体表面点采样，直接监督 density 值

**实现示例**：

```python
# 在训练循环中添加
def compute_3d_loss(model, rod_positions, action_window):
    # rod_positions: (N_nodes, 3) 仿真器的真实节点坐标
    canonical_points = ... # Phase 1 学到的静态形态上的采样点
    predicted_displacement, _ = model.deform(canonical_points, action_window)
    target_displacement = rod_positions - canonical_rest_positions
    return F.mse_loss(predicted_displacement, target_displacement)
```

**优势**：
- 仿真阶段有丰富的 3D 信息，不用白不用
- 直接约束变形场，而不是间接通过 2D 渲染
- 在 sim-to-real 迁移时，3D 监督可以逐步撤掉

**风险**：
- 需要修改数据采集流程，保存 3D 节点坐标
- 如果目标是 sim-to-real，3D 监督可能导致模型过度依赖仿真器精度
- 在真实机器人上部署时没有 3D 监督

---

## 3. 方案组合与优先级建议

### 推荐的探索顺序

```
短期（1-2 天验证）:
  1. 方案 B（线性变形层）→ 最快验证，看是否是 MLP 的问题
  2. 方案 E（课程频率）→ 改动最小，可快速实验

中期（1 周验证）:
  3. 方案 A（RBF 基函数变形）→ 物理上最合理，实现适中
  4. 方案 F（3D 监督）→ 利用仿真器的完整信息

长期（2 周验证）:
  5. 方案 C（多视角）→ 需要改数据采集 + 训练
  6. 方案 D（骨架先验）→ 架构改动较大，但可能是最终方案
```

### 最推荐的组合

**方案 A + 方案 C**：低频变形基 + 多视角约束

- 变形基函数保证光滑性（从模型结构上解决高频问题）
- 多视角保证 3D 约束充分（从数据层面解决精度问题）
- 两者独立、正交，可以分别验证效果

### 实验设计建议

每个方案应对比以下指标：
1. **渲染 PSNR**（2D 图像质量）
2. **3D 点云误差**（如果有 3D 监督）
3. **时序连续性指标**：相邻帧预测的 3D 场之间的差异
4. **尖端预测误差**：单独评估尖端区域的精度
5. **推拉力曲线**：预测的尖端轨迹 vs 真实轨迹

---

## 4. 理论分析：为什么 MLP 变形场产生高频

从信号处理角度分析当前变形 MLP 的问题：

### 4.1 位置编码 + MLP = 高通滤波器

位置编码将坐标 x 映射到高维特征：
```
PE(x) = [x, sin(2⁰x), cos(2⁰x), ..., sin(2⁹x), cos(2⁹x)]
```

最高频率 2⁹ = 512，在单位空间内产生 512 个周期。MLP 的 ReLU 激活函数可以看作分段线性函数，它在相邻 ReLU 节点之间做线性插值。当输入包含 512 周期的高频分量时，MLP 需要在非常细的空间尺度上做出响应——这很容易导致：

- 训练不充分时产生棋盘格状伪影
- 训练充分后仍可能在训练数据稀疏区域（如尖端）过拟合
- ReLU 的不可导性在分段边界处产生尖点

### 4.2 为什么正则化（Smooth-CMSTNF）在尖端失效

尖端有两个特殊性质：
1. **位移最大**：固定端位移 ≈ 0，尖端位移最大 → 需要最大的变形场值
2. **数据最稀疏**：尖端在图像中只占少量像素 → 监督信号最少

Spectral norm 限制了变形场对输入变化的敏感度（Lipschitz 常数），但尖端恰恰需要高敏感度（从 0 到最大位移的变化）。全局正则化在尖端与需求矛盾。

### 4.3 为什么基函数方案能避免这个问题

RBF 基函数 Φ_i(x) = exp(-||x-c_i||²/(2σ²)) 具有：
- **无穷阶可导** → 天然光滑
- **局部支撑** → 远离中心的点不受影响
- **频率受 σ 控制** → σ 越大，变形越光滑

当变形被表达为 `Σ coeff_i * Φ_i(x)` 时，即使系数 coeff_i 有噪声（来自不完美的时序编码），最终变形仍然是光滑的——因为基函数本身吸收了高频噪声。

---

## 5. 与相关工作的对比

| 方法 | 代表工作 | 与本项目的对应 |
|------|---------|---------------|
| D-NeRF | Park et al., 2021 | → C-MSTNF 的 canonical + deformation |
| HyperNeRF | Park et al., 2022 | → 可借鉴其时间切片思想 |
| Nerfies | Park et al., 2021 | → 可借鉴其 deformation basis |
| SoftNeRF | Shan et al., 2024 | → 直接相关的软体机器人方法 |
| BANMo | Yang et al., 2022 | → 骨架先验 + NeRF (方案 D) |
| BARF | Wang et al., 2021 | → 课程式频率学习 (方案 E) |

特别是 **Nerfies** 中使用了类似的变形基函数思想来约束人脸表情变形的光滑性，可以参考其实现。
