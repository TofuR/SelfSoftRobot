# 方向 5: 时序编码与粘弹性迟滞建模 (Temporal Encoding for Viscoelastic Hysteresis)

> **这是论文的核心创新点。** 软体机器人的运动存在显著的迟滞效应（粘弹性），即相同的外力输入在不同加载历史下产生不同的变形。这是与刚性机器人自建模的根本区别，也是现有工作（Chen 2022, Shan 2024, Hu 2025）完全忽略的。

---

## 1. 问题分析

### 1.1 软体机器人的迟滞现象

软体机器人（硅胶/气动/线驱动）的材料具有**粘弹性 (viscoelasticity)**：

```
刚性机器人:     F → x(t) = F/k            (瞬时、可逆、无记忆)
软体机器人:     F → x(t) = f(F, F_history) (延迟、蠕变、路径依赖)
```

具体表现：
1. **加载-卸载不对称**：相同扭矩下，加载和卸载的弯曲角度不同
2. **蠕变 (creep)**：恒定扭矩下，变形持续增加
3. **应力松弛 (stress relaxation)**：恒定变形下，所需应力逐渐减小
4. **频率依赖**：快速加载和慢速加载产生不同变形

### 1.2 现有工作的局限

| 工作 | 时序处理 | 迟滞建模 |
|------|---------|---------|
| Chen 2022 | 无 (单帧) | 无 |
| Hu 2025 | 无 (单帧) | 无 |
| Shan 2024 | 无 (单帧) | 无 |
| MSTNF (我们) | MultiScaleEMA | 隐式 (EMA 有记忆) |
| MS-SCNF (我们) | MultiScaleEMA | 隐式 (EMA 有记忆) |

**所有现有工作都假设瞬时响应**（输入 → 输出是单帧映射）。但真实的软体机器人运动是**路径依赖的**。

### 1.3 为什么这是核心创新

1. **物理上有意义**：粘弹性是软体材料的本征属性
2. **现有工作空白**：没有任何自建模论文处理迟滞
3. **方法论上新颖**：需要将材料科学的迟滞模型融入神经场
4. **实用价值**：准确的迟滞建模 → 更精确的运动控制

---

## 2. 理论基础

### 2.1 经典迟滞模型

**1. Preisach 模型**（最经典）：
```
y(t) = ∬_Ω μ(α,β) γ[α,β]u(t) dα dβ

其中 γ[α,β] 是迟滞算子，α>β 是上/下阈值
```

**2. Bouc-Wen 模型**（微分形式）：
```
ẏ = Aẋ - β|ẏ|ⁿ x - γ ẏ|ẋ|ⁿ⁻¹
```

**3. Maxwell/Wiechert 模型**（粘弹性力学）：
```
弹簧: σ = E·ε
阻尼器: σ = η·dε/dt
组合: 多个 Maxwell 单元并联

当前 MultiScaleEMA 实际上就是一个简化的 Maxwell 模型：
  EMA: s_t = α·s_{t-1} + (1-α)·x_t
  这等价于一个弹簧-阻尼器单元
```

### 2.2 MultiScaleEMA 的局限

当前 MultiScaleEMA 有 4 个 learnable decay rates，但它：

1. **线性**：EMA 是线性算子，无法捕捉非线性迟滞环
2. **无加载方向感知**：不区分加载/卸载
3. **单调记忆衰减**：所有历史以相同速率衰减，无选择性

### 2.3 改进方向

需要让时序编码器能够：
1. 区分加载方向（扭矩增/减）
2. 捕捉非线性输入-输出关系
3. 有选择地保留/遗忘历史信息

---

## 3. 模型设计

### 3.1 方案 A: Hysteresis-Aware EMA (HA-EMA)

**核心思想**：让 EMA 的 decay rate 依赖于输入的变化方向。

```python
class HysteresisAwareEMA(nn.Module):
    """迟滞感知的 EMA — decay rate 随加载方向变化。"""

    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128):
        super().__init__()
        self.n_scales = n_scales
        self.hidden_dim = hidden_dim

        # 每个 scale 有加载/卸载两组 decay rate
        self.raw_decays_loading = nn.Parameter(
            torch.linspace(-1.5, 1.0, n_scales)  # sigmoid → 不同范围
        )
        self.raw_decays_unloading = nn.Parameter(
            torch.linspace(-1.0, 1.5, n_scales)  # 对称初始化
        )

        # 方向检测门控
        self.direction_gate = nn.Sequential(
            nn.Linear(action_dim * 2, 64),  # [当前动作, 速度]
            nn.ReLU(),
            nn.Linear(64, n_scales),
            nn.Sigmoid(),
        )

        # 状态映射
        self.state_mlp = nn.Sequential(
            nn.Linear(n_scales * action_dim + action_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action_window):
        """
        Args:
            action_window: (B, K, D) — 动作历史窗口
        Returns:
            physics_state: (B, hidden_dim)
        """
        B, K, D = action_window.shape

        # 计算速度 (方向信息)
        velocity = action_window[:, -1] - action_window[:, -2]  # (B, D)

        # 方向门控：决定使用 loading 还是 unloading 的 decay rate
        gate_input = torch.cat([action_window[:, -1], velocity], dim=-1)
        gate = self.direction_gate(gate_input)  # (B, n_scales)

        decays_load = torch.sigmoid(self.raw_decays_loading)    # (n_scales,)
        decays_unload = torch.sigmoid(self.raw_decays_unloading) # (n_scales,)
        decays = gate * decays_load + (1 - gate) * decays_unload  # (B, n_scales)

        # 多尺度 EMA（与 MultiScaleEMA 相同逻辑，但 decay 是动态的）
        ema_features = []
        for s in range(self.n_scales):
            alpha = decays[:, s]  # (B,) — 每个 batch 有不同的 decay
            state = torch.zeros(B, D, device=action_window.device)
            for t in range(K):
                alpha_t = alpha.unsqueeze(1)  # (B, 1)
                state = alpha_t * state + (1 - alpha_t) * action_window[:, t]
            ema_features.append(state)

        # 组合特征
        ema_concat = torch.cat(ema_features, dim=-1)  # (B, n_scales * D)
        features = torch.cat([ema_concat, action_window[:, -1], velocity], dim=-1)
        physics_state = self.state_mlp(features)

        return physics_state
```

### 3.2 方案 B: Neural ODE with Memory

**核心思想**：用 Neural ODE 建模连续时间的粘弹性动态。

```python
class ViscoelasticODE(nn.Module):
    """粘弹性 Neural ODE — 连续时间建模。"""

    def __init__(self, action_dim, hidden_dim=128, memory_dim=32):
        super().__init__()
        self.memory_dim = memory_dim

        # ODE 动力学网络: ds/dt = f(s, a, t)
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + memory_dim, 256),
            nn.Tanh(),  # 用 Tanh 保证 Lipschitz 约束
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, hidden_dim),
        )

        # 记忆编码：将历史轨迹编码为 memory vector
        self.memory_encoder = nn.GRU(
            input_size=action_dim,
            hidden_size=memory_dim,
            num_layers=1,
            batch_first=True,
        )

        # 初始状态
        self.init_state = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(self, action_window):
        """
        Args:
            action_window: (B, K, D) — 动作历史
        Returns:
            physics_state: (B, hidden_dim)
        """
        B, K, D = action_window.shape

        # 编码历史为 memory
        _, memory = self.memory_encoder(action_window)  # (1, B, memory_dim)
        memory = memory.squeeze(0)  # (B, memory_dim)

        # ODE 积分
        state = self.init_state.expand(B, -1)
        dt = 1.0 / K

        for t in range(K):
            action_t = action_window[:, t]
            ode_input = torch.cat([state, action_t, memory], dim=-1)
            ds = self.ode_func(ode_input)
            state = state + dt * ds  # Euler 积分

        return state
```

### 3.3 方案 C: PlayaCurl-inspired 不变测度学习

**核心思想**：学习输入-输出迟滞环的拓扑不变量。

```
参考 PlayaCurl (2024):
  - 迟滞环具有拓扑特征（面积、方向、对称性）
  - 学习这些不变量可以更稳定地建模迟滞

思路:
  - 将动作序列 → 延迟嵌入 (delay embedding)
  - 学习嵌入空间中的不变流形
  - 在不变流形上进行预测
```

### 3.4 方案对比

| 方案 | 复杂度 | 可解释性 | 迟滞建模能力 | 训练稳定性 |
|------|--------|---------|-------------|-----------|
| A: HA-EMA | ★★☆ | ★★★ | ★★☆ | ★★★ |
| B: Neural ODE | ★★★ | ★★☆ | ★★★ | ★★☆ |
| C: 不变测度 | ★★★★ | ★★★★ | ★★★★ | ★☆☆ |

**推荐**：先做 A（改动最小，最容易验证），再做 B。

---

## 4. 迟滞数据集

### 4.1 为什么需要专门的迟滞数据

当前数据集（随机动作序列）不能系统地表征迟滞行为。需要设计**特定的加载协议**来激发迟滞效应。

### 4.2 迟滞数据采集协议

```python
# scripts/data/collect_hysteresis.py

def generate_hysteresis_protocols(action_dim=2, max_torque=0.3):
    """生成迟滞表征数据采集协议。"""

    protocols = []

    # Protocol 1: 三角波加载（经典迟滞测试）
    # 固定频率，从 0 → max → 0 → -max → 0
    for freq in [0.1, 0.5, 1.0, 2.0]:  # 不同频率
        for axis in range(action_dim):
            actions = triangular_wave(freq=freq, amplitude=max_torque,
                                      n_cycles=5, axis=axis)
            protocols.append(('triangular', freq, axis, actions))

    # Protocol 2: 阶跃响应（蠕变测试）
    # 突然施加恒定扭矩，维持一段时间
    for level in [0.1, 0.2, 0.3]:
        for hold_time in [0.5, 1.0, 2.0]:
            actions = step_response(level=level, hold_time=hold_time,
                                    ramp_time=0.1)
            protocols.append(('step', level, hold_time, actions))

    # Protocol 3: 应力松弛测试
    # 快速加载到目标位置，然后保持位移
    for target in [0.1, 0.2, 0.3]:
        actions = stress_relaxation(target=target, hold_time=2.0)
        protocols.append(('relaxation', target, actions))

    # Protocol 4: 频率扫描（动态力学分析）
    # 对数扫频
    actions = frequency_sweep(f_start=0.01, f_end=5.0, duration=20.0)
    protocols.append(('sweep', actions))

    return protocols
```

### 4.3 数据采集命令

```bash
# 三角波迟滞数据
python scripts/data/collect_hysteresis.py \
    --mode triangular \
    --freqs 0.1 0.5 1.0 2.0 \
    --amplitude 0.3 \
    --output data/hysteresis/triangular

# 阶跃响应数据
python scripts/data/collect_hysteresis.py \
    --mode step \
    --levels 0.1 0.2 0.3 \
    --hold_times 0.5 1.0 2.0 \
    --output data/hysteresis/step

# 应力松弛数据
python scripts/data/collect_hysteresis.py \
    --mode relaxation \
    --targets 0.1 0.2 0.3 \
    --hold_time 2.0 \
    --output data/hysteresis/relaxation

# 完整迟滞数据集
python scripts/data/collect_hysteresis.py \
    --mode all \
    --output data/hysteresis/full
```

---

## 5. 训练方案

### 5.1 预训练 + 迟滞微调

```bash
# Step 1: 在通用数据上预训练骨架（与 MS-SCNF Phase 1 相同）
python scripts/training/train_ms_scnf.py \
    --data_dir data/sequence_data_3d \
    --phase 1 \
    --n_epochs 200 \
    --save_dir train_log/ms_scnf_hysteresis/phase1

# Step 2: 替换时序编码器为 HA-EMA，在迟滞数据上微调
python scripts/training/train_hysteresis.py \
    --pretrained_skeleton train_log/ms_scnf_hysteresis/phase1/model/best_model.pt \
    --hysteresis_data data/hysteresis/full \
    --general_data data/sequence_data_3d \
    --temporal_type ha_ema \
    --n_epochs 200 \
    --loss_skeleton 1.0 \
    --loss_hysteresis 0.5 \
    --save_dir train_log/ms_scnf_hysteresis/phase2
```

### 5.2 端到端训练

```bash
# 直接用 HA-EMA 从头训练
python scripts/training/train_ms_scnf_hysteresis.py \
    --data_dir data/sequence_data_3d \
    --temporal_type ha_ema \
    --n_epochs 300 \
    --save_dir train_log/ms_scnf_ha_ema
```

---

## 6. 验证与可视化

### 6.1 迟滞环可视化（最关键的验证）

```bash
# 生成迟滞环对比图
python scripts/evaluation/plot_hysteresis_loops.py \
    --checkpoints \
        train_log/ms_scnf_ema/model/best_model.pt \
        train_log/ms_scnf_ha_ema/model/best_model.pt \
    --data_dir data/hysteresis/triangular \
    --output output/hysteresis_loops
```

这会生成：
- **输入扭矩 vs 输出弯曲角** 的迟滞环图
- 对比标准 EMA 和 HA-EMA 的迟滞环拟合质量
- 不同频率下的迟滞环变化

### 6.2 定量指标

```python
# scripts/evaluation/eval_hysteresis.py

def hysteresis_metrics(pred, gt, actions):
    """迟滞专用评估指标。"""

    metrics = {}

    # 1. 迟滞环面积误差（核心指标）
    # 计算预测和 GT 的迟滞环面积
    pred_area = compute_loop_area(actions, pred)
    gt_area = compute_loop_area(actions, gt)
    metrics['loop_area_error'] = abs(pred_area - gt_area) / gt_area

    # 2. 迟滞环宽度
    pred_width = compute_loop_width(actions, pred)
    gt_width = compute_loop_width(actions, gt)
    metrics['loop_width_error'] = abs(pred_width - gt_width) / gt_width

    # 3. 相位滞后
    pred_phase = compute_phase_lag(actions, pred)
    gt_phase = compute_phase_lag(actions, gt)
    metrics['phase_lag_error'] = abs(pred_phase - gt_phase)

    # 4. 加载/卸载分支分别的误差
    loading_mask = actions_diff > 0
    unloading_mask = actions_diff < 0
    metrics['loading_mne'] = mean_node_error(pred[loading_mask], gt[loading_mask])
    metrics['unloading_mne'] = mean_node_error(pred[unloading_mask], gt[unloading_mask])
    metrics['asymmetry_error'] = abs(metrics['loading_mne'] - metrics['unloading_mne'])

    return metrics
```

```bash
# 迟滞评估
python scripts/evaluation/eval_hysteresis.py \
    --checkpoint train_log/ms_scnf_ha_ema/model/best_model.pt \
    --data_dir data/hysteresis/full \
    --save_dir output/hysteresis_eval
```

### 6.3 Notebook 验证

`15_hysteresis_analysis.ipynb`:
```
1. 迟滞数据加载与可视化
   - 三角波：输入扭矩 vs 时间
   - 对应的骨架弯曲角 vs 时间
   - 扭矩-弯曲角 迟滞环

2. 仿真器迟滞特性分析
   - 不同频率下的迟滞环形状
   - 加载-卸载不对称性
   - 蠕变曲线
   - 应力松弛曲线
   - 判断仿真器是否真的有迟滞行为

3. 模型迟滞建模对比
   - MultiScaleEMA (baseline)
   - HA-EMA (方案 A)
   - Neural ODE (方案 B)
   - 每个模型的迟滞环拟合

4. 消融实验
   - EMA scales 数量: 2/4/8
   - 是否有方向门控
   - loading/unloading decay rate 是否独立

5. 时间序列预测
   - 给定动作序列，预测未来 N 帧的骨架
   - 比较有无迟滞建模的预测精度

6. 频率依赖性分析
   - 低频 vs 高频下的模型精度
   - 迟滞环面积随频率的变化
```

---

## 7. 实现文件清单

| 文件 | 用途 |
|------|------|
| `src/models/layers_hysteresis.py` | HA-EMA / Neural ODE 时序编码器 |
| `src/models/model_ms_scnf_hysteresis.py` | 迟滞感知 MS-SCNF |
| `scripts/data/collect_hysteresis.py` | 迟滞数据采集 |
| `scripts/training/train_hysteresis.py` | 迟滞模型训练 |
| `scripts/training/train_ms_scnf_hysteresis.py` | 端到端训练 |
| `scripts/evaluation/eval_hysteresis.py` | 迟滞评估指标 |
| `scripts/evaluation/plot_hysteresis_loops.py` | 迟滞环可视化 |
| `notebooks/15_hysteresis_analysis.ipynb` | 交互式分析 |
| `notebooks/16_hysteresis_ablation.ipynb` | 消融实验 |

---

## 8. 关键验证：仿真器是否有迟滞？

在投入大量精力建模迟滞之前，**必须先验证 PyElastica 仿真器是否真的表现出迟滞行为**。

### 8.1 验证步骤

```bash
# Step 1: 采集三角波数据
python scripts/data/collect_hysteresis.py --mode triangular --freqs 0.5 --output /tmp/hyst_check

# Step 2: 在 notebook 中分析
# notebooks/15_hysteresis_analysis.ipynb Section 2
```

### 8.2 分析内容

```python
# 在 notebook 中：

# 1. 提取骨架弯曲角随扭矩的变化
torques = actions[:, 0]  # 单轴扭矩
tip_x = positions[:, -1, 0]  # 末端 x 坐标

# 2. 画迟滞环
plt.plot(torques, tip_x)
plt.xlabel('Torque (N·m)')
plt.ylabel('Tip X position (m)')
plt.title('Hysteresis Loop')

# 3. 判断条件:
#    - 如果迟滞环宽度 ≈ 0 → 无迟滞（EMA 足够）
#    - 如果有可观测的迟滞环 → 需要建模
#    - 如果 DAMPING_CONSTANT = 0 → 纯弹性，无迟滞
#    - 如果 DAMPING_CONSTANT > 0 → 有粘弹性，可能有迟滞
```

### 8.3 可能的情况

| 仿真器设置 | 迟滞行为 | 应对策略 |
|------------|---------|---------|
| DAMPING=0, 纯弹性 | 无迟滞 | 当前 EMA 足够，方向 5 意义有限 |
| DAMPING>0 | 有迟滞 | HA-EMA 可以改善 |
| 添加粘弹性材料模型 | 强迟滞 | Neural ODE 更适合 |
| 真实世界 | 必然有迟滞 | 所有方案都需要 |

**关键决策点**：如果仿真器没有迟滞，方向 5 需要修改仿真器（添加粘弹性模型）或在真实世界中验证。

### 8.4 给 PyElastica 添加粘弹性

如果仿真器确实没有迟滞，可以修改：

```python
# elastica_env.py 中添加粘弹性阻尼

class ViscoelasticSoftArmEnv(ContinuousSoftArmEnv):
    """带粘弹性的软体臂仿真环境。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 添加 Maxwell 粘弹性单元
        # σ = E·ε + η·dε/dt
        self.n_maxwell_elements = 3
        self.relaxation_times = [0.1, 0.5, 2.0]  # τ = η/E
        self.internal_variables = [np.zeros((3, 31)) for _ in range(self.n_maxwell_elements)]

    def step(self, steps=1):
        # 标准物理步进
        super().step(steps)

        # 更新粘弹性内变量（Maxwell 模型）
        dt = self.dt
        for i, (tau, z) in enumerate(zip(self.relaxation_times, self.internal_variables)):
            dz = (self.current_strain - z) / tau
            self.internal_variables[i] = z + dz * dt * steps

        # 将粘弹性贡献加到位置上
        for z in self.internal_variables:
            self.simulator.rod.position_collection += z * some_factor
```

---

## 9. 创新点总结

### 9.1 论文贡献

1. **首次识别软体机器人自建模中的迟滞问题**
   - 现有工作（Chen 2022, Shan 2024, Hu 2025）都假设瞬时响应
   - 我们指出这是软体与刚性机器人的本质区别

2. **HA-EMA: 迟滞感知的多尺度时序编码**
   - 加载/卸载独立的 decay rates
   - 方向门控机制
   - 保持 EMA 的 Lipschitz 连续性优势

3. **迟滞数据采集协议与评估指标**
   - 系统化的迟滞表征数据
   - 迟滞环面积、相位滞后等专用指标

### 9.2 与其他方向的协同

| 方向 | 与方向 5 的协同 |
|------|---------------|
| 1: 形态发现 | 迟滞建模是时序问题，形态是空间问题，互补 |
| 2: 纯 2D | 迟滞建模增强 2D 学习的时间一致性 |
| 3: 多相机 | 真实世界迟滞更显著，需要多视角观测 |
| 4: Sim-to-Real | 真实材料的迟滞 > 仿真，是迁移的关键挑战 |

### 9.3 建议的论文故事线

```
1. 软体机器人自建模 ≈ 刚性机器人自建模 + 粘弹性迟滞
2. 现有方法忽略迟滞 → 在动态任务中精度下降
3. 我们提出 HA-EMA → 显著提升动态预测精度
4. 在仿真和真实软体机器人上验证
```

---

## 10. 时间规划

| 阶段 | 时间 | 内容 |
|------|------|------|
| Week 1 | 仿真器迟滞验证 | 确认 PyElastica 是否有迟滞行为 |
| Week 2 | 迟滞数据采集 | 三角波、阶跃、松弛数据 |
| Week 3-4 | HA-EMA 实现 | 方案 A 编码器 + 训练 |
| Week 5-6 | 评估与消融 | 迟滞环可视化 + 消融实验 |
| Week 7-8 | Neural ODE (可选) | 方案 B 实现，如果方案 A 不够 |
