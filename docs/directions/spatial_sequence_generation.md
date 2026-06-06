# 方向灵感：空间序列生成 (Spatial Sequential Propagation)

> 来源：PROJECT_HELP.md 中的结构归纳偏置思考
> 核心思想：将软臂沿 Z 轴切片为空间序列，用状态空间模型自下而上传递空间记忆

---

## 一、动机：为什么点云生成是个坏选择？

### 软体机器人的强结构先验

软臂本质上是一个**高度结构化的连续体**：
- 沿 Z 轴（主轴）有明确的**空间因果关系**：底部固定 → 中段跟随 → 尖端响应最大
- 每个横截面（z-slice）近似圆形，半径沿 Z 缓慢变化
- 变形沿着 Z 轴**传播**：底部状态决定中段，中段决定尖端
- 物理上，这是悬臂梁模型：弯矩从底到尖累积

### 纯点云生成的致命缺陷

当前 Flow Matching 把所有点视为**无序集合**：
- 1000 个点之间没有空间关系约束
- 底部的点和尖端的点被同等对待
- 模型不知道"哪些点应该在一起形成截面"
- 导致扇形发散：尖端点失去了与底部点的结构关联

```
当前：1000个独立的3D点 ← 无结构，无约束
```

**关键洞察**：我们不需要生成 1000 个"独立的点"，而是需要生成**沿 Z 轴传播的 N 个截面**。

---

## 二、方法设计：空间序列生成

### 核心思想

```
Action (全局条件)
    ↓
Z-slice 0 (底部) → Z-slice 1 → Z-slice 2 → ... → Z-slice N (尖端)
    截面参数         状态传递      状态传递              截面参数
  (中心+半径+偏移)                                    (中心+半径+偏移)
```

每个 z-slice 的参数：
- **中心点** (x, y)：截面的几何中心
- **半径** r：截面大小（软臂半径，近似恒定）
- **朝向** θ：截面朝向（扭转信息）

### 为什么这比点云好？

| 维度 | 点云生成 | 空间序列生成 |
|------|---------|------------|
| 输出维度 | (N, 3) = 3000 | (K, 5) = ~50（K=10个截面） |
| 空间关系 | 无 | 强因果链（底→尖） |
| 扇形问题 | 容易发散 | 被空间记忆收束 |
| 物理先验 | 无 | 悬臂梁传播 |
| 可解释性 | 低 | 高（每个截面有明确含义） |

---

## 三、架构选项

### 选项 A：Structured State Space Model (S4/Mamba)

```python
class SpatialSequenceModel:
    def __init__(self):
        self.action_encoder = MultiScaleEMA(...)      # action → 全局条件 c
        self.mamba = MambaBlock(d_model=hidden_dim)    # 沿 Z 轴的状态传递
        self.slice_head = nn.Linear(hidden_dim, 5)     # 每个截面: (cx, cy, r, θx, θy)

    def forward(self, action_window):
        c = self.action_encoder(action_window)
        z_positions = torch.linspace(0, L, n_slices)
        z_embed = self.z_position_embed(z_positions)
        sequence = z_embed + c.unsqueeze(1)
        states = self.mamba(sequence)                  # 空间因果传递
        slice_params = self.slice_head(states)
        return slice_params
```

### 选项 B：简单 GRU 沿 Z 轴（推荐先试这个）

```python
class SpatialGRU:
    def __init__(self):
        self.action_encoder = MultiScaleEMA(...)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.slice_head = nn.Linear(hidden_dim, 5)

    def forward(self, action_window):
        c = self.action_encoder(action_window)
        h = self.init_hidden(c)
        slices = []
        for z_pos in z_positions:
            z_emb = self.z_embed(z_pos)
            h = self.gru(z_emb + c, h)  # 状态沿 Z 传递
            slices.append(self.slice_head(h))
        return torch.stack(slices, dim=1)
```

**推荐理由**：极简实现，无额外依赖，序列长度只有 10-20 步。

---

## 四、损失函数

```python
# 1. 截面参数直接回归
loss_center = MSE(pred_centers, gt_centers)
loss_radius = MSE(pred_radii, gt_radii)

# 2. 采样点云的 CD（兼容已有评估）
pred_pc = slices_to_pointcloud(pred_params)
loss_cd = chamfer_distance(pred_pc, gt_pc)

# 3. 空间平滑
loss_smooth = MSE(params[1:] - params[:-1])

# 注意：不再需要 compactness loss！
# 扇形问题自动消失——每个截面被参数化为圆，不可能发散
```

---

## 五、与现有代码的关系

| 已有模块 | 新用途 |
|---------|--------|
| `MultiScaleEMA` | 保持不变，编码 action → 全局条件 |
| `skeleton_heads.py` | 复用为 z-slice 之间的插值 |
| `sdf_utils.py` | GT 截面参数提取 |
| Flow Matching velocity_net | **替换为**空间序列模型 |

---

## 六、实施难度

| 方面 | 难度 |
|------|------|
| 数据准备 | ⭐⭐ 需要从 GT positions 提取截面参数 |
| 模型 | ⭐⭐ GRU 极简，Mamba 需引入依赖 |
| 训练 | ⭐ 输出从 3000→~50 维，更快更稳 |
| 消除扇形 | ⭐ **自动消除** |

**总结**：低风险、高回报。扇形问题从架构层面根本消失，而非靠 loss 补丁。
