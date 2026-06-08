# 方向灵感：拓扑引导的残差流匹配 (Topology-Guided Residual Flow)

> 来源：PROJECT_HELP.md 中的结构归纳偏置思考
> 核心思想：先用物理先验做粗变形，再用 Flow Matching 学习残差细节

---

## 一、动机：为什么让噪声"从零开始"是低效的？

### 当前 Flow Matching 的问题

```
高斯噪声 X₀ ~ N(0, σ²I)  ──[ODE 积分 50步]──→  预测点云 X̂
```

模型需要同时学会三个层次：
1. "软臂大致是圆柱形"（粗结构）
2. "在当前 action 下怎么弯曲"（条件变形）
3. "表面细节长什么样"（精细几何）

三个层次混在一起，模型把容量浪费在学习先验知识上。

### 物理直觉

我们知道软臂的拓扑结构：
- 底部固定，尖端自由
- 截面近似圆形
- 变形是光滑的弯曲/扭转
- 零 action 下是**直的圆柱体**

既然知道这么多先验，为什么不让模型从这些知识出发？

---

## 二、核心设计：两阶段残差流

```
阶段 1: 物理先验粗变形
    基础圆柱体 + Action → MLP/仿射变换 → 粗变形点云 X_coarse

阶段 2: Flow Matching 学习残差
    X_coarse ──[ODE]──→ X_coarse + ΔX_residual
```

模型只需学习**残差** ΔX = X_gt - X_coarse。

| 维度 | 纯 FM | 残差 FM |
|------|-------|---------|
| 起点 | 高斯噪声 | 物理粗变形 |
| 需学习 | 全部 | 仅残差 |
| 扇形风险 | 高 | 低 |
| 训练难度 | 高 | 低 |

---

## 三、阶段 1 设计选项

### 选项 A：MLP 直接预测粗偏移（最简，推荐先试）

```python
class CoarseDeformation:
    def forward(self, action_window, base_cylinder):
        offsets = self.offset_mlp(ema_encoding, z_positions)
        return base_cylinder + offsets
```

### 选项 B：参数化骨架弯曲（更有物理意义）

```python
class CoarseDeformation:
    def forward(self, action_window, base_cylinder):
        bending_angles = self.angle_head(ema_encoding)  # 每段弯曲角
        # 悬臂梁公式：offset ∝ z² × angle
        for z_idx in range(n_slices):
            base_cylinder[:, z_idx, 0] += bending_angles[z_idx] * z[z_idx]**2
        return base_cylinder
```

### 选项 C：逐截面仿射变换矩阵

```python
class CoarseDeformation:
    def forward(self, action_window, base_cylinder):
        transforms = self.transform_head(ema_encoding)  # (B, n_slices, 6)
        return apply_per_slice_transform(base_cylinder, transforms)
```

---

## 四、阶段 2：残差 Flow Matching

```python
def compute_losses(self, batch, phase_spec):
    # 阶段 1: 粗变形
    x_coarse = self.coarse_deformation(action_window, base_cylinder)

    # 阶段 2: 残差目标
    residual_target = gt_pc - x_coarse.detach()

    # OT-sort + 插值（目标改为残差）
    X0 = torch.randn(B, N, 3) * sigma
    t = torch.rand(B, 1)
    X_t = (1 - t) * X0 + t * residual_target
    u_target = residual_target - X0

    u_pred = self.velocity_net(X_t + x_coarse, t, cond, action=current_action)
    loss_fm = MSE(u_pred, u_target)
```

推理时：
```python
x_coarse = self.coarse_deformation(action_window, base_cylinder)
residual = ode_solve(self.velocity_net, X0, cond, n_steps, action=current_action)
return x_coarse + residual  # 粗变形 + 残差 = 最终点云
```

---

## 五、为什么扇形会消失？

1. **粗变形已给出合理弯曲**：尖端已大致在正确位置
2. **残差只需小幅修正**：幅度远小于完整点云
3. **即使 ODE 不完美，基础形状已被物理先验约束**

```
纯 FM:  噪声 → ??? → 扇形（尖端位置不确定）
残差 FM: 物理弯曲 → 小幅修正 → 保持结构
```

---

## 六、与空间序列生成的组合

两者可以**叠加使用**（最强方案）：

```
空间序列生成 → 预测 N 个截面参数 → 生成粗点云
    ↓
残差 Flow Matching → 补充参数化无法表达的细节
```

- 阶段 1：空间序列保证结构，扇形从架构层面消失
- 阶段 2：Flow Matching 补充超出参数化能力的精细变形

---

## 七、实施难度

| 方面 | 难度 |
|------|------|
| 基础圆柱体 | ⭐ |
| 粗变形模块 | ⭐⭐ |
| 残差 Flow | ⭐⭐ 改目标即可 |
| 训练 | ⭐ 分阶段训练 |
| 消除扇形 | ⭐⭐ 大幅减少 |

**总结**：与当前架构改动最小（改 Flow Matching 目标），可快速验证。最强方案是与空间序列生成组合。
