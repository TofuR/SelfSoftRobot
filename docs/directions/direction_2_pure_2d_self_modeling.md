# 方向 2: 纯 2D 自建模 (Pure 2D Self-Modeling)

> 核心思想：真正的自建模机器人只能看到自己的 2D 图像，不应依赖任何 3D GT 数据。这是评估自建模能力的黄金标准。

---

## 1. 问题分析

### 1.1 为什么要去掉 3D GT

当前 MS-SCNF 的优势部分来自 PyElastica 仿真器提供的精确 3D GT 坐标：
- Phase 1 用 3D MSE loss 直接监督骨架回归
- 这相当于告诉机器人"你的骨架在哪里" — 不是自建模

**真正的自建模**（Chen 2022, Hu 2025）只使用：
- 2D 图像（摄像头拍摄）
- 关节/动作指令（自身传感器）

### 1.2 核心挑战

1. **深度歧义**：单视角无法区分 z 轴方向的前后偏移
2. **尺度歧义**：仅从 2D 图像无法确定绝对尺寸
3. **遮挡**：部分形态被自身遮挡
4. **训练信号稀疏**：2D 像素级 loss 比 3D 点级 loss 弱得多

### 1.3 与方向 1 的区别

| | 方向 1 (形态发现) | 方向 2 (纯 2D 自建模) |
|---|---|---|
| 目标 | 发现骨架结构 | 在已有骨架假设下仅用 2D 训练 |
| 骨架 | 可能不固定节点数 | 仍假设 31 节点链式骨架 |
| 重点 | 结构发现能力 | 2D → 3D 的学习能力 |
| 基线 | MS-SCNF (3D GT) | MS-SCNF (3D GT) |

方向 2 可以看作方向 1 路线 A 的扩展，但更系统地研究 2D 监督的全部可能性。

---

## 2. 技术路线

### 2.1 基础方案：MS-SCNF Phase 1→2 合并为单阶段

**核心改动**：将 Phase 1 的 3D skeleton loss 替换为 2D rendering loss。

```
原始 MS-SCNF:
  Phase 1: action → skeleton, loss = MSE(skeleton, GT_3D)      ← 去掉
  Phase 2: action → skeleton → density → render, loss = MSE(render, GT_2D)

纯 2D 方案:
  Single Phase: action → skeleton → density → render, loss = MSE(render, GT_2D)
                                                    + λ * skeleton_regularization
```

### 2.2 增强方案 A：多视角约束

使用 2 个正交相机（xy 轴各一个），从两个视角提供深度约束。

```
Camera 1 (正面, x 轴): 看到 yz 平面的投影
Camera 2 (侧面, y 轴): 看到 xz 平面的投影

两个视角 → 三角化约束 → 消除深度歧义
```

### 2.3 增强方案 B：物理先验正则化

利用软体机器人的物理先验来约束 2D 学习：

1. **长度守恒**：骨架总长度 ≈ 0.5m（材料属性已知）
2. **光滑性**：连续节点间位移平滑（材料连续性）
3. **重力**：base 固定在 z=0，整体向上延伸
4. **不可穿透**：自碰撞检测（骨架不能穿过自身）
5. **局部刚性**：相邻节点距离变化有限

### 2.4 增强方案 C：课程式学习

从简单到困难的训练策略：

```
Stage 1: 小动作范围 (torque ∈ [-0.1, 0.1])
  → 骨架接近直线，容易学习

Stage 2: 中等动作 (torque ∈ [-0.2, 0.2])
  → 弯曲增大

Stage 3: 完整动作范围 (torque ∈ [-0.3, 0.3])
  → 大变形
```

---

## 3. 模型设计

### 3.1 基础模型：MS-SCNF-2D

直接复用 MS-SCNF 架构，仅修改训练 loss。

### 3.2 多视角模型：MS-SCNF-MV

```python
class MSSCNFMultiView(nn.Module):
    """多视角 MS-SCNF — 使用 2 个正交相机消除深度歧义。"""

    def __init__(self, **kwargs):
        super().__init__()
        # 共享的时序编码器和骨架头
        self.temporal = MultiScaleEMA(kwargs['action_dim'],
                                       hidden_dim=kwargs['hidden_dim'])
        self.skeleton_head = SkeletonHead(kwargs['hidden_dim'])

        # 共享的密度场（物理形态唯一，不依赖视角）
        self.density = SkeletonConditionedDensity(
            n_freqs=kwargs['n_freqs'],
            d_filter=kwargs['d_filter'],
        )

    def forward(self, points, action_window):
        """与 MS-SCNF 相同，密度场是视角无关的。"""
        skeleton = self.skeleton_head(self.temporal(action_window))
        return self.density(points, skeleton['fine'])

    def render_view(self, rays_o, rays_d, action_window, near, far, n_samples):
        """渲染特定视角的图像。"""
        pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples)
        raw = self.forward(pts.unsqueeze(0), action_window)
        rendered, _ = OM_rendering(raw.squeeze(0))
        return rendered
```

### 3.3 数据需求

```python
# 数据采集脚本需要支持多视角
# 修改 elastica_env.py 或 collect.py:

# 单视角 (baseline):
python scripts/data/collect.py --n_episodes 200 --output data/sequence_data

# 双视角:
python scripts/data/collect.py --n_episodes 200 --output data/sequence_data_2view \
    --cameras front side
```

---

## 4. 训练方案

### 4.1 基础训练（单视角，无 3D GT）

```bash
python scripts/training/train_ms_scnf_2d.py \
    --data_dir data/sequence_data \
    --mode single_view \
    --n_epochs 300 \
    --lr 5e-4 \
    --batch_size 4 \
    --loss_recon 1.0 \
    --loss_smooth 0.1 \
    --loss_length 0.05 \
    --loss_gravity 0.02 \
    --save_dir train_log/pure_2d/single_view
```

### 4.2 多视角训练

```bash
python scripts/training/train_ms_scnf_2d.py \
    --data_dir data/sequence_data_2view \
    --mode multi_view \
    --n_epochs 300 \
    --loss_recon 1.0 \
    --loss_view_consist 0.5 \
    --loss_smooth 0.1 \
    --loss_length 0.05 \
    --save_dir train_log/pure_2d/multi_view
```

### 4.3 课程式学习

```bash
python scripts/training/train_ms_scnf_2d_curriculum.py \
    --data_dir data/sequence_data \
    --n_stages 3 \
    --stage_epochs 100 \
    --action_ranges 0.1 0.2 0.3 \
    --save_dir train_log/pure_2d/curriculum
```

---

## 5. 实现文件清单

| 文件 | 用途 |
|------|------|
| `src/models/model_ms_scnf_2d.py` | 纯 2D 训练的 MS-SCNF 变体 |
| `src/models/model_ms_scnf_mv.py` | 多视角 MS-SCNF |
| `scripts/training/train_ms_scnf_2d.py` | 纯 2D 训练脚本 |
| `scripts/training/train_ms_scnf_2d_curriculum.py` | 课程式训练脚本 |
| `scripts/data/collect_2view.py` | 双视角数据采集 |
| `notebooks/12_pure_2d_evaluation.ipynb` | 评估 notebook |

---

## 6. 验证方法

### 6.1 定量评估

```bash
# 使用 3D GT（仅评估用，不参与训练）评估骨架精度
python scripts/evaluation/evaluate_3d.py \
    --checkpoint train_log/pure_2d/single_view/model/best_model.pt \
    --data_dir data/sequence_data_3d \
    --save_dir output/pure_2d_eval

# 多视角对比
python scripts/evaluation/compare_views.py \
    --checkpoints train_log/pure_2d/single_view train_log/pure_2d/multi_view \
    --data_dir data/sequence_data_3d \
    --save_dir output/view_comparison
```

### 6.2 关键对比实验

| 实验 | 目的 |
|------|------|
| MS-SCNF (3D GT) vs 纯 2D | 量化 3D GT 的贡献 |
| 单视角 vs 双视角 | 量化多视角的价值 |
| 无正则化 vs +物理先验 | 量化物理先验的价值 |
| 无课程 vs 课程式 | 量化课程学习的价值 |

### 6.3 Notebook 验证

`12_pure_2d_evaluation.ipynb`:
```
1. 加载 3D GT 模型和纯 2D 模型
2. 对比骨架精度 (MNE, EPE, CD)
3. 可视化深度歧义：
   - 单视角模型的 z 轴误差分布
   - 双视角模型的 z 轴误差分布
4. 渲染质量对比 (PSNR, SSIM)
5. 不同正则化的消融实验
6. 课程式学习的训练曲线
```

---

## 7. 预期结果

### 7.1 预期性能

| 方法 | MNE (m) | 渲染 PSNR (dB) |
|------|---------|----------------|
| MS-SCNF (3D GT) | ~0.005 | ~25 |
| 纯 2D (单视角) | ~0.02-0.05 | ~20 |
| 纯 2D (双视角) | ~0.01-0.02 | ~22 |
| 纯 2D (双视角 + 课程) | ~0.008-0.015 | ~23 |

### 7.2 关键结论

- 单视角纯 2D 有明显的深度歧义
- 双视角可以大幅缓解深度问题
- 物理先验正则化进一步提升精度
- 这条路线的价值在于**量化了 2D 自建模的能力边界**

---

## 8. 创新点

1. **首个系统研究软体机器人 2D 自建模能力边界的工作**
2. **多视角约束 + 物理先验的组合**用于纯 2D 软体机器人建模
3. **课程式学习策略**用于软体机器人的大变形学习
4. 为后续 sim-to-real 提供纯 2D 训练基线
