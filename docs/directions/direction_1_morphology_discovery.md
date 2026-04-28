# 方向 1: 形态发现 — 从图像到骨架 (Image → Skeleton Discovery)

> 核心问题：当前 MS-SCNF 假设了 31 个节点的骨架结构和圆柱体形态，这不是真正的自建模。真正的自建模应该让机器人从观测数据中**发现自身结构**。

---

## 1. 问题分析

### 1.1 当前假设 vs 真正的自建模

| 维度 | 当前 MS-SCNF | 真正的自建模 |
|------|-------------|-------------|
| 骨架节点数 | 硬编码 31 (N_ELEMENTS+1) | 从数据中发现 |
| 骨架拓扑 | 硬编码为 1D 链式 | 发现分支/环/链 |
| 截面形状 | 假设圆柱体 | 从图像中学习 |
| 3D 监督 | 使用仿真器 GT | 仅从 2D 图像学习 |

### 1.2 关键挑战

1. **深度歧义**：单视角 2D 图像无法唯一确定 3D 骨架
2. **节点数量不确定**：网络不知道应该输出多少个节点
3. **拓扑不确定**：机器人可能是链式、树状、甚至环形
4. **评估困难**：没有 GT 骨架，难以定量评估发现的骨架质量

### 1.3 可行路线

三条子路线，难度递增：

- **路线 A**：固定节点数，但**不使用 3D GT 监督**，仅从 2D 渲染 loss 学习骨架
- **路线 B**：学习可变数量的骨架点 + 从图像推断节点数
- **路线 C**：完全无监督的形态发现（类似 Chen 2022 的 SDF 学习）

---

## 2. 路线 A：2D 监督的骨架学习

### 2.1 思路

保持 MS-SCNF 的骨架结构，但**去掉 Phase 1 的 3D GT loss**，完全依赖 Phase 2 的 2D 渲染 loss 来学习骨架。

关键改动：
- Phase 1: skeleton MSE loss → **rendering loss**（骨架监督来自渲染质量）
- 增加骨架正则化：平滑性约束 + 长度约束（物理先验）
- 可能需要多视角（2 个相机）来消除深度歧义

### 2.2 模型设计

```
与 MS-SCNF 相同，但训练流程改为：

Phase 1 (Skeleton from Rendering):
  action_window → MultiScaleEMA → SkeletonHead → skeleton (coarse/medium/fine)
  skeleton + query_points → SkeletonConditionedDensity → [vis, density]
  → Volume Rendering → rendered_image
  → Loss = MSE(rendered, GT_image) + λ_smooth * smoothness(skeleton)
                                        + λ_length * length_preservation(skeleton)

Phase 2 (Fine-tune Density):
  Phase 1 的权重初始化 + 继续训练密度场
```

**新增强正则化**：

```python
def length_preservation_loss(skeleton, rest_length=0.5):
    """骨架总长度应接近物理长度 (0.5m)。"""
    segments = skeleton[:, 1:] - skeleton[:, :-1]  # (B, N-1, 3)
    total_length = segments.norm(dim=-1).sum(dim=-1)  # (B,)
    return ((total_length - rest_length) ** 2).mean()

def gravity_prior_loss(skeleton):
    """骨架 base 应在 z≈0, 且整体向上延伸。"""
    base_z = skeleton[:, 0, 2]  # base 的 z 坐标
    tip_z = skeleton[:, -1, 2]  # tip 的 z 坐标
    return (base_z ** 2).mean() + F.relu(-tip_z).mean()  # base 接近 0, tip > 0
```

### 2.3 实现文件

| 文件 | 用途 |
|------|------|
| `src/models/model_ms_scnf_2d.py` | 修改版 MS-SCNF（2D 监督骨架学习） |
| `scripts/training/train_ms_scnf_2d.py` | 训练脚本 |
| `notebooks/09_skeleton_from_2d.ipynb` | 验证 notebook |
| `scripts/evaluation/eval_skeleton_discovery.py` | 评估脚本 |

### 2.4 训练步骤

```bash
# Step 1: 仅用 2D 图像训练骨架 (无 3D GT)
python scripts/training/train_ms_scnf_2d.py \
    --data_dir data/sequence_data \
    --phase 1 \
    --n_epochs 200 \
    --loss_recon 1.0 \
    --loss_smooth 0.1 \
    --loss_length 0.05 \
    --loss_gravity 0.02 \
    --save_dir train_log/ms_scnf_2d_skeleton

# Step 2: 微调密度场 (可选：加入 3D GT 对比)
python scripts/training/train_ms_scnf_2d.py \
    --data_dir data/sequence_data \
    --phase 2 \
    --load_dir train_log/ms_scnf_2d_skeleton/phase1 \
    --n_epochs 100 \
    --save_dir train_log/ms_scnf_2d_skeleton/phase2
```

### 2.5 验证方法

```bash
# 对比 2D 学到的骨架 vs GT 3D 骨架（如果有 3D 数据）
python scripts/evaluation/eval_skeleton_discovery.py \
    --checkpoint train_log/ms_scnf_2d_skeleton/phase1/model/best_model.pt \
    --data_dir_2d data/sequence_data \
    --data_dir_3d data/sequence_data_3d \
    --save_dir output/skeleton_discovery_eval
```

### 2.6 预期结果与风险

**预期**：
- 单视角下，骨架可能存在深度方向偏移（z 轴歧义）
- 渲染质量可以接近 MS-SCNF Phase 2
- 如果加入第 2 个视角，骨架 3D 精度会显著提升

**风险**：
- 单视角深度歧义可能导致骨架退化（投影到相机平面）
- 需要仔细调节正则化权重

---

## 3. 路线 B：可变节点数的骨架发现

### 3.1 思路

不固定节点数为 31，而是让网络自己决定需要多少个控制点来描述形态。

### 3.2 模型设计

**核心模块：Adaptive Skeleton Head**

```python
class AdaptiveSkeletonHead(nn.Module):
    """自适应骨架头 — 从图像特征中发现骨架控制点。"""

    def __init__(self, hidden_dim=128, max_nodes=50, node_dim=3):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_dim = node_dim

        # 预测 N 个候选节点
        self.node_generator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_nodes * node_dim),
        )

        # 预测每个节点的置信度（用于选择有效节点）
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_nodes),
            nn.Sigmoid(),
        )

    def forward(self, physics_state):
        """
        Args:
            physics_state: (B, hidden_dim)
        Returns:
            nodes: (B, max_nodes, 3) — 候选节点坐标
            confidence: (B, max_nodes) — 节点置信度
        """
        B = physics_state.shape[0]
        nodes = self.node_generator(physics_state).reshape(B, self.max_nodes, self.node_dim)
        confidence = self.confidence_head(physics_state)
        return nodes, confidence
```

**训练目标**：

```python
def adaptive_skeleton_loss(nodes, confidence, rendered_img, gt_img, rest_length=0.5):
    """
    nodes: (B, N, 3) — 候选节点
    confidence: (B, N) — 节点置信度
    """
    # 1. 渲染 loss（驱动节点位置学习）
    render_loss = F.mse_loss(rendered_img, gt_img)

    # 2. 稀疏性：鼓励使用更少的节点
    sparsity_loss = confidence.mean()

    # 3. 有效节点的骨架正则化
    valid_nodes = nodes * confidence.unsqueeze(-1)  # 加权
    smooth_loss = curve_smoothness(valid_nodes)
    length_loss = length_preservation_loss(valid_nodes, rest_length)

    # 4. 节点排序：置信度高的节点应该连续
    order_loss = node_ordering_loss(nodes, confidence)

    return render_loss + 0.01 * sparsity_loss + 0.1 * smooth_loss + 0.05 * length_loss
```

### 3.3 实现文件

| 文件 | 用途 |
|------|------|
| `src/models/model_adaptive_skeleton.py` | 自适应骨架模型 |
| `scripts/training/train_adaptive_skeleton.py` | 训练脚本 |
| `notebooks/10_adaptive_skeleton.ipynb` | 验证 notebook |

### 3.4 训练步骤

```bash
# 训练自适应骨架模型
python scripts/training/train_adaptive_skeleton.py \
    --data_dir data/sequence_data \
    --max_nodes 50 \
    --n_epochs 300 \
    --loss_render 1.0 \
    --loss_sparsity 0.01 \
    --loss_smooth 0.1 \
    --save_dir train_log/adaptive_skeleton
```

### 3.5 验证

```bash
# 可视化发现的骨架
python scripts/evaluation/eval_skeleton_discovery.py \
    --checkpoint train_log/adaptive_skeleton/model/best_model.pt \
    --data_dir data/sequence_data \
    --mode adaptive \
    --save_dir output/adaptive_skeleton_eval
```

在 notebook `10_adaptive_skeleton.ipynb` 中验证：
1. 可视化不同动作下发现的骨架
2. 分析置信度分布 → 网络认为需要多少个节点
3. 与 GT 31 节点骨架对比

---

## 4. 路线 C：SDF 形态学习（类 Chen 2022）

### 4.1 思路

参考 Chen 2022 的方法：不预设骨架，直接学习 SDF（Signed Distance Field）。

核心区别：
- Chen 用 5 个 RGB-D 相机 → 3D 点云 → SDF 监督
- 我们只有 2D 灰度图 → 需要从体渲染中隐式学习 SDF

### 4.2 模型设计

```
action_window → MultiScaleEMA → physics_state (128d)

查询 3D 点 x:
  SDF_network(x, physics_state) → signed_distance, feature

体渲染 (VolSDF / NeuS 风格):
  signed_distance → density → rendering
```

**关键改进**：加入 Eikonal 正则化（确保 SDF 有效）和 curvature 正则化（保证光滑）。

```python
class SDFConditionedField(nn.Module):
    """SDF 形态场 — 无骨架假设。"""

    def __init__(self, action_dim, hidden_dim=128, d_filter=128, n_freqs=6):
        super().__init__()
        self.temporal = MultiScaleEMA(action_dim, hidden_dim=hidden_dim)
        self.pos_encoder = PositionalEncoder(3, n_freqs=n_freqs)
        self.sdf_network = MLPDecoder(
            input_dim=self.pos_encoder.output_dim + hidden_dim,
            d_filter=d_filter,
            output_size=1,  # SDF 值
        )
        self.feature_network = nn.Linear(d_filter // 2, 1)  # visibility

    def forward(self, points, action_window):
        """
        Args:
            points: (B, N_rays, N_samples, 3)
            action_window: (B, K, D)
        Returns:
            sdf: (B*N_rays, N_samples) — signed distance
            vis: (B*N_rays, N_samples) — visibility
        """
        state = self.temporal(action_window)  # (B, hidden_dim)

        B, N_rays, N_samples, _ = points.shape
        pts = points.reshape(-1, N_samples, 3)

        pos_enc = self.pos_encoder(pts)  # (B*N_rays, N_samples, enc_dim)
        state_exp = state.unsqueeze(1).expand(-1, N_samples, -1)
        combined = torch.cat([pos_enc, state_exp], dim=-1)

        sdf = self.sdf_network(combined).squeeze(-1)
        vis = torch.sigmoid(self.feature_network(
            self.sdf_network.network[-2](combined)  # 倒数第二层的特征
        )).squeeze(-1)

        # SDF → density (VolSDF conversion)
        sigma = self.sdf_to_density(sdf)

        return vis.reshape(B * N_rays, N_samples), sigma.reshape(B * N_rays, N_samples)
```

### 4.3 实现文件

| 文件 | 用途 |
|------|------|
| `src/models/model_sdf.py` | SDF 形态场模型 |
| `scripts/training/train_sdf.py` | 训练脚本 |
| `notebooks/11_sdf_morphology.ipynb` | 验证 notebook |

### 4.4 训练步骤

```bash
python scripts/training/train_sdf.py \
    --data_dir data/sequence_data \
    --n_epochs 300 \
    --loss_recon 1.0 \
    --loss_eikonal 0.1 \
    --loss_curvature 0.01 \
    --save_dir train_log/sdf_morphology
```

### 4.5 验证

```bash
# 提取 SDF 零水平集 → 3D 形态
python scripts/evaluation/eval_sdf_morphology.py \
    --checkpoint train_log/sdf_morphology/model/best_model.pt \
    --data_dir data/sequence_data \
    --save_dir output/sdf_eval
```

---

## 5. 路线对比与选择建议

| 路线 | 难度 | 创新性 | 实用性 | 推荐顺序 |
|------|------|--------|--------|----------|
| A: 2D 骨架学习 | ★★☆ | ★★☆ | ★★★ | **第 1 个尝试** |
| B: 可变节点 | ★★★ | ★★★ | ★★☆ | 第 2 个尝试 |
| C: SDF 形态 | ★★★★ | ★★★ | ★★★ | 第 3 个尝试 |

**推荐路线 A 先行**：
1. 复用现有 MS-SCNF 架构，改动最小
2. 结果可直接与 MS-SCNF (3D GT) 对比，量化 2D 监督的差距
3. 为路线 B/C 提供基线

---

## 6. Notebook 验证计划

### `09_skeleton_from_2d.ipynb`

```
1. 加载路线 A 训练的模型
2. 提取学到的骨架，与 GT 3D 骨架对比
   - 如果有 3D 数据：计算 MNE, EPE, CD
   - 可视化 GT vs Pred 骨架
3. 渲染质量对比
   - GT image vs Rendered image
   - PSNR, SSIM
4. 深度歧义分析
   - 可视化 z 轴方向误差
   - 统计 xy vs z 方向误差比例
5. 正则化消融实验
   - 无正则化 / +smooth / +length / +gravity
   - 每种配置的骨架质量对比
```

### `10_adaptive_skeleton.ipynb`

```
1. 加载自适应骨架模型
2. 可视化不同动作下的置信度分布
3. 分析网络自动选择的节点数
4. 可视化高置信度节点构成的骨架
5. 与 GT 31 节点对比
```

### `11_sdf_morphology.ipynb`

```
1. 加载 SDF 模型
2. 提取 SDF 零水平集 → 3D 形态
3. 不同动作下的形态变化
4. 与 Chen 2022 的方法对比（如有）
```
