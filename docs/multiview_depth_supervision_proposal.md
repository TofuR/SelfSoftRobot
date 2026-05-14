# 多视角 + 深度监督方案：提升软体机器人自建模的 3D 几何精度与时序一致性

> 日期：2026-05-13
> 状态：分析方案，实施待定

## 1. 背景与问题分析

### 1.1 当前架构的局限

当前所有模型（MSTNF / CMSTNF / Smooth-CMSTNF / MS-SCNF）的训练监督信号来源：

| 监督信号 | MSTNF | CMSTNF | Depth-CMSTNF | MS-SCNF |
|----------|-------|--------|--------------|---------|
| 单视角二值图 | ✓ | ✓ | ✓ | ✓ |
| 3D 骨架坐标 | ✗ | ✗ | ✗ | ✓ |
| 深度图 | ✗ | ✗ | ✓ | ✗ |
| 时序平滑 | ✓ | ✓ | ✓ | ✓ |
| 多视角 | ✗ | ✗ | ✗ | ✗ |

**核心问题**：单视角二值图监督只提供 2D 投影信息，对深度方向约束极弱。体积渲染时，沿射线方向存在大量等价解（不同密度分布可以产生相同的 2D 投影），导致：

1. **3D 几何模糊**：模型可以"欺骗" 2D loss 而不学到正确的 3D 形状
2. **深度歧义**：沿视线方向的厚度无法被单视角约束
3. **时序抖动**：3D 形状在帧间不稳定，因为 2D loss 对深度变化不敏感

### 1.2 已有基础设施

代码库中已部分实现了相关功能：

- **深度监督**：`DepthCMSTNFTrainer` — 深度 loss + 深度引导采样（仅 CMSTNF）
- **多视角采集**：`collect_multiview.py` — 双视角（front + side）数据采集
- **多视角数据集**：`MultiViewDataset` — 双视角图片 + 2D 骨架
- **深度渲染**：`OM_rendering_with_depth()` — 从密度场计算期望深度图

但这些功能是分散的，未统一整合到完整的多视角训练流程中。

### 1.3 硬件假设

- 真实机器人场景，部署 **2-3 个固定相机**，同步采集
- 相机参数已知（内外参标定）
- 可获取二值图（背景分割）和深度图（RGB-D 相机或立体视觉）

## 2. 三个递进方案

### 方案 A：多视角 + 深度监督融合（增量式）

#### 2A.1 核心思路

不改模型架构，在现有训练流程中加入多视角 rendering loss 和深度 loss。每个训练 step 同时从多个视角做 volume rendering，将各视角的 loss 求和。

#### 2A.2 修改清单

**数据层**：
- `collect_multiview.py` 增加 `--depth` 标志，同时保存各视角深度图
- `MultiViewDataset` 扩展：加载多视角图片 + 各相机参数 + 深度图
- 数据 npz 格式扩展：
  ```
  images_front, images_side, ...      # 各视角二值图
  depth_front, depth_side, ...        # 各视角深度图
  camera_params_front, camera_params_side, ...  # 各相机参数
  actions, positions                   # 不变
  ```

**模型层**：无改动。模型 `forward(points, action_window)` 与视角无关，天然支持多视角查询。

**训练层**：
- 新增 `MultiViewTrainer`，继承 `BaseTrainer`：
  ```python
  # 伪代码
  for each view_i in views:
      rays_o_i, rays_d_i = get_rays(camera_params_i)
      raw_i = model(sample_points(rays_o_i, rays_d_i), action_window)
      rendered_i = volume_render(raw_i)
      depth_i = volume_render_depth(raw_i)

      loss_recon += MSE(rendered_i, gt_image_i)
      loss_depth += L1(depth_i, gt_depth_i)

  loss = loss_recon / n_views + w_depth * loss_depth + w_smooth * smoothness_loss
  ```

- 将 `DepthCMSTNFTrainer` 的深度引导采样（coarse-to-fine）通用化，移到 `BaseTrainer`

#### 2A.3 优缺点

| 维度 | 评估 |
|------|------|
| 模型改动 | 无，完全复用现有模型 |
| 训练改动 | 中等，新增一个 Trainer 类 |
| 数据要求 | 需要多视角 + 深度数据 |
| 计算开销 | 随视角数线性增加（每视角独立 rendering） |
| 几何提升 | 中等，多 loss 缓解深度歧义但无显式几何约束 |
| 时序提升 | 间接改善，更准确的几何有助于时序稳定性 |
| 实施时间 | 1-2 周 |

#### 2A.4 适用场景

快速验证多视角 + 深度的效果，确定哪些组件最有价值，为后续更复杂的方案奠定基础。

---

### 方案 B：多视角一致性约束（推荐）

#### 2B.1 核心思路

在方案 A 基础上，加入显式的跨视角几何约束，确保同一 3D 点在不同视角下的预测一致。

#### 2B.2 约束设计

**约束 1：多视角渲染 Loss（来自方案 A）**
- 多视角同时 rendering，各视角 MSE loss 求和
- $L_{\text{recon}} = \sum_{i=1}^{V} \text{MSE}(R_i(\theta), I_i)$

**约束 2：深度监督 Loss**
- 渲染深度图与 GT 深度图 L1 loss
- $L_{\text{depth}} = \sum_{i=1}^{V} \text{L1}(D_i(\theta), D_i^{\text{gt}})$
- 仅在有效深度区域（foreground mask 内）计算

**约束 3：跨视角 SDF/Density 一致性**

这是与方案 A 的关键区别。对同一 3D 点，从不同视角的射线交汇于该点时，模型应给出相同的 density/SDF 值。

实现方式：
```python
# 选择多个视角射线交汇的区域（高 density 区域）
# 对每个交汇点，从不同视角的射线分别查询
# 强制它们的 density/SDF 预测一致
for point in intersection_points:
    values = [model.query(point, action) for _ in views]  # 同一点多次查询
    consistency_loss += variance(values)
```

**约束 4：重投影一致性**

用视角 A 的深度预测，将点投影到视角 B，检查重投影误差：
```python
# 视角 A 渲染深度 → 3D 点
points_3d = unproject(rays_o_A, rays_d_A, depth_A)
# 投影到视角 B
points_2d_B = project(points_3d, camera_B)
# 检查与视角 B 的图像一致性
loss_reproj += MSE(render_from_B_at(points_2d_B), image_B_at(points_2d_B))
```

#### 2B.3 修改清单

**数据层**：
- 同方案 A 的数据扩展
- 额外需要各相机之间的相对位姿（用于重投影计算）

**模型层**：
- 无核心架构改动
- 可选优化：增加一个 `query_density(point, action)` 接口，便于单独查询 3D 点的 density（用于一致性约束）

**训练层**：
- `MultiViewConsistencyTrainer`，核心训练循环：
  ```
  每个训练 step:
  1. 多视角 rendering + 2D loss（约束 1）
  2. 深度 rendering + depth loss（约束 2）
  3. 采样交汇点 + 一致性 loss（约束 3）
  4. 重投影验证 + 重投影 loss（约束 4）
  5. 时序 smoothness loss（现有）
  ```

**采样策略优化**：
- 深度引导采样：第一阶段均匀采样，第二阶段在深度估计附近密集采样
- 交汇点采样：选择多视角射线都经过的高 density 区域

#### 2B.4 优缺点

| 维度 | 评估 |
|------|------|
| 模型改动 | 极小，可选增加 query 接口 |
| 训练改动 | 较大，新增 4 种约束 loss |
| 数据要求 | 同方案 A + 相机相对位姿 |
| 计算开销 | 比方案 A 多 20-30%（交汇点查询 + 重投影） |
| 几何提升 | 显著，显式几何约束大幅减少深度歧义 |
| 时序提升 | 间接但明显，几何一致性有助于时序稳定 |
| 实施时间 | 2-4 周 |

#### 2B.5 适用场景

在方案 A 验证有效后，进一步提升 3D 几何精度。平衡效果和改动量。

---

### 方案 C：混合 SDF + 密度场架构（重设计）

#### 2C.1 核心思路

将当前的纯 density 场替换为 SDF + density 混合表示。SDF 天然定义表面（SDF=0），density 从 SDF 转换而来用于体渲染。这样多视角一致性是内在保证的——同一 3D 点只有一个 SDF 值。

#### 2C.2 架构设计

**模型输出变化**：
```
当前: forward(points, action) → [visibility, density]
改为: forward(points, action) → [SDF, feature]

density = σ(SDF / β)  # β 是可学习的软性参数
rendered = volume_render(density, feature)  # 正常体渲染
```

**新增 Loss**：

1. **Eikonal Loss**：约束 SDF 梯度模为 1
   $L_{\text{eikonal}} = (||\nabla_x \text{SDF}(x)|| - 1)^2$

2. **SDF 表面 Loss**（可选）：在已知 3D 骨架点处约束 SDF 值
   $L_{\text{skeleton}} = \text{SDF}(x_{\text{skeleton}})^2$

3. **深度 Loss**：渲染深度与 GT 深度
   $L_{\text{depth}} = \text{L1}(D(\theta), D^{\text{gt}})$

4. **多视角渲染 Loss**：同方案 A/B

**SDF → Density 转换**：

采用 VolSDF 的方法：
```python
def sdf_to_density(sdf, beta):
    # 使用拉普拉斯分布的 CDF
    return (0.5 + 0.5 * sdf * torch.sigmoid(sdf / beta)) / beta
```

#### 2C.3 具体架构变化

**适用模型**：

以 CMSTNF 为例改造：

```
原 CanonicalField:
  input: canonical_points + action
  output: [visibility, density]

改为 CanonicalSDF:
  input: canonical_points + action
  output: [SDF]
  → sdf_to_density(SDF, learnable_beta)
  → volume_render(density, 1.0)  # visibility 恒为 1，由 density 控制可见性
```

以 MS-SCNF 为例改造：

```
原 SkeletonConditionedDensity:
  input: query_points + skeleton + action
  output: [visibility, density]

改为 SkeletonConditionedSDF:
  input: query_points + skeleton + action
  output: [SDF]
  → 骨架附近的 SDF 值可解析初始化（点到骨架距离 - 半径）
  → sdf_to_density(SDF, learnable_beta)
```

#### 2C.4 训练流程变化

**Phase 1（Canonical/Skeleton）**：
- 学习 SDF 的零等值面形状
- Loss：2D rendering + Eikonal + 深度（可选）
- SDF 的零等值面天然定义了机器人的静止形状

**Phase 2（Deformation）**：
- DeformationField 学习 3D 位移场
- Loss：多视角 rendering + Eikonal + 深度 + 时序 smoothness
- 变形后的 SDF 场仍然满足 Eikonal 约束

#### 2C.5 优缺点

| 维度 | 评估 |
|------|------|
| 模型改动 | 大，输出层和 rendering 逻辑重写 |
| 训练改动 | 大，新增 SDF 相关 loss |
| 数据要求 | 同方案 A |
| 计算开销 | 增加约 30%（Eikonal loss 需要二阶梯度） |
| 几何提升 | 最优，SDF 是最精确的 3D 表示 |
| 时序提升 | 最优，SDF 的梯度约束天然增强平滑性 |
| 实施时间 | 4-8 周 |

#### 2C.6 适用场景

在方案 A/B 验证多视角+深度确实有效后，进行架构升级。或者直接作为长期目标。

## 3. 方案对比总览

| 维度 | 方案 A（增量） | 方案 B（一致性） | 方案 C（SDF 混合） |
|------|---------------|-----------------|-------------------|
| 模型改动 | 无 | 极小 | 大 |
| 训练改动 | 中 | 较大 | 大 |
| 几何精度提升 | 中等 | 显著 | 最优 |
| 时序一致性提升 | 间接 | 间接+直接 | 直接 |
| 计算开销增加 | ~2x | ~2.3x | ~2.6x |
| 实施时间 | 1-2 周 | 2-4 周 | 4-8 周 |
| 风险 | 低 | 中 | 高 |
| 可复用性 | 训练流程可复用 | 训练+约束可复用 | 架构级改动 |

## 4. 推荐实施路径

**渐进式策略**：

```
Phase 1: 方案 A（1-2 周）
├── 扩展数据采集（多视角 + 深度）
├── 实现 MultiViewTrainer
├── 在现有 CMSTNF/MS-SCNF 上验证
└── 评估：几何精度/时序稳定性指标

Phase 2: 方案 B（2-4 周，基于 Phase 1 结果决策）
├── 增加跨视角一致性约束
├── 增加重投影验证
├── 深度引导采样通用化
└── 评估：对比 Phase 1 的提升幅度

Phase 3: 方案 C（4-8 周，长期目标）
├── SDF + density 混合架构设计
├── Eikonal loss + SDF 表面约束
├── 多视角 + 深度 + SDF 综合训练
└── 评估：最终性能 vs 实施成本
```

每个 Phase 有独立的评估节点，如果效果提升不明显可以终止或调整方向。

## 5. 关键技术细节

### 5.1 多视角 Camera 管理

当前代码中相机参数是硬编码的。多视角训练需要一个统一的相机管理模块：

```python
# 建议新增: src/utils/camera_system.py
class MultiCameraSystem:
    """管理多个相机的参数和射线生成。"""

    def __init__(self, camera_configs: list[dict]):
        self.cameras = camera_configs  # 每个 dict 包含 eye, center, up, focal

    def get_all_rays(self, H, W):
        """返回所有相机的 rays_o, rays_d 列表。"""
        return [get_rays(H, W, c['focal'], c['eye'], c['center'], c['up'])
                for c in self.cameras]

    def project(self, points_3d, camera_idx):
        """将 3D 点投影到指定相机。"""
        ...

    def unproject(self, depth, camera_idx):
        """从指定相机的深度图反投影到 3D。"""
        ...
```

### 5.2 数据格式扩展

当前 npz 格式：
```python
{
    'actions': (N, D),
    'images': (N, H, W),
    'positions': (N, 3, 31),  # 可选
}
```

扩展后：
```python
{
    'actions': (N, D),
    'images': (N, V, H, W),          # V 个视角
    'depths': (N, V, H, W),          # V 个视角深度图
    'positions': (N, 3, 31),
    'camera_params': (V, ...),        # 各相机参数
}
```

### 5.3 训练采样策略

多视角训练时，每个 step 的采样策略：

1. **射线采样**：每个视角各采 N_rays 条射线（如 512/view），总计 N_rays × V
2. **点采样**：沿射线采 N_samples 个点（如 64），可选择深度引导采样
3. **交汇点采样**（方案 B）：选择多个视角射线同时经过的区域，采样 K 个交汇点
4. **Foreground 比例**：每个视角独立做前景/背景混合采样

### 5.4 Loss 权重建议

```
L_total = w_recon × L_multi_view_recon
        + w_depth × L_depth
        + w_consist × L_consistency     # 方案 B
        + w_reproj × L_reprojection     # 方案 B
        + w_eikonal × L_eikonal         # 方案 C
        + w_smooth × L_smoothness       # 所有方案

推荐初始权重：
w_recon = 1.0 (per view)
w_depth = 0.1
w_consist = 0.05
w_reproj = 0.1
w_eikonal = 0.1
w_smooth = 0.1
```

## 6. 评估指标

### 6.1 几何精度

| 指标 | 描述 | 需要的 GT |
|------|------|-----------|
| Chamfer Distance | 预测表面与 GT 表面的双向距离 | 3D mesh/点云 |
| IoU | 预测与 GT 的 3D 交并比 | 3D 体素化 |
| Depth MAE | 渲染深度与 GT 深度平均误差 | 深度图 |
| 表面法向一致性 | 预测与 GT 法向量夹角 | 3D mesh |

### 6.2 时序一致性

| 指标 | 描述 |
|------|------|
| 帧间顶点位移 | 连续帧表面点的平均位移 |
| 时序 MSE | 相邻帧预测变化的平滑度 |
| 物理状态平滑度 | temporal encoder 输出的帧间差异 |

### 6.3 效率

| 指标 | 描述 |
|------|------|
| 训练时间/epoch | 多视角 vs 单视角的时间比 |
| 显存占用 | 多视角 rendering 的 GPU 内存开销 |
| 推理速度 | 单帧推理时间（不受训练视角数影响） |
