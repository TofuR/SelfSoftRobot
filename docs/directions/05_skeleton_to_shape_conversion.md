# 方向：骨架→形状转换（Skeleton-to-Shape Conversion）

> 状态：分阶段迁移中（Phase 0 接口先行）
> 优先级：高（★★★）
> 前置：骨架预测模型（[14](14_gt_observed_transition.md) / [15](15_open_loop_windowed_transition.md)）已可用
> 最后更新：2026-06-19（重写为退化式分阶段迁移主线）

---

## 问题

当前模型只预测骨架（中心线 31 节点），形状被简化为**常数半径圆管**。要实现"3D 全身状态"理念，需把骨架升级为完整形态（变半径 / 变截面 / 表面）。但必须**严格符合项目四理念**：骨架为先、低成本（仅相机）、迟滞感知、3D 全身。

**关键现状（2026-06-17 勘误已核实）**：常数半径 + 圆形截面的限制**只在模型侧**——`evaluation/surface_sampling.sample_gt_surface` 与 `data/dataset_skeleton_sdf._sample_surface` 已接受 per-node `radii` 数组（GT/采样侧可变半径已就绪）；缺的是**模型侧**：`SpatialSequenceModel.predict_pointcloud`、`StateTransitionSpatialModel.predict_pointcloud`、`PCSpatialSequenceModel.predict_pointcloud` 仍硬编码 `avg_radius=0.015`、不预测 r_i。

---

## 主线：退化式分阶段迁移

每一阶段**只补一个缺口、增量改动现有代码、去掉新机制严格退化回上一阶段/现状**，绝不破坏当前 gt/open_loop 主线。两条形态路径（密度场渲染 / SDF mesh）**共用同一 `point_to_skeleton_coords` + 同一 `RadiusField`**，只分支输出头，避免与 MS-SCNF 路线分叉。

### Phase 0 — 接口就位（半天，零风险，先于一切）

(a) `src/utils/sdf_utils.compute_gt_sdf` 的 `radius` 参数从 `float` 扩为 `float | (N,)`（`float` 分支严格退化现状）；
(b) `src/heads/skeleton_heads.py::point_to_segment_distance` 增加可选 `return_idx=True`，复用 `point_to_skeleton_coords` 的 `argmin` 逻辑返回 `(dist, closest_idx)`，默认 `False` 不改现有调用方。

这两步是 Phase 1 逐节点半径广播的几何底座，对现有训练零影响。

### Phase 1 — 沿轴变半径场 `radius(s)`（= 旧方案 A 升级）

- 新建 `src/fields/radius_field.py`：`RadiusField(z → (B,31) 半径)`，softplus 保证正 + 偏置令初值 ≈ 0.015（冷启动等价现状）。
- `SkeletonSDFModel.forward` 的 `sdf_prior = dist - rod_radius` 升级为**逐节点广播**（用 Phase 0 的 `closest_idx` gather 出对应节点半径）。
- **迟滞耦合（关键）**：`RadiusField` 的输入必须用 **transition 模型输出的 `z`**（迟滞潜变量），而非 `SkeletonSDFModel` 自带的 `MultiScaleEMA`——否则形态不随迟滞历史演化，违反"迟滞感知"理念。
- **模块边界**：新建 `TransitionConditionedShapeModel` 包装类组合（冻结的 transition + 可训练形态场），**不改 `SkeletonSDFModel.forward` 签名**——保持单一职责，与 `UnifiedTrainer` 的 `freeze_modules` 语义兼容。
- 常数半径时严格退化现状（向后兼容）。

### Phase 2 — 环向激活 `theta`（= 旧方案 B 学习截面，但先解决几何鲁棒性）

- `src/fields/skeleton_density.py` 已计算 `theta` 但丢弃（`self._last_theta`），激活成本极低：加一个 `PositionalEncoder(d_input=1, n_freqs=6)` + 拼入 latent。把形态场从 `(dist, t_axial)` 扩成 `(dist, t_axial, theta)` → 同一 dist 不同 theta 输出不同 density/SDF 残差 → 能表达扁平吸盘 / 充气腔鼓胀 / 接触面压扁。
- **前置几何工程（不可跳过）**：`point_to_skeleton_coords` 用 `cross(tangent, z_hat)`，骨架竖直时切换到 `y_hat`，切换点产生 `theta` 跳变 → 环向不连续。**必须先沿骨架传播平行传输标架（parallel transport frame）** 或对 `theta` unwrap 后再 PE。

### Phase 3 — 时序一致 + 隐式法向（= 新增，缺口 6/8）

- **跨帧形态一致性 loss**（无需 GT 自监督）：对固定查询点集 X，`L_temporal = ‖SDF(X; s_t, z_t) − SDF(X; s_{t-1}, z_{t-1})‖²`，mask 限近表面点，纯平滑。
- **隐式表面法向**：渲染路径下从体渲染 alpha 场梯度近似法向，与多视角 silhouette 边界法向做余弦对齐（SingleView 退化不启用，MultiView 复用 `view_strategy.py` 的 `with_consistency` 开关）。

### 评估闭环（补缺口 7）

新建 `src/evaluation/mesh_metrics.py`：marching cubes 从 SDF 提 mesh + Chamfer/Hausdorff 对 GT mesh，**取代只看中心线 L2**。bbox 由骨架 ± `max(radius)` 自适应，分辨率先 128³。

---

## 与旧方案 A/B/C/D 的映射

| 旧方案 | 现定位 |
|--------|--------|
| **A 解析管状（可变半径）** | → **Phase 0 + Phase 1**（RadiusField 用 transition z）。剩余工作 = "预测 r_i 并接入已存在的 per-node 半径路径" |
| **B 学习截面（椭圆/Fourier/隐式）** | → **Phase 2**（激活 theta = 沿环向可变截面） |
| **C SkeletonSDF（tubular prior + SIREN residual）** | 已实现但效果不好——**原因写入**：常数半径先验下残差 ≈ 0（GT 用同公式），模型收敛到先验即 loss 趋零。改进靠 Phase 1 变半径 + Phase 3 多监督 |
| **D 骨架条件 3DGS（Hu & Yu 2025）** | 长期方向（需 3DGS 基础设施，项目暂无） |

---

## ⚠️ 诚实声明（TOP RISK）

**本项目实物软臂（PyElastica Cosserat rod）半径本就是常数，变半径 / 变截面 / 环向各向异性全部没有 GT**——`RadiusField` / `theta` 会在无监督下学出非物理几何。

**缓解（三管齐下）**：
1. **先在仿真用解析变半径 / 变截面构造合成 GT** 验证机制（Phase 1/2 跑通机制），明确这是"机制验证"非"实物验证"。
2. **几何正则约束到物理合理域**：radius 沿轴二阶差分平滑 + 数值 clamp + **渲染 silhouette 监督**（图像能观测到粗细——是实物唯一可用的形态监督源，符合"仅相机"理念）。
3. **实物部署诚实边界**：实物只走渲染路径（密度场），SDF/mesh 路径仅作仿真自监督与评估；碰撞查询保守取 `max(radius)` 包络（与 [16](16_constraint_oriented_control.md) 一致）。

---

## 落地优先级

1. **立即（半天，零风险）**：Phase 0 接口就位。
2. **短期（1–2 周）**：Phase 1 变半径 `RadiusField`（用 transition z）——最低成本补缺口，向后兼容，立即能表达末端变细 / 根部变粗。
3. **中期（1 月）**：Phase 2 环向 `theta` 激活（**前置 parallel transport frame 几何修复**）。
4. **论文级**：Phase 3 时序一致 + 隐式法向 + `mesh_metrics` 评估闭环 + 方案 D（3DGS）。

---

## 相关文献 / 代码

- 形态机制：`src/fields/skeleton_density.py`（SkeletonConditionedDensity）、`src/models/model_skeleton_sdf.py`（tubular SDF + SIREN）、`src/models/model_ms_scnf.py`（多尺度骨架+渲染）
- 管状 SDF 计算：`src/utils/sdf_utils.py`
- 骨架→点云采样：`src/evaluation/surface_sampling.py`
- 3DGS 机器人自建模：arXiv:2503.05398（Hu, Yu, Tan 2025）= 方案 D
- 控制层消费全身 SDF：[16 约束导向控制](16_constraint_oriented_control.md)
