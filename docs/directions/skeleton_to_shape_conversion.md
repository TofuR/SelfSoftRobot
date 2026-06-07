# 方向：骨架→形状转换（Skeleton-to-Shape Conversion）

> 状态：待探索
> 优先级：高
> 前置：骨架预测模型（SpatialSequence / PCSpatial）已可用

---

## 问题

当前模型只预测骨架（中心线节点坐标），形状被简化为**固定半径的圆管**。实际软臂的截面可能：
- 非圆形（充气后变为椭圆/多边形）
- 沿长度方向半径变化（固定端 vs 自由端）
- 随驱动状态动态变化（弯曲时内侧压缩、外侧拉伸）

**但目前的 `sdf_utils.py` 和 `_sample_surface()` 都假设常数半径 + 圆形截面**。

---

## 方案

### A. 解析管状模型（最简单，已有基础）

当前实现：
- `sdf_utils.py`：逐节段计算 tubular SDF = ||p - closest_point_on_segment|| - radius
- `dataset_pointcloud.py`：`_sample_surface()` 在圆柱坐标下采样表面点

改进：**逐节段可变半径**
- 骨架输出不变 (N nodes × 3D)
- 新增半径预测：每个节点预测一个标量 r_i（或用 SIREN 预测连续半径函数 r(s)）
- SDF 变为：SDF(p) = min_i (||p - s_i|| - r_i)

优点：改动最小，与现有代码兼容
缺点：仍假设圆形截面

### B. 学习截面形状（中等改动）

每个骨架节点不仅预测半径，还预测一个**截面变形参数**：
- 椭圆截面：长轴 a_i, 短轴 b_i, 旋转角 θ_i（每节点 3 个额外参数）
- Fourier 截面：用 Fourier 系数参数化任意闭合截面曲线
- 隐式截面：小 MLP 输入角度 θ → 输出半径 r(θ)

训练信号：3D 点云监督（我们已有 GT 点云数据）

优点：能表示非圆形截面
缺点：训练难度增大，需要足够的截面变化数据

### C. 骨架条件隐式场（SkeletonSDF 已部分实现）

`model_skeleton_sdf.py` 的思路：骨架 + tubular SDF prior + SIREN residual
- Tubular prior 提供粗略几何（固定半径圆管）
- SIREN 学习残差修正（可以修正截面形状、局部变形等）

当前问题：SkeletonSDF 训练效果不好，可能因为 SDF 监督信号不足或网络容量不够

改进方向：
1. 先把骨架预测做准（当前主线），再用 SkeletonSDF 做精细化
2. 增加多尺度 SDF 采样（表面密集 + 远处稀疏）
3. 用 3D 点云作为额外监督（不只是 SDF）

### D. 3D Gaussian Splatting 骨架条件生成 ★

参考 Hu & Yu 2025 (arXiv:2503.05398) 的 3DGS 机器人自建模：
- 骨架节点作为 Gaussian 中心
- 每个节点关联一组 3D Gaussians（位置偏移、协方差、颜色/密度）
- 用可微渲染训练

优点：比 NeRF 快，质量更高，天然支持骨架条件
缺点：需要 3DGS 基础设施（目前项目没有）

---

## 实施建议

1. **短期**（1-2 周）：方案 A，在现有骨架预测基础上加可变半径
2. **中期**（1 月）：方案 B，学习截面形状参数
3. **长期**（论文级）：方案 D，3DGS 骨架条件生成

方案 C 可以和 B 并行探索，看哪种效果更好。

---

## 相关文献

- SkeletonSDF 思路：`src/models/model_skeleton_sdf.py`
- 管状 SDF 计算：`src/utils/sdf_utils.py`
- 骨架→点云采样：`src/evaluation/surface_sampling.py`
- 血管骨架重建：arXiv:2402.12797（联合处理骨架点生成管状表面）
- 隐式管状表面：arXiv:1606.03014（中心线引导隐式表面）
- 3DGS 机器人自建模：arXiv:2503.05398（Hu, Yu, Tan 2025）
