# 方向：从轮廓恢复形状（Shape from Silhouette）

> 状态：待探索
> 优先级：中
> 关联：多视角渲染、骨架条件 Visual Hull、PC-Spatial 修正分支

---

## 问题

我们的仿真中已经生成了**二值轮廓图**（PyVista 渲染的黑白图像）。这些轮廓图目前仅用于体渲染的监督信号（MSTNF / C-MSTNF 的 rendering 模式）。

但轮廓图本身包含了 3D 形状的重要约束：**从多个视角的轮廓可以重建 Visual Hull**——物体 3D 形状的上界。

**问题：能否利用多视角轮廓直接约束/重建 3D 骨架和表面？**

---

## 背景：Visual Hull

Visual Hull = 多个视角轮廓反投影的交集：
1. 每个视角的轮廓 → 定义一个锥体（从相机中心穿过轮廓的半无穷锥）
2. 所有锥体的交集 = Visual Hull
3. Visual Hull 是真实形状的上界（总是 ≥ 真实形状）

局限：
- 凹面无法恢复（被轮廓遮挡）
- 分辨率受视角数量限制
- 需要精确的相机标定

---

## 方案

### A. 骨架条件 Visual Hull ★

Borges et al. (Frontiers 2022) 的 armature-based 方法：
1. 预测 3D 骨架（我们的模型已有）
2. 用骨架定义"骨架条件"：在骨架附近搜索表面
3. 结合 Visual Hull 约束，在骨架邻域内找最优表面

适配到我们：
- 骨架预测 → 定义管状搜索空间（半径 R 内）
- 多视角轮廓 → Visual Hull 约束
- 在管状空间 ∩ Visual Hull 内采样/优化表面

优点：
- Visual Hull 消除了"常数半径"假设
- 骨架先验解决了 Visual Hull 的凹面问题
- 不需要额外训练，纯几何方法

### B. 可微 Visual Hull

将 Visual Hull 构建过程可微化：
1. 3D 点投影到各视角 → 检查是否在轮廓内 → 可微近似（sigmoid）
2. 轮廓一致度 loss = 预测轮廓与真实轮廓的差异
3. 反传梯度优化 3D 形状

适配到我们：
- 骨架节点坐标 → 投影到各视角 → 与轮廓对比
- 直接优化骨架坐标使投影匹配轮廓

优点：端到端可微
缺点：需要仔细处理遮挡和可微近似

### C. 神经隐式 Visual Hull

用隐式场（NeRF/NeuS）结合轮廓监督：
1. 我们的体渲染管线已经支持轮廓监督（MSTNF）
2. 增强：增加轮廓一致性 loss（多视角轮廓应一致）
3. Morphable-SfS (ICRA 2024) 的思路：可变形先验 + 轮廓

适配到我们：
- 已有 `MultiViewStrategy` 的 `consist` loss
- 可以增加更强的轮廓约束

### D. 点云生成 + 轮廓约束

Flow Matching 生成点云 + 轮廓投影约束：
1. Flow Matching 模型生成 3D 点云
2. 将点云投影到各视角，检查是否与轮廓一致
3. 不一致的点被惩罚/移除

优点：结合了学习的生成能力和几何约束
缺点：投影过程需要可微分

---

## 实验计划

### 第一步：评估 Visual Hull 的可行性

1. 用现有 `exp7_multiview` 数据（多视角 + 深度）
2. 从 4-6 个视角的轮廓构建 Visual Hull
3. 将 GT 骨架投影到 Visual Hull 中，检查吻合度
4. 评估 Visual Hull 的分辨率和精度

### 第二步：骨架条件 Visual Hull

1. 用预测骨架定义管状搜索空间
2. 在管状空间内用 Visual Hull 约束表面
3. 输出：带变化半径的 3D 形状

---

## 与当前代码库的对接

| 组件 | 现有 | 需新增 |
|------|------|--------|
| 轮廓图像 | 二值渲染图（已有） | 无 |
| 相机标定 | `src/utils/camera_system.py` | 无 |
| 投影/反投影 | `src/utils/camera.py` | Visual Hull 构建 |
| 轮廓提取 | `src/utils/skeleton_2d.py` | 可适用于轮廓 |
| 多视角策略 | `src/rendering/view_strategy.py`（MultiViewStrategy 已含 with_consistency / w_consist） | Visual Hull 构建 + 骨架条件表面 + 可微轮廓投影 |

---

## 相关文献

- Borges et al. (Frontiers 2022)：Armature-based Visual Hull 重建
- Morphable-SfS (ICRA 2024)：可变形模型 + 轮廓重建
- Segmented SfS (PMC 2022)：分割增强的 Visual Hull
- Lu et al. (ICRA 2024)：可变形形状先验 + 多视角
