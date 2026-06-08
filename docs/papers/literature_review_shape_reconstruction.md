# 文献综述：软体机器人形状重建与自建模 (2024-2026)

> 调研时间：2026-06-08
> 范围：近三年软体机器人 3D 形状重建、骨架→表面、多视角重建、sim-to-real 等方向

---

## 1. 软体机器人 3D 形状重建（从 2D 图像）

### 1a. Dual-Stereo-Vision 三维形态重建
- **Ren, Liu, Zhang et al. — IEEE ROBIO 2024**
- 双对立双目相机系统，重建毫米级缺口管状连续体机器人完整 3D 形态
- arXiv:2408.01615
- **启发**：双目对立配置可覆盖 360° 形态，但硬件成本高

### 1b. 可微渲染的形状重建
- **Li, Boshard et al. — IEEE ICRA 2023**
- 用可微渲染反传图像损失，迭代更新形状参数，无需精确 CAD 模型
- arXiv:2302.14039 | drrobot.cs.columbia.edu
- **启发**：可微渲染替代体渲染，收敛更快；可用于我们模型的渲染监督层

### 1c. SoftNeRF：NeRF 自建模插件
- **Shan et al. — IEEE/RSJ IROS 2024**
- NeRF 自建模框架作为插件，支持形状重建等多种下游任务
- github.com/IRMVLab/soft-nerf
- **启发**：与我们项目最接近的工作。将 NeRF 自建模模块化，可作为 task plugin

### 1d. 局部传感融合形状重建
- **ScienceDirect (Composites and Structures), 2024**
- 融合 Multi-core FBG + IMU 局部应变测量与姿态数据，消除空间误差累积
- **启发**：物理传感 + 视觉融合是另一个方向，但与我们的纯视觉路线不同

### 1e. 视觉伺服综述
- **Toronto Metropolitan University, 2024-2025**
- 系统综述覆盖多相机和无标记视觉形状估计方法
- **启发**：综述性质，了解领域全景

### 1f. Part-Based NeRF 机器人自建模
- **Wiley (Advanced Intelligent Systems), 2025**
- 部件级 NeRF 重建，同时支持形态建模和控制任务
- **启发**：部件级重建思路可用于多段软臂

### 1g. 3D Gaussian Splatting 机器人自建模 ★
- **Hu, Yu, Tan (中山大学) — arXiv March 2025**
- 用 3DGS 替代 NeRF 做机器人自建模，更高保真度和实时性
- arXiv:2503.05398
- **启发**：与我们项目路线最接近。Hu & Yu 来自 Yu 2026 (arXiv) 同一课题组，从 NeRF 演进到 3DGS。如果我们也走 3DGS 路线，这是最直接的参考

---

## 2. 导管/针形状传感（从 X 射线/透视）

### 2a. 双平面透视下的导管形状跟踪
- **Lawson, Chitale et al. — IEEE RA-L 2025**
- 螺旋排列的不透射线标记物，双平面透视同时估计 6-DoF 姿态和形状
- arXiv:2506.09934
- **启发**：标记物设计思路可用于软臂的视觉标记

### 2b. 单平面透视导丝 3D 重建
- **Jianu et al. — RiTA 2024**
- 从单视角透视图像重建 3D 导丝形状，使用 CathSim 仿真器
- arXiv:2311.11209
- **启发**：单视角 3D 重建本身就是欠约束问题，需要强先验或训练数据

### 2c. Guide3D：双平面 X 射线数据集
- **Jianu et al. — ACCV 2024**
- 首个双平面 X 射线标注数据集，用于导丝/导管 3D 重建基准
- arXiv:2410.22224 | airvlab.github.io/guide3d
- **启发**：数据集和方法论可借鉴

### 2d. 合成数据训练导管位置重建
- **MDPI Applied Sciences, 2024**
- 合成数据 + 深度学习，从多张 2D 图像重建导管 3D 位置
- **启发**：与我们的 PyElastica 仿真→渲染→训练 路线一致

---

## 3. 骨架→表面重建（中心线到管状表面）

### 3a. 血管骨架重建几何算法 ★
- **arXiv, February 2024 (arXiv:2402.12797)**
- 联合处理所有骨架点（而非逐点），重建完整管状表面
- **启发**：我们的 `sdf_utils.py` 是逐节段计算 SDF，这种方法可能更精确。但血管的分叉结构比软臂更复杂

### 3b. 可定制管状模型（N-分叉血管）
- **PMC / Medical Imaging, 2023**
- 从中心线 + 半径参数化生成分叉管状几何体
- **启发**：如果我们允许每节段不同半径，可借鉴此参数化

### 3c. 隐式管状表面生成
- **arXiv:1606.03014**
- 从中心线引导生成隐式管状表面，用于冠状动脉重建
- **启发**：与我们的 SkeletonSDF 思路（骨架条件 + 隐式场）高度一致

---

## 4. 多视角/动态场景重建

### 4a. DGS-LRM：单目视频实时可变形 3DGS
- **NeurIPS 2025 (arXiv:2506.09997)**
- 前馈方法，从单目视频实时预测可变形 3DGS，无需逐场景优化
- **启发**：如果从 NeRF 转向 3DGS，这是动态场景的前沿方法

### 4b. MUSt3R：多视角立体 3D 重建
- **Naver Labs — arXiv March 2025**
- 扩展 DUSt3R 到多视角，transformer 预测点图，无需相机标定
- arXiv:2503.01661
- **启发**：无需标定的多视角重建，可能简化我们的多相机设置

### 4c. 可变形 3DGS 动态场景
- **Zhejiang University — CVPR 2024**
- 在规范空间学习 3DGS + 学习变形场，捕获时间动态
- arXiv:2309.13101
- **启发**：规范空间 + 变形场 的思路与我们的 C-MSTNF (canonical + deformation) 一致，但用 3DGS 替代 NeRF

### 4d. NRMVS：非刚性多视角立体
- **Innmann et al. — WACV 2020**
- 从宽基线稀疏视角密集重建动态 3D 场景
- **启发**：基础性工作，方法论可参考

---

## 5. 仿真到真实迁移

### 5a. 零样本 Sim-to-Real 软体机器人感知
- **Yoo et al. (AI4CE Lab) — arXiv March 2023**
- 在仿真中采集高保真点云，零样本迁移到真实 3D 本体感知
- arXiv:2303.04307
- **启发**：与我们的路线几乎一致（仿真→形状模型→感知）。他们的零样本迁移策略值得研究

### 5b. 学习残差物理的 Sim-to-Real ★★
- **Gao, Michelis et al. (ETH Zurich) — IEEE RA-L 2024 (Best Paper Award)**
- 混合框架：解析仿真 + 神经网络学习残差修正，少量真实数据即可
- github.com/srl-ethz/residual_physics_sim2real
- **启发**：直接适用于我们。PyElastica 作为解析仿真器，Neural Field 学习残差。这比纯学习或纯物理都更优雅

### 5c. 零样本 RL 视觉伺服 Sim-to-Real
- **Yang et al. — CoRL 2025**
- RL 视觉伺服，解耦运动学与力学特性，实现零样本迁移
- arXiv:2504.16916
- **启发**：解耦思路可用于我们的模型设计

### 5d. 软体机器人导航策略 Sim-to-Real
- **HAL Science, 2024**
- 仿真框架下引导线驱动软体机器人在静态环境中导航
- **启发**：应用层面的参考

---

## 6. 从轮廓恢复形状（Shape from Silhouette / Visual Hull）

### 6a. Morphable-SfS
- **Lu et al. — IEEE ICRA 2024**
- 可变形模型 + 多视角合成，从轮廓重建 3D 网格
- **启发**：结合形状先验的 SfS，比纯 voxel carving 更精确

### 6b. Armature-Based 3D 形状重建 ★
- **Borges, Rieder et al. — Frontiers in Robotics and AI, 2022**
- 开源实时 3D 重建框架，骨架 + Visual Hull 方法，用于 AR/VR 中传感化软体机器人
- **启发**：骨架条件 + Visual Hull 的组合与我们的路线非常契合。可在骨架预测基础上用 Visual Hull 精细化表面

### 6c. Segmented SfS
- **PMC, 2022**
- 用轮廓分割增强 voxel-based SfS
- **启发**：分割策略可提升重建精度

---

## 综合分析

### 与本项目最相关的工作（按优先级）

| 优先级 | 工作 | 关联原因 |
|--------|------|----------|
| ★★★ | SoftNeRF (IROS 2024) | 直接竞品，NeRF 自建模 |
| ★★★ | 3DGS 自建模 (Hu & Yu 2025) | Yu 2026 课题组的最新演进 |
| ★★☆ | 残差物理 Sim-to-Real (ETH RA-L 2024) | 最佳实践，与 PyElastica 路线一致 |
| ★★☆ | 可微渲染形状重建 (ICRA 2023) | 可替代/补充体渲染监督 |
| ★★☆ | 骨架条件 Visual Hull (Frontiers 2022) | 骨架→形状的工程化方案 |
| ★☆☆ | 零样本 Sim-to-Real (arXiv 2023) | 仿真→感知迁移方法论 |
| ★☆☆ | 管状表面重建 (arXiv 2024) | 精细化骨架→表面 |

### 我们的差异化优势

1. **迟滞建模**：几乎所有上述工作都不考虑时序依赖（迟滞），这是我们的核心差异化
2. **时序编码器**：Gamma/Laguerre 延迟核是原创贡献
3. **预测-修正框架**：PC-Spatial 的两阶段设计是原创
4. **3D 自建模**：大部分工作停留在 2D 形状参数或粗略 3D，我们直接生成 3D 形状

### 我们的不足（相对上述工作）

1. **仅仿真**：所有对比工作都在真实机器人上验证
2. **仅骨架**：退化为骨架预测，丢失了表面/截面信息
3. **无控制闭环**：只做建模，没有走向控制
4. **无实时性考虑**：训练和推理分离，无在线适应
