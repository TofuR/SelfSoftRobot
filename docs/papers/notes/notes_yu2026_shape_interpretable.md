# 阅读笔记：Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control

> Peng Yu, Xin Wang, Ning Tan — arXiv:2603.01751, March 2026
> Sun Yat-sen University, Guangzhou, China

## 一句话概括

用**分段 Bézier 曲线**从多视角图像提取可解释的形状参数，再用 **Neural ODE** 学习形状和末端执行器的动力学，实现软臂的**混合形状-位置控制** + 障碍物避让 + 自运动。

---

## 核心问题

连续体机器人的三大挑战：
1. **感知**：连续变形难以精确测量
2. **建模**：强非线性、高耦合、参数不确定性
3. **控制**：末端控制不够——需要全身形状控制来适应复杂环境

现有方法的不足：
- **模型驱动方法**：依赖精确的物理模型（Cosserat 杆、FEM 等），实际中难以获得
- **离散点方法**：需要在机器人身体上贴大量标记点，不实用
- **端到端视觉方法**（如 DVIK）：隐式表征，无几何语义，无法推理机器人与环境的交互

## 方法架构

框架由五个模块组成（Fig. 2）：

### A. 形状表示：分段 Bézier 曲线
- 用**少量控制点**的 Bézier 曲线参数化机器人全身形状
- 优势：紧凑（只需几个控制点）、连续、平滑、物理意义明确
- 这与我们的 Fourier/B-spline skeleton heads 思路一致

### B. 视觉感知管线
```
多视角图像 → 机器人区域分割 → 骨架提取（形态学操作） → Bézier 曲线拟合 → 形状参数
```
- 从两个正交视角的平面图像提取骨架曲线
- 拟合 Bézier 控制点作为形状状态 (shape state)
- **可解释性**：控制点就是形状的显式编码

### C. Neural ODE 自建模
两个独立的 Neural ODE 模型：
- **Position NODE**：学习 actuation → 末端执行器位置 的动力学
- **Shape NODE**：学习 actuation → 全身形状参数 的动力学

优势：
- 直接从数据学习，无需解析模型
- 数据效率高（NODE 的连续时间特性）
- 可计算 Jacobian（用于控制器设计）

### D. 障碍物避让策略
- 在两个视角图像中检测机器人与障碍物的距离
- 定义**警告距离**：当任一视角中距离低于阈值时激活避让
- 在最近点施加**排斥逃逸速度**（repulsive escape velocity）
- 将局部逃逸速度映射为**全局形状变化**
- 关键洞察：只要在至少一个视角中不碰撞，3D 空间中就不碰撞

### E. 混合控制器
```
u = J_p^† · (K_p · e_p) + J_s^† · (K_s · e_s + v_obstacle)
```
- **Position Controller**：Jacobian 伪逆 × 位置误差
- **Shape Controller**：Jacobian 伪逆 × 形状误差
- **Obstacle Avoidance**：融入形状控制通道的排斥速度
- Jacobian 从 Neural ODE 模型自动计算（不需要手动推导）

## 实验平台

- **硬件**：三段式线驱动连续体机器人（cable-driven, 3-segment continuum robot）
- **感知**：两个正交视角的相机
- **自由度**：每段 2 DOF，共 6 DOF
- **长度**：约 0.3m

## 实验结果

### Task 1: 形状-位置调节 (Shape-Position Regulation)
- 从任意初始形状和位置收敛到参考形状和位置
- 两个视角均收敛到参考形状

### Task 2: 形状-位置轨迹跟踪 (Shape-Position Tracking)
- 平均位置跟踪误差：~0.003m（1% 臂长）
- 最大位置跟踪误差：< 0.006m（2% 臂长）
- 形状误差在图像分辨率的 1.56% 以内

### Task 3: 障碍物感知的形状-位置调节
- 分阶段避让：先正常趋近目标，当距离低于警告距离时激活避让
- 最终同时达到目标形状和位置，并成功避开障碍物

### Task 4: 自运动 (Self-Motion)
- 保持末端位置不变的同时调整全身形状避开动态障碍物
- 在一个视角中可能有视觉重叠，但另一视角保持安全距离 → 3D 无碰撞

### 对比实验（vs. DVIK [Almanan et al.]）
| 维度 | Yu 2026 | DVIK |
|------|---------|------|
| 视角数 | 双视角 | 单视角 |
| 形状表示 | Bézier 控制点（可解释） | 隐式 latent |
| 3D 一致性 | 两视角误差几乎相同 | View 2 误差显著大于 View 1 |
| 避障能力 | 通过形状控制避障 | 无法避障，碰撞 |
| 稳态误差 | 更小 | 更大 |

12 个不同参考形状的统计结果：本文方法两个视角误差分布几乎一致，DVIK 在非控制视角误差显著增大。

## 关键创新

1. **可解释的形状表示**：Bézier 控制点 = 低维连续几何参数，可直接用于控制和避障
2. **Neural ODE 建模**：数据驱动、无需解析模型、可计算 Jacobian
3. **多视角编码**：解决单视角的形状歧义问题
4. **统一框架**：建模 + 控制 + 避障在一个框架内

## 与本项目的深度关联

| 维度 | Yu 2026 | SelfSoftRobot (本项目) |
|------|---------|----------------------|
| **形状表示** | Bézier 控制点 (2D→3D) | 3D 点云 + SDF + Neural Field |
| **感知** | 真实多视角相机 | PyElastica 仿真渲染 |
| **动力学模型** | Neural ODE | Flow Matching ODE |
| **训练信号** | Bézier 控制点（从图像提取） | 点云/深度/渲染图 |
| **控制目标** | 形状 + 位置调节 | 目前仅建模（形状预测） |
| **编码器** | 无（直接用 actuation 做 ODE 输入） | MultiScaleEMA（时序编码） |

### 关键启发

1. **Bézier 形状参数化的优势**：
   - 我们已有 Fourier/B-spline skeleton heads（`src/heads/skeleton_heads.py`）
   - Yu 2026 验证了"少量控制点参数化全身形状"的可行性
   - 可以考虑将 Bézier 参数化作为 Flow Matching 的中间监督

2. **Neural ODE 与 Flow Matching 的对比**：
   - Neural ODE：dstate/dt = f(state, actuation)，物理时间动力学
   - Flow Matching：dX/dt = v(X, t | action)，噪声→数据的生成路径
   - 两者都是 ODE，但语义不同：NODE 建模物理演化，FM 建模生成过程
   - Yu 2026 用 NODE 是因为需要从 actuation 计算 Jacobian 做控制
   - 我们的 FM 更适合从 action 直接生成点云

3. **可解释性很重要**：
   - Yu 2026 的核心卖点是 shape-interpretable
   - 端到端方法的隐式表征无法做障碍物避让
   - 我们的点云输出虽然比隐式密度场更直观，但仍缺乏参数化结构
   - 如果输出 Bézier 控制点而非原始点云，可以直接用于控制

4. **多视角的必要性**：
   - 单视角无法唯一确定 3D 形状（DVIK 的失败案例）
   - 我们的多视角策略（`MultiViewStrategy`）方向正确

5. **控制是最终目标**：
   - Yu 2026 从自建模直接走向控制
   - 我们目前只做到建模（action → shape prediction）
   - 下一步：从形状模型出发设计控制器

## 局限与未来方向（原文指出）

1. **视觉分割依赖**：对光照、遮挡、背景复杂度敏感
2. **Bézier 平滑性约束**：难以捕捉局部高曲率变形
3. **无完整 3D 重建**：多视角 2D 图像推断 3D，但不显式重建
4. **等优先级控制**：形状和位置任务同等优先，未考虑层次化任务分配
5. **无外力干扰处理**：不考虑外力对形状的影响（迟滞/蠕变）
