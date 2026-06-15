# 阅读笔记：Controlling Diverse Robots by Inferring Jacobian Fields with Deep Networks (Neural Jacobian Fields)

> Li, Zhang, Chen, Matusik, Liu, Rus, Sitzmann — Nature, 2025
> MIT CSAIL
> DOI: [10.1038/s41586-025-09170-0](https://doi.org/10.1038/s41586-025-09170-0)
> GitHub: https://github.com/sizhe-li/neural-jacobian-field

## 一句话概括

从多视角 RGB-D 视频自监督学习**visuomotor Jacobian field**——将机器人上任意 3D 点映射到其对每个驱动命令的灵敏度——实现仅需单目相机即可进行闭环控制，无需任何机器人先验知识。

---

## 核心问题

传统机器人控制依赖精确的运动学/动力学模型，这对刚性连杆机器人可行，但对**软体、多材料、仿生机器人**几乎不可能：
- 软体机器人缺乏明确的关节结构，无法用 Denavit-Hartenberg 参数建模
- 多材料机器人的本构关系复杂，FEM 建模代价高昂
- 许多仿生机器人没有内置传感器（proprioception）

核心挑战：能否**仅从视觉观察**学习一个通用表示，使任意形态的机器人都能被控制？

## 方法架构

框架由两个核心组件组成：

### 1. 状态估计模型（State Estimation Model）
- 输入：单张 RGB 图像 $I$
- 输出：**Radiance Field** + **Jacobian Field** 的联合 3D 表示
- 基于 PixelNeRF 架构的单视图到 3D 模块
- **Radiance Field**：编码机器人的 3D 形状和外观
- **Jacobian Field**：编码 3D 空间中每个点 $\mathbf{x}$ 对驱动命令 $\mathbf{u}$ 的灵敏度

### 2. Visuomotor Jacobian Field
- 定义：$J(\mathbf{x}, I) = \partial \mathbf{x} / \partial \mathbf{u}$
- 描述驱动状态变化 $\delta \mathbf{u}$ 如何引起 3D 运动变化 $\delta \mathbf{x}$
- 是传统系统 Jacobian 的**泛化**：不依赖专家设计的状态表示 $q$
- 通过 3D Jacobian 参数化注入**线性**和**空间局部性**归纳偏置，使模型能泛化到未见过的机器人构型

### 训练流程
```
多视角 RGB-D 视频（12 台相机，2-3 小时随机动作）
    ↓
1. 执行随机驱动命令，记录执行前后的多视角捕获
2. 神经 3D 重建：单张 RGB → Jacobian Field + Radiance Field
3. 自监督信号：
   - 将 Jacobian Field 渲染为光流图像
   - 将 Radiance Field 渲染为 RGB-D 图像
   - 与真实观测对比计算 Loss
    ↓
训练完成后：单目相机即可闭环控制
```

### 控制流程
```
单目图像 → 状态估计 → Jacobian Field + Radiance Field
    ↓
给定期望运动 → 梯度优化驱动命令：min ||J · δu - δx_target||
    ↓
执行命令 → 新图像 → 闭环更新
```

## 实验设置

- **测试平台**：多种机器人操纵器，涵盖不同的驱动方式、材料、制造工艺和成本
  - 刚性连杆机器人（传统关节臂）
  - 软体机器人（硅胶等软材料）
  - 多材料混合机器人
  - 仿生机器人
- **训练数据**：12 台消费级 RGB-D 相机，2-3 小时随机动作视频
- **控制**：仅用单目 RGB 相机进行闭环控制

## 关键创新

1. **Visuomotor Jacobian Field 作为通用机器人表示**：将任意 3D 点映射到其对驱动的灵敏度，是传统 Jacobian 矩阵的连续场泛化
2. **纯视觉控制**：不假设机器人的材料、驱动方式或传感能力，仅用相机即可控制
3. **自监督训练**：通过光流和 RGB-D 渲染作为自监督信号，无需人工标注
4. **因果动态结构恢复**：不仅能控制机器人，还能从数据中恢复其因果运动链结构
5. **3D Jacobian 参数化的归纳偏置**：线性和空间局部性使得模型在有限数据下能泛化到新构型

## 与本项目的关联

| 维度 | Neural Jacobian Fields | SelfSoftRobot |
|------|----------------------|---------------|
| **核心表示** | Jacobian Field（灵敏度场） | SDF / 密度场（形状场） |
| **输出目标** | 3D 点对驱动的 Jacobian | 3D 点的 SDF 值 / 占据概率 |
| **用途** | 控制器设计（逆运动学） | 形状预测与重建 |
| **时序建模** | 无（稳态假设） | MultiScaleEMA 等时序编码器 |
| **训练信号** | 光流 + RGB-D（自监督） | 体渲染图像 + 深度 + 3D 点云 |
| **相机需求** | 12 台 RGB-D（训练）+ 单目（推理） | 单/多视角 |
| **机器人类型** | 软体 + 刚性 + 混合 | 软体连续体臂（Cosserat 杆） |

### 关键启发

1. **Jacobian vs. SDF 的互补性**：
   - NJF 回答"如果改变驱动，形状会怎样变化"（微分运动学）
   - 我们的项目回答"给定驱动，形状是什么"（正向运动学/形态学）
   - 两者结合可以实现完整的感知-控制闭环：先建模（我们），再控制（NJF）

2. **稳态假设的局限**：
   - NJF 只建模稳态（steady-state），不处理瞬态动态
   - 软体机器人的迟滞（hysteresis）、蠕变（creep）等时序效应被忽略
   - 我们的 MultiScaleEMA / TemporalGRU 等时序编码器正是为解决此问题而设计

3. **多视角训练代价**：
   - NJF 需要 12 台相机的多视角设置，实验门槛高
   - 我们的项目仅需单视角即可通过体渲染训练，更实用

4. **从 NeRF 到 Jacobian 的思路**：
   - NJF 基于 PixelNeRF 架构，是 NeRF 在机器人领域的又一创造性应用
   - 与 FBV-SM 的思路类似但目标不同：FBV-SM 学密度/可见性，NJF 学 Jacobian
   - 可以考虑在我们的 SDF 场之上额外学习一个 Jacobian 场

## 局限

1. **多视角数据采集代价高**：训练需要 12 台 RGB-D 相机，限制了实际部署的便利性
2. **仅建模稳态**：假设驱动命令执行后机器人达到稳态，不处理瞬态动力学（迟滞、蠕变、振动）
3. **不跨机器人泛化**：每个机器人需要单独训练，学到的 Jacobian Field 无法迁移到不同形态的机器人
4. **无力/触觉感知**：纯视觉方法无法感知接触力、力矩等对软体机器人至关重要的量
5. **无外部负载处理**：训练时使用随机动作，不考虑外部力/负载对形状的影响
6. **计算开销**：PixelNeRF + Jacobian 场的推理计算量较大，实时性受限于 GPU
