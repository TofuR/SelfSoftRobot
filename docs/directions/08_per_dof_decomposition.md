# 方向：单自由度分解与组合（Per-DOF Decomposition）

> 状态：待探索
> 优先级：中
> 灵感来源：Yang et al. (CoRL 2025) 解耦运动学与力学特性

---

## 问题

当前模型将所有驱动通道（action_dim 个 DOF）同时输入时序编码器，联合预测骨架。问题：
1. **通道耦合**：不同 DOF 之间的交互被隐式学习，难以分解
2. **数据效率**：要覆盖所有 DOF 组合，数据量指数增长
3. **调试困难**：预测误差无法归因到具体 DOF

但 PyElastica 仿真可以**独立控制每个 DOF**，我们已有只变化一个维度的数据（`seq_rz`：x 随机 y=0，`seq_zr`：x=0 y 随机）。

---

## 方案

### A. 独立训练 + 组合推理

1. 对每个 DOF 独立训练一个模型：
   - Model_X：只看 action_x 的历史 → 预测 x 方向的变形
   - Model_Y：只看 action_y 的历史 → 预测 y 方向的变形
2. 推理时叠加：skeleton = skeleton_rest + Δskeleton_x + Δskeleton_y

优点：
- 每个子模型更简单，训练更容易
- 数据效率高（每个 DOF 只需一维扫描数据）
- 可解释性好（每个 DOF 的贡献可视化）

缺点：
- **线性叠加假设**：如果 DOF 之间有强非线性耦合，叠加会失败
- 需要验证线性叠加是否适用于 Cosserat 杆

### B. 低秩分解

将 action→skeleton 映射分解为：
- skeleton(s, a) = skeleton_0(s) + Σ_i f_i(a_i) · φ_i(s)
- 其中 s 是弧长参数，a_i 是第 i 个 DOF 的动作值
- φ_i(s) 是第 i 个 DOF 的空间基函数（模态形状）
- f_i(a_i) 是第 i 个 DOF 的响应幅值

这类似于**模态分析**（modal analysis）在柔性体中的应用。

训练：
1. 用 `seq_rz` 数据学习 φ_x(s) 和 f_x(a_x)
2. 用 `seq_zr` 数据学习 φ_y(s) 和 f_y(a_y)
3. 用 `seq_rr` 数据验证叠加精度，必要时学习耦合项

优点：物理直觉清晰，可解释性极强
缺点：需要验证 Cosserat 杆是否满足（近似）线性叠加

### C. 分层模型

1. 底层：每个 DOF 独立预测"模态贡献"
2. 中层：少量参数捕获 DOF 间的耦合（如交叉注意力）
3. 顶层：时序编码器整合所有 DOF 的历史

类似 Yang et al. (CoRL 2025) 的解耦思路：运动学（通用）和力学特性（特定 DOF）分开建模。

---

## 实验验证

### 第一步：验证线性叠加假设

用现有数据检验：
1. 从 `seq_rz` 提取 action_x → skeleton_x 的映射
2. 从 `seq_zr` 提取 action_y → skeleton_y 的映射
3. 从 `seq_rr` 取 (a_x, a_y) 组合，计算 skeleton_pred = skeleton_rest + Δ_x(a_x) + Δ_y(a_y)
4. 比较 skeleton_pred 与 GT skeleton_rr

如果误差小（< 5%），方案 A/B 可行；如果大，需要方案 C 的耦合建模。

### 数据资源

| 数据集 | 内容 | 用于 | 状态 |
|--------|------|------|------|
| `seq_rz_c2_sk` / `seq_rz_c6_sk` | x 随机，y=0（**当前实际仅有此两类**） | 学习 x 方向模态 | ✅ 已有 |
| `seq_zr` | x=0，y 随机 | 学习 y 方向模态 | ⚠️ **需先采集** |
| `seq_zz` | 两个 DOF 都为零 | 基准形状 skeleton_0 | ⚠️ **需先采集** |
| `seq_rr` | 两个 DOF 都随机 | 验证叠加 / 学习耦合 | ⚠️ **需先采集** |

> 注（2026-06-17 勘误）：`data/` 下目前只有 `seq_rz_c2_sk` / `seq_rz_c6_sk`（x 随机、y=0）；`seq_zr` / `seq_zz` / `seq_rr` 尚未采集，故下方"线性叠加验证"实验需先补采数据（见 `scripts/data_collection/collect.py --action-x zero --action-y random` 等）。

---

## 与当前架构的对接

- 时序编码器：每个 DOF 可以有独立的时序编码器（各自捕获该 DOF 的迟滞特性）
- GRU 序列生成：保持不变，但输入从联合特征变为叠加的模态特征
- 训练脚本：需要新的数据加载逻辑（从多个数据集联合训练）

---

## 相关文献

- Yang et al. (CoRL 2025)：解耦运动学与力学，零样本 sim-to-real (arXiv:2504.16916)
- 模态分析（Modal Analysis）：经典结构动力学方法
- 数据集：`data/seq_rz`, `data/seq_zr`, `data/seq_rr`, `data/seq_zz`
