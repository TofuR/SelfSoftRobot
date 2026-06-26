# 方向：视觉辅助部署（Vision-Corrected Deployment）

> 状态：待探索
> 优先级：中高（如果目标是真实机器人部署）
> 关联：PC-Spatial 修正分支、Tang 2026 在线适应、**[13](13_closed_loop_state_transition.md)/[14](14_gt_observed_transition.md)/[15](15_open_loop_windowed_transition.md) 状态转移族**
> 📝 **2026-06-17 更新**：本方向的"预测-修正 / 最小化视觉使用"已被状态转移族**具体落地**，按 s_{t-1} 来源分三档：[14](14_gt_observed_transition.md)=每步视觉重观测（修正最密）、[15](15_open_loop_windowed_transition.md)=观测一次→开环预测 K 步（**正是本方向 Phase 2\"最小视觉修正层\"的精确实现**：每 K 步一次图像骨架化作种子）、[13](13_closed_loop_state_transition.md)=运行中不重观测的无界 rollout。下文 Phase 2 视觉修正层的主参考改为 [15](15_open_loop_windowed_transition.md)。
> 🔁 **2026-06-24 补充**：控制侧落地见 [16 约束导向控制](16_constraint_oriented_control.md)（前向模型作可微黑箱求逆）；实物视觉修正的数据路径已打通（[11 §最小验证平台](11_sim_to_real_transfer.md) + 采集程序 `docs/ref/Main UI-plc/`）。

---

## 问题

当前项目的设计目标是"不用视觉，只从驱动参数预测形状"。但：
1. 纯驱动参数预测的精度有限（迟滞、蠕变、温度漂移）
2. 真实机器人部署时必然有误差（sim-to-real gap）
3. Tang 2026 证明了在线视觉反馈可以显著提升形状控制精度

**问题：如果部署时有相机，如何最小化地利用视觉信息修正预测？**

---

## 方案

### A. 预测-修正框架（PC-Spatial 已部分实现）

当前 PC-Spatial 的设计：
- 预测分支：action_history → 3D skeleton（无视觉）
- 修正分支：2D image features → skeleton residual

问题：
1. 修正分支需要 2D 图像，但 CLAUDE.md 规定"图像/深度仅作监督信号"
2. 修正分支在训练时效果有限

**重新定义设计目标**：
- 训练时：只用 action 监督（保持"物理白盒"理念）
- 推理时：可选地加入视觉修正（作为后处理/在线适应层）

这样既保留了核心贡献（无视觉的自建模），又为实际部署提供了精度提升路径。

### B. 在线策略优化（Tang 2026 启发）

Tang 2026 的核心创新：
1. 离线训练 CNN 策略（图像→动作）
2. 部署时遇到新情况，利用形状模型在线修正策略

适配到我们：
1. 离线训练 action→skeleton 模型（当前主线）
2. 部署时，从相机获取 2D 骨架/轮廓
3. 在线优化：微调模型参数使预测骨架的 2D 投影与观测对齐
4. 微调后的模型适应当前环境（外力、温度等）

优点：不需要预先知道环境变化，可以在线适应
缺点：在线优化需要计算资源；可能过拟合当前帧

### C. 残差物理学习（ETH RA-L 2024）

Gao et al. 的方法：
1. 物理仿真器给出粗略预测
2. 神经网络学习残差修正
3. 少量真实数据即可训练残差网络

适配到我们：
1. PyElastica 仿真 = 物理先验
2. 神经场模型 = 学习映射
3. 真实数据 = 训练残差

关键洞察：我们的模型本质上已经在做"学习 PyElastica 的输出"。加入真实数据后，只需学习 sim-to-real 的残差。

### D. 多任务自监督

训练时同时优化：
1. Action→Skeleton 主任务
2. Skeleton→2D Rendering 辅助任务
3. 对比学习：同一 action 序列的不同视角应该一致

推理时：
- 如果有视觉输入，用辅助任务的解码器提供额外约束
- 如果没有视觉输入，只用主任务

---

## 实施路线

### Phase 1：纯驱动模型（当前主线）
- 完成 Gamma/Laguerre 编码器验证
- 解决超前预测问题
- 目标：骨架预测 mm 级精度

### Phase 2：视觉修正层（可选）
- 训练独立的修正网络：2D 特征 → 骨架残差
- 在推理时作为可选模块接入
- 不影响纯驱动模式的完整性

### Phase 3：真实数据迁移
- 收集真实机器人数据（如果有硬件）
- 训练残差修正网络（ETH 方法）
- 在线适应微调（Tang 2026 方法）

---

## 相关文献

- Tang 2026 (ICRA Poster)：CNN + 在线策略优化适应未知负载
- ETH RA-L 2024：残差物理 sim-to-real (Best Paper Award)
- PC-Spatial 模型：`src/models/model_pc_spatial.py`
- Yu 2026 (arXiv)：视觉感知管线（Bézier + 双视角）
- Yang et al. (CoRL 2025)：解耦 sim-to-real
