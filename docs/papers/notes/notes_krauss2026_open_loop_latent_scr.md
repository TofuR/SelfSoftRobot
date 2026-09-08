# 阅读笔记：Accurate Open-Loop Control of a Soft Continuum Robot Through Visually Learned Latent Representations

> **2026-09-08 复核修正：**本笔记下方的项目对比属于早期 StateTransition 版本，不代表当前 hereditary 模型。二阶递归状态不等于只有一步记忆；限速检查也不等于执行器建模。全文已再次核查，当前研究判断见 [Krauss 重叠分析](../../icra2027/krauss_overlap_and_research_decision_v5.md)。2026-03-20 版本为公开预印本，不据投稿页眉认定录用。

> Henrik Krauss, Johann Licher, Naoya Takeishi, Annika Raatz, Takehisa Yairi — arXiv preprint, 2026-03-20（东京大学 / 莱布尼茨汉诺威大学）
> 链接: https://arxiv.org/abs/2603.19655 · 代码: github.com/UThenrik/visual_oscillators_for_SCR · 数据: zenodo.org/records/17812071
> ⚠️ 2026-08-20 已读全文（arXiv HTML），本笔记为全文版，替代此前摘要版。

## 一句话概括
用视频学到的**机制可解释 2D 振荡器潜动力学**（Visual Oscillator Networks + ABCD 注意力广播解码器），在潜空间做 **single-shooting 开环最优控制**，全程无相机反馈跟踪图像空间航点；在**两段气动软臂（每段 3 腔、2 腔等压联动 → 4 有效输入、垂直于相机轴的平面运动）**上真机执行验证。

## 方法（全文细节）

### 模型
- 编码器 φ（β-VAE）：图像 o → 潜坐标 z；**潜状态 ξ = [z, ż]**；潜速度由编码器 Jacobian 链式得到：`ż = (∂φ/∂o)·ȯ`，ȯ 用观测的中心差分。
- 动力学 f_dyn(z, ż, u) 三个变体：
  - **Koopman**：`ξ_{i+1} = A·ξ_i + B(u_i)`（状态线性 + 输入 MLP）；
  - **MLP**：`ż_{i+1} = f_MLP(ξ_i) + B(u_i)`，`z_{i+1} = z_i + Δt·ż_{i+1}`（积分保证运动学一致）；
  - **振荡器（VON）**：`M·z̈ + D·ż + K·(z−z₀) = B(u)`，symplectic Euler + 隐式阻尼 `Γ = diag(I + Δt·M⁻¹D)`，z₀ 可学习静止位。
- 训练：**多步 rollout 损失** L_d^(H)、L_z^(H)，H 随 epoch 增长（课程式）；**静止态损失** L_s（rest 图像必须编码到 z₀ 且在 rest 驱动下保持平衡）；KL 均值修正到 z₀。

### 控制
- 离散时间 50 Hz，给定初始潜状态，对整段控制序列 u(0:T−1) 做 **single-shooting 开环最优控制**，梯度下降穿透潜 rollout。
- 代价 = 航点跟踪（next/closest 两种活跃航点选择）+ 航点精确项 + 终端项 + 控制增量 ‖Δu‖² + **限速罚** `φ(Δu)=‖max(|Δu|−Δu_max,0)‖²`（尊重底层压力控制器能力）。
- **SCR live simulator**（PyQtGraph）：交互式设计静态/动态/外推目标，把设计出的观测映射为各模型的潜航点。

### 实验
- 数据：两段 15 分钟 50 Hz 采集（正弦激励 0–86 kPa + 阶跃激励），**只用正弦训练、阶跃留作验证**。
- 结果：ABCD 在 upswing 之外改善开环控制。包含全部任务的文中数值为 Koopman+ABCD 1.03e-2、VON 9.80e-3；排除 upswing 后分别为 5.45e-3、6.55e-3。原文结论将 Koopman+ABCD 评为整体最强，但不能以较大的全部任务 MSE 解释其排名。**作者认为 upswing 的较大误差很可能与低层压力控制器无法跟随有关，不能写成已唯一确定的原因。**
- 消融 7 项使用验证集上的压力优化 MAE 与多步图像 MSE 指标评价，并非七组真实控制实验；原文明确观察到 **"多步 MSE 低 ≠ 开环控制好"**。
- 仿真应力测试：静态保持、外推 ramp-up、释放后松弛回静止态，ABCD 模型漂移更小。

## 自述局限与未来工作（原文结论）
1. 从开环走向**闭环或部分反馈稳定**控制（传感/末端反馈纳入模型学习）——明确写为 future work；
2. **应计入底层压力控制器动态**（当前模型假设指令压力=实际压力，upswing 失败的主因）；
3. 潜状态为 [z, ż]，可以经递推携带较早历史的影响；其是否充分表示本平台加载历史需要实验，不能由状态阶数直接判断。部分目标来自验证数据，其余来自模拟器，并非所有目标均由模拟器生成。

## 与本项目的数学对比（详版见 `docs/papers/2026-08-20_discussion_summary_gap_map.md` §2）

| 维度 | Krauss 2026 | 本项目 StateTransition |
|---|---|---|
| 状态 | 学习潜态 ξ=[z,ż]（VAE，需解码器，指标=图像 MSE） | **显式像素骨架**（15 节点，无需编码器，指标=px/mm） |
| 转移 | (z,ż)_{t+1} = f(z,ż,u)，扩展状态递推可携带历史 | 早期 StateTransition：s_t = s_{t−1} + δ_scale·tanh(Δ)，含历史项和潜变量；完整扩展状态也可以写成递推，不能据此宣称对方无长期记忆 |
| 数值与结构约束 | 振荡器 M,D,K 结构及带隐式阻尼的积分；具体稳定性结论需检查相应条件 | 早期模型采用 tanh 有界增量及 delta_scale_max；单步增量有界不保证长期轨迹有界或闭环稳定 |
| 时间 | 50 Hz 物理时间积分 | 帧转移 ~5 Hz（FRAME_DT 0.203s） |
| 控制 | single-shooting 梯度 + 航点/终端/增量/限速 | 同样梯度序列优化，另加**变长 K、避障 keep-out、动作夹在真机执行范围** |
| 执行器动态 | 未包含低层压力控制器动态；作者认为与 upswing 误差有关 | preflight 检查范围/速率/等值约束，但不构成执行器模型；Krauss 同样包含输入增量约束思想 |
| 真机执行 | **已执行开环优化结果**（领先点） | planner 输出尚未上真机 |
| 硬件 | 两段气动、每段 3 腔 2 联动、平面运动 | **几乎同构**（planar-constrained 6ch，[0,1,1,3,4,4]→4 维根动作） |

**修正后的定位含义**：视觉动态开环控制与本项目存在明显重叠。有限递归状态可以近似历史响应；当前 hereditary 的有限松弛 bank 也不等于精确分数阶材料模型。增加重观测或 preflight 不自动形成创新，需要和带同等输入约束、历史传播及观测更新的强基线比较。

## 验证状态
- 2026-08-20 抓取 arXiv HTML 全文（arxiv.org/html/2603.19655），公式编号 (1)–(17)、表 I、图 2–4 均核对。
