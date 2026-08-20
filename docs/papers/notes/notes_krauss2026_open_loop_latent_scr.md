# 阅读笔记：Accurate Open-Loop Control of a Soft Continuum Robot Through Visually Learned Latent Representations

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
- 结果：ABCD 一致降低开环控制误差；Koopman+ABCD 总最优（MSE 1.03e-2），VON 次之（9.80e-3）。**upswing（超出数据范围）任务失败归因于底层压力控制器跟不上快速压力变化——模型未包含执行器动态**。
- 消融 7 项（线性 B、β、rest loss、多步 loss、Rayleigh 阻尼、隐式阻尼等）全部有贡献；明确观察到 **"多步 MSE 低 ≠ 开环控制好"**。
- 仿真应力测试：静态保持、外推 ramp-up、释放后松弛回静止态，ABCD 模型漂移更小。

## 自述局限与未来工作（原文结论）
1. 从开环走向**闭环或部分反馈稳定**控制（传感/末端反馈纳入模型学习）——明确写为 future work；
2. **应计入底层压力控制器动态**（当前模型假设指令压力=实际压力，upswing 失败的主因）；
3. （隐含）潜状态为二阶马尔可夫 [z, ż]，无长程记忆结构；无避障；目标依赖自家模拟器生成。

## 与本项目的数学对比（详版见 `docs/papers/2026-08-20_discussion_summary_gap_map.md` §2）

| 维度 | Krauss 2026 | 本项目 StateTransition |
|---|---|---|
| 状态 | 学习潜态 ξ=[z,ż]（VAE，需解码器，指标=图像 MSE） | **显式像素骨架**（15 节点，无需编码器，指标=px/mm） |
| 转移 | (z,ż)_{t+1} = f(z,ż,u)，**二阶马尔可夫** | s_t = s_{t−1} + δ_scale·tanh(Δ)，Δ 含**分数阶幂律记忆核 + 迟滞潜变量 z**（非马尔可夫/长记忆） |
| 稳定机制 | 振荡器 M,D,K 结构 + symplectic 积分（CON 谱系的 ISS 保证） | tanh 有界增量 + delta_scale_max 硬限 |
| 时间 | 50 Hz 物理时间积分 | 帧转移 ~5 Hz（FRAME_DT 0.203s） |
| 控制 | single-shooting 梯度 + 航点/终端/增量/限速 | 同样梯度序列优化，另加**变长 K、避障 keep-out、动作夹在真机执行范围** |
| 执行器动态 | **忽略（承认是失败主因）** | preflight 显式检查压力范围/速率/等值约束（**领先点**） |
| 真机执行 | **已执行开环优化结果**（领先点） | planner 输出尚未上真机 |
| 硬件 | 两段气动、每段 3 腔 2 联动、平面运动 | **几乎同构**（planar-constrained 6ch，[0,1,1,3,4,4]→4 维根动作） |

**定位含义**：开环部署叙事已被占，但其模型类（二阶马尔可夫 + 指数阻尼）**结构上无法表示幂律长记忆**；其自述缺口（闭环/反馈、执行器动态）与本项目资产（K_max 重观测、preflight）正对。竞争窗口存在但收窄。

## 验证状态
- 2026-08-20 抓取 arXiv HTML 全文（arxiv.org/html/2603.19655），公式编号 (1)–(17)、表 I、图 2–4 均核对。
