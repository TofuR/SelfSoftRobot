# HereditaryOperatorModel v2 可解释性审计与研究路线

日期：2026-09-03
性质：开发集诊断与探索性范围综述；不是独立测试或材料参数辨识报告

## 1. 研究问题与结论

研究目标不是只得到低预测误差，而是回答：气动软臂在快速驱动下为什么具有
历史依赖，能否用具有物理、力学和数学依据的有限状态表达这种迟滞，并用训练外
实验验证这些状态的含义。

当前 HOV2 可以支持一个较窄但真实的结论：

> 动作到全身中心线的系统级记忆，可被压缩到率无关 PI 状态和率相关 Maxwell
> 状态；这些显式状态在当前开发轨迹上显著提高预测精度。

当前证据还不能支持更强的结论：

> 学到的每个时间常数、权重或空间模态就是材料真实松弛谱，或者模型已证明在
> 未见高速条件下仍然有效。

关键原因不是 PI/Maxwell 状态方程缺少依据，而是当前神经读出承担了过多静态
构形和记忆映射，多个 Maxwell 模态还存在强抵消。若论文核心是“可解释的迟滞
状态”，建议保留 HOV2 的状态递推，收紧平衡构形和状态到形状的读出，形成
HOV2.1，而不是继续发展一套独立 ISHSM 动力学。

## 2. 数学上可以直接成立的机制链

### 2.1 率无关路径记忆

对伪驱动 \(e_t\)，play/stop 状态为

\[
p_{j,t}=\operatorname{clip}(p_{j,t-1},e_t-r_j,e_t+r_j),
\qquad q_{j,t}=e_t-p_{j,t}.
\]

更新不显含 \(\Delta t\)。同一路径以不同速度遍历时，只要经过相同的输入顶点，
对应顶点的 \(q_j\) 相同。因此 PI 状态是率无关路径记忆的结构化假设。这是
Prandtl--Ishlinskii 算子的数学性质；把它应用到动作到全身中心线是结构迁移，
不是材料定律的直接推导。

### 2.2 率相关松弛

每个 Maxwell 状态满足

\[
\tau_k\dot h_k+h_k=e,
\qquad d_k=h_k-e.
\]

常输入保持后，亏量满足

\[
d_k(t)=d_k(0^+)\exp(-t/\tau_k),
\]

所以 \(d_k\) 描述“当前平衡目标还未实现的部分”。对正弦输入
\(e(t)=\Re\{E e^{\mathrm i\omega t}\}\)，频域传递关系为

\[
\frac{D(\mathrm i\omega)}{E(\mathrm i\omega)}
=-\frac{\mathrm i\omega\tau_k}{1+\mathrm i\omega\tau_k},
\qquad
\left|\frac{D}{E}\right|
=\frac{\omega\tau_k}{\sqrt{1+(\omega\tau_k)^2}}.
\]

当 \(\omega\tau_k\ll1\) 时，\(d_k\approx-\tau_k\dot e\)，动态亏量很小；
当 \(\omega\tau_k\gtrsim1\) 时，亏量显著。这给出了“高速时不能忽略迟滞”的
直接数学依据，但尚需不同速度的真实实验验证该状态是否对应机器人响应。

### 2.3 当前网络的精确语义重写

原模型写作

\[
y_t=y_{\mathrm{static}}(a_t)+y_{\mathrm{PI}}(q_t)
+y_{\mathrm M}(d_t)+r(a_t,q_t,d_t).
\]

不改变任何预测，可精确重写为

\[
\begin{aligned}
y_t={}&y_{\mathrm{eq}}(a_t)+y_{\mathrm{mem}}(a_t,q_t,d_t),\\
y_{\mathrm{eq}}(a)&=y_{\mathrm{static}}(a)+r(a,0,0),\\
y_{\mathrm{mem}}&=y_{\mathrm{PI}}+y_{\mathrm M}
+\big[r(a,q,d)-r(a,0,0)\big].
\end{aligned}
\]

该重写揭示了当前模型的真实含义：它具有显式、可解释的记忆状态，但平衡构形
实际上主要由神经读出完成。论文若使用现 checkpoint，应称
\(y_{\mathrm{eq}}\) 为“learned equilibrium kinematics”，不能称残差为小修正。

为了顺序无关地把神经记忆增量归到 PI 和 Maxwell 两组，对
\(r_{qd}=r(a,q,d)\)、\(r_{q0}=r(a,q,0)\)、\(r_{0d}=r(a,0,d)\)、
\(r_{00}=r(a,0,0)\) 使用两组 Shapley 分解：

\[
\Delta r_{\mathrm{PI}}
=\tfrac12[(r_{q0}-r_{00})+(r_{qd}-r_{0d})],
\]

\[
\Delta r_{\mathrm M}
=\tfrac12[(r_{0d}-r_{00})+(r_{qd}-r_{q0})].
\]

两者严格满足
\(\Delta r_{\mathrm{PI}}+\Delta r_{\mathrm M}=r_{qd}-r_{00}\)。
这是模型归因，不是材料能量分解。

## 3. 已训练 checkpoint 的实证审计

### 3.1 对象与方法

- checkpoint：HOV2，`residual_scale_max=0.5`，validation-best epoch 190；
- 数据：当前 dev 两条轨迹，675 个计分帧；
- 协议：连续算子状态，只读取动作，不读取真实骨架；
- 单位：所有贡献先经过 checkpoint 反归一化，再以中心线节点毫米报告；
- 干预：固定 checkpoint 后置零某一显式分支及其对应残差输入；不重新拟合。

可复现脚本：`scripts/evaluation/analyze_hereditary_interpretability.py`。完整输出：
`workspace/runs/analysis/hereditary_v2_interpretability_20260903_004/`。

### 3.2 记忆状态确实被模型使用

| 固定 checkpoint 干预 | node mean | endpoint mean | 相对完整模型变化 |
|---|---:|---:|---:|
| 完整模型 | 1.337 mm | 2.370 mm | -- |
| 移除全部 PI 信息 | 1.411 mm | 2.466 mm | +0.075 / +0.095 mm |
| 移除全部 Maxwell 信息 | 1.896 mm | 4.683 mm | +0.559 / +2.313 mm |
| 移除全部记忆信息 | 1.937 mm | 4.714 mm | +0.601 / +2.344 mm |

因此，当前数据上有较强证据表明 Maxwell 型状态信息参与了末端预测；PI 信息的
增益较小但非零。该结果证明模型依赖这些状态，不等同于证明自然系统唯一采用
这些机制。严格的“必要性”还需把对应分支删除后重新训练。

### 3.3 物理输出空间中的实际贡献

| 归因组 | node 平均向量幅度 | endpoint 平均向量幅度 |
|---|---:|---:|
| 显式 PI 分支 | 0.236 mm | 0.404 mm |
| PI 信息总贡献（含残差 Shapley） | 0.424 mm | 0.767 mm |
| 显式 Maxwell 分支 | 0.671 mm | 2.801 mm |
| Maxwell 信息总贡献（含残差 Shapley） | 1.304 mm | 4.016 mm |
| PI+Maxwell 总记忆修正 | 1.399 mm | 4.140 mm |

总记忆修正的区域平均幅度从基部 `0.352 mm`、中部 `1.055 mm` 增长到末端
五节点 `2.789 mm`。这种沿链向末端放大的空间规律与弯曲/角度误差积分后的
几何效应一致，是一个有意义的涌现现象，但仍可能同时包含观测和标定因素。

### 3.4 当前解释性的主要反证

1. 3187 个参数中，神经残差网络与幅度参数共 2670 个，占 `83.8%`。
2. 原残差的实际节点平均幅度为 `34.40 mm`；其中平衡项
   \(r(a,0,0)\) 为 `34.29 mm`。归一化上限 0.5 在 y 方向对应约
   `45.38 mm`，所以“0.5 很小”在物理单位下并不成立。
3. 记忆相关残差 \(r(a,q,d)-r(a,0,0)\) 仍有 `1.306 mm` 节点平均幅度，
   大于显式 Maxwell 分支的 `0.671 mm`。状态递推可读，但状态到形状的映射
   仍有较强黑箱成分。
4. 六个 Maxwell 时标各自贡献幅度之和是最终 Maxwell 合成幅度的 `8.39` 倍，
   且存在最高约 `-0.87` 的模态反相关。这表明时间尺度之间严重抵消，不能把
   单个裸权重解释为已辨识的材料谱。
5. 当前 dev 没有全通道恒定保持段；主要动作速度集中在现有采集轨迹附近。
   因而没有直接数据验证指数松弛，也没有训练外速度外推证据。

### 3.5 已有历史消融能说明什么

- `rest, tau_max=10, n_play=8` 的 continuous/window40 为
  `3.759/1.234 mm`；只改为 equilibrium 后为 `1.373/1.366 mm`。
  这强力支持初始化协议修复了人为状态失配，但不是材料机理证据。
- equilibrium 下把 `tau_max: 10 -> 2 s`，continuous 从 `1.373` 到
  `1.356 mm`，改善较小；它主要增强可辨识性，而非带来巨大精度收益。
- `n_play: 8 -> 2` 没有降低精度，说明当前数据不支持复杂 PI 谱。
- residual 0.5 相对 0.3 稳定改善，但两者都触顶，说明原显式读出容量不足。

## 4. 文献证据及其边界

本轮采用探索性范围综述。检索日期为 2026-09-03，检索源为 Crossref 与
OpenAlex；纳入软气动执行器迟滞、PI/Preisach、率相关模型、广义 Maxwell/
有限内变量和结构化神经状态空间模型，排除只有压电/磁致伸缩对象且不能提供
通用算子数学依据的应用论文。

| 文献 | 能支持的内容 | 不能支持的内容 |
|---|---|---|
| de la Morena et al., 2025 | 软气动弯曲执行器确有黏弹迟滞；GPI、Bouc--Wen、Preisach 可用实验比较 | 本项目全身模型的唯一性 |
| Al Saaideh & Al Janaideh, 2022 | PI 可描述负载相关 PAM 迟滞 | PI 权重等于本项目材料参数 |
| Ru et al., 2022/2023 | 软弯曲气动执行器存在率相关和非对称迟滞，必须显式考虑速度 | 当前固定 Maxwell 网格已识别真实谱 |
| Krikelis et al., 2023/2024 | PI 算子可嵌入结构化非线性状态空间网络并用经典神经优化辨识 | “算子进网络”本身是本项目创新 |
| Simo, 1987；Reese & Govindjee, 1998 | 黏弹性可用平衡/非平衡内部变量和多时标描述 | 从动作到中心线等同于严格有限应变本构 |
| Zhang et al., 2024, TII | PI 与 PCC 串联可形成考虑迟滞的软机械臂运动学与控制 | 当前局部模态等同于 PCC/Cosserat 力学 |

最稳妥的论文定位是“rheology-inspired structured grey-box whole-body
kinematic memory model”。在没有沿腔道压力和材料应力/应变测量时，辨识的是
阀、气路充放气、摩擦、材料、结构和视觉观测共同形成的系统级记忆，不能称为
纯材料本构辨识。

## 5. 推荐的 HOV2.1：保留状态，收紧读出

### 5.1 首选结构

保留已验证稳定的 PI/Maxwell 更新，把输出改为可积的广义形状坐标：

\[
\xi_t=\xi_{\mathrm{eq}}(a_t)
+W_{\mathrm{PI}}q_t+W_{\mathrm M}d_t+\epsilon_{\mathrm{mem}}(q_t,d_t),
\qquad y_t=\mathcal G(\xi_t).
\]

其中 \(\xi\) 可由 8 个分段弯曲/曲率坐标和 2 个分段伸长坐标构成，
\(\mathcal G\) 使用确定性链式几何积分重建 15 个中心线节点。这里吸收的是
ISHSM 最有价值的“几何读出”，而不是保留其当前精度不足的状态更新。

该结构的解释为：PI/Maxwell 决定弯曲和伸长状态的历史偏移，几何方程决定这些
状态如何累积为空间位置。它比“每个时间尺度自由学习一个 15 点位移场”更容易
审计，也自然解释误差为何向末端累积。

### 5.2 必须加入的结构约束

1. **平衡与记忆严格分离**：令
   \(\epsilon_{\mathrm{mem}}(0,0)=0\)，残差不能生成平衡构形。
2. **物理单位约束**：在角度、对数长度或最终毫米空间惩罚 residual RMS，
   不再用归一化坐标中的 0.3/0.5 作为“小”的判断。
3. **去除时标尺度自由度**：固定或单位归一化空间/广义坐标模态，并设定符号
   约定；优先报告组合后的物理贡献。
4. **抑制时标抵消**：对模态正交性、相邻时标平滑性或非负松弛幅度施加约束；
   若采用平衡形状 Jacobian 作为方向，可写
   \(\sum_{c,k}g_{ck}d_{ck}\,\partial y_{\mathrm{eq}}/\partial e_c\)，
   其中 \(g_{ck}\ge0\)。
5. **分阶段辨识**：先用充分保持后的样本训练 \(\xi_{\mathrm{eq}}\)，再训练
   PI/Maxwell，最后只在必要时开启零平衡记忆残差。

## 6. 能形成论文因果链的实验

### A. 同数据重训练消融

| 模型 | 要回答的问题 |
|---|---|
| equilibrium backbone only | 当前动作能解释多少 |
| backbone + PI | 率无关路径记忆是否必要 |
| backbone + Maxwell | 率相关状态是否必要 |
| backbone + PI + Maxwell，无神经记忆残差 | 显式算子是否足够 |
| 完整 HOV2.1 | 小残差是否提供稳定增益 |
| GRU/LSTM | 相同数据下解释性约束的精度代价 |

这些模型必须相同 split、训练预算和 validation-best 选模，并同时报告 node、
endpoint、参数量、推理时间和跨 seed 结果。

### B. 物理机制验证

1. 对同一幅值三角路径使用至少四个安全速率，保持路径顶点一致；留出最快速率
   完全不参与训练。
2. 在若干动作电平执行 step--hold--release，保持时间至少覆盖
   \(5\tau_{\max}\)；当前 \(\tau_{\max}=2 s\)，因此建议观察约 10 s 尾部。
3. 采集“相同当前动作、加载与卸载两种历史”配对样本。
4. 若不能测沿腔道压力，至少记录阀指令、阀反馈/入口压力、时间戳和全身形状；
   结论保持在系统级运动学记忆。

预注册判据：

- 低速时 Maxwell-only 增益应减小，高速时增益应扩大；
- hold 段的 Maxwell 归因应趋零，PI 归因可保持；
- 对应路径顶点的 PI 状态跨速率应稳定；
- 完整模型在训练未见速率上应显著优于 memoryless 与 PI-only；
- 归一化后的快/中/慢时间带贡献应跨 seed 稳定，且抵消比显著低于当前 `8.39`。

只有上述预测被真实数据支持，论文才能把“高速条件下显式迟滞状态仍有效”写成
主要实验结论。

## 7. 检索记录

| 数据库 | 日期 | 查询 | 返回数 | 用途 |
|---|---|---|---:|---|
| Crossref | 2026-09-03 | `soft pneumatic actuator hysteresis Prandtl Ishlinskii` | 20 | PI/PAM 应用 |
| Crossref | 2026-09-03 | `rate-dependent hysteresis soft pneumatic actuator viscoelastic` | 20 | 率相关软执行器 |
| Crossref | 2026-09-03 | `structured nonlinear state-space hysteresis neural operator` | 20 | 结构化辨识 |
| OpenAlex | 2026-09-03 | `soft pneumatic actuator hysteresis viscoelastic` | 20 | 综述与对象证据 |
| OpenAlex | 2026-09-03 | `Prandtl Ishlinskii soft pneumatic actuator` | 20 | PI 与软机械臂 |
| OpenAlex | 2026-09-03 | `grey-box soft robot hysteresis model` | 20 | 灰箱边界 |

## 8. 核心参考文献

1. de la Morena, J., Ramos, F., & Vázquez, A. S. (2025). Hysteresis
   Modeling of Soft Pneumatic Actuators: An Experimental Review. *Actuators*,
   14(7), 321. https://doi.org/10.3390/act14070321
2. Al Saaideh, M., & Al Janaideh, M. (2022). On Prandtl--Ishlinskii
   Hysteresis Modeling of a Loaded Pneumatic Artificial Muscle. *ASME Letters
   in Dynamic Systems and Control*, 2(3). https://doi.org/10.1115/1.4054779
3. Ru, H., Huang, J., Chen, W., & Xiong, C. (2022). Modeling and
   identification of rate-dependent and asymmetric hysteresis of soft bending
   pneumatic actuator. *Mechanism and Machine Theory*, 181, 105169.
   https://doi.org/10.1016/j.mechmachtheory.2022.105169
4. Krikelis, K., Pei, J.-S., van Berkel, K., & Schoukens, M. (2024).
   Identification of structured nonlinear state-space models for hysteretic
   systems using neural network hysteresis operators. *Measurement*, 224,
   113966. https://doi.org/10.1016/j.measurement.2023.113966
5. Simo, J. C. (1987). On a fully three-dimensional finite-strain
   viscoelastic damage model. *CMAME*, 60(2), 153--173.
   https://doi.org/10.1016/0045-7825(87)90107-1
6. Reese, S., & Govindjee, S. (1998). A theory of finite viscoelasticity and
   numerical aspects. *IJSS*, 35(26--27), 3455--3482.
   https://doi.org/10.1016/S0020-7683(97)00217-5
7. Zhang et al. (2024). Kinematic Modeling and Control for an Elephant-Trunk
   Soft Manipulator Considering Hysteresis. *IEEE Transactions on Industrial
   Informatics*. https://doi.org/10.1109/TII.2024.3403250

更完整的公式--引用对应关系见
`docs/papers/hereditary_v2_theoretical_foundation.md`。
