# 方法如何写：从任务定义到可执行更新

以下公式按 `96e684f` 实现整理。符号是论文建议记法；先说明设计动机，再给公式，再说明如何落到代码。已有机制、可证明的局部性质和待实现扩展分开标识。

## 1. 任务、传感与动作合同

令四维有效动作为 \(u_k\in\mathbb R^4\)，六腔下发压力为 \(a_k=M\operatorname{diag}(s)u_k\in\mathbb R^6\)。\(M\) 是每行一个 1 的腔道映射矩阵，\(s\) 来自训练尺度。当前包动作界为 0–150 kPa，名义变化率上限 50 kPa/s，实际还与 GUI 各腔限制取交集。不能把映射当作模型自动适应任意腔道语义。

目标中心线为 \(X^*=[x_0^*,\ldots,x_{N-1}^*]\)，\(N=15\)，基点固定；本平台验证受控平面运动。当前图像 \(I(t_j)\) 只提供部分形状证据 \(\mathcal E_j\)，目标是依靠历史与这些证据，使**真实**形状接近目标；训练/规划中使用预测形状，实验另用独立真值评价。

建议图 2 同时画出：动作/ACK、时间戳历史、模型状态、当前预测、局部证据、状态候选、动作候选、提交门。独立评价原图/NDI画为旁路，不可连入控制。

## 2. 预测模型：历史状态与几何读出

### 2.1 显式历史递推

设驱动为 \(e_c(u_c)\)，可学习单调分段线性函数；play 阈值为 \(r_i\)，Maxwell 衰减为 \(\alpha_j=\exp(-\Delta t/\tau_j)\)。状态为 \(z=(p,h)\)。

\[
p_{c,i,k}^{-}=\operatorname{clip}\big(p_{c,i,k-1}^{+},\ e_c(u_{c,k})-r_i,\ e_c(u_{c,k})+r_i\big),
\tag{1}
\]
\[
h_{c,j,k}^{-}=\alpha_j h_{c,j,k-1}^{+}+(1-\alpha_j)e_c(u_{c,k}).
\tag{2}
\]

每输入 2 个 play 与 6 个 Maxwell 状态，总维度 \(4(2+6)=32\)。play 提供加载路径相关的持久记忆，Maxwell 提供有限时间尺度松弛；二者是**系统级建模结构**，不能由结构名称推出真实材料机理。

实际持压 \(\delta t\) 时使用 \(\alpha_j(\delta t)=\alpha_j^{\delta t/\Delta t}\)。新动作先将 play 投影到新驱动允许区间，Maxwell按真实已过时间积分。预测读出 `observe` 不推进时间，避免同一图像既预测又校正时重复推进状态。

### 2.2 形状由几何参数生成

定义 \(q=e(u)-p\)，\(d=h-e(u)\)。当前无神经残差部署包可写成

\[
\xi_k=\xi_{\rm ref}(u_k)+W_pq_k+W_hd_k,
\quad \hat X_k=G(\xi_k).
\tag{3}
\]

\(\xi\) 是 14 个局部弯曲角与 2 个段对数长度参数。矩阵含训练中确定的方向与尺度；\(\xi_{\rm ref}\) 可为单调样条驱动的参考项。对第 \(i\) 段：

\[
\theta_i=\sum_{r=1}^{i}\beta_r,\quad
\ell_i=\ell_i^0\exp(\operatorname{clip}(\lambda_{s(i)},-0.25,0.25)),
\quad
\hat x_i=\hat x_0+\sum_{r=1}^{i}\ell_r[\cos\theta_r,\sin\theta_r]^T.
\tag{4}
\]

这里的“全身形状”是中心线几何；GUI 管身由模型半径扩展，并非当前训练了三维体积神经场。不要沿用早期 NeRF/分数阶编码器的公式。

### 2.3 学习与部署的分工

离线学习动作—形状序列的多步预测，按实际配置报告 skeleton/bend/length/endpoint 等损失及权重、初始状态约定、episode 与 burn-in。不能将未核对的统一损失写成已采用。模型选择、归一化和几何基均限定于训练/验证合同，测试集不参与。

在线权重 \(\theta\) 固定，变化的是 \(z\)。这使状态校正与重新辨识参数有明确区别。预测精度、参数数量、训练效率都是待比较指标，不是结构命名即可证明的贡献。

## 3. 从部分图像构造误差，而非恢复完整骨架

用初始化得到并固定的相似变换 \(T(x)=s_cRx+b\) 将模型坐标转为像素。沿预测中心线内部线段的两侧法向寻找候选边缘，按对比度、梯度方向、颜色、歧义和去重规则保留
\(\mathcal E_j=\{(y_m,i_m)\}\)。\(i_m\) 是关联线段，不是把残段重新编号成完整臂身。

设线段端点 \(v_i=T(\hat x_i)\)，像素半径 \(R_p=s_cR_0\)。

\[
\eta_m=\operatorname{clip}\!\left(
\frac{(y_m-v_i)^T(v_{i+1}-v_i)}{\|v_{i+1}-v_i\|^2},0,1\right),\quad
c_m=(1-\eta_m)v_i+\eta_mv_{i+1},
\]
\[
r_m(z)=\frac{\|c_m(z)-y_m\|-R_p}{\sigma_{\rm px}}.
\tag{5}
\]

当前 \(\sigma_{\rm px}=2\)。这比较的是可信边缘到预测管身的距离，不是中心线点到像素的距离。每次局部更新固定像素—线段关联；分段投影和 play 切换使导数只在当前分支内成立。

遮挡区域无需人工标注。覆盖率只统计可用内部线段；当前 15 节点的 14 段中，检测器排除首段和末段。缺证据只标为“可能遮挡/失配”，没有遮挡概率分类器。复杂纹理、相似颜色遮挡物、背景边缘仍可能产生错误关联，必须做实验证明适用范围。

## 4. 带历史先验的状态校正

从图像时间的历史缓冲取得 \(z_j^-\)，定义

\[
\delta z^*=\arg\min_{\delta z}\ \sum_{m\in\mathcal E_j}\rho(r_m(z_j^-+\delta z))
+\|\delta z/\sigma_z\|^2,
\tag{6}
\]

并对更新范围及有效状态域施加限制。当前采用一次鲁棒加权 Gauss–Newton 候选：

\[
\delta z=-(J_z^TWJ_z+\sigma_z^{-2}I)^{-1}J_z^TWr,
\quad W_{mm}=\frac{2}{\max(|r_m|,2)}.
\tag{7}
\]

代码使用 \(\rho(r)=r^2\)（\(|r|\le2\)），否则 \(4|r|-4\)，\(\sigma_z=0.08\)，最大状态增量无穷范数 0.12。候选先投影有效 play/Maxwell 范围，再在 1、1/2、1/4、1/8 比例上检验“图像稳健误差＋先验”下降；未下降则保留原状态。不是把每次 Gauss–Newton 都视为成功。

**为什么必须保留先验？** \(z\) 为 32 维，而当前固定输入下 \(G\circ\xi\) 至多经 16 维几何参数传递。单帧完整形状 Jacobian 的秩上界已是 16，部分边缘只会进一步减少约束。因此不能声称一帧唯一恢复 p/h；先验保留未被观测约束的方向，历史传播提供后续信息。时间序列是否能充分辨识状态，需要跨时刻可观测性和预测实验，不能由正则项自动推出。

## 5. 迟到图像与已确认动作的时间对齐

若图像时刻 \(t_j\) 早于当前动作边界，先在该时刻校正，再依序重放 \((t,a^{\rm applied})\) 记录，得到当前候选状态。写为

\[
z_{\rm now}^{c}=\mathcal F_{a_{[t_j,t_{\rm now}]}}(z_j^-+\delta z^*).
\tag{8}
\]

注意使用 ACK 确认的实际**下发值**，不是计划请求值，也不是传感器实测气压。图像时间是主机接收时间，当前不是硬件曝光同步。实验需测接收延迟与图像年龄，特别是快速动作时。

## 6. 剩余动作的约束局部修订

旧剩余序列为 \(U^0\)，参考形状为 \(\bar X\)。以 \(\Psi\) 将最多 8 个动作块的增量 \(\eta\) 展开到全序列：\(U=U^0+\Psi\eta\)，优化变量最多 32 维。原序列块内仍可变化，限制的是**修订增量**。

\[
\hat X_{1:H}=\mathcal R(z_{\rm now}^{c},U^0),\quad
b=\operatorname{vec}(\hat X_{1:H,1:N-1}-\bar X_{1:H,1:N-1}).
\]
\[
\min_\eta\ \frac{1}{2m}\|b+J_U\Psi\eta\|^2+\frac{\lambda}{2}\|\eta\|^2
\quad\text{s.t.}\quad U_{\min}\le U^0+\Psi\eta\le U_{\max},
\]
\[
-\Delta U_{\rm fall}\le D(U^0+\Psi\eta;u_{\rm last})\le\Delta U_{\rm rise},
\quad \|\eta\|_\infty\le0.08.
\tag{9}
\]

\(m\) 为残差标量数量，\(D\) 包含从最后下发动作到第一条剩余动作的接续。当前 \(\lambda=2\)，以 SLSQP 求解该二次目标与线性约束，并合并重复约束行。完整非线性 rollout 回溯检查均方误差下降和动作可行性后才接受。不要把该方法写成没有线性化误差的全局求解器。

解析链式导数来自式(1)–(4)，沿时间递推状态敏感度并批量计算读出。**没有 A 预计算缓存，也没有依赖 GPU。** A 是对照路线：缓存名义轨迹附近的线性影响，需独立统计失配与投影后未改善的拒绝原因。

初始规划外层尝试不同 H、复用候选，必要时加四维终态压力 shooting；均值/最大节点误差达标即停。若 shooting 被采用，其可行预测轨迹成为后续跟踪参考。当前执行逐步缩短有限后缀，不是无限期延长视野的标准滚动 MPC，也没有到达后的额外目标保持控制器。

## 7. 状态与动作作为一次有期限的提交

令本步发令时间为 \(t_k^{cmd}\)，预留 \(\epsilon=3\) ms：

\[
D_k=t_k^{cmd}+\Delta t-\epsilon,\qquad
(z^+,U^+)=
\begin{cases}
(z^c,U^c),&t^{finish}<D_k\ \land\ \text{snapshot version valid},\\
(z^-,U^{fallback}),&\text{otherwise}.
\end{cases}
\tag{10}
\]

反馈线程只修改私有快照。超时整个状态/动作事务丢弃，旧合法后缀继续；并发任务最多一个，未结束则本拍不再排队。至少间隔模型 dt 发下一条命令，不突发补发。额外等图默认 0，但必须是 ACK 后的新图像。

“状态已提交”“状态值发生改变”“优化器候选下降”“后缀实际改变”是四件不同的事，要分别记录。当前连续无有效图像/边缘默认 3 次停止归零；连续无法及时提交默认 10 次，可配置。全遮挡没有稳定跟踪保证；归零后的物理运动也需在故障实验中评价。

## 8. 可以证明什么，不能证明什么

- 可由代码合同和测试证明：期限后候选不提交；任务不积压；候选动作在声明约束内；被采用的局部候选改善了所用预测目标。
- 可给出数学说明：单帧状态估计欠定，局部正则解存在；当前分支敏感度链式递推成立。
- 尚不能证明：唯一恢复真实内部状态、全遮挡可观测性、全局收敛、真实轨迹误差单调下降、闭环渐近稳定、递归可行或碰撞安全。
- 可作为后续分析：时窗信息矩阵 \(\sum_j\Phi_j^TJ_j^TW_jJ_j\Phi_j\)、可见段位置对小奇异值的影响、模型失配和迟到输入下的误差界。现实现未基于这些量自动调节风险/探测动作，不应把建议写成已实现模块。

## 9. 公式到代码的索引

| 内容 | 实现 |
|---|---|
| 离线模型、几何读出 | [model_hereditary_geometry.py](../../../src/models/model_hereditary_geometry.py) |
| 冻结递推、读出和解析导数 | [hereditary_math.py](../../../real_validation/runtime/hereditary_math.py) |
| 时间传播、状态校正、初始规划、历史重放 | [hereditary_deployment.py](../../../real_validation/runtime/hereditary_deployment.py) |
| 局部可见边缘 | [partial_edges.py](../../../real_validation/perception/partial_edges.py) |
| 压力/速率约束与动作块 | [hereditary_bounds.py](../../../real_validation/runtime/hereditary_bounds.py) |
| 期限快照与提交 | [deadline_feedback.py](../../../real_validation/execution/deadline_feedback.py) |
| 执行顺序、失效终止、逐步审计 | [hereditary_executor.py](../../../real_validation/execution/hereditary_executor.py)、[experiment_archive.py](../../../real_validation/execution/experiment_archive.py) |
