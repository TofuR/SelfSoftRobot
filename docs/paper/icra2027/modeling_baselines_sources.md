# 全身形态建模基准：论文、官方源码与适配建议

核查日期：2026-09-12。机器可读记录：[modeling_sources.json](modeling_sources.json)。

## 1. 选择结论与当前任务

主表建议包含线性/二次多项式、静态 MLP、窗口 MLP，以及 **Chen 2025 压力＋方向 MLP、Park 2024 因果 TCN、Schäfke 2024 GRU/LSTM 架构适配**。前三篇提供相互独立的近期依据，分别检验末次方向、有限历史卷积和门控递归记忆。Yu 2026 的分段二次 Bézier 适合做同一预测器的几何读出对照。Krauss 的振子/Koopman 核适合后续增加结构化动态基线；SoftNeRF 的完整视觉自建模与当前监督、输出预算差别较大，列作表示扩展。

本文记录文献核查与源码获取的原始审阅范围。当前已接入Chen方向网络、Park TCN、SPONGE官方GRU/LSTM类、Bézier读出及VON方程适配，完成真实数据短运行；精确实现与验证范围见[实验协议](modeling_experiments.md)和[验证记录](modeling_validation.md)。原作者完整训练程序和原平台性能仍未复现。

当前统一任务为

\[
U_t=[u_{t-L+1},\ldots,u_t]\in\mathbb R^{L\times d_u},\qquad
\widehat S_t=F_\theta(U_t)\in\mathbb R^{N\times d_s}.
\]

`d_u` 取主线统一后的独立压力通道数，`d_s` 取统一评价坐标维数。输入是截至当前时刻的压力窗口，标签是当前骨架；原论文中的一步预测目标、当前实测形状、图像、速度、窗口外状态不能直接带入这个任务。压力命令与实测气压必须明确选择并对所有方法一致，保留主线已经核对的数据时间对齐。

建议对所有方法声明以下接口约束：

- 每个窗口独立初始化。GRU/LSTM 的 `h/c`、振子的 `z/v`、HOV 分支的历史状态都不能携带窗口之前的信息；固定零值、训练得到的共享初值或仅由窗口首个压力产生的初值，应明确记录。固定窗口成绩只支持有限历史预算下的结论。
- DirectionMLP 从窗口内的相邻差分得到方向；首个差分不能读取 `u[t-L]`。默认窗口起点方向为零，持压时继承该窗口内最后一个非零方向；阈值和初始化属于本项目选择。`L=1` 时方向全零。
- 对 RNN，窗口顺序为旧→新，取末时刻输出；TCN 只在左边补齐，不进行双向卷积。训练/验证/测试先按独立采集序列划分，再生成窗口，归一化与读出拟合只使用训练集。
- 静态 MLP 使用 `u_t`，方向模型使用 `u_t,d_t`，二者是相同可用预算下主动压缩信息的基线。窗口 MLP、TCN、GRU、HOV 均可利用整个 `U_t`。

## 2. 官方来源与获取结果

“官方”依据是论文正文链接，或论文→作者项目页→仓库的可追溯链接；GitHub 搜索命中本身不构成官方身份。已下载 **3 个有模型源码的官方仓库**：

| 方法/仓库 | 论文至仓库的证据 | 本地目录（相对项目根目录） | 固定 commit | 许可与可用程度 |
|---|---|---|---|---|
| Schäfke / SPONGE | [论文](https://arxiv.org/abs/2411.05616) I 节脚注→[作者项目页](https://tlhabich.github.io/sponge/rnn_mpc)→[官方目录](https://github.com/tlhabich/sponge/tree/main/rnn_mpc/software) | `workspace/third_party/modeling_baselines/sponge` | `fffb24a063475a3bca09f04968be0d165ad3f90a` | 根目录 MIT；模型、训练/HPO 与控制代码存在。最适合直接抽取模型类。 |
| Krauss / visual_oscillators_for_SCR | [开环论文](https://arxiv.org/abs/2603.19655) II-C 脚注及 III-A→[官方仓库](https://github.com/UThenrik/visual_oscillators_for_SCR) | `workspace/third_party/modeling_baselines/visual_oscillators_for_SCR` | `994bca02a1788cde19b0317b2836409af4057bfc` | 未发现许可证；VAE、ABCD、Koopman、振子及训练 notebook 存在。当前代码对应 VON 表示论文，见第 7 节。 |
| SoftNeRF | [出版论文](https://doi.org/10.1109/IROS58592.2024.10801344) 摘要明确提供 [官方仓库](https://github.com/IRMVLab/soft-nerf) | `workspace/third_party/modeling_baselines/softnerf` | `317f80f651edba17613f9e7567d17a9008cfbb79` | 未发现许可证；条件 SDF、SIREN 编码、渲染器与数据类存在，未找到完整训练入口。 |

SPONGE 稀疏检出根目录文件和 `rnn_mpc/`；Krauss 稀疏检出根目录文件和 `configs/`，未检出大体积 `results/`；SoftNeRF 为完整浅克隆工作树。稀疏克隆的首次一体化初始化失败，之后通过 `clone --no-checkout`、独立设置 sparse-checkout、checkout 成功恢复。各目录均保留独立 Git 元数据，未改第三方源码。

静态验证：SPONGE 2、Krauss 3、SoftNeRF 20 个 Python 文件通过 AST 语法解析；三个第三方工作树均干净。**这证明源码已获取且可静态阅读，不证明依赖兼容、训练入口可运行或复现成功。** 文件大小、SHA-256、源码行号和获取范围均在 JSON 与 `workspace/third_party/modeling_baselines/_provenance/` 中记录。

未发现许可证的仓库只作为本地来源审阅，不将“公开可下载”写成“已有明确开源复用许可”。后续主线可依据论文独立实现并引用；直接复制或分发这些源码前需明确作者授权。SPONGE 抽取代码应保留 MIT 声明。

| 其余优先论文 | 本次核查范围 | 结论 |
|---|---|---|
| Chen 2025 | 原文 HTML、当前 arXiv 摘要、题名及编号的 GitHub 仓库检索 | 本次未找到论文可验证关联的官方代码；可根据 III-A 独立实现。 |
| Yu 2026 | 原文 HTML、当前 arXiv 摘要、题名及编号的 GitHub 仓库检索 | 本次未找到官方代码；曲线表示与 NODE 方程足够支持明确范围的思想复现。 |
| Park 2024 | 原文 HTML、当前 arXiv 摘要、题名及编号的 GitHub 仓库检索 | 本次未找到官方代码；TCN 块结构可按原文实现。搜索到其他作者的 Transformer 页面，不作为本论文官方代码。 |

编号检索产生大量聚合页和弱相关结果；未据此认定仓库身份，也未穷尽所有作者页面。因此这里的状态是“本次未找到”，并非断言论文没有发布代码。arXiv 页面的 `html_feedback`、LaTeXML 链接属于平台工具，已从模型代码线索中排除。

## 3. Chen 2025：方向条件全身 MLP

**论文**：[Hysteresis-Aware Neural Network Modeling and Whole-Body Reinforcement Learning Control of Soft Robots](https://arxiv.org/abs/2504.13582)，核对本地 v2 原文 III-A、III-B、V-A。[已有原文卡片](../../papers/review_partial_observation_20260910/core_papers.md#chen2025)。

原文输入为三腔压力与各腔压力变化方向共 6 维；**4 个隐藏层，每层 128 单元**，输出 `3n` 个关键点坐标，再用 B-spline 表示连续形态。训练数据来自真实 OptiTrack 测量；每次设压等待 3 s，再以 20 Hz 记录 1 s，主要是准静态方向迟滞数据。

建议适配：

\[
d_{j,k}=\begin{cases}
\operatorname{sgn}(u_{j,k}-u_{j-1,k}),&|u_{j,k}-u_{j-1,k}|>\epsilon,\\
d_{j-1,k},&\text{otherwise},
\end{cases}\quad d_{0,k}=0,
\]
\[
\widehat S_t=\operatorname{reshape}\left(W_o\,\sigma(W_4\sigma(W_3\sigma(W_2\sigma(W_1[u_t,d_t]+b_1)+b_2)+b_3)+b_4))+b_o\right).
\]

层结构：`2*d_u → 128 → 128 → 128 → 128 → N*d_s`。隐藏层选 ReLU、输出线性，是明确标注的适配选择；所核对的建模段没有给出隐藏层激活函数，不能将控制策略的 Tanh 当作建模网络设定。持压方向规则同样是适配选择。对照静态 MLP 保持同样隐藏层，仅删除方向输入。

原文式 (3)–(5) 还给出运动范围加权损失。将坐标展平为 `D=N*d_s` 后，适配形式为

\[
r_i=\max_{\rm train} y_i-\min_{\rm train} y_i,\quad
w_i=1+\frac{D r_i}{\sum_jr_j},\quad
\mathcal L=\frac1D\sum_iw_i(\hat y_i-y_i)^2.
\]

若分母为零则使用全一权重。注意原文是运动范围越大权重越大，不是逆范围归一化；原文 `n` 在关键点与输出维度之间的记号略有歧义，上式明确使用展平坐标维度。主表建议统一损失，以便比较架构；原文加权损失作为额外配置，权重只能从训练数据计算。

**命名**：`Chen2025-inspired DirectionMLP`。用本项目动态序列和骨架监督重训练、改变腔数/点数或损失后，是论文结构与方向特征的适配，不是完整 HAW-NN＋RL 原实验复现。方向只能表示当前分支，不能先验覆盖持压松弛与更早的嵌套加载路径，这正是其与固定历史方法的可检验差异。

## 4. Park 2024：有限窗口因果 TCN

**论文**：[Hysteresis Compensation of Flexible Continuum Manipulator Using RGBD Sensing and Temporal Convolutional Network](https://arxiv.org/abs/2402.11319)，核对 IV-C、式 (19)–(20)、Fig. 8。[已有卡片](../../papers/review_partial_observation_20260910/core_papers.md#park2024)。

原方法学习 Bowden 传动连续体机器人命令与实际关节量的映射，再做迟滞补偿。输入历史维数 5，输出维数 5；RGB-D 和标记参与测量，并不是当前动作输入之外可以免费加入本基准的观测。原文比较 FNN、LSTM、TCN、TCN-LSTM，窗口取 `10,50,65,80,100,120`，报告每个模型 3 次随机初始化。原文约 800 参数的 TCN 不应被直接替换成大通道网络后还称为原配置。

明确可实现的原文结构：

\[
h^{(b+1)}=h^{(b)}+
\operatorname{ReLU}\!\left(C^{(b)}_{k=3,d=2^b,2}
\operatorname{ReLU}(C^{(b)}_{k=3,d=2^b,1}h^{(b)})\right).
\]

每个 `C` 使用左侧 padding `2*2^b`，保持时间长度；该公式给出建议的激活放置，原文未完全明确残差相加之后是否还加 ReLU。取最后一个时刻，经线性头输出 `N*d_s`。推荐起始配置为 `channels=d_u`，同宽残差直接相加；扩宽配置另设 `1×1` 输入投影，并标为容量对照。

两次卷积/块的感受野为

\[
R_B=1+2(k-1)\sum_{b=0}^{B-1}2^b=1+4(2^B-1),\qquad
B=\left\lceil\log_2\left(1+\frac{L-1}{4}\right)\right\rceil.
\]

`L=10/50/80` 对应 `B=2/4/5`，与原文例子一致。`L=1` 可定义为零残差块＋线性头，并在配置中记录。原文式 (20) 排版的括号容易误读，建议按上述感受野条件选择块数。五维等宽、5 块、每块两个带偏置卷积的参数量是 `5×2×(3×5×5+5)=800`；因此**等宽 5 是根据参数量作出的合理推断**，不是原文明示的通道配置。新加骨架头后不能再报告“800 参数”。

**命名**：`Park2024-inspired causal TCN`。推荐同时运行原文风格的小通道配置和与 GRU 大致匹配容量的配置；只调整本模型的宽度而不给其他模型相当调参预算，会使“架构优劣”的解释不充分。报告窗口持续时间以及采样周期，原文的 80 步不能直接解释为本平台最优。

## 5. Schäfke 2024：最适合抽取的官方 GRU/LSTM

**论文**：[Learning-Based Nonlinear Model Predictive Control of Articulated Soft Robots Using Recurrent Neural Networks](https://arxiv.org/abs/2411.05616)，DOI [10.1109/LRA.2024.3495579](https://doi.org/10.1109/LRA.2024.3495579)，III-A、Algorithm 1。[官方模型文件](../../../workspace/third_party/modeling_baselines/sponge/rnn_mpc/software/rnn_training/NN_fcn.py)。

源码接口：

| 类/函数 | 文件行号 | 输入与输出 | 本基准建议 |
|---|---|---|---|
| `GRU(input_dim, output_dim, hidden_dim, num_layer, dropout)` | `NN_fcn.py:156` | `forward(x, ht)`；`x:[B,L,input_dim]`，`ht:[layers,B,H]`；返回 `[B,output_dim], ht` | `input_dim=d_u`，`output_dim=N*d_s`，每窗零初始化，直接取末输出。 |
| `LSTM(...)` | `NN_fcn.py:173` | `forward(x, ht, ct)`；返回 `out,ht,ct` | 同样只输入窗口压力，每窗重置 `h,c`。 |
| `NNstep_window(...)` | `NN_fcn.py:326` | 历史状态/动作预热后，自回归预测未来状态 | 只用于理解原论文训练协议；当前窗末回归另写训练适配器。 |

两个类都是 `nn.GRU/nn.LSTM(batch_first=True) → out[:,-1,:] → nn.Linear(H,output_dim)`。原论文使用系统状态（角度，论文形式还包含角速度）与目标压力共同预测下一状态，并通过预热初始化隐藏状态；官方保存的 `model/config.txt` 则设置 `neglegt_qdyn=True`、单层 GRU、`hidden_dim=59`、`window=100`、`prediction=20`。必须区分论文一般表达、该已存配置和本项目适配。

建议主表 `H=64`、单层、dropout=0，验证集选择 `H∈{32,64,128}`；这是本项目初始配置，不是原文“最佳”。多层时才将 PyTorch 的层间 dropout 纳入调参。隐藏状态与输入同设备、同 dtype。该单文件顶层还导入 Ray、pandas、scipy、torchinfo 等；主线应保留 MIT 声明抽取模型类，避免仅为使用 GRU 继承整套 HPO/硬件环境。

**命名**：`Schafke2024-adapted GRU` / `Schafke2024-adapted LSTM`。将输入改为纯压力并删除原测量状态预热，是为了统一输入预算；成绩属于该门控架构在本项目任务上的重训练结果，不代表原论文观察器或 NMPC 的性能。每窗重置也不支持“已经比较任意长在线 RNN 记忆”的结论。

## 6. Yu 2026：分段二次 Bézier 读出与 NODE 的区别

**论文**：[Shape-Interpretable Visual Self-Modeling Enables Geometry-Aware Continuum Robot Control](https://arxiv.org/abs/2603.01751)，II-B、III-A 式 (4)–(11)、III-B 式 (12)–(17)。

原文使用 **M 段相连的二次 Bézier**，共有 `2M+1` 个控制点。不是单条任意阶 Bézier，也不是三次样条。第 `i` 段为

\[
B_i(s)=(1-s)^2P_{2i}+2(1-s)sP_{2i+1}+s^2P_{2i+2},\quad s\in[0,1].
\]

原文以三段为例，共 7 点；固定基点 `P0` 后，每个二维视角有 12 个变量；`K` 个视角的形状变量总维数为 `4MK`。形状通过骨架有序分段、端点确定及最小二乘求中间控制点获得。

**适合当前任务的读出对照**：保持窗口编码器 `E(U_t)` 不变，线性输出 `2M*d_s` 个可变控制点，再用固定基点和确定性 Bézier 采样得到 `N` 点骨架。`M` 按训练/验证数据选定，不强制本项目两段软臂也使用对方三段数目。分段连接、弧长重采样与节点顺序要固定；共享端点保证位置连续，不自动保证切向连续。固定基点只能来自部署几何约定，不能取当前测试标签。

若增加控制点辅助监督，只能从训练骨架拟合；主读出仍对原始骨架评分。同时报告“将 GT 投影到该 Bézier 族后再重建”的表示误差下界，帮助区分表示容量和压力映射误差。该投影只能用于离线诊断，不能成为测试时输入。

原文 NODE 是

\[
\dot x_s=f_\theta(x_s,u,t),\qquad
\hat x_s(t+\Delta t)=x_s(t)+\int_t^{t+\Delta t}f_\theta(x_s(\tau),u,\tau)d\tau,
\]

还单独建模末端位置动力学。它需要初始形状状态；在纯动作窗任务中用共享/动作生成初态、MLP 向量场与 RK4 来实现，会改变状态初始化和监督协议，隐藏层宽度也需自行选定。当前 **`Yu2026-inspired Bezier readout` 只比较形状表示，不应命名为“Yu NODE 复现”**。若以后实现动作窗 NODE，应单独列为 `Bezier-NODE architecture adaptation`，明确新的初态与时间定义。

## 7. Krauss：官方 VON 模型核与后续开环版本

开环论文官方链接确实指向所下载仓库。但此 commit 的 README 指定的是 **Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video**，arXiv [2511.18322](https://arxiv.org/abs/2511.18322)，2026 RA-L，DOI [10.1109/LRA.2026.3703241](https://doi.org/10.1109/LRA.2026.3703241)。本次也保存并核对了该原文。其机器人同样为分段气动连续体，视觉表示与潜在动力学对本项目有直接参考价值。

源码：[models.py](../../../workspace/third_party/modeling_baselines/visual_oscillators_for_SCR/models.py)。

| 类/接口 | 行号 | 所需输入与结构 |
|---|---|---|
| `VAE.encode/decode` | 507 / 549 / 565 | 图像 `[B,C,H,W]` 与潜变量；默认配置图像 32×32，编码卷积通道 32、64、128，核宽 4、步幅 2。 |
| `AttentionBroadcastDecoder` | 165 | `forward(z)` 返回图像；注意力结构和潜坐标一致性属于原表示方案。 |
| `KoopmanModel(config)` | 600 | `forward(z_t,z_dot_t,u_t,dt)`；`xi=[z,v]`，`xi_next=K*xi+B(u)`；`B` 为 `d_u→16→2*d_z`、LeakyReLU。代码的 `dt` 参数不参与离散 Koopman 更新。 |
| `FullyCoupledOscNet(config,device)` | 651 | 可学习质量逆、刚度、阻尼；力网络 `d_u→32→32→d_z`、LeakyReLU。 |
| `HarmonicOscillatorDynamics(config,device)` | 868 | `forward(z_t,z_dot_t,u_t,dt)` 返回下一 `z,v`；支持 `analytical` 或 `symplectic_euler`。 |
| `create_dynamics_model(config,device)` | 930 | 工厂接口；配置和底层实现允许的模式需一致。 |

源码振子的关系为

\[
M\ddot z+D\dot z+K(z-z_0)=B(u),\quad
v^+=v+\Delta tM^{-1}[B(u)-K(z-z_0)-Dv],\quad
z^+=z+\Delta t v^+.
\]

这对应 `step_symplectic_euler` 的**显式阻尼**。开环论文式 (8) 引入隐式阻尼，另有多步课程、修改的静止状态/KL 项及 MLP 动力学对照，不能因为仓库链接相同就声称这些更新已被完整实现。所检出的 `models.py` 没有独立的开环 MLP 动力学类；训练 notebook 与默认配置反映的是 VON 表示学习流程。默认 `delta_t=0.01667` 也不能移植成当前数据时间间隔。

默认配置还有 `num_actuation_delays=4`；训练 notebook 将动力学的有效动作维数扩展为基础通道数的四倍。因此直接使用模型类时，配置 `actuation_dim` 必须匹配实际传入的张量宽度。当前窗口适配可取消额外延迟拼接，或仅用窗口内的延迟样本；不能让 notebook 的全序列预处理为窗口起点补入更早历史。

**可行的动作预算适配**：抽象保留振子/Koopman 更新方程，每窗 `z=v=0`，依次消费窗口压力，最后用 `Linear(2*d_z,N*d_s)` 读出骨架。起始 `d_z=8`，也可验证 `4/8/16`；所有可学习参数计入预算。窗口内离散更新作为因果编码器使用，目标仍固定为 `S_t`；若强调其物理时间含义，则必须与主线压力记录的区间约定一致，不能无意多积分一个未来间隔。

该方案命名为 `Krauss-VON-inspired action-only oscillator` / `Krauss-inspired Koopman core`；移除了图像编码、图像导数、ABCD 和相应视觉损失，骨架头也是新增的。它比较的是结构化动态核，而不是视觉潜表示自建模的完整性能。原文图像速度使用中心差分；原始视觉训练协议中的这一做法更不能直接加入截至 `t` 的在线输入。许可证尚不明确，建议依据论文独立实现方程，源码作为接口参考。

## 8. SoftNeRF：条件形状表示的可抽取部分

**论文**：Jiwei Shan 等，IROS 2024，[SoftNeRF: A Self-Modeling Soft Robot Plugin for Various Tasks](https://doi.org/10.1109/IROS58592.2024.10801344)。本地出版 PDF 的摘要给出官方仓库，正文提出控制信号条件化的混合显式网格/隐式 SDF、可微渲染及误差引导采样。

源码文件：[fields.py](../../../workspace/third_party/modeling_baselines/softnerf/models/fields.py)、[neus.py](../../../workspace/third_party/modeling_baselines/softnerf/models/neus.py)、[dataset.py](../../../workspace/third_party/modeling_baselines/softnerf/models/dataset.py)。

- `kinematicNetwork`（`fields.py:25`）：`forward(inputs, training=True)`。仓库配置是 4 维动作→128→128→128→32；每层为 `sin(30*Linear(...))`，第四层也使用正弦。构造函数虽然接受 `n_layers`，实际层数固定为四个，不能只改配置就假定已改变网络深度。
- `SDFNetwork`（`:79`）：`forward(inputs, kinematic_feature, training=True)`，空间查询点编码后拼接运动特征，输出 SDF 与几何特征；`sdf(x,kinematic_feature)` 取首通道。输入层额外的运动特征维数硬编码为 32，需与压力编码匹配。
- `NeuSModel`（`neus.py:10`）/`forward_`（`:103`）：需要相机射线、near/far、驱动信号及配置。`NeuSRenderer.render`（`renderer.py:23`）包装这一入口。
- `Dataset` 需要 `cameras_sphere.npz`、图像/Mask 与 `action.txt`；还存在作者机器绝对路径、固定 1300×1300 图像大小和 CUDA 设备设置。依赖涉及 `tinycudann`、`nerfacc` 等，不是放入本项目 NPZ 即可训练的通用接口。根目录未发现训练 runner，README 只有标题说明，未核实完整误差引导采样训练流程的可运行性。

原表示可写作 `g=E(u), sdf(x,u)=G([HashGrid(x),g])`，再通过 SDF 体渲染监督 RGB/Mask。若仅抽象出 SIREN 压力编码并添加骨架线性头，它应叫 **`SoftNeRF-inspired SIREN encoder`**，只是编码器适配，不能代表 SoftNeRF 的混合表示、形状重建或采样方法。若用 `MLP([x,E(U)])` 直接预测二维 occupancy，也属于新的条件隐式表示基线；它没有原论文的三维网格/渲染训练。

因此当前主表不必为了增加近期方法数量而把上述轻量网络标成完整 SoftNeRF；可在有统一图像监督与相机标定预算时单独设置视觉表示表。原论文没有直接输出与本项目同语义的有序骨架节点，添加骨架提取/读出会引入新的误差来源。

### 可实施的二维条件 SDF 扩展

如果主线希望保留与 SoftNeRF 更接近的空间查询机制，建议实现 `SoftNeRF-inspired 2D conditional SDF`，而不是仅使用 SIREN 骨架头：

\[
g_t=E_\theta(U_t)\in\mathbb R^{32},\qquad
\hat d(x,U_t)=G_\psi([\gamma(x),g_t]),\quad x\in\mathbb R^2.
\]

`E` 可用展平压力窗的 SIREN（`L*d_u→128→128→128→32`）或现有因果 GRU；两者分别记录，窗口 SIREN 改变了原静态动作编码输入。`G` 可用 4 个 128 单元 Softplus/ReLU 隐藏层及标量线性输出，`gamma` 用固定频率坐标编码。这些宽度/激活/二维频率编码是本项目建议，未声称为原文 HashGrid 配置。空间查询点属于输出采样坐标，不是额外的机器人观测。

从训练集真实 Mask 计算负内正外的 `d_GT=EDT(1-M)-EDT(M)`，坐标和距离采用相同各向同性尺度；用统一截断半径对距离归一化。每帧采样边界附近、内部、外部三类点，比例固定或只由验证集选择；以 L1/Huber 距离损失训练，可选同监督条件下的 occupancy 损失。Mask 本身若由 GT 骨架膨胀生成，应称为管状几何代理，不能当独立视觉真值。

测试时使用固定栅格查询并以 `d_hat<0` 生成 Mask，再进行统一骨架化、基点定向与弧长重采样。空 Mask、断裂、多分支、基点/末端无法确定都须记录为提取失败；报告失败率、成功样本误差以及预先规定的全样本处理，避免只保留容易提取的形态。不可用测试 GT 选择路径、调阈值或定位末端。该输出链的误差包含场预测和提取器两部分，应同时报告 native-mask 与提取骨架精度。

该方案增加了 Mask/SDF 监督，宜作为单独的表示实验，或向其他方法提供相同训练监督。它是依据论文思想独立实现的二维条件场，不是官方三维 SDF/渲染训练的等价复现；许可证状态也不因替换输入输出而改变。

## 9. 给实施者的最小适配组合

| 优先级 | 建议实现名 | 原文明确保留的部分 | 本项目变更/新增 | 可作出的比较 |
|---|---|---|---|---|
| P0 | `Chen2025-inspired DirectionMLP` | 压力＋方向、4×128 隐藏层、关键点输出 | 动态数据、窗口内方向规则、激活与统一损失 | 末次加载方向是否足够。 |
| P0 | `Park2024-inspired causal TCN` | 两次因果膨胀卷积/块、k=3、dilation=2^b、末时刻特征 | 压力维数、骨架读出、通道宽度推断与调参 | 有限压力历史能否被轻量卷积有效利用。 |
| P0 | `Schafke2024-adapted GRU/LSTM` | 官方门控层＋末时刻线性读出 | 仅动作输入、每窗重置、当前骨架标签 | 同输入预算下门控记忆与 HOV 的差异。 |
| P1 | `Yu2026-inspired Bezier readout` | 分段二次 Bézier、固定基点、共享端点 | 动作编码器预测控制点、统一骨架监督 | 结构化读出的表示误差与泛化。 |
| P1 | `Krauss-VON-inspired action-only oscillator` | 振子方程与可学习强迫项 | 去除视觉初始化/监督，新增骨架头 | 另一种结构化动态核与 HOV 的差异。 |
| P2 | `SoftNeRF-inspired SIREN encoder` | 正弦动作编码结构 | 骨架头替代空间 SDF/渲染 | 编码器对照；无法评价完整 SoftNeRF。 |

至少 Chen＋Park 两个近期方法适配有直接原文结构依据；再加入官方 SPONGE GRU，可提供有源码来源的时序对照。MLP、线性、二次多项式作为基础方法独立标注；窗口 MLP 明确 flatten 顺序。简单地添加一个通用 GRU、一个普通 Bézier 层而不披露上述具体对应与变更，不足以声称复现了近期论文。

HOV 消融建议独立重训练无 play、无 Maxwell、两者都无、直接骨架读出等配置，固定数据划分、窗口、训练预算与验证选择准则。完整 HOV 的参数/状态规模与普通网络不同时，分别报告原文风格配置和合理容量对照；少参数并不自动证明更强的泛化。

## 10. 评价与显著性建议

本文件不实现指标或统计代码，仅给后续对比的解释约束：

1. **骨架与末端**：固定节点语义、坐标系和尺度，报告逐点欧氏距离均值、RMSE、末端欧氏误差及按序列分布；`sqrt(mean(||e||²))` 与逐坐标 RMSE 有不同尺度，名称需写清。不能对每个测试预测额外做刚体对齐或用标签弧长修正后再评分。
2. **Mask**：骨架模型使用共同、预先固定的管径/投影/光栅化器得到 Mask 时，IoU、Dice、边界距离评价的是“骨架预测＋共同几何读出”，不是模型独立学习了整体宽度。Mask GT 应来自实际图像标注/可靠分割；若由 GT 骨架膨胀得到，只能称几何代理。真正图像/SDF输出另列 native-mask 评价并记录额外训练监督。用验证集确定阈值与宽度，不在测试集选择。
3. **递进实验**：独立序列上的总体形态精度→相同当前压力及末次方向但不同窗口历史的配对→改变动作速率/持压时间与窗口长度→重训练 HOV 分支消融→Bézier/直接读出及数据量分析。纯建模主表可先完整观测评分，遮挡下状态补偿作为后续独立实验。
4. **统计单位**：建议至少 5 个训练种子作为起点，报告每种子的序列级指标；实际所需样本量取决于独立试验数量和效应大小。相邻帧、重叠窗口和种子×帧组合不是独立重复。先对相同测试序列和种子配对，再按独立采集/日期进行分组置换或分层 bootstrap；必要时先对种子平均后做序列配对检验，并单独报告种子变异。多个序列共享同次采集条件时按更高层级聚类。
5. **报告结论**：预先指定主指标（如独立序列平均骨架误差）、主对照与检验假设；报告差值、95% CI、效应量与经 Holm 校正的多重比较 p 值，避免只给“显著优于”。只有很少独立采集时，应报告描述性差异与不确定性，不能用大量相关帧代替独立证据。预期趋势保持为待检验假设。

## 11. 来源定位与复核范围

优先读取并核对了 [core_papers.md](../../papers/review_partial_observation_20260910/core_papers.md)、[source_manifest.json](../../papers/review_partial_observation_20260910/source_manifest.json)、[catalog.json](../../papers/review_partial_observation_20260910/catalog.json) 及它们所指本地原文。原始 HTML/PDF 路径与 SHA-256 继承后逐文件验证，新增网络证据与每个已检出文件的 SHA-256 另行保存。

`modeling_sources.json` 区分 `paper_verified`、`official_code_verified`、`downloaded`、`runtime_tested` 和 `adaptation_status`；其 `official_interfaces` 给出固定 commit 下的文件行号。下载日志记录失败后恢复的过程，而不是把第一次失败写成最终未获取。来源核查未执行第三方训练/安装/硬件脚本。后续已将MIT许可的SPONGE两个模型类提取到本项目并运行验证；未验证作者权重、原平台数据或原论文数值。
