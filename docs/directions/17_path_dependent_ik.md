# 方向 17:路径依赖逆运动学 — 软体臂 IK 是函数还是泛函

> 状态:**计划**(2026-07-16 立)。物理验证已用原始数据跑通(§3.1);建模对照实验待跑(§3.2 CLI 已给)。
> 关联:[12 科学问题 B(IK 可逆性)](12_scientific_problems_soft_robot_self_modeling.md)、[14 gt](14_gt_observed_transition.md)、[15 open_loop](15_open_loop_windowed_transition.md)、[16 约束导向控制](16_constraint_oriented_control.md)、[literature §4.1/§五](../background/literature.md)
> 一句话:**所有已发表软体自建模都在准静态下采数据、假设 action→shape 是无记忆函数;动态/循环加载下它物理上错了,我们量化这个错并修。**

---

## 〇、为什么是这个(不是缝补)

经 14-agent 对抗压测(文献缺口图 + 4 个创新假设 + 怀疑者批判),结论:

- **真正的空格**(无人占):{迟滞被建在可微前向场内} × {该场用于逆动作序列规划} × {单相机免标定 2D 像素 state} × {路径依赖 IK 失败演示}。每个邻居漏 ≥2 轴。
- **被压测毙掉、不要走的路**:
  - ❌ "精度≠可逆性 / 收缩稳定前向模型"(contraction/K_max):被一个**硬编码超参 confound**(gt `delta_scale_max=inf` vs open_loop `=1.0`),且仓库内两个 horizon run 自相矛盾(67392× vs 272×)。是"你设计 A 稳定、B 发散再证明 A 稳定"的循环论证。
  - ❌ "自认证可靠视野 K_self + 辨识定理":定理只有断言无证明(grep 零命中),z 懒惰(范数 0.0016,per-window 重置→无跨帧记忆)。
- **本方向(路径依赖 IK)**是 [literature §4.1](../background/literature.md) 最高优先级 + [12 问题 B](12_scientific_problems_soft_robot_self_modeling.md),且**刚画的实物 hysteresis loop 正是它缺的经验锚点**。

---

## 一、科学问题

**软体臂的逆运动学是"函数"还是"泛函"?**
- 函数思维(传统 IK):`target_shape → find action a*`。假设相同动作永远产生相同形状(马尔可夫)。
- 泛函思维(粘弹性 IK):`target_shape + 加载历史/当前状态 → find action 序列`。同一目标从不同历史出发需不同驱动轨迹;加载速率、路径曲率都影响最终形状。

**物理事实**(literature §1.3):硅胶是粘弹性体,`当前形状 = f(加载历史)`(非马尔可夫)。所有已发表工作(Tang/Yu/SoftNeRF/3DGS/Flow-Matching)都在**准静态下采数据(等稳定再拍)**,从而**把迟滞整个排除**——一旦动态/循环/变速加载,它们的 action→shape 映射**系统性失败**。

---

## 二、三问检验(literature §五的底线,三个都肯定=可发)

| # | 检验 | 怎么做 | 现状 |
|---|------|--------|------|
| ① | **迟滞真实且可观测?** | 实物 load/unload 不重合 + 速率依赖 | ✅ **已用原始数据验证**(§3.1) |
| ② | **不考虑迟滞的方法何时失败?** | 无记忆模型在"反方向"帧误差 = loop 宽度 | ⏳ 训 window=1 对照(§3.2) |
| ③ | **考虑迟滞能做什么新的?** | history-aware IK:为当前加载态规划正确动作序列 | ⏳ 接 inverse_plan(§3.3) |

---

## 三、实验设计

### 3.1 Exp1 — 物理验证(✅ 已完成,**不用训练**,raw NDI/pressure/cam0 即可)

数据:`real_capture/data/raw/seq_20260627_173114`(准静态,0.5s settle,11 周期)、`seq_20260627_172916`(动态,0.2s settle,25 周期)。两者都是 ch0 在 0↔150 kPa 的三角波加载。脚本 `scripts/experiments/exp5b_hysteresis_loop_real.py`。

**实测结果(2026-07-16,响应轴=NDI x)**:

| 信号 | 173114(准静态) | 172916(动态) | 解读 |
|---|---|---|---|
| **周期内路径依赖**(75kPa 处 load/unload 位移差) | **1.53 mm** | **2.06 mm** | ★ 强信号:同一气压、加载 vs 卸载→不同形状 |
| **速率依赖**(动态 loop / 准静态 loop) | — | 2.06 > 1.53 | ★ 强信号:同目标不同速率→不同形状 |
| **跨周期"第二次B≠第一次B"**(峰值漂移) | +0.44 mm(std 0.38) | +0.26 mm | ⚠️ 弱(~NDI 噪声底 0.74),**不靠这个** |
| loop 面积随周期 | -8% | +22% | 预条件化弱/不一致 |

**关键诚实结论**:
- **强证据 = 周期内 loop + 速率依赖**。这才是无记忆模型会失败的地方。
- **弱证据 = 跨周期 Mullins**(这块硅胶上 ~0.4mm,接近噪声)。论文**不要**把卖点压在"第二次 B≠第一次 B"上,压在"同气压 load/unload 不同形状"上。
- 输出图:`output/exp5b_hysteresis_loop/hysteresis_loop_real.png`(主图)+ `_172916.png`(动态对比)+ `time_series_real.png`。

### 3.2 Exp2 — 无记忆模型失败(⏳ 待跑,CLI 已给)

**干净 ablation**:同架构、同数据、同 `delta_scale_max`、同 z,只改 `--window_size`。审稿人挑不出 confound(吸取 contraction thesis 被 confound 的教训)。

```bash
# 无记忆 baseline(只看当前动作,看不出 load/unload 方向)
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode gt --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train \
    --window_size 1
# 对照:已有 exp_20260714_7(window_size=40,有记忆)
# 备选更彻底(纯前馈,无 s_{t-1}/无 z):
#   python scripts/training/train_spatial_sequence.py --data_dir <同上>
```

**评估**:在**方向反转帧**(loading↔unloading 切换处)比两模型误差。预期:
- window=1:误差 ≈ loop 宽度(1.5–4.25 mm),因为它只看当前气压→预测单一形状。
- window=40:误差显著降低(它能从历史知道方向)。

**评估集**:标准 val(方向反转帧多)或循环数据转 npz(`masks_to_transition_npz` 处理 172916/173114)。

### 3.3 Exp3 — history-aware IK(⏳ 接现有 inverse_plan)

给定 `目标形状 + 当前加载状态`,用 `scripts/control/inverse_plan.py` 反求正确的**动作序列**(不是单一动作)。这是 test ③——迟滞感知方法能做的新事。
- 关键:逆规划要 **condition 在当前状态**(s_init)上,因为路径依赖下逆映射依赖起点。
- 已有:variable-K + 避障。需补:**对比"有记忆前向模型"vs"无记忆前向模型"做逆规划的质量**——同样的目标,无记忆模型规划的序列在真机上达不到(因为它假设的形状是错的)。

### 3.4 Exp4(/stretch) — 记忆视界 h / 临界记忆长度 T*

[12 问题 A/B](12_scientific_problems_soft_robot_self_modeling.md):从形状解码历史的最大步数 h;消除 IK 歧义所需的最小历史 T*。理论预测 `h ≈ T* ≈ τ_max/Δt`。**风险**:当前 z 懒惰,可能要先修 z(§5)才能做。

---

## 四、关于"泛函逆运算"的方法论(回应疑虑)

当前 `inverse_plan.py` = **可微模型上的轨迹优化**(shooting + BPTT + Adam)。它**是可微的**(梯度穿过模型),但是**逐实例在线优化**,不是**学出来的逆泛函**。

| IK 方法 | 性质 | 路径依赖 IK 适用性 | 计划 |
|---|---|---|---|
| **轨迹优化**(现有) | 可微模型 BPTT 求 action 序列 | ✅ 天然处理状态依赖+约束 | 主方法 |
| **Jacobian 伪逆** | a←a+αJ†(s_tgt−s) | ✅ 我们**有可微 Jacobian**(autograd 免费) | **补一个对比**(显得非黑箱) |
| **学习型逆网络** | 监督训 shape+state→action | 需 state 条件化 | stretch |
| **Diffusion/Flow IK** | 建 action 分布(多模态逆) | ✅ 迟滞下 IK 非单射→多解 | stretch(对应问题 B) |
| **CMA-ES** | 无梯度进化 | 梯度失效兜底 | 备选 |

**定位**:卖点不是"学出逆泛函",而是**"路径依赖 IK 需要 state-conditioned 方法"**——轨迹优化为主 + Jacobian 为辅对比。逐实例优化对路径依赖甚至是**必要的**(逆映射依赖当前状态,不是固定函数)。

---

## 五、风险(go/no-go 前必须正面对付)

1. **lazy z 是真问题**:z 范数≈0、per-window 重置→z 是 cond 的函数,**不是真记忆**。
   - **不要**把"z 建模迟滞"当卖点(审稿人读 `model_state_transition.py` 会毙)。
   - 路径依赖的机制**挂在 action-history 窗口 + 分数阶核**(确实携带历史),不是 z。
   - Exp2(window=1 vs 40)若显示"有窗口就有用",说明窗口是有效机制,z 不必背锅。若连窗口都没用→方向重评。
2. **单构型**(1-DOF 两段硅胶)、单序列、10214 帧。**不声称泛化**(那是 Koopman/NJF 的)。
3. **未上真机**:所有控制结论是模型内验证。规划动作从未执行。Exp3 只到模型内,真机闭环是后续里程碑。
4. **contraction thesis 教训**:任何"模型 A 比 B 好"的对照,先确认没有隐藏超参差异(delta_scale_max、window、训练 regime 全一致)。

---

## 六、差异化(一句话 delta)

| 已发表 | 我们的 delta |
|---|---|
| Yu 2026(Bézier+NODE) | 他把形状当**控制目标**;我们当**加载历史载体**,证明 IK 路径依赖 |
| Tang 2026(在线优化) | 他适应**外部干扰**;我们理解**内在物理极限** |
| Hysteresis-Aware NN(2504.13582) | 他在末端**补偿**迟滞;我们**全身形状**量化+用于规划 |
| SoftNeRF/FBV-SM | 他们**准静态+标定**;我们动态+免标定 |
| NJF/Koopman(Nature) | 他们**RGB-D 多相机/泛化**;我们单相机+物理定律 |

**不声称**(已被占):骨架曲线参数化、coarse-fine、两阶段+在线 FT、33 构型泛化、NeRF→机器人范式、3DGS、纯可微 SDF-IK、分数阶核单独、免标定管线单独。

---

## 七、投稿目标与执行顺序

- **目标**:RA-L / ICRA(应用,4–6 页,聚焦"准静态方法动态失败 + 记忆模型修复 + history-aware IK");长版 T-RO(并入 [12 问题 A/B] 的信息论/可逆性理论)。
- **执行**:
  1. Exp2(window=1 vs 40)— **go/no-go**,数据/CLI 现成。
  2. Exp3(history-aware IK 对比无记忆 IK 的规划质量)。
  3. (stretch)Jacobian IK 对比;修 z 或明确机制挂在窗口上。
  4. 真机闭环(里程碑,需 PLC+RealSense+NDI)。

---

## 八、待核实

- web 搜索 2025-26 是否有未记录的"迟滞感知软体自建模+IK"同类工作(今天限流,明天 19:48 UTC 后查),确认空格未被占。
