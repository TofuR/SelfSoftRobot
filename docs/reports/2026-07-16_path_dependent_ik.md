# 报告:路径依赖逆运动学 —— 下一篇论文的方向

> 日期:2026-07-16 · 分支 `feat/real-data-transition`
> 性质:**前瞻/方向报告**(下一次汇报的核心内容)。把 07-15 的"控制形状"工程进展,升级为**论文级的科学问题**。
> 关联:[方向 17](../directions/17_path_dependent_ik.md)(完整计划)·[汇报 §11](../presentations/2026-07-15_shape_control_planning.md)·[literature §4.1/§五](../background/literature.md)·[科学问题 12 B](../directions/12_scientific_problems_soft_robot_self_modeling.md)
> 实物迟滞图:`output/exp5b_hysteresis_loop/hysteresis_loop_real.png`

---

## 摘要(一句话)

**所有已发表软体机器人自建模都在准静态下采数据、假设 action→shape 是无记忆函数;但在动态/循环加载下,这个假设物理上错了——软体臂的逆运动学是路径依赖的(泛函,非函数)。** 我们已经用原始数据验证了迟滞真实存在(实物 hysteresis loop,同气压 load/unload 形状差 1.5–2mm,且速率依赖);下一步是用"无记忆对照模型"证明现有方法会失败、用"带记忆模型"修好,并接到 history-aware 逆规划。

---

## 一、为什么"控制更准"不够 —— 论文创新在哪

07-15 的"控制形状"(变长 K + 避障 + 观测解耦)是扎实的工程。但若只讲"控制更准/到达 4.3px",**会被当成增量**——因为骨架参数化、coarse-fine、两阶段+在线 FT、33 构型泛化等**已被已发表工作占据**(见 §五"不声称"清单)。

经对 2025-26 全部已发表工作的缺口分析,**真正的空格**:

> **{迟滞被建在可微前向场内} × {该场用于逆动作序列规划} × {单相机免标定 2D 像素 state} × {路径依赖 IK 失败演示}**

每个最近的邻居都**漏 ≥2 轴**:

| 已发表 | 有 | 漏 |
|---|---|---|
| Hysteresis-Aware NN (arXiv 2504.13582) | 迟滞建模 | 不规划 / 不免标定 / 只到末端 |
| Yu 2026 (Bézier+NODE) | 可微逆规划 | 假设准静态 / 双相机标定 |
| Tang 2026 (在线优化) | 全身形状控制 | 无时序建模 / 不做视野认证 |
| SoftNeRF / FBV-SM | 可微场+规划 | 无迟滞 / 需标定多视角 |
| NJF (Nature 2025) / Koopman | 泛化+控制 | 需 RGB-D 多相机 / 无迟滞 |

**物理根基**:硅胶是粘弹性体,`当前形状 = f(加载历史)`(非马尔可夫)。所有已发表工作**等稳定再拍照**→ 把迟滞整个排除。一旦动态/循环/变速加载,它们的映射**系统性失败**。

---

## 二、科学问题

**软体臂的 IK 是"函数"还是"泛函"?**
- 函数(传统 IK):`target → action a*`,假设相同动作永远产生相同形状。
- 泛函(粘弹性 IK):`target + 加载历史/当前状态 → action 序列`;同目标从不同历史出发需不同驱动轨迹;**速率、路径曲率都影响最终形状**。

---

## 三、三问检验 + 当前进展(literature §五底线:三个肯定=可发)

### ① 迟滞真实且可观测?—— ✅ 已用原始数据验证(不用训练)

数据:`real_capture/data/raw/seq_20260627_173114`(准静态 0.5s settle)、`seq_20260627_172916`(动态 0.2s settle),ch0 在 0↔150 kPa 三角波循环加载。脚本 `scripts/experiments/exp5b_hysteresis_loop_real.py`。

| 信号 | 173114 准静态 | 172916 动态 | 结论 |
|---|---|---|---|
| **周期内路径依赖**(75kPa 处 load/unload 位移差) | **1.53 mm** | **2.06 mm** | ★ 强:同气压、加载 vs 卸载→不同形状 |
| **速率依赖**(动态/准静态) | — | 2.06 > 1.53 | ★ 强:同目标不同速率→不同形状 |
| 跨周期"第二次B≠第一次B"(峰值漂移) | +0.44 mm | +0.26 mm | ⚠️ 弱(≈NDI 噪声底 0.74),**不靠** |

图:`output/exp5b_hysteresis_loop/{hysteresis_loop_real.png, hysteresis_loop_real_172916.png, time_series_real.png}`。

> **诚实**:论文卖点压在"同气压 load/unload 不同形状"+"速率依赖"上,**不**压在跨周期 Mullins(这块硅胶上 ~0.4mm 太弱)。

### ② 不考虑迟滞的方法何时失败?—— ⏳ 待训无记忆对照

**干净 ablation**(同架构、同数据、同 `delta_scale_max`、同 z,只改 `--window_size`,避开 confound):

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \
    --mode gt --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train \
    --window_size 1
# 对照: train_log/gt_transition/exp_20260714_7 (window_size=40)
```

**预期**:window=1(只看当前动作→预测单一形状)在**方向反转帧**误差 ≈ loop 宽度(1.5–4.25 mm);window=40(看历史→知 load/unload 方向)显著降低。评估集:标准 val 方向反转帧,或循环数据转 npz(`masks_to_transition_npz` 处理 172916/173114)。

### ③ 考虑迟滞能做什么新的?—— ⏳ history-aware IK

给定 `目标 + 当前加载状态`,用 `scripts/control/inverse_plan.py` 反求**动作序列**(condition 在 s_init)。对比"有记忆前向模型"vs"无记忆前向模型"做逆规划的质量——无记忆模型规划的序列在真机达不到(它假设的形状是错的)。

---

## 四、IK 方法的诚实定位(回应"优化 vs 可微泛函"疑虑)

当前 `inverse_plan.py` = **可微模型上的轨迹优化**(shooting + BPTT + Adam)。它**是可微的**(梯度穿过模型),但是**逐实例在线优化**,不是**学出的逆泛函**。

| IK 方法 | 性质 | 路径依赖 IK 适用 | 计划 |
|---|---|---|---|
| 轨迹优化(现有) | 可微模型 BPTT 求 action 序列 | ✅ 天然处理状态依赖+约束 | 主方法 |
| Jacobian 伪逆 | a←a+αJ†(s_tgt−s) | ✅ autograd 免费给 Jacobian | **补对比**(非黑箱) |
| 学习型逆网络 | 监督训 shape+state→action | 需 state 条件化 | stretch |
| Diffusion/Flow IK | 建 action 分布(多模态) | ✅ 迟滞下 IK 非单射→多解 | stretch |
| CMA-ES | 无梯度进化 | 兜底 | 备选 |

**定位**:卖点不是"学出逆泛函",而是**"路径依赖 IK 需要 state-conditioned 方法"**——轨迹优化为主 + Jacobian 为辅。逐实例优化对路径依赖甚至是**必要的**(逆映射依赖当前状态)。

---

## 五、诚实的风险与边界(go/no-go 前必须正视)

1. **lazy z 是真问题**:z 范数≈0、per-window 重置→是 cond 的函数,**不是真记忆**。**不要**把"z 建模迟滞"当卖点;机制挂 action-history 窗口 + 分数阶核。Exp2(window=1 vs 40)是 go/no-go:有窗口就有用→窗口是有效机制;没用→方向重评。
2. **"gt 爆炸 67392× vs open_loop 1.7×"不是科学发现**:一部分来自 open_loop 的 `delta_scale_max=1.0` 钳位(工程手段),且仓库另一个 run 是 272×。07-15 汇报里它是"训练-推理鸿沟"的**工程观察**(有用),但论文**不**把它当"精度≠可逆性"的科学主张。
3. **单构型**(1-DOF 两段硅胶)、单序列。**不声称泛化**。
4. **未上真机**:控制结论都是模型内验证,规划动作从未执行。真机闭环是后续里程碑。

---

## 六、差异化(一句话)

- vs **Yu 2026**:他把形状当**控制目标**;我们当**加载历史载体**,证明 IK 路径依赖。
- vs **Tang 2026**:他在线适应**外部干扰**;我们理解**内在物理极限**。
- vs **Hysteresis-Aware NN**:他在末端**补偿**迟滞;我们**全身形状**量化+用于规划。
- vs **SoftNeRF/FBV-SM**:他们**准静态+标定**;我们动态+免标定。
- vs **NJF/Koopman**:他们**RGB-D 多相机/泛化**;我们单相机+物理定律。

**不声称**(已被占):骨架曲线参数化、coarse-fine、两阶段+在线 FT、33 构型泛化、NeRF→机器人范式、3DGS、纯可微 SDF-IK、分数阶核单独、免标定管线单独。

---

## 七、下一步执行顺序

1. **Exp2(window=1 vs 40)** —— go/no-go,数据/CLI 现成。
2. **Exp3**(history-aware IK vs 无记忆 IK 规划质量)。
3. (stretch)Jacobian IK 对比;修 z 或明确机制挂窗口。
4. 真机闭环(里程碑)。

**投稿目标**:RA-L / ICRA(应用 4–6 页:准静态方法动态失败 + 记忆模型修复 + history-aware IK);长版 T-RO(并入信息论/可逆性理论)。

---

## 八、待核实

web 搜索 2025-26 是否有未记录的"迟滞感知软体自建模+IK"同类工作(今天限流,明天 19:48 UTC 后查),确认空格未被占。

---

*详细计划:`docs/directions/17_path_dependent_ik.md`。迟滞图:`output/exp5b_hysteresis_loop/`。汇报讲稿 §11:`docs/presentations/2026-07-15_shape_control_planning.md`。*
