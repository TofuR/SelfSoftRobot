# 04 · 实验计划(Exp A–G)

> **总览**:7 个实验,分三档。🟢 数据/代码现成 · 🟡 需少量新代码或已有实验待跑 · 🔴 需硬件。
> **公共资产**:训练数据 `data/real_seq/seq_20260627_163921_n15_sam2_clean/`(15 节点,SAM2 mask,10214 帧);GT checkpoint `train_log/gt_transition/exp_20260714_7`(best_loss 0.00077,NDI 末端 mean 0.77mm);open_loop checkpoint `train_log/open_loop_transition/exp_20260714_8`(best_loss 0.080,⚠️ 末期 NaN,多步 rollout mm 不可信);原始帧 `real_capture/data/raw/{seq_20260627_163921, seq_20260627_172916, seq_20260627_173114}`;当前配置 `real_validation/checkpoints/current/config.json`(OpenLoop, action_dim=1, window=40, fractional, n_nodes=15, z_dim=16)。

---

## Exp A — 时序编码器六路消融 🟢(最高优先)

**目标**:同一架构只换 `encoder_type` ∈ {ema, gamma, gru, transformer, tcn, fractional},在 open_loop 上训练,比全身骨架 px 误差、`drift_by_k`、末端误差。

```bash
for enc in ema gamma gru transformer tcn fractional; do
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode open_loop \
    --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train \
    --encoder_type $enc
done
```

**输出**:每编码器 `train_log/open_loop_transition/exp_*/eval_*`;汇总表:
- 验证集骨架 MSE(px²)与 tip MSE
- `drift_by_k`(k=10/20/40/80)
- 训练曲线收敛性(transformer/tcn 在小数据上是否欠拟合)

**判定**:GL 在**长视野 drift + 速率泛化**(Exp B)胜过通用序列模型 → P3 立;若全面不敌 → 重评 P3。

---

## Exp B — 速率泛化 🟡(物理接地的直接检验)

**目标**:训练速率 vs 测试速率错开,检验记忆核的跨速率外推。

**阶段 1(手头数据)**:两档已有 —— `seq_20260627_173114`(准静态 0.5s settle)、`seq_20260627_172916`(动态 0.2s settle)。先做"训准静态→测动态"与反向,只测 GL vs GRU(最省)。
**阶段 2(需硬件)**:P2 采集 3 档速率(见 C1),画"误差 vs 速率"曲线。

**实现要点**:
1. 两序列都是 ch0 0↔150 kPa 三角波,需先转 npz(`masks_to_transition_npz.py`),再 train/val 划分。
2. 注意:两序列的 npz 骨架源不同(准静态 vs 动态),先核 `pc_center/pc_scale` 一致或分别设。
3. 评估集:方向反转帧 + 变速段,分解误差。

**预期**:GL 幂律核在中速外推优于指数衰减 EMA;GRU 次之(泛化不如物理接地但强拟合)。若 GL 胜 → P3 有实证;若全都不胜 → 物理接地主张降级。

---

## Exp C — 学习 α vs 实测弛豫幂律指数 🟡(物理接地"钉子")

**目标**:证明模型学到的分数阶参数对应材料真实弛豫谱。

**做法**:
1. 从 NDI 阶跃弛豫(加载后保持段)拟合 `x(t) - x(∞) ~ t^(-α)` → 实测 α。
2. 读 GL 核 `model.temporal.alphas`(sigmoid 后)。
3. 对比 + 与 T* ≈ τ_max/Δt 预测互证(方向 12 §四)。

**数据**:现用 `seq_20260627_173114` 的保持段;P2 加专用阶跃序列(C2)更好。
**注意**:多阶次(n_orders=4)的 α 是谱(不是单值),对比时用"主导阶次 α 或加权平均"与实测比较,论文里明确是"谱匹配"而非单点吻合。

---

## Exp D — IK 歧义集量化(函数 vs 泛函形式化)🟡(最深科学空缺)

**目标**:定量回答"迟滞下 IK 是函数还是泛函"。

**做法**(基于 open_loop forward model + shooting):
1. 选目标形态 `S*`(取录制帧保证可达)。
2. 用 `openloop_planner`/`inverse_plan` 从**不同初始历史/状态**求解,得 M 组动作序列。
3. 测歧义集:序列间分散度(动作空间) + 对应 rollout 形态的差异。
4. 逐步增加"已知最近 k 步历史"(window 条件),测歧义何时消失 → 临界记忆长度 T*。
5. 对照 window=1 无记忆前向模型:是否"伪造"唯一解但实际不可达。

**产出**:"歧义集直径 vs 已知历史步数 k"曲线 + T* 估计。这是论文 P1 空缺里最有分量的图。

---

## Exp E — 历史感知 IK 规划质量(window=1 vs 40)🟢(方向 17 Exp2)

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt \
    --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train --window_size 1
```
**对照**:已训 `exp_20260714_7`(window=40)。
**评估**:方向反转帧误差;window=1 预期误差 ≈ 迟滞环宽度(1.5–4.25mm),window=40 显著更低。
**注意**:保持 delta_scale_max 等一致(contraction 教训)。

---

## Exp F — 可信视野认证形式化 🟢(K_max → "信任视野")

**已有**:`eval_horizon.py`(open_loop drift 1.7× @300步;K_max@10px≈124 步 ≈25s)、`real_validation/planning/auto_k.py`(step_budget 从学到的 delta_scale 现算)。
**做**:把 K_safe/auto_k 转成论文的"**开环信任视野**":给定容差 → 认证 K → 作为规划 horizon + 滚动重观测频率。产出:
- "信任视野 vs 容差"曲线(px 容差 2/5/10/20 → K_max)
- 记忆编码器对比下的信任视野(GL vs GRU vs EMA:谁在同样容差下支持更长开环?)
- deploy_manifest 的 `k_safe_table_px` 即此表的部署形态

> ⚠️ 不要复活方向 17 毙掉的"自认证可靠视野 K_self"(循环论证);信任视野是**从数据测出**的表,不是模型自证。

---

## Exp G — 免标定端到端实机演示(NDI mm)🔴(需硬件)

**目标**:把整条链路做成论文 Demo 图:相机 → 分割(SAM2/white_on_blue)→ 15 点骨架(免标定 px)→ 记忆自模型 → 开环规划 → 真机执行 → NDI 实测,报 **prediction-to-execution gap**。

**步骤**:见 `docs/real_data/deployment.md` §11(采集 → 前处理 → 训练 → 视野认证 → manifest → 工作台执行)。新采集必须:多速率(Exp B)+ 阶跃弛豫(Exp C)+ 固定相机拍无臂静态背景(配准参考,§8 #1)。

**指标**:
- 末端:N 次执行末态 vs NDI → gap mm(对比 GT-actions 上界 0.77mm 与 open_loop 期望)
- 全身:执行末帧骨架 vs 预测 → px(对比 `drift_by_k` 预测)
- 避障 demo(若有 3D/冗余):最小净距 vs 障碍

## Exp H — 多视角自标定验证(3D 升级,L2/L3)🔴(需硬件)

> 设计见 [`06_multiview_self_calibration.md`](06_multiview_self_calibration.md)。这里列评估。

**目标**:证明"身体自我标定 + 学习式免标定"达到可比传统标定的 3D mm 精度,且全程无标定板/流程。

**H1 自标定精度**:
- 用 L2 自标定外参三角化 → 3D 骨架;重投影误差(px) + 与 NDI 交叉的末端 mm 误差。
- 对照:传统标定(L1)三角化的 3D → 同一序列同指标 → 证明 L2 可比。

**H2 L2 vs L3 对照**:
- L2(身体/场景自标定)vs L3(DUSt3R/MUSt3R 类)同序列同指标 → 两独立通道差异。
- 结论决定部署用哪个 / 是否需要互证 fallback。

**H3 3D 全身避障 demo(可选,恢复 P5)**:
- 3D 冗余下"到达目标 + 全身避障"(零空间可用);预测最小净距 vs 执行后实测。

**产出**:"免标定 vs 传统标定 3D 精度"对照表 + L2/L3 一致性 → 论文差异化 #3 的 3D 版证据。

---

## 汇总时间线

| 阶段 | 实验 | 依赖 | 闸门 |
|---|---|---|---|
| 第 1 周 | A 六路消融、B 手头两档、E window 对比 | 无(离线) | A 中 GL 长视野/速率胜 → 继续;否则重评 |
| 第 2–3 周 | C 物理接地、D 歧义集、F 信任视野 | A/B 的 checkpoint | D 歧义集显著 → 泛函主张立 |
| 第 4 周+ | G 实机 + B/C 新数据 | 硬件 | G 的 gap 报诚实值,超容差报"不可达" |
| 3D 升级 | H 自标定验证(D1 脚本→D2 三角化→H1/H2)→ H3 3D 避障 demo | 多相机 + 自标定脚本 | H1 与 NDI/传统标定交叉达标 → 3D demo 才可信 |

> **诚实边界贯穿**:所有"规划"结论在真机执行前都是模型内验证;投稿前需真机闭环 + 近期文献再核(速率定量评估的"首次"声称)。
