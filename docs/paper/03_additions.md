# 03 · 要添加到项目的内容(逐项映射到代码)

> **原则**:能复用就复用,不新造轮子。所有"实验类"添加项都建立在现有代码 + 现有数据上,先把它们跑出来(go/no-go),再决定投入"模型修复/数据重采"的更深项。
> **工作量标注**:🟢 数据/代码现成,半天–1天 · 🟡 需少量新代码或已有实验待跑,1–3天 · 🔴 需新数据/大改,数天–周。

---

## A. 实验类(核心,大部分 🟢/🟡)

### A1. ★★★ 时序编码器六路系统性消融 —— 最快的 go/no-go

**为什么**:这是软体形态预测领域唯一没人做的系统对比(只有 GRU-vs-LSTM)。项目编码器套件天然齐备。若 GL 分数阶在此消融中胜出(尤其速率泛化),论文 P3 立住;若不胜,提前止损。

**怎么做**:同一架构 `StateTransitionSpatialModel`,只换 `encoder_type`:
```bash
for enc in ema gamma gru transformer tcn fractional; do
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode open_loop \
    --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train \
    --encoder_type $enc --tag encoder_ablation_$enc
done
```
**指标**:验证集全身骨架 px 误差、NDI 末端 mm(若含 NDI)、`drift_by_k` 曲线、**速率泛化**(见 A2)。
**对应代码**:`src/encoders/{multi_scale_ema,gamma_laguerre,temporal_gru,temporal_transformer,temporal_tcn,fractional_memory}.py`;`src/models/model_state_transition.py` `_ENCODERS` dict。

> 💡 **论文卖点**:这是"物理接地的记忆核 vs 通用序列模型"的正面证据,且无先例。即使 GL 只在中/长视野+速率泛化上胜出,也是可写的结论(物理接地在需要长记忆处胜出)。

### A2. ★★★ 速率泛化实验 —— 物理接地主张的证据

**为什么**:迟滞的速率依赖是 P1 的核心。训练一个速率、测试不同速率,能直接检验"GL 幂律核匹配物理弛豫谱 → 外推到未见速率"的假说(方向 12 理论预测)。

**怎么做**:
1. 已有数据:`seq_20260627_173114`(准静态,0.5s settle)+ `seq_20260627_172916`(动态,0.2s settle)。→ 先做"训准静态、测动态"与反向,看 GL vs GRU 的差距。
2. 需更多速率覆盖 → 新采集(见 C1),但**先用手头两档速率验证可行性**。
3. 评估在**方向反转帧 + 变速段**分解误差,画"误差 vs 速率"曲线。

**对应代码**:`train_transition.py`(数据目录换 seq 即可);误差分解可复用 `src/evaluation/transition_metrics.py` 的 rollout + `drift_by_k`。

### A3. ★★★ 学习 α vs 实测弛豫幂律指数 —— 物理接地的"钉子"

**为什么**:P3 的"物理接地"四个字需要一个可核验的锚:模型学到的分数阶 α 应该对应材料实测的弛豫幂律指数。若吻合,审稿人无法说这是 story-telling。

**怎么做**:
1. 从 NDI 阶跃响应(加载后保持)拟合 `x(t) ~ t^(-α)` 的实测 α。
2. 读 GL 核学到的 `model.temporal.alphas`(sigmoid 后 ∈ (0,1))。
3. 对比两者 + 与"窗口长度→记忆视界 h"的预测(方向 12 §四 T* ≈ τ_max/Δt)互证。
4. 若无现成 NDI 阶跃数据,用 `real_capture/data/raw/seq_20260627_163921` 的 pressure step 段(173114/172916 的三角波 + 保持段即可)。

**对应代码**:`src/encoders/fractional_memory.py` 的 `alphas` 属性;NDI 时序在 `real_capture/data/raw/<seq>/ndi.csv`。

### A4. ★★★ IK 歧义集量化(函数 vs 泛函的形式化)—— 最深的科学空缺

**为什么**:P1 空缺残余里最强的一环。前向映射路径依赖已有人量化(Cho 9±6.5%、Chen 3.4%),但**逆映射的歧义集(前像集直径)无人量化**。这是"IK 是函数还是泛函"的定量回答。

**怎么做**(基于现有 forward model + planner):
1. 用已训练的 open_loop 前向模型做可微 rollout。
2. 对同一目标形态 `S*`(取录制帧,保证可达),用 `openloop_planner` 的 shooting 从**不同初始历史/状态**求解 → 得到多组动作序列。
3. 统计这些序列的分散度 = 歧义集直径;逐步增加"已知最近 k 步历史"看歧义何时消失 → 测临界记忆长度 T*。
4. 对比不同 `window_size`(如 1 vs 40)前向模型 → 无记忆模型是否"伪造"出唯一但错误的解。

**对应代码**:`real_validation/openloop_planner.py`(rollout + shooting);`scripts/control/inverse_plan.py`;`real_validation/planning/auto_k.py`。

> 💡 **论文贡献句**:这是把"迟滞下 IK 不适定"从直觉变成可测量量纲(歧义集直径 / T*)的首次尝试。

### A5. 🟡 历史感知 IK 规划质量对比(window=1 vs 40)

方向 17 Exp2/Exp3 已规划,数据/CLI 现成:
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py --mode gt \
    --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/train --window_size 1
```
评估:同架构、同数据、同 delta_scale_max、同 z,只改 window_size(干净消融,吸取 contraction confound 教训)。预期:window=1 在方向反转帧误差 ≈ 迟滞环宽度;window=40 显著降低。→ **P2 的证据,不作头条**。

### A6. 🟢 可信视野认证形式化(K_max / auto_k → "信任视野")

已有:`eval_horizon.py`(K_max@px)、`real_validation/planning/auto_k.py`(step_budget 从学到的 delta_scale 现算)。论文把"观测一次、预测 K 步"表述为**开环信任视野**,并用 K_safe/deploy_manifest 落地为可认证量。此项主要是**论文化转写 + 图表**。

---

## B. 模型修复类(🟡/🔴)

### B1. 修 z 懒惰 或 弱化 z 声称

- **方案 a(推荐,成本低)**:论文把记忆机制明确挂在 **动作历史窗口 + 分数阶 GL 核** 上,不再声称"z 建模迟滞"。改动 = 删/改模型 docstring 与论文叙述,**代码可不改**(z 仍是 latent,只是不作为卖点)。
- **方案 b(强,成本高)**:序列级训练(episode 内 rollout,z 跨帧演化),让 z 成为真记忆。需要 Stage 1 训练循环(当前 UnifiedTrainer 是 per-frame shuffle,Stage 0 限制下 z 退化为 cond 的函数)。**若 A1/A2 显示窗口+GL 已够,可不做 b。**

---

## C. 数据类(🔴,需硬件)

### C1. 多速率 + 循环 + 方向反转数据集(P2 采集协议已写)

- 现有协议:`docs/real_data/deployment.md` §11.2 + §7(覆盖 loading/unloading/hold/反转/**变速**)。
- 新增要点:**每个速率的 settle/ramp 档位**,至少 2–3 档,支撑 A2 速率泛化曲线。
- 3 腔道 6 通道升级为"冗余全身避障 demo"(P5)所需,但 **P5 已降级**,可延后。

### C2. NDI 阶跃弛豫采集(支撑 A3)

在 P2 采集中加入"阶跃 + 保持"序列(给不同气压阶跃,NDI 记录弛豫),直接出 A3 的实测 α。

---

## 四、优先级与执行顺序(go/no-go 逻辑)

```
第 1 周(go/no-go,纯离线)
  A1 六路消融(最优先)  → 若 GL 在长视野/速率泛化胜出 → 论文 P3 立
  A2 速率泛化(用手头两档) → 确认"物理接地"是否有实证
  A5 window=1 vs 40     → 确认 P2 证据强度

第 2–3 周(方法加固)
  A3 学习 α vs 实测幂律(物理接地钉子)
  A4 IK 歧义集量化(最深空缺)
  A6 信任视野形式化

第 4 周+(需要硬件)
  C1/C2 多速率+阶跃采集 → 重训 gt/open_loop → 真机执行 → prediction-to-execution gap
  (可选)B1-b 序列级训练让 z 成真记忆;P5 demo 延后
```

> **风险闸门**:A1 若 GL 全面不敌 GRU → 重评 P3(可能把论文改为"系统消融 + 物理接地记忆在特定 regime 的价值",仍可写,但降一档);A4 若歧义集很小 → "泛函"主张弱化,回落到"记忆提升开环规划精度"的工程叙事。
