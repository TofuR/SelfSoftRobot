# 方向：全 GT 驱动单步状态转移框架（独立于自回归 rollout）

> 状态：已实现（Stage 0），冒烟测试通过
> 模型：`src/models/model_gt_transition.py` 的 `GTObservedTransitionModel`
> 姊妹方向：[13_closed_loop_state_transition.md](13_closed_loop_state_transition.md)（自回归闭环，未来扩展）
> 核心思想：**前一状态 s_{t-1} 永远来自真实观测**（仿真 GT / 实物图像骨架化），模型做单步转移，z 跨帧演化
> 创建：2026-06-15

---

## 〇、为什么单独立一个框架

[方向 13](13_closed_loop_state_transition.md) 的 `StateTransitionSpatialModel` 是"自回归闭环"框架——推理时把模型自己的预测喂回（rollout），适用于"无法每步观测真实状态、需一路推下去"的场景。

但**实际部署并非如此**。软体机器人的真实部署是：

```
每一步：
  采集图像 → 骨架化 → 得到真实 s_{t-1}
  输入 (真实 s_{t-1}, action_t) → 模型 → 预测 s_t
```

**前一状态永远是真实观测**（仿真里是 GT `positions[t-1]`，实物里是图像骨架化结果），不是模型自己的预测。模型本质是**单步状态转移**——给定真实的前一状态和当前动作，预测这一步的状态。这像一个**观测驱动的滤波器/预测器**，而非一路推到底的世界模型。

因此纯自回归 rollout（方向 13 的核心验证方式）**不是当前场景**，而是"跑通当前管线后"的未来扩展方向。本框架（14）专注当前真实场景。

### 与方向 13 的对比

| 维度 | 方向 13（自回归闭环） | **方向 14（全 GT 驱动）** |
|------|---------------------|--------------------------|
| 前一状态 s_{t-1} | 模型自己的预测（rollout 喂回） | **真实观测**（GT / 图像骨架化） |
| 推理方式 | 自回归一路推 T 步 | **单步转移，每步喂真实前一状态** |
| train/inference | 不一致（train 喂 GT，inference 喂预测）→ 需 scheduled sampling | **完全一致**（都喂真实 s_{t-1}） |
| s 误差累积 | 有（rollout 漂移，实测 1170×） | **无**（每步重置为真实观测） |
| 误差风险源 | s 和 z 都可能漂移 | **仅 z**（z 无 GT，跨帧演化） |
| 适用场景 | 无法每步观测（如预测未来 N 步） | **常规部署**（每步都能采集图像） |
| 定位 | 未来扩展 | **当前主线** |

---

## 一、设计要点

### 1. s_{t-1} 永远真实

```
ŝ_t = F(真实 s_{t-1}, z_{t-1}, action_t)
```

- 仿真：`prev_skeleton = positions[t-1]`（GT，已在 `.npz`，无需重采）
- 实物：`prev_skeleton = 图像骨架化(s_{t-1})`（Stage 2 接入感知前端）

**性质**：s 每步重置为真实观测 → s 不累积漂移。train 与 inference 完全一致（都喂真实前一状态），**无需 scheduled sampling**（`teacher_forcing_ratio = 1.0`）。

### 2. z 跨帧演化（保留，用户决策）

z 是可学习迟滞潜变量，跨帧演化 `z_t = Φ_z(z_{t-1}, a_t, s_{t-1})`，无 GT，端到端从 skeleton loss 学。在"每步真实 s"下，z 是**唯一跨帧、唯一无 GT 的状态**。

**为什么保留 z**：s_{t-1}（位置）和 action 虽是真实输入，但 z 可编码**位置+动作之外的深度历史**——例如内部应力方向（充气中 vs 放气中），这些信息位置本身不直接体现。z 是这层历史的低维潜表示。

**风险**：z 无 GT、跨帧演化，可能漂移。但在全 GT 驱动下，z 漂移的影响是**逐步纠正的**（每步用真实 s 重置位置基准），不致失控。冒烟测试证实 z 收敛有界（见 §三）。

### 3. 数据集：单帧 vs episode

- **z 训练需要序列**：z 跨帧演化必须用 episode 模式（z 在序列内 BPTT 学习）。
- **s 学习不需要顺序**：单帧模式每样本独立即可（prev 来自 GT，样本间可 shuffle）。
- 本框架用 **episode 模式**（为 z 训练），但 s 仍每步取 GT（TF=1.0）。

---

## 二、实现（全部复用方向 13 基础设施）

`GTObservedTransitionModel` 继承 `StateTransitionSpatialModel`，**复用全部 forward / forward_sequence / z_module**，仅固化训练 spec 为"全 GT 驱动"身份：

| 组件 | 来源 | 说明 |
|------|------|------|
| forward / forward_sequence / z_module | 继承父类 | 零改动复用 |
| training_spec | **固化** | episode 模式 + `teacher_forcing_ratio=1.0` + episode_len |
| gt_observed_mode buffer | **新增** | 标识本模型，供 model_loader 从 config.json 区分 |

### 文件清单

| 文件 | 类型 | 内容 |
|------|------|------|
| `src/models/model_gt_transition.py` | 新建 | `GTObservedTransitionModel`（继承，固化 spec） |
| `src/models/__init__.py` | 编辑 | lazy export |
| `src/utils/model_loader.py` | 编辑 | state_transition 分支按 config.json `model` 字段区分子类 |
| `scripts/training/train_gt_transition.py` | 新建 | 训练入口（episode, cuda1, 短 epoch） |
| `scripts/evaluation/eval_gt_transition.py` | 新建 | **观测驱动评估**（s 每步真实 + z 演化，监测 z 漂移） |

> model_loader 用 config.json 的 `model` 字段（而非 state_dict key）区分本模型与方向 13 模型——二者继承关系导致 state_dict key 完全相同，无法靠 key 区分。

---

## 三、冒烟测试结果（cuda1, 5 mini-batch + 15 步评估）

训练 loss 持续下降（4.3e-5 → 7.8e-6），观测驱动评估：

| 指标 | 值 | 含义 |
|------|-----|------|
| per-step MSE | ~6e-6，15 步**几乎不增长** | s 每步真实 → **部署精度稳定，无 s 累积漂移** |
| z norm | 0.64 → 1.33，**收敛**（13 步后稳定） | z 跨帧演化但**有界收敛** |
| z drift ratio | 2.08× | z 启动后稳定，非单调发散 |

**对比方向 13 自回归 rollout**（漂移比 1170×）：全 GT 驱动下 s 漂移消失，仅 z 有微小收敛性演化——**这正是当前部署场景的预期**。

---

## 四、与现有方向体系的整合

- 本框架是 [13_closed_loop_state_transition.md](13_closed_loop_state_transition.md) 的**当前主线**，13 退为未来扩展（无法每步观测时的自回归预测）。
- z 复用方向 13 的可学习潜变量设计（方案 A），无 GT，端到端学。
- 部署层（[10_vision_corrected_deployment](10_vision_corrected_deployment.md)）天然契合：图像骨架化提供真实 s_{t-1}，闭环模型做单步转移。

---

## 五、未来扩展（记录，非当前）

- **2D→3D 状态获取**（Stage 2）：实物上从图像骨架化得到真实 s_{t-1}（替代仿真 GT）。需感知前端。
- **纯自回归 rollout**（方向 13）：当无法每步观测时（预测未来、快速控制跳过观测），把模型预测喂回。这是已实现但当前不主用的扩展。
- **z 收缩正则**：若长序列上 z 漂移失控，对 Φ_z 加谱约束（与方向 13 Stage 1 共享）。
