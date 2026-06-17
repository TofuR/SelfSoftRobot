"""OpenLoopTransitionModel — 窗口开环状态转移框架（autoregressive within window）。

定位（与姊妹方向的区别，详见 docs/directions/15_open_loop_windowed_transition.md）:

  GTObservedTransitionModel（方向 14，当前主线）：每一步的前一状态 s_{t-1} 永远来自真实观测
    （teacher_forcing_ratio=1.0）。s 不累积漂移；仅 z 跨帧演化。部署语义"每步都观测"。

  StateTransitionSpatialModel（方向 13）：纯自回归 rollout——从 1 帧 GT 种子一路喂自身预测到
    序列末尾，误差无界累积（实测漂移比 1170×），适用于"无法每步观测、需多步前瞻"的未来场景。

  本模型 OpenLoopTransitionModel（方向 15）：**窗口开环**——每个窗口 K 步，仅以 1 帧真实观测
    （init_skeleton = positions[start_t-1]）作种子，窗口内剩余 K 步把模型自身预测喂回
    （teacher_forcing_ratio 退火到 0.0；s 与 z 在窗口内自演化）。窗口结束、下一窗口重新以 GT 种子
    锚定 → 把 rollout 漂移**约束在 K 步内**。部署语义："观测一次 → 开环预测 K 步"。
    迟滞由窗口内累积的潜轨迹 z 编码。

与"只有最近几十步影响当前状态"的关系（重要，避免过度声称）:
  本框架不是纯"action-history-only"——它用 1 帧 GT 种子锚定绝对位姿。正确表述是：
  "每 K 步一个绝对锚点约束位姿漂移；K 步潜轨迹 z 编码路径依赖（迟滞）"。
  纯冷启动（s_0=0）才会退化为方向 13 批判的稳态假设（迟滞下欠定），故保留 1 帧种子。

z 的范围（关键设计决策）:
  z 在**每个窗口**重新初始化（z_0 = z_init(cond)），是窗口内记忆，**不跨窗口携带**。
  若跨窗口携带 z，会退化成"有界累积的方向 13"，违背"只看一个 window"的定位。

实现:
  继承 StateTransitionSpatialModel，复用全部 forward / z_module（零参数增量，与
  GTObservedTransitionModel 同构——三者 state_dict key 完全相同）。仅固化 training_spec 为
  "窗口开环"身份 + open_loop_mode 标识 buffer（供 model_loader 从 config.json 区分）。

训练:
  - 推荐：从 GTObservedTransitionModel checkpoint 热启动（单步动力学已学好，56× 优于 copy），
    再按需退火 tf 1.0→0.0（staircase 优先，避免中段 0<tf<1 的速度输入混入）。
    先试 tf=0.0 直接跳变（成本最低；per-frame 误差 ~1e-8 下漂移缓慢，退火未必必要）。
  - dense supervision（窗口内每步 loss，给无 GT 的 z 每步直接梯度）。
  - z 无 GT，不加 loss；监测 ‖z‖（trainer 已输出 z_norm_monitor，漂移先于 skeleton loss 失稳）。
"""

import torch

from .model_state_transition import StateTransitionSpatialModel
from src.training.spec import TrainingSpec, PhaseSpec


class OpenLoopTransitionModel(StateTransitionSpatialModel):
    """窗口开环状态转移模型（每窗口 1 帧 GT 种子 + K 步自回归 rollout）。

    继承 StateTransitionSpatialModel，复用全部 forward / z_module（零参数增量）。
    仅固化训练 spec 为"窗口开环"身份。构造参数与父类一致（额外接受 episode_len）。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                name="open_loop_transition",
                dataset_type="state_transition",
                supervision_mode="spatial_sequence",
                # episode 路径只计算 skeleton + spatial_smooth（无 action_window_next，不算 smooth）
                active_losses=["skeleton", "spatial_smooth"],
                forward_attr="forward",
                # 窗口开环：z 在窗口内 K 步演化，s 从 1 帧 GT 种子开始自回归 rollout。
                # 默认 teacher_forcing_ratio=0.0 + tf_anneal_epochs=0 = 直接纯闭环
                # （成本最低的首试）。训练入口可开启退火（--tf_anneal_epochs --tf_start）。
                use_episode_mode=True,
                teacher_forcing_ratio=0.0,
                tf_anneal_epochs=0,
                tf_min=0.0,
                tf_schedule="staircase",
                episode_len=40,
                dense_step_weight="uniform",
            ),
        ],
    )

    def __init__(self, action_dim=2, n_nodes=31, hidden_dim=128, window_size=20,
                 n_orders=4, encoder_type="fractional", z_dim=16, episode_len=40):
        super().__init__(
            action_dim=action_dim, n_nodes=n_nodes, hidden_dim=hidden_dim,
            window_size=window_size, n_orders=n_orders, encoder_type=encoder_type,
            z_dim=z_dim,
        )
        # 窗口开环模式标识（非参数，仅用于 model_loader 从 config.json 区分本模型；
        # 三类模型继承同一基类 → state_dict key 完全相同，无法靠 key 区分）
        self.register_buffer('open_loop_mode', torch.tensor(True))
        self.episode_len = episode_len
        # 同步 spec 的 episode_len（K = 窗口开环步数）
        self.training_spec.phases[0].episode_len = episode_len
