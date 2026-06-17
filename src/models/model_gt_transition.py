"""GTObservedTransitionModel — 全 GT 驱动的单步状态转移框架。

定位（与 model_state_transition.py 的区别）:
  StateTransitionSpatialModel 是"自回归闭环"框架——推理时把模型自己的预测喂回，
  适用于"无法每步观测真实状态、需一路推下去"的未来场景（rollout）。

  本模型（GTObservedTransitionModel）是"全 GT 驱动"框架——**前一状态 s_{t-1} 永远
  来自真实观测**：
    - 仿真：positions[t-1]（GT）
    - 实物：采集图像 → 骨架化得到真实 s_{t-1}
  模型做单步转移 ŝ_t = F(真实 s_{t-1}, z_{t-1}, a_t)。train 与 inference 完全一致
  （都喂真实前一状态），不涉及纯自回归 rollout。

  因此：
    - s 不累积漂移（每步重置为真实观测）
    - 误差累积风险转移到 z 上（z 无 GT，跨帧演化）→ 仅需监测 ‖z‖，不漂移即可
    - teacher_forcing_ratio 恒为 1.0（s 总是真实，无需 scheduled sampling）

z 的处理（继承自 StateTransitionSpatialModel）:
  z 是可学习迟滞潜变量，跨帧演化（编码位置+动作之外的深度历史，如内部应力方向），
  无 GT，端到端从 skeleton loss 学。在"每步真实 s"下，z 是唯一跨帧、唯一无 GT 的状态。

实现:
  继承 StateTransitionSpatialModel，复用全部 forward / z_module。
  仅固化 training_spec（episode 模式 + TF=1.0）+ 加 gt_observed_mode 标识 buffer（供
  model_loader 从 config.json 的 model 字段区分，不依赖 state_dict key）。

训练:
  - episode 模式：z 在序列内逐步演化（BPTT），s 每步取 GT
  - L_skeleton / L_spatial_smooth 同 StateTransitionSpatialModel
  - z 无 GT，不加 loss
"""

import torch

from .model_state_transition import StateTransitionSpatialModel
from src.training.spec import TrainingSpec, PhaseSpec


class GTObservedTransitionModel(StateTransitionSpatialModel):
    """全 GT 驱动的单步状态转移模型（s_{t-1} 永远真实，z 跨帧演化）。

    继承 StateTransitionSpatialModel，仅固化训练 spec 为"全 GT 驱动"身份。
    构造参数与父类完全一致（额外接受 episode_len 用于固化 spec）。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                name="gt_transition",
                dataset_type="state_transition",
                supervision_mode="spatial_sequence",
                # episode 路径只计算 skeleton + spatial_smooth
                # （无 action_window_next，时序 smooth 无法沿用逐帧实现，故不声明）
                active_losses=["skeleton", "spatial_smooth"],
                forward_attr="forward",
                # 全 GT 驱动窗口模式：z 在窗口内 K 步演化（每步喂真实 s），
                # dense supervision（每步预测都算 loss，给无 GT 的 z 直接梯度），
                # s 恒真实（TF=1.0），样本自包含可打乱。
                # episode_len 默认对齐 action_window（=window_size），K 可调。
                use_episode_mode=True,
                teacher_forcing_ratio=1.0,
                episode_len=40,
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
        # 全 GT 驱动模式标识（非参数，仅用于 model_loader 从 config.json 区分本模型与
        # StateTransitionSpatialModel；二者 state_dict key 完全相同，无法靠 key 区分）
        self.register_buffer('gt_observed_mode', torch.tensor(True))
        self.episode_len = episode_len
        # 同步 spec 的 episode_len（允许构造时覆盖默认 20）
        self.training_spec.phases[0].episode_len = episode_len
