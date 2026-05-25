"""Multi-Scale Temporal Neural Field (MSTNF).

时序编码: 动作窗口 → MultiScaleEMA → physics_state
空间查询: (x,y,z) + physics_state → [visibility, density]
渲染: 体渲染 → 2D 图像
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder
from .mixins import TemporalMixin
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.training.spec import PhaseSpec, TrainingSpec


class MSTNFModel(nn.Module, TemporalMixin):
    """Multi-Scale Temporal Neural Field 完整模型。

    时序编码: 动作窗口 → MultiScaleEMA → physics_state
    空间查询: (x,y,z) + physics_state → [visibility, density]
    渲染: 体渲染 → 2D 图像

    Args:
        action_dim: 动作维度。
        window_size: 时序窗口长度。
        n_scales: EMA 尺度数。
        hidden_dim: 物理状态/隐层维度。
        d_filter: 空间 MLP 隐层维度。
        n_freqs: 位置编码频率数。
    """

    training_spec = TrainingSpec(
        phases=[PhaseSpec("full", forward_attr="forward", data_mode="sequence",
                          active_losses=["recon", "depth", "smooth"])],
    )

    def __init__(
        self,
        action_dim,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        d_filter=128,
        n_freqs=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 时序编码器
        self.temporal = MultiScaleEMA(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # 空间编码
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=n_freqs, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * n_freqs)

        # 空间解码器: pos_enc + physics_state + current_action → [visibility, density]
        # action skip connection: 即使 EMA 编码尚未成熟，解码器也能直接感知当前动作
        self.decoder = MLPDecoder(
            input_dim=pos_enc_dim + hidden_dim + action_dim,
            d_filter=d_filter,
            output_size=2,
        )

        # density bias: 0.0 中性初始化
        # softplus 在负值区也有非零梯度，不需要负偏置来控制稀疏性
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

    def encode_temporal(self, action_window):
        """编码动作窗口为物理状态（向后兼容别名，推荐使用 encode()）。"""
        return self.encode(action_window)

    def decode_spatial(self, points, physics_state, current_action=None):
        """查询空间场。

        Args:
            points: (N, N_samples, 3) 3D 查询点。
            physics_state: (N, Hidden_Dim) 物理状态。
            current_action: (N, Action_Dim) 当前动作 (skip connection)。

        Returns:
            output: (N, N_samples, 2) [visibility, density]
        """
        n_samples = points.shape[1]

        x_pos = self.pos_encoder(points)
        state_expanded = physics_state.unsqueeze(1).expand(-1, n_samples, -1)

        parts = [x_pos, state_expanded]
        if current_action is not None:
            action_expanded = current_action.unsqueeze(1).expand(-1, n_samples, -1)
        else:
            action_expanded = torch.zeros(
                physics_state.shape[0], n_samples, self.action_dim,
                device=physics_state.device, dtype=physics_state.dtype,
            )
        parts.append(action_expanded)
        latent = torch.cat(parts, dim=-1)
        return self.decoder(latent)

    def forward(self, points, action_window):
        """统一前向接口，兼容 MultiViewTrainer 调用约定。

        Args:
            points: (N_rays, n_samples, 3) 3D 查询点。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            output: (B*N_rays, n_samples, 2) [visibility, density]
        """
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        physics_state = self.temporal(action_window)  # (B, Hidden)
        current_action = action_window[:, -1, :]       # (B, D)

        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        state_expanded = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.hidden_dim)
        action_expanded = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.action_dim)

        return self.decode_spatial(pts_expanded, state_expanded, action_expanded)
