"""动作条件变形场：输出 3D 位移 (Δx, Δy, Δz)。

使用 MultiScaleEMA 编码动作历史，低频位置编码保证变形光滑。
"""

import torch
import torch.nn as nn
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.models.layers import PositionalEncoder, MLPDecoder


class DeformationField(nn.Module):
    """动作条件变形场。

    Args:
        action_dim: 动作维度。
        window_size: 时序窗口长度。
        n_scales: EMA 尺度数。
        hidden_dim: 物理状态维度。
        d_filter: MLP 隐层维度。
        deform_n_freqs: 变形位置编码频率数。
    """

    def __init__(self, action_dim, window_size=20, n_scales=4, hidden_dim=128,
                 d_filter=128, deform_n_freqs=6):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.temporal = MultiScaleEMA(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        self.deform_encoder = PositionalEncoder(d_input=3, n_freqs=deform_n_freqs, log_space=True)
        deform_enc_dim = 3 * (1 + 2 * deform_n_freqs)

        self.deform_mlp = MLPDecoder(
            input_dim=deform_enc_dim + hidden_dim + action_dim,
            d_filter=d_filter,
            output_size=3,
        )

        with torch.no_grad():
            self.deform_mlp.net[-1].bias.zero_()

    def forward(self, points, action_window):
        """计算变形位移。

        Args:
            points: (N_rays, n_samples, 3) 空间查询点（不含 batch 维度）。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            displacement: (B*N_rays, n_samples, 3) 3D 位移。
            physics_state: (B, Hidden) 物理状态（用于 smoothness loss）。
        """
        B, K, D = action_window.shape
        physics_state = self.temporal(action_window)  # (B, Hidden)
        current_action = action_window[:, -1, :]  # (B, D)

        N_rays = points.shape[0]
        n_samples = points.shape[1]

        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)

        x_deform = self.deform_encoder(pts_expanded)
        x_deform_flat = x_deform.reshape(-1, x_deform.shape[-1])

        state_expanded = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.hidden_dim)
        state_for_mlp = state_expanded.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.hidden_dim)

        action_expanded = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, D)
        action_for_mlp = action_expanded.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, D)

        latent = torch.cat([x_deform_flat, state_for_mlp, action_for_mlp], dim=-1)
        displacement = self.deform_mlp(latent)
        displacement = displacement.reshape(B * N_rays, n_samples, 3)

        return displacement, physics_state
