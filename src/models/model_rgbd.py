"""RGB-D Neural Field — 原生深度信息驱动的神经场模型。

与 CMSTNF 的关键区别:
  - 深度图作为模型输入（通过 DepthEncoder 编码为条件特征）
  - 单阶段端到端训练（不需要两阶段 canonical + deformation）
  - Depth-aware volume rendering

架构:
  深度图 → DepthEncoder → f_depth
  驱动序列 → ActuatorMLPEncoder + TemporalEMA → f_act
  f_condition = concat(f_depth, f_act)
  3D points + f_condition → NeuralField → [visibility, density]
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder, DepthEncoder, ActuatorMLPEncoder


class TemporalEMA(nn.Module):
    """简单的指数移动平均时序编码器（轻量版 MultiScaleEMA）。"""

    def __init__(self, action_dim, hidden_dim=64, ema_decay=0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ema_decay = nn.Parameter(torch.tensor(ema_decay))
        self.encoder = ActuatorMLPEncoder(action_dim, feat_dim=hidden_dim)

    def forward(self, action_window):
        B, K, D = action_window.shape
        features = self.encoder(action_window.reshape(B * K, D)).reshape(B, K, -1)
        decay = torch.sigmoid(self.ema_decay)
        state = features[:, 0]
        for t in range(1, K):
            state = decay * state + (1 - decay) * features[:, t]
        return state


class RGBDNeuralField(nn.Module):
    """RGB-D Neural Field: 深度图条件化的 3D 神经场。

    单阶段端到端训练，深度信息从第一轮就参与。

    Args:
        action_dim: 驱动信号维度。
        depth_feat_dim: DepthEncoder 输出维度。
        hidden_dim: 时序编码器隐层维度。
        d_filter: MLP 解码器滤波器维度。
        n_freqs: 位置编码频率数。
        window_size: 动作窗口长度。
    """

    def __init__(
        self,
        action_dim=2,
        depth_feat_dim=64,
        hidden_dim=64,
        d_filter=128,
        n_freqs=10,
        window_size=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth_feat_dim = depth_feat_dim

        self.depth_encoder = DepthEncoder(feat_dim=depth_feat_dim)
        self.temporal = TemporalEMA(action_dim, hidden_dim=hidden_dim)
        self.action_encoder = ActuatorMLPEncoder(action_dim, feat_dim=hidden_dim)

        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=n_freqs, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * n_freqs)

        condition_dim = depth_feat_dim + hidden_dim + action_dim
        self.field = MLPDecoder(
            input_dim=pos_enc_dim + condition_dim,
            d_filter=d_filter,
            output_size=2,
        )

        with torch.no_grad():
            self.field.net[-1].bias[1] = 0.0

    def forward(self, points, action_window, depth_map=None):
        """查询神经场。

        Args:
            points: (N_rays, n_samples, 3) 3D 查询点。
            action_window: (B, K, D) 动作序列窗口。
            depth_map: (B, 1, H, W) 深度图（可选）。

        Returns:
            output: (B*N_rays, n_samples, 2) [visibility, density]
        """
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        temporal_state = self.temporal(action_window)
        current_action = action_window[:, -1, :]
        action_feat = self.action_encoder(current_action)

        if depth_map is not None:
            depth_feat = self.depth_encoder(depth_map)
        else:
            depth_feat = torch.zeros(B, self.depth_feat_dim, device=points.device)

        condition = torch.cat([depth_feat, action_feat, current_action], dim=-1)

        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(
            B * N_rays, n_samples, 3)

        pos_enc = self.pos_encoder(pts_expanded)

        condition_expanded = condition.unsqueeze(1).expand(-1, N_rays, -1).reshape(
            B * N_rays, -1)
        condition_per_point = condition_expanded.unsqueeze(1).expand(
            -1, n_samples, -1)

        mlp_input = torch.cat([pos_enc, condition_per_point], dim=-1)
        output = self.field(mlp_input.reshape(-1, mlp_input.shape[-1]))
        return output.reshape(B * N_rays, n_samples, 2)

    def forward_rendering(self, points, action_window, depth_map=None):
        return self.forward(points, action_window, depth_map)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        state_t = self.temporal(action_windows_t)
        state_t1 = self.temporal(action_windows_t1)
        return torch.mean((state_t1 - state_t) ** 2)
