"""TemporalTCN — 时序卷积网络编码器。

1D 因果膨胀卷积在动作窗口上操作，指数级增长的膨胀率提供多尺度时间感受野。

与加权求和编码器的关键区别：
  卷积核是可学习的局部时间模式检测器，保持序列顺序。
  因果约束确保只有过去动作影响当前状态。
  膨胀卷积以 O(log K) 层数覆盖整个窗口。

接口兼容 MultiScaleEMA / FractionalMemory / GammaLaguerre：
  forward(action_window) -> (B, hidden_dim)
  compute_smoothness(aw_t, aw_t1) -> scalar
  decays property (synthetic, for TemporalMixin logging)
"""

import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _CausalConv1dBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            channels, channels, kernel_size,
            dilation=dilation, padding=0,
        ))
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = nn.functional.pad(x, (self.padding, 0))
        return self.relu(self.conv(x)) + res


class TemporalTCN(nn.Module):
    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128,
                 kernel_size=3):
        super().__init__()
        self.action_dim = action_dim
        self.n_scales = n_scales
        self.window_size = window_size
        self.kernel_size = kernel_size

        n_channels = n_scales * action_dim
        self.input_proj = nn.Linear(action_dim, n_channels)

        n_layers = max(1, math.ceil(math.log2(max(window_size, 2))))
        self.tcn_layers = nn.ModuleList([
            _CausalConv1dBlock(n_channels, kernel_size, dilation=2 ** i)
            for i in range(n_layers)
        ])

        self._synthetic_decays = nn.Parameter(
            torch.linspace(0.2, 0.95, n_scales).logit()
        )

        mlp_input_dim = n_channels + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def decays(self):
        return torch.sigmoid(self._synthetic_decays)

    def forward(self, action_window):
        B, K, D = action_window.shape

        x = self.input_proj(action_window)  # (B, K, n_channels)
        x = x.transpose(1, 2)  # (B, n_channels, K)

        for layer in self.tcn_layers:
            x = layer(x)

        tcn_feat = x[:, :, -1]  # (B, n_channels) — 取最后时间步（因果）

        current_action = action_window[:, -1, :]
        velocity = (action_window[:, -1, :] - action_window[:, -2, :]
                    if K >= 2 else torch.zeros_like(current_action))

        features = torch.cat([tcn_feat, current_action, velocity], dim=-1)
        return self.state_mlp(features)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        s_t = self.forward(action_windows_t)
        s_t1 = self.forward(action_windows_t1)
        return torch.mean((s_t1 - s_t) ** 2)
