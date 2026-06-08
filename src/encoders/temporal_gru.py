"""TemporalGRU — GRU 时序编码器。

单向 GRU 沿动作窗口的时间维度逐步处理，最终隐状态捕获加载历史的累积效应。
物理类比：隐状态 = 内部应力/应变状态，随加载历史逐步演化。

与加权求和类编码器（EMA/Fractional/Gamma）的关键区别：
  GRU 的门控机制天然保留了序列顺序——充气 [0.1→0.4] 和放气 [0.4→0.1]
  产生完全不同的隐状态，无需手动设计加权核。

接口兼容 MultiScaleEMA / FractionalMemory / GammaLaguerre：
  forward(action_window) -> (B, hidden_dim)
  compute_smoothness(aw_t, aw_t1) -> scalar
  decays property (synthetic, for TemporalMixin logging)
"""

import torch
import torch.nn as nn


class TemporalGRU(nn.Module):
    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.n_scales = n_scales
        self.window_size = window_size

        gru_hidden = n_scales * action_dim
        self.gru = nn.GRU(
            input_size=action_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self._synthetic_decays = nn.Parameter(
            torch.linspace(0.2, 0.95, n_scales).logit()
        )

        mlp_input_dim = gru_hidden + 2 * action_dim
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

        _, h_n = self.gru(action_window)
        gru_feat = h_n.squeeze(0)  # (B, gru_hidden)

        current_action = action_window[:, -1, :]
        velocity = (action_window[:, -1, :] - action_window[:, -2, :]
                    if K >= 2 else torch.zeros_like(current_action))

        features = torch.cat([gru_feat, current_action, velocity], dim=-1)
        return self.state_mlp(features)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        s_t = self.forward(action_windows_t)
        s_t1 = self.forward(action_windows_t1)
        return torch.mean((s_t1 - s_t) ** 2)
