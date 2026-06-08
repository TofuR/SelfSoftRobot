"""TemporalTransformer — 自注意力时序编码器。

在动作窗口上应用 Transformer Encoder，用可学习 CLS token 聚合全局信息。
正弦位置编码提供时间感知。

与加权求和编码器的关键区别：
  自注意力机制让每个时间步都能看到完整历史，且通过位置编码区分顺序。
  全局注意力 = 最强的顺序保持能力，适合作为"上限基线"。

接口兼容 MultiScaleEMA / FractionalMemory / GammaLaguerre：
  forward(action_window) -> (B, hidden_dim)
  compute_smoothness(aw_t, aw_t1) -> scalar
  decays property (synthetic, for TemporalMixin logging)
"""

import math
import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128,
                 n_heads=4, n_layers=2):
        super().__init__()
        self.action_dim = action_dim
        self.n_scales = n_scales
        self.window_size = window_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        model_dim = n_scales * action_dim
        if model_dim % n_heads != 0:
            model_dim = n_heads * (model_dim // n_heads + 1)

        self.model_dim = model_dim
        self.input_proj = nn.Linear(action_dim, model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=4 * model_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self._synthetic_decays = nn.Parameter(
            torch.linspace(0.2, 0.95, n_scales).logit()
        )

        mlp_input_dim = model_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _sinusoidal_pe(self, length, dim, device):
        position = torch.arange(length, device=device).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float()
            * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    @property
    def decays(self):
        return torch.sigmoid(self._synthetic_decays)

    def forward(self, action_window):
        B, K, D = action_window.shape

        x = self.input_proj(action_window)  # (B, K, model_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, K+1, model_dim)

        pe = self._sinusoidal_pe(K + 1, self.model_dim, x.device)
        x = x + pe.unsqueeze(0)

        x = self.transformer(x)  # (B, K+1, model_dim)
        cls_out = x[:, 0, :]  # (B, model_dim)

        current_action = action_window[:, -1, :]
        velocity = (action_window[:, -1, :] - action_window[:, -2, :]
                    if K >= 2 else torch.zeros_like(current_action))

        features = torch.cat([cls_out, current_action, velocity], dim=-1)
        return self.state_mlp(features)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        s_t = self.forward(action_windows_t)
        s_t1 = self.forward(action_windows_t1)
        return torch.mean((s_t1 - s_t) ** 2)
