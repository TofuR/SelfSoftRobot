"""SDF-based 3D Self-Model with temporal action encoding.

Fuses Chen 2022's dual-encoder SDF architecture with our EMA temporal encoding
to model viscoelastic hysteresis in soft continuum arms.

Architecture:
  3D query point -> SIREN Coordinate Encoder -> spatial_feat
  action window -> MultiScaleEMA -> temporal_state -> Linear -> state_feat
  concat(spatial_feat, state_feat) -> SIREN Fusion MLP -> SDF value
"""

import torch
import torch.nn as nn
import numpy as np


class SirenLayer(nn.Module):
    """SIREN layer: sin(w0 * (Wx + b))."""

    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


class TemporalSDFModel(nn.Module):
    """SDF model with EMA temporal encoding for viscoelastic hysteresis."""

    def __init__(
        self,
        action_dim=2,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        sdf_hidden=256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        half = sdf_hidden // 2

        # SIREN coordinate encoder（首层 w0=30，后续层 w0=1，与原始论文一致）
        self.coord_encoder = nn.Sequential(
            SirenLayer(3, half, w0=30, is_first=True),
            SirenLayer(half, half, w0=1),
            SirenLayer(half, half, w0=1),
        )

        # EMA temporal encoder (from model_mstnf.py)
        from src.models.model_mstnf import MultiScaleEMA
        self.temporal = MultiScaleEMA(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )
        self.state_proj = nn.Linear(hidden_dim, half)

        # SIREN fusion MLP（首层 w0=30，后续 w0=1）
        self.fusion = nn.Sequential(
            SirenLayer(sdf_hidden, sdf_hidden, w0=30),
            SirenLayer(sdf_hidden, sdf_hidden, w0=1),
            SirenLayer(sdf_hidden, sdf_hidden, w0=1),
            SirenLayer(sdf_hidden, 1, is_last=True),
        )

    def forward(self, coords, action_window):
        """Predict SDF for 3D points given action history.

        Args:
            coords: (N, 3) query coordinates.
            action_window: (B, K, D) action history.

        Returns:
            sdf: (B*N, 1) predicted SDF value.
        """
        B = action_window.shape[0]
        N = coords.shape[0]

        spatial_feat = self.coord_encoder(coords)  # (N, half)
        temporal_state = self.temporal(action_window)  # (B, hidden)
        state_feat = self.state_proj(temporal_state)  # (B, half)

        spatial_expanded = spatial_feat.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        state_expanded = state_feat.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)

        combined = torch.cat([spatial_expanded, state_expanded], dim=-1)
        return self.fusion(combined)

    def forward_with_grad(self, coords, action_window):
        """Forward with coords requiring grad (for Eikonal loss)."""
        coords = coords.clone().detach().requires_grad_(True)
        sdf = self.forward(coords, action_window)
        return sdf, coords

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        state_t = self.temporal(action_windows_t)
        state_t1 = self.temporal(action_windows_t1)
        return torch.mean((state_t1 - state_t) ** 2)

    def get_learned_decays(self):
        return self.temporal.decays.detach().cpu().numpy()
