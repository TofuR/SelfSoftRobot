"""Smooth-CMSTNF — 在 CMSTNF 架构上增加正则化约束。

三种正则化手段:
  1. 变形 Jacobian 惩罚: ∂D/∂x 应小，保证空间光滑
  2. 时序梯度惩罚: D(x, a_t) 与 D(x, a_{t+1}) 差异应正比于 ||a_t - a_{t+1}||
  3. 光谱归一化: 限制变形 MLP 的 Lipschitz 常数，防止输入微小变化导致输出跳变

与 CMSTNFModel 的唯一区别是变形 MLP 使用了光谱归一化 + 额外的正则化方法。
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from .layers import PositionalEncoder, MLPDecoder
from .model_mstnf import MultiScaleEMA
from .model_cmstnf import CanonicalField
from src.training.spec import PhaseSpec, TrainingSpec


class SpectralMLPDecoder(nn.Module):
    """带光谱归一化的 MLP 解码器，限制 Lipschitz 常数。

    光谱归一化确保网络对输入变化不过度敏感——
    如果 action 变化 ε，输出的 displacement 最多变化 L*ε（L 受控）。
    """

    def __init__(self, input_dim, d_filter=128, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, d_filter * 2)),
            nn.ReLU(),
            spectral_norm(nn.Linear(d_filter * 2, d_filter * 2)),
            nn.ReLU(),
            spectral_norm(nn.Linear(d_filter * 2, d_filter)),
            nn.ReLU(),
            spectral_norm(nn.Linear(d_filter, d_filter // 2)),
            nn.ReLU(),
            spectral_norm(nn.Linear(d_filter // 2, output_size)),
        )

    def forward(self, x):
        return self.net(x)


class SmoothCMSTNFModel(nn.Module):
    """Smooth CMSTNF：canonical + 光滑变形场。

    与 CMSTNFModel 接口完全一致，额外提供正则化方法。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("canonical", freeze_modules=["deform"],
                      forward_attr="forward_canonical", data_mode="canonical",
                      active_losses=["recon"]),
            PhaseSpec("deformation", freeze_modules=["canonical"],
                      forward_attr="forward", data_mode="sequence",
                      active_losses=["recon", "depth", "smooth"]),
        ],
    )

    def __init__(self, action_dim, window_size=20, n_scales=4, hidden_dim=128,
                 d_filter=128, n_freqs=10, deform_n_freqs=6):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.canonical = CanonicalField(d_filter=d_filter, n_freqs=n_freqs)

        # 时序编码器（复用 EMA）
        self.temporal = MultiScaleEMA(
            action_dim=action_dim, n_scales=n_scales,
            window_size=window_size, hidden_dim=hidden_dim,
        )

        # 变形场（使用光谱归一化 MLP）
        self.deform_encoder = PositionalEncoder(d_input=3, n_freqs=deform_n_freqs, log_space=True)
        deform_enc_dim = 3 * (1 + 2 * deform_n_freqs)
        self.deform_mlp = SpectralMLPDecoder(
            input_dim=deform_enc_dim + hidden_dim + action_dim,
            d_filter=d_filter, output_size=3,
        )
        with torch.no_grad():
            self.deform_mlp.net[-1].bias.zero_()

    def _compute_displacement(self, points, action_window):
        B, K, D = action_window.shape
        physics_state = self.temporal(action_window)
        current_action = action_window[:, -1, :]

        N_rays = points.shape[0]
        n_samples = points.shape[1]

        pts_exp = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        x_deform = self.deform_encoder(pts_exp).reshape(-1, self.deform_encoder.d_output)

        state_exp = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.hidden_dim)
        state_flat = state_exp.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.hidden_dim)

        action_exp = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, D)
        action_flat = action_exp.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, D)

        latent = torch.cat([x_deform, state_flat, action_flat], dim=-1)
        displacement = self.deform_mlp(latent).reshape(B * N_rays, n_samples, 3)
        return displacement, physics_state

    # ── 接口与 CMSTNFModel 一致 ──

    def forward_canonical(self, points):
        return self.canonical(points)

    def forward(self, points, action_window):
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        displacement, _ = self._compute_displacement(points, action_window)
        pts_exp = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        return self.canonical(pts_exp + displacement)

    def compute_smoothness(self, aw_t, aw_t1):
        s_t = self.temporal(aw_t)
        s_t1 = self.temporal(aw_t1)
        return torch.mean((s_t1 - s_t) ** 2)

    def compute_jacobian_penalty(self, points, action_window, eps=1e-3):
        """变形 Jacobian 惩罚：∂D/∂x 应该小。

        通过有限差分近似 Jacobian 的 Frobenius 范数。
        惩罚空间上邻近点的变形差异过大。
        """
        B, K, D = action_window.shape
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        # 只在少量点上计算（节省显存）
        n_check = min(64, N_rays)
        pts_check = points[:n_check]

        # 基准变形
        disp0, _ = self._compute_displacement(pts_check, action_window)

        # 沿 x, y, z 方向扰动
        penalty = torch.tensor(0.0, device=points.device)
        for axis in range(3):
            pts_shifted = pts_check.clone()
            pts_shifted[:, :, axis] += eps
            disp_shifted, _ = self._compute_displacement(pts_shifted, action_window)
            # Jacobian 列的范数
            penalty = penalty + ((disp_shifted - disp0) / eps).pow(2).mean()

        return penalty

    def compute_temporal_gradient_penalty(self, points, aw_t, aw_t1):
        """时序梯度惩罚：变形的变化应正比于动作变化。

        ||D(x, a_t) - D(x, a_{t+1})|| / ||a_t - a_{t+1}|| 应有界。
        """
        action_diff = (aw_t[:, -1, :] - aw_t1[:, -1, :]).norm(dim=-1).mean()
        if action_diff < 1e-8:
            return torch.tensor(0.0, device=points.device)

        disp_t, _ = self._compute_displacement(points[:64], aw_t)
        disp_t1, _ = self._compute_displacement(points[:64], aw_t1)

        deform_diff = (disp_t - disp_t1).norm() / action_diff
        return deform_diff.pow(2)

    def freeze_canonical(self):
        for p in self.canonical.parameters():
            p.requires_grad = False

    def unfreeze_canonical(self):
        for p in self.canonical.parameters():
            p.requires_grad = True

    def get_learned_decays(self):
        return self.temporal.decays.detach().cpu().numpy()
