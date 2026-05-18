"""SkeletonSDF — 参数化骨架 + 截面 SDF 全身 3D 自建模。

架构：
  Action Window -> MultiScaleEMA -> physics_state
                                      |
                           SkeletonHead(bspline/fourier/...) -> skeleton curve (31x3)
                                      |
              对每个查询点 x:
                dist = point_to_segment_distance(x, skeleton)
                sdf_prior = dist - radius（管状 SDF 先验）
                residual = SIREN(sdf_prior, 位置编码(x), physics_state)
                final_sdf = sdf_prior + residual

  训练信号：
    - 骨架 loss: 31 节点 L2（GT 来自 PyElastica positions）
    - SDF loss: |pred_sdf - gt_sdf|（GT 解析计算: dist_to_skeleton - radius）
    - Eikonal loss: ||grad SDF|| = 1
    - 表面 loss: SDF = 0 在表面上

  优势：
    - 参数化骨架保证拓扑连通（不会断裂）
    - SDF 截面提供完整 3D 体积信息（不仅中心线）
    - 管状先验 + 残差学习 = 收敛快 + 精度高
"""

import torch
import torch.nn as nn
import numpy as np

from .model_mstnf import MultiScaleEMA
from .skeleton_heads import (
    create_skeleton_head, downsample_skeleton, point_to_segment_distance,
)


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


class SkeletonSDFModel(nn.Module):
    """参数化骨架 + 截面 SDF 模型。

    Args:
        action_dim: 驱动维度。
        skeleton_mode: 骨架参数化方式（point/fourier/bspline/catmullrom）。
        rod_radius: 软臂半径（用于管状 SDF 先验）。
        sdf_residual: 是否使用 SIREN 残差修正 SDF 先验。
    """

    def __init__(
        self,
        action_dim,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        n_coarse=4,
        n_medium=10,
        n_fine=31,
        skeleton_mode="bspline",
        fourier_n_freq=8,
        bspline_n_ctrl=10,
        catmullrom_n_ctrl=10,
        rod_radius=0.015,
        sdf_residual=True,
        sdf_hidden=128,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_fine = n_fine
        self.skeleton_mode = skeleton_mode
        self.rod_radius = rod_radius
        self.sdf_residual = sdf_residual

        self.temporal = MultiScaleEMA(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        skel_kwargs = dict(hidden_dim=hidden_dim, n_coarse=n_coarse,
                           n_medium=n_medium, n_fine=n_fine)
        if skeleton_mode == "fourier":
            skel_kwargs["n_freq"] = fourier_n_freq
        elif skeleton_mode in ("bspline", "catmullrom"):
            skel_kwargs["n_ctrl"] = (
                bspline_n_ctrl if skeleton_mode == "bspline" else catmullrom_n_ctrl)
        self.skeleton_head = create_skeleton_head(skeleton_mode, **skel_kwargs)

        if sdf_residual:
            pos_enc_dim = 3 * (1 + 2 * 4)  # 27
            self.state_proj = nn.Linear(hidden_dim, 32)
            input_dim = 1 + pos_enc_dim + 32  # 60
            self.sdf_net = nn.Sequential(
                SirenLayer(input_dim, sdf_hidden, w0=30, is_first=True),
                SirenLayer(sdf_hidden, sdf_hidden, w0=1),
                SirenLayer(sdf_hidden, sdf_hidden, w0=1),
                SirenLayer(sdf_hidden, 1, is_last=True),
            )

    def skeleton_config(self):
        cfg = {"skeleton_mode": self.skeleton_mode, "n_fine": self.n_fine,
               "rod_radius": self.rod_radius}
        if self.skeleton_mode == "fourier":
            cfg["fourier_n_freq"] = self.skeleton_head.n_freq
        elif self.skeleton_mode == "bspline":
            cfg["bspline_n_ctrl"] = self.skeleton_head.n_ctrl
        elif self.skeleton_mode == "catmullrom":
            cfg["catmullrom_n_ctrl"] = self.skeleton_head.n_ctrl
        return cfg

    def encode(self, action_window):
        return self.temporal(action_window)

    def predict_skeleton(self, action_window):
        return self.skeleton_head(self.encode(action_window))

    def _positional_encode(self, x, n_freqs=4):
        enc = [x]
        for k in range(n_freqs):
            freq = 2 ** k * x
            enc.append(torch.sin(freq))
            enc.append(torch.cos(freq))
        return torch.cat(enc, dim=-1)

    def forward(self, query_points, action_window):
        """预测查询点的 SDF 值。

        Args:
            query_points: (N, n_samples, 3) 空间查询点。
            action_window: (B, K, D) 动作序列。

        Returns:
            sdf: (B*N, n_samples, 1) SDF 值。
        """
        state = self.encode(action_window)
        skeleton = self.skeleton_head(state)['fine']  # (B, 31, 3)

        N, n_samples, _ = query_points.shape
        B = skeleton.shape[0]

        pts = query_points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, n_samples, 3)
        skel_exp = skeleton.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, 3)

        dist = point_to_segment_distance(pts, skel_exp[:, :-1, :], skel_exp[:, 1:, :])
        sdf_prior = dist - self.rod_radius

        if not self.sdf_residual:
            return sdf_prior.unsqueeze(-1)

        state_feat = self.state_proj(state)
        pos_enc = self._positional_encode(pts, n_freqs=4)

        sdf_in = torch.cat([
            sdf_prior.unsqueeze(-1),
            pos_enc,
            state_feat.unsqueeze(1).expand(-1, N * n_samples, -1)
                .reshape(B * N, n_samples, 32),
        ], dim=-1)

        residual = self.sdf_net(sdf_in)
        return sdf_prior.unsqueeze(-1) + residual

    def compute_skeleton_loss(self, pred_dict, gt_positions):
        losses = {}
        losses['fine'] = ((pred_dict['fine'] - gt_positions) ** 2).mean()
        gt_medium = downsample_skeleton(gt_positions, pred_dict['medium'].shape[-2])
        losses['medium'] = ((pred_dict['medium'] - gt_medium) ** 2).mean()
        gt_coarse = downsample_skeleton(gt_positions, pred_dict['coarse'].shape[-2])
        losses['coarse'] = ((pred_dict['coarse'] - gt_coarse) ** 2).mean()
        return losses

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        return self.temporal.compute_smoothness(action_windows_t, action_windows_t1)

    def get_learned_decays(self):
        return self.temporal.decays.detach().cpu().numpy()
