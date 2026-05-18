"""MS-SCNF — Multi-Scale Skeleton-Conditioned Neural Field。

核心架构：
  Action Window → MultiScaleEMA → physics_state →
      ├── SkeletonHead → coarse(4×3) / medium(10×3) / fine(31×3)
      │                        ↓
      │               SkeletonConditionedDensity → [vis, density]

skeleton_mode 选择骨架参数化方式（详见 skeleton_heads.py）：
  "point" / "fourier" / "bspline" / "catmullrom"

训练分两阶段：
  Phase 1: 仅 SkeletonHead（3D L2 loss，GT 来自仿真器）
  Phase 2: 联合 SkeletonHead + DensityField（3D + 2D rendering loss）
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder
from .model_mstnf import MultiScaleEMA
from .skeleton_heads import (
    SkeletonHead, SKELETON_MODES, create_skeleton_head,
    downsample_skeleton, point_to_segment_distance,
)
from src.training.spec import PhaseSpec, TrainingSpec


class SkeletonConditionedDensity(nn.Module):
    """骨架条件密度场：查询点密度取决于到骨架曲线的距离。"""

    def __init__(self, n_freqs=6, d_filter=128):
        super().__init__()
        self.dist_encoder = PositionalEncoder(d_input=1, n_freqs=n_freqs, log_space=True)
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=4, log_space=True)

        dist_enc_dim = 1 * (1 + 2 * n_freqs)
        pos_enc_dim = 3 * (1 + 2 * 4)
        input_dim = dist_enc_dim + pos_enc_dim

        self.decoder = MLPDecoder(input_dim=input_dim, d_filter=d_filter, output_size=2)
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

    def forward(self, query_points, skeleton):
        N, n_samples, _ = query_points.shape
        B = skeleton.shape[0]
        n_seg = skeleton.shape[1] - 1

        pts = query_points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, n_samples, 3)
        skel_expanded = skeleton.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, 3)
        seg_start = skel_expanded[:, :-1, :]
        seg_end = skel_expanded[:, 1:, :]

        dist = point_to_segment_distance(pts, seg_start, seg_end)
        dist = dist.unsqueeze(-1)

        dist_enc = self.dist_encoder(dist)
        pos_enc = self.pos_encoder(pts)
        latent = torch.cat([dist_enc, pos_enc], dim=-1)

        return self.decoder(latent)


class MSSCNFModel(nn.Module):
    """MS-SCNF 完整模型。

    skeleton_mode 选择骨架参数化：
      "point" / "fourier" / "bspline" / "catmullrom"
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("skeleton", forward_attr="forward", data_mode="sequence",
                      active_losses=["skeleton"]),
            PhaseSpec("joint", forward_attr="forward", data_mode="sequence",
                      active_losses=["skeleton", "recon", "smooth"]),
        ],
    )

    def __init__(
        self,
        action_dim,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        d_filter=128,
        n_freqs=10,
        n_coarse=4,
        n_medium=10,
        n_fine=31,
        deform_n_freqs=6,
        skeleton_mode="point",
        fourier_n_freq=8,
        bspline_n_ctrl=10,
        catmullrom_n_ctrl=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.n_fine = n_fine
        self.skeleton_mode = skeleton_mode

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
            self.skeleton_head = create_skeleton_head("fourier", **skel_kwargs)
        elif skeleton_mode == "bspline":
            skel_kwargs["n_ctrl"] = bspline_n_ctrl
            self.skeleton_head = create_skeleton_head("bspline", **skel_kwargs)
        elif skeleton_mode == "catmullrom":
            skel_kwargs["n_ctrl"] = catmullrom_n_ctrl
            self.skeleton_head = create_skeleton_head("catmullrom", **skel_kwargs)
        else:
            self.skeleton_head = create_skeleton_head("point", **skel_kwargs)

        self.density = SkeletonConditionedDensity(
            n_freqs=deform_n_freqs,
            d_filter=d_filter,
        )

        self.canonical = nn.Module()

    def skeleton_config(self):
        """返回骨架配置 dict，用于保存到实验 log。"""
        cfg = {"skeleton_mode": self.skeleton_mode, "n_fine": self.n_fine}
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
        state = self.encode(action_window)
        return self.skeleton_head(state)

    def forward(self, points, action_window):
        state = self.encode(action_window)
        skel_dict = self.skeleton_head(state)
        skeleton = skel_dict['fine']
        return self.density(points, skeleton)

    def forward_canonical(self, points):
        B = 1
        N, n_samples, _ = points.shape
        device = points.device
        return torch.zeros(N, n_samples, 2, device=device)

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

    def freeze_canonical(self):
        pass

    def get_learned_decays(self):
        return self.temporal.decays.detach().cpu().numpy()
