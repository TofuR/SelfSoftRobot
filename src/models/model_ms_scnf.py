"""MS-SCNF — Multi-Scale Skeleton-Conditioned Neural Field。

核心架构：
  Action Window → MultiScaleEMA → physics_state →
      ├── SkeletonHead → coarse(4×3) / medium(10×3) / fine(31×3)
      │                        ↓
      │               SkeletonConditionedDensity → [vis, density]
"""

import torch
import torch.nn as nn
from .mixins import TemporalMixin, SkeletonMixin
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.heads.skeleton_heads import create_skeleton_head
from src.fields.skeleton_density import SkeletonConditionedDensity
from src.training.spec import PhaseSpec, TrainingSpec


class MSSCNFModel(nn.Module, TemporalMixin, SkeletonMixin):
    """MS-SCNF 完整模型。

    skeleton_mode 选择骨架参数化：
      "point" / "fourier" / "bspline" / "catmullrom"
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("skeleton", freeze_modules=["density"],
                      supervision_mode="skeleton", dataset_type="sequence",
                      dataset_kwargs={"return_3d": True, "return_pairs": False},
                      active_losses=["skeleton"],
                      save_modules=["temporal", "skeleton_head"]),
            PhaseSpec("joint",
                      use_gt_skeleton=True,
                      dataset_type="multiview_depth",
                      supervision_mode="rendering",
                      freeze_modules=["temporal", "skeleton_head"],
                      dataset_kwargs={"return_3d": True},
                      active_losses=["recon", "depth", "reproj", "consist", "smooth"],
                      load_modules={}),
        ],
    )

    # Loss weights for skeleton multi-scale weighting
    _w_skeleton_fine = 1.0
    _w_skeleton_medium = 0.3
    _w_skeleton_coarse = 0.1

    def compute_losses(self, batch, phase_spec):
        """模型层 loss 计算：skeleton + smooth（recon 由 ViewStrategy 处理）。

        Args:
            batch: 统一 dict batch。
            phase_spec: 当前 PhaseSpec。

        Returns:
            dict[str, torch.Tensor]: loss 名到标量 Tensor 的映射。
        """
        losses = {}
        active = set(phase_spec.active_losses)
        device = self.temporal.decays.device

        if "skeleton" in active:
            aw = batch["action_window"].to(device)
            gt_positions = batch["gt_positions"].to(device)
            # GT positions: (B, 3, N) → permute to (B, N, 3)
            if gt_positions.shape[-1] != 3 and gt_positions.shape[1] == 3:
                gt_positions = gt_positions.permute(0, 2, 1)

            pred_dict = self.predict_skeleton(aw)
            skel_losses = self.compute_skeleton_loss(pred_dict, gt_positions)

            losses["skeleton"] = (
                self._w_skeleton_fine * skel_losses["fine"]
                + self._w_skeleton_medium * skel_losses["medium"]
                + self._w_skeleton_coarse * skel_losses["coarse"]
            )

        if "smooth" in active and "action_window_next" in batch and batch["action_window_next"] is not None:
            aw_t = batch["action_window"].to(device)
            aw_t1 = batch["action_window_next"].to(device)
            losses["smooth"] = self.compute_smoothness(aw_t, aw_t1)

        return losses

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

    def forward(self, points, action_window, gt_skeleton=None):
        if gt_skeleton is not None:
            skel_dict = self.skeleton_head.fit_to_points(gt_skeleton)
            skeleton = skel_dict['fine']
        else:
            state = self.encode(action_window)
            skeleton = self.skeleton_head(state)['fine']
        return self.density(points, skeleton)

    def forward_canonical(self, points):
        B = 1
        N, n_samples, _ = points.shape
        device = points.device
        return torch.zeros(N, n_samples, 2, device=device)

    def freeze_canonical(self):
        pass
