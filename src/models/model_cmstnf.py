"""Canonical MSTNF (C-MSTNF) — D-NeRF 范式的形态先验 + 变形场。

训练分两阶段:
  Phase 1: 训练 CanonicalField — 用零动作数据学习机器人静止形态
  Phase 2: 训练 DeformationField — 冻结 canonical，用运动数据学习动作变形

查询流程:
  世界空间点 → DeformationField → canonical 坐标 → CanonicalField → [vis, dens]
"""

import torch
import torch.nn as nn
from src.fields.canonical import CanonicalField
from src.fields.deformation import DeformationField
from src.training.spec import PhaseSpec, TrainingSpec


class CMSTNFModel(nn.Module):
    """Canonical MSTNF 完整模型。

    包含两个子模块:
      - canonical: CanonicalField，表示静止态形态
      - deform: DeformationField，学习动作引起的 3D 变形

    Phase 1: 只训练 canonical，使用零动作数据
    Phase 2: 冻结 canonical，训练 deform，使用运动数据
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("canonical", freeze_modules=["deform"],
                      forward_attr="forward_canonical", data_mode="canonical",
                      active_losses=["recon"],
                      save_modules=["canonical"]),
            PhaseSpec("deformation", freeze_modules=["canonical"],
                      forward_attr="forward", data_mode="sequence",
                      active_losses=["recon", "smooth"],
                      load_modules={"canonical": "canonical"}),
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
        deform_n_freqs=6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.canonical = CanonicalField(d_filter=d_filter, n_freqs=n_freqs)
        self.deform = DeformationField(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=n_scales,
            hidden_dim=hidden_dim,
            d_filter=d_filter,
            deform_n_freqs=deform_n_freqs,
        )

    def forward_canonical(self, points):
        """Phase 1 用：直接查 canonical field。

        Args:
            points: (N, n_samples, 3)

        Returns:
            output: (N, n_samples, 2) [visibility, density]
        """
        return self.canonical(points)

    def forward(self, points, action_window):
        """Phase 2 用：变形 → canonical。

        Args:
            points: (N_rays, n_samples, 3) 世界空间查询点。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            output: (B*N_rays, n_samples, 2) [visibility, density]
        """
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        displacement, _ = self.deform(points, action_window)  # (B*N_rays, n_samples, 3)
        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        return self.canonical(pts_expanded + displacement)

    def forward_with_state(self, points, action_window):
        """Phase 2 用，额外返回 physics_state 用于 smoothness loss。

        Returns:
            output: (B*N_rays, n_samples, 2)
            physics_state: (B, Hidden)
        """
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        displacement, physics_state = self.deform(points, action_window)
        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        output = self.canonical(pts_expanded + displacement)
        return output, physics_state

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        """变形场的时序平滑 loss：相邻帧变形位移应连续。"""
        dummy_points = torch.zeros(1, 1, 3, device=action_windows_t.device)
        _, state_t = self.deform(dummy_points, action_windows_t)
        _, state_t1 = self.deform(dummy_points, action_windows_t1)
        return torch.mean((state_t1 - state_t) ** 2)

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """模型层 loss 计算。处理 "smooth" loss。"""
        losses = {}
        active = set(phase_spec.active_losses)

        if "smooth" in active and batch.get("action_window_next") is not None:
            aw_t = batch["action_window"]
            aw_t1 = batch["action_window_next"]
            losses["smooth"] = self.compute_smoothness(aw_t, aw_t1)

        return losses

    def freeze_canonical(self):
        """冻结 canonical 参数，Phase 2 开始时调用。"""
        for p in self.canonical.parameters():
            p.requires_grad = False

    def unfreeze_canonical(self):
        """解冻 canonical 参数（如需要 fine-tune）。"""
        for p in self.canonical.parameters():
            p.requires_grad = True

    def get_learned_decays(self):
        """返回 EMA 学到的衰减率。"""
        return self.deform.temporal.decays.detach().cpu().numpy()
