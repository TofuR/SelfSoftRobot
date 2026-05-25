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
import torch.nn.functional as F

from .layers import SirenLayer
from .mixins import TemporalMixin
from src.training.spec import TrainingSpec, PhaseSpec


class TemporalSDFModel(nn.Module, TemporalMixin):
    """SDF model with EMA temporal encoding for viscoelastic hysteresis."""

    # 训练需求声明：单阶段 direct_3d 监督，启用 sdf/normal/eikonal loss
    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("full", dataset_type="sdf", supervision_mode="direct_3d",
                      active_losses=["sdf", "normal", "eikonal"]),
        ],
    )

    def __init__(
        self,
        action_dim=2,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        sdf_hidden=256,
        w_sdf=3e3,
        w_normal=1e2,
        w_eikonal=5e1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # SDF loss 权重
        self.w_sdf = w_sdf
        self.w_normal = w_normal
        self.w_eikonal = w_eikonal

        half = sdf_hidden // 2

        # SIREN coordinate encoder（首层 w0=30，后续层 w0=1，与原始论文一致）
        self.coord_encoder = nn.Sequential(
            SirenLayer(3, half, w0=30, is_first=True),
            SirenLayer(half, half, w0=1),
            SirenLayer(half, half, w0=1),
        )

        # EMA temporal encoder
        from src.encoders.multi_scale_ema import MultiScaleEMA
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

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """计算 SDF 模型的损失。

        从 SDFTrainer.compute_loss() 迁移而来。根据 phase_spec.active_losses
        决定计算哪些 loss。

        Args:
            batch: 统一 dict batch，包含:
                "action_window": (B, K, D) 动作序列
                "coords": (B, N, 3) 查询坐标
                "gt_sdf": (B, N, 1) GT SDF 值
                "gt_normals": (B, N, 3) GT 法向量
            phase_spec: 当前 PhaseSpec。

        Returns:
            dict[str, torch.Tensor]: loss 名到标量 Tensor 的映射。
        """
        active = set(phase_spec.active_losses)

        device = next(self.parameters()).device

        action_window = batch["action_window"].to(device)
        coords = batch["coords"].to(device)
        gt_sdf = batch["gt_sdf"].to(device)
        gt_normals = batch.get("gt_normals")
        if gt_normals is not None:
            gt_normals = gt_normals.to(device)

        # B=1 时 squeeze batch 维度，与原始 SDFTrainer 行为一致
        if coords.dim() == 3 and coords.shape[0] == 1:
            coords = coords.squeeze(0)        # (N, 3)
            gt_sdf = gt_sdf.squeeze(0)        # (N, 1)
            if gt_normals is not None:
                gt_normals = gt_normals.squeeze(0)  # (N, 3)

        coords = coords.requires_grad_(True)

        pred_sdf = self.forward(coords, action_window)  # (B*N, 1) or (N, 1)

        # 确保 gt_sdf 与 pred_sdf 形状匹配
        gt_sdf = gt_sdf.reshape(-1, 1)
        if gt_normals is not None:
            gt_normals = gt_normals.reshape(-1, 3)

        # 计算 SDF 梯度（用于 normal 和 eikonal loss）
        gradient = torch.autograd.grad(
            outputs=pred_sdf,
            inputs=coords,
            grad_outputs=torch.ones_like(pred_sdf),
            create_graph=True,
        )[0]

        losses = {}

        # SDF L1 回归: 所有点
        if "sdf" in active:
            losses["sdf"] = torch.abs(pred_sdf - gt_sdf).mean() * self.w_sdf

        # 法向量 loss: 仅表面点 (gt_sdf == 0)
        if "normal" in active and gt_normals is not None:
            is_surface = (gt_sdf.abs() < 1e-6).float()
            if is_surface.sum() > 0:
                cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
                losses["normal"] = (
                    (is_surface * (1 - cos_sim)).sum()
                    / (is_surface.sum() + 1e-8)
                    * self.w_normal
                )
            else:
                losses["normal"] = torch.tensor(0.0, device=device)

        # Eikonal: 梯度模 = 1
        if "eikonal" in active:
            if self.w_eikonal > 0:
                losses["eikonal"] = (
                    ((gradient.norm(dim=-1) - 1) ** 2).mean() * self.w_eikonal
                )
            else:
                losses["eikonal"] = torch.tensor(0.0, device=device)

        return losses
