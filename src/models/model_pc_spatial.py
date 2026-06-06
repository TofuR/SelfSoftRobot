"""PCSpatialSequenceModel — 预测-修正空间序列模型。

两阶段架构:
  Phase 1 (Predictive):
    Action Window → FractionalMemory → GRU(z₀→z_K) → 预测中心线
    纯 3D GT 监督，学习 action_history → 3D_skeleton 的映射

  Phase 2 (Corrective):
    Phase 1 预测 + 图像编码器 → 残差修正 → 最终中心线
    修正分支学习视觉观测与预测之间的差异

核心思想:
  预测分支提供基于驱动历史的强先验（仿真预训练），
  修正分支通过视觉观测弥补模型误差（sim-to-real）。
  类似 Kalman 滤波：模型预测 + 观测修正。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins import TemporalMixin
from src.encoders.fractional_memory import FractionalMemory
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.training.spec import TrainingSpec, PhaseSpec
from src.data.dataset_pointcloud import _sample_surface


class PCSpatialSequenceModel(nn.Module, TemporalMixin):
    """预测-修正空间序列模型。"""

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                name="predictive",
                dataset_type="spatial_sequence",
                supervision_mode="spatial_sequence",
                active_losses=["skeleton", "spatial_smooth", "smooth"],
                forward_attr="forward_predictive",
                freeze_modules=["correction"],
            ),
            PhaseSpec(
                name="corrective",
                dataset_type="spatial_sequence",
                supervision_mode="spatial_sequence",
                active_losses=["skeleton", "spatial_smooth", "smooth"],
                forward_attr="forward_corrective",
                freeze_modules=[],
            ),
        ],
    )

    def __init__(
        self,
        action_dim=2,
        n_nodes=31,
        hidden_dim=128,
        window_size=20,
        n_orders=4,
        encoder_type="fractional",
        n_views=2,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        self.register_buffer('pc_center', torch.zeros(1, 1, 3))
        self.register_buffer('pc_scale', torch.ones(1, 1, 3))
        self.register_buffer('action_norm_factor', torch.tensor(1.0))

        # ── 预测分支 ──
        EncoderClass = FractionalMemory if encoder_type == "fractional" else MultiScaleEMA
        self.temporal = EncoderClass(
            action_dim=action_dim,
            n_scales=n_orders,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )
        self.z_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.slice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3),
        )
        self.init_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 修正分支 ──
        self.correction = nn.ModuleDict({
            "image_encoder": nn.Sequential(
                nn.Conv2d(n_views, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, hidden_dim), nn.ReLU(),
            ),
            "correction_head": nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_nodes * 3),
            ),
        })

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center).float()
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        self.pc_center = center.view(1, 1, 3)
        self.pc_scale = scale.view(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    def _get_z_positions(self, device):
        return torch.linspace(-1, 1, self.n_nodes, device=device)

    def _predict(self, action_window):
        """预测分支：GRU 沿 Z 轴生成中心线。"""
        device = action_window.device
        B = action_window.shape[0]
        cond = self.encode(action_window)
        h = self.init_hidden(cond)
        z_positions = self._get_z_positions(device)
        skeleton = []
        for i in range(self.n_nodes):
            z_emb = self.z_embed(z_positions[i:i + 1].unsqueeze(0).expand(B, -1))
            h = self.gru(cond + z_emb, h)
            skeleton.append(self.slice_head(h))
        return torch.stack(skeleton, dim=1)

    def _correct(self, pred, images):
        """修正分支：从图像编码残差修正。"""
        if images is None:
            return pred
        B = pred.shape[0]
        if isinstance(images, list):
            imgs = torch.stack(images, dim=1).float()
        else:
            if images.dim() == 4:
                imgs = images.unsqueeze(1)
            elif images.dim() == 5:
                imgs = images
            else:
                return pred
        if imgs.max() > 1.5:
            imgs = imgs.float() / 255.0
        img_feat = self.correction["image_encoder"](imgs)
        delta = self.correction["correction_head"](img_feat).view(B, self.n_nodes, 3)
        return pred + delta

    def forward_predictive(self, batch):
        return self._predict(batch["action_window"])

    def forward_corrective(self, batch):
        pred = self._predict(batch["action_window"])
        return self._correct(pred, batch.get("images"))

    def compute_losses(self, batch, phase_spec):
        losses = super().compute_losses(batch, phase_spec)
        active = set(phase_spec.active_losses)
        device = next(self.parameters()).device
        gt_skeleton = batch["gt_skeleton"].to(device)

        if phase_spec.name == "corrective" and batch.get("images") is not None:
            pred = self.forward_corrective(batch)
        else:
            pred = self._predict(batch["action_window"].to(device))

        if "skeleton" in active:
            losses["skeleton"] = F.mse_loss(pred, gt_skeleton)
        if "spatial_smooth" in active:
            pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
            gt_delta = gt_skeleton[:, 1:, :] - gt_skeleton[:, :-1, :]
            losses["spatial_smooth"] = F.mse_loss(pred_delta, gt_delta)
        return losses

    @torch.no_grad()
    def predict_skeleton(self, action_window, images=None):
        device = next(self.parameters()).device
        action_window = action_window.to(device)
        norm = self.action_norm_factor.item()
        if norm > 1.01:
            action_window = action_window / norm
        pred = self._predict(action_window)
        if images is not None:
            pred = self._correct(pred, images)
        return pred * self.pc_scale.to(device) + self.pc_center.to(device)

    @torch.no_grad()
    def predict_pointcloud(self, action_window, n_points=1000, avg_radius=0.015):
        skeleton = self.predict_skeleton(action_window)
        B = skeleton.shape[0]
        all_points = []
        for b in range(B):
            skel_np = skeleton[b].cpu().numpy().T
            pts, _ = _sample_surface(skel_np, avg_radius, n_points)
            if len(pts) < n_points:
                pad = np.tile(pts[-1:], (n_points - len(pts), 1))
                pts = np.concatenate([pts, pad], axis=0)
            all_points.append(torch.from_numpy(pts[:n_points]).float())
        return torch.stack(all_points, dim=0).to(skeleton.device)
