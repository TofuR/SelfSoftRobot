"""SpatialSequenceModel — 基于 GRU 空间序列的中心线预测模型。

架构:
  Action Window → FractionalMemory → physics_state (全局条件 c)
                                       |
  Z 位置嵌入 ─→ GRU(z₀→z_K) ─→ 每节点 xyz 预测
                                       |
                                 与 GT 中心线对比 (skeleton MSE + spatial_smooth)

关键设计:
  - GRU 沿 Z 轴自下而上传播空间状态（悬臂梁因果性）
  - 底部节点由 action 条件决定初始状态，尖端节点由递推产生
  - 扇形问题从架构层面消失：每个节点有明确的 z 位置先验

训练:
  - L_skeleton: 预测中心线与 GT 的 MSE
  - L_spatial_smooth: 相邻节点位移连续性约束
  - L_smooth: 时序平滑（TemporalMixin 提供）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins import TemporalMixin
from src.encoders.fractional_memory import FractionalMemory
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.encoders.gamma_laguerre import GammaLaguerreMemory
from src.encoders.temporal_gru import TemporalGRU
from src.encoders.temporal_transformer import TemporalTransformer
from src.encoders.temporal_tcn import TemporalTCN
from src.training.spec import TrainingSpec, PhaseSpec
from src.data.dataset_pointcloud import _sample_surface


class SpatialSequenceModel(nn.Module, TemporalMixin):
    """空间序列生成模型：GRU 沿 Z 轴传播，预测中心线节点坐标。

    继承 TemporalMixin 获得 encode() 和 compute_smoothness()。

    Args:
        action_dim: 驱动维度。
        n_nodes: 中心线节点数（与 GT positions 的 N 一致）。
        hidden_dim: 隐层维度。
        window_size: 时序窗口长度。
        n_orders: FractionalMemory 的分数阶个数。
        encoder_type: "fractional" 或 "ema"。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                "spatial",
                dataset_type="spatial_sequence",
                supervision_mode="spatial_sequence",
                active_losses=["skeleton", "spatial_smooth", "smooth"],
                forward_attr="forward",
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
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type

        # 点云归一化参数（由 set_normalization 设置）
        self.register_buffer('pc_center', torch.zeros(1, 1, 3))
        self.register_buffer('pc_scale', torch.ones(1, 1, 3))
        self.register_buffer('action_norm_factor', torch.tensor(1.0))

        # 时序编码器（TemporalMixin 依赖 self.temporal）
        _ENCODERS = {
            "ema": MultiScaleEMA,
            "fractional": FractionalMemory,
            "gamma": GammaLaguerreMemory,
            "gru": TemporalGRU,
            "transformer": TemporalTransformer,
            "tcn": TemporalTCN,
        }
        EncoderClass = _ENCODERS.get(encoder_type, FractionalMemory)
        self.temporal = EncoderClass(
            action_dim=action_dim,
            n_scales=n_orders,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # Z 位置嵌入：将 z 坐标映射到 hidden_dim 维空间
        self.z_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU：沿 Z 轴的空间状态传播
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 每节点的 xyz 预测头
        self.slice_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        # 初始隐藏状态：从 action 条件生成
        self.init_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        """设置归一化参数（从数据集获取）。"""
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center).float()
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        self.pc_center = center.view(1, 1, 3)
        self.pc_scale = scale.view(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    def _get_z_positions(self, device):
        """获取归一化后的 z 位置序列 [-1, 1]。"""
        return torch.linspace(-1, 1, self.n_nodes, device=device)

    def forward(self, batch_or_action_window):
        """预测中心线节点坐标。

        Args:
            batch_or_action_window: (B, K, D) action_window 张量，
                                    或包含 "action_window" 键的 dict batch。

        Returns:
            skeleton_pred: (B, n_nodes, 3) 归一化空间。
        """
        if isinstance(batch_or_action_window, dict):
            action_window = batch_or_action_window["action_window"]
        else:
            action_window = batch_or_action_window

        device = action_window.device
        B = action_window.shape[0]

        cond = self.encode(action_window)  # (B, hidden_dim)
        h = self.init_hidden(cond)  # (B, hidden_dim)

        z_positions = self._get_z_positions(device)
        skeleton = []

        for i in range(self.n_nodes):
            z_emb = self.z_embed(
                z_positions[i:i + 1].unsqueeze(0).expand(B, -1))  # (B, hidden_dim)
            gru_input = cond + z_emb
            h = self.gru(gru_input, h)
            skeleton.append(self.slice_head(h))

        return torch.stack(skeleton, dim=1)  # (B, n_nodes, 3)

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """计算训练损失。"""
        losses = super().compute_losses(batch, phase_spec)
        active = set(phase_spec.active_losses)

        device = next(self.parameters()).device
        action_window = batch["action_window"].to(device)
        gt_skeleton = batch["gt_skeleton"].to(device)

        pred = self.forward(action_window)

        if "skeleton" in active:
            losses["skeleton"] = F.mse_loss(pred, gt_skeleton)

        if "spatial_smooth" in active:
            pred_delta = pred[:, 1:, :] - pred[:, :-1, :]
            gt_delta = gt_skeleton[:, 1:, :] - gt_skeleton[:, :-1, :]
            losses["spatial_smooth"] = F.mse_loss(pred_delta, gt_delta)

        return losses

    @torch.no_grad()
    def predict_skeleton(self, action_window):
        """推理：预测中心线（物理坐标）。"""
        device = next(self.parameters()).device
        action_window = action_window.to(device)
        norm = self.action_norm_factor.item()
        if norm > 1.01:
            action_window = action_window / norm

        pred = self.forward(action_window)
        return pred * self.pc_scale.to(device) + self.pc_center.to(device)

    @torch.no_grad()
    def predict_pointcloud(self, action_window, n_points=1000, avg_radius=0.015):
        """从预测中心线采样表面点（兼容评估流程）。"""
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
