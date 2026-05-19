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
import torch.nn.functional as F

from .layers import SirenLayer, PositionalEncoder
from .mixins import TemporalMixin, SkeletonMixin
from .model_mstnf import MultiScaleEMA
from .skeleton_heads import (
    create_skeleton_head, downsample_skeleton, point_to_segment_distance,
)
from src.training.spec import TrainingSpec, PhaseSpec


class SkeletonSDFModel(nn.Module, TemporalMixin, SkeletonMixin):
    """参数化骨架 + 截面 SDF 模型。

    Args:
        action_dim: 驱动维度。
        skeleton_mode: 骨架参数化方式（point/fourier/bspline/catmullrom）。
        rod_radius: 软臂半径（用于管状 SDF 先验）。
        sdf_residual: 是否使用 SIREN 残差修正 SDF 先验。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec("skeleton", dataset_type="skeleton_sdf",
                      supervision_mode="direct_3d",
                      freeze_modules=["sdf_net", "state_proj"],
                      active_losses=["skeleton"],
                      save_modules=["temporal", "skeleton_head"]),
            PhaseSpec("joint", dataset_type="skeleton_sdf",
                      supervision_mode="direct_3d",
                      active_losses=["skeleton", "sdf", "normal", "eikonal"],
                      load_modules={"temporal": "skeleton", "skeleton_head": "skeleton"}),
        ],
    )

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
            self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=4, log_space=True)
            pos_enc_dim = self.pos_encoder.d_output  # 27
            self.state_proj = nn.Linear(hidden_dim, 32)
            input_dim = 1 + pos_enc_dim + 32  # 60
            self.sdf_net = nn.Sequential(
                SirenLayer(input_dim, sdf_hidden, w0=30, is_first=True),
                SirenLayer(sdf_hidden, sdf_hidden, w0=1),
                SirenLayer(sdf_hidden, sdf_hidden, w0=1),
                SirenLayer(sdf_hidden, 1, is_last=True),
            )

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
        pos_enc = self.pos_encoder(pts)

        sdf_in = torch.cat([
            sdf_prior.unsqueeze(-1),
            pos_enc,
            state_feat.unsqueeze(1).expand(-1, N * n_samples, -1)
                .reshape(B * N, n_samples, 32),
        ], dim=-1)

        residual = self.sdf_net(sdf_in)
        return sdf_prior.unsqueeze(-1) + residual

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """计算骨架 + SDF 相关 loss。

        根据 phase_spec.active_losses 决定计算哪些 loss：
          "skeleton" — 多尺度骨架 L2 + 二阶差分平滑
          "sdf"      — SDF L1 回归
          "normal"   — 表面法向量余弦相似度
          "eikonal"  — SDF 梯度范数约束
          "smooth"   — 时序平滑（委托给 TemporalMixin）

        Args:
            batch: dict，包含:
                "action_window"     (B, K, D)  动作序列
                "coords"            (1, N, 3)  空间查询坐标（SDF 阶段）
                "gt_sdf"            (1, N)      GT SDF 值
                "gt_normals"        (1, N, 3)  GT 法向量
                "gt_positions"      (B, n_fine, 3)  GT 骨架位置
            phase_spec: 当前 PhaseSpec。

        Returns:
            dict[str, torch.Tensor]: loss 名到标量 Tensor 的映射。
        """
        # 先让 TemporalMixin 处理 "smooth" 等
        losses = super().compute_losses(batch, phase_spec)
        active = set(phase_spec.active_losses)

        device = next(self.parameters()).device
        action_window = batch["action_window"].to(device)

        # ── 骨架 loss ──
        if "skeleton" in active:
            gt_positions = batch["gt_positions"].to(device)
            # gt_positions: (B, n_fine, 3)，squeeze(0) 用于 B=1 情况已在 trainer 里做，
            # 但这里保持通用：直接传给 compute_skeleton_loss
            if gt_positions.dim() == 3 and gt_positions.shape[0] == 1:
                gt_pos = gt_positions  # keep B dim for compute_skeleton_loss
            else:
                gt_pos = gt_positions

            pred_dict = self.predict_skeleton(action_window)
            skel_losses = self.compute_skeleton_loss(pred_dict, gt_pos)

            # loss 权重（从 config 属性或默认值）
            w_fine = getattr(self, "w_skel_fine", 1.0)
            w_medium = getattr(self, "w_skel_medium", 0.3)
            w_coarse = getattr(self, "w_skel_coarse", 0.1)
            w_smooth = getattr(self, "w_skel_smooth", 0.01)

            losses["skeleton"] = (
                skel_losses["fine"] * w_fine
                + skel_losses["medium"] * w_medium
                + skel_losses["coarse"] * w_coarse
            )

            # 骨架二阶差分平滑
            skel_fine = pred_dict["fine"]  # (B, n_fine, 3)
            if skel_fine.shape[1] >= 3:
                second_order = (
                    skel_fine[:, 2:] - 2 * skel_fine[:, 1:-1] + skel_fine[:, :-2]
                )
                losses["skel_smooth"] = (second_order ** 2).mean() * w_smooth
            else:
                losses["skel_smooth"] = torch.tensor(0.0, device=device)

        # ── SDF / Normal / Eikonal losses ──
        has_sdf_losses = active & {"sdf", "normal", "eikonal"}
        if has_sdf_losses:
            coords = batch["coords"].to(device)
            if coords.dim() == 3 and coords.shape[0] == 1:
                coords = coords.squeeze(0)  # (N, 3)
            coords = coords.requires_grad_(True)

            gt_sdf = batch["gt_sdf"].to(device)
            if gt_sdf.dim() == 2 and gt_sdf.shape[0] == 1:
                gt_sdf = gt_sdf.squeeze(0)  # (N,)

            gt_normals = batch["gt_normals"].to(device)
            if gt_normals.dim() == 3 and gt_normals.shape[0] == 1:
                gt_normals = gt_normals.squeeze(0)  # (N, 3)

            # forward 接受 (N, n_samples, 3)，coords.squeeze(0) 后为 (N, 3)
            # unsqueeze(0) -> (1, N, 3) 会被 forward 理解为 N_q=1, n_samples=N, 3
            # 这与 trainer 中 query = coords.unsqueeze(0) 一致
            query = coords.unsqueeze(0)
            pred_sdf = self(query, action_window).squeeze(-1)  # (B, N)

            # SDF L1 回归
            if "sdf" in active:
                w_sdf = getattr(self, "w_sdf", 3e3)
                losses["sdf"] = torch.abs(pred_sdf - gt_sdf).mean() * w_sdf

            # 梯度（共享计算图，用于 normal 和 eikonal）
            gradient = torch.autograd.grad(
                pred_sdf.sum(), coords, create_graph=True,
            )[0]

            # Normal loss: 表面点的梯度方向 vs GT 法向量
            if "normal" in active:
                w_normal = getattr(self, "w_normal", 10.0)
                is_surface = (gt_sdf.abs() < 1e-6).float()
                n_surf = is_surface.sum()
                if n_surf > 0 and gt_normals.abs().sum() > 0:
                    cos_sim = F.cosine_similarity(gradient, gt_normals, dim=-1)
                    losses["normal"] = (
                        (is_surface * (1 - cos_sim)).sum() / (n_surf + 1e-8)
                        * w_normal
                    )
                else:
                    losses["normal"] = torch.tensor(0.0, device=device)

            # Eikonal loss: 梯度范数应为 1
            if "eikonal" in active:
                w_eikonal = getattr(self, "w_eikonal", 50.0)
                losses["eikonal"] = (
                    ((gradient.norm(dim=-1) - 1) ** 2).mean() * w_eikonal
                )

        return losses
