"""MS-SCNF — Multi-Scale Skeleton-Conditioned Neural Field。

核心架构：
  Action Window → MultiScaleEMA → physics_state →
      ├── SkeletonHead → coarse(4×3) / medium(10×3) / fine(31×3)
      │                        ↓
      │               SkeletonConditionedDensity → [vis, density]

skeleton_mode 选择骨架参数化方式：
  "point"      — 原始方案，独立预测每个节点（93 参数）
  "fourier"    — 截断 Fourier 级数，天然光滑（51 参数 @ n_freq=8）
  "bspline"    — 三次 B-spline，局部控制 + C² 连续（30 参数 @ n_ctrl=10）
  "catmullrom" — Catmull-Rom 样条，插值型曲线通过控制点（30 参数 @ n_ctrl=10）

训练分两阶段：
  Phase 1: 仅 SkeletonHead（3D L2 loss，GT 来自仿真器）
  Phase 2: 联合 SkeletonHead + DensityField（3D + 2D rendering loss）
"""

import numpy as np
import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder
from .model_mstnf import MultiScaleEMA
from src.training.spec import PhaseSpec, TrainingSpec


# =============================================================================
# 共享 trunk 网络
# =============================================================================

class _Trunk(nn.Module):
    """共享特征提取 trunk。"""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# 骨架头: point（原始方案）
# =============================================================================

class SkeletonHead(nn.Module):
    """独立预测每个节点坐标（原始方案）。参数量: (4+10+31)*3 = 135"""

    def __init__(self, hidden_dim=128, n_coarse=4, n_medium=10, n_fine=31):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_medium = n_medium
        self.n_fine = n_fine
        self.mode = "point"

        self.trunk = _Trunk(hidden_dim)
        self.coarse_head = nn.Linear(256, n_coarse * 3)
        self.medium_head = nn.Linear(256, n_medium * 3)
        self.fine_head = nn.Linear(256, n_fine * 3)

    def forward(self, physics_state):
        feat = self.trunk(physics_state)
        B = physics_state.shape[0]
        return {
            'coarse': self.coarse_head(feat).reshape(B, self.n_coarse, 3),
            'medium': self.medium_head(feat).reshape(B, self.n_medium, 3),
            'fine': self.fine_head(feat).reshape(B, self.n_fine, 3),
        }


# =============================================================================
# 骨架头: fourier（截断 Fourier 级数）
# =============================================================================

class FourierSkeletonHead(nn.Module):
    """Fourier 级数骨架头：天然光滑，带限防止高频振荡。

    x(s) = a0 + Σ_k (a_k cos(2πks) + b_k sin(2πks))，s ∈ [0,1]
    参数量: 3*(1+2*n_freq)，n_freq=8 → 51
    """

    def __init__(self, hidden_dim=128, n_freq=8,
                 n_coarse=4, n_medium=10, n_fine=31):
        super().__init__()
        self.n_freq = n_freq
        self.n_coarse = n_coarse
        self.n_medium = n_medium
        self.n_fine = n_fine
        self.mode = "fourier"

        self.trunk = _Trunk(hidden_dim)
        self.head = nn.Linear(256, 3 * (1 + 2 * n_freq))
        self.register_buffer('eval_matrix', self._build_eval_matrix(n_fine, n_freq))

    @staticmethod
    def _build_eval_matrix(n_eval, n_freq):
        s = np.linspace(0, 1, n_eval)
        basis = [np.ones(n_eval)]
        for k in range(1, n_freq + 1):
            basis.append(np.cos(2 * np.pi * k * s))
            basis.append(np.sin(2 * np.pi * k * s))
        return torch.tensor(np.stack(basis, axis=-1), dtype=torch.float32)

    def forward(self, physics_state):
        feat = self.trunk(physics_state)
        B = physics_state.shape[0]
        coeffs = self.head(feat).reshape(B, 3, 1 + 2 * self.n_freq)
        fine = torch.matmul(coeffs, self.eval_matrix.T).transpose(1, 2)
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


# =============================================================================
# 骨架头: bspline（三次 B-spline）
# =============================================================================

class BSplineSkeletonHead(nn.Module):
    """三次 B-spline 骨架头：局部控制 + C² 连续。

    预测 n_ctrl 个控制点，B-spline 求值得到 n_fine 个骨架节点。
    参数量: n_ctrl*3，n_ctrl=10 → 30
    """

    def __init__(self, hidden_dim=128, n_ctrl=10, degree=3,
                 n_coarse=4, n_medium=10, n_fine=31):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.degree = degree
        self.n_coarse = n_coarse
        self.n_medium = n_medium
        self.n_fine = n_fine
        self.mode = "bspline"

        self.trunk = _Trunk(hidden_dim)
        self.head = nn.Linear(256, n_ctrl * 3)
        self.register_buffer('basis_matrix', self._build_basis(n_ctrl, degree, n_fine))

    @staticmethod
    def _build_basis(n_ctrl, degree, n_eval):
        from scipy.interpolate import BSpline as SciBSpline

        # clamped 节点向量
        n_knots = n_ctrl + degree + 1
        n_internal = n_ctrl - degree - 1
        if n_internal > 0:
            internal = np.linspace(0, 1, n_internal + 2)[1:-1]
        else:
            internal = np.array([])
        knots = np.concatenate([
            np.zeros(degree + 1), internal, np.ones(degree + 1)
        ])
        knots = knots[:n_knots]

        eval_pts = np.linspace(0, 1, n_eval)
        basis = np.zeros((n_eval, n_ctrl))
        for j in range(n_ctrl):
            c = np.zeros(n_ctrl)
            c[j] = 1.0
            spl = SciBSpline(knots, c, degree, extrapolate=False)
            for i, t in enumerate(eval_pts):
                basis[i, j] = spl(t) if t <= 1.0 else 0.0
        return torch.tensor(basis, dtype=torch.float32)

    def forward(self, physics_state):
        feat = self.trunk(physics_state)
        B = physics_state.shape[0]
        ctrl = self.head(feat).reshape(B, self.n_ctrl, 3)
        fine = torch.matmul(
            self.basis_matrix.unsqueeze(0).expand(B, -1, -1), ctrl
        )
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


# =============================================================================
# 骨架头: catmullrom（Catmull-Rom 样条）
# =============================================================================

class CatmullRomSkeletonHead(nn.Module):
    """Catmull-Rom 样条骨架头：插值型，曲线精确通过控制点。

    参数量: n_ctrl*3，n_ctrl=10 → 30
    """

    def __init__(self, hidden_dim=128, n_ctrl=10,
                 n_coarse=4, n_medium=10, n_fine=31):
        super().__init__()
        self.n_ctrl = n_ctrl
        self.n_coarse = n_coarse
        self.n_medium = n_medium
        self.n_fine = n_fine
        self.mode = "catmullrom"

        self.trunk = _Trunk(hidden_dim)
        self.head = nn.Linear(256, n_ctrl * 3)
        self.register_buffer('eval_matrix', self._build_eval(n_ctrl, n_fine))

    @staticmethod
    def _build_eval(n_ctrl, n_eval):
        n_segs = n_ctrl - 1
        n_per_seg = n_eval // n_segs
        remainder = n_eval - n_per_seg * n_segs

        rows = []
        for seg in range(n_segs):
            n_pts = n_per_seg + (remainder if seg == n_segs - 1 else 0)
            i0, i1, i2, i3 = max(seg - 1, 0), seg, seg + 1, min(seg + 2, n_ctrl - 1)
            for j in range(n_pts):
                t = j / max(n_pts - 1, 1)
                t2, t3 = t * t, t * t * t
                w0 = -0.5 * t3 + t2 - 0.5 * t
                w1 = 1.5 * t3 - 2.5 * t2 + 1.0
                w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                w3 = 0.5 * t3 - 0.5 * t2
                row = np.zeros(n_ctrl)
                row[i0] += w0; row[i1] += w1; row[i2] += w2; row[i3] += w3
                rows.append(row)

        mat = np.array(rows[:n_eval])
        return torch.tensor(mat, dtype=torch.float32)

    def forward(self, physics_state):
        feat = self.trunk(physics_state)
        B = physics_state.shape[0]
        ctrl = self.head(feat).reshape(B, self.n_ctrl, 3)
        fine = torch.matmul(
            self.eval_matrix.unsqueeze(0).expand(B, -1, -1), ctrl
        )
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


# =============================================================================
# 骨架头工厂
# =============================================================================

SKELETON_MODES = {
    "point": SkeletonHead,
    "fourier": FourierSkeletonHead,
    "bspline": BSplineSkeletonHead,
    "catmullrom": CatmullRomSkeletonHead,
}


def downsample_skeleton(skeleton, n_target):
    """将细尺度骨架均匀下采样到 n_target 个节点。

    始终包含首尾节点。

    Args:
        skeleton: (..., N, 3) 骨架坐标。
        n_target: 目标节点数。

    Returns:
        (..., n_target, 3) 下采样后的骨架。
    """
    N = skeleton.shape[-2]
    if n_target >= N:
        return skeleton
    indices = torch.linspace(0, N - 1, n_target, device=skeleton.device).long()
    return skeleton[..., indices, :]


def point_to_segment_distance(points, seg_start, seg_end):
    """计算点到线段的最短距离（可微）。

    Args:
        points: (..., M, 3) 查询点。
        seg_start: (..., S, 3) 线段起点。
        seg_end: (..., S, 3) 线段终点。

    Returns:
        (..., M) 每个查询点到最近线段的距离。
    """
    # seg_vec: (..., S, 3)
    seg_vec = seg_end - seg_start
    seg_len_sq = (seg_vec ** 2).sum(-1, keepdim=True).clamp(min=1e-8)

    # points: (..., M, 1, 3), seg_start: (..., 1, S, 3)
    diff = points.unsqueeze(-2) - seg_start.unsqueeze(-3)
    t = (diff * seg_vec.unsqueeze(-3)).sum(-1, keepdim=True) / seg_len_sq.unsqueeze(-3)
    t = t.clamp(0, 1)

    # 投影点
    projection = seg_start.unsqueeze(-3) + t * seg_vec.unsqueeze(-3)
    dist = ((points.unsqueeze(-2) - projection) ** 2).sum(-1)
    # dist: (..., M, S)，取最近线段
    return dist.min(-1)[0].sqrt()


class SkeletonConditionedDensity(nn.Module):
    """骨架条件密度场：查询点密度取决于到骨架曲线的距离。

    距骨架越近密度越高，远离骨架处密度趋零。
    """

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
        """根据骨架计算查询点的 [visibility, density]。

        Args:
            query_points: (N, n_samples, 3) 3D 查询点。
            skeleton: (B, N_nodes, 3) 骨架节点。

        Returns:
            output: (B*N, n_samples, 2)
        """
        N, n_samples, _ = query_points.shape
        B = skeleton.shape[0]
        n_seg = skeleton.shape[1] - 1

        # 扩展到 batch: pts (B*N, n_samples, 3), seg (B*N, n_seg, 3)
        pts = query_points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N, n_samples, 3)
        skel_expanded = skeleton.unsqueeze(1).expand(-1, N, -1, -1).reshape(B * N, -1, 3)
        seg_start = skel_expanded[:, :-1, :]
        seg_end = skel_expanded[:, 1:, :]

        # point_to_segment_distance: (B*N, n_samples, 3) vs (B*N, n_seg, 3) → (B*N, n_samples)
        dist = point_to_segment_distance(pts, seg_start, seg_end)
        dist = dist.unsqueeze(-1)  # (B*N, n_samples, 1)

        dist_enc = self.dist_encoder(dist)
        pos_enc = self.pos_encoder(pts)
        latent = torch.cat([dist_enc, pos_enc], dim=-1)

        return self.decoder(latent)


class MSSCNFModel(nn.Module):
    """MS-SCNF 完整模型。

    skeleton_mode 选择骨架参数化：
      "point"      — 独立节点（默认）
      "fourier"    — Fourier 级数
      "bspline"    — B-spline
      "catmullrom" — Catmull-Rom 样条
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
            self.skeleton_head = FourierSkeletonHead(**skel_kwargs)
        elif skeleton_mode == "bspline":
            skel_kwargs["n_ctrl"] = bspline_n_ctrl
            self.skeleton_head = BSplineSkeletonHead(**skel_kwargs)
        elif skeleton_mode == "catmullrom":
            skel_kwargs["n_ctrl"] = catmullrom_n_ctrl
            self.skeleton_head = CatmullRomSkeletonHead(**skel_kwargs)
        else:
            self.skeleton_head = SkeletonHead(**skel_kwargs)

        self.density = SkeletonConditionedDensity(
            n_freqs=deform_n_freqs,
            d_filter=d_filter,
        )

        self.canonical = nn.Module()  # placeholder for interface compatibility

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
        """动作窗口 → 物理状态。"""
        return self.temporal(action_window)

    def predict_skeleton(self, action_window):
        """从动作窗口预测多尺度骨架（部署用）。

        Args:
            action_window: (B, K, D)

        Returns:
            dict with 'coarse', 'medium', 'fine', each (B, N, 3)。
        """
        state = self.encode(action_window)
        return self.skeleton_head(state)

    def forward(self, points, action_window):
        """训练用前向传播：骨架条件密度场。

        Args:
            points: (N_rays, n_samples, 3) 空间查询点。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            output: (B*N_rays, n_samples, 2) [vis, density]
        """
        state = self.encode(action_window)
        skel_dict = self.skeleton_head(state)
        skeleton = skel_dict['fine']
        return self.density(points, skeleton)

    def forward_canonical(self, points):
        """Phase 1 placeholder（MS-SCNF 的 Phase 1 是骨架回归，不需要此方法）。"""
        B = 1
        N, n_samples, _ = points.shape
        device = points.device
        return torch.zeros(N, n_samples, 2, device=device)

    def compute_skeleton_loss(self, pred_dict, gt_positions):
        """多尺度骨架 L2 loss。

        Args:
            pred_dict: predict_skeleton() 输出的 dict。
            gt_positions: (B, N_full, 3) GT 骨架坐标。

        Returns:
            dict of losses: 'fine', 'medium', 'coarse'。
        """
        losses = {}
        losses['fine'] = ((pred_dict['fine'] - gt_positions) ** 2).mean()

        gt_medium = downsample_skeleton(gt_positions, pred_dict['medium'].shape[-2])
        losses['medium'] = ((pred_dict['medium'] - gt_medium) ** 2).mean()

        gt_coarse = downsample_skeleton(gt_positions, pred_dict['coarse'].shape[-2])
        losses['coarse'] = ((pred_dict['coarse'] - gt_coarse) ** 2).mean()

        return losses

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        """时序平滑 loss。"""
        return self.temporal.compute_smoothness(action_windows_t, action_windows_t1)

    def freeze_canonical(self):
        """接口兼容 no-op。"""
        pass

    def get_learned_decays(self):
        """返回 EMA 衰减率。"""
        return self.temporal.decays.detach().cpu().numpy()
