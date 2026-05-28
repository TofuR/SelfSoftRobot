"""骨架曲线参数化模块 — 可复用的骨架回归头。

提供 4 种骨架参数化方式，输出统一接口：
  forward(physics_state) -> dict('fine', 'medium', 'coarse') 各为 (B, N, 3)

可供 MSSCNFModel、SkeletonSDFModel 等模型复用。
"""

import numpy as np
import torch
import torch.nn as nn


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


def downsample_skeleton(skeleton, n_target):
    """将细尺度骨架均匀下采样到 n_target 个节点。"""
    N = skeleton.shape[-2]
    if n_target >= N:
        return skeleton
    indices = torch.linspace(0, N - 1, n_target, device=skeleton.device).long()
    return skeleton[..., indices, :]


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

    def fit_to_points(self, gt_points):
        """GT 坐标不需要参数化，直接返回多尺度。"""
        return {
            'fine': gt_points,
            'medium': downsample_skeleton(gt_points, self.n_medium),
            'coarse': downsample_skeleton(gt_points, self.n_coarse),
        }


class FourierSkeletonHead(nn.Module):
    """Fourier 级数骨架头：天然光滑，带限防止高频振荡。

    x(s) = a0 + sum_k (a_k cos(2*pi*k*s) + b_k sin(2*pi*k*s))，s in [0,1]
    参数量: 3*(1+2*n_freq)，n_freq=8 -> 51
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

    def fit_to_points(self, gt_points):
        """GT → Fourier 系数（伪逆拟合）→ 重建带限光滑骨架。

        forward: fine = (coeffs @ eval_matrix.T).T，即 fine = eval_matrix @ coeffs.T
        逆: coeffs.T = pinv(eval_matrix) @ fine
        重建: fine_recon = eval_matrix @ pinv(eval_matrix) @ gt（最小二乘投影）
        """
        if not hasattr(self, '_proj_matrix'):
            pinv = torch.linalg.pinv(self.eval_matrix)  # (n_basis, n_fine)
            self._proj_matrix = self.eval_matrix @ pinv  # (n_fine, n_fine)
        B = gt_points.shape[0]
        fine = torch.matmul(
            self._proj_matrix.unsqueeze(0).expand(B, -1, -1), gt_points)
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


class BSplineSkeletonHead(nn.Module):
    """三次 B-spline 骨架头：局部控制 + C2 连续。

    预测 n_ctrl 个控制点，B-spline 求值得到 n_fine 个骨架节点。
    参数量: n_ctrl*3，n_ctrl=10 -> 30
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

    def fit_to_points(self, gt_points):
        """GT → B-spline 控制点（伪逆拟合）→ 重建光滑骨架。"""
        if not hasattr(self, '_proj_matrix'):
            pinv = torch.linalg.pinv(self.basis_matrix)  # (n_ctrl, n_fine)
            self._proj_matrix = self.basis_matrix @ pinv  # (n_fine, n_fine)
        B = gt_points.shape[0]
        fine = torch.matmul(
            self._proj_matrix.unsqueeze(0).expand(B, -1, -1), gt_points)
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


class CatmullRomSkeletonHead(nn.Module):
    """Catmull-Rom 样条骨架头：插值型，曲线精确通过控制点。

    参数量: n_ctrl*3，n_ctrl=10 -> 30
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

    def fit_to_points(self, gt_points):
        """GT → CatmullRom 控制点（伪逆拟合）→ 重建插值光滑骨架。"""
        if not hasattr(self, '_proj_matrix'):
            pinv = torch.linalg.pinv(self.eval_matrix)  # (n_ctrl, n_fine)
            self._proj_matrix = self.eval_matrix @ pinv  # (n_fine, n_fine)
        B = gt_points.shape[0]
        fine = torch.matmul(
            self._proj_matrix.unsqueeze(0).expand(B, -1, -1), gt_points)
        return {
            'fine': fine,
            'medium': downsample_skeleton(fine, self.n_medium),
            'coarse': downsample_skeleton(fine, self.n_coarse),
        }


SKELETON_MODES = {
    "point": SkeletonHead,
    "fourier": FourierSkeletonHead,
    "bspline": BSplineSkeletonHead,
    "catmullrom": CatmullRomSkeletonHead,
}


def create_skeleton_head(mode="point", **kwargs):
    """骨架头工厂函数。"""
    if mode not in SKELETON_MODES:
        raise ValueError(
            f"Unknown skeleton_mode '{mode}', choose from {list(SKELETON_MODES)}")
    return SKELETON_MODES[mode](**kwargs)


def point_to_segment_distance(points, seg_start, seg_end):
    """计算点到线段的最短距离（可微）。

    Args:
        points: (..., M, 3) 查询点。
        seg_start: (..., S, 3) 线段起点。
        seg_end: (..., S, 3) 线段终点。

    Returns:
        (..., M) 每个查询点到最近线段的距离。
    """
    seg_vec = seg_end - seg_start
    seg_len_sq = (seg_vec ** 2).sum(-1, keepdim=True).clamp(min=1e-8)

    diff = points.unsqueeze(-2) - seg_start.unsqueeze(-3)
    t = (diff * seg_vec.unsqueeze(-3)).sum(-1, keepdim=True) / seg_len_sq.unsqueeze(-3)
    t = t.clamp(0, 1)

    projection = seg_start.unsqueeze(-3) + t * seg_vec.unsqueeze(-3)
    dist = ((points.unsqueeze(-2) - projection) ** 2).sum(-1)
    return dist.min(-1)[0].sqrt()


def point_to_skeleton_coords(points, seg_start, seg_end):
    """计算查询点相对于骨架曲线的局部柱坐标（可微）。

    在最近骨架线段上建立局部参考系，返回径向距离、轴向参数和环向角度。

    Args:
        points: (..., M, 3) 查询点。
        seg_start: (..., S, 3) 线段起点。
        seg_end: (..., S, 3) 线段终点。

    Returns:
        (dist, t_axial, theta):
          dist:    (..., M) 到最近线段的径向距离。
          t_axial: (..., M) 归一化轴向参数 in [0, 1]。
          theta:   (..., M) 环向角度 in [-pi, pi]。
    """
    n_seg = seg_start.shape[-2]
    seg_vec = seg_end - seg_start                          # (..., S, 3)
    seg_len_sq = (seg_vec ** 2).sum(-1, keepdim=True).clamp(min=1e-8)

    diff = points.unsqueeze(-2) - seg_start.unsqueeze(-3)  # (..., M, S, 3)
    t_local = (diff * seg_vec.unsqueeze(-3)).sum(-1, keepdim=True) \
              / seg_len_sq.unsqueeze(-3)                    # (..., M, S, 1)
    t_local = t_local.clamp(0, 1)

    projection = seg_start.unsqueeze(-3) + t_local * seg_vec.unsqueeze(-3)
    dist_sq = ((points.unsqueeze(-2) - projection) ** 2).sum(-1)  # (..., M, S)

    # 最近线段
    closest_idx = dist_sq.argmin(dim=-1)                    # (..., M)
    min_dist_sq = dist_sq.gather(-1, closest_idx.unsqueeze(-1)).squeeze(-1)
    dist = min_dist_sq.sqrt()

    # 轴向参数: (seg_idx + t_local) / n_seg
    t_local_gathered = t_local.squeeze(-1).gather(-1, closest_idx.unsqueeze(-1)).squeeze(-1)
    t_axial = (closest_idx.float() + t_local_gathered) / n_seg
    t_axial = t_axial.clamp(0.0, 1.0)

    # 环向角度
    idx_for_proj = closest_idx.unsqueeze(-1).unsqueeze(-1).expand(
        *closest_idx.shape, 1, 3)                           # (..., M, 1, 3)
    proj_gathered = projection.gather(-2, idx_for_proj).squeeze(-2)  # (..., M, 3)

    tangent = seg_vec / seg_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (..., S, 3)
    idx_for_tangent = closest_idx.unsqueeze(-1).expand(
        *closest_idx.shape, 3)                              # (..., M, 3)
    tangent_gathered = tangent.gather(-2, idx_for_tangent)  # (..., M, 3)

    radial = points - proj_gathered                         # (..., M, 3)
    # 投影到切平面
    radial_proj = radial - (radial * tangent_gathered).sum(-1, keepdim=True) * tangent_gathered

    # 参考方向: cross(tangent, z_hat)，退化时用 y_hat
    z_hat = torch.zeros_like(tangent_gathered)
    z_hat[..., 2] = 1.0
    ref = torch.cross(tangent_gathered, z_hat, dim=-1)
    degenerate = ref.norm(dim=-1, keepdim=True) < 1e-6
    y_hat = torch.zeros_like(tangent_gathered)
    y_hat[..., 1] = 1.0
    ref_alt = torch.cross(tangent_gathered, y_hat, dim=-1)
    ref = torch.where(degenerate, ref_alt, ref)
    ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    theta = torch.atan2(
        (radial_proj * torch.cross(tangent_gathered, ref, dim=-1)).sum(-1),
        (radial_proj * ref).sum(-1))

    return dist, t_axial, theta
