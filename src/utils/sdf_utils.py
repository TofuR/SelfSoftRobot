"""GT SDF 生成工具 — 从骨架 + 半径解析计算管状 SDF。

对于管状结构（中心线 + 常数半径），SDF 可以精确计算：
  GT SDF(x) = dist_to_skeleton(x) - radius

不需要 directors（SDF 对截面旋转不变）。
"""

import torch
import numpy as np
from src.heads.skeleton_heads import point_to_segment_distance


def compute_gt_sdf(query_points, skeleton, radius):
    """解析计算管状 SDF。

    Args:
        query_points: (M, 3) 查询点。
        skeleton: (B, N, 3) 或 (N, 3) 骨架节点。
        radius: float, 管半径。

    Returns:
        sdf: (M,) 或 (B, M) SDF 值。SDF=0 在表面，<0 在管内，>0 在管外。
    """
    squeeze = False
    if skeleton.dim() == 2:
        skeleton = skeleton.unsqueeze(0)
        squeeze = True

    B, N, _ = skeleton.shape
    M = query_points.shape[0]

    pts = query_points.unsqueeze(0).expand(B, -1, -1)
    seg_start = skeleton[:, :-1, :]
    seg_end = skeleton[:, 1:, :]

    dist = point_to_segment_distance(pts, seg_start, seg_end)
    sdf = dist - radius

    return sdf.squeeze(0) if squeeze else sdf


def sample_sdf_training_data(positions, radius, n_surface=1000, n_near=1000,
                             n_off=1000, near_sigma=0.02, bbox_margin=0.05):
    """为 SDF 训练采样查询点并计算 GT SDF。

    Args:
        positions: (N, 3) 骨架节点坐标。
        radius: float 管半径。
        n_surface: 表面采样点数。
        n_near: 近表面采样点数。
        n_off: 远离表面采样点数。
        near_sigma: 近表面扰动标准差。
        bbox_margin: bounding box 外扩余量。

    Returns:
        dict with keys:
            'query_points': (n_total, 3) float32
            'gt_sdf': (n_total,) float32
            'gt_normals': (n_total, 3) float32（仅表面点有值）
            'n_surface': int
    """
    positions = np.asarray(positions, dtype=np.float32)
    radius = float(radius)
    mins = positions.min(axis=0) - bbox_margin
    maxs = positions.max(axis=0) + bbox_margin

    query_list = []
    sdf_list = []
    normal_list = []

    # 1. 表面点：沿骨架每段法平面圆采样
    n_segs = len(positions) - 1
    n_per_seg = max(1, n_surface // max(n_segs, 1))

    surface_pts_list = []
    surface_normals_list = []

    for i in range(n_segs):
        p1, p2 = positions[i], positions[i + 1]
        seg_dir = p2 - p1
        seg_len = np.linalg.norm(seg_dir)
        if seg_len < 1e-8:
            continue
        tangent = seg_dir / seg_len

        if abs(tangent[2]) < 0.99:
            perp1 = np.cross(tangent, np.array([0, 0, 1.0]))
        else:
            perp1 = np.cross(tangent, np.array([1.0, 0, 0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)

        center = (p1 + p2) / 2
        angles = np.linspace(0, 2 * np.pi, n_per_seg, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)
        pts_seg = center[np.newaxis, :] + radius * (
            cos_a[:, np.newaxis] * perp1[np.newaxis, :] +
            sin_a[:, np.newaxis] * perp2[np.newaxis, :]
        )
        normals_seg = (cos_a[:, np.newaxis] * perp1[np.newaxis, :] +
                       sin_a[:, np.newaxis] * perp2[np.newaxis, :])
        surface_pts_list.append(pts_seg)
        surface_normals_list.append(normals_seg)

    surface_pts = np.concatenate(surface_pts_list, axis=0) if surface_pts_list else np.zeros((0, 3))
    surface_normals = np.concatenate(surface_normals_list, axis=0) if surface_normals_list else np.zeros((0, 3))
    n_surf = len(surface_pts)

    # 2. 近表面点
    near_pts = surface_pts + np.random.randn(*surface_pts.shape).astype(np.float32) * near_sigma

    positions_t = torch.from_numpy(positions)
    near_pts_t = torch.from_numpy(near_pts)
    near_sdf = compute_gt_sdf(near_pts_t, positions_t, radius).numpy()

    # 3. 远离表面点
    off_pts = np.random.uniform(mins, maxs, size=(n_off, 3)).astype(np.float32)
    off_pts_t = torch.from_numpy(off_pts)
    off_sdf = compute_gt_sdf(off_pts_t, positions_t, radius).numpy()

    all_queries = np.concatenate([surface_pts, near_pts, off_pts], axis=0)
    all_sdf = np.concatenate([
        np.zeros(n_surf, dtype=np.float32),
        near_sdf,
        off_sdf,
    ])
    all_normals = np.zeros_like(all_queries)
    all_normals[:n_surf] = surface_normals

    return {
        'query_points': all_queries,
        'gt_sdf': all_sdf,
        'gt_normals': all_normals,
        'n_surface': n_surf,
    }
