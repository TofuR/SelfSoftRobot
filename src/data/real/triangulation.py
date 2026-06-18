"""triangulation.py — 多视角 2D 骨架 → 3D 骨架（线性 DLT 三角化）。

输入每视角的 2D 骨架 (V,N,31,2) [col,row] + camera_params(V,10)，逐节点三角化出
世界系 3D 骨架 (N,31,3)。投影矩阵由 camera_params_format.projection_matrix 重建，
与 src/utils/skeleton_2d.py::project_3d_to_2d 投影约定完全一致。

2D 骨架由 src/utils/skeleton_2d.extract_skeleton_2d 提供（实物管线复用，零重复）。
"""

import numpy as np

from src.calibration.camera_params_format import projection_matrix


def _projection_matrices(camera_params, H, W):
    """camera_params(V,10) → list of P(3,4) world→pixel。"""
    cp = np.asarray(camera_params, float).reshape(-1, 10)
    return [projection_matrix(row, H, W) for row in cp]


def triangulate_point(pixels, Ps):
    """DLT 三角化单点。

    Args:
        pixels: list of (col, row)（≥2 个视角）。
        Ps: 对应 list of (3,4) 投影矩阵。

    Returns:
        (3,) 世界坐标；数值不稳定时返回 nan。
    """
    A = []
    for (col, row), P in zip(pixels, Ps):
        A.append(row * P[2] - P[1])
        A.append(P[0] - col * P[2])
    A = np.asarray(A, float)
    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.full(3, np.nan)
    return X[:3] / X[3]


def triangulate_skeletons(skeletons_2d, camera_params, H, W):
    """多视角 2D 骨架 → 3D 骨架。

    Args:
        skeletons_2d: (V, N, 31, 2) [col,row]，或 list[V] of (N,31,2)。
            全 0 的 2D 点（extract_skeleton_2d 无前景时的返回）视为无效。
        camera_params: (V, 10)。
        H, W: 图像尺寸（构 K 的主点 cx=W/2, cy=H/2）。

    Returns:
        skeletons_3d (N, 31, 3)：世界系米；无效节点为 nan。
    """
    sk = np.asarray(skeletons_2d, float)
    if sk.ndim != 4:
        sk = np.stack(sk, axis=0)             # (V,N,31,2)
    V, N, J, _ = sk.shape
    Ps = _projection_matrices(camera_params, H, W)

    out = np.full((N, J, 3), np.nan, float)
    for n in range(N):
        for j in range(J):
            pts, pPs = [], []
            for v in range(V):
                col, row = sk[v, n, j]
                if not np.isfinite(col) or (col == 0.0 and row == 0.0):
                    continue
                pts.append((col, row))
                pPs.append(Ps[v])
            if len(pts) >= 2:
                out[n, j] = triangulate_point(pts, pPs)
    return out
