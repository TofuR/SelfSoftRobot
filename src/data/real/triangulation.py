"""triangulation.py — 多视角 2D 骨架 → 3D 骨架（线性 DLT 三角化）。

输入每视角的 2D 骨架 (V,N,31,2) [col,row] + camera_params(V,10)，逐节点三角化出
世界系 3D 骨架 (N,31,3)。投影矩阵由 camera_params_format.projection_matrix 重建，
与 src/utils/skeleton_2d.py::project_3d_to_2d 投影约定完全一致（该投影函数留在 src/）。

2D 骨架由 real_validation/perception/skeleton.extract_skeleton_2d 提供
（唯一实现，src/utils/skeleton_2d.py 为其薄壳）。
"""

import numpy as np

from src.calibration.camera_params_format import (
    projection_matrix, camera_params_to_K_Rt)


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
            全 0 的 2D 点（perception.skeleton.extract_skeleton_2d 无前景时的返回）
            视为无效。
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


def planar_lift_skeletons(skeletons_2d, camera_params, plane_point, plane_normal,
                          H, W):
    """单相机 2D 骨架 → 3D（射线-平面相交，平面弯曲假设）。

    适用于 1-DOF 平面弯曲 + 相机正对弯曲平面：中心线落在一个已知平面 P 上，
    对每个 2D 点反投影出射线，与 P 求交即得唯一 3D 点。等价于"深度恒定"假设
    （P 平行像面时），但更通用（P 可任意朝向）。

    Args:
        skeletons_2d: (1,N,J,2) 或 (N,J,2) [col,row]。全 0 视为无效。
        camera_params: (1,10)（仅单相机）。
        plane_point: (3,) 平面上一点（世界系，米，如臂基座）。
        plane_normal: (3,) 平面法向（世界系；正对安装时=相机 view_dir）。
        H, W: 图像尺寸。

    Returns:
        (N, J, 3) 世界系骨架；无效点（无前景/射线平行平面/交在相机后方）为 nan。
    """
    cp = np.asarray(camera_params, float).reshape(-1, 10)
    if cp.shape[0] != 1:
        raise ValueError("planar_lift 仅支持单相机（camera_params 应为 (1,10)）")
    K, R, t = camera_params_to_K_Rt(cp[0], H, W)
    eye = -R.T @ t                                  # 相机世界系位置
    Kinv = np.linalg.inv(K)
    p0 = np.asarray(plane_point, float).reshape(3)
    n = np.asarray(plane_normal, float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)

    sk = np.asarray(skeletons_2d, float)
    if sk.ndim == 4:                                # (1,N,J,2) → (N,J,2)
        sk = sk[0]
    N, J, _ = sk.shape

    out = np.full((N, J, 3), np.nan, float)
    for ni in range(N):
        for j in range(J):
            col, row = sk[ni, j]
            if not np.isfinite(col) or (col == 0.0 and row == 0.0):
                continue
            d_cam = Kinv @ np.array([col, row, 1.0])     # 相机系射线方向
            d_world = R.T @ d_cam                         # 世界系
            denom = float(d_world @ n)
            if abs(denom) < 1e-9:
                continue                                  # 射线平行平面
            lam = float((p0 - eye) @ n) / denom
            if lam <= 0:
                continue                                  # 交点在相机后方
            out[ni, j] = eye + lam * d_world
    return out
