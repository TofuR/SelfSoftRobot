"""triangulation.py — 多视角 2D 骨架 → 3D 骨架（线性 DLT 三角化）。

输入每视角的 2D 骨架 (V,T,J,2) [col,row] + camera_params(V,10)，逐节点三角化出
世界系 3D 骨架 (T,J,3)。投影矩阵由 camera_params_format.projection_matrix 重建，
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
        skeletons_2d: (V,T,J,2) [col,row]，或 list[V] of (T,J,2)。
            全 0 的 2D 点（perception.skeleton.extract_skeleton_2d 无前景时的返回）
            视为无效。
        camera_params: (V, 10)。
        H, W: 图像尺寸（构 K 的主点 cx=W/2, cy=H/2）。

    Returns:
        skeletons_3d (T,J,3)：世界系米；无效节点为 nan。
    """
    sk = np.asarray(skeletons_2d, float)
    if sk.ndim != 4:
        sk = np.stack(sk, axis=0)             # (V,T,J,2)
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


def triangulate_skeletons_with_quality(skeletons_2d, camera_params, H, W,
                                       max_reprojection_error_px=5.0,
                                       projection_matrices=None):
    """三角化并返回节点级可见性、重投影误差和置信度。

    第一版仍由调用方保证跨视角 node j 的对应。重投影误差超过阈值或落在任一
    参与相机后方的节点会被拒绝为 NaN，而不是作为训练 GT。
    """
    sk = np.asarray(skeletons_2d, float)
    if sk.ndim != 4:
        sk = np.stack(sk, axis=0)
    V, T, J, _ = sk.shape
    Ps = ([np.asarray(value, float).reshape(3, 4)
           for value in projection_matrices]
          if projection_matrices is not None
          else _projection_matrices(camera_params, H, W))
    if len(Ps) != V:
        raise ValueError(f"投影矩阵数量 {len(Ps)} 与视角数 {V} 不一致")
    visibility = np.isfinite(sk).all(axis=-1) & ~np.all(sk == 0.0, axis=-1)
    points = np.full((T, J, 3), np.nan, np.float32)
    reprojection_error = np.full((V, T, J), np.nan, np.float32)
    confidence = np.zeros((T, J), np.float32)
    view_count = visibility.sum(axis=0).astype(np.uint8)
    threshold = float(max_reprojection_error_px)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("max_reprojection_error_px 必须为正数")

    for t in range(T):
        for j in range(J):
            view_ids = np.flatnonzero(visibility[:, t, j])
            if len(view_ids) < 2:
                continue
            X = triangulate_point([sk[v, t, j] for v in view_ids],
                                  [Ps[v] for v in view_ids])
            if not np.isfinite(X).all():
                continue
            Xh = np.r_[X, 1.0]
            errors = []
            positive_depth = True
            for v in view_ids:
                projected = Ps[v] @ Xh
                if projected[2] <= 1e-9:
                    positive_depth = False
                    break
                uv = projected[:2] / projected[2]
                error = float(np.linalg.norm(uv - sk[v, t, j]))
                reprojection_error[v, t, j] = error
                errors.append(error)
            mean_error = float(np.mean(errors)) if errors else float("inf")
            if not positive_depth or mean_error > threshold:
                reprojection_error[:, t, j] = np.nan
                continue
            points[t, j] = X
            geometry_factor = min(1.0, len(view_ids) / max(2.0, float(V)))
            confidence[t, j] = geometry_factor * np.exp(-mean_error / threshold)

    source_mask = np.where(np.isfinite(points).all(axis=-1), 2, 0).astype(np.uint8)
    return {
        "positions_3d": points,
        "positions_2d": np.transpose(sk, (1, 0, 2, 3)).astype(np.float32),
        "visibility": np.transpose(visibility, (1, 0, 2)),
        "reprojection_error": np.transpose(reprojection_error, (1, 0, 2)),
        "position_confidence": confidence,
        "view_count": view_count,
        "source_mask": source_mask,
    }


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
