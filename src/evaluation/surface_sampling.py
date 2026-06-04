"""surface_sampling.py — GT 表面采样 + 模型输出→点云转换。

两个核心功能：
  sample_gt_surface        — 从 positions+radii 解析生成 GT 表面点云
  model_output_to_pointcloud — 统一调度，把任意模型输出转为 (N,3) 点云
"""

import numpy as np


def sample_gt_surface(positions, radii, n_points=1000, seed=42):
    """从骨架节点 + 半径解析采样表面点云。

    复用 dataset_pointcloud._sample_surface 的圆柱采样逻辑，
    但使用固定 seed 并返回原始物理坐标（不做归一化）。

    Args:
        positions: (3, N_nodes) 骨架节点坐标。
        radii:     (N_nodes,) 或 scalar，杆体半径。
        n_points:  目标采样点数。
        seed:      随机种子（保证可重复性）。

    Returns:
        np.ndarray: (n_points, 3) 表面点云，物理坐标（米）。
    """
    rng = np.random.RandomState(seed)
    N = positions.shape[1]
    n_segs = N - 1
    n_per_seg = max(1, n_points // n_segs)

    # radii 可以是标量或数组
    radii = np.atleast_1d(np.asarray(radii, dtype=np.float32))
    if len(radii) == 1:
        radii = np.full(N, radii[0])
    elif len(radii) == n_segs:
        # (N-1,) → 扩展到 (N,)，取相邻平均
        radii = np.concatenate([radii, [radii[-1]]])

    pts_list = []
    for i in range(n_segs):
        p1, p2 = positions[:, i], positions[:, i + 1]
        r = (radii[i] + radii[i + 1]) / 2  # 段平均半径
        seg_vec = p2 - p1
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            continue
        tangent = seg_vec / seg_len

        # 构建法平面正交基
        ref = (np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.99
               else np.array([1.0, 0.0, 0.0]))
        perp1 = np.cross(tangent, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)

        # 沿线段插值 + 圆周采样
        n_seg = min(n_per_seg, n_points - len(pts_list))
        if n_seg <= 0:
            break
        t_param = rng.rand(n_seg)
        centers = p1[:, None] * (1 - t_param[None, :]) + p2[:, None] * t_param[None, :]
        angles = rng.rand(n_seg) * 2 * np.pi
        offsets = r * (
            np.cos(angles)[:, None] * perp1[None, :] +
            np.sin(angles)[:, None] * perp2[None, :])
        pts_list.append(centers.T + offsets)

    if not pts_list:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(pts_list, axis=0).astype(np.float32)


def model_output_to_pointcloud(model_type, query_result, model, config):
    """把模型查询结果统一转换为 (N, 3) 点云（物理坐标）。

    Args:
        model_type:  str，模型类型标识（"flowmatch", "mstnf" 等）。
        query_result: dict，query.py 对应函数的返回值。
        model:       nn.Module，原始模型（用于获取归一化参数等）。
        config:      dict，评估配置（含 density_threshold 等）。

    Returns:
        np.ndarray: (N, 3) 点云，物理坐标（米）。
    """
    density_threshold = config.get("density_threshold", 0.5)

    if model_type == "flowmatch":
        # FlowMatch: 反归一化
        pc = query_result["points"]  # (N, 3) numpy, normalized
        scale = model.pc_scale.cpu().numpy()  # (1, 1, 3)
        center = model.pc_center.cpu().numpy()  # (1, 1, 3)
        return pc * scale[0] + center[0]

    elif model_type in ("mstnf", "cmstnf", "ms_scnf"):
        # 密度场：阈值提取高密度点
        points = query_result["points"]    # (grid³, 3)
        density = query_result["density"]  # (grid³,)
        visibility = query_result.get("visibility", np.ones_like(density))
        mask = (visibility > 0.5) & (density > density_threshold)
        return points[mask]

    elif model_type in ("sdf", "skeleton_sdf"):
        # SDF: marching cubes 顶点
        vertices = query_result.get("vertices")
        if vertices is not None and len(vertices) > 0:
            return vertices
        # fallback: 从 SDF grid 采样接近零的点
        sdf_grid = query_result["sdf_grid"]
        threshold = 0.01
        mask = np.abs(sdf_grid) < threshold
        x = query_result["x"]
        y = query_result["y"]
        z = query_result["z"]
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return np.stack([xx[mask], yy[mask], zz[mask]], axis=-1).astype(np.float32)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
