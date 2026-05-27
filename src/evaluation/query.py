"""query.py — 统一的模型查询引擎。

3 个标准查询函数，供可视化脚本和训练 validation 共用：
  query_density_field — density 网格查询（MSTNF, CMSTNF, MS-SCNF）
  query_sdf_field     — SDF 网格 + marching cubes（SDF, SkeletonSDF）
  query_skeleton      — 骨架预测（MS-SCNF, SkeletonSDF）
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def query_density_field(model, action_window, bounds, grid_res, device,
                        gt_skeleton=None, batch_size=50000, n_samples=1):
    """在 3D 网格上查询密度场。

    Args:
        model: 支持 forward(pts, aw, gt_skeleton=...) 的模型。
        action_window: (1, K, D) 动作窗口。
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)。
        grid_res: 每轴分辨率。
        device: 计算设备。
        gt_skeleton: (1, N, 3) GT 骨架 tensor 或 None。
        batch_size: 分块大小（避免 OOM）。
        n_samples: 每个查询点的采样数（通常 1）。

    Returns:
        dict: {points: (N, 3), density: (N,), visibility: (N,)}
    """
    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[2], bounds[3], grid_res)
    z = np.linspace(bounds[4], bounds[5], grid_res)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords_np = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    n_pts = len(coords_np)
    all_vis = np.zeros(n_pts, dtype=np.float32)
    all_dens = np.zeros(n_pts, dtype=np.float32)
    n_batches = (n_pts + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_pts)
        pts = torch.from_numpy(coords_np[start:end]).to(device)
        pts = pts.unsqueeze(1).expand(-1, n_samples, -1)  # (N, 1, 3)

        raw = model(pts, action_window, gt_skeleton=gt_skeleton)

        # 处理可能的 batch 维度展开
        if raw.dim() == 3 and raw.shape[0] == action_window.shape[0] * (end - start):
            raw = raw.reshape(end - start, n_samples, 2)

        vis = torch.sigmoid(raw[..., 0]).mean(dim=-1).cpu().numpy()
        dens = F.softplus(raw[..., 1]).mean(dim=-1).cpu().numpy()

        all_vis[start:end] = vis
        all_dens[start:end] = dens

    return {'points': coords_np, 'density': all_dens, 'visibility': all_vis}


@torch.no_grad()
def query_sdf_field(model, action_window, bounds, grid_res, device,
                    gt_skeleton=None, batch_size=50000):
    """在 3D 网格上查询 SDF 场并执行 marching cubes。

    Args:
        model: 支持 forward(coords, aw, gt_skeleton=...) 的 SDF 模型。
        action_window: (1, K, D) 动作窗口。
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)。
        grid_res: 每轴分辨率。
        device: 计算设备。
        gt_skeleton: (1, N, 3) GT 骨架 tensor 或 None（SkeletonSDF 用）。
        batch_size: 分块大小。

    Returns:
        dict: {sdf_grid: (res,res,res), x, y, z, vertices, faces}
    """
    from skimage import measure

    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[2], bounds[3], grid_res)
    z = np.linspace(bounds[4], bounds[5], grid_res)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords_np = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    n_pts = len(coords_np)
    all_sdf = np.zeros(n_pts, dtype=np.float32)
    n_batches = (n_pts + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_pts)
        pts = torch.from_numpy(coords_np[start:end]).to(device)

        # SkeletonSDF 需要 (N, 1, 3)，SDF 需要 (N, 3)
        if hasattr(model, 'skeleton_head'):
            query = pts.unsqueeze(0)  # (1, N, 3)
            pred = model(query, action_window, gt_skeleton=gt_skeleton)
            sdf_vals = pred.squeeze(-1).squeeze(0)  # (N,)
        else:
            sdf_vals = model(pts, action_window).squeeze(-1)  # (N,)

        all_sdf[start:end] = sdf_vals.cpu().numpy()

    sdf_grid = all_sdf.reshape(grid_res, grid_res, grid_res)

    result = {
        'sdf_grid': sdf_grid,
        'x': x, 'y': y, 'z': z,
        'vertices': None, 'faces': None,
    }

    # Marching cubes 提取零等值面
    spacing = (
        (x[-1] - x[0]) / max(grid_res - 1, 1),
        (y[-1] - y[0]) / max(grid_res - 1, 1),
        (z[-1] - z[0]) / max(grid_res - 1, 1),
    )
    try:
        verts, faces, _, _ = measure.marching_cubes(sdf_grid, level=0, spacing=spacing)
        verts[:, 0] += x[0]
        verts[:, 1] += y[0]
        verts[:, 2] += z[0]
        result['vertices'] = verts
        result['faces'] = faces
    except Exception:
        pass

    return result


@torch.no_grad()
def query_skeleton(model, action_window, device=None):
    """预测多尺度骨架。

    Args:
        model: 有 predict_skeleton() 的模型（SkeletonMixin 提供）。
        action_window: (1, K, D) 动作窗口。
        device: 计算设备（默认从 action_window 推断）。

    Returns:
        dict: {fine: (1,N,3), medium: (1,M,3), coarse: (1,C,3)}
              值为 numpy arrays。
    """
    if device is None:
        device = action_window.device
    skel_dict = model.predict_skeleton(action_window)
    return {k: v.cpu().numpy() for k, v in skel_dict.items()}
