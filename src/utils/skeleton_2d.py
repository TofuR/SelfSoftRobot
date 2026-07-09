"""skeleton_2d.py — 2D 骨架提取与 3D→2D 投影工具。

提供:
  - 从二值图像提取 2D 中心线/骨架
  - 将 3D 骨架点投影到相机像素坐标
  - 计算 2D 投影骨架 loss（Phase 1 监督信号）
"""

import numpy as np
import torch


def _perpendicular_tip_fix(skeleton, binary_img, n_points):
    """末端 node0 的"垂直于局部轴切片质心"修正（修倾斜 cap 的 corner 偏移 + node0-1-2 折角）。

    根因: 逐行质心对倾斜管的末端 cap 做**水平**切片, 最底行落在 cap 角落而非中点
    (弯管 cap 倾斜时, 底部几行变窄且偏向一侧→node0 落角落, node0-1-2 形成非物理尖折角)。
    修法: body 段保留(直管段水平切片本来就对), 仅重算 tip——从 body 节点估**局部轴方向**,
    在尖端做**垂直于轴**的切片(对管左右对称)→质心=局部中心线中点=cap 中点, 与倾斜无关;
    再沿 body→node0 重布 node1-2 消折角。body(node3+)不动。

    实测(实物 10116 帧): 34% 帧(M0 末端误差>4px)从 mean 6.94px→2.01px(-71%); body 不变;
    0 失败; 仅 1.3% 易帧小幅回退(≤3.5px)。详见 scripts/real/compare_skeleton_methods.py。

    仅在 n_points>=5 且 mask 非空足够时生效, 否则原样返回(skeleton 不变)。
    """
    sk = skeleton.astype(np.float64)
    if n_points < 5 or np.abs(sk).max() == 0:
        return skeleton
    ys, xs = np.where(binary_img > 0.5)
    if len(xs) < 10:
        return skeleton
    pts = np.column_stack([xs.astype(float), ys.astype(float)])  # (col, row)
    far = sk[min(7, n_points - 1)]       # body 节点(偏 base)
    near = sk[min(3, n_points - 1)]      # body 节点(偏 tip)
    seg = near - far                      # 指向 tip 的局部轴方向
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return skeleton
    d = seg / L
    proj = (pts - far) @ d
    w = float(binary_img.sum(1).max())    # 管径估计(最大行宽)
    slab = proj >= proj.max() - 0.4 * w   # 尖端垂直切片
    if int(slab.sum()) < 3:
        return skeleton
    node0 = pts[slab].mean(0)             # 垂直切片质心 = cap 中心线中点
    sk[0] = node0
    a = sk[min(3, n_points - 1)]          # 沿 body→node0 重布 node1,2 消折角
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32)


def extract_skeleton_2d(binary_img, n_points=31, tip_fix=False):
    """从二值图像提取 2D 中心线骨架。

    对图像每一行（从底到顶）计算白色像素的质心列坐标，
    然后沿弧长均匀重采样到 n_points 个点。

    Args:
        binary_img: (H, W) 二值图像，1=前景。
        n_points: 采样点数。
        tip_fix: 是否对末端 node0 做"垂直于局部轴切片质心"修正。默认 False
            (保持原有行为, 供 sim 等已验证管线)。实物管在弯曲时逐行质心会把 node0
            落到倾斜 cap 的角落, 置 True 可修正(见 _perpendicular_tip_fix)。

    Returns:
        skeleton_2d: (n_points, 2) 像素坐标 [col, row]，从底部到顶部排列。
                     若图像无前景，返回全零。
    """
    H, W = binary_img.shape
    coords = []

    for row in range(H - 1, -1, -1):
        white_cols = np.where(binary_img[row] > 0.5)[0]
        if len(white_cols) > 0:
            center_col = white_cols.mean()
            coords.append([center_col, float(row)])

    if len(coords) < 2:
        return np.zeros((n_points, 2), dtype=np.float32)

    coords = np.array(coords, dtype=np.float32)

    # 沿弧长均匀重采样
    diffs = np.diff(coords, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return np.zeros((n_points, 2), dtype=np.float32)

    target_lens = np.linspace(0, total_len, n_points)

    resampled = np.zeros((n_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lens, cum_len, coords[:, 0])
    resampled[:, 1] = np.interp(target_lens, cum_len, coords[:, 1])

    if tip_fix:
        resampled = _perpendicular_tip_fix(resampled, binary_img, n_points)
    return resampled


def batch_extract_skeleton_2d(images, n_points=31, tip_fix=False):
    """批量提取 2D 骨架。

    Args:
        images: (T, H, W) 二值图像序列。
        n_points: 采样点数。
        tip_fix: 末端 node0 垂直切片修正(见 extract_skeleton_2d), 默认 False。

    Returns:
        skeletons: (T, n_points, 2) 像素坐标。
    """
    T = images.shape[0]
    skeletons = np.zeros((T, n_points, 2), dtype=np.float32)
    for t in range(T):
        skeletons[t] = extract_skeleton_2d(images[t], n_points, tip_fix=tip_fix)
    return skeletons


def project_3d_to_2d(points_3d, eye, center, up, focal, H, W):
    """将 3D 点投影到相机像素坐标（可微，支持 torch）。

    与 camera.py 的 get_rays 保持一致的投影约定。

    Args:
        points_3d: (..., 3) 世界坐标下的 3D 点（torch tensor）。
        eye: 相机位置 (3,)。
        center: 注视点 (3,)。
        up: 上方向 (3,)。
        focal: 焦距。
        H, W: 图像尺寸。

    Returns:
        points_2d: (..., 2) 像素坐标 [col, row]。
    """
    eye = torch.tensor(eye, dtype=points_3d.dtype, device=points_3d.device)
    center = torch.tensor(center, dtype=points_3d.dtype, device=points_3d.device)
    up = torch.tensor(up, dtype=points_3d.dtype, device=points_3d.device)
    focal = torch.tensor(float(focal), dtype=points_3d.dtype, device=points_3d.device)

    view_dir = center - eye
    view_dir = view_dir / torch.norm(view_dir)

    right = torch.linalg.cross(view_dir, up)
    right = right / torch.norm(right)

    true_up = torch.linalg.cross(right, view_dir)
    true_up = true_up / torch.norm(true_up)

    p_rel = points_3d - eye
    p_right = (p_rel * right).sum(dim=-1)
    p_up = (p_rel * true_up).sum(dim=-1)
    p_view = (p_rel * view_dir).sum(dim=-1)

    p_view = p_view.clamp(min=1e-6)

    col = focal * p_right / p_view + W * 0.5
    row = -focal * p_up / p_view + H * 0.5

    return torch.stack([col, row], dim=-1)


def compute_2d_skeleton_loss(pred_skeleton_3d, skeleton_2d_list, camera_list):
    """计算多视角 2D 骨架投影 loss。

    将预测的 3D 骨架投影到每个相机视角，与提取的 2D 骨架对比。

    Args:
        pred_skeleton_3d: (B, N, 3) 预测的 3D 骨架。
        skeleton_2d_list: list of (B, N, 2) 每个视角的 2D 骨架（torch tensor）。
        camera_list: list of dict，每个含 eye, center, up, focal, H, W。

    Returns:
        loss: scalar，所有视角的 L2 loss 之和。
    """
    total_loss = torch.tensor(0.0, device=pred_skeleton_3d.device)

    for skel_2d, cam in zip(skeleton_2d_list, camera_list):
        projected = project_3d_to_2d(
            pred_skeleton_3d,
            cam['eye'], cam['center'], cam['up'],
            cam['focal'], cam['H'], cam['W'],
        )
        # 只对有效骨架点计算 loss（非零点）
        mask = (skel_2d.sum(dim=-1) > 0.1)
        if mask.any():
            total_loss = total_loss + ((projected[mask] - skel_2d[mask]) ** 2).mean()

    return total_loss
