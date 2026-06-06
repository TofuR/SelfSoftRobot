"""pointcloud_losses.py — 点云对比损失函数。

补充 src/training/metrics_3d.py 中没有的点云损失：
  - chamfer_distance_with_details: 带方向分量的 CD（用于日志记录）
  - emd_approx: 近似 Earth Mover's Distance（沿主轴排序，适合杆状结构）

训练用 CD 直接复用 metrics_3d.chamfer_distance（已可微）。
"""

import torch
import torch.nn.functional as F


def chamfer_distance_with_details(pred: torch.Tensor, gt: torch.Tensor):
    """双向 Chamfer Distance，返回各方向分量（用于日志记录）。

    Args:
        pred: (B, N1, 3) 预测点云。
        gt:   (B, N2, 3) 目标点云。

    Returns:
        dict: {"cd": 总 CD, "cd_pred": pred→gt 分量, "cd_gt": gt→pred 分量}
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist_matrix = (diff ** 2).sum(-1)

    cd_pred = dist_matrix.min(dim=2)[0].mean()
    cd_gt = dist_matrix.min(dim=1)[0].mean()

    return {
        "cd": (cd_pred + cd_gt) / 2,
        "cd_pred": cd_pred,
        "cd_gt": cd_gt,
    }


def emd_approx(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """近似 Earth Mover's Distance。

    对细长杆状结构有效：沿主轴（Z轴）对两组点排序，
    然后计算排序后点对的 MSE。

    注意：仅适用于形状大致沿某一主轴延伸的情况（如 Cosserat rod）。

    Args:
        pred: (B, N1, 3) 预测点云。
        gt:   (B, N2, 3) 目标点云。

    Returns:
        标量，近似 EMD。
    """
    n = min(pred.shape[1], gt.shape[1])

    # 按 z 坐标排序
    pred_idx = pred[:, :, 2].argsort(dim=-1)
    pred_sorted = torch.gather(
        pred, 1,
        pred_idx.unsqueeze(-1).expand(-1, -1, 3),
    )[:, :n]

    gt_idx = gt[:, :, 2].argsort(dim=-1)
    gt_sorted = torch.gather(
        gt, 1,
        gt_idx.unsqueeze(-1).expand(-1, -1, 3),
    )[:, :n]

    return F.mse_loss(pred_sorted, gt_sorted)


def compactness_loss(pred_points, n_z_bins=10):
    """紧密度损失：惩罚每个 z-band 内点在 x,y 方向的过度扩散。

    直接攻击"扇形"问题：末端点云散开时，同一 z 层的点在 x,y 方向
    分布过广。该损失鼓励同一 z 层的点保持紧凑。

    Args:
        pred_points: (B, N, 3) 预测点云。
        n_z_bins: z 方向分 bin 数。

    Returns:
        标量，各 z-bin 内 x,y 标准差的平均值。
    """
    B, N, _ = pred_points.shape
    z = pred_points[:, :, 2]  # (B, N)

    total_loss = 0.0
    count = 0
    for b in range(B):
        z_b = z[b]
        pts_b = pred_points[b]

        z_min = z_b.min()
        z_max = z_b.max()
        if z_max - z_min < 1e-6:
            continue

        bin_edges = torch.linspace(z_min, z_max + 1e-6, n_z_bins + 1,
                                   device=z_b.device)

        for i in range(n_z_bins):
            mask = (z_b >= bin_edges[i]) & (z_b < bin_edges[i + 1])
            n_in = mask.sum()
            if n_in < 3:
                continue
            xy = pts_b[mask, :2]
            total_loss += xy.std(dim=0).mean()
            count += 1

    return total_loss / max(count, 1)


def cross_section_circularity_loss(pred_points, n_z_bins=10):
    """截面圆度损失：鼓励每个 z 截面的点分布接近圆形。

    原理：对每个 z-bin 内的点：
      1. 计算中心点 (mean of x,y)
      2. 计算每个点到中心的距离
      3. 惩罚距离的标准差（标准差越小 → 越接近圆）

    对于软体机器人的圆柱形结构，理想情况下每个截面应该是圆形。

    Args:
        pred_points: (B, N, 3) 预测点云。
        n_z_bins: z 方向分 bin 数。

    Returns:
        标量，各 z-bin 内半径标准差的平均值。
    """
    B, N, _ = pred_points.shape
    z = pred_points[:, :, 2]

    total_loss = 0.0
    count = 0
    for b in range(B):
        z_b = z[b]
        pts_b = pred_points[b]

        z_min = z_b.min()
        z_max = z_b.max()
        if z_max - z_min < 1e-6:
            continue

        bin_edges = torch.linspace(z_min, z_max + 1e-6, n_z_bins + 1,
                                   device=z_b.device)

        for i in range(n_z_bins):
            mask = (z_b >= bin_edges[i]) & (z_b < bin_edges[i + 1])
            n_in = mask.sum()
            if n_in < 4:
                continue
            xy = pts_b[mask, :2]
            center = xy.mean(dim=0)
            radii = torch.norm(xy - center, dim=-1)
            total_loss += radii.std()
            count += 1

    return total_loss / max(count, 1)
