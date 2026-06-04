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
