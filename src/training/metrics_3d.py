"""metrics_3d.py — 3D 几何评估指标（纯函数，无模型依赖）。

所有函数接受 (pred, gt) 张量对，形状统一为 (B, N, 3) 或 (N, 3)。
"""

import torch


def chamfer_distance(pred, gt):
    """双向 Chamfer Distance。

    对每个 pred 点找最近 gt 点的距离均值，反之亦然，取平均。

    Args:
        pred: (B, N1, 3) 或 (N1, 3)。
        gt: (B, N2, 3) 或 (N2, 3)。

    Returns:
        标量 Chamfer Distance。
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    # pred: (B, N1, 1, 3), gt: (B, 1, N2, 3)
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)
    dist_matrix = (diff ** 2).sum(-1)

    min_pred_to_gt = dist_matrix.min(dim=2)[0].mean(dim=1)
    min_gt_to_pred = dist_matrix.min(dim=1)[0].mean(dim=1)

    return (min_pred_to_gt.mean() + min_gt_to_pred.mean()) / 2


def endpoint_error(pred, gt):
    """末端节点 L2 误差（取最后一个节点）。

    Args:
        pred: (B, N, 3) 或 (N, 3)。
        gt: (B, N, 3) 或 (N, 3)。

    Returns:
        标量平均末端误差。
    """
    tip_pred = pred[:, -1] if pred.dim() == 3 else pred[-1]
    tip_gt = gt[:, -1] if gt.dim() == 3 else gt[-1]
    return ((tip_pred - tip_gt) ** 2).sum(-1).sqrt().mean()


def mean_node_error(pred, gt):
    """所有节点平均 L2 误差。

    Args:
        pred: (B, N, 3) 或 (N, 3)。
        gt: (B, N, 3) 或 (N, 3)。

    Returns:
        标量平均节点误差。
    """
    return ((pred - gt) ** 2).sum(-1).sqrt().mean()


def curve_smoothness(skeleton):
    """骨架曲线二阶差分 L2 范数（越小越平滑）。

    Args:
        skeleton: (B, N, 3) 或 (N, 3)。

    Returns:
        标量平滑度指标。
    """
    if skeleton.dim() == 2:
        skeleton = skeleton.unsqueeze(0)

    second_diff = skeleton[:, 2:] - 2 * skeleton[:, 1:-1] + skeleton[:, :-2]
    return (second_diff ** 2).sum(-1).sqrt().mean()
