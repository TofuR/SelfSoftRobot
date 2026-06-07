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


def node_errors(pred, gt):
    """逐节点 L2 误差。

    Args:
        pred: (B, N, 3) 或 (N, 3)。
        gt:   (B, N, 3) 或 (N, 3)。

    Returns:
        (B, N) 或 (N,) — 每个节点的 L2 距离（米）。
    """
    return ((pred - gt) ** 2).sum(-1).sqrt()


def max_node_error(pred, gt):
    """最差节点 L2 误差。

    Args:
        pred: (B, N, 3) 或 (N, 3)。
        gt:   (B, N, 3) 或 (N, 3)。

    Returns:
        标量最大节点误差。
    """
    return node_errors(pred, gt).max()


def evaluate_skeleton(pred, gt, arm_length=0.5, rod_radius=0.015):
    """骨架综合评估 — 绝对/相对/逐节点指标一次性计算。

    Args:
        pred:       (B, N, 3) 或 (N, 3) 预测骨架（米，世界坐标）。
        gt:         (B, N, 3) 或 (N, 3) GT 骨架（米，世界坐标）。
        arm_length: 臂长（米），默认 0.5。
        rod_radius: 杆半径（米），默认 0.015。

    Returns:
        dict 包含:
          绝对指标 (m): mean_node_err, endpoint_err, max_node_err, chamfer_distance
          相对指标 (%): mean_pct_arm, endpoint_pct_arm, mean_pct_radius, endpoint_pct_radius
          逐节点:      per_node_err (N,) ndarray — 每节点平均 L2（跨 batch）
    """
    import numpy as np

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
        gt = gt.unsqueeze(0)

    # 逐节点误差 (B, N)
    errs = node_errors(pred, gt)

    # 聚合
    mean_err = errs.mean().item()
    ep_err = errs[:, -1].mean().item()
    max_err = errs.max().item()
    cd = chamfer_distance(pred, gt).item()

    # 逐节点平均（跨 batch）
    per_node = errs.mean(dim=0).cpu().numpy()  # (N,)

    return {
        # 绝对指标 (m)
        'mean_node_err': mean_err,
        'endpoint_err': ep_err,
        'max_node_err': max_err,
        'chamfer_distance': cd,
        # 相对指标 (%)
        'mean_pct_arm': mean_err / arm_length * 100,
        'endpoint_pct_arm': ep_err / arm_length * 100,
        'mean_pct_radius': mean_err / rod_radius * 100,
        'endpoint_pct_radius': ep_err / rod_radius * 100,
        # 逐节点
        'per_node_err': per_node,
    }
