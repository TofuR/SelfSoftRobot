"""shape_metrics.py — 3D 形状比较指标（纯 numpy 实现）。

三个核心指标，统一签名 (pred, gt) -> float：
  chamfer_distance   — 双向平均最近邻距离
  f_score            — 精度-覆盖率调和平均（阈值化）
  hausdorff_distance — 双向最大最近邻距离（最坏情况）
"""

import numpy as np
from scipy.spatial.distance import cdist


def chamfer_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """双向 Chamfer Distance。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。

    Returns:
        float: (mean(pred→gt) + mean(gt→pred)) / 2。
    """
    dists = cdist(pred, gt)  # (N, M)
    cd_pred = dists.min(axis=1).mean()  # pred→gt
    cd_gt = dists.min(axis=0).mean()    # gt→pred
    return float((cd_pred + cd_gt) / 2)


def f_score(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    """F-Score @threshold。

    Precision: pred 中有 GT 邻居（<threshold）的比例。
    Recall:    GT 中有 pred 邻居（<threshold）的比例。
    F-Score:   precision 和 recall 的调和平均。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。
        threshold: 距离阈值（米）。

    Returns:
        float: F-Score ∈ [0, 1]。
    """
    dists = cdist(pred, gt)  # (N, M)
    precision = (dists.min(axis=1) < threshold).mean()
    recall = (dists.min(axis=0) < threshold).mean()
    if precision + recall < 1e-8:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def hausdorff_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    """双向 Hausdorff Distance（最大最近邻距离）。

    Args:
        pred: (N, 3) 预测点云。
        gt:   (M, 3) GT 点云。

    Returns:
        float: max(max(pred→gt), max(gt→pred))。
    """
    dists = cdist(pred, gt)  # (N, M)
    hd_pred = dists.min(axis=1).max()  # pred→gt 最远点
    hd_gt = dists.min(axis=0).max()    # gt→pred 最远点
    return float(max(hd_pred, hd_gt))
