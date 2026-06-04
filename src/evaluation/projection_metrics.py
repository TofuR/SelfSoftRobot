"""projection_metrics.py — 投影 F1 评价。

将 3D 点云投影到相机视角，与 GT 二值图像做像素级比较。
解决纯 3D 指标（CD, F-Score）无法惩罚"扇形扩散"的问题：
过度预测的区域投影后产生额外像素，拉低 precision → F1 下降。

核心函数：
  project_points_to_mask — 3D 点云 → 2D 二值 mask（含膨胀填充）
  mask_f1_score          — 两个 mask 间的像素级 F1 / IoU
  projection_f1          — 便捷封装：投影 + 比较
"""

import numpy as np
import torch

from src.utils.camera_system import MultiCameraSystem


def project_points_to_mask(points_3d, cam_config, H=None, W=None, dilation=1):
    """将 3D 点投影到相机视角生成二值 mask。

    Args:
        points_3d: (N, 3) numpy array, 3D 点坐标（物理坐标，米）。
        cam_config: dict, 相机配置 {eye, center, up, focal, H, W}。
        H, W: 可选覆盖图像尺寸（默认用 cam_config 中的值）。
        dilation: int, 膨胀迭代次数（填充点云稀疏投影产生的空隙）。

    Returns:
        mask: (H, W) numpy uint8 array, 1=有点投影到该像素。
    """
    h = H or cam_config['H']
    w = W or cam_config['W']

    if len(points_3d) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # 用 MultiCameraSystem.project 做针孔投影
    pts_t = torch.from_numpy(np.asarray(points_3d, dtype=np.float32))
    cam_sys = MultiCameraSystem([cam_config])
    pixels, depths = cam_sys.project(pts_t, view_idx=0, device='cpu')

    # 只保留相机前方的点（深度 > 0）
    valid = depths.numpy() > 0
    pixels = pixels.numpy()[valid]

    mask = np.zeros((h, w), dtype=np.uint8)
    if len(pixels) == 0:
        return mask

    # 四舍五入到最近像素，裁剪到图像边界
    px = np.clip(np.round(pixels[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.round(pixels[:, 1]).astype(int), 0, h - 1)
    mask[py, px] = 1

    # 膨胀：稀疏点云投影会留下空隙，膨胀填充
    if dilation > 0:
        from scipy.ndimage import binary_dilation
        mask = binary_dilation(mask, iterations=dilation).astype(np.uint8)

    return mask


def mask_f1_score(pred_mask, gt_mask, threshold=0.5):
    """两个二值 mask 之间的像素级 F1 / IoU。

    Args:
        pred_mask: (H, W) 预测 mask（任意数值类型）。
        gt_mask: (H, W) GT mask。
        threshold: 二值化阈值。

    Returns:
        dict: {precision, recall, f1, iou} 各为 float ∈ [0, 1]。
    """
    pred = (np.asarray(pred_mask, dtype=np.float32) > threshold).astype(np.uint8)
    gt = (np.asarray(gt_mask, dtype=np.float32) > threshold).astype(np.uint8)

    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    iou = tp / max(tp + fp + fn, 1)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
    }


def projection_f1(pred_points_3d, gt_image, cam_config, dilation=1):
    """投影 F1：3D 点云投影到相机视角，与 GT 图像做像素级比较。

    解决的问题：纯 3D Chamfer Distance 不惩罚"扇形扩散"——
    模型在末端多预测的点虽然离 GT 表面较远，但数量少时对 CD 均值影响小。
    投影 F1 直接惩罚这些多余像素（precision 下降）。

    Args:
        pred_points_3d: (N, 3) 预测的 3D 点云 (numpy, 物理坐标)。
        gt_image: (H, W) GT 二值图像 (float32 or uint8)。
        cam_config: dict, 相机配置 {eye, center, up, focal, H, W}。
        dilation: int, 投影 mask 膨胀迭代次数。

    Returns:
        dict: {precision, recall, f1, iou} 各为 float ∈ [0, 1]。
    """
    H, W = gt_image.shape[:2]
    pred_mask = project_points_to_mask(
        pred_points_3d, cam_config, H=H, W=W, dilation=dilation)
    return mask_f1_score(pred_mask, gt_image)
