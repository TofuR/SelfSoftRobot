"""Complete-shape initialization from pixels, requiring operator review.

No model state or synthetic occlusion coordinates enter this extractor. Do not
bridge separated components: partial observations belong to the edge observer.
"""
import cv2
import numpy as np
from .skeleton import extract_centerline_2d


def extract_initial_shape(image, n_nodes, polarity='bright'):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('需要 BGR 相机图像')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flag = cv2.THRESH_BINARY if polarity == 'bright' else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
    if polarity == 'bright':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask[(hsv[..., 1] > 135) | (hsv[..., 2] < 95)] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if not candidates or stats[candidates[0], cv2.CC_STAT_AREA] < 200:
        raise ValueError('未提取到完整臂身，请检查明暗方向、背景或手动描画')
    largest = candidates[0]
    area = stats[largest, cv2.CC_STAT_AREA]
    if len(candidates) > 1 and stats[candidates[1], cv2.CC_STAT_AREA] > max(200, .15*area):
        raise ValueError('存在多个较大分离区域：可能遮挡或背景干扰；请露出全臂后重试')
    mask = np.uint8(labels == largest)*255
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise ValueError('候选形状接触图像边界，请让完整臂身进入视野后重试')
    curve, info = extract_centerline_2d(mask, n_points=n_nodes, endpoint_fix=False, return_info=True)
    if not info['success'] or info['arc_length_px'] < 30:
        raise ValueError('完整中心线提取失败，请检查图像或手动描画')
    widths = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    xy = np.rint(curve[1:-1]).astype(int)
    radius = float(np.median(widths[xy[:, 1], xy[:, 0]]))
    if info['arc_length_px'] < 5*radius:
        raise ValueError('候选区域不像细长臂身，请检查背景或手动描画')
    return curve, mask, dict(arc_length_px=info['arc_length_px'], radius_px=radius,
                            base_rule='upper endpoint; operator must review or reverse', area_px=int(area))
