"""segmentation.py — 彩色图 → 二值剪影（替代 PyVista 的干净二值图）。

实物硅胶半透明、易高光，分割是最易出错的环节。按可用条件选方法：
  - 'backlight'   背光剪影，臂成暗块 → 亮度反相阈值（最稳，推荐）
  - 'bg_subtract' 减去参考背景 → 阈值（需先拍无臂背景）
  - 'color'       HSV 颜色阈值（臂为特定色，如染色/涂层）

统一输出 (H,W) 二值（1=前景=臂），形态学清理 + 取最大连通区，可直接喂
src/utils/skeleton_2d.extract_skeleton_2d。
"""

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc


def _clean(mask):
    """形态学开闭 + 取最大连通区（假设画面中主要前景是臂）。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return (lbl == keep).astype(np.uint8)


def segment_backlight(gray, thresh=60):
    """背光：臂为暗 → 前景 = gray < thresh。"""
    mask = (gray < thresh).astype(np.uint8)
    return _clean(mask)


def segment_bg_subtract(gray, bg_gray, thresh=25):
    """背景减：|gray - bg| > thresh → 前景。"""
    d = cv2.absdiff(gray, bg_gray) if cv2 else np.abs(gray.astype(int) -
                                                      bg_gray.astype(int))
    return _clean((d > thresh).astype(np.uint8))


def segment_color(bgr, lower_hsv, upper_hsv):
    """HSV 颜色阈值。lower/upper_hsv: 各 (3,) uint8。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv, np.uint8),
                       np.array(upper_hsv, np.uint8))
    return _clean((mask > 0).astype(np.uint8))


def segment_views(images_bgr, method="backlight", bg=None,
                  color_bounds=None, gray_thresh=60, bg_thresh=25):
    """(V,N,H,W,3) BGR → (V,N,H,W) 二值。

    Args:
        method: 'backlight'|'bg_subtract'|'color'。
        bg: (V,H,W) 背景灰度（bg_subtract 用）。
        color_bounds: (lower_hsv, upper_hsv)（color 用）。
        gray_thresh / bg_thresh: 各方法阈值。

    Returns:
        masks (V,N,H,W) uint8 {0,1}。
    """
    V, N = images_bgr.shape[:2]
    H, W = images_bgr.shape[2:4]
    masks = np.zeros((V, N, H, W), np.uint8)
    for v in range(V):
        for n in range(N):
            bgr = images_bgr[v, n]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if cv2 else \
                bgr.mean(axis=-1)
            if method == "backlight":
                masks[v, n] = segment_backlight(gray, gray_thresh)
            elif method == "bg_subtract":
                masks[v, n] = segment_bg_subtract(gray, bg[v], bg_thresh)
            elif method == "color":
                lo, hi = color_bounds
                masks[v, n] = segment_color(bgr, lo, hi)
            else:
                raise ValueError(f"未知分割方法: {method}")
    return masks


def masks_to_skeletons_2d(masks, n_points=31):
    """(V,N,H,W) 二值 → (V,N,n_points,2) 2D 骨架，复用 skeleton_2d。

    返回 [col,row]；无前景帧为全 0（与 extract_skeleton_2d 约定一致，三角化时跳过）。
    """
    from src.utils.skeleton_2d import batch_extract_skeleton_2d

    V, N = masks.shape[:2]
    out = np.zeros((V, N, n_points, 2), np.float32)
    for v in range(V):
        out[v] = batch_extract_skeleton_2d(masks[v], n_points)
    return out
