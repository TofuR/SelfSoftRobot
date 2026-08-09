"""彩色图 → 二值剪影（唯一实现）。

实物硅胶半透明、易高光，分割是最易出错的环节。按可用条件选方法：
  - 'backlight'      背光剪影，臂成暗块 → 亮度反相阈值（最稳，推荐）
  - 'bg_subtract'    减去参考背景 → 阈值（需先拍无臂背景）
  - 'color'          HSV 颜色阈值（臂为特定色，如染色/涂层）
  - 'white_on_blue'  白半透明硅胶臂 + 蓝静态墙背景 + 白气管场景（专用，diag 校准）

统一输出 (H,W) 二值（1=前景=臂），形态学清理 + 取最大连通区，可直接喂
real_validation.perception.skeleton.extract_skeleton_2d。
src/data/real/segmentation.py 是本模块的薄壳。

⚠️ 在线部署必须使用与训练一致的参数。真实参数在
   real_capture/data/derived/<seq>/segment_meta.json（实测 val=100，非默认 120）。
"""

import glob
import os

import numpy as np

from ._compat import _CV2_ERR, cv2


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


# -------------------------------- 'white_on_blue' 实物硅胶臂专用
def build_median_background(cam_dir, n_bg=500):
    """从图像目录均匀采样 n_bg 帧灰度 → per-pixel median = 静态背景。

    机器人移动占每像素 <50% 时间 → 中值趋近真实静态背景（蓝墙+座+静态管），
    无需单独拍无臂背景。返回 (bg_gray, frame_paths)。
    """
    fs = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    if not fs:
        raise FileNotFoundError(f"无帧: {cam_dir}")
    idx = np.linspace(0, len(fs) - 1, min(n_bg, len(fs))).astype(int)
    stack = np.stack([cv2.imread(fs[i], cv2.IMREAD_GRAYSCALE) for i in idx])
    return np.median(stack, axis=0).astype(np.uint8), fs


def segment_white_on_blue(bgr, bg_gray, sat=100, val=120, diff=25, dil=35,
                          open_k=5, close_k=15,
                          min_area_frac=0.003, min_h_frac=0.15):
    """白半透明硅胶臂（+ 蓝静态背景 + 白气管）专用分割。

    管线（diag 校准）:
      HSV白(S<sat,V>val) ∩ dilate(背景差, dil)
        → OPEN(open_k 去细管) → CLOSE(close_k 填体) → fillholes
        → 面积≥min_area_frac·Frame 且 高≥min_h_frac·H 的最大连通区

    半透明臂内部与蓝底对比低 → 背景差只抓边；HSV白 抓臂主体 + 杂白(座/眩光)。
    两者交集 = 动且白 = 臂；OPEN 按宽度去细管；取最大连通区。

    Returns: (H,W) uint8 {0,1}。
    """
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    from scipy.ndimage import binary_fill_holes
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < sat) & (hsv[:, :, 2] > val)).astype(np.uint8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    moved = (cv2.absdiff(gray, bg_gray) > diff).astype(np.uint8)
    moved = cv2.dilate(moved, np.ones((dil, dil), np.uint8)) if dil > 1 else moved
    m = (white & moved).astype(np.uint8)
    if open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    m = binary_fill_holes(m > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros((H, W), np.uint8)
    if n > 1:
        cands = [(int(stats[i, cv2.CC_STAT_AREA]), i) for i in range(1, n)
                 if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * H * W
                 and stats[i, cv2.CC_STAT_HEIGHT] >= min_h_frac * H]
        if cands:
            cands.sort(reverse=True)
            out[lbl == cands[0][1]] = 1
    return out


def segment_views(images_bgr, method="backlight", bg=None,
                  color_bounds=None, gray_thresh=60, bg_thresh=25,
                  white_on_blue_params=None):
    """(V,N,H,W,3) BGR → (V,N,H,W) 二值。

    Args:
        method: 'backlight'|'bg_subtract'|'color'|'white_on_blue'。
        bg: (V,H,W) 背景灰度（bg_subtract / white_on_blue 用）。
        color_bounds: (lower_hsv, upper_hsv)（color 用）。
        gray_thresh / bg_thresh: backlight / bg_subtract 阈值。
        white_on_blue_params: dict（white_on_blue 用；None=默认）。

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
            elif method == "white_on_blue":
                params = white_on_blue_params or {}
                masks[v, n] = segment_white_on_blue(bgr, bg[v], **params)
            else:
                raise ValueError(f"未知分割方法: {method}")
    return masks


def masks_to_skeletons_2d(masks, n_points=31, tip_fix=True):
    """(V,N,H,W) 二值 → (V,N,n_points,2) 2D 骨架，复用 skeleton 模块。

    返回 [col,row]；无前景帧为全 0（与 extract_skeleton_2d 约定一致，三角化时跳过）。
    tip_fix=True(默认): 末端 node0 垂直切片修正(修弯管 cap 角落偏移), 实物默认开。
    """
    from .skeleton import batch_extract_skeleton_2d

    V, N = masks.shape[:2]
    out = np.zeros((V, N, n_points, 2), np.float32)
    for v in range(V):
        out[v] = batch_extract_skeleton_2d(masks[v], n_points, tip_fix=tip_fix)
    return out
