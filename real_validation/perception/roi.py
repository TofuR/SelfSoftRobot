"""ROI 坐标恢复、边界约束与当前 mask 的方形候选。"""

from __future__ import annotations

import math

import numpy as np


def clamp_roi_xywh(roi_xywh, frame_size_wh, *, minimum_side: int = 16,
                   square: bool = False) -> tuple[int, int, int, int]:
    width, height = (int(value) for value in frame_size_wh)
    x, y, roi_width, roi_height = (int(round(float(value))) for value in roi_xywh)
    if width <= 0 or height <= 0:
        raise ValueError("frame_size_wh 必须为正")
    minimum = max(1, int(minimum_side))
    if square:
        side = min(max(minimum, roi_width, roi_height), width, height)
        roi_width = roi_height = side
    else:
        roi_width = min(max(minimum, roi_width), width)
        roi_height = min(max(minimum, roi_height), height)
    x = min(max(0, x), width - roi_width)
    y = min(max(0, y), height - roi_height)
    return x, y, roi_width, roi_height


def crop_frame(frame, roi_xywh):
    values = np.asarray(frame)
    if values.ndim < 2:
        raise ValueError("frame 至少需要二维")
    x, y, width, height = clamp_roi_xywh(
        roi_xywh, (values.shape[1], values.shape[0]), minimum_side=1)
    return values[y:y + height, x:x + width]


def roi_local_to_camera(points, roi_xywh) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] < 2:
        raise ValueError("points 最后一维至少为2")
    result = values.copy()
    result[..., 0] += float(roi_xywh[0])
    result[..., 1] += float(roi_xywh[1])
    return result.astype(np.float32)


def camera_to_roi_local(points, roi_xywh) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1] < 2:
        raise ValueError("points 最后一维至少为2")
    result = values.copy()
    result[..., 0] -= float(roi_xywh[0])
    result[..., 1] -= float(roi_xywh[1])
    return result.astype(np.float32)


def suggest_square_roi(mask, *, source_offset_xy=(0, 0), source_size_wh=None,
                       padding_px: float = 32.0,
                       minimum_side: int = 64) -> tuple[int, int, int, int]:
    binary = np.asarray(mask) > 0
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("当前 mask 为空，无法生成 ROI 候选")
    offset_x, offset_y = (float(value) for value in source_offset_xy)
    x0 = float(xs.min()) + offset_x - float(padding_px)
    y0 = float(ys.min()) + offset_y - float(padding_px)
    x1 = float(xs.max() + 1) + offset_x + float(padding_px)
    y1 = float(ys.max() + 1) + offset_y + float(padding_px)
    side = max(float(minimum_side), x1 - x0, y1 - y0)
    side = int(math.ceil(side / 16.0) * 16)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    candidate = (int(round(center_x - side / 2.0)),
                 int(round(center_y - side / 2.0)), side, side)
    if source_size_wh is None:
        return candidate
    return clamp_roi_xywh(candidate, source_size_wh,
                          minimum_side=minimum_side, square=True)
