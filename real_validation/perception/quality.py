"""在线单帧质量门控。

离线管线对"坏帧"的处置一律是**时间插值修复**(clean_outlier_skeletons /
clean_transition_npz / repair_masks),那需要未来帧,在线不可复现。
在线只能**拒帧**:verdict=reject 的帧不进模型、不更新 anchor，但仍写入隐藏评价流。

阈值分两类:
  - **数据相关**(area_median_px):无默认值，必须由调用方从 manifest 提供。
    理由:仓库里"mask 面积中位数"有 4 个互相矛盾的值(8562 white_on_blue /
    6718 outliers / 6323-7099 qc / 运行时重算)，而部署 checkpoint 用的 SAM2 mask
    一个统计都没有。在这里写死任何一个都会变成第 5 套约定。
  - **策略常量**(比例、行阈值、速度上界):有默认值，与离线判据对齐。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._compat import _CV2_ERR, cv2

Q_OK = "ok"
Q_DEGRADED = "degraded"
Q_REJECT = "reject"

R_EMPTY_MASK = "empty_mask"
R_AREA_LOW = "area_ratio_low"
R_AREA_HIGH = "area_ratio_high"
R_HEIGHT_LOW = "height_frac_low"
R_TOP_ROW_HIGH = "top_row_high"
R_SECOND_BLOB = "second_blob_present"
R_TIP_FIX_SKIPPED = "tip_fix_skipped"
R_NODE_STEP_HIGH = "node_step_high"
R_FRAME_STALE = "frame_stale"
R_REGISTRATION_DISPLACED = "registration_displaced"

_REJECT_REASONS = frozenset({
    R_EMPTY_MASK, R_AREA_LOW, R_AREA_HIGH, R_HEIGHT_LOW, R_TOP_ROW_HIGH,
    R_NODE_STEP_HIGH, R_FRAME_STALE, R_REGISTRATION_DISPLACED,
})


@dataclass(frozen=True)
class QualityThresholds:
    """area_median_px 无默认值 —— 它是数据相关量，必须显式提供。"""
    area_median_px: float
    area_ratio_min: float = 0.7
    area_ratio_max: float = 1.3
    min_height_frac: float = 0.15
    max_top_row: int = 20
    max_second_blob_ratio: float = 0.15
    max_node_step_px: float = 4.0
    max_frame_age_s: float = 0.5
    max_registration_displacement_px: float = 2.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.area_median_px) or self.area_median_px <= 0:
            raise ValueError("area_median_px 必须是正的有限值")
        if self.area_ratio_min > self.area_ratio_max:
            raise ValueError("area_ratio_min 不能大于 area_ratio_max")


@dataclass(frozen=True)
class FrameQuality:
    verdict: str
    reasons: tuple[str, ...] = ()
    flags: dict[str, Any] = field(default_factory=dict)


def _blob_stats(mask):
    """返回 (area, height, top_row, second_ratio)。无前景时 (0, 0, H, 0.0)。"""
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    height_total = binary.shape[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return 0, 0, height_total, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    largest = 1 + int(order[0])
    area = int(stats[largest, cv2.CC_STAT_AREA])
    box_height = int(stats[largest, cv2.CC_STAT_HEIGHT])
    top_row = int(stats[largest, cv2.CC_STAT_TOP])
    second = float(areas[order[1]]) / float(area) if len(order) > 1 and area else 0.0
    return area, box_height, top_row, second


def assess_frame(mask, skeleton, skeleton_info: dict, thresholds: QualityThresholds,
                 *, prev_skeleton=None, frame_age_s: float | None = None,
                 registration_displacement_px: float | None = None) -> FrameQuality:
    """对单帧给出 ok / degraded / reject 判决与全部标志。

    flags 里的值全部是 Python 标量(不是 numpy 标量、不含 NaN)，因为它们会经
    io.atomic_write_json(allow_nan=False) 落进 run 目录。
    """
    reasons: list[str] = []
    area, box_height, top_row, second_ratio = _blob_stats(mask)
    frame_height = int(np.asarray(mask).shape[0])
    area_ratio = float(area) / float(thresholds.area_median_px)
    height_frac = float(box_height) / float(frame_height) if frame_height else 0.0

    if area == 0:
        reasons.append(R_EMPTY_MASK)
    else:
        if area_ratio < thresholds.area_ratio_min:
            reasons.append(R_AREA_LOW)
        if area_ratio > thresholds.area_ratio_max:
            reasons.append(R_AREA_HIGH)
        if height_frac < thresholds.min_height_frac:
            reasons.append(R_HEIGHT_LOW)
        if top_row > thresholds.max_top_row:
            reasons.append(R_TOP_ROW_HIGH)
        if second_ratio > thresholds.max_second_blob_ratio:
            reasons.append(R_SECOND_BLOB)

    if skeleton_info.get("tip_fix_requested") and not skeleton_info.get("tip_fix_applied"):
        reasons.append(R_TIP_FIX_SKIPPED)

    max_step = 0.0
    if prev_skeleton is not None:
        current = np.asarray(skeleton, dtype=np.float64)[:, :2]
        previous = np.asarray(prev_skeleton, dtype=np.float64)[:, :2]
        if current.shape != previous.shape:
            raise ValueError(f"骨架形状不同：{current.shape} != {previous.shape}")
        max_step = float(np.linalg.norm(current - previous, axis=1).max())
        if max_step > thresholds.max_node_step_px:
            reasons.append(R_NODE_STEP_HIGH)

    if frame_age_s is not None and float(frame_age_s) > thresholds.max_frame_age_s:
        reasons.append(R_FRAME_STALE)

    displacement = registration_displacement_px
    if displacement is not None:
        value = float(displacement)
        if not math.isfinite(value) or value > thresholds.max_registration_displacement_px:
            reasons.append(R_REGISTRATION_DISPLACED)

    if any(reason in _REJECT_REASONS for reason in reasons):
        verdict = Q_REJECT
    elif reasons:
        verdict = Q_DEGRADED
    else:
        verdict = Q_OK

    flags: dict[str, Any] = {
        "mask_area_px": int(area),
        "mask_area_ratio": float(area_ratio),
        "blob_height_frac": float(height_frac),
        "top_row": int(top_row),
        "second_blob_ratio": float(second_ratio),
        "tip_fix_applied": bool(skeleton_info.get("tip_fix_applied", False)),
        "tip_fix_reason": str(skeleton_info.get("tip_fix_reason", "")),
        "n_valid_rows": int(skeleton_info.get("n_valid_rows", 0)),
        "max_node_step_px": float(max_step),
        "frame_age_s": None if frame_age_s is None else float(frame_age_s),
        "registration_displacement_px": (
            None if displacement is None or not math.isfinite(float(displacement))
            else float(displacement)),
        "verdict": verdict,
    }
    return FrameQuality(verdict=verdict, reasons=tuple(reasons), flags=flags)
