"""由软体机器人已知直径建立图像像素到毫米的近似尺度。"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


DEFAULT_ROBOT_DIAMETER_MM = 16.0


@dataclass(frozen=True)
class DiameterScale:
    diameter_mm: float
    diameter_px: float
    mm_per_px: float
    source: str


def estimate_diameter_px(rows: Iterable[Mapping[str, object]]) -> tuple[float, str]:
    """从逐帧骨架QC取稳健主体直径，优先使用中心线中段宽度。"""
    materialized = list(rows)
    for field in ("body_width_px", "tip_width_px"):
        values = []
        for row in materialized:
            if str(row.get("hard_invalid", "false")).lower() in {"1", "true"}:
                continue
            try:
                value = float(row.get(field, "nan"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0:
                values.append(value)
        if values:
            return float(np.median(values)), field
    raise ValueError("骨架QC中缺少有效的主体直径像素测量")


def _scalar(npz, key):
    if key not in npz:
        return None
    value = float(np.asarray(npz[key]).reshape(()))
    return value if math.isfinite(value) and value > 0 else None


def resolve_diameter_scale(npz, data_dir: str, *,
                           diameter_mm: float = DEFAULT_ROBOT_DIAMETER_MM,
                           diameter_px: float | None = None) -> DiameterScale:
    """解析评价尺度：显式像素直径 -> NPZ合同 -> 同序列骨架QC。"""
    diameter_mm = float(diameter_mm)
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("robot diameter mm必须为正有限值")

    source = "cli"
    measured_px = float(diameter_px) if diameter_px is not None else None
    if measured_px is None:
        measured_px = _scalar(npz, "robot_diameter_px")
        source = "npz:robot_diameter_px"
    if measured_px is None:
        qc_path = os.path.join(os.path.dirname(os.path.normpath(data_dir)),
                               "qc_skeleton", "skeleton_metrics.csv")
        with open(qc_path, newline="", encoding="utf-8") as stream:
            measured_px, field = estimate_diameter_px(csv.DictReader(stream))
        source = f"qc_skeleton:{field}"
    if not math.isfinite(measured_px) or measured_px <= 0:
        raise ValueError("robot diameter px必须为正有限值")
    return DiameterScale(diameter_mm, measured_px,
                         diameter_mm / measured_px, source)
