"""一次真实验证实验的相机与在线感知合同。"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .io import stable_digest


@dataclass(frozen=True)
class ValidationSetup:
    """绑定当前 run、当前相机和当前模型的在线感知配置。

    ROI 使用源相机像素 ``[x, y, width, height]``。骨架提取在 ROI 局部执行，
    随后恢复到源相机像素，再进入机器人毫米坐标变换。
    """

    camera_backend: str
    camera_index: int
    camera_identity: str
    frame_size_wh: tuple[int, int]
    roi_xywh: tuple[int, int, int, int]
    segmentation_method: str
    segment_params: dict[str, Any]
    model_checkpoint_hash: str
    state_coordinate_frame: str
    state_length_unit: str
    robot_diameter_mm: float
    node_order: str = "base_to_tip"
    background_path: str | None = None
    background_sha256: str | None = None
    mask_area_reference_px: float | None = None
    base_side: str = "top"
    segment_lengths: tuple[float, ...] = (1.0, 1.0)
    skeleton_frame_transform: dict[str, Any] | None = None
    setup_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: dt.datetime.now(
        dt.timezone.utc).isoformat())
    schema_version: int = 2

    def __post_init__(self) -> None:
        if int(self.schema_version) != 2:
            raise ValueError("ValidationSetup schema_version 必须为 2")
        width, height = (int(value) for value in self.frame_size_wh)
        x, y, roi_width, roi_height = (int(value) for value in self.roi_xywh)
        if width <= 0 or height <= 0:
            raise ValueError("frame_size_wh 必须为正")
        if roi_width <= 0 or roi_height <= 0:
            raise ValueError("ROI 宽高必须为正")
        if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
            raise ValueError("ROI 必须位于源相机画面内")
        if self.camera_index < 0:
            raise ValueError("camera_index 必须为非负整数")
        if self.segmentation_method not in {"backlight", "white_on_blue"}:
            raise ValueError(f"未知在线分割方法: {self.segmentation_method}")
        if self.state_coordinate_frame not in {
                "camera_pixel_v1", "robot_planar_mm_v1"}:
            raise ValueError("未知模型状态坐标")
        expected_unit = ("mm" if self.state_coordinate_frame ==
                         "robot_planar_mm_v1" else "px")
        if self.state_length_unit != expected_unit:
            raise ValueError("state_length_unit 与模型状态坐标不一致")
        if self.node_order != "base_to_tip":
            raise ValueError("node_order 必须为 base_to_tip")
        if not math.isfinite(float(self.robot_diameter_mm)) or \
                float(self.robot_diameter_mm) <= 0:
            raise ValueError("robot_diameter_mm 必须是正有限值")
        if self.mask_area_reference_px is not None and (
                not math.isfinite(float(self.mask_area_reference_px)) or
                float(self.mask_area_reference_px) <= 0):
            raise ValueError("mask_area_reference_px 必须是正有限值")
        if self.base_side not in {"top", "bottom", "left", "right"}:
            raise ValueError("base_side 必须是 top/bottom/left/right")
        lengths = tuple(float(value) for value in self.segment_lengths)
        if not lengths or any(not math.isfinite(value) or value <= 0
                              for value in lengths):
            raise ValueError("segment_lengths 必须是非空正有限数组")
        object.__setattr__(self, "frame_size_wh", (width, height))
        object.__setattr__(self, "roi_xywh", (x, y, roi_width, roi_height))
        object.__setattr__(self, "segment_lengths", lengths)
        object.__setattr__(self, "segment_params", dict(self.segment_params))

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())

    @property
    def ready_for_anchor(self) -> bool:
        return (self.segmentation_method == "backlight" or
                bool(self.background_path and self.background_sha256))

    def updated(self, **changes) -> "ValidationSetup":
        """产生新 setup 身份；调用方据此使旧 Anchor 和 Plan 失效。"""
        return replace(self, setup_id=uuid.uuid4().hex,
                       created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                       **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ValidationSetup":
        data = dict(value)
        data["frame_size_wh"] = tuple(data["frame_size_wh"])
        data["roi_xywh"] = tuple(data["roi_xywh"])
        data["segment_lengths"] = tuple(data.get("segment_lengths", (1.0, 1.0)))
        return cls(**{key: item for key, item in data.items()
                      if key in cls.__dataclass_fields__})
