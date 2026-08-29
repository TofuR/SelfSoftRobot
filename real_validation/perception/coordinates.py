"""机器人平面坐标与相机像素之间的可逆相似变换。

模型状态使用 ``robot_planar_mm_v1``：基座为原点，纵轴从基座指向机器人末端，
横轴与纵轴正交，长度单位为毫米。相机 ROI 只属于图像处理层；裁剪偏移先恢复到
源图像像素，再通过本模块进入模型坐标。

一次采集序列或一次在线实验只建立一个固定变换。固定变换保留机器人运动产生的
平移和弯曲，同时吸收相机平面内的平移、旋转与统一缩放。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


CAMERA_PIXEL_FRAME = "camera_pixel_v1"
ROBOT_PLANAR_FRAME = "robot_planar_mm_v1"
TRANSFORM_SCHEMA_VERSION = 1


def _unit_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(2)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-9:
        raise ValueError(f"{name} 必须是非零有限二维向量")
    return vector / length


@dataclass(frozen=True)
class SkeletonFrameTransform:
    """``camera_pixel_v1`` ↔ ``robot_planar_mm_v1`` 的相似变换。

    ``axial_axis_camera`` 是相机图像中基座指向末端的单位向量；横轴取其顺时针
    90 度方向。机器人坐标按 ``[lateral_mm, axial_mm]`` 排列。
    """

    origin_camera_px: tuple[float, float]
    axial_axis_camera: tuple[float, float]
    pixels_per_mm: float
    source: str = "unknown"
    frame_id: str = ROBOT_PLANAR_FRAME
    schema_version: int = TRANSFORM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        origin = tuple(float(v) for v in self.origin_camera_px)
        if len(origin) != 2 or not all(math.isfinite(v) for v in origin):
            raise ValueError("origin_camera_px 必须是两个有限值")
        axial = _unit_vector(self.axial_axis_camera, "axial_axis_camera")
        scale = float(self.pixels_per_mm)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("pixels_per_mm 必须是正有限数")
        if self.frame_id != ROBOT_PLANAR_FRAME:
            raise ValueError(f"frame_id 必须是 {ROBOT_PLANAR_FRAME}")
        if int(self.schema_version) != TRANSFORM_SCHEMA_VERSION:
            raise ValueError(
                f"不支持 SkeletonFrameTransform schema {self.schema_version}")
        object.__setattr__(self, "origin_camera_px", origin)
        object.__setattr__(self, "axial_axis_camera",
                           tuple(float(v) for v in axial))
        object.__setattr__(self, "pixels_per_mm", scale)

    @property
    def lateral_axis_camera(self) -> np.ndarray:
        axial = np.asarray(self.axial_axis_camera, dtype=np.float64)
        return np.asarray((axial[1], -axial[0]), dtype=np.float64)

    @property
    def camera_basis(self) -> np.ndarray:
        """列向量依次为横轴、纵轴，shape=(2,2)。"""
        return np.column_stack((self.lateral_axis_camera,
                                np.asarray(self.axial_axis_camera,
                                           dtype=np.float64)))

    def camera_to_model(self, points) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] < 2:
            raise ValueError("camera points 最后一维至少为2")
        xy = values[..., :2]
        centered = xy - np.asarray(self.origin_camera_px, dtype=np.float64)
        result = centered @ self.camera_basis / self.pixels_per_mm
        return result.astype(np.float32)

    def model_to_camera(self, points) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1] < 2:
            raise ValueError("model points 最后一维至少为2")
        xy = values[..., :2]
        result = (xy * self.pixels_per_mm) @ self.camera_basis.T
        result += np.asarray(self.origin_camera_px, dtype=np.float64)
        return result.astype(np.float32)

    def camera_length_to_model(self, value_px: float) -> float:
        return float(value_px) / self.pixels_per_mm

    def model_length_to_camera(self, value_mm: float) -> float:
        return float(value_mm) * self.pixels_per_mm

    def camera_to_model_matrix(self) -> np.ndarray:
        linear = self.camera_basis.T / self.pixels_per_mm
        origin = np.asarray(self.origin_camera_px, dtype=np.float64)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = linear
        matrix[:2, 2] = -linear @ origin
        return matrix

    def model_to_camera_matrix(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_model_matrix())

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["camera_to_model_matrix"] = self.camera_to_model_matrix().tolist()
        value["model_to_camera_matrix"] = self.model_to_camera_matrix().tolist()
        value["length_unit"] = "mm"
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SkeletonFrameTransform":
        return cls(
            origin_camera_px=tuple(value["origin_camera_px"]),
            axial_axis_camera=tuple(value["axial_axis_camera"]),
            pixels_per_mm=float(value["pixels_per_mm"]),
            source=str(value.get("source", "unknown")),
            frame_id=str(value.get("frame_id", ROBOT_PLANAR_FRAME)),
            schema_version=int(value.get("schema_version", TRANSFORM_SCHEMA_VERSION)),
        )


def _skeleton_xy(positions) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[1] >= 2:
            return values[:, :2][None, ...]
        if values.shape[0] >= 2:
            return values[:2, :].T[None, ...]
    if values.ndim == 3:
        if values.shape[1] >= 2 and values.shape[2] > 3:
            return values[:, :2, :].transpose(0, 2, 1)
        if values.shape[2] >= 2:
            return values[..., :2]
    raise ValueError(f"骨架形状必须是 (N,2/3)、(2/3,N)、(T,2/3,N) 或 (T,N,2/3)，实际 {values.shape}")


def estimate_skeleton_frame(positions_camera_px, *, robot_diameter_mm: float,
                            robot_diameter_px: float,
                            source: str = "skeleton") -> SkeletonFrameTransform:
    """从一帧或一段 base→tip 骨架估计固定机器人坐标。

    序列输入使用各帧基座和基座附近切向的稳健中位数。单帧输入用于在线锚定；
    得到的变换随后由当前实验固定保存并复用。
    """
    xy = _skeleton_xy(positions_camera_px)
    valid = np.isfinite(xy).all(axis=(1, 2))
    valid &= np.linalg.norm(xy[:, 0] - xy[:, -1], axis=1) > 1e-6
    xy = xy[valid]
    if not len(xy):
        raise ValueError("没有可用于建立机器人坐标的有效骨架")
    node_count = xy.shape[1]
    if node_count < 3:
        raise ValueError("建立机器人坐标至少需要3个骨架节点")
    inward = max(1, min(node_count - 2, int(round((node_count - 1) * 0.2))))
    base = xy[:, 0]
    near_base = xy[:, inward]
    directions = near_base - base
    lengths = np.linalg.norm(directions, axis=1)
    directions = directions[lengths > 1e-6] / lengths[lengths > 1e-6, None]
    if not len(directions):
        raise ValueError("基座附近节点无法确定机器人纵轴")
    reference = directions[0]
    directions[np.sum(directions * reference, axis=1) < 0] *= -1.0
    axial = _unit_vector(np.median(directions, axis=0), "序列纵轴")
    diameter_mm = float(robot_diameter_mm)
    diameter_px = float(robot_diameter_px)
    if diameter_mm <= 0 or diameter_px <= 0:
        raise ValueError("机器人直径 mm/px 必须为正数")
    return SkeletonFrameTransform(
        origin_camera_px=tuple(np.median(base, axis=0).tolist()),
        axial_axis_camera=tuple(axial.tolist()),
        pixels_per_mm=diameter_px / diameter_mm,
        source=source,
    )


def estimate_body_diameter_px(mask, skeleton_camera_px) -> float:
    """从 mask 主体中段的距离变换估计直径像素值。"""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("估计机器人像素直径需要 scipy") from error
    binary = np.asarray(mask) > 0
    skeleton = _skeleton_xy(skeleton_camera_px)[0]
    if not binary.any() or len(skeleton) < 3:
        raise ValueError("mask/骨架不足以估计机器人直径")
    start = max(1, int(math.floor(0.2 * len(skeleton))))
    stop = min(len(skeleton) - 1, int(math.ceil(0.8 * len(skeleton))))
    points = np.rint(skeleton[start:stop]).astype(np.int64)
    points[:, 0] = np.clip(points[:, 0], 0, binary.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, binary.shape[0] - 1)
    distance = distance_transform_edt(binary)
    diameters = 2.0 * distance[points[:, 1], points[:, 0]]
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    if not len(diameters):
        raise ValueError("机器人主体直径估计失败")
    return float(np.median(diameters))


def frame_transform_displacement_px(reference: SkeletonFrameTransform,
                                    current: SkeletonFrameTransform,
                                    arm_extent_mm: float) -> float:
    """把原点、方向和尺度变化汇总为机器人工作范围内的最大像素位移。"""
    extent = max(float(arm_extent_mm), 1.0)
    probes = np.asarray(((0.0, 0.0), (0.0, extent),
                         (-0.25 * extent, 0.5 * extent),
                         (0.25 * extent, 0.5 * extent)), dtype=np.float64)
    ref_px = reference.model_to_camera(probes)
    current_px = current.model_to_camera(probes)
    return float(np.linalg.norm(ref_px - current_px, axis=1).max())


def transform_positions_to_model(positions_camera_px,
                                 transform: SkeletonFrameTransform) -> np.ndarray:
    """``(T,3,N)`` 相机像素骨架转换为同形状毫米骨架。"""
    positions = np.asarray(positions_camera_px, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1] < 2:
        raise ValueError(f"positions 必须是 (T,3,N)，实际 {positions.shape}")
    xy = positions[:, :2, :].transpose(0, 2, 1)
    canonical = transform.camera_to_model(xy)
    result = np.zeros_like(positions, dtype=np.float32)
    result[:, :2, :] = canonical.transpose(0, 2, 1)
    return result


def transform_positions_to_camera(positions_model,
                                  transform: SkeletonFrameTransform) -> np.ndarray:
    """``(T,3,N)`` 毫米骨架转换为同形状相机像素骨架。"""
    positions = np.asarray(positions_model, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1] < 2:
        raise ValueError(f"positions 必须是 (T,3,N)，实际 {positions.shape}")
    xy = positions[:, :2, :].transpose(0, 2, 1)
    camera = transform.model_to_camera(xy)
    result = np.zeros_like(positions, dtype=np.float32)
    result[:, :2, :] = camera.transpose(0, 2, 1)
    return result
