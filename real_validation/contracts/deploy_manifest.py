"""deploy_manifest.json 的数据契约与读取(修 B3)。

把部署所需的隐式知识显式化:action_scale_kpa(kPa 上界,训练时 npz 的 /hi6)、
train_dt(实测采样周期)、mask_source(在线只允许匹配的源)、segment_params(分割参数指纹)、
camera 指纹、k_safe_table_px(视野认证表)和planning_displacement_px_p95(实测形态位移表)。由 scripts/utils/build_deploy_manifest.py
从已有实验生成;工作台只读。

缺 manifest 或缺关键字段时:**fail-closed 阻断规划**(action_scale_kpa 缺失不能用
or 1.0 回退 —— 单位 bug 是活的,kPa 0-150 直接除 ≈1.0 的 norm_factor 喂进 [0,1]
训练域;回退会把 OOD 固化成"默认正确",且错误单位的 plan 会被存档 replay 成假工件)。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import (channel_equalities_from_sources,
                     validate_hardware_action_contract)

REQUIRED = (
    "checkpoint_sha256", "action_scale_kpa", "channel_map", "train_dt_nominal_s",
    "mask_source", "n_nodes", "window_size", "z_dim", "episode_len",
    "action_dim", "encoder_type", "hidden_dim", "n_scales",
    "node_order", "spatial_propagation_direction", "gl_kernel_alignment",
    "model_contract_version",
)


@dataclass(frozen=True)
class DeployManifest:
    schema_version: int = 2
    checkpoint_sha256: str | None = None
    action_scale_kpa: tuple[float, ...] | None = None
    channel_map: tuple[int, ...] | None = None
    channel_source6: tuple[int, ...] = ()
    channel_equalities: tuple[tuple[int, int], ...] = ()
    action_expansion6: tuple[int, ...] = ()
    train_dt_nominal_s: float | None = None
    train_dt_measured_s: float | None = None
    train_dt_std_s: float | None = None
    mask_source: str | None = None
    mask_source_provenance: str | None = None
    segment_params: dict[str, Any] | None = None
    camera: dict[str, Any] | None = None
    reference_frame: str | None = None
    reference_frame_sha256: str | None = None
    mask_area_median_px: int | None = None
    registration_residual_max_px: float = 2.0
    k_safe_table_px: dict[str, int] | None = None
    k_safe_table: dict[str, int] | None = None
    k_safe_unit: str | None = None
    planning_displacement_px_p95: dict[str, float] | None = None
    planning_displacement_p95: dict[str, float] | None = None
    robot_diameter_mm: float | None = None
    robot_diameter_px: float | None = None
    mm_per_px: float | None = None
    state_coordinate_frame: str = "camera_pixel_v1"
    state_length_unit: str = "px"
    node_order: str = "base_to_tip"
    spatial_propagation_direction: str = "base_to_tip"
    gl_kernel_alignment: str = "current_at_window_end"
    model_contract_version: int = 2
    train_sequences: tuple[str, ...] = ()
    n_nodes: int | None = None
    window_size: int | None = None
    z_dim: int | None = None
    episode_len: int | None = None
    action_dim: int | None = None
    encoder_type: str | None = None
    hidden_dim: int | None = None
    n_scales: int | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) != 2:
            raise ValueError("deploy_manifest schema_version 必须为 2")
        missing = [name for name in REQUIRED if getattr(self, name) is None]
        if missing:
            raise ValueError(f"deploy_manifest 缺必填字段: {missing}")
        if self.action_scale_kpa is not None:
            scale = tuple(float(v) for v in self.action_scale_kpa)
            if len(scale) != self.action_dim or any(
                    v <= 0 or not math.isfinite(v) for v in scale):
                raise ValueError("action_scale_kpa 必须是 action_dim 个正数")
            object.__setattr__(self, "action_scale_kpa", scale)
        sources, expansion = validate_hardware_action_contract(
            self.action_dim, self.channel_map, self.channel_equalities,
            self.action_expansion6, self.channel_source6)
        equalities = channel_equalities_from_sources(sources) if sources else ()
        if equalities and self.action_scale_kpa is None:
            raise ValueError("通道来源合同要求 action_scale_kpa")
        if self.channel_map is not None:
            object.__setattr__(self, "channel_map", tuple(int(v) for v in self.channel_map))
        object.__setattr__(self, "channel_source6", sources)
        object.__setattr__(self, "channel_equalities", equalities)
        object.__setattr__(self, "action_expansion6", expansion)
        if self.train_sequences is not None:
            object.__setattr__(self, "train_sequences", tuple(self.train_sequences))
        if self.planning_displacement_px_p95 is not None:
            table = {str(int(k)): float(v)
                     for k, v in self.planning_displacement_px_p95.items()}
            if any(int(k) <= 0 or value <= 0 or not math.isfinite(value)
                   for k, value in table.items()):
                raise ValueError("planning_displacement_px_p95需要正步数和正有限距离")
            object.__setattr__(self, "planning_displacement_px_p95", table)
        if self.planning_displacement_p95 is not None:
            table = {str(int(k)): float(v)
                     for k, v in self.planning_displacement_p95.items()}
            if any(int(k) <= 0 or value <= 0 or not math.isfinite(value)
                   for k, value in table.items()):
                raise ValueError("planning_displacement_p95需要正步数和正有限距离")
            object.__setattr__(self, "planning_displacement_p95", table)
        scale_values = (self.robot_diameter_mm, self.robot_diameter_px, self.mm_per_px)
        if any(value is not None for value in scale_values):
            if any(value is None or float(value) <= 0 or not math.isfinite(float(value))
                   for value in scale_values):
                raise ValueError("直径尺度合同需要三个正有限值")
            expected = float(self.robot_diameter_mm) / float(self.robot_diameter_px)
            if abs(expected - float(self.mm_per_px)) > max(1e-6, expected * 1e-5):
                raise ValueError("mm_per_px 与 robot_diameter_mm/px 不一致")
        allowed_frames = {
            "camera_pixel_v1": "px",
            "robot_planar_mm_v1": "mm",
        }
        if self.state_coordinate_frame not in allowed_frames:
            raise ValueError(f"未知 state_coordinate_frame: {self.state_coordinate_frame}")
        if self.state_length_unit != allowed_frames[self.state_coordinate_frame]:
            raise ValueError("state_length_unit 与 state_coordinate_frame 不一致")
        if self.node_order != "base_to_tip" or \
                self.spatial_propagation_direction != "base_to_tip":
            raise ValueError("节点合同与空间传播必须统一为 base_to_tip")
        if self.gl_kernel_alignment != "current_at_window_end":
            raise ValueError("GL 权重 w0 必须对齐窗口末尾的当前动作")
        if int(self.model_contract_version) != 2:
            raise ValueError("model_contract_version 必须为 2")
        if self.state_coordinate_frame == "robot_planar_mm_v1" and \
                self.robot_diameter_mm is None:
            raise ValueError("robot_planar_mm_v1 需要 robot_diameter_mm")
        if self.k_safe_table is not None:
            table = {str(key): int(value) for key, value in self.k_safe_table.items()}
            if any(value <= 0 for value in table.values()):
                raise ValueError("k_safe_table 的 K 必须为正整数")
            if self.k_safe_unit != self.state_length_unit:
                raise ValueError("k_safe_unit 与 state_length_unit 不一致")
            object.__setattr__(self, "k_safe_table", table)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeployManifest":
        missing = [name for name in ("schema_version",) + REQUIRED
                   if name not in value]
        if missing:
            raise ValueError(f"deploy_manifest 缺必填字段: {missing}")
        return cls(**{k: v for k, v in value.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "DeployManifest":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} 顶层必须是对象")
        return cls.from_dict(payload)
