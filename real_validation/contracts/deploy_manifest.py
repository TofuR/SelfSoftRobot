"""deploy_manifest.json 的数据契约与读取(修 B3)。

把部署所需的隐式知识显式化:action_scale_kpa(kPa 上界,训练时 npz 的 /hi6)、
train_dt(实测采样周期)、mask_source(在线只允许匹配的源)、segment_params(分割参数指纹)、
camera 指纹、k_safe_table_px(视野认证表)。由 scripts/utils/build_deploy_manifest.py
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

from .models import (normalize_channel_equalities,
                     validate_hardware_action_contract)

REQUIRED = (
    "checkpoint_sha256", "action_scale_kpa", "channel_map", "train_dt_nominal_s",
    "mask_source", "n_nodes", "window_size", "z_dim", "episode_len",
    "action_dim", "encoder_type", "hidden_dim", "n_scales",
)


@dataclass(frozen=True)
class DeployManifest:
    schema_version: int = 1
    checkpoint_sha256: str | None = None
    action_scale_kpa: tuple[float, ...] | None = None
    channel_map: tuple[int, ...] | None = None
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
        missing = [name for name in REQUIRED if getattr(self, name) is None]
        if missing:
            raise ValueError(f"deploy_manifest 缺必填字段: {missing}")
        if self.action_scale_kpa is not None:
            scale = tuple(float(v) for v in self.action_scale_kpa)
            if len(scale) != self.action_dim or any(
                    v <= 0 or not math.isfinite(v) for v in scale):
                raise ValueError("action_scale_kpa 必须是 action_dim 个正数")
            object.__setattr__(self, "action_scale_kpa", scale)
        equalities = normalize_channel_equalities(self.channel_equalities)
        if equalities:
            if self.action_scale_kpa is None:
                raise ValueError("channel_equalities 要求 action_scale_kpa")
        expansion = validate_hardware_action_contract(
            self.action_dim, self.channel_map, equalities, self.action_expansion6)
        if self.channel_map is not None:
            object.__setattr__(self, "channel_map", tuple(int(v) for v in self.channel_map))
        object.__setattr__(self, "channel_equalities", equalities)
        object.__setattr__(self, "action_expansion6", expansion)
        if self.train_sequences is not None:
            object.__setattr__(self, "train_sequences", tuple(self.train_sequences))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DeployManifest":
        return cls(**{k: v for k, v in value.items()
                      if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "DeployManifest":
        with open(path, "r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} 顶层必须是对象")
        return cls.from_dict(payload)
