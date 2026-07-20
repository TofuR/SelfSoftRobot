"""验证工作台的稳定数据契约。

计划始终保存为六通道命令，模型自身的动作维度通过 ``channel_map`` 显式映射；
因此 1/3/6 通道模型可以共用执行器，同时不会静默补零掩盖维度错误。
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .io import stable_digest

SCHEMA_VERSION = 1
N_HARDWARE_CHANNELS = 6


def _finite_vector(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} 必须有 {size} 个值，实际为 {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 含 NaN/Inf")
    return result


def _vec6(values: Iterable[float], name: str) -> tuple[float, ...]:
    return _finite_vector(values, N_HARDWARE_CHANNELS, name)


@dataclass(frozen=True)
class ModelDescriptor:
    checkpoint: str
    checkpoint_hash: str
    model_type: str
    action_dim: int
    n_nodes: int
    history_steps: int
    model_class: str = ""
    k_train: int | None = None
    k_safe: int | None = None
    data_dir: str | None = None
    normalization: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim not in (1, 3, 6):
            raise ValueError("实机工作台当前只接受 action_dim=1/3/6")
        if self.n_nodes <= 0 or self.history_steps <= 0:
            raise ValueError("n_nodes 与 history_steps 必须为正数")
        if self.k_safe is not None and self.k_safe <= 0:
            raise ValueError("k_safe 必须为正数")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelDescriptor":
        data = dict(value)
        data.pop("schema_version", None)
        return cls(**data)


@dataclass(frozen=True)
class Anchor:
    state: tuple[tuple[float, ...], ...]
    action_history: tuple[tuple[float, ...], ...]
    frame_id: str = "model"
    timestamp: float = field(default_factory=time.time)
    anchor_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "unknown"
    quality: float | None = None
    state_space: str = "model_normalized"
    action_units: str = "kpa"

    def __post_init__(self) -> None:
        state = tuple(tuple(float(v) for v in node) for node in self.state)
        history = tuple(tuple(float(v) for v in action) for action in self.action_history)
        if not state or any(len(node) not in (2, 3) for node in state):
            raise ValueError("anchor state 必须是非空的 N×2 或 N×3 节点")
        if any(not all(math.isfinite(v) for v in node) for node in state):
            raise ValueError("anchor state 含 NaN/Inf")
        if any(not action for action in history):
            raise ValueError("action_history 中不能有空动作")
        if any(not all(math.isfinite(v) for v in action) for action in history):
            raise ValueError("action_history 含 NaN/Inf")
        if self.state_space not in {"model", "model_normalized"}:
            raise ValueError("anchor state_space 必须是 model 或 model_normalized")
        if self.action_units not in {"kpa", "model_normalized"}:
            raise ValueError("anchor action_units 必须是 kpa 或 model_normalized")
        if any(not all(math.isfinite(v) for v in action) for action in history):
            raise ValueError("action_history 含 NaN/Inf")
        if self.quality is not None and not math.isfinite(self.quality):
            raise ValueError("anchor quality 必须为有限值")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "action_history", history)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Anchor":
        data = dict(value)
        data.pop("schema_version", None)
        data["state"] = tuple(tuple(row) for row in data["state"])
        data["action_history"] = tuple(tuple(row) for row in data["action_history"])
        return cls(**data)


@dataclass(frozen=True)
class ScenePrimitive:
    kind: str
    frame_id: str
    geometry: dict[str, Any]
    name: str = ""
    safety_margin: float = 0.0
    primitive_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.kind not in {
            "target_point", "target_circle", "target_rectangle", "target_polygon",
            "obstacle_circle", "obstacle_aabb", "obstacle_polygon", "obstacle_mask",
            "waypoint", "gate", "observation_port", "workspace",
        }:
            raise ValueError(f"未知 scene primitive: {self.kind}")
        if not self.frame_id:
            raise ValueError("scene primitive 必须声明 frame_id")
        if self.safety_margin < 0:
            raise ValueError("safety_margin 不能为负")


@dataclass(frozen=True)
class Scene:
    name: str = "untitled"
    primitives: tuple[ScenePrimitive, ...] = ()
    dimension: int = 2
    revision: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError("scene dimension 只能是 2 或 3")
        object.__setattr__(self, "primitives", tuple(self.primitives))

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())

    def with_primitive(self, primitive: ScenePrimitive) -> "Scene":
        return Scene(name=self.name, primitives=self.primitives + (primitive,),
                     dimension=self.dimension)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "dimension": self.dimension,
            "revision": self.revision,
            "primitives": [asdict(item) for item in self.primitives],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scene":
        return cls(
            name=value.get("name", "untitled"),
            dimension=int(value.get("dimension", 2)),
            revision=value.get("revision", uuid.uuid4().hex),
            primitives=tuple(ScenePrimitive(**item) for item in value.get("primitives", [])),
        )


@dataclass(frozen=True)
class SafetyPolicy:
    pressure_min6: tuple[float, ...] = (0.0,) * N_HARDWARE_CHANNELS
    pressure_max6: tuple[float, ...] = (200.0,) * N_HARDWARE_CHANNELS
    rise_rate6: tuple[float, ...] = (100.0,) * N_HARDWARE_CHANNELS
    fall_rate6: tuple[float, ...] = (100.0,) * N_HARDWARE_CHANNELS
    initial_action6: tuple[float, ...] = (0.0,) * N_HARDWARE_CHANNELS
    ack_timeout_s: float = 1.0
    required_groups: tuple[int, ...] = (1, 2)
    pause_policy: str = "zero"

    def __post_init__(self) -> None:
        for field_name in ("pressure_min6", "pressure_max6", "rise_rate6",
                           "fall_rate6", "initial_action6"):
            object.__setattr__(self, field_name, _vec6(getattr(self, field_name), field_name))
        if any(lo > hi for lo, hi in zip(self.pressure_min6, self.pressure_max6)):
            raise ValueError("pressure_min6 不能大于 pressure_max6")
        if any(value < 0 or value > 500
               for value in self.pressure_min6 + self.pressure_max6):
            raise ValueError("压力安全范围必须位于阀的 0..500 kPa 物理范围")
        if any(value < lo or value > hi for value, lo, hi in zip(
                self.initial_action6, self.pressure_min6, self.pressure_max6)):
            raise ValueError("initial_action6 必须位于配置的压力范围内")
        if any(rate < 0 for rate in self.rise_rate6 + self.fall_rate6):
            raise ValueError("压力变化速率不能为负")
        if self.ack_timeout_s <= 0:
            raise ValueError("ack_timeout_s 必须为正数")
        if not set(self.required_groups).issubset({1, 2}):
            raise ValueError("required_groups 只能包含 1/2")
        if self.pause_policy not in {"zero", "hold"}:
            raise ValueError("pause_policy 只能是 zero 或 hold")

    @property
    def digest(self) -> str:
        return stable_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SafetyPolicy":
        data = dict(value)
        data.pop("schema_version", None)
        for name in ("pressure_min6", "pressure_max6", "rise_rate6",
                     "fall_rate6", "initial_action6", "required_groups"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


@dataclass(frozen=True)
class ActionPlan:
    actions6: tuple[tuple[float, ...], ...]
    step_interval_s: float
    model_action_dim: int
    channel_map: tuple[int, ...]
    model_hash: str
    scene_digest: str
    anchor_id: str
    safety_digest: str
    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    random_seed: int | None = None
    predicted_states_path: str | None = None
    loss_terms: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = tuple(_vec6(action, "actions6 row") for action in self.actions6)
        if not actions:
            raise ValueError("计划至少需要一个动作")
        object.__setattr__(self, "actions6", actions)
        object.__setattr__(self, "channel_map", tuple(int(i) for i in self.channel_map))
        if self.step_interval_s <= 0:
            raise ValueError("step_interval_s 必须为正数")

    @property
    def horizon(self) -> int:
        return len(self.actions6)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ActionPlan":
        data = dict(value)
        data.pop("schema_version", None)
        data["actions6"] = tuple(tuple(row) for row in data["actions6"])
        data["channel_map"] = tuple(data["channel_map"])
        return cls(**data)
