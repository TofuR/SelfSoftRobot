"""验证工作台的稳定数据契约。

计划始终保存为六通道命令，模型自身的动作维度通过 ``channel_map`` 显式映射；
因此 1..6 通道模型可以共用执行器，同时不会静默补零掩盖维度错误。
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
CHANNEL_EQUALITY_TOLERANCE = 0.5


def _finite_vector(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} 必须有 {size} 个值，实际为 {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} 含 NaN/Inf")
    return result


def _vec6(values: Iterable[float], name: str) -> tuple[float, ...]:
    return _finite_vector(values, N_HARDWARE_CHANNELS, name)


def normalize_channel_sources(sources=None, *, pairs=(),
                              size: int = N_HARDWARE_CHANNELS) -> tuple[int, ...]:
    """规范化硬件来源图；链式关系压平到根，循环 fail-closed。"""
    if sources is None:
        values = list(range(size))
        for item in pairs or ():
            values_pair = tuple(item)
            if len(values_pair) != 2:
                raise ValueError("每个 channel equality 必须是 [leader, follower]")
            leader, follower = int(values_pair[0]), int(values_pair[1])
            if leader == follower or leader not in range(size) or follower not in range(size):
                raise ValueError(f"channel equality 必须引用两个不同的 0..{size - 1} 通道")
            values[follower] = leader
    else:
        values = tuple(int(value) for value in sources)
        if len(values) != size or any(value not in range(size) for value in values):
            raise ValueError(f"channel_source6 必须是 {size} 个 0..{size - 1} 通道下标")

    def root(start):
        seen = set()
        current = start
        while values[current] != current:
            if current in seen:
                raise ValueError("channel_source6 不能包含循环")
            seen.add(current)
            current = values[current]
        return current

    return tuple(root(channel) for channel in range(size))


def channel_equalities_from_sources(sources, *,
                                    size: int = N_HARDWARE_CHANNELS
                                    ) -> tuple[tuple[int, int], ...]:
    normalized = normalize_channel_sources(sources, size=size)
    return tuple((source, channel) for channel, source in enumerate(normalized)
                 if channel != source)


def normalize_channel_equalities(pairs: Iterable[Iterable[int]] | None,
                                 *, size: int = N_HARDWARE_CHANNELS
                                 ) -> tuple[tuple[int, int], ...]:
    """旧 pair 合同兼容入口；内部统一为来源图。"""
    return channel_equalities_from_sources(
        normalize_channel_sources(pairs=pairs, size=size), size=size)


def apply_channel_sources(values: Iterable[float], sources,
                          *, size: int = N_HARDWARE_CHANNELS
                          ) -> tuple[float, ...]:
    vector = _finite_vector(values, size, "action")
    normalized = normalize_channel_sources(sources, size=size)
    return tuple(vector[source] for source in normalized)


def apply_channel_equalities(values: Iterable[float], pairs,
                             *, size: int = N_HARDWARE_CHANNELS
                             ) -> tuple[float, ...]:
    return apply_channel_sources(
        values, normalize_channel_sources(pairs=pairs, size=size), size=size)


def channel_source_residuals(values: Iterable[float], sources,
                             *, size: int = N_HARDWARE_CHANNELS
                             ) -> tuple[float, ...]:
    vector = _finite_vector(values, size, "action")
    return tuple(abs(vector[source] - vector[channel])
                 for source, channel in channel_equalities_from_sources(
                     sources, size=size))


def channel_equality_residuals(values: Iterable[float], pairs,
                               *, size: int = N_HARDWARE_CHANNELS
                               ) -> tuple[float, ...]:
    return channel_source_residuals(
        values, normalize_channel_sources(pairs=pairs, size=size), size=size)


def hardware_action_expansion(channel_map, pairs=(), channel_sources=None) -> tuple[int, ...]:
    """返回六个硬件通道各自读取的模型动作列；未驱动通道为 -1。"""
    mapping = tuple(int(value) for value in channel_map)
    lookup = {channel: index for index, channel in enumerate(mapping)}
    sources = normalize_channel_sources(channel_sources or None, pairs=pairs)
    return tuple(lookup.get(source, -1) for source in sources)


def validate_hardware_action_contract(action_dim, channel_map, pairs=(),
                                      action_expansion6=(), channel_sources=None
                                      ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """校验模型独立通道到六通道硬件的来源与展开关系。"""
    if channel_map is None:
        if pairs or channel_sources:
            raise ValueError("通道来源合同要求显式 channel_map")
        return (), ()
    mapping = tuple(int(value) for value in channel_map)
    if len(mapping) != int(action_dim) or len(set(mapping)) != len(mapping) or any(
            value not in range(N_HARDWARE_CHANNELS) for value in mapping):
        raise ValueError("channel_map 必须是不重复的 0..5 通道,长度等于 action_dim")
    sources = normalize_channel_sources(channel_sources or None, pairs=pairs)
    roots = tuple(channel for channel, source in enumerate(sources) if channel == source)
    constrained = bool(channel_sources) or bool(pairs)
    if constrained and mapping != roots:
        raise ValueError(
            f"channel_map={mapping} 必须等于 channel_source6 根通道 {roots}")
    expected = hardware_action_expansion(mapping, channel_sources=sources)
    expansion = tuple(int(value) for value in action_expansion6 or ())
    if expansion and expansion != expected:
        raise ValueError(
            f"action_expansion6={expansion} 与 channel_source6 推导值 {expected} 不同")
    return (sources if constrained else ()), expected


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
    # ---- P1b 新增(全部带默认值;缺 manifest 时为 None,由 preflight/planner 阻断) ----
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
    camera_fingerprint: dict[str, Any] | None = None
    reference_frame_hash: str | None = None
    k_safe_table_px: dict[str, int] | None = None
    registration_residual_max_px: float = 2.0
    provenance: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim not in range(1, N_HARDWARE_CHANNELS + 1):
            raise ValueError("实机工作台只接受 action_dim=1..6")
        if self.n_nodes <= 0 or self.history_steps <= 0:
            raise ValueError("n_nodes 与 history_steps 必须为正数")
        if self.k_safe is not None and self.k_safe <= 0:
            raise ValueError("k_safe 必须为正数")
        if self.action_scale_kpa is not None:
            values = tuple(float(v) for v in self.action_scale_kpa)
            if len(values) != self.action_dim:
                raise ValueError("action_scale_kpa 长度必须等于 action_dim")
            if any(v <= 0 or not math.isfinite(v) for v in values):
                raise ValueError("action_scale_kpa 必须全为正有限值")
            object.__setattr__(self, "action_scale_kpa", values)
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
    prev_state: tuple[tuple[float, ...], ...] | None = None   # ★P1b:s_{t-2},速度项需要
    frame_id: str = "model"
    frame_ref: str = ""                                       # ★P1b:隐藏评价流的帧引用
    timestamp: float = field(default_factory=time.time)
    anchor_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: str = "unknown"
    quality: dict[str, Any] = field(default_factory=dict)     # ★P1b:float → 标志集
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
        if self.prev_state is not None:
            prev = tuple(tuple(float(v) for v in node) for node in self.prev_state)
            if not prev or any(len(node) not in (2, 3) for node in prev):
                raise ValueError("anchor prev_state 必须是非空的 N×2 或 N×3 节点")
            if any(not all(math.isfinite(v) for v in node) for node in prev):
                raise ValueError("anchor prev_state 含 NaN/Inf")
            object.__setattr__(self, "prev_state", prev)
        if self.state_space not in {"model", "model_normalized"}:
            raise ValueError("anchor state_space 必须是 model 或 model_normalized")
        if self.action_units not in {"kpa", "model_normalized"}:
            raise ValueError("anchor action_units 必须是 kpa 或 model_normalized")
        if not isinstance(self.quality, dict):
            raise ValueError("anchor quality 必须是 dict(标志集)")
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
        if data.get("prev_state") is not None:
            data["prev_state"] = tuple(tuple(row) for row in data["prev_state"])
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
            "target_skeleton",
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

    def without_primitive(self, primitive_id: str) -> "Scene":
        """按 primitive_id 移除一个原语(B7:原来只能追加,交互式编辑无法删除)。"""
        kept = tuple(item for item in self.primitives if item.primitive_id != primitive_id)
        if len(kept) == len(self.primitives):
            raise KeyError(f"primitive_id 不存在: {primitive_id}")
        return Scene(name=self.name, primitives=kept, dimension=self.dimension)

    def replace_primitive(self, primitive_id: str, new_primitive: "ScenePrimitive") -> "Scene":
        """按 primitive_id 替换一个原语。"""
        replaced = tuple(new_primitive if item.primitive_id == primitive_id else item
                         for item in self.primitives)
        if replaced == self.primitives:
            raise KeyError(f"primitive_id 不存在: {primitive_id}")
        return Scene(name=self.name, primitives=replaced, dimension=self.dimension)

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
    pressure_max6: tuple[float, ...] = (150.0,) * N_HARDWARE_CHANNELS   # P1b:对齐训练上界(原 200)
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
    channel_source6: tuple[int, ...] = ()
    channel_equalities: tuple[tuple[int, int], ...] = ()
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
        mapping = tuple(int(i) for i in self.channel_map)
        sources, _expansion = validate_hardware_action_contract(
            self.model_action_dim, mapping, self.channel_equalities,
            channel_sources=self.channel_source6)
        equalities = channel_equalities_from_sources(sources) if sources else ()
        object.__setattr__(self, "channel_map", mapping)
        object.__setattr__(self, "channel_source6", sources)
        object.__setattr__(self, "channel_equalities", equalities)
        for step, action in enumerate(actions):
            residuals = channel_source_residuals(action, sources) if sources else ()
            if any(value > CHANNEL_EQUALITY_TOLERANCE for value in residuals):
                raise ValueError(f"计划第 {step} 步违反 channel_equalities: {residuals}")
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
        data["channel_source6"] = tuple(data.get("channel_source6", ()))
        data["channel_equalities"] = tuple(
            tuple(pair) for pair in data.get("channel_equalities", ()))
        return cls(**data)
