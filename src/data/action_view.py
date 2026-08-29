"""六维原始动作到模型动作视图的显式合同。

采集与 NPZ 保留硬件动作 ``actions(T, 6)``；训练数据集只把独立通道投影给模型。
合同必须在同一数据目录的所有 NPZ 中一致，避免把不同通道语义静默混入一次训练。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import glob
import json
import os
from typing import Iterable

import numpy as np


def _scalar_int(data, key, default):
    if key not in data:
        return default
    return int(np.asarray(data[key]).item())


def _json_pairs(data, key):
    if key not in data:
        return ()
    raw = np.asarray(data[key]).item()
    value = json.loads(str(raw))
    return tuple(tuple(int(v) for v in pair) for pair in value)


def parse_action_channels(value) -> tuple[int, ...] | None:
    """解析 CLI/config 值；``auto``/None 返回 None，由调用方区分。"""
    if value is None or value == "auto":
        return None
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
        if not fields:
            raise ValueError("action_channels 不能为空")
        channels = tuple(int(field) for field in fields)
    else:
        channels = tuple(int(field) for field in value)
    if len(set(channels)) != len(channels) or any(channel < 0 for channel in channels):
        raise ValueError("action_channels 必须是互不重复的非负通道")
    return channels


def normalize_channel_sources(sources=None, equalities=()) -> tuple[int, ...]:
    if sources is None:
        values = list(range(6))
        for leader, follower in equalities or ():
            leader, follower = int(leader), int(follower)
            if leader == follower or leader not in range(6) or follower not in range(6):
                raise ValueError("channel_equalities 必须引用两个不同的 0..5 通道")
            values[follower] = leader
    else:
        values = tuple(int(value) for value in sources)
        if len(values) != 6 or any(value not in range(6) for value in values):
            raise ValueError("channel_source6 必须是 6 个 0..5 通道下标")

    def root(start):
        seen = set()
        current = start
        while values[current] != current:
            if current in seen:
                raise ValueError("channel_source6 不能包含循环")
            seen.add(current)
            current = values[current]
        return current

    return tuple(root(channel) for channel in range(6))


def source_equalities(sources):
    normalized = normalize_channel_sources(sources)
    return tuple((source, channel) for channel, source in enumerate(normalized)
                 if channel != source)


def source_model_channels(sources):
    normalized = normalize_channel_sources(sources)
    return tuple(channel for channel, source in enumerate(normalized)
                 if channel == source)


def source_expansion6(sources, channels):
    lookup = {int(channel): index for index, channel in enumerate(channels)}
    try:
        return tuple(lookup[source] for source in normalize_channel_sources(sources))
    except KeyError as error:
        raise ValueError(f"根通道 ch{int(error.args[0])} 未进入模型动作") from error


@dataclass(frozen=True)
class ActionViewContract:
    raw_action_dim: int
    model_action_channels: tuple[int, ...]
    channel_source6: tuple[int, ...] = ()
    channel_equalities: tuple[tuple[int, int], ...] = ()
    action_expansion6: tuple[int, ...] = ()

    def __post_init__(self):
        if self.raw_action_dim <= 0:
            raise ValueError("raw_action_dim 必须为正")
        channels = tuple(int(value) for value in self.model_action_channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("model_action_channels 必须非空且互不重复")
        if any(value < 0 or value >= self.raw_action_dim for value in channels):
            raise ValueError("model_action_channels 超出原始动作维度")

        if self.channel_source6:
            if self.raw_action_dim != 6:
                raise ValueError("channel_source6 当前要求原始硬件动作维度为 6")
            sources = normalize_channel_sources(self.channel_source6)
        elif self.channel_equalities:
            sources = normalize_channel_sources(equalities=self.channel_equalities)
        else:
            sources = ()

        if sources:
            expected_channels = source_model_channels(sources)
            if channels != expected_channels:
                raise ValueError(
                    f"model_action_channels={channels} 与 channel_source6 根通道 "
                    f"{expected_channels} 不一致")
            equalities = source_equalities(sources)
            expansion = source_expansion6(sources, channels)
            supplied_equalities = tuple(tuple(int(v) for v in pair)
                                        for pair in self.channel_equalities)
            if supplied_equalities and supplied_equalities != equalities:
                raise ValueError("channel_equalities 与 channel_source6 不一致")
            supplied_expansion = tuple(int(v) for v in self.action_expansion6)
            if supplied_expansion and supplied_expansion != expansion:
                raise ValueError("action_expansion6 与 channel_source6 不一致")
        else:
            equalities = ()
            expansion = tuple(int(v) for v in self.action_expansion6)

        object.__setattr__(self, "model_action_channels", channels)
        object.__setattr__(self, "channel_source6", sources)
        object.__setattr__(self, "channel_equalities", equalities)
        object.__setattr__(self, "action_expansion6", expansion)

    @property
    def model_action_dim(self) -> int:
        return len(self.model_action_channels)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["model_action_dim"] = self.model_action_dim
        return value


def contract_from_npz(path: str, action_channels=None) -> ActionViewContract:
    explicit = parse_action_channels(action_channels)
    use_stored_view = action_channels == "auto"
    with np.load(path, allow_pickle=False) as data:
        if "actions" not in data:
            raise ValueError(f"NPZ 缺少 actions: {path}")
        actions = data["actions"]
        if actions.ndim != 2:
            raise ValueError(f"actions 必须是二维数组: {path}")
        raw_dim = _scalar_int(data, "raw_action_dim", actions.shape[1])
        if raw_dim != actions.shape[1]:
            raise ValueError(
                f"raw_action_dim={raw_dim} 与 actions 宽度 {actions.shape[1]} 不同: {path}")
        stored = (tuple(int(v) for v in np.asarray(data["model_action_channels"]).tolist())
                  if "model_action_channels" in data else None)
        legacy = (tuple(int(v) for v in np.asarray(data["independent_channels"]).tolist())
                  if "independent_channels" in data else None)
        channels = (explicit or ((stored or legacy) if use_stored_view else None)
                    or tuple(range(raw_dim)))
        stored_model_dim = _scalar_int(data, "model_action_dim", len(channels))
        if action_channels is not None and stored_model_dim != len(channels):
            raise ValueError(
                f"model_action_dim={stored_model_dim} 与动作视图宽度 {len(channels)} 不同: {path}")
        if explicit is not None and stored is not None and explicit != stored:
            raise ValueError(
                f"显式 action_channels={explicit} 与 NPZ 合同 {stored} 不一致: {path}")

        equalities = _json_pairs(data, "channel_equalities")
        expansion = (tuple(int(v) for v in np.asarray(data["action_expansion6"]).tolist())
                     if "action_expansion6" in data else ())
        sources = (tuple(int(v) for v in np.asarray(data["channel_source6"]).tolist())
                   if "channel_source6" in data else None)
        if sources is None and expansion and stored:
            sources = tuple(stored[index] for index in expansion)
        if action_channels is None:
            sources, equalities, expansion = None, (), ()
    return ActionViewContract(
        raw_dim, channels, channel_source6=tuple(sources or ()),
        channel_equalities=equalities, action_expansion6=expansion)


def resolve_action_contract(data_dir: str, action_channels=None) -> ActionViewContract:
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files in {data_dir}")
    contracts = [contract_from_npz(path, action_channels) for path in paths]
    first = contracts[0]
    for path, contract in zip(paths[1:], contracts[1:]):
        if contract != first:
            raise ValueError(
                f"同一数据目录的 action view 合同不一致: {paths[0]} vs {path}")
    return first


def project_actions(actions, channels: Iterable[int]) -> np.ndarray:
    values = np.asarray(actions)
    selected = tuple(int(value) for value in channels)
    if values.ndim != 2:
        raise ValueError("actions 必须为 (T,D)")
    if any(value < 0 or value >= values.shape[1] for value in selected):
        raise ValueError("动作视图通道超出 actions 宽度")
    return values[:, selected]
