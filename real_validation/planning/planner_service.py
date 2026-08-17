"""GUI/CLI 共用规划服务适配层(模型动作 → 6 通道计划)。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..contracts.models import (
    ActionPlan,
    Anchor,
    ModelDescriptor,
    SafetyPolicy,
    Scene,
    apply_channel_equalities,
    normalize_channel_equalities,
)


def expand_model_actions(actions: Sequence[Sequence[float]],
                         channel_map: Sequence[int], channel_equalities=()
                         ) -> tuple[tuple[float, ...], ...]:
    mapping = tuple(int(channel) for channel in channel_map)
    equalities = normalize_channel_equalities(channel_equalities)
    if len(set(mapping)) != len(mapping) or any(channel not in range(6) for channel in mapping):
        raise ValueError("channel_map 必须是 0..5 内不重复的通道")
    expanded = []
    for step, action in enumerate(actions):
        values = tuple(float(value) for value in action)
        if len(values) != len(mapping):
            raise ValueError(f"第 {step} 步动作维度与 channel_map 不同")
        row = [0.0] * 6
        for value, channel in zip(values, mapping):
            row[channel] = value
        expanded.append(apply_channel_equalities(row, equalities))
    return tuple(expanded)


def build_plan(*, model_actions: Sequence[Sequence[float]], channel_map: Sequence[int],
               step_interval_s: float, model: ModelDescriptor, anchor: Anchor,
               scene: Scene, safety: SafetyPolicy, random_seed: int | None = None,
               predicted_states_path: str | None = None,
               loss_terms: dict[str, float] | None = None,
               metadata: dict[str, Any] | None = None) -> ActionPlan:
    return ActionPlan(
        actions6=expand_model_actions(
            model_actions, channel_map, model.channel_equalities),
        step_interval_s=step_interval_s,
        model_action_dim=model.action_dim,
        channel_map=tuple(channel_map),
        model_hash=model.checkpoint_hash,
        scene_digest=scene.digest,
        anchor_id=anchor.anchor_id,
        safety_digest=safety.digest,
        channel_equalities=model.channel_equalities,
        random_seed=random_seed,
        predicted_states_path=predicted_states_path,
        loss_terms=dict(loss_terms or {}),
        metadata=dict(metadata or {}),
    )
