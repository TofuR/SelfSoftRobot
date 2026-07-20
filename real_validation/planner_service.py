"""GUI/CLI 共用规划服务的第一版适配层。"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .models import ActionPlan, Anchor, ModelDescriptor, SafetyPolicy, Scene


def expand_model_actions(actions: Sequence[Sequence[float]],
                         channel_map: Sequence[int]) -> tuple[tuple[float, ...], ...]:
    mapping = tuple(int(channel) for channel in channel_map)
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
        expanded.append(tuple(row))
    return tuple(expanded)


def build_plan(*, model_actions: Sequence[Sequence[float]], channel_map: Sequence[int],
               step_interval_s: float, model: ModelDescriptor, anchor: Anchor,
               scene: Scene, safety: SafetyPolicy, random_seed: int | None = None,
               predicted_states_path: str | None = None,
               loss_terms: dict[str, float] | None = None,
               metadata: dict[str, Any] | None = None) -> ActionPlan:
    return ActionPlan(
        actions6=expand_model_actions(model_actions, channel_map),
        step_interval_s=step_interval_s,
        model_action_dim=model.action_dim,
        channel_map=tuple(channel_map),
        model_hash=model.checkpoint_hash,
        scene_digest=scene.digest,
        anchor_id=anchor.anchor_id,
        safety_digest=safety.digest,
        random_seed=random_seed,
        predicted_states_path=predicted_states_path,
        loss_terms=dict(loss_terms or {}),
        metadata=dict(metadata or {}),
    )


class PlannerService:
    """有界单 worker 规划服务，防止重复点击并发占满 GPU。"""

    def __init__(self):
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="real-validation-planner")
        self._future: concurrent.futures.Future | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def submit(self, planner: Callable[..., ActionPlan], *args, **kwargs):
        with self._lock:
            if self._future is not None and not self._future.done():
                raise RuntimeError("已有规划任务在运行")
            self._future = self._pool.submit(planner, *args, **kwargs)
            return self._future

    def cancel(self) -> bool:
        with self._lock:
            return bool(self._future and self._future.cancel())

    def close(self) -> None:
        self.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
