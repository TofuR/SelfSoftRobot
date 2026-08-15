"""严格隔离控制观测与隐藏评价真值。"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque


@dataclass(frozen=True)
class ObservationDecision:
    timestamp: float
    source: str
    allowed: bool
    reason: str
    checkpoint: str | None = None


@dataclass
class ObservationPolicy:
    mode: str = "anchor_only"
    period_steps: int | None = None
    checkpoints: set[str] = field(default_factory=set)
    _last_observed_step: int | None = None
    audit: list[ObservationDecision] = field(default_factory=list)

    def decide(self, *, step: int, timestamp: float, source: str,
               checkpoint: str | None = None, force: bool = False) -> ObservationDecision:
        if force:
            allowed, reason = True, "operator_anchor"
        elif self.mode == "always":
            allowed, reason = True, "always"
        elif self.mode == "periodic" and self.period_steps:
            allowed = self._last_observed_step is None or (
                step - self._last_observed_step >= self.period_steps)
            reason = "periodic_due" if allowed else "periodic_hidden"
        elif self.mode == "checkpoint":
            allowed = checkpoint is not None and checkpoint in self.checkpoints
            reason = "checkpoint" if allowed else "outside_checkpoint"
        elif self.mode == "anchor_only":
            allowed = self._last_observed_step is None
            reason = "initial_anchor" if allowed else "hidden_after_anchor"
        else:
            raise ValueError(f"无效 observation policy: {self.mode}")
        if allowed:
            self._last_observed_step = step
        decision = ObservationDecision(timestamp, source, allowed, reason, checkpoint)
        self.audit.append(decision)
        return decision

    def require_allowed(self, decision: ObservationDecision) -> None:
        if not decision.allowed:
            raise PermissionError(f"隐藏评价观测不能进入模型: {decision.reason}")


class ActionHistoryBuffer:
    """只保存实际 applied command 的有界 H 步历史。"""

    def __init__(self, history_steps: int, action_dim: int, channel_map):
        if history_steps <= 0 or action_dim <= 0:
            raise ValueError("history_steps/action_dim 必须为正数")
        self.history_steps = int(history_steps)
        self.action_dim = int(action_dim)
        self.channel_map = tuple(int(channel) for channel in channel_map)
        if len(self.channel_map) != self.action_dim:
            raise ValueError("channel_map 长度与 action_dim 不同")
        self._values = deque(maxlen=self.history_steps)

    def append_applied6(self, applied6) -> None:
        values = tuple(float(value) for value in applied6)
        if len(values) != 6:
            raise ValueError("applied command 必须是六通道")
        self._values.append(tuple(values[channel] for channel in self.channel_map))

    @property
    def ready(self) -> bool:
        return len(self._values) == self.history_steps

    def snapshot(self):
        if not self.ready:
            raise RuntimeError(f"动作历史仅 {len(self._values)}/{self.history_steps} 步")
        return tuple(self._values)
