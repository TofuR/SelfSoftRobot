"""Metric selection and early-stop state shared by training phases."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .spec import ValidationSpec


@dataclass(frozen=True)
class SelectionDecision:
    evaluation_index: int
    improved: bool
    should_stop: bool
    best_value: float
    best_epoch: int
    bad_evaluations: int
    reason: str


class SelectionState:
    """Track best validation metric and patience in validation units."""

    def __init__(self, spec: ValidationSpec):
        self.spec = spec
        self.evaluations = 0
        self.bad_evaluations = 0
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None

    def _improved(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.spec.selection_mode == "min":
            return value < self.best_value - self.spec.min_delta
        return value > self.best_value + self.spec.min_delta

    def observe(
        self,
        value: float,
        *,
        epoch: int,
        early_stopping_allowed: bool = True,
    ) -> SelectionDecision:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"validation metric 必须是有限值: {value!r}")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= 0:
            raise ValueError("epoch 必须是正整数")

        self.evaluations += 1
        improved = self._improved(value)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_evaluations = 0
            reason = "improved"
        elif self.evaluations <= self.spec.warmup_evaluations:
            self.bad_evaluations = 0
            reason = "warmup"
        elif self.spec.early_stopping_patience_evaluations is None:
            self.bad_evaluations = 0
            reason = "early_stopping_disabled"
        elif not early_stopping_allowed:
            self.bad_evaluations = 0
            reason = "early_stopping_gated"
        else:
            self.bad_evaluations += 1
            reason = "patience_wait"

        patience = self.spec.early_stopping_patience_evaluations
        should_stop = bool(
            patience is not None and early_stopping_allowed and
            self.evaluations > self.spec.warmup_evaluations and
            self.bad_evaluations >= patience)
        if should_stop:
            reason = "patience_exhausted"
        return SelectionDecision(
            evaluation_index=self.evaluations,
            improved=improved,
            should_stop=should_stop,
            best_value=float(self.best_value),
            best_epoch=int(self.best_epoch),
            bad_evaluations=self.bad_evaluations,
            reason=reason,
        )
