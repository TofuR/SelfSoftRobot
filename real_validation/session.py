"""实验会话状态机与 run 目录管理。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .io import atomic_write_json, read_json
from .models import ActionPlan, Anchor, ModelDescriptor, SafetyPolicy, Scene
from .preflight import PreflightResult, validate_plan


class SessionState(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    READY = "ready"
    ARMED = "armed"
    EXECUTING = "executing"
    PAUSED = "paused"
    REANCHOR = "reanchor"
    COMPLETED = "completed"
    ABORTING = "aborting"
    ZEROED = "zeroed"
    ERROR = "error"


_TRANSITIONS = {
    SessionState.IDLE: {SessionState.PLANNING, SessionState.READY, SessionState.ERROR},
    SessionState.PLANNING: {SessionState.READY, SessionState.IDLE, SessionState.ERROR},
    SessionState.READY: {SessionState.ARMED, SessionState.PLANNING, SessionState.IDLE,
                         SessionState.ERROR},
    SessionState.ARMED: {SessionState.EXECUTING, SessionState.READY, SessionState.ABORTING},
    SessionState.EXECUTING: {SessionState.PAUSED, SessionState.REANCHOR,
                             SessionState.COMPLETED, SessionState.ABORTING,
                             SessionState.ERROR},
    SessionState.PAUSED: {SessionState.EXECUTING, SessionState.REANCHOR,
                          SessionState.ABORTING, SessionState.ZEROED},
    SessionState.REANCHOR: {SessionState.PLANNING, SessionState.ABORTING,
                            SessionState.ERROR},
    SessionState.COMPLETED: {SessionState.ZEROED, SessionState.IDLE},
    SessionState.ABORTING: {SessionState.ZEROED, SessionState.ERROR},
    SessionState.ZEROED: {SessionState.IDLE},
    SessionState.ERROR: {SessionState.ZEROED, SessionState.IDLE},
}


@dataclass
class ExperimentSession:
    run_dir: Path
    model: ModelDescriptor | None = None
    anchor: Anchor | None = None
    scene: Scene = field(default_factory=Scene)
    safety: SafetyPolicy = field(default_factory=SafetyPolicy)
    plan: ActionPlan | None = None
    state: SessionState = SessionState.IDLE
    events: list[dict[str, Any]] = field(default_factory=list)
    replay_only: bool = False

    @classmethod
    def create(cls, root: str | Path, prefix: str = "run") -> "ExperimentSession":
        root_path = Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for suffix in range(10000):
            name = f"{prefix}_{stamp}" if suffix == 0 else f"{prefix}_{stamp}_{suffix:02d}"
            run_dir = root_path / name
            try:
                run_dir.mkdir()
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError("无法创建唯一 run 目录")
        session = cls(run_dir=run_dir)
        session._record("session_created")
        session.save_snapshot()
        return session

    @classmethod
    def load_for_replay(cls, run_dir: str | Path) -> "ExperimentSession":
        directory = Path(run_dir).resolve()
        value = read_json(directory / "experiment.json")
        session = cls(
            run_dir=directory,
            model=ModelDescriptor.from_dict(value["model"]) if value.get("model") else None,
            anchor=Anchor.from_dict(value["anchor"]) if value.get("anchor") else None,
            scene=Scene.from_dict(value.get("scene", {})),
            safety=SafetyPolicy.from_dict(value.get("safety", {})),
            plan=ActionPlan.from_dict(value["plan"]) if value.get("plan") else None,
            state=SessionState.IDLE,
            events=list(value.get("events", [])),
            replay_only=True,
        )
        session._record("opened_for_replay")
        return session

    def transition(self, target: SessionState, reason: str = "") -> None:
        if target == self.state:
            return
        if target not in _TRANSITIONS[self.state]:
            raise RuntimeError(f"非法状态转移: {self.state.value} -> {target.value}")
        previous = self.state
        self.state = target
        self._record("state_transition", previous=previous.value,
                     current=target.value, reason=reason)
        self.save_snapshot()

    def configure_model(self, model: ModelDescriptor) -> None:
        if self.state not in {SessionState.IDLE, SessionState.READY}:
            raise RuntimeError("只能在 idle/ready 状态切换模型")
        changed = self.model != model
        self.model = model
        if changed:
            self.anchor = None
            self.plan = None
            self._record("model_changed", checkpoint_hash=model.checkpoint_hash)
            self._return_to_idle_if_ready("model changed")
        self.save_snapshot()

    def set_anchor(self, anchor: Anchor) -> None:
        self._guard_editable("anchor")
        self.anchor = anchor
        self.plan = None
        self._record("anchor_changed", anchor_id=anchor.anchor_id)
        self._return_to_idle_if_ready("anchor changed")
        self.save_snapshot()

    def set_scene(self, scene: Scene) -> None:
        self._guard_editable("scene")
        self.scene = scene
        self.plan = None
        self._record("scene_changed", scene_digest=scene.digest)
        self._return_to_idle_if_ready("scene changed")
        self.save_snapshot()

    def set_safety(self, safety: SafetyPolicy) -> None:
        self._guard_editable("safety")
        self.safety = safety
        self.plan = None
        self._record("safety_changed", safety_digest=safety.digest)
        self._return_to_idle_if_ready("safety changed")
        self.save_snapshot()

    def _guard_editable(self, field_name: str) -> None:
        """B16:执行中禁止改 scene/anchor/safety —— 否则 experiment.json 的 plan 被清空
        而命令正在下发,执行记录与实际下发计划脱钩(溯源腐败)。"""
        if self.state not in {SessionState.IDLE, SessionState.READY}:
            raise RuntimeError(f"只能在 idle/ready 状态修改 {field_name},当前 {self.state.value}")

    def invalidate_model(self, reason: str = "") -> None:
        """B15:清除模型 descriptor(加载失败时调用),防操作员误用旧 runtime。"""
        if self.state not in {SessionState.IDLE, SessionState.READY}:
            raise RuntimeError("只能在 idle/ready 状态清除模型")
        self.model = None
        self.anchor = None
        self.plan = None
        self._record("model_invalidated", reason=reason)
        self._return_to_idle_if_ready(reason or "model invalidated")
        self.save_snapshot()

    def begin_planning(self) -> None:
        if self.replay_only:
            raise RuntimeError("replay session 只读；请新建实验后重新规划")
        if self.model is None or self.anchor is None:
            raise RuntimeError("规划前必须加载模型并建立 anchor")
        self.plan = None
        self.transition(SessionState.PLANNING, "planning started")

    def _return_to_idle_if_ready(self, reason: str) -> None:
        if self.state == SessionState.READY:
            self.transition(SessionState.IDLE, reason)

    def accept_plan(self, plan: ActionPlan) -> PreflightResult:
        if self.model is None or self.anchor is None:
            raise RuntimeError("接受计划前必须加载模型并建立 anchor")
        result = validate_plan(plan, self.model, self.anchor, self.scene, self.safety)
        if result.ok:
            self.plan = plan
            if self.state == SessionState.PLANNING:
                self.transition(SessionState.READY, "plan accepted")
            elif self.state == SessionState.IDLE:
                self.transition(SessionState.READY, "plan imported")
            else:
                self.save_snapshot()
        return result

    def arm(self) -> None:
        if self.replay_only:
            raise RuntimeError("replay session 永远不能 Arm")
        if self.plan is None or self.model is None or self.anchor is None:
            raise RuntimeError("没有可执行计划")
        validate_plan(self.plan, self.model, self.anchor, self.scene,
                      self.safety).require_ok()
        self.transition(SessionState.ARMED, "operator confirmed")

    def _record(self, event: str, **payload: Any) -> None:
        self.events.append({"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "event": event, **payload})

    def save_snapshot(self) -> None:
        atomic_write_json(self.run_dir / "experiment.json", {
            "state": self.state.value,
            "run_dir": str(self.run_dir),
            "model": self.model.to_dict() if self.model else None,
            "anchor": self.anchor.to_dict() if self.anchor else None,
            "scene": self.scene.to_dict(),
            "safety": self.safety.to_dict(),
            "plan": self.plan.to_dict() if self.plan else None,
            "events": self.events,
            "replay_only": self.replay_only,
        })
