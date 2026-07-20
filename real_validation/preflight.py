"""计划执行前的纯函数安全检查。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import ActionPlan, Anchor, ModelDescriptor, SafetyPolicy, Scene


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    message: str
    step: int | None = None
    channel: int | None = None


@dataclass(frozen=True)
class PreflightResult:
    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def require_ok(self) -> None:
        if self.issues:
            details = "; ".join(issue.message for issue in self.issues[:5])
            raise ValueError(f"计划未通过 preflight: {details}")


def validate_plan(plan: ActionPlan, model: ModelDescriptor, anchor: Anchor,
                  scene: Scene, safety: SafetyPolicy) -> PreflightResult:
    issues: list[PreflightIssue] = []

    def add(code: str, message: str, step=None, channel=None) -> None:
        issues.append(PreflightIssue(code, message, step, channel))

    if plan.model_hash != model.checkpoint_hash:
        add("stale_model", "计划对应的 checkpoint 已变化")
    if plan.scene_digest != scene.digest:
        add("stale_scene", "计划对应的 scene 已变化")
    if plan.anchor_id != anchor.anchor_id:
        add("stale_anchor", "计划对应的 anchor 已变化")
    if plan.safety_digest != safety.digest:
        add("stale_safety", "计划对应的安全配置已变化")
    if plan.model_action_dim != model.action_dim:
        add("action_dim", f"计划动作维度 {plan.model_action_dim} 与模型 {model.action_dim} 不同")
    if len(anchor.state) != model.n_nodes:
        add("anchor_nodes", f"anchor 节点数 {len(anchor.state)} 与模型 {model.n_nodes} 不同")
    if len(anchor.action_history) < model.history_steps:
        add("history_short", f"动作历史仅 {len(anchor.action_history)} 步，模型需要 {model.history_steps} 步")
    if any(len(action) != model.action_dim for action in anchor.action_history):
        add("history_dim", "anchor 动作历史维度与模型 action_dim 不同")
    if len(plan.channel_map) != model.action_dim:
        add("channel_map", "channel_map 长度必须等于模型 action_dim")
    if len(set(plan.channel_map)) != len(plan.channel_map):
        add("channel_map", "channel_map 不能包含重复硬件通道")
    if any(channel < 0 or channel >= 6 for channel in plan.channel_map):
        add("channel_map", "channel_map 必须位于 0..5")
    if model.k_safe is not None and plan.horizon > model.k_safe:
        add("k_safe", f"计划 K={plan.horizon} 超过 checkpoint 的 K_safe={model.k_safe}")
    predicted_clearance = plan.metadata.get("predicted_min_obstacle_clearance")
    if predicted_clearance is not None and float(predicted_clearance) < 0:
        add("predicted_collision",
            f"预测轨迹侵入障碍 {abs(float(predicted_clearance)):.3g} 个模型坐标单位")

    mapped = set(plan.channel_map)
    previous = safety.initial_action6
    for step, action in enumerate(plan.actions6):
        for channel, value in enumerate(action):
            if not math.isfinite(value):
                add("non_finite", f"第 {step} 步 ch{channel} 含 NaN/Inf", step, channel)
                continue
            if value < safety.pressure_min6[channel] or value > safety.pressure_max6[channel]:
                add("pressure_bound", f"第 {step} 步 ch{channel}={value:g} kPa 越界", step, channel)
            if channel not in mapped and abs(value) > 1e-9:
                add("inactive_channel", f"第 {step} 步未映射 ch{channel} 必须锁零", step, channel)
            delta = value - previous[channel]
            rate = safety.rise_rate6[channel] if delta >= 0 else safety.fall_rate6[channel]
            if rate > 0 and abs(delta) > rate * plan.step_interval_s + 1e-9:
                add("slew_rate", f"第 {step} 步 ch{channel} 压力变化超过 {rate:g} kPa/s",
                    step, channel)
        previous = action
    return PreflightResult(tuple(issues))
