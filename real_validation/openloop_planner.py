"""OpenLoopTransitionModel 的工作台 shooting planner。

这里只实现当前实验主线所需的最小目标：平面末端点/圆区域与全身圆障碍。
其他 scene primitive 会明确拒绝，不会被静默忽略。
"""

from __future__ import annotations

import math
import concurrent.futures
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .runtime import plan_rollout

from .models import Anchor, ModelDescriptor, SafetyPolicy, Scene
from .planner_service import build_plan


@dataclass(frozen=True)
class ShootingConfig:
    horizon: int = 20
    n_iter: int = 400
    learning_rate: float = 0.05
    n_restarts: int = 4
    w_path: float = 0.2
    w_smooth: float = 0.01
    w_monotonic: float = 1.0
    w_obstacle: float = 1.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.n_iter <= 0 or self.n_restarts <= 0:
            raise ValueError("horizon/n_iter/n_restarts 必须为正数")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正数")


def _model_geometry(model):
    try:
        center = model.pc_center.view(3)
        scale = model.pc_scale.view(3)
    except (AttributeError, RuntimeError) as error:
        raise ValueError("模型缺少 pc_center/pc_scale，不能进行实物坐标规划") from error
    if torch.any(scale.abs() < 1e-12):
        raise ValueError("模型 pc_scale 含零")
    return center, scale


def _state_tensor(anchor: Anchor, descriptor: ModelDescriptor, model, device):
    state = np.asarray(anchor.state, dtype=np.float32)
    if state.shape[0] != descriptor.n_nodes:
        raise ValueError("anchor 节点数与模型不一致")
    if state.shape[1] == 2:
        state = np.column_stack((state, np.zeros(len(state), dtype=np.float32)))
    value = torch.as_tensor(state, dtype=torch.float32, device=device)
    if anchor.state_space == "model":
        center, scale = _model_geometry(model)
        value = (value - center.to(device)) / scale.to(device)
    return value.unsqueeze(0)


def _target(scene: Scene, model, device):
    targets = [item for item in scene.primitives if item.kind.startswith("target_")]
    if len(targets) != 1:
        raise ValueError("第一版 planner 要求 scene 中恰好一个 target_point/target_circle")
    item = targets[0]
    if item.kind not in {"target_point", "target_circle"}:
        raise ValueError(f"当前 planner 尚不支持 {item.kind}")
    xy = item.geometry.get("xy", item.geometry.get("center"))
    if not isinstance(xy, (list, tuple)) or len(xy) != 2:
        raise ValueError(f"{item.kind} geometry 需要 xy=[x,y] 或 center=[x,y]")
    point = torch.tensor(xy, dtype=torch.float32, device=device)
    radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
    if radius < 0:
        raise ValueError("目标半径不能为负")
    if item.frame_id == "model":
        target_space = "model"
    elif item.frame_id == "model_normalized":
        target_space = "model_normalized"
    else:
        raise ValueError(f"目标坐标 {item.frame_id} 尚未转换到 model/model_normalized")
    node = int(item.geometry.get("node", 0))
    return point, radius, node, item, target_space


def _obstacles(scene: Scene):
    supported = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.kind != "obstacle_circle":
            raise ValueError(f"当前 planner 尚不支持 {item.kind}")
        if item.frame_id != "model":
            raise ValueError("圆障碍必须先转换到 model 坐标")
        center = item.geometry.get("center", item.geometry.get("xy"))
        radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
        if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
            raise ValueError("obstacle_circle 需要 center=[x,y] 与正 radius")
        supported.append((float(center[0]), float(center[1]),
                          radius + float(item.safety_margin)))
    return supported


def _project_actions(raw, lower, upper, rise, fall, initial, dt):
    rows = []
    previous = initial
    for raw_row in raw:
        bounded = torch.maximum(lower, torch.minimum(upper, raw_row))
        delta = bounded - previous
        delta = torch.maximum(-fall * dt, torch.minimum(rise * dt, delta))
        previous = previous + delta
        rows.append(previous)
    return torch.stack(rows)


class OpenLoopShootingPlanner:
    def __init__(self, runtime):
        self.runtime = runtime

    def plan(self, *, anchor: Anchor, scene: Scene, safety: SafetyPolicy,
             channel_map: tuple[int, ...], step_interval_s: float,
             output_dir: str | Path, config: ShootingConfig = ShootingConfig(),
             cancel_event: threading.Event | None = None):
        descriptor: ModelDescriptor = self.runtime.descriptor
        model = self.runtime.model
        if scene.dimension != 2:
            raise ValueError("当前 checkpoint/planner 只认证 2D 平面任务")
        if descriptor.model_type != "state_transition" or not hasattr(model, "init_z_from_action"):
            raise ValueError("当前工作台 shooting planner 只支持 state-transition/OpenLoop checkpoint")
        if descriptor.model_class and descriptor.model_class != "OpenLoopTransitionModel":
            raise ValueError(
                f"部署主线要求 OpenLoopTransitionModel，当前为 {descriptor.model_class}")
        if len(channel_map) != descriptor.action_dim:
            raise ValueError("channel_map 长度必须等于模型 action_dim")
        if descriptor.k_safe is not None and config.horizon > descriptor.k_safe:
            raise ValueError(f"K={config.horizon} 超过 K_safe={descriptor.k_safe}")
        if anchor.action_units not in {"kpa", "model_normalized"}:
            raise ValueError("未知 anchor action_units")

        device = next(model.parameters()).device
        state = _state_tensor(anchor, descriptor, model, device)
        target_xy, target_radius, target_node, target_item, target_space = _target(
            scene, model, device)
        if target_node < 0 or target_node >= descriptor.n_nodes:
            raise ValueError("target node 超出模型节点范围")
        obstacles = _obstacles(scene)
        center, scale = _model_geometry(model)
        center, scale = center.to(device), scale.to(device)
        norm = float(self.runtime.info["norm_factor"])
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("action norm_factor 必须为正数")

        history = torch.tensor(anchor.action_history, dtype=torch.float32, device=device)
        if history.shape[1] != descriptor.action_dim:
            raise ValueError("anchor action history 维度与模型不同")
        if len(history) < descriptor.history_steps:
            raise ValueError("anchor action history 不足 H 步")
        history = history[-descriptor.history_steps:]
        if anchor.action_units == "kpa":
            history = history / norm

        mapped = torch.tensor(channel_map, dtype=torch.long, device=device)
        lo = torch.tensor(safety.pressure_min6, device=device)[mapped]
        hi = torch.tensor(safety.pressure_max6, device=device)[mapped]
        rise = torch.tensor(safety.rise_rate6, device=device)[mapped]
        fall = torch.tensor(safety.fall_rate6, device=device)[mapped]
        initial = torch.tensor(safety.initial_action6, device=device)[mapped]
        seed_last = history[-1] * norm

        torch.manual_seed(config.random_seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(config.random_seed)
        was_training = model.training
        model.train()  # cuDNN RNN backward 需要 train；当前模型无 dropout/BN
        best = None
        try:
            for restart in range(config.n_restarts):
                if cancel_event and cancel_event.is_set():
                    raise concurrent.futures.CancelledError("planner cancelled")
                if restart == 0:
                    initial_raw = seed_last.repeat(config.horizon, 1)
                    init_name = "repeat"
                elif restart == 1:
                    initial_raw = torch.zeros(config.horizon, descriptor.action_dim, device=device)
                    init_name = "zero"
                else:
                    initial_raw = lo + torch.rand(config.horizon, descriptor.action_dim,
                                                  device=device) * (hi - lo)
                    init_name = "random"
                raw = initial_raw.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
                loss_curve = []
                for _ in range(config.n_iter):
                    if cancel_event and cancel_event.is_set():
                        raise concurrent.futures.CancelledError("planner cancelled")
                    optimizer.zero_grad(set_to_none=True)
                    physical = _project_actions(raw, lo, hi, rise, fall, initial,
                                                step_interval_s)
                    normalized = physical / norm
                    buffer = torch.cat((history, normalized), dim=0)
                    predictions = plan_rollout(model, buffer, len(history) - 1,
                                               config.horizon, descriptor.history_steps, state)
                    tip_xy = predictions[:, target_node, :2]
                    if target_space == "model":
                        tip_xy = tip_xy * scale[:2] + center[:2]
                    distances = torch.linalg.vector_norm(tip_xy - target_xy, dim=1)
                    errors = torch.relu(distances - target_radius).square()
                    terminal = errors[-1]
                    path_loss = errors.mean()
                    monotonic = (torch.relu(errors[1:] - errors[:-1]).square().mean()
                                 if config.horizon > 1 else errors.new_zeros(()))
                    smooth = ((normalized[1:] - normalized[:-1]).square().mean()
                              if config.horizon > 1 else errors.new_zeros(()))
                    obstacle = errors.new_zeros(())
                    if obstacles:
                        physical_states = predictions * scale + center
                        for cx, cy, radius in obstacles:
                            distance = torch.linalg.vector_norm(
                                physical_states[:, :, :2] -
                                physical_states.new_tensor((cx, cy)), dim=2)
                            obstacle = obstacle + torch.relu(radius - distance).square().mean()
                    loss = (terminal + config.w_path * path_loss +
                            config.w_monotonic * monotonic + config.w_smooth * smooth +
                            config.w_obstacle * obstacle)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("planner loss 出现 NaN/Inf")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([raw], 1.0)
                    optimizer.step()
                    loss_curve.append(float(loss.detach().cpu()))
                with torch.no_grad():
                    physical = _project_actions(raw, lo, hi, rise, fall, initial,
                                                step_interval_s)
                    normalized = physical / norm
                    predictions = plan_rollout(
                        model, torch.cat((history, normalized), dim=0), len(history) - 1,
                        config.horizon, descriptor.history_steps, state)
                    final_tip = predictions[-1, target_node, :2]
                    if target_space == "model":
                        final_tip = final_tip * scale[:2] + center[:2]
                    final_distance = float(torch.linalg.vector_norm(final_tip - target_xy).cpu())
                    candidate = {
                        "actions": physical.detach().cpu().numpy(),
                        "predictions": predictions.detach().cpu().numpy(),
                        "loss_curve": loss_curve,
                        "final_distance_normalized": final_distance,
                        "init": init_name,
                    }
                if best is None or candidate["final_distance_normalized"] < best["final_distance_normalized"]:
                    best = candidate
        finally:
            model.train(was_training)

        assert best is not None
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        predictions_path = output / "predicted_states.npz"
        states_model = (best["predictions"] * scale.detach().cpu().numpy() +
                        center.detach().cpu().numpy())
        minimum_clearance = None
        if obstacles:
            values = []
            for cx, cy, radius in obstacles:
                distance = np.linalg.norm(states_model[:, :, :2] -
                                          np.asarray((cx, cy)), axis=2)
                values.append(float((distance - radius).min()))
            minimum_clearance = min(values)
        np.savez_compressed(predictions_path,
                            states_normalized=best["predictions"],
                            states_model=states_model)
        return build_plan(
            model_actions=best["actions"], channel_map=channel_map,
            step_interval_s=step_interval_s, model=descriptor, anchor=anchor,
            scene=scene, safety=safety, random_seed=config.random_seed,
            predicted_states_path=predictions_path.name,
            loss_terms={"final_target_distance": best["final_distance_normalized"]},
            metadata={
                "planner": "openloop_shooting_v1", "target_id": target_item.primitive_id,
                "target_node": target_node, "best_init": best["init"],
                "loss_curve": best["loss_curve"], "n_iter": config.n_iter,
                "n_restarts": config.n_restarts,
                "predicted_min_obstacle_clearance": minimum_clearance,
            })
