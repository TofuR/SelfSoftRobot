"""OpenLoopTransitionModel 的工作台 shooting planner。

这里只实现当前实验主线所需的最小目标：平面末端点/圆区域与全身圆障碍。
其他 scene primitive 会明确拒绝，不会被静默忽略。
"""

from __future__ import annotations

import math
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..runtime import plan_rollout

from ..contracts.models import Anchor, ModelDescriptor, SafetyPolicy, Scene
from .planner_service import build_plan
from .units import kPa_to_model, model_to_kPa


@dataclass(frozen=True)
class ShootingConfig:
    horizon: int | None = 20         # 固定 K;auto_k=True 时必须显式设 None
    auto_k: bool = False
    k_min: int = 4
    k_max: int = 40
    n_iter: int = 400
    learning_rate: float = 0.05
    n_restarts: int = 4
    w_path: float = 0.2
    w_smooth: float = 0.01
    w_monotonic: float = 1.0
    w_obstacle: float = 1.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.auto_k:
            if self.horizon is not None:
                raise ValueError("auto_k=True 时 horizon 必须为 None(互斥)")
        elif self.horizon is None or self.horizon <= 0:
            raise ValueError("auto_k=False 时 horizon 必须为正整数")
        if self.auto_k and self.k_min > self.k_max:
            raise ValueError("k_min 不能大于 k_max")
        if self.n_iter <= 0 or self.n_restarts <= 0 or self.learning_rate <= 0:
            raise ValueError("n_iter/n_restarts/learning_rate 必须为正数")


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
    """解析唯一 target 原语 → dict。

    返回:
      - target_point/circle:{kind, point:(2,), radius, node, item, space}
      - target_skeleton:{kind, nodes:(N,2) tensor, weights|None, tolerance, item, space="model"}
    """
    targets = [item for item in scene.primitives if item.kind.startswith("target_")]
    if len(targets) != 1:
        raise ValueError("planner 要求 scene 中恰好一个 target 原语")
    item = targets[0]
    if item.kind in {"target_point", "target_circle"}:
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
        return {"kind": item.kind, "point": point, "radius": radius,
                "node": node, "item": item, "space": target_space}
    if item.kind == "target_skeleton":
        nodes = item.geometry.get("nodes")
        if not isinstance(nodes, (list, tuple)) or not nodes:
            raise ValueError("target_skeleton geometry 需要非空 nodes=[[x,y]×N]")
        weights = item.geometry.get("weights")
        tolerance = float(item.geometry.get("tolerance_px", 0.0))
        if item.frame_id != "model":
            raise ValueError("target_skeleton 必须已在 model 坐标")
        return {"kind": "target_skeleton", "nodes": torch.tensor(
            nodes, dtype=torch.float32, device=device), "weights": weights,
            "tolerance": tolerance, "item": item, "space": "model"}
    raise ValueError(f"当前 planner 尚不支持 {item.kind}")


def _obstacles(scene: Scene):
    """解析 obstacle 原语 → [("circle",(cx,cy),r) | ("aabb",(x0,y0,x1,y1),0)]。

    safety_margin 折进几何(circle 加 r;aabb 扩盒)。
    """
    supported = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.kind == "obstacle_circle":
            if item.frame_id != "model":
                raise ValueError("圆障碍必须先转换到 model 坐标")
            center = item.geometry.get("center", item.geometry.get("xy"))
            radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
            if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
                raise ValueError("obstacle_circle 需要 center=[x,y] 与正 radius")
            supported.append(("circle", (float(center[0]), float(center[1])),
                              radius + float(item.safety_margin)))
        elif item.kind == "obstacle_aabb":
            if item.frame_id != "model":
                raise ValueError("AABB 障碍必须先转换到 model 坐标")
            lo = item.geometry.get("min")
            hi = item.geometry.get("max")
            if not isinstance(lo, (list, tuple)) or not isinstance(hi, (list, tuple)) \
                    or len(lo) != 2 or len(hi) != 2:
                raise ValueError("obstacle_aabb 需要 min=[x,y] 与 max=[x,y]")
            margin = float(item.safety_margin)
            supported.append(("aabb", (float(lo[0]) - margin, float(lo[1]) - margin,
                                       float(hi[0]) + margin, float(hi[1]) + margin), 0.0))
        else:
            raise ValueError(f"当前 planner 尚不支持 {item.kind}")
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


def _forward_normalized(raw, model, history, k_effective, history_steps, state,
                        lo, hi, rise, fall, initial, dt, scale_kpa, norm):
    """物理动作 → 归一化模型输入 → rollout。返回 (physical_kPa, normalized, predictions)。

    优化循环与 no_grad 终评共用(消除重复)。
    """
    physical = _project_actions(raw, lo, hi, rise, fall, initial, dt)
    normalized = kPa_to_model(physical, action_scale_kpa=scale_kpa,
                              action_norm_factor=norm)
    predictions = plan_rollout(
        model, torch.cat((history, normalized), dim=0), len(history) - 1,
        k_effective, history_steps, state)
    return physical, normalized, predictions


def _skeleton_dists(predictions, target_nodes, scale, center):
    """全身目标:preds (K,N,3) → 各节点到目标的 (K,N) 距离(px 空间)。"""
    physical_nodes = predictions * scale[:3] + center[:3]
    return torch.linalg.vector_norm(
        physical_nodes[:, :, :2] - target_nodes[..., :2].unsqueeze(0), dim=2)


def _resolve_k(config, descriptor, model, target, state, center, scale):
    """固定 K 或 auto_k(step_budget 从学到的 delta_scale 现算)→ (k_effective, gap_px)。"""
    if config.auto_k:
        from .auto_k import (gap_px_point, gap_px_skeleton,
                             select_k_by_gap, step_budget_px)
        budget = step_budget_px(model)
        if target["kind"] == "target_skeleton":
            now_px = (state.squeeze(0).detach().cpu().numpy()
                      * scale.detach().cpu().numpy() + center.detach().cpu().numpy())
            gap = gap_px_skeleton(now_px, target["nodes"].cpu().numpy(),
                                  target["tolerance"])
        else:
            tip_px = (state[0, target["node"], :2].detach().cpu().numpy()
                      * scale[:2].cpu().numpy() + center[:2].cpu().numpy())
            gap = gap_px_point(tip_px, target["point"].cpu().numpy(),
                               target["radius"])
        k = select_k_by_gap(gap, budget, config.k_min, config.k_max)
        return min(k, descriptor.k_safe or k), gap
    k = config.horizon
    if descriptor.k_safe is not None and k > descriptor.k_safe:
        raise ValueError(f"K={k} 超过 K_safe={descriptor.k_safe}")
    return k, None


class OpenLoopShootingPlanner:
    def __init__(self, runtime):
        self.runtime = runtime

    def plan(self, *, anchor: Anchor, scene: Scene, safety: SafetyPolicy,
             channel_map: tuple[int, ...], step_interval_s: float,
             output_dir: str | Path, config: ShootingConfig = ShootingConfig(),
             cancel_event: threading.Event | None = None):
        from .obstacles import obstacle_term_ext

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
        if descriptor.action_scale_kpa is None:
            raise ValueError("checkpoint 缺少 action_scale_kpa(deploy_manifest 缺失);"
                             "单位链不可知,阻断规划(fail-closed)")
        if anchor.action_units not in {"kpa", "model_normalized"}:
            raise ValueError("未知 anchor action_units")

        device = next(model.parameters()).device
        state = _state_tensor(anchor, descriptor, model, device)
        target = _target(scene, model, device)
        target_item = target["item"]
        if target["kind"] != "target_skeleton" and (
                target["node"] < 0 or target["node"] >= descriptor.n_nodes):
            raise ValueError("target node 超出模型节点范围")
        obstacles = _obstacles(scene)
        center, scale = _model_geometry(model)
        center, scale = center.to(device), scale.to(device)
        norm = float(self.runtime.info["norm_factor"])
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("action norm_factor 必须为正数")
        action_scale_kpa = torch.as_tensor(
            descriptor.action_scale_kpa, dtype=torch.float32, device=device)

        history = torch.tensor(anchor.action_history, dtype=torch.float32, device=device)
        if history.shape[1] != descriptor.action_dim:
            raise ValueError("anchor action history 维度与模型不同")
        if len(history) < descriptor.history_steps:
            raise ValueError("anchor action history 不足 H 步")
        history = history[-descriptor.history_steps:]
        if anchor.action_units == "kpa":
            # 兼容旧标注:真实 kPa → 训练域 [0,1] → /norm_factor
            history = kPa_to_model(history, action_scale_kpa=action_scale_kpa,
                                   action_norm_factor=norm)
        # model_normalized(npz 来源,offline_anchor 新标注)直接用:已是模型单位

        # ---- 变长 K(B17):step_budget 从学到的 delta_scale 现算 ----
        k_effective, auto_k_gap_px = _resolve_k(config, descriptor, model, target,
                                                state, center, scale)

        mapped = torch.tensor(channel_map, dtype=torch.long, device=device)
        lo = torch.tensor(safety.pressure_min6, device=device)[mapped]
        hi = torch.tensor(safety.pressure_max6, device=device)[mapped]
        rise = torch.tensor(safety.rise_rate6, device=device)[mapped]
        fall = torch.tensor(safety.fall_rate6, device=device)[mapped]
        initial = torch.tensor(safety.initial_action6, device=device)[mapped]
        # 修 I1:history 在 kpa 分支已换成模型单位(训练域 [0,1]),repeat warm-start 的 raw
        # 是 kPa 空间(_project_actions 用 kPa 的 lo/hi 投影,physical 存为 plan 按 kPa 校验)。
        # 旧式 history[-1]*norm 仍在模型域(≈0..1),repeat 退化成 zero;须 ×scale 回 kPa。
        seed_last = torch.as_tensor(
            model_to_kPa(history[-1:], action_scale_kpa=action_scale_kpa,
                         action_norm_factor=norm),
            dtype=torch.float32, device=device).reshape(-1)

        torch.manual_seed(config.random_seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(config.random_seed)
        was_training = model.training
        model.train()  # cuDNN RNN backward 需要 train；当前模型无 dropout/BN
        start_wall = time.perf_counter()
        temporal = getattr(model, "temporal", None)
        if temporal is not None and hasattr(temporal, "build_weight_cache"):
            temporal.build_weight_cache(descriptor.history_steps, device=device,
                                        dtype=torch.float32)
        best = None
        try:
            for restart in range(config.n_restarts):
                if cancel_event and cancel_event.is_set():
                    raise concurrent.futures.CancelledError("planner cancelled")
                if restart == 0:
                    initial_raw = seed_last.repeat(k_effective, 1)
                    init_name = "repeat"
                elif restart == 1:
                    initial_raw = torch.zeros(k_effective, descriptor.action_dim, device=device)
                    init_name = "zero"
                else:
                    initial_raw = lo + torch.rand(k_effective, descriptor.action_dim,
                                                  device=device) * (hi - lo)
                    init_name = "random"
                raw = initial_raw.detach().clone().requires_grad_(True)
                optimizer = torch.optim.Adam([raw], lr=config.learning_rate)
                loss_curve = []
                for _ in range(config.n_iter):
                    if cancel_event and cancel_event.is_set():
                        raise concurrent.futures.CancelledError("planner cancelled")
                    optimizer.zero_grad(set_to_none=True)
                    physical, normalized, predictions = _forward_normalized(
                        raw, model, history, k_effective, descriptor.history_steps, state,
                        lo, hi, rise, fall, initial, step_interval_s,
                        action_scale_kpa, norm)
                    if target["kind"] == "target_skeleton":
                        dists = _skeleton_dists(predictions, target["nodes"], scale, center)
                        weights = target["weights"]
                        if weights is not None:
                            w = torch.as_tensor(weights, dtype=torch.float32, device=device)
                            errors = ((torch.relu(dists - target["tolerance"]).square()
                                       * w).sum(1) / max(1.0, float(w.sum())))
                        else:
                            errors = torch.relu(dists - target["tolerance"]).square().mean(1)
                    else:
                        tip_xy = predictions[:, target["node"], :2]
                        if target["space"] == "model":
                            tip_xy = tip_xy * scale[:2] + center[:2]
                        distances = torch.linalg.vector_norm(tip_xy - target["point"], dim=1)
                        errors = torch.relu(distances - target["radius"]).square()
                    terminal = errors[-1]
                    path_loss = errors.mean()
                    monotonic = (torch.relu(errors[1:] - errors[:-1]).square().mean()
                                 if k_effective > 1 else errors.new_zeros(()))
                    smooth = ((normalized[1:] - normalized[:-1]).square().mean()
                              if k_effective > 1 else errors.new_zeros(()))
                    obstacle = errors.new_zeros(())
                    if obstacles:
                        obstacle = obstacle_term_ext(predictions, scale[:3], center[:3],
                                                     obstacles)
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
                    physical, normalized, predictions = _forward_normalized(
                        raw, model, history, k_effective, descriptor.history_steps, state,
                        lo, hi, rise, fall, initial, step_interval_s,
                        action_scale_kpa, norm)
                    if target["kind"] == "target_skeleton":
                        dists = _skeleton_dists(predictions, target["nodes"], scale, center)
                        final_distance = float(dists.mean().cpu())
                    else:
                        final_tip = predictions[-1, target["node"], :2]
                        if target["space"] == "model":
                            final_tip = final_tip * scale[:2] + center[:2]
                        final_distance = float(torch.linalg.vector_norm(
                            final_tip - target["point"]).cpu())
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
            if temporal is not None and hasattr(temporal, "invalidate_weight_cache"):
                temporal.invalidate_weight_cache()
        duration_s = time.perf_counter() - start_wall

        assert best is not None
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        predictions_path = output / "predicted_states.npz"
        states_model = (best["predictions"] * scale.detach().cpu().numpy() +
                        center.detach().cpu().numpy())
        minimum_clearance = None
        if obstacles:
            values = []
            for kind, geom, radius in obstacles:
                xy = states_model[:, :, :2]
                if kind == "circle":
                    cx, cy = geom
                    distance = np.linalg.norm(xy - np.asarray((cx, cy), dtype=np.float64), axis=2)
                    values.append(float((distance - radius).min()))
                else:  # aabb:2D 有符号距离
                    x0, y0, x1, y1 = geom
                    cx = np.maximum(x0 - xy[..., 0], xy[..., 0] - x1)
                    cy = np.maximum(y0 - xy[..., 1], xy[..., 1] - y1)
                    outside = np.sqrt(np.maximum(cx, 0) ** 2 + np.maximum(cy, 0) ** 2)
                    inside = np.minimum(np.maximum(cx, cy), 0.0)
                    values.append(float((outside + inside).min()))
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
                "planner": "openloop_shooting_v2", "target_id": target_item.primitive_id,
                "target_kind": target["kind"],
                "best_init": best["init"],
                "loss_curve": best["loss_curve"], "n_iter": config.n_iter,
                "n_restarts": config.n_restarts, "k_effective": k_effective,
                "auto_k": config.auto_k, "auto_k_gap_px": auto_k_gap_px,
                "duration_s": duration_s,
                "predicted_min_obstacle_clearance": minimum_clearance,
            })
