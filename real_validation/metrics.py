"""预测—实机执行的独立、可离线复算指标。"""

from __future__ import annotations

import numpy as np

from .models import SafetyPolicy, Scene


def _states(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] not in (2, 3):
        raise ValueError(f"{name} 必须为 K×N×2/3")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 含 NaN/Inf；应先按质量门控筛除无效帧")
    return array


def evaluate_prediction(predicted_states, observed_states, scene: Scene | None = None,
                        tip_node: int = 0) -> dict:
    predicted = _states(predicted_states, "predicted_states")
    observed = _states(observed_states, "observed_states")
    if predicted.shape != observed.shape:
        raise ValueError(f"预测与观测形状不同: {predicted.shape} != {observed.shape}")
    if tip_node < 0 or tip_node >= predicted.shape[1]:
        raise ValueError("tip_node 越界")
    dimensions = min(predicted.shape[2], observed.shape[2])
    distances = np.linalg.norm(predicted[..., :dimensions] -
                               observed[..., :dimensions], axis=2)
    per_step = distances.mean(axis=1)
    metrics = {
        "steps": int(predicted.shape[0]),
        "nodes": int(predicted.shape[1]),
        "mne": float(distances.mean()),
        "p90": float(np.percentile(distances, 90)),
        "max_error": float(distances.max()),
        "terminal_mne": float(per_step[-1]),
        "terminal_tip_error": float(distances[-1, tip_node]),
        "error_by_k": per_step.tolist(),
    }
    if scene is not None:
        metrics.update(_scene_metrics(observed, scene, tip_node))
    return metrics


def _scene_metrics(observed: np.ndarray, scene: Scene, tip_node: int) -> dict:
    if scene.dimension != 2:
        raise ValueError("第一版 scene metrics 只认证 2D 平面约束")
    xy = observed[..., :2]
    clearances = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.frame_id != "model":
            raise ValueError(f"scene metrics 尚不支持 {item.kind}@{item.frame_id}")
        if item.kind == "obstacle_circle":
            center = item.geometry.get("center", item.geometry.get("xy"))
            radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
            radius += float(item.safety_margin)
            distance = np.linalg.norm(xy - np.asarray(center, dtype=np.float64), axis=2)
            clearances.append(distance - radius)
        elif item.kind == "obstacle_aabb":
            lo = np.asarray(item.geometry.get("min"), dtype=np.float64)
            hi = np.asarray(item.geometry.get("max"), dtype=np.float64)
            margin = float(item.safety_margin)
            lo -= margin
            hi += margin
            # AABB 2D 有符号距离(盒外正 / 盒内负)
            cx = np.maximum(lo[0] - xy[..., 0], xy[..., 0] - hi[0])
            cy = np.maximum(lo[1] - xy[..., 1], xy[..., 1] - hi[1])
            outside = np.sqrt(np.maximum(cx, 0) ** 2 + np.maximum(cy, 0) ** 2)
            inside = np.minimum(np.maximum(cx, cy), 0.0)
            clearances.append(outside + inside)
        else:
            raise ValueError(f"scene metrics 尚不支持 {item.kind}")
    result = {}
    if clearances:
        clearance = np.stack(clearances, axis=0)
        result["minimum_obstacle_clearance"] = float(clearance.min())
        result["collision"] = bool(np.any(clearance < 0))

    targets = [item for item in scene.primitives if item.kind.startswith("target_")]
    if len(targets) == 1:
        target = targets[0]
        if target.frame_id != "model":
            raise ValueError("任务成功评价要求 target 已转换到 model 坐标")
        if target.kind in {"target_point", "target_circle"}:
            center = target.geometry.get("xy", target.geometry.get("center"))
            radius = float(target.geometry.get("radius", target.geometry.get("r", 0.0)))
            terminal_distance = float(np.linalg.norm(
                xy[-1, tip_node] - np.asarray(center, dtype=np.float64)))
            result["terminal_target_distance"] = terminal_distance
            result["target_success"] = bool(terminal_distance <= radius)
        elif target.kind == "target_skeleton":
            nodes = np.asarray(target.geometry.get("nodes"), dtype=np.float64)
            tolerance = float(target.geometry.get("tolerance_px", 0.0))
            dists = np.linalg.norm(xy[-1] - nodes[:, :2], axis=1)
            result["terminal_skeleton_mne"] = float(dists.mean())
            result["target_success"] = bool(dists.mean() <= tolerance)
    return result


def evaluate_command_safety(actions6, step_interval_s: float,
                            safety: SafetyPolicy) -> dict:
    actions = np.asarray(actions6, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 6 or not np.isfinite(actions).all():
        raise ValueError("actions6 必须是有限的 K×6")
    if step_interval_s <= 0:
        raise ValueError("step_interval_s 必须为正数")
    lower = np.asarray(safety.pressure_min6)
    upper = np.asarray(safety.pressure_max6)
    bound_mask = (actions < lower) | (actions > upper)
    previous = np.vstack((np.asarray(safety.initial_action6), actions[:-1]))
    delta = actions - previous
    rise = np.asarray(safety.rise_rate6) * step_interval_s
    fall = np.asarray(safety.fall_rate6) * step_interval_s
    # rate=0 与采集 limiter 保持一致，表示不限速。
    slew_mask = ((delta > rise) & (rise > 0)) | ((-delta > fall) & (fall > 0))
    return {
        "pressure_violation_count": int(bound_mask.sum()),
        "slew_violation_count": int(slew_mask.sum()),
        "pressure_safe": bool(not bound_mask.any()),
        "slew_safe": bool(not slew_mask.any()),
    }
