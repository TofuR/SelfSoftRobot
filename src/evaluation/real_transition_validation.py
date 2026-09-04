"""Native-unit validation metrics for real transition models."""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch

from src.data.action_view import project_actions
from src.evaluation.transition_metrics import build_action_window


def per_frame_rollout(model, mode, actions, positions, window_size, norm_factor,
                      device, K=40, max_steps=None, anchor_index=0):
    """Predict every usable row with GT-observed or windowed OpenLoop semantics."""
    if K <= 0:
        raise ValueError("OpenLoop validation K 必须为正整数")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("validation max_steps 必须为正整数或 null")
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < T:
        raise ValueError("anchor_index 必须位于 validation 范围内")
    actions_norm = actions / norm_factor
    pc_center = model.pc_center.view(3).detach().cpu().numpy()
    pc_scale = model.pc_scale.view(3).detach().cpu().numpy()
    N = positions.shape[2]

    def to_norm(pos_3N):
        skeleton = pos_3N.T.astype(np.float32)
        skeleton = (skeleton - pc_center) / pc_scale
        return torch.from_numpy(skeleton).float().unsqueeze(0).to(device)

    def action_window(t):
        value = build_action_window(actions_norm, t, window_size)
        return torch.from_numpy(value).float().unsqueeze(0).to(device)

    pred = np.zeros((T, N, 3), np.float32)
    k_in_window = np.full(T, -1, np.int32)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if mode in ("gt", "onestep"):
                z_t = model.init_z_from_action(action_window(0))
                for t in range(T):
                    prev = to_norm(positions[max(t - 1, 0)])
                    prev2 = to_norm(positions[max(t - 2, 0)])
                    out = model.forward(action_window(t), prev, prev2, z_t)
                    pred[t] = out["skeleton"].squeeze(0).cpu().numpy()
                    z_t = out["latent_z"]
            elif mode == "open_loop":
                t = 1
                while t < T:
                    z_t = model.init_z_from_action(action_window(t))
                    s_roll = to_norm(positions[t - 1])
                    s_prev = s_roll
                    for k in range(K):
                        current = t + k
                        if current >= T:
                            break
                        out = model.forward(
                            action_window(current), s_roll, s_prev, z_t)
                        pred[current] = out["skeleton"].squeeze(0).cpu().numpy()
                        k_in_window[current] = k
                        z_t = out["latent_z"]
                        s_prev = s_roll
                        s_roll = out["skeleton"]
                    t += K
            elif mode in ("ishsm", "ishsm_zero", "ishsm_periodic"):
                # ``ishsm`` observes frame 0 once. ``ishsm_periodic`` observes
                # one skeleton every K frames. Neither mode consumes a dense
                # skeleton history between anchors.
                if T >= anchor_index + 2:
                    z_t = None
                    last_anchor = anchor_index
                    for t in range(anchor_index + 1, T):
                        current_window = action_window(t)
                        should_anchor = (
                            z_t is None or
                            (mode == "ishsm_periodic" and
                             (t - 1 - anchor_index) % K == 0))
                        if should_anchor:
                            if mode == "ishsm_zero":
                                z_t = model.init_z_from_action(current_window)
                            else:
                                observation = to_norm(positions[t - 1])
                                if (mode == "ishsm_periodic" and z_t is not None and
                                        hasattr(model, "assimilate_observation")):
                                    z_t = model.assimilate_observation(
                                        current_window, observation, z_t)
                                else:
                                    z_t = model.init_rollout_state(
                                        current_window, observation)
                            last_anchor = t - 1
                        out = model.forward(current_window, prev_z=z_t)
                        pred[t] = out["skeleton"].squeeze(0).cpu().numpy()
                        k_in_window[t] = t - last_anchor - 1
                        z_t = out["latent_z"]
            else:
                raise ValueError(f"未知 transition validation mode: {mode!r}")
    finally:
        model.train(was_training)
    pred_world = pred * pc_scale + pc_center
    return pred_world, k_in_window


def transition_state_unit(data_dir) -> str:
    files = sorted(glob.glob(os.path.join(str(data_dir), "*.npz")))
    if not files:
        raise FileNotFoundError(f"val 目录没有 NPZ: {data_dir}")
    with np.load(files[0], allow_pickle=False) as data:
        unit = (str(data["state_length_unit"].item())
                if "state_length_unit" in data else "px")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", unit):
        raise ValueError(f"非法 state_length_unit: {unit!r}")
    return unit


def evaluate_native_node_metrics(
    model,
    data_dir,
    config,
    device,
    *,
    mode,
    window_len=None,
    max_steps=None,
    seq_idx=0,
    return_details=False,
):
    """Return native-unit node metrics shared by training and the evaluation CLI."""
    files = sorted(glob.glob(os.path.join(str(data_dir), "*.npz")))
    if not files:
        raise FileNotFoundError(f"val 目录没有 NPZ: {data_dir}")
    if seq_idx is None:
        results = [evaluate_native_node_metrics(
            model, data_dir, config, device, mode=mode,
            window_len=window_len, max_steps=max_steps, seq_idx=index,
            return_details=True) for index in range(len(files))]
        units = {details["unit"] for _, details in results}
        if len(units) != 1:
            raise ValueError(f"validation 文件长度单位不一致: {sorted(units)}")
        unit = units.pop()
        rows = np.array([
            metrics["validation.prediction_rows"] for metrics, _ in results],
            dtype=np.float64)
        total_rows = float(rows.sum())
        if total_rows <= 0:
            raise ValueError("所有 validation 序列均无模型预测行")
        aggregated = {
            f"validation.node_mean_{unit}": float(sum(
                metrics[f"validation.node_mean_{unit}"] * count
                for (metrics, _), count in zip(results, rows)) / total_rows),
            f"validation.endpoint_{unit}": float(sum(
                metrics[f"validation.endpoint_{unit}"] * count
                for (metrics, _), count in zip(results, rows)) / total_rows),
            f"validation.max_node_{unit}": float(max(
                metrics[f"validation.max_node_{unit}"]
                for metrics, _ in results)),
            "validation.prediction_rows": total_rows,
        }
        if not return_details:
            return aggregated
        return aggregated, {
            "unit": unit,
            "sequences": [details for _, details in results],
            "selected_npz": [details["selected_npz"] for _, details in results],
        }
    if seq_idx < 0 or seq_idx >= len(files):
        raise IndexError(f"seq_idx 越界: {seq_idx}; files={len(files)}")
    with np.load(files[seq_idx], allow_pickle=False) as raw:
        raw_actions = raw["actions"].astype(np.float32)
        positions = raw["positions"].astype(np.float32)
        evaluation_mask = (raw["evaluation_mask"].astype(bool)
                           if "evaluation_mask" in raw else None)
        unit = (str(raw["state_length_unit"].item())
                if "state_length_unit" in raw else "px")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", unit):
        raise ValueError(f"非法 state_length_unit: {unit!r}")
    if len(raw_actions) != len(positions):
        raise ValueError(
            f"validation actions/positions 帧数不一致: "
            f"{len(raw_actions)} != {len(positions)}")
    if evaluation_mask is not None and len(evaluation_mask) != len(positions):
        raise ValueError("validation evaluation_mask 帧数与 positions 不一致")

    action_view = config.get("action_view") or {}
    channels = tuple(action_view.get(
        "model_action_channels", range(raw_actions.shape[1])))
    actions = project_actions(raw_actions, channels).astype(np.float32)
    action_dim = getattr(model, "action_dim", actions.shape[1])
    if actions.shape[1] != action_dim:
        raise ValueError(
            f"validation 动作视图 {actions.shape[1]}D 与模型 {action_dim}D 不一致")
    norm = getattr(model, "action_norm_factor", 1.0)
    norm_factor = norm.item() if isinstance(norm, torch.Tensor) else float(norm)
    if not np.isfinite(norm_factor) or norm_factor <= 0:
        raise ValueError(f"非法 action_norm_factor: {norm_factor!r}")
    window_size = int(config.get("temporal", {}).get(
        "window_size", getattr(model, "window_size", 40)))
    K = int(window_len or getattr(model, "episode_len", window_size))
    anchor_index = 0
    if (mode in ("ishsm", "ishsm_zero", "ishsm_periodic") and
            evaluation_mask is not None and evaluation_mask.any()):
        anchor_index = max(int(np.flatnonzero(evaluation_mask)[0]) - 1, 0)
    pred_world, k_in_window = per_frame_rollout(
        model, mode, actions, positions, window_size, norm_factor, device,
        K=K, max_steps=max_steps, anchor_index=anchor_index)
    T = pred_world.shape[0]
    gt_world = positions[:T].transpose(0, 2, 1)
    prediction_valid = (
        k_in_window >= 0 if mode in (
            "open_loop", "ishsm", "ishsm_zero", "ishsm_periodic")
        else np.ones(T, dtype=bool))
    if evaluation_mask is not None:
        prediction_valid &= evaluation_mask[:T]
    if not prediction_valid.any():
        raise ValueError("当前 validation 范围没有模型预测行")
    node_errors = np.sqrt(
        ((pred_world[:, :, :2] - gt_world[:, :, :2]) ** 2).sum(-1))
    per_frame_node = node_errors.mean(axis=1)
    per_frame_endpoint = node_errors[:, -1]
    metrics = {
        f"validation.node_mean_{unit}": float(
            per_frame_node[prediction_valid].mean()),
        f"validation.endpoint_{unit}": float(
            per_frame_endpoint[prediction_valid].mean()),
        f"validation.max_node_{unit}": float(
            node_errors[prediction_valid].max()),
        "validation.prediction_rows": float(prediction_valid.sum()),
    }
    if not return_details:
        return metrics
    return metrics, {
        "unit": unit,
        "per_frame_node": per_frame_node,
        "prediction_valid": prediction_valid,
        "horizon": k_in_window,
        "anchor_index": anchor_index,
        "selected_npz": files[seq_idx],
    }
