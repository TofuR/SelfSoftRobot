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
                      device, K=40, max_steps=None):
    """Predict every usable row with GT-observed or windowed OpenLoop semantics."""
    if K <= 0:
        raise ValueError("OpenLoop validation K 必须为正整数")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("validation max_steps 必须为正整数或 null")
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)
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
    """Return the same native-unit node mean used by the periodic watcher."""
    files = sorted(glob.glob(os.path.join(str(data_dir), "*.npz")))
    if not files:
        raise FileNotFoundError(f"val 目录没有 NPZ: {data_dir}")
    if seq_idx < 0 or seq_idx >= len(files):
        raise IndexError(f"seq_idx 越界: {seq_idx}; files={len(files)}")
    with np.load(files[seq_idx], allow_pickle=False) as raw:
        raw_actions = raw["actions"].astype(np.float32)
        positions = raw["positions"].astype(np.float32)
        unit = (str(raw["state_length_unit"].item())
                if "state_length_unit" in raw else "px")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", unit):
        raise ValueError(f"非法 state_length_unit: {unit!r}")
    if len(raw_actions) != len(positions):
        raise ValueError(
            f"validation actions/positions 帧数不一致: "
            f"{len(raw_actions)} != {len(positions)}")

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
    pred_world, k_in_window = per_frame_rollout(
        model, mode, actions, positions, window_size, norm_factor, device,
        K=K, max_steps=max_steps)
    T = pred_world.shape[0]
    gt_world = positions[:T].transpose(0, 2, 1)
    prediction_valid = (
        k_in_window >= 0 if mode == "open_loop" else np.ones(T, dtype=bool))
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
        "selected_npz": files[seq_idx],
    }
