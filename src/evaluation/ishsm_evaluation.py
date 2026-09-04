"""Deployment-aligned rollout helpers for ISHSM analysis."""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.transition_metrics import build_action_window


def rollout_ishsm_sequence(
        model, actions, positions, *, protocol, window_size, norm_factor,
        device, reanchor_interval=None, anchor_index=0):
    """Roll out one sequence; frame zero is an observation and is never scored."""
    if protocol not in {"single_anchor", "zero_init", "periodic", "h0"}:
        raise ValueError(f"未知 ISHSM protocol: {protocol}")
    if protocol == "periodic" and (
            reanchor_interval is None or reanchor_interval <= 0):
        raise ValueError("periodic protocol 需要正 reanchor_interval")
    actions = np.asarray(actions, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    actions_norm = actions / float(norm_factor)
    total = len(actions_norm)
    if positions.shape[0] != total:
        raise ValueError("actions/positions 帧数不一致")
    anchor_index = int(anchor_index)
    if not 0 <= anchor_index < total:
        raise ValueError("anchor_index 必须位于序列范围内")

    pc_center = model.pc_center.view(3).detach().cpu().numpy()
    pc_scale = model.pc_scale.view(3).detach().cpu().numpy()
    predictions = np.full((total, positions.shape[2], 3), np.nan, np.float32)
    states = np.full((total, model.z_dim), np.nan, np.float32)
    horizon = np.full(total, -1, np.int32)

    def window(t):
        value = build_action_window(actions_norm, t, window_size)
        return torch.from_numpy(value).float().unsqueeze(0).to(device)

    def normalized_observation(t):
        skeleton = positions[t].T.astype(np.float32)
        skeleton = (skeleton - pc_center) / pc_scale
        return torch.from_numpy(skeleton).float().unsqueeze(0).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            z_t = None
            last_anchor = anchor_index
            start = 1 if protocol == "h0" else anchor_index + 1
            for t in range(start, total):
                action_window = window(t)
                if protocol == "h0":
                    z_t = torch.zeros(
                        1, model.z_dim, dtype=action_window.dtype,
                        device=device)
                    pred_norm = model.decode_state(action_window[:, -1], z_t)
                else:
                    should_anchor = (
                        z_t is None or
                        (protocol == "periodic" and
                         (t - 1 - anchor_index) % int(reanchor_interval) == 0))
                    if should_anchor:
                        if protocol == "zero_init":
                            z_t = model.init_z_from_action(action_window)
                        else:
                            observation = normalized_observation(t - 1)
                            if (protocol == "periodic" and z_t is not None and
                                    hasattr(model, "assimilate_observation")):
                                z_t = model.assimilate_observation(
                                    action_window, observation, z_t)
                            else:
                                z_t = model.init_rollout_state(
                                    action_window, observation)
                        last_anchor = t - 1
                    out = model.forward(action_window, prev_z=z_t)
                    pred_norm = out["skeleton"]
                    z_t = out["latent_z"]
                predictions[t] = (
                    pred_norm.squeeze(0).cpu().numpy() * pc_scale + pc_center)
                states[t] = z_t.squeeze(0).cpu().numpy()
                horizon[t] = 0 if protocol == "h0" else t - last_anchor - 1
    finally:
        model.train(was_training)
    return {"predictions": predictions, "states": states, "horizon": horizon}
