#!/usr/bin/env python3
"""Interrogate a trained HereditaryOperatorModel in physical output units.

This is an analysis-only tool: it does not modify parameters or redefine the
forward pass.  It reports realized branch/mode contributions, fixed-checkpoint
counterfactual interventions, state activity, and empirical step-hold traces.
Raw operator weights are deliberately not treated as a material spectrum.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.action_view import project_actions  # noqa: E402
from src.evaluation.transition_metrics import build_action_window  # noqa: E402
from src.utils.model_loader import load_model  # noqa: E402


def _residual(model, action, q, d):
    batch = action.shape[0]
    value = torch.cat(
        [action, q.reshape(batch, -1), d.reshape(batch, -1)], dim=-1)
    scale = torch.clamp(
        model.residual_scale, max=model.residual_scale_max)
    return scale * model.residual(value).view(batch, model.n_nodes, 3)


def _decompose_step(model, action_window, previous_state):
    """Mirror ``forward`` and expose its strictly additive branches."""
    action = action_window[:, -1]
    if previous_state is None:
        p, h = model._burn_in(action_window[:, :-1])
    else:
        p, h = model._unpack_state(previous_state)
    drive = model.drive(action)
    p, q = model.play.step(p, drive)
    h = model.maxwell.step(h, drive)
    deficit = h - drive.unsqueeze(-1)

    static = model.static_bias.unsqueeze(0) + torch.einsum(
        "bc,cnd->bnd", drive, model.static_dirs)
    play_modes = model.play_modes(static)
    maxwell_modes = model.maxwell_modes(static)
    play_pair = (
        model.play.weights[None, :, :, None, None]
        * q[:, :, :, None, None]
        * play_modes.permute(1, 0, 2, 3)[:, None, :, :, :]
    )
    maxwell_pair = (
        model.maxwell.weights[None, :, :, None, None]
        * deficit[:, :, :, None, None]
        * maxwell_modes.permute(1, 0, 2, 3)[:, None, :, :, :]
    )
    play = play_pair.sum(dim=(1, 2))
    maxwell = maxwell_pair.sum(dim=(1, 2))
    residual = _residual(model, action, q, deficit)
    residual_no_play = _residual(model, action, torch.zeros_like(q), deficit)
    residual_no_maxwell = _residual(
        model, action, q, torch.zeros_like(deficit))
    residual_no_memory = _residual(
        model, action, torch.zeros_like(q), torch.zeros_like(deficit))
    residual_memory = residual - residual_no_memory
    residual_play_shapley = 0.5 * (
        (residual_no_maxwell - residual_no_memory)
        + (residual - residual_no_play))
    residual_maxwell_shapley = 0.5 * (
        (residual_no_play - residual_no_memory)
        + (residual - residual_no_maxwell))
    full = static + play + maxwell + residual

    return {
        "full": full,
        "static_only": static,
        "structured_only": static + play + maxwell,
        "no_explicit_play": full - play,
        "no_explicit_maxwell": full - maxwell,
        "no_residual": full - residual,
        "no_play_information": static + maxwell + residual_no_play,
        "no_maxwell_information": static + play + residual_no_maxwell,
        "no_memory_information": static + residual_no_memory,
        "component_play": play,
        "component_maxwell": maxwell,
        "component_residual": residual,
        "component_residual_equilibrium": residual_no_memory,
        "component_residual_memory": residual_memory,
        "component_residual_play_shapley": residual_play_shapley,
        "component_residual_maxwell_shapley": residual_maxwell_shapley,
        "component_total_play_information": play + residual_play_shapley,
        "component_total_maxwell_information": (
            maxwell + residual_maxwell_shapley),
        "component_total_memory": play + maxwell + residual_memory,
        "play_pair": play_pair,
        "maxwell_pair": maxwell_pair,
        "q": q,
        "deficit": deficit,
        "drive": drive,
        "state": model._pack_state(p, h),
    }


def _physical_prediction(value, scale, center):
    return value.detach().cpu().numpy() * scale + center


def _physical_delta(value, scale):
    return value.detach().cpu().numpy() * scale


def _magnitude(values):
    xy_norm = np.linalg.norm(values[..., :2], axis=-1)
    return {
        "node_vector_mean_mm": float(xy_norm.mean()),
        "node_vector_rms_mm": float(np.sqrt(np.mean(xy_norm ** 2))),
        "endpoint_vector_mean_mm": float(xy_norm[:, -1].mean()),
        "endpoint_vector_rms_mm": float(np.sqrt(
            np.mean(xy_norm[:, -1] ** 2))),
        "peak_node_vector_mm": float(xy_norm.max()),
        "per_node_vector_mean_mm": xy_norm.mean(axis=0).tolist(),
    }


def _prediction_metric(prediction, ground_truth):
    error = np.linalg.norm(
        prediction[..., :2] - ground_truth[..., :2], axis=-1)
    return {
        "node_mean_mm": float(error.mean()),
        "endpoint_mean_mm": float(error[:, -1].mean()),
        "node_p90_mm": float(np.quantile(error.mean(axis=1), 0.9)),
        "endpoint_p90_mm": float(np.quantile(error[:, -1], 0.9)),
    }


def _correlation_matrix(components):
    names = list(components)
    flattened = []
    for name in names:
        values = components[name][..., :2].reshape(-1).astype(np.float64)
        values -= values.mean()
        flattened.append(values)
    matrix = np.eye(len(names), dtype=np.float64)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            denom = np.linalg.norm(flattened[i]) * np.linalg.norm(flattened[j])
            value = float(flattened[i] @ flattened[j] / denom) if denom else 0.0
            matrix[i, j] = matrix[j, i] = value
    return {"labels": names, "pearson_flattened": matrix.tolist()}


def _step_hold(model, normalized_actions, scale, center, window_size, steps):
    quantiles = np.quantile(normalized_actions, [0.25, 0.5, 0.75], axis=0)
    traces = []
    for channel in range(model.action_dim):
        baseline = quantiles[1].astype(np.float32)
        baseline[channel] = quantiles[0, channel]
        target = baseline.copy()
        target[channel] = quantiles[2, channel]
        if abs(float(target[channel] - baseline[channel])) < 1e-8:
            continue
        history = np.repeat(baseline[None, :], window_size, axis=0)
        history_tensor = torch.from_numpy(history).float().unsqueeze(0)
        state = model.init_z_from_action(history_tensor)
        for index in range(steps):
            window = history.copy()
            window[-1] = target
            out = _decompose_step(
                model, torch.from_numpy(window).float().unsqueeze(0), state)
            state = out["state"]
            row = {
                "channel": channel,
                "step": index,
                "time_s": index * float(model.dt),
                "input_low": float(baseline[channel]),
                "input_high": float(target[channel]),
            }
            for name in ("component_play", "component_maxwell",
                         "component_residual"):
                delta = _physical_delta(out[name], scale)[0]
                row[f"{name}_endpoint_norm_mm"] = float(
                    np.linalg.norm(delta[-1, :2]))
            row["full_endpoint_x_mm"] = float(
                _physical_prediction(out["full"], scale, center)[0, -1, 0])
            row["full_endpoint_y_mm"] = float(
                _physical_prediction(out["full"], scale, center)[0, -1, 1])
            traces.append(row)
            history[:] = target
    return traces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--step-hold-steps", type=int, default=61)
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=False)
    info = load_model(args.checkpoint, device=args.device)
    model = info["model"].eval()
    if type(model).__name__ != "HereditaryOperatorModel":
        raise TypeError("checkpoint must contain HereditaryOperatorModel")
    config = info.get("saved_config") or {}
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    window_size = int(config.get("window_size", model.window_size))
    norm_factor = float(model.action_norm_factor.detach().cpu())
    scale = model.pc_scale.detach().cpu().numpy().reshape(1, 1, 3)
    center = model.pc_center.detach().cpu().numpy().reshape(1, 1, 3)

    prediction_names = (
        "full", "static_only", "structured_only", "no_explicit_play",
        "no_explicit_maxwell", "no_residual", "no_play_information",
        "no_maxwell_information", "no_memory_information")
    collected = {name: [] for name in prediction_names}
    components = {name: [] for name in (
        "component_play", "component_maxwell", "component_residual",
        "component_residual_equilibrium", "component_residual_memory",
        "component_residual_play_shapley",
        "component_residual_maxwell_shapley",
        "component_total_play_information",
        "component_total_maxwell_information",
        "component_total_memory")}
    play_pairs, maxwell_pairs, q_values, deficit_values = [], [], [], []
    ground_truth, input_actions = [], []
    per_frame_rows = []

    with torch.no_grad():
        for path_string in sorted(glob.glob(str(Path(args.data_dir) / "*.npz"))):
            with np.load(path_string, allow_pickle=False) as raw:
                raw_actions = raw["actions"].astype(np.float32)
                actions = project_actions(raw_actions, channels).astype(np.float32)
                positions = raw["positions"].astype(np.float32).transpose(0, 2, 1)
                mask = (raw["evaluation_mask"].astype(bool)
                        if "evaluation_mask" in raw else
                        np.ones(len(actions), dtype=bool))
            normalized_actions = actions / norm_factor
            state = None
            sequence_outputs = {name: [] for name in prediction_names}
            sequence_components = {name: [] for name in components}
            sequence_play_pairs, sequence_maxwell_pairs = [], []
            sequence_q, sequence_deficit = [], []
            for frame in range(len(actions)):
                window = build_action_window(
                    normalized_actions, frame, window_size)
                tensor = torch.from_numpy(window).float().unsqueeze(0).to(args.device)
                if state is None:
                    state = model.init_z_from_action(tensor)
                out = _decompose_step(model, tensor, state)
                state = out["state"]
                for name in prediction_names:
                    sequence_outputs[name].append(
                        _physical_prediction(out[name], scale, center)[0])
                for name in components:
                    sequence_components[name].append(
                        _physical_delta(out[name], scale)[0])
                sequence_play_pairs.append(
                    _physical_delta(out["play_pair"], scale)[0])
                sequence_maxwell_pairs.append(
                    _physical_delta(out["maxwell_pair"], scale)[0])
                sequence_q.append(out["q"].detach().cpu().numpy()[0])
                sequence_deficit.append(
                    out["deficit"].detach().cpu().numpy()[0])
            sequence_outputs = {
                name: np.asarray(value)
                for name, value in sequence_outputs.items()
            }
            sequence_components = {
                name: np.asarray(value)
                for name, value in sequence_components.items()
            }
            sequence_play_pairs = np.asarray(sequence_play_pairs)
            sequence_maxwell_pairs = np.asarray(sequence_maxwell_pairs)
            sequence_q = np.asarray(sequence_q)
            sequence_deficit = np.asarray(sequence_deficit)
            valid_frames = np.flatnonzero(mask)
            for frame in valid_frames:
                full_error = np.linalg.norm(
                    sequence_outputs["full"][frame, ..., :2]
                    - positions[frame, ..., :2], axis=-1)
                per_frame_rows.append({
                    "sequence": Path(path_string).name,
                    "frame": int(frame),
                    "node_mean_mm": float(full_error.mean()),
                    "endpoint_mm": float(full_error[-1]),
                    **{
                        f"{name}_node_norm_mm": float(np.linalg.norm(
                            sequence_components[name][frame, ..., :2],
                            axis=-1).mean())
                        for name in components
                    },
                })
            for name in prediction_names:
                collected[name].append(sequence_outputs[name][valid_frames])
            for name in components:
                components[name].append(sequence_components[name][valid_frames])
            play_pairs.append(sequence_play_pairs[valid_frames])
            maxwell_pairs.append(sequence_maxwell_pairs[valid_frames])
            q_values.append(sequence_q[valid_frames])
            deficit_values.append(sequence_deficit[valid_frames])
            ground_truth.append(positions[valid_frames])
            input_actions.append(normalized_actions[valid_frames])

    collected = {name: np.concatenate(value) for name, value in collected.items()}
    components = {name: np.concatenate(value) for name, value in components.items()}
    play_pairs = np.concatenate(play_pairs)
    maxwell_pairs = np.concatenate(maxwell_pairs)
    q_values = np.concatenate(q_values)
    deficit_values = np.concatenate(deficit_values)
    ground_truth = np.concatenate(ground_truth)
    input_actions = np.concatenate(input_actions)

    metrics = {
        name: _prediction_metric(value, ground_truth)
        for name, value in collected.items()
    }
    full_metric = metrics["full"]
    interventions = {}
    for name, values in metrics.items():
        if name == "full":
            continue
        interventions[name] = {
            **values,
            "delta_node_mean_vs_full_mm": (
                values["node_mean_mm"] - full_metric["node_mean_mm"]),
            "delta_endpoint_mean_vs_full_mm": (
                values["endpoint_mean_mm"] - full_metric["endpoint_mean_mm"]),
        }

    play_by_threshold = play_pairs.sum(axis=1)
    maxwell_by_tau = maxwell_pairs.sum(axis=1)
    play_by_channel = play_pairs.sum(axis=2)
    maxwell_by_channel = maxwell_pairs.sum(axis=2)
    thresholds = model.play.thresholds.detach().cpu().numpy()
    taus = model.maxwell.taus.detach().cpu().numpy()
    q_ratio = np.abs(q_values) / thresholds[None, None, :]

    report = {
        "schema": "hereditary_interpretability_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "data_role": "development; not independent material identification",
        "frames": int(len(ground_truth)),
        "model": {
            "action_dim": model.action_dim,
            "n_nodes": model.n_nodes,
            "burnin_mode": model.burnin_mode,
            "dt_s": float(model.dt),
            "play_thresholds": thresholds.tolist(),
            "maxwell_taus_s": taus.tolist(),
            "residual_scale_raw": float(model.residual_scale.detach().cpu()),
            "residual_scale_effective": min(
                float(model.residual_scale.detach().cpu()),
                float(model.residual_scale_max)),
            "residual_scale_max": float(model.residual_scale_max),
            "drive_hinge_weights": model.drive.weights.detach().cpu().tolist(),
            "parameter_counts": {
                "total": int(sum(value.numel() for value in model.parameters())),
                "residual_network_and_scale": int(sum(
                    value.numel() for name, value in model.named_parameters()
                    if name.startswith("residual.") or name == "residual_scale")),
            },
            "normalization": {
                "pc_center": center.reshape(-1).tolist(),
                "pc_scale": scale.reshape(-1).tolist(),
                "action_norm_factor": norm_factor,
            },
        },
        "input_normalized_quantiles": {
            str(q): np.quantile(input_actions, q, axis=0).tolist()
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
        "prediction_metrics": metrics,
        "fixed_checkpoint_interventions": interventions,
        "realized_family_contributions": {
            name: _magnitude(value) for name, value in components.items()
        },
        "realized_play_by_threshold": [
            {"threshold": float(value), **_magnitude(play_by_threshold[:, i])}
            for i, value in enumerate(thresholds)
        ],
        "realized_maxwell_by_tau": [
            {"tau_s": float(value), **_magnitude(maxwell_by_tau[:, i])}
            for i, value in enumerate(taus)
        ],
        "realized_play_by_channel": [
            {"channel": i, **_magnitude(play_by_channel[:, i])}
            for i in range(model.action_dim)
        ],
        "realized_maxwell_by_channel": [
            {"channel": i, **_magnitude(maxwell_by_channel[:, i])}
            for i in range(model.action_dim)
        ],
        "mode_cancellation": {
            "play_sum_mode_mean_over_family_mean": float(sum(
                _magnitude(play_by_threshold[:, i])["node_vector_mean_mm"]
                for i in range(len(thresholds))) /
                _magnitude(components["component_play"])["node_vector_mean_mm"]),
            "maxwell_sum_mode_mean_over_family_mean": float(sum(
                _magnitude(maxwell_by_tau[:, i])["node_vector_mean_mm"]
                for i in range(len(taus))) /
                _magnitude(components["component_maxwell"])["node_vector_mean_mm"]),
            "play_mode_correlation": _correlation_matrix({
                f"r={thresholds[i]:.6g}": play_by_threshold[:, i]
                for i in range(len(thresholds))
            }),
            "maxwell_mode_correlation": _correlation_matrix({
                f"tau={taus[i]:.6g}s": maxwell_by_tau[:, i]
                for i in range(len(taus))
            }),
        },
        "state_activity": {
            "play_mean_abs_over_threshold": q_ratio.mean(axis=0).tolist(),
            "play_saturation_fraction_ge_0_95": (
                q_ratio >= 0.95).mean(axis=0).tolist(),
            "maxwell_deficit_rms": np.sqrt(
                np.mean(deficit_values ** 2, axis=0)).tolist(),
        },
        "component_correlation": _correlation_matrix(components),
        "interpretation_guardrails": [
            "All magnitudes are realized output-space contributions, not raw material spectra.",
            "Fixed-checkpoint removal tests show model reliance, not component necessity after retraining.",
            "Residual receives q and d; information interventions zero both its matching input and explicit branch.",
            "The residual is exactly split into an equilibrium term r(a,0,0) and a memory-dependent difference r(a,q,d)-r(a,0,0).",
            "Development trajectories cannot establish material causality or out-of-distribution rate generalization.",
        ],
    }

    step_rows = _step_hold(
        model.cpu(), input_actions, scale, center, window_size,
        args.step_hold_steps)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    with (output_dir / "per_frame.csv").open(
            "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_frame_rows[0].keys())
        writer.writeheader()
        writer.writerows(per_frame_rows)
    if step_rows:
        with (output_dir / "step_hold.csv").open(
                "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=step_rows[0].keys())
            writer.writeheader()
            writer.writerows(step_rows)
    (output_dir / "ANALYSIS_COMPLETE").touch(exist_ok=False)
    print(json.dumps({
        "prediction_metrics": report["prediction_metrics"],
        "fixed_checkpoint_interventions": report["fixed_checkpoint_interventions"],
        "realized_family_contributions": report["realized_family_contributions"],
        "realized_play_by_threshold": report["realized_play_by_threshold"],
        "realized_maxwell_by_tau": report["realized_maxwell_by_tau"],
        "component_correlation": report["component_correlation"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
