#!/usr/bin/env python3
"""Compare HOV2.1 variants and diagnose their geometry/memory bottlenecks.

The report is development-only.  It uses the validation-selected checkpoints,
evaluates both continuous-state and cold-restart-40 protocols, performs paired
moving-block comparisons, and measures the best reconstruction attainable by
the frozen generalized-coordinate subspace.  Counterfactual branch removals
are fixed-checkpoint reliance tests, not retrained ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.action_view import project_actions  # noqa: E402
from src.evaluation.real_transition_validation import per_frame_rollout  # noqa: E402
from src.evaluation.transition_metrics import build_action_window  # noqa: E402
from src.models.model_ishsm import (  # noqa: E402
    generalized_to_skeleton,
    skeleton_to_generalized,
)
from src.utils.model_loader import load_model  # noqa: E402


PROTOCOLS = (("continuous", "gt"), ("cold_restart_40", "open_loop"))


def _metric(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = np.linalg.norm(prediction[..., :2] - truth[..., :2], axis=-1)
    return {
        "node_mean_mm": float(error.mean()),
        "node_p90_mm": float(np.quantile(error, 0.9)),
        "endpoint_mean_mm": float(error[:, -1].mean()),
        "endpoint_p90_mm": float(np.quantile(error[:, -1], 0.9)),
        "per_node_mean_mm": error.mean(axis=0).tolist(),
    }


def _decode_physical(model, action: torch.Tensor, memory: torch.Tensor):
    normalized = model._decode_generalized(action, memory)
    return (normalized * model.pc_scale.to(normalized) +
            model.pc_center.to(normalized))


def _decompose_step(model, action_window: torch.Tensor, previous_state):
    action = action_window[:, -1]
    if previous_state is None:
        p, h = model._burn_in(action_window[:, :-1])
    else:
        p, h = model._unpack_state(previous_state)
    drive = model.drive(action)
    p, q = model.play.step(p, drive)
    h = model.maxwell.step(h, drive)
    deficit = h - drive.unsqueeze(-1)
    pi, maxwell, _, _ = model._structured_memory(q, deficit)
    zero_q = torch.zeros_like(q)
    zero_deficit = torch.zeros_like(deficit)
    residual = model._memory_residual(q, deficit)
    residual_no_pi = model._memory_residual(zero_q, deficit)
    residual_no_maxwell = model._memory_residual(q, zero_deficit)
    zero_memory = torch.zeros_like(pi)
    memories = {
        "full": pi + maxwell + residual,
        "equilibrium_only": zero_memory,
        "pi_only": pi,
        "maxwell_only": maxwell,
        "no_pi_information": maxwell + residual_no_pi,
        "no_maxwell_information": pi + residual_no_maxwell,
        "no_neural_residual": pi + maxwell,
    }
    predictions = {
        name: _decode_physical(model, action, value)
        for name, value in memories.items()
    }
    return {
        "state": model._pack_state(p, h),
        "predictions": predictions,
        "memories": memories,
        "q": q,
        "deficit": deficit,
    }


def _load_sequence(path: Path, channels) -> tuple:
    with np.load(path, allow_pickle=False) as raw:
        actions = project_actions(
            raw["actions"], channels).astype(np.float32)
        positions = raw["positions"].astype(np.float32).transpose(0, 2, 1)
        mask = (raw["evaluation_mask"].astype(bool)
                if "evaluation_mask" in raw else
                np.ones(len(actions), dtype=bool))
    return actions, positions, mask


def _evaluate_protocols(model, config, data_dir: Path, device):
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    window_size = int(config.get("window_size", model.window_size))
    norm = float(model.action_norm_factor.detach().cpu())
    rows = []
    for path in sorted(data_dir.glob("*.npz")):
        actions, positions, mask = _load_sequence(path, channels)
        for protocol, mode in PROTOCOLS:
            prediction, horizon = per_frame_rollout(
                model, mode, actions, positions.transpose(0, 2, 1),
                window_size, norm, device, K=40)
            valid = mask.copy()
            if mode == "open_loop":
                valid &= horizon >= 0
            error = np.linalg.norm(
                prediction[..., :2] - positions[..., :2], axis=-1)
            for frame in np.flatnonzero(valid):
                rows.append({
                    "protocol": protocol,
                    "sequence": path.name,
                    "frame": int(frame),
                    "horizon": (int(horizon[frame]) if mode == "open_loop"
                                else int(frame - np.flatnonzero(mask)[0])),
                    "node": float(error[frame].mean()),
                    "endpoint": float(error[frame, -1]),
                })
    aggregate = {}
    keyed = {}
    for protocol, _ in PROTOCOLS:
        selected = [row for row in rows if row["protocol"] == protocol]
        aggregate[protocol] = {
            "frames": len(selected),
            "node_mean_mm": float(np.mean([row["node"] for row in selected])),
            "endpoint_mean_mm": float(np.mean(
                [row["endpoint"] for row in selected])),
        }
        for row in selected:
            keyed[(protocol, row["sequence"], row["frame"])] = {
                "node": row["node"], "endpoint": row["endpoint"],
                "horizon": row["horizon"],
            }
    return {"aggregate": aggregate, "rows": keyed, "csv_rows": rows}


def _evaluate_interventions(model, config, data_dir: Path, device):
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    window_size = int(config.get("window_size", model.window_size))
    norm = float(model.action_norm_factor.detach().cpu())
    names = ("full", "equilibrium_only", "pi_only", "maxwell_only",
             "no_pi_information", "no_maxwell_information",
             "no_neural_residual")
    predictions = {name: [] for name in names}
    truths = []
    memory_norms = defaultdict(list)
    q_values, deficit_values = [], []
    with torch.no_grad():
        for path in sorted(data_dir.glob("*.npz")):
            actions, positions, mask = _load_sequence(path, channels)
            normalized = actions / norm
            state = None
            sequence = {name: [] for name in names}
            for frame in range(len(actions)):
                window = torch.from_numpy(build_action_window(
                    normalized, frame, window_size)).float().unsqueeze(0)
                window = window.to(device)
                if state is None:
                    state = model.init_z_from_action(window)
                output = _decompose_step(model, window, state)
                state = output["state"]
                for name in names:
                    sequence[name].append(
                        output["predictions"][name][0].cpu().numpy())
                for name, value in output["memories"].items():
                    memory_norms[name].append(float(
                        torch.linalg.vector_norm(value[0]).cpu()))
                q_values.append(output["q"][0].cpu().numpy())
                deficit_values.append(output["deficit"][0].cpu().numpy())
            valid = np.flatnonzero(mask)
            for name in names:
                predictions[name].append(np.asarray(sequence[name])[valid])
            truths.append(positions[valid])
    truth = np.concatenate(truths)
    prediction = {name: np.concatenate(value)
                  for name, value in predictions.items()}
    metrics = {name: _metric(value, truth)
               for name, value in prediction.items()}
    full = metrics["full"]
    for name, value in metrics.items():
        value["delta_node_mean_vs_full_mm"] = (
            value["node_mean_mm"] - full["node_mean_mm"])
        value["delta_endpoint_mean_vs_full_mm"] = (
            value["endpoint_mean_mm"] - full["endpoint_mean_mm"])
    return {
        "metrics": metrics,
        "memory_generalized_norm": {
            name: {"mean": float(np.mean(value)),
                   "p90": float(np.quantile(value, 0.9)),
                   "max": float(np.max(value))}
            for name, value in memory_norms.items()
        },
        "state_activity": {
            "play_mean_abs_over_threshold": (
                np.abs(np.asarray(q_values)) /
                model.play.thresholds.detach().cpu().numpy()[None, None, :]
            ).mean(axis=0).tolist(),
            "maxwell_deficit_rms": np.sqrt(
                np.mean(np.asarray(deficit_values) ** 2, axis=0)).tolist(),
        },
    }


def _oracle_geometry(model, config, fit_dir: Path, data_dir: Path):
    del fit_dir
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    norm = float(model.action_norm_factor.detach().cpu())
    outputs = defaultdict(list)
    truths = []
    for path in sorted(data_dir.glob("*.npz")):
        with torch.no_grad():
            actions, positions, mask = _load_sequence(path, channels)
            action = torch.from_numpy(actions / norm)
            truth = torch.from_numpy(positions)
            bend, length, _ = skeleton_to_generalized(
                truth, model.reference_segment_lengths,
                model.section_intervals)
            bend_reference, _ = model._reference(action)
            residual = bend - bend_reference
            coefficients = residual @ torch.linalg.pinv(model.bend_basis).T
            projected = bend_reference + coefficients @ model.bend_basis.T
            angle_projection = generalized_to_skeleton(
                projected, length, model.reference_segment_lengths,
                model.section_intervals, model.base_position)
            full = generalized_to_skeleton(
                bend, length, model.reference_segment_lengths,
                model.section_intervals, model.base_position)
        # Optimize one coefficient vector per scored frame directly in node
        # space.  This is an optimistic representational audit: it uses the
        # true shape and is not a deployable predictor.
        scored = torch.as_tensor(np.flatnonzero(mask))
        coefficient = torch.nn.Parameter(coefficients[scored].clone())
        optimizer = torch.optim.Adam([coefficient], lr=0.03)
        target = truth[scored]
        reference_scored = bend_reference[scored]
        length_scored = length[scored]
        for _ in range(250):
            optimizer.zero_grad()
            candidate = generalized_to_skeleton(
                reference_scored + coefficient @ model.bend_basis.T,
                length_scored, model.reference_segment_lengths,
                model.section_intervals, model.base_position)
            loss = (candidate[..., :2] - target[..., :2]).square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            optimized = generalized_to_skeleton(
                reference_scored + coefficient @ model.bend_basis.T,
                length_scored, model.reference_segment_lengths,
                model.section_intervals, model.base_position)
        valid = np.flatnonzero(mask)
        outputs["angle_projection_in_current_subspace"].append(
            angle_projection.numpy()[valid])
        outputs["node_optimized_current_subspace"].append(
            optimized.detach().numpy())
        outputs["all_local_bends_oracle"].append(full.numpy()[valid])
        truths.append(positions[valid])
    truth = np.concatenate(truths)
    return {name: _metric(np.concatenate(values), truth)
            for name, values in outputs.items()}


def _paired_improvement(baseline, candidate, protocol, *, samples=10000,
                        block_length=20, seed=42):
    paired = defaultdict(lambda: {"node": [], "endpoint": []})
    for (name, sequence, frame), baseline_values in baseline["rows"].items():
        if name != protocol:
            continue
        key = (protocol, sequence, frame)
        if key not in candidate["rows"]:
            continue
        for metric in ("node", "endpoint"):
            paired[sequence][metric].append((
                frame, baseline_values[metric] - candidate["rows"][key][metric]))
    if not paired:
        raise ValueError(f"没有配对帧: {protocol}")
    rng = np.random.default_rng(seed)
    output = {}
    for metric in ("node", "endpoint"):
        arrays = [np.asarray([value for _, value in sorted(paired[sequence][metric])])
                  for sequence in sorted(paired)]
        estimate = float(np.concatenate(arrays).mean())
        bootstrap = np.empty(samples)
        for index in range(samples):
            selected = []
            for values in arrays:
                starts = rng.integers(
                    0, len(values),
                    size=int(np.ceil(len(values) / block_length)))
                indices = np.concatenate([
                    (start + np.arange(block_length)) % len(values)
                    for start in starts])[:len(values)]
                selected.append(values[indices])
            bootstrap[index] = np.concatenate(selected).mean()
        output[metric] = {
            "improvement_mm": estimate,
            "ci95_mm": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "frames": int(sum(map(len, arrays))),
        }
    return output


def _load_baseline(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[(row["protocol"], row["sequence"], int(row["frame"]))] = {
                "node": float(row["node"]),
                "endpoint": float(row["endpoint"]),
                "horizon": int(row["horizon"]),
            }
    return {"rows": rows}


def _model_summary(model, config, checkpoint):
    geometry = model.geometry_report()
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_selection": "minimum aggregate dev validation.node_mean_mm",
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "n_bend_modes": model.n_bend_modes,
        "n_length_coordinates": model.n_sections,
        "residual_mode": model.residual_mode,
        "burnin_mode": model.burnin_mode,
        "play_thresholds": model.play.thresholds.detach().cpu().tolist(),
        "play_weights": model.play.weights.detach().cpu().tolist(),
        "maxwell_taus_s": model.maxwell.taus.detach().cpu().tolist(),
        "maxwell_gains": model.maxwell_gains.detach().cpu().tolist(),
        "residual_coordinate_scales": (
            geometry["residual_coordinate_scales"].tolist()),
        "best_epoch": config["phases"][0]["validation_selection"]["best_epoch"],
        "best_dev_node_mm": config["phases"][0]["validation_selection"]["best_value"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-checkpoint", required=True)
    parser.add_argument("--memory-checkpoint", required=True)
    parser.add_argument("--baseline-csv", required=True)
    parser.add_argument("--fit-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)

    runs = {}
    for name, checkpoint in (("explicit", args.explicit_checkpoint),
                             ("memory", args.memory_checkpoint)):
        info = load_model(checkpoint, device=args.device)
        if info["model_type"] != "hereditary_geometry":
            raise TypeError(f"{name} checkpoint 不是 hereditary_geometry")
        model = info["model"].eval()
        config = info.get("saved_config") or {}
        protocol = _evaluate_protocols(
            model, config, Path(args.data_dir), device)
        runs[name] = {
            "model": _model_summary(model, config, checkpoint),
            "aggregate": protocol["aggregate"],
            "interventions": _evaluate_interventions(
                model, config, Path(args.data_dir), device),
            "geometry_oracles": _oracle_geometry(
                model, config, Path(args.fit_dir), Path(args.data_dir)),
            "rows": protocol["rows"],
            "csv_rows": protocol["csv_rows"],
        }

    baseline = _load_baseline(Path(args.baseline_csv))
    comparisons = {
        "explicit_vs_memory": {
            protocol: _paired_improvement(runs["memory"], runs["explicit"], protocol)
            for protocol, _ in PROTOCOLS
        },
        "explicit_vs_hereditary_v2_r05": {
            protocol: _paired_improvement(baseline, runs["explicit"], protocol)
            for protocol, _ in PROTOCOLS
        },
    }
    report = {
        "schema": "hov21_analysis_v1",
        "data_role": "development model selection; reserved test not read",
        "metric": "mean Euclidean planar skeleton-node error in millimetres",
        "protocols": {
            "continuous": "operator state continues across each sequence",
            "cold_restart_40": "operator state rebuilt from action history every 40 frames",
        },
        "bootstrap": {
            "method": "sequence-stratified circular moving blocks",
            "samples": 10000,
            "block_length_frames": 20,
            "seed": 42,
        },
        "runs": {
            name: {key: value for key, value in run.items()
                   if key not in {"rows", "csv_rows"}}
            for name, run in runs.items()
        },
        "comparisons": comparisons,
        "guardrails": [
            "Dev metrics select structure; the reserved test split was not read.",
            "Fixed-checkpoint interventions measure reliance, not retrained necessity.",
            "Operator gains and directions are system-level kinematic memory parameters, not material constants.",
            "The geometry oracle uses true per-frame generalized coordinates and is a representation bound, not a deployable predictor.",
        ],
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    with (output / "per_frame.csv").open(
            "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("run", "protocol", "sequence", "frame",
                                "horizon", "node", "endpoint"))
        writer.writeheader()
        for name, run in runs.items():
            for row in run["csv_rows"]:
                writer.writerow({"run": name, **row})
    (output / "ANALYSIS_COMPLETE").touch(exist_ok=False)
    print(json.dumps({
        "aggregate": {name: run["aggregate"] for name, run in runs.items()},
        "comparisons": comparisons,
        "interventions": {
            name: run["interventions"]["metrics"] for name, run in runs.items()},
        "geometry_oracles": {
            name: run["geometry_oracles"] for name, run in runs.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
