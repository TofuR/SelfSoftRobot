#!/usr/bin/env python3
"""Evaluate ISHSM under sparse-observation protocols and archive JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.action_view import project_actions  # noqa: E402
from src.evaluation.ishsm_evaluation import rollout_ishsm_sequence  # noqa: E402
from src.models.model_ishsm import skeleton_to_generalized  # noqa: E402
from src.utils.model_loader import load_model, _load_config_json  # noqa: E402


def ishsm_evaluation_protocols():
    """Return explicit observation protocols used in every ISHSM report."""
    return [
        ("h0", "h0", None),
        ("zero_init", "zero_init", None),
        ("single_anchor", "single_anchor", None),
        ("periodic_1", "periodic", 1),
        ("periodic_5", "periodic", 5),
        ("periodic_10", "periodic", 10),
        ("periodic_20", "periodic", 20),
        ("periodic_40", "periodic", 40),
        ("periodic_80", "periodic", 80),
    ]


def _metrics(model, predictions, positions, valid, horizon):
    gt = positions.transpose(0, 2, 1)
    error = np.linalg.norm(predictions[..., :2] - gt[..., :2], axis=-1)
    selected = error[valid]
    n_nodes = error.shape[1]
    thirds = np.array_split(np.arange(n_nodes), 3)
    pred_t = torch.from_numpy(predictions[valid]).float()
    gt_t = torch.from_numpy(gt[valid]).float()
    pred_b, pred_l, _ = skeleton_to_generalized(
        pred_t, model.reference_segment_lengths.cpu(), model.section_intervals)
    gt_b, gt_l, _ = skeleton_to_generalized(
        gt_t, model.reference_segment_lengths.cpu(), model.section_intervals)
    angle_delta = torch.atan2(torch.sin(pred_b - gt_b), torch.cos(pred_b - gt_b))
    segment_pred = np.linalg.norm(
        predictions[valid, 1:] - predictions[valid, :-1], axis=-1)
    segment_gt = np.linalg.norm(gt[valid, 1:] - gt[valid, :-1], axis=-1)
    section_abs = []
    start = 0
    for count in model.section_intervals:
        section_abs.append(np.abs(
            segment_pred[:, start:start + count].sum(1) -
            segment_gt[:, start:start + count].sum(1)))
        start += count
    by_horizon = {}
    for value in np.unique(horizon[valid]):
        mask = valid & (horizon == value)
        by_horizon[str(int(value))] = {
            "frames": int(mask.sum()),
            "node_mean_mm": float(error[mask].mean()),
            "endpoint_mean_mm": float(error[mask, -1].mean()),
        }
    return {
        "frames": int(valid.sum()),
        "node_mean_mm": float(selected.mean()),
        "node_p90_mm": float(np.quantile(selected, 0.90)),
        "max_node_mm": float(selected.max()),
        "endpoint_mean_mm": float(selected[:, -1].mean()),
        "endpoint_p90_mm": float(np.quantile(selected[:, -1], 0.90)),
        "base_third_mean_mm": float(selected[:, thirds[0]].mean()),
        "middle_third_mean_mm": float(selected[:, thirds[1]].mean()),
        "tip_third_mean_mm": float(selected[:, thirds[2]].mean()),
        "bend_mae_rad": float(angle_delta.abs().mean()),
        "section_log_length_mae": float((pred_l - gt_l).abs().mean()),
        "section_length_mae_mm": [float(values.mean()) for values in section_abs],
        "by_horizon": by_horizon,
    }, error


def evaluate(checkpoint, data_dir, output, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有分析目录: {output}")
    output.mkdir(parents=True)
    info = load_model(checkpoint, device=device)
    if info["model_type"] != "ishsm":
        raise ValueError("eval_ishsm.py 只接受 ISHSM checkpoint")
    model = info["model"]
    config = _load_config_json(checkpoint) or {}
    selection_protocol = (config.get("evaluation") or {}).get(
        "ishsm_validation_protocol", "legacy_unspecified")
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    norm_factor = float(model.action_norm_factor.item())
    window_size = int(config.get("window_size", model.window_size))
    protocols = ishsm_evaluation_protocols()
    summaries = {name: [] for name, _, _ in protocols}
    csv_rows = []
    trajectory_payload = {}
    files = sorted(Path(data_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"没有 NPZ: {data_dir}")
    for seq_index, path in enumerate(files):
        with np.load(path, allow_pickle=False) as raw:
            actions = project_actions(raw["actions"], channels).astype(np.float32)
            positions = raw["positions"].astype(np.float32)
            evaluation_mask = (raw["evaluation_mask"].astype(bool)
                               if "evaluation_mask" in raw else
                               np.ones(len(actions), dtype=bool))
        if evaluation_mask.shape != (len(actions),):
            raise ValueError(
                f"{path}: evaluation_mask 必须为 ({len(actions)},)，"
                f"得到 {evaluation_mask.shape}")
        if not evaluation_mask.any():
            raise ValueError(f"{path}: evaluation_mask 没有任何计分帧")
        anchor_index = max(int(np.flatnonzero(evaluation_mask)[0]) - 1, 0)
        for name, protocol, interval in protocols:
            result = rollout_ishsm_sequence(
                model, actions, positions, protocol=protocol,
                reanchor_interval=interval, window_size=window_size,
                norm_factor=norm_factor, device=torch.device(device),
                anchor_index=anchor_index)
            valid = evaluation_mask & (result["horizon"] >= 0)
            metrics, errors = _metrics(
                model, result["predictions"], positions, valid,
                result["horizon"])
            metrics["sequence"] = path.name
            summaries[name].append(metrics)
            prefix = f"seq{seq_index}_{name}"
            trajectory_payload[f"{prefix}_states"] = result["states"]
            trajectory_payload[f"{prefix}_horizon"] = result["horizon"]
            for frame in np.flatnonzero(valid):
                state = result["states"][frame]
                csv_rows.append({
                    "protocol": name, "sequence": path.name,
                    "frame": int(frame), "horizon": int(result["horizon"][frame]),
                    "node_mean_mm": float(errors[frame].mean()),
                    "endpoint_mm": float(errors[frame, -1]),
                    "state_norm": float(np.linalg.norm(state)),
                })

    aggregate = {}
    for name, sequence_metrics in summaries.items():
        weights = np.asarray([m["frames"] for m in sequence_metrics], dtype=float)
        total = weights.sum()
        aggregate[name] = {
            key: float(sum(m[key] * w for m, w in zip(sequence_metrics, weights)) / total)
            for key in ("node_mean_mm", "endpoint_mean_mm", "bend_mae_rad",
                        "section_log_length_mae")
        }
        aggregate[name]["frames"] = int(total)

    report = model.state_report()
    summary = {
        "schema": "ishsm_evaluation_v2",
        "checkpoint": str(Path(checkpoint).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "checkpoint_selection": (
            "minimum dev validation.node_mean_mm under " +
            selection_protocol),
        "protocol_definitions": {
            "single_anchor": (
                "observe one skeleton immediately before the first scored frame, "
                "then use actions only"),
            "periodic_K": (
                "observe one skeleton every K frames; no dense skeleton history"),
        },
        "protocol_aliases": {
            "window40": "periodic_40 (legacy report name only)",
        },
        "model": {
            "n_bend_modes": model.n_bend_modes,
            "n_length_states": model.n_length_states,
            "taus_s": report["taus"].tolist(),
            "decays_per_step": report["decays"].tolist(),
            "excitation": report["excitation"].tolist(),
            "observation_update": report["observation_update"],
            "observation_gains": report["observation_gains"].tolist(),
            "training_reanchor_intervals": list(
                report["training_reanchor_intervals"]),
            "tau_parameterization": report["tau_parameterization"],
            "observation_projection": report["observation_projection"],
            "tip_dls_lambda_mm2": report["tip_dls_lambda_mm2"],
        },
        "aggregate": aggregate,
        "per_sequence": summaries,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    with (output / "per_frame.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    np.savez_compressed(output / "state_trajectories.npz", **trajectory_payload)
    (output / "EVALUATION_COMPLETE").touch(exist_ok=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = evaluate(args.checkpoint, args.data_dir, args.out, args.device)
    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
