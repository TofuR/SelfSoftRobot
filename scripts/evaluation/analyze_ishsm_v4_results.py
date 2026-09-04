#!/usr/bin/env python3
"""Aggregate ISHSM v4 protocol runs and the fair Hereditary baseline."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.action_view import project_actions  # noqa: E402
from src.evaluation.real_transition_validation import per_frame_rollout  # noqa: E402
from src.utils.model_loader import load_model  # noqa: E402


RUN_NAMES = {
    "v4_main_s42": "run_20260903_000_main_s42_dev",
    "v4_main_s43": "run_20260903_001_main_s43_dev",
    "v4_hard_s42": "run_20260903_002_hard_s42_dev",
    "v4_indtau_s42": "run_20260903_003_indtau_s42_dev",
    "v3_geom_s42": "run_20260903_004_v3_geom_s42_reval_dev",
    "v3_geom_s43": "run_20260903_005_v3_geom_s43_reval_dev",
}


def _load_protocol_run(path: Path):
    if not (path / "EVALUATION_COMPLETE").is_file():
        raise RuntimeError(f"评估未完成: {path}")
    with (path / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    rows = {}
    with (path / "per_frame.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["protocol"], row["sequence"], int(row["frame"]))
            rows[key] = {
                "node": float(row["node_mean_mm"]),
                "endpoint": float(row["endpoint_mm"]),
                "horizon": int(row["horizon"]),
            }
    return {"summary": summary, "rows": rows}


def _paired_improvement(
        baseline, candidate, protocol_baseline, protocol_candidate=None,
        *, bootstrap_samples=10000, block_length=20, seed=42):
    """Return baseline-candidate improvement with stratified circular blocks."""
    protocol_candidate = protocol_candidate or protocol_baseline
    pairs = defaultdict(lambda: {"node": [], "endpoint": []})
    for (protocol, sequence, frame), base_values in baseline["rows"].items():
        if protocol != protocol_baseline:
            continue
        candidate_key = (protocol_candidate, sequence, frame)
        if candidate_key not in candidate["rows"]:
            continue
        candidate_values = candidate["rows"][candidate_key]
        for metric in ("node", "endpoint"):
            pairs[sequence][metric].append(
                (frame, base_values[metric] - candidate_values[metric]))
    if not pairs:
        raise ValueError(
            f"没有配对帧: {protocol_baseline} -> {protocol_candidate}")

    rng = np.random.default_rng(seed)
    output = {}
    for metric in ("node", "endpoint"):
        arrays = []
        for sequence in sorted(pairs):
            ordered = sorted(pairs[sequence][metric])
            arrays.append(np.asarray([value for _, value in ordered]))
        estimate = float(np.concatenate(arrays).mean())
        samples = np.empty(bootstrap_samples, dtype=np.float64)
        for sample_index in range(bootstrap_samples):
            sampled = []
            for values in arrays:
                n_values = len(values)
                starts = rng.integers(
                    0, n_values,
                    size=int(np.ceil(n_values / block_length)))
                indices = np.concatenate([
                    (start + np.arange(block_length)) % n_values
                    for start in starts
                ])[:n_values]
                sampled.append(values[indices])
            samples[sample_index] = np.concatenate(sampled).mean()
        output[metric] = {
            "improvement_mm": estimate,
            "ci95_mm": [float(value) for value in np.quantile(
                samples, [0.025, 0.975])],
            "frames": int(sum(len(values) for values in arrays)),
        }
    return output


def _horizon_slices(run, protocol="single_anchor"):
    bounds = ((0, 19), (20, 39), (40, 79), (80, None))
    output = {}
    for lower, upper in bounds:
        selected = [
            values for (name, _, _), values in run["rows"].items()
            if name == protocol and values["horizon"] >= lower and
            (upper is None or values["horizon"] <= upper)
        ]
        label = f"{lower}-{upper}" if upper is not None else f"{lower}+"
        output[label] = {
            "frames": len(selected),
            "node_mean_mm": float(np.mean([v["node"] for v in selected])),
            "endpoint_mean_mm": float(np.mean(
                [v["endpoint"] for v in selected])),
        }
    return output


def _evaluate_hereditary(checkpoint, data_dir, device):
    info = load_model(str(checkpoint), device=device)
    if info["model_type"] != "hereditary":
        raise ValueError("Hereditary 对照 checkpoint 类型不匹配")
    model = info["model"]
    config = info.get("saved_config") or {}
    channels = tuple((config.get("action_view") or {}).get(
        "model_action_channels", range(model.action_dim)))
    window_size = int(config.get("window_size", model.window_size))
    norm_factor = float(model.action_norm_factor.item())
    rows = []
    for path_string in sorted(glob.glob(os.path.join(str(data_dir), "*.npz"))):
        with np.load(path_string, allow_pickle=False) as raw:
            actions = project_actions(
                raw["actions"], channels).astype(np.float32)
            positions = raw["positions"].astype(np.float32)
            evaluation_mask = (raw["evaluation_mask"].astype(bool)
                               if "evaluation_mask" in raw else
                               np.ones(len(actions), dtype=bool))
        for protocol, mode in (("continuous", "gt"),
                               ("cold_restart_40", "open_loop")):
            predictions, horizon = per_frame_rollout(
                model, mode, actions, positions, window_size, norm_factor,
                torch.device(device), K=40, max_steps=800)
            gt = positions[:len(predictions)].transpose(0, 2, 1)
            errors = np.linalg.norm(
                predictions[..., :2] - gt[..., :2], axis=-1)
            valid = evaluation_mask[:len(predictions)].copy()
            if mode == "open_loop":
                valid &= horizon >= 0
            first_scored = int(np.flatnonzero(evaluation_mask)[0])
            for frame in np.flatnonzero(valid):
                rows.append({
                    "protocol": protocol,
                    "sequence": Path(path_string).name,
                    "frame": int(frame),
                    "horizon": (int(horizon[frame]) if mode == "open_loop"
                                else int(frame - first_scored)),
                    "node": float(errors[frame].mean()),
                    "endpoint": float(errors[frame, -1]),
                })
    aggregate = {}
    keyed_rows = {}
    for protocol in ("continuous", "cold_restart_40"):
        selected = [row for row in rows if row["protocol"] == protocol]
        aggregate[protocol] = {
            "frames": len(selected),
            "node_mean_mm": float(np.mean([row["node"] for row in selected])),
            "endpoint_mean_mm": float(np.mean(
                [row["endpoint"] for row in selected])),
        }
        for row in selected:
            keyed_rows[(protocol, row["sequence"], row["frame"])] = {
                "node": row["node"], "endpoint": row["endpoint"],
                "horizon": row["horizon"],
            }
    report = model.hysteresis_report()
    raw_residual = float(model.residual_scale.detach().cpu())
    model_report = {
        "parameters": int(sum(parameter.numel()
                              for parameter in model.parameters())),
        "burnin_mode": model.burnin_mode,
        "play_thresholds": report["play_thresholds"].tolist(),
        "maxwell_taus_s": report["maxwell_taus"].tolist(),
        "residual_scale_raw": raw_residual,
        "residual_scale_effective": min(
            raw_residual, float(model.residual_scale_max)),
        "residual_scale_max": float(model.residual_scale_max),
    }
    return {
        "summary": {"aggregate": aggregate, "model": model_report},
        "rows": keyed_rows,
        "csv_rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--hereditary-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.analysis_root)
    runs = {
        name: _load_protocol_run(root / directory)
        for name, directory in RUN_NAMES.items()
    }
    hereditary = _evaluate_hereditary(
        Path(args.hereditary_checkpoint), Path(args.data_dir), args.device)

    comparisons = {
        "v4_main_vs_v3_s42": {
            protocol: _paired_improvement(
                runs["v3_geom_s42"], runs["v4_main_s42"], protocol)
            for protocol in ("zero_init", "single_anchor", "periodic_1",
                             "periodic_5", "periodic_40")
        },
        "v4_main_vs_hard_s42": {
            protocol: _paired_improvement(
                runs["v4_hard_s42"], runs["v4_main_s42"], protocol)
            for protocol in ("zero_init", "single_anchor", "periodic_1",
                             "periodic_5", "periodic_40")
        },
        "v4_shared_vs_independent_s42": {
            protocol: _paired_improvement(
                runs["v4_indtau_s42"], runs["v4_main_s42"], protocol)
            for protocol in ("zero_init", "single_anchor", "periodic_1",
                             "periodic_5", "periodic_40")
        },
        "v4_main_internal": {
            "single_vs_h0": _paired_improvement(
                runs["v4_main_s42"], runs["v4_main_s42"],
                "h0", "single_anchor"),
            "single_vs_zero": _paired_improvement(
                runs["v4_main_s42"], runs["v4_main_s42"],
                "zero_init", "single_anchor"),
            "periodic1_vs_single": _paired_improvement(
                runs["v4_main_s42"], runs["v4_main_s42"],
                "single_anchor", "periodic_1"),
            "periodic5_vs_single": _paired_improvement(
                runs["v4_main_s42"], runs["v4_main_s42"],
                "single_anchor", "periodic_5"),
        },
        "v4_main_vs_hereditary_continuous": _paired_improvement(
            hereditary, runs["v4_main_s42"], "continuous", "zero_init"),
    }

    compact_runs = {}
    for name, run in runs.items():
        compact_runs[name] = {
            "checkpoint": run["summary"]["checkpoint"],
            "model": run["summary"]["model"],
            "aggregate": run["summary"]["aggregate"],
        }
    output = {
        "schema": "ishsm_v4_analysis_v1",
        "data_role": "dev_model_selection_and_development",
        "bootstrap": {
            "method": "sequence-stratified circular moving-block",
            "samples": 10000,
            "block_length_frames": 20,
            "seed": 42,
        },
        "runs": compact_runs,
        "hereditary": hereditary["summary"],
        "comparisons": comparisons,
        "v4_main_s42_horizon_slices": _horizon_slices(
            runs["v4_main_s42"]),
        "seed_difference_absolute": {
            protocol: {
                metric: abs(
                    runs["v4_main_s42"]["summary"]["aggregate"][protocol][metric] -
                    runs["v4_main_s43"]["summary"]["aggregate"][protocol][metric])
                for metric in ("node_mean_mm", "endpoint_mean_mm")
            }
            for protocol in ("zero_init", "single_anchor", "periodic_1",
                             "periodic_5", "periodic_40")
        },
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    hereditary_csv = output_path.with_name("hereditary_protocols.csv")
    with hereditary_csv.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("protocol", "sequence", "frame", "horizon",
                                "node", "endpoint"))
        writer.writeheader()
        writer.writerows(hereditary["csv_rows"])
    print(json.dumps({
        "hereditary": output["hereditary"],
        "comparisons": output["comparisons"],
        "horizon_slices": output["v4_main_s42_horizon_slices"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
