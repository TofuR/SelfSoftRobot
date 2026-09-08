#!/usr/bin/env python3
"""Compare completed HOV2.1/HOV2.2 checkpoints on the same dev protocols."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.analyze_hov21_results import (
    PROTOCOLS,
    _evaluate_interventions,
    _evaluate_protocols,
    _model_summary,
    _paired_improvement,
)
from src.utils.model_loader import load_model


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("参数必须为 NAME=PATH")
    return name, Path(path)


def _comparison(value: str) -> tuple[str, str]:
    candidate, separator, baseline = value.partition(":")
    if not separator or not candidate or not baseline:
        raise argparse.ArgumentTypeError("比较必须为 CANDIDATE:BASELINE")
    return candidate, baseline


def _load_csv(path: Path) -> dict:
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows[(row["protocol"], row["sequence"], int(row["frame"]))] = {
                "node": float(row["node"]),
                "endpoint": float(row["endpoint"]),
                "horizon": int(row["horizon"]),
            }
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=_named_path, required=True)
    parser.add_argument("--baseline", action="append", type=_named_path,
                        default=[])
    parser.add_argument("--compare", action="append", type=_comparison,
                        required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    evaluated = {}
    csv_rows = []
    for name, checkpoint in args.run:
        info = load_model(str(checkpoint), device=args.device)
        if info["model_type"] != "hereditary_geometry":
            raise TypeError(f"{name}: checkpoint 不是 hereditary_geometry")
        model = info["model"].eval()
        config = info.get("saved_config") or {}
        protocols = _evaluate_protocols(
            model, config, Path(args.data_dir), device)
        summary = _model_summary(model, config, checkpoint)
        ones = torch.ones(1, model.action_dim, device=device)
        zeros = torch.zeros_like(ones)
        with torch.no_grad():
            summary.update({
                "model_contract_version": model.model_contract_version,
                "bend_basis_kind": getattr(model, "bend_basis_kind", "pod"),
                "coordinate_loss_basis": model.coordinate_loss_basis,
                "drive_normalization": model.drive_normalization,
                "drive_at_zero": model.drive(zeros)[0].cpu().tolist(),
                "drive_at_one": model.drive(ones)[0].cpu().tolist(),
            })
        evaluated[name] = {
            "model": summary,
            "aggregate": protocols["aggregate"],
            "interventions": _evaluate_interventions(
                model, config, Path(args.data_dir), device),
            "rows": protocols["rows"],
        }
        csv_rows.extend({"run": name, **row}
                        for row in protocols["csv_rows"])

    sources = {name: run for name, run in evaluated.items()}
    sources.update({name: _load_csv(path) for name, path in args.baseline})
    comparisons = {}
    for candidate, baseline in args.compare:
        if candidate not in sources or baseline not in sources:
            raise KeyError(f"未知比较源: {candidate}:{baseline}")
        comparisons[f"{candidate}_vs_{baseline}"] = {
            protocol: _paired_improvement(
                sources[baseline], sources[candidate], protocol)
            for protocol, _ in PROTOCOLS
        }

    report = {
        "schema": "hov22_analysis_v1",
        "data_role": "development model selection; reserved test not read",
        "metric": "mean Euclidean planar skeleton-node error in millimetres",
        "runs": {
            name: {key: value for key, value in run.items() if key != "rows"}
            for name, run in evaluated.items()
        },
        "comparisons": comparisons,
        "bootstrap": {
            "method": "sequence-stratified circular moving blocks",
            "samples": 10000,
            "block_length_frames": 20,
            "seed": 42,
        },
        "guardrails": [
            "All structure decisions use dev; reserved test was not read.",
            "Positive paired improvement means candidate has lower error.",
            "Fixed-checkpoint interventions measure reliance, not retrained necessity.",
            "System-level kinematic memory parameters are not material constants.",
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
        writer.writerows(csv_rows)
    (output / "ANALYSIS_COMPLETE").touch(exist_ok=False)
    print(json.dumps({
        "aggregate": {name: run["aggregate"]
                      for name, run in evaluated.items()},
        "comparisons": comparisons,
        "drive": {name: {
            "normalization": run["model"]["drive_normalization"],
            "at_one": run["model"]["drive_at_one"],
        } for name, run in evaluated.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
