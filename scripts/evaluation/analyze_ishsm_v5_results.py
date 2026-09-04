#!/usr/bin/env python3
"""Compare ISHSM v5 projections and 200-epoch Hereditary baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.analyze_ishsm_v4_results import (  # noqa: E402
    _evaluate_hereditary,
    _horizon_slices,
    _load_protocol_run,
    _paired_improvement,
)


def _load_hereditary_csv(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["protocol"], row["sequence"], int(row["frame"]))
            rows[key] = {
                "node": float(row["node"]),
                "endpoint": float(row["endpoint"]),
                "horizon": int(row["horizon"]),
            }
    return {"rows": rows}


def _write_hereditary_csv(path: Path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("protocol", "sequence", "frame", "horizon",
                                "node", "endpoint"))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tipdls-eval", required=True)
    parser.add_argument("--modal-eval", required=True)
    parser.add_argument("--hereditary-r03-checkpoint", required=True)
    parser.add_argument("--hereditary-r05-checkpoint", required=True)
    parser.add_argument("--previous-hereditary-csv", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    tipdls = _load_protocol_run(Path(args.tipdls_eval))
    modal = _load_protocol_run(Path(args.modal_eval))
    hereditary_r03 = _evaluate_hereditary(
        Path(args.hereditary_r03_checkpoint), Path(args.data_dir), args.device)
    hereditary_r05 = _evaluate_hereditary(
        Path(args.hereditary_r05_checkpoint), Path(args.data_dir), args.device)
    hereditary_100 = _load_hereditary_csv(
        Path(args.previous_hereditary_csv))

    protocols = ("zero_init", "single_anchor", "periodic_1",
                 "periodic_5", "periodic_40")
    comparisons = {
        "tipdls_vs_modal": {
            protocol: _paired_improvement(modal, tipdls, protocol)
            for protocol in protocols
        },
        "hereditary_r05_vs_r03": {
            protocol: _paired_improvement(
                hereditary_r03, hereditary_r05, protocol)
            for protocol in ("continuous", "cold_restart_40")
        },
        "hereditary_r03_200_vs_previous_100": {
            protocol: _paired_improvement(
                hereditary_100, hereditary_r03, protocol)
            for protocol in ("continuous", "cold_restart_40")
        },
        "hereditary_r05_vs_ishsm_modal_action_only":
            _paired_improvement(
                modal, hereditary_r05, "zero_init", "continuous"),
        "hereditary_r05_internal_continuous_vs_cold":
            _paired_improvement(
                hereditary_r05, hereditary_r05,
                "cold_restart_40", "continuous"),
    }

    output = {
        "schema": "ishsm_v5_hereditary_200_analysis_v1",
        "data_role": "dev_model_selection_and_development",
        "bootstrap": {
            "method": "sequence-stratified circular moving-block",
            "samples": 10000,
            "block_length_frames": 20,
            "seed": 42,
        },
        "ishsm_tipdls": {
            "checkpoint": tipdls["summary"]["checkpoint"],
            "model": tipdls["summary"]["model"],
            "aggregate": tipdls["summary"]["aggregate"],
        },
        "ishsm_modal": {
            "checkpoint": modal["summary"]["checkpoint"],
            "model": modal["summary"]["model"],
            "aggregate": modal["summary"]["aggregate"],
        },
        "hereditary_r03": hereditary_r03["summary"],
        "hereditary_r05": hereditary_r05["summary"],
        "comparisons": comparisons,
        "horizon_slices": {
            "ishsm_tipdls_single": _horizon_slices(tipdls),
            "ishsm_modal_single": _horizon_slices(modal),
            "hereditary_r05_continuous": _horizon_slices(
                hereditary_r05, "continuous"),
        },
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    _write_hereditary_csv(
        output_path.with_name("hereditary_r03_protocols.csv"),
        hereditary_r03["csv_rows"])
    _write_hereditary_csv(
        output_path.with_name("hereditary_r05_protocols.csv"),
        hereditary_r05["csv_rows"])
    (output_path.parent / "ANALYSIS_COMPLETE").touch(exist_ok=False)
    print(json.dumps({
        "hereditary_r03": output["hereditary_r03"],
        "hereditary_r05": output["hereditary_r05"],
        "comparisons": comparisons,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
