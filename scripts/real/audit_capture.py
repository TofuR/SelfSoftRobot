"""训练前审计 real_capture 原始序列的完整性、同步和动作合同。

输出 ``capture_audit.json``、``timing_quality.png`` 和
``action_coverage.png``。只读原始数据，不修改或清洗记录。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.registry import (  # noqa: E402
    ProjectPaths, canonical_intermediate, canonical_output,
    resolve_raw_sequence,
)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def numeric(rows, columns):
    return np.asarray([[float(row[column]) for column in columns]
                       for row in rows], dtype=np.float64)


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def strict_time_summary(values):
    values = np.asarray(values, dtype=float)
    dt = np.diff(values)
    return {
        "finite": bool(np.isfinite(values).all()),
        "strictly_increasing": bool((dt > 0).all()),
        "start_s": float(values[0]) if len(values) else None,
        "end_s": float(values[-1]) if len(values) else None,
        "duration_s": float(values[-1] - values[0]) if len(values) else 0.0,
        "dt_s": quantiles(dt),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="real_capture训练前数据质量审计")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--out", default=None,
                        help="显式输出目录；必须位于 workspace")
    parser.add_argument("--workspace-root", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(workspace_root=args.workspace_root)
    seq = str(resolve_raw_sequence(paths, args.seq, camera=args.camera))
    seq_name = os.path.basename(seq)
    out = str(canonical_output(paths, args.out or (
        canonical_intermediate(paths, seq_name, "capture-audit-v1") /
        "qc_capture")))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(seq, "meta.json"), encoding="utf-8") as stream:
        meta = json.load(stream)

    image_paths = sorted(glob.glob(os.path.join(seq, args.camera, "*.png")))
    frame_ids = np.asarray([int(os.path.splitext(os.path.basename(path))[0])
                            for path in image_paths], dtype=np.int64)
    frame_times = np.atleast_1d(np.loadtxt(os.path.join(seq, "frame_times.txt")))
    actions_rows = read_csv(os.path.join(seq, "actions6.csv"))
    commands_rows = read_csv(os.path.join(seq, "commands.csv"))
    samples_rows = read_csv(os.path.join(seq, "samples.csv"))
    ndi_path = os.path.join(seq, "ndi.csv")
    ndi_available = os.path.isfile(ndi_path)
    ndi_rows = read_csv(ndi_path) if ndi_available else []
    actions = numeric(actions_rows, [f"c{i}" for i in range(6)])
    action_t = numeric(actions_rows, ["t_sec"])[:, 0]
    command_applied = numeric(commands_rows, [f"action_command{i}" for i in range(6)])
    command_requested = numeric(commands_rows, [f"requested{i}" for i in range(6)])
    command_t = numeric(commands_rows, ["t_command"])[:, 0]
    sample_frame_idx = numeric(samples_rows, ["frame_idx"])[:, 0].astype(np.int64)
    sample_command_ids = numeric(samples_rows, ["command_id"])[:, 0].astype(np.int64)
    sample_grab_t = numeric(samples_rows, ["t_grab"])[:, 0]
    frame_age = numeric(samples_rows, ["frame_age"])[:, 0]

    counts = {
        "images": len(image_paths), "frame_times": len(frame_times),
        "actions": len(actions_rows), "commands": len(commands_rows),
        "samples": len(samples_rows), "ndi": len(ndi_rows),
    }
    expected = counts["images"]
    # commands.csv 是命令事件日志，不是逐帧表。停止采集附近可能有一个已 ACK、但尚未
    # 来得及配对图像的尾命令；真正的一致性应通过 samples.command_id 外键检查。
    count_match = {name: value == expected for name, value in counts.items()
                   if name != "commands" and (name != "ndi" or ndi_available)}
    frame_contiguous = bool(len(frame_ids) and np.array_equal(
        frame_ids, np.arange(frame_ids[0], frame_ids[0] + len(frame_ids))))
    sample_contiguous = bool(np.array_equal(
        sample_frame_idx, np.arange(len(sample_frame_idx))))

    sources = tuple(int(value) for value in meta.get("channel_source6", range(6)))
    equalities = [(source, channel) for channel, source in enumerate(sources)
                  if source != channel]
    residuals = {
        f"ch{follower}=ch{leader}": float(np.max(np.abs(
            actions[:, leader] - actions[:, follower]), initial=0.0))
        for leader, follower in equalities
    }
    tolerance = float(meta.get("channel_equality_tolerance_kpa", 0.5))
    command_by_id = {int(row["command_id"]): row for row in commands_rows}
    duplicate_command_ids = len(command_by_id) != len(commands_rows)
    missing_command_ids = sorted(set(sample_command_ids) - set(command_by_id))
    sampled_commands = [command_by_id[value] for value in sample_command_ids
                        if value in command_by_id]
    unpaired_command_ids = sorted(set(command_by_id) - set(sample_command_ids))
    sampled_applied = (numeric(sampled_commands,
                               [f"action_command{i}" for i in range(6)])
                       if sampled_commands else np.empty((0, 6)))
    statuses = {}
    for row in commands_rows:
        status = row.get("communication_status", "")
        statuses[status] = statuses.get(status, 0) + 1

    ndi_columns = list(ndi_rows[0]) if ndi_rows else []
    ndi_indices = sorted({int(name.split("_")[0][3:]) for name in ndi_columns
                          if name.startswith("ndi") and name.endswith("_x")})
    ndi_summary = {}
    for index in ndi_indices:
        xyz = numeric(ndi_rows, [f"ndi{index}_{axis}" for axis in "xyz"])
        quality = numeric(ndi_rows, [f"ndi{index}_quality"])[:, 0]
        finite_xyz = np.isfinite(xyz).all(axis=1)
        ndi_summary[f"ndi{index}"] = {
            "finite_xyz_ratio": float(finite_xyz.mean()),
            "quality": quantiles(quality),
            "xyz_mm": {axis: quantiles(xyz[:, column])
                       for column, axis in enumerate("xyz")},
        }

    sync = {
        "actions_vs_frame_times_max_abs_s": float(np.max(
            np.abs(action_t - frame_times), initial=0.0))
            if len(action_t) == len(frame_times) else None,
        "samples_vs_frame_times_max_abs_s": float(np.max(
            np.abs(sample_grab_t - frame_times), initial=0.0))
            if len(sample_grab_t) == len(frame_times) else None,
        "sampled_commands_vs_actions_max_abs_kpa": float(np.max(
            np.abs(sampled_applied - actions), initial=0.0))
            if len(sampled_applied) == len(actions) else None,
        "requested_vs_applied_abs_kpa": quantiles(
            np.abs(command_requested - command_applied).ravel()),
    }
    max_frame_age = float(meta.get("max_frame_age", np.inf))
    ndi_age_columns = [name for name in (ndi_rows[0] if ndi_rows else {})
                       if name.endswith("_age")]
    # ndi ages live in samples.csv, not ndi.csv.
    ndi_age_columns = [name for name in (samples_rows[0] if samples_rows else {})
                       if name.startswith("ndi") and name.endswith("_age")]
    age_summary = {
        "frame_age_s": quantiles(frame_age),
        "frame_age_over_limit": int(np.sum(frame_age > max_frame_age)),
        "ndi_age_s": {name: quantiles(numeric(samples_rows, [name])[:, 0])
                      for name in ndi_age_columns},
    }

    issues = []
    if not all(count_match.values()):
        issues.append({"severity": "critical", "code": "count_mismatch",
                       "detail": counts})
    if duplicate_command_ids or missing_command_ids:
        issues.append({"severity": "critical", "code": "command_link_failure",
                       "duplicate_command_ids": duplicate_command_ids,
                       "missing_sample_command_ids": missing_command_ids[:20]})
    if not frame_contiguous or not sample_contiguous:
        issues.append({"severity": "critical", "code": "non_contiguous_frames"})
    for name, values in (("frame_times", frame_times), ("actions", action_t),
                         ("commands", command_t), ("samples", sample_grab_t)):
        if not strict_time_summary(values)["strictly_increasing"]:
            issues.append({"severity": "critical", "code": f"{name}_time_order"})
    bad_residuals = {key: value for key, value in residuals.items()
                     if value > tolerance}
    if bad_residuals:
        issues.append({"severity": "critical", "code": "channel_source_violation",
                       "detail": bad_residuals})
    sampled_statuses = {}
    for row in sampled_commands:
        status = row.get("communication_status", "")
        sampled_statuses[status] = sampled_statuses.get(status, 0) + 1
    non_ack = expected - sampled_statuses.get("ack", 0)
    if non_ack:
        issues.append({"severity": "high", "code": "non_ack_commands",
                       "count": int(non_ack), "statuses": sampled_statuses})
    if age_summary["frame_age_over_limit"]:
        issues.append({"severity": "high", "code": "stale_frames",
                       "count": age_summary["frame_age_over_limit"]})

    result = {
        "schema_version": 1,
        "sequence": seq,
        "intended_grain": "one synchronized command/frame/NDI sample per frame_idx",
        "counts": counts,
        "count_match_to_images": count_match,
        "frame_ids_contiguous": frame_contiguous,
        "sample_frame_idx_contiguous": sample_contiguous,
        "time": {
            "frame_times": strict_time_summary(frame_times),
            "actions": strict_time_summary(action_t),
            "commands": strict_time_summary(command_t),
            "samples": strict_time_summary(sample_grab_t),
        },
        "sync": sync,
        "command_link": {
            "sampled_command_count": len(sampled_commands),
            "sampled_status": sampled_statuses,
            "missing_sample_command_ids": missing_command_ids,
            "unpaired_command_ids": unpaired_command_ids,
            "note": ("Unpaired acknowledged commands are allowed at capture boundaries; "
                     "they are not part of the frame-aligned training table."),
        },
        "action": {
            "channel_source6": list(sources),
            "model_action_channels": [i for i, source in enumerate(sources)
                                      if i == source],
            "equality_tolerance_kpa": tolerance,
            "equality_residual_max_kpa": residuals,
            "per_channel_kpa": {f"ch{i}": quantiles(actions[:, i])
                                for i in range(6)},
        },
        "communication_status": statuses,
        "age": age_summary,
        "ndi": {"available": ndi_available, "streams": ndi_summary},
        "issues": issues,
        "ready_for_image_preprocessing": not any(
            issue["severity"] == "critical" for issue in issues),
        "next_stage": "crop_and_segmentation",
    }
    with open(os.path.join(out, "capture_audit.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time_minutes = frame_times / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(time_minutes, frame_age * 1000, lw=.6)
    axes[0].axhline(max_frame_age * 1000, color="#B54708", ls="--", lw=1)
    axes[0].set_ylabel("frame age [ms]"); axes[0].grid(alpha=.2)
    for name in ndi_age_columns:
        axes[1].plot(time_minutes, numeric(samples_rows, [name])[:, 0] * 1000,
                     lw=.55, label=name)
    axes[1].set_ylabel("NDI age [ms]"); axes[1].legend(); axes[1].grid(alpha=.2)
    for index in ndi_indices:
        quality = numeric(ndi_rows, [f"ndi{index}_quality"])[:, 0]
        axes[2].plot(time_minutes, quality, lw=.55, label=f"ndi{index}")
    axes[2].set_ylabel("NDI quality"); axes[2].set_xlabel("time [min]")
    axes[2].legend(); axes[2].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "timing_quality.png"), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for index in range(6):
        axes[0].plot(time_minutes, actions[:, index], lw=.5, label=f"ch{index}")
    axes[0].set_ylabel("applied command [kPa]"); axes[0].set_xlabel("time [min]")
    axes[0].legend(ncol=6); axes[0].grid(alpha=.2)
    roots = [i for i, source in enumerate(sources) if i == source]
    axes[1].hist([actions[:, index] for index in roots], bins=30,
                 label=[f"ch{index}" for index in roots], histtype="step", lw=1.4)
    axes[1].set_xlabel("applied command [kPa]"); axes[1].set_ylabel("samples")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "action_coverage.png"), dpi=130)
    plt.close(fig)

    print(f"审计完成：frames={expected} duration={result['time']['frame_times']['duration_s']:.1f}s")
    print(f"  counts={counts} status={statuses} equalities={residuals}")
    print(f"  critical={sum(i['severity'] == 'critical' for i in issues)} "
          f"high={sum(i['severity'] == 'high' for i in issues)}")
    print(f"  output={out}")


if __name__ == "__main__":
    main()
