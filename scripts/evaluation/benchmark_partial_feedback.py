#!/usr/bin/env python3
"""Sequential full-feedback benchmark: same checkpoint, task, images and square.

Times span in-memory image -> predicted state -> edge evidence -> state update
-> validated absolute suffix. File reads, reference scoring and plots are out
of this window. Exposure age and hardware communication are not measured.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.control.hereditary_feedback import ActionBounds, physical_shape, correct_state, correct_suffix, batched_rollout
from src.control.hereditary_fast import FrozenHereditary, CachedSuffixA, fast_suffix_b
from src.control.partial_image import extract_edges, extract_edges_vectorized, project_camera
from src.evaluation.partial_replay_data import load_replay_data
from src.utils.model_loader import load_model
from scripts.evaluation.replay_partial_observation import write_json, write_csv


def timing_summary(values):
    values = np.asarray(values, dtype=float)
    return {**dict(zip(("p50", "p95", "p99", "max"), np.percentile(values, [50, 95, 99, 100]).tolist())),
            "over_100ms": int(np.sum(values >= 100)), "over_200ms": int(np.sum(values >= 200)),
            "samples": len(values)}


def prepare(args):
    source = Path(args.source_run)
    config = json.loads((source/"config.json").read_text())
    audit = json.loads((source/"data_audit.json").read_text())
    info = load_model(config["checkpoint"], device="cpu")
    model = info["model"].eval()
    model.requires_grad_(False)
    data = load_replay_data(config["data"], config["raw"], info["saved_config"],
                            model_dt=float(model.dt), norm_factor=float(model.action_norm_factor))
    with np.load(source/"traces.npz") as values:
        trace = {k: values[k] for k in values.files}
    start, stop = audit["task_local_slice"]
    if not np.array_equal(trace["raw_indices"], data.raw_indices[start:stop]):
        raise ValueError("source replay and raw sequence differ")
    images = []
    for path in data.images[start:stop]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"cannot decode {path}")
        images.append(image)
    scale = trace["action_unit_to_kpa"]
    channels = info["saved_config"]["action_view"]["model_action_channels"]
    bounds = ActionBounds(np.asarray(data.meta["lo6"])[channels]/scale,
                          np.asarray(data.meta["hi6"])[channels]/scale,
                          np.asarray(data.meta["rise_rates6"])[channels]*float(model.dt)/scale,
                          np.asarray(data.meta["fall_rates6"])[channels]*float(model.dt)/scale)
    return model, FrozenHereditary(model), data, trace, images, bounds, audit, config


def controller_call(method, model, engine, cache, k, state, old, previous, reference, bounds, *, blocks=8):
    if method == "torch_b":
        return correct_suffix(model, state, old, previous, reference, bounds, blocks=blocks)
    if method == "batched_b":
        return correct_suffix(model, state, old, previous, reference, bounds, blocks=blocks, rollout_fn=batched_rollout)
    if method == "fast_b":
        result, info = fast_suffix_b(engine, state.numpy(), old.numpy(), previous.numpy(), reference.numpy(), bounds, blocks=blocks)
    elif method == "cached_a":
        result, info = cache.correct(k, state.numpy(), old.numpy(), previous.numpy(), bounds)
    else:
        raise ValueError(method)
    return torch.as_tensor(result, dtype=old.dtype), info


def replay(method, model, engine, data, trace, images, bounds, audit, source_config, args, destination):
    destination.mkdir()
    state = torch.tensor(trace["initial_state"])
    plan = torch.tensor(trace["initial_plan"])
    reference = torch.tensor(trace["reference"])
    start = audit["task_local_slice"][0]
    last = torch.tensor(data.actions[start-1])
    matrix = torch.tensor(data.model_to_camera)
    lower = torch.tensor(bounds.lower, dtype=torch.float32)
    upper = torch.tensor(bounds.upper, dtype=torch.float32)
    detector = extract_edges if args.detector == "scalar" else extract_edges_vectorized
    x, y, w, h = audit["occluder_xywh"]
    cache = None
    if method == "cached_a":
        cache = CachedSuffixA(engine, trace["initial_state"], trace["initial_plan"], trace["reference"],
                               blocks=source_config["blocks"],
                               max_model_discrepancy_mm=args.cache_limit_mm)
    # Warmup on a discarded state/plan copy, using an existing archived snapshot.
    # No measurement is injected into the real replay state during warmup.
    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        with torch.no_grad():
            warm = model.step_state(plan[:1], state[None])["latent_z"][0]
            warm_prediction = project_camera(physical_shape(model, warm, plan[0]), matrix).numpy()
        warm_image = images[0].copy()
        warm_image[y:y+h, x:x+w] = audit["occluder_bgr"]
        warm_edges = detector(warm_image, warm_prediction, radius=data.radius_px, search=source_config["search_px"])
        def warm_residual(z):
            return warm_edges.residual(project_camera(physical_shape(model, z, plan[0]), matrix), data.radius_px)
        warm, _ = correct_state(model, warm, plan[0], warm_residual, lower, upper,
                                 prior_std=source_config["prior_std"], max_delta=source_config["state_max_delta"])
        controller_call(method, model, engine, cache, 1, warm, plan[1:], plan[0], reference[1:], bounds,
                        blocks=source_config["blocks"])
    warmup_ms = (time.perf_counter()-warmup_started)*1000
    rows, steps, issued, shapes, states, suffixes = [], [], [], [], [], []
    for k, raw in enumerate(images):
        started = time.perf_counter()
        image = raw.copy()
        image[y:y+h, x:x+w] = np.asarray(audit["occluder_bgr"], np.uint8)
        action, old = plan[0].clone(), plan[1:].clone()
        with torch.no_grad():
            state = model.step_state(action[None], state[None])["latent_z"][0]
            before = physical_shape(model, state, action)
            predicted = project_camera(before, matrix).numpy()
        predicted_at = time.perf_counter()
        evidence = detector(image, predicted, radius=data.radius_px, search=source_config["search_px"])
        detected_at = time.perf_counter()

        def residual(z):
            return evidence.residual(project_camera(physical_shape(model, z, action), matrix), data.radius_px)

        if data.pairing[start+k]["pair_valid"]:
            state, observer = correct_state(model, state, action, residual, lower, upper,
                                             prior_std=source_config["prior_std"],
                                             max_delta=source_config["state_max_delta"])
        else:
            observer = {"accepted": False, "count": 0, "reason": "invalid_time_pair"}
        corrected_at = time.perf_counter()
        if len(old) and observer["count"]:
            plan, control = controller_call(method, model, engine, cache, k+1, state, old, action,
                                             reference[k+1:], bounds, blocks=source_config["blocks"])
        else:
            plan = old
            control = {"accepted": False, "reason": "no_suffix_or_evidence"}
        if not bounds.valid(plan.numpy(), action.numpy()):
            raise AssertionError("candidate pressure/rate violation")
        committed_at = time.perf_counter()
        # Everything below is diagnostic, outside the measured control path.
        with torch.no_grad():
            corrected_shape = physical_shape(model, state, action).numpy()
        row = {"step": k, "raw_frame": int(trace["raw_indices"][k]),
               "horizon": len(old), "prediction_ms": (predicted_at-started)*1000,
               "image_ms": (detected_at-predicted_at)*1000,
               "observer_ms": (corrected_at-detected_at)*1000,
               "controller_commit_ms": (committed_at-corrected_at)*1000,
               "feedback_ms": (committed_at-started)*1000,
               "edge_count": len(evidence.pixels), "state_accepted": observer["accepted"],
               "suffix_accepted": control["accepted"], "cache_stale": control.get("cache_stale", False),
               "injection_discrepancy_mm": float(np.linalg.norm(corrected_shape[1:]-trace["reference"][k, 1:], axis=-1).mean())}
        rows.append(row)
        steps.append({"step": k, "observer": observer, "control": control})
        issued.append(action.numpy())
        shapes.append(corrected_shape)
        states.append(state.numpy())
        padded = np.full_like(trace["initial_plan"], np.nan)
        padded[k+1:] = plan.numpy()
        suffixes.append(padded)
        if k % 20 == 0 or k == len(images)-1:
            print(f"{method} {k+1}/{len(images)}: feedback={row['feedback_ms']:.1f}ms edges={row['edge_count']} update={control['accepted']}", flush=True)
    if not bounds.valid(np.asarray(issued), last.numpy()):
        raise AssertionError("issued sequence pressure/rate violation")
    timing = {key: timing_summary([row[key] for row in rows[:-1]]) for key in
              ("feedback_ms", "prediction_ms", "image_ms", "observer_ms", "controller_commit_ms")}
    summary = {"method": method, "detector": args.detector, "timing": timing,
               "warmup_ms": warmup_ms,
               "timing_scope": "frame in memory to validated absolute suffix; exposure, IO, scoring, rendering, ACK excluded",
               "state_updates": sum(row["state_accepted"] for row in rows),
               "suffix_updates": sum(row["suffix_accepted"] for row in rows),
               "cache_stale": sum(row["cache_stale"] for row in rows),
               "initial_cache_ms": cache.precompute_ms if cache else 0.,
               "cache_megabytes": cache.cache_bytes/1e6 if cache else 0.,
               "injection_discrepancy_mm": float(np.mean([row["injection_discrepancy_mm"] for row in rows])),
               "pressure_contract_passed": True,
               "not_physical_control": True}
    write_csv(destination/"per_frame.csv", rows)
    write_json(destination/"steps.json", steps)
    write_json(destination/"summary.json", summary)
    np.savez_compressed(destination/"traces.npz", actions=issued, shapes=shapes, states=states, suffixes=suffixes)
    (destination/"COMPLETE").touch()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--methods", nargs="+", choices=["torch_b", "batched_b", "fast_b", "cached_a"],
                        default=["torch_b", "batched_b", "fast_b", "cached_a"])
    parser.add_argument("--detector", choices=["scalar", "vectorized"], default="vectorized")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cache-limit-mm", type=float, default=.5)
    args = parser.parse_args()
    if args.threads < 1 or args.warmup < 1:
        parser.error("threads and warmup must be positive")
    if not np.isfinite(args.cache_limit_mm) or args.cache_limit_mm <= 0:
        parser.error("cache-limit-mm must be finite and positive")
    if len(args.methods) != len(set(args.methods)):
        parser.error("methods must not contain duplicates")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    (output/"status.txt").write_text("running\n")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    try:
        with threadpool_limits(args.threads):
            prepared = prepare(args)
            write_json(output/"config.json", vars(args))
            write_json(output/"environment.json", {"created_at": datetime.now(timezone.utc).isoformat(),
                       "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                       "platform": platform.platform(), "processor": platform.processor(),
                       "threadpools": threadpool_info(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())})
            (output/"commands.sh").write_text(shlex.join([sys.executable, *sys.argv])+"\n")
            summaries = []
            for method in args.methods:
                summaries.append(replay(method, *prepared, args, output/method))
            write_json(output/"summary.json", summaries)
        (output/"status.txt").write_text("complete\n")
        (output/"COMPLETE").touch()
        print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        (output/"status.txt").write_text("failed\n")
        raise


if __name__ == "__main__":
    main()
