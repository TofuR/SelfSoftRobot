#!/usr/bin/env python3
"""Audit unified20 fits and measure frozen, single-core CPU B1 inference.

Run with the selfsr Python and -B. All writes are confined to OUTPUT.
Uses the training run's source snapshot, manifests, checkpoints and normalization.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
OUTPUT = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005/efficiency"
sys.path.insert(0, str(RUN / "source"))
import numpy as np
import torch
from src.benchmarks.modeling_models import make_model
from src.benchmarks.modeling_fast_training import cache_windows

SEEDS = list(range(100, 120))
MAIN = ["hov", "chen_direction", "oscillator", "koopman", "pcc", "base", "linear", "window"]
SOURCES = {}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def track(path):
    path = Path(path)
    key = rel(path)
    if key not in SOURCES:
        SOURCES[key] = dict(path=key, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def read(path):
    return json.loads(track(path).read_text())


def write_json(name, obj):
    (OUTPUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(name, rows):
    require(bool(rows), f"Empty output: {name}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, key, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = []
    for value, group in groups.items():
        item = {key: value, "n_models": len(group)}
        for field in fields:
            values = np.asarray([r[field] for r in group if r.get(field) is not None], dtype=float)
            item[field + "_n"] = len(values)
            for suffix, result in (("mean", np.mean(values) if len(values) else None),
                                   ("sd", np.std(values, ddof=1) if len(values) > 1 else None),
                                   ("median", np.median(values) if len(values) else None),
                                   ("min", np.min(values) if len(values) else None),
                                   ("max", np.max(values) if len(values) else None)):
                item[field + "_" + suffix] = float(result) if result is not None else None
        output.append(item)
    return output


def training_evidence(protocol):
    rows, curves, checks, stages = [], [], [], []
    jobs = read(RUN / "formal_plan.json")
    require(len(jobs) == 286 and protocol["seeds"] == SEEDS, "Formal population mismatch")
    for job in jobs:
        name, seed = job["name"], job["seed"]
        folder = RUN / job["id"]
        manifest = read(folder / "run_manifest.json")
        config = read(folder / "resolved_config.json")
        history = read(folder / "history.json")
        require(config == job["config"], f"Config mismatch: {folder}")
        require(manifest["status"] == "complete" and manifest["seed"] == seed
                and manifest["model"] == name and manifest["device"] == "cpu", str(folder))
        require((manifest["supervised_windows"], manifest["validation_windows"]) == (8988, 2958), str(folder))
        require(seed in SEEDS if seed is not None else name.startswith("linear"), str(folder))
        epochs = [h["epoch"] for h in history]
        require(epochs == ([1, *range(5, 101, 5)] if seed is not None else [1]), f"Checks missing: {folder}")
        values = {h["epoch"]: h["validation_node_mean_mm"] for h in history}
        best = min(values.values())
        best_epoch = min(values, key=values.get)
        require(abs(best - manifest["best_validation_node_mean_mm"]) < 1e-8
                and best_epoch == manifest["best_epoch"], f"Best mismatch: {folder}")
        require(all(a["elapsed_seconds"] < b["elapsed_seconds"] for a, b in zip(history, history[1:])), str(folder))
        require(abs(history[-1]["cumulative_training_seconds"] - manifest["training_seconds"]) < 1e-8, str(folder))
        completed = datetime.fromisoformat(track(folder / "COMPLETE").read_text().strip())
        start = datetime.fromisoformat(manifest["started_at"])
        job_wall = (completed - start).total_seconds()
        wall = manifest["wall_seconds"]
        require(job_wall >= wall - .002 and wall >= history[-1]["elapsed_seconds"], f"Clock inconsistency: {folder}")
        init = manifest.get("model_initialization_and_prior_seconds")
        memory = manifest.get("memory_initialization_seconds")
        memory_requested = bool(config.get("memory_readout_init"))
        require(not memory_requested or memory is not None, f"Missing memory timer: {folder}")
        epoch0 = manifest.get("epoch0_validation_node_mean_mm")
        first = values[1]
        within = next(h for h in history if h["validation_node_mean_mm"] <= 1.05 * best)
        residual = None if init is None else wall - init - (memory if memory_requested else 0.) - manifest["training_seconds"]
        require(residual is None or residual >= -1e-5, f"Negative timing residual: {folder}")
        row = dict(model=name, seed=seed, main_method=name in MAIN,
                   train_windows=8988, val_windows=2958, epochs=config["epochs"],
                   parameters=manifest["parameter_count_including_fitted_reference"],
                   batch_size=config["batch_size"], initial_lr=config["lr"],
                   minibatch_updates_per_epoch=math.ceil(8988 / config["batch_size"]) if seed is not None else 0,
                   configured_reference_prefit_steps=config.get("prior_steps") if name.startswith("hov") else None,
                   device="cpu", concurrent_workers=protocol["workers"], threads_per_worker=protocol["threads"],
                   epoch0_val_mm=epoch0, epoch0_recorded=epoch0 is not None,
                   epoch0_history_recorded=0 in epochs, epoch0_elapsed_seconds=None,
                   epoch0_stage="after_reference_and_memory_initialization" if memory_requested else
                                ("after_closed_form_fit" if seed is None else "after_model_initialization"),
                   epoch1_val_mm=first, best_val_mm=best, best_epoch=best_epoch,
                   selected_epoch_ge80=best_epoch >= 80, selected_epoch100=best_epoch == 100,
                   epoch1_to_best_improvement_mm=first-best,
                   epoch1_to_best_improvement_pct=100*(first-best)/first,
                   epoch1_excess_over_best_pct=100*(first-best)/best,
                   first_recorded_epoch_within5pct_best=within["epoch"],
                   initialization_within5pct_best=None if epoch0 is None else epoch0 <= 1.05*best,
                   elapsed_seconds_to_epoch1=history[0]["elapsed_seconds"],
                   elapsed_seconds_to_within5pct_best=within["elapsed_seconds"],
                   elapsed_seconds_to_best=next(h["elapsed_seconds"] for h in history if h["epoch"] == best_epoch),
                   model_initialization_and_prior_seconds=init,
                   memory_initialization_seconds=memory, memory_initialization_applicable=memory_requested,
                   minibatch_training_seconds=manifest["training_seconds"],
                   other_recorded_fit_seconds=residual, recorded_fit_wall_seconds=wall,
                   task_wall_to_COMPLETE_seconds=job_wall,
                   postfit_replay_and_export_wall_seconds=job_wall-wall,
                   complete_offline_pipeline_seconds=None,
                   started_at=manifest["started_at"], complete_marker_at=completed.isoformat(),
                   source_manifest=rel(folder / "run_manifest.json"), source_history=rel(folder / "history.json"))
        for epoch in (5, 10, 20, 50, 80, 95, 100):
            row[f"epoch{epoch}_val_mm"] = values.get(epoch)
        for cutoff in (20, 50, 80):
            early_best = min((v for e, v in values.items() if e <= cutoff), default=None) if seed is not None else None
            row[f"best_through_epoch{cutoff}_mm"] = early_best
            row[f"after_epoch{cutoff}_best_improvement_mm"] = None if early_best is None else early_best-best
            row[f"after_epoch{cutoff}_best_improvement_pct"] = None if early_best is None else 100*(early_best-best)/early_best
        row["epoch80_to_best_improvement_pct"] = None if 80 not in values else 100*(values[80]-best)/values[80]
        rows.append(row)
        stages.append(dict(model=name, seed=seed, stage=row["epoch0_stage"], val_mm=epoch0,
                           evidence="recorded_manifest" if epoch0 is not None else "not_recorded",
                           elapsed_seconds=None, source=rel(folder / "run_manifest.json")))
        for h in history:
            curves.append(dict(model=name, seed=seed, **h, source=rel(folder / "history.json")))
        checks.append(dict(check="formal_manifest_history_and_timing_consistent", model=name, seed=seed, passed=True))
    for name in protocol["stochastic_models"]:
        require(sorted(r["seed"] for r in rows if r["model"] == name) == SEEDS, f"Seed coverage: {name}")
    fields = [k for k, v in rows[-1].items() if isinstance(v, (int, float, bool)) and k not in ("seed", "main_method")]
    fields += ["memory_initialization_seconds", "complete_offline_pipeline_seconds", "epoch0_elapsed_seconds"]
    summary = aggregate(rows, "model", list(dict.fromkeys(fields)))
    return rows, curves, summary, stages, checks


def hardware():
    cpu_lines = Path("/proc/cpuinfo").read_text().splitlines()
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), hostname=platform.node(),
                cpu=next(s.split(":", 1)[1].strip() for s in cpu_lines if s.startswith("model name")),
                logical_cpus=os.cpu_count(), affinity=list(os.sched_getaffinity(0)),
                load_average=list(os.getloadavg()), platform=platform.platform(), python=sys.version,
                python_executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                torch_config=torch.__config__.show(), timer="time.perf_counter_ns",
                timer_resolution_seconds=time.get_clock_info("perf_counter").resolution,
                thread_environment={k: os.environ[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})


def load_data(protocol):
    manifest_path = Path(protocol["dataset_manifest"])
    meta = read(manifest_path)
    require(meta["H"] == 20 and meta["dt"] == .2 and meta["length_unit"] == "mm", "Dataset mismatch")
    data = {}
    for role in ("val", "test"):
        sequences = []
        for row in meta["files"]:
            if row["role"] != role:
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = manifest_path.parent / path
            with np.load(track(path), allow_pickle=False) as archive:
                seq = {k: archive[k].copy() for k in archive.files}
            seq["record"] = row
            seq["source_path"] = rel(path)
            sequences.append(seq)
        x, y, groups = cache_windows(sequences, 20, "cpu")
        require(len(x) == 2958 and len(sequences) == 3, "Wrong data population")
        data[role] = dict(x=x, y=y, groups=groups, sequences=sequences)
    return data


def frozen_feature_class():
    path = track(RUN / "source/scripts/experiments/analyze_modeling_plugin_convergence.py")
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FeatureMLP"]
    require(len(nodes) == 1, "Missing frozen FeatureMLP")
    namespace = {"nn": torch.nn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["FeatureMLP"]


class FeatureLinear(torch.nn.Module):
    """Identical intercept-first readout to the frozen training worker."""
    def __init__(self, inputs):
        super().__init__()
        self.register_buffer("coefficients", torch.zeros(inputs + 1, 45))

    def forward(self, x):
        return (torch.cat([torch.ones_like(x[:, :1]), x], dim=1) @ self.coefficients).reshape(-1, 15, 3)


class OneStep:
    def __init__(self, name, seed, model, checkpoint, norm, cached=False):
        self.name, self.seed, self.model, self.cached = name, seed, model, cached
        self.mode = "hov_cached_step" if cached else ("hov_h20_full" if name == "hov" else name + "_h20")
        self.center = torch.tensor(checkpoint["center"], dtype=torch.float32)
        self.scale = float(checkpoint["scale"])
        self.mean = self.std = None
        if name in ("window", "base", "linear"):
            variant = checkpoint["config"]["variant"]
            self.mean = torch.tensor(norm["features"][variant]["mean"], dtype=torch.float32).reshape(1, -1)
            self.std = torch.tensor(norm["features"][variant]["std"], dtype=torch.float32).reshape(1, -1)
        self.queue = None
        self.state = None
        self.states = None
        self.outputs, self.expected = [], None
        self.mutation_error = 0.

    def __call__(self, action):
        if self.cached:
            output = self.model.core.step_state(action, self.state)
            self.state = output["latent_z"]
            return output["skeleton"] * self.scale + self.center
        self.queue = torch.cat([self.queue[:, 1:], action.unsqueeze(1)], dim=1)
        x = self.queue
        if self.mean is not None:
            x = x.flatten(1) if self.name == "window" else x[:, -1]
            x = (x - self.mean) / self.std
        return self.model(x) * self.scale + self.center


def prepare_inference(protocol, norm, data, count, main8):
    feature_class = frozen_feature_class()
    test = data["test"]
    # Three contiguous segments; each starts at its split-local first H20 window.
    indices = []
    for group in range(3):
        available = np.flatnonzero(test["groups"] == group)
        take = count // 3 + (group < count % 3)
        require(take <= len(available), "Too many timing samples for contiguous test segments")
        indices.extend(available[:take].tolist())
    samples = []
    for i, index in enumerate(indices):
        group = int(test["groups"][index])
        offset = int(np.flatnonzero(test["groups"] == group)[0])
        seq = test["sequences"][group]
        frame = index - offset + 19
        samples.append(dict(sample=i, test_window_index=index, group=group,
                            frame_id=int(seq["frame_ids"][frame]), source=seq["source_path"],
                            segment_start=i == 0 or int(test["groups"][indices[i-1]]) != group))
    windows = [test["x"][i:i+1].contiguous() for i in indices]
    tasks, stages, checks, checkpoint_rows = [], [], [], []
    names = MAIN if main8 else ["hov", "window"]
    for name in names:
        for seed in ([None] if name == "linear" else SEEDS):
            folder = RUN / "formal" / name / ("closed_form" if seed is None else f"seed_{seed}")
            checkpoint_path = track(folder / "best_eval_model.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            require(checkpoint["config"] == read(folder / "resolved_config.json") and checkpoint["model"] == name,
                    f"Checkpoint provenance: {folder}")
            cfg = checkpoint["config"]
            require(checkpoint["center"] == norm["center"] and checkpoint["scale"] == norm["scale"], "Target normalization mismatch")
            if name in ("base", "window", "linear"):
                dimension = len(norm["features"][cfg["variant"]]["mean"])
                model = FeatureLinear(dimension) if name == "linear" else feature_class(dimension, cfg["hidden"])
            else:
                model, _ = make_model(name, cfg, normalization=(np.array(checkpoint["center"]), checkpoint["scale"]),
                                      geometry_config=checkpoint["geometry_config"])
            if name == "hov":
                # Saved geometry_config holds original prefitted coefficients. Do not load
                # jointly optimized state_dict until this supplemental reference evaluation.
                model.eval().requires_grad_(False)
                pred = torch.cat([model.core.decode_equilibrium(x[:, -1]) * checkpoint["scale"] +
                                  torch.tensor(checkpoint["center"]) for x in data["val"]["x"].split(512)])
                reference_error = float(torch.linalg.vector_norm(pred-data["val"]["y"], dim=-1).mean())
                stages.append(dict(model=name, seed=seed, stage="prefitted_reference_from_saved_geometry",
                                   val_mm=reference_error, evidence="supplemental_validation_reconstruction",
                                   elapsed_seconds=None, source=rel(checkpoint_path)))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            manifest = read(folder / "run_manifest.json")
            require(checkpoint["selected_epoch"] == manifest["best_epoch"]
                    and abs(checkpoint["validation_node_mean_mm"] - manifest["best_validation_node_mean_mm"]) < 1e-8,
                    "Checkpoint selection mismatch")
            prediction_path = track(RUN / "evaluation" / name / folder.name / "predictions.npz")
            with np.load(prediction_path, allow_pickle=False) as archive:
                expected = archive["prediction_mm"][indices].copy()
                require(np.array_equal(archive["frame_ids"][indices], [s["frame_id"] for s in samples]), "Prediction alignment mismatch")
            task = OneStep(name, seed, model, checkpoint, norm)
            task.expected = expected
            tasks.append(task)
            if name == "hov":
                cached = OneStep(name, seed, model, checkpoint, norm, cached=True)
                # Each cache is prepared at B1 from exactly this window's first 19
                # commands, starting from equilibrium at its first command.
                cached.states = [model.core(w[:, :19])["latent_z"].clone() for w in windows]
                cached.expected = expected
                tasks.append(cached)
            checkpoint_rows.append(dict(model=name, seed=seed, path=rel(checkpoint_path),
                                        sha256=SOURCES[rel(checkpoint_path)]["sha256"],
                                        selected_epoch=checkpoint["selected_epoch"],
                                        parameters=manifest["parameter_count_including_fitted_reference"],
                                        frozen_parameters=all(not p.requires_grad for p in model.parameters())))
        print(f"prepared frozen {name}: {1 if name == 'linear' else 20} models", flush=True)
    return tasks, windows, samples, stages, checks, checkpoint_rows


def measure(tasks, windows, samples, warmup, repetitions):
    raw, checks = [], []
    rng = np.random.default_rng(20260913)
    base_order = rng.permutation(len(tasks)).tolist()
    gc.collect()
    gc_enabled = gc.isenabled()
    gc.disable()
    before_hardware = hardware()
    started = time.perf_counter()
    try:
        for i, (window, sample) in enumerate(zip(windows, samples)):
            offset = i % len(tasks)
            order = base_order[offset:] + base_order[:offset]
            if (i // len(tasks)) % 2:
                order = list(reversed(order))
            action = window[:, -1]
            for position, task_index in enumerate(order):
                task = tasks[task_index]
                if task.cached:
                    task.state = task.states[i]
                    previous_state = task.state
                    previous_copy = previous_state.clone()
                elif sample["segment_start"]:
                    task.queue = torch.cat([window[:, :1], window[:, :19]], dim=1)
                start = time.perf_counter_ns()
                result = task(action)
                latency_ns = time.perf_counter_ns() - start
                # All assertions, comparisons, hashing and sample serialization are untimed.
                require(tuple(result.shape) == (1, 15, 3) and bool(torch.isfinite(result).all()), task.mode)
                task.outputs.append(result.clone())
                if task.cached:
                    task.mutation_error = max(task.mutation_error, float((previous_state-previous_copy).abs().max()))
                else:
                    require(torch.equal(task.queue, window), f"Window update mismatch: {task.mode}")
                if i >= warmup:
                    raw.append(dict(mode=task.mode, model=task.name, seed=task.seed,
                                    repetition=i-warmup, sample=i, order_position=position,
                                    test_window_index=sample["test_window_index"], group=sample["group"],
                                    frame_id=sample["frame_id"], latency_ns=latency_ns,
                                    latency_ms=latency_ns/1e6, benchmark_elapsed_seconds=time.perf_counter()-started))
            if i == warmup-1 or (i+1-warmup) % 50 == 0:
                print(f"timing: warmup={min(i+1,warmup)}/{warmup}; measured={max(0,i+1-warmup)}/{repetitions}", flush=True)
    finally:
        if gc_enabled:
            gc.enable()
    after_hardware = hardware()
    by_task = {(t.mode, t.seed): t for t in tasks}
    for task in tasks:
        actual = torch.cat(task.outputs).numpy()
        error = float(np.max(np.abs(actual-task.expected)))
        require(error <= .003, f"Frozen prediction mismatch: {task.mode}/{task.seed}: {error}")
        checks.append(dict(check="b1_output_matches_saved_test_predictions", model=task.name,
                           mode=task.mode, seed=task.seed, n_windows=len(windows),
                           max_abs_coordinate_mm=error, atol_mm=.003, passed=True))
        if task.cached:
            full = torch.cat(by_task["hov_h20_full", task.seed].outputs)
            error = float((torch.cat(task.outputs)-full).abs().max())
            require(error <= 1e-4 and task.mutation_error == 0., f"H20 cache mismatch: {task.seed}: {error}")
            checks.append(dict(check="cached_H19_plus_one_equals_same_full_H20", model="hov", seed=task.seed,
                               n_windows=len(windows), max_abs_coordinate_mm=error, atol_mm=1e-4,
                               input_state_max_mutation=task.mutation_error, passed=True))
    groups = defaultdict(list)
    for row in raw:
        groups[row["mode"], row["seed"]].append(row)
    per_seed = []
    for (mode, seed), group in groups.items():
        values = np.asarray([r["latency_ms"] for r in group])
        require(len(values) == repetitions and (values > 0).all(), f"Timing sample count: {mode}/{seed}")
        p50, p95 = np.quantile(values, [.5, .95], method="linear")
        halves = [np.asarray([r["latency_ms"] for r in group if (r["repetition"] < repetitions//2) == first])
                  for first in (True, False)]
        per_seed.append(dict(mode=mode, model=group[0]["model"], seed=seed, n_calls=len(values),
                             p50_ms=float(p50), p95_ms=float(p95), mean_ms=float(values.mean()),
                             min_ms=float(values.min()), max_ms=float(values.max()),
                             first_half_p50_ms=float(np.median(halves[0])), second_half_p50_ms=float(np.median(halves[1]))))
    summary = aggregate(per_seed, "mode", ["p50_ms", "p95_ms", "mean_ms", "first_half_p50_ms", "second_half_p50_ms"])
    for row in summary:
        values = [r["latency_ms"] for r in raw if r["mode"] == row["mode"]]
        row["pooled_n_calls"] = len(values)
        row["pooled_p50_ms"], row["pooled_p95_ms"] = map(float, np.quantile(values, [.5, .95]))
    return raw, per_seed, summary, checks, dict(before=before_hardware, after=after_hardware)


def file_schema(tables):
    definitions = {
        "seed": "Training seed 100..119; null means one deterministic closed-form fit, never 20 independent seeds.",
        "recorded_fit_wall_seconds": "Original monotonic timer: fit_impl entry through final training_state save; includes reference/model init, memory init, epoch0 validation, Adam, scheduled validation and in-loop writes. Stops before selected-checkpoint replay/export.",
        "task_wall_to_COMPLETE_seconds": "COMPLETE UTC timestamp minus manifest.started_at: includes recorded fit plus selected checkpoint replay, validation_predictions export and final manifest; wall-clock timestamp resolution, ends just before writing COMPLETE itself.",
        "minibatch_training_seconds": "Recorded cumulative Adam/minibatch section time; excludes initialization, scheduled validation and file writes. Closed-form fits have no Adam updates.",
        "model_initialization_and_prior_seconds": "Recorded bundled model construction/reference prefit time; includes initial manifest write; standalone reference timer unavailable.",
        "memory_initialization_seconds": "Recorded memory readout initialization timer; null when not performed/unrecorded, see applicable flag.",
        "complete_offline_pipeline_seconds": "Unknown: shared dataset loading, window/feature building, normalization, worker startup, tuning and queue time are not all timed per fit.",
        "epoch0_val_mm": "Recorded manifest initialization validation error. For HOV it is after reference AND memory initialization; for linear it is already after the closed-form solve.",
        "epoch0_elapsed_seconds": "Unrecorded; never reconstructed by interpolating training curves.",
        "best_epoch": "Earliest strict minimum at recorded epochs 1,5,...,100; epoch0 was diagnostic and ineligible for checkpoint selection.",
        "epoch80_to_best_improvement_pct": "100*(E80-Ebest)/E80; includes possible epoch80 fluctuation.",
        "after_epoch80_best_improvement_pct": "100*(min(E1..E80)-Ebest)/min(E1..E80), using scheduled checks only; measures additional best-checkpoint improvement after epoch80.",
        "p50_ms": "Per frozen model empirical median of measured calls, warmup excluded; numpy linear quantile.",
        "p95_ms": "Per frozen model empirical 95th percentile, warmup excluded; numpy linear quantile.",
        "order_position": "Zero-based call position in the rotated/reversed frozen-model order for this sample.",
        "latency_ns": "Actual perf_counter_ns duration of one CPU B1 call, includes Python wrapper, required queue/state update, feature normalization if applicable and 15x3 mm output.",
        "test_window_index": "Index in concatenated split-local H20 test windows in dataset manifest order; join to input_windows.csv.",
        "configured_reference_prefit_steps": "Frozen prior_steps setting for HOV construction; no old-run step count is borrowed.",
    }
    schema = {"schema": "unified20_efficiency_files_v1", "version": 1,
              "nulls": "JSON null / CSV empty = unavailable or not applicable; not numerical zero.",
              "aggregates": "Each *_mean/sd/median/min/max aggregates per-model values, *_n counts nonnull models; SD uses ddof=1 and is null for n=1. Pooled quantiles are separately labeled.",
              "definitions": definitions, "files": {}}
    keys = {"training_per_seed.csv": ["model", "seed"], "training_history_raw.csv": ["model", "seed", "epoch"],
            "training_summary.csv": ["model"], "initialization_stages.csv": ["model", "seed", "stage"],
            "inference_raw.csv": ["mode", "seed", "repetition"], "inference_per_seed.csv": ["mode", "seed"],
            "inference_summary.csv": ["mode"], "input_windows.csv": ["sample"], "checkpoints.csv": ["model", "seed"]}
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        columns = {}
        for field in fields:
            values = [row.get(field) for row in rows]
            types = sorted({type(v).__name__ for v in values if v is not None})
            columns[field] = dict(types=types, nullable=any(v is None for v in values))
        schema["files"][name] = dict(rows=len(rows), primary_key=keys[name], columns=columns)
    schema["files"].update({"summary.json": {"schema": "unified20_efficiency_summary_v1", "role": "complete findings, protocol, hardware, provenance and QA index"},
                           "validation.json": {"role": "all checks and readiness assessment"},
                           "sources.json": {"role": "source SHA256 inventory"},
                           "summary.md": {"role": "human-readable findings and limitations"},
                           "run.log": {"role": "actual command stdout/stderr"}})
    return schema


def markdown(result):
    rows = {r["model"]: r for r in result["training"]["summary"]}
    hov, window = rows["hov"], rows["window"]
    lines = ["# Unified20 早期收敛与实际推理成本", "", f"来源：`{rel(RUN)}`；正式随机种子 100–119。", "",
             f"HOV 初始化后验证误差 {hov['epoch0_val_mm_mean']:.4f}±{hov['epoch0_val_mm_sd']:.4f} mm，"
             f"首 epoch {hov['epoch1_val_mm_mean']:.4f}±{hov['epoch1_val_mm_sd']:.4f} mm，"
             f"最佳 {hov['best_val_mm_mean']:.4f}±{hov['best_val_mm_sd']:.4f} mm。"
             f"首 epoch 到最佳的下降为 {hov['epoch1_to_best_improvement_pct_mean']:.2f}%；"
             f"前80 epoch已见最佳到全程最佳新增下降 {hov['after_epoch80_best_improvement_pct_mean']:.2f}%。", "",
             f"HOV 最佳 epoch≥80 的模型有 {round(hov['selected_epoch_ge80_mean']*20)}/20，"
             f"最佳在100的有 {round(hov['selected_epoch100_mean']*20)}/20。早期已有较低误差，"
             "固定100 epoch预算不构成完全收敛的证明。", "",
             "|方法|独立拟合数|epoch0 mm|epoch1 mm|最佳val mm|首轮→最佳下降%|80后新增最佳改善%|最佳epoch 均值|记录fit wall s|至COMPLETE任务wall s|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def cell(row, field, digits=3):
        value, sd = row.get(field+"_mean"), row.get(field+"_sd")
        return "—" if value is None else f"{value:.{digits}f}" + (f"±{sd:.{digits}f}" if sd is not None else "")
    for name in MAIN:
        row = rows[name]
        columns = [cell(row, f) for f in ("epoch0_val_mm", "epoch1_val_mm", "best_val_mm",
                   "epoch1_to_best_improvement_pct", "after_epoch80_best_improvement_pct", "best_epoch",
                   "recorded_fit_wall_seconds", "task_wall_to_COMPLETE_seconds")]
        lines.append(f"|{name}|{row['n_models']}|" + "|".join(columns) + "|")
    lines += ["", "均值±样本SD。linear只有一次确定性拟合；其epoch0已在闭式求解后，epoch1没有Adam更新。", "",
              f"CPU原始拟合为8 worker并发、每worker单线程。HOV记录fit wall {cell(hov,'recorded_fit_wall_seconds')} s，"
              f"其中模型/reference预拟合 {cell(hov,'model_initialization_and_prior_seconds')} s，"
              f"记忆初始化 {cell(hov,'memory_initialization_seconds')} s，"
              f"正式minibatch训练 {cell(hov,'minibatch_training_seconds')} s。"
              "这些是共享CPU负载下的任务墙钟时间，不能当作独占CPU计算时间；并发任务时长之和也不能当作项目历时。", "",
              "记录fit计时覆盖初始化、训练、验证及训练内文件写入，结束于最佳checkpoint回放前。"
              "至COMPLETE任务wall用原始UTC时间戳补充回放和结果保存耗时。共享数据读取、窗口/特征构建、"
              "mean/std计算和worker启动没有独立完整计时；全离线管线耗时保留null。", "",
              "history没有epoch0行；全部280次随机拟合的manifest保存了初始化后的验证指标。"
              "HOV该指标包含reference及记忆初始化。初始化checkpoint和epoch0时间点没有保存；"
              "reference单独指标由checkpoint中保存的原始geometry_config重建后在2958个验证窗口补算，"
              "标记为supplemental_validation_reconstruction；未将它当作当时记录的history0。", "",
              "|实际计时路径|冻结模型数|每模型次数|各模型p50的均值±SD ms|各模型p95的均值±SD ms|",
              "|---|---:|---:|---:|---:|"]
    for row in result["inference"]["summary"]:
        lines.append(f"|{row['mode']}|{row['n_models']}|{result['inference']['protocol']['repetitions']}|{cell(row,'p50_ms',4)}|{cell(row,'p95_ms',4)}|")
    h = result["inference"]["hardware"]["before"]
    p = result["inference"]["protocol"]
    lines += ["", f"硬件：{h['cpu']}；绑核 {h['affinity']}；PyTorch {h['torch']}；CPU float32、batch1、"
              f"intra/inter-op单线程。每冻结模型/路径预热{p['warmup']}次、实测{p['repetitions']}次。"
              "模型及seed逐样本轮换，并周期反转顺序。表中先计算每seed分位数再汇总；另存池化分位数。", "",
              "HOV全重算包含窗口更新、H20平衡起点预烧入、一步状态更新、几何读出与毫米反归一化。"
              "缓存一步从该H20前19步准备的状态开始，包含状态更新、状态返回、几何读出与毫米输出；"
              "缓存准备不在一步计时内。window包含4通道窗口更新、80维展平、训练mean/std标准化、"
              "64/64 Tanh MLP和15×3毫米骨架输出。其余主方法亦包含窗口更新和各自特征处理。", "",
              f"缓存一致性：20个seed各{p['warmup']+p['repetitions']}个窗口，"
              f"与同一H20完整重算的最大坐标差 {result['inference']['cache_max_abs_coordinate_mm']:.8g} mm；"
              "输入缓存状态不被原位修改。长历史连续流式状态具有不同的起始条件，本任务未将其视作H20精度评价。", "",
              "计时输入是内存中的真实test压力指令，跨3条序列分别重置；磁盘读取、加载模型、缓存预计算、"
              "正确性检查和原始记录写盘均在计时外。测量未做系统隔离，保留机器负载和前后半段分位数供检查。"
              "上述延迟衡量本实现的模型求值成本，不代表闭环控制Hz、有效采样频率或实测控制吞吐。", "",
              "文件字段、单位、null语义与主键见schema.json；全部检查见validation.json；来源哈希见sources.json。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--main8", action="store_true", help="Also time the remaining main methods; linear has one frozen fit")
    parser.add_argument("--cpu", type=int, default=None, help="Pin to this allowed logical CPU; default first allowed CPU")
    args = parser.parse_args()
    require(args.warmup >= 50 and args.repetitions >= 300, "Require at least 50 warmup and 300 timed calls")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260913)
    original_affinity = sorted(os.sched_getaffinity(0))
    cpu = original_affinity[0] if args.cpu is None else args.cpu
    require(cpu in original_affinity, "CPU outside allowed affinity")
    os.sched_setaffinity(0, {cpu})
    track(Path(__file__))
    protocol = read(RUN / "protocol.json")
    norm = read(RUN / "normalization.json")
    for path in (RUN / "source/src").rglob("*.py"):
        track(path)
    track(RUN / "source/scripts/experiments/run_modeling_unified_repetitions.py")
    rows, curves, training_summary, stages, checks = training_evidence(protocol)
    print(f"training audit complete: {len(rows)} fits, {len(curves)} history rows", flush=True)
    # Save completed read-only analysis before the potentially longer benchmark.
    write_csv("training_per_seed.csv", rows)
    write_csv("training_history_raw.csv", curves)
    write_csv("training_summary.csv", training_summary)
    with torch.inference_mode():
        data = load_data(protocol)
        tasks, windows, samples, reference_stages, prepared_checks, checkpoints = prepare_inference(
            protocol, norm, data, args.warmup + args.repetitions, args.main8)
        raw, per_seed, inference_summary, inference_checks, machine = measure(tasks, windows, samples, args.warmup, args.repetitions)
    checks += prepared_checks + inference_checks
    stages += reference_stages
    tables = {"training_per_seed.csv": rows, "training_history_raw.csv": curves,
              "training_summary.csv": training_summary, "initialization_stages.csv": stages,
              "inference_raw.csv": raw, "inference_per_seed.csv": per_seed,
              "inference_summary.csv": inference_summary, "input_windows.csv": samples, "checkpoints.csv": checkpoints}
    for name, records in tables.items():
        write_csv(name, records)
    cache_checks = [c for c in checks if c["check"] == "cached_H19_plus_one_equals_same_full_H20"]
    require(len(cache_checks) == 20, "Missing cache validations")
    benchmark_protocol = dict(device="cpu", intra_threads=1, interop_threads=1, dtype="float32", batch=1,
                              original_affinity=original_affinity, pinned_cpu=cpu, warmup=args.warmup,
                              repetitions=args.repetitions, seeds=SEEDS, main8=args.main8,
                              command=[sys.executable, "-B", rel(Path(__file__)), *sys.argv[1:]],
                              model_order="RNG20260913 shuffled task list, cyclic shift every sample, reverse each full cycle",
                              sampling="Three contiguous test segments in manifest order; reset window at each sequence boundary",
                              cached_state="Independent same-H20 prefix of 19 commands at equilibrium start, precomputed B1; no long-history stream",
                              included={"hov_h20_full": "queue update + H20 burn-in + state update + geometry + output denormalization",
                                        "hov_cached_step": "one step_state + cache assignment + geometry + output denormalization",
                                        "window_h20": "queue update + flatten80 + training mean/std + frozen MLP + output denormalization",
                                        "others": "queue update + model-specific features/normalization + frozen forward + output denormalization"},
                              excluded=["disk I/O", "loading", "cache preburn/setup", "validation/logging", "sensor/vision", "communication", "controller"],
                              output="float32 CPU tensor [1,15,3], millimeters; resident normalized pressure commands",
                              gc="disabled during measurement, ordinary refcount cleanup retained",
                              system_isolation=False, quantile="numpy method=linear; seed first, then mean/sample SD",
                              timing_interpretation="model call latency, not closed-loop control rate or streaming accuracy")
    result = dict(schema="unified20_efficiency_summary_v1", generated_at=datetime.now(timezone.utc).isoformat(),
                  source_run=rel(RUN), training=dict(summary=training_summary, n_formal_fits=len(rows),
                      n_stochastic_fits=280, n_deterministic_fits=6,
                      timing_scope="CPU 8 concurrent workers x 1 thread; original recorded fit includes reference and memory initialization; COMPLETE extends to final replay/export; shared preparation costs unknown",
                      initialization_summary=aggregate([s for s in stages if s["model"] == "hov"], "stage", ["val_mm"])),
                  inference=dict(summary=inference_summary, per_seed=per_seed, protocol=benchmark_protocol,
                                 hardware=machine, cache_max_abs_coordinate_mm=max(c["max_abs_coordinate_mm"] for c in cache_checks)),
                  validation=dict(assessment="share_with_caveats", passed=len(checks), failed=0,
                                  caveats=["fixed split and fixed 100-epoch budget", "shared CPU walltime and unrecorded shared preprocessing",
                                           "epoch0 scalar recorded, initialization checkpoint/time unavailable", "no system-isolation or closed-loop timing claim"]),
                  files={name: rel(OUTPUT/name) for name in tables})
    write_json("summary.json", result)
    write_json("schema.json", file_schema(tables))
    write_json("validation.json", dict(**result["validation"], checks=checks))
    # Verify all source content is unchanged before publishing the final inventory.
    for source in SOURCES.values():
        require(hashlib.sha256((ROOT/source["path"]).read_bytes()).hexdigest() == source["sha256"],
                f"Source changed during analysis: {source['path']}")
    write_json("sources.json", dict(source_run=rel(RUN), sources=list(SOURCES.values()), all_unchanged=True))
    (OUTPUT / "summary.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(dict(status="complete", output=rel(OUTPUT), measured_calls=len(raw),
                          frozen_checkpoints=len(checkpoints), passed_checks=len(checks)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
