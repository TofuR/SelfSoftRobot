#!/usr/bin/env python3
"""Frozen CPU modeling study: smoke -> LR validation -> 20 repeats -> test.

Only this script and --run receive writes. Run ``prepare`` to freeze the study,
then ``launch`` for a detached, resumable pipeline. No CUDA probing is performed.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = ROOT / "workspace/runs/training/modeling_three_seq_20260913_001928"
DEFAULT_RUN = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
SCRIPT = Path(__file__).resolve()
PLUGIN_SOURCE = Path("scripts/experiments/analyze_modeling_plugin_convergence.py")
SEEDS = list(range(100, 120))
TUNING_SEED = 900001
VARIANTS = ["base", "path", "time", "both", "static_capacity", "window"]
COUNTS = dict(zip(VARIANTS, [7405, 7917, 8941, 9453, 9453, 12269]))
NATIVE = ["hov", "chen_direction", "oscillator", "koopman", "pcc",
          "hov_no_play", "hov_no_maxwell", "hov_no_memory"]
MODELS = NATIVE + VARIANTS
LINEARS = ["linear"] + ["linear_" + name for name in VARIANTS[1:]]
ALL_MODELS = MODELS + LINEARS
FORMAL_FITS = 286
RIDGES = [1e-6, 1e-5, 1e-4, 1e-3, .01]
ALIASES = {
    "main": {name: name for name in NATIVE[:5]} | {"mlp": "base", "linear": "linear", "window": "window"},
    "ablation": {name: name for name in NATIVE[5:]},
    "plugin_mlp": {name: name for name in VARIANTS},
    "plugin_linear": dict(zip(VARIANTS, LINEARS)),
}
CAPACITY_KEYS = ["hidden", "chen_hidden", "latent", "force_hidden", "n_play", "n_maxwell",
                 "ridge", "prior_steps", "calibrate_reference", "calibrate_length_directions",
                 "memory_readout_init", "memory_ridge", "reference_pair_interactions",
                 "reference_pair_length_interactions"]
_DATA = None
_RUN = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_once(path, value):
    if path.exists():
        require(read(path) == value, f"Frozen content differs: {path}")
    else:
        write(path, value)


def runtime(run, threads=1):
    """Import numerical libraries after limiting every BLAS/Torch thread pool."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(run / "source"))
    global np, torch, make_model, Polynomial, cache_windows, fit_normalization
    global initialize_memory_readout, feature_bank, FeatureMLP, FeatureLinear, skeleton_metrics
    import numpy as np
    import torch
    from src.benchmarks.modeling_models import make_model, Polynomial, fit_normalization
    from src.benchmarks.modeling_fast_training import cache_windows
    from src.benchmarks.modeling_memory_initialization import initialize_memory_readout
    from src.evaluation.modeling_benchmark_metrics import skeleton_metrics
    from src.operators.static_drive import MonotoneSplineDrive
    from src.operators.play_bank import PlayBank
    from src.operators.maxwell_bank import MaxwellBank
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    # Execute the exact two reusable definitions, excluding the legacy script's
    # module-level GPU, thread, output-directory, and experiment side effects.
    source = run / "source" / PLUGIN_SOURCE
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in {"feature_bank", "FeatureMLP"}]
    require(len(nodes) == 2, "Missing original plugin definitions")
    namespace = dict(torch=torch, np=np, math=math, nn=torch.nn,
                     MonotoneSplineDrive=MonotoneSplineDrive, PlayBank=PlayBank, MaxwellBank=MaxwellBank)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    feature_bank, FeatureMLP = namespace["feature_bank"], namespace["FeatureMLP"]
    class LinearReadout(torch.nn.Module):
        def __init__(self, inputs):
            super().__init__()
            self.register_buffer("coefficients", torch.zeros(inputs + 1, 45))

        def features(self, x):
            return torch.cat([torch.ones_like(x[:, :1]), x], dim=1)

        def forward(self, x):
            return (self.features(x) @ self.coefficients).reshape(-1, 15, 3)
    FeatureLinear = LinearReadout


def validate_manifest(path):
    meta = read(path)
    require(meta["schema"] == "shape_modeling_temporal_pool_v1" and meta["H"] == 20
            and meta["dt"] == .2 and meta["length_unit"] == "mm"
            and meta["node_order"] == "base_to_tip", "Dataset protocol mismatch")
    groups = sorted({r["group"] for r in meta["files"]})
    require(len(groups) == 3 and len(meta["files"]) == 9, "Expected three sequences and nine slices")
    for group in groups:
        parts = sorted([r for r in meta["files"] if r["group"] == group], key=lambda r: r["start"])
        n = parts[0]["original_frames"]
        require([p["role"] for p in parts] == ["train", "val", "test"], "Split ordering")
        require([p["start"] for p in parts] == [0, round(.6*n), round(.8*n)]
                and [p["stop"] for p in parts] == [round(.6*n), round(.8*n), n], "Expected per-sequence 6:2:2")
    return meta


def load_roles(manifest, roles):
    """Validate requested arrays and chronology; open only the requested roles."""
    meta = validate_manifest(manifest)
    result = {}
    for role in roles:
        seq = []
        for row in meta["files"]:
            if row["role"] != role:
                continue
            source = Path(row["path"])
            if not source.is_absolute():
                source = manifest.parent / source
            with np.load(source, allow_pickle=False) as archive:
                s = {k: archive[k].copy() for k in archive.files}
            n = row["frames"]
            require(s["actions"].shape == (n, 4) and s["positions"].shape == (n, 15, 3), "Array shape mismatch")
            require(np.isfinite(s["actions"]).all() and np.isfinite(s["positions"]).all(), "Nonfinite input")
            require(np.array_equal(s["frame_ids"], np.arange(row["start"], row["stop"]))
                    and np.all(np.diff(s["timestamps"]) > 0), "Sequence continuity mismatch")
            s["record"] = row
            s["manifest_dir"] = manifest.parent
            seq.append(s)
        x, y, groups = cache_windows(seq, 20, "cpu")
        require(len(x) == meta["scored_counts"][role], f"Wrong {role} window count")
        result[role] = dict(sequences=seq, x=x, y=y, groups=groups,
                            frame_ids=np.concatenate([s["frame_ids"][19:] for s in seq]),
                            timestamps=np.concatenate([s["timestamps"][19:] for s in seq]))
    return result


def prepare(run, study, workers, threads):
    require(run.parent == ROOT / "workspace/runs/training" and run != study,
            "Output must be a separate training run directory")
    if (run / "protocol.json").exists():
        protocol = read(run / "protocol.json")
        require(protocol["study"] == str(study) and protocol["workers"] == workers
                and protocol["threads"] == threads, "Existing allocation/protocol differs")
        require((run / "source/scripts/experiments" / SCRIPT.name).read_bytes() == SCRIPT.read_bytes(),
                "Script differs from frozen source; use the frozen version or a new run")
        require((run / "PREPARED").exists(), "Incomplete preparation requires inspection")
        return protocol
    require(not run.exists() or not any(run.iterdir()), "Refusing an existing nonempty output directory")
    run.mkdir(parents=True, exist_ok=True)
    manifest = study / "data/dataset_manifest.json"
    meta = validate_manifest(manifest)
    sources = sorted((ROOT / "src").rglob("*.py")) + [ROOT / PLUGIN_SOURCE, SCRIPT]
    for source in sources:
        destination = run / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    common = dict(history=20, H=20, dt=.2, batch_size=256, epochs=100,
                  minimum_epochs=100, early_stop_checks=100000, endpoint_weight=.25,
                  validation_interval=5, schedule_lr=True, scheduler_patience=4,
                  scheduler_factor=.5, scheduler_min_lr=1e-5, optimizer="Adam",
                  gradient_clip=10., aggregation="pooled_frames", device="cpu", threads=threads,
                  train_stride=1, eval_stride=1, max_train_windows=None, max_eval_windows=None,
                  study_id=run.name, run_kind="unified_fixed_budget_repetition")
    configs = {}
    for name in NATIVE:
        source = study / "formal" / name / "seed_1/resolved_config.json"
        old = read(source)
        require(old["model"] == name, f"Reference key mismatch: {name}")
        configs[name] = dict(common, **{k: old[k] for k in CAPACITY_KEYS if k in old}, model=name,
                             initial_lr_reference=old["lr"], capacity_source=str(source))
    for name in VARIANTS:
        configs[name] = dict(common, model="feature_mlp", variant=name, hidden=64,
                             expected_parameter_count=COUNTS[name], feature_standardization="train_mean_std")
    for name, variant in zip(LINEARS, VARIANTS):
        configs[name] = dict(common, model="feature_linear", variant=variant,
                             feature_standardization="train_mean_std", ridge=1e-4)
    contrasts = [("main", "hov", name) for name in ["chen_direction", "oscillator", "koopman", "pcc", "base", "window", "linear"]]
    contrasts += [("ablation", "hov", name) for name in NATIVE[5:]]
    contrasts += [("plugin", "base", name) for name in VARIANTS[1:]]
    protocol = dict(schema="modeling_unified20_v1", frozen_at=stamp(), study=str(study), run=str(run),
                    dataset_manifest=str(manifest), seeds=SEEDS, tuning_seed=TUNING_SEED,
                    stochastic_models=MODELS, deterministic_models=LINEARS,
                    formal_unique_fits=FORMAL_FITS, stochastic_fits=280, deterministic_fits=6, aliases=ALIASES,
                    common=common, workers=workers, threads=threads, device="cpu",
                    lr_candidates=[.001, .003], tuning_epochs=100, smoke_epochs=2,
                    ridge_candidates=RIDGES, ridge_selection="minimum validation mean-node mm; smaller ridge breaks exact ties",
                    lr_selection="minimum best pooled validation mean-node mm; lower LR breaks exact ties",
                    checkpoint_selection="epoch 1, 5, 10, ..., 100; strict best validation mean-node mm",
                    feature_implementation=str(PLUGIN_SOURCE), source_snapshot="source",
                    history="H20; independent split-local causal windows, current-frame target, stride 1",
                    normalization="target center/scale from all training frames; feature mean/std from training windows",
                    test_gate="all 286 formal train/val fits at full budget and frozen config before loading test arrays",
                    seed_policy="all seeds 100..119 retained; interrupted jobs retry the same seed/config and retain prior attempt",
                    linear="six deterministic CPU ridge fits, one per representation; train standardization shared with MLP; endpoint output weight 1 + 15*0.25",
                    masks=dict(models=NATIVE + ["base", "window", "linear"], frames=2958,
                               radius_mm=8., stride=1, metrics=["iou", "dice"], renderer="existing render_tube",
                               cache="shared read-only uint8 mmap decoded once per test target frame"),
                    uncertainty="training randomness conditional on the historically selected three-sequence temporal split",
                    convergence="save initialization and all scheduled val checks; report late improvement and selected epoch",
                    statistics=dict(metric="pooled mean_node_mm", contrasts=contrasts,
                                    wilcoxon="two-sided exact signed-rank sign enumeration, average tied ranks, zero_method=wilcox",
                                    multiplicity="separate Holm families: main7, ablation3, plugin5",
                                    sensitivity="two-sided exact binomial sign test, same Holm families",
                                    bootstrap="paired seed differences, 20000 resamples, RNG20260913, percentile95",
                                    linear="one fixed reference in main contrast; all randomness is HOV's 20 fits; plugin linear descriptive"),
                    outputs=["frozen_configs.json", "training_validation_plan.json", "status.json", "pipeline.log",
                             "raw_validation.csv", "raw_test.csv", "raw_test_by_sequence.csv", "raw_paired_differences.csv",
                             "paired_statistics.json", "formal/*/seed_*/history.json", "formal/*/seed_*/best_eval_model.pt"])
    write(run / "protocol.json", protocol)
    write(run / "candidate_configs.json", configs)
    write(run / "aliases.json", ALIASES)
    write(run / "training_validation_plan.json", dict(
        frozen_at=protocol["frozen_at"], smoke=dict(models=ALL_MODELS, epochs=2, seed=TUNING_SEED, lr=.001),
        tuning=dict(models=MODELS, lrs=[.001, .003], epochs=100, seed=TUNING_SEED, fits=28),
        linear_tuning=dict(models=LINEARS, ridges=RIDGES, fits=len(LINEARS)*len(RIDGES)),
        formal=dict(models=MODELS, seeds=SEEDS, fits=280, deterministic_linear_fits=6),
        evaluation=dict(role="test", gate="TRAIN_VAL_COMPLETE.json and verified 286 completed fits")))
    (run / "TRAINING_VALIDATION_PLAN.md").write_text(
        "# 统一20次重复训练计划\n\n"
        "数据：已有3序列5Hz，各序列按时间6:2:2划分后pool；H20，split内历史，stride1。"
        "训练/验证有效窗口8988/2958；test在统一完成检查后开放。\n\n"
        "1. CPU冒烟：14个随机模型和6个线性表示，seed900001，神经模型2 epoch，全量train/val，保留既定先验拟合步骤。\n"
        "2. 独立LR验证：14个随机模型，seed900001，每个LR0.001/0.003各100 epoch；"
        "按最佳val均值选LR，平局取较小LR。随后写frozen_configs.json及formal_plan.json。\n"
        "线性六表示比较ridge=1e-6/1e-5/1e-4/1e-3/0.01，仅用val选择。\n"
        "3. 正式：固定seed100..119，每个模型20次、每次100 epoch；linear六表示各确定性闭式一次，共286次。\n"
        "4. 核验全部正式配置、checkpoint、完成标记及训练预算，之后统一test并输出raw及配对检验。\n\n"
        "Adam，batch256，归一化坐标MSE+0.25端点MSE，梯度裁剪10；epoch1及每5epoch验证。"
        "ReduceLROnPlateau factor0.5/patience4/min_lr1e-5；固定100epoch。"
        "选择指标为pooled mean node Euclidean error(mm)。全部MLP为64/64 Tanh；"
        "base/path/time/both/static_capacity/window参数7405/7917/8941/9453/9453/12269。"
        "主静态MLP=插件base，主window=插件window，直接引用同一checkpoint。"
        "非MLP容量和HOV先验设置来自旧seed1配置；所有先验及标准化只用train。\n\n"
        "保存全部正式seed、窗口模型、最佳checkpoint、最终训练状态、验证曲线、初始化及收敛诊断。"
        "中断保留旧attempt，继续时仅重跑相同seed及配置。"
        "主对照7/消融3/MLP插件5项比较分族Wilcoxon+Holm，符号检验敏感性；"
        "主线性对照为同一固定解，差值随机性仅来自HOV20次，插件线性仅描述。配对bootstrap95%CI。"
        "主对照及消融全2958帧IoU/Dice，固定8mm，原render_tube，目标mask共享缓存。"
        "统计解释限定为固定数据划分上的训练随机性；旧研究曾参与序列和容量选择。\n", encoding="utf-8")
    runtime(run, threads)
    data = load_roles(manifest, ("train", "val"))
    center, scale = fit_normalization(data["train"]["sequences"])
    norms, arrays = {}, {}
    raw = {role: feature_bank(d["x"]) for role, d in data.items()}
    for variant in VARIANTS:
        mean, std = raw["train"][variant].mean(0), raw["train"][variant].std(0)
        std = np.where(std < 1e-6, 1., std)
        norms[variant] = dict(mean=mean.tolist(), std=std.tolist())
        for role in data:
            arrays[f"{role}_{variant}"] = ((raw[role][variant] - mean) / std).astype(np.float32)
        model = FeatureMLP(raw["train"][variant].shape[1], 64)
        require(sum(p.numel() for p in model.parameters()) == COUNTS[variant], f"MLP capacity: {variant}")
    normalization = dict(center=center.tolist(), scale=scale, features=norms)
    write(run / "normalization.json", normalization)
    np.savez(run / "training_features.npz", **arrays)
    write(run / "data_check.json", dict(checked_at=stamp(), roles=["train", "val"],
                                       windows={r: len(d["x"]) for r, d in data.items()},
                                       feature_parameter_counts=COUNTS,
                                       slices=[{k: row[k] for k in ("group", "role", "start", "stop", "frames")}
                                               for row in meta["files"]]))
    (run / "PREPARED").write_text(stamp() + "\n")
    write(run / "status.json", dict(phase="prepared", status="ready", updated_at=stamp(), formal_total=286,
                                   workers=workers, threads=threads, stochastic_fits=280, deterministic_fits=6))
    return protocol


def worker_init(run_string):
    global _RUN, _DATA
    _RUN = Path(run_string)
    protocol = read(_RUN / "protocol.json")
    runtime(_RUN, protocol["threads"])
    _DATA = load_roles(Path(protocol["dataset_manifest"]), ("train", "val"))
    norm = read(_RUN / "normalization.json")
    _DATA["center"] = torch.tensor(norm["center"], dtype=torch.float32)
    _DATA["scale"] = norm["scale"]
    with np.load(_RUN / "training_features.npz", allow_pickle=False) as arrays:
        _DATA["features"] = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}


def jobs_for(phase, configs):
    jobs = []
    for name in ALL_MODELS:
        if name in LINEARS:
            for ridge in (RIDGES if phase == "tuning" else [configs[name]["ridge"]]):
                config = dict(configs[name], seed=None, lr=.001, ridge=ridge, epochs=1, run_kind=f"unified_{phase}")
                suffix = f"ridge_{ridge:g}" if phase == "tuning" else "closed_form"
                jobs.append(dict(id=f"{phase}/{name}/{suffix}", phase=phase, name=name, seed=None, config=config))
            continue
        for seed in (SEEDS if phase == "formal" else [TUNING_SEED]):
            for lr in ([.001, .003] if phase == "tuning" else [configs[name].get("lr", .001)]):
                config = dict(configs[name], seed=seed, lr=lr,
                              epochs=2 if phase == "smoke" else 100,
                              run_kind=f"unified_{phase}")
                suffix = f"lr_{lr:g}" if phase == "tuning" else f"seed_{seed}"
                jobs.append(dict(id=f"{phase}/{name}/{suffix}", phase=phase, name=name, seed=seed, config=config))
    # Distribute seeds and model families over workers, including slow models.
    if phase == "formal":
        jobs.sort(key=lambda j: (j["seed"] if j["seed"] is not None else 99, ALL_MODELS.index(j["name"])))
    return jobs


def complete_job(run, job):
    destination = run / job["id"]
    if not (destination / "COMPLETE").exists():
        return False
    result = read(destination / "run_manifest.json")
    history = read(destination / "history.json")
    expected = 1 if job["name"] in LINEARS else job["config"]["epochs"]
    require(result["status"] == "complete" and result["current_epoch"] == expected
            and read(destination / "resolved_config.json") == job["config"]
            and history[-1]["epoch"] == expected and math.isfinite(result["best_validation_node_mean_mm"])
            and (destination / "best_eval_model.pt").exists()
            and (destination / "training_state.pt").exists(), f"Invalid completed job: {job['id']}")
    return True


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prediction(model, x, center, scale, batch_size=256):
    model.eval()
    with torch.inference_mode():
        pred = torch.cat([model(batch) * scale + center for batch in x.split(batch_size)])
    require(bool(torch.isfinite(pred).all()), "Nonfinite prediction")
    return pred


def fit_job(job):
    destination = _RUN / job["id"]
    if complete_job(_RUN, job):
        return read(destination / "run_manifest.json")
    if destination.exists():
        retained = _RUN / "interrupted_attempts" / job["id"] / f"{time.time_ns()}"
        retained.parent.mkdir(parents=True, exist_ok=True)
        destination.rename(retained)
    destination.mkdir(parents=True)
    write(destination / "resolved_config.json", job["config"])
    with (destination / "train.log").open("a", buffering=1) as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            try:
                return fit_impl(job, destination)
            except BaseException:
                write(destination / "failure.json", dict(at=stamp(), error=traceback.format_exc(), job=job))
                raise


def fit_impl(job, destination):
    cfg, name = job["config"], job["name"]
    if cfg["seed"] is not None:
        seed_all(cfg["seed"])
    start = time.perf_counter()
    x, y = _DATA["train"]["x"], _DATA["train"]["y"]
    vx, vy = _DATA["val"]["x"], _DATA["val"]["y"]
    center, scale = _DATA["center"], _DATA["scale"]
    state = dict(status="running", started_at=stamp(), pid=os.getpid(), job_id=job["id"],
                 model=name, seed=cfg["seed"], current_epoch=0, best_epoch=None,
                 supervised_windows=len(x), validation_windows=len(vx), device="cpu")
    write(destination / "run_manifest.json", state)
    if name in VARIANTS + LINEARS:
        variant = cfg["variant"]
        x, vx = _DATA["features"][f"train_{variant}"], _DATA["features"][f"val_{variant}"]
        model, geometry = (FeatureLinear(x.shape[1]) if name in LINEARS else FeatureMLP(x.shape[1], 64)), None
    else:
        model, geometry = make_model(name, cfg, _DATA["train"]["sequences"], (center.numpy(), scale))
    state["model_initialization_and_prior_seconds"] = time.perf_counter() - start
    if cfg.get("memory_readout_init"):
        t = time.perf_counter()
        state["memory_initialization"] = initialize_memory_readout(model, x, y, ridge=cfg["memory_ridge"])
        state["memory_initialization_seconds"] = time.perf_counter() - t
    target = (y - center) / scale
    parameters = [p for p in model.parameters() if p.requires_grad]
    state["parameter_count"] = sum(p.numel() for p in model.parameters())
    state["trainable_parameter_count"] = sum(p.numel() for p in parameters)
    state["closed_form_coefficients"] = model.coefficients.numel() if name in LINEARS else 0
    state["parameter_count"] += state["closed_form_coefficients"]
    reference_names = {"reference_bend_bias", "reference_bend_dirs", "reference_length_bias",
                       "reference_length_dirs", "reference_drive_weights"}
    state["fitted_reference_buffer_count"] = sum(v.numel() for k, v in model.named_buffers()
                                                if k.rsplit(".", 1)[-1] in reference_names)
    state["parameter_count_including_fitted_reference"] = state["parameter_count"] + state["fitted_reference_buffer_count"]
    if name in VARIANTS:
        require(state["parameter_count"] == COUNTS[name], f"Wrong parameter count: {name}")
    native_counts = dict(chen_direction=56493, pcc=1156, koopman=3981, oscillator=7037)
    if name in native_counts:
        require(state["parameter_count"] == native_counts[name], f"Wrong native capacity: {name}")
    if name in LINEARS:
        # Closed-form minimizer with the same relative endpoint weight as the
        # neural objective. One ridge coefficient is retained from the old fit.
        design = model.features(x).double()
        penalty = torch.eye(design.shape[1], dtype=torch.float64) * cfg["ridge"]
        penalty[0, 0] = 0
        output_weights = torch.ones(45, dtype=torch.float64)
        output_weights[-3:] = 1 + 15 * cfg["endpoint_weight"]
        gram = design.T @ design / len(design)
        rhs = design.T @ target.flatten(1).double() / len(design)
        coefficient = torch.linalg.solve(gram[None] + penalty[None] / output_weights[:, None, None],
                                         rhs.T[..., None]).squeeze(-1).T
        model.coefficients.copy_(coefficient.float())
    optimizer = torch.optim.Adam(parameters, lr=cfg["lr"]) if parameters else None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=4, min_lr=1e-5) if optimizer else None
    epochs = cfg["epochs"] if optimizer else 1
    require(optimizer is not None or name in LINEARS, f"Unexpected deterministic model: {name}")
    epoch0 = float(torch.linalg.vector_norm(prediction(model, vx, center, scale) - vy, dim=-1).mean())
    state["epoch0_validation_node_mean_mm"] = epoch0
    history, best, best_epoch, training_seconds = [], float("inf"), 0, 0.
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        loss_sum = 0.
        if optimizer:
            model.train()
            for indices in torch.randperm(len(x)).split(cfg["batch_size"]):
                optimizer.zero_grad(set_to_none=True)
                pred, goal = model(x[indices]), target[indices]
                loss = (pred - goal).square().mean() + cfg["endpoint_weight"] * (pred[:, -1] - goal[:, -1]).square().mean()
                require(bool(torch.isfinite(loss)), "Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 10.)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(indices)
        training_seconds += time.perf_counter() - epoch_start
        if epoch % 5 and epoch not in (1, epochs):
            continue
        pred = prediction(model, vx, center, scale)
        metric = float(torch.linalg.vector_norm(pred - vy, dim=-1).mean())
        if metric < best:
            best, best_epoch = metric, epoch
            torch.save(dict(schema="unified_shape_checkpoint_v1", model=name, state_dict=model.state_dict(),
                            config=cfg, geometry_config=geometry, center=center.tolist(), scale=scale,
                            selected_epoch=epoch, validation_node_mean_mm=metric), destination / "best_eval_model.pt")
        history.append(dict(epoch=epoch, train_loss=loss_sum / len(x) if optimizer else None,
                            validation_node_mean_mm=metric, best_validation_node_mean_mm=best,
                            lr=optimizer.param_groups[0]["lr"] if optimizer else None,
                            epoch_seconds=time.perf_counter() - epoch_start,
                            cumulative_training_seconds=training_seconds, elapsed_seconds=time.perf_counter() - start))
        if scheduler:
            scheduler.step(metric)
        state.update(current_epoch=epoch, best_epoch=best_epoch, best_validation_node_mean_mm=best,
                     wall_seconds=time.perf_counter() - start)
        write(destination / "history.json", history)
        write(destination / "run_manifest.json", state)
        print(f"{job['id']} epoch={epoch}/{epochs} val={metric:.6f} best={best:.6f}", flush=True)
        if epoch % 25 == 0 or epoch == epochs:
            import random
            torch.save(dict(epoch=epoch, model=model.state_dict(), config=cfg,
                            optimizer=optimizer.state_dict() if optimizer else None,
                            scheduler=scheduler.state_dict() if scheduler else None,
                            torch_rng_state=torch.get_rng_state(), numpy_rng_state=np.random.get_state(),
                            python_rng_state=random.getstate()), destination / "training_state.pt")
    recent = [row["validation_node_mean_mm"] for row in history[-3:]]
    trend = (recent[0] - min(recent)) / max(abs(recent[0]), 1e-8)
    state.update(status="complete", completed_at=stamp(), wall_seconds=time.perf_counter() - start,
                 training_seconds=training_seconds, stop_reason="fixed_budget" if optimizer else "closed_form",
                 convergence=dict(selected_epoch=best_epoch, last_validation_node_mean_mm=recent[-1],
                                  recent_relative_improvement=trend, selected_in_last_20_epochs=best_epoch >= 80,
                                  epoch0_to_best_improvement_mm=epoch0 - best,
                                  assessment="still_improving" if trend > .02 else "late_checks_flat_or_noisy",
                                  diagnostic="last three scheduled checks; fixed budget, not a guarantee of convergence"))
    best_ckpt = torch.load(destination / "best_eval_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    selected_pred = prediction(model, vx, center, scale).numpy()
    selected_metric = float(torch.linalg.vector_norm(torch.from_numpy(selected_pred) - vy, dim=-1).mean())
    require(abs(selected_metric - best) < 1e-6, "Checkpoint replay differs from selected validation")
    np.savez_compressed(destination / "validation_predictions.npz", prediction_mm=selected_pred,
                        groups=_DATA["val"]["groups"], frame_ids=_DATA["val"]["frame_ids"])
    write(destination / "run_manifest.json", state)
    (destination / "COMPLETE").write_text(stamp() + "\n")
    return state


def stage(run, name, jobs, workers):
    pending = [j for j in jobs if not complete_job(run, j)]
    done, failed = len(jobs) - len(pending), []
    start = time.monotonic()
    def status(active):
        write(run / "status.json", dict(phase=name, status="running", completed=done, total=len(jobs),
                                       failed=failed, active_jobs=active, updated_at=stamp(),
                                       elapsed_seconds=time.monotonic() - start, pid=os.getpid(),
                                       workers=workers, formal_total=286))
    status([])
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=worker_init, initargs=(str(run),)) as pool:
        # Bounded submission leaves active_jobs useful and avoids losing a long
        # queue of already dispatched work on an interrupted controller.
        iterator, active = iter(pending), {}
        for _ in range(min(workers, len(pending))):
            job = next(iterator)
            active[pool.submit(fit_job, job)] = job
        while active:
            finished, _ = concurrent.futures.wait(active, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                job = active.pop(future)
                try:
                    result = future.result()
                    done += 1
                    print(f"DONE {job['id']} val={result['best_validation_node_mean_mm']:.6f} wall={result['wall_seconds']:.1f}s", flush=True)
                except BaseException:
                    failed.append(job["id"])
                    print(traceback.format_exc(), flush=True)
                try:
                    next_job = next(iterator)
                except StopIteration:
                    pass
                else:
                    active[pool.submit(fit_job, next_job)] = next_job
            status([job["id"] for job in active.values()])
    require(not failed, f"Stage {name} failed; preserve jobs and rerun same plan: {failed}")
    write(run / f"{name}_complete.json", dict(completed_at=stamp(), fits=len(jobs), elapsed_seconds=time.monotonic() - start))
    return [read(run / j["id"] / "run_manifest.json") for j in jobs]


def freeze_configs(run, candidates):
    selections, audit = {}, {}
    for name in MODELS:
        choices = []
        for lr in [.001, .003]:
            result = read(run / "tuning" / name / f"lr_{lr:g}" / "run_manifest.json")
            choices.append(dict(lr=lr, val=result["best_validation_node_mean_mm"], epoch=result["best_epoch"]))
        selected = min(choices, key=lambda x: (x["val"], x["lr"]))
        selections[name] = dict(candidates[name], lr=selected["lr"])
        audit[name] = dict(candidates=choices, selected_lr=selected["lr"])
    for name in LINEARS:
        choices = []
        for ridge in RIDGES:
            result = read(run / "tuning" / name / f"ridge_{ridge:g}" / "run_manifest.json")
            choices.append(dict(ridge=ridge, val=result["best_validation_node_mean_mm"]))
        selected = min(choices, key=lambda x: (x["val"], x["ridge"]))
        selections[name] = dict(candidates[name], lr=.001, ridge=selected["ridge"])
        audit[name] = dict(candidates=choices, selected_ridge=selected["ridge"])
    path = run / "frozen_configs.json"
    if path.exists():
        frozen = read(path)
        require(frozen["configs"] == selections and frozen["selection_audit"] == audit, "Frozen selections differ")
    else:
        frozen = dict(frozen_at=stamp(), tuning_seed=TUNING_SEED, selection_role="val", configs=selections,
                      selection_audit=audit, formal_seeds=SEEDS, aliases=ALIASES)
        write(path, frozen)
    jobs = jobs_for("formal", selections)
    require(len(jobs) == 286 and len({j["id"] for j in jobs}) == 286, "Duplicate/missing formal jobs")
    write_once(run / "formal_plan.json", jobs)
    return jobs


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def training_gate(run, jobs):
    require(len(jobs) == 286 and all(complete_job(run, j) for j in jobs), "Formal train/val barrier incomplete")
    rows = []
    for job in jobs:
        state = read(run / job["id"] / "run_manifest.json")
        rows.append(dict(model=job["name"], seed=job["seed"], lr=job["config"]["lr"],
                         parameter_count=state["parameter_count_including_fitted_reference"],
                         bare_parameter_count=state["parameter_count"], trainable_parameter_count=state["trainable_parameter_count"],
                         best_epoch=state["best_epoch"], val_mean_node_mm=state["best_validation_node_mean_mm"],
                         epoch0_val_mean_node_mm=state["epoch0_validation_node_mean_mm"],
                         wall_seconds=state["wall_seconds"], checkpoint=f"{job['id']}/best_eval_model.pt",
                         **state["convergence"]))
    write_csv(run / "raw_validation.csv", rows)
    path = run / "TRAIN_VAL_COMPLETE.json"
    if not path.exists():
        write(path, dict(verified_at=stamp(), formal_fits=286, stochastic_seeds=SEEDS,
                         deterministic_fits=6, stochastic_fits=280, configs="frozen_configs.json", plan="formal_plan.json"))


def summarize_errors(pred, target):
    frame = skeleton_metrics(pred, target)
    row = {k: float(v.mean()) for k, v in frame.items()}
    errors = np.linalg.norm(pred.astype(np.float64) - target, axis=-1)
    row.update(node_global_rmse_mm=float(np.sqrt(np.mean(errors**2))),
               endpoint_rmse_mm=float(np.sqrt(np.mean(errors[:, -1]**2))))
    return row, frame


def evaluate(run, jobs):
    # This call is also mandatory for direct --stage test invocation.
    training_gate(run, jobs)
    protocol = read(run / "protocol.json")
    write(run / "test_access.json", dict(first_or_resumed_access_at=stamp(), gate="TRAIN_VAL_COMPLETE.json", role="test"))
    test = load_roles(Path(protocol["dataset_manifest"]), ("test",))["test"]
    norm = read(run / "normalization.json")
    center, scale = torch.tensor(norm["center"], dtype=torch.float32), norm["scale"]
    features = feature_bank(test["x"])
    for name in VARIANTS:
        values = norm["features"][name]
        features[name] = torch.from_numpy(((features[name] - np.asarray(values["mean"], dtype=np.float32)) /
                                         np.asarray(values["std"], dtype=np.float32)).astype(np.float32))
    target = test["y"].numpy()
    np.savez_compressed(run / "test_targets.npz", target_mm=target, groups=test["groups"],
                        frame_ids=test["frame_ids"], timestamps=test["timestamps"])
    rows, by_sequence = [], []
    for index, job in enumerate(jobs):
        name, source = job["name"], run / job["id"]
        dest = evaluation_dir(run, job)
        if (dest / "COMPLETE").exists():
            row = read(dest / "metrics.json")
            groups = read(dest / "sequence_metrics.json")
        else:
            checkpoint = torch.load(source / "best_eval_model.pt", map_location="cpu", weights_only=False)
            require(checkpoint["config"] == job["config"], "Checkpoint configuration differs")
            if name in VARIANTS + LINEARS:
                x = features[job["config"]["variant"]]
                model = FeatureLinear(x.shape[1]) if name in LINEARS else FeatureMLP(x.shape[1], 64)
            else:
                x = test["x"]
                model, _ = make_model(name, job["config"], normalization=(center.numpy(), scale),
                                      geometry_config=checkpoint["geometry_config"])
            model.load_state_dict(checkpoint["state_dict"])
            pred = prediction(model, x, center, scale).numpy()
            metrics, frame = summarize_errors(pred, target)
            fit = read(source / "run_manifest.json")
            row = dict(model=name, seed=job["seed"], test_frames=len(pred),
                       parameter_count=fit["parameter_count_including_fitted_reference"], bare_parameter_count=fit["parameter_count"],
                       best_epoch=fit["best_epoch"], val_mean_node_mm=fit["best_validation_node_mean_mm"],
                       checkpoint=f"{job['id']}/best_eval_model.pt", **metrics)
            groups = []
            for i, sequence in enumerate(test["sequences"]):
                mask = test["groups"] == i
                measured, _ = summarize_errors(pred[mask], target[mask])
                groups.append(dict(model=name, seed=job["seed"], group=sequence["record"]["group"],
                                   test_frames=int(mask.sum()), **measured))
            dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / "predictions.npz", prediction_mm=pred, groups=test["groups"],
                                frame_ids=test["frame_ids"], **frame)
            write(dest / "metrics.json", row)
            write(dest / "sequence_metrics.json", groups)
            (dest / "COMPLETE").write_text(stamp() + "\n")
        rows.append(row)
        by_sequence.extend(groups)
        write(run / "status.json", dict(phase="test", status="running", completed=index + 1,
                                       total=len(jobs), pid=os.getpid(), updated_at=stamp()))
    write_csv(run / "raw_test.csv", rows)
    write_csv(run / "raw_test_by_sequence.csv", by_sequence)
    statistics(run, rows)
    write(run / "skeleton_evaluation_complete.json", dict(completed_at=stamp(), fits=FORMAL_FITS,
                                                         mask_status="pending"))
    mask_evaluation(run, jobs, test)
    return rows


def evaluation_dir(run, job):
    return run / "evaluation" / job["name"] / ("closed_form" if job["seed"] is None else f"seed_{job['seed']}")


def mask_worker_init(run_string):
    global _RUN, _MASK, _MASK_META, render_tube
    _RUN = Path(run_string)
    runtime(_RUN, read(_RUN / "protocol.json")["threads"])
    from src.evaluation.modeling_benchmark_metrics import render_tube
    _MASK = np.load(_RUN / "mask_targets.npy", mmap_mode="r", allow_pickle=False)
    _MASK_META = read(_RUN / "mask_cache.json")


def mask_job(job):
    destination = evaluation_dir(_RUN, job)
    if (destination / "MASK_COMPLETE").exists():
        return read(destination / "mask_metrics.json")
    with np.load(destination / "predictions.npz", allow_pickle=False) as archive:
        pred, groups, ids = archive["prediction_mm"], archive["groups"], archive["frame_ids"]
    require(len(pred) == 2958 and np.array_equal(ids, np.asarray(_MASK_META["frame_ids"])), "Mask target alignment")
    iou, dice = np.empty(len(pred)), np.empty(len(pred))
    start = time.perf_counter()
    for index, points in enumerate(pred):
        projection = _MASK_META["sequences"][int(groups[index])]
        target_mask = _MASK[index]
        estimate = render_tube(points, np.asarray(projection["model_to_mask"]), target_mask.shape,
                               projection["radius_px"])
        intersection = int(np.count_nonzero(estimate & target_mask))
        n_pred = int(np.count_nonzero(estimate))
        n_target = _MASK_META["foreground_counts"][index]
        union = n_pred + n_target - intersection
        total = n_pred + n_target
        iou[index] = intersection / union if union else 1.
        dice[index] = 2 * intersection / total if total else 1.
    per_sequence = []
    for index, projection in enumerate(_MASK_META["sequences"]):
        mask = groups == index
        per_sequence.append(dict(group=projection["group"], mask_frames=int(mask.sum()),
                                 mask_iou=float(iou[mask].mean()), mask_dice=float(dice[mask].mean())))
    result = dict(model=job["name"], seed=job["seed"], mask_frames=len(pred), radius_mm=8., mask_stride=1,
                  mask_iou=float(iou.mean()), mask_dice=float(dice.mean()), by_sequence=per_sequence,
                  elapsed_seconds=time.perf_counter() - start)
    np.savez_compressed(destination / "mask_frame_scores.npz", frame_ids=ids, groups=groups, mask_iou=iou, mask_dice=dice)
    write(destination / "mask_metrics.json", result)
    (destination / "MASK_COMPLETE").write_text(stamp() + "\n")
    return result


def mask_evaluation(run, jobs, test):
    """Decode all targets once, then reuse one shared mmap across eight workers."""
    import cv2
    cv2.setNumThreads(1)
    cache_path = run / "mask_cache.json"
    if not cache_path.exists():
        require(len(test["x"]) == 2958, "Expected all 2958 mask targets")
        shape = test["sequences"][0]["record"]["mask_shape"]
        masks = np.lib.format.open_memmap(run / "mask_targets.npy", mode="w+", dtype=bool,
                                         shape=(2958, *shape))
        projections, foreground_counts = [], []
        offset = 0
        for sequence in test["sequences"]:
            row, matrix = sequence["record"], sequence["model_to_mask"]
            linear = matrix[:2, :2]
            require(np.allclose(matrix[2], [0, 0, 1]) and
                    np.allclose(linear.T @ linear, np.eye(2) * np.sum(linear[:, 0]**2)), "Mask similarity projection")
            require(row["mask_shape"] == shape, "Mask cache shape mismatch")
            projections.append(dict(group=row["group"], model_to_mask=matrix.tolist(),
                                    radius_px=float(8. * np.linalg.norm(linear[:, 0]))))
            directory = Path(row["masks"])
            if not directory.is_absolute():
                directory = ROOT / directory
            for frame_id in sequence["frame_ids"][19:]:
                target = cv2.imread(str(directory / f"{int(frame_id):05d}.png"), cv2.IMREAD_GRAYSCALE)
                require(target is not None and list(target.shape) == shape, f"Mask dimensions at {frame_id}")
                masks[offset] = target > 0
                foreground_counts.append(int(np.count_nonzero(masks[offset])))
                offset += 1
                if offset % 100 == 0:
                    write(run / "status.json", dict(phase="test_mask_cache", status="running", completed=offset,
                                                     total=2958, updated_at=stamp(), pid=os.getpid()))
        require(offset == 2958, "Incomplete mask target cache")
        masks.flush()
        del masks
        write(cache_path, dict(created_at=stamp(), frames=2958, radius_mm=8., stride=1,
                               sequences=projections, frame_ids=test["frame_ids"].tolist(),
                               foreground_counts=foreground_counts, target_file="mask_targets.npy"))
    else:
        cached = read(cache_path)
        require(cached["frame_ids"] == test["frame_ids"].tolist() and cached["frames"] == 2958, "Cached mask frames differ")
    mask_models = read(run / "protocol.json")["masks"]["models"]
    requested = [job for job in jobs if job["name"] in mask_models]
    require(len(requested) == 201, "Mask evaluation requires 201 unique main/ablation fits")
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=read(run / "protocol.json")["workers"],
            mp_context=multiprocessing.get_context("spawn"), initializer=mask_worker_init, initargs=(str(run),)) as pool:
        futures = {pool.submit(mask_job, job): job for job in requested}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            write(run / "status.json", dict(phase="test_masks", status="running", completed=len(results),
                                             total=len(requested), updated_at=stamp(), pid=os.getpid()))
            print(f"MASK {result['model']} seed={result['seed']} IoU={result['mask_iou']:.6f}", flush=True)
    lookup = {(result["model"], result["seed"]): result for result in results}
    rows, by_sequence = [], []
    for job in jobs:
        dest = evaluation_dir(run, job)
        row, groups = read(dest / "metrics.json"), read(dest / "sequence_metrics.json")
        if (job["name"], job["seed"]) in lookup:
            mask_result = lookup[(job["name"], job["seed"])]
            row.update({key: mask_result[key] for key in ("mask_iou", "mask_dice", "mask_frames")})
            for group in groups:
                matching = next(g for g in mask_result["by_sequence"] if g["group"] == group["group"])
                group.update(matching)
            write(dest / "metrics.json", row)
            write(dest / "sequence_metrics.json", groups)
        rows.append(row)
        by_sequence.extend(groups)
    write_csv(run / "raw_test.csv", rows)
    write_csv(run / "raw_test_by_sequence.csv", by_sequence)
    write(run / "mask_evaluation_complete.json", dict(completed_at=stamp(), unique_fits=201,
                                                      frames_per_fit=2958, radius_mm=8., metrics=["iou", "dice"]))


def holm(values):
    order = np.argsort(values)
    adjusted, running = np.empty(len(values)), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def statistics(run, rows):
    from scipy import stats
    lookup = {(r["model"], r["seed"]): r["mean_node_mm"] for r in rows}
    results, raw = [], []
    rng = np.random.default_rng(20260913)
    samples = rng.integers(0, len(SEEDS), size=(20000, len(SEEDS)))
    for family, reference, alternative in read(run / "protocol.json")["statistics"]["contrasts"]:
        alt_values = [lookup[(alternative, None if alternative in LINEARS else seed)] for seed in SEEDS]
        delta = np.asarray([lookup[(reference, seed)] - v for seed, v in zip(SEEDS, alt_values)])
        active = delta[delta != 0]
        ranks = np.rint(2 * stats.rankdata(np.abs(active), method="average")).astype(int)
        counts = np.zeros(int(ranks.sum()) + 1, dtype=np.int64)
        counts[0] = 1
        for rank in ranks:
            previous = counts.copy()
            counts[rank:] += previous[:-rank]
        observed = int(ranks[active > 0].sum())
        p = min(1., 2 * min(counts[:observed + 1].sum(), counts[observed:].sum()) / 2**len(active))
        sign_p = float(stats.binomtest(int((active > 0).sum()), len(active)).pvalue) if len(active) else 1.
        ci = np.quantile(delta[samples].mean(1), [.025, .975])
        results.append(dict(family=family, reference=reference, alternative=alternative,
                            n_pairs=20, nonzero_pairs=len(active), mean_reference_minus_alternative_mm=float(delta.mean()),
                            positive_pairs=int((delta > 0).sum()), wilcoxon_exact_p=float(p), sign_test_exact_p=sign_p,
                            bootstrap95_lower_mm=float(ci[0]), bootstrap95_upper_mm=float(ci[1]),
                            reference_fits=20, alternative_fits=1 if alternative in LINEARS else 20))
        for seed, value, alt_value in zip(SEEDS, delta, alt_values):
            raw.append(dict(family=family, reference=reference, alternative=alternative, seed=seed,
                            reference_mean_node_mm=lookup[(reference, seed)], alternative_mean_node_mm=alt_value,
                            alternative_seed=None if alternative in LINEARS else seed,
                            reference_minus_alternative_mm=float(value)))
    for family, size in dict(main=7, ablation=3, plugin=5).items():
        group = [row for row in results if row["family"] == family]
        require(len(group) == size, "Statistical family size mismatch")
        for key in ("wilcoxon_exact_p", "sign_test_exact_p"):
            for row, adjusted in zip(group, holm([r[key] for r in group])):
                row[key.replace("_exact_p", "_holm_p")] = adjusted
    write_csv(run / "raw_paired_differences.csv", raw)
    write(run / "paired_statistics.json", dict(metric="pooled mean_node_mm", holm_family_sizes=dict(main=7, ablation=3, plugin=5),
                                               scope="training seeds conditional on fixed selected temporal split", contrasts=results))


def pipeline(run, stop_after):
    with (run / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(run / "protocol.json")
        require((run / "PREPARED").exists(), "Run prepare first")
        require((run / "source/scripts/experiments" / SCRIPT.name).read_bytes() == SCRIPT.read_bytes(), "Frozen script mismatch")
        (run / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        write(run / "process.json", dict(pid=os.getpid(), process_group=os.getpgrp(), started_at=stamp(),
                                          command=sys.argv, log=str(run / "pipeline.log"), workers=protocol["workers"]))
        candidates = read(run / "candidate_configs.json")
        try:
            smoke = stage(run, "smoke", jobs_for("smoke", candidates), protocol["workers"])
            estimates = [r["model_initialization_and_prior_seconds"] + r.get("memory_initialization_seconds", 0.)
                         + r["training_seconds"] * 50 for r in smoke if r["model"] not in LINEARS]
            # Two epochs overstate steady-state overhead and omit most val I/O;
            # publish a planning estimate with an explicit uncertainty range.
            serial = sum(estimates)
            write(run / "cost_estimate.json", dict(measured_at=stamp(), smoke=smoke,
                  formal_estimated_serial_seconds=serial * 20, tuning_estimated_serial_seconds=serial * 2,
                  ideal_parallel_hours=serial * 22 / protocol["workers"] / 3600,
                  planning_hours_range=[serial * 22 / protocol["workers"] / 3600 * .7,
                                        serial * 22 / protocol["workers"] / 3600 * 2.],
                  basis="2 full-data epochs; prior fitting measured separately; shared CPU contention and validation add uncertainty"))
            if stop_after == "smoke":
                write(run / "status.json", dict(phase="smoke", status="complete", updated_at=stamp()))
                return
            stage(run, "tuning", jobs_for("tuning", candidates), protocol["workers"])
            formal = freeze_configs(run, candidates)
            if stop_after == "tuning":
                write(run / "status.json", dict(phase="tuning", status="complete", updated_at=stamp()))
                return
            stage(run, "formal", formal, protocol["workers"])
            training_gate(run, formal)
            if stop_after == "train":
                write(run / "status.json", dict(phase="train", status="complete", formal_fits=286, updated_at=stamp()))
                return
            runtime(run, protocol["threads"])
            evaluate(run, formal)
            write(run / "status.json", dict(phase="complete", status="complete", formal_fits=286, test_fits=286,
                                             updated_at=stamp(), pid=os.getpid()))
        except BaseException:
            write(run / "status.json", dict(phase="pipeline", status="failed", updated_at=stamp(),
                                             pid=os.getpid(), error=traceback.format_exc()))
            raise


def launch(run, stop_after):
    require((run / "PREPARED").exists(), "Run prepare first")
    # Check the OS lock, which is released even if a prior controller crashed.
    with (run / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)
    command = [sys.executable, "-B", str(SCRIPT), "pipeline", "--run", str(run), "--stop-after", stop_after]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    with (run / "pipeline.log").open("a", buffering=1) as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (run / "pipeline.pid").write_text(str(process.pid) + "\n")
    write(run / "launch.json", dict(pid=process.pid, command=command, launched_at=stamp(), log=str(run / "pipeline.log")))
    print(json.dumps(dict(pid=process.pid, run=str(run), log=str(run / "pipeline.log")), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "launch", "pipeline", "test", "status"])
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--stop-after", choices=["smoke", "tuning", "train", "test"], default="test")
    args = parser.parse_args()
    run = args.run.resolve()
    require(args.workers >= 1 and args.threads >= 1, "Positive CPU allocation required")
    if args.command == "prepare":
        prepare(run, args.study.resolve(), args.workers, args.threads)
        print(json.dumps(dict(run=str(run), status="prepared", formal_unique_fits=286)))
    elif args.command == "launch":
        launch(run, args.stop_after)
    elif args.command == "pipeline":
        pipeline(run, args.stop_after)
    elif args.command == "test":
        with (run / "pipeline.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            protocol = read(run / "protocol.json")
            jobs = read(run / "formal_plan.json")
            training_gate(run, jobs)
            runtime(run, protocol["threads"])
            evaluate(run, jobs)
    else:
        print(json.dumps(read(run / "status.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
