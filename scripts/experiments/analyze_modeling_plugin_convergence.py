#!/usr/bin/env python3
"""Train-only memory feature transfer and reconstruction of HOV epoch zero.

Writes only its own analysis directory and plugin_convergence.{json,md}.
The existing models, training runs and paper draft are read-only inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

# Keep the entire process within the four-thread allocation, including BLAS.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from scipy import stats

from src.benchmarks.modeling_data import write_json
from src.benchmarks.modeling_fast_training import cache_windows
from src.benchmarks.modeling_models import fit_normalization, make_model
from src.benchmarks.modeling_memory_initialization import initialize_memory_readout
from src.benchmarks.modeling_runner import _seed
from src.operators.static_drive import MonotoneSplineDrive
from src.operators.play_bank import PlayBank
from src.operators.maxwell_bank import MaxwellBank

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
STUDY = ROOT / "workspace/runs/training/modeling_three_seq_20260913_001928"
OUTPUT = ROOT / "workspace/runs/analysis/modeling_mechanisms_20260913_001/plugin"
REPORT = ROOT / "workspace/reports/modeling_mechanisms_20260913_001/plugin_convergence"
SEEDS = list(range(5))
VARIANTS = ["base", "path", "time", "both", "static_capacity", "window"]
NAMES = {"base": "基础", "path": "+路径记忆", "time": "+时间记忆",
         "both": "+双记忆", "static_capacity": "静态容量控制", "window": "完整窗口"}
MODEL_NAMES = {"hov": "HOV", "hov_no_play": "HOV 无路径记忆",
               "hov_no_maxwell": "HOV 无时间记忆", "hov_no_memory": "HOV 无记忆",
               "chen_direction": "Chen 方向网络", "bezier_gru": "Yu Bézier–GRU 适配",
               "mlp": "静态 MLP", "linear": "线性", "koopman": "Koopman",
               "oscillator": "Krauss 振子适配", "pcc": "PCC"}


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def load_roles(roles):
    """Load just requested label partitions; no image or mask inventory scans."""
    path = STUDY / "data/dataset_manifest.json"
    meta = read(path)
    assert meta["dt"] == .2 and meta["H"] == 20
    assert meta["length_unit"] == "mm" and meta["node_order"] == "base_to_tip"
    result = {}
    for role in roles:
        seq = []
        for record in meta["files"]:
            if record["role"] != role:
                continue
            source = Path(record["path"])
            if not source.is_absolute():
                source = path.parent / source
            with np.load(source, allow_pickle=False) as data:
                s = {key: data[key].copy() for key in data.files}
            s["record"] = record
            assert s["actions"].shape == (record["frames"], 4)
            assert s["positions"].shape == (record["frames"], 15, 3)
            assert np.isfinite(s["positions"]).all() and np.isfinite(s["actions"]).all()
            seq.append(s)
        assert len(seq) == 3
        x, y, groups = cache_windows(seq, 20, "cpu")
        result[role] = {"sequences": seq, "x": x, "y": y, "groups": groups}
    return meta, result


def metrics(pred, target):
    p = np.asarray(pred, dtype=np.float64).reshape(-1, 15, 3)
    y = np.asarray(target, dtype=np.float64).reshape(-1, 15, 3)
    distance = np.linalg.norm(p - y, axis=-1)
    assert np.isfinite(distance).all()
    return {"mean_node_mm": float(distance.mean()),
            "node_rmse_mm": float(np.sqrt(np.mean(distance ** 2))),
            "endpoint_mm": float(distance[:, -1].mean()),
            "endpoint_rmse_mm": float(np.sqrt(np.mean(distance[:, -1] ** 2)))}


def predict_geometry(model, windows, center, scale, reference=False):
    model.eval()
    with torch.inference_mode():
        pieces = [(model.core.decode_equilibrium(z[:, -1]) if reference else model(z))
                  * scale + center for z in windows.split(512)]
    return torch.cat(pieces).numpy()


def summarize_existing_history():
    curves, timing, summary = [], [], []
    for folder in sorted((STUDY / "formal").iterdir()):
        if not folder.is_dir():
            continue
        for seed in SEEDS:
            run = folder / f"seed_{seed}"
            manifest, history = read(run / "run_manifest.json"), read(run / "history.json")
            config = read(run / "resolved_config.json")
            assert manifest["status"] == "complete" and config["seed"] == seed
            previous_epoch = previous_time = 0
            for item in history:
                curves.append(dict(model=folder.name, model_label=MODEL_NAMES.get(folder.name, folder.name),
                                   seed=seed, **item))
                delta_epochs = item["epoch"] - previous_epoch
                timing.append(dict(model=folder.name, seed=seed,
                                   start_epoch=previous_epoch + 1, end_epoch=item["epoch"],
                                   mean_train_seconds_per_epoch=(item["cumulative_training_seconds"] - previous_time) / delta_epochs,
                                   logged_epoch_including_validation_seconds=item["epoch_seconds"]))
                previous_epoch, previous_time = item["epoch"], item["cumulative_training_seconds"]
            first, best = history[0]["validation_node_mean_mm"], manifest["best_validation_node_mean_mm"]
            threshold = 1.05 * best
            within = next((r for r in history if r["validation_node_mean_mm"] <= threshold), history[-1])
            init_seconds = manifest.get("model_initialization_and_prior_seconds", 0.)
            memory_seconds = manifest.get("memory_initialization_seconds", 0.)
            total_training = history[-1]["cumulative_training_seconds"]
            summary.append(dict(model=folder.name, model_label=MODEL_NAMES.get(folder.name, folder.name),
                                seed=seed, epoch1_val_node_mm=first, best_val_node_mm=best,
                                epoch1_to_best_improvement_mm=first-best,
                                epoch1_to_best_improvement_pct=100*(first-best)/first,
                                best_epoch=manifest["best_epoch"], logged_checks=len(history),
                                first_recorded_epoch_within_5pct_best=within["epoch"],
                                first_recorded_wall_seconds_within_5pct_best=within["elapsed_seconds"],
                                model_initialization_and_prior_seconds=init_seconds,
                                memory_initialization_seconds=memory_seconds,
                                official_training_seconds=total_training,
                                mean_training_seconds_per_epoch=total_training/history[-1]["epoch"],
                                wall_seconds=manifest["wall_seconds"],
                                other_wall_seconds=manifest["wall_seconds"]-init_seconds-memory_seconds-total_training,
                                parameter_count_including_fitted_reference=manifest.get("parameter_count_including_fitted_reference"),
                                supervised_windows=manifest["supervised_windows"],
                                source=str(run.relative_to(ROOT))))
    return {"curves": curves, "timing_intervals": timing, "summary": summary}


def convergence(trainval):
    """Reconstruct the genuine pre-epoch model from train, not its best weights."""
    out = OUTPUT / "epoch0"
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for relative in ["src/models/model_ishsm.py", "src/models/model_hereditary_geometry.py",
                     "src/models/model_hereditary_operator.py", "src/benchmarks/modeling_models.py",
                     "src/benchmarks/modeling_memory_initialization.py",
                     "src/benchmarks/modeling_geometry_calibration.py",
                     "src/operators/static_drive.py", "src/operators/play_bank.py", "src/operators/maxwell_bank.py"]:
        archived = STUDY / "formal/hov/seed_0/source" / relative
        equal = archived.read_bytes() == (ROOT / relative).read_bytes()
        checks.append(dict(file=relative, matches_formal_source=equal))
        if not equal:
            raise RuntimeError(f"Formal source differs: {relative}")
    existing = summarize_existing_history()
    train, val = trainval["train"], trainval["val"]
    center_np, scale = fit_normalization(train["sequences"])
    center = torch.tensor(center_np)
    stages, reconstructions = [], []
    for seed in SEEDS:
        run = STUDY / "formal/hov" / f"seed_{seed}"
        cfg = read(run / "resolved_config.json")
        _seed(seed)
        start = time.perf_counter()
        model, geometry = make_model("hov", cfg, train["sequences"], (center_np, scale))
        prior_seconds = time.perf_counter() - start
        checkpoint = torch.load(run / "best_eval_model.pt", map_location="cpu", weights_only=False)
        differences = {}
        for key, value in geometry.items():
            other = checkpoint["geometry_config"][key]
            if isinstance(value, (list, tuple, float, int)) and not isinstance(value, bool):
                a, b = np.asarray(value, dtype=float), np.asarray(other, dtype=float)
                differences[key] = float(np.max(np.abs(a-b))) if a.size else 0.
            else:
                if value != other:
                    raise AssertionError(f"Reconstruction mismatch: {key}")
        max_difference = max(differences.values())
        assert max_difference < 1e-5, (seed, differences)
        seedpath = out / f"seed_{seed}"
        seedpath.mkdir(exist_ok=True)
        for stage, reference in [("prefitted_reference", True), ("prefit_random_memory", False)]:
            pred = predict_geometry(model, val["x"], center, scale, reference=reference)
            stages.append(dict(seed=seed, stage=stage, **metrics(pred, val["y"].numpy())))
        t = time.perf_counter()
        init = initialize_memory_readout(model, train["x"], train["y"], ridge=cfg["memory_ridge"])
        memory_seconds = time.perf_counter()-t
        pred = predict_geometry(model, val["x"], center, scale)
        measured = metrics(pred, val["y"].numpy())
        stages.append(dict(seed=seed, stage="epoch0_after_memory_initialization", **measured))
        torch.save(dict(model="hov", seed=seed, state_dict=model.state_dict(), config=cfg,
                        geometry_config=geometry, center=center_np.tolist(), scale=scale,
                        initialization=init, validation_metrics=measured,
                        stage="after train-only priors and memory ridge; before formal Adam"),
                   seedpath / "epoch0_model.pt")
        np.savez_compressed(seedpath / "epoch0_val_predictions.npz", prediction=pred,
                            target=val["y"].numpy(), groups=val["groups"])
        formal = next(r for r in existing["summary"] if r["model"] == "hov" and r["seed"] == seed)
        item = dict(seed=seed, epoch0_val_node_mm=measured["mean_node_mm"],
                    epoch1_val_node_mm=formal["epoch1_val_node_mm"], best_val_node_mm=formal["best_val_node_mm"],
                    epoch0_to_best_improvement_pct=100*(measured["mean_node_mm"]-formal["best_val_node_mm"])/measured["mean_node_mm"],
                    reconstructed_prior_seconds_cpu=prior_seconds,
                    reconstructed_memory_initialization_seconds_cpu=memory_seconds,
                    prior_configuration_max_abs_difference=max_difference,
                    prior_step_setting=cfg["prior_steps"], coordinate_adam_steps=cfg["prior_steps"],
                    geometry_adam_steps=max(cfg["prior_steps"]//2, 1),
                    total_prefit_adam_steps=cfg["prior_steps"]+max(cfg["prior_steps"]//2, 1),
                    prior_fullbatch_training_frames=len(train["x"]),
                    initialization=init, formal_manifest=str(run / "run_manifest.json"),
                    checkpoint=str(seedpath / "epoch0_model.pt"))
        reconstructions.append(item)
        write_json(seedpath / "reconstruction.json", item)
        print(f"epoch0 seed={seed}: {measured['mean_node_mm']:.5f} mm; epoch1={formal['epoch1_val_node_mm']:.5f}; best={formal['best_val_node_mm']:.5f}", flush=True)
    result = dict(existing, epoch0_stages=stages, epoch0=reconstructions, source_checks=checks,
                  historical_timing_device="Original GPU jobs with two concurrent workers per GPU; prefit is CPU",
                  reconstruction_device="CPU, two Torch/BLAS threads; timings distinct from original runs")
    write_json(OUTPUT / "convergence.json", result)
    return result


def feature_bank(x):
    """Portable, fixed HOV operators, with equilibrium initialization per window."""
    drive = MonotoneSplineDrive(4, output_normalization="unit_range")
    play = PlayBank(4, 2, (.02, .5))
    temporal = MaxwellBank(4, 6, .2, (.6, 2.))
    with torch.inference_mode():
        e = drive(x)
        p = e[:, 0, :, None].repeat(1, 1, 2)
        h = e[:, 0, :, None].repeat(1, 1, 6)
        for step in range(1, x.shape[1]):
            p, q = play.step(p, e[:, step])
            h = temporal.step(h, e[:, step])
        current = x[:, -1]
        q = q.flatten(1)
        d = (h-e[:, -1, :, None]).flatten(1)
        # Same 32 extra scalars as both memory branches, but only current input.
        static = torch.cat([current**2, current**3] +
                           [f(math.pi*k*current) for k in (1, 2, 3) for f in (torch.sin, torch.cos)], 1)
        result = {"base": current, "path": torch.cat([current, q], 1),
                  "time": torch.cat([current, d], 1), "both": torch.cat([current, q, d], 1),
                  "static_capacity": torch.cat([current, static], 1), "window": x.flatten(1)}
    assert result["both"].shape[1] == result["static_capacity"].shape[1] == 36
    # Verify fixed recurrences with a direct implementation for a small subset.
    ep = e[:8].numpy()
    pn = np.repeat(ep[:, 0, :, None], 2, axis=-1)
    hn = np.repeat(ep[:, 0, :, None], 6, axis=-1)
    for t in range(1, ep.shape[1]):
        pn = np.clip(pn, ep[:, t, :, None]-play.thresholds.numpy(), ep[:, t, :, None]+play.thresholds.numpy())
        hn = temporal.decays.numpy()*hn+(1-temporal.decays.numpy())*ep[:, t, :, None]
    assert np.allclose(q[:8].numpy(), (ep[:, -1, :, None]-pn).reshape(8, -1), atol=1e-6)
    assert np.allclose(d[:8].numpy(), (hn-ep[:, -1, :, None]).reshape(8, -1), atol=1e-6)
    return {k: v.numpy().copy() for k, v in result.items()}


class FeatureMLP(nn.Module):
    def __init__(self, inputs, width):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 45))

    def forward(self, x):
        return self.network(x).reshape(-1, 15, 3)


def fit_linear(x, target, alpha):
    # Ridge is normalized by number of windows. Intercept is not penalized.
    augmented = np.c_[x.astype(np.float64), np.ones(len(x))]
    y = target.reshape(len(target), -1).astype(np.float64)
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0.
    coef = np.linalg.solve(augmented.T @ augmented / len(x) + penalty,
                           augmented.T @ y / len(x))
    return coef


def fit_mlp(x, y, vx, vy, scale, center, config, seed, destination):
    destination.mkdir(parents=True, exist_ok=True)
    _seed(seed)
    model = FeatureMLP(x.shape[1], config["width"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=3, min_lr=1e-5)
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    tvx = torch.from_numpy(vx)
    center_t = torch.tensor(center)
    history, best, best_state, best_epoch = [], float("inf"), None, 0
    train_seconds = 0.
    start = time.perf_counter()
    for epoch in range(1, config["epochs"]+1):
        model.train()
        t = time.perf_counter()
        order = torch.randperm(len(tx))
        losses = []
        for batch in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(tx[batch])
            loss = (prediction-ty[batch]).square().mean()+.25*(prediction[:, -1]-ty[batch, -1]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            losses.append(float(loss.detach())*len(batch))
        train_seconds += time.perf_counter()-t
        if epoch != 1 and epoch % 5 and epoch != config["epochs"]:
            continue
        model.eval()
        with torch.inference_mode():
            prediction = torch.cat([model(z)*scale+center_t for z in tvx.split(1024)]).numpy()
        score = metrics(prediction, vy)["mean_node_mm"]
        if score < best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch, val_mean_node_mm=score, best_val_mean_node_mm=best,
                            train_loss=sum(losses)/len(tx), cumulative_training_seconds=train_seconds,
                            elapsed_seconds=time.perf_counter()-start, lr=optimizer.param_groups[0]["lr"]))
        scheduler.step(score)
    seconds = time.perf_counter()-start
    model.load_state_dict(best_state)
    torch.save(dict(state_dict=best_state, config=config, seed=seed, input_dim=x.shape[1],
                    best_epoch=best_epoch, val_mean_node_mm=best), destination / "model.pt")
    result = dict(seed=seed, config=config, input_dim=x.shape[1],
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  best_epoch=best_epoch, val_mean_node_mm=best, fit_seconds=seconds,
                  optimizer_seconds=train_seconds, checkpoint=str(destination / "model.pt"))
    write_json(destination / "history.json", history)
    write_json(destination / "fit.json", result)
    return result


def plugins(trainval, epochs):
    started = stamp()
    train, val = trainval["train"], trainval["val"]
    t = time.perf_counter()
    raw = {role: feature_bank(trainval[role]["x"]) for role in ("train", "val")}
    feature_seconds = time.perf_counter()-t
    norm, features = {}, {"train": {}, "val": {}}
    for variant in VARIANTS:
        mean = raw["train"][variant].mean(0)
        std = raw["train"][variant].std(0)
        std = np.where(std < 1e-6, 1., std)
        norm[variant] = dict(mean=mean.tolist(), std=std.tolist())
        for role in features:
            features[role][variant] = ((raw[role][variant]-mean)/std).astype(np.float32)
    center, scale = fit_normalization(train["sequences"])
    y = (train["y"].numpy()-center)/scale
    vy = val["y"].numpy()
    normalization = dict(features=norm, target_center_xyz=center.tolist(), target_scale=scale,
                         role="train only", feature_build_seconds_train_val=feature_seconds)
    write_json(OUTPUT / "normalization.json", normalization)
    # A single screening seed is declared in advance; every family/variant gets
    # the same complete grid and epoch budget. All five seeds are retrained.
    linear_grid = [1e-6, 1e-4, 1e-2, 1.]
    mlp_grid = [dict(width=w, lr=lr, epochs=epochs, batch_size=512)
                for w in (64, 128) for lr in (.001, .003)]
    protocol = dict(started_at=started, device="cpu", torch_threads=2, blas_threads=2,
                    seeds=SEEDS, screening_seed=0, linear_ridge_grid=linear_grid,
                    mlp_grid=mlp_grid, variants=VARIANTS, history=20, dt=.2,
                    split="formal fixed three-sequence temporal 6:2:2; split-local H20; pooled frames",
                    selection="lowest val pooled mean node distance; identical grid and budget per variant; freeze before opening test labels",
                    labels="same 15-node visual skeletons as formal run, mm",
                    initialization="linear direct ridge; MLP PyTorch default random initialization; fixed operator drive and time/threshold grids; train-only standardization",
                    operator_drive="fixed equal-weight MonotoneSplineDrive (5 hinges, unit_range); no transferred HOV fitted weights",
                    plugin="concatenate current u with q=e-p (8), h-e (24), or both (32); fit ordinary output readout/MLP",
                    state_initialization="independent H20 window, p0=h0=e(u0), update remaining 19 commands",
                    recurrence=dict(path="p_t=clamp(p_(t-1), e_t-r, e_t+r); q_t=e_t-p_t",
                                    time="h_t=exp(-dt/tau) h_(t-1)+(1-exp(-dt/tau)) e_t; d_t=h_t-e_t",
                                    thresholds=[.02, .5], taus=np.geomspace(.6, 2., 6).tolist()),
                    static_capacity="32 additional current-input functions: u^2,u^3, sin(k*pi*u),cos(k*pi*u), k=1,2,3; same input dimension as dual memory",
                    window="flatten the same 20x4 causal command window",
                    endpoint_weight=.25, target_dimensions=45,
                    secondary_analysis=True)
    write_json(OUTPUT / "protocol.json", protocol)
    screens, frozen = [], {}
    for family in ("linear", "mlp"):
        for variant in VARIANTS:
            x, vx = features["train"][variant], features["val"][variant]
            candidates = []
            grid = linear_grid if family == "linear" else mlp_grid
            for candidate, hyper in enumerate(grid):
                dest = OUTPUT / "screening" / family / variant / f"candidate_{candidate}"
                if family == "linear":
                    t = time.perf_counter()
                    coef = fit_linear(x, y, hyper)
                    pred = (np.c_[vx, np.ones(len(vx))] @ coef).reshape(-1, 15, 3)*scale+center
                    score = metrics(pred, vy)["mean_node_mm"]
                    result = dict(seed=0, config=dict(ridge=hyper), input_dim=x.shape[1],
                                  parameter_count=int(coef.size), best_epoch=0,
                                  val_mean_node_mm=score, fit_seconds=time.perf_counter()-t)
                    dest.mkdir(parents=True, exist_ok=True)
                    write_json(dest / "fit.json", result)
                else:
                    result = fit_mlp(x, y, vx, vy, scale, center, hyper, 0, dest)
                row = dict(family=family, variant=variant, candidate=candidate, **result)
                screens.append(row)
                candidates.append(row)
                print(f"screen {family}/{variant}/{candidate}: val={result['val_mean_node_mm']:.5f}, seconds={result['fit_seconds']:.1f}", flush=True)
                write_json(OUTPUT / "screening_results.json", screens)
            selected = min(candidates, key=lambda r: (r["val_mean_node_mm"], r["candidate"]))
            frozen[f"{family}/{variant}"] = dict(config=selected["config"], candidate=selected["candidate"],
                                                  selection_val_mean_node_mm=selected["val_mean_node_mm"])
    # Test data have not been loaded at this point.
    freeze = dict(frozen_at=stamp(), selections=frozen, seeds=SEEDS,
                  test_opened=False, label_access_so_far=["train", "val"], protocol=protocol)
    write_json(OUTPUT / "frozen_plugin_configs.json", freeze)
    fitted = []
    for family in ("linear", "mlp"):
        for variant in VARIANTS:
            x, vx = features["train"][variant], features["val"][variant]
            config = frozen[f"{family}/{variant}"]["config"]
            for seed in SEEDS:
                dest = OUTPUT / "formal" / family / variant / f"seed_{seed}"
                if family == "linear":
                    _seed(seed)
                    dest.mkdir(parents=True, exist_ok=True)
                    t = time.perf_counter()
                    coef = fit_linear(x, y, config["ridge"])
                    seconds = time.perf_counter()-t
                    pred = (np.c_[vx, np.ones(len(vx))]@coef).reshape(-1, 15, 3)*scale+center
                    np.savez_compressed(dest / "model.npz", coefficients=coef)
                    result = dict(seed=seed, config=config, input_dim=x.shape[1], parameter_count=int(coef.size),
                                  best_epoch=0, val_mean_node_mm=metrics(pred, vy)["mean_node_mm"],
                                  fit_seconds=seconds, optimizer_seconds=0., checkpoint=str(dest / "model.npz"))
                    write_json(dest / "fit.json", result)
                else:
                    result = fit_mlp(x, y, vx, vy, scale, center, config, seed, dest)
                fitted.append(dict(family=family, variant=variant, label=f"{family.upper()} {NAMES[variant]}", **result))
                print(f"fit {family}/{variant}/seed{seed}: val={result['val_mean_node_mm']:.5f}", flush=True)
                write_json(OUTPUT / "fitted_runs.json", fitted)
    write_json(OUTPUT / "test_access.json", dict(first_test_label_load=stamp(), freeze_file=str(OUTPUT / "frozen_plugin_configs.json"),
                                                 formal_fits_completed=len(fitted), config_updates_after_test=0))
    _, loaded = load_roles(("test",))
    test = loaded["test"]
    test_raw = feature_bank(test["x"])
    results, curves = [], []
    for row in fitted:
        family, variant = row["family"], row["variant"]
        z = ((test_raw[variant]-np.asarray(norm[variant]["mean"]))/np.asarray(norm[variant]["std"])).astype(np.float32)
        dest = Path(row["checkpoint"]).parent
        if family == "linear":
            coef = np.load(row["checkpoint"])["coefficients"]
            pred = (np.c_[z, np.ones(len(z))]@coef).reshape(-1, 15, 3)*scale+center
        else:
            ck = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
            model = FeatureMLP(z.shape[1], ck["config"]["width"])
            model.load_state_dict(ck["state_dict"])
            model.eval()
            with torch.inference_mode():
                pred = torch.cat([model(a)*scale+torch.tensor(center) for a in torch.from_numpy(z).split(1024)]).numpy()
            for point in read(dest / "history.json"):
                curves.append(dict(family=family, variant=variant, seed=row["seed"], label=row["label"], **point))
        measured = metrics(pred, test["y"].numpy())
        per_sequence = []
        for i, sequence in enumerate(test["sequences"]):
            ix = test["groups"] == i
            per_sequence.append(dict(group=sequence["record"]["group"], n_frames=int(ix.sum()),
                                     **metrics(pred[ix], test["y"].numpy()[ix])))
        result = dict(row, **measured, test_windows=len(pred), per_sequence=per_sequence,
                      feature_constants_learned=0, feature_standardization_scalars=2*z.shape[1],
                      stochastic_repetitions=family == "mlp")
        distance = np.linalg.norm(pred.astype(np.float64)-test["y"].numpy(), axis=-1)
        np.savez_compressed(dest / "test_predictions.npz", prediction=pred, target=test["y"].numpy(),
                            groups=test["groups"], node_distance_mm=distance)
        write_json(dest / "test_metrics.json", result)
        results.append(result)
        print(f"test {family}/{variant}/seed{row['seed']}: {measured['mean_node_mm']:.5f} mm", flush=True)
    output = dict(protocol=protocol, frozen=freeze, screening=screens, runs=results, curves=curves,
                  normalization=normalization, train_windows=len(train["x"]), val_windows=len(val["x"]), test_windows=len(test["x"]))
    write_json(OUTPUT / "plugin_results.json", output)
    return output


def holm(values):
    order = np.argsort(values)
    corrected = np.empty(len(values))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values)-rank)*values[index])
        corrected[index] = min(1., running)
    return corrected.tolist()


def summarize_plugin(result):
    summaries, contrasts = [], []
    for family in ("linear", "mlp"):
        base = sorted([r for r in result["runs"] if r["family"] == family and r["variant"] == "base"], key=lambda r:r["seed"])
        for variant in VARIANTS:
            group = sorted([r for r in result["runs"] if r["family"] == family and r["variant"] == variant], key=lambda r:r["seed"])
            assert [r["seed"] for r in group] == SEEDS
            row = dict(family=family, variant=variant, label=group[0]["label"], seeds=SEEDS,
                       independent_stochastic_repetitions=5 if family == "mlp" else 0,
                       parameter_count=group[0]["parameter_count"], input_dim=group[0]["input_dim"],
                       config=group[0]["config"])
            for metric in ("mean_node_mm", "node_rmse_mm", "endpoint_mm", "fit_seconds", "val_mean_node_mm"):
                values = [r[metric] for r in group]
                row[metric+"_mean"] = float(np.mean(values))
                row[metric+"_sd"] = float(np.std(values, ddof=1))
            summaries.append(row)
            if variant != "base":
                delta = np.array([a["mean_node_mm"]-b["mean_node_mm"] for a,b in zip(base, group)])
                p = float(stats.wilcoxon(delta, alternative="two-sided", method="exact").pvalue) if family == "mlp" else None
                contrasts.append(dict(family=family, variant=variant, contrast=f"{family}/{variant} vs {family}/base",
                                      improvement_mm=float(delta.mean()), improvement_by_seed_mm=delta.tolist(),
                                      improvement_pct=100*float(delta.mean())/np.mean([r["mean_node_mm"] for r in base]),
                                      better_seeds=int((delta>0).sum()), n_seeds=5,
                                      wilcoxon_two_sided_p=p,
                                      inferential_status="paired optimization seeds; fixed dataset" if family == "mlp" else "deterministic fit, repeated seed labels are not independent evidence"))
    stochastic = [r for r in contrasts if r["family"] == "mlp"]
    for row, corrected in zip(stochastic, holm([r["wilcoxon_two_sided_p"] for r in stochastic])):
        row["wilcoxon_holm_p"] = corrected
    for row in contrasts:
        row.setdefault("wilcoxon_holm_p", None)
    return summaries, contrasts


def report(convergence_result, plugin_result):
    summary, contrasts = summarize_plugin(plugin_result)
    conv = convergence_result
    hov = [r for r in conv["summary"] if r["model"] == "hov"]
    mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
    epoch0 = mean(conv["epoch0"], "epoch0_val_node_mm")
    epoch1 = mean(hov, "epoch1_val_node_mm")
    best = mean(hov, "best_val_node_mm")
    before = mean([r for r in conv["epoch0_stages"] if r["stage"] == "prefitted_reference"], "mean_node_mm")
    dual = {r["family"]:r for r in contrasts if r["variant"] == "both"}
    summaries_by_key = {(r["family"], r["variant"]): r for r in summary}
    findings = [
        dict(id="prefit", title="首个正式 epoch 前已完成监督学习", status="verified",
             text=f"prior_steps=500 对应 500 步坐标拟合加 250 步几何微调，共 750 步全批量 Adam；随后在 8,988 个训练窗口上以 32 个特征拟合 16 维几何记忆读出。原日志平均参考初始化 {mean(hov, 'model_initialization_and_prior_seconds'):.3f} s，记忆初始化 {mean(hov, 'memory_initialization_seconds'):.3f} s。"),
        dict(id="epoch_zero", title="重建 epoch 0 并分离初始化收益", status="verified_with_cpu_reconstruction",
             text=f"五个固定 seed 的验证节点误差：预拟合静态参考 {before:.4f} mm，记忆初始化后的 epoch 0 为 {epoch0:.4f} mm，epoch 1 为 {epoch1:.4f} mm，最佳 checkpoint 为 {best:.4f} mm。epoch 1 到最佳平均相对改善 {mean(hov,'epoch1_to_best_improvement_pct'):.2f}%。"),
        dict(id="online", title="早期达到较低误差支持初始化有效，尚不构成在线学习证据", status="interpretation",
             text="该结果来自完整训练集上的离线预拟合、岭回归和正式优化。在线学习还需流式数据、每次更新时间、遗忘、漂移适应和闭环稳定性实验。epoch 1 也不能单独判断容量饱和。"),
        dict(id="linear_transfer", title="记忆特征在线性基础模型上的迁移", status="measured",
             text=f"双记忆相对基础线性测试节点误差变化为减少 {dual['linear']['improvement_mm']:.4f} mm（{dual['linear']['improvement_pct']:.2f}%）。线性解为确定性结果，五个 seed 不提供五次独立统计证据。"),
        dict(id="mlp_transfer", title="记忆特征在 MLP 基础模型上的迁移", status="measured",
             text=f"双记忆相对基础 MLP 测试节点误差平均减少 {dual['mlp']['improvement_mm']:.4f} mm（{dual['mlp']['improvement_pct']:.2f}%），{dual['mlp']['better_seeds']}/5 个 seed 改善；双侧精确 Wilcoxon p={dual['mlp']['wilcoxon_two_sided_p']:.4f}，Holm 后 p={dual['mlp']['wilcoxon_holm_p']:.4f}。"),
    ]
    for family in ("linear", "mlp"):
        both_row = summaries_by_key[family, "both"]
        static_row = summaries_by_key[family, "static_capacity"]
        window_row = summaries_by_key[family, "window"]
        control_interpretation = ("完整窗口的误差更低；双记忆以更紧凑输入保留了有效历史信息。"
                                  if window_row["mean_node_mm_mean"] < both_row["mean_node_mm_mean"] else
                                  "在本线性读出中，固定递推特征比展开窗口取得更低误差。")
        findings.append(dict(id=f"{family}_controls", title=f"{family.upper()} 的容量与窗口控制", status="measured",
                             text=f"双记忆测试节点误差 {both_row['mean_node_mm_mean']:.4f} mm、{both_row['parameter_count']} 参数；静态容量控制 {static_row['mean_node_mm_mean']:.4f} mm、{static_row['parameter_count']} 参数；完整窗口 {window_row['mean_node_mm_mean']:.4f} mm、{window_row['parameter_count']} 参数。双记忆使用36维输入（当前4维+记忆32维），窗口使用80维输入。{control_interpretation}"))
    definitions = dict(
        node_mean="先计算每帧15个节点的三维欧氏距离（标签为平面骨架），再对全部测试帧和节点平均；单位 mm。",
        node_rmse="sqrt(mean_{frame,node}(||prediction-label||_2^2))，单位 mm。",
        endpoint="每帧末端欧氏距离，再对所有帧平均；单位 mm。",
        repetitions="固定 split，固定 seeds 0..4；MLP seed 改变初始化与批次顺序，线性求解没有随机自由度。统计单位为优化 seed，不是帧或独立机器人。",
        uncertainty="均值±样本标准差（ddof=1）；同一时间序列的帧相关，未把帧视为独立重复。5 seed 双侧精确 Wilcoxon 最小 p=.0625。",
        tests="主指标节点平均误差，MLP五个非基础变体分别与基础比较，配对双侧精确 Wilcoxon；五项比较内 Holm 校正；线性不计算显著性。",
        prior_steps="500 坐标优化 + floor(500/2)=250 几何优化；每一步均使用8,988训练窗口对应的当前帧。此前还有线性岭回归热启动。",
        epoch0="训练集参考预拟合与记忆读出岭回归之后、正式 Adam minibatch 之前。不是未经训练的模型。",
        epoch_time="原 history 累计 optimizer 时间差 / epoch 数给出区间每epoch训练均值；epoch_seconds仅记录有验证的epoch，含验证/保存，不能解释为所有epoch的纯训练时间。",
        timer_scope="正式日志来自GPU并发训练；重建和插件来自CPU双线程，跨设备耗时不作直接速度排名。插件线性fit_seconds为求解时间，MLP为优化及验证/最佳权重恢复时间；特征构建、标准化、测试和文件写入时间另计。",
        plugin_scope="迁移的是固定递推特征的方程/结构，未迁移HOV的已训练权重，未使用HOV的几何解码器；输入为[u,q,h-e]，线性/MLP直接预测骨架坐标。",
        time_memory="分支统称时间记忆；h为时间记忆状态，读出为h-e。",
        standardization="各输入列均值/标准差只用train计算，输出使用原正式fit_normalization的train全帧中心和单一尺度；val/test复用固定统计量。",
        limitations=["已有三序列为此前挑选的子集，本分析是固定数据上的探索性机制补充，无法评估对未知机器人或独立序列的泛化。",
                     "插件为小网格验证集选型，不能声称各基础网络已到达其全局最佳性能。",
                     "路径与时间特征具有相关性；预测改善说明历史表征有用，不独立证明材料物理机制已被辨识。",
                     "静态容量控制具有相同输入维数；若验证选到不同宽度，总参数量仍可能不同，需结合表中配置判断。",
                     "H20覆盖19个采样间隔，即3.8s，并在窗口首命令平衡初始化；长于窗口的真实历史未被观察。",
                     "使用视觉中心线标签及固定像素毫米比例，不是独立外部三维测量；此分析未计算mask指标。"])
    stages = list(conv["epoch0_stages"])
    for r in hov:
        stages.extend([dict(seed=r["seed"], stage="epoch1", mean_node_mm=r["epoch1_val_node_mm"]),
                       dict(seed=r["seed"], stage="best_epoch", mean_node_mm=r["best_val_node_mm"])])
    timing_rows = []
    for r in conv["summary"]:
        for phase, key in [("参考/模型初始化", "model_initialization_and_prior_seconds"),
                           ("记忆读出初始化", "memory_initialization_seconds"),
                           ("正式优化", "official_training_seconds"), ("其他I/O与验证", "other_wall_seconds")]:
            timing_rows.append(dict(model=r["model"], model_label=r["model_label"], seed=r["seed"], phase=phase, seconds=r[key]))
    curves = [dict(model=r["model"], seed=r["seed"], epoch=r["epoch"],
                   series=f"{r['model_label']} seed {r['seed']}",
                   model_label=r["model_label"], val_node_mm=r["validation_node_mean_mm"],
                   wall_seconds=r["elapsed_seconds"], training_seconds=r["cumulative_training_seconds"]) for r in conv["curves"]]
    # Scalar-only chart rows are directly consumable by the canonical report
    # helper. Explicit seed series keep separate runs from being joined.
    plugin_chart_rows = [{key:r[key] for key in ("family", "variant", "seed", "label", "parameter_count",
                                                 "mean_node_mm", "node_rmse_mm", "endpoint_mm", "fit_seconds")}
                         for r in plugin_result["runs"]]
    plugin_curve_rows = [dict(r, series=f"{r['label']} seed {r['seed']}") for r in plugin_result["curves"]]
    metric_rows = []
    for r in plugin_result["runs"]:
        for metric in ("mean_node_mm", "node_rmse_mm", "endpoint_mm"):
            metric_rows.append(dict(family=r["family"], variant=r["variant"], label=r["label"], seed=r["seed"], metric=metric, value=r[metric]))
    charts = [
        dict(id="convergence_epoch", title="正式训练验证曲线（五个seed全量）", kind="line", x="epoch", y="val_node_mm", color="series", unit="mm", rows=curves),
        dict(id="convergence_wall", title="包含预拟合的原训练墙钟时间", kind="line", x="wall_seconds", y="val_node_mm", color="series", unit="mm", rows=curves),
        dict(id="epoch0_stages", title="HOV预拟合、记忆初始化与正式优化", kind="bar", x="stage", y="mean_node_mm", color="seed", unit="mm", rows=stages),
        dict(id="training_time_components", title="原训练时间分解", kind="bar", x="model_label", y="seconds", color="phase", unit="s", rows=timing_rows),
        dict(id="per_epoch_timing", title="每epoch纯训练耗时（日志区间平均）", kind="line", x="end_epoch", y="mean_train_seconds_per_epoch", color="model", unit="s", rows=conv["timing_intervals"]),
        dict(id="plugin_node", title="基础模型与记忆插件：逐seed测试节点误差", kind="bar", x="label", y="mean_node_mm", color="seed", unit="mm", rows=plugin_chart_rows),
        dict(id="plugin_endpoint", title="基础模型与记忆插件：逐seed末端误差", kind="bar", x="label", y="endpoint_mm", color="seed", unit="mm", rows=plugin_chart_rows),
        dict(id="plugin_capacity", title="插件参数量与测试误差", kind="scatter", x="parameter_count", y="mean_node_mm", color="label", unit="mm", rows=plugin_chart_rows),
        dict(id="plugin_fit_time", title="CPU拟合耗时（线性求解/MLP优化与验证）", kind="bar", x="label", y="fit_seconds", color="seed", unit="s", rows=plugin_chart_rows),
        dict(id="plugin_curves", title="插件MLP逐seed验证学习曲线", kind="line", x="epoch", y="val_mean_node_mm", color="series", unit="mm", rows=plugin_curve_rows),
        dict(id="plugin_improvement", title="插件相对基础模型的测试改善", kind="bar", x="contrast", y="improvement_mm", color="family", unit="mm", rows=contrasts),
    ]
    for chart in charts:
        chart["rows"] = [{key:value for key,value in row.items()
                           if value is None or isinstance(value, (str, float, int, bool))}
                          for row in chart["rows"]]
    artifact = dict(schema="modeling_plugin_convergence_v1", generated_at=stamp(), findings=findings,
                    definitions=definitions, charts=charts,
                    tables=[dict(id="original_convergence", title="原正式训练逐seed收敛汇总", rows=conv["summary"]),
                            dict(id="epoch0", title="HOV阶段0重建", rows=conv["epoch0"]),
                            dict(id="plugin_summary", title="插件测试汇总（均值和样本SD）", rows=summary),
                            dict(id="plugin_seeds", title="插件逐seed完整结果", rows=plugin_result["runs"]),
                            dict(id="plugin_contrasts", title="插件相对基础模型统计", rows=contrasts),
                            dict(id="validation_screen", title="验证集完整网格", rows=plugin_result["screening"])],
                    provenance=dict(source_study=str(STUDY), dataset_manifest=str(STUDY / "data/dataset_manifest.json"),
                                    analysis_script=str(Path(__file__).resolve()), output=str(OUTPUT),
                                    frozen_configs=str(OUTPUT / "frozen_plugin_configs.json"),
                                    protocol=plugin_result["protocol"], source_checks=conv["source_checks"],
                                    epoch0_reconstruction="Current implementation byte-equal to archived formal source; priors refit on train and compared with saved prior configuration; weights before formal training saved separately.",
                                    mask_hashes_scanned=False, training_model_files_changed=False),
                    long_tables=dict(plugin_metrics=metric_rows), validation_status="Share with caveats")
    write_json(REPORT.with_suffix(".json"), artifact)
    lines = ["# 模型收敛与记忆插件分析", "", f"生成时间：{stamp()}。全部正式结果使用固定 seeds 0–4。", ""]
    for f in findings:
        lines.extend([f"## {f['title']}", "", f["text"], ""])
    lines.extend(["## 插件逐方法汇总", "", "| 基础/插件 | 节点误差 mm | 末端误差 mm | 参数量 | CPU拟合 s |", "|---|---:|---:|---:|---:|"])
    for r in summary:
        lines.append(f"| {r['label']} | {r['mean_node_mm_mean']:.4f} ± {r['mean_node_mm_sd']:.4f} | {r['endpoint_mm_mean']:.4f} ± {r['endpoint_mm_sd']:.4f} | {r['parameter_count']} | {r['fit_seconds_mean']:.3f} |")
    lines.extend(["", "## 实验结构与协议", "", plugin_result["protocol"]["plugin"], "",
                  "1. 从原三序列固定6:2:2分割读取train、val；沿用5Hz、H20及相同中心线标签。",
                  "2. 使用原play与时间递推方程构建固定记忆特征，仅用训练集估计标准化参数。",
                  "3. 线性岭回归4个正则系数；MLP宽度64/128、学习率0.001/0.003，所有变体同网格、同预算。",
                  f"4. 预先指定seed 0做验证选型；MLP每次{plugin_result['protocol']['mlp_grid'][0]['epochs']} epoch。选型后写出冻结配置，再用0–4全部重训。",
                  "5. 所有模型训练完成后才读取测试标签，一次性评估并保存逐帧预测。",
                  "6. 静态容量控制加入32维仅依赖当前输入的多项式/三角基；窗口控制输入同一个20×4窗口。", "",
                  "## 指标、时间和结论边界", ""])
    for key, value in definitions.items():
        if isinstance(value, str):
            lines.append(f"- **{key}**：{value}")
    lines.extend(["", "## 局限", ""]+[f"- {s}" for s in definitions["limitations"]])
    lines.extend(["", "## 可复核文件", "", f"- 分析脚本：`{Path(__file__).resolve()}`",
                  f"- 原始长表、预测、检查点和冻结配置：`{OUTPUT}`",
                  f"- HTML输入JSON：`{REPORT.with_suffix('.json')}`", ""])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.with_suffix(".md").write_text("\n".join(lines))
    # Structural and completeness checks, not redundant data-file hash scans.
    assert len(plugin_result["runs"]) == 2*len(VARIANTS)*5
    assert len(conv["epoch0"]) == 5
    assert all(c["rows"] for c in charts)
    json.dumps(artifact, allow_nan=False)
    write_json(OUTPUT / "COMPLETE.json", dict(completed_at=stamp(), formal_plugin_runs=len(plugin_result["runs"]),
                                             epoch0_reconstructions=5, report=str(REPORT.with_suffix(".json")),
                                             checks="Finite metrics; all fixed seeds; 60 fits; feature recurrence cross-check; source/prior reconstruction checks"))
    print(json.dumps({"summary": summary, "contrasts": contrasts}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "convergence", "plugin", "report"), default="all")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("all", "convergence", "plugin"):
        _, data = load_roles(("train", "val"))
        print(f"loaded train={len(data['train']['x'])}, val={len(data['val']['x'])}; CPU threads={torch.get_num_threads()}", flush=True)
    if args.phase in ("all", "convergence"):
        convergence(data)
    if args.phase in ("all", "plugin"):
        plugins(data, args.epochs)
    if args.phase in ("all", "report"):
        report(read(OUTPUT / "convergence.json"), read(OUTPUT / "plugin_results.json"))


if __name__ == "__main__":
    main()
