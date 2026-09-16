#!/usr/bin/env python3
"""Frozen 20-seed extension of memory features and window-MLP evaluation.

All writes belong to modeling_paper_revision_20260913_002/plugin. Original
checkpoints, datasets, reports and paper drafts remain read-only inputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from scipy import stats
import analyze_modeling_plugin_convergence as prior
from src.benchmarks.modeling_data import write_json
from src.evaluation.modeling_benchmark_metrics import skeleton_metrics, render_tube, mask_metrics

OLD = prior.OUTPUT
STUDY = prior.STUDY
OUT = ROOT / "workspace/runs/analysis/modeling_paper_revision_20260913_002/plugin"
REPORT = ROOT / "workspace/reports/modeling_paper_revision_20260913_002/plugin"
SEEDS = list(range(20))
VARIANTS = prior.VARIANTS
LABELS = {"base": "静态 MLP（插件基座）", "path": "MLP + 路径记忆", "time": "MLP + 时间记忆",
          "both": "MLP + 双记忆", "static_capacity": "MLP + 静态容量特征", "window": "窗口 MLP"}
_TRAIN = None


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def freeze_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = prior.read(OLD / "frozen_plugin_configs.json")
    protocol = dict(
        schema="fixed_seed_extension_v1", declared_at=prior.stamp(), final_seeds=SEEDS,
        reused_seeds=list(range(5)), new_seeds=list(range(5, 20)), variants=VARIANTS,
        original_freeze=rel(OLD / "frozen_plugin_configs.json"),
        selections={v: frozen["selections"][f"mlp/{v}"] for v in VARIANTS},
        dataset_manifest=rel(STUDY / "data/dataset_manifest.json"),
        original_protocol=frozen["protocol"],
        primary_metric="pooled mean_node_mm; paired scalar per training seed",
        repetitions="120 MLP fits total; 30 archived + 90 new; all final seeds reported",
        sampling_rule="Fixed seeds 0..19 before new fitting; no p-dependent stopping, selection or replacement",
        statistical_tests="5 variants vs base: paired two-sided exact Wilcoxon, Holm across five primary contrasts",
        bootstrap="paired seed-difference percentile CI; 20000 resamples; RNG seed 20260913; 95%",
        scope="Training randomness conditional on the fixed three-sequence split; not independent acquisition or new-data generalization",
        prior_results_seen=True,
        extension_status="Secondary extension motivated after seeing the initial five repeats; sample size fixed before new runs",
        linear="Deterministic closed form: report one fit per variant, no repeated-seed inference",
        CPU_processes=2, threads_per_process=2,
        test_policy="Existing seed0..4 tests were previously seen; all new train/val fits finish before extension test evaluation",
        formal_rmse="mean over frames of sqrt(mean over nodes of squared Euclidean error))",
        plugin_global_rmse="sqrt(mean over all frames and nodes of squared Euclidean error)); separately named node_global_rmse_mm",
    )
    path = OUT / "frozen_extension_protocol.json"
    if path.exists():
        old = prior.read(path)
        for k in ("final_seeds", "new_seeds", "variants", "selections", "sampling_rule", "statistical_tests"):
            assert old[k] == protocol[k], k
        return old
    write_json(path, protocol)
    return protocol


def prepare_training():
    protocol = freeze_protocol()
    if (OUT / "training_features.npz").exists():
        return protocol
    _, roles = prior.load_roles(("train", "val"))
    normalization = prior.read(OLD / "normalization.json")
    center = np.asarray(normalization["target_center_xyz"], dtype=np.float32)
    scale = normalization["target_scale"]
    actual_center, actual_scale = prior.fit_normalization(roles["train"]["sequences"])
    assert np.allclose(center, actual_center, rtol=0, atol=1e-6)
    assert abs(scale - actual_scale) < 1e-6
    payload = dict(y=(roles["train"]["y"].numpy()-center)/scale,
                   vy=roles["val"]["y"].numpy(), center=center, scale=np.array(scale))
    for role in ("train", "val"):
        raw = prior.feature_bank(roles[role]["x"])
        for v in VARIANTS:
            norm = normalization["features"][v]
            mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
            if role == "train":
                recomputed = raw[v].std(0)
                recomputed = np.where(recomputed < 1e-6, 1., recomputed)
                assert np.allclose(mean, raw[v].mean(0), atol=1e-6)
                assert np.allclose(std, recomputed, atol=1e-6)
            payload[f"{role}_{v}"] = ((raw[v]-mean)/std).astype(np.float32)
    np.savez(OUT / "training_features.npz", **payload)
    write_json(OUT / "normalization.json", normalization)
    write_json(OUT / "training_data_check.json", dict(
        train_windows=len(payload["y"]), val_windows=len(payload["vy"]),
        target_normalization_matches=True, train_feature_normalization_matches=True,
        source=rel(OLD / "normalization.json")))
    return protocol


def worker(seed):
    global _TRAIN
    if _TRAIN is None:
        with np.load(OUT / "training_features.npz") as data:
            _TRAIN = {key: data[key].copy() for key in data.files}
    data = _TRAIN
    configs = prior.read(OUT / "frozen_extension_protocol.json")["selections"]
    rows = []
    for v in VARIANTS:
        dest = OUT / "formal/mlp" / v / f"seed_{seed}"
        if (dest / "fit.json").exists() and (dest / "model.pt").exists():
            fit = prior.read(dest / "fit.json")
            assert fit["config"] == configs[v]["config"] and fit["seed"] == seed
            assert prior.read(dest / "history.json")[-1]["epoch"] == 100
        else:
            fit = prior.fit_mlp(data[f"train_{v}"], data["y"], data[f"val_{v}"], data["vy"],
                                float(data["scale"]), data["center"], configs[v]["config"], seed, dest)
        rows.append(dict(variant=v, **fit))
        print(f"FIT seed={seed:02d} {v:15s} val={fit['val_mean_node_mm']:.6f} seconds={fit['fit_seconds']:.1f}", flush=True)
    return rows


def train():
    protocol = prepare_training()
    start = time.perf_counter()
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(worker, seed): seed for seed in protocol["new_seeds"]}
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
            write_json(OUT / "training_progress.json", dict(completed_new_fits=len(rows), planned_new_fits=90,
                       complete_seeds=sorted({r["seed"] for r in rows}), updated_at=prior.stamp()))
    assert len(rows) == 90
    write_json(OUT / "training_complete.json", dict(completed_at=prior.stamp(), new_fits=90,
               reused_fits=30, elapsed_seconds=time.perf_counter()-start, final_seeds=SEEDS))


def checkpoint_dir(v, seed):
    return (OLD if seed < 5 else OUT) / "formal/mlp" / v / f"seed_{seed}"


def summarize_frame_errors(pred, target):
    """Match requested primary table: per-frame RMSE first, then frame mean."""
    frame = skeleton_metrics(pred, target)
    result = {key: float(value.mean()) for key, value in frame.items()}
    distance = np.linalg.norm(pred.astype(np.float64)-target, axis=-1)
    result.update(node_global_rmse_mm=float(np.sqrt(np.mean(distance**2))),
                  endpoint_rmse_mm=float(np.sqrt(np.mean(distance[:, -1]**2))),
                  node_p50_mm=float(np.quantile(distance, .5)), node_p95_mm=float(np.quantile(distance, .95)),
                  endpoint_p50_mm=float(np.quantile(distance[:, -1], .5)), endpoint_p95_mm=float(np.quantile(distance[:, -1], .95)))
    return result, frame


def signed_rank_exact(delta):
    """Exact sign randomization of Wilcoxon ranks, including zero/tied cases."""
    delta = np.asarray(delta, dtype=float)
    active = delta[delta != 0]
    if not len(active):
        return dict(statistic=0., pvalue=1., nonzero_pairs=0, absolute_rank_ties=False)
    ranks2 = np.rint(2*stats.rankdata(np.abs(active), method="average")).astype(int)
    counts = np.zeros(int(ranks2.sum())+1, dtype=np.int64)
    counts[0] = 1
    for rank in ranks2:
        previous = counts.copy()
        counts[rank:] += previous[:-rank]
    observed = int(ranks2[active > 0].sum())
    p = min(1., 2*min(counts[:observed+1].sum(), counts[observed:].sum())/2**len(active))
    ties = len(np.unique(np.abs(active))) != len(active)
    if not ties and len(active) == len(delta):
        scipy_p = float(stats.wilcoxon(delta, alternative="two-sided", method="exact").pvalue)
        assert abs(p-scipy_p) < 1e-14
    return dict(statistic=min(observed, int(ranks2.sum())-observed)/2, pvalue=float(p),
                nonzero_pairs=len(active), absolute_rank_ties=ties)


def paired_statistics(rows):
    base = sorted((r for r in rows if r["variant"] == "base"), key=lambda r:r["seed"])
    assert [r["seed"] for r in base] == SEEDS
    rng = np.random.default_rng(20260913)
    bootstrap_indices = rng.integers(0, len(SEEDS), size=(20000, len(SEEDS)))
    contrasts = []
    for v in VARIANTS[1:]:
        group = sorted((r for r in rows if r["variant"] == v), key=lambda r:r["seed"])
        assert [r["seed"] for r in group] == SEEDS
        b = np.asarray([r["mean_node_mm"] for r in base])
        x = np.asarray([r["mean_node_mm"] for r in group])
        delta = b-x
        ci = np.quantile(delta[bootstrap_indices].mean(1), [.025, .975])
        result = signed_rank_exact(delta)
        contrasts.append(dict(variant=v, label=LABELS[v], n_pairs=len(delta),
            improvement_mm=float(delta.mean()), improvement_sd_mm=float(delta.std(ddof=1)),
            improvement_pct=float(100*delta.mean()/b.mean()), better_seeds=int((delta>0).sum()),
            bootstrap95_lower_mm=float(ci[0]), bootstrap95_upper_mm=float(ci[1]),
            improvement_by_seed_mm=delta.tolist(), wilcoxon_exact_p=result["pvalue"],
            signed_rank_statistic=result["statistic"], nonzero_pairs=result["nonzero_pairs"],
            absolute_rank_ties=result["absolute_rank_ties"],
            exact_method="Exact Wilcoxon signed-rank sign enumeration; zero_method=wilcox; average tied ranks"))
    adjusted = prior.holm([r["wilcoxon_exact_p"] for r in contrasts])
    for row, adjusted_p in zip(contrasts, adjusted):
        row["holm_p"] = adjusted_p
        row["holm_significant_005"] = adjusted_p < .05
    return contrasts


def aggregate(rows, key, metrics):
    summaries = []
    for name in dict.fromkeys(r[key] for r in rows):
        group = [r for r in rows if r[key] == name]
        row = {key:name, "label":group[0]["label"], "n_seeds":len(group),
               "parameter_count":group[0]["parameter_count"]}
        for metric in metrics:
            values = np.asarray([r[metric] for r in group], dtype=float)
            row[metric+"_mean"] = float(values.mean())
            row[metric+"_sd"] = float(values.std(ddof=1)) if len(values)>1 else None
        summaries.append(row)
    return summaries


def test_mlp(roles):
    test = roles["test"]
    raw = prior.feature_bank(test["x"])
    normalization = prior.read(OUT / "normalization.json")
    center = np.asarray(normalization["target_center_xyz"], dtype=np.float32)
    scale = normalization["target_scale"]
    target = test["y"].numpy()
    rows, validations, curves = [], [], []
    for v in VARIANTS:
        norm = normalization["features"][v]
        z = ((raw[v]-np.asarray(norm["mean"]))/np.asarray(norm["std"])).astype(np.float32)
        for seed in SEEDS:
            src = checkpoint_dir(v, seed)
            dest = OUT / "evaluation/mlp" / v / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            checkpoint = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
            model = prior.FeatureMLP(z.shape[1], checkpoint["config"]["width"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.inference_mode():
                pred = torch.cat([model(batch)*scale+torch.tensor(center)
                                  for batch in torch.from_numpy(z).split(1024)]).numpy()
            measured, frame_metrics = summarize_frame_errors(pred, target)
            fit = prior.read(src / "fit.json")
            row = dict(variant=v, label=LABELS[v], seed=seed,
                parameter_count=fit["parameter_count"], input_dim=fit["input_dim"], best_epoch=fit["best_epoch"],
                val_mean_node_mm=fit["val_mean_node_mm"], fit_seconds=fit["fit_seconds"],
                checkpoint=rel(src / "model.pt"), reused_checkpoint=seed<5, test_frames=len(pred), **measured)
            np.savez_compressed(dest / "predictions.npz", prediction_mm=pred, target_mm=target,
                                groups=test["groups"], **frame_metrics)
            write_json(dest / "metrics.json", row)
            rows.append(row)
            for point in prior.read(src / "history.json"):
                curves.append(dict(variant=v, label=LABELS[v], seed=seed, **point))
            if seed < 5:
                with np.load(src / "test_predictions.npz") as old:
                    assert np.array_equal(old["target"], target)
                    assert np.array_equal(old["groups"], test["groups"])
                    delta = float(np.max(np.abs(pred-old["prediction"])))
                    assert delta < .002, (v, seed, delta)
                legacy = prior.read(src / "test_metrics.json")
                validations.append(dict(variant=v, seed=seed, max_prediction_abs_difference_mm=delta,
                    node_score_difference_mm=abs(measured["mean_node_mm"]-legacy["mean_node_mm"]),
                    global_rmse_difference_mm=abs(measured["node_global_rmse_mm"]-legacy["node_rmse_mm"])))
        print(f"EVAL MLP {v}: all20 seeds", flush=True)
    assert len(rows) == 120
    write_json(OUT / "plugin_seed_metrics.json", rows)
    write_json(OUT / "reused_prediction_validation.json", validations)
    write_json(OUT / "plugin_learning_curves.json", curves)
    return rows, curves, validations


MAIN_LABELS = {"hov":"HOV", "chen_direction":"Chen 方向网络适配", "oscillator":"Krauss 振子适配",
               "koopman":"Koopman", "pcc":"PCC", "mlp":"静态 MLP（正式主对照）", "linear":"线性回归", "window":"窗口 MLP"}


def window_masks(seed):
    import cv2
    cv2.setNumThreads(1)
    _, roles = prior.load_roles(("test",))
    test = roles["test"]
    dest = OUT / "evaluation/mlp/window" / f"seed_{seed}"
    with np.load(dest / "predictions.npz") as data:
        prediction = data["prediction_mm"].copy()
        assert np.array_equal(data["target_mm"], test["y"].numpy())
    started = time.perf_counter()
    values, groups, ids = [], [], []
    for group, seq in enumerate(test["sequences"]):
        indices = np.flatnonzero(test["groups"] == group)
        frame_ids = seq["frame_ids"][19:]
        assert len(frame_ids) == len(indices)
        matrix = seq["model_to_mask"]
        radius_px = 8*np.linalg.norm(matrix[:2, 0])
        record = seq["record"]
        for index, frame_id in zip(indices, frame_ids):
            path = Path(record["masks"]) / f"{int(frame_id):05d}.png"
            target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            assert target is not None and list(target.shape) == record["mask_shape"]
            rendered = render_tube(prediction[index], matrix, target.shape, radius_px)
            measured = mask_metrics(rendered, target>0, boundary_tolerance_px=2.)
            measured.pop("boundary_tolerance_px")
            values.append(measured); groups.append(group); ids.append(int(frame_id))
    arrays = {f"mask_{k}":np.asarray([v[k] for v in values]) for k in values[0]}
    np.savez_compressed(dest / "mask_metrics.npz", frame_ids=np.asarray(ids), groups=np.asarray(groups), **arrays)
    record = dict(seed=seed, frames=len(values), radius_mm=8., boundary_tolerance_px=2., stride=1,
                  source="Existing full SAM2 masks; identical fixed tube adapter and metric functions",
                  evaluated_at=prior.stamp(), elapsed_seconds=time.perf_counter()-started,
                  **{k:float(v.mean()) for k,v in arrays.items()})
    write_json(dest / "mask_metrics.json", record)
    print(f"MASK window seed={seed} IoU={record['mask_iou']:.6f} seconds={record['elapsed_seconds']:.1f}", flush=True)
    return record


def main_comparison(plugin_rows, roles):
    test = roles["test"]
    rows, checks = [], []
    for name in MAIN_LABELS:
        if name == "window":
            continue
        for seed in range(5):
            run = STUDY / "formal" / name / f"seed_{seed}"
            cfg, manifest = prior.read(run / "resolved_config.json"), prior.read(run / "run_manifest.json")
            assert cfg["epochs"] == 100 and cfg["endpoint_weight"] == .25
            assert cfg["history"] == 20 and cfg["dataset_manifest"] == str(STUDY / "data/dataset_manifest.json")
            prediction, target, masks = [], [], []
            evaluation = STUDY / "evaluations" / name / f"seed_{seed}"
            for group, seq in enumerate(test["sequences"]):
                path = evaluation / f"{seq['record']['group']}_predictions.npz"
                with np.load(path) as data:
                    selected = test["groups"] == group
                    assert np.array_equal(data["target_mm"], test["y"].numpy()[selected])
                    assert np.array_equal(data["frame_ids"], seq["frame_ids"][19:])
                    assert np.array_equal(data["mask_frame_ids"], data["frame_ids"])
                    prediction.append(data["prediction_mm"].copy()); target.append(data["target_mm"].copy())
                    masks.append({k:data[k].copy() for k in data.files if k.startswith("mask_") and k != "mask_frame_ids"})
            pred, y = np.concatenate(prediction), np.concatenate(target)
            measured, _ = summarize_frame_errors(pred, y)
            mask_summary = {k:float(np.concatenate([m[k] for m in masks]).mean()) for k in masks[0]}
            row = dict(model=name, label=MAIN_LABELS[name], seed=seed, test_frames=len(pred),
                parameter_count=manifest["parameter_count_including_fitted_reference"],
                best_epoch=manifest["best_epoch"], val_mean_node_mm=manifest["best_validation_node_mean_mm"],
                stochastic=name != "linear", source=rel(evaluation), **measured, **mask_summary)
            rows.append(row)
            checks.append(dict(model=name, seed=seed, matching_targets=True, matching_frame_ids=True,
                complete_mask_coverage=True, n_frames=len(y)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as executor:
        mask_results = list(executor.map(window_masks, range(5)))
    for seed in range(5):
        row = next(r for r in plugin_rows if r["variant"] == "window" and r["seed"] == seed)
        mask = next(r for r in mask_results if r["seed"] == seed)
        metrics = {k:row[k] for k in ("mean_node_mm", "endpoint_mm", "node_rmse_mm", "node_global_rmse_mm",
                    "max_node_mm", "chamfer_mm", "node_p50_mm", "node_p95_mm", "endpoint_p50_mm", "endpoint_p95_mm", "endpoint_rmse_mm")}
        rows.append(dict(model="window", label=MAIN_LABELS["window"], seed=seed, test_frames=row["test_frames"],
            parameter_count=row["parameter_count"], best_epoch=row["best_epoch"], val_mean_node_mm=row["val_mean_node_mm"],
            stochastic=True, source=rel(OUT / "evaluation/mlp/window" / f"seed_{seed}"), **metrics,
            **{k:mask[k] for k in mask if k.startswith("mask_")}))
    write_json(OUT / "main_comparison_seed_metrics.json", rows)
    write_json(OUT / "main_comparison_data_validation.json", checks)
    return rows, checks


def protocol_audit():
    screens = prior.read(STUDY / "screening_summary.json")["candidates"]
    audit = []
    for name in MAIN_LABELS:
        if name == "window":
            configs = [r["config"] for r in prior.read(OLD / "screening_results.json")
                       if r["family"] == "mlp" and r["variant"] == "window"]
            cfg = prior.read(OLD / "frozen_plugin_configs.json")["selections"]["mlp/window"]["config"]
        else:
            configs = [r["config"] for r in screens if r["model"] == name]
            cfg = prior.read(STUDY / "formal" / name / "seed_0/resolved_config.json")
        audit.append(dict(model=name, label=MAIN_LABELS[name], candidates_recorded=len(configs),
            candidate_lr=sorted(set(c["lr"] for c in configs)),
            candidate_width=sorted(set(c.get("width", c.get("hidden", 0)) for c in configs)),
            screening_seed=0 if name=="window" else 101,
            epochs=100, batch_size=cfg["batch_size"], endpoint_weight=.25,
            loss="MSE(all normalized node coordinates) + 0.25*MSE(normalized endpoint coordinates)",
            lr=cfg["lr"], scheduler="ReduceLROnPlateau", lr_factor=.5,
            scheduler_patience=3 if name=="window" else 4, validation_interval=5,
            initial_validation_epoch=1, gradient_clip_norm=10.,
            feature_standardization="train-only per-column mean/std" if name=="window" else "pressure / 150 kPa",
            target_normalization="same train-only center and scalar scale",
            comparison_scope="fixed data, labels, causal targets, 100 epochs, loss and validation selection; architecture-specific validation grids"))
    return audit


def evaluate():
    protocol = freeze_protocol()
    completion = prior.read(OUT / "training_complete.json")
    assert completion["new_fits"] == 90
    write_json(OUT / "test_access.json", dict(extension_first_test_access=prior.stamp(),
        all_new_fits_complete=True, original_tests_previously_seen=True, configs_changed_after_extension_start=0))
    _, roles = prior.load_roles(("test",))
    assert len(roles["test"]["x"]) == 2958
    rows, curves, validations = test_mlp(roles)
    contrasts = paired_statistics(rows)
    write_json(OUT / "plugin_statistics.json", contrasts)
    main_rows, main_checks = main_comparison(rows, roles)
    audit = protocol_audit()
    write_json(OUT / "protocol_audit.json", audit)
    make_report(protocol, rows, curves, validations, contrasts, main_rows, main_checks, audit)


def make_report(protocol, rows, curves, validations, contrasts, main_rows, main_checks, audit):
    metric_names = ["mean_node_mm", "endpoint_mm", "node_rmse_mm", "node_global_rmse_mm",
                    "max_node_mm", "chamfer_mm", "val_mean_node_mm", "fit_seconds", "best_epoch"]
    plugin_summary = aggregate(rows, "variant", metric_names)
    main_summary = aggregate(main_rows, "model", ["mean_node_mm", "endpoint_mm", "node_rmse_mm",
        "node_global_rmse_mm", "chamfer_mm", "max_node_mm", "mask_iou", "mask_dice", "mask_boundary_f1", "val_mean_node_mm"])
    linear = []
    for v in VARIANTS:
        src = OLD / "formal/linear" / v / "seed_0"
        with np.load(src / "test_predictions.npz") as data:
            measured, _ = summarize_frame_errors(data["prediction"], data["target"])
        old = prior.read(src / "test_metrics.json")
        linear.append(dict(variant=v, label="线性 "+prior.NAMES[v], n_fits=1,
            parameter_count=old["parameter_count"], input_dim=old["input_dim"],
            independent_stochastic_repetitions=0, source=rel(src), **measured))
    for row in linear:
        row["improvement_pct_vs_base"] = 100*(linear[0]["mean_node_mm"]-row["mean_node_mm"])/linear[0]["mean_node_mm"]
    for row in plugin_summary:
        c = next((c for c in contrasts if c["variant"] == row["variant"]), None)
        row.update(holm_p=c["holm_p"] if c else None,
            improvement_pct_vs_base=c["improvement_pct"] if c else 0.,
            input_dim=next(r["input_dim"] for r in rows if r["variant"] == row["variant"]))
    for row in main_summary:
        row["independent_stochastic_repetitions"] = 0 if row["model"] == "linear" else 5

    window5 = next(r for r in main_summary if r["model"] == "window")
    hov5 = next(r for r in main_summary if r["model"] == "hov")
    node_window = [next(r["mean_node_mm"] for r in main_rows if r["model"]=="window" and r["seed"]==s) for s in range(5)]
    node_hov = [next(r["mean_node_mm"] for r in main_rows if r["model"]=="hov" and r["seed"]==s) for s in range(5)]
    window_hov = dict(n_pairs=5, difference_window_minus_hov_mm=float(np.mean(node_window)-np.mean(node_hov)),
                     **signed_rank_exact(np.asarray(node_window)-np.asarray(node_hov)),
                     scope="Descriptive supplementary pairwise comparison; five seeds only; no unadjusted significance claim")
    both = next(r for r in plugin_summary if r["variant"] == "both")
    base = next(r for r in plugin_summary if r["variant"] == "base")
    both_c = next(r for r in contrasts if r["variant"] == "both")
    window20 = next(r for r in plugin_summary if r["variant"] == "window")
    static_control = next(r for r in plugin_summary if r["variant"] == "static_capacity")
    findings = [
        dict(id="dual_memory", title="双记忆插件的20次固定重复", status="measured", text=
             f"静态插件基座的测试节点误差为 {base['mean_node_mm_mean']:.4f}±{base['mean_node_mm_sd']:.4f} mm；"
             f"加入双记忆后为 {both['mean_node_mm_mean']:.4f}±{both['mean_node_mm_sd']:.4f} mm，"
             f"改善 {both_c['improvement_pct']:.2f}%，{both_c['better_seeds']}/20 个seed改善。"
             f"配对双侧精确Wilcoxon p={both_c['wilcoxon_exact_p']:.8g}，五项Holm后p={both_c['holm_p']:.8g}，"
             f"平均误差减少的95% seed bootstrap CI为[{both_c['bootstrap95_lower_mm']:.4f}, {both_c['bootstrap95_upper_mm']:.4f}] mm。"),
        dict(id="comparison_window", title="窗口MLP纳入主对照", status="measured", text=
             f"固定seed0..4的窗口MLP为 {window5['mean_node_mm_mean']:.4f}±{window5['mean_node_mm_sd']:.4f} mm，"
             f"参数量 {window5['parameter_count']}；HOV为 {hov5['mean_node_mm_mean']:.4f}±{hov5['mean_node_mm_sd']:.4f} mm，"
             f"参数量 {hov5['parameter_count']}（包含已拟合参考系数）。窗口MLP测试误差较低，应保留这一结果。"),
        dict(id="capacity_window", title="输入压缩与精度权衡", status="measured", text=
             f"20次重复中，双记忆MLP使用36维输入/{both['parameter_count']}参数，"
             f"窗口MLP使用80维输入/{window20['parameter_count']}参数，节点误差分别为"
             f"{both['mean_node_mm_mean']:.4f}和{window20['mean_node_mm_mean']:.4f} mm。"
             f"36维静态容量控制为{static_control['mean_node_mm_mean']:.4f} mm，参数量{static_control['parameter_count']}。"),
        dict(id="transfer_scope", title="可移植的是记忆特征结构", status="interpretation", text=
             "固定路径与时间递推特征可以接入普通线性读出和MLP，适配后用原训练集重新拟合输出。"
             "实验验证结构可复用；没有迁移HOV训练权重，没有使用HOV几何解码器。"),
        dict(id="statistics_scope", title="重复实验的统计范围", status="verified", text=
             "初始5次结果此前已观察；本次在新增训练前固定最终seed0..19，全部结果纳入。"
             "统计置信区间反映固定三序列划分下的优化随机性。确定性线性只报告一次解。"),
    ]
    definitions = dict(
        split="Same three-sequence chronological 60/20/20 split, pooled by role; 5 Hz; H20; 8988 train, 2958 val, 2958 test windows",
        node_mean="mean_{t,j} ||prediction[t,j]-target[t,j]||_2, mm",
        endpoint="mean_t ||prediction[t,tip]-target[t,tip]||_2, mm",
        node_rmse="mean_t sqrt(mean_j ||prediction[t,j]-target[t,j]||_2^2), mm; requested frame-mean definition for all new tables",
        node_global_rmse="sqrt(mean_{t,j} ||prediction[t,j]-target[t,j]||_2^2), mm; separately retained legacy/global statistic",
        mask="per-frame IoU/Dice/boundary F1 then pooled over all 2958 test frames; fixed 8 mm tube radius, 2 px boundary tolerance",
        variability="mean±sample SD, ddof=1; MLP20 seeds0..19; main comparison5 seeds0..4",
        significance="paired two-sided exact Wilcoxon on pooled seed node error; five contrasts vs plugin base; Holm adjustment; bootstrap paired differences 20000 resamples",
        plugin_training="same frozen 100-epoch FeatureMLP implementation and selected LR/width; reused0..4; trained5..19; all new fits completed before extension test evaluation",
        loss="L=MSE(normalized full skeleton)+0.25*MSE(normalized endpoint); same formula and output normalization as formal models",
        static_models="Formal static MLP: 22957 parameters, independent main-table validation choice. Plugin base MLP: 7405 parameters, independent plugin-grid validation choice. They occupy separate tables.",
        tuning="Each plugin variant received the same 2 widths × 2 learning rates on val before freezing. Main baseline search uses archived architecture-specific grids; not an identical optimizer trajectory or exhaustive optimum.",
        implementation_differences="Window/plugin batch512 and scheduler patience3; formal batch256 and patience4; window/plugin additionally use train-only feature standardization. No configuration retuning in this extension.",
        mask_labels="Skeleton and mask targets derive from the same SAM2 annotations; mask is complementary geometric agreement, not independent truth",
        scope="Conditional training-seed variability on a previously chosen three-sequence subset; generalization to new acquisitions requires new held-out experiments")
    charts = []
    def chart(identifier, title, kind, x, y, data, unit="mm", color=None, note=""):
        item = dict(id=identifier, title=title, kind=kind, x=x, y=y, rows=data, unit=unit, note=note)
        if color:
            item["color"] = color
        charts.append(item)
    chart("plugin_node20", "记忆插件的20次平均节点误差", "bar", "label", "mean_node_mm_mean", plugin_summary,
          note="全量seed0..19；SD字段为mean_node_mm_sd")
    chart("plugin_endpoint20", "记忆插件的20次末端误差", "bar", "label", "endpoint_mm_mean", plugin_summary)
    chart("plugin_seed_scatter", "六个变体的全部20次重复", "scatter", "seed", "mean_node_mm", rows, color="label")
    diff_rows = [dict(seed=s, variant=c["variant"], label=c["label"], improvement_mm=c["improvement_by_seed_mm"][s])
                 for c in contrasts for s in SEEDS]
    chart("plugin_paired_improvement", "相对插件基座的逐seed误差减少", "scatter", "seed", "improvement_mm", diff_rows, color="label")
    chart("plugin_capacity", "参数量与节点精度", "scatter", "parameter_count", "mean_node_mm_mean", plugin_summary, color="label")
    curve_rows = []
    for v in VARIANTS:
        for epoch in sorted(set(c["epoch"] for c in curves)):
            values = [c["val_mean_node_mm"] for c in curves if c["variant"]==v and c["epoch"]==epoch]
            assert len(values) == 20
            curve_rows.append(dict(variant=v, label=LABELS[v], epoch=epoch, mean=float(np.mean(values)), sd=float(np.std(values,ddof=1))))
    chart("plugin_validation_curves", "固定配置的20次验证曲线", "line", "epoch", "mean", curve_rows, color="label")
    chart("plugin_linear", "确定性线性读出的记忆插件", "bar", "label", "mean_node_mm", linear,
          note="每个变体一次解析拟合，不计算seed显著性")
    chart("main_node5", "更新主对照的节点误差（5 seeds）", "bar", "label", "mean_node_mm_mean", main_summary)
    chart("main_endpoint5", "更新主对照的末端误差（5 seeds）", "bar", "label", "endpoint_mm_mean", main_summary)
    chart("main_rmse5", "统一逐帧RMSE口径后的主对照", "bar", "label", "node_rmse_mm_mean", main_summary,
          note="每帧对节点平方误差开方，再在全部测试帧上平均")
    mask_rows = [dict(label=r["label"], metric=m, mean=r[m+"_mean"], sd=r[m+"_sd"])
                 for r in main_summary for m in ("mask_iou", "mask_dice")]
    chart("main_mask5", "全帧轮廓一致性（5 seeds）", "bar", "label", "mean", mask_rows, unit="score", color="metric")
    scalar_contrasts = [{k:v for k,v in row.items() if not isinstance(v, (list, dict))} for row in contrasts]
    scalar_audit = [{k:json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k,v in row.items()} for row in audit]
    table_data = [("plugin_summary", "插件MLP：20次重复", plugin_summary),
        ("plugin_seeds", "插件MLP：全部seed", rows), ("plugin_contrasts", "五项配对统计与Holm校正", scalar_contrasts),
        ("linear_descriptive", "线性读出的描述性对照", linear),
        ("main_comparison", "主对照：同一全量测试集5次重复", main_summary),
        ("main_seeds", "主对照：各seed完整指标", main_rows), ("protocol_audit", "训练与验证协议核对", scalar_audit)]
    tables = [dict(id=i, title=t, rows=r, columns=list(r[0])) for i,t,r in table_data]
    for identifier, _, data in table_data:
        with (OUT / f"{identifier}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    # Recompute the principal paired mean independently from aggregate values.
    assert abs(base["mean_node_mm_mean"]-both["mean_node_mm_mean"]-both_c["improvement_mm"]) < 1e-12
    assert all(len([r for r in rows if r["variant"]==v]) == 20 for v in VARIANTS)
    assert max(v["node_score_difference_mm"] for v in validations) < .002
    validation = dict(status="passed", complete_stochastic_fits=120, newly_trained_fits=90,
        reused_stochastic_fits=30, final_seeds=SEEDS, complete_test_windows_per_run=2958,
        all_new_histories_end_at_100=True, no_hyperparameter_rescreening=True,
        five_main_window_masks_complete=True, formal_targets_and_frame_ids_match=all(c["matching_targets"] and c["matching_frame_ids"] for c in main_checks),
        reused_prediction_max_abs_difference_mm=max(v["max_prediction_abs_difference_mm"] for v in validations),
        paired_mean_crosscheck=True, exact_wilcoxon_crosschecked_scipy_for_no_ties=True,
        old_outputs_modified=False, per_frame_vs_global_rmse_separately_named=True,
        old_csv_rmse_note="The previous summary.csv uses GLOBAL sqrt aggregation (HOV seed0 1.895734); requested frame-mean aggregation yields 1.690182. Both are provided under distinct names.")
    write_json(OUT / "validation.json", validation)
    payload = dict(schema="modeling_plugin_repetitions_v2", generated_at=prior.stamp(),
        protocol=protocol, definitions=definitions, findings=findings, charts=charts, tables=tables,
        plugin_summary=plugin_summary, plugin_contrasts=contrasts, main_comparison=main_summary,
        window_hov_comparison=window_hov, validation=validation,
        sources=[dict(id="original_plugin", path=rel(OLD), role="frozen validation choices and archived seeds0..4"),
                 dict(id="original_main", path=rel(STUDY), role="original primary model checkpoints and predictions"),
                 dict(id="extension", path=rel(OUT), role="new fits, unified evaluation, statistics and validation")])
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json(REPORT.with_suffix(".json"), payload)
    md = ["# 记忆插件重复统计扩充与窗口MLP主对照", "", *[f["text"]+"\n" for f in findings],
          "## 预先固定的扩充协议", "", "最终使用seed0..19：复用原0..4，新增5..19。六个MLP变体各20次，共120次；本轮新增90次训练。学习率、宽度、100 epoch训练预算和验证选模均沿用原冻结配置。所有新增模型拟合结束后统一测试2958个窗口。", "",
          "这是在初始5次结果已观察后的固定样本量扩充，未使用p值决定停止、增加样本或选择seed。独立统计单位是训练seed，测试帧不是独立重复。", "",
          "## 插件结果：均值±样本标准差", "",
          "|变体|输入维度|参数|节点误差/mm|末端误差/mm|改善/%|Holm p|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in plugin_summary:
        p = "—" if r["holm_p"] is None else f"{r['holm_p']:.7g}"
        md.append(f"|{r['label']}|{r['input_dim']}|{r['parameter_count']}|{r['mean_node_mm_mean']:.4f}±{r['mean_node_mm_sd']:.4f}|{r['endpoint_mm_mean']:.4f}±{r['endpoint_mm_sd']:.4f}|{r['improvement_pct_vs_base']:.2f}|{p}|")
    md += ["", "|对照插件基座|改善均值/mm|seed差95% bootstrap CI/mm|改善seed数|精确双侧p|Holm p|", "|---|---:|---:|---:|---:|---:|"]
    for r in contrasts:
        md.append(f"|{r['label']}|{r['improvement_mm']:.4f}|[{r['bootstrap95_lower_mm']:.4f}, {r['bootstrap95_upper_mm']:.4f}]|{r['better_seeds']}/20|{r['wilcoxon_exact_p']:.7g}|{r['holm_p']:.7g}|")
    md += ["", "五项主指标比较采用配对双侧精确Wilcoxon并进行Holm调整。Bootstrap为配对seed差的20000次百分位重采样，随机数seed固定20260913。CI未作多重比较调整；显著性判断使用Holm结果。", "",
           "## 更新主对照：同一固定划分的seed0..4", "",
           "|方法|参数|节点误差/mm|逐帧RMSE均值/mm|末端误差/mm|Mask IoU|Mask Dice|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in main_summary:
        md.append(f"|{r['label']}|{r['parameter_count']}|{r['mean_node_mm_mean']:.4f}±{r['mean_node_mm_sd']:.4f}|{r['node_rmse_mm_mean']:.4f}±{r['node_rmse_mm_sd']:.4f}|{r['endpoint_mm_mean']:.4f}±{r['endpoint_mm_sd']:.4f}|{r['mask_iou_mean']:.4f}±{r['mask_iou_sd']:.4f}|{r['mask_dice_mean']:.4f}±{r['mask_dice_sd']:.4f}|")
    md += ["", "线性为确定性解析解，五个归档seed的相同输出仅用于与既有主表对齐，不能解释为五次随机重复。HOV参数量包含已拟合参考系数。", "",
           "主表静态MLP保持原正式模型的22957参数与1.96878 mm结果；插件基座使用独立验证选出的7405参数模型，两者分表报告。", "",
           "RMSE统一为每帧sqrt(mean节点平方欧氏误差)后对帧平均。核算发现旧summary.csv实际上使用全局sqrt口径，故本次对所有主表模型重算，并在node_global_rmse_mm单列保留全局值。节点均值与末端均值口径一致。", "",
           "Mask在全部2958帧、每个seed上使用同一render_tube与mask_metrics函数计算，固定半径8 mm、边界容差2 px、stride1。旧主对照复用其已保存的全帧mask分数；窗口MLP重新渲染并评分，不拟合新的轮廓参数。", "",
           "## 公平性核对与适用范围", "",
           "损失一致：L=MSE(归一化全部节点坐标)+0.25 MSE(归一化末端坐标)。输出归一化、train/val/test划分、H20、当前目标帧、5Hz采样及100epoch预算一致。窗口MLP对宽度64/128与学习率0.001/0.003共4种配置进行了完整验证筛选，提供了充分的验证机会。", "",
           "各方法保留各自的验证选型。插件/窗口批大小512、调度器耐心值3，原正式模型分别为256与4；窗口输入额外使用训练集逐列标准化。不能将这一对照描述为每项优化设置完全相同，也不能根据不同硬件上的训练耗时排名。原正式各架构的已记录候选数与范围见protocol_audit表。", "",
           "两类记忆作为固定递推特征输入线性/MLP读出，随后拟合下游网络，支持模块化复用。各插件变体单独在相同验证网格选择容量；时间分支或静态容量控制选到128宽度，双记忆与基座选到64。因此跨变体比较仍应结合参数量。", "",
           "线性插件沿用一次确定性拟合的描述性结果：", "", "|变体|参数|节点误差/mm|相对基础改善/%|", "|---|---:|---:|---:|"]
    for r in linear:
        md.append(f"|{r['label']}|{r['parameter_count']}|{r['mean_node_mm']:.4f}|{r['improvement_pct_vs_base']:.2f}|")
    md += ["", "## 可用于初稿的结果段落", "", findings[0]["text"]+
           "这些结果说明，以显式递推计算的路径与时间记忆能够作为通用历史特征接入基础预测器。"
           "与36维静态容量特征对照结合，结果支持历史表示对预测的贡献。"
           "窗口MLP使用更高维的完整历史输入，在本数据上取得更低节点误差；双记忆提供的是紧凑且可递推的表示。", "",
           "显著性与置信区间只反映固定三序列划分下训练随机性的影响。该子集此前根据观察选取，新增seed不产生新的独立采集数据，结论范围限于本数据与冻结协议。", "",
           "## 复现与验证", "", "```bash",
           "/Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/extend_modeling_plugin_repetitions.py",
           "```", "", "脚本可复用完整的新拟合并重新评价。`--train-only`只训练，`--evaluate-only`只评价，`--report-only`从已存评价生成报告。",
           "", f"配置与结果目录：`{rel(OUT)}`。JSON含通用charts/tables、全部20seed统计和完整主表指标。`validation.json`记录预测复算、同标签/帧ID、覆盖率、统计交叉核对。"]
    REPORT.with_suffix(".md").write_text("\n".join(md)+"\n", encoding="utf-8")
    write_json(OUT / "COMPLETE.json", dict(completed_at=prior.stamp(), report_json=rel(REPORT.with_suffix(".json")),
        report_md=rel(REPORT.with_suffix(".md")), mlp_fits=120, new_mlp_fits=90, test_evaluations=120,
        full_window_mask_evaluations=5, validation="passed"))
    print(f"COMPLETE {REPORT.with_suffix('.json')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        make_report(prior.read(OUT / "frozen_extension_protocol.json"), prior.read(OUT / "plugin_seed_metrics.json"),
            prior.read(OUT / "plugin_learning_curves.json"), prior.read(OUT / "reused_prediction_validation.json"),
            prior.read(OUT / "plugin_statistics.json"), prior.read(OUT / "main_comparison_seed_metrics.json"),
            prior.read(OUT / "main_comparison_data_validation.json"), prior.read(OUT / "protocol_audit.json"))
        sys.exit(0)
    if not args.evaluate_only:
        train()
    if not args.train_only:
        evaluate()
