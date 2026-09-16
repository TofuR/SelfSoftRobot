#!/usr/bin/env python3
"""Frozen unified20 HOV geometry analysis; run with the selfsr Python and -B.

All seeds 100..119 and all 2958 test windows are mandatory. The only output
directory is analysis/modeling_unified20_20260913_005/geometry. Mathematical
helpers below are copied from the two representation analyses and the fixed
input matching protocol; their old result directories are never opened.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from scipy.spatial import cKDTree
import scipy
import torch
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
OUT = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005/geometry"
SOURCE = RUN / "source"
sys.path.insert(0, str(SOURCE))
from src.benchmarks.modeling_models import make_model
from src.models.model_ishsm import generalized_to_skeleton

SEEDS = list(range(100, 120))
KINDS = ("reference", "joint_linear", "full")
CATEGORY = dict(opposite="current pressure close, opposite direction",
                same_direction="current pressure and direction close, different history",
                same_recent_two="recent two inputs close, different earlier history")
TOLERANCES = (1., 2., 5., 10.)


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(name, data):
    (OUT / name).write_text(json.dumps(plain(data), indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text())


def write_csv(name, rows):
    if not rows:
        raise ValueError(f"Empty output: {name}")
    with (OUT / name).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(plain(rows))


def stats(values):
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(n=int(a.size), mean=float(a.mean()),
                sd=float(a.std(ddof=1)) if a.size > 1 else 0.,
                min=float(a.min()), max=float(a.max()))


def summarize(rows, fields):
    return {field: stats([r[field] for r in rows]) for field in fields}


def cosine(a, b):
    aa, bb = np.sum(a*a), np.sum(b*b)
    return float(np.sum(a*b)/np.sqrt(aa*bb)) if aa > 0 and bb > 0 else None


def array(x):
    return x.detach().cpu().numpy().astype(np.float64)


def load_inputs(manifest):
    """Follow the manifest order used by 004; training labels are not loaded."""
    meta = read_json(manifest)
    assert meta["H"] == 20 and meta["dt"] == .2
    assert meta["length_unit"] == "mm" and meta["node_order"] == "base_to_tip"
    assert meta["action_scale_kpa"] == [150]*4
    groups, inventory = {}, []
    for split in ("train", "test"):
        groups[split] = []
        for row in meta["files"]:
            if row["role"] != split:
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = manifest.parent / path
            with np.load(path, allow_pickle=False) as d:
                a = d["actions"].astype(np.float32)
                ids, times = d["frame_ids"], d["timestamps"]
                assert a.shape == (row["frames"], 4) and np.isfinite(a).all()
                assert np.array_equal(ids, np.arange(row["start"], row["stop"]))
                assert np.all(np.diff(times) > 0)
                windows = np.stack([a[t-19:t+1] for t in range(19, len(a))])
                g = dict(group=row["group"], actions=a, windows=windows,
                         frames=ids[19:], times=times[19:])
                if split == "test":
                    g["target"] = d["positions"][19:].astype(np.float64)
                    assert np.isfinite(g["target"]).all()
                groups[split].append(g)
                inventory.append(dict(split=split, group=row["group"], path=path,
                                      raw_frames=len(a), scored_frames=len(windows)))
        assert len(groups[split]) == 3
        assert sum(len(g["windows"]) for g in groups[split]) == meta["scored_counts"][split]
    assert sum(len(g["windows"]) for g in groups["train"]) == 8988
    assert sum(len(g["windows"]) for g in groups["test"]) == 2958
    return groups, inventory, meta


def select_references(groups):
    a = np.concatenate([g["windows"][:, -1] for g in groups]).astype(np.float64)
    ids = np.concatenate([g["frames"] for g in groups])
    gi = np.concatenate([np.full(len(g["frames"]), i) for i, g in enumerate(groups)])
    mean, sd = a.mean(0), a.std(0)
    assert np.all(sd > 0)
    z = (a-mean)/sd
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    # Resolve PCA sign ambiguity by the largest-magnitude loading (first tie).
    pc = vt[0] * (1 if vt[0, np.argmax(abs(vt[0]))] >= 0 else -1)
    scores = z @ pc
    quantiles = np.array([.1, .3, .5, .7, .9])
    thresholds = np.quantile(scores, quantiles)
    indices = np.array([np.argmin(abs(scores-v)) for v in thresholds])
    refs = np.vstack([mean, a[indices]]).astype(np.float32)
    selection = dict(selection_population="8988 training-window current inputs, pooled frames",
        rule="mean plus actual input nearest standardized PC1 quantiles .1/.3/.5/.7/.9; first pooled index breaks ties",
        pca_sign="largest absolute loading positive; first channel breaks ties",
        mean=mean, sd_population=sd, pc1=pc, singular_values=s,
        quantiles=quantiles, quantile_scores=thresholds, selected_scores=scores[indices],
        pooled_indices=indices, frame_ids=ids[indices], group_indices=gi[indices],
        group_names=[g["group"] for g in groups],
        reference_labels=["train_mean", "pc1_q10", "pc1_q30", "pc1_q50", "pc1_q70", "pc1_q90"],
        reference_actions=refs, reference_pressure_kpa=refs.astype(float)*150)
    np.savez_compressed(OUT / "training_reference_inputs.npz", action=a,
                        frame_ids=ids, group_index=gi, pc1_score=scores)
    write_json("reference_selection.json", selection)
    return refs, selection


def prepare_pairs(groups):
    pairs, coverage = [], []
    for tol in TOLERANCES:
        offset = 0
        for group_index, g in enumerate(groups):
            selected, counts = select_pairs(g, tol)
            for category, rows in selected.items():
                coverage.append(dict(tolerance_kpa=tol, category=category,
                    group_index=group_index, group=g["group"],
                    candidate_pairs=counts[category], selected_pairs=len(rows),
                    active_channels=np.flatnonzero(np.ptp(g["actions"], axis=0) > .01).tolist()))
                for row in rows:
                    pairs.append(dict(pair_id=len(pairs), tolerance_kpa=tol,
                        category=category, group_index=group_index, group=g["group"],
                        pooled_i=offset+row["i"], pooled_j=offset+row["j"],
                        frame_i=int(g["frames"][row["i"]]), frame_j=int(g["frames"][row["j"]]), **row))
            offset += len(g["windows"])
    write_csv("matched_pairs.csv", pairs)
    write_json("matched_pair_coverage.json", coverage)
    return pairs, coverage


def jacobian(core, geo):
    """Analytic J at each reference: frames x nodes x xy x physical coordinates."""
    b, ell = geo["bend_reference"], geo["length_reference"]
    section = np.repeat(np.arange(core.n_sections), core.section_intervals)
    lam = ell[:, section]
    lengths = array(core.reference_segment_lengths)*np.exp(np.clip(lam, -.25, .25))
    theta = np.cumsum(b, axis=1)
    tangent = np.stack([np.cos(theta), np.sin(theta)], -1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], -1)
    bend = normal[..., None]*np.cumsum(array(core.bend_basis), axis=0)[None, :, None, :]
    length = tangent[..., None]*np.eye(core.n_sections)[section][None, :, None, :]
    length *= ((lam > -.25) & (lam < .25))[:, :, None, None]
    segment = lengths[:, :, None, None]*np.concatenate([bend, length], axis=-1)
    return np.concatenate([np.zeros((len(b), 1, 2, core.generalized_dim)),
                           np.cumsum(segment, axis=1)], axis=1)


def readouts(core):
    scale = array(core.generalized_coordinate_scale)
    wp = array(core.play.weights)[:, :, None]*array(core.pi_mode_directions)*scale
    wh = array(core.maxwell_gains)[:, :, None]*array(core.maxwell_mode_directions)*scale
    return wp, wh


def collect(core, model, windows):
    pools = {k: [] for k in ("drive", "q", "d", "memory_path", "memory_time", "memory_joint", "full_xyz_mm")}
    qa = dict(stop_recurrence_max=0., time_convolution_max=0., core_vs_model_max_mm=0.,
              path_readout_max=0., time_readout_max=0., memory_additivity_max=0.)
    wp, wh = readouts(core)
    alpha = array(core.maxwell.decays)
    for start in range(0, len(windows), 256):
        a = torch.from_numpy(windows[start:start+256])
        with torch.inference_mode():
            e = core.drive(a.flatten(0, 1)).reshape(len(a), 20, 4)
            p = e[:, 0, :, None].repeat(1, 1, core.n_play)
            h = e[:, 0, :, None].repeat(1, 1, core.n_maxwell)
            q = torch.zeros_like(p)
            for t in range(1, 20):
                qprevious = q
                p, q = core.play.step(p, e[:, t])
                qcheck = torch.clamp(qprevious + (e[:, t]-e[:, t-1])[:, :, None],
                                     -core.play.thresholds, core.play.thresholds)
                qa["stop_recurrence_max"] = max(qa["stop_recurrence_max"], float((q-qcheck).abs().max()))
                h = core.maxwell.step(h, e[:, t])
            result = core._state_output(a[:, -1], p, h, q, e[:, -1])
            full = model(a)*core.pc_scale + core.pc_center
            manual = result["skeleton"]*core.pc_scale + core.pc_center
        values = dict(drive=array(e), q=array(q), d=array(h-e[:, -1, :, None]),
            memory_path=array(result["pi_generalized"]), memory_time=array(result["maxwell_generalized"]),
            memory_joint=array(result["memory_generalized"]), full_xyz_mm=array(full))
        expanded = -np.einsum("btc,tk->bck", np.diff(values["drive"], axis=1),
                              alpha[None, :]**np.arange(19, 0, -1)[:, None])
        errors = dict(time_convolution_max=np.max(abs(expanded-values["d"])),
            core_vs_model_max_mm=np.max(abs(array(manual)-values["full_xyz_mm"])),
            path_readout_max=np.max(abs(np.einsum("bck,ckg->bg", values["q"], wp)-values["memory_path"])),
            time_readout_max=np.max(abs(np.einsum("bck,ckg->bg", values["d"], wh)-values["memory_time"])),
            memory_additivity_max=np.max(abs(values["memory_joint"]-values["memory_path"]-values["memory_time"])))
        for key, value in errors.items():
            qa[key] = max(qa[key], float(value))
        for key, value in values.items():
            pools[key].append(value)
    assert qa["stop_recurrence_max"] < 2e-7, qa
    assert qa["time_convolution_max"] < 2e-6, qa
    assert qa["core_vs_model_max_mm"] < 2e-4, qa
    assert max(qa[k] for k in ("path_readout_max", "time_readout_max", "memory_additivity_max")) < 1e-6, qa
    return {k: np.concatenate(v) for k, v in pools.items()}, qa


def analyze_seed(seed, core, model, windows, target, groups, ids, pairs):
    raw, qa = collect(core, model, windows)
    geo = geometry_arrays(core, windows[:, -1], raw["memory_joint"])
    j = jacobian(core, geo)
    path = np.einsum("bndg,bg->bnd", j, raw["memory_path"])
    temporal = np.einsum("bndg,bg->bnd", j, raw["memory_time"])
    linear = path+temporal
    ref, full = geo["reference"], raw["full_xyz_mm"][..., :2]
    assert np.all(raw["full_xyz_mm"][..., 2] == 0)
    predictions = np.stack([ref, ref+linear, full])
    residual = target[..., :2]-ref
    coordinate_errors = predictions-target[None, ..., :2]
    node_errors = np.linalg.norm(coordinate_errors, axis=-1)
    energy = np.sum(coordinate_errors**2, axis=(-2, -1))
    denominator = float(energy[0].sum())
    assert denominator > 0
    rows = []
    for k, kind in enumerate(KINDS):
        rows.append(dict(seed=seed, prediction=kind, test_frames=len(windows),
            mean_node_mm=float(node_errors[k].mean()), nonbase_mean_node_mm=float(node_errors[k, :, 1:].mean()),
            endpoint_mm=float(node_errors[k, :, -1].mean()),
            residual_squared_energy_mm2=float(energy[k].sum()), reference_squared_energy_mm2=denominator,
            residual_energy_reduction=1-float(energy[k].sum())/denominator,
            residual_energy_reduction_pct=100*(1-float(energy[k].sum())/denominator)))
    remainder = full-(ref+linear)
    rem_node = np.linalg.norm(remainder, axis=-1)
    exact = full-ref
    branch = dict(seed=seed, test_frames=len(windows), path_time_cosine=cosine(path, temporal),
        path_residual_cosine=cosine(path, residual), time_residual_cosine=cosine(temporal, residual),
        joint_linear_residual_cosine=cosine(linear, residual), full_residual_cosine=cosine(exact, residual),
        linear_full_displacement_cosine=cosine(linear, exact),
        path_squared_displacement_mm2=float(np.sum(path**2)),
        time_squared_displacement_mm2=float(np.sum(temporal**2)),
        path_time_inner_product_mm2=float(np.sum(path*temporal)),
        linear_minus_full_mean_node_mm=rows[1]["mean_node_mm"]-rows[2]["mean_node_mm"],
        linear_full_mean_node_distance_mm=float(rem_node.mean()),
        linear_full_endpoint_distance_mm=float(rem_node[:, -1].mean()),
        linear_full_node_distance_p95_mm=float(np.quantile(rem_node, .95)),
        linear_full_relative_displacement_rms=float(np.sqrt(np.sum(remainder**2)/np.sum(exact**2))),
        reference_length_clipped_frames=int(np.any(abs(geo["length_reference"]) >= .25, axis=1).sum()),
        length_clip_affected_frames=int((~geo["unclipped"]).sum()),
        max_abs_cumulative_bend_change_rad=float(abs(geo["delta_theta"]).max()),
        max_abs_log_length_change=float(abs(geo["delta_lam"]).max()))
    # Independent autograd and central differences at eight predetermined frames.
    qa.update(jacobian_checks(core, windows[:, -1], raw["memory_joint"], geo))
    qa["analytic_J_action_max_mm"] = float(abs(np.einsum("bndg,bg->bnd", j, raw["memory_joint"])-geo["linear"]).max())
    qa["joint_float32_additivity_max_mm"] = float(abs(linear-geo["linear"]).max())
    qa["float64_geometry_vs_full_max_mm"] = float(abs(ref+geo["exact"]-full).max())
    assert qa["analytic_J_action_max_mm"] < 1e-10
    assert qa["joint_float32_additivity_max_mm"] < 1e-4
    assert qa["float64_geometry_vs_full_max_mm"] < 2e-4
    if geo["unclipped"].any():
        mask = geo["unclipped"]
        violation = np.linalg.norm(geo["exact"]-geo["linear"], axis=-1)-geo["bounds"]
        qa["taylor_bound_max_violation_mm"] = float(violation[mask].max())
        assert qa["taylor_bound_max_violation_mm"] < 1e-9
    with np.load(RUN / f"evaluation/hov/seed_{seed}/predictions.npz", allow_pickle=False) as saved:
        assert np.array_equal(saved["groups"], groups)
        assert np.array_equal(saved["frame_ids"], ids)
        qa["saved_prediction_max_difference_mm"] = float(abs(raw["full_xyz_mm"]-saved["prediction_mm"]).max())
        assert qa["saved_prediction_max_difference_mm"] < 2e-4
    metric = read_json(RUN / f"evaluation/hov/seed_{seed}/metrics.json")
    qa["saved_mean_node_difference_mm"] = abs(rows[2]["mean_node_mm"]-metric["mean_node_mm"])
    assert qa["saved_mean_node_difference_mm"] < 2e-5
    qa["seed"] = seed
    ii, jj = np.array([r["pooled_i"] for r in pairs]), np.array([r["pooled_j"] for r in pairs])
    target_delta = target[ii, :, :2]-target[jj, :, :2]
    prediction_delta = predictions[:, ii]-predictions[:, jj]
    pair_error = np.linalg.norm(prediction_delta-target_delta[None], axis=-1)
    pair_rows = []
    for tol in TOLERANCES:
        for category in CATEGORY:
            mask = np.array([p["tolerance_kpa"] == tol and p["category"] == category for p in pairs])
            if not mask.any():
                continue
            for k, kind in enumerate(KINDS):
                pair_rows.append(dict(seed=seed, tolerance_kpa=tol, category=category,
                    prediction=kind, pairs=int(mask.sum()),
                    delta_mean_node_mm=float(pair_error[k, mask].mean()),
                    delta_endpoint_mm=float(pair_error[k, mask, -1].mean()),
                    observed_delta_mean_node_mm=float(np.linalg.norm(target_delta[mask], axis=-1).mean())))
    raw.update(seed=np.array(seed), frame_ids=ids, group_index=groups,
        reference_xy_mm=ref, joint_linear_xy_mm=ref+linear,
        path_displacement_xy_mm=path, time_displacement_xy_mm=temporal,
        nonlinear_displacement_xy_mm=exact, linear_full_difference_xy_mm=remainder,
        J_reference_xy=j, reference_bend=geo["bend_reference"], reference_log_length=geo["length_reference"],
        delta_theta_rad=geo["delta_theta"], delta_log_length=geo["delta_lam"],
        unclipped=geo["unclipped"], taylor_bound_mm=geo["bounds"],
        prediction_names=np.array(KINDS), node_error_mm=node_errors,
        residual_squared_energy_per_frame_mm2=energy,
        pair_id=np.arange(len(pairs)), pair_prediction_delta_xy_mm=prediction_delta,
        pair_node_delta_error_mm=pair_error)
    np.savez_compressed(OUT / f"seed_{seed}_test_geometry.npz", **raw)
    write_json(f"seed_{seed}_metrics.json", dict(predictions=rows, geometry=branch, validation=qa, matched_pairs=pair_rows))
    return rows, branch, qa, pair_rows


def rank_metrics(matrix):
    """Uncentered rank-one energy; normalization treats each row as a curve."""
    norms = np.linalg.norm(matrix, axis=-1)
    active = norms > 1e-12
    if not active.any():
        return dict(rank1_energy=None, normalized_rank1_energy=None, active_curves=0)
    s = np.linalg.svd(matrix, compute_uv=False)
    sn = np.linalg.svd(matrix[active]/norms[active, None], compute_uv=False)
    return dict(rank1_energy=float(s[0]**2/np.sum(s**2)),
                normalized_rank1_energy=float(sn[0]**2/np.sum(sn**2)), active_curves=int(active.sum()))


def kernels_seed(seed, core, refs):
    wp, wh = readouts(core)
    alpha = array(core.maxwell.decays)
    phi = alpha[:, None]**(np.arange(19)[None]+1)
    generalized = -np.einsum("ckg,kl->cgl", wh, phi)
    geo = geometry_arrays(core, refs, np.zeros((len(refs), core.generalized_dim)))
    j = jacobian(core, geo)[:, 1:]
    kernel = np.einsum("rndg,cgl->rcnld", j, generalized)
    # Unit increment injected 0..18 lags ago, equilibrium deficit initially zero.
    unit = []
    for lag in range(19):
        deficit = -alpha.copy()
        for _ in range(lag):
            deficit *= alpha
        unit.append(deficit)
    assert np.max(abs(np.stack(unit).T+phi)) < 1e-14
    ranks, curves = [], []
    for r in range(len(refs)):
        for c in range(4):
            for d, coordinate in enumerate(("x", "y")):
                matrix = kernel[r, c, :, :, d]
                ranks.append(dict(seed=seed, reference_index=r, channel=c, coordinate=coordinate,
                                  **rank_metrics(matrix)))
                for ni, curve in enumerate(matrix):
                    peak = int(np.argmax(abs(curve)))
                    curves.append(dict(seed=seed, reference_index=r, channel=c, node=ni+1,
                        coordinate=coordinate, gain_lag0_mm_per_delta_e=float(curve[0]),
                        gain_lag18_mm_per_delta_e=float(curve[-1]),
                        peak_signed_gain_mm_per_delta_e=float(curve[peak]), peak_abs_lag=peak,
                        peak_abs_time_s=peak*.2, sampled_l2_gain=float(np.linalg.norm(curve)),
                        sign_change_count=int(np.sum(curve[1:]*curve[:-1] < 0))))
    return dict(alpha=alpha, taus_s=array(core.maxwell.taus), W_path=wp, W_time=wh,
                generalized_coordinate_scale=array(core.generalized_coordinate_scale),
                play_weights=array(core.play.weights), play_directions=array(core.pi_mode_directions),
                time_gains=array(core.maxwell_gains), time_directions=array(core.maxwell_mode_directions),
                play_thresholds=array(core.play.thresholds), phi=phi,
                generalized_kernel=generalized, J_reference=j, kernel_xy_mm_per_delta_e=kernel,
                reference_xy_mm=geo["reference"], reference_bend=geo["bend_reference"],
                reference_log_length=geo["length_reference"], bend_basis=array(core.bend_basis),
                reference_segment_lengths_mm=array(core.reference_segment_lengths),
                base_position_mm=array(core.base_position)), ranks, curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    threadpool_limits(limits=args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    # Invalidate a previous completion marker while a complete rerun is in progress.
    if (OUT / "COMPLETE.json").exists():
        (OUT / "COMPLETE.json").unlink()
    protocol = read_json(RUN / "protocol.json")
    assert protocol["seeds"] == SEEDS
    assert read_json(RUN / "status.json")["status"] == "complete"
    assert read_json(RUN / "skeleton_evaluation_complete.json")["fits"] == 286
    for seed in SEEDS:
        for folder in ("formal", "evaluation"):
            assert (RUN / f"{folder}/hov/seed_{seed}/COMPLETE").is_file()
    manifest = Path(protocol["dataset_manifest"])
    groups, inventory, meta = load_inputs(manifest)
    refs, selection = select_references(groups["train"])
    pairs, coverage = prepare_pairs(groups["test"])
    print(f"Fixed input selections: 6 references; {len(pairs)} pair/category/tolerance records", flush=True)
    windows = np.concatenate([g["windows"] for g in groups["test"]])
    target = np.concatenate([g["target"] for g in groups["test"]])
    ids = np.concatenate([g["frames"] for g in groups["test"]])
    gi = np.concatenate([np.full(len(g["frames"]), i) for i, g in enumerate(groups["test"])])
    times = np.concatenate([g["times"] for g in groups["test"]])
    assert np.all(target[..., 2] == 0), "Planar metric requires zero target z"
    assert len(set(zip(gi.tolist(), ids.tolist()))) == 2958
    with np.load(RUN / "test_targets.npz", allow_pickle=False) as d:
        for key, value in (("target_mm", target), ("groups", gi), ("frame_ids", ids), ("timestamps", times)):
            assert np.array_equal(value, d[key]), key
    ii = np.array([p["pooled_i"] for p in pairs])
    jj = np.array([p["pooled_j"] for p in pairs])
    np.savez_compressed(OUT / "test_inputs_targets.npz", windows=windows, target_xyz_mm=target,
        frame_ids=ids, group_index=gi, timestamps=times,
        group_names=np.array([g["group"] for g in groups["test"]]),
        pair_id=np.arange(len(pairs)), pair_pooled_i=ii, pair_pooled_j=jj,
        pair_observed_delta_xy_mm=target[ii, :, :2]-target[jj, :, :2])
    observation = ["Fixed three-sequence within-sequence temporal split, H20 split-local equilibrium initialization, nominal dt=0.2 s.",
        "Seed uncertainty describes optimization randomness conditional on this dataset and protocol; test frames and seeds are not independent robot experiments.",
        "Reference is the jointly trained reference inside each full HOV. Its residual includes reference approximation, measurement and visual annotation errors.",
        "The kernel is the local time-memory displacement per normalized drive increment Delta e with reference input held fixed; pressure slope and static reference response are not included.",
        "Branch directions and kernel similarity describe the fitted representation. They do not uniquely identify physical hysteresis or material relaxation mechanisms.",
        "Pair categories and tolerances overlap; disjointness is only within each sequence/category/tolerance. Matching is approximate and observations are temporally correlated."]
    write_json("protocol.json", dict(schema="unified20_geometry_v1", source_run=RUN,
        source_status=read_json(RUN / "status.json"), dataset_manifest=manifest, inventory=inventory,
        draft_section="docs/icra2027/draft.md section 3.5", seeds=SEEDS,
        test_frames=2958, training_reference_input_frames=8988,
        source_implementation=SOURCE, threads=args.threads, interop_threads=1,
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                         scipy=scipy.__version__, torch=torch.__version__),
        threadpools=threadpool_info(),
        metric="per seed: pooled mean Euclidean node error over 2958 frames and all 15 nodes in xy mm, including base; z is verified zero",
        energy="eta=1-sum_frame,node,xy((target-prediction)^2)/sum_frame,node,xy((target-reference)^2); report seed mean and sample SD ddof=1",
        path_time_cosine="dot(vec(J Wp q), vec(J Wh d))/(norm(vec(J Wp q))*norm(vec(J Wh d))) on all test frames",
        linear_full_difference="signed difference of mean-node errors and separate mean Euclidean distance between predictions",
        kernel_formula="K[s,r,c,n,l,d]=-sum_g,k J[s,r,n,d,g]*W_time[s,c,k,g]*alpha[s,k]**(l+1)",
        kernel_lags=list(range(19)), kernel_lag_seconds=(np.arange(19)*.2),
        kernel_nodes=list(range(1, 15)), kernel_coordinates=["x", "y"],
        kernel_rank="uncentered SVD, 14 nodes x 19 lags per seed/reference/channel/coordinate; row normalization excludes norms<=1e-12",
        pair_protocol=dict(source="scripts/experiments/analyze_modeling_history_mechanisms.py::select_pairs",
            tolerance_kpa=TOLERANCES, primary_tolerance_kpa=5, min_separation_frames=20,
            min_history_rms_kpa=20, direction_deadband_kpa=.3, active_channel_input_range=.01,
            categories=CATEGORY, greedy_order=["current_gap_kpa", "local_i", "local_j"],
            history="first 19 inputs within each H20 window; pressure scale=150 kPa",
            selection="pressure histories and chronology only, once shared by all seeds and prediction compositions"),
        math_provenance=["analyze_modeling_representation.py::geometry_arrays,jacobian_checks",
                         "analyze_modeling_branch_interpretation.py::metrics, time-kernel formula, reference selection",
                         "analyze_modeling_history_mechanisms.py::select_pairs"],
        observation_conditions=observation, inherited_subset_selection=meta["subset_selection"]))
    metrics, branches, validations, pair_metrics, kernel_ranks, kernel_curves, parameters = [], [], [], [], [], [], []
    provenance = []
    for seed in SEEDS:
        path = RUN / f"formal/hov/seed_{seed}/best_eval_model.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["model"] == "hov" and ckpt["config"]["seed"] == seed
        assert ckpt["config"]["study_id"] == RUN.name
        assert ckpt["config"] == read_json(path.parent / "resolved_config.json")
        model, _ = make_model("hov", ckpt["config"], normalization=(ckpt["center"], ckpt["scale"]),
                              geometry_config=ckpt["geometry_config"])
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        core = model.core
        assert core.residual_mode == "none" and core.burnin_mode == "equilibrium"
        assert core.n_bend_modes == 14 and core.generalized_dim == 16 and core.n_maxwell == 6
        rows, branch, qa, pm = analyze_seed(seed, core, model, windows, target, gi, ids, pairs)
        kp, kr, kc = kernels_seed(seed, core, refs)
        assert np.all(np.isfinite(kp["kernel_xy_mm_per_delta_e"]))
        metrics += rows
        branches.append(branch)
        validations.append(qa)
        pair_metrics += pm
        kernel_ranks += kr
        kernel_curves += kc
        parameters.append(kp)
        info = path.stat()
        provenance.append(dict(seed=seed, checkpoint=path, bytes=info.st_size, mtime_ns=info.st_mtime_ns,
            selected_epoch=ckpt["selected_epoch"], config=ckpt["config"]))
        write_json("progress.json", dict(completed_seeds=[r["seed"] for r in branches],
                                         elapsed_seconds=time.monotonic()-started))
        print(f"seed {seed}: reference={rows[0]['mean_node_mm']:.6f}, linear={rows[1]['mean_node_mm']:.6f}, "
              f"full={rows[2]['mean_node_mm']:.6f} mm; path/time cosine={branch['path_time_cosine']:.6f}", flush=True)
    kernels = {key: np.stack([p[key] for p in parameters]) for key in parameters[0]}
    kernels.update(seeds=np.array(SEEDS), reference_actions=refs,
        reference_labels=np.array(selection["reference_labels"]), channels=np.arange(4),
        node_ids=np.arange(1, 15), lags=np.arange(19), lag_seconds=np.arange(19)*.2,
        coordinates=np.array(["x", "y"]), section_intervals=np.array(core.section_intervals))
    np.savez_compressed(OUT / "kernels.npz", **kernels)
    write_csv("seed_prediction_metrics.csv", metrics)
    write_csv("seed_geometry_metrics.csv", branches)
    write_csv("matched_pair_seed_metrics.csv", pair_metrics)
    write_csv("kernel_rank.csv", kernel_ranks)
    write_csv("kernel_node_metrics.csv", kernel_curves)
    write_json("checkpoint_provenance.json", provenance)
    write_json("validation.json", dict(status="passed", per_seed=validations))
    metric_fields = [k for k in metrics[0] if k not in ("seed", "prediction", "test_frames")]
    branch_fields = [k for k in branches[0] if k not in ("seed", "test_frames")]
    rank_summary = []
    for r in range(6):
        for c in range(4):
            for d in ("x", "y"):
                part = [v for v in kernel_ranks if v["reference_index"] == r and v["channel"] == c and v["coordinate"] == d]
                active = [v for v in part if v["rank1_energy"] is not None]
                rank_summary.append(dict(reference_index=r, channel=c, coordinate=d,
                    defined_seeds=len(active), **(summarize(active, ["rank1_energy", "normalized_rank1_energy"]) if active else {})))
    pair_summary = []
    for tol in TOLERANCES:
        for category in CATEGORY:
            selected = [p for p in pairs if p["tolerance_kpa"] == tol and p["category"] == category]
            item = dict(tolerance_kpa=tol, category=category, pairs=len(selected))
            if selected:
                item["input_conditions"] = summarize(selected, ["current_gap_kpa", "history_rms_kpa", "recent_two_gap_kpa", "time_gap_s"])
                item["predictions"] = {kind: summarize([p for p in pair_metrics if p["tolerance_kpa"] == tol and p["category"] == category and p["prediction"] == kind],
                    ["delta_mean_node_mm", "delta_endpoint_mm", "observed_delta_mean_node_mm"]) for kind in KINDS}
            pair_summary.append(item)
    result = dict(schema="unified20_geometry_v1", completed_seeds=SEEDS, test_frames_per_seed=2958,
        geometry_evaluations=20*2958,
        predictions={kind: summarize([r for r in metrics if r["prediction"] == kind], metric_fields) for kind in KINDS},
        geometry=summarize(branches, branch_fields), kernel_rank_by_reference_channel_coordinate=rank_summary,
        kernel_descriptive_range={d: summarize([r for r in kernel_ranks if r["coordinate"] == d and r["rank1_energy"] is not None],
            ["rank1_energy", "normalized_rank1_energy"]) for d in ("x", "y")},
        kernel_descriptive_range_grain="480 seed/reference/channel cells per coordinate; descriptive coverage, not independent replicates",
        matched_pairs=pair_summary, observation_conditions=observation,
        validation="validation.json; saved prediction, readout, recurrence, analytic Jacobian, decoder and Taylor checks passed",
        runtime_seconds=time.monotonic()-started)
    write_json("summary.json", result)
    write_schema(kernels)
    verify_saved()
    write_handoff(result)
    write_json("COMPLETE.json", dict(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
        seeds=SEEDS, test_frames_per_seed=2958, total_seed_frames=20*2958,
        runtime_seconds=time.monotonic()-started, verification="passed"))
    print(json.dumps(plain(dict(predictions=result["predictions"], geometry=result["geometry"],
                               output=OUT, runtime_seconds=time.monotonic()-started)), indent=2), flush=True)


def write_schema(kernels):
    """Record every archive field with its actual shape and dtype."""
    schemas = {}
    for name in ("training_reference_inputs.npz", "test_inputs_targets.npz", "seed_100_test_geometry.npz"):
        with np.load(OUT / name, allow_pickle=False) as d:
            schemas[name.replace("seed_100", "seed_{seed}")] = {
                k: dict(shape=list(d[k].shape), dtype=str(d[k].dtype)) for k in d.files}
    schemas["kernels.npz"] = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in kernels.items()}
    write_json("schema.json", dict(schema="unified20_geometry_v1", archives=schemas,
        axes=dict(S="20 seeds 100..119", T="2958 pooled test frames in manifest order",
            R="6 references: training mean, PC1 q10/30/50/70/90", C="4 drive channels 0..3",
            N="15 nodes including base for predictions; 14 nodes 1..14 for kernels",
            G="16 physical generalized coordinates: 14 bend-mode radians, 2 log lengths",
            K="6 time modes", L="19 lags, 0..18; alpha exponent L+1", D="x,y planar mm",
            P="matched pair records; join pair_id to matched_pairs.csv", V="reference,joint_linear,full"),
        kernel_axes=dict(kernel_xy_mm_per_delta_e="S,R,C,N,L,D", J_reference="S,R,N,D,G",
            W_time="S,C,K,G", W_path="S,C,2,G", phi="S,K,L", generalized_kernel="S,C,G,L"),
        raw_axes=dict(J_reference_xy="T,15,2,G", node_error_mm="V,T,15",
            pair_prediction_delta_xy_mm="V,P,15,2", pair_node_delta_error_mm="V,P,15",
            residual_squared_energy_per_frame_mm2="V,T", drive="T,20,C"),
        units="geometry and Euclidean errors mm; squared energy mm^2; kernels mm/unit Delta e; lag time s; pair pressure kPa",
        json=dict(summary="seed aggregates {n,mean,sd,min,max}; sd uses ddof=1",
            seed_metrics="predictions[3], geometry, validation, matched_pairs",
            protocol="source, formulas, selection rules, versions and observation conditions",
            reference_selection="PCA normalization/loadings, selected actual inputs and source frame keys",
            validation="per-seed all-frame agreements and fixed-frame independent derivative checks",
            checkpoint_provenance="20 exact checkpoint paths, selected epochs, configs and file metadata"),
        csv=dict(seed_prediction_metrics="60 rows: seed x prediction composition",
            seed_geometry_metrics="20 rows: stacked branch cosine and linear/full discrepancy",
            matched_pairs="one fixed input-selected pair per sequence/tolerance/category; pair_id aligns NPZ",
            matched_pair_seed_metrics="seed x tolerance x category x prediction, including count and shape-difference error",
            kernel_rank="seed x reference x channel x coordinate; uncentered and row-normalized rank-one energies",
            kernel_node_metrics="seed x reference x channel x non-base node x coordinate; gain and absolute-peak lag")))


def verify_saved():
    """Recompute headline values and K=-J W alpha from delivered archives."""
    result = read_json(OUT / "summary.json")
    values = {kind: [] for kind in KINDS}
    energies = {kind: [] for kind in KINDS}
    cosines = []
    with np.load(OUT / "test_inputs_targets.npz", allow_pickle=False) as d:
        target = d["target_xyz_mm"][..., :2]
        groups, ids = d["group_index"], d["frame_ids"]
        observed = d["pair_observed_delta_xy_mm"]
    for seed in SEEDS:
        with np.load(OUT / f"seed_{seed}_test_geometry.npz", allow_pickle=False) as d:
            assert np.array_equal(groups, d["group_index"]) and np.array_equal(ids, d["frame_ids"])
            predictions = [d["reference_xy_mm"], d["joint_linear_xy_mm"], d["full_xyz_mm"][..., :2]]
            denominator = np.square(target-predictions[0]).sum()
            for kind, pred in zip(KINDS, predictions):
                values[kind].append(float(np.linalg.norm(pred-target, axis=-1).mean()))
                energies[kind].append(1-float(np.square(target-pred).sum())/denominator)
            cosines.append(cosine(d["path_displacement_xy_mm"], d["time_displacement_xy_mm"]))
            np.testing.assert_allclose(d["pair_node_delta_error_mm"],
                np.linalg.norm(d["pair_prediction_delta_xy_mm"]-observed[None], axis=-1), rtol=0, atol=1e-12)
    for kind in KINDS:
        for metric, raw in (("mean_node_mm", values[kind]), ("residual_energy_reduction", energies[kind])):
            for field, value in stats(raw).items():
                np.testing.assert_allclose(result["predictions"][kind][metric][field], value, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["geometry"]["path_time_cosine"]["mean"], np.mean(cosines), rtol=0, atol=1e-12)
    with np.load(OUT / "kernels.npz", allow_pickle=False) as d:
        assert d["kernel_xy_mm_per_delta_e"].shape == (20, 6, 4, 14, 19, 2)
        expected = -np.einsum("srndg,sckg,skl->srcnld", d["J_reference"], d["W_time"],
                              d["alpha"][:, :, None]**(d["lags"][None, None]+1))
        np.testing.assert_allclose(d["kernel_xy_mm_per_delta_e"], expected, rtol=1e-12, atol=1e-12)
    write_json("saved_artifact_verification.json", dict(status="passed", seeds=20, frames_per_seed=2958,
        checks=["20 per-seed archives and frame identities", "headline means and sample SDs from saved predictions",
                "energy reductions from squared coordinate residuals", "branch cosine from saved displacements",
                "every pair/node delta error", "all kernels from saved alpha/W/J"]))


def write_handoff(result):
    """Compact numerical handoff generated by the same reproducible run."""
    def display(value, precision=6):
        return f"{value['mean']:.{precision}f} ± {value['sd']:.{precision}f}"

    lines = ["# Unified20 HOV geometry numerical handoff", "",
        "完成 seeds 100–119；每 seed 全部 2958 测试帧。以下 ± 为20次训练的样本标准差（ddof=1）。",
        "骨架误差按15节点（含基点）逐帧汇总，坐标为平面 mm。", "",
        "| 预测组成 | 骨架误差/mm | 末端误差/mm | 残差平方能量减少率/% |",
        "|---|---:|---:|---:|"]
    for kind in KINDS:
        row = result["predictions"][kind]
        lines.append(f"| {kind} | {display(row['mean_node_mm'])} | {display(row['endpoint_mm'])} | {display(row['residual_energy_reduction_pct'])} |")
    geometry = result["geometry"]
    lines += ["", f"测试集堆叠路径/时间位移余弦：{display(geometry['path_time_cosine'])}。",
        f"一阶减 full 的骨架误差差值：{display(geometry['linear_minus_full_mean_node_mm'])} mm。",
        f"一阶与 full 预测间的平均节点距离：{display(geometry['linear_full_mean_node_distance_mm'])} mm。",
        "这组观察支持局部一阶几何读出保留了大部分预测收益，且两分支的全局修正方向重合较小。", "",
        "参考输入固定为8988个训练窗口当前输入的均值，以及标准化PC1分数q10/30/50/70/90附近的实际输入；选择索引和PCA符号规则已保存。",
        "时间核 K=-J W_time alpha^(lag+1)，单位mm/单位归一化驱动增量；覆盖6参考×4通道×14非基点×19 lag（0–3.6 s）×xy。",
        "每个参考/通道/坐标的20-seed汇总见summary.json；完整节点增益与绝对峰值lag见kernel_node_metrics.csv。", ""]
    for coord in ("x", "y"):
        row = result["kernel_descriptive_range"][coord]
        rank, norm = row["rank1_energy"], row["normalized_rank1_energy"]
        lines.append(f"{coord}坐标480个seed/参考/通道单元：原始秩一能量平均{100*rank['mean']:.3f}%（范围{100*rank['min']:.3f}–{100*rank['max']:.3f}%）；逐节点曲线L2归一化后平均{100*norm['mean']:.3f}%。这些单元用于描述覆盖范围，独立重复数仍为20。")
    lines += ["", "固定5 kPa协议的配对结果（全部1/2/5/10 kPa容差保存在summary.json）：", "",
        "| 条件 | 帧对数 | 当前输入最大通道差均值/kPa | 历史RMS差均值/kPa | reference形态差值误差/mm | full形态差值误差/mm |",
        "|---|---:|---:|---:|---:|---:|"]
    for row in result["matched_pairs"]:
        if row["tolerance_kpa"] == 5 and row["pairs"]:
            ic, p = row["input_conditions"], row["predictions"]
            lines.append(f"| {row['category']} | {row['pairs']} | {ic['current_gap_kpa']['mean']:.6f} | {ic['history_rms_kpa']['mean']:.6f} | {display(p['reference']['delta_mean_node_mm'])} | {display(p['full']['delta_mean_node_mm'])} |")
    lines += ["", "在上述固定5 kPa条件下，反向组的形态差值预测改善；同向和最近两步相近组的full误差略高于reference。结论依赖历史条件与当前观测覆盖。", "",
        "观察条件：", ""] + [f"- {v}" for v in result["observation_conditions"]]
    lines += ["", "输出文件与schema：", "",
        "- `summary.json`：三种预测组成、位移几何、时间核秩和配对的汇总；数值统计为{n,mean,sd,min,max}。",
        "- `seed_{100..119}_test_geometry.npz`：逐测试帧状态q/d、真实驱动、物理广义坐标、J、预测和分支位移、误差/能量及配对预测差值。",
        "- `seed_{100..119}_metrics.json`、`seed_prediction_metrics.csv`、`seed_geometry_metrics.csv`：每seed数值及QA。",
        "- `kernels.npz`：kernel[20,6,4,14,19,2]；alpha[20,6]；W_time[20,4,6,16]；J_reference[20,6,14,2,16]，以及参数因子、参考几何与轴标识。",
        "- `kernel_rank.csv`、`kernel_node_metrics.csv`：960个秩记录、13440个节点/坐标增益与峰值记录。",
        "- `test_inputs_targets.npz`、`training_reference_inputs.npz`、`reference_selection.json`：原始分析输入、目标和固定参考选择。",
        "- `matched_pairs.csv`、`matched_pair_coverage.json`、`matched_pair_seed_metrics.csv`：输入配对索引、候选/选择覆盖、每seed差值误差。",
        "- `protocol.json`、`checkpoint_provenance.json`：公式、协议、软件环境、checkpoint路径/选中epoch/文件元数据。",
        "- `schema.json`：逐NPZ字段的精确shape/dtype、轴含义和单位；`validation.json`及`saved_artifact_verification.json`：数值核验。",
        "- `COMPLETE.json`：成功完成标记。", "",
        "复现（cwd `/Data5/ddf/projects/SelfSoftRobot`）：", "", "```bash",
        "/Data5/ddf/environments/conda_envs/selfsr/bin/python -B scripts/experiments/analyze_unified20_geometry.py --threads 1",
        "```", "", "运行使用004冻结source的模型实现；逐seed full预测与004保存预测完全一致，解析导数通过autograd/有限差分核验，保存的alpha/W/J可重构全部时间核。", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


# Reused pure mathematical functions, copied from the sources named below.


# Mathematical source: scripts/experiments/analyze_modeling_representation.py::geometry_arrays
def geometry_arrays(core, action, memory):
    """Physical reference, exact displacement and analytic J_ref @ memory.

    Uses the calibrated reference, including static pressure pair terms, and
    exactly the decoder's piecewise bounded length convention.
    """
    with torch.inference_mode():
        b, ell = core._reference(torch.as_tensor(action, dtype=torch.float32))
    b, ell = b.numpy().astype(float), ell.numpy().astype(float)
    basis = core.bend_basis.detach().numpy().astype(float)
    reference_lengths = core.reference_segment_lengths.detach().numpy().astype(float)
    section = np.repeat(np.arange(core.n_sections), core.section_intervals)
    theta = np.cumsum(b, axis=1)
    lam = ell[:, section]
    lengths = reference_lengths * np.exp(np.clip(lam, -.25, .25))
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    delta_theta = np.cumsum(memory[:, :core.n_bend_modes] @ basis.T, axis=1)
    delta_lam = memory[:, core.n_bend_modes:][:, section]
    derivative_mask = (lam > -.25) & (lam < .25)
    first_segments = lengths[:, :, None] * (
        tangent * (delta_lam * derivative_mask)[:, :, None]
        + normal * delta_theta[:, :, None])
    new_lengths = reference_lengths * np.exp(np.clip(lam + delta_lam, -.25, .25))
    new_theta = theta + delta_theta
    actual_segments = new_lengths[:, :, None] * np.stack(
        [np.cos(new_theta), np.sin(new_theta)], axis=-1) - lengths[:, :, None] * tangent

    def integrate(x):
        return np.concatenate([np.zeros((len(x), 1, 2)), np.cumsum(x, axis=1)], axis=1)

    unclipped = np.all((np.abs(lam) < .25) & (np.abs(lam + delta_lam) < .25), axis=1)
    # Integral Taylor remainder: |exp(z)-1-z| <= |z|^2 exp(max(Re z,0))/2.
    segment_bound = .5 * lengths * (delta_theta ** 2 + delta_lam ** 2) * np.exp(np.maximum(delta_lam, 0))
    bounds = np.column_stack([np.zeros(len(memory)), np.cumsum(segment_bound, axis=1)])
    reference = integrate(lengths[:, :, None] * tangent) + core.base_position.detach().numpy()[:2]
    return dict(reference=reference, exact=integrate(actual_segments), linear=integrate(first_segments),
                bounds=bounds, unclipped=unclipped, delta_theta=delta_theta,
                delta_lam=delta_lam, bend_reference=b, length_reference=ell)


# Mathematical source: scripts/experiments/analyze_modeling_representation.py::jacobian_checks
def jacobian_checks(core, a, m, geo):
    """Independent torch autograd and finite difference checks in float64."""
    indices = np.linspace(0, len(a) - 1, 8, dtype=int)
    max_autograd = max_finite = max_decoder = 0.
    basis = core.bend_basis.detach().double()
    ref_lengths = core.reference_segment_lengths.detach().double()
    for idx in indices:
        b = torch.from_numpy(geo["bend_reference"][idx:idx + 1])
        ell = torch.from_numpy(geo["length_reference"][idx:idx + 1])
        mem = torch.from_numpy(m[idx])

        def decode(v):
            return generalized_to_skeleton(
                b + (v[:core.n_bend_modes] @ basis.T)[None],
                ell + v[core.n_bend_modes:][None], ref_lengths,
                core.section_intervals, core.base_position.detach().double())[0, :, :2]

        zero = torch.zeros_like(mem)
        jac = torch.autograd.functional.jacobian(decode, zero)
        auto = (jac @ mem).detach().numpy()
        eps = 1e-3
        finite = ((decode(eps * mem) - decode(-eps * mem)) / (2 * eps)).detach().numpy()
        exact = (decode(mem) - decode(zero)).detach().numpy()
        max_autograd = max(max_autograd, float(np.max(np.abs(auto - geo["linear"][idx]))))
        max_finite = max(max_finite, float(np.max(np.abs(finite - geo["linear"][idx]))))
        max_decoder = max(max_decoder, float(np.max(np.abs(exact - geo["exact"][idx]))))
    assert max_autograd < 1e-8
    assert max_finite < 1e-5
    assert max_decoder < 1e-8
    return dict(checked_windows=8, analytic_vs_autograd_max_mm=max_autograd,
                analytic_vs_finite_difference_max_mm=max_finite,
                numpy_vs_torch_decoder_max_mm=max_decoder)


# Mathematical source: scripts/experiments/analyze_modeling_history_mechanisms.py::select_pairs
def select_pairs(group, tolerance):
    w = group['windows']
    active = np.ptp(group['actions'], axis=0) > .01
    current = w[:, -1, active].astype(np.float64)*150
    # Matching uses pressure only; disjoint greedy selection uses current-input
    # distance and chronology, never any measured or predicted shape.
    pairs = cKDTree(current).query_pairs(tolerance, p=np.inf, output_type='ndarray')
    if not len(pairs):
        return {k: [] for k in CATEGORY}, {k: 0 for k in CATEGORY}
    pairs = pairs[np.abs(pairs[:, 1]-pairs[:, 0]) >= 20]
    candidates = {k: [] for k in CATEGORY}
    for i, j in pairs:
        hist_gap = float(np.sqrt(np.mean((w[i, :-1, active]-w[j, :-1, active])**2))*150)
        if hist_gap < 20:
            continue
        now_gap = float(np.max(np.abs(current[i]-current[j])))
        delta_i = (w[i, -1, active]-w[i, -2, active])*150
        delta_j = (w[j, -1, active]-w[j, -2, active])*150
        direction_i = np.where(np.abs(delta_i) >= .3, np.sign(delta_i), 0)
        direction_j = np.where(np.abs(delta_j) >= .3, np.sign(delta_j), 0)
        row = dict(i=int(i), j=int(j), current_gap_kpa=now_gap,
                   history_rms_kpa=hist_gap,
                   recent_two_gap_kpa=float(np.max(np.abs(w[i,-2:]-w[j,-2:]))*150),
                   time_gap_s=float(abs(group['times'][j]-group['times'][i])))
        if np.any(direction_i*direction_j < 0):
            candidates['opposite'].append(row)
        if np.array_equal(direction_i, direction_j):
            candidates['same_direction'].append(row)
        if row['recent_two_gap_kpa'] <= tolerance:
            candidates['same_recent_two'].append(row)
    selected = {}
    for key, rows in candidates.items():
        used, selected[key] = set(), []
        for row in sorted(rows, key=lambda r: (r['current_gap_kpa'], r['i'], r['j'])):
            if row['i'] not in used and row['j'] not in used:
                used.update((row['i'], row['j']))
                selected[key].append(row)
    return selected, {k: len(v) for k, v in candidates.items()}


if __name__ == "__main__":
    main()
