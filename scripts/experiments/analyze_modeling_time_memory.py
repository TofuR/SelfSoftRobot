#!/usr/bin/env python3
"""Frozen-checkpoint time-memory diagnostics; all outputs stay in the time task.

Run with the selfsr Python environment from the repository root.  No training,
source-data mutation, interpolation of labels, or checkpoint selection occurs.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/selfsr_time_memory_mpl")

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmarks.modeling_models import make_model  # noqa: E402

STUDY = ROOT / "workspace/runs/training/modeling_three_seq_20260913_001928"
OUT = ROOT / "workspace/runs/analysis/modeling_mechanisms_20260913_001/time"
REPORT = ROOT / "workspace/reports/modeling_mechanisms_20260913_001"
RAW = ROOT / "workspace/data/raw/real"
PROCESSED = ROOT / "workspace/data/processed/real"
SEEDS = range(5)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(plain(rows))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def checkpoint(name="hov", seed=0):
    path = STUDY / "formal" / name / f"seed_{seed}" / "best_eval_model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = make_model(saved["model"], saved["config"],
                          normalization=(saved["center"], saved["scale"]),
                          geometry_config=saved["geometry_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), path


def raw_timing(seq):
    root = RAW / seq
    data = np.loadtxt(root / "actions6.csv", delimiter=",", skiprows=1)
    meta = read_json(root / "meta.json")
    result = dict(seq=seq, raw_path=str(root), meta=meta, time=data[:, 0],
                  actions=data[:, [1, 2, 4, 6]] / 150.)
    if (root / "samples.csv").exists() and (root / "commands.csv").exists():
        samples = read_csv(root / "samples.csv")
        commands = read_csv(root / "commands.csv")
        by_id = {r["command_id"]: r for r in commands}
        rows = [by_id[r["command_id"]] for r in samples]
        assert len(samples) == len(data)
        assert np.array_equal([int(r["frame_idx"]) for r in samples], np.arange(len(data)))
        exposure = np.array([float(r["t_grab"]) - float(r.get("frame_age0", r["frame_age"]))
                             for r in samples])
        issue = np.array([float(r["t_command"]) for r in rows])
        ack = np.array([float(r["t_command_ack"]) for r in rows])
        ack_ok = np.array([r["communication_status"] == "ack" for r in rows])
        issued = np.array([[float(r[f"action_command{i}"]) for i in [0, 1, 3, 5]] for r in rows])
        assert np.max(np.abs(issued / 150 - result["actions"])) < 1e-5
        result.update(exposure=exposure, issue=issue, ack=ack,
                      valid=ack_ok & (exposure >= ack - 1e-6), ack_ok=ack_ok)
        result["timing_summary"] = dict(
            sequence=seq, frames=len(data), nominal_dt_s=meta["action_interval_s"],
            actual_mean_hz=float(1 / np.mean(np.diff(data[:, 0]))),
            grab_interval_median_s=float(np.median(np.diff(data[:, 0]))),
            grab_interval_p05_s=float(np.quantile(np.diff(data[:, 0]), .05)),
            grab_interval_p95_s=float(np.quantile(np.diff(data[:, 0]), .95)),
            camera_after_ack_median_s=float(np.median(exposure - ack)),
            camera_before_ack_frames=int(np.sum(exposure < ack - 1e-6)),
            non_ack_frames=int(np.sum(~ack_ok)),
            command_to_ack_median_s=float(np.median(ack - issue)),
            camera_before_issue_frames=int(np.sum(exposure < issue - 1e-6)))
    return result


def native_data():
    """Use source train+val to recover all frames, not the incomplete aggregate."""
    result, inventory = [], []
    combined = PROCESSED / "seq_20260819_10hz_n15_sam2_robot_mm"
    for suffix in ("182253", "182519"):
        seq = f"seq_20260819_{suffix}"
        source = PROCESSED / f"{seq}_n15_sam2_robot_mm"
        manifest = read_json(source / "dataset_manifest.json")
        arrays, inputs, camera, source_files = [], [], [], []
        combined_difference = 0.
        for role in ("train", "val"):
            path = source / role / f"{seq}_{role}.npz"
            source_files.append(str(path))
            with np.load(path, allow_pickle=False) as d:
                y = d["positions"].transpose(0, 2, 1)
                arrays.append(y.copy())
                camera.append(d["positions_camera_px"].transpose(0, 2, 1).copy())
                inputs.append(d["actions"][:, d["model_action_channels"]].copy())
                old = combined / role / path.name
                if old.exists():
                    with np.load(old, allow_pickle=False) as c:
                        combined_difference = max(combined_difference,
                            float(np.max(np.abs(d["positions"] - c["positions"]))))
        record = raw_timing(seq)
        y, x = np.concatenate(arrays), np.concatenate(inputs)
        assert len(y) == len(record["time"])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        discrepancy = float(np.max(np.abs(x - record["actions"])) * 150)
        assert discrepancy < .001
        qc = read_csv(source / "qc_skeleton/skeleton_metrics.csv")
        assert len(qc) == len(y)
        interpolated = np.array([r["interpolated"].lower() == "true" for r in qc])
        invalid = np.array([r["hard_invalid"].lower() == "true" for r in qc])
        assert np.array_equal([int(r["frame"]) for r in qc], np.arange(len(y)))
        record.update(actions=x, positions=y, camera=np.concatenate(camera),
                      label_valid=~interpolated & ~invalid, source_files=source_files)
        result.append(record)
        inventory.append(dict(sequence=seq, nominal_hz=10, frames=len(y),
            label_source="existing single-frame SAM2 centerlines; not new relabel run",
            label_interpolated_frames=int(interpolated.sum()),
            label_hard_invalid_frames=int(invalid.sum()),
            source_to_combined_max_difference_mm=combined_difference,
            action_to_raw_max_difference_kpa=discrepancy,
            source_files=source_files, manifest=str(source / "dataset_manifest.json"),
            calibration="fixed 1.25 px/mm, base (370,110), planar",
            use="frozen-model diagnostic on recordings absent from formal three-sequence fit",
            generated_at=manifest.get("created_at")))
    return result, inventory


def formal_data():
    meta = read_json(STUDY / "data/dataset_manifest.json")
    sequences = []
    for row in meta["files"]:
        if row["role"] != "test":
            continue
        with np.load(row["path"], allow_pickle=False) as d:
            sequences.append(dict(seq=row["group"], **{k: d[k].copy() for k in d.files}))
    return sequences


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float32)


def windows(actions, endpoints, history, stride=1):
    ids = endpoints[:, None] - np.arange(history - 1, -1, -1)[None, :] * stride
    return actions[ids], ids


def initial(core, action):
    e = core.drive(action)
    return e[..., None].expand(-1, -1, core.n_play).clone(), e[..., None].expand(
        -1, -1, core.n_maxwell).clone()


def propagate_h(core, h, e, elapsed):
    elapsed = tensor(elapsed).reshape(-1, 1, 1)
    assert bool(torch.all(elapsed >= -1e-6))
    decay = torch.exp(-elapsed.clamp_min(0) / core.maxwell.taus)
    return decay * h + (1 - decay) * e[..., None]


@torch.inference_mode()
def consume(core, actions, dt=.2, switch=None, exposure=None, initialization="equilibrium"):
    """Window starts at first exposure; no initial artificial action interval.

    nominal: u_j acts for dt before its target; event: switch timestamp is a
    commanded/acknowledged ZOH boundary, readout is at actual image exposure.
    For decimation, omitted commands are unobserved; coarse ZOH is an estimate.
    """
    p, h = initial(core, actions[:, 0])
    if initialization == "rest":
        p.zero_()
        h.zero_()
    eprev = core.drive(actions[:, 0])
    if switch is None:
        for j in range(1, actions.shape[1]):
            e = core.drive(actions[:, j])
            p, _ = core.play.step(p, e)
            h = propagate_h(core, h, e, dt)
    else:
        # All windows use exactly the same first/last image exposure in both
        # views. Commands between retained ones are deliberately not supplied.
        last = exposure[:, 0]
        for j in range(1, actions.shape[1]):
            boundary = switch[:, j]
            h = propagate_h(core, h, eprev, boundary - last)
            e = core.drive(actions[:, j])
            p, _ = core.play.step(p, e)
            eprev, last = e, boundary
        h = propagate_h(core, h, eprev, exposure[:, -1] - last)
    e = core.drive(actions[:, -1])
    return core._state_output(actions[:, -1], p, h, e[..., None] - p, e), p, h


def physical(core, value):
    return value * core.pc_scale + core.pc_center


def errors(pred, target):
    d = np.linalg.norm(pred - target, axis=-1)
    return d.mean(-1), d[:, -1]


def summary_rows(rows, keys, metrics):
    buckets = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        buckets.setdefault(key, []).append(row)
    out = []
    for key, bucket in buckets.items():
        item = dict(zip(keys, key))
        for metric in metrics:
            values = np.array([r[metric] for r in bucket], dtype=float)
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        item["repetitions"] = len(bucket)
        out.append(item)
    return out


def spot_overlays(native):
    import cv2
    panels = []
    rows = []
    for seq in native:
        for idx in np.linspace(0, len(seq["actions"]) - 1, 3, dtype=int):
            path = RAW / seq["seq"] / "cam0" / f"{idx:05d}.png"
            im = cv2.imread(str(path))
            if im is None:
                raise FileNotFoundError(path)
            pts = np.rint(seq["camera"][idx, :, :2]).astype(np.int32)
            cv2.polylines(im, [pts], False, (0, 180, 255), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(im, tuple(pt), 2, (220, 70, 40), -1)
            crop = im[68:368, 220:520]
            cv2.putText(crop, f"{seq['seq'][-6:]} frame {idx}", (7, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, .48, (25, 25, 25), 1, cv2.LINE_AA)
            panels.append(crop)
            rows.append(dict(sequence=seq["seq"], frame=int(idx), image_path=str(path),
                             selection="first/middle/last by index; no error-based selection"))
    cv2.imwrite(str(OUT / "native10hz_label_spotcheck.png"),
                np.concatenate([np.concatenate(panels[:3], 1), np.concatenate(panels[3:], 1)], 0))
    return rows


@torch.inference_mode()
def native_comparison(native):
    metrics, frame_rows, alignment = [], [], []
    variants = [("nominal_10Hz_H39", 39, 1, .1, None),
                ("decimated_5Hz_H20", 20, 2, .2, None),
                ("event_issue_10Hz_H39", 39, 1, None, "issue"),
                ("event_issue_5Hz_H20", 20, 2, None, "issue"),
                ("event_ack_10Hz_H39", 39, 1, None, "ack"),
                ("event_ack_5Hz_H20", 20, 2, None, "ack"),
                ("wrong_dt_10Hz_H39_dt0.2", 39, 1, .2, None)]
    for s in native:
        ends = np.arange(38, len(s["actions"]))
        ids = ends[:, None] - np.arange(38, -1, -1)[None, :]
        valid = s["valid"][ids].all(1) & s["label_valid"][ends]
        # No command is moved backwards through the initializing exposure.
        valid &= (s["issue"][ids[:, 1]] >= s["exposure"][ids[:, 0]] - 1e-6)
        ends = ends[valid]
        s["scored_ends"] = ends
        alignment.append(dict(sequence=s["seq"], available_frames=len(s["actions"]),
            excluded_initial_context=38, excluded_timing_or_label_windows=int((~valid).sum()),
            scored_common_frames=len(ends), even_endpoint_frames=int((ends % 2 == 0).sum()),
            odd_endpoint_frames=int((ends % 2 == 1).sum()),
            nominal_span_s=3.8,
            actual_exposure_span_mean_s=float(np.mean(s["exposure"][ends] - s["exposure"][ends - 38])),
            actual_exposure_span_p05_s=float(np.quantile(s["exposure"][ends] - s["exposure"][ends - 38], .05)),
            actual_exposure_span_p95_s=float(np.quantile(s["exposure"][ends] - s["exposure"][ends - 38], .95))))
    for name in ("hov", "hov_no_maxwell"):
        for seed in SEEDS:
            model, _ = checkpoint(name, seed)
            core = model.core
            for s in native:
                ends = s["scored_ends"]
                for variant, history, stride, dt, boundary in variants:
                    if variant.startswith("wrong_dt") and (name != "hov" or seed != 0):
                        continue
                    all_pred = []
                    for start in range(0, len(ends), 256):
                        end = ends[start:start + 256]
                        a, ids = windows(s["actions"], end, history, stride)
                        output, _, _ = consume(core, tensor(a), dt=dt,
                            switch=None if boundary is None else s[boundary][ids],
                            exposure=None if boundary is None else s["exposure"][ids])
                        all_pred.append(physical(core, output["skeleton"]).numpy())
                    pred = np.concatenate(all_pred)
                    node, tip = errors(pred, s["positions"][ends])
                    metrics.append(dict(sequence=s["seq"], model=name, seed=seed, protocol=variant,
                        frames=len(ends), node_mean_mm=float(node.mean()), endpoint_mean_mm=float(tip.mean()),
                        node_p95_mm=float(np.quantile(node, .95))))
                    # Full native per-frame errors are the long-form plotting source.
                    frame_rows.extend(dict(sequence=s["seq"], model=name, seed=seed, protocol=variant,
                        frame=int(end), exposure_s=float(s["exposure"][end]),
                        node_mm=float(node[k]), endpoint_mm=float(tip[k])) for k, end in enumerate(ends))
                    if seed == 0 and variant in ("nominal_10Hz_H39", "decimated_5Hz_H20"):
                        np.savez_compressed(OUT / f"{s['seq']}_{name}_{variant}_seed0.npz",
                            frame_ids=ends, prediction_mm=pred, target_mm=s["positions"][ends],
                            exposure_s=s["exposure"][ends])
            print(f"native completed {name} seed={seed}", flush=True)
    pooled = []
    for name in ("hov", "hov_no_maxwell"):
        for seed in SEEDS:
            for variant, *_ in variants:
                subset = [r for r in metrics if r["model"] == name and r["seed"] == seed and r["protocol"] == variant]
                if not subset:
                    continue
                total = sum(r["frames"] for r in subset)
                pooled.append(dict(model=name, seed=seed, protocol=variant, frames=total,
                    node_mean_mm=sum(r["node_mean_mm"] * r["frames"] for r in subset) / total,
                    endpoint_mean_mm=sum(r["endpoint_mean_mm"] * r["frames"] for r in subset) / total))
    write_csv("native10hz_per_frame", frame_rows)
    write_csv("native10hz_per_sequence_seed", metrics)
    write_csv("native10hz_pooled_seed", pooled)
    return pooled, metrics, alignment


@torch.inference_mode()
def formal_states(formal):
    spectrum, halfstep, duration, hold_rows, geometry_rows, initial_rows = [], [], [], [], [], []
    consistency = []
    baseline_rows = []
    representative = []
    correlation_rows, cancellation_rows = [], []
    for seed in SEEDS:
        model, _ = checkpoint("hov", seed)
        core = model.core
        sum_node = 0.
        count = 0
        all_d, all_components, all_q, all_actions = [], [], [], []
        seed_half = []
        for s in formal:
            ends = np.arange(19, len(s["actions"]))
            picks = set(np.linspace(0, len(ends) - 1, min(24, len(ends)), dtype=int).tolist())
            for start in range(0, len(ends), 256):
                end = ends[start:start + 256]
                a, _ = windows(s["actions"], end, 20)
                a = tensor(a)
                out, p, h = consume(core, a, dt=.2)
                ref = core(a)
                maxdiff = float(torch.max(torch.abs(out["skeleton"] - ref["skeleton"])))
                consistency.append(maxdiff)
                pred = physical(core, out["skeleton"]).numpy()
                node, _ = errors(pred, s["positions"][end])
                sum_node += float(node.sum())
                count += len(end)
                e = core.drive(a[:, -1])
                d, q = h - e[..., None], e[..., None] - p
                all_d.append(d)
                all_q.append(q)
                all_components.append(out["maxwell_generalized_components"])
                all_actions.append(a[:, -1])
                # One 0.2-second constant-input update equals two 0.1 updates;
                # only the first input initializes equilibrium, so H20->H39.
                fine = torch.cat([a[:, :1], a[:, 1:].repeat_interleave(2, dim=1)], dim=1)
                fine_out, pf, hf = consume(core, fine, dt=.1)
                half_error = torch.linalg.vector_norm(
                    physical(core, out["skeleton"]) - physical(core, fine_out["skeleton"]), dim=-1)
                seed_half.append((float(half_error.sum()), half_error.numel(),
                    float(half_error.max()), float(torch.max(torch.abs(h - hf))),
                    float(torch.max(torch.abs(p - pf)))))
                if seed == 0:
                    # Log-initialization diagnostic; explicitly not a new fit.
                    rest_out, _, _ = consume(core, a, dt=.2, initialization="rest")
                    rest_pred = physical(core, rest_out["skeleton"]).numpy()
                    rn, _ = errors(rest_pred, s["positions"][end])
                    initial_rows.extend(dict(sequence=s["seq"], frame=int(s["frame_ids"][end[j]]),
                        equilibrium_mm=float(node[j]), zero_init_mm=float(rn[j])) for j in range(len(end)))
                    for j in range(len(end)):
                        if start + j in picks:
                            representative.append(dict(sequence=s["seq"], frame=int(s["frame_ids"][end[j]])))
                # All samples enter measured branch amplitude, no winner cases.
        all_d, all_q = torch.cat(all_d), torch.cat(all_q)
        all_components, all_actions = torch.cat(all_components), torch.cat(all_actions)
        pi, tm, _, _ = core._structured_memory(all_q, all_d)
        # Similar exponentials are a useful basis, not independently identified
        # material modes. Report their observed collinearity and cancellation.
        for c in range(core.action_dim):
            corr = np.corrcoef(all_d[:, c].numpy(), rowvar=False)
            for i in range(core.n_maxwell):
                for j in range(core.n_maxwell):
                    correlation_rows.append(dict(seed=seed, channel=c, tau_a=i+1, tau_b=j+1,
                        tau_a_s=float(core.maxwell.taus[i]), tau_b_s=float(core.maxwell.taus[j]),
                        pearson_r=float(corr[i, j])))
        normalized_components = all_components / core.generalized_coordinate_scale
        normalized_net = tm / core.generalized_coordinate_scale
        component_rms_sum = float(torch.sqrt(torch.mean(
            torch.sum(normalized_components**2, dim=-1), dim=0)).sum())
        net_rms = float(torch.sqrt(torch.mean(torch.sum(normalized_net**2, dim=-1))))
        cancellation_rows.append(dict(seed=seed, component_rms_sum=component_rms_sum,
            net_rms=net_rms, component_to_net_rms_ratio=component_rms_sum/max(net_rms, 1e-12),
            coordinates="checkpoint-normalized 14 bend plus 2 log-length coordinates"))
        full = physical(core, core._decode_generalized(all_actions, pi + tm))
        no_time = physical(core, core._decode_generalized(all_actions, pi))
        effect = torch.linalg.vector_norm(full - no_time, dim=-1)
        baseline_rows.append(dict(seed=seed, frames=count, node_mean_mm=sum_node/count,
                                  time_correction_mean_node_mm=float(effect.mean()),
                                  time_correction_tip_mm=float(effect[:, -1].mean())))
        halfstep.append(dict(seed=seed, windows=count, H_coarse=20, H_fine=39,
            physical_duration_s=3.8, mean_node_difference_mm=sum(x[0] for x in seed_half)/sum(x[1] for x in seed_half),
            max_node_difference_mm=max(x[2] for x in seed_half),
            max_h_difference=max(x[3] for x in seed_half), max_p_difference=max(x[4] for x in seed_half)))
        for k, tau in enumerate(core.maxwell.taus):
            component = all_components[:, :, k].sum(1)
            removed = physical(core, core._decode_generalized(all_actions, pi + tm - component))
            influence = torch.linalg.vector_norm(full - removed, dim=-1)
            for c in range(core.action_dim):
                scale = core.generalized_coordinate_scale
                coeff = core.maxwell_gains[c, k] * core.maxwell_mode_directions[c, k] * scale
                spectrum.append(dict(seed=seed, channel=c, tau_index=k, tau_s=float(tau),
                    state_rms=float(torch.sqrt(torch.mean(all_d[:, c, k] ** 2))),
                    bend_readout_l2_rad=float(torch.linalg.vector_norm(coeff[:core.n_bend_modes])),
                    length_readout_l2_log=float(torch.linalg.vector_norm(coeff[core.n_bend_modes:])),
                    bend_contribution_rms_rad=float(torch.sqrt(torch.mean(all_components[:, c, k, :core.n_bend_modes] ** 2))),
                    length_contribution_rms_log=float(torch.sqrt(torch.mean(all_components[:, c, k, core.n_bend_modes:] ** 2)))))
            for n in range(15):
                geometry_rows.append(dict(seed=seed, tau_index=k, tau_s=float(tau), node=n+1,
                    removal_mean_displacement_mm=float(influence[:, n].mean())))
        if seed == 0:
            # Deterministic, equally spaced initial states from each test sequence.
            selected = []
            offset = 0
            for s in formal:
                num = len(s["actions"]) - 19
                selected.extend((offset + np.linspace(0, num-1, min(24, num), dtype=int)).tolist())
                offset += num
            d0, q0, a0 = all_d[selected], all_q[selected], all_actions[selected]
            initial_effect = effect[selected]
            for t in np.arange(0., 8.0001, .1):
                decay = torch.exp(-float(t) / core.maxwell.taus)
                pi0, tm0, _, _ = core._structured_memory(q0, d0 * decay)
                y = physical(core, core._decode_generalized(a0, pi0 + tm0))
                asymptote = physical(core, core._decode_generalized(a0, pi0))
                geom = torch.linalg.vector_norm(y-asymptote, dim=-1)
                for k, tau in enumerate(core.maxwell.taus):
                    hold_rows.append(dict(elapsed_s=float(t), tau_s=float(tau), tau_index=k,
                        expected_remaining_fraction=float(decay[k]),
                        measured_state_rms=float(torch.sqrt(torch.mean((d0[..., k] * decay[k])**2))),
                        start_state_rms=float(torch.sqrt(torch.mean(d0[..., k]**2))),
                        initial_states=len(selected), evidence="counterfactual_constant_command_model_response"))
                duration.append(dict(elapsed_s=float(t), initial_states=len(selected),
                    mean_node_time_correction_mm=float(geom.mean()),
                    p10_node_time_correction_mm=float(torch.quantile(geom.mean(1), .1)),
                    p90_node_time_correction_mm=float(torch.quantile(geom.mean(1), .9)),
                    mean_tip_time_correction_mm=float(geom[:, -1].mean()),
                    initial_mean_node_correction_mm=float(initial_effect.mean())))
    write_csv("tau_channel_spectrum", spectrum)
    write_csv("tau_node_geometric_effect", geometry_rows)
    write_csv("zoh_halfstep_consistency", halfstep)
    write_csv("counterfactual_hold_state_decay", hold_rows)
    write_csv("counterfactual_hold_geometry", duration)
    write_csv("initialization_sensitivity_seed0", initial_rows)
    write_csv("tau_state_correlations", correlation_rows)
    write_csv("time_readout_cancellation", cancellation_rows)
    return dict(spectrum=spectrum, geometry=geometry_rows, halfstep=halfstep,
                hold=hold_rows, duration=duration, baseline=baseline_rows,
                initialization=initial_rows, selected_states=representative,
                correlations=correlation_rows, cancellation=cancellation_rows,
                reconstruction_max_normalized_difference=max(consistency))


def hold_inventory():
    """All-channel range-bounded contiguous runs, never a per-step drift rule."""
    rows, runs = [], []
    for path in sorted(RAW.glob("*/meta.json")):
        raw = np.loadtxt(path.parent / "actions6.csv", delimiter=",", skiprows=1)
        x, t = raw[:, 1:], raw[:, 0]
        for tolerance in (.001, .5, 2.):
            start, segments = 0, []
            for j in range(1, len(x)):
                if np.max(np.ptp(x[start:j+1], axis=0)) > tolerance:
                    segments.append((start, j-1))
                    start = j
            segments.append((start, len(x)-1))
            durations = [t[b]-t[a] for a, b in segments]
            accepted = [(a, b) for a, b in segments if t[b]-t[a] >= 1.]
            rows.append(dict(sequence=path.parent.name, command_range_tolerance_kpa=tolerance,
                frames=len(x), hold_min_duration_s=1., hold_count=len(accepted),
                longest_segment_s=float(max(durations))))
            for a, b in accepted:
                runs.append(dict(sequence=path.parent.name, tolerance_kpa=tolerance,
                    first_frame=a, last_frame=b, duration_s=float(t[b]-t[a]),
                    max_command_kpa=float(np.max(np.abs(x[a:b+1])))))
    write_csv("hold_inventory", rows)
    write_csv("hold_segments", runs)
    return rows, runs


def zero_pressure_hold():
    seq = "seq_20260819_183526"
    raw = raw_timing(seq)
    assert np.max(np.abs(raw["actions"])) == 0
    ndi = read_csv(RAW / seq / "ndi.csv")
    x = np.array([[float(r[f"ndi0_{a}"]) for a in "xyz"] for r in ndi])
    y = np.array([[float(r[f"ndi1_{a}"]) for a in "xyz"] for r in ndi])
    # Both sensor displacements and relative translation are shown; no claim
    # that either sensor is the base or robot tip without sensor calibration.
    channels = {"NDI0": x, "NDI1": y, "NDI1_minus_NDI0": y-x}
    rows, summary = [], []
    for name, xyz in channels.items():
        displacement = xyz - xyz[:3].mean(0)
        radial = np.linalg.norm(displacement, axis=-1)
        for j, r in enumerate(ndi):
            rows.append(dict(sequence=seq, frame=j, sensor=name,
                elapsed_s=float(r["t_sec"])-float(ndi[0]["t_sec"]),
                dx_mm=float(displacement[j, 0]), dy_mm=float(displacement[j, 1]),
                dz_mm=float(displacement[j, 2]), displacement_mm=float(radial[j]),
                pressure_kpa=0.))
        summary.append(dict(sensor=name, frames=len(x), duration_s=float(raw["time"][-1]-raw["time"][0]),
            first3_to_last3_mm=float(np.linalg.norm(xyz[-3:].mean(0)-xyz[:3].mean(0))),
            maximum_displacement_mm=float(radial.max()), rms_displacement_mm=float(np.sqrt(np.mean(radial**2))),
            unknown_preceding_history=True))
    write_csv("zero_pressure_ndi_hold", rows)
    write_csv("zero_pressure_ndi_summary", summary)
    return rows, summary


def charts_and_tables(pooled, by_seq, alignment, states, timing, inventory, holds, hold_runs, ndi, ndi_summary):
    charts = []
    def chart(identifier, title, kind, x, y, color, unit, rows, note):
        charts.append(dict(id=identifier, title=title, kind=kind, x=x, y=y, color=color,
                           unit=unit, rows=plain(rows), note=note))
    native = summary_rows(pooled, ["model", "protocol"], ["node_mean_mm", "endpoint_mean_mm"])
    chart("native_rate_prediction", "原生10Hz与抽样5Hz的形态预测", "bar", "protocol", "node_mean_mm_mean", "model", "mm", native,
          "5个固定seed；帧数加权合并两条记录；误差线为seed样本标准差。H39与H20起止图像相同；wrong_dt仅seed0。")
    chart("native_by_sequence", "不同记录与时间约定的预测误差", "bar", "protocol", "node_mean_mm_mean", "sequence_model", "mm",
          [dict(r, sequence_model=r["sequence"][-6:]+" / "+r["model"]) for r in
           summary_rows(by_seq, ["sequence", "model", "protocol"], ["node_mean_mm", "endpoint_mean_mm"])],
          "两条记录分别呈现，避免总体平均掩盖变化；不以误差筛选记录。")
    chart("tau_channel_activation", "各通道时间尺度上的状态幅值", "line", "tau_s", "state_rms_mean", "channel", "normalized drive",
          summary_rows(states["spectrum"], ["channel", "tau_index", "tau_s"],
                       ["state_rms", "bend_contribution_rms_rad", "length_contribution_rms_log"]),
          "全部2958个正式测试窗口；tau是预先固定的0.6–2.0秒网格，不是从材料试验独立辨识的常数。")
    chart("tau_geometric_profile", "各时间尺度对沿臂几何的影响", "line", "node", "removal_mean_displacement_mm_mean", "tau_s", "mm",
          summary_rows(states["geometry"], ["node", "tau_index", "tau_s"], ["removal_mean_displacement_mm"]),
          "单独移除一个tau的全部通道几何读出后，与完整预测比较；保留其余项，非重训，不可将这些位移相加。")
    chart("tau_state_correlation", "不同时间尺度的状态相关性", "heatmap", "tau_a", "tau_b", "pearson_r_mean", "Pearson r",
          summary_rows(states["correlations"], ["tau_a", "tau_b", "tau_a_s", "tau_b_s"], ["pearson_r"]),
          "逐通道、逐seed计算后平均，共4通道×5seed；横纵轴1–6分别对应固定tau网格。相关性高表示基函数难独立辨识，不等于无预测价值。")
    chart("counterfactual_decay", "恒定命令下时间记忆的指数衰减", "line", "elapsed_s", "expected_remaining_fraction", "tau_s", "fraction",
          states["hold"], "从真实测试历史得到72个初始状态后保持当前命令；曲线为模型反事实传播，不是实测持压曲线。")
    chart("counterfactual_geometry", "恒定命令下时间分支的几何修正", "line", "elapsed_s", "mean_node_time_correction_mm", None, "mm",
          states["duration"], "真实历史初始化、随后模型持压8秒；路径记忆保持，时间修正相对同一参考与路径项计算。")
    chart("halfstep_consistency", "5Hz到10Hz零阶保持细分的一致性", "bar", "seed", "max_node_difference_mm", None, "mm",
          states["halfstep"], "每个0.2秒命令原样保持并拆成两个0.1秒；H20→H39，3.8秒一致；仅数值一致性。")
    chart("zero_pressure_ndi", "零压短平台的NDI位移", "line", "elapsed_s", "displacement_mm", "sensor", "mm", ndi,
          "183526，39帧；相对起始3帧均值；传感器安装位置与此前驱动历史未恢复，不拟合材料时间常数。")
    tables = [dict(id="data_inventory", title="可用数据与标签来源", rows=inventory),
              dict(id="timing", title="原生10Hz实际时序", rows=timing),
              dict(id="alignment", title="共同目标帧与历史时长", rows=alignment),
              dict(id="native_metrics", title="逐seed、逐协议总体误差", rows=pooled),
              dict(id="hold_inventory", title="全部原始记录的持压片段搜索", rows=holds),
              dict(id="hold_segments", title="满足至少1秒的片段", rows=hold_runs),
              dict(id="ndi_summary", title="零压片段NDI统计", rows=ndi_summary),
              dict(id="readout_cancellation", title="时间读出的叠加与抵消", rows=states["cancellation"]),
              dict(id="formal_reference", title="正式模型复建与时间分支幅值", rows=states["baseline"])]
    return charts, tables


def render_figures(charts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#2563a6", "#c88732", "#b2594c", "#6f7d3c", "#a86b91", "#6d7581"]
    chosen = [c for c in charts if c["id"] in ("counterfactual_decay", "counterfactual_geometry",
              "tau_geometric_profile", "zero_pressure_ndi", "halfstep_consistency")]
    artifacts = []
    for c in chosen:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
        groups = {}
        for r in c["rows"]:
            group = str(r[c["color"]]) if c["color"] else "all"
            groups.setdefault(group, []).append(r)
        for i, (g, rows) in enumerate(groups.items()):
            rows = sorted(rows, key=lambda r: r[c["x"]])
            x, y = [r[c["x"]] for r in rows], [r[c["y"]] for r in rows]
            if c["kind"] == "bar":
                ax.bar(x, y, color=colors[i % len(colors)], width=.65)
            else:
                label = f"{float(g):.3f} s" if c["color"] == "tau_s" else g
                ax.plot(x, y, color=colors[i % len(colors)], label=label,
                        linestyle=["-", "--", "-.", ":"][i % 4], linewidth=1.8)
        ax.set(xlabel=c["x"], ylabel=c["unit"], title=c["id"].replace("_", " "))
        ax.grid(axis="y", color="#e4e7eb", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        if c["color"]:
            ax.legend(title=c["color"], fontsize=8, ncol=2)
        path = OUT / f"{c['id']}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        artifacts.append(str(path))
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        raise ValueError("Use at most four CPU threads")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    native, inventory = native_data()
    spots = spot_overlays(native)
    holds, hold_runs = hold_inventory()
    ndi, ndi_summary = zero_pressure_hold()
    print("Data alignment/QC and held-command inventory complete", flush=True)
    pooled, by_seq, alignment = native_comparison(native)
    states = formal_states(formal_data())
    timing = [s["timing_summary"] for s in native]
    charts, tables = charts_and_tables(pooled, by_seq, alignment, states, timing, inventory,
                                       holds, hold_runs, ndi, ndi_summary)
    artifacts = render_figures(charts)
    lookup = {(r["model"], r["protocol"]): r for r in
              summary_rows(pooled, ["model", "protocol"], ["node_mean_mm", "endpoint_mean_mm"])}
    def value(model, protocol):
        return lookup[(model, protocol)]["node_mean_mm_mean"]
    nominal10, nominal5 = value("hov", "nominal_10Hz_H39"), value("hov", "decimated_5Hz_H20")
    issue10, ack10 = value("hov", "event_issue_10Hz_H39"), value("hov", "event_ack_10Hz_H39")
    notime10 = value("hov_no_maxwell", "nominal_10Hz_H39")
    maxhalf = max(r["max_node_difference_mm"] for r in states["halfstep"])
    adjacent_corr = [r["pearson_r"] for r in states["correlations"] if r["tau_b"] == r["tau_a"] + 1]
    cancellation_ratio = np.mean([r["component_to_net_rms_ratio"] for r in states["cancellation"]])
    effect_mean = np.mean([r["time_correction_mean_node_mm"] for r in states["baseline"]])
    assert states["reconstruction_max_normalized_difference"] < 2e-6
    assert maxhalf < .001
    findings = [
        dict(id="native_available", level="observed", text=
             "182253/182519有原生命令、时间日志与逐帧骨架；完整来源合计4211帧，旧10Hz合并目录遗漏182253的227帧val，故使用源train+val。两条标签QC均未记录跨帧插值。"),
        dict(id="native_prediction", level="frozen_model_diagnostic", text=
             f"相同起止帧、tau和初始化下，HOV名义10Hz(H39)节点误差{nominal10:.4f} mm，抽样5Hz(H20)为{nominal5:.4f} mm；重训无时间分支模型在原生10Hz为{notime10:.4f} mm。"),
        dict(id="timing_sensitivity", level="diagnostic", text=
             f"按日志命令发出时刻进行ZOH积分，原生10Hz误差{issue10:.4f} mm；以ACK时刻为生效边界则为{ack10:.4f} mm。两者用于量化输入生效时刻的不确定性，未据测试误差重新选模型。"),
        dict(id="halfstep", level="numerical_property", text=
             f"正式测试窗口的0.2秒ZOH细分为两个0.1秒，最大节点差{maxhalf:.3g} mm；这是指数递推的半步一致性，不是新10Hz观测或执行速度泛化。"),
        dict(id="tau", level="model_structure", text=
             "6个时间尺度固定为0.6000、0.7634、0.9712、1.2356、1.5720、2.0000秒。报告其状态幅值、弯曲/长度读出和沿臂几何影响；这些网格与读出不能解释成已独立辨识的材料松弛谱。"),
        dict(id="basis_identifiability", level="diagnostic", text=
             f"全部正式测试窗口中，相邻tau的时间偏差状态相关系数均值{np.mean(adjacent_corr):.4f}；归一化几何坐标中，各通道/尺度贡献的RMS范数之和约为合成贡献的{cancellation_ratio:.2f}倍。净时间修正平均{effect_mean:.4f} mm，说明分支有几何作用，但单尺度幅值存在抵消和辨识歧义。"),
        dict(id="hold_evidence", level="limited_observation", text=
             "对全部13条原始记录搜索至少1秒的全通道恒压片段；0.001、0.5、2 kPa范围阈值均只找到183526的39帧全零压段。前史缺失、无该段骨架，不能作为非零持压暂态验证。提供NDI位移及模型反事实持压衰减。"),
    ]
    limitations = [
        "原生10Hz标签为现有SAM2中心线，未重跑新标签流程；原图仅按首/中/末抽查6张；历史分割模型可能含序列传播信息，视觉标签并非独立三维真值。",
        "此处为训练于三条5Hz记录的冻结模型跨记录诊断；两条10Hz记录不属于该模型训练数据，但可能曾用于项目早期开发。不得称完全未见的确认性测试。",
        "采集平均约9.05–9.08Hz，名义3.8秒窗口的实际时间更长；两种视图对齐同一首尾图像，日志积分单独呈现。",
        "抽样5Hz保留每隔一帧的命令与标签，丢失中间输入变化，真实机器人的执行轨迹没有改变。合并两种奇偶相位，每个目标图像只计一次。",
        "生效边界由命令发出或ACK日志近似；未测得阀后实时压力。ACK之前曝光或非ACK命令涉及的历史窗口统一排除。",
        "各窗口在第一个观测时刻假设p=h=e(u0)，其前史未知；3.8秒只相当于最大tau的1.9倍，最慢模式仍保留约15%的初始状态差。",
        "时间尺度预先固定，邻近指数基高度相关；激活或移除某尺度造成位移，均不等价于唯一的真实物理机制识别。",
        "反事实持压响应由模型计算，不包含新观测；路径与参考项保持，状态d单调衰减不要求多个带符号几何读出之和的范数严格单调。",
        "仅5个训练seed且只有2条10Hz记录；表中标准差是优化重复差异，不代表记录/机器人总体不确定性，不做帧独立显著性检验。",
        "零压NDI片段没有可恢复的历史或同时骨架；传感器位置未作基座/末端假设，不能据此分离材料、气路、测量漂移。",
    ]
    definitions = dict(
        time_memory="分支名称为时间记忆；h为时间记忆状态，d=h-e为时间读出所使用的偏差。",
        recurrence="h_next=exp(-dt/tau)*h+(1-exp(-dt/tau))*e；恒定e时d(t+s)=exp(-s/tau)*d(t)。",
        tau_policy="从冻结checkpoint读取taus，改变dt时不重建tau网格或训练读出。",
        timing_nominal="首帧设置p=h=e(u0)；随后H-1次更新：原生H39×0.1s与抽样H20×0.2s均为3.8s。",
        timing_event="在首张图像曝光时刻初始化；按日志输入生效边界传播，到末张图像曝光停止；不将当前输入提前作用整个曝光间隔。分别用issue/ack边界。",
        label_policy="只取实际目标帧骨架；任何hard_invalid/interpolated目标不计分；不插值图像/骨架，也不把低频数据命名为新10Hz观测。",
        mean_node_error="每帧15个对应节点三维欧氏距离的均值（坐标为平面毫米）；对两序列全部有效帧求均值，再对5seed求mean±sample SD。",
        endpoint_error="第15节点欧氏距离，先池化帧再求seed统计。",
        hold_detection="按时间顺序构造最大连续片段，要求片段内每个原始通道max-min<=阈值且首末采集时间差>=1s；阈值0.001/0.5/2kPa。",
        geometry_spectrum="冻结模型，移除单个tau所有通道的几何读出后比较节点位移；该曲线为局部功能影响，不是误差增益或可加分解。")
    definitions["cancellation_ratio"] = "先除以checkpoint几何坐标scale；计算sum_c,k sqrt(mean_t ||component_t,c,k||²) / sqrt(mean_t ||sum_c,k component_t,c,k||²)，不混合弧度与对数长度的原始单位。"
    definitions["state_correlation"] = "每通道、每seed计算测试窗口d=h-e的六尺度Pearson相关，再平均20张相关矩阵；这是同一输入驱动产生的基函数相关性。"
    payload = dict(schema="modeling_mechanism_analysis_v1", module="time_memory",
        findings=findings, definitions=definitions, charts=charts, tables=tables,
        limitations=limitations, provenance=dict(study=str(STUDY),
            formal_manifest=str(STUDY / "data/dataset_manifest.json"),
            checkpoints=[str(STUDY / "formal" / m / f"seed_{s}" / "best_eval_model.pt")
                         for m in ("hov", "hov_no_maxwell") for s in SEEDS],
            native_sources=inventory, label_spotchecks=spots, output_directory=str(OUT),
            script=str(Path(__file__).resolve()), command="python scripts/experiments/analyze_modeling_time_memory.py --threads 4",
            device="cpu", threads=args.threads, elapsed_seconds=time.perf_counter()-start,
            created_at=datetime.now(timezone.utc).isoformat(),
            versions=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__),
            raw_files=[str(RAW / s / f) for s in ["seq_20260819_182253", "seq_20260819_182519", "seq_20260819_183526"]
                       for f in ["meta.json", "commands.csv", "samples.csv", "actions6.csv", "frame_times.txt"]],
            source_code=["src/operators/maxwell_bank.py", "src/operators/play_bank.py",
                         "src/models/model_hereditary_operator.py", "src/models/model_hereditary_geometry.py",
                         "src/benchmarks/modeling_geometry_calibration.py", "src/benchmarks/modeling_models.py"],
            reconstruction_max_normalized_difference=states["reconstruction_max_normalized_difference"],
            counterfactual_initial_frames=states["selected_states"], figures=artifacts,
            inference_only=True, label_interpolation=False,
            terms={"branch": "时间记忆", "h": "时间记忆状态", "d": "h-e"}))
    write_json(REPORT / "time_memory.json", payload)
    write_json(OUT / "analysis_summary.json", {k:v for k,v in payload.items() if k not in ("charts", "tables")})
    write_json(OUT / "native_alignment.json", alignment)
    md = ["# 时间记忆与原生10Hz分析", "", "本报告使用冻结权重，仅做诊断与状态传播；分支称为时间记忆，递推变量h称为时间记忆状态。", ""]
    md += ["## 主要结果", ""] + ["- " + r["text"] for r in findings]
    md += ["", "## 方法与时间约定", ""] + [f"- **{k}**：{v}" for k,v in definitions.items()]
    md += ["", "## 同一目标帧上的预测误差", "", "| 模型 | 协议 | 节点误差 mm | 末端误差 mm | seed数 |", "|---|---|---:|---:|---:|"]
    for r in lookup.values():
        sd = r["node_mean_mm_sd"]
        err = f"{r['node_mean_mm_mean']:.4f}" + (f" ± {sd:.4f}" if sd is not None else "")
        md.append(f"| {r['model']} | {r['protocol']} | {err} | {r['endpoint_mean_mm_mean']:.4f} | {r['repetitions']} |")
    md += ["", "两种奇偶抽样相位共享原生目标帧，但每帧只计一次。无时间分支为正式独立重训模型，完整模型的分支置零与历史置换由主分析另行提供。", "", "## 证据范围", ""]
    md += ["- " + s for s in limitations]
    md += ["", "## 记录与图数据", "", f"- 通用图表数据：`{REPORT / 'time_memory.json'}`。", f"- 完整长表与预测：`{OUT}`。", "- `native10hz_per_frame.csv`为所有目标帧、模型、seed与协议的误差；统计表未进行样本筛选。", "- `tau_channel_spectrum.csv`保留状态幅值、弯曲与分段长度读出；`tau_node_geometric_effect.csv`保留沿臂结果。", "- `native10hz_label_spotcheck.png`为按索引选取的原图叠加骨架。", ""]
    (REPORT / "time_memory.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(dict(seconds=time.perf_counter()-start, findings=findings,
                          json=str(REPORT / 'time_memory.json')), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
