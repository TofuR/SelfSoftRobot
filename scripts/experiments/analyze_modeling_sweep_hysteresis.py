#!/usr/bin/env python3
"""Reproducible, CPU-only held-cycle NDI analysis of two early pressure sweeps.

No raw data are changed. Fixed protocol is written before any outcome fitting.
All positions are NDI probe positions, not whole-body or certified tip labels.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/selfsr_sweep_matplotlib")

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "workspace/runs/analysis/modeling_mechanisms_20260913_001/sweeps"
DEFAULT_REPORT = ROOT / "workspace/reports/modeling_mechanisms_20260913_001/sweep_hysteresis"
SEQUENCES = ("seq_20260627_172916", "seq_20260627_173114")
MODEL_NAMES = {
    "static_cubic": "静态三次基函数",
    "direction_cubic": "方向特征（三次基函数，Chen 思想适配）",
    "play": "静态 + 路径记忆",
    "time": "静态 + 时间记忆",
    "dual": "静态 + 双记忆",
}
MODEL_EN = {"static_cubic": "Static cubic", "direction_cubic": "Direction cubic",
            "play": "+ Path memory", "time": "+ Time memory", "dual": "+ Dual memory"}
COLORS = {"static_cubic": "#777777", "direction_cubic": "#A68100", "play": "#C2642B",
          "time": "#526C30", "dual": "#2368A2"}
PROTOCOL = {
    "version": "sweep_hysteresis_v1",
    "sequences": list(SEQUENCES),
    "target": "NDI probe position (x,y,z), mm; no tip or whole-body identity assumption",
    "alignment": "Exact one-to-one join of actions6.csv and ndi.csv on t_sec; fail on missing timestamps",
    "cycle": "Consecutive recorded 0 kPa samples enclosing exactly one 150 kPa peak and monotone up/down branches",
    "split": "Per sequence, chronological complete cycles: [0,floor(.6N)) train, [floor(.6N),floor(.8N)) val, remainder test",
    "fit_rows": "Each cycle uses [start_row,end_row), so no labeled frame belongs to two splits",
    "loop_rows": "Both endpoints included for geometric branch interpolation only; adjacent cycles share a boundary",
    "pressure_grid_kpa": list(range(151)),
    "static_features": ["1", "e", "e^2", "e^3"],
    "drive": "e = c0 / 150; c0 is recorded pressure command, not independently measured chamber pressure",
    "direction_features": ["s", "s*e", "s*e^2", "s*e^3"],
    "direction_rule": "Sign of latest nonzero recorded pressure difference, causal; first row 0",
    "play_thresholds_normalized": [0.05, 0.1, 0.2, 0.35, 0.5],
    "time_constants_s": [0.6, 1.2, 2.4, 4.8, 9.6],
    "memory_readout": "Each q or (h-e) multiplies 1,e,e^2; fixed pressure-dependent linear readout",
    "play_update": "p_t=clip(p_(t-1),e_t-r,e_t+r); q_t=e_t-p_t",
    "time_update": "h_t=exp(-dt_t/tau)*h_(t-1)+(1-exp(-dt_t/tau))*e_t; d_t=h_t-e_t",
    "state_initialization": "First recorded e: p=e and h=e; propagate all preceding recorded inputs, including initial partial cycle, without any NDI feedback; never reset at split boundaries",
    "timing_assumption": "Use true differences between recorded t_sec values with current-level ZOH approximation; exact pneumatic switching times/unmeasured pressure lag are unavailable",
    "baseline": "Subtract first 0 kPa NDI position in the first training cycle of each sequence; no val/test target calibration",
    "normalization": "Feature center/std learned on pooled training frames only; intercept unpenalized",
    "ridge_objective": "mean(sum_xyz((XW-y)^2)) + lambda*sum(nonintercept standardized coefficients^2)",
    "ridge_grid": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    "selection": "For each fixed feature family select lambda by pooled validation mean Euclidean 3D error; ties prefer larger lambda; freeze before test; no train+val refit",
    "primary_metric": "Frame-pooled mean Euclidean 3D position error (mm), with per-sequence and per-cycle results",
    "inference_scope": "Descriptive chronological held-cycle fit within two recordings; serial cycles are not independent acquisition trials; no significance test",
    "excluded_search": "No outcome-based cycle dropping, feature search, tau/threshold search, test-based tuning or seed selection",
}


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(x) for x in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def records(frame):
    return clean(frame.to_dict("records"))


def integrate(values, grid):
    return float(np.trapezoid(values, grid) if hasattr(np, "trapezoid") else np.trapz(values, grid))


def load_data():
    frames, cycles, provenance = [], [], []
    for seq in SEQUENCES:
        path = ROOT / "workspace/data/raw/real" / seq
        a, n = (pd.read_csv(path / name) for name in ("actions6.csv", "ndi.csv"))
        meta = json.loads((path / "meta.json").read_text())
        assert a.t_sec.is_unique and n.t_sec.is_unique
        assert np.all(np.diff(a.t_sec) > 0) and np.all(np.diff(n.t_sec) > 0)
        assert set(a.t_sec) == set(n.t_sec), "Non-identical timestamp sets require explicit alignment review"
        d = a.merge(n, on="t_sec", validate="one_to_one", sort=True)
        assert np.isfinite(d[["t_sec", "c0", "x", "y", "z"]]).all().all()
        assert np.allclose(d[[f"c{i}" for i in range(1, 6)]], 0)
        p, t = d.c0.to_numpy(), d.t_sec.to_numpy()
        zero = np.flatnonzero(np.isclose(p, 0))
        sequence_cycles = []
        for start, end in zip(zero[:-1], zero[1:]):
            segment = p[start:end + 1]
            peak = start + int(np.argmax(segment))
            assert np.isclose(p[peak], 150) and np.count_nonzero(segment == 150) == 1
            assert np.all(np.diff(p[start:peak + 1]) > 0) and np.all(np.diff(p[peak:end + 1]) < 0)
            duration = t[end] - t[start]
            sequence_cycles.append(dict(sequence=seq, cycle_id=len(sequence_cycles) + 1,
                start_row=int(start), peak_row=int(peak), end_row=int(end),
                start_t_sec=t[start], peak_t_sec=t[peak], end_t_sec=t[end], duration_s=duration,
                loading_duration_s=t[peak]-t[start], unloading_duration_s=t[end]-t[peak],
                loading_slope_kpa_s=150/(t[peak]-t[start]), unloading_slope_kpa_s=-150/(t[end]-t[peak]),
                mean_abs_command_slope_kpa_s=300/duration, loop_rows=int(end-start+1),
                fitting_rows=int(end-start)))
        count = len(sequence_cycles)
        assert count >= 5, "Insufficient full cycles for planned split"
        b1, b2 = int(np.floor(.6 * count)), int(np.floor(.8 * count))
        d["sequence"], d["source_row"] = seq, np.arange(len(d))
        d["cycle_id"], d["split"] = -1, "partial_or_terminal"
        d["dt_s"] = np.r_[np.nan, np.diff(t)]
        d["command_slope_kpa_s"] = np.r_[np.nan, np.diff(p)/np.diff(t)]
        for j, cycle in enumerate(sequence_cycles):
            cycle["split"] = "train" if j < b1 else "val" if j < b2 else "test"
            mask = (d.source_row >= cycle["start_row"]) & (d.source_row < cycle["end_row"])
            d.loc[mask, ["cycle_id", "split"]] = [cycle["cycle_id"], cycle["split"]]
        baseline_row = sequence_cycles[0]["start_row"]
        baseline = d.loc[baseline_row, ["x", "y", "z"]].to_numpy(float)
        for axis, base in zip("xyz", baseline):
            d[f"{axis}_relative_mm"] = d[axis] - base
        dt = np.diff(t)
        provenance.append(dict(sequence=seq, raw_directory=str(path.relative_to(ROOT)),
            actions_rows=len(a), ndi_rows=len(n), matched_rows=len(d), maximum_timestamp_offset_s=0.0,
            nonfinite_position_rows=0, dropped_quality_rows=0, complete_cycles=count,
            train_cycles=b1, val_cycles=b2-b1, test_cycles=count-b2,
            excluded_from_fit_rows=int((d.split == "partial_or_terminal").sum()),
            first_full_cycle_start_row=int(baseline_row), baseline_xyz_mm=baseline,
            baseline_source="first 0 kPa row in first training cycle",
            recorded_dt_median_s=np.median(dt), recorded_dt_min_s=dt.min(), recorded_dt_max_s=dt.max(),
            nominal_action_interval_s=meta["action_interval_s"], settle_s=meta["settle_s"],
            complete_cycle_mean_abs_command_slope_kpa_s=np.mean([c["mean_abs_command_slope_kpa_s"] for c in sequence_cycles]),
            meta=meta))
        frames.append(d)
        cycles.extend(sequence_cycles)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(cycles), provenance


def measure_loops(frames, cycles, prediction_columns=None):
    grid = np.array(PROTOCOL["pressure_grid_kpa"], float)
    curve_rows, summaries = [], []
    xyz = ["x", "y", "z"] if prediction_columns is None else prediction_columns
    for cycle in cycles.to_dict("records"):
        d = frames[frames.sequence == cycle["sequence"]].set_index("source_row")
        start, peak, end = (cycle[k] for k in ("start_row", "peak_row", "end_row"))
        up, down = d.loc[start:peak], d.loc[peak:end].iloc[::-1]
        up_xyz = np.column_stack([np.interp(grid, up.c0, up[c]) for c in xyz])
        down_xyz = np.column_stack([np.interp(grid, down.c0, down[c]) for c in xyz])
        delta = down_xyz - up_xyz
        gap = np.linalg.norm(delta, axis=1)
        summary = dict(cycle)
        summary.update(gap_75kpa_mm=float(gap[75]), pressure_mean_gap_mm=integrate(gap, grid)/150,
            gap_max_mm=float(gap.max()), gap_max_pressure_kpa=float(grid[gap.argmax()]),
            signed_x_loop_area_mm_kpa=integrate(delta[:, 0], grid),
            absolute_x_branch_area_mm_kpa=integrate(np.abs(delta[:, 0]), grid),
            closure_gap_mm=float(gap[0]))
        # This closed-loop convention integrates x_down-x_up; direction is explicit.
        summaries.append(summary)
        for k, pressure in enumerate(grid):
            row = dict(sequence=cycle["sequence"], cycle_id=cycle["cycle_id"], split=cycle["split"],
                pressure_kpa=pressure, gap_3d_mm=gap[k], delta_x_mm=delta[k, 0],
                delta_y_mm=delta[k, 1], delta_z_mm=delta[k, 2],
                mean_abs_command_slope_kpa_s=cycle["mean_abs_command_slope_kpa_s"])
            row.update({f"{branch}_{axis}_mm": values[k, j]
                for branch, values in (("loading", up_xyz), ("unloading", down_xyz)) for j, axis in enumerate("xyz")})
            curve_rows.append(row)
    return pd.DataFrame(curve_rows), pd.DataFrame(summaries)


def feature_data(frames):
    feature_frames = []
    rs = np.array(PROTOCOL["play_thresholds_normalized"])
    taus = np.array(PROTOCOL["time_constants_s"])
    for _, d in frames.groupby("sequence", sort=False):
        e, t = d.c0.to_numpy()/150, d.t_sec.to_numpy()
        direction = np.zeros(len(d))
        for i in range(1, len(d)):
            direction[i] = np.sign(e[i]-e[i-1]) or direction[i-1]
        play = np.full(len(rs), e[0])
        h = np.full(len(taus), e[0])
        q, deficit = np.zeros((len(d), len(rs))), np.zeros((len(d), len(taus)))
        for i in range(len(d)):
            play = np.clip(play, e[i]-rs, e[i]+rs)
            q[i] = e[i]-play
            if i:
                decay = np.exp(-(t[i]-t[i-1])/taus)
                h = decay*h + (1-decay)*e[i]
            deficit[i] = h-e[i]
        out = d[["sequence", "source_row", "t_sec", "c0", "split", "cycle_id"]].copy()
        for power in range(4):
            out[f"static_e{power}"] = e**power
            out[f"direction_e{power}"] = direction*e**power
        for j, r in enumerate(rs):
            for power in range(3):
                out[f"play_r{r:g}_e{power}"] = q[:, j]*e**power
        for j, tau in enumerate(taus):
            for power in range(3):
                out[f"time_tau{tau:g}_e{power}"] = deficit[:, j]*e**power
        feature_frames.append(out)
    return pd.concat(feature_frames, ignore_index=True)


def fit_models(frames, features, run):
    train, val, test = (frames.split.to_numpy() == s for s in ("train", "val", "test"))
    assert not np.any(train & val) and not np.any(train & test) and not np.any(val & test)
    y = frames[[f"{a}_relative_mm" for a in "xyz"]].to_numpy()
    static = [c for c in features if c.startswith("static_")]
    path = [c for c in features if c.startswith("play_")]
    temporal = [c for c in features if c.startswith("time_")]
    directional = [c for c in features if c.startswith("direction_")]
    families = dict(static_cubic=static, direction_cubic=static+directional,
                    play=static+path, time=static+temporal, dual=static+path+temporal)
    selected, val_rows, predictions, coefficient_rows = {}, [], [], []
    # All choices are made using train/val targets. No test outcome is consulted here.
    for name, cols in families.items():
        started = time.perf_counter()
        x = features[cols].to_numpy(float)
        mean, std = x[train].mean(0), x[train].std(0)
        mean[0], std[0] = 0., 1.
        std[std < 1e-10] = 1.
        z = (x-mean)/std
        gram, rhs = z[train].T @ z[train]/train.sum(), z[train].T @ y[train]/train.sum()
        penalty = np.eye(len(cols)); penalty[0, 0] = 0
        choices = []
        for lam in PROTOCOL["ridge_grid"]:
            coef = np.linalg.solve(gram+lam*penalty, rhs)
            error = np.linalg.norm(z[val] @ coef-y[val], axis=1)
            choices.append((float(error.mean()), -lam, coef))
            val_rows.append(dict(model=name, lambda_ridge=lam, validation_mean_3d_mm=error.mean(),
                                 feature_count=len(cols), coefficient_count=3*len(cols)))
        best, neg_lam, coef = min(choices, key=lambda item: item[:2])
        raw_coef = coef/std[:, None]
        raw_coef[0] -= mean @ raw_coef
        selected[name] = dict(model=name, label=MODEL_NAMES[name], lambda_ridge=-neg_lam,
            validation_mean_3d_mm=best, features=cols, feature_count=len(cols), coefficient_count=3*len(cols),
            feature_mean=mean, feature_std=std, standardized_coefficients=coef,
            raw_coefficients=raw_coef, fitting_seconds=time.perf_counter()-started)
        for j, col in enumerate(cols):
            for axis, k in zip("xyz", range(3)):
                coefficient_rows.append(dict(model=name, feature=col, axis=axis,
                    standardized_coefficient_mm=coef[j, k], raw_feature_coefficient_mm=raw_coef[j, k]))
    write_json(run / "frozen_models.json", selected)
    pd.DataFrame(val_rows).to_csv(run / "validation_grid.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(run / "coefficients.csv", index=False)
    for name, model in selected.items():
        x = features[model["features"]].to_numpy(float)
        z = (x-model["feature_mean"])/model["feature_std"]
        pred = z @ model["standardized_coefficients"]
        row = frames[["sequence", "source_row", "t_sec", "c0", "cycle_id", "split"]].copy()
        row["model"] = name
        for j, axis in enumerate("xyz"):
            row[f"true_{axis}_relative_mm"] = y[:, j]
            row[f"pred_{axis}_relative_mm"] = pred[:, j]
            row[f"pred_{axis}_mm"] = pred[:, j]+frames[axis]-y[:, j]
        row["error_3d_mm"] = np.linalg.norm(pred-y, axis=1)
        for group, prefix in (("path", "play_"), ("time", "time_")):
            mask = np.array([c.startswith(prefix) for c in model["features"]])
            # Raw feature contributions give zero at q/d=0; centering is absorbed in intercept.
            contribution = x[:, mask] @ model["raw_coefficients"][mask]
            for j, axis in enumerate("xyz"):
                row[f"{group}_contribution_{axis}_mm"] = contribution[:, j]
            row[f"{group}_contribution_norm_mm"] = np.linalg.norm(contribution, axis=1)
        predictions.append(row)
    pred_frame = pd.concat(predictions, ignore_index=True)
    pred_frame.to_csv(run / "predictions.csv", index=False)
    return selected, pred_frame, pd.DataFrame(val_rows)


def error_summary(predictions, selected):
    rows, per_cycle = [], []
    valid = predictions[predictions.split.isin(["train", "val", "test"])]
    for (model, split, seq, cycle), d in valid.groupby(["model", "split", "sequence", "cycle_id"], sort=False):
        per_cycle.append(dict(model=model, label=MODEL_NAMES[model], split=split, sequence=seq,
            cycle_id=cycle, frames=len(d), mean_3d_mm=d.error_3d_mm.mean(),
            rms_3d_mm=np.sqrt(np.mean(d.error_3d_mm**2))))
    per_cycle = pd.DataFrame(per_cycle)
    for model in MODEL_NAMES:
        for split in ("train", "val", "test"):
            for seq in ("pooled", *SEQUENCES):
                d = valid[(valid.model == model) & (valid.split == split)]
                c = per_cycle[(per_cycle.model == model) & (per_cycle.split == split)]
                if seq != "pooled":
                    d, c = d[d.sequence == seq], c[c.sequence == seq]
                rows.append(dict(model=model, label=MODEL_NAMES[model], split=split, sequence=seq,
                    frames=len(d), cycles=len(c), mean_3d_mm=d.error_3d_mm.mean(),
                    rms_3d_mm=np.sqrt(np.mean(d.error_3d_mm**2)), p95_3d_mm=d.error_3d_mm.quantile(.95),
                    cycle_mean_std_mm=c.mean_3d_mm.std(ddof=1),
                    lambda_ridge=selected[model]["lambda_ridge"],
                    coefficient_count=selected[model]["coefficient_count"]))
    return pd.DataFrame(rows), per_cycle


def make_charts(frames, loops, summaries, errors, per_cycle, predictions, pred_loops):
    charts = []
    def add(cid, title, kind, x, y, color, unit, df, note):
        charts.append(dict(id=cid, title=title, kind=kind, x=x, y=y, color=color, unit=unit,
                           rows=records(df), note=note))
    for seq in SEQUENCES:
        sid = seq[-6:]
        d, loop = frames[frames.sequence == seq], loops[loops.sequence == seq]
        add(f"pressure_{sid}", f"{sid}：记录压力指令与时间", "line", "t_sec", "c0", None, "kPa", d,
            "t_sec 为记录时间；两条记录的扫压速度不同，不能把比较解释成纯采样率效应。")
        lr = []
        for branch in ("loading", "unloading"):
            a = loop.copy(); a["branch"] = branch; a["probe_x_mm"] = a[f"{branch}_x_mm"]
            a["series"] = a.cycle_id.astype(str) + ":" + branch; lr.append(a)
        add(f"loops_x_{sid}", f"{sid}：逐周期 NDI x 投影回线", "line", "pressure_kpa", "probe_x_mm", "series", "mm", pd.concat(lr),
            "每个完整周期均展示；横轴为 kPa。纵轴是 NDI 探头坐标。峰值由两支共享。")
        a = loop.copy(); a["series"] = a.cycle_id.astype(str)
        add(f"gap_pressure_{sid}", f"{sid}：同压加载与卸载的三维位置间距", "line", "pressure_kpa", "gap_3d_mm", "series", "mm", a,
            "每周期在0–150 kPa均匀1 kPa网格线性插值；插值增加曲线密度，不增加独立样本量。")
        dsum = summaries[summaries.sequence == seq]
        add(f"cycle_gap_{sid}", f"{sid}：逐周期同压平均间距", "scatter", "cycle_id", "pressure_mean_gap_mm", "split", "mm", dsum,
            "每点一个完整周期；完整循环数和时间顺序均保留；不据此作独立试次显著性检验。")
        add(f"cycle_area_{sid}", f"{sid}：逐周期 NDI x 投影面积", "scatter", "cycle_id", "signed_x_loop_area_mm_kpa", "split", "mm·kPa", dsum,
            "A=∫(x卸载−x加载)dP。几何投影量，不是能耗。")
        a = per_cycle[(per_cycle.sequence == seq) & (per_cycle.split == "test")].copy()
        add(f"held_cycle_error_{sid}", f"{sid}：留出周期的 NDI 预测误差", "bar", "cycle_id", "mean_3d_mm", "model", "mm", a,
            "配置仅用验证周期选择。每个柱为一个测试周期的20个不重复帧。")
        first = int(a.cycle_id.min())
        first_pred = predictions[(predictions.sequence == seq) & (predictions.cycle_id == first)]
        traces = []
        observed = first_pred[first_pred.model == "static_cubic"].copy()
        observed["model"], observed["probe_x_relative_mm"] = "Observed", observed.true_x_relative_mm
        traces.append(observed)
        for model, row in first_pred.groupby("model", sort=False):
            row = row.copy(); row["probe_x_relative_mm"] = row.pred_x_relative_mm; traces.append(row)
        add(f"held_trace_{sid}", f"{sid}：首个测试周期 NDI x 预测", "line", "t_sec", "probe_x_relative_mm", "model", "mm", pd.concat(traces),
            "预先固定展示首个测试周期；只减去训练期首个零压基准。")
        a = pred_loops[(pred_loops.sequence == seq) & (pred_loops.split == "test")].copy()
        grouped = a.groupby(["model", "pressure_kpa"], as_index=False).gap_3d_mm.mean()
        observed = loop[loop.split == "test"].groupby("pressure_kpa", as_index=False).gap_3d_mm.mean()
        observed["model"] = "Observed"
        add(f"held_gap_fit_{sid}", f"{sid}：测试周期的同压回线间距拟合", "line", "pressure_kpa", "gap_3d_mm", "model", "mm", pd.concat([grouped, observed]),
            "先在每周期匹配压力，再平均间距；两支共用峰值使150 kPa处间距按定义为0。")
    a = errors[(errors.split == "test")].copy()
    add("test_error", "NDI 预测器测试误差", "bar", "model", "mean_3d_mm", "sequence", "mm", a,
        "pooled按帧汇总，快速序列100帧、慢速序列40帧；两次独立记录，7个留出周期。")
    a = summaries[["sequence", "cycle_id", "mean_abs_command_slope_kpa_s", "gap_75kpa_mm", "pressure_mean_gap_mm", "split"]]
    add("speed_gap", "记录扫压速度与75 kPa三维间距", "scatter", "mean_abs_command_slope_kpa_s", "gap_75kpa_mm", "sequence", "mm", a,
        "横轴为记录指令绝对变化速度kPa/s。两速度各仅一次连续记录，不能因果归因于材料速率效应。")
    dual = predictions[(predictions.model == "dual") & (predictions.split == "test")]
    branch_rows = []
    for group in ("path", "time"):
        a = dual.copy(); a["branch"] = group; a["contribution_norm_mm"] = a[f"{group}_contribution_norm_mm"]
        branch_rows.append(a)
    add("dual_contributions", "双记忆预测器测试期分支贡献", "scatter", "c0", "contribution_norm_mm", "branch", "mm", pd.concat(branch_rows),
        "固定模型的输出贡献幅值；分支可能相互抵消、特征相关，贡献幅值不等于独立因果效应。")
    return charts


def save_figures(run, frames, loops, summaries, errors, predictions, pred_loops):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18,
                         "figure.dpi": 140, "savefig.dpi": 160})
    files = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = run / f"{name}.{ext}"; fig.savefig(path, bbox_inches="tight"); files.append(str(path.relative_to(ROOT)))
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axrow, seq in zip(axes, SEQUENCES):
        l = loops[loops.sequence == seq]; s = summaries[summaries.sequence == seq]
        for _, cycle in l.groupby("cycle_id"):
            axrow[0].plot(cycle.pressure_kpa, cycle.loading_x_mm, color="#2368A2", alpha=.2, lw=1)
            axrow[0].plot(cycle.pressure_kpa, cycle.unloading_x_mm, color="#C2642B", alpha=.2, lw=1, ls="--")
            axrow[1].plot(cycle.pressure_kpa, cycle.gap_3d_mm, color="#2368A2", alpha=.2, lw=1)
        avg = l.groupby("pressure_kpa").mean(numeric_only=True)
        axrow[0].plot(avg.index, avg.loading_x_mm, color="#2368A2", label="Loading")
        axrow[0].plot(avg.index, avg.unloading_x_mm, color="#C2642B", label="Unloading", ls="--")
        axrow[1].plot(avg.index, avg.gap_3d_mm, color="#2368A2", lw=2, label="Cycle mean")
        axrow[1].axvline(75, color="#777777", ls=":")
        for split, color, marker in (("train", "#777777", "o"), ("val", "#A68100", "s"), ("test", "#2368A2", "^")):
            a = s[s.split == split]
            axrow[2].scatter(a.cycle_id, a.pressure_mean_gap_mm, color=color, marker=marker, label=split)
        axrow[0].set(xlabel="Pressure command (kPa)", ylabel="NDI probe x (mm)", title=f"{seq[-6:]} | {len(s)} full cycles", ylim=(-15.5, 3.5))
        axrow[1].set(xlabel="Matched pressure (kPa)", ylabel="3D branch separation (mm)", title=f"Mean command speed {s.mean_abs_command_slope_kpa_s.mean():.2f} kPa/s", ylim=(0, 7.5))
        axrow[2].set(xlabel="Complete cycle index", ylabel="Pressure-mean 3D gap (mm)", title="Chronological cycle split")
        for ax in axrow: ax.legend(fontsize=8)
    save(fig, "01_observed_hysteresis")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, seq in zip(axes, ("pooled", *SEQUENCES)):
        d = errors[(errors.sequence == seq) & (errors.split == "test")].set_index("model").loc[list(MODEL_NAMES)]
        ax.bar(range(5), d.mean_3d_mm, color=[COLORS[m] for m in MODEL_NAMES])
        ax.set_xticks(range(5), [MODEL_EN[m] for m in MODEL_NAMES], rotation=25, ha="right")
        ax.set(ylabel="Mean 3D NDI error (mm)", title=f"{seq[-6:]} | {int(d.frames.iloc[0])} test frames", ylim=(0, errors[errors.split == 'test'].mean_3d_mm.max()*1.2))
        for i, v in enumerate(d.mean_3d_mm): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    save(fig, "02_held_cycle_fitting")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axrow, seq in zip(axes, SEQUENCES):
        pred = predictions[(predictions.sequence == seq) & (predictions.split == "test")]
        first = pred.cycle_id.min(); pred = pred[pred.cycle_id == first]
        observed = pred[pred.model == "static_cubic"]
        axrow[0].plot(observed.t_sec, observed.true_x_relative_mm, color="#222222", marker="o", ms=3, label="Observed")
        for model in MODEL_NAMES:
            d = pred[pred.model == model]
            axrow[0].plot(d.t_sec, d.pred_x_relative_mm, color=COLORS[model], label=MODEL_EN[model], ls="--" if model != "dual" else "-")
            d = pred_loops[(pred_loops.sequence == seq) & (pred_loops.model == model) & (pred_loops.split == "test")]
            avg = d.groupby("pressure_kpa").gap_3d_mm.mean()
            axrow[1].plot(avg.index, avg.values, color=COLORS[model], label=MODEL_EN[model], ls="--" if model != "dual" else "-")
        l = loops[(loops.sequence == seq) & (loops.split == "test")].groupby("pressure_kpa").gap_3d_mm.mean()
        axrow[1].plot(l.index, l.values, color="#222222", label="Observed", lw=2)
        axrow[0].set(xlabel="Recorded time (s)", ylabel="NDI probe x from train baseline (mm)", title=f"{seq[-6:]} | first held cycle {first}", ylim=(-1, 18))
        axrow[1].set(xlabel="Matched pressure (kPa)", ylabel="3D branch separation (mm)", title=f"{seq[-6:]} | held-cycle mean loop gap", ylim=(-.1, 6.2))
        for ax in axrow: ax.legend(fontsize=8)
    save(fig, "03_test_traces_and_loop_fit")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--report-stem", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    run, report = args.run_dir, args.report_stem
    run.mkdir(parents=True, exist_ok=True); report.parent.mkdir(parents=True, exist_ok=True)
    write_json(run / "protocol.json", PROTOCOL)
    frames, cycles, provenance = load_data()
    cycles.to_csv(run / "cycle_split.csv", index=False)
    frames.to_csv(run / "aligned_frames.csv", index=False)
    write_json(run / "data_inventory.json", provenance)
    print("Fixed cycle split:", cycles.groupby(["sequence", "split"]).size().to_dict(), flush=True)
    features = feature_data(frames)
    features.to_csv(run / "history_features.csv", index=False)
    selected, predictions, val_grid = fit_models(frames, features, run)
    errors, per_cycle = error_summary(predictions, selected)
    errors.to_csv(run / "fit_metrics.csv", index=False)
    per_cycle.to_csv(run / "fit_metrics_per_cycle.csv", index=False)
    loops, summaries = measure_loops(frames, cycles)
    loops.to_csv(run / "matched_pressure_loops.csv", index=False)
    summaries.to_csv(run / "cycle_metrics.csv", index=False)
    pred_loop_frames, pred_summary_frames = [], []
    for model in MODEL_NAMES:
        d = predictions[predictions.model == model].copy()
        pl, ps = measure_loops(d, cycles, [f"pred_{axis}_mm" for axis in "xyz"])
        pl["model"], ps["model"] = model, model
        pred_loop_frames.append(pl); pred_summary_frames.append(ps)
    pred_loops, pred_summaries = pd.concat(pred_loop_frames), pd.concat(pred_summary_frames)
    pred_loops.to_csv(run / "predicted_matched_pressure_loops.csv", index=False)
    pred_summaries.to_csv(run / "predicted_cycle_metrics.csv", index=False)
    sequence_summary = []
    for seq in SEQUENCES:
        d = summaries[summaries.sequence == seq]
        row = dict(sequence=seq, complete_cycles=len(d), frames=len(frames[frames.sequence == seq]),
            mean_command_speed_kpa_s=d.mean_abs_command_slope_kpa_s.mean())
        for metric in ("gap_75kpa_mm", "pressure_mean_gap_mm", "signed_x_loop_area_mm_kpa", "absolute_x_branch_area_mm_kpa", "closure_gap_mm"):
            row.update({metric+"_mean": d[metric].mean(), metric+"_std": d[metric].std(ddof=1),
                        metric+"_min": d[metric].min(), metric+"_max": d[metric].max()})
        sequence_summary.append(row)
    seq_summary = pd.DataFrame(sequence_summary)
    seq_summary.to_csv(run / "sequence_summary.csv", index=False)
    # Across-cycle reproducibility is descriptive dispersion at the same pressure/branch.
    branch_repeat = []
    for (seq, pressure), d in loops.groupby(["sequence", "pressure_kpa"], sort=False):
        for branch in ("loading", "unloading"):
            xyz = d[[f"{branch}_{axis}_mm" for axis in "xyz"]].to_numpy()
            center = xyz.mean(0); err = np.linalg.norm(xyz-center, axis=1)
            branch_repeat.append(dict(sequence=seq, pressure_kpa=pressure, branch=branch, cycles=len(d),
                mean_x_mm=center[0], mean_y_mm=center[1], mean_z_mm=center[2],
                mean_3d_distance_to_cycle_mean_mm=err.mean(), rms_3d_distance_to_cycle_mean_mm=np.sqrt(np.mean(err**2))))
    repeat = pd.DataFrame(branch_repeat); repeat.to_csv(run / "branch_repeatability.csv", index=False)
    # Fit loop gap and signed projected area, not just frame-position accuracy.
    loop_fit = pred_summaries.merge(summaries[["sequence", "cycle_id", "gap_75kpa_mm", "pressure_mean_gap_mm", "signed_x_loop_area_mm_kpa"]],
        on=["sequence", "cycle_id"], suffixes=("_pred", "_observed"), validate="many_to_one")
    for metric in ("gap_75kpa_mm", "pressure_mean_gap_mm", "signed_x_loop_area_mm_kpa"):
        loop_fit[metric+"_absolute_error"] = abs(loop_fit[metric+"_pred"]-loop_fit[metric+"_observed"])
    loop_fit.to_csv(run / "loop_fit_metrics.csv", index=False)
    loop_fit_summary = loop_fit[loop_fit.split == "test"].groupby(["model", "sequence"], as_index=False).agg(
        test_cycles=("cycle_id", "count"),
        gap75_absolute_error_mm=("gap_75kpa_mm_absolute_error", "mean"),
        mean_gap_absolute_error_mm=("pressure_mean_gap_mm_absolute_error", "mean"),
        projected_area_absolute_error_mm_kpa=("signed_x_loop_area_mm_kpa_absolute_error", "mean"))
    loop_fit_summary.to_csv(run / "loop_fit_summary.csv", index=False)
    charts = make_charts(frames, loops, summaries, errors, per_cycle, predictions, pred_loops)
    r75 = repeat[repeat.pressure_kpa == 75].copy()
    r75["sequence_branch"] = r75.sequence.str[-6:]+":"+r75.branch
    charts.extend([
        dict(id="repeatability75", title="75 kPa同分支的跨周期位置离散", kind="bar",
             x="sequence_branch", y="mean_3d_distance_to_cycle_mean_mm", color=None, unit="mm", rows=records(r75),
             note="每柱为同一序列同一分支各周期位置到分支均值的平均三维距离，属于记录内重复性。"),
        dict(id="loop_gap_fit_error", title="测试周期的回线平均间距拟合误差", kind="bar",
             x="model", y="mean_gap_absolute_error_mm", color="sequence", unit="mm", rows=records(loop_fit_summary),
             note="每周期先计算预测与实测全压力平均间距的绝对差，再对测试周期平均。"),
        dict(id="loop_area_fit_error", title="测试周期的NDI x投影面积拟合误差", kind="bar",
             x="model", y="projected_area_absolute_error_mm_kpa", color="sequence", unit="mm·kPa", rows=records(loop_fit_summary),
             note="评价回线几何拟合；不是控制补偿或能耗评估。"),
    ])
    figures = save_figures(run, frames, loops, summaries, errors, predictions, pred_loops)
    pooled = errors[(errors.sequence == "pooled") & (errors.split == "test")].set_index("model")
    findings = []
    for row in sequence_summary:
        findings.append(dict(id="observed_"+row["sequence"][-6:],
            text=f"{row['sequence'][-6:]} 含 {row['complete_cycles']} 个完整周期；记录扫压速度均值 {row['mean_command_speed_kpa_s']:.2f} kPa/s；75 kPa 加卸载三维间距 {row['gap_75kpa_mm_mean']:.3f} ± {row['gap_75kpa_mm_std']:.3f} mm；全压力均值 {row['pressure_mean_gap_mm_mean']:.3f} mm。",
            evidence="sequence_summary.csv; cycle_metrics.csv", status="descriptive_within_recording"))
    static_error, dual_error = pooled.loc["static_cubic", "mean_3d_mm"], pooled.loc["dual", "mean_3d_mm"]
    best_name = pooled.mean_3d_mm.idxmin()
    findings.append(dict(id="held_cycle_fitting",
        text=f"固定完整周期划分后，静态三次基函数测试误差 {static_error:.3f} mm，双记忆 {dual_error:.3f} mm，相对降低 {(1-dual_error/static_error)*100:.1f}%；本组固定配置中测试误差最低为 {MODEL_NAMES[best_name]}（{pooled.loc[best_name, 'mean_3d_mm']:.3f} mm）。",
        evidence="fit_metrics.csv; frozen_models.json", status="descriptive_held_cycle"))
    slow_test = errors[(errors.sequence == SEQUENCES[1]) & (errors.split == "test")].set_index("model")
    findings.append(dict(id="incremental_memory_evidence",
        text=f"方向特征、路径记忆、时间记忆的测试误差分别为 {pooled.loc['direction_cubic','mean_3d_mm']:.3f}、{pooled.loc['play','mean_3d_mm']:.3f}、{pooled.loc['time','mean_3d_mm']:.3f} mm。双记忆相对时间记忆仅进一步降低 {(1-dual_error/pooled.loc['time','mean_3d_mm'])*100:.2f}%；慢速测试上时间记忆{slow_test.loc['time','mean_3d_mm']:.4f} mm、双记忆{slow_test.loc['dual','mean_3d_mm']:.4f} mm。本数据对时间特征的增益证据较强，对路径分支额外作用的证据有限。",
        evidence="fit_metrics.csv", status="descriptive_held_cycle"))
    rep_lookup = r75.set_index(["sequence", "branch"])
    rep_strings = [f"{seq[-6:]}加载/卸载{rep_lookup.loc[(seq,'loading'),'mean_3d_distance_to_cycle_mean_mm']:.3f}/{rep_lookup.loc[(seq,'unloading'),'mean_3d_distance_to_cycle_mean_mm']:.3f} mm" for seq in SEQUENCES]
    findings.append(dict(id="repeatability",
        text="75 kPa同一分支各周期到分支均值的平均三维距离："+"，".join(rep_strings)+"，均小于对应加卸载间距。这支持记录内存在可重复的路径相关位置差异，尚未分离测量噪声与真实周期波动。",
        evidence="branch_repeatability.csv; sequence_summary.csv", status="descriptive_within_recording"))
    findings.append(dict(id="scope", text="双速度扫压支持检验当前输入的多值响应及历史特征的预测作用；单一幅值三角波没有嵌套反转或长时保持，无法独立辨识路径与时间机制，也不能把各分支读出解释为材料物理参数。", status="limitation"))
    findings.append(dict(id="not_compensation", text="本分析衡量 NDI 位置与回线的前向拟合；没有执行逆补偿或反馈控制，不能称为消除迟滞影响或控制改善。", status="scope"))
    definitions = {
        "measurement": "NDI探头位置(x,y,z)，单位mm；不等同全身形态或已认证末端位置。",
        "gap_3d": "D_c(P)=||y_c,unload(P)-y_c,load(P)||_2；各完整周期按相同压力线性插值匹配。",
        "mean_gap": "(1/150)∫_0^150 D_c(P)dP，使用1 kPa网格梯形积分；不是简单平均非均匀时间采样点。",
        "gap75": "D_c(75 kPa)。原始15 kPa步进包含75 kPa，不依赖网格外推。",
        "signed_x_area": "A_x=∫_0^150[x_unload(P)-x_load(P)]dP，单位mm·kPa，为有向几何投影面积；非能耗。",
        "absolute_x_area": "∫_0^150|x_unload(P)-x_load(P)|dP，交叉回线也不发生正负抵消。",
        "endpoint": "两支共享同一峰值记录，所以150 kPa处差异按定义为0；0 kPa比较周期首尾，可见漂移/未闭合。",
        "speed": "每完整周期总指令变化300 kPa/真实周期时长，另列上升150/Δt和下降−150/Δt；不是实测腔压导数。",
        "statistics": "±为同一次连续记录中逐周期样本标准差；不作独立重复试验置信区间或显著性推断。",
        "generalization": "两种速度均包含训练数据；测试为相同幅值及激励模式的未来周期，不是未见速度或未见加载路径的泛化评估。",
        "baseline": PROTOCOL["baseline"], "memory": "路径记忆用play/stop；时间记忆用真实dt的一阶递推。两者均有压力依赖线性读出，是同类结构的单输入NDI验证。",
        "direction": "方向模型为静态三次基函数+方向与0至3次压力幂的交互；借鉴Chen方向编码思想，并非Chen原文MLP或原实验复现。",
        "capacity": "静态/方向/路径/时间/双记忆分别有4/8/19/19/34个线性特征（每个输出坐标一组系数）；本小型验证不是容量完全匹配的正式模型排名。",
        "settling": "meta记录settle_s=0.19/0.49 s，仅为采集等待设定；单次记录不能据此称严格准静态。",
    }
    tables = [dict(id="sequence_summary", title="扫压与回线统计", rows=records(seq_summary)),
        dict(id="splits", title="完整周期分割", rows=records(cycles)),
        dict(id="test_metrics", title="测试期NDI预测误差", rows=records(errors[errors.split == "test"])),
        dict(id="all_fit_metrics", title="训练验证测试误差", rows=records(errors)),
        dict(id="cycle_metrics", title="逐周期实测回线", rows=records(summaries)),
        dict(id="held_cycle_metrics", title="逐测试周期预测误差", rows=records(per_cycle[per_cycle.split == "test"])),
        dict(id="validation_grid", title="固定岭系数候选的验证结果", rows=records(val_grid)),
        dict(id="loop_fit", title="逐测试周期回线指标拟合", rows=records(loop_fit[loop_fit.split == "test"])),
        dict(id="loop_fit_summary", title="回线拟合误差汇总", rows=records(loop_fit_summary)),
        dict(id="branch_repeatability", title="跨周期同压同分支重复性", rows=records(repeat))]
    write_json(report.with_suffix(".json"), dict(schema="modeling_mechanisms_sweep_v1", findings=findings,
        definitions=definitions, charts=charts, tables=tables, protocol=PROTOCOL,
        provenance=dict(sources=provenance, script="scripts/experiments/analyze_modeling_sweep_hysteresis.py",
            output_directory=str(run.relative_to(ROOT)), static_figures=figures,
            literature_source="docs/paper/icra2027/modeling_baselines_sources.md, Chen方向编码条目；未采用2026-07-16报告的文献评语或回线数值")))
    lines = ["# 早期周期扫压：NDI迟滞回线与历史特征拟合", "", *[f["text"]+"\n" for f in findings],
        "## 数据与预先固定的评估协议", "", "两条序列均为单通道c0的0↔150 kPa三角扫压，其余通道为0；actions6.csv与ndi.csv按t_sec严格一对一配准，727行全部匹配，未进行标签筛选或平滑。读取NDI原始xyz，单位mm。两种速度均包含训练数据，测试针对相同幅值与激励模式的未来完整周期，不代表未见速度或加载路径的泛化。"]
    for p in provenance:
        lines.append(f"\n- {p['sequence']}：{p['matched_rows']}行，实际dt中位数{p['recorded_dt_median_s']:.3f}s，完整周期{p['complete_cycles']}；train/val/test={p['train_cycles']}/{p['val_cycles']}/{p['test_cycles']}。{p['excluded_from_fit_rows']}个头尾非完整周期/边界帧不计入拟合指标。")
    lines += ["", "每周期用于拟合的记录区间为左闭右开，因此跨分割共享的低压边界没有重复标签。回线计算包含两个端点，便于比较加载与卸载分支。初始不足一个周期的驱动记录仅用于递推状态预热。传感器坐标偏移由每条序列第一个训练周期的首个零压NDI位置去除，此后不再用验证或测试目标校准。", "",
        "五种预测器使用固定三次静态基函数；方向对照增加方向与各次幂的交互。路径记忆阈值为归一化压力的0.05、0.10、0.20、0.35、0.50（即7.5、15、30、52.5、75 kPa）；时间记忆时间常数为0.6、1.2、2.4、4.8、9.6 s，以真实dt更新。每个记忆量与1、e、e²相乘，形成可解释的压力依赖读出。所有阈值、时间常数与特征在拟合前固定，仅岭系数在相同验证集上选择。未用train+val重拟合。", "",
        "## 回线定义与重复性", "", "同一压力逐周期线性插值，计算三维间距D(P)、75 kPa间距、0–150 kPa压力平均间距，以及NDI x有向投影面积∫(x卸载−x加载)dP。面积单位mm·kPa，是几何量；不代表能耗。插值网格为1 kPa，原始步长15 kPa，没有增加独立观测数。", "",
        "| 序列 | 周期 | 平均扫压速度 kPa/s | 75 kPa间距 mm（周期均值±SD） | 全压力平均间距 mm | x投影面积 mm·kPa（均值±SD） |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in sequence_summary:
        lines.append(f"| {r['sequence'][-6:]} | {r['complete_cycles']} | {r['mean_command_speed_kpa_s']:.2f} | {r['gap_75kpa_mm_mean']:.3f} ± {r['gap_75kpa_mm_std']:.3f} | {r['pressure_mean_gap_mm_mean']:.3f} | {r['signed_x_loop_area_mm_kpa_mean']:.2f} ± {r['signed_x_loop_area_mm_kpa_std']:.2f} |")
    lines += ["", "## 按完整周期留出的预测结果", "", "主指标为测试帧汇总的三维欧氏误差均值；7个测试周期、140个不重复测试帧，其中快速记录100帧、慢速记录40帧。所有逐周期结果保留，不报告伪独立显著性。", "",
        "| 预测器 | 系数数目 | 验证选择λ | 验证误差 mm | 测试误差 mm | 快速测试 mm | 慢速测试 mm |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODEL_NAMES:
        m = selected[model]
        vals = [errors[(errors.model == model) & (errors.split == "test") & (errors.sequence == s)].mean_3d_mm.iloc[0] for s in SEQUENCES]
        lines.append(f"| {MODEL_NAMES[model]} | {m['coefficient_count']} | {m['lambda_ridge']:g} | {m['validation_mean_3d_mm']:.4f} | {pooled.loc[model,'mean_3d_mm']:.4f} | {vals[0]:.4f} | {vals[1]:.4f} |")
    lines += ["", "上述方向对照只借鉴Chen的方向编码思想，采用三次线性读出，不是论文原始网络的复现。记忆结构也是单输入NDI位置验证，不是正式四通道HOV的跨本体迁移或控制补偿。特征数量不同，性能差异同时含容量与表示影响；分支贡献大小不能直接解释成独立物理效应。", "",
        "## 证据边界与后续采集", "", "- 两种扫压速度各只有一次记录，周期重复属于同一段连续实验；传感噪声、漂移、气动动态与材料历史响应尚未分离。", "- 采集间隔改变的同时，15 kPa步进的加载速度发生改变。这里不构成固定动作轨迹的纯降采样实验，也不构成严格的材料率效应辨识。", "- 初始NDI偏移只从第一个训练零压样本去除。测试期漂移保留在误差中；初始记忆由首条记录初始化，未记录的更早历史不可恢复。", "- 三角波只有单一幅值的主回线，没有嵌套反转、不同幅值小回线或长时间持压。方向、路径记忆与时间记忆可能相关，本数据只能比较这些固定预测特征，不能验证全部路径记忆公理或唯一分离两种机制。", "- 当前时间模型按记录时间作当前驱动电平近似；更准确的阶跃/曝光时间、实测腔压均未记录，短时间常数不作物理参数解释。", "- meta中的settle_s为采集等待设置，不能据此称严格准静态。150 kPa处两支共用峰值，间距按定义为0；零压处包含周期首尾漂移。", "- 若进一步检验机制，应增加同一加载路径的多速度独立重复、长时持压、嵌套反转与不同预条件；若要说明消除迟滞影响，还需实际逆补偿或闭环控制实验。", "",
        "## 结果与复现文件", "", "运行：`OMP_NUM_THREADS=4 /Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/analyze_modeling_sweep_hysteresis.py`。", "", f"原始分割、逐帧配准、历史特征、验证网格、冻结模型、预测与循环指标均在`{run.relative_to(ROOT)}`。同名JSON包含findings、definitions、charts（含完整图表行数据）、tables、provenance，供汇总HTML使用。", ""]
    for image in figures:
        if image.endswith(".png"):
            relative = os.path.relpath(ROOT / image, report.parent)
            lines.append(f"![{Path(image).stem}]({relative})\n")
    report.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(errors[(errors.split == "test")].to_string(index=False), flush=True)
    print(seq_summary.to_string(index=False), flush=True)
    print(f"Wrote {report.with_suffix('.json')} ({len(charts)} chart specs) and Markdown", flush=True)


if __name__ == "__main__":
    main()
