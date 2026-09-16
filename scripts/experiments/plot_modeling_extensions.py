#!/usr/bin/env python3
"""Export the 006 extension figures from completed, source-backed analyses.

Only this script, docs/icra2027/figures/extensions006, and the matching report
figures directory are written. Model code, results and original figures are
read-only. --figures supports incremental rendering while plugin fits finish.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/selfsr-extensions006-mpl")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits
from plot_unified20_results import BLUE, ORANGE, GRAY, INK, EN

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
OLD = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005"
ANALYSIS = ROOT / "workspace/runs/analysis/modeling_extensions_20260913_006"
FIG = ROOT / "docs/icra2027/figures/extensions006"
REPORT = ROOT / "workspace/reports/modeling_extensions_20260913_006/figures"
SEEDS = list(range(100, 120))
MAIN = ["linear", "pcc", "base", "koopman", "oscillator", "chen_direction", "hov"]
VARIANTS = ["base", "path", "time", "both", "static_capacity"]
NAMES = ["initialization_test_stages", "internal_memory_mlp", "internal_memory_plugins", "prediction_summary",
         "memory_plugin", "learning_and_inference"]
LABELS = {"base": "Base", "path": "+ Path", "time": "+ Time", "both": "+ Both",
          "static_capacity": "Polynomial\ncontrol"}
GRID = "#E3E7EB"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11.5, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.titlepad": 12,
    "svg.fonttype": "none", "svg.hashsalt": "modeling_extensions006", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A8AFB8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.grid": False})


def read(path):
    return json.loads(path.read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value.relative_to(ROOT))
    return value


def stat(values):
    a = np.asarray(values, dtype=float)
    assert len(a) and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=.7)
    ax.set_axisbelow(True)


def values(rows, model, metric, *, deterministic=False):
    group = rows.loc[rows.model == model].sort_values("seed")
    if deterministic:
        assert len(group) == 1, (model, len(group))
    else:
        assert group.seed.astype(int).tolist() == SEEDS, model
    assert (group.test_frames == 2958).all(), model
    a = group[metric].to_numpy(dtype=float)
    assert np.isfinite(a).all(), (model, metric)
    return a


def title(fig, headline, subtitle):
    fig.suptitle(headline, x=.06, y=.974, ha="left", fontsize=15.5, weight="bold")
    # Maintain a physical header gap on the shorter single-row canvases.
    subtitle_y = min(.918, .974 - 31 / (72 * fig.get_figheight()))
    fig.text(.06, subtitle_y, subtitle, fontsize=10)


def finish(fig, name, caption, details):
    fig.canvas.draw()
    for ext in ("svg", "pdf", "png"):
        meta = {"Date": None} if ext == "svg" else ({"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        path = FIG / f"{name}.{ext}"
        fig.savefig(path, dpi=250, metadata=meta)
        shutil.copyfile(path, REPORT / path.name)
        assert path.read_bytes() == (REPORT / path.name).read_bytes()
    details["canvas_inches"] = fig.get_size_inches().tolist()
    details["formats"] = ["svg", "pdf", "png"]
    details["png_dpi"] = 250
    plt.close(fig)
    return dict(id=name, caption=caption, formats=["svg", "pdf", "png"], path=str((FIG / f"{name}.png").relative_to(ROOT))), details


def initialization():
    source = ANALYSIS / "initialization"
    summary = read(source / "summary.json")
    assert summary["status"] == "complete" and summary["independent_initialization_fits"] == 1
    assert summary["subsequent_joint_epochs"] == 0
    assert summary["observed_prefit"]["joint_Adam_steps"] == 0
    stages = pd.read_csv(source / "stage_metrics.csv")
    stages = stages.loc[stages.role == "test"].set_index("stage")
    assert len(stages) == 2 and (stages.frames == 2958).all()
    final = pd.read_csv(source / "final_hov_per_seed.csv").sort_values("seed")
    assert final.seed.tolist() == SEEDS and (final.frames == 2958).all()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.4))
    fig.subplots_adjust(left=.085, right=.97, top=.78, bottom=.28, wspace=.28)
    title(fig, "Test prediction at initialization and after joint training",
          "Reference fitting, memory initialization, and joint optimization  •  shared test set")
    payload = {}
    for ax, metric, heading in zip(axes, ["mean_node_mm", "endpoint_mm"], ["(a) Skeleton error", "(b) Endpoint error"]):
        initial = [float(stages.loc[k, metric]) for k in ("reference_prefit", "hov_joint_epoch0")]
        finals = final[metric].to_numpy()
        np.testing.assert_allclose(finals.mean(), summary["final_hov_summary"][metric]["mean"], atol=1e-12, rtol=0)
        for i, (v, color) in enumerate(zip(initial, [GRAY, ORANGE])):
            ax.scatter(i, v, marker="D", s=48, color=color, edgecolors=INK, linewidths=.5, zorder=3)
            ax.annotate(f"{v:.4f}", (i, v), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=10)
        ax.scatter(2+np.linspace(-.085, .085, 20), finals, color=BLUE, s=14, alpha=.38, zorder=2)
        ax.errorbar(2, finals.mean(), yerr=finals.std(ddof=1), fmt="o", color=BLUE, ms=6, capsize=4, zorder=4)
        ax.annotate(f"{finals.mean():.4f} ± {finals.std(ddof=1):.4f}", (2, finals.mean()),
                    xytext=(0, 16), textcoords="offset points", ha="center", fontsize=10)
        all_values = np.r_[initial, finals]
        span = np.ptp(all_values)
        ax.set(title=heading, ylabel="Error (mm; lower is better)", xlim=(-.3, 2.42),
            ylim=(all_values.min()-.18*span, all_values.max()+.24*span), xticks=[0, 1, 2],
            xticklabels=["Reference prefit", "Memory initialization\nBefore joint training", "After joint training"])
        grid(ax)
        payload[metric] = dict(reference_prefit=stat([initial[0]]), zero_joint_epoch=stat([initial[1]]),
                               final=stat(finals), zero_joint_minus_final_mm=stat(initial[1]-finals))
    fig.text(.06, .114, "Diamonds: two stages of one initialization. Final points and whiskers: 20 repetitions and mean ± SD.", fontsize=9.5)
    fig.text(.06, .065, "Reference and memory initialization use the training set; subsequent optimization uses the full geometric loss.", fontsize=9.5)
    return finish(fig, "initialization_test_stages",
        "Two stages of one reconstructed training-only initialization (reference prefit; reference plus memory ridge initialization at zero subsequent joint epochs), followed by the 20 final HOV fits. All stages use the same 2958 test targets. Initialization diamonds have no cross-seed SD; final points and whiskers summarize seeds 100–119. Epoch zero already contains reference and memory fitting.",
        dict(sources=[source / "summary.json", source / "stage_metrics.csv", source / "final_hov_per_seed.csv"],
             metrics=payload, actual_prefit_updates=summary["observed_prefit"],
             seed_rule="One actual initialization reconstruction with seed100; two stages share that fit. Final models retain seeds100..119.",
             units="Euclidean distances in mm; skeleton is pooled over all frames and 15 nodes, endpoint is node14",
             limitations=["The fixed initialization scalar is not 20 independent initializations.",
                "Stage-to-final differences are conditional on this one initialization and the previously inspected dataset.",
                "This figure uses test errors; the learning figure uses validation errors. Neither represents zero training-data exposure."]))


def prediction():
    path = SOURCE / "raw_test.csv"
    raw = pd.read_csv(path)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.0), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, top=.79, bottom=.20, wspace=.21)
    title(fig, "Whole-shape prediction accuracy", "Fixed test split  •  20 training seeds per stochastic model  •  one deterministic linear fit")
    payload = {}
    for ax, metric, heading, xlabel in zip(axes, ["mean_node_mm", "endpoint_mm", "mask_iou"],
        ["(a) Skeleton error", "(b) Endpoint error", "(c) Silhouette overlap"],
        ["Mean node distance (mm)", "Endpoint distance (mm)", "Mask IoU"]):
        payload[metric] = {}
        limits = []
        for i, model in enumerate(MAIN):
            a = values(raw, model, metric, deterministic=model == "linear")
            payload[metric][model] = stat(a)
            color = BLUE if model == "hov" else GRAY
            ax.scatter(a, i+np.linspace(-.12, .12, len(a)) if len(a)>1 else [i], color=color, s=13, alpha=.35)
            ax.errorbar(a.mean(), i, xerr=a.std(ddof=1) if len(a)>1 else 0,
                        fmt="o", color=color, ms=5.5, capsize=3, mec=INK, mew=.5)
            limits.extend(a)
        margin = np.ptp(limits)*.09
        ax.set(title=heading, xlabel=xlabel, xlim=(min(limits)-margin, max(limits)+margin))
        grid(ax, "x")
    axes[0].set(yticks=range(len(MAIN)), yticklabels=[EN[m] for m in MAIN])
    axes[0].invert_yaxis()
    fig.text(.06, .065, "Dots: all training repeats. Centers/whiskers: mean ± sample SD. Distances decrease with accuracy; IoU increases.", fontsize=9.5)
    return finish(fig, "prediction_summary", "Main-model prediction metrics on all 2958 fixed test frames. Seven displayed models; all 20 seeds are retained for stochastic models and the deterministic linear model is shown once. Dots and mean ± sample SD; focused point-plot axes.",
        dict(sources=[path], models=MAIN, metrics=payload,
             units=dict(mean_node_mm="mm, 15-node frame-pooled Euclidean distance", endpoint_mm="mm, endpoint node14", mask_iou="unitless intersection over union"),
             selection="Fixed method list from the task; retain every seed and test target",
             limitations=["Seed SD describes training randomness conditional on the fixed temporal split.", "Visual masks and skeletons share annotation sources."]))


def input_plugin():
    path = SOURCE / "raw_test.csv"
    raw = pd.read_csv(path)
    contrasts = [r for r in read(SOURCE / "paired_statistics.json")["contrasts"]
                 if r["family"] == "plugin" and r["alternative"] in VARIANTS[1:]]
    assert len(contrasts) == 4
    lookup = {r["alternative"]: r for r in contrasts}
    linear = ["linear", "linear_path", "linear_time", "linear_both", "linear_static_capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.235, wspace=.32)
    title(fig, "Memory features at the prediction input", "Fixed history encoding  •  linear readout and 64/64 MLP  •  20 training repetitions")
    mlp_stats, linear_stats = {}, {}
    for i, variant in enumerate(VARIANTS):
        a = values(raw, variant, "mean_node_mm")
        b = values(raw, linear[i], "mean_node_mm", deterministic=True)
        mlp_stats[variant], linear_stats[variant] = stat(a), stat(b)
        axes[0].scatter(i+.10+np.linspace(-.055, .055, 20), a, s=10, color=BLUE, alpha=.26)
        axes[0].errorbar(i+.10, a.mean(), yerr=a.std(ddof=1), fmt="o", color=BLUE, capsize=3,
                         label="MLP: mean ± SD" if i==0 else None)
        axes[0].plot(i-.13, b[0], marker="s", mfc="white", mec=GRAY, color=GRAY, ls="none",
                     label="Linear: one fit" if i==0 else None)
    axes[0].set(title="(a) Test skeleton error", xticks=range(5),
        xticklabels=["Current", "+ Path", "+ Time", "+ Both", "Static\nfeatures"], ylabel="Mean node distance (mm)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    grid(axes[0])
    base = values(raw, "base", "mean_node_mm")
    extrema = [0.]
    for i, variant in enumerate(VARIANTS[1:]):
        row = lookup[variant]
        d = base-values(raw, variant, "mean_node_mm")
        e, lo, hi = [row[k] for k in ("mean_reference_minus_alternative_mm", "bootstrap95_lower_mm", "bootstrap95_upper_mm")]
        np.testing.assert_allclose(d.mean(), e, atol=1e-12, rtol=0)
        axes[1].errorbar(e, i, xerr=[[e-lo], [hi-e]], fmt="o", color=BLUE, capsize=4)
        axes[1].annotate(f"{e:+.3f}", (hi, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9)
        extrema += [lo, hi]
    span = np.ptp(extrema)
    axes[1].set(title="(b) Paired MLP effects", yticks=range(4),
        yticklabels=["+ Path", "+ Time", "+ Both", "Static features"],
        xlabel="Current-input MLP − variant error (mm)",
        xlim=(min(extrema)-.09*span, max(extrema)+.27*span))
    axes[1].invert_yaxis(); axes[1].axvline(0, color=INK, lw=.8); grid(axes[1], "x")
    fig.text(.06, .085, "Left: mean ± sample SD of 20 MLP seeds. Right: paired-seed bootstrap 95% CIs; positive values mean lower error.", fontsize=9.5)
    fig.text(.06, .04, "Fixed memory features are concatenated with current pressure; the prediction network is trained from scratch.", fontsize=9.5)
    return finish(fig, "memory_plugin", "Fixed memory features supplied at the input of linear and MLP predictors. Five displayed feature variants. Left: all 20 MLP seeds and sample SD, one fit per linear variant. Right: original paired-bootstrap 95% intervals for four displayed MLP contrasts; positive differences favor the feature variant.",
        dict(sources=[path, SOURCE / "paired_statistics.json"], variants=VARIANTS,
             mlp_metrics_mm=mlp_stats, linear_metrics_mm=linear_stats, paired_statistics=contrasts,
             units="Pooled test mean-node Euclidean distance in mm", ci="Use the existing 004 paired bootstrap95 intervals; no statistical family is refit for display",
             limitations=["The original inferential plugin family remains five comparisons, even though four are displayed here.",
                "These encoders provide input features; the 006 internal-branch experiment jointly trains its drive transformation and network."]))


def learning():
    folder = OLD / "efficiency"
    history = pd.read_csv(folder / "training_history_raw.csv")
    training = pd.read_csv(folder / "training_summary.csv").set_index("model")
    latency = pd.read_csv(folder / "inference_summary.csv").set_index("mode")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.subplots_adjust(left=.115, right=.97, top=.83, bottom=.165, hspace=.59, wspace=.37)
    title(fig, "Learning trajectories and inference cost", "20 training seeds per model  •  current validation errors  •  CPU inference timings")
    styles = {"hov": (BLUE, "-", "o"), "chen_direction": (GRAY, "--", "s"), "base": (INK, ":", "^")}
    for mi, (model, (color, style, marker)) in enumerate(styles.items()):
        rows = history.loc[history.model == model]
        groups = rows.groupby("epoch")
        for epoch, block in groups:
            assert sorted(block.seed.astype(int)) == SEEDS, (model, epoch)
        aggregate = groups.validation_node_mean_mm.agg(["mean", "std"])
        for ax, early in ((axes[0, 0], True), (axes[0, 1], False)):
            g = aggregate.loc[aggregate.index <= 10] if early else aggregate.loc[aggregate.index >= 10]
            if early:
                ax.errorbar(g.index+(mi-1)*.16, g["mean"], yerr=g["std"], fmt=marker,
                            color=color, ms=5, capsize=3, label=EN[model])
            else:
                ax.plot(g.index, g["mean"], color=color, ls=style, marker=marker, ms=2.5, lw=1.4)
                ax.fill_between(g.index, g["mean"]-g["std"], g["mean"]+g["std"], color=color, alpha=.1, linewidth=0)
    axes[0, 0].set(title="(a) Early validation checkpoints", yscale="log", xlabel="Joint-training epoch",
                  ylabel="Skeleton error (mm; log scale)", xticks=[1, 5, 10], xlim=(.4, 10.6))
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0, 1].set(title="(b) Validation trajectory", xlabel="Joint-training epoch", ylabel="Skeleton error (mm)", xticks=[10, 25, 50, 75, 100])
    for ax in axes[0]: grid(ax)
    train_notes = {}
    for i, model in enumerate(MAIN):
        row = training.loc[model]
        mu, sd = row.task_wall_to_COMPLETE_seconds_mean, row.task_wall_to_COMPLETE_seconds_sd
        train_notes[model] = dict(mean_seconds=float(mu), sd_seconds=None if pd.isna(sd) else float(sd), n=int(row.n_models))
        axes[1, 0].errorbar(mu, i, xerr=0 if pd.isna(sd) else sd, fmt="o", color=BLUE if model=="hov" else GRAY, capsize=3, ms=5)
        axes[1, 0].annotate(f"{mu:.2f}", (mu, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=8.5)
    axes[1, 0].set(title="(c) CPU training time", xscale="log", xlim=(.025, 150),
                  yticks=range(7), yticklabels=[EN[m] for m in MAIN], xlabel="Task wall time (s; log scale)")
    axes[1, 0].invert_yaxis(); grid(axes[1, 0], "x")
    modes = ["linear_h20", "pcc_h20", "base_h20", "koopman_h20", "oscillator_h20", "chen_direction_h20", "hov_h20_full", "hov_cached_step"]
    labels = ["Linear", "PCC", "MLP", "Koopman", "Krauss", "Chen", "HOV: full H20", "HOV: cached step"]
    latency_notes = {}
    for i, mode in enumerate(modes):
        row = latency.loc[mode]
        p50, p95 = float(row.p50_ms_mean), float(row.p95_ms_mean)
        latency_notes[mode] = dict(mean_p50_ms=p50, mean_p95_ms=p95, n_models=int(row.n_models))
        color = BLUE if mode.startswith("hov") else GRAY
        axes[1, 1].plot([p50, p95], [i, i], color=color, lw=1.3)
        axes[1, 1].scatter(p50, i, color=color, s=28, marker="o", label="p50" if i==0 else None)
        axes[1, 1].scatter(p95, i, edgecolor=color, facecolor="white", s=27, marker="s", label="p95" if i==0 else None)
    axes[1, 1].set(title="(d) CPU, one thread, batch size 1", xscale="log", xlim=(.05, 3.3),
                  yticks=range(8), yticklabels=labels, xlabel="Per-call latency (ms; log scale)")
    axes[1, 1].invert_yaxis(); grid(axes[1, 1], "x"); axes[1, 1].legend(frameon=False, ncol=2, loc="upper right")
    fig.text(.06, .082, "Learning curves: mean ± SD over 20 repetitions. Training time includes initialization, optimization, validation, and saving.", fontsize=9)
    fig.text(.06, .043, "Latency: mean of per-model p50/p95, 500 calls after 50 warmups. Cached-step latency omits history-state preparation and external sensing/actuation.", fontsize=9)
    return finish(fig, "learning_and_inference", "Current validation-error summaries for HOV, Chen and MLP at early checkpoints and epochs10–100. All 20 training seeds retained. Original task wall times for seven methods under eight-worker concurrency, including initialization and export. CPU single-thread per-call latency for eight evaluation modes, summarized as means of each model's p50/p95. Cached-state preparation is excluded from step latency.",
        dict(sources=[folder / "training_history_raw.csv", folder / "training_summary.csv", folder / "inference_summary.csv"],
             learning_models=list(styles), fitting_models=MAIN, inference_modes=modes,
             training_time=train_notes, inference_time=latency_notes,
             units=dict(learning="validation mean-node mm", training="task wall seconds", inference="milliseconds per call"),
             limitations=["Three early scheduled checkpoints are shown as points with SD, not a densely sampled learning curve.",
                "Later curves are current validation errors, not running-best trajectories or test results.",
                "Task wall time reflects the original concurrent environment and is not exclusive compute time.",
                "The one-process initialization reconstruction cannot be compared directly to these wall times as a percentage speedup."]))


def internal():
    folder = ANALYSIS / "internal_plugins"
    if not (folder / "COMPLETE.json").is_file():
        raise RuntimeError(f"Internal plugin evaluation is still running: {folder / 'summary.json'}")
    summary = read(folder / "summary.json")
    assert summary["schema"] == "internal_memory_plugin_results_v1"
    assert summary["protocol"]["seeds"] == SEEDS
    run = Path(summary["source_run"])
    raw = pd.read_csv(run / "raw_test.csv")
    assert len(raw) == 160
    family_variants = {"mlp": ["base", "both", "static_capacity"], "koopman": VARIANTS}
    means = {r["model"]: r for r in summary["models"]}
    comparisons = {r["alternative"]: r for r in summary["statistics"]}
    assert len(means) == 8 and len(comparisons) == 6
    arrays, errors, bounds = {}, [], [0.]
    resampling = np.random.default_rng(20260913).integers(0, 20, size=(20000, 20))
    for family, variants in family_variants.items():
        base = values(raw, f"{family}_base", "mean_node_mm")
        for variant in variants:
            name = f"{family}_{variant}"
            a = values(raw, name, "mean_node_mm")
            assert means[name]["n"] == 20 and means[name]["test_frames"] == 2958
            assert raw.loc[raw.model == name, "parameter_count"].eq(means[name]["parameters"]).all()
            arrays[name] = a
            errors.extend(a)
            for field, value in stat(a).items():
                if field != "n":
                    np.testing.assert_allclose(value, means[name]["mean_node_mm"][field], rtol=0, atol=1e-12)
            if variant != "base":
                r = comparisons[name]
                assert r["n_pairs"] == 20 and r["reference"] == f"{family}_base"
                delta = base-a
                ci = np.quantile(delta[resampling].mean(1), [.025, .975])
                np.testing.assert_allclose(delta, r["seed_differences_mm"], rtol=0, atol=1e-12)
                np.testing.assert_allclose(ci, [r["bootstrap95_lower_mm"], r["bootstrap95_upper_mm"]], rtol=0, atol=1e-12)
                bounds += list(ci)
        assert means[f"{family}_both"]["parameters"] == means[f"{family}_static_capacity"]["parameters"]
    fig = plt.figure(figsize=(12.2, 8.0))
    title(fig, "Trainable memory inside two prediction backbones", "20 paired training seeds per configuration  •  100-epoch budget  •  all 2,958 test targets")
    gs = fig.add_gridspec(2, 2, left=.08, right=.97, top=.80, bottom=.18, hspace=.35, wspace=.25, height_ratios=[1, 1])
    e_span, d_span = np.ptp(errors), np.ptp(bounds)
    e_limits = (min(errors)-.12*e_span, max(errors)+.20*e_span)
    d_limits = (min(bounds)-.17*d_span, max(bounds)+.25*d_span)
    for col, (family, variants) in enumerate(family_variants.items()):
        ax = fig.add_subplot(gs[0, col])
        effect_ax = fig.add_subplot(gs[1, col], sharex=ax)
        for i, variant in enumerate(variants):
            name = f"{family}_{variant}"
            a = arrays[name]
            color = GRAY if variant == "base" else ORANGE if variant == "static_capacity" else BLUE
            marker = "s" if variant == "static_capacity" else "o"
            ax.scatter(i+np.linspace(-.085, .085, 20), a, color=color, s=13, alpha=.3, zorder=2)
            ax.errorbar(i, a.mean(), yerr=a.std(ddof=1), fmt=marker, color=color, capsize=3, ms=5.5, zorder=3)
            ax.annotate(f"{a.mean():.3f}", (i, a.max()), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
            if variant == "base":
                effect_ax.text(i, 0, "reference", ha="center", va="bottom", color=GRAY, fontsize=8.5)
                continue
            r = comparisons[name]
            e, lo, hi = [r[k] for k in ("mean_reference_minus_alternative_mm", "bootstrap95_lower_mm", "bootstrap95_upper_mm")]
            effect_ax.errorbar(i, e, yerr=[[e-lo], [hi-e]], fmt=marker, color=color, ms=5.5, capsize=4)
            effect_ax.annotate(f"{e:+.4f}", (i, hi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.set(title=f"({'ab'[col]}) {family.upper() if family=='mlp' else 'Koopman'} internal branch",
               ylabel="Test skeleton error (mm)", ylim=e_limits, xlim=(-.35, len(variants)-.65))
        ax.tick_params(labelbottom=False)
        effect_ax.set(ylabel="Base − variant error (mm)", ylim=d_limits,
                      xticks=range(len(variants)), xticklabels=[LABELS[v] for v in variants])
        effect_ax.axhline(0, color=INK, lw=.9)
        grid(ax); grid(effect_ax)
    fig.text(.06, .08, "Top: all seed results and mean ± sample SD. Bottom: paired-seed percentile-bootstrap 95% CIs; positive values indicate improvement.", fontsize=9.5)
    fig.text(.06, .038, "MLP: memory enters the second hidden layer. Koopman: memory enters the shape readout. Both and static capacity have matched parameter counts.", fontsize=9.5)
    return finish(fig, "internal_memory_plugins", "Trainable memory branches inside MLP and Koopman, using all20 matched training seeds. MLP displays base/both/static-capacity; Koopman displays base/path/time/both/static-capacity. Upper axes show test skeleton error with all seeds and mean ± sample SD; lower axes show paired base-minus-variant improvements and 20000-resample percentile95 intervals. Both-memory and static-capacity controls have identical parameter counts within each family.",
        dict(sources=[folder / "summary.json", folder / "COMPLETE.json", run / "raw_test.csv"],
             variants=family_variants, models=summary["models"], comparisons=summary["statistics"],
             seed_rule="Exactly seeds100..119 for all8 configurations; seed differences explicitly verified",
             paired_ci_validation="Recomputed all6 source CIs using RNG20260913, 20000 paired-seed resamples; atol1e-12",
             units="Test Euclidean mean-node mm; all15 nodes pooled over2958 frames before seed aggregation",
             protocol=summary["protocol"],
             limitations=["CI reflects optimization randomness on this previously inspected fixed dataset, not fresh-data or independent-robot generalization.",
                "MLP2 and Koopman4 contrasts are separate Holm families in the source analysis; displayed percentile intervals are not simultaneous intervals.",
                "Equal parameter counts do not make the memory and static branches identical in function class or conditioning.",
                "Trainable internal memory and the fixed input-feature plugin are different experiments."]))


def internal_mlp():
    source = ANALYSIS / "internal_plugins/summary.json"
    summary = read(source)
    raw = pd.read_csv(Path(summary["source_run"]) / "raw_test.csv")
    variants = ["base", "both", "static_capacity"]
    comparisons = {r["alternative"]: r for r in summary["statistics"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.27, top=.74, wspace=.34)
    title(fig, "Trainable memory inside an MLP", "20 paired training seeds  •  100 epochs  •  2,958 test targets")
    for i, variant in enumerate(variants):
        a = values(raw, "mlp_"+variant, "mean_node_mm")
        color = GRAY if variant == "base" else BLUE if variant == "both" else ORANGE
        axes[0].scatter(i+np.linspace(-.09, .09, 20), a, color=color, s=14, alpha=.35)
        axes[0].errorbar(i, a.mean(), yerr=a.std(ddof=1), fmt="o", color=color, capsize=3)
        axes[0].annotate(f"{a.mean():.3f}", (i, a.max()), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(title="(a) Prediction accuracy", xticks=range(3),
        xticklabels=[LABELS[v] for v in variants], ylabel="Skeleton error (mm)", ylim=(1.36, 2.07))
    for i, variant in enumerate(variants[1:]):
        r = comparisons["mlp_"+variant]
        e, lo, hi = [r[k] for k in ("mean_reference_minus_alternative_mm", "bootstrap95_lower_mm", "bootstrap95_upper_mm")]
        color = BLUE if variant == "both" else ORANGE
        axes[1].errorbar(e, i, xerr=[[e-lo], [hi-e]], fmt="o", color=color, capsize=4)
        axes[1].annotate(f"{e:+.4f}", (e, i), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set(title="(b) Paired improvements", yticks=range(2), yticklabels=["+ Both", "Polynomial control"],
        xlabel="Base − variant error (mm)", xlim=(-.08, .60), ylim=(1.45, -.45))
    axes[1].axvline(0, color=INK, lw=.8)
    grid(axes[0]); axes[1].grid(axis="x", color=GRID, linewidth=.7)
    fig.text(.06, .12, "(a) All 20 seeds and mean ± SD.  (b) Paired-mean differences and bootstrap 95% CIs.", fontsize=9)
    fig.text(.06, .065, "Memory features enter the second hidden layer and are learned jointly with the prediction network.", fontsize=9)
    return finish(fig, "internal_memory_mlp", "MLP internal memory: all20 seeds, test skeleton means and sample SD, paired bootstrap95 intervals. Original MLP2 Holm family retained; Koopman extension is reported in the supplement.",
        dict(sources=[source, Path(summary["source_run"])/"raw_test.csv"], models=[r for r in summary["models"] if r["family"]=="mlp"],
             comparisons=[r for r in summary["statistics"] if r["family"]=="mlp_internal"],
             statistical_family="Original MLP2; no p values recalculated", supplementary_figure="internal_memory_plugins"))


FUNCTIONS = dict(initialization_test_stages=initialization, internal_memory_plugins=internal, internal_memory_mlp=internal_mlp,
                 prediction_summary=prediction, memory_plugin=input_plugin, learning_and_inference=learning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", choices=NAMES, default=NAMES,
                        help="Render only these figures; preserve other completed extension outputs.")
    args = parser.parse_args()
    threadpool_limits(limits=1)
    FIG.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    catalog_path, notes_path = FIG / "figure_extensions.json", FIG / "figure_extensions_notes.json"
    catalog = {r["id"]: r for r in read(catalog_path)} if catalog_path.exists() else {}
    details = read(notes_path).get("figures", {}) if notes_path.exists() else {}
    for name in args.figures:
        record, note = FUNCTIONS[name]()
        catalog[name], details[name] = record, plain(note)
        print(f"Exported {name}: SVG/PDF/PNG and report copies", flush=True)
    appendix = []
    for name in ("prediction_summary", "memory_plugin", "learning_and_inference"):
        path = ROOT / "docs/icra2027/figures/unified20" / f"{name}.png"
        assert path.is_file()
        appendix.append(dict(id=name+"_complete_comparison", path=str(path.relative_to(ROOT)),
            formats=["svg", "pdf", "png"], caption="Original complete comparison including Window MLP, retained for appendix use."))
    ordered = [catalog[n] for n in NAMES if n in catalog]
    notes = dict(schema="modeling_extensions_figures_v1", script="scripts/experiments/plot_modeling_extensions.py",
        generated_figures=list(catalog), figures=details, appendix_catalog=appendix,
        replacement_map={n: {"main": f"docs/icra2027/figures/extensions006/{n}.png", "appendix": f"docs/icra2027/figures/unified20/{n}.png"}
                         for n in ("prediction_summary", "memory_plugin", "learning_and_inference")},
        export=dict(formats=["svg", "pdf", "png"], png_dpi=250, font="DejaVu Sans", threads=1,
                    paper=str(FIG.relative_to(ROOT)), report=str(REPORT.relative_to(ROOT))),
        interpretation="Source populations and uncertainty differ by experiment; every figure specifies units, seed counts, fixed initialization status and conditioning.")
    for path, payload in ((catalog_path, ordered), (notes_path, notes)):
        path.write_text(json.dumps(plain(payload), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        shutil.copyfile(path, REPORT / path.name)
    print(json.dumps(dict(figures=len(ordered), pending=[n for n in NAMES if n not in catalog], catalog=str(catalog_path)), indent=2))


if __name__ == "__main__":
    main()
