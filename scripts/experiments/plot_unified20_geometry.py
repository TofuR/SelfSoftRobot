#!/usr/bin/env python3
"""Plot the completed unified20 geometry analysis, using only its saved data.

Exports three figures and figure_geometry_notes.json to the paper figure folder
and copies those exact files to the report's figures folder. Run with selfsr
Python and -B. All seeds, channels and fixed input-selected references remain
in the analysis. The shared style module's main entry point is never called.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/selfsr-unified20-geometry-mpl")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from threadpoolctl import threadpool_limits

from plot_unified20_results import BLUE, ORANGE, GRAY, INK

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005/geometry"
FIG = ROOT / "docs/icra2027/figures/unified20"
COPY = ROOT / "workspace/reports/modeling_unified20_20260913_005/figures"
OUT = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005"
SEEDS = np.arange(100, 120)
KINDS = ["reference", "joint_linear", "full"]
PAIRS = ["opposite", "same_direction", "same_recent_two"]
LIGHT = "#E3E7EB"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5,
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "axes.titlelocation": "left", "axes.titlepad": 12,
    "svg.fonttype": "none", "svg.hashsalt": "unified20_geometry_v1", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A8AFB8", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans", "savefig.dpi": 250,
})


def read_json(name):
    return json.loads((DATA / name).read_text())


def read_csv(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


def stat(a, axis=0):
    return dict(mean=np.mean(a, axis=axis), sd=np.std(a, axis=axis, ddof=1))


def formatted(a, digits=3):
    return f"{np.mean(a):.{digits}f} ± {np.std(a, ddof=1):.{digits}f}"


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=LIGHT, linewidth=.7)
    ax.set_axisbelow(True)


def rank1(a):
    s = np.linalg.svd(a, compute_uv=False)
    return s[0]**2 / np.sum(s**2)


def load_data():
    assert read_json("COMPLETE.json")["status"] == "complete"
    summary = read_json("summary.json")
    assert summary["completed_seeds"] == SEEDS.tolist()
    assert summary["test_frames_per_seed"] == 2958
    with np.load(DATA / "test_inputs_targets.npz", allow_pickle=False) as d:
        target = d["target_xyz_mm"][..., :2]
        ids, groups = d["frame_ids"], d["group_index"]
    profiles, terms, cosines, distances = [], [], [], []
    for seed in SEEDS:
        with np.load(DATA / f"seed_{seed}_test_geometry.npz", allow_pickle=False) as d:
            assert np.array_equal(d["frame_ids"], ids) and np.array_equal(d["group_index"], groups)
            assert d["prediction_names"].tolist() == KINDS
            errors = d["node_error_mm"]
            assert errors.shape == (3, 2958, 15)
            profiles.append(errors.mean(axis=1))
            residual = target-d["reference_xy_mm"]
            p, h = d["path_displacement_xy_mm"], d["time_displacement_xy_mm"]
            denominator = np.sum(residual**2)
            # Exact signed energy terms at the fitted readouts, all test frames.
            path = (2*np.sum(residual*p)-np.sum(p*p))/denominator*100
            temporal = (2*np.sum(residual*h)-np.sum(h*h))/denominator*100
            overlap = -2*np.sum(p*h)/denominator*100
            linear = (1-np.sum((residual-p-h)**2)/denominator)*100
            full = (1-np.sum((target-d["full_xyz_mm"][..., :2])**2)/denominator)*100
            np.testing.assert_allclose(path+temporal+overlap, linear, rtol=0, atol=1e-10)
            terms.append([path, temporal, overlap, linear, full])
            cosines.append(np.sum(p*h)/np.sqrt(np.sum(p*p)*np.sum(h*h)))
            distances.append(np.linalg.norm(d["linear_full_difference_xy_mm"], axis=-1).mean())
    profiles, terms = np.array(profiles), np.array(terms)
    for k, kind in enumerate(KINDS):
        np.testing.assert_allclose(profiles[:, k].mean(axis=1).mean(),
            summary["predictions"][kind]["mean_node_mm"]["mean"], rtol=0, atol=1e-12)
    for i, kind in ((3, "joint_linear"), (4, "full")):
        np.testing.assert_allclose(terms[:, i].mean(),
            summary["predictions"][kind]["residual_energy_reduction_pct"]["mean"], rtol=0, atol=1e-10)
    pair_rows = read_csv("matched_pair_seed_metrics.csv")
    paired = {}
    for category in PAIRS:
        selected = [r for r in pair_rows if float(r["tolerance_kpa"]) == 5 and r["category"] == category]
        values = []
        for kind in KINDS:
            rows = sorted([r for r in selected if r["prediction"] == kind], key=lambda r: int(r["seed"]))
            assert [int(r["seed"]) for r in rows] == SEEDS.tolist()
            values.append([float(r["delta_mean_node_mm"]) for r in rows])
        assert len({int(r["pairs"]) for r in selected}) == 1
        paired[category] = dict(n=int(selected[0]["pairs"]), error=np.array(values).T)
    assert [paired[c]["n"] for c in PAIRS] == [216, 58, 38]
    with np.load(DATA / "kernels.npz", allow_pickle=False) as d:
        kernels = {k: d[k] for k in ("kernel_xy_mm_per_delta_e", "alpha", "phi", "reference_actions",
                                    "node_ids", "lags", "lag_seconds", "seeds", "taus_s")}
    assert np.array_equal(kernels["seeds"], SEEDS)
    assert kernels["kernel_xy_mm_per_delta_e"].shape == (20, 6, 4, 14, 19, 2)
    assert np.array_equal(kernels["lags"], np.arange(19))
    assert np.array_equal(kernels["node_ids"], np.arange(1, 15))
    # Rank is computed per seed, before averaging, then checked against saved CSV.
    lateral = kernels["kernel_xy_mm_per_delta_e"][..., 0]
    ranks = np.empty((20, 6, 4, 2))
    for s in range(20):
        for r in range(6):
            for c in range(4):
                matrix = lateral[s, r, c]
                norms = np.linalg.norm(matrix, axis=1)
                assert np.all(norms > 1e-12)
                ranks[s, r, c] = [rank1(matrix), rank1(matrix/norms[:, None])]
    rank_rows = read_csv("kernel_rank.csv")
    for row in rank_rows:
        if row["coordinate"] != "x":
            continue
        s, r, c = int(row["seed"])-100, int(row["reference_index"]), int(row["channel"])
        np.testing.assert_allclose(ranks[s, r, c],
            [float(row["rank1_energy"]), float(row["normalized_rank1_energy"])], rtol=0, atol=1e-12)
    basis_rank = np.array([[rank1(phi), rank1(phi/np.linalg.norm(phi, axis=1)[:, None])]
                           for phi in kernels["phi"]])
    basis_context = read_json("time_basis_structure.json")
    assert basis_context["seeds"] == SEEDS.tolist()
    np.testing.assert_allclose(basis_rank.mean(0),
        [basis_context["raw_rank1_mean"], basis_context["row_normalized_rank1_mean"]],
        rtol=0, atol=1e-12)
    assert np.isfinite(ranks).all() and np.isfinite(lateral).all()
    return dict(summary=summary, profiles=profiles, terms=terms, cosines=np.array(cosines),
        distances=np.array(distances), pairs=paired, kernels=kernels, lateral=lateral,
        ranks=ranks, basis_rank=basis_rank, basis_context=basis_context)


def finish(fig, name, caption):
    """Write one owned figure and return an independently mergeable catalog row."""
    assert name in ("geometry_mechanism", "time_kernel_structure", "time_kernel_gain")
    fig.canvas.draw()
    for ext in ("svg", "pdf", "png"):
        metadata = {"Date": None} if ext == "svg" else (
            {"CreationDate": None, "ModDate": None} if ext == "pdf" else None)
        path = FIG / f"{name}.{ext}"
        fig.savefig(path, dpi=250, metadata=metadata, facecolor="white")
        shutil.copyfile(path, COPY / path.name)
        assert path.read_bytes() == (COPY / path.name).read_bytes()
    plt.close(fig)
    return dict(id=name, caption=caption, formats=["svg", "pdf", "png"],
                path=str((FIG / f"{name}.png").relative_to(ROOT)))


def geometry_figure(data):
    fig = plt.figure(figsize=(12.8, 12.1))
    fig.suptitle("Geometry of history-dependent prediction", x=.07, y=.974, ha="left", fontsize=17, weight="bold")
    fig.text(.07, .941, "20 frozen HOV fits  •  2,958 test frames per fit  •  mean ± sample SD across seeds", fontsize=11)
    gs = fig.add_gridspec(3, 1, left=.16, right=.96, top=.885, bottom=.105,
                         hspace=.72, height_ratios=[1.1, 1.05, 1.0])
    ax = fig.add_subplot(gs[0])
    profile = data["profiles"]
    x = np.arange(15)
    colors = [GRAY, ORANGE, BLUE]
    labels = ["Reference", "Joint linear", "Full"]
    # Draw dashed joint last so nearly coincident full/joint curves remain visible.
    for k in (0, 2, 1):
        mu, sd = profile[:, k].mean(0), profile[:, k].std(0, ddof=1)
        ax.fill_between(x, mu-sd, mu+sd, color=colors[k], alpha=.14, linewidth=0)
        ax.errorbar(x, mu, yerr=sd, color=colors[k], lw=1.9,
                    ls="--" if k == 1 else "-", marker=["s", "o", None][k],
                    mfc="white" if k == 1 else colors[k], markersize=4, capsize=2,
                    label=f"{labels[k]}  ({profile[:, k].mean():.3f} mm pooled)", zorder=4 if k == 1 else 3)
    ax.set(title="(a) Prediction error along the arm", xlabel="Node index (base to tip)",
           ylabel="Mean node error (mm)", xlim=(-.15, 14.3), ylim=(0, 4.95),
           xticks=[0, 2, 4, 6, 8, 10, 12, 14], xticklabels=["0 (base)", "2", "4", "6", "8", "10", "12", "14 (tip)"])
    handles, texts = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in (0, 2, 1)], [texts[i] for i in (0, 2, 1)],
              loc="upper left", frameon=False, handlelength=3)
    grid(ax)
    ax.text(1, -.31, f"Joint linear − full pooled error: {formatted(profile[:, 1].mean(1)-profile[:, 2].mean(1), 4)} mm",
            ha="right", transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(gs[1])
    t = data["terms"]
    contributions = t[:, :3].mean(0)
    cumulative = np.cumsum(t[:, :3], axis=1)
    end = cumulative.mean(0)
    start = np.r_[0., end[:-1]]
    width = .62
    for i, color in enumerate((BLUE, ORANGE, GRAY)):
        ax.bar(i, contributions[i], bottom=start[i], width=width, color=color,
               edgecolor=INK, linewidth=.65, zorder=3)
        ax.errorbar(i, end[i], yerr=cumulative[:, i].std(ddof=1), fmt="none", color=INK, capsize=3, lw=1, zorder=4)
        if i < 2:
            ax.plot([i+width/2, i+1-width/2], [end[i], end[i]], color=GRAY, lw=1, ls=":")
        ax.text(i, max(start[i], end[i])+3.5, f"{contributions[i]:+.2f} pp", ha="center", fontsize=11)
    for i, k in ((3, 3), (4, 4)):
        mu, sd = t[:, k].mean(), t[:, k].std(ddof=1)
        ax.bar(i, mu, width=width, color="white" if i == 3 else BLUE,
               edgecolor=ORANGE if i == 3 else INK, linewidth=1.6 if i == 3 else .65,
               hatch="//" if i == 3 else None, zorder=3)
        ax.errorbar(i, mu, yerr=sd, fmt="none", color=INK, capsize=3, lw=1, zorder=4)
        ax.text(i, mu+3.5, f"{mu:.2f}%", ha="center", fontsize=11)
    ax.axhline(0, color=INK, linewidth=.8)
    ax.set(title="(b) Residual-energy reduction from the fitted geometric corrections",
        ylabel="Reference residual energy removed (%)", ylim=(0, 68), xlim=(-.6, 4.6),
        xticks=range(5), xticklabels=["Path term", "Time term", "Overlap term", "Joint linear", "Full"])
    grid(ax)
    ax.text(.985, .95, f"cos(path, time) = {formatted(data['cosines'], 3)}", ha="right", va="top", transform=ax.transAxes, fontsize=10)
    ax.text(0, -.21, r"$\eta(p+h)=[2r^Tp-\|p\|^2+2r^Th-\|h\|^2-2p^Th]/\|r\|^2$",
            transform=ax.transAxes, fontsize=11, va="top")
    ax.text(1, -.22, "Whiskers: SD of energy level\npp = percentage points", transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5)

    sub = gs[2].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=.14)
    ax = fig.add_subplot(sub[0])
    table = fig.add_subplot(sub[1], sharey=ax)
    ax.set_title("(c) Shape-difference prediction for matched inputs", loc="left", pad=26)
    ax.text(0, 1.065, "Current pressure within 5 kPa; history RMS gap ≥ 20 kPa; separation ≥ 20 frames",
            transform=ax.transAxes, fontsize=10, va="bottom")
    pair_notes = []
    for row, category in enumerate(PAIRS):
        errors = data["pairs"][category]["error"]
        for k, color, marker, offset in ((1, ORANGE, "D", .105), (2, BLUE, "o", -.105)):
            effect = errors[:, 0]-errors[:, k]
            yy = row+offset
            ax.scatter(effect, yy+np.linspace(-.035, .035, 20), color=color, s=10, alpha=.3, zorder=3)
            ax.errorbar(effect.mean(), yy, xerr=effect.std(ddof=1), fmt=marker,
                        mfc="white" if k == 1 else color, mec=color, color=color,
                        capsize=3, ms=5.5, lw=1.3, zorder=4)
        for k in range(3):
            table.text([.15, .49, .83][k], row, f"{errors[:, k].mean():.4f}",
                       ha="center", va="center", color=colors[k], fontsize=11)
        pair_notes.append(dict(category=category, pairs=data["pairs"][category]["n"],
            error_mm={kind: stat(errors[:, k]) for k, kind in enumerate(KINDS)},
            reference_minus_joint_linear_mm=stat(errors[:, 0]-errors[:, 1]),
            reference_minus_full_mm=stat(errors[:, 0]-errors[:, 2])))
    ax.axvline(0, color=INK, linewidth=1)
    ax.set(xlim=(-.07, .545), ylim=(2.55, -.55), yticks=range(3),
        yticklabels=["Opposite direction\n216 pairs", "Same direction\n58 pairs", "Recent two inputs close\n38 pairs"],
        xticks=[-.05, 0, .1, .2, .3, .4, .5],
        xlabel="Reference − corrected error (mm); positive = improvement")
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.xaxis.label.set_size(10)
    grid(ax, "x")
    table.set_xlim(0, 1)
    table.axis("off")
    table.text(.49, 1.13, "Mean shape-difference error (mm)", transform=table.transAxes, ha="center", fontsize=10)
    for k, label in enumerate(("Reference", "Joint linear", "Full")):
        table.text([.15, .49, .83][k], 1.015, label, transform=table.transAxes,
                   ha="center", color=colors[k], fontsize=10)
    for row in (.5, 1.5):
        table.axhline(row, color=LIGHT, linewidth=.8)
    ax.legend(handles=[Line2D([], [], marker="D", mfc="white", mec=ORANGE, color=ORANGE, ls="none", label="Joint linear"),
                       Line2D([], [], marker="o", color=BLUE, ls="none", label="Full")],
              loc="lower right", frameon=False, fontsize=9, ncol=2, columnspacing=1)
    fig.text(.07, .04, "Energy terms use the same frozen readouts; the waterfall is an algebraic decomposition. Pair sets overlap across categories.\n"
             "Uncertainty reflects training randomness on this fixed dataset; visual residuals also include reference-fit and annotation errors.", fontsize=9.5, va="bottom")
    info = finish(fig, "geometry_mechanism", "20 frozen HOV fits, each evaluated on all 2958 test frames. (a) Per-node reference, joint-linear and full errors; bands and whiskers are seed sample SD. (b) Algebraic path/time/overlap decomposition of reference residual-energy reduction at the fitted readouts. (c) All three input-defined 5-kPa pair categories, showing paired-seed signed effects and absolute shape-difference errors; categories overlap. Seed variation is conditional on the fixed dataset.")
    return dict(**info,
        question="Where does joint memory improve the prediction, how do the fitted displacement terms reduce residual energy, and under which fixed input matches does the benefit hold?",
        panels=dict(a="15-node mean error profiles; per-seed frame means, then seed mean and sample SD; dashed/open joint overlays solid full",
                    b="Signed waterfall of path, time and overlap contributions to residual-energy reduction; joint and full totals; SD is of each cumulative energy level",
                    c="All three fixed 5-kPa categories; paired-seed reference-minus-corrected effects and SD, seed dots, exact absolute mean errors and pair counts"),
        node_profiles_mm={kind: stat(profile[:, k]) for k, kind in enumerate(KINDS)},
        energy_terms_pct={k: stat(t[:, i]) for i, k in enumerate(("path", "time", "overlap", "joint_linear", "full"))},
        energy_terms_per_seed=[dict(seed=int(s), **dict(zip(("path", "time", "overlap", "joint_linear", "full"), t[i]))) for i, s in enumerate(SEEDS)],
        path_time_cosine=stat(data["cosines"]), linear_full_mean_node_distance_mm=stat(data["distances"]),
        matched_pairs_5kpa=pair_notes,
        supported_conclusions=["The first-order joint correction retains nearly all full-model error and residual-energy benefit on all 2958 test frames.",
            "Path and time terms align with the fitted-reference residual; their nonzero overlap contributes a signed correction in the energy identity.",
            "At the fixed 5-kPa match, the opposite-direction category improves; same-direction and recent-two-close categories have slightly greater error than the reference."],
        limitations=["Reference is the fitted reference inside full HOV, not a separately retrained static model.",
            "The path/time waterfall terms are fixed-readout algebra, not retrained ablations or separately identified causal physical mechanisms.",
            "A small stacked branch cosine does not imply statistical independence or node/frame-wise orthogonality.",
            "The fixed three-sequence temporal split and visual labels restrict generalization; seed SD is not uncertainty over independent robot experiments.",
            "The displayed 5-kPa protocol was fixed by the preceding analysis; its three categories overlap, with disjoint pairs only within each sequence/category/tolerance."])


def time_figure(data):
    fig = plt.figure(figsize=(12.8, 9.5))
    fig.suptitle("Time-kernel structure across drive channels", x=.07, y=.974, ha="left", fontsize=17, weight="bold")
    pressure = data["kernels"]["reference_actions"][0]*150
    fig.text(.07, .937, "Lateral displacement per drive increment  •  20 frozen fits  •  14 non-base nodes  •  19 lags", fontsize=11)
    fig.text(.07, .901, "(a) Mean kernels at the training-mean input (ref0)", fontsize=12.5, weight="medium")
    fig.text(.07, .873, "Reference pressures (kPa): ["+", ".join(f"{v:.2f}" for v in pressure)+"]; common signed scale", fontsize=10)
    gs = fig.add_gridspec(2, 5, left=.075, right=.94, top=.826, bottom=.205,
        width_ratios=[1, 1, 1, 1, .065], height_ratios=[1.2, 1], hspace=.85, wspace=.19)
    average = data["lateral"][:, 0].mean(axis=0)
    limit = float(np.ceil(np.max(abs(average))/5)*5)
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", [BLUE, "#FFFFFF", ORANGE], N=257)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    axes, rank_axes = [], []
    for c in range(4):
        ax = fig.add_subplot(gs[0, c], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        axes.append(ax)
        im = ax.imshow(average[c], cmap=cmap, norm=norm, aspect="auto", origin="lower",
                       interpolation="nearest", extent=(-.1, 3.7, .5, 14.5))
        ax.set(title=f"Channel {c}", xlabel="Increment lag (s)",
               xticks=[0, 1.2, 2.4, 3.6], yticks=[1, 4, 7, 10, 14])
        ax.axhline(7.5, color=INK, lw=.55, ls=(0, (4, 3)), alpha=.6)
        if c == 0:
            ax.set_ylabel("Node index (base to tip)")
        else:
            ax.tick_params(labelleft=False)
    color_ax = fig.add_subplot(gs[0, 4])
    bar = fig.colorbar(im, cax=color_ax, ticks=[-limit, -limit/2, 0, limit/2, limit])
    bar.set_label(r"$K^x$ (mm / unit $\Delta e_c$)", fontsize=10)
    for c in range(4):
        ax = fig.add_subplot(gs[1, c], sharey=rank_axes[0] if rank_axes else None)
        rank_axes.append(ax)
        for k, color, marker, offset in ((0, BLUE, "o", -.12), (1, ORANGE, "D", .12)):
            a = data["ranks"][:, :, c, k]*100
            for r in range(6):
                ax.scatter(r+offset+np.linspace(-.035, .035, 20), a[:, r], s=7,
                           color=color, alpha=.20, linewidths=0, zorder=2)
            ax.errorbar(np.arange(6)+offset, a.mean(0), yerr=a.std(0, ddof=1),
                        fmt=marker, color=color, mfc="white" if k else color,
                        ms=4.5, capsize=2, lw=1, zorder=3)
        ax.set(xlim=(-.48, 5.48), ylim=(50, 102), yticks=[50, 60, 70, 80, 90, 100],
               xticks=range(6), xticklabels=["Mean", "Q10", "Q30", "Q50", "Q70", "Q90"],
               xlabel="Training reference input")
        ax.tick_params(axis="x", labelsize=8.7, rotation=40)
        ax.get_xticklabels()[0].set_weight("bold")
        if c == 0:
            ax.set_ylabel("Rank-one energy (%)")
        else:
            ax.tick_params(labelleft=False)
        grid(ax)
    top = rank_axes[0].get_position().y1
    fig.text(.07, top+.078, "(b) Per-seed kernel rank at all six fixed training references", fontsize=12.5)
    fig.text(.07, top+.045, "Uncentered SVD of each 14-node × 19-lag matrix; common focused vertical scale", fontsize=10)
    fig.legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="none", label="Raw kernel"),
                        Line2D([], [], marker="D", mfc="white", color=ORANGE, ls="none", label="Each node curve normalized to unit L2")],
               loc="lower left", bbox_to_anchor=(.066, top+.005), frameon=False, ncol=2, fontsize=10)
    basis = data["basis_rank"].mean(0)*100
    fig.text(.07, .103,
        r"$K^x_{c,n}[\ell]=-\sum_{g,k}J^x_{n,g}(u_{\rm ref})\,W_{c,k,g}\,\alpha_k^{\ell+1}$"
        f"     Shared six-exponential basis: rank-one energy {basis[0]:.2f}% raw / {basis[1]:.2f}% mode-normalized.", fontsize=10.5)
    fig.text(.07, .052, "High kernel rank-one energy also reflects the shared time basis, fitted readout, and reference Jacobian.\n"
             "It does not by itself identify a new physical relaxation law. Points: all seeds; whiskers: mean ± sample SD.", fontsize=10)
    info = finish(fig, "time_kernel_structure", "All four drive channels and 14 non-base nodes. (a) Mean signed lateral time-memory kernels at the training-mean reference, with a common symmetric blue-white-orange scale and lags 0–3.6 s. (b) Per-seed raw and node-normalized rank-one energies at all six input-selected training references. The common six-exponential basis is itself close to rank one; kernel rank does not uniquely identify a physical relaxation law.")
    ranks = data["ranks"]
    per_cell = [dict(reference=r, channel=c, raw=stat(ranks[:, r, c, 0]*100),
                     node_normalized=stat(ranks[:, r, c, 1]*100)) for r in range(6) for c in range(4)]
    return dict(**info,
        question="How do signed lateral time kernels vary by drive channel, node, lag and fixed training reference, and how much of waveform similarity remains after node normalization?",
        panels=dict(a="Four channel heatmaps, arithmetic seed-mean lateral kernels at reference index 0, all 14 non-base nodes and 19 lags; one common symmetric blue-white-orange normalization",
                    b="Per-seed rank-one energies of raw and unit-node-L2 kernels, all six fixed training references in each channel; dots are all seeds and whiskers sample SD"),
        heatmap=dict(reference_index=0, reference_pressure_kpa=pressure, channels=[0, 1, 2, 3], coordinate="x",
            nodes=list(range(1, 15)), lags=list(range(19)), lag_seconds=data["kernels"]["lag_seconds"],
            mean_kernel_xy_units="mm per unit normalized drive increment Delta e, current reference held fixed",
            common_vmin=-limit, common_vmax=limit, vcenter=0,
            color_limits_rule="ceil(max absolute value over all four ref0 seed-mean heatmaps / 5) * 5; no clipping or channel-specific normalization",
            colormap_stops=[BLUE, "#FFFFFF", ORANGE], grid_rule="lag centers at 0:0.2:3.6 s; node centers 1:14; dashed line is the section boundary after node 7"),
        ranks_pct_by_reference_channel=per_cell,
        ranks_pct_per_seed=ranks*100,
        rank_axes="seed,reference,channel,{raw,node_normalized}", rank_plot_ylim=[50, 102],
        exponential_basis_rank_pct=dict(raw=stat(data["basis_rank"][:, 0]*100),
                                       mode_normalized=stat(data["basis_rank"][:, 1]*100)),
        supporting_basis_file="geometry/time_basis_structure.json",
        supporting_basis_context=data["basis_context"],
        supported_conclusions=["The four channels have distinct signed node/lag readouts under a common scale.",
            "Raw lateral rank-one energy is high, while per-node normalization exposes substantial channel/reference dependence in relative waveform similarity.",
            "The stored shared exponential basis already has about 97% raw rank-one energy on these 19 lags; architecture is relevant to interpreting kernel rank.",
            f"The mean raw lateral output rank-one energy over 20 seeds x 6 references x 4 channels is {100*ranks[..., 0].mean():.6f}%; it is a joint result of structure, learned readout and geometry."],
        limitations=["Heatmaps show the seed mean; kernel rank is computed for each seed, not from the mean heatmap.",
            "These are local time-memory kernels at fixed reference input, not measured physical impulse responses or the full pressure-to-shape derivative.",
            "Units are per normalized drive increment; conversion to pressure response requires the drive derivative and reference-map contribution.",
            "The six reference inputs are fixed from training inputs, and all four channels and 14 non-base nodes are included; coverage is conditional on these references.",
            "The 19-lag horizon is 0–3.6 seconds under nominal 0.2-second updates; no longer-time conclusion is inferred.",
            "Raw rank weights large-amplitude node curves more; node normalization instead emphasizes waveform shape, including small-gain nodes.",
            "High rank-one energy can arise from the shared exponential basis combined with W and J; it does not establish a newly learned or uniquely identified physical law.",
            "The basis rank and output rank describe different matrices. They are displayed descriptively and are never subtracted as a correction, excess-rank estimate or independent-discovery statistic."])


def gain_figure(data):
    """Factor every seed/reference/channel before summarizing the ref0 curves."""
    lateral = data["lateral"]
    gains = np.empty((20, 6, 4, 14))
    waves = np.empty((20, 6, 4, 19))
    sign_indices = np.empty((20, 6, 4), dtype=int)
    reconstruction = np.empty((20, 6, 4))
    for s in range(20):
        for r in range(6):
            for c in range(4):
                matrix = lateral[s, r, c]
                u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
                idx = int(np.argmax(abs(vt[0])))
                sign = 1 if vt[0, idx] >= 0 else -1
                gains[s, r, c] = sign*sigma[0]*u[:, 0]
                waves[s, r, c] = sign*vt[0]
                sign_indices[s, r, c] = idx
                estimate = np.outer(gains[s, r, c], waves[s, r, c])
                reconstruction[s, r, c] = 1-np.sum((matrix-estimate)**2)/np.sum(matrix**2)
    np.testing.assert_allclose(np.linalg.norm(waves, axis=-1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(reconstruction, data["ranks"][..., 0], rtol=0, atol=1e-12)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9))
    fig.subplots_adjust(left=.08, right=.97, top=.71, bottom=.25, wspace=.27)
    fig.suptitle("Node gains and dominant time waveforms", x=.07, y=.96, ha="left", fontsize=17, weight="bold")
    fig.text(.07, .899, "Training-mean input (ref0)  •  all four channels  •  mean ± sample SD of 20 per-seed SVD factors", fontsize=11)
    palette = [BLUE, ORANGE, GRAY, INK]
    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", "-.", ":"]
    for c, color in enumerate(palette):
        g, f = gains[:, 0, c], waves[:, 0, c]
        for ax, x, a in ((axes[0], np.arange(1, 15), g),
                         (axes[1], data["kernels"]["lag_seconds"], f)):
            mu, sd = a.mean(0), a.std(0, ddof=1)
            ax.fill_between(x, mu-sd, mu+sd, color=color, alpha=.13, linewidth=0)
            ax.plot(x, mu, color=color, ls=styles[c], lw=1.8, marker=markers[c],
                    ms=4.2, markevery=2, mfc="white" if c in (1, 3) else color,
                    label=f"Channel {c}")
    axes[0].set(title="(a) Signed gain at each node", xlabel="Node index (base to tip)",
                ylabel=r"Dominant gain $g_{c,n}$ (mm / unit $\Delta e_c$)",
                xlim=(.8, 14.25), xticks=[1, 4, 7, 10, 14])
    axes[1].set(title="(b) Dominant waveform shared across nodes", xlabel="Increment lag (s)",
                ylabel=r"Unit-L2 waveform $f_c[\ell]$", xlim=(-.04, 3.64),
                xticks=[0, .6, 1.2, 1.8, 2.4, 3, 3.6])
    axes[0].axvline(7.5, color=GRAY, lw=.8, ls=":")
    for ax in axes:
        ax.axhline(0, color=INK, lw=.8)
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.063, .86),
               frameon=False, ncol=4, handlelength=3, columnspacing=2.5)
    fig.text(.07, .139,
        r"Per seed and channel: $K^x_{c,n}[\ell]\approx g_{c,n}\,f_c[\ell]$, $\|f_c\|_2=1$ (19 samples). "
        "Sign: largest-absolute waveform sample is positive (earliest lag breaks ties).", fontsize=10)
    fig.text(.07, .066, "These are dominant-component gains under one fixed reference, not exact full-kernel amplitudes or universal arm profiles.\n"
             "The shared waveform is within each channel; channel/reference dependence and reconstruction coverage are reported in the notes.", fontsize=10)
    info = finish(fig, "time_kernel_gain", "At the fixed training-mean reference (ref0), dominant SVD node gains and corresponding time waveforms for all four channels, using all 14 non-base nodes and 19 lags. SVD is performed separately for each seed/channel; f has discrete L2 norm one, and its largest-absolute sample is made positive with earliest-lag tie breaking. Curves/bands are mean ± sample SD across 20 fits. These are approximate dominant-component gains conditional on the reference and factor normalization, not unique physical or universal spatial laws.")
    coverage = [dict(reference=r, channel=c, rank1_pct=stat(reconstruction[:, r, c]*100),
        mean_gain_by_node=stat(gains[:, r, c]), mean_waveform=stat(waves[:, r, c]),
        mean_peak_abs_gain_node=int(np.argmax(abs(gains[:, r, c].mean(0)))+1),
        waveform_sign_pivot_lags=sign_indices[:, r, c]) for r in range(6) for c in range(4)]
    return dict(**info,
        question="What spatial gain profile and common-in-node temporal shape make up the dominant component of each signed lateral time kernel?",
        selection=dict(reference=0, channels=[0, 1, 2, 3], nodes=list(range(1, 15)), lags=list(range(19)), seeds=SEEDS),
        panels=dict(a="Dominant signed node gains g=sigma1*u1 at the training-mean reference, four channels; SD across per-seed factorizations",
                    b="Corresponding temporal factors f=v1, shared across the nodes of each channel; no waveform is assumed common to all channels"),
        factorization=dict(formula="K approximately outer(g,f), g=sigma1*u1, f=v1; per seed/reference/channel",
            normalization="sum over the 19 discrete lag samples of f^2 = 1, with no dt weighting",
            sign="Largest absolute waveform sample positive; first/earliest lag breaks exact ties; same sign applied to g",
            units=dict(g="mm per unit Delta e because discrete waveform has unit L2 norm", f="dimensionless"),
            averaging="Factor each seed first, apply deterministic sign, then mean and sample SD; product of mean factors need not equal the mean kernel"),
        plotted_per_seed=dict(gain_ref0=gains[:, 0], waveform_ref0=waves[:, 0],
            axes="seed,channel,node or lag", rank1_pct_ref0=reconstruction[:, 0]*100),
        cross_reference_coverage=coverage,
        supported_conclusions=["At ref0, channel 0 and channel 1 mean dominant gains change sign along the arm, whereas channels 2 and 3 retain opposite signed mean gains across the 14 nodes.",
            "Distal gain magnitudes are larger at ref0, with distinct spatial profiles and dominant time waveforms across channels.",
            "The dominant-component approximation is assessed with per-seed reconstruction energy; the full signed heatmaps retain effects excluded by the first component."],
        limitations=["The spatial profiles shown are conditional on the training-mean reference. Five further fixed training references are quantified in the notes and the rank figure; ref0 gain ordering or sign patterns are not asserted to be universal.",
            "Gains depend on the unit-L2 waveform and deterministic sign convention. They are not peak full-kernel values, impulse integrals, or calibrated pressure sensitivities.",
            "Each channel has its own waveform; apparent agreement within one channel does not establish equality of waveforms across channels.",
            "A rank-one approximation emphasizes high-energy nodes. Small-gain node waveform differences can remain substantial even with high global reconstruction energy.",
            "The shared exponential basis, fitted readout and reference Jacobian jointly determine the factors. SVD identifies a descriptive component, not a new or uniquely identified physical law.",
            "Seed SD is conditional on the fixed dataset, references, local linearization, horizon and normalization."],
        verification=dict(unit_L2_all_480_waveforms="passed atol=1e-12",
                          reconstruction_energy_all_480_cells="equals saved raw rank1, atol=1e-12"))


def main():
    threadpool_limits(limits=1)
    FIG.mkdir(parents=True, exist_ok=True)
    COPY.mkdir(parents=True, exist_ok=True)
    data = load_data()
    geometry = geometry_figure(data)
    kernel = time_figure(data)
    gain = gain_figure(data)
    notes = dict(schema="unified20_geometry_figures_v1", source=str(DATA.relative_to(ROOT)),
        script=str(Path(__file__).resolve().relative_to(ROOT)), seeds=SEEDS, test_frames_per_seed=2958,
        aggregation="Pool all test frames within each seed first; report mean and sample SD ddof=1 across 20 frozen seeds. All 15 nodes count in error metrics; kernels use nodes 1..14.",
        selection_protocol=dict(reference_selection=read_json("reference_selection.json"),
            pair_selection=read_json("protocol.json")["pair_protocol"],
            displayed_pair_tolerance_kpa=5, displayed_pair_categories=PAIRS,
            pair_input_conditions=[r for r in data["summary"]["matched_pairs"] if r["tolerance_kpa"] == 5],
            selection_order="Reference inputs, categories, 5-kPa display tolerance and all-channel coverage are fixed by the completed analysis/request before plotting."),
        style=dict(font="DejaVu Sans", language="English", palette=[BLUE, ORANGE, GRAY, INK],
            policy="Blue and orange plus neutrals; dash/open markers distinguish overlapping corrections; centered diverging colors encode kernel sign."),
        figures=dict(geometry_mechanism=geometry, time_kernel_structure=kernel, time_kernel_gain=gain),
        checks=dict(seed_and_frame_identity="passed", energy_decomposition_identity="passed for all 20 seeds, atol=1e-10 percentage points",
            agreement_with_geometry_summary="passed", all_480_lateral_rank_cells_against_saved_csv="passed, atol=1e-12",
            provided_time_basis_structure_vs_phi="passed, atol=1e-12; descriptive context only",
            exact_report_copies="passed for all nine image files", all_pair_categories="216/58/38 pairs retained at 5 kPa",
            finite_arrays="passed"),
        outputs=dict(paper_directory=str(FIG.relative_to(ROOT)), report_directory=str(COPY.relative_to(ROOT)),
            figures=[f"{name}.{ext}" for name in ("geometry_mechanism", "time_kernel_structure", "time_kernel_gain") for ext in ("svg", "pdf", "png")],
            notes="figure_geometry_notes.json"),
        reproduce="/Data5/ddf/environments/conda_envs/selfsr/bin/python -B scripts/experiments/plot_unified20_geometry.py")
    path = FIG / "figure_geometry_notes.json"
    path.write_text(json.dumps(plain(notes), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    shutil.copyfile(path, COPY / path.name)
    catalog = [{key: record[key] for key in ("id", "caption", "formats", "path")}
               for record in (geometry, kernel, gain)]
    (OUT / "figure_geometry.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(figures=3, formats=["svg", "pdf", "png"], source=str(DATA),
        paper=str(FIG), report=str(COPY), notes=str(path), checks=notes["checks"]), indent=2))


if __name__ == "__main__":
    main()
