#!/usr/bin/env python3
"""Audit saved HOV/MLP memory ablations and export the draft2 section 3.5 evidence.

Run from any directory with the selfsr Python environment. All generated artifacts
are confined to OUTPUT. Statistical observations are paired training seeds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "workspace/runs/analysis"
MLP = ANALYSIS / "modeling_internal_mlp_single_memory_20260913_007"
HOV = ANALYSIS / "modeling_unified20_20260913_005"
OUTPUT = ANALYSIS / "draft2_experiment_extensions_20260914_009/reuse"
SEEDS = np.arange(100, 120)
METRICS = {"mean_node_mm": "骨架", "endpoint_mm": "末端"}
MODELS = {
    "HOV": {"none": "hov_no_memory", "path": "hov_no_maxwell",
            "time": "hov_no_play", "both": "hov"},
    "MLP": {"none": "mlp_base", "path": "mlp_path",
            "time": "mlp_time", "both": "mlp_both"},
}
VARIANT_ZH = {"none": "无记忆", "path": "仅路径", "time": "仅时间", "both": "双记忆"}
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_REPLICATES = 20000
BOOT = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 20, size=(BOOTSTRAP_REPLICATES, 20))


def read(path):
    return json.loads(Path(path).read_text())


def table(path):
    return pd.read_csv(path, float_precision="round_trip")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def close(actual, expected, context, atol=1e-12):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def require(condition, context):
    if not condition:
        raise ValueError(context)


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        pvalues[order] * np.arange(len(pvalues), 0, -1)))
    return adjusted


def paired_statistics(reference, alternative):
    delta = reference - alternative
    active = delta[delta != 0]
    if len(active):
        # Integer doubled ranks allow exact sign enumeration with average ties.
        ranks = np.rint(2 * stats.rankdata(np.abs(active))).astype(int)
        counts = np.zeros(ranks.sum() + 1, dtype=np.int64)
        counts[0] = 1
        for rank in ranks:
            previous = counts.copy()
            counts[rank:] += previous[:-rank]
        observed = ranks[active > 0].sum()
        pvalue = min(1.0, 2 * min(counts[:observed + 1].sum(), counts[observed:].sum()) / 2 ** len(active))
        if len(np.unique(np.abs(active))) == len(active):
            close(pvalue, stats.wilcoxon(active, method="exact", alternative="two-sided").pvalue,
                  "Independent SciPy signed-rank cross-check")
        sign_pvalue = stats.binomtest(int((active > 0).sum()), len(active)).pvalue
    else:
        pvalue = sign_pvalue = 1.0
    ci = np.quantile(delta[BOOT].mean(axis=1), [.025, .975])
    return {
        "n_pairs": len(delta),
        "mean_reference_minus_alternative_mm": float(delta.mean()),
        "bootstrap95_lower_mm": float(ci[0]), "bootstrap95_upper_mm": float(ci[1]),
        "positive_pairs": int((delta > 0).sum()), "negative_pairs": int((delta < 0).sum()),
        "zero_pairs": int((delta == 0).sum()), "wilcoxon_exact_p": float(pvalue),
        "sign_test_exact_p": float(sign_pvalue),
    }


def load_and_audit():
    mlp_summary, hov_summary = read(MLP / "summary.json"), read(HOV / "summary.json")
    hov_run = ROOT / hov_summary["run"]
    frames = {"MLP": table(MLP / "raw_test.csv"), "HOV": table(hov_run / "raw_test.csv")}
    summaries = {"MLP": mlp_summary, "HOV": hov_summary}
    summary_tables = {"MLP": table(MLP / "model_summary.csv").set_index("model"),
                      "HOV": table(HOV / "model_summary.csv").set_index("model")}
    with np.load(hov_run / "test_targets.npz", allow_pickle=False) as saved:
        identity = {k: saved[k].copy() for k in ["target_mm", "groups", "frame_ids"]}
    require(identity["target_mm"].shape == (2958, 15, 3), "Unexpected target shape")
    require(len(set(zip(identity["groups"], identity["frame_ids"]))) == 2958,
            "Duplicate test identities")
    audit_rows, node_rows, sequences, values, parameters = [], [], [], {}, {}
    for backbone in MODELS:
        model_ids = list(MODELS[backbone].values())
        if backbone == "MLP":
            model_ids.append("mlp_static_capacity")
        selected = frames[backbone][frames[backbone].model.isin(model_ids)].copy()
        require(not selected.duplicated(["model", "seed"]).any(), "Duplicate model/seed")
        require(len(selected) == 20 * len(model_ids), "Unexpected model/seed coverage")
        summary_lookup = {r["model"]: r for r in summaries[backbone]["models"]}
        for model in model_ids:
            rows = selected[selected.model == model].sort_values("seed")
            require(np.array_equal(rows.seed.to_numpy(), SEEDS), f"Seed mismatch: {model}")
            require((rows.test_frames == 2958).all(), f"Frame count mismatch: {model}")
            require(np.isfinite(rows[list(METRICS)].to_numpy()).all(), f"Nonfinite metric: {model}")
            values[model] = {metric: rows[metric].to_numpy() for metric in METRICS}
            lookup = summary_tables[backbone].loc[model]
            parameters[model] = int(lookup["parameters" if backbone == "MLP" else "active_fitted_parameters"])
            require(int(lookup["n"]) == 20 and summary_lookup[model]["n"] == 20,
                    f"Summary sample size: {model}")
            for metric in METRICS:
                for statistic, value in [("mean", rows[metric].mean()), ("sd", rows[metric].std(ddof=1))]:
                    close(value, lookup[f"{metric}_{statistic}"], f"CSV summary: {model}/{metric}")
                    close(value, summary_lookup[model][metric][statistic], f"JSON summary: {model}/{metric}")
            for row in rows.to_dict("records"):
                seed = int(row["seed"])
                metric_path = (Path(row["metrics_source"]) if backbone == "MLP" else
                               hov_run / f"evaluation/{model}/seed_{seed}/metrics.json")
                saved_metric = read(metric_path)
                require(saved_metric["model"] == model and saved_metric["seed"] == seed,
                        f"Metric identity mismatch: {metric_path}")
                prediction_path = metric_path.with_name("predictions.npz")
                with np.load(prediction_path, allow_pickle=False) as saved:
                    for key in ["groups", "frame_ids"]:
                        require(np.array_equal(saved[key], identity[key]), f"Test order mismatch: {prediction_path}")
                    if "target_mm" in saved:
                        require(np.array_equal(saved["target_mm"], identity["target_mm"]),
                                f"Test targets mismatch: {prediction_path}")
                    prediction = saved["prediction_mm"].astype(np.float64)
                require(prediction.shape == identity["target_mm"].shape and np.isfinite(prediction).all(),
                        f"Invalid predictions: {prediction_path}")
                error = np.linalg.norm(prediction - identity["target_mm"].astype(np.float64), axis=-1)
                recomputed = {"mean_node_mm": error.mean(), "endpoint_mm": error[:, -1].mean()}
                for metric, actual in recomputed.items():
                    close(actual, row[metric], f"Predictions vs raw CSV: {model}/{seed}/{metric}")
                    close(actual, saved_metric[metric], f"Predictions vs metric JSON: {model}/{seed}/{metric}")
                audit_rows.append({"backbone": backbone, "model": model, "seed": seed,
                                   "test_frames": 2958, "metrics_source": relative(metric_path),
                                   "prediction_source": relative(prediction_path), "status": "pass"})
                if backbone == "MLP" and model != "mlp_static_capacity":
                    for node, node_error in enumerate(error.mean(axis=0)):
                        node_rows.append({"model": model, "seed": seed, "node_index": node,
                                          "normalized_node_position": node / 14, "mean_error_mm": float(node_error)})
                    for group in np.unique(identity["groups"]):
                        group_error = error[identity["groups"] == group]
                        sequences.append({"model": model, "seed": seed, "group": int(group),
                                          "test_frames": len(group_error),
                                          "mean_node_mm": float(group_error.mean()),
                                          "endpoint_mm": float(group_error[:, -1].mean())})
    audit = {"status": "pass", "seeds": SEEDS.tolist(), "paired_units": 20,
             "audited_model_seed_rows": len(audit_rows), "primary_model_seed_rows": 160,
             "common_test_targets": 2958, "target_nodes": 15,
             "group_frame_counts": {str(g): int((identity["groups"] == g).sum()) for g in np.unique(identity["groups"])},
             "checks": ["unique model/seed keys", "complete seeds 100..119",
                        "CSV and JSON mean/sample-SD agreement", "common groups and frame order",
                        "MLP targets exactly equal HOV common target array",
                        "both metrics recomputed from all saved prediction arrays"]}
    return values, parameters, audit, audit_rows, node_rows, sequences, mlp_summary, hov_summary, hov_run


def verify_existing(values, mlp_summary, hov_summary, hov_run):
    mlp_saved = read(MLP / "paired_statistics.json")["contrasts"]
    require(mlp_saved == mlp_summary["statistics"], "MLP statistical JSON copies disagree")
    hov_saved = read(hov_run / "paired_statistics.json")
    require(hov_saved == hov_summary["statistics"], "HOV statistical JSON copies disagree")
    mlp_csv = table(MLP / "paired_statistics.csv")
    records, lookup = [], {}
    groups = [("MLP", mlp_saved), ("HOV", [r for r in hov_saved["contrasts"] if r["family"] == "ablation"])]
    for backbone, saved_rows in groups:
        for row in saved_rows:
            ref, alt = row["reference"], row["alternative"]
            reference, alternative = values[ref]["mean_node_mm"], values[alt]["mean_node_mm"]
            recomputed = paired_statistics(reference, alternative)
            for key, value in recomputed.items():
                if key in row:
                    close(value, row[key], f"Existing {backbone} statistics {ref}/{alt}/{key}")
            if "seeds" in row:
                require(row["seeds"] == SEEDS.tolist(), "Stored statistical seed order")
                close(reference - alternative, row["seed_differences_mm"], "Stored seed differences")
            if backbone == "MLP":
                csv_row = mlp_csv[(mlp_csv.reference == ref) & (mlp_csv.alternative == alt)]
                require(len(csv_row) == 1, "MLP statistical CSV row missing/duplicated")
                for key in [*recomputed, "wilcoxon_holm_p", "sign_test_holm_p"]:
                    close(csv_row.iloc[0][key], row[key], f"MLP statistical CSV {key}")
            records.append({"backbone": backbone, "reference": ref, "alternative": alt,
                            "family": row["family"], "status": "pass",
                            "max_ci_abs_difference_mm": max(abs(recomputed[k] - row[k]) for k in
                                                            ["bootstrap95_lower_mm", "bootstrap95_upper_mm"])})
            lookup[(backbone, ref, alt)] = row
        for family in {r["family"] for r in saved_rows}:
            family_rows = [r for r in saved_rows if r["family"] == family]
            for field in ["wilcoxon", "sign_test"]:
                adjusted = holm([r[f"{field}_exact_p"] for r in family_rows])
                close(adjusted, [r[f"{field}_holm_p"] for r in family_rows], f"Existing Holm family {family}")
    return lookup, records


def comparisons(values, parameters, existing):
    comparison_rows, seed_rows, contrasts = [], [], []
    for backbone, model_map in MODELS.items():
        for metric in METRICS:
            baseline = values[model_map["none"]][metric]
            for variant, model in model_map.items():
                errors = values[model][metric]
                seed_pct = 100 * (baseline - errors) / baseline
                bootstrap_pct = 100 * (baseline[BOOT].mean(axis=1) - errors[BOOT].mean(axis=1)) / baseline[BOOT].mean(axis=1)
                ci = np.quantile(bootstrap_pct, [.025, .975])
                comparison_rows.append({"backbone": backbone, "variant": variant, "model": model,
                    "baseline_model": model_map["none"], "metric": metric, "n_seeds": 20,
                    "test_frames_per_seed": 2958, "active_fitted_parameters": parameters[model],
                    "mean_error_mm": float(errors.mean()), "sd_error_mm": float(errors.std(ddof=1)),
                    "baseline_mean_error_mm": float(baseline.mean()),
                    "improvement_pct": float(100 * (baseline.mean() - errors.mean()) / baseline.mean()),
                    "improvement_pct_ci95_lower": float(ci[0]), "improvement_pct_ci95_upper": float(ci[1]),
                    "mean_seed_improvement_pct": float(seed_pct.mean()),
                    "improved_seed_count": int((errors < baseline).sum()),
                    "equal_seed_count": int((errors == baseline).sum())})
                for seed, error, base, pct in zip(SEEDS, errors, baseline, seed_pct):
                    seed_rows.append({"backbone": backbone, "variant": variant, "model": model,
                                      "metric": metric, "seed": int(seed), "error_mm": float(error),
                                      "baseline_error_mm": float(base), "improvement_pct": float(pct)})
            for reference_variant, alternative_variant in [("none", "path"), ("none", "time"),
                    ("none", "both"), ("path", "both"), ("time", "both")]:
                ref, alt = model_map[reference_variant], model_map[alternative_variant]
                result = {"backbone": backbone, "metric": metric, "reference_variant": reference_variant,
                          "alternative_variant": alternative_variant, "reference": ref, "alternative": alt,
                          **paired_statistics(values[ref][metric], values[alt][metric])}
                original = existing.get((backbone, ref, alt)) or existing.get((backbone, alt, ref))
                if metric == "mean_node_mm" and original:
                    # Existing HOV contrasts use full-minus-ablation; publish the reverse sign.
                    reverse = original["reference"] != ref
                    result.update({"family": original["family"], "statistics_origin": "reused_verified",
                        "family_size": {"ablation": 3, "mlp_base_extensions": 4,
                                        "mlp_both_single_mechanism": 2}[original["family"]],
                        "wilcoxon_exact_p": original["wilcoxon_exact_p"],
                        "wilcoxon_holm_p": original["wilcoxon_holm_p"],
                        "sign_test_exact_p": original["sign_test_exact_p"],
                        "sign_test_holm_p": original["sign_test_holm_p"],
                        "mean_reference_minus_alternative_mm": (-1 if reverse else 1) * original["mean_reference_minus_alternative_mm"],
                        "bootstrap95_lower_mm": -original["bootstrap95_upper_mm"] if reverse else original["bootstrap95_lower_mm"],
                        "bootstrap95_upper_mm": -original["bootstrap95_lower_mm"] if reverse else original["bootstrap95_upper_mm"],
                        "source": relative(HOV / "summary.json" if backbone == "HOV" else MLP / "paired_statistics.csv")})
                else:
                    result.update({"family": "supplemental_endpoint_10" if metric == "endpoint_mm" else "supplemental_hov_single_vs_none_2",
                                   "statistics_origin": "supplemental", "source": "paired_seed_metrics.csv"})
                contrasts.append(result)
    for family in {r["family"] for r in contrasts if r["statistics_origin"] == "supplemental"}:
        rows = [r for r in contrasts if r["family"] == family]
        for field in ["wilcoxon", "sign_test"]:
            for row, adjusted in zip(rows, holm([r[f"{field}_exact_p"] for r in rows])):
                row[f"{field}_holm_p"] = float(adjusted)
                row["family_size"] = len(rows)
    return comparison_rows, seed_rows, contrasts


def along_arm(node_rows, values):
    frame = pd.DataFrame(node_rows)
    result = []
    baseline = frame[frame.model == "mlp_base"].pivot(index="seed", columns="node_index", values="mean_error_mm").loc[SEEDS]
    for variant, model in MODELS["MLP"].items():
        matrix = frame[frame.model == model].pivot(index="seed", columns="node_index", values="mean_error_mm").loc[SEEDS]
        close(matrix.mean(axis=1).to_numpy(), values[model]["mean_node_mm"], "Along-arm skeleton reconciliation")
        close(matrix[14].to_numpy(), values[model]["endpoint_mm"], "Along-arm endpoint reconciliation")
        delta = baseline.to_numpy() - matrix.to_numpy()
        ci = np.quantile(delta[BOOT].mean(axis=1), [.025, .975], axis=0)
        for node in matrix.columns:
            base = baseline[node].mean()
            result.append({"variant": variant, "model": model, "node_index": int(node),
                "normalized_node_position": node / 14, "mean_error_mm": float(matrix[node].mean()),
                "sd_error_mm": float(matrix[node].std(ddof=1)),
                "baseline_mean_error_mm": float(base),
                "improvement_pct": float(100 * (base - matrix[node].mean()) / base) if base > 0 else None,
                "mean_baseline_minus_variant_mm": float(delta[:, node].mean()),
                "delta_ci95_lower_mm": float(ci[0, node]), "delta_ci95_upper_mm": float(ci[1, node]),
                "improved_seed_count": int((delta[:, node] > 0).sum())})
    return result


def plot(comparison_rows, seed_rows):
    # Contract: two metric panels; four discrete branches; own-baseline percent
    # improvement. Dots show 20 paired seed ratios; large symbols/intervals show
    # ratio of means and its paired-bootstrap CI. Two colors plus marker shapes.
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / ".matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#242424", "text.color": "#242424",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True)
    colors, markers = {"HOV": "#2563A6", "MLP": "#D97732"}, {"HOV": "s", "MLP": "o"}
    jitter = np.linspace(-.045, .045, 20)
    for ax, metric, title in zip(axes, METRICS, ["(a) Skeleton error reduction", "(b) Endpoint error reduction"]):
        for bi, backbone in enumerate(MODELS):
            for vi, variant in enumerate(VARIANT_ZH):
                y = 3 - vi + (.16 if bi == 0 else -.16)
                row = next(r for r in comparison_rows if r["backbone"] == backbone and r["metric"] == metric and r["variant"] == variant)
                points = [r["improvement_pct"] for r in seed_rows if r["backbone"] == backbone and r["metric"] == metric and r["variant"] == variant]
                ax.scatter(points, y + jitter, s=8, c=colors[backbone], alpha=.28, edgecolors="none", zorder=2)
                mean = row["improvement_pct"]
                ax.errorbar(mean, y, xerr=[[mean - row["improvement_pct_ci95_lower"]],
                    [row["improvement_pct_ci95_upper"] - mean]], fmt=markers[backbone],
                    color=colors[backbone], markerfacecolor=colors[backbone] if backbone == "HOV" else "white",
                    markersize=5, linewidth=1.1, capsize=2.5, zorder=3)
                ax.text(max(points + [row["improvement_pct_ci95_upper"]]) + 1.0, y, f"{mean:.2f}",
                        fontsize=8, va="center", color=colors[backbone])
        ax.axvline(0, color="#777777", linewidth=.8, zorder=0)
        ax.grid(axis="x", color="#E6E6E6", linewidth=.6)
        ax.set_axisbelow(True)
        ax.set(title=title, xlabel="Reduction from own no-memory baseline (%)",
               xlim=(-1.5, 56), ylim=(-.48, 3.5), xticks=[0, 10, 20, 30, 40, 50],
               yticks=[3, 2, 1, 0], yticklabels=["No memory", "Path only", "Time only", "Dual memory"])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    handles = [Line2D([0], [0], marker=markers[b], color=colors[b], linestyle="none",
                         markerfacecolor=colors[b] if b == "HOV" else "white", markersize=6, label=b) for b in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.56, .98), ncol=2, frameon=False)
    fig.suptitle("Memory reuse across HOV and a coordinate MLP", fontsize=11, y=1.025)
    fig.text(.5, .015, "20 paired training seeds · fixed 2,958 test targets · small dots: individual seeds\n"
             "Large markers: reduction from mean errors; whiskers: paired-seed bootstrap 95% CI", ha="center", fontsize=8)
    fig.subplots_adjust(left=.14, right=.985, bottom=.22, top=.82, wspace=.16)
    for suffix in ["png", "svg"]:
        fig.savefig(OUTPUT / f"memory_reuse_comparison.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def findings(summary):
    rows, contrasts = summary["comparison"], summary["paired_contrasts"]
    def cell(backbone, variant, metric="mean_node_mm"):
        return next(r for r in rows if (r["backbone"], r["variant"], r["metric"]) == (backbone, variant, metric))
    def contrast(backbone, ref, metric="mean_node_mm"):
        return next(r for r in contrasts if (r["backbone"], r["reference_variant"], r["alternative_variant"], r["metric"]) == (backbone, ref, "both", metric))
    mp, mt = contrast("MLP", "path"), contrast("MLP", "time")
    mep, met = contrast("MLP", "path", "endpoint_mm"), contrast("MLP", "time", "endpoint_mm")
    hp, ht = contrast("HOV", "path"), contrast("HOV", "time")
    lines = ["# 3.5 MLP记忆复用：与3.4的数值对照", "", "## 可插入3.5的段落", "",
        "为考察该历史表示能否用于直接预测骨架坐标的基础网络，在两层64单元的MLP第二隐藏层融合路径与时间记忆特征，并比较无记忆、仅路径、仅时间和双记忆四种配置。",
        "",
        f"以各自的无记忆配置为基准，MLP仅路径、仅时间和双记忆的骨架误差分别降低"
        f"{cell('MLP','path')['improvement_pct']:.2f}%、{cell('MLP','time')['improvement_pct']:.2f}%和{cell('MLP','both')['improvement_pct']:.2f}%，"
        f"末端误差分别降低{cell('MLP','path','endpoint_mm')['improvement_pct']:.2f}%、"
        f"{cell('MLP','time','endpoint_mm')['improvement_pct']:.2f}%和{cell('MLP','both','endpoint_mm')['improvement_pct']:.2f}%。"
        f"相应地，3.4节HOV的三种记忆配置使骨架误差降低{cell('HOV','path')['improvement_pct']:.2f}%、"
        f"{cell('HOV','time')['improvement_pct']:.2f}%和{cell('HOV','both')['improvement_pct']:.2f}%，"
        f"末端误差降低{cell('HOV','path','endpoint_mm')['improvement_pct']:.2f}%、"
        f"{cell('HOV','time','endpoint_mm')['improvement_pct']:.2f}%和{cell('HOV','both','endpoint_mm')['improvement_pct']:.2f}%。"
        "两种基础模型中，单独加入任一记忆分支均改善平均骨架和末端预测，双记忆均取得四种配置中的最低平均误差。",
        "",
        f"在20个配对训练seed中，MLP双记忆相对仅路径的骨架改善出现在{mp['positive_pairs']}/20次，"
        f"平均差值为{mp['mean_reference_minus_alternative_mm']:.3f} mm（95% CI "
        f"[{mp['bootstrap95_lower_mm']:.3f}, {mp['bootstrap95_upper_mm']:.3f}] mm，Holm校正p={mp['wilcoxon_holm_p']:.3g}）；"
        f"相对仅时间的改善出现在{mt['positive_pairs']}/20次，平均差值为{mt['mean_reference_minus_alternative_mm']:.3f} mm"
        f"（95% CI [{mt['bootstrap95_lower_mm']:.3f}, {mt['bootstrap95_upper_mm']:.3f}] mm，Holm校正p={mt['wilcoxon_holm_p']:.3g}）。"
        f"HOV双记忆相对仅路径和仅时间的骨架误差则均在20/20次配对中改善，平均差值分别为"
        f"{hp['mean_reference_minus_alternative_mm']:.3f} mm和{ht['mean_reference_minus_alternative_mm']:.3f} mm。"
        f"MLP双记忆相对两个单分支的末端误差均在20/20次配对中改善；相对仅路径与仅时间的平均差值分别为"
        f"{mep['mean_reference_minus_alternative_mm']:.3f} mm（95% CI [{mep['bootstrap95_lower_mm']:.3f}, {mep['bootstrap95_upper_mm']:.3f}] mm）"
        f"和{met['mean_reference_minus_alternative_mm']:.3f} mm（95% CI [{met['bootstrap95_lower_mm']:.3f}, {met['bootstrap95_upper_mm']:.3f}] mm）。"
        "这些结果支持路径与时间历史特征可在直接输出节点坐标的MLP中发挥预测作用，并在本实验中联合使用获得进一步的平均改善。"
        "区间以训练seed为重采样单位，反映固定数据及既定训练协议下的训练随机性；对新记录及其他网络结构的适用性仍需进一步评价。",
        "",
        f"双记忆MLP的平均骨架误差为{cell('MLP','both')['mean_error_mm']:.3f} mm，低于HOV的"
        f"{cell('HOV','both')['mean_error_mm']:.3f} mm；末端误差亦分别为"
        f"{cell('MLP','both','endpoint_mm')['mean_error_mm']:.3f} mm和{cell('HOV','both','endpoint_mm')['mean_error_mm']:.3f} mm。"
        "这一数值关系属于两个完整模型的描述性比较；模型结构、参数规模和融合位置等同时变化，归因需要进一步的受控比较。",
        "", "## 提议图注", "",
        "图5　记忆特征在HOV与直接坐标MLP中的复用对照。（a）骨架平均节点距离、（b）末端距离相对于各自无记忆配置的降低百分比，"
        "正值表示改善。HOV四种配置对应3.4节的参考形态及记忆分支消融，MLP四种配置对应3.5节的隐藏层融合。"
        "大标记表示20次训练平均误差的相对降低量，细小散点表示每个seed相对于同seed基准的改善率，"
        "横向误差线表示20,000次配对seed重采样的逐项95%百分位区间。两种统计量分别为均值的比值与逐seed比值，"
        "大标记无需等于散点的算术平均。所有配置采用seed 100–119及相同的2,958个测试目标；区间刻画固定测试集条件下的训练随机性。",
        "", "## 数值表", "",
        "改善率 = 100 ×（无记忆平均误差 − 该配置平均误差）/ 无记忆平均误差。误差为均值±样本标准差（mm）。",
        "", "|基体|配置|有效拟合参数|骨架误差|骨架改善%|末端误差|末端改善%|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for backbone in MODELS:
        for variant in VARIANT_ZH:
            a, b = cell(backbone, variant), cell(backbone, variant, "endpoint_mm")
            lines.append(f"|{backbone}|{VARIANT_ZH[variant]}|{a['active_fitted_parameters']}|"
                         f"{a['mean_error_mm']:.6f}±{a['sd_error_mm']:.6f}|{a['improvement_pct']:.4f}|"
                         f"{b['mean_error_mm']:.6f}±{b['sd_error_mm']:.6f}|{b['improvement_pct']:.4f}|")
    lines += ["", "## 双记忆相对单记忆的配对结果", "",
              "差值 = 单记忆误差 − 双记忆误差；正值支持双记忆。", "",
              "|基体|指标|单分支|平均差值/mm|95% CI/mm|改善/20|Holm p|检验族|",
              "|---|---|---|---:|---|---:|---:|---|"]
    for backbone in MODELS:
        for metric, label in METRICS.items():
            for ref in ["path", "time"]:
                r = contrast(backbone, ref, metric)
                lines.append(f"|{backbone}|{label}|{VARIANT_ZH[ref]}|{r['mean_reference_minus_alternative_mm']:.6f}|"
                    f"[{r['bootstrap95_lower_mm']:.6f}, {r['bootstrap95_upper_mm']:.6f}]|{r['positive_pairs']}|"
                    f"{r['wilcoxon_holm_p']:.8g}|{r['family']} ({r['family_size']})|")
    lines += ["", "## 统计口径与核对", "",
        f"已核对{summary['audit']['audited_model_seed_rows']}个模型/seed结果：主要比较160个，另20个为原MLP统计族中的静态容量控制。"
        "所有骨架与末端数值均由保存的预测数组复算，并与逐seed CSV、metrics.json及汇总均值/标准差一致。"
        "共同测试目标按记录的数量为2457、238、263；主指标按目标帧汇总，骨架指标在15个节点上平均，末端为索引14。",
        "",
        "原骨架Wilcoxon-Holm结果及差值CI全部复用并复算验证：HOV ablation族含双记忆与三个消融的3项比较；"
        "MLP base_extensions族含路径、时间、双记忆及静态容量控制的4项比较；MLP双记忆与两个单记忆的比较为2项一族。"
        "HOV原差值方向为完整模型减消融，此处交换符号及区间上下限以统一为改善方向。",
        "",
        "新增末端比较共10项（两个基体各5项），使用单独Holm族supplemental_endpoint_10；"
        "HOV两个单分支相对无记忆的新增骨架比较组成supplemental_hov_single_vs_none_2。"
        "这些为固定既有数据上的补充分析。所有p值均为双侧精确符号秩检验，零差剔除、并列取平均秩；"
        "另保存符号检验作为敏感性核对。区间均为逐项95%百分位区间，未作同时覆盖校正。"
        "bootstrap RNG seed=20260913，20,000次，按完整配对seed共同重采样；测试帧与沿臂节点均不充当独立重复。",
        "", "## 证据边界与写作建议", "",
        "3.5的研究问题宜表述为“该历史表示能否用于直接输出骨架坐标的MLP”。已检验的两种结构支持这一可用性；"
        "对任意基础网络的通用性需要更广泛的结构证据。分支消融检验的是预测贡献，特定材料或物理机制的辨识需要额外受控实验。",
        "",
        "两基体中，两个单分支相对各自无记忆的骨架与末端改善均为20/20。MLP双记忆对仅路径的骨架改善为16/20，"
        "其余4个seed为104、112、115、117，宜写作“进一步降低平均骨架误差”，并同时交代配对一致性。"
        "MLP与HOV的各分支参数规模不同，当前证据中的分支收益包含模型配置变化的影响。",
        "",
        f"静态容量控制与双记忆MLP均为9473个参数；控制的骨架误差为{summary['capacity_control']['mean_node_mm']:.6f} mm，"
        f"相对基础MLP的原Holm p={summary['capacity_control']['wilcoxon_holm_p']:.6f}。"
        "该控制为历史输入带来收益提供补充支持；其具体构造所覆盖的容量解释范围有限。",
        "", "## 沿臂补充证据", "",
        "mlp_along_arm.csv给出全部15个节点的20-seed误差均值、标准差、相对MLP基准的改善及配对差值区间；"
        "mlp_along_arm_seed.csv保留逐seed值。横向位置node_index/14表示节点序号比例。"
        "节点0为基端索引，直接坐标MLP在该点的预测误差也纳入统计；该描述性剖面与整体骨架及末端误差已逐seed核对。"
        "仅路径和仅时间的平均改善出现在节点1–14，双记忆的平均改善出现在节点2–14。"
        "双记忆在节点0的平均误差由0.301增至0.354 mm，节点1由1.452增至1.469 mm；"
        "节点7由1.614降至1.295 mm，末端由4.362降至2.326 mm。沿臂收益因此存在位置差异。",
        "", "## 来源与复现", "",
        f"- MLP汇总与配对统计：`{relative(MLP)}`。",
        f"- HOV汇总：`{relative(HOV)}`；逐seed结果：`{summary['sources']['hov_training']}`。",
        "- 逐数组来源见source_audit.csv；完整检验来源与校正族见paired_contrasts.csv和summary.json。",
        "- 运行：`/Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/analyze_draft2_memory_reuse.py`。",
        "- 图：memory_reuse_comparison.png与同名SVG。",
    ]
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    values, parameters, audit, audit_rows, nodes, sequences, mlp_summary, hov_summary, hov_run = load_and_audit()
    existing, verified = verify_existing(values, mlp_summary, hov_summary, hov_run)
    comparison_rows, seed_rows, contrasts = comparisons(values, parameters, existing)
    along_rows = along_arm(nodes, values)
    capacity = existing[("MLP", "mlp_base", "mlp_static_capacity")]
    cross_model = {metric: {"hov_mean_mm": float(values["hov"][metric].mean()),
        "mlp_mean_mm": float(values["mlp_both"][metric].mean()),
        "hov_minus_mlp_mm": float(values["hov"][metric].mean() - values["mlp_both"][metric].mean())}
        for metric in METRICS}
    summary = {"schema": "draft2_memory_reuse_analysis_v1", "sources": {
        "mlp_analysis": relative(MLP), "hov_analysis": relative(HOV), "hov_training": relative(hov_run),
        "mlp_training_single": relative(mlp_summary["source_run"]),
        "mlp_training_base_and_both": relative(mlp_summary["reused_source_run"])},
        "audit": audit, "existing_statistics_verification": verified,
        "method": {"unit": "paired training seed", "seeds": SEEDS.tolist(),
            "scope": "training randomness conditional on fixed observed data, split and training protocol",
            "skeleton": "mean Euclidean node distance over 2958 targets and 15 nodes, in mm",
            "endpoint": "mean Euclidean distance at node 14 over 2958 targets, in mm",
            "improvement_pct": "100 * (mean baseline error - mean variant error) / mean baseline error",
            "individual_seed_pct": "100 * (baseline error for seed - variant error for seed) / baseline error for seed",
            "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "rng_seed": BOOTSTRAP_SEED,
                          "interval": "pointwise percentile 95%; paired resampling of training seeds"},
            "existing_holm_families": {"ablation": 3, "mlp_base_extensions": 4, "mlp_both_single_mechanism": 2},
            "supplemental_holm_families": {"supplemental_endpoint_10": 10, "supplemental_hov_single_vs_none_2": 2}},
        "comparison": comparison_rows, "paired_contrasts": contrasts,
        "cross_model_descriptive": cross_model,
        "capacity_control": {"parameters": parameters["mlp_static_capacity"],
            "mean_node_mm": float(values["mlp_static_capacity"]["mean_node_mm"].mean()),
            "wilcoxon_holm_p": capacity["wilcoxon_holm_p"]},
        "chart_contract": {"question": "How much does each memory configuration reduce error within each backbone?",
            "family": "comparison and uncertainty", "variant": "two-panel horizontal dots and paired-bootstrap intervals",
            "unit": "20 seed pairs per backbone/branch/metric", "palette": {"HOV": "#2563A6", "MLP": "#D97732"},
            "markers": {"HOV": "filled square", "MLP": "open circle"},
            "exports": ["memory_reuse_comparison.png", "memory_reuse_comparison.svg"]}}
    for filename, records in [("comparison.csv", comparison_rows), ("paired_seed_metrics.csv", seed_rows),
                              ("paired_contrasts.csv", contrasts), ("source_audit.csv", audit_rows),
                              ("mlp_along_arm.csv", along_rows), ("mlp_along_arm_seed.csv", nodes),
                              ("mlp_by_sequence_seed.csv", sequences)]:
        pd.DataFrame(records).to_csv(OUTPUT / filename, index=False)
    plot(comparison_rows, seed_rows)
    (OUTPUT / "findings.md").write_text(findings(summary))
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"Saved: {OUTPUT}")
    print(f"Audit passed: {len(audit_rows)} saved fits; {len(verified)} existing paired contrasts verified.")
    print(pd.DataFrame(comparison_rows)[["backbone", "variant", "metric", "mean_error_mm", "improvement_pct"]].to_string(index=False))
    print(pd.DataFrame(contrasts).query("reference_variant != 'none'")[["backbone", "metric", "reference_variant",
          "mean_reference_minus_alternative_mm", "bootstrap95_lower_mm", "bootstrap95_upper_mm",
          "positive_pairs", "wilcoxon_holm_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
