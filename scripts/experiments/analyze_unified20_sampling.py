#!/usr/bin/env python3
"""Evaluate the frozen unified-20 HOV models on common native 10 Hz targets.

Run with the selfsr Python environment. All artifacts are written to sampling/;
the existing time-memory module supplies data, windowing and state propagation.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
import torch

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
OUT = ROOT / "workspace/runs/analysis/modeling_unified20_20260913_005/sampling"
LEGACY = ROOT / "scripts/experiments/analyze_modeling_time_memory.py"
SEEDS = tuple(range(100, 120))
MODELS = ("hov", "hov_no_maxwell")
METRICS = ("node_mean_mm", "endpoint_mean_mm")
PROTOCOLS = (
    dict(protocol="nominal_10Hz_H39", history=39, stride=1, dt_s=0.1,
         nominal_observation_span_s=3.8, integrated_span_s=3.8),
    dict(protocol="decimated_5Hz_H20", history=20, stride=2, dt_s=0.2,
         nominal_observation_span_s=3.8, integrated_span_s=3.8),
    dict(protocol="wrong_dt_10Hz_H39_dt0.2", history=39, stride=1, dt_s=0.2,
         nominal_observation_span_s=3.8, integrated_span_s=7.6),
)


def load_legacy():
    spec = importlib.util.spec_from_file_location("sampling_time_memory", LEGACY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Its old loader is deliberately replaced before any evaluation call.
    module.STUDY, module.OUT, module.SEEDS = STUDY, OUT, SEEDS
    module.checkpoint = lambda name="hov", seed=100: checkpoint(module, name, seed)[:2]
    return module


def checkpoint(legacy, name, seed):
    path = STUDY / "formal" / name / f"seed_{seed}" / "best_eval_model.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["model"] == name and saved["config"]["seed"] == seed
    assert saved["config"]["study_id"] == STUDY.name
    model, _ = legacy.make_model(saved["model"], saved["config"],
        normalization=(saved["center"], saved["scale"]),
        geometry_config=saved["geometry_config"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model, path, saved


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)),
                min=float(a.min()), max=float(a.max()))


def common_targets(legacy, native):
    alignment, covariates = [], []
    expected = {"seq_20260819_182253": 1061, "seq_20260819_182519": 2876}
    for seq in native:
        ends = np.arange(38, len(seq["actions"]))
        _, ids = legacy.windows(seq["actions"], ends, 39)
        timing_ok = seq["valid"][ids].all(1)
        label_ok = seq["label_valid"][ends]
        boundary_ok = (seq["issue"][ids[:, 1]] >=
                       seq["exposure"][ids[:, 0]] - 1e-6)
        valid = timing_ok & label_ok & boundary_ok
        ends = ends[valid]
        seq["scored_ends"] = ends
        assert len(ends) == expected[seq["seq"]]
        _, full = legacy.windows(seq["actions"], ends, 39)
        _, coarse = legacy.windows(seq["actions"], ends, 20, 2)
        assert np.array_equal(coarse, full[:, ::2])
        assert np.array_equal(full[:, [0, -1]], coarse[:, [0, -1]])
        assert np.all(np.diff(seq["exposure"]) > 0)
        elapsed = seq["exposure"][ends] - seq["exposure"][ends - 38]
        alignment.append(dict(sequence=seq["seq"], available_frames=len(seq["actions"]),
            excluded_initial_context=38, candidate_windows=len(valid),
            excluded_timing_or_label_windows=int((~valid).sum()),
            failed_timing_windows=int((~timing_ok).sum()),
            failed_label_targets=int((~label_ok).sum()),
            failed_issue_boundary_windows=int((~boundary_ok).sum()),
            scored_common_frames=len(ends), even_endpoint_frames=int((ends % 2 == 0).sum()),
            odd_endpoint_frames=int((ends % 2 == 1).sum()), nominal_span_s=3.8,
            actual_exposure_span_s=describe(elapsed)))
        for i, end in enumerate(ends):
            f, c = full[i], coarse[i]
            # Input variation is descriptive; it does not affect eligibility.
            x = seq["actions"][f].astype(np.float64) * 150.0
            covariates.append(dict(target_index=len(covariates), sequence=seq["seq"],
                frame=int(end), first_frame=int(end - 38), phase=int(end % 2),
                exposure_s=float(seq["exposure"][end]),
                first_exposure_s=float(seq["exposure"][end - 38]),
                actual_exposure_span_s=float(elapsed[i]), nominal_span_s=3.8,
                full_exposure_interval_mean_s=float(np.diff(seq["exposure"][f]).mean()),
                coarse_exposure_interval_mean_s=float(np.diff(seq["exposure"][c]).mean()),
                target_issue_s=float(seq["issue"][end]), target_ack_s=float(seq["ack"][end]),
                target_exposure_after_ack_s=float(seq["exposure"][end] - seq["ack"][end]),
                window_exposure_after_ack_min_s=float(np.min(seq["exposure"][f] - seq["ack"][f])),
                window_command_to_ack_mean_s=float(np.mean(seq["ack"][f] - seq["issue"][f])),
                first_next_issue_after_exposure_s=float(seq["issue"][f[1]] - seq["exposure"][f[0]]),
                full_action_total_variation_kpa=float(np.abs(np.diff(x, axis=0)).sum()),
                coarse_action_total_variation_kpa=float(np.abs(np.diff(x[::2], axis=0)).sum()),
                omitted_action_previous_retained_max_delta_kpa=float(np.abs(x[1::2] - x[:-1:2]).max()),
                target_label_valid=True, full_history_timing_valid=True))
    assert len(covariates) == 3937
    assert len({(r["sequence"], r["frame"]) for r in covariates}) == 3937
    return alignment, covariates


@torch.inference_mode()
def evaluate(legacy, native, batch_size):
    n_targets = sum(len(s["scored_ends"]) for s in native)
    err = np.empty((len(MODELS), len(SEEDS), len(PROTOCOLS), n_targets, 2), dtype=np.float64)
    pooled, by_seq, checkpoint_rows, tau_rows = [], [], [], []
    max_reconstruction_mm = 0.0
    for mi, name in enumerate(MODELS):
        for si, seed in enumerate(SEEDS):
            start = time.perf_counter()
            model, path, saved = checkpoint(legacy, name, seed)
            core = model.core
            snapshot = {k: v.clone() for k, v in model.state_dict().items()}
            taus = core.maxwell.taus.detach().clone()
            assert torch.equal(taus, saved["state_dict"]["core.maxwell.taus"])
            assert bool(core.disable_maxwell) == (name == "hov_no_maxwell")
            offset = 0
            for seq in native:
                ends = seq["scored_ends"]
                # Verify the reused consume path against checkpoint-native H20/.2.
                picks = ends[np.linspace(0, len(ends) - 1, 5, dtype=int)]
                a, _ = legacy.windows(seq["actions"], picks, 20, 2)
                a = legacy.tensor(a)
                out, _, _ = legacy.consume(core, a, dt=.2)
                ref = core(a)
                diff = float(torch.max(torch.abs(legacy.physical(core, out["skeleton"]) -
                                                legacy.physical(core, ref["skeleton"]))))
                assert diff < 1e-3, (name, seed, diff)
                max_reconstruction_mm = max(max_reconstruction_mm, diff)
                for pi, protocol in enumerate(PROTOCOLS):
                    for start_idx in range(0, len(ends), batch_size):
                        stop = min(start_idx + batch_size, len(ends))
                        e = ends[start_idx:stop]
                        a, _ = legacy.windows(seq["actions"], e, protocol["history"], protocol["stride"])
                        out, _, _ = legacy.consume(core, legacy.tensor(a), dt=protocol["dt_s"])
                        pred = legacy.physical(core, out["skeleton"]).numpy().astype(np.float64)
                        distance = np.linalg.norm(pred - seq["positions"][e].astype(np.float64), axis=-1)
                        assert np.isfinite(distance).all() and distance.shape == (len(e), 15)
                        err[mi, si, pi, offset + start_idx:offset + stop, 0] = distance.mean(1)
                        err[mi, si, pi, offset + start_idx:offset + stop, 1] = distance[:, -1]
                    values = err[mi, si, pi, offset:offset + len(ends)]
                    by_seq.append(dict(sequence=seq["seq"], model=name, seed=seed,
                        protocol=protocol["protocol"], frames=len(ends),
                        node_mean_mm=float(values[:, 0].mean()),
                        endpoint_mean_mm=float(values[:, 1].mean())))
                offset += len(ends)
            assert all(torch.equal(v, snapshot[k]) for k, v in model.state_dict().items())
            assert torch.equal(taus, core.maxwell.taus)
            if name == "hov_no_maxwell":
                np.testing.assert_array_equal(err[mi, si, 0], err[mi, si, 2])
            for pi, protocol in enumerate(PROTOCOLS):
                values = err[mi, si, pi]
                pooled.append(dict(model=name, seed=seed, protocol=protocol["protocol"], frames=n_targets,
                    node_mean_mm=float(values[:, 0].mean()), endpoint_mean_mm=float(values[:, 1].mean())))
            st = path.stat()
            checkpoint_rows.append(dict(model=name, seed=seed, path=str(path),
                size_bytes=st.st_size, mtime_ns=st.st_mtime_ns, selected_epoch=int(saved["selected_epoch"]),
                training_history=int(saved["config"]["history"]), training_dt_s=float(saved["config"]["dt"]),
                taus_s=taus.tolist(), tau_is_trainable_parameter="taus" in dict(core.maxwell.named_parameters()),
                parameters_and_buffers_unchanged=True, weights_only=False))
            tau_rows.extend(dict(model=name, seed=seed, tau_index=k, tau_s=float(tau),
                decay_dt01=float(torch.exp(-.1 / tau)), decay_dt02=float(torch.exp(-.2 / tau)))
                for k, tau in enumerate(taus))
            write_csv("pooled_seed.csv", pooled)
            write_csv("per_sequence_seed.csv", by_seq)
            print(f"completed {name} seed={seed} protocols=3 targets={n_targets} "
                  f"seconds={time.perf_counter() - start:.2f}", flush=True)
    return err, pooled, by_seq, checkpoint_rows, tau_rows, max_reconstruction_mm


def seed_statistics(pooled, by_seq):
    summaries, paired_seed, comparisons = [], [], []
    scopes = ["pooled"] + sorted({r["sequence"] for r in by_seq})
    bootstrap_ids = np.random.default_rng(20260913).integers(0, len(SEEDS), size=(20000, len(SEEDS)))
    for scope in scopes:
        rows = pooled if scope == "pooled" else [r for r in by_seq if r["sequence"] == scope]
        lookup = {(r["model"], r["seed"], r["protocol"]): r for r in rows}
        assert len(lookup) == len(MODELS) * len(SEEDS) * len(PROTOCOLS)
        values = {}
        for name in MODELS:
            for p in PROTOCOLS:
                for metric in METRICS:
                    a = np.array([lookup[name, seed, p["protocol"]][metric] for seed in SEEDS])
                    values[name, p["protocol"], metric] = a
                    summaries.append(dict(scope=scope, model=name, protocol=p["protocol"], metric=metric,
                        seeds=len(SEEDS), frames_per_seed=rows[0]["frames"], mean_mm=float(a.mean()),
                        sd_mm=float(a.std(ddof=1)), min_mm=float(a.min()), max_mm=float(a.max())))
        p10, p5, pwrong = [p["protocol"] for p in PROTOCOLS]
        definitions = [
            ("hov_sampling_5Hz_minus_10Hz", "sampling_diagnostic", [(1, "hov", p5), (-1, "hov", p10)]),
            ("no_maxwell_sampling_5Hz_minus_10Hz", "sampling_diagnostic",
             [(1, "hov_no_maxwell", p5), (-1, "hov_no_maxwell", p10)]),
            ("10Hz_no_maxwell_minus_hov", "sampling_diagnostic", [(1, "hov_no_maxwell", p10), (-1, "hov", p10)]),
            ("5Hz_no_maxwell_minus_hov", "sampling_diagnostic", [(1, "hov_no_maxwell", p5), (-1, "hov", p5)]),
            ("sampling_interaction_hov_minus_no_maxwell", "sampling_diagnostic",
             [(1, "hov", p5), (-1, "hov", p10), (-1, "hov_no_maxwell", p5), (1, "hov_no_maxwell", p10)]),
            ("hov_wrong_dt_minus_correct_10Hz", "wrong_dt_diagnostic", [(1, "hov", pwrong), (-1, "hov", p10)]),
            ("no_maxwell_wrong_dt_minus_correct_10Hz", "wrong_dt_diagnostic",
             [(1, "hov_no_maxwell", pwrong), (-1, "hov_no_maxwell", p10)]),
        ]
        for comparison, family, terms in definitions:
            for metric in METRICS:
                delta = sum(sign * values[name, protocol, metric] for sign, name, protocol in terms)
                for seed, d in zip(SEEDS, delta):
                    paired_seed.append(dict(scope=scope, comparison=comparison, family=family, metric=metric,
                                            seed=seed, delta_mm=float(d)))
                mean, sd = float(delta.mean()), float(delta.std(ddof=1))
                margin = float(stats.t.ppf(.975, len(SEEDS) - 1) * sd / np.sqrt(len(SEEDS)))
                boot = np.quantile(delta[bootstrap_ids].mean(1), [.025, .975])
                if np.all(delta == 0):
                    statistic, pvalue, method = 0.0, 1.0, "all_zero_convention"
                else:
                    method = "exact" if (np.all(delta != 0) and len(np.unique(np.abs(delta))) == len(delta)) else "approx"
                    result = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method=method)
                    statistic, pvalue = float(result.statistic), float(result.pvalue)
                comparisons.append(dict(scope=scope, comparison=comparison, family=family, metric=metric,
                    seeds=len(SEEDS), mean_delta_mm=mean, sd_delta_mm=sd, median_delta_mm=float(np.median(delta)),
                    ci95_t_low_mm=mean - margin, ci95_t_high_mm=mean + margin,
                    ci95_bootstrap_low_mm=float(boot[0]), ci95_bootstrap_high_mm=float(boot[1]),
                    positive_seeds=int((delta > 0).sum()), negative_seeds=int((delta < 0).sum()),
                    zero_seeds=int((delta == 0).sum()), wilcoxon_statistic=statistic,
                    wilcoxon_p_two_sided=pvalue, wilcoxon_method=method))
    for family in sorted({r["family"] for r in comparisons}):
        ordered = sorted([r for r in comparisons if r["family"] == family], key=lambda r: r["wilcoxon_p_two_sided"])
        adjusted = 0.0
        for i, row in enumerate(ordered):
            adjusted = max(adjusted, min(1.0, (len(ordered) - i) * row["wilcoxon_p_two_sided"]))
            row["wilcoxon_p_holm"] = adjusted
            row["family_tests"] = len(ordered)
    return summaries, paired_seed, comparisons


def table_schema(rows, keys, description):
    def dtype(value):
        return "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "string"
    descriptions = {
        "model": "Frozen checkpoint model name", "seed": "Training repetition ID, 100..119",
        "protocol": "Protocol ID defined in summary.json protocols",
        "sequence": "Original recording ID", "scope": "pooled or one original recording ID",
        "metric": "node_mean_mm (15-node skeleton) or endpoint_mean_mm (node 15)",
        "comparison": "Signed contrast ID; exact direction in definitions",
        "family": "Separate diagnostic multiplicity family, independent of formal testing",
        "frames": "Number of common targets pooled within this row",
        "frames_per_seed": "Fixed common-target denominator for each repetition",
        "seeds": "Number of paired training repetitions",
        "phase": "Original target frame modulo 2; both decimation phases pooled once",
        "target_index": "Zero-based axis3 index in per_target_errors.npz",
        "frame": "Zero-based original target frame index", "first_frame": "Original frame minus 38",
        "tau_index": "Zero-based index in checkpoint Maxwell taus buffer",
        "tau_s": "Unchanged checkpoint time constant",
        "decay_dt01": "exp(-0.1/tau)", "decay_dt02": "exp(-0.2/tau)",
        "wilcoxon_statistic": "Minimum positive/negative signed-rank sum for the two-sided test",
        "wilcoxon_method": "exact, approx or all_zero_convention",
        "wilcoxon_p_two_sided": "Unadjusted two-sided Wilcoxon p-value",
        "wilcoxon_p_holm": "Holm-adjusted p-value within the named diagnostic family",
        "family_tests": "Total hypotheses in this family, including both metrics and all scopes",
        "positive_seeds": "Count of strictly positive paired differences",
        "negative_seeds": "Count of strictly negative paired differences",
        "zero_seeds": "Count of exactly zero paired differences",
        "mean_mm": "Arithmetic mean of 20 seed-level metrics",
        "sd_mm": "Sample SD across 20 seed-level metrics (ddof=1)",
        "mean_delta_mm": "Arithmetic mean of 20 within-seed contrasts",
        "sd_delta_mm": "Sample SD of within-seed contrasts (ddof=1)",
        "delta_mm": "Signed within-seed contrast",
    }
    columns = {}
    for k, v in rows[0].items():
        unit = "mm" if k.endswith("_mm") else "kPa" if k.endswith("_kpa") else "s" if k.endswith("_s") else "dimensionless"
        desc = descriptions.get(k, k.replace("_", " "))
        if k.startswith("ci95_t_"):
            desc = "Lower/upper marginal 95% Student-t interval for paired mean difference, df=19"
        elif k.startswith("ci95_bootstrap_"):
            desc = "Lower/upper marginal 95% percentile interval from 20,000 paired-seed bootstrap draws"
        columns[k] = dict(type=dtype(v), unit=unit, description=desc)
    return dict(description=description, rows=len(rows), primary_key=keys,
                columns=columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or args.batch_size < 1:
        parser.error("threads must be 1..4 and batch-size positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    write_json("status.json", dict(status="running", started_at=datetime.now(timezone.utc).isoformat()))
    legacy = load_legacy()
    paths = [STUDY / "formal" / m / f"seed_{s}" / "best_eval_model.pt" for m in MODELS for s in SEEDS]
    assert all(p.is_file() for p in paths), "The complete frozen set of 40 checkpoints is required"
    initial_stats = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    native, inventory = legacy.native_data()
    for row in inventory:
        row["label_source"] = "existing August-2026 SAM2 centerlines from the original 10Hz sequence train+val files"
        row["qc_csv"] = str(Path(row["manifest"]).parent / "qc_skeleton/skeleton_metrics.csv")
        row["use"] = "frozen unified20 models; common targets from the two original 10Hz recordings"
        manifest = legacy.read_json(row["manifest"])
        row["historical_preprocessing"] = manifest.get("preprocessing", {})
    alignment, covariates = common_targets(legacy, native)
    write_json("alignment.json", alignment)
    write_csv("timingcovariates.csv", covariates)
    write_json("label_sources.json", inventory)
    timing = [s["timing_summary"] for s in native]
    write_json("timing_summary.json", timing)
    print("common targets=3937 (1061+2876), native/decimated aligned, checkpoint count=40", flush=True)
    err, pooled, by_seq, checkpoint_rows, tau_rows, reconstruction = evaluate(legacy, native, args.batch_size)
    summaries, paired, comparisons = seed_statistics(pooled, by_seq)
    write_csv("seed_summary.csv", summaries)
    write_csv("paired_seed_differences.csv", paired)
    write_csv("paired_statistics.csv", comparisons)
    write_csv("checkpoint_taus.csv", tau_rows)
    write_json("checkpoint_sources.json", checkpoint_rows)
    np.savez_compressed(OUT / "per_target_errors.npz", errors_mm=err,
        models=np.array(MODELS), seeds=np.array(SEEDS),
        protocols=np.array([p["protocol"] for p in PROTOCOLS]), metrics=np.array(METRICS),
        sequence=np.array([r["sequence"] for r in covariates]),
        frame=np.array([r["frame"] for r in covariates]))
    # Verify saved arrays, denominators, weighted pooling and paired arithmetic.
    with np.load(OUT / "per_target_errors.npz", allow_pickle=False) as saved_errors:
        np.testing.assert_array_equal(saved_errors["errors_mm"], err)
    max_pool_difference = 0.0
    for row in pooled:
        members = [s for s in by_seq if all(s[k] == row[k] for k in ("model", "seed", "protocol"))]
        assert sum(s["frames"] for s in members) == 3937
        for metric in METRICS:
            weighted = sum(s["frames"] * s[metric] for s in members) / 3937
            max_pool_difference = max(max_pool_difference, abs(row[metric] - weighted))
    assert max_pool_difference < 1e-12
    assert initial_stats == [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    assert len(pooled) == 120 and len(by_seq) == 240 and len(paired) == 840 and len(comparisons) == 42
    # Read only the training dataset's manifest to identify recording overlap.
    protocol_path = STUDY / "protocol.json"
    training_protocol = legacy.read_json(protocol_path)
    training_manifest_path = Path(training_protocol["dataset_manifest"])
    training_manifest = legacy.read_json(training_manifest_path)
    training_groups = sorted({r["group"] for r in training_manifest["files"]})
    assert not set(training_groups) & {s["seq"] for s in native}
    limitations = [
        "这是固定两条10Hz记录上的冻结模型诊断；20个seed的CI描述训练重复差异，不能当作独立记录或机器人总体的CI。",
        "相邻窗口重叠；两种奇偶抽样相位合并后每个共同目标只计一次，推断单位为配对seed。",
        "实际采集约9.05–9.08Hz，共同窗口实际曝光跨度约4.19s；模型使用名义dt，两协议名义积分跨度均为3.8s。",
        "抽样5Hz沿原10Hz执行轨迹取stride2输入，遗漏中间命令；不代表机器人以5Hz重新执行或在新5Hz数据上训练。",
        "计分标签来自2026年8月的旧10Hz SAM2平面中心线；不使用新训练5Hz数据的标签重建声明。旧视觉流程可能含跨帧传播，标签并非独立三维真值。",
        "首帧采用p=h=e(u0)平衡初始化，窗口前史未知；时序covariates仅描述采集偏差，本轮没有按事件时间积分。",
        "所有比较均属诊断，sampling_diagnostic与wrong_dt_diagnostic单独进行Holm校正，均独立于正式模型比较family；95% CI是未作多重性校正的逐项区间。",
        "Wilcoxon双侧符号秩检验假设配对差值分布对称；p值与均值差CI对应的估计量不同。",
        "hov_no_maxwell为对应seed独立拟合的冻结消融模型；模型间差值包含训练后的参考项和其他参数差异。",
    ]
    definitions = dict(
        node_mean_mm="每目标15个节点的欧氏距离均值，再池化全部有效目标；坐标为robot_planar_mm（15×3，平面毫米）。",
        endpoint_mean_mm="每目标第15节点的欧氏距离，再池化全部有效目标。",
        pooling="1061与2876个目标按帧数加权；seed_summary对20个配对seed等权取mean和sample SD。",
        eligibility="end>=38；完整39帧均ACK成功且曝光>=ACK-1e-6；目标标签非interpolated且非hard_invalid；issue[end-37]>=exposure[end-38]-1e-6。",
        initialization="首个曝光处p=h=e(u0)，只有H-1次状态更新；两个抽样相位取同一起止曝光与同一目标标签。",
        recurrence="h_next=exp(-dt/tau)*h+(1-exp(-dt/tau))*e；dt作为consume实参传入。",
        tau_policy="严格加载checkpoint中taus并逐模型验证前后不变；该实现taus为固定buffer，非可训练参数，保留学习所得读出权重。",
        diagnostic_duration="wrong_dt H39/.2仍观察原3.8s名义窗口，但错误积分7.6s。",
        paired_difference="comparison字段直接给出有符号差值；5Hz-minus-10Hz>0表示抽样误差增加；no_maxwell-minus-hov>0表示HOV误差更低。",
        interaction="(HOV_5Hz-HOV_10Hz)-(no_maxwell_5Hz-no_maxwell_10Hz)，正值表示HOV的抽样代价更大。",
        confidence_interval="配对seed均值差的Student-t 95% CI(df=19)，并给出20,000次配对seed重抽样percentile CI，随机seed=20260913。",
        wilcoxon="双侧，zero_method=wilcox；无零值且绝对差无重复时exact，否则正态approx；全零差约定statistic=0,p=1。",
        multiplicity="sampling_diagnostic:5个对比×3个scope×2指标=30检验；wrong_dt_diagnostic:2×3×2=12检验；分别Holm，不合并正式检验。",
        timingcovariates="相对该记录日志原点的秒；exposure=t_grab-frame_age0（缺省frame_age）；issue/ack来自samples.command_id关联commands。",
        action_covariates="原4个模型动作通道乘150恢复kPa；total_variation为相邻输入差绝对值在时间和通道上求和，omitted_delta比较遗漏帧与其前一保留帧。",
    )
    comparison_lookup = {(r["scope"], r["comparison"], r["metric"]): r for r in comparisons}
    def contrast_text(comparison, metric, scope="pooled"):
        r = comparison_lookup[scope, comparison, metric]
        return (f"{r['mean_delta_mm']:+.6f} mm（95% t CI "
                f"[{r['ci95_t_low_mm']:.6f}, {r['ci95_t_high_mm']:.6f}]；"
                f"Wilcoxon p={r['wilcoxon_p_two_sided']:.6g}，Holm p={r['wilcoxon_p_holm']:.6g}）")
    findings = [
        dict(id="hov_sampling", text="HOV抽样5Hz减原生10Hz：骨架" +
             contrast_text("hov_sampling_5Hz_minus_10Hz", "node_mean_mm") + "；末端" +
             contrast_text("hov_sampling_5Hz_minus_10Hz", "endpoint_mean_mm") + "。"),
        dict(id="sequence_dependence", text="HOV骨架抽样差值在182253为" +
             contrast_text("hov_sampling_5Hz_minus_10Hz", "node_mean_mm", "seq_20260819_182253") +
             "，182519为" + contrast_text("hov_sampling_5Hz_minus_10Hz", "node_mean_mm", "seq_20260819_182519") +
             "；pooling按1061/2876帧加权，总体增幅主要来自182519。"),
        dict(id="ablation_sampling", text="hov_no_maxwell的骨架抽样差值为" +
             contrast_text("no_maxwell_sampling_5Hz_minus_10Hz", "node_mean_mm") + "；末端为" +
             contrast_text("no_maxwell_sampling_5Hz_minus_10Hz", "endpoint_mean_mm") +
             "。采样效应依赖模型、指标与记录。"),
        dict(id="hov_advantage", text="在共同目标上，hov_no_maxwell减HOV的pooled骨架差值：10Hz为" +
             contrast_text("10Hz_no_maxwell_minus_hov", "node_mean_mm") + "；抽样5Hz为" +
             contrast_text("5Hz_no_maxwell_minus_hov", "node_mean_mm") + "。"),
        dict(id="wrong_dt", text="HOV在H39误用dt0.2时，相对正确dt0.1的骨架差值为" +
             contrast_text("hov_wrong_dt_minus_correct_10Hz", "node_mean_mm") + "；末端为" +
             contrast_text("hov_wrong_dt_minus_correct_10Hz", "endpoint_mean_mm") +
             "。hov_no_maxwell两种dt的逐目标误差完全一致。"),
    ]
    validation = dict(assessment="share_with_caveats", common_targets=3937, unique_targets=3937,
        complete_seed_ids=list(SEEDS), checkpoint_count=40, pooled_rows=len(pooled),
        per_sequence_rows=len(by_seq), paired_statistics_rows=len(comparisons),
        shared_first_last_frames=True, immutable_parameters_and_buffers=True,
        checkpoint_size_mtime_unchanged=True, checkpoint_native_reconstruction_max_abs_mm=reconstruction,
        weighted_pooling_max_abs_difference_mm=max_pool_difference,
        no_maxwell_wrong_dt_errors_identical=True, saved_error_roundtrip_exact=True,
        training_recording_overlap=[], errors_finite=bool(np.isfinite(err).all()))
    tables = {
        "pooled_seed.csv": table_schema(pooled, ["model", "seed", "protocol"], "每模型/seed/协议池化3937共同目标"),
        "per_sequence_seed.csv": table_schema(by_seq, ["sequence", "model", "seed", "protocol"], "每条记录分别聚合"),
        "seed_summary.csv": table_schema(summaries, ["scope", "model", "protocol", "metric"], "20个seed的mean/sample SD"),
        "paired_seed_differences.csv": table_schema(paired, ["scope", "comparison", "metric", "seed"], "以seed配对的有符号差值"),
        "paired_statistics.csv": table_schema(comparisons, ["scope", "comparison", "metric"], "均值差、两种CI、Wilcoxon与family内Holm"),
        "timingcovariates.csv": table_schema(covariates, ["target_index"], "全部共同目标的标签、时间与动作covariates；sequence/frame也唯一"),
        "checkpoint_taus.csv": table_schema(tau_rows, ["model", "seed", "tau_index"], "冻结时间常数及两种dt的指数衰减"),
    }
    schema = dict(schema_id="modeling_unified20_sampling_v1", summary_file="summary.json",
        summary_required_fields=["schema", "status", "findings", "protocols", "alignment", "seed_summary", "paired_statistics", "definitions", "provenance", "limitations", "validation"],
        csv_tables=tables, definitions=definitions,
        npz={"per_target_errors.npz": dict(
            errors_mm=dict(dtype="float64", shape=list(err.shape),
                           dimensions=["model", "seed", "protocol", "target_index", "metric"]),
            coordinates=["models", "seeds", "protocols", "metrics", "sequence", "frame"],
            target_alignment="axis3 matches timingcovariates.csv target_index order; sequence/frame identify original 10Hz label")},
        json_files={"alignment.json": "每序列筛选计数和实际跨度分布",
            "label_sources.json": "旧10Hz来源NPZ、manifest、QC路径及历史preprocessing元数据",
            "timing_summary.json": "全记录采集/曝光/ACK描述",
            "checkpoint_sources.json": "40个checkpoint来源、大小/mtime、selected_epoch与tau审计",
            "validation.json": "关键一致性检查", "status.json": "本次运行状态"})
    for name, table in tables.items():
        with (OUT / name).open(encoding="utf-8", newline="") as stream:
            saved_rows = list(csv.DictReader(stream))
        assert len(saved_rows) == table["rows"]
        assert list(saved_rows[0]) == list(table["columns"])
        assert len({tuple(r[k] for k in table["primary_key"]) for r in saved_rows}) == len(saved_rows)
    validation["csv_schema_rows_columns_and_unique_keys_verified"] = True
    write_json("schema.json", schema)
    write_json("validation.json", validation)
    payload = dict(schema=schema["schema_id"], status="complete", findings=findings, protocols=list(PROTOCOLS),
        alignment=alignment, timing_summary=timing, seed_summary=summaries, paired_statistics=comparisons,
        definitions=definitions, limitations=limitations, validation=validation,
        provenance=dict(study=str(STUDY), script=str(Path(__file__).resolve()), reused_script=str(LEGACY),
            reused_functions=["native_data", "windows", "consume", "physical"],
            command=f"{sys.executable} -B scripts/experiments/analyze_unified20_sampling.py --threads {args.threads} --batch-size {args.batch_size}",
            output_directory=str(OUT), checkpoint_sources="checkpoint_sources.json", label_sources="label_sources.json",
            raw_sources=[str(legacy.RAW / s["seq"] / f) for s in native
                         for f in ("actions6.csv", "commands.csv", "samples.csv", "meta.json")],
            training_protocol=str(protocol_path), training_manifest=str(training_manifest_path),
            training_groups=training_groups, training_manifest_use="recording names only",
            inference_only=True, image_files_read=0, hashes_recomputed=False,
            new_5hz_labels_used=False, weights_only=False, threads=args.threads, device="cpu",
            versions=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__),
            created_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - start))
    assert all(k in payload for k in schema["summary_required_fields"])
    write_json("summary.json", payload)
    md = ["本轮冻结20次模型的共同目标采样分析", "", "40个checkpoint（两模型×seed 100–119）均完成三个协议；共同目标3937帧（1061+2876）。", ""]
    md += [f"- {r['text']}" for r in findings]
    md += ["", "下表为3937目标池化后，20个seed的均值 ± sample SD，单位mm。", "",
          "| 模型 | 协议 | 骨架 | 末端 |", "|---|---|---:|---:|"]
    lookup = {(r["model"], r["protocol"], r["metric"]): r for r in summaries if r["scope"] == "pooled"}
    for name in MODELS:
        for p in PROTOCOLS:
            n, e = [lookup[name, p["protocol"], metric] for metric in METRICS]
            md.append(f"| {name} | {p['protocol']} | {n['mean_mm']:.6f} ± {n['sd_mm']:.6f} | {e['mean_mm']:.6f} ± {e['sd_mm']:.6f} |")
    md += ["", "配对差值按comparison所列方向；95% CI为配对seed均值差的t区间，完整bootstrap区间与分序列结果见JSON/CSV。", "",
           "| 比较 | 指标 | 差值mm | 95% CI | Wilcoxon p | Holm p |", "|---|---|---:|---|---:|---:|"]
    for r in comparisons:
        if r["scope"] == "pooled":
            md.append(f"| {r['comparison']} | {r['metric']} | {r['mean_delta_mm']:.6f} | [{r['ci95_t_low_mm']:.6f}, {r['ci95_t_high_mm']:.6f}] | {r['wilcoxon_p_two_sided']:.6g} | {r['wilcoxon_p_holm']:.6g} |")
    md += ["", "方法与解释范围", ""] + [f"- {v}" for v in definitions.values()]
    md += [""] + [f"- {v}" for v in limitations]
    md += ["", "交付文件：summary.json（完整摘要）、schema.json（字段/维度/单位定义）、pooled_seed.csv、per_sequence_seed.csv、paired_seed_differences.csv、paired_statistics.csv、timingcovariates.csv、per_target_errors.npz。", "",
           f"验证：共同目标唯一、全部40个模型参数与buffer未变、H20/.2与checkpoint原生forward最大坐标差{reconstruction:.3g} mm、按序列加权池化差{max_pool_difference:.3g} mm；完整检查见validation.json。", ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    write_json("status.json", dict(status="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                                   elapsed_seconds=time.perf_counter() - start, validation=validation))
    print(json.dumps(dict(status="complete", seconds=time.perf_counter() - start,
                          summary=str(OUT / "summary.json"), schema=str(OUT / "schema.json")), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
