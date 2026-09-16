#!/usr/bin/env python3
"""Rebuild HOV from train, verify validation, then evaluate zero joint epochs.

Only this script's dedicated 006 initialization directory receives outputs.
Original trained checkpoints are never loaded. Reference prefitting still learns
from train: 500 coordinate Adam steps, 250 geometry Adam steps, then memory ridge.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

PROCESS_START = time.perf_counter()
sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "workspace/runs/training/modeling_unified20_20260913_004"
OUT = ROOT / "workspace/runs/analysis/modeling_extensions_20260913_006/initialization"
sys.path.insert(0, str(RUN / "source"))
import numpy as np
import torch
from scipy import stats
from torch.optim.optimizer import register_optimizer_step_post_hook
from src.benchmarks.modeling_fast_training import cache_windows
from src.benchmarks.modeling_models import fit_normalization, make_model
from src.benchmarks.modeling_memory_initialization import initialize_memory_readout

IMPORT_SECONDS = time.perf_counter() - PROCESS_START
SEED = 100
FINAL_SEEDS = list(range(100, 120))
STAGES = ["reference_prefit", "hov_joint_epoch0"]
EXPECTED_REFERENCE_VAL = 2.04559063911438
EXPECTED_EPOCH0_VAL = 1.6309987306594849
SCHEMA = "hov_initialization_only_v1"
TIMINGS, CHECKS, ACCESSES = [], [], []
FILES = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(name, condition, **details):
    row = dict(check=name, passed=bool(condition), **details)
    CHECKS.append(row)
    require(condition, f"Check failed: {row}")


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name, rows, key):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    require(len({tuple(r[k] for k in key) for r in rows}) == len(rows), f"Duplicate rows: {name}")
    FILES[name] = dict(format="csv", rows=len(rows), primary_key=key, columns=fields)


def save_arrays(name, **arrays):
    np.savez_compressed(OUT / name, **arrays)
    FILES[name] = dict(format="npz", arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()})


def log(message):
    print(message, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(stamp() + " " + message + "\n")


@contextmanager
def timed(stage):
    before = time.perf_counter()
    yield
    TIMINGS.append(dict(stage=stage, wall_seconds=time.perf_counter()-before))


def hardware():
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    return dict(at=stamp(), cpu=cpu, hostname=platform.node(), platform=platform.platform(),
                logical_cpus=os.cpu_count(), affinity=list(os.sched_getaffinity(0)),
                load_average=list(os.getloadavg()), python=sys.version, executable=sys.executable,
                torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(), concurrent_fits_in_this_process=1,
                cpu_isolation=False, timer="time.perf_counter")


def load_role(manifest_path, role, test_gate=False):
    require(role != "test" or test_gate, "Test access requires both validation gates")
    meta = read(manifest_path)
    sequences = []
    for record in meta["files"]:
        if record["role"] != role:
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        ACCESSES.append(dict(role=role, path=rel(path), opened_at=stamp(), validation_gate_passed=test_gate))
        with np.load(path, allow_pickle=False) as archive:
            sequence = {k: archive[k].copy() for k in archive.files}
        require(sequence["actions"].shape == (record["frames"], 4), "Action dimensions")
        require(sequence["positions"].shape == (record["frames"], 15, 3), "Target dimensions")
        require(np.isfinite(sequence["actions"]).all() and np.isfinite(sequence["positions"]).all(), "Nonfinite data")
        require(np.array_equal(sequence["frame_ids"], np.arange(record["start"], record["stop"])), "Chronology mismatch")
        sequence["record"] = record
        sequences.append(sequence)
    x, y, groups = cache_windows(sequences, 20, "cpu")
    expected = 8988 if role == "train" else 2958
    check(f"{role}_window_count", len(x) == expected and len(sequences) == 3, windows=len(x))
    return dict(sequences=sequences, x=x, y=y, groups=groups,
                frame_ids=np.concatenate([q["frame_ids"][19:] for q in sequences]),
                timestamps=np.concatenate([q["timestamps"][19:] for q in sequences]))


def predict(model, x, center, scale, stage):
    model.eval()
    with torch.inference_mode():
        pieces = []
        for batch in x.split(512 if stage == "reference_prefit" else 256):
            normalized = model.core.decode_equilibrium(batch[:, -1]) if stage == "reference_prefit" else model(batch)
            pieces.append(normalized * scale + center)
        result = torch.cat(pieces)
    require(bool(torch.isfinite(result).all()), "Nonfinite predictions")
    return result


def metric_data(prediction, target):
    distances = np.linalg.norm(np.asarray(prediction, dtype=np.float64)-np.asarray(target, dtype=np.float64), axis=-1)
    frame_rmse = np.sqrt(np.mean(distances**2, axis=1))
    frame = dict(mean_node_mm=distances.mean(axis=1), endpoint_mm=distances[:, -1], node_rmse_mm=frame_rmse)
    pooled = dict(mean_node_mm=float(distances.mean()), endpoint_mm=float(distances[:, -1].mean()),
                  node_rmse_mm=float(frame_rmse.mean()), node_global_rmse_mm=float(np.sqrt(np.mean(distances**2))),
                  endpoint_rmse_mm=float(np.sqrt(np.mean(distances[:, -1]**2))))
    return pooled, frame


def record_stage_metrics(role, data, predictions, stage_rows, sequence_rows, frame_rows):
    for stage, pred in predictions.items():
        metrics, frame = metric_data(pred, data["y"].numpy())
        stage_rows.append(dict(stage=stage, role=role, seed=SEED, independent_initialization_fits=1,
                               frames=len(pred), **metrics))
        for group, seq in enumerate(data["sequences"]):
            mask = data["groups"] == group
            values, _ = metric_data(pred[mask], data["y"].numpy()[mask])
            sequence_rows.append(dict(stage=stage, role=role, seed=SEED, group=group,
                                      sequence=seq["record"]["group"], frames=int(mask.sum()), **values))
        if role == "test":
            for i in range(len(pred)):
                frame_rows.append(dict(stage=stage, group=int(data["groups"][i]), frame_id=int(data["frame_ids"][i]),
                                       test_window_index=i, **{k: float(v[i]) for k, v in frame.items()}))


def checkpoint(model, config, geometry, norm, stage, val):
    path = OUT / f"{stage}.pt"
    torch.save(dict(schema="hov_initialization_checkpoint_v1", model="hov", stage=stage,
                    prediction_mode="decode_equilibrium" if stage == "reference_prefit" else "full_H20",
                    seed=SEED, joint_epochs=0, config=dict(config, epochs=0), source_config=config,
                    state_dict={k: v.detach().clone() for k, v in model.state_dict().items()},
                    geometry_config=geometry, center=norm["center"], scale=norm["scale"],
                    validation_node_mean_mm=val, trained_from="train sequences only; fresh make_model",
                    prefit_adam_steps=750, memory_initialized=stage == "hov_joint_epoch0"), path)
    FILES[path.name] = dict(format="torch checkpoint", schema="hov_initialization_checkpoint_v1",
                            joint_epochs=0, seed=SEED, prediction_mode="decode_equilibrium" if stage == "reference_prefit" else "full_H20")


def compare_final(test, stage_metrics):
    with (RUN / "raw_test.csv").open() as stream:
        original = {int(r["seed"]): r for r in csv.DictReader(stream) if r["model"] == "hov"}
    require(sorted(original) == FINAL_SEEDS, "Final HOV seed coverage")
    final_rows, differences = [], []
    metric_error = 0.
    for seed in FINAL_SEEDS:
        path = RUN / f"evaluation/hov/seed_{seed}/predictions.npz"
        with np.load(path, allow_pickle=False) as archive:
            check("final_prediction_targets_aligned", np.array_equal(archive["groups"], test["groups"])
                  and np.array_equal(archive["frame_ids"], test["frame_ids"]), seed=seed)
            pred = archive["prediction_mm"]
            require(pred.shape == (2958, 15, 3), "Final prediction count")
        metrics, _ = metric_data(pred, test["y"].numpy())
        for metric, value in metrics.items():
            metric_error = max(metric_error, abs(value-float(original[seed][metric])))
        final_rows.append(dict(seed=seed, frames=2958, selected_epoch=int(original[seed]["best_epoch"]),
                               source=rel(path), **metrics))
        for stage in STAGES:
            baseline = stage_metrics[stage]["mean_node_mm"]
            differences.append(dict(family="initialization_vs_final_hov_diagnostic", stage=stage, final_seed=seed,
                                    baseline_fit_seed=SEED, baseline_independent_fits=1,
                                    baseline_mean_node_mm=baseline, final_mean_node_mm=metrics["mean_node_mm"],
                                    baseline_minus_final_mm=baseline-metrics["mean_node_mm"]))
    check("final_metrics_match_original_raw", metric_error < 1e-10, max_abs_mm=metric_error)
    rng = np.random.default_rng(20260913)
    bootstrap_indices = rng.integers(0, 20, size=(20000, 20))
    comparisons = []
    for stage in STAGES:
        delta = np.array([r["baseline_minus_final_mm"] for r in differences if r["stage"] == stage])
        positive, negative = int((delta > 0).sum()), int((delta < 0).sum())
        nonzero = positive+negative
        pvalue = float(stats.binomtest(positive, nonzero, .5, alternative="two-sided").pvalue) if nonzero else 1.
        boot = delta[bootstrap_indices].mean(axis=1)
        lower, upper = np.quantile(boot, [.025, .975])
        mean, sd = float(delta.mean()), float(delta.std(ddof=1))
        margin = float(stats.t.ppf(.975, 19)*sd/math.sqrt(20))
        comparisons.append(dict(family="initialization_vs_final_hov_diagnostic", stage=stage, metric="mean_node_mm",
                                n_final_seeds=20, independent_baseline_fits=1, mean_baseline_minus_final_mm=mean,
                                paired_difference_sd_mm=sd, conditional_bootstrap95_mm=[float(lower), float(upper)],
                                conditional_t95_mm=[mean-margin, mean+margin], positive_pairs=positive,
                                negative_pairs=negative, zero_pairs=20-nonzero, sign_test_two_sided_p=pvalue))
    adjusted = 0.
    for rank, row in enumerate(sorted(comparisons, key=lambda r: r["sign_test_two_sided_p"])):
        adjusted = max(adjusted, min(1., (len(comparisons)-rank)*row["sign_test_two_sided_p"]))
        row["sign_test_holm_p"] = adjusted
    final_summary = {metric: dict(mean=float(np.mean([r[metric] for r in final_rows])),
                                 sd=float(np.std([r[metric] for r in final_rows], ddof=1)))
                     for metric in ("mean_node_mm", "endpoint_mm", "node_rmse_mm", "node_global_rmse_mm", "endpoint_rmse_mm")}
    return final_rows, differences, comparisons, final_summary


def render_summary(result):
    rows = {r["stage"]: r for r in result["stage_metrics"] if r["role"] == "test"}
    final = result["final_hov_summary"]
    names = {"reference_prefit": "reference预拟合后", "hov_joint_epoch0": "reference＋记忆初始化，0联合epoch"}
    lines = ["# HOV 初始化后、0联合epoch测试", "", "以seed100从004冻结源码和训练集完整重建一次。初始化基线只有一个实际拟合结果；两阶段共享同一次reference预拟合。", "",
             "0联合epoch表示跳过后续100 epoch联合优化。reference已执行500步坐标目标Adam及250步几何目标Adam，记忆读出再进行训练集ridge闭式拟合；模型已从训练数据学习。", "",
             f"验证门：reference={result['validation_values']['reference_prefit']:.12f} mm（目标{EXPECTED_REFERENCE_VAL:.12f}），"
             f"完整初始化={result['validation_values']['hov_joint_epoch0']:.12f} mm（目标{EXPECTED_EPOCH0_VAL:.12f}）；均在1e-6 mm内。通过两门后才打开test数组。", "",
             "|阶段|独立拟合数|测试骨架/mm|末端/mm|逐帧节点RMSE均值/mm|全局节点RMSE/mm|末端RMSE/mm|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    metrics = ["mean_node_mm", "endpoint_mm", "node_rmse_mm", "node_global_rmse_mm", "endpoint_rmse_mm"]
    for stage in STAGES:
        lines.append(f"|{names[stage]}|1|"+"|".join(f"{rows[stage][m]:.6f}" for m in metrics)+"|")
    lines.append("|正式HOV最终模型|20|"+"|".join(f"{final[m]['mean']:.6f}±{final[m]['sd']:.6f}" for m in metrics)+"|")
    lines += ["", "|条件对比（初始化阶段 − 正式最终HOV）|骨架均值差/mm|条件配对bootstrap95%CI/mm|正/负/零|双侧符号检验p|新族Holm p|",
              "|---|---:|---:|---:|---:|---:|"]
    for row in result["comparisons"]:
        lo, hi = row["conditional_bootstrap95_mm"]
        lines.append(f"|{names[row['stage']]}|{row['mean_baseline_minus_final_mm']:.6f}|[{lo:.6f}, {hi:.6f}]|"
                     f"{row['positive_pairs']}/{row['negative_pairs']}/{row['zero_pairs']}|"
                     f"{row['sign_test_two_sided_p']:.9g}|{row['sign_test_holm_p']:.9g}|")
    lines += ["", "正差表示正式最终HOV误差更低。每项将同一固定初始化标量与20个最终seed逐一作差，"
              "CI仅描述固定初始化、同一数据划分条件下最终训练随机性的差异。初始基线没有20个独立拟合，"
              "不为其构造跨seed标准差。两项骨架符号检验独立构成新的诊断族（Holm n=2），不并入既有15项检验；"
              "均值差CI为20,000次配对seed重抽样百分位区间，未作多重区间校正，另存t区间。", "",
              "|阶段|记录|目标数|骨架/mm|末端/mm|",
              "|---|---|---:|---:|---:|"]
    for row in result["per_sequence_metrics"]:
        if row["role"] == "test":
            lines.append(f"|{names[row['stage']]}|{row['sequence']}|{row['frames']}|{row['mean_node_mm']:.6f}|{row['endpoint_mm']:.6f}|")
    lines += ["", "记录按2457/238/263帧计权；帧、记录和重叠窗口不作为额外独立重复。旧test参与数据子集及部分容量选择的局限继续成立。", "",
              f"CPU：{result['hardware_before']['cpu']}；PyTorch {result['hardware_before']['torch']}；单进程、单线程。", "",
              "|构建/评价阶段|实际墙钟/s|", "|---|---:|"]
    lines += [f"|{r['stage']}|{r['wall_seconds']:.6f}|" for r in TIMINGS]
    wall = result["walltime"]
    lines += ["", f"完整构建至test门（含数据/窗口、训练归一化核对、两阶段拟合、val检查、checkpoint保存与重载核验）："
              f"{wall['complete_build_to_test_gate_seconds']:.6f} s。模型/reference构建加记忆初始化本身："
              f"{wall['model_build_and_memory_fit_seconds']:.6f} s。数值库导入另记{IMPORT_SECONDS:.6f} s。", "",
              "本次单进程环境与原8任务并发拟合环境不同；上述墙钟不用于推算原训练节省百分比。"
              "reference计时包含实际750次Adam步骤及模型构建；其余阶段见stages.csv。", "",
              "reference测试使用重新预拟合模型的decode_equilibrium；完整初始化使用相同H20窗口的完整forward。"
              "原正式checkpoint权重未被读取；正式对照使用其保存的逐帧预测，并独立复算坐标指标。"
              "输出含两阶段checkpoint、val/test预测、逐帧指标、逐记录指标、配对差值及schema。"]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-incomplete", action="store_true", help="Replace only this script's incomplete output; completed output is immutable")
    args = parser.parse_args()
    if OUT.exists() and any(OUT.iterdir()):
        require(args.resume_incomplete and (OUT/"protocol.json").exists()
                and read(OUT/"protocol.json").get("schema") == SCHEMA and not (OUT/"COMPLETE.json").exists(),
                "Refusing to overwrite an existing result directory")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    machine = hardware()
    cfg_path = RUN / "formal/hov/seed_100/resolved_config.json"
    cfg = read(cfg_path)
    norm = read(RUN/"normalization.json")
    original_protocol = read(RUN/"protocol.json")
    manifest_path = Path(original_protocol["dataset_manifest"])
    meta = read(manifest_path)
    require(cfg["seed"] == SEED and cfg["prior_steps"] == 500 and cfg["memory_ridge"] == .001
            and cfg["history"] == 20 and cfg["dt"] == .2 and meta["H"] == 20, "Unexpected frozen configuration")
    historical = [read(RUN/f"formal/hov/seed_{seed}/run_manifest.json")["epoch0_validation_node_mean_mm"] for seed in FINAL_SEEDS]
    check("all_twenty_historical_initialization_vals_identical", all(v == EXPECTED_EPOCH0_VAL for v in historical), n_manifests=20)
    protocol = dict(schema=SCHEMA, started_at=stamp(), source_run=rel(RUN), source_snapshot=rel(RUN/"source"),
                    config_source=rel(cfg_path), normalization_source=rel(RUN/"normalization.json"),
                    dataset_manifest=rel(manifest_path), source_config=cfg, actual_rebuild_seeds=[SEED],
                    independent_initialization_fits=1, subsequent_joint_epochs=0, final_comparator_seeds=FINAL_SEEDS,
                    fit_data="train only", validation_gate_atol_mm=1e-6,
                    reference_validation_target_mm=EXPECTED_REFERENCE_VAL, epoch0_validation_target_mm=EXPECTED_EPOCH0_VAL,
                    reference_validation_target_source="005 prior supplemental reference validation; used as reproduction target only",
                    initialization="p=h=e(first input), q=d=0; H20 contains 19 updates; split-local windows",
                    optimizer_audit="global Adam post-step observer records actual update counts and learning rates",
                    statistics=dict(family="initialization_vs_final_hov_diagnostic", family_size=2, metric="mean_node_mm",
                                    direction="fixed stage baseline minus final seed", inference_unit="final training seed conditional on one fixed initialization",
                                    test="exact two-sided binomial sign test; Holm across two stage contrasts",
                                    bootstrap_samples=20000, bootstrap_rng=20260913, confidence=.95,
                                    scope="conditional mean difference CI; not twenty initialization fits or independent datasets"),
                    hardware_before=machine, numerical_import_seconds=IMPORT_SECONDS)
    write_json("protocol.json", protocol)
    log("Rebuild seed100 from train; no original checkpoint weights will be loaded")
    with timed("load_train_arrays_and_H20_windows"):
        train = load_role(manifest_path, "train")
    with timed("verify_train_normalization"):
        center_np, scale = fit_normalization(train["sequences"])
        check("normalization_reproduced_from_train", np.array_equal(center_np, np.asarray(norm["center"], dtype=np.float32))
              and scale == norm["scale"], scale=scale)
    center = torch.tensor(norm["center"], dtype=torch.float32)
    with timed("load_validation_arrays_and_H20_windows"):
        val = load_role(manifest_path, "val")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    adam_rates = []
    def observed_step(optimizer, unused_args, unused_kwargs):
        require(isinstance(optimizer, torch.optim.Adam), "Unexpected optimizer")
        adam_rates.append(float(optimizer.param_groups[0]["lr"]))
    hook = register_optimizer_step_post_hook(observed_step)
    try:
        with timed("fresh_make_model_including_reference_500plus250_Adam"):
            model, geometry = make_model("hov", cfg, train["sequences"], (center.numpy(), scale))
        check("actual_reference_Adam_steps", adam_rates == [.03]*500+[.003]*250,
              total=len(adam_rates), counts_by_lr=dict(Counter(map(str, adam_rates))))
        with timed("reference_validation"):
            reference_val = predict(model, val["x"], center, scale, STAGES[0])
            ref_score = float(torch.linalg.vector_norm(reference_val-val["y"], dim=-1).mean())
        check("reference_validation_reproduced", abs(ref_score-EXPECTED_REFERENCE_VAL) <= 1e-6,
              actual_mm=ref_score, expected_mm=EXPECTED_REFERENCE_VAL, absolute_difference_mm=abs(ref_score-EXPECTED_REFERENCE_VAL))
        with timed("save_reference_checkpoint"):
            checkpoint(model, cfg, geometry, norm, STAGES[0], ref_score)
        before_memory = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with timed("ridge_memory_initialization_train_only"):
            memory = initialize_memory_readout(model, train["x"], train["y"], ridge=cfg["memory_ridge"])
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, before_memory[k])]
        check("memory_changes_only_reported_train_readouts", set(changed) == set(memory["modified_parameters"]), changed=changed)
        check("memory_is_closed_form_and_joint_updates_zero", len(adam_rates) == 750 and memory["n_windows"] == 8988,
              subsequent_Adam_steps=len(adam_rates)-750, solver=memory["solver"], ridge=memory["ridge"])
    finally:
        hook.remove()
    with timed("epoch0_validation"):
        zero_val = predict(model, val["x"], center, scale, STAGES[1])
        zero_score = float(torch.linalg.vector_norm(zero_val-val["y"], dim=-1).mean())
    check("epoch0_validation_reproduced", abs(zero_score-EXPECTED_EPOCH0_VAL) <= 1e-6,
          actual_mm=zero_score, expected_mm=EXPECTED_EPOCH0_VAL, absolute_difference_mm=abs(zero_score-EXPECTED_EPOCH0_VAL))
    with timed("save_epoch0_checkpoint"):
        checkpoint(model, cfg, geometry, norm, STAGES[1], zero_score)
    models = {}
    with timed("reload_own_checkpoints_and_validate_outputs"):
        for stage, expected in [(STAGES[0], reference_val), (STAGES[1], zero_val)]:
            saved = torch.load(OUT/f"{stage}.pt", map_location="cpu", weights_only=True)
            restored, _ = make_model("hov", saved["config"], normalization=(center.numpy(), scale), geometry_config=saved["geometry_config"])
            restored.load_state_dict(saved["state_dict"], strict=True)
            restored.eval().requires_grad_(False)
            actual = predict(restored, val["x"], center, scale, stage)
            check("saved_checkpoint_validation_replay", torch.equal(actual, expected), stage=stage,
                  max_abs_coordinate_mm=float((actual-expected).abs().max()))
            models[stage] = restored
    build_seconds = time.perf_counter()-start
    gate_at = stamp()
    write_json("validation.json", dict(status="validation_gates_passed", passed_at=gate_at, checks=CHECKS))
    log(f"Validation gates passed: reference={ref_score:.12f}, zero_joint_epoch={zero_score:.12f}; opening test")
    with timed("load_test_after_validation_gates"):
        test = load_role(manifest_path, "test", test_gate=True)
    with timed("evaluate_two_frozen_stages_on_test"):
        test_predictions = {stage: predict(models[stage], test["x"], center, scale, stage).numpy() for stage in STAGES}
    stage_rows, sequence_rows, frame_rows = [], [], []
    val_predictions = {STAGES[0]: reference_val.numpy(), STAGES[1]: zero_val.numpy()}
    with timed("metrics_and_prediction_exports"):
        for role, data, predictions in [("val", val, val_predictions), ("test", test, test_predictions)]:
            record_stage_metrics(role, data, predictions, stage_rows, sequence_rows, frame_rows)
            save_arrays(f"{role}_predictions.npz", target_mm=data["y"].numpy(), groups=data["groups"],
                        frame_ids=data["frame_ids"], timestamps=data["timestamps"],
                        reference_prefit_mm=predictions[STAGES[0]], hov_joint_epoch0_mm=predictions[STAGES[1]])
        write_csv("stage_metrics.csv", stage_rows, ["stage", "role"])
        write_csv("per_sequence_metrics.csv", sequence_rows, ["stage", "role", "group"])
        write_csv("per_frame_metrics.csv", frame_rows, ["stage", "group", "frame_id"])
    test_metrics = {r["stage"]: r for r in stage_rows if r["role"] == "test"}
    with timed("final_HOV_comparison_and_conditional_statistics"):
        final_rows, differences, comparisons, final_summary = compare_final(test, test_metrics)
        write_csv("final_hov_per_seed.csv", final_rows, ["seed"])
        write_csv("paired_differences.csv", differences, ["stage", "final_seed"])
    with timed("verify_saved_test_arrays_and_sequence_aggregation"):
        with np.load(OUT/"test_predictions.npz", allow_pickle=False) as saved:
            for stage in STAGES:
                key = stage+"_mm"
                check("saved_test_prediction_exact", np.array_equal(saved[key], test_predictions[stage]), stage=stage)
                recomputed, _ = metric_data(saved[key], saved["target_mm"])
                check("saved_test_metrics_exact", all(recomputed[k] == test_metrics[stage][k] for k in recomputed), stage=stage)
        for row in stage_rows:
            pieces = [q for q in sequence_rows if q["stage"] == row["stage"] and q["role"] == row["role"]]
            weighted = sum(q["mean_node_mm"]*q["frames"] for q in pieces)/row["frames"]
            check("sequence_frame_weighting", abs(weighted-row["mean_node_mm"]) < 1e-12,
                  stage=row["stage"], role=row["role"], absolute_difference_mm=abs(weighted-row["mean_node_mm"]))
    check("all_test_opens_after_validation_gates", all(r["validation_gate_passed"] and r["opened_at"] >= gate_at
          for r in ACCESSES if r["role"] == "test"), gate_passed_at=gate_at)
    write_csv("stages.csv", TIMINGS, ["stage"])
    by_time = {r["stage"]: r["wall_seconds"] for r in TIMINGS}
    result = dict(schema=SCHEMA, status="complete", seed=SEED, independent_initialization_fits=1,
                  subsequent_joint_epochs=0, validation_values=dict(zip(STAGES, [ref_score, zero_score])),
                  observed_prefit=dict(configured_coordinate_steps=500, actual_coordinate_Adam_steps=500,
                                       actual_geometry_Adam_steps=250, total_Adam_steps=len(adam_rates), joint_Adam_steps=0),
                  memory_initialization=memory, stage_metrics=stage_rows, per_sequence_metrics=sequence_rows,
                  final_hov_summary=final_summary, comparisons=comparisons, hardware_before=machine, hardware_after=hardware(),
                  walltime=dict(complete_build_to_test_gate_seconds=build_seconds,
                                model_build_and_memory_fit_seconds=by_time["fresh_make_model_including_reference_500plus250_Adam"]+by_time["ridge_memory_initialization_train_only"],
                                numerical_import_seconds=IMPORT_SECONDS, analysis_seconds_before_report=time.perf_counter()-start,
                                process_seconds_before_report=time.perf_counter()-PROCESS_START),
                  validation=dict(passed=len(CHECKS), failed=0, tolerance_mm=1e-6),
                  sources=dict(formal_run=rel(RUN), dataset_manifest=rel(manifest_path), original_checkpoint_weights_loaded=False),
                  files={name: rel(OUT/name) for name in FILES})
    protocol.update(validation_gate_passed_at=gate_at, data_access_order=ACCESSES,
                    observed_prefit=result["observed_prefit"], complete_build_to_test_gate_seconds=build_seconds)
    write_json("protocol.json", protocol)
    write_json("summary.json", result)
    write_json("validation.json", dict(status="complete", assessment="conditional diagnostic; single fixed initialization fit",
                                       passed=len(CHECKS), failed=0, checks=CHECKS))
    FILES.update({"summary.json": dict(format="json", schema=SCHEMA), "protocol.json": dict(format="json", schema=SCHEMA),
                  "validation.json": dict(format="json", fields=["status", "assessment", "passed", "failed", "checks"]),
                  "summary.md": dict(format="markdown"), "run.log": dict(format="text"), "COMPLETE.json": dict(format="json")})
    write_json("schema.json", dict(schema="hov_initialization_files_v1", files=FILES,
        units=dict(coordinates="mm, robot_planar_mm, base-to-tip, 15x3", time="seconds", errors="mm", probabilities="unitless"),
        definitions=dict(mean_node_mm="Mean of Euclidean node errors per frame, then pooled frames",
                         endpoint_mm="Mean Euclidean error at node14", node_rmse_mm="Frame-node RMSE then frame mean",
                         node_global_rmse_mm="Sqrt of mean squared Euclidean errors over all frame-node pairs",
                         endpoint_rmse_mm="Sqrt of mean squared endpoint Euclidean error",
                         seed="One reconstruction seed100; final_seed100..119 indexes final joint-trained models",
                         uncertainty="Baseline fixed. Bootstrap/t CI resamples final training seeds; no initialization variance estimated",
                         checkpoint_reference="Use decode_equilibrium on last input; unused initial memory weights are not part of the reference prediction",
                         null="Not applicable or unrecorded; CSV empty / JSON null")))
    (OUT/"summary.md").write_text(render_summary(result), encoding="utf-8")
    write_json("COMPLETE.json", dict(status="complete", completed_at=stamp(), actual_rebuild_seeds=[SEED],
                                     test_frames=2958, passed_checks=len(CHECKS),
                                     process_wall_seconds=time.perf_counter()-PROCESS_START))
    log(f"Complete: zero-joint-epoch test skeleton={test_metrics[STAGES[1]]['mean_node_mm']:.9f} mm; "
        f"final HOV mean={final_summary['mean_node_mm']['mean']:.9f} mm; {len(CHECKS)} checks passed")


if __name__ == "__main__":
    main()
