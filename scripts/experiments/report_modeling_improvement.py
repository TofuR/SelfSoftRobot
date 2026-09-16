#!/usr/bin/env python3
"""Report the complete fixed three-sequence, five-seed modeling improvement study.

Usage: python scripts/experiments/report_modeling_improvement.py --study PATH
Optional: --wait-seconds 3600 --learning-curves

Only saved training metadata, evaluation arrays and latency JSON are read.
No training, checkpoint loading, dataset hashing or metric selection is done.
All required inputs are checked before replacing any files under results/.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.benchmarks.modeling_seed_summary import summarize_seeds
from src.evaluation.modeling_benchmark_metrics import skeleton_metrics

METRICS = ("mean_node_mm", "node_rmse_mm", "endpoint_mm", "mask_iou", "mask_dice")
LABELS = ("节点均距/mm ↓", "节点RMSE/mm ↓", "末端误差/mm ↓", "IoU ↑", "Dice ↑")
MODEL_LABELS = {
    "hov": "HOV", "hov_no_play": "HOV 无PI", "hov_no_maxwell": "HOV 无Maxwell",
    "hov_no_memory": "HOV 静态参考", "chen_direction": "Chen2025-inspired DirectionMLP",
    "bezier_gru": "Yu Bézier–GRU（表示适配）",
    "oscillator": "Krauss-VON-inspired oscillator adaptation", "koopman": "Koopman adaptation",
    "pcc": "PCC几何适配", "mlp": "静态MLP", "linear": "线性回归",
}
RUNTIME_KEYS = {
    "seed", "run_kind", "command", "device", "threads", "study_id", "model",
    "dataset_manifest", "dataset_manifest_sha256", "frozen_config_sha256",
    "epochs", "minimum_epochs", "early_stop_checks", "schedule_lr",
}
SCOPE = ("固定三序列事后选择的子集；每个 seed 内合并所有评分帧，再对预定的五个 seed 等权汇总。"
         "样本标准差 ddof=1，仅描述固定划分下的训练随机性，不能作为跨序列、跨划分或跨日泛化证据。"
         "100 epoch 是优化预算，不代表渐近最优或已充分收敛。")


class IncompleteResults(RuntimeError):
    """Required inputs have not finished; retrying may resolve this condition."""


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IncompleteResults(f"缺少文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise IncompleteResults(f"JSON 尚未完整写入或格式错误：{path}: {exc}") from exc


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"{label} 必须是有限数值")
    require(minimum is None or value >= minimum, f"{label} 小于允许值 {minimum}")
    return value


def design(study):
    plan = read_json(study / "study_plan.json")
    seeds = plan.get("repeat_seeds", plan.get("seeds"))
    require(isinstance(seeds, list) and len(seeds) == 5
            and all(type(seed) is int and seed >= 0 for seed in seeds)
            and len(set(seeds)) == 5, "study_plan 必须声明五个唯一 repeat_seeds/seeds")
    comparisons, ablations = plan.get("comparisons"), plan.get("ablations")
    for label, models, count in (("comparisons", comparisons, 8), ("ablations", ablations, 4)):
        require(isinstance(models, list) and len(models) == count
                and all(isinstance(m, str) and m and Path(m).name == m and m not in (".", "..")
                        for m in models) and len(set(models)) == count and "hov" in models,
                f"study_plan.{label} 必须声明含 hov 的 {count} 个唯一模型")
    groups = plan.get("groups")
    require(isinstance(groups, list) and len(groups) == 3 and len(set(groups)) == 3,
            "study_plan.groups 必须声明三个固定序列")
    base = plan.get("base_config", {})
    require(base.get("history") == 20 and base.get("epochs") == 100,
            "此报告要求 study_plan.base_config 中 history=20、epochs=100")
    require(base.get("aggregation") == "pooled_frames", "此报告要求 pooled_frames 验证选择")
    return plan, seeds, comparisons, ablations, list(dict.fromkeys(comparisons + ablations))


def completed(directory, manifest_name):
    manifest = read_json(directory / manifest_name)
    if manifest.get("status") == "failed":
        raise ValueError(f"已失败：{directory}: {manifest.get('error', '见 manifest')}")
    if manifest.get("status") != "complete" or not (directory / "COMPLETE").is_file():
        raise IncompleteResults(f"尚未完成：{directory}（status={manifest.get('status')}）")
    return manifest


def frozen_configs(study, models):
    source = read_json(study / "frozen_configs.json")
    require(isinstance(source, dict), "frozen_configs.json 必须为对象")
    require(source.get("selection_role", "val") == "val", "冻结配置必须来自验证选择")
    configs = source.get("configs", source.get("selected_configs", source))
    result = {}
    for model in models:
        if model not in configs:
            raise IncompleteResults(f"frozen_configs.json 缺少模型 {model}")
        row = configs[model]
        require(isinstance(row, dict), f"{model} 冻结配置必须为对象")
        result[model] = row.get("config", row)
        require(isinstance(result[model], dict), f"{model} config 必须为对象")
    return result


def training_row(study, model, seed, selected):
    run = study / "formal" / model / f"seed_{seed}"
    state = completed(run, "run_manifest.json")
    config = read_json(run / "resolved_config.json")
    history = read_json(run / "history.json")
    require(config.get("model") == model and config.get("seed") == seed
            and state.get("seed") == seed, f"训练模型/seed 与目录不符：{run}")
    require(config.get("history") == 20 and config.get("epochs") == 100
            and config.get("aggregation") == "pooled_frames", f"训练预算/历史/验证口径不符：{run}")
    require(state.get("selection", {}).get("role") == "val", f"训练未按验证集选取：{run}")
    for key, value in selected.items():
        if key not in RUNTIME_KEYS:
            require(key in config and config[key] == value, f"冻结配置不符：{run}: {key}")
    require(isinstance(history, list) and bool(history), f"训练 history 为空：{run}")
    epochs = [finite(row.get("epoch"), f"{run}: epoch", minimum=1) for row in history]
    require(epochs == sorted(set(epochs)), f"history epoch 重复或无序：{run}")
    for row in history:
        finite(row.get("validation_node_mean_mm"), f"{run}: validation_node_mean_mm", minimum=0)
    best = finite(state.get("best_epoch"), f"{run}: best_epoch", minimum=1)
    require(best in epochs and state.get("current_epoch") == epochs[-1], f"训练 epoch 记录不一致：{run}")
    best_val = finite(state.get("best_validation_node_mean_mm"), f"{run}: best validation", minimum=0)
    selected_val = next(row["validation_node_mean_mm"] for row in history if row["epoch"] == best)
    require(math.isclose(best_val, selected_val, rel_tol=1e-6, abs_tol=1e-8), f"选中 epoch 的验证值不符：{run}")
    wall = finite(state.get("wall_seconds"), f"{run}: wall_seconds", minimum=0)
    params = finite(state.get("parameter_count"), f"{run}: parameter_count", minimum=0)
    trainable = finite(state.get("trainable_parameter_count"), f"{run}: trainable_parameter_count", minimum=0)
    fitted = finite(state.get("fitted_reference_buffer_count"), f"{run}: fitted_reference_buffer_count", minimum=0)
    inclusive = finite(state.get("parameter_count_including_fitted_reference"),
                       f"{run}: parameter_count_including_fitted_reference", minimum=0)
    require(inclusive == params + fitted and trainable <= params, f"拟合参数计数不一致：{run}")
    convergence = state.get("convergence", {})
    flags = []
    if convergence.get("assessment"):
        flags.append(convergence["assessment"])
    else:
        flags.append("convergence_not_recorded")
    if trainable == 0:
        flags.append("closed_form_or_frozen")
    else:
        if epochs[-1] < 100:
            flags.append("stopped_before_budget")
        if best == epochs[-1]:
            flags.append("best_at_last_validation")
    row = dict(model=model, seed=seed, training_run=str(run), wall_seconds=wall,
               parameter_count=params, trainable_parameter_count=trainable, selected_epoch=best,
               fitted_reference_buffer_count=fitted, parameter_count_including_fitted_reference=inclusive,
               dt=finite(config.get("dt"), f"{run}: dt", minimum=1e-12),
               completed_epoch=epochs[-1], validation_node_mean_mm=best_val,
               stop_reason=state.get("stop_reason", "not_recorded"), convergence_flags=";".join(flags),
               recent_relative_improvement=convergence.get("recent_relative_improvement"),
               initialization_seconds=state.get("model_initialization_and_prior_seconds"),
               memory_initialization_seconds=state.get("memory_initialization_seconds"))
    return row, state, history


def latency_rows(study, models, seeds, training):
    jobs = {}
    for path in (study / "jobs").glob("*.json"):
        job = read_json(path)
        if job.get("task") == "latency" and job.get("output"):
            output = Path(job["output"])
            jobs[str(output.resolve() if output.is_absolute() else (study / output).resolve())] = job
    rows, contract, physical_ids, unverifiable = [], None, set(), False
    for model in models:
        paths = list((study / "latency" / model).glob("seed_*.json"))
        if not paths:
            raise IncompleteResults(f"缺少延迟：latency/{model}/seed_<n>.json（允许仅 seed0）")
        for path in sorted(paths):
            result = read_json(path)
            seed = result.get("seed")
            require(type(seed) is int and seed in seeds and path.name == f"seed_{seed}.json"
                    and result.get("model") == model, f"延迟模型/seed 不符：{path}")
            require(result.get("schema") == "modeling_latency_v1"
                    and result.get("history") == 20 and result.get("action_dim") == 4
                    and str(result.get("device", "")).startswith("cuda")
                    and result.get("cuda_synchronized") is True
                    and result.get("eval_mode") is True and result.get("inference_mode") is True
                    and result.get("training_cache_used") is False
                    and "full-window" in result.get("prediction_semantics", ""),
                    f"延迟不是同步 GPU H20 完整窗口协议：{path}")
            current = {k: result.get(k) for k in ("device", "device_name", "torch_version", "cuda_version",
                        "threads", "dtype", "warmup", "repeats", "timing_scope", "prediction_semantics")}
            require(all(v is not None for v in current.values()), f"延迟协议字段不完整：{path}")
            require(contract is None or contract == current, f"不同硬件型号或计时协议：{path}")
            contract = current
            job = jobs.get(str(path.resolve()), {})
            physical = result.get("gpu_uuid", result.get("physical_gpu", job.get("gpu")))
            if physical is None:
                unverifiable = True
            else:
                physical_ids.add(str(physical))
            train = training[(model, seed)]
            require(result.get("selected_epoch") == train["selected_epoch"], f"延迟 checkpoint epoch 不符：{path}")
            expected_checkpoint = Path(train["training_run"]) / "best_eval_model.pt"
            require(Path(result.get("checkpoint", "")).resolve() == expected_checkpoint.resolve(),
                    f"延迟来自其他训练 checkpoint：{path}")
            measurements = result.get("measurements", [])
            require(len(measurements) == 2 and {v.get("batch_size") for v in measurements} == {1, 256},
                    f"延迟需要 B1、B256 两组结果：{path}")
            row = dict(model=model, seed=seed, source=str(path), physical_gpu=physical,
                       device_name=result["device_name"], selected_epoch=result["selected_epoch"])
            for item in measurements:
                batch = item["batch_size"]
                require(item.get("input_shape") == [batch, 20, 4], f"延迟输入形状不符：{path}")
                for key in ("mean_ms", "p50_ms", "p95_ms", "throughput_windows_per_second"):
                    row[f"latency_b{batch}_{key}"] = finite(item.get(key), f"{path}: {key}", minimum=0)
            rows.append(row)
    require(len(physical_ids) <= 1, "延迟测量来自不同物理 GPU，不能作为同卡比较")
    note = ("GPU 型号与计时协议一致；部分记录缺少物理 GPU 标识，无法完全核验同卡。" if unverifiable else
            f"所有已提供延迟记录来自同一物理 GPU（标识 {next(iter(physical_ids))}）。")
    return rows, dict(protocol=contract, physical_gpu_ids=sorted(physical_ids),
                      physical_gpu_verified=not unverifiable, note=note,
                      aggregation="Mean/sample SD of available per-seed latency summaries; missing seeds are not imputed")


def collect(study):
    plan, seeds, comparisons, ablations, models = design(study)
    selected = frozen_configs(study, models)
    frozen = read_json(study / "frozen_configs.json")
    require(frozen.get("repeat_seeds", seeds) == seeds, "冻结 seed 与 study_plan 不一致")
    screening_path = study / "screening_summary.json"
    screening = read_json(screening_path) if screening_path.exists() else None
    if screening is not None:
        require(screening.get("selection_role") == "val", "筛选记录必须来自验证集")
    training, histories, evaluations = {}, {}, {}
    pending = []
    for model in models:
        for seed in seeds:
            try:
                row, state, history = training_row(study, model, seed, selected[model])
                evaluation = study / "evaluations" / model / f"seed_{seed}"
                evaluated = completed(evaluation, "evaluation_manifest.json")
                require(evaluated.get("role") == "test", f"正式汇总要求 test 评估：{evaluation}")
                require(evaluated.get("selected_epoch") == row["selected_epoch"], f"评估选中 epoch 不符：{evaluation}")
                for key in ("dataset_manifest_sha256", "checkpoint_sha256"):
                    require(bool(state.get(key)) and evaluated.get(key) == state[key], f"训练/评估 {key} 不符：{evaluation}")
                require(Path(evaluated.get("run", "")).resolve() == Path(row["training_run"]),
                        f"评估来自其他训练目录：{evaluation}")
                if not (evaluation / "records.json").is_file() or not list(evaluation.glob("*_predictions.npz")):
                    raise IncompleteResults(f"评估缺少 records/预测数组：{evaluation}")
                training[(model, seed)], histories[(model, seed)] = row, history
                evaluations[(model, seed)] = evaluation
            except IncompleteResults as exc:
                pending.append(str(exc))
    if pending:
        raise IncompleteResults(f"尚有 {len(pending)} 个模型/seed 未就绪：\n" + "\n".join(pending))
    latency, latency_info = latency_rows(study, models, seeds, training)
    declared_latency = frozen.get("latency", {})
    if declared_latency.get("physical_gpu") is not None:
        require(not latency_info["physical_gpu_ids"] or
                latency_info["physical_gpu_ids"] == [str(declared_latency["physical_gpu"])],
                "实际延迟 GPU 与冻结配置不一致")
    require(len({row["dt"] for row in training.values()}) == 1, "训练 dt 不一致")
    summaries = {}
    for family, names in (("comparison", comparisons), ("ablation", ablations)):
        result = summarize_seeds([evaluations[(m, s)] for m in names for s in seeds], seeds,
                                 reference="hov", metrics=METRICS, models=names, exploratory_t=True)
        require(result["status"] == "complete", f"{family} 有缺失指标或 seed，停止汇总")
        for row in result["per_seed"]:
            require({v["group"] for v in row["coverage"]} == set(plan["groups"]),
                    f"实际评估序列与固定三序列不符：{row['evaluation']}")
            require(Path(row["evaluation"]) == evaluations[(row["model"], row["seed"])],
                    "评估元数据模型/seed 与目录不符")
        result.update(study=str(study), generated_at=datetime.now(timezone.utc).isoformat(),
                      study_plan=plan, selected_configs={m: selected[m] for m in names},
                      model_labels={m: MODEL_LABELS.get(m, m) for m in names},
                      baseline_sources=str(ROOT / "docs/paper/icra2027/modeling_baselines_sources.md"),
                      frozen_selection=frozen.get("selection"), screening_summary=screening,
                      training_workers=frozen.get("training_workers"), declared_latency=declared_latency,
                      reporting_scope_zh=SCOPE, exploratory_t=True,
                      exploratory_t_caveat="配对 t 检验及其 Holm 校正仅供探索；五个 seed 无法可靠检验差值正态假设，不作为确认性显著性证据。",
                      training=[training[(m, s)] for m in names for s in seeds],
                      latency=[row for row in latency if row["model"] in names], latency_metadata=latency_info)
        summaries[family] = result
    return summaries, histories


def describe(values):
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def fmt_p(value):
    return "—" if value is None else f"{value:.3g}"


def mean_sd(mean, sd, digits=4):
    return f"{fmt(mean, digits)} ± {fmt(sd, digits)}"


def tables(summaries):
    pools, training, latency, aggregate = {}, {}, {}, {}
    for family, result in summaries.items():
        for row in result["per_seed"]:
            pools[(row["model"], row["seed"])] = row
        for row in result["training"]:
            training[(row["model"], row["seed"])] = row
        for row in result["latency"]:
            latency[(row["model"], row["seed"])] = row
        for row in result["summary"]:
            aggregate[(row["model"], row["metric"])] = row
    per_seed, per_sequence, summary = [], [], []
    for key, pool in pools.items():
        row = dict(training[key], model_label=MODEL_LABELS.get(key[0], key[0]), frames=pool["frames"], mask_frames=pool["mask_frames"],
                   evaluation=pool["evaluation"], **{m: pool["metrics"][m] for m in METRICS})
        row.update({k: v for k, v in latency.get(key, {}).items() if k.startswith("latency_")})
        per_seed.append(row)
        for coverage in pool["coverage"]:
            path = Path(pool["evaluation"]) / f"{coverage['group']}_predictions.npz"
            with np.load(path, allow_pickle=False) as arrays:
                metrics = skeleton_metrics(arrays["prediction_mm"], arrays["target_mm"])
                values = dict(mean_node_mm=float(metrics["mean_node_mm"].mean()),
                              node_rmse_mm=float(np.sqrt(np.mean(metrics["node_rmse_mm"] ** 2))),
                              endpoint_mm=float(metrics["endpoint_mm"].mean()),
                              mask_iou=float(arrays["mask_iou"].mean()), mask_dice=float(arrays["mask_dice"].mean()))
            per_sequence.append(dict(model=key[0], model_label=MODEL_LABELS.get(key[0], key[0]), seed=key[1], group=coverage["group"],
                                     frames=coverage["frames"], mask_frames=coverage["mask_frames"],
                                     **values, source=str(path)))
    models = list(dict.fromkeys(key[0] for key in pools))
    for model in models:
        rows = [r for r in per_seed if r["model"] == model]
        latencies = [r for r in latency.values() if r["model"] == model]
        row = dict(model=model, model_label=MODEL_LABELS.get(model, model), families=";".join(f for f, result in summaries.items() if model in result["models"]),
                   n_seeds=len(rows), seeds=json.dumps([r["seed"] for r in rows]),
                   frames_per_seed=rows[0]["frames"], mask_frames_per_seed=rows[0]["mask_frames"],
                   latency_n_seeds=len(latencies), latency_seeds=json.dumps([r["seed"] for r in latencies]))
        for metric in METRICS:
            values = aggregate[(model, metric)]
            row[f"{metric}_mean"], row[f"{metric}_std"] = values["mean"], values["std"]
        for metric in ("wall_seconds", "parameter_count", "trainable_parameter_count", "fitted_reference_buffer_count",
                       "parameter_count_including_fitted_reference", "selected_epoch", "validation_node_mean_mm"):
            row[f"{metric}_mean"], row[f"{metric}_std"] = describe([r[metric] for r in rows])
        for batch in (1, 256):
            for statistic in ("mean_ms", "p50_ms", "p95_ms", "throughput_windows_per_second"):
                metric = f"latency_b{batch}_{statistic}"
                row[f"{metric}_mean"], row[f"{metric}_std"] = describe([r[metric] for r in latencies])
        row["convergence_flags"] = "; ".join(f"seed{r['seed']}:{r['convergence_flags']}" for r in rows)
        summary.append(row)
    return summary, per_seed, per_sequence


def markdown_table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)])


def report(summaries, summary, per_sequence, curves):
    comparison = summaries["comparison"]
    by_model = {row["model"]: row for row in summary}
    hov = by_model["hov"]
    best_name = min((m for m in comparison["models"] if m != "hov"), key=lambda m: by_model[m]["mean_node_mm_mean"])
    best = by_model[best_name]
    display = lambda model: MODEL_LABELS.get(model, model)
    workers = comparison.get("training_workers") or {}
    text = ["# 固定三序列建模改进实验", "",
            f"HOV 节点均距为 **{mean_sd(hov['mean_node_mm_mean'], hov['mean_node_mm_std'])} mm**；"
            f"主比较中节点均距最低的基线为 {display(best_name)}，{mean_sd(best['mean_node_mm_mean'], best['mean_node_mm_std'])} mm。"
            f"均值差（HOV−该基线）为 {hov['mean_node_mm_mean'] - best['mean_node_mm_mean']:+.4f} mm。",
            "", SCOPE, "",
            f"全部 seed：{comparison['seeds']}；H=20、dt={comparison['training'][0]['dt']} s、最多100 epoch。"
            "每个 checkpoint 由 pooled-frame 验证节点均距选取；本表来自固定 test 帧。"
            "节点均距与末端误差为欧氏距离均值；节点 RMSE 为所有节点欧氏误差平方的均值再开方。"
            "IoU/Dice 在实际评分 mask 帧上平均，取值0–1，不是累计像素交并比。", "",
            "标签为二维视觉骨架，采用固定 1.25 px/mm 图像标定；毫米误差反映该标注与标定下的二维形状一致性，"
            "不等同于独立 NDI 测量的空间精度。", "",
            "Chen为方向特征与网络结构适配；Yu为Bézier几何读出与GRU组合，非原论文NODE复现；"
            "Krauss为action-only振子适配。命名与范围依据 modeling_baselines_sources.md。", ""]
    for family, title in (("comparison", "主比较：全部五个 seed 的均值 ± 样本标准差"),
                          ("ablation", "消融：共享参考设计，分别训练")):
        result = summaries[family]
        text += [f"## {title}", "", markdown_table(["模型", *LABELS], [
            [display(m), *(mean_sd(by_model[m][f"{key}_mean"], by_model[m][f"{key}_std"]) for key in METRICS)]
            for m in result["models"]]), ""]
    ablation_names = [m for m in summaries['ablation']['models'] if m != 'hov']
    reductions = [
        f"{display(m)}：{100 * (1 - hov['mean_node_mm_mean'] / by_model[m]['mean_node_mm_mean']):.1f}%"
        for m in ablation_names if by_model[m]['mean_node_mm_mean'] > 0]
    ablation_node_tests = [r for r in summaries['ablation']['paired_tests']['comparisons']
                           if r['metric'] == 'mean_node_mm']
    all_pairs_improved = all(d > 0 for r in ablation_node_tests for d in r['differences'])
    text += ["## 结果解释", "",
             "完整模型相对于各消融的平均节点误差降幅为：" + "；".join(reductions) + "。",
             "完整模型在五个种子上均优于三个消融，支持两个记忆分支在本实验中的贡献。" if all_pairs_improved else
             "各消融的逐种子差异保存在统计JSON中，均值差与运行间波动应共同判断。", "",
             f"HOV存储系数总量为{hov['parameter_count_including_fitted_reference_mean']:.0f}，"
             f"{display(best_name)}为{best['parameter_count_including_fitted_reference_mean']:.0f}；"
             f"HOV参数量约为后者的{100 * hov['parameter_count_including_fitted_reference_mean'] / best['parameter_count_including_fitted_reference_mean']:.2f}%。"
             "本轮体现了模型规模与预测精度的取舍，分序列结果进一步显示不同运动模式下的差异。", "",
             f"HOV完整窗口B1推理中位耗时{hov['latency_b1_p50_ms_mean']:.3f} ms，"
             f"{display(best_name)}为{best['latency_b1_p50_ms_mean']:.3f} ms；"
             "参数量与实际运行时间需要分别评价。", ""]
    text += ["## 配对统计的解释边界", "",
             "参考为 HOV，差值定义为模型−HOV。五个非零配对的双侧精确 Wilcoxon 最小 p=0.0625；"
             "存在零差值时下界更大，因此本设计无法在0.05水平获得该精确检验显著性。"
             "主比较与消融各自对全部模型×五指标进行 Holm 校正。配对 t 检验显式启用 exploratory_t=True，"
             "另成 Holm 家族；五个 seed 下正态假设难以核验，其 p 值与未调整95%区间仅供探索。"
             "统计单位为固定划分上的配对 training_seed。下表仅展示主指标，其余四指标完整统计见两个 JSON。", ""]
    tests = []
    for family, result in summaries.items():
        for row in result["paired_tests"]["comparisons"]:
            if row["metric"] != "mean_node_mm":
                continue
            w, t = row.get("wilcoxon") or {}, row.get("t_test") or {}
            tests.append([family, display(row["model"]), fmt(row["mean_difference"]), fmt_p(w.get("p_value")),
                          fmt_p(w.get("p_holm")), fmt_p(t.get("p_value")), fmt_p(t.get("p_holm")), t.get("status", "—")])
    text += [markdown_table(["家族", "模型", "Δ节点均距/mm", "Wilcoxon p", "Holm p", "探索t p", "探索t Holm p", "t状态"], tests), "",
             "## 分序列检查", "", "以下对各序列分别计算五个 seed 的节点均距均值与样本标准差；总表仍按帧合并，不平均这三行。", ""]
    groups = comparison["study_plan"]["groups"]
    text += [markdown_table(["序列", "HOV/mm", f"{display(best_name)}/mm"], [
        [group, *(mean_sd(*describe([r["mean_node_mm"] for r in per_sequence if r["model"] == m and r["group"] == group]))
                  for m in ("hov", best_name))] for group in groups]), "",
        "## 训练成本与收敛", "", "时间为每次训练 wall time（包含初始化/prior与验证）。"
        f"冻结并发安排为 GPU {workers.get('gpus', '未记录')}，每GPU {workers.get('workers_per_gpu', '未记录')} 个worker；"
        "时间受并发资源竞争影响，不能解释为独占GPU的训练速度。"
        "可训练计数为优化器更新的参数；存储总量包括消融中停用的参数及保存为buffer的训练拟合参考系数，其他固定几何元数据不计入。"
        "选中 epoch 是验证最优位置；短预算曲线平坦或波动不等同于充分收敛。", "",
        markdown_table(["模型", "训练秒均值±SD", "可训练参数", "拟合参考buffer", "存储系数总量", "选中epoch均值±SD", "仍改善/末次最优的次数"], [
            [display(r["model"]), mean_sd(r["wall_seconds_mean"], r["wall_seconds_std"], 2),
             fmt(r["trainable_parameter_count_mean"], 0), fmt(r["fitted_reference_buffer_count_mean"], 0),
             fmt(r["parameter_count_including_fitted_reference_mean"], 0), mean_sd(r["selected_epoch_mean"], r["selected_epoch_std"], 1),
             f"{r['convergence_flags'].count('still_improving')}/5；{r['convergence_flags'].count('best_at_last_validation')}/5"] for r in summary]), "",
        "逐seed收敛标志保存在 per_seed.csv。线性回归的225个系数由闭式求解确定。", ""]
    if curves:
        text += ["![全部预定seed的验证学习曲线](learning_curves.png)", "", "每条线对应一个预定 seed 的实际验证记录，未插值或剔除轨迹。", ""]
    text += ["## GPU 完整窗口延迟", "", comparison["latency_metadata"]["note"], "",
             "冻结延迟安排：" + json.dumps(comparison.get("declared_latency", {}), ensure_ascii=False) + "。", "",
             "H20完整窗口包含 burn-in、算子和几何解码及毫米输出转换；输入驻留GPU，计时同步。"
             "B256为整批延迟，不是单样本或增量状态步延迟。表中p50/p95为可用seed的各自分位数之均值；"
             "若仅seed0则直接报告该次测量，不构造五seed延迟SD。", "",
             markdown_table(["模型", "延迟seed", "B1 p50/ms", "B1 p95/ms", "B256 p50/ms", "B256 p95/ms", "B256 windows/s"], [
                 [display(r["model"]), r["latency_seeds"], *(fmt(r[f"latency_b{b}_{p}_ms_mean"]) for b, p in ((1, "p50"), (1, "p95"), (256, "p50"), (256, "p95"))),
                  fmt(r["latency_b256_throughput_windows_per_second_mean"], 1)] for r in summary]), "",
             "## 冻结配置与结论范围", "",
             "完整选中配置保存在 comparison.json / ablation.json 的 selected_configs，来源为 frozen_configs.json。"
             "以下列出主要拟合设置；CSV保留逐seed成本、选中epoch、评分帧数和逐序列指标。", ""]
    keys = ("lr", "hidden", "chen_hidden", "latent", "n_play", "n_maxwell", "ridge", "prior_steps",
            "calibrate_reference", "calibrate_length_directions", "reference_pair_interactions",
            "reference_pair_length_interactions", "memory_readout_init", "memory_ridge")
    configs = {m: c for result in summaries.values() for m, c in result["selected_configs"].items()}
    used_keys = {"chen_direction": ("lr", "chen_hidden"), "bezier_gru": ("lr", "hidden"),
                 "oscillator": ("lr", "latent", "force_hidden"), "koopman": ("lr", "latent", "force_hidden"),
                 "pcc": ("lr", "hidden"), "mlp": ("lr", "hidden"), "linear": ("ridge",)}
    text += [markdown_table(["模型", "冻结设置"], [[display(m), "; ".join(f"{k}={configs[m][k]}" for k in used_keys.get(m, keys) if k in configs[m])] for m in by_model]), "",
             "静态参考拟合、参考系数校正、静态压力对耦合和记忆读出初始化分别按上述开关记录；"
             "静态消融保留相同参考设置。预测精度、训练初始化成本与完整窗口延迟分别报告。", ""]
    screening = comparison.get("screening_summary") or {}
    originals = [r for r in screening.get("candidates", [])
                 if r.get("model") == "hov" and r.get("candidate_plan") == "pilot_plan.json"]
    final = screening.get("selected", {}).get("hov")
    if len(originals) == 1 and final:
        original = originals[0]
        for item in (original, final):
            finite(item.get("validation_node_mean_mm"), "筛选 HOV 验证值", minimum=0)
        text += [f"验证开发（seed={screening.get('screening_seed')}）：原HOV节点均距 "
                 f"{original['validation_node_mean_mm']:.4f} mm → 最终冻结HOV {final['validation_node_mean_mm']:.4f} mm。"
                 "这是同一验证集上的配置选择记录，不能代替正式五seed测试证据。", ""]
    text += [
             "当前结果仅支持本固定划分和预算下的比较。标签来自同源图像标注，几何与mask指标相关。"
             "每个冻结模型均汇总预定五次重复；后续若需泛化结论，应采用独立新序列/采集日验证。", ""]
    return "\n".join(text)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def learning_curves(path, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(m for m, _ in histories))
    fig, axes = plt.subplots(math.ceil(len(models) / 3), 3, figsize=(15, 3.2 * math.ceil(len(models) / 3)), squeeze=False)
    for axis, model in zip(axes.flat, models):
        for (name, seed), history in histories.items():
            if name == model:
                axis.plot([r["epoch"] for r in history], [r["validation_node_mean_mm"] for r in history],
                          label=f"seed {seed}", linewidth=1,
                          marker="o" if len(history) == 1 else None)
        axis.set(title=model, xlabel="Epoch", ylabel="Pooled validation node mean (mm)")
        if model == "linear":
            axis.set_xlabel("Fit completion")
            axis.text(.04,.92,"Closed-form fit",transform=axis.transAxes,va="top",fontsize=9)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(models):]:
        axis.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate(study, *, curves=False):
    summaries, histories = collect(study)
    summary, per_seed, per_sequence = tables(summaries)
    # Stage everything first: incomplete inputs cannot create a partial report.
    with tempfile.TemporaryDirectory(prefix=".modeling-report-", dir=study) as temporary:
        staging = Path(temporary)
        for family, result in summaries.items():
            (staging / f"{family}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name, rows in (("summary", summary), ("per_seed", per_seed), ("per_sequence", per_sequence)):
            write_csv(staging / f"{name}.csv", rows)
        if curves:
            learning_curves(staging / "learning_curves.png", histories)
        (staging / "report.md").write_text(report(summaries, summary, per_sequence, curves), encoding="utf-8")
        output = study / "results"
        output.mkdir(exist_ok=True)
        for path in staging.iterdir():
            os.replace(path, output / path.name)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=0, help="最多等待的秒数；默认缺失即失败")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--learning-curves", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or args.wait_seconds < 0 or not 0 < args.poll_seconds <= 60:
        parser.error("wait-seconds 必须非负有限；poll-seconds 必须在(0,60]")
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            output = generate(args.study.resolve(), curves=args.learning_curves)
            print(f"完整汇总已写入：{output}")
            return 0
        except IncompleteResults as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                parser.exit(2, f"结果未就绪，未写入报告：{exc}\n")
            print(f"等待结果：{exc}", file=sys.stderr, flush=True)
            time.sleep(min(args.poll_seconds, remaining))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            parser.exit(2, f"结果校验失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
