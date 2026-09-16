#!/usr/bin/env python3
"""Paper efficiency evidence from frozen logs and bounded CPU inference.

Reads the original three-sequence study and completed window-MLP runs.
Writes only efficiency.{json,md} and its own analysis directory. No training.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy import stats
from src.benchmarks.modeling_models import make_model, NeuralShape

STUDY = ROOT / "workspace/runs/training/modeling_three_seq_20260913_001928"
PLUGIN = ROOT / "workspace/runs/analysis/modeling_mechanisms_20260913_001/plugin"
REPORT = ROOT / "workspace/reports/modeling_paper_revision_20260913_002/efficiency"
OUTPUT = ROOT / "workspace/runs/analysis/modeling_paper_revision_20260913_002/efficiency"
SEEDS = list(range(5))
MODELS = ["hov", "mlp", "chen_direction", "oscillator", "window_mlp"]
LABELS = {"hov": "HOV", "mlp": "静态 MLP（正式）", "chen_direction": "Chen 方向网络",
          "oscillator": "Krauss 振子适配", "window_mlp": "窗口 MLP（H20）"}
MODE_LABELS = {**LABELS, "hov": "HOV 重算 H20 窗口", "hov_cached": "HOV 缓存状态一步"}


def read(path):
    return json.loads(Path(path).read_text())


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.


def aggregate(rows, keys, fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    output = []
    for group, records in grouped.items():
        result = dict(zip(keys, group))
        result["n_seeds"] = len(records)
        for field in fields:
            values = [r[field] for r in records if r[field] is not None]
            avg, sd = mean_sd(values) if values else (None, None)
            result[field + "_mean"] = avg
            result[field + "_sd"] = sd
        output.append(result)
    return output


def training_evidence():
    curves, summary, settings, checks = [], [], [], []
    old = read(PLUGIN / "convergence.json")
    old_index = {(r["model"], r["seed"]): r for r in old["summary"]}
    normalization = read(PLUGIN / "normalization.json")
    old_plugin = read(PLUGIN / "plugin_results.json")
    assert (old_plugin["train_windows"], old_plugin["val_windows"], old_plugin["test_windows"]) == (8988, 2958, 2958)
    manifest = read(STUDY / "data/dataset_manifest.json")
    assert manifest["H"] == 20 and manifest["dt"] == .2
    for model in MODELS:
        for seed in SEEDS:
            if model == "window_mlp":
                folder = PLUGIN / "formal/mlp/window" / f"seed_{seed}"
                meta = read(folder / "fit.json")
                cfg = meta["config"]
                history = read(folder / "history.json")
                best, best_epoch = meta["val_mean_node_mm"], meta["best_epoch"]
                init = memory = None
                total, optimizer = meta["fit_seconds"], meta["optimizer_seconds"]
                parameters = meta["parameter_count"]
                batch = cfg["batch_size"]
                device = "CPU 双线程（原插件任务）"
                scope = "优化、验证与恢复最佳权重；不含模型/优化器构建、特征预处理、磁盘写入"
                architecture = f"80→{cfg['width']}→{cfg['width']}→45，Tanh，按训练集逐特征标准化"
                train_key = "val_mean_node_mm"
                prefit_steps = 0
            else:
                folder = STUDY / "formal" / model / f"seed_{seed}"
                meta = read(folder / "run_manifest.json")
                cfg = read(folder / "resolved_config.json")
                history = read(folder / "history.json")
                assert meta["status"] == "complete" and cfg["seed"] == seed
                assert meta["supervised_windows"] == 8988 and meta["validation_windows"] == 2958
                best, best_epoch = meta["best_validation_node_mean_mm"], meta["best_epoch"]
                init = meta.get("model_initialization_and_prior_seconds", 0.)
                memory = meta.get("memory_initialization_seconds", 0.)
                total = meta["wall_seconds"]
                optimizer = history[-1]["cumulative_training_seconds"]
                parameters = meta["parameter_count_including_fitted_reference"]
                batch = cfg["batch_size"]
                device = "CPU 预拟合 + GPU 正式优化（原并发任务）" if model == "hov" else "GPU 正式优化（原并发任务）"
                scope = "记录任务墙钟：数据读取/窗口构建、模型初始化/预拟合、优化、验证与文件写入"
                architecture = {
                    "hov": "4 输入；路径 4×2、时间 4×6；16 维几何读出；含参考拟合系数",
                    "mlp": f"4→{cfg['hidden']}→{cfg['hidden']}→45，Tanh，当前输入",
                    "chen_direction": f"8→{cfg['chen_hidden']}×4 隐藏层→45，ReLU，当前输入及最近方向",
                    "oscillator": f"4→{cfg['force_hidden']}→{cfg['force_hidden']}→{cfg['latent']} 力网络，{cfg['latent']} 维振子，32→45 读出",
                }[model]
                train_key = "validation_node_mean_mm"
                prefit_steps = 750 if model == "hov" else 0
                ref = old_index[model, seed]
                assert abs(history[0][train_key] - ref["epoch1_val_node_mm"]) < 1e-10
                assert abs(best - ref["best_val_node_mm"]) < 1e-10
                checks.append(dict(check="original_logs_match_prior_convergence", model=model, seed=seed, passed=True))
            assert history[0]["epoch"] == 1 and history[-1]["epoch"] == 100
            assert cfg["epochs"] == 100 and batch in (256, 512)
            assert abs(min(r[train_key] for r in history)-best) < 1e-8
            checkpoints = {r["epoch"]: r for r in history}
            first = history[0][train_key]
            within = next(r for r in history if r[train_key] <= 1.05 * best)
            near = [r for r in history if r[train_key] <= 1.10 * best]
            end_gap = (checkpoints[80][train_key] - best) / checkpoints[80][train_key] * 100
            residual = total - optimizer - (init or 0) - (memory or 0)
            assert residual >= -1e-5
            row = dict(model=model, label=LABELS[model], seed=seed, parameters=parameters,
                batch_size=batch, minibatch_steps_per_epoch=math.ceil(8988 / batch),
                prefitting_fullbatch_steps=prefit_steps,
                supervised_presentations_through_epoch1=(prefit_steps + 1) * 8988,
                train_windows=8988, val_windows=2958, epochs=100, device=device,
                epoch1_val_mm=first, epoch5_val_mm=checkpoints[5][train_key],
                epoch10_val_mm=checkpoints[10][train_key], best_val_mm=best,
                epoch1_minus_best_mm=first-best,
                epoch1_excess_over_best_pct=100 * (first-best) / best,
                epoch1_to_best_reduction_pct=100 * (first-best) / first,
                epoch80_to_best_reduction_pct=end_gap, best_epoch=best_epoch,
                first_recorded_epoch_within5pct_best=within["epoch"],
                first_recorded_epoch_within10pct_best=near[0]["epoch"],
                recorded_seconds_to_within5pct_best=within["elapsed_seconds"],
                recorded_seconds_to_epoch1=history[0]["elapsed_seconds"],
                prior_and_model_init_seconds=init, memory_init_seconds=memory,
                optimizer_seconds=optimizer, other_recorded_seconds=residual,
                recorded_total_seconds=total, recorded_scope=scope,
                average_minibatch_epoch_seconds=optimizer/100,
                full_offline_pipeline_seconds=None,
                source=relative(folder))
            summary.append(row)
            for item in history:
                curves.append(dict(model=model, label=LABELS[model], seed=seed,
                    epoch=item["epoch"], val_mm=item[train_key],
                    excess_over_best_pct=100*(item[train_key]-best)/best,
                    optimizer_seconds=item["cumulative_training_seconds"],
                    recorded_seconds=item["elapsed_seconds"],
                    formal_minibatch_updates=item["epoch"]*math.ceil(8988/batch),
                    prefit_fullbatch_updates=prefit_steps,
                    device=device, source=relative(folder / "history.json")))
            if seed == 0:
                settings.append(dict(model=model, label=LABELS[model], architecture=architecture,
                    parameters=parameters, batch_size=batch, initial_lr=cfg["lr"],
                    epochs=100, updates_per_epoch=math.ceil(8988/batch), prefit_fullbatch_steps=prefit_steps,
                    device=device, history=20, nominal_dt_s=.2,
                    scheduler_patience_checks=3 if model == "window_mlp" else 4,
                    validation_batch_size=1024 if model == "window_mlp" else batch,
                    validation_interval="1, 5, 10, …, 100", recorded_time_scope=scope,
                    source=relative(folder / ("fit.json" if model == "window_mlp" else "resolved_config.json"))))
    metrics = [k for k,v in summary[0].items() if isinstance(v, (int,float)) and k != "seed"]
    metrics += ["full_offline_pipeline_seconds"]
    averaged = aggregate(summary, ["model", "label", "device", "recorded_scope"], metrics)
    averaged_curves = aggregate(curves, ["model", "label", "epoch", "device"],
        ["val_mm", "excess_over_best_pct", "recorded_seconds", "optimizer_seconds"])
    stage_names = {"prefitted_reference": "参考预拟合后", "epoch0_after_memory_initialization": "记忆初始化后（epoch 0）"}
    stages = [dict(seed=r["seed"], stage=stage_names[r["stage"]], val_mm=r["mean_node_mm"])
              for r in old["epoch0_stages"] if r["stage"] in stage_names]
    stages += [dict(seed=r["seed"], stage=stage, val_mm=r[key]) for r in summary if r["model"] == "hov"
               for stage,key in [("正式 epoch 1", "epoch1_val_mm"), ("最佳验证 checkpoint", "best_val_mm")]]
    tuning = []
    screen = read(STUDY / "screening_summary.json")["candidates"]
    for model in MODELS:
        if model == "window_mlp":
            candidates = [r for r in read(PLUGIN / "screening_results.json") if r["family"] == "mlp" and r["variant"] == "window"]
            total = sum(r["fit_seconds"] for r in candidates)
            source = relative(PLUGIN / "screening_results.json")
            scope = "窗口变体 4 候选拟合时长之和；CPU；预处理单独记录且共享"
        else:
            candidates = [r for r in screen if r["model"] == model]
            total = sum(r["wall_seconds"] for r in candidates)
            source = relative(STUDY / "screening_summary.json")
            scope = "本轮选择汇总中同名模型的候选任务墙钟之和；含其预拟合；并发任务时长不能视为项目历时"
        formal = [r["recorded_total_seconds"] for r in summary if r["model"] == model]
        tuning.append(dict(model=model,label=LABELS[model],recorded_candidates=len(candidates),
            candidate_run_seconds_sum=total, five_formal_run_seconds_sum=sum(formal),
            known_candidate_plus_five_run_seconds_sum=total+sum(formal),
            full_research_pipeline_seconds=None, scope=scope, source=source))
    paired=[]
    hov=[r for r in summary if r["model"]=="hov"]
    for model in MODELS[1:]:
        baseline=[r for r in summary if r["model"]==model]
        differences=np.array([b["epoch1_val_mm"]-h["epoch1_val_mm"] for h,b in zip(hov,baseline)])
        test=stats.wilcoxon(differences,alternative="two-sided",method="exact")
        paired.append(dict(model=model,label=LABELS[model],metric="epoch1_val_mm",
            baseline_minus_hov_mean_mm=float(differences.mean()),
            positive_pairs=int((differences>0).sum()),n=5,wilcoxon_two_sided_exact_p=float(test.pvalue),
            comparison_scope="整套训练流程到正式 epoch 1；初始化与batch不同；不是等计算量比较"))
    sorted_p=sorted(enumerate(paired),key=lambda item:item[1]["wilcoxon_two_sided_exact_p"])
    corrected=0.
    for rank,(_,row) in enumerate(sorted_p):
        corrected=max(corrected,min(1.,(len(paired)-rank)*row["wilcoxon_two_sided_exact_p"]))
        row["holm_p"]=corrected
    return dict(summary=averaged,seeds=summary,curves=curves,curve_summary=averaged_curves,
        settings=settings,initialization_stages=stages,
        initialization_summary=aggregate(stages,["stage"],["val_mm"]),
        tuning_cost=tuning,early_paired_tests=paired,validation=checks,
        window_shared_feature_build_seconds=normalization["feature_build_seconds_train_val"],
        other_unrecorded_costs=["原图采集及分割/骨架提取", "全项目模型开发与更早调参", "所有方法的测试/部署开销",
                              "窗口 MLP 数据读取、模型/优化器构建、标准化及保存总时长未独立记录"])


def hardware_snapshot():
    cpu="unknown"
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            cpu=line.split(":",1)[1].strip();break
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),cpu=cpu,
        logical_cpus=os.cpu_count(),load_average_1_5_15=list(os.getloadavg()),
        affinity_cpus=len(os.sched_getaffinity(0)),platform=platform.platform(),
        python=platform.python_version(),torch=torch.__version__,
        torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads())


class ForwardMM:
    def __init__(self, model, center, scale, mean=None, std=None):
        self.model=model.eval()
        self.center=torch.tensor(center,dtype=torch.float32)
        self.scale=float(scale)
        self.mean=None if mean is None else torch.tensor(mean,dtype=torch.float32).reshape(1,20,4)
        self.std=None if std is None else torch.tensor(std,dtype=torch.float32).reshape(1,20,4)

    def __call__(self, window):
        if self.mean is not None:
            window=(window-self.mean)/self.std
        return self.model(window)*self.scale+self.center


def load_forward(model,seed):
    if model=="window_mlp":
        ck=torch.load(PLUGIN/f"formal/mlp/window/seed_{seed}/model.pt",map_location="cpu",weights_only=True)
        norm=read(PLUGIN/"normalization.json")
        # NeuralShape(window_mlp) and the trained FeatureMLP use the identical
        # named Sequential network. Strict loading and saved-output agreement
        # below guard against an architecture or normalization mismatch.
        instance=NeuralShape("window_mlp",20,hidden=ck["config"]["width"])
        instance.load_state_dict(ck["state_dict"],strict=True)
        assert sum(p.numel() for p in instance.parameters())==12269
        return ForwardMM(instance,norm["target_center_xyz"],norm["target_scale"],
                         norm["features"]["window"]["mean"],norm["features"]["window"]["std"])
    ck=torch.load(STUDY/f"formal/{model}/seed_{seed}/best_eval_model.pt",map_location="cpu",weights_only=True)
    instance,_=make_model(ck["model"],ck["config"],normalization=(ck["center"],ck["scale"]),geometry_config=ck["geometry_config"])
    instance.load_state_dict(ck["state_dict"],strict=True)
    if model=="mlp":
        assert sum(p.numel() for p in instance.parameters())==22957
    return ForwardMM(instance,ck["center"],ck["scale"])


def inference_evidence(warmup,runs):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start_hardware=hardware_snapshot()
    path=STUDY/"data/test/seq_20260819_172644.npz"
    with np.load(path,allow_pickle=False) as data:
        commands=data["actions"].astype(np.float32)
    assert len(commands)>=warmup+runs+20
    # Contiguous time samples with no sequence transition. All tensors and
    # window slices are already resident; only model transforms are timed.
    windows=[torch.from_numpy(commands[i:i+20]).unsqueeze(0) for i in range(warmup+runs+1)]
    raw,per_seed,checks=[],[],[]
    for seed in SEEDS:
        forward={model:load_forward(model,seed) for model in MODELS}
        with torch.inference_mode():
            small=torch.cat(windows[:32])
            for model,call in forward.items():
                actual=call(small).numpy()
                if model=="window_mlp":
                    prediction_file=PLUGIN/f"formal/mlp/window/seed_{seed}/test_predictions.npz"
                    with np.load(prediction_file,allow_pickle=False) as d:
                        expected=d["prediction"][:32]
                else:
                    prediction_file=STUDY/f"evaluations/{model}/seed_{seed}/seq_20260819_172644_predictions.npz"
                    with np.load(prediction_file,allow_pickle=False) as d:
                        expected=d["prediction_mm"][:32]
                difference=float(np.max(np.abs(actual-expected)))
                assert difference<.003,(model,seed,difference)
                checks.append(dict(check="restored_forward_agrees_saved_prediction",model=model,seed=seed,
                    n_windows=32,max_abs_coordinate_mm=difference,passed=True,source=relative(prediction_file)))
            hov=forward["hov"]
            core=hov.model.core
            state=core(windows[0])["latent_z"]
            # Exact one-step continuation equals a 21-command forward pass
            # with the same initialization, rather than a newly reset H20.
            next_step=core.step_state(windows[1][:,-1],state)
            continuation=core(torch.from_numpy(commands[:21]).unsqueeze(0))
            error=float((next_step["skeleton"]-continuation["skeleton"]).abs().max()*hov.scale)
            assert error<1e-4,error
            checks.append(dict(check="cached_continuation_matches_full_prefix",model="hov",seed=seed,
                n_windows=1,max_abs_coordinate_mm=error,passed=True))
            random=np.random.default_rng(9090+seed)
            modes=[*MODELS,"hov_cached"]
            timings={mode:[] for mode in modes}
            for i in range(warmup+runs):
                window=windows[i+1]
                for mode in random.permutation(modes):
                    before=time.perf_counter_ns()
                    if mode=="hov_cached":
                        item=core.step_state(window[:,-1],state)
                        state=item["latent_z"]
                        result=item["skeleton"]*hov.scale+hov.center
                    else:
                        result=forward[mode](window)
                    elapsed=(time.perf_counter_ns()-before)/1e6
                    if i>=warmup:
                        timings[mode].append(elapsed)
                        raw.append(dict(mode=mode,label=MODE_LABELS[mode],seed=seed,run=i-warmup,
                            latency_ms=elapsed,command_index=i+20))
                assert result.shape==(1,15,3)
            for mode,values in timings.items():
                p50,p95=np.quantile(values,[.5,.95])
                per_seed.append(dict(mode=mode,label=MODE_LABELS[mode],seed=seed,n_calls=len(values),
                    p50_ms=float(p50),p95_ms=float(p95),mean_ms=float(np.mean(values)),
                    p50_latency_reciprocal_hz=float(1000/p50),p95_latency_reciprocal_hz=float(1000/p95),
                    p50_fraction_of_50hz_period_pct=float(p50/20*100),
                    p95_fraction_of_50hz_period_pct=float(p95/20*100),
                    p50_fraction_of_100hz_period_pct=float(p50/10*100),
                    p95_fraction_of_100hz_period_pct=float(p95/10*100)))
        print(f"timing seed {seed} complete",flush=True)
    summary=aggregate(per_seed,["mode","label"],["p50_ms","p95_ms","mean_ms",
        "p50_latency_reciprocal_hz","p95_latency_reciprocal_hz",
        "p50_fraction_of_50hz_period_pct","p95_fraction_of_50hz_period_pct",
        "p50_fraction_of_100hz_period_pct","p95_fraction_of_100hz_period_pct"])
    for row in summary:
        values=[r["latency_ms"] for r in raw if r["mode"]==row["mode"]]
        row["pooled_p50_ms"],row["pooled_p95_ms"]=map(float,np.quantile(values,[.5,.95]))
        row["pooled_n_calls"]=len(values)
        row["reciprocal_of_mean_p50_hz"]=1000/row["p50_ms_mean"]
        row["reciprocal_of_mean_p95_hz"]=1000/row["p95_ms_mean"]
    return dict(summary=summary,seeds=per_seed,raw=raw,validation=checks,
        hardware_before=start_hardware,hardware_after=hardware_snapshot(),
        protocol=dict(device="CPU",threads=1,interop_threads=1,dtype="float32",batch=1,seeds=SEEDS,
            warmup_per_seed_per_mode=warmup,runs_per_seed_per_mode=runs,
            inputs="内存中的真实连续归一化压力窗口 1×20×4；返回 1×15×3 毫米坐标",
            included="模型必要的方向/历史处理、窗口标准化（适用时）、前向及坐标反归一化；缓存模式含状态更新",
            excluded="磁盘I/O、图像获取、分割/骨架提取、主机设备复制、窗口队列维护、通信、规划/优化控制器、参数学习",
            cached_state="首个H20初始化不计时；后续状态连续传递。用于递推计算量，未重评估长期递推精度。",
            order="每个时间步固定随机种子随机排列各方法，分散同机并发负载的顺序影响",
            concurrency="同机存在其它分析任务；无绑核/CPU隔离/实时调度保证",
            frequency="1000 / 延迟毫秒是每次模型求值的串行延迟倒数；50/100Hz为假设时间预算，不是闭环实测频率",
            physics_dt="权重的模型时间步仍为0.2s；运算频率与模型有效采样率、控制稳定性不同",
            source=relative(path)))


def chart(id,title,rows,x,y,kind="bar",color=None,unit="mm",description=""):
    return dict(id=id,title=title,kind=kind,x=x,y=y,color=color,unit=unit,description=description,rows=rows)


def assemble(training,inference):
    conv={r["model"]:r for r in training["summary"]}
    timing={r["mode"]:r for r in inference["summary"]}
    h=conv["hov"];cached=timing["hov_cached"];window=timing["hov"]
    stages={r["stage"]:r["val_mm_mean"] for r in training["initialization_summary"]}
    early=(f"在三条序列固定划分及五次重复实验中，HOV 在完成参考形态与记忆读出的离线初始化后，"
           f"验证节点误差为 {stages['记忆初始化后（epoch 0）']:.3f} mm；首个联合优化 epoch 后为 "
           f"{h['epoch1_val_mm_mean']:.3f}±{h['epoch1_val_mm_sd']:.3f} mm，100 epoch 内所选最佳模型为 "
           f"{h['best_val_mm_mean']:.3f}±{h['best_val_mm_sd']:.3f} mm。首轮至最佳的误差下降为 "
           f"{h['epoch1_to_best_reduction_pct_mean']:.1f}%，低于静态 MLP、Chen 方向网络、Krauss 振子适配和窗口 MLP "
           f"对应的 {conv['mlp']['epoch1_to_best_reduction_pct_mean']:.1f}%、"
           f"{conv['chen_direction']['epoch1_to_best_reduction_pct_mean']:.1f}%、"
           f"{conv['oscillator']['epoch1_to_best_reduction_pct_mean']:.1f}% 和 "
           f"{conv['window_mlp']['epoch1_to_best_reduction_pct_mean']:.1f}%。"
           "这说明结构化模型配合分阶段拟合，能够在联合优化早期建立较准确的形态预测，后续训练主要进一步细化误差。")
    runtime=(f"在同机 CPU 单线程、batch size 为1的条件下，使用五个训练种子分别预热 "
             f"{inference['protocol']['warmup_per_seed_per_mode']} 次并计时 {inference['protocol']['runs_per_seed_per_mode']} 次，"
             f"HOV 缓存状态单步预测的 p50/p95 延迟分别为 "
             f"{cached['p50_ms_mean']:.3f}/{cached['p95_ms_mean']:.3f} ms（各种子分位数的均值），"
             f"重算20步历史窗口则为 {window['p50_ms_mean']:.3f}/{window['p95_ms_mean']:.3f} ms。"
             f"以平均 p95 延迟估计，一次递推模型求值占假设50 Hz更新周期的 "
             f"{cached['p95_fraction_of_50hz_period_pct_mean']:.1f}%，占100 Hz周期的 "
             f"{cached['p95_fraction_of_100hz_period_pct_mean']:.1f}%，为反馈计算保留了时间余量。"
             "该计时包含状态更新与骨架输出，测量时存在同机并发任务；闭环频率仍需结合视觉、通信及规划耗时实测。")
    definitions=dict(
        validation_metric="每帧15节点欧氏距离平均后，汇总全部2958个验证目标帧；mm；不按序列等权平均",
        repetitions="固定种子0–4、同三序列train/val/test时序6:2:2；均值±样本SD，ddof=1",
        first_gap="首轮距最佳：E1−Ebest；相对最佳超额=(E1−Ebest)/Ebest；后续下降率=(E1−Ebest)/E1",
        best="本次100 epoch内记录的最低验证误差；不是已知全局最优或无限训练的收敛极限",
        first_threshold="首次记录到Eepoch≤1.05×Ebest的epoch；验证只在1/5/10/…/100发生，非精确首次越过时刻",
        optimizer_efficiency="只支持联合优化早期精度及初始化有效性；每epoch的更新次数、初始化、批次及硬件必须同时给出",
        sample_efficiency="需固定计算/选择协议下的训练数据量—误差曲线；现有全训练集预拟合与单一数据量不能证明样本效率",
        physical_law_efficiency="早期低误差可与结构先验相容，不能独立证明学到了真实材料规律或唯一物理参数",
        timing_aggregation="主表先对每个seed的200次求p50/p95，再报告5个分位数的均值±SD；另存1000次池化分位数，不混淆",
        frequency="f_latency=1000/L_ms；周期占比=L_ms/(1000/f_target)×100%；是模型计算预算，不是实际闭环控制频率",
        formal_vs_window="静态MLP=原正式22957参数；窗口MLP=旧插件window分支12269参数。插件base的7405参数MLP不在效率主比较中",
        full_cost="原正式总时长包含其预拟合和任务内开销；窗口只有拟合区段完整时长。调参另列已记录任务时间之和；全离线管线耗时未知，禁止补零",
        architecture_scope="Chen和Krauss是本数据/骨架监督下的实现适配；计时结论针对当前实现及选定配置")
    charts=[
        chart("efficiency_validation_curves","各方法早期与后续验证精度",training["curve_summary"],"epoch","val_mm_mean","line","label",
              description="五种模型、各五种子均值；首个HOV正式epoch已完成全训练集初始化；窗口MLP batch512，其余batch256。"),
        chart("efficiency_validation_early","前20个正式epoch验证精度",[r for r in training["curve_summary"] if r["epoch"]<=20],"epoch","val_mm_mean","line","label"),
        chart("efficiency_remaining_gap","首轮至各自最佳的后续下降比例",training["summary"],"label","epoch1_to_best_reduction_pct_mean",unit="%"),
        chart("efficiency_relative_best_curve","各轮相对自身最佳的误差超额",training["curve_summary"],"epoch","excess_over_best_pct_mean","line","label",unit="%"),
        chart("efficiency_hov_stages","HOV初始化及正式优化阶段",training["initialization_summary"],"stage","val_mm_mean"),
        chart("efficiency_gpu_recorded_wall","原GPU任务：包含预拟合的记录墙钟时间",[r for r in training["curve_summary"] if r["model"]!="window_mlp"],"recorded_seconds_mean","val_mm_mean","line","label",
              description="原并发任务；HOV参考预拟合在CPU执行且已计入。不得把窗口CPU拟合时长接入这条共同时间轴。"),
        chart("efficiency_window_cpu_wall","窗口MLP原CPU任务：拟合区段时间",[r for r in training["curve_summary"] if r["model"]=="window_mlp"],"recorded_seconds_mean","val_mm_mean","line","label"),
        chart("efficiency_cpu_latency","同机CPU单线程B1前向延迟",inference["summary"],"label","p50_ms_mean",unit="ms"),
        chart("efficiency_cpu_p95","同机CPU单线程B1前向p95延迟",inference["summary"],"label","p95_ms_mean",unit="ms"),
        chart("efficiency_100hz_budget","模型求值占假设100Hz周期的比例（p95）",inference["summary"],"label","p95_fraction_of_100hz_period_pct_mean",unit="%"),
    ]
    return dict(schema="modeling_paper_efficiency_v1",generated_at=datetime.now(timezone.utc).isoformat(),
        sources=[relative(STUDY/"data/dataset_manifest.json"),relative(STUDY/"screening_summary.json"),
                 relative(PLUGIN/"convergence.json"),relative(PLUGIN/"plugin_results.json"),relative(PLUGIN/"normalization.json")],
        definitions=definitions,training=training,inference={k:v for k,v in inference.items() if k!="raw"},
        paper_paragraphs=dict(early_accuracy=early,recursive_inference=runtime,
            interpretation="建议使用‘结构化先验与分阶段拟合带来的早期预测精度’或‘较少联合优化即可达到较低误差’。若讨论优化效率，须同时报告初始化成本。样本效率、在线学习和真实规律辨识需要各自的额外实验。"),
        charts=charts,tables=[dict(id=key,title=title,rows=rows) for key,title,rows in [
            ("efficiency_configs","效率比较的真实模型配置",training["settings"]),
            ("efficiency_training_summary","训练与早期精度汇总",training["summary"]),
            ("efficiency_training_seeds","逐种子训练指标",training["seeds"]),
            ("efficiency_known_tuning_cost","已记录选型与五次正式拟合成本",training["tuning_cost"]),
            ("efficiency_cpu_summary","推理时间、延迟倒数与周期占比",inference["summary"]),
            ("efficiency_cpu_seeds","五种子推理分位数",inference["seeds"]),
            ("efficiency_early_tests","正式首轮的描述性配对比较",training["early_paired_tests"])]])


def markdown(result):
    training=result["training"];inference=result["inference"]
    lines=["# 论文效率论据", "", "本分析比较 HOV、原正式静态 MLP、Chen 方向网络、Krauss 振子适配和窗口 MLP。所有结果保留 seeds 0–4。", "",
           "## 可用于论文的表述", "", result["paper_paragraphs"]["early_accuracy"], "", result["paper_paragraphs"]["recursive_inference"], "",
           "## 早期精度与成本", "", "误差为验证集全体目标帧的节点平均距离，均值±种子SD。最佳仅指100 epoch内验证选中的checkpoint。", "",
           "| 模型 | 参数 | batch / 更新数每轮 | epoch1 mm | epoch5 mm | 最佳 mm | 首轮→最佳下降 | 首轮超出最佳 | 记录总耗时 s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in training["summary"]:
        lines.append(f"| {r['label']} | {int(r['parameters_mean'])} | {int(r['batch_size_mean'])} / {int(r['minibatch_steps_per_epoch_mean'])} | {r['epoch1_val_mm_mean']:.3f}±{r['epoch1_val_mm_sd']:.3f} | {r['epoch5_val_mm_mean']:.3f} | {r['best_val_mm_mean']:.3f}±{r['best_val_mm_sd']:.3f} | {r['epoch1_to_best_reduction_pct_mean']:.2f}% | {r['epoch1_excess_over_best_pct_mean']:.2f}% | {r['recorded_total_seconds_mean']:.3f}±{r['recorded_total_seconds_sd']:.3f} |")
    lines += ["", "HOV、静态MLP、Chen和Krauss总耗时来自原GPU并发任务，包含任务内预拟合、优化和其余开销；HOV参考预拟合在CPU完成。窗口MLP来自原CPU双线程任务，仅记录优化、验证与恢复最佳权重区段。因此不能依据该列跨任务排列训练速度。", "",
              "HOV正式第一轮前执行500步坐标预拟合、250步几何微调及全训练集记忆读出拟合。750次全批量优化各使用8988个训练目标帧，已重复使用同一组标签；首轮误差不能作为仅使用一次数据的证据。", "",
              "| 模型 | 配置 | 学习率 | 训练设备 |",
              "|---|---|---:|---|"]
    for r in training["settings"]:
        lines.append(f"| {r['label']} | {r['architecture']} | {r['initial_lr']} | {r['device']} |")
    lines += ["", "窗口MLP使用训练集逐特征标准化、batch512、每轮18次更新；其余batch256、每轮36次。各模型同为100轮且验证检查点相同，但不是等更新次数或等计算量实验。窗口MLP的ReduceLROnPlateau patience为3次检查，原正式为4次。", "",
              "## 计入初始化及已记录调参", "",
              "| 模型 | 模型/参考初始化 s | 记忆初始化 s | 正式优化 s | 其余记录时间 s |",
              "|---|---:|---:|---:|---:|"]
    for r in training["summary"]:
        def fmt(key):
            v=r.get(key+"_mean");return "未记录" if v is None else f"{v:.3f}"
        lines.append(f"| {r['label']} | {fmt('prior_and_model_init_seconds')} | {fmt('memory_init_seconds')} | {fmt('optimizer_seconds')} | {fmt('other_recorded_seconds')} |")
    lines += ["", "窗口预处理共享构建六种特征的train/val用时 " + f"{training['window_shared_feature_build_seconds']:.4f} s，未逐模型单列标准化/数据读取/模型构建成本。全离线流程还涉及采集与标签提取，不能用这些部分计时替代全流程耗时。", "",
              "| 模型 | 已记录候选数 | 候选耗时之和 s | 五次正式耗时之和 s |",
              "|---|---:|---:|---:|"]
    for r in training["tuning_cost"]:
        lines.append(f"| {r['label']} | {r['recorded_candidates']} | {r['candidate_run_seconds_sum']:.3f} | {r['five_formal_run_seconds_sum']:.3f} |")
    lines += ["", "表中为已保存的候选记录小计。并发运行时各任务墙钟相加不等于实际等待历时，且不包含更早的研究开发；不同任务设备与计时范围也有差异。", "",
              "## 同机单线程推理", "", f"硬件：{inference['hardware_before']['cpu']}；PyTorch {inference['hardware_before']['torch']}；开始时1/5/15分钟负载 {inference['hardware_before']['load_average_1_5_15']}。", "",
              "每个种子、每个模式预热20次、计时200次，模式顺序逐时间步随机排列。主表先计算各seed分位数，再给五个分位数的均值±SD。", "",
              "| 方法/模式 | p50 ms | p95 ms | 1/平均p50 Hz | 1/平均p95 Hz | p95占50Hz周期 | p95占100Hz周期 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in inference["summary"]:
        lines.append(f"| {r['label']} | {r['p50_ms_mean']:.4f}±{r['p50_ms_sd']:.4f} | {r['p95_ms_mean']:.4f}±{r['p95_ms_sd']:.4f} | {r['reciprocal_of_mean_p50_hz']:.1f} | {r['reciprocal_of_mean_p95_hz']:.1f} | {r['p95_fraction_of_50hz_period_pct_mean']:.2f}% | {r['p95_fraction_of_100hz_period_pct_mean']:.2f}% |")
    lines += ["", "计时含必要的历史/方向处理、窗口标准化和毫米坐标输出。HOV缓存模式还含一步状态递推，首窗口初始化不计时。保存状态的连续预测不等同于每次重置20步窗口，长期递推误差未由本计时实验验证。", "",
              "延迟倒数衡量串行模型求值的计算上限估计。模型时间步仍为0.2s，未测量高频闭环的预测精度、稳定性及相机/通信/规划耗时；本文可使用50或100Hz周期中的模型计算占比描述计算余量。", "",
              "## 结论口径与核验", "",
              result["paper_paragraphs"]["interpretation"], "",
              "五对首轮比较使用双侧精确Wilcoxon，并对四项比较作Holm校正；统计单位是训练seed。初始化与batch不同，该比较评价现有拟合流程的首轮表现，不能归因于某个独立机制。", "",
              "已核对25份训练日志的首/末轮及最佳检查点，复核原正式曲线与旧收敛报告一致，严格装载25个模型并核对32个窗口的保存预测；另验证5个HOV缓存单步与相同前史完整前向等价。原始数据未修改。", "",
              "## 文件与复现", "",
              "```bash", "/Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/analyze_modeling_paper_efficiency.py", "```", "",
              f"- JSON：`{relative(REPORT.with_suffix('.json'))}`", f"- 原始时延、曲线与分种子指标：`{relative(OUTPUT)}`", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--runs",type=int,default=200)
    parser.add_argument("--reuse-timing",action="store_true",help="Rebuild narrative from this task's saved timing; no rerun.")
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    training=training_evidence()
    write_json(OUTPUT/"training.json",training)
    if args.reuse_timing:
        inference=read(OUTPUT/"inference.json")
    else:
        inference=inference_evidence(args.warmup,args.runs)
        write_json(OUTPUT/"inference.json",inference)
    result=assemble(training,inference)
    write_json(REPORT.with_suffix(".json"),result)
    REPORT.with_suffix(".md").write_text(markdown(result))
    for name,rows in [("training_seeds",training["seeds"]),("training_curves",training["curves"]),
                      ("training_summary",training["summary"]),("inference_samples",inference["raw"]),
                      ("inference_seeds",inference["seeds"]),("inference_summary",inference["summary"])]:
        write_csv(OUTPUT/f"{name}.csv",rows)
    checks=training["validation"]+inference["validation"]
    assert all(row["passed"] for row in checks)
    write_json(OUTPUT/"validation.json",dict(status="passed",checks=checks,
        models=MODELS,seeds=SEEDS,curves=len(training["curves"]),inference_samples=len(inference["raw"]),
        note="No training, test selection, source-image scans or dataset mutation."))
    write_json(OUTPUT/"COMPLETE.json",dict(completed_at=datetime.now(timezone.utc).isoformat(),
        report=relative(REPORT.with_suffix(".json")),script=relative(Path(__file__))))
    print(json.dumps({"training":[{k:r[k] for k in ('label','epoch1_val_mm_mean','best_val_mm_mean','epoch1_to_best_reduction_pct_mean','recorded_total_seconds_mean')} for r in training['summary']],
                      "timing":[{k:r[k] for k in ('label','p50_ms_mean','p95_ms_mean','p95_fraction_of_100hz_period_pct_mean')} for r in inference['summary']]},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
