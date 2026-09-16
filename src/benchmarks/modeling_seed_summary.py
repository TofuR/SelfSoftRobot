"""Frame-pooled metrics and paired training-seed statistics on one fixed split.

``pool_seed`` reads modeling_runner's evaluation_manifest.json, records.json,
and *_predictions.npz. ``metrics.json`` can replace records.json as a record
list, a {"records": [...]} object, or a run-level object whose metadata is in
the evaluation manifest / resolved training config. Sequence summary numbers
are never used to compute pooled metrics. Only evaluation arrays and JSON
metadata are read; dataset files and annotation images are not needed.

``summarize_seeds`` requires the seeds chosen before observing results. Each
model gets a row for every seed; missing seeds prevent its aggregate and paired
tests. STD uses ddof=1. IoU/Dice are means over scored mask frames, not ratios
of pooled pixels (the evaluator does not save pixel contingency counts).

``benchmark_latency`` reconstructs shape_modeling_checkpoint_v1 with the
existing make_model interface and measures full, independent history-20
windows at B=1 and B=256, including physical-unit output conversion.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from scipy import stats

__all__ = ["DEFAULT_METRICS", "pool_seed", "summarize_seeds",
           "paired_seed_tests", "holm_adjust", "benchmark_latency"]

_SKELETON = ("mean_node_mm", "endpoint_mm", "node_rmse_mm", "max_node_mm", "chamfer_mm")
_MASK = ("mask_iou", "mask_dice", "mask_precision", "mask_recall", "mask_boundary_f1")
DEFAULT_METRICS = (*_SKELETON, "node_p50_mm", "node_p95_mm",
                   "endpoint_p50_mm", "endpoint_p95_mm", *_MASK)
_SCOPE = ("The statistical unit is a paired training seed on the same fixed "
          "train/validation/test split. Uncertainty concerns training randomness "
          "conditional on this split; it does not establish cross-sequence, "
          "cross-split, or cross-day generalization. Frames are pooled measurements, "
          "not independent statistical replicates.")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(output, result):
    if output is not None:
        path = Path(output)
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return array.astype(np.float64, copy=False)


def _frame_ids(value, name):
    ids = np.asarray(value)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or
            np.any(ids < 0) or (len(ids) > 1 and np.any(ids[1:] <= ids[:-1]))):
        raise ValueError(f"{name} must be unique, increasing nonnegative integer IDs")
    return ids


def _digest(array, dtype):
    # Fingerprint evaluated targets/IDs to verify the same paired observations.
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def _metadata(directory):
    manifest_path = directory / "evaluation_manifest.json"
    if not manifest_path.exists():
        manifest_path = directory / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation: {directory}")
    records_path = directory / "records.json"
    if not records_path.exists():
        records_path = directory / "metrics.json"
    payload = _read_json(records_path)
    if isinstance(payload, list):
        rows, extra = payload, {}
    elif isinstance(payload, dict):
        rows, extra = payload.get("records", []), payload
    else:
        raise ValueError("records.json/metrics.json must contain records or an object")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid evaluation records")
    sources = [manifest, extra, *rows]
    # Run-level metrics exports may leave model/seed/history in this config.
    if not all(any(key in source for source in sources)
               for key in ("model", "seed", "history", "eval_stride")):
        run = Path(manifest.get("run", "."))
        if not run.is_absolute():
            run = directory / run
        config = run / "resolved_config.json"
        if config.exists():
            sources.append(_read_json(config))

    def agreed(key, required=False):
        values = [source[key] for source in sources if key in source]
        if not values and required:
            raise ValueError(f"Missing evaluation metadata: {key}")
        if len({json.dumps(value, sort_keys=True) for value in values}) > 1:
            raise ValueError(f"Inconsistent evaluation metadata: {key}")
        return values[0] if values else None

    model, seed = agreed("model", True), agreed("seed", True)
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a nonempty string")
    seed = _integer(seed, "seed")
    required = ("role", "dataset_manifest_sha256", "history", "eval_stride", "masks_enabled")
    optional = ("fold", "mask_adapter", "radius_mm", "mask_stride", "boundary_tolerance_px",
                "evaluation_code_hashes", "evidence_level")
    protocol = {key: agreed(key, key in required) for key in (*required, *optional)}
    if protocol["role"] not in ("val", "test"):
        raise ValueError("Evaluation role must be val or test")
    for key in ("history", "eval_stride"):
        _integer(protocol[key], key, 1)
    if not isinstance(protocol["masks_enabled"], bool):
        raise ValueError("masks_enabled must be boolean")
    if not protocol["dataset_manifest_sha256"]:
        raise ValueError("A fixed dataset_manifest_sha256 is required")
    return model, seed, protocol, agreed("run_kind"), rows


def pool_seed(evaluation):
    """Return a JSON-serializable pooled row for one model/seed evaluation dir.

    Euclidean node errors and Chamfer errors are recomputed from mm coordinates.
    All sequences must have the same node count. Quantiles use concatenated
    node/endpoint arrays; RMSE is sqrt(mean(squared Euclidean node errors)).
    Mask metrics pool their own mask_frame_ids, which can have a lower cadence.
    Invalid arrays, partial mask coverage, and inconsistent metadata raise.
    """
    from src.evaluation.modeling_benchmark_metrics import skeleton_metrics

    directory = Path(evaluation).resolve()
    model, seed, protocol, run_kind, rows = _metadata(directory)
    files = sorted(directory.glob("*_predictions.npz"))
    if not files:
        raise ValueError(f"No prediction arrays in {directory}")
    by_group = {}
    for row in rows:
        if "group" in row:
            if row["group"] in by_group:
                raise ValueError("Duplicate sequence record")
            by_group[row["group"]] = row
    if by_group and set(by_group) != {p.name[:-len("_predictions.npz")] for p in files}:
        raise ValueError("Sequence records and prediction files disagree")
    pooled = {key: [] for key in _SKELETON}
    node_errors, mask_arrays, coverage = [], {}, []
    node_count, mask_keys = None, None
    for path in files:
        group = path.name[:-len("_predictions.npz")]
        with np.load(path, allow_pickle=False) as arrays:
            pred = _finite(arrays["prediction_mm"], "prediction_mm")
            target = _finite(arrays["target_mm"], "target_mm")
            values = skeleton_metrics(pred, target)
            ids = _frame_ids(arrays["frame_ids"], "frame_ids")
            mids = _frame_ids(arrays["mask_frame_ids"], "mask_frame_ids")
            if len(ids) != len(pred) or not np.isin(mids, ids).all():
                raise ValueError("Frame IDs do not match prediction/mask coverage")
            if node_count is not None and node_count != pred.shape[1]:
                raise ValueError("Sequences must have the same node count")
            node_count = pred.shape[1]
            keys = {key for key in _MASK if key in arrays}
            if protocol["masks_enabled"]:
                if not len(mids) or not {"mask_iou", "mask_dice"} <= keys:
                    raise ValueError("Enabled masks require scored frames, mask_iou and mask_dice")
                stride = protocol["mask_stride"]
                if stride is not None:
                    _integer(stride, "mask_stride", 1)
                    if not np.array_equal(mids, ids[::stride]):
                        raise ValueError("Mask frame coverage disagrees with mask_stride")
            elif len(mids) or keys:
                raise ValueError("Mask arrays disagree with masks_enabled")
            if mask_keys is not None and keys != mask_keys:
                raise ValueError("Inconsistent mask metric availability across sequences")
            mask_keys = keys
            for key in keys:
                scores = _finite(arrays[key], key)
                if scores.shape != (len(mids),) or np.any((scores < 0) | (scores > 1)):
                    raise ValueError(f"{key} must have one [0,1] score per mask frame")
                mask_arrays.setdefault(key, []).append(scores)
            for key, value in values.items():
                pooled[key].append(value)
            node_errors.append(np.linalg.norm(pred - target, axis=-1).ravel())
            row = by_group.get(group, {})
            for key, actual in (("frames", len(ids)), ("mask_frames", len(mids))):
                if key in row and row[key] != actual:
                    raise ValueError(f"{group}: {key} disagrees with saved arrays")
            coverage.append(dict(group=group, frames=len(ids), mask_frames=len(mids),
                                 nodes=node_count, frame_ids_sha256=_digest(ids, "<i8"),
                                 mask_frame_ids_sha256=_digest(mids, "<i8"),
                                 target_mm_sha256=_digest(target, "<f8")))
    errors = np.concatenate(node_errors)
    endpoints = np.concatenate(pooled["endpoint_mm"])
    metrics = {key: float(np.concatenate(value).mean()) for key, value in pooled.items()}
    metrics["node_rmse_mm"] = float(np.sqrt(np.mean(errors ** 2)))
    for label, q in (("p50", .5), ("p95", .95)):
        metrics[f"node_{label}_mm"] = float(np.quantile(errors, q))
        metrics[f"endpoint_{label}_mm"] = float(np.quantile(endpoints, q))
    metrics.update({key: float(np.concatenate(value).mean()) for key, value in mask_arrays.items()})
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Nonfinite pooled metric")
    return dict(model=model, seed=seed, status="complete", evaluation=str(directory),
                run_kind=run_kind, protocol=protocol, coverage=coverage,
                frames=sum(row["frames"] for row in coverage),
                mask_frames=sum(row["mask_frames"] for row in coverage), metrics=metrics)


def _design(per_seed, seeds, models, metrics):
    seeds = [_integer(seed, "seed") for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Predeclared seeds must be nonempty and unique")
    metrics = list(DEFAULT_METRICS if metrics is None else metrics)
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(DEFAULT_METRICS):
        raise ValueError("Specify unique supported metrics")
    models = sorted({row["model"] for row in per_seed}) if models is None else list(models)
    if (not models or len(set(models)) != len(models) or
            any(not isinstance(model, str) or not model for model in models)):
        raise ValueError("Specify unique model names (models is required for empty evaluations)")
    index, contract = {}, None
    for row in per_seed:
        key = (row["model"], row["seed"])
        if key in index:
            raise ValueError(f"Duplicate model/seed: {key}")
        if row["model"] not in models or row["seed"] not in seeds:
            raise ValueError(f"Unexpected model/seed outside the predeclared design: {key}")
        if row.get("status") != "complete":
            raise ValueError("Pass complete pool_seed rows; missing seeds are filled automatically")
        current = json.dumps([row["protocol"], row["coverage"]], sort_keys=True)
        if contract is not None and current != contract:
            raise ValueError("Evaluations must share a fixed split, protocol, targets and frame coverage")
        contract = current
        for value in row["metrics"].values():
            if not np.isscalar(value) or not math.isfinite(float(value)):
                raise ValueError("Metrics must be finite scalars; seeds cannot be dropped")
        index[key] = row
    return seeds, models, metrics, index


def holm_adjust(pvalues):
    """Holm FWER adjustment in input order; None reserves an untested hypothesis.

    The family size includes None entries (treated as p=1 internally), and their
    returned values remain None. This prevents missing data shrinking a planned
    comparison family. Call separately for primary and exploratory test families.
    """
    pvalues = list(pvalues)
    for value in pvalues:
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("p-values must be finite values in [0,1] or None")
    order = sorted(range(len(pvalues)), key=lambda i: 1 if pvalues[i] is None else pvalues[i])
    adjusted, previous = [None] * len(pvalues), 0.
    for rank, i in enumerate(order):
        p = 1. if pvalues[i] is None else pvalues[i]
        previous = min(1., max(previous, (len(pvalues) - rank) * p))
        if pvalues[i] is not None:
            adjusted[i] = float(previous)
    return adjusted


def _wilcoxon_exact(differences):
    # Wilcox zero convention. Twice the average ranks are integers even at ties.
    d = differences[differences != 0]
    n = len(d)
    ranks = np.rint(2 * stats.rankdata(np.abs(d))).astype(int)
    positive = int(ranks[d > 0].sum())
    total = int(ranks.sum())
    tail = min(positive, total - positive)
    counts = [1]
    for rank in ranks:
        updated = [0] * (len(counts) + int(rank))
        for score, count in enumerate(counts):
            updated[score] += count
            updated[score + rank] += count
        counts = updated
    p = min(1., 2 * sum(counts[:tail + 1]) / (2 ** n))
    return dict(method="exact conditional signed-rank sign enumeration", alternative="two-sided",
                zero_method="wilcox", n_nonzero=n, n_zero=len(differences) - n,
                has_ties=len(set(np.abs(d))) < n, statistic=tail / 2,
                p_value=p, minimum_attainable_p=min(1., 2. ** (1 - n)),
                rank_biserial=(2 * positive - total) / total if total else 0.,
                assumption="Independent seed pairs with symmetric differences under the null")


def _paired_t(differences):
    result = dict(method="paired t-test", exploratory=True, statistic=None, p_value=None,
                  mean_difference_ci95=None, ci_adjustment="unadjusted exploratory interval",
                  assumption="Independent seed pairs and approximately normal paired differences")
    if len(differences) < 2:
        return dict(result, status="insufficient_seeds")
    sd = float(np.std(differences, ddof=1))
    if sd == 0:
        return dict(result, status="undefined_zero_variance")
    mean = float(differences.mean())
    se = sd / math.sqrt(len(differences))
    t = mean / se
    margin = float(stats.t.ppf(.975, len(differences) - 1)) * se
    return dict(result, status="ok", statistic=t,
                p_value=float(2 * stats.t.sf(abs(t), len(differences) - 1)),
                mean_difference_ci95=[mean - margin, mean + margin])


def paired_seed_tests(per_seed, seeds, reference, metrics=None, *, models=None,
                      alpha=.05, exploratory_t=False):
    """Compare every model to reference by identical predeclared seed IDs.

    ``per_seed`` contains pool_seed results. Differences are model-reference;
    improvement reverses the sign for errors. Exact Wilcoxon handles ties/zeros
    by its conditional sign distribution, not a normal approximation. With five
    nonzero pairs the smallest two-sided p is .0625. Holm covers all requested
    model x metric comparisons. Optional paired t tests form a separate,
    explicitly exploratory Holm family; zero-variance t tests are undefined.
    """
    per_seed = list(per_seed)
    seeds, models, metrics, index = _design(per_seed, seeds, models, metrics)
    if reference not in models:
        raise ValueError("reference must be one of the declared models")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    comparisons = []
    for model in models:
        if model == reference:
            continue
        for metric in metrics:
            direction = "higher" if metric in _MASK else "lower"
            values = [[index.get((name, seed), {}).get("metrics", {}).get(metric)
                       for seed in seeds] for name in (model, reference)]
            missing = [seed for i, seed in enumerate(seeds)
                       if any(value[i] is None for value in values)]
            row = dict(model=model, reference=reference, metric=metric, direction=direction,
                       seeds=seeds, model_values=values[0], reference_values=values[1],
                       missing_seeds=missing, n_pairs=len(seeds) - len(missing),
                       differences=None, mean_difference=None, median_difference=None,
                       std_difference=None, mean_improvement=None, cohen_dz=None,
                       wilcoxon=None, t_test=None, significant=False, inference_eligible=False,
                       status="incomplete" if missing else "complete")
            if not missing:
                d = np.asarray(values[0]) - np.asarray(values[1])
                sd = float(np.std(d, ddof=1)) if len(d) > 1 else None
                eligible = (len(seeds) >= 2 and all(
                    index[(name, seed)]["protocol"]["role"] == "test" and
                    index[(name, seed)].get("run_kind") not in ("smoke", "screening", "screen")
                    for name in (model, reference) for seed in seeds))
                row.update(differences=d.tolist(), mean_difference=float(d.mean()),
                           median_difference=float(np.median(d)), std_difference=sd,
                           mean_improvement=float(d.mean()) * (1 if direction == "higher" else -1),
                           cohen_dz=float(d.mean() / sd) if sd else None,
                           wilcoxon=_wilcoxon_exact(d), inference_eligible=eligible,
                           status="complete" if eligible else "diagnostic")
                if exploratory_t:
                    row["t_test"] = _paired_t(d)
            comparisons.append(row)
    for field in ("wilcoxon", "t_test"):
        adjusted = holm_adjust([(row[field] or {}).get("p_value") for row in comparisons])
        for row, p in zip(comparisons, adjusted):
            if row[field] is not None:
                row[field].update(p_holm=p, family_size=len(comparisons),
                                  reject_holm=bool(p is not None and p <= alpha and
                                                   row.get("inference_eligible", False)))
                if field == "wilcoxon":
                    row["significant"] = row[field]["reject_holm"]
    return dict(reference=reference, seeds=seeds, alpha=alpha, statistical_scope=_SCOPE,
                statistical_unit="training_seed", difference_definition="model - reference",
                family="all declared non-reference models x requested metrics",
                exploratory_t_family="separate Holm family" if exploratory_t else None,
                five_seed_wilcoxon_minimum_two_sided_p=.0625, comparisons=comparisons)


def summarize_seeds(evaluations, seeds, *, reference=None, metrics=None, models=None,
                    exploratory_t=False, alpha=.05, output=None):
    """Pool each evaluation, then give every fixed seed equal weight.

    Optional ``models`` declares even models with no completed evaluations.
    Missing seeds are explicit rows; affected mean/std and tests remain null.
    Unexpected/duplicate seeds, invalid data or mismatched evaluation coverage
    raise instead of selecting a subset. ``output`` is an optional JSON filename;
    the same JSON-serializable dict is returned whether or not it is written.
    """
    pools = [pool_seed(directory) for directory in evaluations]
    seeds, models, metrics, index = _design(pools, seeds, models, metrics)
    rows, summaries = [], []
    for model in models:
        rows.extend(index.get((model, seed), dict(model=model, seed=seed, status="missing",
                                                  evaluation=None, metrics=None)) for seed in seeds)
        for metric in metrics:
            values = [index.get((model, seed), {}).get("metrics", {}).get(metric) for seed in seeds]
            missing = [seed for seed, value in zip(seeds, values) if value is None]
            summaries.append(dict(model=model, metric=metric, seeds=seeds, values=values,
                                  missing_seeds=missing, n_seeds=len(seeds),
                                  n_available=len(seeds) - len(missing),
                                  status="incomplete" if missing else "complete",
                                  mean=None if missing else float(np.mean(values)),
                                  std=None if missing or len(seeds) < 2 else float(np.std(values, ddof=1))))
    tests = (paired_seed_tests(pools, seeds, reference, metrics, models=models,
                               exploratory_t=exploratory_t, alpha=alpha)
             if reference is not None else None)
    result = dict(schema="modeling_seed_summary_v1", seeds=seeds, models=models,
                  metrics=metrics, std_ddof=1, statistical_scope=_SCOPE,
                  statistical_unit="training_seed",
                  pooling="All evaluated frames per seed; node quantiles use all node errors; "
                          "mask scores average scored mask frames; seeds have equal weight",
                  status="complete" if all(row["status"] == "complete" for row in summaries) else "incomplete",
                  per_seed=rows, summary=summaries, paired_tests=tests)
    _write_json(output, result)
    return result


def benchmark_latency(run, device, output, warmup=20, repeats=100):
    """Measure a saved model at B1/B256, H20, four action channels on one device.

    ``run`` is a completed training directory or checkpoint path. ``output`` is
    a JSON filename (None returns results only). Input generation, transfers,
    reconstruction and warmup are outside timing. Each timed call includes
    model(actions) and conversion to mm; CUDA is synchronized before and after.
    HOV's ordinary forward recomputes window burn-in, operators and geometry on
    every invocation. This is full-window prediction, not a stateful step API.
    Incompatible checkpoint history is rejected rather than silently changed.
    """
    import torch
    from src.benchmarks.modeling_models import make_model

    warmup = _integer(warmup, "warmup")
    repeats = _integer(repeats, "repeats", 1)
    run = Path(run).resolve()
    if run.is_dir():
        if not (run / "COMPLETE").exists():
            raise ValueError("Training run is incomplete")
        checkpoint_path = run / "best_eval_model.pt"
    else:
        checkpoint_path = run
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "shape_modeling_checkpoint_v1":
        raise ValueError("Unsupported checkpoint schema")
    config = checkpoint["config"]
    if config.get("history") != 20:
        raise ValueError("Latency protocol requires a checkpoint trained with history=20")
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("Latency benchmark supports cpu and cuda devices")
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda", torch.cuda.current_device())
    geometry = checkpoint.get("geometry_config")
    if (checkpoint["model"].startswith("hov") or checkpoint["model"] == "pcc") and geometry is None:
        raise ValueError("Checkpoint must contain geometry_config for reconstruction")
    center_np = _finite(checkpoint["center"], "checkpoint center")
    scale = float(checkpoint["scale"])
    if center_np.shape != (3,) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid checkpoint normalization")
    previous_threads = torch.get_num_threads()
    threads = _integer(config.get("threads", previous_threads), "threads", 1)
    measurements = []
    try:
        torch.set_num_threads(threads)
        with torch.random.fork_rng(devices=[]):
            model, _ = make_model(checkpoint["model"], config,
                                  normalization=(checkpoint["center"], scale), geometry_config=geometry)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(target_device).eval()
        center = torch.tensor(center_np, dtype=torch.float32, device=target_device)
        generator = torch.Generator(device="cpu").manual_seed(0)
        inputs = torch.rand(256, 20, 4, generator=generator, dtype=torch.float32).to(target_device)

        def synchronize():
            if target_device.type == "cuda":
                torch.cuda.synchronize(target_device)

        with torch.inference_mode():
            for batch in (1, 256):
                actions = inputs[:batch]
                for _ in range(warmup):
                    prediction = model(actions) * scale + center
                synchronize()
                elapsed = []
                for _ in range(repeats):
                    synchronize()
                    start = time.perf_counter_ns()
                    prediction = model(actions) * scale + center
                    synchronize()
                    elapsed.append((time.perf_counter_ns() - start) / 1e6)
                if (prediction.ndim != 3 or prediction.shape[0] != batch or
                        prediction.shape[-1] != 3 or not torch.isfinite(prediction).all().item()):
                    raise ValueError("Model must return finite (B,N,3) predictions")
                mean_ms = float(np.mean(elapsed))
                if mean_ms <= 0:
                    raise ValueError("Timer resolution is insufficient")
                measurements.append(dict(batch_size=batch, input_shape=[batch, 20, 4],
                                         output_shape=list(prediction.shape),
                                         p50_ms=float(np.quantile(elapsed, .5)),
                                         p95_ms=float(np.quantile(elapsed, .95)), mean_ms=mean_ms,
                                         throughput_windows_per_second=1000 * batch / mean_ms,
                                         samples_ms=elapsed))
        result = dict(schema="modeling_latency_v1", checkpoint=str(checkpoint_path),
                      model=checkpoint["model"], seed=config.get("seed"),
                      selected_epoch=checkpoint.get("selected_epoch"), device=str(target_device),
                      device_name=torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "cpu",
                      torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                      threads=threads, dtype="float32", history=20, action_dim=4,
                      warmup=warmup, repeats=repeats, input_seed=0,
                      input_distribution="Uniform [0,1); B1 uses the first B256 window",
                      eval_mode=True, inference_mode=True,
                      cuda_synchronized=target_device.type == "cuda",
                      prediction_semantics="Independent full-window prediction; this is not stateful incremental step latency",
                      timing_scope="Device-resident input -> complete model forward -> physical mm output; "
                                   "HOV includes window burn-in, hereditary operators and geometry on every call",
                      excluded_from_timing=["checkpoint loading", "model reconstruction", "input generation",
                                            "host/device transfers", "warmup", "output validation"],
                      training_cache_used=False,
                      throughput_definition="batch_size * repeats / sum(measured batch seconds)",
                      measurements=measurements)
    finally:
        torch.set_num_threads(previous_threads)
    _write_json(output, result)
    return result
