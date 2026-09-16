"""Physical skeleton/mask metrics and group-paired benchmark inference.

Skeleton coordinates are already in robot-frame millimetres, ordered
``base_to_tip``. Calibration and tube radius must be fixed independently of
the test data by the caller. This module performs no alignment or fitting.
Only NumPy and SciPy are required; model/training packages are not imported.
"""

from collections.abc import Mapping
import hashlib
from numbers import Integral

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

__all__ = ["skeleton_metrics", "mask_metrics", "render_tube", "compare_models"]

_EXACT_MAX_GROUPS = 16
_PERMUTATION_SAMPLES = 50000
_BOOTSTRAP_SAMPLES = 10000
_ALPHA = 0.05


def _finite_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nonnegative_scalar(value, name):
    array = _finite_array(value, name)
    if array.ndim != 0 or array.item() < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")
    return float(array)


def skeleton_metrics(pred, target) -> dict[str, np.ndarray]:
    """Return five per-frame ``(T,)`` errors in mm for equal ``(T,N,3)`` arrays.

    T and N must be positive, values finite, nodes corresponding and ordered
    base_to_tip (last node is the endpoint). Inputs must already be physical
    mm; units and ordering cannot be inferred from coordinates.

    ``mean_node_mm`` averages node Euclidean errors; ``node_rmse_mm`` is
    sqrt(mean of squared node Euclidean errors), not coordinate-wise RMSE.
    ``endpoint_mm`` and ``max_node_mm`` measure last-node and worst-node error.
    ``chamfer_mm`` is half the sum of the two mean nearest-node Euclidean
    distances, using the sampled skeleton nodes, not squared distances or
    continuous curve distances. Frames are never averaged here.
    """
    pred = _finite_array(pred, "pred")
    target = _finite_array(target, "target")
    if (pred.ndim != 3 or pred.shape[-1] != 3 or
            pred.shape != target.shape or min(pred.shape[:2]) == 0):
        raise ValueError("pred and target must have identical nonempty (T,N,3) shapes")
    errors = np.linalg.norm(pred - target, axis=-1)
    chamfer = np.empty(pred.shape[0], dtype=np.float64)
    # Allocate a node-pair matrix per frame, not a T*N*N*3 trajectory tensor.
    for frame, (p, t) in enumerate(zip(pred, target)):
        distances = np.linalg.norm(p[:, None, :] - t[None, :, :], axis=-1)
        chamfer[frame] = (distances.min(axis=1).mean() +
                          distances.min(axis=0).mean()) / 2
    return {
        "mean_node_mm": errors.mean(axis=1),
        "node_rmse_mm": np.sqrt(np.mean(errors ** 2, axis=1)),
        "endpoint_mm": errors[:, -1].copy(),
        "max_node_mm": errors.max(axis=1),
        "chamfer_mm": chamfer,
    }


def _binary_mask(value, name):
    array = np.asarray(value)
    if (array.ndim != 2 or 0 in array.shape or
            array.dtype.kind not in "biuf" or
            not np.all((array == 0) | (array == 1))):
        raise ValueError(f"{name} must be a nonempty 2D binary mask (bool or 0/1)")
    return array.astype(bool, copy=False)


def mask_metrics(pred, target, boundary_tolerance_px=2) -> dict[str, float]:
    """Score equal 2D binary masks; probabilities and 0/255 masks are rejected.

    Both empty: all five scores are 1. Exactly one empty: all five are 0,
    including precision/recall with a zero denominator. Otherwise use ordinary
    foreground IoU, Dice, precision, and recall.

    Boundaries are foreground pixels removed by one 3x3 (8-neighbour) erosion,
    treating pixels outside the image as background. Boundary precision and
    recall count pixels within Euclidean distance <= boundary_tolerance_px
    of the opposite boundary. Tolerances <=4 px use dilation by a discrete
    Euclidean disk; larger tolerances use SciPy's distance transform. ``boundary_f1``
    is their harmonic mean. Tolerance is a finite nonnegative scalar, in px;
    the declared value is included in the output for auditability.
    """
    pred = _binary_mask(pred, "pred")
    target = _binary_mask(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target masks must have identical shapes")
    tolerance = _nonnegative_scalar(boundary_tolerance_px, "boundary_tolerance_px")
    n_pred, n_target = int(pred.sum()), int(target.sum())
    if not n_pred or not n_target:
        score = float(n_pred == n_target)
        return dict.fromkeys(("iou", "dice", "precision", "recall", "boundary_f1"), score) | {
            "boundary_tolerance_px": tolerance,
        }
    intersection = int(np.count_nonzero(pred & target))
    structure = np.ones((3, 3), dtype=bool)
    p_boundary = pred & ~binary_erosion(pred, structure=structure, border_value=0)
    t_boundary = target & ~binary_erosion(target, structure=structure, border_value=0)
    if tolerance <= 4:
        radius = int(tolerance)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        # dx²+dy² <= tolerance². Compare distances to preserve EDT's floating
        # threshold at values such as sqrt(13), whose square rounds below 13.
        disk = np.sqrt(xx * xx + yy * yy) <= tolerance
        precision = float(np.mean(binary_dilation(t_boundary, structure=disk,
                                                  border_value=0)[p_boundary]))
        recall = float(np.mean(binary_dilation(p_boundary, structure=disk,
                                               border_value=0)[t_boundary]))
    else:
        precision = float(np.mean(distance_transform_edt(~t_boundary)[p_boundary] <= tolerance))
        recall = float(np.mean(distance_transform_edt(~p_boundary)[t_boundary] <= tolerance))
    boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": float(intersection / (n_pred + n_target - intersection)),
        "dice": float(2 * intersection / (n_pred + n_target)),
        "precision": float(intersection / n_pred),
        "recall": float(intersection / n_target),
        "boundary_f1": float(boundary_f1),
        "boundary_tolerance_px": tolerance,
    }


def render_tube(skeleton, homography, shape, radius_px) -> np.ndarray:
    """Rasterize one ``(N,3)`` base_to_tip skeleton to a boolean ``(H,W)`` mask.

    A supplied nonsingular 3x3 homography maps [robot X mm, robot Y mm, 1]
    to homogeneous [pixel x/column, pixel y/row, w]; Z is ignored. The tube
    is the union of round-ended projected segments with fixed radius_px.
    A pixel is foreground iff its integer centre (column,row) is within
    radius_px of a segment. Fractional radii/subpixel positions are supported;
    a single node produces a disk and radius zero covers centres on the curve.
    Off-image geometry is clipped by the image, not clamped to its border.

    Require N>=1, positive integer image dimensions and finite inputs. Reject
    singular homographies and segments touching/crossing the projective horizon.
    The caller must supply calibration and radius chosen independently of test
    targets; this function has no target-dependent calibration parameters.
    """
    skeleton = _finite_array(skeleton, "skeleton")
    homography = _finite_array(homography, "homography")
    radius = _nonnegative_scalar(radius_px, "radius_px")
    if skeleton.ndim != 2 or skeleton.shape[1] != 3 or skeleton.shape[0] == 0:
        raise ValueError("skeleton must have nonempty shape (N,3)")
    if (not isinstance(shape, (tuple, list, np.ndarray)) or np.ndim(shape) != 1 or len(shape) != 2 or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v <= 0
                for v in shape)):
        raise ValueError("shape must contain two positive integers (H,W)")
    if homography.shape != (3, 3):
        raise ValueError("homography must have shape (3,3)")
    scale = np.max(np.abs(homography))
    if scale == 0:
        raise ValueError("homography must be nonsingular")
    homography = homography / scale  # Homographies are invariant to scalar multiples.
    if np.linalg.matrix_rank(homography) < 3:
        raise ValueError("homography must be nonsingular")
    homogeneous = np.column_stack((skeleton[:, :2], np.ones(len(skeleton)))) @ homography.T
    weights = homogeneous[:, 2]
    if (not np.isfinite(homogeneous).all() or np.any(weights == 0) or
            np.any(np.signbit(weights[1:]) != np.signbit(weights[:-1]))):
        raise ValueError("skeleton touches or crosses the homography horizon")
    with np.errstate(over="ignore", invalid="ignore"):
        pixels = homogeneous[:, :2] / weights[:, None]
    if not np.isfinite(pixels).all():
        raise ValueError("homography projection must be finite")
    height, width = map(int, shape)
    mask = np.zeros((height, width), dtype=bool)
    starts, ends = (pixels[:-1], pixels[1:]) if len(pixels) > 1 else (pixels, pixels)
    for start, end in zip(starts, ends):
        lower = np.maximum(np.minimum(start, end) - radius, [0, 0])
        upper = np.minimum(np.maximum(start, end) + radius, [width - 1, height - 1])
        if np.any(lower > upper):
            continue
        x0, y0 = np.ceil(lower).astype(int)
        x1, y1 = np.floor(upper).astype(int)
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        delta = end - start
        length = np.hypot(*delta)
        if length == 0:
            distance = np.hypot(xx - start[0], yy - start[1])
        else:
            direction = delta / length
            dx, dy = xx - start[0], yy - start[1]
            along = dx * direction[0] + dy * direction[1]
            # Cross-product distance preserves exact collinearity, including
            # radius-zero diagonal pixels that projection subtraction can lose.
            distance = np.abs(dx * delta[1] - dy * delta[0]) / length
            distance = np.where(along <= 0, np.hypot(dx, dy), distance)
            distance = np.where(along >= length,
                                np.hypot(xx - end[0], yy - end[1]), distance)
        mask[y0:y1 + 1, x0:x1 + 1] |= distance <= radius
    return mask


def _identifier(value, name):
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    raise ValueError(f"{name} must be a nonempty string or integer")


def _sort_identifier(value):
    return (type(value).__name__, value)


def _sign_flip(differences, rng):
    n = len(differences)
    scale = np.max(np.abs(differences))
    normalized = differences / scale if scale else differences
    observed = abs(normalized.mean())
    tolerance = 32 * np.finfo(float).eps * np.mean(np.abs(normalized))
    exact = n <= _EXACT_MAX_GROUPS
    samples = 2 ** n if exact else _PERMUTATION_SAMPLES
    hits = 0
    # Bound temporary memory for large numbers of independent groups.
    batch_size = max(1, min(2048, 1000000 // n))
    for offset in range(0, samples, batch_size):
        count = min(batch_size, samples - offset)
        if exact:
            codes = np.arange(offset, offset + count, dtype=np.uint64)
            signs = 2 * ((codes[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float) - 1
        else:
            signs = 2 * rng.integers(0, 2, size=(count, n)) - 1
        null_statistics = np.abs(np.mean(signs * normalized, axis=1))
        hits += int(np.count_nonzero(null_statistics >= observed - tolerance))
    p_value = hits / samples if exact else (hits + 1) / (samples + 1)
    return float(p_value), "exact" if exact else "monte_carlo", samples


def _bootstrap_ci(differences, rng):
    n = len(differences)
    means = np.empty(_BOOTSTRAP_SAMPLES)
    batch_size = max(1, min(2048, 1000000 // n))
    for offset in range(0, _BOOTSTRAP_SAMPLES, batch_size):
        count = min(batch_size, _BOOTSTRAP_SAMPLES - offset)
        indices = rng.integers(0, n, size=(count, n))
        means[offset:offset + count] = differences[indices].mean(axis=1)
    return np.quantile(means, [_ALPHA / 2, 1 - _ALPHA / 2]).tolist()


def compare_models(records, reference, metrics, seed=0) -> dict:
    """Compare every model against reference using independent paired groups.

    records is an iterable of mappings with ``model`` (nonempty string),
    ``seed``, ``group`` (integer/string identifiers), and ``metrics`` (mapping
    of finite scalar group summaries). Supply exactly one record per
    (model,group,seed). Reduce frames within each group/seed before calling;
    arrays and duplicate records are rejected to prevent frame pseudoreplication.
    All models must have exactly the reference's (group,seed) coverage. Seed
    sets may differ between groups; within a group they must match across models.

    metrics is a nonempty sequence of names, or {name: "lower"/"higher"}.
    Names iou/dice/precision/recall/boundary_f1 default to higher-is-better;
    other names default to lower-is-better. Differences are always model minus
    reference. Seeds are averaged per group, then groups weighted equally.

    Two-sided paired sign-flip tests enumerate all 2**n assignments for n<=16;
    otherwise use 50,000 draws and the plus-one p-value correction. This test
    assumes exchangeable model labels (symmetric differences under the null).
    A paired group bootstrap (10,000 draws) gives a pointwise 95% percentile
    mean-difference CI. Holm correction covers ALL model/metric comparisons.
    Significance requires adjusted p<=.05 AND a CI excluding zero. Fewer than
    six groups always yields status "inconclusive" and significant=False.

    The JSON-serializable report contains method metadata, auditable paired
    group means and differences, raw/adjusted p-values, and directional status
    (improved/worse/no_evidence/inconclusive). CIs are not multiplicity-adjusted.
    Ordering and RNG streams are stable under record/metric input reordering.
    """
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference must be a nonempty model name")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    higher = {"iou", "dice", "precision", "recall", "boundary_f1"}
    if isinstance(metrics, (str, bytes)):
        raise ValueError("metrics must be a sequence of names or a direction mapping")
    names = list(metrics)
    if (not names or any(not isinstance(name, str) or not name for name in names) or
            len(set(names)) != len(names)):
        raise ValueError("metrics must contain unique nonempty names")
    directions = {name: (metrics[name] if isinstance(metrics, Mapping) else
                         "higher" if name in higher else "lower") for name in names}
    if any(direction not in ("lower", "higher") for direction in directions.values()):
        raise ValueError("metric directions must be 'lower' or 'higher'")
    models = {}
    for record in records:
        if not isinstance(record, Mapping) or not {"model", "seed", "group", "metrics"} <= record.keys():
            raise ValueError("each record requires model, seed, group, and metrics")
        model = record["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        group = _identifier(record["group"], "group")
        record_seed = _identifier(record["seed"], "record seed")
        values = record["metrics"]
        if not isinstance(values, Mapping) or any(name not in values for name in names):
            raise ValueError("each record must contain all requested metrics")
        scalars = {}
        for name in names:
            value = _finite_array(values[name], f"metric {name}")
            if value.ndim != 0:
                raise ValueError(f"metric {name} must be a scalar group/seed summary, not frames")
            scalars[name] = float(value)
        pairs = models.setdefault(model, {})
        if (group, record_seed) in pairs:
            raise ValueError(f"duplicate (model,group,seed) record for {model!r}")
        pairs[group, record_seed] = scalars
    if reference not in models or len(models) < 2:
        raise ValueError("records must contain the reference and at least one other model")
    coverage = set(models[reference])
    for model, pairs in models.items():
        if set(pairs) != coverage:
            missing, extra = coverage - set(pairs), set(pairs) - coverage
            raise ValueError(f"paired group/seed coverage mismatch for {model!r}: "
                             f"{len(missing)} missing, {len(extra)} extra")
    groups = sorted({group for group, _ in coverage}, key=_sort_identifier)
    group_seeds = {group: sorted((s for g, s in coverage if g == group), key=_sort_identifier)
                   for group in groups}
    comparisons = []
    for model in sorted(set(models) - {reference}):
        for name in sorted(names):
            paired = []
            for group in groups:
                seeds = group_seeds[group]
                model_mean = float(np.mean([models[model][group, s][name] for s in seeds]))
                ref_mean = float(np.mean([models[reference][group, s][name] for s in seeds]))
                paired.append({"group": group, "seeds": seeds, "model_mean": model_mean,
                               "reference_mean": ref_mean, "difference": model_mean - ref_mean})
            differences = np.array([p["difference"] for p in paired])
            if not np.isfinite(differences).all():
                raise ValueError("group means and differences must be finite")
            # Stable per-comparison RNG, separate bootstrap and permutation streams.
            digest = hashlib.sha256((model + "\0" + name).encode()).digest()
            entropy = [int(seed)] + np.frombuffer(digest, dtype="<u4").tolist()
            streams = np.random.SeedSequence(entropy).spawn(2)
            p_value, method, samples = _sign_flip(differences, np.random.default_rng(streams[0]))
            ci = _bootstrap_ci(differences, np.random.default_rng(streams[1]))
            comparisons.append({
                "model": model, "reference": reference, "metric": name,
                "direction": directions[name], "n_groups": len(groups),
                "paired_groups": paired,
                "model_mean": float(np.mean([p["model_mean"] for p in paired])),
                "reference_mean": float(np.mean([p["reference_mean"] for p in paired])),
                "mean_difference": float(differences.mean()), "ci_95": ci,
                "p_value": p_value, "permutation_method": method,
                "permutation_samples": samples,
            })
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["p_value"])
    adjusted = 0.0
    for rank, index in enumerate(order):
        result = comparisons[index]
        adjusted = min(1.0, max(adjusted, (len(order) - rank) * result["p_value"]))
        result["p_value_holm"] = float(adjusted)
        low, high = result["ci_95"]
        significant = len(groups) >= 6 and adjusted <= _ALPHA and (low > 0 or high < 0)
        result["significant"] = bool(significant)
        if len(groups) < 6:
            result["status"] = "inconclusive"
        elif significant:
            improvement = (result["mean_difference"] < 0 if result["direction"] == "lower"
                           else result["mean_difference"] > 0)
            result["status"] = "improved" if improvement else "worse"
        else:
            result["status"] = "no_evidence"
    return {
        "reference": reference, "seed": int(seed), "independent_unit": "group",
        "difference_convention": "model_minus_reference",
        "aggregation": "equal seed means within group, equal group means across groups",
        "alpha": _ALPHA, "alternative": "two-sided", "minimum_groups": 6,
        "exact_max_groups": _EXACT_MAX_GROUPS,
        "bootstrap_samples": _BOOTSTRAP_SAMPLES, "ci_method": "paired_group_percentile",
        "ci_level": 0.95, "ci_multiplicity_adjusted": False,
        "correction": "holm", "correction_family": "all model/metric comparisons",
        "n_comparisons": len(comparisons), "comparisons": comparisons,
    }
