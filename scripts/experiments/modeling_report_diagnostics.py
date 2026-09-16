"""Bounded diagnostics for the saved three-sequence, five-seed HTML report.

Only study_plan.json and the main comparisons' saved prediction NPZs are read.
The public helper returns ordinary Python values and performs no file writes.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_NODES = 15
_FRAMES_PER_SEED = 2958
_SEEDS = (0, 1, 2, 3, 4)


def _plan_names(plan: dict, key: str, count: int) -> list[str]:
    names = plan.get(key)
    if not (
        isinstance(names, list)
        and len(names) == count
        and all(isinstance(name, str) and name not in ("", ".", "..")
                and Path(name).name == name for name in names)
        and len(set(names)) == count
    ):
        raise ValueError(f"study_plan.{key} must contain {count} unique names")
    return names


def _read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        prediction = np.asarray(arrays["prediction_mm"], dtype=np.float64)
        target = np.asarray(arrays["target_mm"], dtype=np.float64)
        frame_ids = arrays["frame_ids"]
    if not (
        prediction.ndim == 3
        and prediction.shape[1:] == (_NODES, 3)
        and 0 < len(prediction) <= _FRAMES_PER_SEED
        and target.shape == prediction.shape
        and frame_ids.shape == (len(prediction),)
        and np.issubdtype(frame_ids.dtype, np.integer)
    ):
        raise ValueError(f"Invalid prediction/target/frame_ids shapes or dtype: {path}")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError(f"Nonfinite coordinates: {path}")
    if not np.all(frame_ids[1:] > frame_ids[:-1]):
        raise ValueError(f"frame_ids must be unique and chronologically ordered: {path}")
    return prediction, target, frame_ids


def compute_diagnostics(study: Path) -> dict:
    """Return node profiles, a common-grid empirical CDF, and fixed shape examples.

    Euclidean errors use all three saved millimetre coordinates. For each seed,
    corresponding-node errors are pooled over all three sequences' frames;
    each node then gets a five-seed mean and sample SD (ddof=1). CDF observations
    are the per-frame mean-node errors from all five repeated predictions.

    Examples use seed 0 and scored-frame index ``len(frame_ids) // 2`` for every
    sequence (the later central frame for even lengths), determined solely by
    frame order. Targets and frame IDs must agree exactly across all models and
    seeds before one GT shape per sequence is emitted. Model order follows the
    study plan. Missing or inconsistent inputs raise an exception.

    Output is bounded to 120 node rows, at most 496 CDF rows, and 405 shape rows.
    Only the plan and 8 models x 5 seeds x 3 prediction NPZs are required.
    """
    study = Path(study)
    plan = json.loads((study / "study_plan.json").read_text(encoding="utf-8"))
    models = _plan_names(plan, "comparisons", 8)
    groups = _plan_names(plan, "groups", 3)
    declared_seeds = plan.get("repeat_seeds")
    if not (isinstance(declared_seeds, list)
            and all(type(seed) is int for seed in declared_seeds)
            and sorted(declared_seeds) == list(_SEEDS)):
        raise ValueError("study_plan.repeat_seeds must declare seeds 0, 1, 2, 3, 4")

    references = {}
    example_predictions = {}
    node_profiles = []
    frame_errors = {}
    for model in models:
        seed_node_means = []
        model_frame_errors = []
        for seed in _SEEDS:
            node_sums = np.zeros(_NODES, dtype=np.float64)
            frames = 0
            for group in groups:
                path = (study / "evaluations" / model / f"seed_{seed}"
                        / f"{group}_predictions.npz")
                prediction, target, frame_ids = _read_predictions(path)
                if group not in references:
                    references[group] = (target, frame_ids)
                else:
                    ref_target, ref_ids = references[group]
                    if not (np.array_equal(frame_ids, ref_ids)
                            and np.array_equal(target, ref_target)):
                        raise ValueError(f"Targets or frame IDs differ from reference: {path}")
                errors = np.linalg.norm(prediction - target, axis=-1)
                if not np.isfinite(errors).all():
                    raise ValueError(f"Nonfinite Euclidean errors: {path}")
                node_sums += errors.sum(axis=0)
                frames += len(frame_ids)
                model_frame_errors.append(errors.mean(axis=1))
                if seed == 0:
                    example_predictions[(group, model)] = prediction[len(frame_ids) // 2, :, :2].copy()
            if frames != _FRAMES_PER_SEED:
                raise ValueError(f"{model}/seed_{seed}: expected 2958 scored frames, got {frames}")
            seed_node_means.append(node_sums / frames)

        means = np.mean(seed_node_means, axis=0)
        stds = np.std(seed_node_means, axis=0, ddof=1)
        node_profiles.extend(
            dict(model=model, node=node + 1, mean_mm=float(means[node]),
                 seed_std_mm=float(stds[node]), frames_per_seed=_FRAMES_PER_SEED)
            for node in range(_NODES)
        )
        frame_errors[model] = np.sort(np.concatenate(model_frame_errors))

    thresholds = np.linspace(0.0, 6.0, 61)
    actual_max = max(float(values[-1]) for values in frame_errors.values())
    if actual_max > thresholds[-1]:
        thresholds = np.append(thresholds, actual_max)
    frame_error_cdf = []
    for model in models:
        values = frame_errors[model]
        fractions = np.searchsorted(values, thresholds, side="right") / len(values)
        frame_error_cdf.extend(
            dict(model=model, error_mm=float(threshold), fraction=float(fraction),
                 n_predictions=int(len(values)))
            for threshold, fraction in zip(thresholds, fractions)
        )

    shape_examples = []
    example_frames = []
    for group in groups:
        target, frame_ids = references[group]
        middle = len(frame_ids) // 2
        frame_id = int(frame_ids[middle])
        example_frames.append(dict(group=group, frame_id=frame_id, scored_index=middle))
        shapes = [("GT", target[middle, :, :2])]
        shapes.extend((model, example_predictions[(group, model)]) for model in models)
        for model, xy in shapes:
            shape_examples.extend(
                dict(group=group, frame_id=frame_id, model=model, node=node + 1,
                     x_mm=float(point[0]), y_mm=float(point[1]))
                for node, point in enumerate(xy)
            )

    return {
        "node_profiles": node_profiles,
        "frame_error_cdf": frame_error_cdf,
        "shape_examples": shape_examples,
        "notes": {
            "model_order": models,
            "seeds": list(_SEEDS),
            "frames_per_sequence": {group: len(references[group][1]) for group in groups},
            "node_profiles": (
                "Corresponding-node Euclidean errors in mm use all 3 coordinates. "
                "Pool all 2958 frames per seed, then take the mean and sample SD "
                "(ddof=1) across five seeds. Nodes are 1-based in saved order. "
                "Seed SD describes training randomness on the fixed split."
            ),
            "frame_error_cdf": (
                "Fraction of per-frame mean-node errors <= error_mm, pooling all "
                "five seeds: 14790 repeated predictions of 2958 scored frames per model. "
                "This CDF is descriptive; repeated predictions and temporal frames "
                "are not independent samples. All models share linspace(0, 6, 61), "
                "plus the global observed maximum when it exceeds 6 mm."
            ),
            "shape_examples": (
                "Seed 0; scored-frame index n // 2 per sequence, using chronological "
                "frame order (later central frame for even n). Coordinates are saved "
                "x/y in mm. GT is target_mm from the first model in plan order, "
                "emitted once per sequence after exact target/frame-ID equality "
                "checks across all models and seeds. GT denotes the saved visual "
                "skeleton label."
            ),
            "shape_seed": 0,
            "shape_frames": example_frames,
        },
    }
