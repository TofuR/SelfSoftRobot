"""Exact EDT equivalence and an opt-in CPU benchmark using synthetic masks.

Run tests with ``python -m unittest tests.test_modeling_mask_fast_boundary``.
Run the synthetic benchmark with
``python -m tests.test_modeling_mask_fast_boundary --benchmark``.
Neither path reads dataset masks or imports training/model code.
"""
import json
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from src.evaluation import modeling_benchmark_metrics as metrics


def _original_edt_metrics(pred, target, boundary_tolerance_px=2):
    """Original implementation as an independent numerical/timing oracle."""
    pred = metrics._binary_mask(pred, "pred")
    target = metrics._binary_mask(target, "target")
    if pred.shape != target.shape:
        raise ValueError("pred and target masks must have identical shapes")
    tolerance = metrics._nonnegative_scalar(boundary_tolerance_px, "boundary_tolerance_px")
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
    precision = float(np.mean(distance_transform_edt(~t_boundary)[p_boundary] <= tolerance))
    recall = float(np.mean(distance_transform_edt(~p_boundary)[t_boundary] <= tolerance))
    return dict(iou=float(intersection / (n_pred + n_target - intersection)),
                dice=float(2 * intersection / (n_pred + n_target)),
                precision=float(intersection / n_pred), recall=float(intersection / n_target),
                boundary_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
                boundary_tolerance_px=tolerance)


class FastBoundaryTests(unittest.TestCase):
    def assert_equivalent(self, pred, target, tolerances):
        for tolerance in tolerances:
            with self.subTest(shape=pred.shape, tolerance=tolerance):
                self.assertEqual(metrics.mask_metrics(pred, target, tolerance),
                                 _original_edt_metrics(pred, target, tolerance))

    def test_random_masks_and_noninteger_tolerances(self):
        rng = np.random.default_rng(219)
        for shape in ((1, 1), (1, 19), (23, 1), (7, 11), (31, 47), (128, 96)):
            for density in (.01, .15, .5, .95):
                pred = rng.random(shape) < density
                target = rng.random(shape) < 1 - density
                self.assert_equivalent(pred, target, (0., .2, .999, 1., 1.4, 1.5, 2., 2.5, 3.9, 4., 4.1, 8.))

    def test_edges_corners_full_empty_and_thin_masks(self):
        empty = np.zeros((19, 23), bool)
        full = np.ones_like(empty)
        border = full.copy()
        border[1:-1, 1:-1] = False
        line = empty.copy()
        line[0, :] = True
        corners = empty.copy()
        corners[0, 0] = corners[0, -1] = corners[-1, 0] = corners[-1, -1] = True
        rectangle = empty.copy()
        rectangle[:10, :15] = True
        diagonal = np.eye(19, 23, dtype=bool)
        masks = (empty, full, border, line, corners, rectangle, diagonal)
        for pred in masks:
            for target in masks:
                self.assert_equivalent(pred, target, (0., .9, np.sqrt(2), 2., 3.1, 4., 6.))

    def test_inclusive_lattice_distances_and_adjacent_float_thresholds(self):
        for dy, dx in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2), (4, 0)):
            pred = np.zeros((13, 13), bool)
            target = pred.copy()
            pred[3, 3], target[3 + dy, 3 + dx] = True, True
            distance = float(np.sqrt(dx * dx + dy * dy))
            tolerances = [distance, np.nextafter(distance, np.inf)]
            if distance:
                tolerances.append(np.nextafter(distance, 0.))
            self.assert_equivalent(pred, target, tolerances)
            self.assertEqual(metrics.mask_metrics(pred, target, distance)["boundary_f1"], 1.)
            if distance:
                self.assertEqual(metrics.mask_metrics(pred, target, np.nextafter(distance, 0.))["boundary_f1"], 0.)

    def test_small_tolerance_uses_dilation_and_large_tolerance_uses_edt(self):
        pred = np.eye(12, dtype=bool)
        target = np.flipud(pred)
        with patch.object(metrics, "distance_transform_edt", side_effect=AssertionError("EDT on fast path")):
            for tolerance in (0., .5, 2., 3.2, 4.):
                metrics.mask_metrics(pred, target, tolerance)
        with patch.object(metrics, "binary_dilation", side_effect=AssertionError("Dilation on fallback")), \
                patch.object(metrics, "distance_transform_edt", wraps=distance_transform_edt) as edt:
            for tolerance in (np.nextafter(4., np.inf), 5., 20.):
                self.assert_equivalent(pred, target, [tolerance])
            self.assertEqual(edt.call_count, 6)

    def test_noncontiguous_inputs_and_inputs_are_preserved(self):
        rng = np.random.default_rng(41)
        pred = (rng.random((37, 53)) > .8).astype(np.uint8)[::-1, ::2]
        target = (rng.random((37, 53)) > .4).astype(np.float32)[::-1, ::2]
        before_pred, before_target = pred.copy(), target.copy()
        self.assert_equivalent(pred, target, (0., 2., 2.5, 4., 5.))
        np.testing.assert_array_equal(pred, before_pred)
        np.testing.assert_array_equal(target, before_target)


def benchmark_synthetic(repeats=7, frames=4):
    """Compare complete mask_metrics calls; generation is outside timed regions.

    Alternating method order reduces warmup/order bias. Report medians of full
    frame-pass means, rather than selecting a best time. No speed assertion is
    part of the tests because CPU load and mask geometry affect the speedup.
    """
    results = []
    for height, width in ((256, 256), (512, 512), (720, 1280)):
        yy, xx = np.ogrid[:height, :width]
        pairs = []
        for frame in range(frames):
            center = width * (.45 + .12 * np.sin(yy / height * 4 + frame * .3))
            radius = max(3., width * .015)
            valid = (yy >= height * .06) & (yy <= height * .94)
            target = (np.abs(xx - center) <= radius) & valid
            pred = (np.abs(xx - center - 1.5 - np.sin(yy / height * 7)) <= radius * 1.02) & valid
            pairs.append((pred, target))
        functions = {"edt": _original_edt_metrics, "dilation": metrics.mask_metrics}
        for pred, target in pairs:
            assert functions["edt"](pred, target, 2.) == functions["dilation"](pred, target, 2.)
        durations = {name: [] for name in functions}
        for repeat in range(repeats):
            order = ("edt", "dilation") if repeat % 2 == 0 else ("dilation", "edt")
            for name in order:
                start = time.perf_counter_ns()
                for pred, target in pairs:
                    functions[name](pred, target, 2.)
                durations[name].append((time.perf_counter_ns() - start) / 1e6 / frames)
        original, fast = (float(np.median(durations[name])) for name in ("edt", "dilation"))
        results.append(dict(shape=[height, width], tolerance_px=2., synthetic_frames=frames,
                            repeats=repeats, edt_ms_per_mask=original, dilation_ms_per_mask=fast,
                            speedup=original / fast))
    return dict(scope="CPU synthetic curved tubes; complete mask scoring calls", results=results)


if __name__ == "__main__":
    if sys.argv[1:] == ["--benchmark"]:
        print(json.dumps(benchmark_synthetic(), indent=2))
    else:
        unittest.main()
