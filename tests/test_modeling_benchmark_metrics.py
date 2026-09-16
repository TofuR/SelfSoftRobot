"""Scientific contracts for physical metrics and independent-group inference."""

import itertools
import json
import unittest

import numpy as np

from src.evaluation.modeling_benchmark_metrics import (
    compare_models,
    mask_metrics,
    render_tube,
    skeleton_metrics,
)


class SkeletonMetricsTests(unittest.TestCase):
    def test_physical_units_node_rmse_and_per_frame_max(self):
        target = np.zeros((2, 2, 3))
        pred = np.array([[[0, 0, 0], [3, 4, 0]], [[0, 0, 12], [0, 0, 0]]])
        actual = skeleton_metrics(pred, target)
        expected = {
            "mean_node_mm": [2.5, 6], "node_rmse_mm": np.sqrt([12.5, 72]),
            "endpoint_mm": [5, 0], "max_node_mm": [5, 12], "chamfer_mm": [1.25, 3],
        }
        self.assertEqual(set(actual), set(expected))
        for name, values in expected.items():
            np.testing.assert_allclose(actual[name], values)
            self.assertEqual(actual[name].shape, (2,))

    def test_identical_single_node_and_translation(self):
        skeleton = np.array([[[1., 2., 3.]], [[4., 5., 6.]]])
        for values in skeleton_metrics(skeleton, skeleton).values():
            np.testing.assert_array_equal(values, [0, 0])
        for values in skeleton_metrics(skeleton + [3, 4, 0], skeleton).values():
            np.testing.assert_allclose(values, [5, 5])

    def test_chamfer_is_symmetric_unsquared_and_averages_both_directions(self):
        pred = np.array([[[0, 0, 0], [2, 0, 0]]])
        target = np.array([[[0, 0, 0], [6, 0, 0]]])
        # pred->target mean=1, target->pred mean=2: Chamfer=1.5 mm.
        self.assertEqual(skeleton_metrics(pred, target)["chamfer_mm"][0], 1.5)
        self.assertEqual(skeleton_metrics(target, pred)["chamfer_mm"][0], 1.5)

    def test_reversal_changes_correspondence_and_endpoint_but_not_chamfer(self):
        target = np.array([[[0, 0, 0], [0, 0, 3], [0, 0, 10]]])
        result = skeleton_metrics(target[:, ::-1], target)
        self.assertEqual(result["endpoint_mm"][0], 10)
        self.assertEqual(result["chamfer_mm"][0], 0)
        self.assertAlmostEqual(result["mean_node_mm"][0], 20 / 3)

    def test_input_validation(self):
        good = np.zeros((2, 4, 3))
        bad = [np.zeros((4, 3)), np.zeros((0, 4, 3)), np.zeros((2, 0, 3)),
               np.zeros((2, 4, 2)), np.zeros((1, 4, 3)),
               np.full_like(good, np.nan), np.full_like(good, np.inf),
               good.astype(complex), np.full(good.shape, "0"), np.zeros(())]
        for value in bad:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises(ValueError):
                    skeleton_metrics(value, good)


class MaskMetricsTests(unittest.TestCase):
    def test_overlap_precision_recall_and_dice(self):
        pred = np.array([[1, 1, 1], [0, 0, 0]])
        target = np.array([[1, 0, 0], [1, 0, 0]])
        result = mask_metrics(pred, target, boundary_tolerance_px=0)
        self.assertAlmostEqual(result["iou"], 1 / 4)
        self.assertAlmostEqual(result["dice"], 2 / 5)
        self.assertAlmostEqual(result["precision"], 1 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 2)
        self.assertAlmostEqual(result["boundary_f1"], 2 / 5)
        self.assertEqual(result["boundary_tolerance_px"], 0)

    def test_empty_conventions(self):
        empty = np.zeros((3, 3), bool)
        full = np.ones((3, 3), bool)
        for pred, target, score in [(empty, empty, 1), (empty, full, 0), (full, empty, 0)]:
            result = mask_metrics(pred, target)
            for key in ("iou", "dice", "precision", "recall", "boundary_f1"):
                self.assertEqual(result[key], score)
                self.assertIsInstance(result[key], float)

    def test_full_masks_and_image_border_are_real_boundaries(self):
        full = np.ones((5, 5), bool)
        ring = full.copy()
        ring[1:-1, 1:-1] = False
        self.assertEqual(mask_metrics(full, full, 0)["boundary_f1"], 1)
        self.assertEqual(mask_metrics(full, ring, 0)["boundary_f1"], 1)
        self.assertEqual(mask_metrics([[1]], [[1]], 0)["boundary_f1"], 1)

    def test_inclusive_euclidean_boundary_tolerance(self):
        pred = np.zeros((8, 8), bool)
        target = pred.copy()
        pred[2, 2], target[4, 2] = True, True
        self.assertEqual(mask_metrics(pred, target)["boundary_f1"], 1)
        self.assertEqual(mask_metrics(pred, target, 1.999)["boundary_f1"], 0)
        target[:] = False
        target[3, 3] = True
        self.assertEqual(mask_metrics(pred, target, 1)["boundary_f1"], 0)
        self.assertEqual(mask_metrics(pred, target, np.sqrt(2))["boundary_f1"], 1)

    def test_boundary_f1_is_harmonic_mean_of_both_matches(self):
        pred = np.zeros((10, 10), bool)
        target = pred.copy()
        pred[1, 1] = pred[8, 8] = target[1, 1] = True
        self.assertAlmostEqual(mask_metrics(pred, target, 0)["boundary_f1"], 2 / 3)

    def test_input_validation(self):
        good = np.ones((3, 3), bool)
        for bad in [np.zeros((2, 3)), np.zeros((0, 3)), np.zeros((1, 3, 3)),
                    np.full((3, 3), .5), np.full((3, 3), 255),
                    np.full((3, 3), np.nan), np.full((3, 3), -1), good.astype(complex)]:
            with self.subTest(mask=bad):
                with self.assertRaises(ValueError):
                    mask_metrics(bad, good)
        for tolerance in [-1, np.nan, np.inf, [2], True, "2"]:
            with self.subTest(tolerance=tolerance):
                with self.assertRaises(ValueError):
                    mask_metrics(good, good, tolerance)


class RenderTubeTests(unittest.TestCase):
    def test_homography_units_axes_and_fixed_radius(self):
        skeleton = np.array([[0., 0., 100], [2., 0., -100]])
        h = np.array([[2, 0, 3], [0, 3, 4], [0, 0, 1.]])
        actual = render_tube(skeleton, h, (10, 12), 1)
        yy, xx = np.mgrid[:10, :12]
        expected = np.hypot(xx - np.clip(xx, 3, 7), yy - 4) <= 1
        self.assertEqual(actual.dtype, np.dtype(bool))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, render_tube(skeleton, -h * 1e-20, (10, 12), 1))

    def test_perspective_division(self):
        h = np.array([[1., 0, 0], [0, 1, 0], [.5, 0, 1]])
        actual = render_tube([[2, 4, 0]], h, (5, 5), 0)
        expected = np.zeros((5, 5), bool)
        expected[2, 1] = True
        np.testing.assert_array_equal(actual, expected)

    def test_single_and_repeated_nodes_fractional_radius(self):
        skeleton = [[2.5, 2.5, 0]]
        self.assertFalse(render_tube(skeleton, np.eye(3), (6, 6), .7).any())
        actual = render_tube(skeleton, np.eye(3), (6, 6), .71)
        self.assertEqual(actual.sum(), 4)
        np.testing.assert_array_equal(actual, render_tube(skeleton * 3, np.eye(3), (6, 6), .71))

    def test_continuous_diagonal_and_round_ends(self):
        expected = np.zeros((6, 6), bool)
        expected[np.arange(1, 5), np.arange(1, 5)] = True
        for radius in (0, .1):
            actual = render_tube([[1, 1, 0], [4, 4, 0]], np.eye(3), (6, 6), radius)
            np.testing.assert_array_equal(actual, expected)

    def test_off_image_geometry_is_not_clamped_but_crossing_segment_is_visible(self):
        self.assertFalse(render_tube([[-3, 1, 0], [-3, 4, 0]], np.eye(3), (6, 6), 1).any())
        actual = render_tube([[-10, 2, 0], [10, 2, 0]], np.eye(3), (5, 5), 0)
        expected = np.zeros((5, 5), bool)
        expected[2] = True
        np.testing.assert_array_equal(actual, expected)
        # An off-image centre can still contribute a tube cap.
        self.assertTrue(render_tube([[-.5, 2, 0]], np.eye(3), (5, 5), .6)[2, 0])

    def test_validation_and_projective_horizon(self):
        defaults = dict(skeleton=[[1, 1, 0]], homography=np.eye(3), shape=(5, 5), radius_px=1)
        bad_inputs = [dict(skeleton=[]), dict(skeleton=[[1, 2]]),
                      dict(skeleton=[[np.nan, 0, 0]]), dict(homography=np.zeros((3, 3))),
                      dict(homography=np.eye(2)), dict(homography=np.full((3, 3), np.inf)),
                      dict(shape=(0, 5)), dict(shape=(5., 5)), dict(shape=(True, 5)),
                      dict(shape=np.array(5)),
                      dict(radius_px=-1), dict(radius_px=np.inf), dict(radius_px=[1])]
        for bad in bad_inputs:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    render_tube(**(defaults | bad))
        h = np.array([[1., 0, 1], [0, 1, 0], [1, 0, 0]])
        for skeleton in [[[0, 1, 0]], [[-1, 1, 0], [1, 1, 0]]]:
            with self.assertRaisesRegex(ValueError, "horizon"):
                render_tube(skeleton, h, (5, 5), 1)


def make_records(differences, seeds=(0,), metric="mean_node_mm", model="candidate"):
    records = []
    for group, difference in enumerate(differences):
        for seed in seeds:
            baseline = 10 + group + seed / 100
            for name, value in [("reference", baseline), (model, baseline + difference)]:
                records.append({"model": name, "seed": seed, "group": f"g{group:03d}",
                                "metrics": {metric: value}})
    return records


class PairedStatisticsTests(unittest.TestCase):
    def compare(self, differences, **kwargs):
        return compare_models(make_records(differences), "reference", ["mean_node_mm"], **kwargs)

    def test_exact_known_p_and_direction(self):
        for difference, status in [(-2, "improved"), (2, "worse")]:
            result = self.compare([difference] * 6)["comparisons"][0]
            self.assertEqual(result["mean_difference"], difference)
            self.assertEqual(result["ci_95"], [difference, difference])
            self.assertEqual(result["p_value"], 2 / 64)
            self.assertEqual(result["p_value_holm"], 2 / 64)
            self.assertEqual(result["permutation_method"], "exact")
            self.assertEqual(result["permutation_samples"], 64)
            self.assertTrue(result["significant"])
            self.assertEqual(result["status"], status)

    def test_higher_metric_and_explicit_direction(self):
        records = make_records([.1] * 6, metric="dice")
        result = compare_models(records, "reference", ["dice"])["comparisons"][0]
        self.assertEqual(result["status"], "improved")
        result = compare_models(records, "reference", {"dice": "lower"})["comparisons"][0]
        self.assertEqual(result["status"], "worse")

    def test_small_n_is_inconclusive_despite_interval_excluding_zero(self):
        for n in [1, 5]:
            result = self.compare([-1] * n)["comparisons"][0]
            self.assertEqual(result["ci_95"], [-1, -1])
            self.assertEqual(result["status"], "inconclusive")
            self.assertFalse(result["significant"])

    def test_equal_models_and_balanced_differences(self):
        for differences in [[0] * 6, [-1, 1] * 3]:
            result = self.compare(differences)["comparisons"][0]
            self.assertEqual(result["p_value"], 1)
            self.assertEqual(result["mean_difference"], 0)
            self.assertEqual(result["status"], "no_evidence")
            self.assertFalse(result["significant"])

    def test_exact_permutation_matches_independent_enumeration_with_ties(self):
        differences = np.array([-3., -2, 0, 1, 1, 2])
        statistics = [abs(np.mean(np.array(signs) * differences))
                      for signs in itertools.product((-1, 1), repeat=6)]
        expected = np.mean(np.array(statistics) >= abs(differences.mean()) - 1e-14)
        result = self.compare(differences)["comparisons"][0]
        self.assertEqual(result["p_value"], expected)

    def test_average_seeds_then_equal_groups_not_seed_pseudoreplication(self):
        records = make_records([-1] * 5, seeds=range(20))
        result = compare_models(records, "reference", ["mean_node_mm"])["comparisons"][0]
        self.assertEqual(result["n_groups"], 5)
        self.assertEqual(result["p_value"], 2 / 32)
        self.assertEqual(result["status"], "inconclusive")
        # Different seed counts across groups must not change group weights.
        records = make_records([0, 0], seeds=(0, 1, 2))
        records = [r for r in records if r["group"] == "g000" or r["seed"] == 0]
        for record in records:
            if record["model"] == "candidate":
                record["metrics"]["mean_node_mm"] += 3 if record["group"] == "g000" else 9
        result = compare_models(records, "reference", ["mean_node_mm"])["comparisons"][0]
        self.assertEqual(result["mean_difference"], 6)
        self.assertEqual([p["difference"] for p in result["paired_groups"]], [3, 9])

    def test_opposing_seed_effects_are_averaged_before_inference(self):
        records = make_records([0] * 6, seeds=(0, 1))
        for record in records:
            if record["model"] == "candidate":
                record["metrics"]["mean_node_mm"] += -2 if record["seed"] == 0 else 2
        result = compare_models(records, "reference", ["mean_node_mm"])["comparisons"][0]
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result["ci_95"], [0, 0])
        self.assertEqual(result["p_value"], 1)

    def test_holm_covers_models_and_metrics_and_controls_significance(self):
        records = make_records([-1] * 6)
        for record in records:
            record["metrics"]["endpoint_mm"] = record["metrics"]["mean_node_mm"]
        records += [dict(record, model="other") for record in records if record["model"] == "candidate"]
        report = compare_models(records, "reference", ["mean_node_mm", "endpoint_mm"])
        self.assertEqual(report["n_comparisons"], 4)
        for result in report["comparisons"]:
            self.assertEqual(result["p_value"], .03125)
            self.assertEqual(result["p_value_holm"], .125)
            self.assertEqual(result["ci_95"], [-1, -1])
            self.assertFalse(result["significant"])
            self.assertEqual(result["status"], "no_evidence")

    def test_holm_step_down_with_unequal_p_values(self):
        records = make_records([-1] * 8)
        for record in records:
            record["metrics"]["equal"] = 0.
        results = compare_models(records, "reference", ["equal", "mean_node_mm"])["comparisons"]
        by_metric = {r["metric"]: r for r in results}
        self.assertEqual(by_metric["mean_node_mm"]["p_value_holm"], 4 / 256)
        self.assertTrue(by_metric["mean_node_mm"]["significant"])
        self.assertEqual(by_metric["equal"]["p_value_holm"], 1)

    def test_paired_group_bootstrap_bounds_and_reproducibility(self):
        records = make_records([-4, -3, -2, -1, 0, 1])
        first = compare_models(records, "reference", ["mean_node_mm"], seed=12)
        reordered = compare_models(records[::-1], "reference", ["mean_node_mm"], seed=12)
        self.assertEqual(first, reordered)
        result = first["comparisons"][0]
        low, high = result["ci_95"]
        self.assertGreaterEqual(low, -4)
        self.assertLessEqual(high, 1)
        self.assertLess(low, result["mean_difference"])
        self.assertGreater(high, result["mean_difference"])
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)

    def test_monte_carlo_branch_reproducible_and_p_never_zero(self):
        first = self.compare([-1] * 17, seed=4)
        self.assertEqual(first, self.compare([-1] * 17, seed=4))
        result = first["comparisons"][0]
        self.assertEqual(result["permutation_method"], "monte_carlo")
        self.assertEqual(result["permutation_samples"], 50000)
        self.assertGreater(result["p_value"], 0)
        self.assertLess(result["p_value"], .001)
        self.assertEqual(result["status"], "improved")

    def test_bootstrap_agrees_with_known_binomial_distribution(self):
        # Resampling six differences [0,0,0,0,0,6] gives a mean distributed
        # Binomial(6,1/6), whose 2.5% and 97.5% quantiles are exactly 0 and 3.
        result = self.compare([0, 0, 0, 0, 0, 6], seed=9)["comparisons"][0]
        self.assertEqual(result["ci_95"], [0, 3])
        self.assertEqual(result["mean_difference"], 1)
        self.assertEqual(result["p_value"], 1)
        self.assertFalse(result["significant"])

    def test_group_seed_coverage_mismatch_raises_even_with_equal_counts(self):
        records = make_records([-1] * 6, seeds=(0, 1))
        for field, value in [("group", "extra"), ("seed", 99)]:
            bad = [dict(r) for r in records]
            bad[-1][field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                    compare_models(bad, "reference", ["mean_node_mm"])
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            compare_models(records[:-1], "reference", ["mean_node_mm"])

    def test_duplicate_frame_arrays_and_invalid_records_rejected(self):
        records = make_records([-1] * 6)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_models(records + records[:1], "reference", ["mean_node_mm"])
        for value in [[1, 2], np.array(1 + 2j), np.nan, np.inf, "1", True]:
            bad = [dict(r, metrics=dict(r["metrics"])) for r in records]
            bad[0]["metrics"]["mean_node_mm"] = value
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    compare_models(bad, "reference", ["mean_node_mm"])
        for metrics in [[], ["missing"], ["mean_node_mm"] * 2, "mean_node_mm",
                        {"mean_node_mm": "sideways"}]:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    compare_models(records, "reference", metrics)
        for bad_records in [[], records[:1], [{}], [dict(records[0], group=np.nan)]]:
            with self.assertRaises(ValueError):
                compare_models(bad_records, "reference", ["mean_node_mm"])
        for seed in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                compare_models(records, "reference", ["mean_node_mm"], seed=seed)


if __name__ == "__main__":
    unittest.main()
