"""Synthetic evaluation arrays and CPU-only checkpoint/latency contracts."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import stats
import torch

from src.benchmarks.modeling_seed_summary import (
    benchmark_latency, holm_adjust, paired_seed_tests, pool_seed, summarize_seeds,
)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(root, model="candidate", seed=0, offset=0., masks=True, layout="records"):
    directory = root / f"{model}_{seed}"
    directory.mkdir()
    protocol = dict(role="test", dataset_manifest_sha256="fixed-synthetic-split", history=20,
                    eval_stride=1, fold=0, run_kind="formal")
    manifest = dict(status="complete", **protocol, masks_enabled=masks, mask_stride=2,
                    mask_adapter="fixed_radius_planar_tube", radius_mm=8.,
                    boundary_tolerance_px=2., evaluation_code_hashes={"synthetic": "v1"},
                    evidence_level="within_sequence")
    records = []
    for group, errors in (("short", np.array([[1., 3.]])),
                          ("long", np.tile([7., 9.], (9, 1)))):
        n = len(errors)
        target = np.zeros((n, 2, 3))
        target[:, 1, 1] = 20.
        prediction = target.copy()
        prediction[:, :, 0] = errors + offset
        ids = np.arange(20, 20 + n)
        mids = ids[::2] if masks else np.array([], dtype=int)
        arrays = dict(prediction_mm=prediction, target_mm=target, frame_ids=ids,
                      mask_frame_ids=mids, mean_node_mm=np.full(n, -999.))
        if masks:
            for key, maximum in (("iou", 1.), ("dice", .8), ("precision", .9),
                                 ("recall", .7), ("boundary_f1", .6)):
                arrays[f"mask_{key}"] = np.full(len(mids), maximum if group == "long" else 0.)
        np.savez(directory / f"{group}_predictions.npz", **arrays)
        records.append(dict(model=model, seed=seed, group=group, **protocol,
                            frames=n, mask_frames=len(mids),
                            metrics={"mean_node_mm": -999., "node_p95_mm": -999.}))
    if layout == "records":
        _json(directory / "records.json", records)
    elif layout == "metrics_list":
        _json(directory / "metrics.json", records)
    elif layout == "metrics_records":
        _json(directory / "metrics.json", {"records": records})
    elif layout == "metrics_run":
        manifest.update(model=model, seed=seed)
        _json(directory / "metrics.json", {"mean_node_mm": -999.})
    elif layout == "metrics_config":
        manifest["run"] = "run"
        (directory / "run").mkdir()
        _json(directory / "run/resolved_config.json", dict(model=model, seed=seed, **protocol))
        _json(directory / "metrics.json", {"metrics": {"node_p95_mm": -999.}})
    else:
        raise AssertionError(layout)
    _json(directory / "evaluation_manifest.json", manifest)
    (directory / "COMPLETE").write_text("complete\n")
    return directory


def _change_npz(path, change):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    change(arrays)
    np.savez(path, **arrays)


def _paired_rows(differences, seeds=None):
    seeds = list(range(len(differences))) if seeds is None else seeds
    rows = []
    for seed, difference in zip(seeds, differences):
        for model, value in (("reference", 10.), ("candidate", 10. + difference)):
            rows.append(dict(model=model, seed=seed, status="complete", run_kind="formal",
                             protocol={"role": "test"}, coverage=[{"group": "fixed"}],
                             metrics={"mean_node_mm": value, "mask_dice": value / 20}))
    return rows


class PoolSeedTests(unittest.TestCase):
    def test_frame_weighting_rmse_and_quantiles_come_from_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = pool_seed(_evaluation(Path(tmp)))
        errors = np.concatenate(([1., 3.], np.tile([7., 9.], 9)))
        metrics = row["metrics"]
        self.assertEqual(row["frames"], 10)
        self.assertEqual(row["mask_frames"], 6)
        self.assertAlmostEqual(metrics["mean_node_mm"], errors.mean())
        self.assertNotAlmostEqual(metrics["mean_node_mm"], (2. + 8.) / 2)
        self.assertAlmostEqual(metrics["node_rmse_mm"], np.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(metrics["endpoint_mm"], (3. + 9. * 9) / 10)
        self.assertAlmostEqual(metrics["node_p95_mm"], np.quantile(errors, .95))
        self.assertAlmostEqual(metrics["node_p50_mm"], np.quantile(errors, .5))
        self.assertAlmostEqual(metrics["endpoint_p95_mm"], np.quantile([3.] + [9.] * 9, .95))
        self.assertNotAlmostEqual(metrics["node_p95_mm"], np.mean([2.9, 8.9]))
        self.assertAlmostEqual(metrics["mask_iou"], 5 / 6)
        self.assertAlmostEqual(metrics["mask_dice"], 4 / 6)
        self.assertAlmostEqual(metrics["mask_boundary_f1"], .5)
        self.assertIn("chamfer_mm", metrics)

    def test_metrics_json_exports_and_config_metadata_are_supported(self):
        for layout in ("metrics_list", "metrics_records", "metrics_run", "metrics_config"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                row = pool_seed(_evaluation(Path(tmp), seed=7, layout=layout))
                self.assertEqual(row["seed"], 7)
                self.assertEqual(row["model"], "candidate")
                self.assertAlmostEqual(row["metrics"]["mean_node_mm"], 7.4)

    def test_invalid_or_partial_arrays_fail_instead_of_dropping_frames(self):
        changes = {
            "nonfinite": lambda a: a["prediction_mm"].__setitem__((0, 0, 0), np.nan),
            "duplicate IDs": lambda a: a["frame_ids"].__setitem__(1, a["frame_ids"][0]),
            "missing mask metric": lambda a: a.pop("mask_dice"),
            "invalid score": lambda a: a["mask_iou"].__setitem__(0, 1.1),
            "mask cadence": lambda a: a["mask_frame_ids"].__setitem__(0, 21),
            "inconsistent nodes": lambda a: a.update(prediction_mm=a["prediction_mm"][:, :1],
                                                     target_mm=a["target_mm"][:, :1]),
            "wrong frame count": lambda a: a.update(frame_ids=a["frame_ids"][:-1]),
        }
        for label, change in changes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                directory = _evaluation(Path(tmp))
                _change_npz(directory / "long_predictions.npz", change)
                with self.assertRaises(ValueError):
                    pool_seed(directory)

    def test_mask_metrics_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = pool_seed(_evaluation(Path(tmp), masks=False))
        self.assertEqual(row["mask_frames"], 0)
        self.assertFalse(any(key.startswith("mask_") for key in row["metrics"]))

    def test_manifest_record_and_file_mismatches_are_rejected(self):
        for fault in ("incomplete", "mixed seeds", "missing sequence", "wrong count"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                directory = _evaluation(Path(tmp))
                if fault == "incomplete":
                    path = directory / "evaluation_manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["status"] = "failed"
                    _json(path, manifest)
                elif fault == "missing sequence":
                    (directory / "short_predictions.npz").unlink()
                else:
                    path = directory / "records.json"
                    records = json.loads(path.read_text())
                    records[0]["seed" if fault == "mixed seeds" else "frames"] = 99
                    _json(path, records)
                with self.assertRaises(ValueError):
                    pool_seed(directory)


class SeedSummaryTests(unittest.TestCase):
    def test_fixed_seed_order_equal_weight_and_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directories = [_evaluation(root, seed=seed, offset=float(seed)) for seed in (1, 0, 2)]
            output = root / "summary.json"
            report = summarize_seeds(directories, [2, 0, 1], metrics=["mean_node_mm"], output=output)
            self.assertEqual(json.loads(output.read_text()), report)
        self.assertEqual([row["seed"] for row in report["per_seed"]], [2, 0, 1])
        summary = report["summary"][0]
        np.testing.assert_allclose(summary["values"], [9.4, 7.4, 8.4])
        self.assertAlmostEqual(summary["mean"], 8.4)
        self.assertAlmostEqual(summary["std"], 1.)
        self.assertEqual(report["status"], "complete")
        self.assertIn("training randomness", report["statistical_scope"])
        self.assertIn("does not establish cross-sequence", report["statistical_scope"])

    def test_missing_and_entirely_absent_models_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _evaluation(Path(tmp), seed=0)
            report = summarize_seeds([directory], [0, 1, 2, 3, 4],
                                     models=["candidate", "reference"], reference="reference",
                                     metrics=["mean_node_mm"])
        self.assertEqual(len(report["per_seed"]), 10)
        self.assertEqual(sum(row["status"] == "missing" for row in report["per_seed"]), 9)
        self.assertTrue(all(row["mean"] is None and row["std"] is None for row in report["summary"]))
        comparison = report["paired_tests"]["comparisons"][0]
        self.assertEqual(comparison["status"], "incomplete")
        self.assertIsNone(comparison["wilcoxon"])
        self.assertFalse(comparison["significant"])
        empty = summarize_seeds([], [0, 1], models=["reference"], metrics=["mean_node_mm"])
        self.assertEqual(len(empty["per_seed"]), 2)

    def test_duplicate_unplanned_seeds_and_invalid_design_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = _evaluation(Path(tmp), seed=0)
            for dirs, seeds in (([directory, directory], [0]), ([directory], [1]),
                                ([directory], [0, 0]), ([directory], []), ([directory], [True])):
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    summarize_seeds(dirs, seeds, metrics=["mean_node_mm"])

    def test_different_split_targets_frame_ids_or_protocol_cannot_be_paired(self):
        for fault in ("split", "target", "frames", "radius"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                first = _evaluation(root, seed=0)
                second = _evaluation(root, seed=1)
                if fault in ("split", "radius"):
                    path = second / "evaluation_manifest.json"
                    manifest = json.loads(path.read_text())
                    manifest["dataset_manifest_sha256" if fault == "split" else "radius_mm"] = (
                        "other-split" if fault == "split" else 7.)
                    _json(path, manifest)
                    if fault == "split":
                        path = second / "records.json"
                        records = json.loads(path.read_text())
                        for row in records:
                            row["dataset_manifest_sha256"] = "other-split"
                        _json(path, records)
                else:
                    def change(arrays):
                        if fault == "target":
                            arrays["target_mm"] += .1
                        else:
                            arrays["frame_ids"] += 1
                            arrays["mask_frame_ids"] += 1
                    _change_npz(second / "long_predictions.npz", change)
                with self.assertRaisesRegex(ValueError, "fixed split"):
                    summarize_seeds([first, second], [0, 1], metrics=["mean_node_mm"])

    def test_one_seed_has_no_sample_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = summarize_seeds([_evaluation(Path(tmp))], [0], metrics=["mean_node_mm"])
        self.assertIsNone(report["summary"][0]["std"])


class PairedSeedTests(unittest.TestCase):
    def test_five_seed_exact_wilcoxon_floor_and_signed_effects(self):
        seeds = [9, 2, 5, 1, 7]
        rows = _paired_rows([-1., -2., -3., -4., -5.], seeds)
        report = paired_seed_tests(list(reversed(rows)), seeds, "reference", ["mean_node_mm"])
        row = report["comparisons"][0]
        self.assertEqual(row["differences"], [-1., -2., -3., -4., -5.])
        self.assertEqual(row["mean_difference"], -3.)
        self.assertEqual(row["mean_improvement"], 3.)
        self.assertEqual(row["wilcoxon"]["p_value"], .0625)
        self.assertEqual(row["wilcoxon"]["minimum_attainable_p"], .0625)
        self.assertEqual(row["wilcoxon"]["p_holm"], .0625)
        self.assertEqual(row["wilcoxon"]["rank_biserial"], -1.)
        self.assertFalse(row["significant"])

    def test_ties_and_zeros_match_brute_force_sign_enumeration(self):
        for differences in ([-1., -1., -2., 0., 3.], [1., 1., 1., 1., 1.],
                            [-3., 2., 4., -1., 5.], [0., 0., 0., 0., 0.]):
            with self.subTest(differences=differences):
                d = np.array([value for value in differences if value != 0])
                ranks = stats.rankdata(abs(d))
                total = ranks.sum()
                observed = min(ranks[d > 0].sum(), ranks[d < 0].sum())
                distribution = [sum(rank * sign for rank, sign in zip(ranks, signs))
                                for signs in itertools.product((0, 1), repeat=len(d))]
                expected = sum(min(value, total - value) <= observed for value in distribution) / len(distribution)
                row = paired_seed_tests(_paired_rows(differences), range(5), "reference",
                                        ["mean_node_mm"], exploratory_t=True)["comparisons"][0]
                self.assertAlmostEqual(row["wilcoxon"]["p_value"], expected)
                self.assertEqual(row["wilcoxon"]["n_zero"], 5 - len(d))
                if not len(d):
                    self.assertEqual(row["wilcoxon"]["p_value"], 1.)
                    self.assertIsNone(row["cohen_dz"])
                    self.assertEqual(row["t_test"]["status"], "undefined_zero_variance")
                    self.assertIsNone(row["t_test"]["p_value"])

    def test_holm_adjustment_order_monotonicity_and_missing_hypotheses(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03, .002]), [.06, .03, .06, .008])
        self.assertEqual(holm_adjust([.01, None, .02]), [.03, None, .04])
        self.assertEqual(holm_adjust([]), [])
        for invalid in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                holm_adjust([invalid])

    def test_holm_family_covers_models_metrics_and_missing_models(self):
        report = paired_seed_tests(_paired_rows([1., 2., 3., 4., 5.]), range(5), "reference",
                                   ["mean_node_mm", "mask_dice"],
                                   models=["reference", "candidate", "absent"])
        self.assertEqual(len(report["comparisons"]), 4)
        for row in report["comparisons"][:2]:
            self.assertEqual(row["wilcoxon"]["family_size"], 4)
            self.assertEqual(row["wilcoxon"]["p_holm"], .25)
        dice = report["comparisons"][1]
        self.assertEqual(dice["direction"], "higher")
        self.assertGreater(dice["mean_improvement"], 0.)

    def test_paired_t_is_optional_exploratory_and_matches_scipy(self):
        d = [1., 2., 3., 4., 5.]
        rows = _paired_rows(d)
        primary = paired_seed_tests(rows, range(5), "reference", ["mean_node_mm"])["comparisons"][0]
        self.assertIsNone(primary["t_test"])
        row = paired_seed_tests(rows, range(5), "reference", ["mean_node_mm"],
                                exploratory_t=True)["comparisons"][0]
        expected = stats.ttest_rel(np.array(d) + 10., np.full(5, 10.))
        self.assertAlmostEqual(row["t_test"]["p_value"], expected.pvalue)
        self.assertTrue(row["t_test"]["exploratory"])
        self.assertIn("normal", row["t_test"]["assumption"])
        self.assertTrue(row["t_test"]["reject_holm"])
        self.assertFalse(row["significant"])
        lo, hi = row["t_test"]["mean_difference_ci95"]
        self.assertLess(lo, 3.)
        self.assertGreater(hi, 3.)

    def test_missing_pairs_never_use_available_case_test(self):
        rows = _paired_rows([1., 2., 3., 4., 5.])[:-1]
        row = paired_seed_tests(rows, range(5), "reference", ["mean_node_mm"])["comparisons"][0]
        self.assertEqual(row["missing_seeds"], [4])
        self.assertIsNone(row["wilcoxon"])
        self.assertIsNone(row["mean_difference"])

    def test_validation_and_smoke_tests_are_diagnostic(self):
        for kind in ("validation", "smoke"):
            rows = _paired_rows(range(1, 9))
            for row in rows:
                if kind == "validation":
                    row["protocol"]["role"] = "val"
                else:
                    row["run_kind"] = "smoke"
            result = paired_seed_tests(rows, range(8), "reference", ["mean_node_mm"])["comparisons"][0]
            self.assertLess(result["wilcoxon"]["p_value"], .05)
            self.assertEqual(result["status"], "diagnostic")
            self.assertFalse(result["significant"])


class LatencyTests(unittest.TestCase):
    def _checkpoint(self, root, name="window_mlp", history=20):
        from src.benchmarks.modeling_models import make_model
        config = dict(model=name, seed=3, history=history, hidden=4, dt=.2, threads=1)
        geometry = (dict(action_dim=4, n_nodes=3, window_size=history, n_play=1, n_maxwell=1,
                         n_bend_modes=2, section_intervals=[1, 1], residual_mode="none",
                         reference_kind="linear", dt=.2) if name == "hov" else None)
        model, metadata = make_model(name, config, normalization=([1., 2., 3.], 2.),
                                     geometry_config=geometry)
        run = root / name
        run.mkdir()
        torch.save(dict(schema="shape_modeling_checkpoint_v1", model=name, config=config,
                        state_dict=model.state_dict(), geometry_config=metadata,
                        center=[1., 2., 3.], scale=2., selected_epoch=4), run / "best_eval_model.pt")
        (run / "COMPLETE").write_text("complete\n")
        return run

    def test_actual_cpu_checkpoint_shapes_timing_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._checkpoint(root)
            output = root / "latency.json"
            before_threads = torch.get_num_threads()
            before_rng = torch.random.get_rng_state().clone()
            with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU only")), \
                    patch("torch.cuda.is_available", side_effect=AssertionError("No GPU query")):
                result = benchmark_latency(run, "cpu", output, warmup=1, repeats=3)
            self.assertEqual(json.loads(output.read_text()), result)
            self.assertEqual(torch.get_num_threads(), before_threads)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
        self.assertEqual(result["history"], 20)
        self.assertFalse(result["training_cache_used"])
        self.assertFalse(result["cuda_synchronized"])
        self.assertIn("full-window", result["prediction_semantics"])
        self.assertEqual([row["batch_size"] for row in result["measurements"]], [1, 256])
        for row in result["measurements"]:
            batch = row["batch_size"]
            self.assertEqual(row["input_shape"], [batch, 20, 4])
            self.assertEqual(row["output_shape"], [batch, 15, 3])
            self.assertEqual(len(row["samples_ms"]), 3)
            self.assertGreater(row["p50_ms"], 0.)
            self.assertGreaterEqual(row["p95_ms"], row["p50_ms"])
            self.assertAlmostEqual(row["throughput_windows_per_second"],
                                   batch * 3 * 1000 / sum(row["samples_ms"]))

    def test_eval_inference_mode_inputs_and_denormalization_are_inside_timing(self):
        calls, recorded_outputs, counter = [], [], []

        class Tiny(torch.nn.Module):
            def forward(self, actions):
                calls.append((tuple(actions.shape), self.training, torch.is_inference_mode_enabled(),
                              torch.is_grad_enabled(), actions.clone()))
                return torch.zeros(len(actions), 2, 3)

        def tick():
            counter.append(len(calls))
            return len(counter) * 1_000_000

        original_finite = torch.isfinite

        def finite(value):
            recorded_outputs.append(value.clone())
            return original_finite(value)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self._checkpoint(root, name="mean")
            checkpoint = torch.load(run / "best_eval_model.pt", weights_only=True)
            checkpoint["state_dict"] = {}
            torch.save(checkpoint, run / "best_eval_model.pt")
            with patch("src.benchmarks.modeling_models.make_model", return_value=(Tiny(), None)), \
                    patch("src.benchmarks.modeling_seed_summary.time.perf_counter_ns", side_effect=tick), \
                    patch("torch.isfinite", side_effect=finite):
                result = benchmark_latency(run, "cpu", None, warmup=2, repeats=3)
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(not training and inference and not grad for _, training, inference, grad, _ in calls))
        self.assertEqual(counter, [2, 3, 3, 4, 4, 5, 7, 8, 8, 9, 9, 10])
        torch.testing.assert_close(calls[0][4][0], calls[5][4][0])
        for tensor in recorded_outputs:
            torch.testing.assert_close(tensor, torch.tensor([1., 2., 3.]).expand_as(tensor))
        for row in result["measurements"]:
            self.assertEqual(row["p50_ms"], 1.)
            self.assertEqual(row["p95_ms"], 1.)
            self.assertEqual(row["throughput_windows_per_second"], 1000 * row["batch_size"])

    def test_hov_reconstruction_calls_full_burnin_and_geometry_every_time(self):
        from src.benchmarks.modeling_models import make_model
        rebuilt = []

        def factory(*args, **kwargs):
            self.assertIsNone(kwargs.get("train_sequences"))
            self.assertIsNotNone(kwargs["geometry_config"])
            model, metadata = make_model(*args, **kwargs)
            model.core._burn_in = unittest.mock.Mock(wraps=model.core._burn_in)
            model.core._decode_generalized = unittest.mock.Mock(wraps=model.core._decode_generalized)
            rebuilt.append(model)
            return model, metadata

        with tempfile.TemporaryDirectory() as tmp:
            run = self._checkpoint(Path(tmp), name="hov")
            with patch("src.benchmarks.modeling_models.make_model", side_effect=factory), \
                    patch("src.benchmarks.modeling_models.fit_ishsm_priors_from_arrays",
                          side_effect=AssertionError("Reconstruction must use saved priors")):
                result = benchmark_latency(run, "cpu", None, warmup=1, repeats=2)
        self.assertEqual(rebuilt[0].core._burn_in.call_count, 6)
        self.assertEqual(rebuilt[0].core._decode_generalized.call_count, 6)
        for call in rebuilt[0].core._burn_in.call_args_list:
            self.assertEqual(call.args[0].shape[1:], (19, 4))
        self.assertEqual(result["measurements"][1]["output_shape"], [256, 3, 3])

    def test_bad_history_options_and_checkpoint_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wrong_history = self._checkpoint(root, history=5)
            with self.assertRaisesRegex(ValueError, "history=20"):
                benchmark_latency(wrong_history, "cpu", None, warmup=0, repeats=1)
            run = self._checkpoint(root, name="mean")
            for options in ({"warmup": -1}, {"repeats": 0}, {"repeats": True}):
                with self.assertRaises(ValueError):
                    benchmark_latency(run, "cpu", None, **options)
            checkpoint = torch.load(run / "best_eval_model.pt", weights_only=True)
            checkpoint["state_dict"] = {"wrong": torch.zeros(1)}
            torch.save(checkpoint, run / "best_eval_model.pt")
            before_threads = torch.get_num_threads()
            with self.assertRaises(RuntimeError):
                benchmark_latency(run, "cpu", None, warmup=0, repeats=1)
            self.assertEqual(torch.get_num_threads(), before_threads)
            (run / "COMPLETE").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                benchmark_latency(run, "cpu", None)


if __name__ == "__main__":
    unittest.main()
