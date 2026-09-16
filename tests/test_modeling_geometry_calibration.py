"""CPU contracts for calibrated reference fitting and hereditary ablations."""
import io
import json
import unittest

import torch

from src.benchmarks.modeling_geometry_calibration import (
    CalibratedGeometry, make_calibrated_geometry,
)
from src.benchmarks.modeling_models import AblatedGeometry


def geometry_config(reference_kind="monotone_spline"):
    bias = [0.0] * 14
    bias[0] = 1.4
    config = dict(
        action_dim=4, n_nodes=15, window_size=5, dt=0.2,
        n_play=2, n_maxwell=3, tau_range=(0.6, 2.0),
        n_bend_modes=14, bend_basis_kind="local",
        bend_basis=torch.eye(14).tolist(), section_intervals=(7, 7),
        generalized_coordinate_scale=[0.05] * 14 + [0.01, 0.01],
        reference_segment_lengths=[10.0] * 14,
        reference_bend_bias=bias,
        reference_bend_dirs=[[(-1) ** c * 0.01] * 14 for c in range(4)],
        reference_length_bias=[0.0, 0.0],
        reference_length_dirs=[[0.01, -0.01]] * 4,
        reference_kind=reference_kind, base_position=[2.0, 3.0, 0.0],
        residual_mode="none", drive_normalization="unit_range",
        burnin_mode="equilibrium",
    )
    if reference_kind == "monotone_spline":
        config.update(reference_knots=[0.0, 0.25, 0.5, 0.75, 1.0],
                      reference_drive_weights=[[0.4] * 5] * 4)
    return config


class GeometryCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)

    def test_initial_prediction_and_state_match_fitted_core(self):
        for kind in ("linear", "monotone_spline"):
            with self.subTest(reference_kind=kind):
                config = geometry_config(kind)
                original = AblatedGeometry(**config)
                calibrated = CalibratedGeometry(**config)
                calibrated.load_state_dict(original.state_dict(), strict=True)
                actions = torch.rand(3, 5, 4)
                before, after = original(actions), calibrated(actions)
                for key in before:
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
                self.assertEqual(set(original.state_dict()), set(calibrated.state_dict()))
                added = set(dict(calibrated.named_parameters())) - set(dict(original.named_parameters()))
                self.assertEqual(added, set(calibrated.calibration_parameter_names))
                self.assertEqual(calibrated.geometry_report()["calibration_parameter_count"], 72)
                self.assertTrue((calibrated.reference_bend_dirs < 0).any())

    def test_zero_memory_is_exactly_the_calibrated_static_reference(self):
        model = CalibratedGeometry(**geometry_config())
        with torch.no_grad():
            model.reference_bend_bias.add_(0.01)
            model.reference_length_bias.add_(0.02)
        action = torch.rand(3, 4)
        drive = model.drive(action)
        state = model._pack_state(drive[..., None].expand(-1, -1, model.n_play),
                                  drive[..., None].expand(-1, -1, model.n_maxwell))
        torch.testing.assert_close(model.observe_state(action, state),
                                   model.decode_equilibrium(action), rtol=0, atol=0)
        output = model(action[:, None].expand(-1, 5, -1))
        torch.testing.assert_close(output["memory_generalized"],
                                   torch.zeros_like(output["memory_generalized"]), atol=1e-7, rtol=0)
        self.assertFalse(any(isinstance(module, torch.nn.Linear) for module in model.modules()))

    def test_branch_ablation_removes_state_influence_from_complete_output(self):
        for branch in ("play", "maxwell"):
            with self.subTest(branch=branch):
                model = CalibratedGeometry(**geometry_config(), **{f"disable_{branch}": True})
                action = torch.full((2, 4), 0.5)
                drive = model.drive(action)
                p = drive[..., None].repeat(1, 1, model.n_play)
                h = drive[..., None].repeat(1, 1, model.n_maxwell)
                state = model._pack_state(p, h)
                changed = model._pack_state(p - 0.02 if branch == "play" else p,
                                            h + 0.03 if branch == "maxwell" else h)
                torch.testing.assert_close(model.observe_state(action, state),
                                           model.observe_state(action, changed), rtol=0, atol=0)
                # The remaining branch still influences the complete output.
                active = model._pack_state(p if branch == "play" else p - 0.02,
                                          h + 0.03 if branch == "play" else h)
                self.assertGreater(float((model.observe_state(action, active) -
                                          model.observe_state(action, state)).abs().max()), 1e-6)
                frozen = [model.pi_mode_directions_raw, *model.play.parameters()] if branch == "play" else [
                    model.maxwell_mode_directions_raw, model.maxwell_gain_raw, *model.maxwell.parameters()]
                self.assertTrue(all(not parameter.requires_grad for parameter in frozen))

    def test_static_ablation_trains_reference_and_ignores_history(self):
        model, _ = make_calibrated_geometry(geometry_config(), ([0, 0, 0], 100.0), static=True)
        trainable = {name for name, p in model.core.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, set(model.core.calibration_parameter_names))
        actions = torch.rand(4, 5, 4)
        other = torch.rand_like(actions)
        other[:, -1] = actions[:, -1]
        torch.testing.assert_close(model(actions), model(other), rtol=0, atol=0)

        teacher, _ = make_calibrated_geometry(geometry_config(), ([0, 0, 0], 100.0), static=True)
        with torch.no_grad():
            teacher.core.reference_bend_bias[0].add_(0.03)
            teacher.core.reference_length_bias.add_(0.01)
            target = teacher(actions)
        frozen = {name: value.clone() for name, value in model.core.named_buffers()}
        initial_loss = (model(actions) - target).square().mean().item()
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss = (model(actions) - target).square().mean()
            loss.backward()
            for name in trainable:
                gradient = getattr(model.core, name).grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(float(gradient.abs().sum()), 0)
            optimizer.step()
        self.assertLess((model(actions) - target).square().mean().item(), initial_loss)
        for name, value in frozen.items():
            torch.testing.assert_close(dict(model.core.named_buffers())[name], value, rtol=0, atol=0)

    def test_fixed_monotone_drive_and_positive_bounded_lengths(self):
        model = CalibratedGeometry(**geometry_config())
        buffers = dict(model.named_buffers())
        for name in ("reference_drive_weights", "reference_knots", "reference_length_dirs",
                     "reference_segment_lengths", "bend_basis", "base_position"):
            self.assertIn(name, buffers)
        action = torch.linspace(0, 1, 11)[:, None].repeat(1, 4)
        reference_drive = (torch.relu(action[..., None] - model.reference_knots) *
                           model.reference_drive_weights).sum(-1)
        self.assertTrue((reference_drive.diff(dim=0) >= 0).all())
        self.assertTrue((model.play.weights > 0).all())
        self.assertTrue((model.maxwell_gains > 0).all())
        with torch.no_grad():
            model.reference_length_bias.copy_(torch.tensor([100.0, -100.0]))
        skeleton = model.decode_equilibrium(action)
        lengths = torch.linalg.vector_norm(skeleton[:, 1:] - skeleton[:, :-1], dim=-1)
        expected = 10 * torch.exp(torch.tensor([0.25] * 7 + [-0.25] * 7))
        torch.testing.assert_close(lengths, expected.expand_as(lengths), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(skeleton[:, 0], model.base_position.expand(len(action), -1))
        self.assertTrue((skeleton[..., 2] == 0).all())

    def test_factory_checkpoint_roundtrip_and_configuration_isolation(self):
        for static in (False, True):
            with self.subTest(static=static):
                config = geometry_config()
                original = json.dumps(config)
                model, metadata = make_calibrated_geometry(config, ([2, 3, 0], 50), static=static)
                self.assertEqual(json.dumps(config), original)
                with torch.no_grad():
                    model.core.reference_bend_dirs.add_(0.004)
                stream = io.BytesIO()
                torch.save(model.state_dict(), stream)
                stream.seek(0)
                # Reconstruction needs only the returned metadata and normalization.
                rebuilt, _ = make_calibrated_geometry(json.loads(json.dumps(metadata)), ([2, 3, 0], 50))
                rebuilt.load_state_dict(torch.load(stream, weights_only=True), strict=True)
                actions = torch.rand(3, 5, 4)
                torch.testing.assert_close(model(actions), rebuilt(actions), rtol=0, atol=0)
                self.assertTrue(rebuilt.core.reference_bend_dirs.requires_grad)

    def test_optional_length_directions_and_residual_contract(self):
        model = CalibratedGeometry(**geometry_config(), calibrate_length_directions=True)
        self.assertTrue(model.reference_length_dirs.requires_grad)
        self.assertEqual(model.geometry_report()["calibration_parameter_count"], 80)
        config = geometry_config()
        config["residual_mode"] = "memory"
        with self.assertRaisesRegex(ValueError, "residual_mode"):
            make_calibrated_geometry(config, ([0, 0, 0], 1))

    def test_pair_zero_initialization_preserves_predictions_and_default_state(self):
        for lengths in (False, True):
            with self.subTest(length_interactions=lengths):
                torch.manual_seed(23)
                original = CalibratedGeometry(**geometry_config())
                torch.manual_seed(23)
                paired = CalibratedGeometry(**geometry_config(), reference_pair_interactions=True,
                                            reference_pair_length_interactions=lengths)
                actions = torch.rand(3, 5, 4)
                for key, value in original(actions).items():
                    torch.testing.assert_close(value, paired(actions)[key], rtol=0, atol=0)
                self.assertFalse(any("reference_pair" in key for key in original.state_dict()))
                self.assertEqual(tuple(paired.reference_pair_coefficients.shape), (6, 14))
                self.assertEqual(paired.geometry_report()["calibration_parameter_count"],
                                 168 if lengths else 156)

    def test_pair_products_use_only_the_corresponding_raw_pressure_channels(self):
        model = CalibratedGeometry(**geometry_config(), reference_pair_interactions=True,
                                   reference_pair_length_interactions=True)
        expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.assertEqual([tuple(pair) for pair in model.reference_pair_indices.T.tolist()], expected_pairs)
        for row, (i, j) in enumerate(expected_pairs):
            with self.subTest(pair=(i, j)), torch.no_grad():
                model.reference_pair_coefficients.zero_()
                model.reference_pair_length_coefficients.zero_()
                model.reference_pair_coefficients[row, 3] = -0.4
                model.reference_pair_length_coefficients[row, 1] = 0.2
                action = torch.full((4, 4), 0.1)
                action[:, i], action[:, j] = 0.3, 0.7
                unrelated = [k for k in range(4) if k not in (i, j)]
                action[1, unrelated] = 0.9
                action[2, i] = 0
                action[3, j] = 0
                bend, length = model._reference(action)
                base_bend, base_length = AblatedGeometry._reference(model, action)
                expected_bend, expected_length = torch.zeros_like(bend), torch.zeros_like(length)
                expected_bend[:2, 3] = -0.4 * 0.3 * 0.7
                expected_length[:2, 1] = 0.2 * 0.3 * 0.7
                torch.testing.assert_close(bend - base_bend, expected_bend, atol=1e-7, rtol=0)
                torch.testing.assert_close(length - base_length, expected_length, atol=1e-7, rtol=0)

    def test_pair_terms_have_gradients_in_every_ablation(self):
        for flags in ({}, {"disable_play": True}, {"disable_maxwell": True},
                      {"disable_play": True, "disable_maxwell": True}):
            with self.subTest(flags=flags):
                model = CalibratedGeometry(**geometry_config(), **flags,
                                           reference_pair_interactions=True,
                                           reference_pair_length_interactions=True)
                actions = 0.2 + 0.6 * torch.rand(3, 5, 4)
                model(actions)["skeleton"][..., 0].sum().backward()
                for name in ("reference_pair_coefficients", "reference_pair_length_coefficients"):
                    parameter = getattr(model, name)
                    self.assertTrue(parameter.requires_grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertTrue((parameter.grad.abs().sum(-1) > 0).all())

    def test_pair_static_checkpoint_zero_memory_and_geometry_constraints(self):
        config = dict(geometry_config(), reference_pair_interactions=True,
                      reference_pair_length_interactions=True)
        model, metadata = make_calibrated_geometry(config, ([0, 0, 0], 1), static=True)
        self.assertFalse(any(p.requires_grad for p in model.core.drive.parameters()))
        with torch.no_grad():
            model.core.reference_pair_coefficients.fill_(-0.003)
            model.core.reference_pair_length_coefficients[:, 0] = 100
            model.core.reference_pair_length_coefficients[:, 1] = -100
        action = torch.full((2, 4), 0.5)
        window = action[:, None].repeat(1, 5, 1)
        window[:, :-1] = torch.rand(2, 4, 4)
        output = model.core(window)
        torch.testing.assert_close(output["memory_generalized"],
                                   torch.zeros_like(output["memory_generalized"]), rtol=0, atol=0)
        torch.testing.assert_close(output["skeleton"], model.core.decode_equilibrium(action), rtol=0, atol=0)
        skeleton = output["skeleton"]
        lengths = torch.linalg.vector_norm(skeleton[:, 1:] - skeleton[:, :-1], dim=-1)
        expected = 10 * torch.exp(torch.tensor([0.25] * 7 + [-0.25] * 7))
        torch.testing.assert_close(lengths, expected.expand_as(lengths), atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(skeleton[:, 0], model.core.base_position.expand(2, -1))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        rebuilt, saved = make_calibrated_geometry(json.loads(json.dumps(metadata)), ([0, 0, 0], 1))
        rebuilt.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertTrue(saved["reference_pair_interactions"])
        self.assertTrue(saved["reference_pair_length_interactions"])
        self.assertTrue(rebuilt.core.reference_pair_coefficients.requires_grad)
        self.assertTrue(rebuilt.core.reference_pair_length_coefficients.requires_grad)
        torch.testing.assert_close(model(window), rebuilt(window), rtol=0, atol=0)
        # At an explicit zero-memory state the full model has the same static term.
        full = CalibratedGeometry(**config)
        full.load_state_dict(model.core.state_dict(), strict=True)
        drive = full.drive(action)
        state = full._pack_state(drive[..., None].repeat(1, 1, full.n_play),
                                 drive[..., None].repeat(1, 1, full.n_maxwell))
        torch.testing.assert_close(full.observe_state(action, state),
                                   full.decode_equilibrium(action), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
