"""CPU synthetic recovery and readout-only initialization contracts."""
import json
import unittest
from unittest.mock import patch

import torch

from src.benchmarks.modeling_geometry_calibration import CalibratedGeometry
from src.benchmarks.modeling_memory_initialization import initialize_memory_readout
from src.benchmarks.modeling_models import AblatedGeometry, GeometryWindow


def _model(calibrated=False, reduced=False, **flags):
    basis = [[1., .3], [.2, 1.], [.5, -.1], [.1, .4]] if reduced else torch.eye(4).tolist()
    modes = 2 if reduced else 4
    core = (CalibratedGeometry if calibrated else AblatedGeometry)(
        action_dim=2, n_nodes=5, window_size=6, n_play=1, n_maxwell=1,
        n_bend_modes=modes, bend_basis=basis, section_intervals=(2, 2), dt=.2,
        reference_segment_lengths=[10.] * 4, reference_bend_bias=[1., .03, -.02, .01],
        reference_bend_dirs=[[.1, 0., .01, 0.], [-.1, .01, 0., .02]],
        reference_length_bias=[.02, -.01], reference_kind="linear",
        generalized_coordinate_scale=[.06] * modes + [.02, .03], residual_mode="none", **flags)
    core.set_normalization(torch.tensor([1., 2., 3.]), torch.tensor([12., 12., 12.]), 1.)
    return GeometryWindow(core).double()


def _coefficients(core):
    return torch.cat(((core.pi_mode_directions * core.play.weights[..., None]).flatten(0, 1),
                      (core.maxwell_mode_directions * core.maxwell_gains[..., None]).flatten(0, 1)))


def _training(model, n=240):
    generator = torch.Generator().manual_seed(123)
    actions = torch.rand(n, 6, 2, generator=generator, dtype=torch.float64)
    with torch.no_grad():
        # Keep geometry in the locally invertible, unclipped range.
        model.core.play.raw_weights.fill_(-1.5)
        model.core.maxwell_gain_raw.fill_(-1.2)
        expected = _coefficients(model.core).clone()
        physical = model(actions) * model.core.pc_scale + model.core.pc_center
    return actions, physical, expected


class MemoryInitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(8)

    def test_known_readout_recovery_for_local_reduced_and_calibrated_geometry(self):
        for calibrated, reduced in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(calibrated=calibrated, reduced=reduced):
                model = _model(calibrated, reduced)
                actions, physical, expected = _training(model)
                with torch.no_grad():
                    model.core.pi_mode_directions_raw.normal_()
                    model.core.maxwell_mode_directions_raw.normal_()
                    model.core.play.raw_weights.fill_(-.3)
                    model.core.maxwell_gain_raw.fill_(-.3)
                metadata = initialize_memory_readout(model, actions, physical, ridge=1e-10, batch_size=31)
                torch.testing.assert_close(_coefficients(model.core), expected, atol=2e-8, rtol=2e-7)
                with torch.no_grad():
                    actual = model(actions) * model.core.pc_scale + model.core.pc_center
                torch.testing.assert_close(actual, physical, atol=2e-8, rtol=2e-8)
                self.assertEqual(metadata["n_features"], 4)
                self.assertEqual(metadata["n_windows"], 240)
                self.assertEqual(metadata["status"], "initialized")
                json.dumps(metadata, allow_nan=False)

    def test_only_readout_changes_zero_memory_modes_and_gradients_are_preserved(self):
        model = _model(calibrated=True)
        actions, physical, _ = _training(model)
        model.train()
        model.core.drive.eval()
        before_flags = [module.training for module in model.modules()]
        before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        with torch.no_grad():
            equilibrium = model.core.decode_equilibrium(actions[:, -1]).clone()
        metadata = initialize_memory_readout(model, actions, physical)
        changed = set(metadata["modified_parameters"])
        for name, tensor in model.state_dict().items():
            if name not in changed:
                torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        self.assertEqual([module.training for module in model.modules()], before_flags)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        with torch.no_grad():
            core = model.core
            q = torch.zeros(3, 2, 1, dtype=torch.float64)
            pi, maxwell, *_ = core._structured_memory(q, q)
            self.assertEqual(torch.count_nonzero(pi + maxwell), 0)
            torch.testing.assert_close(core.decode_equilibrium(actions[:, -1]), equilibrium, rtol=0, atol=0)
        self.assertFalse(metadata["centered"])
        self.assertFalse(metadata["intercept"])

    def test_float32_training_path_runs_without_autograd(self):
        model = _model(calibrated=True, reduced=True)
        actions, physical, expected = _training(model, 120)
        model.float()
        grad_modes = []
        hook = model.core.register_forward_pre_hook(
            lambda module, args: grad_modes.append(torch.is_grad_enabled()))
        try:
            initialize_memory_readout(model, actions.float(), physical.float(),
                                     ridge=1e-7, batch_size=23)
        finally:
            hook.remove()
        self.assertEqual(grad_modes, [False] * 6)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        torch.testing.assert_close(_coefficients(model.core).double(), expected, atol=3e-5, rtol=3e-4)

    def test_disabled_branches_are_excluded_and_remain_frozen(self):
        for disabled in ("play", "maxwell"):
            with self.subTest(disabled=disabled):
                model = _model(calibrated=True, **{f"disable_{disabled}": True})
                actions, physical, expected = _training(model)
                before = {key: value.clone() for key, value in model.state_dict().items()}
                metadata = initialize_memory_readout(model, actions, physical, ridge=1e-10)
                self.assertEqual(metadata["active_branches"], ["maxwell" if disabled == "play" else "play"])
                self.assertEqual(metadata["n_features"], 2)
                for name, value in model.state_dict().items():
                    if name not in metadata["modified_parameters"]:
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                torch.testing.assert_close(_coefficients(model.core), expected, atol=1e-8, rtol=1e-7)
                q = torch.ones(2, 2, 1, dtype=torch.float64)
                parts = model.core._structured_memory(q, q)
                self.assertEqual(torch.count_nonzero(parts[0 if disabled == "play" else 1]), 0)
                directions = model.core.pi_mode_directions_raw if disabled == "play" else model.core.maxwell_mode_directions_raw
                self.assertFalse(directions.requires_grad)

    def test_all_disabled_and_static_skip_without_accessing_data(self):
        for model in (_model(disable_play=True, disable_maxwell=True), GeometryWindow(_model().core, static=True)):
            with patch.object(model.core, "forward", side_effect=AssertionError("Must skip")):
                metadata = initialize_memory_readout(model, None, None)
            self.assertEqual(metadata["status"], "skipped")
            self.assertEqual(metadata["modified_parameters"], [])

    def test_ridge_is_normalized_by_sample_count_and_batch_partition(self):
        model = _model()
        actions, physical, _ = _training(model, 100)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        initialize_memory_readout(model, actions, physical, ridge=.05, batch_size=13)
        expected = _coefficients(model.core).clone()
        model.load_state_dict(initial)
        initialize_memory_readout(model, actions.repeat(3, 1, 1), physical.repeat(3, 1, 1),
                                  ridge=.05, batch_size=71)
        torch.testing.assert_close(_coefficients(model.core), expected, atol=1e-12, rtol=1e-10)

    def test_unexcited_features_are_finite_and_zero_memory_remains_zero(self):
        model = _model()
        actions = torch.full((10, 6, 2), .4, dtype=torch.float64)
        with torch.no_grad():
            physical = model.core.decode_equilibrium(actions[:, -1]) * model.core.pc_scale + model.core.pc_center
        metadata = initialize_memory_readout(model, actions, physical, ridge=0)
        self.assertEqual(metadata["unexcited_features"], 4)
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in model.parameters()))
        with torch.no_grad():
            self.assertEqual(torch.count_nonzero(model.core(actions)["memory_generalized"]), 0)

    def test_invalid_inputs_leave_parameters_and_modes_unchanged(self):
        model = _model()
        actions, physical, _ = _training(model, 10)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        bad = actions.clone()
        bad[-1, 0, 0] = float("nan")
        for x, y, options in ((bad, physical, {"batch_size": 3}), (actions, physical, {"ridge": -1}),
                               (actions, physical, {"batch_size": 0}), (actions[:0], physical[:0], {}),
                               (actions, physical[:, :2], {})):
            with self.assertRaises(ValueError):
                initialize_memory_readout(model, x, y, **options)
            self.assertTrue(model.training)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        with self.assertRaises(TypeError):
            initialize_memory_readout(torch.nn.Linear(2, 2), actions, physical)


if __name__ == "__main__":
    unittest.main()
