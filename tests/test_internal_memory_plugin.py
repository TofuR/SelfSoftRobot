"""CPU contracts for the internal MLP/Koopman memory API and gradients."""
import copy
import io
import json
import unittest

import torch
from torch import nn

from src.benchmarks.modeling_foundations import KoopmanShape
from src.benchmarks.modeling_memory_plugin import (
    FEATURE_DIMS, InternalMemoryModel, MemoryEncoder,
    count_parameters, fit_plugin_normalization, parameter_count,
)
from src.operators.maxwell_bank import MaxwellBank
from src.operators.play_bank import PlayBank
from src.operators.static_drive import MonotoneSplineDrive


def actions(n=24, history=20, dtype=torch.float32):
    generator = torch.Generator().manual_seed(80231)
    return .03+.94*torch.rand(n, history, 4, generator=generator, dtype=dtype)


def original_bank_features(encoder, windows):
    """Independent reference through the actual repository drive and banks."""
    drive = MonotoneSplineDrive(4, 5, output_normalization="unit_range").to(windows)
    drive.load_state_dict(encoder.drive.state_dict())
    play = PlayBank(4, 2, r_range=(.02, .5)).to(windows)
    maxwell = MaxwellBank(4, 6, dt=encoder.dt, tau_range=(.6, 2.)).to(windows)
    torch.testing.assert_close(encoder.thresholds, play.thresholds, rtol=0, atol=0)
    torch.testing.assert_close(encoder.alpha, maxwell.decays, rtol=0, atol=0)
    e = drive(windows)
    p = e[:, 0, :, None].repeat(1, 1, 2)
    h = e[:, 0, :, None].repeat(1, 1, 6)
    q = torch.zeros_like(p)
    for t in range(1, windows.shape[1]):
        p, q = play.step(p, e[:, t])
        h = maxwell.step(h, e[:, t])
    d = h-e[:, -1, :, None]
    return torch.cat([q.flatten(1), d.flatten(1)], dim=1), drive


class InternalMemoryPluginTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.train = actions(80)
        cls.normalization = fit_plugin_normalization(cls.train)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_encoder_values_and_action_phi_gradients_against_actual_banks(self):
        for dtype, tolerance in ((torch.float32, 4e-6), (torch.float64, 2e-12)):
            for variant in ("path", "time", "both"):
                with self.subTest(dtype=dtype, variant=variant):
                    encoder = MemoryEncoder(variant).to(dtype=dtype)
                    with torch.no_grad():
                        encoder.drive.raw_weights.copy_(torch.linspace(-1.4, .4, 20).reshape(4, 5))
                    a = actions(12, dtype=dtype).requires_grad_()
                    b = a.detach().clone().requires_grad_()
                    actual = encoder(a)
                    bank, drive = original_bank_features(encoder, b)
                    expected = bank[:, :8] if variant == "path" else bank[:, 8:] if variant == "time" else bank
                    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                    generator = torch.Generator().manual_seed(990)
                    probe = torch.randn(actual.shape, dtype=dtype, generator=generator)
                    actual_grad = torch.autograd.grad((actual*probe).sum(), (a, encoder.drive.raw_weights))
                    expected_grad = torch.autograd.grad((expected*probe).sum(), (b, drive.raw_weights))
                    for got, reference in zip(actual_grad, expected_grad):
                        torch.testing.assert_close(got, reference, atol=tolerance, rtol=tolerance)
                        self.assertTrue(torch.isfinite(got).all())
                        self.assertGreater(float(got.abs().sum()), 0.)

    def test_equilibrium_constant_windows_are_exactly_zero(self):
        constants = torch.tensor([[0., .1, .5, 1.], [.2, .4, .6, .8]])
        for history in (1, 20):
            window = constants[:, None].repeat(1, history, 1)
            for variant in ("base", "path", "time", "both"):
                with self.subTest(history=history, variant=variant):
                    encoder = MemoryEncoder(variant, history=history)
                    features = encoder(window)
                    self.assertEqual(features.shape, (2, FEATURE_DIMS[variant]))
                    self.assertEqual(int(torch.count_nonzero(features)), 0)
        # A nonconstant past followed by a hold need not erase path memory.
        window = torch.ones(1, 20, 4)*.4
        window[:, 0] = .8
        self.assertGreater(float(MemoryEncoder("path")(window).abs().sum()), 0.)

    def test_feature_order_static_powers_and_history_dependence(self):
        a = actions()
        both = MemoryEncoder("both")
        q, d = MemoryEncoder("path")(a), MemoryEncoder("time")(a)
        torch.testing.assert_close(both(a), torch.cat([q, d], 1), atol=0, rtol=0)
        static = MemoryEncoder("static_capacity")
        phi = static.drive(a[:, -1])
        independent = torch.stack([phi[:, c]**power for c in range(4) for power in range(1, 9)], 1)
        torch.testing.assert_close(static(a), independent)
        changed = a.clone()
        changed[:, :-1] = .1
        torch.testing.assert_close(static(a), static(changed), atol=0, rtol=0)
        self.assertGreater(float((both(a)-both(changed)).abs().sum()), 0.)
        self.assertEqual(parameter_count(both), 20)
        self.assertEqual(parameter_count(static), 20)
        self.assertEqual(parameter_count(MemoryEncoder("base")), 0)
        torch.testing.assert_close(static.drive(torch.zeros(2, 4)), torch.zeros(2, 4), atol=0, rtol=0)
        # The original float32 drive can round the normalized endpoint by one ULP.
        torch.testing.assert_close(static.drive(torch.ones(2, 4)), torch.ones(2, 4), atol=2e-7, rtol=0)

    def test_exact_parameter_counts_and_all_base_parameters_shared_by_seed(self):
        expected = {
            "mlp": dict(zip(FEATURE_DIMS, [7405, 7937, 8961, 9473, 9473])),
            "koopman": dict(zip(FEATURE_DIMS, [3981, 4361, 5081, 5441, 5441])),
        }
        window = actions(5)
        for family in expected:
            for seed in (100, 119):
                torch.manual_seed(seed)
                reference = InternalMemoryModel(family, "base", self.normalization)
                reference_prediction = reference(window)
                for variant, count in expected[family].items():
                    with self.subTest(family=family, variant=variant, seed=seed):
                        torch.manual_seed(seed)
                        model = InternalMemoryModel(family, variant, self.normalization)
                        self.assertEqual(model.parameter_count(), count)
                        self.assertEqual(model.count_parameters(), count)
                        self.assertEqual(count_parameters(model), count)
                        self.assertEqual(sum(p.numel() for p in model.parameters()), count)
                        self.assertEqual(model.memory_dim, FEATURE_DIMS[variant])
                        self.assertEqual(model(window).shape, (5, 15, 3))
                        torch.testing.assert_close(model(window), reference_prediction, rtol=0, atol=0)
                        self.assertEqual(set(model.base.state_dict()), set(reference.base.state_dict()))
                        for name, value in model.base.state_dict().items():
                            torch.testing.assert_close(value, reference.base.state_dict()[name], atol=0, rtol=0)
                        if variant != "base":
                            self.assertIsNone(model.D.bias)
                            self.assertEqual(int(torch.count_nonzero(model.D.weight)), 0)
                            self.assertEqual(model.D.weight.shape, (64 if family == "mlp" else 45, FEATURE_DIMS[variant]))

    def test_koopman_matches_original_and_does_not_standardize_pressure_lift(self):
        a = actions(7)
        for seed in (23, 101):
            torch.manual_seed(seed)
            original = KoopmanShape(hidden=128, latent=16)
            torch.manual_seed(seed)
            model = InternalMemoryModel("koopman", "both", self.normalization)
            self.assertIs(type(model.base), KoopmanShape)
            torch.testing.assert_close(model(a), original(a), atol=0, rtol=0)
            altered_norm = copy.deepcopy(self.normalization)
            altered_norm["input_mean"] = [7., 8., 9., 10.]
            altered_norm["input_std"] = [2., 3., 4., 5.]
            torch.manual_seed(seed)
            changed = InternalMemoryModel("koopman", "both", altered_norm)
            torch.testing.assert_close(changed(a), original(a), atol=0, rtol=0)
        with torch.no_grad():
            model.D.weight.fill_(.003)
        expected = original(a)+model.D(model.standardized_memory(a)).reshape(-1, 15, 3)
        torch.testing.assert_close(model(a), expected, atol=0, rtol=0)

    def test_mlp_exact_baseline_and_hidden_layer_injection(self):
        a = actions(7)
        torch.manual_seed(29)
        independent = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 45))
        torch.manual_seed(29)
        model = InternalMemoryModel("mlp", "both", self.normalization)
        current = (a[:, -1]-model.input_mean)/model.input_std
        torch.testing.assert_close(model(a), independent(current).reshape(-1, 15, 3), atol=0, rtol=0)
        with torch.no_grad():
            model.D.weight.copy_(torch.linspace(-.04, .04, model.D.weight.numel()).reshape_as(model.D.weight))
        memory = (model.encoder(a)-model.memory_mean)/model.memory_std
        h1 = torch.tanh(independent[0](current))
        expected = independent[4](torch.tanh(independent[2](h1)+model.D(memory))).reshape(-1, 15, 3)
        torch.testing.assert_close(model(a), expected, atol=0, rtol=0)

    def test_normalization_json_population_statistics_rng_and_sequence_interface(self):
        rng = torch.random.get_rng_state().clone()
        normalization = fit_plugin_normalization(self.train)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(json.loads(json.dumps(normalization, allow_nan=False)), normalization)
        current = self.train[:, -1].double()
        torch.testing.assert_close(torch.tensor(normalization["input_mean"], dtype=torch.float64), current.mean(0), atol=0, rtol=0)
        torch.testing.assert_close(torch.tensor(normalization["input_std"], dtype=torch.float64), current.std(0, unbiased=False), atol=0, rtol=0)
        for variant, dim in FEATURE_DIMS.items():
            raw = MemoryEncoder(variant)(self.train).double()
            self.assertEqual(len(normalization["memory_mean"][variant]), dim)
            self.assertEqual(len(normalization["memory_std"][variant]), dim)
            if dim == 0:
                continue
            torch.testing.assert_close(torch.tensor(normalization["memory_mean"][variant], dtype=torch.float64), raw.mean(0), atol=0, rtol=0)
            torch.testing.assert_close(torch.tensor(normalization["memory_std"][variant], dtype=torch.float64), raw.std(0, unbiased=False).clamp_min(1e-6), atol=0, rtol=0)
        sequence = actions(2).reshape(-1, 4)
        windowed = torch.stack([sequence[t-19:t+1] for t in range(19, len(sequence))])
        self.assertEqual(fit_plugin_normalization(sequence), fit_plugin_normalization(windowed.numpy()))
        constant = fit_plugin_normalization(torch.full((4, 20, 4), .4))
        self.assertEqual(constant["input_std"], [1e-6]*4)
        self.assertEqual(constant["memory_std"]["both"], [1e-6]*32)

    def test_phi_gradient_and_optimizer_step_with_fixed_statistics(self):
        a = actions(16)
        target = torch.linspace(-.4, .6, 16*45).reshape(16, 15, 3)
        for family in ("mlp", "koopman"):
            for variant in ("path", "time", "both", "static_capacity"):
                with self.subTest(family=family, variant=variant):
                    torch.manual_seed(830)
                    model = InternalMemoryModel(family, variant, self.normalization)
                    # Zero D blocks the drive gradient at initialization by design.
                    (model(a)-target).square().mean().backward()
                    self.assertIsNotNone(model.encoder.drive.raw_weights.grad)
                    self.assertEqual(float(model.encoder.drive.raw_weights.grad.abs().sum()), 0.)
                    self.assertGreater(float(model.D.weight.grad.abs().sum()), 0.)
                    model.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        model.D.weight.copy_(torch.linspace(-.02, .025, model.D.weight.numel()).reshape_as(model.D.weight))
                    before = model.encoder.drive.raw_weights.detach().clone()
                    buffers = {name: value.clone() for name, value in model.named_buffers()}
                    optimizer = torch.optim.SGD(model.parameters(), lr=.03)
                    model.train()
                    loss = (model(a)-target).square().mean()
                    loss.backward()
                    gradient = model.encoder.drive.raw_weights.grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertTrue(torch.all(gradient.abs().sum(1) > 1e-9))
                    optimizer.step()
                    self.assertGreater(float((model.encoder.drive.raw_weights-before).abs().sum()), 0.)
                    for name, value in model.named_buffers():
                        torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)

    def test_checkpoint_roundtrip_and_dtype(self):
        a = actions(5, dtype=torch.float64)
        for family in ("mlp", "koopman"):
            model = InternalMemoryModel(family, "both", self.normalization).double()
            with torch.no_grad():
                model.D.weight.fill_(.025)
            stream = io.BytesIO()
            torch.save(model.state_dict(), stream)
            stream.seek(0)
            restored = InternalMemoryModel(family, "both", self.normalization).double()
            restored.load_state_dict(torch.load(stream, weights_only=True))
            torch.testing.assert_close(restored(a), model(a), rtol=0, atol=0)
            self.assertEqual(restored(a).dtype, torch.float64)
            self.assertFalse(any(b.requires_grad for b in restored.buffers()))

    def test_invalid_variants_inputs_and_normalization_fail_clearly(self):
        with self.assertRaises(ValueError):
            MemoryEncoder("unknown")
        with self.assertRaises(ValueError):
            MemoryEncoder("time", history=0)
        with self.assertRaises(ValueError):
            MemoryEncoder("time", dt=0.)
        with self.assertRaises(ValueError):
            MemoryEncoder("both")(torch.zeros(2, 19, 4))
        with self.assertRaises(ValueError):
            fit_plugin_normalization(torch.empty(0, 20, 4))
        with self.assertRaises(ValueError):
            fit_plugin_normalization(torch.full((1, 20, 4), float("nan")))
        with self.assertRaises(ValueError):
            InternalMemoryModel("unknown", "base", self.normalization)
        bad = copy.deepcopy(self.normalization)
        bad["memory_std"]["both"][0] = 0.
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            InternalMemoryModel("mlp", "both", bad)


if __name__ == "__main__":
    unittest.main()
