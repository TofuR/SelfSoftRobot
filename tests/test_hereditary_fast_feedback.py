"""Numerical parity and full-suffix contracts for accelerated inference."""
import unittest

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from tests.test_hereditary_geometry_model import _model
from src.control.hereditary_fast import FrozenHereditary, CachedSuffixA, fast_suffix_b
from src.control.hereditary_feedback import ActionBounds, block_basis, rollout, batched_rollout
from src.control.partial_image import extract_edges, extract_edges_vectorized


class FastFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.limits = threadpool_limits(1)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def setUp(self):
        self.model = _model(residual_mode="none").eval()
        self.model.requires_grad_(False)
        self.fast = FrozenHereditary(self.model)
        generator = np.random.default_rng(17)
        self.actions = generator.uniform(.32, .38, (7, 4)).astype(np.float32)
        self.state = self.model.init_z_from_action(torch.tensor(self.actions[:2])[None])[0].detach().numpy()
        self.bounds = ActionBounds(np.zeros(4), np.ones(4), np.ones(4)*.1, np.ones(4)*.1)

    def test_forward_batched_and_numpy_match_all_frames(self):
        z, u = torch.tensor(self.state), torch.tensor(self.actions)
        expected = rollout(self.model, z, u).detach().numpy()
        np.testing.assert_allclose(batched_rollout(self.model, z, u).detach(), expected, atol=2e-5)
        np.testing.assert_allclose(self.fast.rollout(self.state, self.actions), expected, atol=2e-5)
        expected_state = self.model.step_state(u[:1], z[None])["latent_z"][0].detach().numpy()
        np.testing.assert_allclose(self.fast.step(self.state, self.actions[0]), expected_state, atol=1e-7)

    def test_action_and_state_jacobians_match_autograd(self):
        z, u = torch.tensor(self.state), torch.tensor(self.actions)
        basis = block_basis(len(u), 4, 3)
        _, jac_u = self.fast.rollout(self.state, self.actions, action_directions=basis.reshape(len(u), 4, -1))
        _, jac_z = self.fast.rollout(self.state, self.actions, initial_directions=np.eye(self.fast.z_dim))
        fn = lambda eta: batched_rollout(self.model, z, u+(torch.tensor(basis, dtype=torch.float32)@eta).reshape_as(u))
        expected_u = torch.autograd.functional.jacobian(fn, torch.zeros(basis.shape[1]), vectorize=True).detach().numpy()
        expected_z = torch.autograd.functional.jacobian(lambda s: batched_rollout(self.model, s, u), z, vectorize=True).detach().numpy()
        np.testing.assert_allclose(jac_u, expected_u, atol=3e-5, rtol=3e-4)
        np.testing.assert_allclose(jac_z, expected_z, atol=3e-5, rtol=3e-4)

    def test_length_clamp_and_reverse_input(self):
        self.model.reference_length_bias.fill_(.5)
        fast = FrozenHereditary(self.model)
        actions = torch.tensor(self.actions[::-1].copy())
        expected = rollout(self.model, torch.tensor(self.state), actions).detach().numpy()
        np.testing.assert_allclose(fast.rollout(self.state, actions.numpy()), expected, atol=2e-5)

    def test_rejects_unrepresented_neural_residual(self):
        with self.assertRaisesRegex(ValueError, 'residual_mode'):
            FrozenHereditary(_model(residual_mode="memory"))

    def test_fast_b_reduces_error_with_same_constraints(self):
        old = self.actions.astype(float)
        target = self.fast.rollout(self.state, old+.015)
        before_state, before_old = self.state.copy(), old.copy()
        new, info = fast_suffix_b(self.fast, self.state, old, old[0], target, self.bounds, blocks=3)
        self.assertTrue(info["solver_success"])
        self.assertTrue(info["accepted"])
        self.assertLess(info["mse_after_mm2"], info["mse_before_mm2"])
        self.assertTrue(self.bounds.valid(new, old[0]))
        np.testing.assert_array_equal(self.state, before_state)
        np.testing.assert_array_equal(old, before_old)

    def test_a_accounts_for_already_modified_suffix(self):
        old = self.actions.astype(float)
        target = self.fast.rollout(self.state, old)
        cache = CachedSuffixA(self.fast, self.state, old, target, blocks=3)
        perturbed = old+.01
        new, info = cache.correct(0, self.state, perturbed, old[0], self.bounds)
        self.assertTrue(info["accepted"])
        self.assertFalse(info["cache_stale"])
        self.assertLess(info["mse_after_mm2"], info["mse_before_mm2"])
        self.assertEqual(new.shape, old.shape)
        np.testing.assert_array_equal(cache.nominal, old)
        # Nominal state with nominal actions has zero compensation.
        same, zero = cache.correct(0, self.state, old, old[0], self.bounds)
        np.testing.assert_allclose(same, old, atol=1e-12)
        self.assertFalse(zero["accepted"])

    def test_a_stale_cache_rejects_without_mutating_working_plan(self):
        old = self.actions.astype(float)
        cache = CachedSuffixA(self.fast, self.state, old, self.fast.rollout(self.state, old),
                              blocks=3, max_model_discrepancy_mm=1e-12)
        changed = old+.06
        new, info = cache.correct(0, self.state, changed, old[0], self.bounds)
        self.assertTrue(info["cache_stale"])
        np.testing.assert_array_equal(new, changed)

    def test_vectorized_edges_match_reference_including_occlusion(self):
        image = np.full((220, 200, 3), 25, np.uint8)
        cv2.rectangle(image, (80, 15), (100, 205), (220, 220, 220), -1)
        hidden = np.zeros(image.shape[:2], bool)
        hidden[90:130, 60:120] = True
        image[hidden] = 35
        center = np.stack([np.ones(15)*90, np.linspace(20, 200, 15)], axis=1).astype(np.float32)
        for mask in (None, hidden):
            a = extract_edges(image, center, oracle_hidden=mask)
            b = extract_edges_vectorized(image, center, oracle_hidden=mask)
            np.testing.assert_array_equal(a.pixels, b.pixels)
            np.testing.assert_array_equal(a.segments, b.segments)
            np.testing.assert_allclose(a.strengths, b.strengths)


if __name__ == "__main__":
    unittest.main()
