"""GL 时间对齐与骨架空间传播合同。"""

import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch


class FractionalMemoryDirectionTest(unittest.TestCase):
    def test_w0_is_applied_to_current_action_at_window_end(self):
        from src.encoders.fractional_memory import FractionalMemory

        encoder = FractionalMemory(
            action_dim=1, n_orders=1, window_size=4, hidden_dim=4)
        encoder.state_mlp = torch.nn.Identity()
        actions = torch.tensor([[[10.0], [20.0], [30.0], [40.0]]])
        alpha = encoder.alphas[0]
        weights = encoder._compute_gl_weights(alpha, 4)
        weights = weights / (weights.abs().sum() + 1e-8)
        expected = torch.sum(weights * actions[0, :, 0].flip(0))

        encoded = encoder(actions)

        self.assertTrue(torch.allclose(encoded[0, 0], expected))
        self.assertEqual(float(encoded[0, 1]), 40.0)
        self.assertEqual(float(encoded[0, 2]), 10.0)

    def test_deployment_copy_matches_training_encoder(self):
        from real_validation.runtime.model import FractionalMemory as RuntimeMemory
        from src.encoders.fractional_memory import FractionalMemory

        source = FractionalMemory(2, n_orders=2, window_size=5, hidden_dim=8)
        runtime = RuntimeMemory(2, n_orders=2, window_size=5, hidden_dim=8)
        runtime.load_state_dict(source.state_dict())
        actions = torch.randn(3, 5, 2)
        with torch.no_grad():
            self.assertTrue(torch.equal(source(actions), runtime(actions)))


class SpatialDirectionContractTest(unittest.TestCase):
    def test_centerline_is_base_to_tip(self):
        from real_validation.perception.skeleton import extract_skeleton_2d

        mask = np.zeros((80, 40), dtype=np.uint8)
        mask[10:70, 17:23] = 1
        skeleton = extract_skeleton_2d(mask, n_points=15, tip_fix=True)
        self.assertLess(skeleton[0, 1], skeleton[-1, 1])

    def test_state_transition_gru_sequence_is_base_to_tip(self):
        from src.models.model_state_transition import StateTransitionSpatialModel

        model = StateTransitionSpatialModel(
            action_dim=1, n_nodes=5, hidden_dim=4, window_size=3,
            n_orders=1, z_dim=2)
        captured = {}

        def capture_input(_module, inputs):
            captured["sequence"] = inputs[0].detach().clone()

        handle = model.gru.register_forward_pre_hook(capture_input)
        try:
            model(torch.zeros(1, 3, 1))
        finally:
            handle.remove()

        positions = model._get_z_positions(torch.device("cpu"))
        expected_delta = model.z_embed(positions.view(5, 1))[1:] - \
            model.z_embed(positions.view(5, 1))[:-1]
        actual_delta = captured["sequence"][0, 1:] - captured["sequence"][0, :-1]
        self.assertEqual(model.node_order, "base_to_tip")
        self.assertEqual(model.spatial_propagation_direction, "base_to_tip")
        self.assertTrue(torch.allclose(actual_delta, expected_delta, atol=1e-6))

    def test_transition_dataset_requires_explicit_base_to_tip_metadata(self):
        from src.data.dataset_spatial import StateTransitionDataset

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "seq.npz"
            np.savez_compressed(
                path, positions=np.zeros((4, 3, 5), dtype=np.float32),
                actions=np.ones((4, 1), dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "node_order=base_to_tip"):
                StateTransitionDataset(temporary, seq_len=2)


if __name__ == "__main__":
    unittest.main()
