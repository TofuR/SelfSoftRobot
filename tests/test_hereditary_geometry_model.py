"""Contract tests for the HOV2.1 operator-to-geometry fusion."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.training.train_transition import build_parser
from src.models.model_hereditary_geometry import HereditaryGeometryModel
from src.models.model_ishsm import generalized_to_skeleton
from src.training.trainer_unified import UnifiedTrainer
from src.utils.model_loader import load_model


def _model(*, residual_mode="memory"):
    torch.manual_seed(7)
    basis, _ = torch.linalg.qr(torch.randn(14, 8), mode="reduced")
    return HereditaryGeometryModel(
        action_dim=4,
        n_nodes=15,
        window_size=6,
        n_play=2,
        n_maxwell=3,
        dt=0.1,
        tau_range=(0.3, 2.0),
        burnin_mode="equilibrium",
        n_bend_modes=8,
        section_intervals=(7, 7),
        bend_basis=basis,
        generalized_coordinate_scale=torch.tensor(
            [0.08] * 8 + [0.01, 0.01]),
        reference_segment_lengths=torch.ones(14),
        reference_bend_bias=torch.zeros(14),
        reference_bend_dirs=torch.zeros(4, 14),
        reference_length_bias=torch.zeros(2),
        reference_length_dirs=torch.zeros(4, 2),
        base_position=torch.tensor([2.0, 3.0, 0.0]),
        residual_mode=residual_mode,
        episode_len=4,
    )


class HereditaryGeometryContractTests(unittest.TestCase):
    def test_constant_equilibrium_has_zero_memory_and_matches_h0(self):
        model = _model()
        window = torch.full((2, 6, 4), 0.4)
        state = model.init_z_from_action(window)
        result = model.forward(window, prev_z=state)

        expected = model.decode_equilibrium(window[:, -1])
        self.assertTrue(torch.allclose(
            result["memory_generalized"],
            torch.zeros_like(result["memory_generalized"]), atol=1e-7))
        self.assertTrue(torch.allclose(result["skeleton"], expected, atol=1e-6))

    def test_memory_residual_is_strictly_zero_at_zero_operator_state(self):
        model = _model(residual_mode="memory")
        q = torch.zeros(3, model.action_dim, model.n_play)
        deficit = torch.zeros(3, model.action_dim, model.n_maxwell)
        residual = model._memory_residual(q, deficit)
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_memory_residual_obeys_physical_coordinate_bounds(self):
        model = _model(residual_mode="memory")
        q = torch.randn(5, model.action_dim, model.n_play)
        deficit = torch.randn(5, model.action_dim, model.n_maxwell)
        residual = model._memory_residual(q, deficit)
        bounds = torch.tensor([0.05] * 8 + [0.02, 0.02])
        self.assertTrue(torch.all(residual.abs() <= bounds + 1e-7))

    def test_explicit_only_ablation_has_no_neural_residual_parameters(self):
        model = _model(residual_mode="none")
        keys = set(dict(model.named_parameters()))
        self.assertFalse(any(key.startswith("memory_residual_net") for key in keys))
        window = torch.rand(2, 6, 4)
        output = model.forward(window)
        self.assertTrue(torch.equal(
            output["memory_residual_generalized"],
            torch.zeros_like(output["memory_residual_generalized"])))

    def test_readout_directions_are_unit_norm_and_gains_nonnegative(self):
        model = _model()
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(model.pi_mode_directions, dim=-1),
            torch.ones(model.action_dim, model.n_play), atol=1e-6))
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(model.maxwell_mode_directions, dim=-1),
            torch.ones(model.action_dim, model.n_maxwell), atol=1e-6))
        self.assertTrue(torch.all(model.play.weights >= 0))
        self.assertTrue(torch.all(model.maxwell_gains >= 0))

    def test_model_has_no_free_point_coordinate_readout(self):
        model = _model()
        keys = set(dict(model.named_parameters()))
        self.assertNotIn("static_bias", keys)
        self.assertFalse(any(key.startswith("play_modes") for key in keys))
        self.assertFalse(any(key.startswith("maxwell_modes") for key in keys))
        self.assertFalse(any(key.startswith("residual.") for key in keys))
        self.assertIn("memory_residual_net.0.weight", keys)

    def test_cli_exposes_hov21_as_a_hereditary_readout(self):
        args = build_parser().parse_args([
            "--mode", "hereditary_geo", "--data_dir", "dummy"])
        self.assertEqual(args.hov21_residual, "none")
        self.assertEqual(args.n_bend_modes, 8)
        self.assertEqual(args.section_intervals, (7, 7))

    def test_loader_round_trips_geometry_and_operator_contract(self):
        model = _model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = (
                root / "phase_hereditary_geometry" / "model" /
                "best_eval_model.pt")
            checkpoint.parent.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint)
            (root / "config.json").write_text(json.dumps({
                "model": "HereditaryGeometryModel",
                "action_dim": 4,
                "n_nodes": 15,
                "window_size": 6,
                "n_play": 2,
                "n_maxwell": 3,
                "dt": 0.1,
                "tau_max": 2.0,
                "burnin_mode": "equilibrium",
                "n_bend_modes": 8,
                "section_intervals": [7, 7],
                "h0_reference": "fit_only_frozen_linear",
                "hov21_residual": "memory",
                "hov21_bend_residual_max_rad": 0.05,
                "hov21_length_residual_max_log": 0.02,
                "episode_len": 4,
            }), encoding="utf-8")

            info = load_model(str(checkpoint), device="cpu")

        self.assertEqual(info["model_type"], "hereditary_geometry")
        loaded = info["model"]
        self.assertTrue(torch.equal(loaded.bend_basis, model.bend_basis))
        self.assertTrue(torch.equal(
            loaded.generalized_coordinate_scale,
            model.generalized_coordinate_scale))
        self.assertEqual(loaded.residual_mode, "memory")

    def test_bend_auxiliary_uses_declared_modal_coordinates(self):
        model = _model(residual_mode="none")
        # Construct a bend perturbation orthogonal to the represented POD
        # subspace.  It must not leak into the modal-coordinate auxiliary loss.
        candidate = torch.randn(14)
        orthogonal = candidate - model.bend_basis @ (
            model.bend_basis.T @ candidate)
        orthogonal = orthogonal / orthogonal.norm()
        pred = generalized_to_skeleton(
            torch.zeros(1, 14), torch.zeros(1, 2),
            model.reference_segment_lengths, model.section_intervals,
            model.base_position)
        gt = generalized_to_skeleton(
            orthogonal.unsqueeze(0) * 0.02, torch.zeros(1, 2),
            model.reference_segment_lengths, model.section_intervals,
            model.base_position)

        losses = model.compute_sequence_aux_losses(
            pred.unsqueeze(1), gt.unsqueeze(1))

        self.assertLess(float(losses["bend"]), 1e-10)
        self.assertGreater(float(losses["endpoint"]), 0.0)

    def test_episode_training_backpropagates_geometry_and_residual_losses(self):
        model = _model(residual_mode="memory")
        trainer = UnifiedTrainer(model, config={"loss_weights": {
            "skeleton": 1.0,
            "spatial_smooth": 0.5,
            "bend": 0.05,
            "length": 0.1,
            "endpoint": 1.0,
            "residual_bend": 0.1,
            "residual_length": 0.1,
        }})
        trainer.device = torch.device("cpu")
        windows = torch.rand(2, 4, 6, 4)
        with torch.no_grad():
            state = model.init_z_from_action(windows[:, 0])
            predictions = []
            for step in range(4):
                output = model.forward(windows[:, step], prev_z=state)
                state = output["latent_z"]
                predictions.append(output["skeleton"])
        target = torch.stack(predictions, dim=1) + 0.01
        batch = {
            "action_windows": windows,
            "gt_skeletons": target,
            "init_skeleton": target[:, 0],
        }
        losses = trainer._compute_sequence_losses(
            batch, model.training_spec.phases[0])
        self.assertIn("residual_bend", losses)
        self.assertIn("residual_length", losses)
        losses["total"].backward()
        self.assertIsNotNone(model.pi_mode_directions_raw.grad)


if __name__ == "__main__":
    unittest.main()
