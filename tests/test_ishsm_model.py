import unittest
from unittest import mock
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.models.model_ishsm import (
    ISHSMModel,
    fit_ishsm_priors_from_arrays,
    generalized_to_skeleton,
    skeleton_to_generalized,
)
from src.training.trainer_unified import UnifiedTrainer
from src.evaluation.real_transition_validation import (
    evaluate_native_node_metrics, per_frame_rollout)
from src.utils.model_loader import load_model
from scripts.training.train_transition import build_parser


class ISHSMGeometryTests(unittest.TestCase):
    def test_generalized_shape_round_trip_with_two_section_stretch(self):
        bend = torch.tensor([
            [1.20, 0.04, -0.02, 0.03, 0.01, -0.01, 0.02,
             0.03, -0.02, 0.01, 0.02, -0.03, 0.01, 0.02],
            [1.45, -0.02, 0.03, -0.01, 0.04, 0.01, -0.03,
             0.02, 0.01, -0.02, 0.03, 0.01, -0.01, 0.02],
        ])
        log_section_scale = torch.tensor([[0.03, -0.02], [-0.01, 0.04]])
        ref_lengths = torch.linspace(11.8, 13.1, 14)
        base = torch.tensor([[2.0, -3.0, 0.0], [-1.0, 4.0, 0.0]])

        skeleton = generalized_to_skeleton(
            bend, log_section_scale, ref_lengths, (7, 7), base)
        got_bend, got_scale, got_base = skeleton_to_generalized(
            skeleton, ref_lengths, (7, 7))

        self.assertTrue(torch.allclose(got_bend, bend, atol=1e-5))
        self.assertTrue(torch.allclose(got_scale, log_section_scale, atol=1e-5))
        self.assertTrue(torch.allclose(got_base, base, atol=1e-6))

    def test_invalid_section_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            generalized_to_skeleton(
                torch.zeros(1, 14), torch.zeros(1, 2),
                torch.ones(14), (6, 7), torch.zeros(1, 3))


class ISHSMStateTests(unittest.TestCase):
    def _model(self):
        basis = torch.eye(14)[:, :8]
        model = ISHSMModel(
            action_dim=4,
            n_nodes=15,
            n_bend_modes=8,
            section_intervals=(7, 7),
            dt=0.1,
            tau_range=(0.3, 2.0),
            bend_basis=basis,
            reference_segment_lengths=torch.ones(14),
            reference_bend_bias=torch.zeros(14),
            reference_bend_dirs=torch.zeros(4, 14),
            reference_length_bias=torch.zeros(2),
            reference_length_dirs=torch.zeros(4, 2),
        )
        model.set_normalization(np.zeros(3), np.ones(3), 1.0)
        return model

    def test_anchor_observation_recovers_known_8_plus_2_state(self):
        model = self._model()
        z_bend = torch.tensor([[0.20, -0.10, 0.05, 0.03, -0.02, 0.01, 0.04, -0.03]])
        z_length = torch.tensor([[0.02, -0.01]])
        bend = torch.zeros(1, 14)
        bend[:, :8] = z_bend
        skeleton = generalized_to_skeleton(
            bend, z_length, torch.ones(14), (7, 7), torch.zeros(1, 3))
        action_window = torch.zeros(1, 4, 4)

        state = model.init_rollout_state(action_window, skeleton)

        self.assertEqual(tuple(state.shape), (1, 10))
        self.assertTrue(torch.allclose(state[:, :8], z_bend, atol=1e-5))
        self.assertTrue(torch.allclose(state[:, 8:], z_length, atol=1e-5))

    def test_tip_dls_projection_reduces_unrepresented_anchor_tip_error(self):
        common = dict(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7),
            bend_basis=torch.eye(14)[:, :8],
            reference_segment_lengths=torch.full((14,), 10.0),
            reference_bend_bias=torch.zeros(14),
            reference_bend_dirs=torch.zeros(4, 14),
            reference_length_bias=torch.zeros(2),
            reference_length_dirs=torch.zeros(4, 2),
            observation_update="hard")
        modal = ISHSMModel(**common, observation_projection="modal")
        tip_dls = ISHSMModel(
            **common, observation_projection="tip_dls",
            tip_dls_lambda_mm2=1.0)
        for model in (modal, tip_dls):
            model.set_normalization(np.zeros(3), np.ones(3), 1.0)

        # This turning angle is outside the first eight POD coordinates, so
        # the ordinary modal projection cannot reproduce its endpoint.
        observed_bend = torch.zeros(1, 14)
        observed_bend[:, 10] = 0.20
        observed = generalized_to_skeleton(
            observed_bend, torch.zeros(1, 2), torch.full((14,), 10.0),
            (7, 7), torch.zeros(1, 3))
        action_window = torch.zeros(1, 4, 4)

        modal_state = modal.init_rollout_state(action_window, observed)
        dls_state = tip_dls.init_rollout_state(action_window, observed)
        modal_reconstruction = modal.decode_state(
            action_window[:, -2], modal_state)
        dls_reconstruction = tip_dls.decode_state(
            action_window[:, -2], dls_state)
        modal_tip_error = torch.linalg.vector_norm(
            modal_reconstruction[:, -1, :2] - observed[:, -1, :2])
        dls_tip_error = torch.linalg.vector_norm(
            dls_reconstruction[:, -1, :2] - observed[:, -1, :2])

        self.assertLess(dls_tip_error.item(), 0.2 * modal_tip_error.item())
        self.assertTrue(torch.isfinite(dls_reconstruction).all())

    def test_constant_action_makes_nonzero_state_decay(self):
        model = self._model()
        with torch.no_grad():
            model.excitation.zero_()
        prev_state = torch.full((1, 10), 0.2)
        action_window = torch.full((1, 4, 4), 0.4)

        out = model.forward(action_window, prev_z=prev_state)

        self.assertTrue(torch.all(out["latent_z"].abs() < prev_state.abs()))
        self.assertTrue(torch.all(model.taus >= 0.3))
        self.assertTrue(torch.all(model.taus <= 2.0))

    def test_shared_bending_tau_keeps_spatial_modes_but_reduces_time_parameters(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7), tau_parameterization="shared_bending")

        self.assertEqual(model.raw_taus.numel(), 3)
        self.assertEqual(tuple(model.taus.shape), (10,))
        self.assertTrue(torch.allclose(
            model.taus[:8], model.taus[0].expand(8)))
        self.assertFalse(torch.isclose(model.taus[8], model.taus[9]))

    def test_grouped_innovation_observer_applies_bounded_causal_update(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7),
            observation_update="innovation", observation_gain_init=0.25,
            bend_basis=torch.eye(14)[:, :8],
            reference_segment_lengths=torch.ones(14),
            reference_bend_bias=torch.zeros(14),
            reference_bend_dirs=torch.zeros(4, 14),
            reference_length_bias=torch.zeros(2),
            reference_length_dirs=torch.zeros(4, 2))
        model.set_normalization(np.zeros(3), np.ones(3), 1.0)
        observed_state = torch.tensor([[
            0.20, -0.10, 0.05, 0.03, -0.02,
            0.01, 0.04, -0.03, 0.02, -0.01]])
        bend = torch.zeros(1, 14)
        bend[:, :8] = observed_state[:, :8]
        skeleton = generalized_to_skeleton(
            bend, observed_state[:, 8:], torch.ones(14),
            (7, 7), torch.zeros(1, 3))
        predicted = torch.full((1, 10), 0.1)

        updated = model.assimilate_observation(
            torch.zeros(1, 4, 4), skeleton, predicted)

        gains = model.observation_gains
        self.assertEqual(tuple(gains.shape), (10,))
        self.assertTrue(torch.all((gains > 0.0) & (gains < 1.0)))
        self.assertTrue(torch.allclose(gains[:8], gains[0].expand(8)))
        self.assertTrue(torch.allclose(gains[8:], gains[8].expand(2)))
        self.assertTrue(torch.allclose(
            updated, predicted + gains * (observed_state - predicted),
            atol=1e-5))

    def test_persistent_anchor_component_survives_constant_action(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7), use_persistent_state=True,
            persistence_init=0.25,
            bend_basis=torch.eye(14)[:, :8],
            reference_segment_lengths=torch.ones(14),
            reference_bend_bias=torch.zeros(14),
            reference_bend_dirs=torch.zeros(4, 14),
            reference_length_bias=torch.zeros(2),
            reference_length_dirs=torch.zeros(4, 2))
        model.set_normalization(np.zeros(3), np.ones(3), 1.0)
        bend = torch.zeros(1, 14)
        bend[:, 0] = 0.2
        anchor = generalized_to_skeleton(
            bend, torch.tensor([[0.02, -0.01]]), torch.ones(14),
            (7, 7), torch.zeros(1, 3))
        action_window = torch.zeros(1, 4, 4)

        state = model.init_rollout_state(action_window, anchor)
        initial = model.decode_state(action_window[:, -1], state)
        for _ in range(100):
            state = model.forward(action_window, prev_z=state)["latent_z"]

        transient, persistent = model.unpack_memory_state(state)
        self.assertEqual(tuple(state.shape), (1, 20))
        self.assertTrue(torch.allclose(initial, anchor, atol=1e-5))
        self.assertLess(transient.abs().max().item(), 1e-4)
        self.assertGreater(persistent.abs().max().item(), 0.01)

    def test_forward_does_not_use_skeleton_after_state_is_initialized(self):
        model = self._model()
        action_window = torch.zeros(1, 4, 4)
        state = torch.zeros(1, 10)
        a = model.forward(
            action_window, prev_skeleton=torch.zeros(1, 15, 3), prev_z=state)
        b = model.forward(
            action_window, prev_skeleton=torch.randn(1, 15, 3), prev_z=state)
        self.assertTrue(torch.allclose(a["skeleton"], b["skeleton"]))
        self.assertTrue(torch.allclose(a["latent_z"], b["latent_z"]))

    def test_single_anchor_validation_reads_observation_once(self):
        model = self._model()
        positions = np.zeros((6, 3, 15), dtype=np.float32)
        positions[:, 0, :] = np.arange(15, dtype=np.float32)
        actions = np.zeros((6, 4), dtype=np.float32)

        with mock.patch.object(
                model, "init_rollout_state",
                wraps=model.init_rollout_state) as init_from_anchor:
            _, horizon = per_frame_rollout(
                model, "ishsm", actions, positions, window_size=4,
                norm_factor=1.0, device=torch.device("cpu"), K=40)

        init_from_anchor.assert_called_once()
        self.assertTrue(np.array_equal(horizon, [-1, 0, 1, 2, 3, 4]))

    def test_periodic_validation_reanchors_without_dense_history(self):
        model = self._model()
        actions = np.zeros((6, 4), dtype=np.float32)
        positions = np.zeros((6, 3, 15), dtype=np.float32)
        positions[:, 0, :] = np.arange(15, dtype=np.float32)
        with mock.patch.object(
                model, "init_rollout_state",
                wraps=model.init_rollout_state) as init_from_anchor, \
                mock.patch.object(
                    model, "assimilate_observation",
                    wraps=model.assimilate_observation) as assimilate:
            _, horizon = per_frame_rollout(
                model, "ishsm_periodic", actions, positions, window_size=4,
                norm_factor=1.0, device=torch.device("cpu"), K=2)

        self.assertEqual(init_from_anchor.call_count, 1)
        # init_rollout_state delegates to one assimilation with no prediction;
        # the two later calls are true periodic innovation corrections.
        self.assertEqual(assimilate.call_count, 3)
        self.assertIsNone(
            assimilate.call_args_list[0].kwargs.get("predicted_state"))
        self.assertTrue(all(
            call.args[2] is not None for call in assimilate.call_args_list[1:]))
        self.assertTrue(np.array_equal(horizon, [-1, 0, 1, 0, 1, 0]))

    def test_validation_excludes_read_only_context_mask(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as td:
            positions = np.zeros((6, 3, 15), dtype=np.float32)
            positions[:, 0, :] = np.arange(15, dtype=np.float32)
            np.savez_compressed(
                Path(td) / "dev.npz",
                actions=np.zeros((6, 4), dtype=np.float32),
                positions=positions,
                model_action_channels=np.arange(4),
                evaluation_mask=np.array(
                    [False, False, False, True, True, True]),
                state_length_unit=np.array("mm"),
            )
            metrics = evaluate_native_node_metrics(
                model, td, {"action_view": {
                    "model_action_channels": [0, 1, 2, 3]}},
                torch.device("cpu"), mode="ishsm")

        self.assertEqual(metrics["validation.prediction_rows"], 3.0)

    def test_validation_can_aggregate_all_dev_sequences(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as td:
            for name in ("a", "b"):
                positions = np.zeros((5, 3, 15), dtype=np.float32)
                positions[:, 0, :] = np.arange(15, dtype=np.float32)
                np.savez_compressed(
                    Path(td) / f"{name}.npz",
                    actions=np.zeros((5, 4), dtype=np.float32),
                    positions=positions,
                    model_action_channels=np.arange(4),
                    evaluation_mask=np.array(
                        [False, False, True, True, True]),
                    state_length_unit=np.array("mm"),
                )
            metrics = evaluate_native_node_metrics(
                model, td, {"action_view": {
                    "model_action_channels": [0, 1, 2, 3]}},
                torch.device("cpu"), mode="ishsm", seq_idx=None)

        self.assertEqual(metrics["validation.prediction_rows"], 6.0)

    def test_no_dynamic_length_ablation_has_only_eight_states(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7), use_dynamic_length=False)
        self.assertEqual(model.z_dim, 8)
        self.assertEqual(model.state_report()["n_length_states"], 0)

    def test_episode_trainer_initializes_from_anchor_and_adds_aux_losses(self):
        model = self._model()
        actions = torch.zeros(1, 2, 4, 4)
        bend = torch.zeros(1, 14)
        lengths = torch.zeros(1, 2)
        anchor = generalized_to_skeleton(
            bend, lengths, torch.ones(14), (7, 7), torch.zeros(1, 3))
        batch = {
            "action_windows": actions,
            "gt_skeletons": anchor.unsqueeze(1).repeat(1, 2, 1, 1),
            "init_skeleton": anchor,
        }
        trainer = UnifiedTrainer(
            model, config={"loss_weights": {"bend": 0.2, "length": 0.1}})
        trainer.device = next(model.parameters()).device

        with mock.patch.object(
                model, "init_rollout_state",
                wraps=model.init_rollout_state) as init_from_anchor:
            losses = trainer._compute_sequence_losses(
                batch, model.training_spec.phases[0])

        init_from_anchor.assert_called_once()
        self.assertIn("bend", losses)
        self.assertIn("length", losses)
        self.assertIn("endpoint", losses)

    def test_episode_trainer_causally_reanchors_at_selected_sparse_interval(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7),
            observation_update="innovation", observation_gain_init=0.25,
            training_reanchor_intervals=(2,))
        model.set_normalization(np.zeros(3), np.ones(3), 1.0)
        actions = torch.zeros(1, 4, 4, 4)
        skeleton = generalized_to_skeleton(
            torch.zeros(1, 14), torch.zeros(1, 2), torch.ones(14),
            (7, 7), torch.zeros(1, 3))
        batch = {
            "action_windows": actions,
            "gt_skeletons": skeleton.unsqueeze(1).repeat(1, 4, 1, 1),
            "init_skeleton": skeleton,
        }
        trainer = UnifiedTrainer(model, config={"loss_weights": {}})
        trainer.device = next(model.parameters()).device

        with mock.patch.object(
                model, "assimilate_observation",
                wraps=model.assimilate_observation) as assimilate:
            trainer._compute_sequence_losses(
                batch, model.training_spec.phases[0])

        self.assertEqual(assimilate.call_count, 2)
        second_observation = assimilate.call_args_list[1].args[1]
        self.assertTrue(torch.equal(second_observation, batch["gt_skeletons"][:, 1]))


class ISHSMPriorFitTests(unittest.TestCase):
    def test_fit_priors_produces_orthonormal_fit_only_basis(self):
        rng = np.random.default_rng(7)
        actions = rng.uniform(0.0, 1.0, size=(80, 4)).astype(np.float32)
        ref_lengths = np.linspace(12.0, 13.0, 14).astype(np.float32)
        basis = np.eye(14, dtype=np.float32)[:, :3]
        coeff = rng.normal(0.0, 0.04, size=(80, 3)).astype(np.float32)
        bend = np.zeros((80, 14), dtype=np.float32)
        bend[:, 0] = 1.4 + 0.1 * actions[:, 0]
        bend += coeff @ basis.T
        section_scale = np.stack(
            [0.02 * actions[:, 1], -0.01 * actions[:, 2]], axis=1)
        skeletons = generalized_to_skeleton(
            torch.from_numpy(bend), torch.from_numpy(section_scale),
            torch.from_numpy(ref_lengths), (7, 7), torch.zeros(80, 3),
        ).numpy()

        priors = fit_ishsm_priors_from_arrays(
            actions, skeletons, n_bend_modes=8, section_intervals=(7, 7))

        phi = priors["bend_basis"]
        self.assertEqual(phi.shape, (14, 8))
        self.assertTrue(np.allclose(phi.T @ phi, np.eye(8), atol=1e-5))
        self.assertEqual(priors["reference_bend_dirs"].shape, (4, 14))
        self.assertEqual(priors["reference_length_dirs"].shape, (4, 2))
        self.assertGreater(priors["bend_explained_energy"], 0.95)

    def test_monotone_spline_h0_fits_nonlinear_reference_better_than_linear(self):
        rng = np.random.default_rng(11)
        actions = rng.uniform(0.0, 1.0, size=(240, 4)).astype(np.float32)
        bend = np.zeros((240, 14), dtype=np.float32)
        bend[:, 0] = 1.2 + 0.5 * actions[:, 0] ** 2
        bend[:, 1] = 0.2 * np.maximum(actions[:, 1] - 0.35, 0.0)
        skeletons = generalized_to_skeleton(
            torch.from_numpy(bend), torch.zeros(240, 2), torch.ones(14),
            (7, 7), torch.zeros(240, 3)).numpy()

        linear = fit_ishsm_priors_from_arrays(
            actions, skeletons, n_bend_modes=4,
            section_intervals=(7, 7), reference_kind="linear")
        spline = fit_ishsm_priors_from_arrays(
            actions, skeletons, n_bend_modes=4,
            section_intervals=(7, 7), reference_kind="monotone_spline",
            n_reference_knots=5, reference_fit_steps=400)

        self.assertLess(spline["reference_fit_mse"],
                        0.65 * linear["reference_fit_mse"])
        self.assertTrue(np.all(spline["reference_drive_weights"] >= 0.0))
        self.assertEqual(spline["reference_knots"].shape, (5,))

    def test_geometry_h0_objective_prioritizes_reconstructed_endpoint(self):
        actions = np.zeros((240, 4), dtype=np.float32)
        actions[:, 0] = np.linspace(0.0, 1.0, len(actions))
        bend = np.zeros((len(actions), 14), dtype=np.float32)
        bend[:, 0] = 1.1 + 0.25 * actions[:, 0] ** 2
        bend[:, -1] = 0.25 * np.sqrt(actions[:, 0])
        skeletons = generalized_to_skeleton(
            torch.from_numpy(bend), torch.zeros(len(actions), 2),
            torch.ones(14), (7, 7), torch.zeros(len(actions), 3)).numpy()

        coordinate = fit_ishsm_priors_from_arrays(
            actions, skeletons, n_bend_modes=4,
            section_intervals=(7, 7), reference_kind="monotone_spline",
            n_reference_knots=5, reference_fit_steps=400,
            reference_fit_objective="coordinate")
        geometry = fit_ishsm_priors_from_arrays(
            actions, skeletons, n_bend_modes=4,
            section_intervals=(7, 7), reference_kind="monotone_spline",
            n_reference_knots=5, reference_fit_steps=400,
            reference_fit_objective="geometry",
            reference_geometry_weight=1.0,
            reference_endpoint_weight=4.0)

        self.assertLess(
            geometry["reference_fit_endpoint_rmse_mm"],
            coordinate["reference_fit_endpoint_rmse_mm"])
        self.assertGreater(geometry["reference_fit_node_rmse_mm"], 0.0)

    def test_transition_cli_accepts_ishsm_mode(self):
        args = build_parser().parse_args([
            "--mode", "ishsm", "--data_dir", "dummy", "--n_bend_modes", "8"])
        self.assertEqual(args.mode, "ishsm")
        self.assertEqual(args.n_bend_modes, 8)
        self.assertEqual(args.h0_reference, "monotone_spline")
        self.assertEqual(args.h0_fit_objective, "geometry")
        self.assertEqual(args.endpoint_loss_weight, 1.0)
        self.assertFalse(args.use_persistent_state)
        self.assertEqual(args.ishsm_validation_protocol, "single_anchor")
        self.assertEqual(args.ishsm_observation_update, "innovation")
        self.assertEqual(args.ishsm_observation_gain_init, 0.25)
        self.assertEqual(args.ishsm_training_reanchor_intervals, (0, 5, 10, 20))
        self.assertEqual(args.ishsm_tau_parameterization, "shared_bending")
        self.assertEqual(args.ishsm_observation_projection, "tip_dls")
        self.assertEqual(args.ishsm_tip_dls_lambda_mm2, 1.0)

        enabled = build_parser().parse_args([
            "--mode", "ishsm", "--data_dir", "dummy",
            "--enable_persistent_state"])
        self.assertTrue(enabled.use_persistent_state)

    def test_loader_strictly_round_trips_ishsm_contract_and_buffers(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7), dt=0.1, tau_range=(0.3, 2.0))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "phase_ishsm" / "model" / "best_eval_model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint)
            (root / "config.json").write_text(json.dumps({
                "model": "ISHSMModel", "action_dim": 4, "n_nodes": 15,
                "n_bend_modes": 8, "section_intervals": [7, 7],
                "dt": 0.1, "tau_min": 0.3, "tau_max": 2.0,
                "episode_len": 40, "use_dynamic_length": True,
                "window_size": 40,
            }), encoding="utf-8")

            loaded = load_model(str(checkpoint), device="cpu")

        self.assertEqual(loaded["model_type"], "ishsm")
        self.assertTrue(torch.equal(
            loaded["model"].bend_basis, model.bend_basis))

    def test_loader_round_trips_monotone_persistent_v2(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=4,
            section_intervals=(7, 7), reference_kind="monotone_spline",
            reference_knots=torch.linspace(0, 1, 5),
            reference_drive_weights=torch.ones(4, 5),
            use_persistent_state=True, persistence_init=0.1)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "phase_ishsm" / "model" / "best_eval_model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint)
            (root / "config.json").write_text(json.dumps({
                "model": "ISHSMModel", "action_dim": 4, "n_nodes": 15,
                "n_bend_modes": 4, "section_intervals": [7, 7],
                "dt": 0.1, "tau_min": 0.3, "tau_max": 2.0,
                "episode_len": 40, "use_dynamic_length": True,
                "use_persistent_state": True, "persistence_init": 0.1,
                "h0_reference": "fit_only_frozen_monotone_spline",
                "window_size": 40,
            }), encoding="utf-8")

            loaded = load_model(str(checkpoint), device="cpu")["model"]

        self.assertTrue(loaded.use_persistent_state)
        self.assertEqual(loaded.reference_kind, "monotone_spline")
        self.assertTrue(torch.equal(
            loaded.reference_drive_weights, model.reference_drive_weights))

    def test_loader_round_trips_v4_observer_and_shared_tau_contract(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7),
            observation_update="innovation", observation_gain_init=0.25,
            training_reanchor_intervals=(0, 5, 10, 20),
            tau_parameterization="shared_bending")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "phase_ishsm" / "model" / "best_eval_model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint)
            (root / "config.json").write_text(json.dumps({
                "model": "ISHSMModel", "action_dim": 4, "n_nodes": 15,
                "n_bend_modes": 8, "section_intervals": [7, 7],
                "dt": 0.1, "tau_min": 0.3, "tau_max": 2.0,
                "episode_len": 40, "use_dynamic_length": True,
                "use_persistent_state": False,
                "observation_update": "innovation",
                "observation_gain_init": 0.25,
                "training_reanchor_intervals": [0, 5, 10, 20],
                "tau_parameterization": "shared_bending",
                "window_size": 40,
            }), encoding="utf-8")

            loaded = load_model(str(checkpoint), device="cpu")["model"]

        self.assertEqual(loaded.observation_update, "innovation")
        self.assertEqual(loaded.training_reanchor_intervals, (0, 5, 10, 20))
        self.assertEqual(loaded.tau_parameterization, "shared_bending")
        self.assertTrue(torch.equal(
            loaded.raw_observation_gains, model.raw_observation_gains))

    def test_loader_round_trips_tip_dls_projection_contract(self):
        model = ISHSMModel(
            action_dim=4, n_nodes=15, n_bend_modes=8,
            section_intervals=(7, 7),
            observation_update="innovation",
            tau_parameterization="shared_bending",
            observation_projection="tip_dls", tip_dls_lambda_mm2=2.5)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "phase_ishsm" / "model" / "best_eval_model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(model.state_dict(), checkpoint)
            (root / "config.json").write_text(json.dumps({
                "model": "ISHSMModel", "action_dim": 4, "n_nodes": 15,
                "n_bend_modes": 8, "section_intervals": [7, 7],
                "dt": 0.1, "tau_min": 0.3, "tau_max": 2.0,
                "episode_len": 40, "use_dynamic_length": True,
                "use_persistent_state": False,
                "observation_update": "innovation",
                "observation_gain_init": 0.25,
                "training_reanchor_intervals": [0, 5, 10, 20],
                "tau_parameterization": "shared_bending",
                "observation_projection": "tip_dls",
                "tip_dls_lambda_mm2": 2.5,
                "window_size": 40,
            }), encoding="utf-8")

            loaded = load_model(str(checkpoint), device="cpu")["model"]

        self.assertEqual(loaded.observation_projection, "tip_dls")
        self.assertEqual(loaded.tip_dls_lambda_mm2, 2.5)


if __name__ == "__main__":
    unittest.main()
