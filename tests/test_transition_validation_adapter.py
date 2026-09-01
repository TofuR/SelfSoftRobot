import csv
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from scripts.training.train_transition import configure_transition_validation
from src.evaluation.real_transition_validation import evaluate_native_node_metrics
from src.evaluation.transition_metrics import evaluate_transition_rollout
from src.training.spec import PhaseSpec
from src.training.validation_adapters import transition_validation_adapter


class _CopyTransitionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.action_dim = 1
        self.window_size = 2
        self.episode_len = 2
        self.register_buffer("pc_center", torch.zeros(1, 1, 3))
        self.register_buffer("pc_scale", torch.ones(1, 1, 3))
        self.register_buffer("action_norm_factor", torch.tensor(1.0))

    def init_z_from_action(self, action_window):
        return torch.zeros(action_window.shape[0], 1, device=action_window.device)

    def forward(self, action_window, prev, prev2, latent_z):
        del action_window, prev2
        return {"skeleton": prev + self.anchor * 0, "latent_z": latent_z}


class TransitionValidationAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp.name) / "val"
        self.data_dir.mkdir()
        positions = np.zeros((6, 3, 3), np.float32)
        positions[:, 0, 0] = np.arange(6)
        positions[:, 0, 1] = 2 * np.arange(6)
        positions[:, 0, 2] = 3 * np.arange(6)
        np.savez(
            self.data_dir / "seq_20260901_000000_val.npz",
            actions=np.zeros((6, 1), np.float32),
            positions=positions,
            state_length_unit=np.array("mm"),
        )
        self.model = _CopyTransitionModel()
        self.config = {
            "temporal": {"window_size": 2},
            "action_view": {"model_action_channels": [0]},
            "evaluation": {"transition_validation_max_steps": 6},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_native_node_mean_matches_external_csv_aggregation(self):
        metrics, details = evaluate_native_node_metrics(
            self.model, self.data_dir, self.config, torch.device("cpu"),
            mode="gt", max_steps=6, return_details=True)
        csv_path = Path(self.temp.name) / "per_frame.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("is_prediction", "node_mean_mm"))
            writer.writeheader()
            for valid, value in zip(
                    details["prediction_valid"], details["per_frame_node"]):
                writer.writerow({
                    "is_prediction": int(valid),
                    "node_mean_mm": float(value),
                })

        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        csv_values = [float(row["node_mean_mm"]) for row in rows
                      if row["is_prediction"] == "1"]
        csv_score = sum(csv_values) / len(csv_values)

        self.assertAlmostEqual(
            metrics["validation.node_mean_mm"], csv_score)
        self.assertAlmostEqual(csv_score, 5.0 / 3.0)

    def test_adapter_uses_openloop_semantics_from_phase_name(self):
        phase = PhaseSpec("open_loop_transition", episode_len=2)
        metrics = transition_validation_adapter(
            model=self.model,
            phase_spec=phase,
            data_dir=self.data_dir,
            epoch=1,
            exp_dir=self.temp.name,
            device=torch.device("cpu"),
            config=self.config,
        )

        self.assertIn("validation.node_mean_mm", metrics)
        self.assertEqual(metrics["validation.prediction_rows"], 5.0)

    def test_training_cli_contract_enables_validation_only_when_requested(self):
        phase = PhaseSpec("gt_transition")
        config = {}
        args = SimpleNamespace(
            val_dir=str(self.data_dir),
            validation_interval=3,
            validation_max_steps=5,
            validation_min_delta=0.01,
            validation_warmup=2,
            early_stopping_patience=4,
            allow_early_stop_before_tf_anneal=False,
        )

        data_dirs, adapters = configure_transition_validation(
            args, phase, config)

        self.assertEqual(data_dirs, {"gt_transition": str(self.data_dir)})
        self.assertIs(adapters["gt_transition"], transition_validation_adapter)
        self.assertEqual(
            phase.validation.selection_metric, "validation.node_mean_mm")
        self.assertEqual(phase.validation.eval_interval_epochs, 3)
        self.assertEqual(
            phase.validation.early_stopping_patience_evaluations, 4)
        self.assertEqual(
            config["evaluation"]["transition_validation_max_steps"], 5)

        args.val_dir = None
        data_dirs, adapters = configure_transition_validation(args, phase, config)
        self.assertIsNone(data_dirs)
        self.assertIsNone(adapters)
        self.assertIsNone(phase.validation)

    def test_rollout_diagnostic_does_not_scale_mm_state_twice(self):
        result = evaluate_transition_rollout(
            self.model,
            self.data_dir,
            self.config,
            torch.device("cpu"),
            n_seqs=1,
            windows_per_seq=1,
            K=2,
        )

        self.assertAlmostEqual(
            result["summary"]["mean_node_mm"], 3.0, places=5)
        self.assertEqual(
            result["summary"]["physical_scales"], [{
                "state_unit": "mm",
                "source": "state_length_unit:mm",
                "state_to_m": 0.001,
            }])


if __name__ == "__main__":
    unittest.main()
