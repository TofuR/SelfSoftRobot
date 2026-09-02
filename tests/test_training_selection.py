import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.training.selection import SelectionState
from src.training.spec import PhaseSpec, TrainingSpec, ValidationSpec
from src.training.trainer_unified import UnifiedTrainer


class TrainingSelectionTest(unittest.TestCase):
    def test_validation_contract_rejects_ambiguous_semantics(self):
        with self.assertRaisesRegex(ValueError, "dataset_role"):
            ValidationSpec(selection_metric="node_mean", dataset_role="test")
        with self.assertRaisesRegex(ValueError, "selection_mode"):
            ValidationSpec(selection_metric="node_mean", selection_mode="median")
        with self.assertRaisesRegex(ValueError, "正整数"):
            ValidationSpec(selection_metric="node_mean", eval_interval_epochs=0)
        with self.assertRaisesRegex(ValueError, "正整数或 null"):
            ValidationSpec(
                selection_metric="node_mean",
                early_stopping_patience_evaluations=0)
        with self.assertRaisesRegex(ValueError, "min_delta"):
            ValidationSpec(selection_metric="node_mean", min_delta=float("nan"))
        with self.assertRaisesRegex(ValueError, "bool"):
            ValidationSpec(
                selection_metric="node_mean", restore_best_at_end="yes")

    def test_min_selection_uses_min_delta_warmup_and_validation_patience(self):
        state = SelectionState(ValidationSpec(
            selection_metric="validation.node_mean_mm",
            min_delta=0.1,
            warmup_evaluations=1,
            early_stopping_patience_evaluations=2,
        ))

        first = state.observe(1.0, epoch=2)
        second = state.observe(0.95, epoch=4)
        third = state.observe(0.94, epoch=6)

        self.assertTrue(first.improved)
        self.assertFalse(second.improved)
        self.assertEqual(second.bad_evaluations, 1)
        self.assertTrue(third.should_stop)
        self.assertEqual(third.best_epoch, 2)
        self.assertEqual(third.reason, "patience_exhausted")

    def test_max_selection_and_early_stop_gate(self):
        state = SelectionState(ValidationSpec(
            selection_metric="validation.f_score",
            selection_mode="max",
            early_stopping_patience_evaluations=1,
        ))
        self.assertTrue(state.observe(0.5, epoch=1).improved)
        gated = state.observe(
            0.4, epoch=2, early_stopping_allowed=False)
        self.assertFalse(gated.should_stop)
        self.assertEqual(gated.bad_evaluations, 0)
        self.assertEqual(gated.reason, "early_stopping_gated")
        stopped = state.observe(0.4, epoch=3)
        self.assertTrue(stopped.should_stop)

    def test_validation_contract_is_recorded_in_resolved_phase_config(self):
        validation = ValidationSpec(
            selection_metric="validation.rollout_node_mean_mm",
            eval_interval_epochs=5,
            early_stopping_patience_evaluations=4,
        )
        trainer = UnifiedTrainer.__new__(UnifiedTrainer)
        trainer.model = torch.nn.Linear(1, 1)
        trainer.model_tag = "open_loop_transition"
        trainer.device = torch.device("cpu")
        trainer.views = None
        trainer.phase = type("Phase", (), {"spec": type("Spec", (), {
            "phases": [PhaseSpec("open_loop", validation=validation)],
        })()})()
        trainer.config = {
            "optimization": {
                "lr": 1e-3,
                "batch_size": 2,
                "n_epochs": 10,
                "scheduler_patience": 3,
                "seed": 42,
            },
        }

        config = trainer._build_exp_config({}, None)

        recorded = config["phases"][0]["validation"]
        self.assertEqual(
            recorded["selection_metric"],
            "validation.rollout_node_mean_mm")
        self.assertEqual(recorded["lr_scheduler_metric"],
                         recorded["selection_metric"])
        self.assertEqual(recorded["early_stopping_patience_evaluations"], 4)

    def _tiny_trainer(self, validation):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.training_spec = TrainingSpec(phases=[PhaseSpec(
                    "tiny", supervision_mode="skeleton",
                    validation=validation)])

            def forward(self, value):
                return value * self.weight

        trainer = UnifiedTrainer(
            TinyModel(), config={
                "optimization": {
                    "lr": 0.1,
                    "batch_size": 1,
                    "num_workers": 0,
                    "n_epochs": 10,
                    "scheduler_patience": 2,
                    "seed": 42,
                },
                "logging": {"checkpoint_interval": 0},
                "evaluation": {"eval_interval": 0},
            }, model_tag="tiny")
        trainer._create_loader = lambda *_: ([{}], object())
        trainer._compute_losses = lambda *_: {
            "total": trainer.model.weight.square().sum(),
        }
        trainer._write_model_card = lambda *_: None
        return trainer

    def test_declared_validation_fails_closed_without_val_or_adapter(self):
        trainer = self._tiny_trainer(ValidationSpec(
            selection_metric="validation.score"))
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "val 数据目录"):
                trainer.train(
                    {"sequence": root}, exp_dir=str(Path(root) / "run"))
            val = Path(root) / "val"
            val.mkdir()
            trainer = self._tiny_trainer(ValidationSpec(
                selection_metric="validation.score"))
            with self.assertRaisesRegex(ValueError, "evaluator adapter"):
                trainer.train(
                    {"sequence": root}, exp_dir=str(Path(root) / "run2"),
                    validation_data_dirs={"tiny": str(val)})

    def test_engine_selects_validation_checkpoint_stops_and_restores_best(self):
        validation = ValidationSpec(
            selection_metric="validation.score",
            early_stopping_patience_evaluations=2,
        )
        trainer = self._tiny_trainer(validation)
        scores = iter((3.0, 2.0, 2.1, 2.2))
        epochs = []

        def validator(**kwargs):
            epochs.append(kwargs["epoch"])
            self.assertFalse(kwargs["model"].training)
            return {"validation.score": next(scores)}

        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "run"
            val = Path(root) / "val"
            val.mkdir()
            trainer.train(
                {"sequence": root}, exp_dir=str(run),
                validation_data_dirs={"tiny": str(val)},
                validation_adapters={"tiny": validator},
            )

            phase = run / "phase_tiny"
            best = torch.load(
                phase / "model/best_eval_model.pt",
                map_location="cpu", weights_only=True)
            final = torch.load(
                phase / "model/final_model.pt",
                map_location="cpu", weights_only=True)
            self.assertEqual(epochs, [1, 2, 3, 4])
            self.assertTrue(torch.equal(
                trainer.model.state_dict()["weight"], best["weight"]))
            self.assertFalse(torch.equal(best["weight"], final["weight"]))
            records = [json.loads(line) for line in
                       (phase / "validation_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 4)
            resolved = json.loads((run / "config.json").read_text())
            selection = resolved["phases"][0]["validation_selection"]
            self.assertEqual(selection["best_epoch"], 2)
            self.assertTrue(selection["stopped_early"])
            self.assertTrue((phase / "checkpoints/model_epoch_0004.pt").is_file())

    def test_phase_without_validation_keeps_training_loss_behavior(self):
        trainer = self._tiny_trainer(None)
        with tempfile.TemporaryDirectory() as root:
            run = Path(root) / "run"
            trainer.train(
                {"sequence": root}, exp_dir=str(run),
                n_epochs_per_phase={"tiny": 2})
            model_dir = run / "phase_tiny/model"
            self.assertTrue((model_dir / "best_model.pt").is_file())
            self.assertTrue((model_dir / "final_model.pt").is_file())
            self.assertFalse((model_dir / "best_eval_model.pt").exists())
            resolved = json.loads((run / "config.json").read_text())
            self.assertNotIn(
                "validation_selection", resolved["phases"][0])


if __name__ == "__main__":
    unittest.main()
