import unittest

import torch

from src.training.selection import SelectionState
from src.training.spec import PhaseSpec, ValidationSpec
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


if __name__ == "__main__":
    unittest.main()
