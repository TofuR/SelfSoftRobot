import os
import tempfile
import unittest

import torch

from src.training.trainer_unified import UnifiedTrainer


class TrainingMonitorLossTests(unittest.TestCase):
    def test_monitor_values_never_enter_optimization_total(self):
        skeleton = torch.tensor(2.0, requires_grad=True)
        smooth = torch.tensor(0.5, requires_grad=True)
        z_monitor = torch.tensor(100.0, requires_grad=True)
        old_total = torch.tensor(999.0, requires_grad=True)

        total = UnifiedTrainer._sum_optimization_losses({
            "skeleton": skeleton,
            "spatial_smooth": smooth,
            "z_norm_monitor": z_monitor,
            "tf_ratio_monitor": torch.tensor(1.0),
            "total": old_total,
        })

        self.assertEqual(total.item(), 2.5)
        total.backward()
        self.assertEqual(skeleton.grad.item(), 1.0)
        self.assertEqual(smooth.grad.item(), 1.0)
        self.assertIsNone(z_monitor.grad)
        self.assertIsNone(old_total.grad)

    def test_periodic_checkpoint_contains_weights_and_training_state(self):
        trainer = object.__new__(UnifiedTrainer)
        trainer.model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        with tempfile.TemporaryDirectory() as root:
            current, archive, training_state = trainer._save_periodic_checkpoint(
                root, "gt_transition", 5, optimizer, scheduler, 0.125)
            self.assertTrue(os.path.isfile(current))
            self.assertTrue(os.path.isfile(archive))
            self.assertTrue(os.path.isfile(training_state))
            self.assertTrue(archive.endswith("model_epoch_0005.pt"))
            state = torch.load(
                training_state, map_location="cpu", weights_only=False)
            self.assertEqual(state["phase"], "gt_transition")
            self.assertEqual(state["epoch"], 5)
            self.assertAlmostEqual(state["best_loss"], 0.125)
            self.assertIn("model_state_dict", state)
            self.assertIn("optimizer_state_dict", state)
            self.assertIn("scheduler_state_dict", state)


if __name__ == "__main__":
    unittest.main()
