import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.training.train_transition import _default_gt_checkpoint_candidates
from src.registry.paths import ProjectPaths
from src.training.trainer_unified import UnifiedTrainer


class _NoPhaseStrategy:
    spec = type("Spec", (), {"phases": []})()

    @staticmethod
    def iterate_phases():
        return iter(())


class TestTrainingWorkspacePaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        (self.repo / "config").mkdir(parents=True)
        (self.repo / "config/paths.local.toml").write_text(
            "schema_version = 1\n"
            "[paths]\nworkspace_root = 'artifacts'\n"
            "[compat]\ntraining_roots = ['old_train_log']\n",
            encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_default_training_run_uses_workspace_and_not_cwd(self):
        model = torch.nn.Linear(1, 1)
        trainer = UnifiedTrainer.__new__(UnifiedTrainer)
        trainer.model = model
        trainer.model_tag = "gt_transition"
        trainer.phase = _NoPhaseStrategy()
        trainer.views = None
        trainer.config = {"optimization": {}}
        trainer._build_exp_config = lambda *_: {"model": "gt_transition"}
        trainer._write_model_card = lambda *_: None

        old_cwd = Path.cwd()
        try:
            os.chdir(self.temp.name)
            with patch(
                    "src.training.trainer_unified.ProjectPaths.load",
                    return_value=ProjectPaths.load(
                        repo_root=self.repo, environ={})):
                trainer.train({})
        finally:
            os.chdir(old_cwd)

        runs = list((self.repo / "artifacts/runs/training/gt_transition").iterdir())
        self.assertEqual(len(runs), 1)
        self.assertTrue(runs[0].name.startswith("exp_"))
        self.assertTrue((runs[0] / "config.json").is_file())
        self.assertFalse((Path(self.temp.name) / "train_log").exists())

    def test_gt_candidates_include_canonical_and_legacy_without_cwd(self):
        canonical = (self.repo /
                     "artifacts/runs/training/gt_transition/exp_new/"
                     "phase_gt_transition/model/best_model.pt")
        legacy = (self.repo /
                  "old_train_log/gt_transition/exp_old/"
                  "phase_gt_transition/model/best_model.pt")
        for checkpoint in (canonical, legacy):
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()

        paths = ProjectPaths.load(repo_root=self.repo, environ={})
        old_cwd = Path.cwd()
        try:
            os.chdir(self.temp.name)
            candidates = _default_gt_checkpoint_candidates(paths)
        finally:
            os.chdir(old_cwd)

        self.assertEqual(set(map(Path, candidates)), {canonical, legacy})


if __name__ == "__main__":
    unittest.main()
