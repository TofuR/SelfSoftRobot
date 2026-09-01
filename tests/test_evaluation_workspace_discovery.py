from pathlib import Path
import tempfile
import unittest

from scripts.evaluation.evaluate_shape import (
    scan_checkpoints as scan_shape_checkpoints,
    scan_data_dirs as scan_shape_data_dirs,
)
from scripts.evaluation.inspect_real_data import auto_find_npz
from scripts.evaluation.visualize_3d_shape import (
    parse_checkpoint_path,
    scan_checkpoints as scan_visualization_checkpoints,
)
from src.registry.paths import ProjectPaths


class EvaluationWorkspaceDiscoveryTest(unittest.TestCase):
    def test_canonical_training_and_processed_assets_are_discovered(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            paths = ProjectPaths.load(repo_root=repo, environ={})
            checkpoint = paths.training_run(
                "gt_transition", "run_001") / "phase_gt/model/best_model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            data_dir = paths.processed_dataset("real", "dataset_a") / "train"
            data_dir.mkdir(parents=True)
            npz = data_dir / "dataset_a_train.npz"
            npz.write_bytes(b"fixture")

            self.assertIn(str(checkpoint), scan_shape_checkpoints(paths))
            self.assertIn(str(checkpoint), scan_visualization_checkpoints(paths))
            self.assertIn(str(data_dir), scan_shape_data_dirs(paths))
            self.assertEqual(auto_find_npz(paths), str(npz))

    def test_canonical_checkpoint_path_is_parsed(self):
        value = parse_checkpoint_path(
            "/repo/workspace/runs/training/open_loop_transition/"
            "run_001/phase_open_loop_transition/model/best_model.pt")
        self.assertEqual(
            value,
            ("open_loop_transition", "run_001", "open_loop_transition"),
        )


if __name__ == "__main__":
    unittest.main()
