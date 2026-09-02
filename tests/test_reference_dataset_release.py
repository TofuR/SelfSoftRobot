import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.real.publish_reference_dataset import publish_reference_release
from src.registry.datasets import DatasetSelector
from src.registry.manifests import validate_dataset_manifest
from src.registry.paths import ProjectPaths


class ReferenceDatasetReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})
        for sequence in ("seq_train", "seq_test"):
            raw = self.paths.raw_sequence("real", sequence)
            (raw / "cam0").mkdir(parents=True)
            for frame in range(4):
                (raw / "cam0" / f"{frame:05d}.png").write_bytes(b"png")
            (raw / "frame_times.txt").write_text(
                "0.0\n0.1\n0.2\n0.3\n", encoding="utf-8")
            (raw / "meta.json").write_text(json.dumps({
                "frames": 4,
                "start_iso": "2026-09-01T10:00:00",
                "stop_iso": "2026-09-01T10:00:01",
                "action_interval_s": 0.1,
                "camera_count": 1,
                "ndi_count": 0,
            }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _npz(self, name: str) -> Path:
        path = self.repo / "source" / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            positions=np.zeros((4, 3, 15), np.float32),
            actions=np.zeros((4, 6), np.float32),
            state_coordinate_frame=np.array("robot_planar_mm_v1"),
            state_length_unit=np.array("mm"),
            node_order=np.array("base_to_tip"),
            n_points=np.array(15),
            raw_action_dim=np.array(6),
            model_action_dim=np.array(4),
            model_action_channels=np.asarray((0, 1, 3, 5)),
            channel_source6=np.asarray((0, 1, 1, 3, 3, 5)),
            action_expansion6=np.asarray((0, 1, 1, 2, 2, 3)),
        )
        return path

    @patch("scripts.real.publish_reference_dataset._git_commit",
           return_value="3157b12")
    def test_publishes_non_overwriting_registered_splits(self, _commit):
        sources = {role: self._npz(role) for role in ("train", "val", "test")}
        manifest_path = publish_reference_release(
            paths=self.paths,
            dataset_id="reference_v1",
            train_npz=sources["train"],
            val_npz=sources["val"],
            test_npz=sources["test"],
            train_sequence="seq_train",
            test_sequence="seq_test",
            command="publish reference_v1",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_dataset_manifest(manifest)
        selector = DatasetSelector(self.paths)
        for role in ("train", "val", "test"):
            artifact = selector.resolve("reference_v1", role)
            self.assertEqual(artifact.path.name, f"{role}.npz")
        self.assertEqual(
            manifest["contracts"]["evaluation"]["test_policy"],
            "frozen_final_only")
        self.assertEqual(manifest["sources"][0]["raw_manifest_uri"],
                         "artifact://data/raw/real/seq_train/raw_manifest.json")
        serialized = json.dumps(manifest)
        self.assertNotIn(str(self.repo), serialized)
        self.assertEqual(
            manifest["recipe"]["parameters"]["source_roles"]["train"],
            "repo://source/train.npz")

        with self.assertRaises(FileExistsError):
            publish_reference_release(
                paths=self.paths,
                dataset_id="reference_v1",
                train_npz=sources["train"],
                val_npz=sources["val"],
                test_npz=sources["test"],
                train_sequence="seq_train",
                test_sequence="seq_test",
                command="publish reference_v1",
            )


if __name__ == "__main__":
    unittest.main()
