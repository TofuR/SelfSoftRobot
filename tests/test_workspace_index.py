from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from src.registry.paths import ProjectPaths
from src.registry.workspace_index import (
    WorkspaceIndexBuilder,
    write_missing_legacy_dataset_manifests,
    write_mainline_legacy_manifests,
    write_workspace_index,
)


class WorkspaceIndexTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})

        dataset = self.paths.processed_dataset("real", "dataset_a")
        (dataset / "train").mkdir(parents=True)
        (dataset / "val").mkdir()
        (dataset / "train/a.npz").write_bytes(b"train fixture")
        (dataset / "val/a.npz").write_bytes(b"val fixture")
        (dataset / "dataset_manifest.json").write_text(json.dumps({
            "schema_version": 2,
            "dataset_id": "dataset_a",
            "source": {"sequence": "seq_a"},
        }), encoding="utf-8")
        missing = self.paths.processed_dataset("real", "dataset_without_manifest")
        (missing / "train").mkdir(parents=True)
        (missing / "train/seq_20260819_182253_train.npz").write_bytes(
            b"legacy fixture")

        self.run = (self.paths.runs_root / "training/real_pipeline/seq_a/"
                    "trial_20260901_000")
        self.run.mkdir(parents=True)
        (self.run / "config.json").write_text(json.dumps({
            "trial": {
                "id": "trial_20260901_000",
                "created_at": "2026-09-01T12:00:00+08:00",
            },
            "data": {
                "train_dir": "data/real_seq/dataset_a/train",
                "dataset_manifest": "data/real_seq/dataset_a/dataset_manifest.json",
            },
            "training": {"stages": {"gt": {"epochs": 2}}},
        }), encoding="utf-8")
        (self.run / "commands.sh").write_text("python train.py\n", encoding="utf-8")
        (self.run / "status.txt").write_text("complete\n", encoding="utf-8")
        checkpoint = self.run / "stages/gt/model/best_eval_model.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        (self.run / "artifacts.json").write_text(json.dumps({
            "stages": {"gt": {
                "best_checkpoint": "stages/gt/model/best_eval_model.pt",
            }},
        }), encoding="utf-8")

        fixed = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.index = WorkspaceIndexBuilder(
            self.paths, now=lambda: fixed).build()

    def tearDown(self):
        self.temp.cleanup()

    def test_builds_lightweight_dataset_to_run_index(self):
        self.assertFalse(self.index["policy"]["hash_payload_files"])
        dataset = next(item for item in self.index["datasets"]
                       if item["dataset_id"] == "dataset_a")
        self.assertEqual(dataset["manifest_contract"], "historical")
        self.assertEqual(dataset["source_sequence_ids"], ["seq_a"])
        self.assertEqual(dataset["split_files"], {
            "train": 1, "val": 1, "test": 0})
        missing = next(item for item in self.index["datasets"]
                       if item["dataset_id"] == "dataset_without_manifest")
        self.assertEqual(missing["manifest_contract"], "missing")
        run_uri = self.paths.artifact_uri(self.run)
        self.assertEqual(
            self.index["reverse_index"]["dataset_to_runs"]["dataset_a"],
            [run_uri])

    def test_writes_refreshable_index_and_non_overwriting_mainline_manifest(self):
        target = write_workspace_index(self.paths, self.index)
        self.assertEqual(json.loads(target.read_text()), self.index)
        written = write_mainline_legacy_manifests(self.paths, self.index)
        self.assertEqual(written, [self.run / "legacy_run_manifest.json"])
        manifest = json.loads(written[0].read_text())
        self.assertFalse(manifest["provenance"]["hash_payload_files"])
        self.assertEqual(manifest["datasets"][0]["dataset_id"], "dataset_a")
        self.assertEqual(manifest["stages"][0]["selection_basis"], "validation")
        self.assertEqual(
            write_mainline_legacy_manifests(self.paths, self.index), [])

    def test_backfills_observational_dataset_manifest_without_payload_hash(self):
        written = write_missing_legacy_dataset_manifests(self.paths, self.index)
        self.assertEqual(len(written), 1)
        manifest = json.loads(written[0].read_text())
        self.assertEqual(manifest["kind"], "legacy_processed_dataset")
        self.assertEqual(
            manifest["source_sequence_ids"], ["seq_20260819_182253"])
        self.assertFalse(manifest["provenance"]["hash_payload_files"])
        self.assertFalse(manifest["provenance"]["open_payload_files"])
        self.assertNotIn("sha256", json.dumps(manifest))

        refreshed = WorkspaceIndexBuilder(self.paths).build()
        dataset = next(item for item in refreshed["datasets"]
                       if item["dataset_id"] == "dataset_without_manifest")
        self.assertEqual(dataset["manifest_contract"], "historical")
        self.assertTrue(dataset["manifest_uri"].endswith(
            "/legacy_dataset_manifest.json"))
        self.assertEqual(
            write_missing_legacy_dataset_manifests(self.paths, refreshed), [])


if __name__ == "__main__":
    unittest.main()
