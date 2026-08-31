from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from src.registry.inventory import LegacyInventoryBuilder, write_legacy_inventory
from src.registry.paths import ProjectPaths


class TestLegacyInventory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()

        raw = self.repo / "real_capture/data/raw/seq_b"
        (raw / "cam0").mkdir(parents=True)
        (raw / "meta.json").write_text("{}", encoding="utf-8")
        (raw / "actions6.csv").write_text("", encoding="utf-8")

        derived = self.repo / "real_capture/data/derived/seq_b"
        derived.mkdir(parents=True)
        (derived / "preprocess_manifest.json").write_text("{}", encoding="utf-8")

        dataset = self.repo / "data/real_seq/dataset_b"
        dataset.mkdir(parents=True)
        (dataset / "dataset_manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "source": {"sequence": "seq_b"},
            "splits": {
                "train": [{"frames": 80}],
                "val": [{"frames": 20}],
            },
            "quality_control": {"training_ready": True},
        }), encoding="utf-8")
        (self.repo / "data/real_seq/dataset_without_manifest").mkdir()

        run = self.repo / "train_log/model_b/exp_001"
        (run / "evaluations").mkdir(parents=True)
        (run / "config.json").write_text("{}", encoding="utf-8")
        (run / "RUN_COMPLETE").touch()

        validation = self.repo / "real_validation/runs/run_001"
        validation.mkdir(parents=True)
        (validation / "setup.json").write_text("{}", encoding="utf-8")

        analysis = self.repo / "output/topic_a"
        analysis.mkdir(parents=True)
        (analysis / "summary.json").write_text("{}", encoding="utf-8")

        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})
        fixed = datetime(2026, 8, 31, tzinfo=timezone.utc)
        self.inventory = LegacyInventoryBuilder(
            self.paths, now=lambda: fixed).build()

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_maps_legacy_assets_without_absolute_paths(self):
        self.assertEqual(
            self.inventory["raw_sequences"][0]["sequence_id"], "seq_b")
        self.assertEqual(
            self.inventory["raw_sequences"][0]["uri"], "legacy://raw/0/seq_b")
        self.assertEqual(
            next(item for item in self.inventory["processed_datasets"]
                 if item["dataset_id"] == "dataset_b")["source_sequence_ids"],
            ["seq_b"])
        self.assertEqual(
            next(item for item in self.inventory["processed_datasets"]
                 if item["dataset_id"] == "dataset_b")["splits"]["train"],
            {"files": 1, "frames": 80})
        missing = next(
            item for item in self.inventory["processed_datasets"]
            if item["dataset_id"] == "dataset_without_manifest")
        self.assertEqual(missing["manifest_error"], "missing")
        self.assertEqual(
            self.inventory["training_runs"][0]["operational_status"],
            "complete")
        serialized = json.dumps(self.inventory)
        self.assertNotIn(str(self.repo), serialized)
        self.assertIn("repo://real_capture/data/raw", serialized)

    def test_snapshot_writes_only_inside_workspace_and_is_refreshable(self):
        target = write_legacy_inventory(self.paths, self.inventory)
        self.assertEqual(target, self.paths.registry_root / "legacy_inventory.json")
        self.assertEqual(json.loads(target.read_text()), self.inventory)
        refreshed = dict(self.inventory, generated_at="later")
        self.assertEqual(
            write_legacy_inventory(self.paths, refreshed), target)
        self.assertEqual(json.loads(target.read_text())["generated_at"], "later")
        with self.assertRaises(ValueError):
            write_legacy_inventory(
                self.paths, self.inventory, target=self.repo / "inventory.json")


if __name__ == "__main__":
    unittest.main()
