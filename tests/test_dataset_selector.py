import json
from pathlib import Path
import tempfile
import unittest

from src.registry.datasets import DatasetSelector
from src.registry.manifests import sha256_file
from src.registry.paths import ProjectPaths


class DatasetSelectorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})
        self.selector = DatasetSelector(self.paths)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _write_npz(path: Path, value: bytes = b"npz") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    def test_canonical_manifest_is_preferred_and_hash_checked(self):
        dataset = self.paths.processed_dataset("real", "dataset_a")
        data = self._write_npz(dataset / "splits" / "train" / "a.npz")
        manifest = {
            "splits": {"train": [{
                "uri": self.paths.artifact_uri(data),
                "sha256": sha256_file(data),
            }]},
        }
        (dataset / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
        legacy = self._write_npz(
            self.repo / "data" / "real_seq" / "dataset_a" / "train" / "a.npz",
            b"legacy")

        selected = self.selector.resolve("dataset_a", "train")

        self.assertEqual(selected.path, data.resolve())
        self.assertEqual(selected.source, "canonical")
        self.assertNotEqual(selected.path, legacy.resolve())

    def test_legacy_absolute_manifest_survives_path_migration(self):
        dataset = self.repo / "data" / "real_seq" / "dataset_a"
        data = self._write_npz(dataset / "val" / "a.npz")
        manifest = {"splits": {"val": [{
            "path": "/old/host/data/real_seq/dataset_a/val/a.npz",
        }]}}
        (dataset / "dataset_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        selected = self.selector.resolve("dataset_a", "val")

        self.assertEqual(selected.path, data.resolve())
        self.assertEqual(selected.source, "legacy")

    def test_filename_selection_and_missing_role_fail_closed(self):
        dataset = self.repo / "data" / "real_seq" / "dataset_a"
        first = self._write_npz(dataset / "train" / "a.npz")
        second = self._write_npz(dataset / "train" / "b.npz")

        self.assertEqual(
            self.selector.resolve(
                "dataset_a", "train", filename="b.npz").path,
            second.resolve())
        self.assertEqual(
            self.selector.resolve("dataset_a", "train").path,
            first.resolve())
        with self.assertRaisesRegex(FileNotFoundError, "dataset artifact"):
            self.selector.resolve("dataset_a", "test")

    def test_hash_mismatch_fails_closed(self):
        dataset = self.paths.processed_dataset("real", "dataset_a")
        data = self._write_npz(dataset / "splits" / "train" / "a.npz")
        manifest = {"splits": {"train": [{
            "uri": self.paths.artifact_uri(data),
            "sha256": "0" * 64,
        }]}}
        (dataset / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "hash 不匹配"):
            self.selector.resolve("dataset_a", "train")


if __name__ == "__main__":
    unittest.main()
