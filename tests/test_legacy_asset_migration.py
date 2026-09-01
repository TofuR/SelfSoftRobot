import tempfile
from pathlib import Path
import subprocess
import unittest

from scripts.maintenance.migrate_legacy_assets import (
    apply_migration,
    build_ledger,
    remove_aliases,
    rollback_migration,
    verify_migration,
)
from src.registry.paths import ProjectPaths


class LegacyAssetMigrationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "--allow-empty", "-q", "-m", "init"],
            cwd=self.repo, check=True)
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})
        (self.repo / "real_capture/data/raw/seq_a/cam0").mkdir(parents=True)
        (self.repo / "real_capture/data/raw/seq_a/cam0/00000.png").write_bytes(b"raw")
        (self.repo / "data/real_seq/dataset_a/train").mkdir(parents=True)
        (self.repo / "data/real_seq/dataset_a/train/a.npz").write_bytes(b"data")
        (self.repo / "train_log/study_a/run_a").mkdir(parents=True)
        (self.repo / "train_log/study_a/run_a/config.json").write_text(
            '{"data":"data/real_seq/dataset_a/train"}', encoding="utf-8")
        (self.repo / "output/report_a").mkdir(parents=True)
        (self.repo / "output/report_a/result.txt").write_text("ok", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_apply_verify_remove_aliases(self):
        ledger = build_ledger(self.paths)
        apply_migration(self.paths, ledger)
        verify_migration(self.paths, ledger)

        raw_source = self.repo / "real_capture/data/raw/seq_a"
        raw_target = self.paths.raw_sequence("real", "seq_a")
        self.assertTrue(raw_source.is_symlink())
        self.assertEqual(raw_source.resolve(), raw_target.resolve())
        self.assertEqual((raw_target / "cam0/00000.png").read_bytes(), b"raw")

        remove_aliases(self.paths, ledger)
        self.assertFalse(raw_source.exists())
        verify_migration(self.paths, ledger, require_alias=False)

    def test_rollback_restores_original_tree(self):
        ledger = build_ledger(self.paths)
        apply_migration(self.paths, ledger)
        rollback_migration(self.paths, ledger)

        source = self.repo / "data/real_seq/dataset_a/train/a.npz"
        target = self.paths.processed_dataset("real", "dataset_a")
        self.assertEqual(source.read_bytes(), b"data")
        self.assertFalse(target.exists())

    def test_changed_source_is_rejected(self):
        ledger = build_ledger(self.paths)
        source = self.repo / "real_capture/data/raw/seq_a/cam0/00000.png"
        source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "已变化"):
            apply_migration(self.paths, ledger)


if __name__ == "__main__":
    unittest.main()
