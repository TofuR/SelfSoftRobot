import os
from pathlib import Path
import tempfile
import unittest

from src.registry.paths import ProjectPaths


class TestProjectPaths(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "repo"
        (self.root / "config").mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, text, name="paths.local.toml"):
        path = self.root / "config" / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_default_is_repo_workspace_and_cwd_independent(self):
        old_cwd = Path.cwd()
        try:
            os.chdir(self.temp.name)
            paths = ProjectPaths.load(repo_root=self.root, environ={})
        finally:
            os.chdir(old_cwd)
        self.assertEqual(paths.workspace_root, self.root / "workspace")
        self.assertEqual(
            paths.processed_dataset("real", "dataset_a"),
            self.root / "workspace/data/processed/real/dataset_a")

    def test_precedence_explicit_then_env_then_config(self):
        config = self.write_config(
            "schema_version = 1\n[paths]\nworkspace_root = 'from_config'\n")
        from_config = ProjectPaths.load(
            repo_root=self.root, config_path=config, environ={})
        self.assertEqual(from_config.workspace_root, self.root / "from_config")

        from_env = ProjectPaths.load(
            repo_root=self.root, config_path=config,
            environ={"SSR_WORKSPACE_ROOT": "from_env"})
        self.assertEqual(from_env.workspace_root, self.root / "from_env")

        explicit = ProjectPaths.load(
            repo_root=self.root, config_path=config,
            workspace_root="from_cli",
            environ={"SSR_WORKSPACE_ROOT": "from_env"})
        self.assertEqual(explicit.workspace_root, self.root / "from_cli")

    def test_external_workspace_and_artifact_uri_roundtrip(self):
        external = Path(self.temp.name) / "large_disk" / "ssr"
        paths = ProjectPaths.load(
            repo_root=self.root, workspace_root=external, environ={})
        dataset = paths.processed_dataset("real", "release_001")
        uri = paths.artifact_uri(dataset / "manifest.json")
        self.assertEqual(
            uri, "artifact://data/processed/real/release_001/manifest.json")
        self.assertEqual(paths.resolve_artifact_uri(uri), dataset / "manifest.json")

    def test_artifact_uri_rejects_escape_and_external_path(self):
        paths = ProjectPaths.load(repo_root=self.root, environ={})
        with self.assertRaises(ValueError):
            paths.resolve_artifact_uri("artifact://data/../outside")
        with self.assertRaises(ValueError):
            paths.artifact_uri(self.root / "data/legacy")
        with self.assertRaises(ValueError):
            paths.raw_sequence("real", "../escape")

    def test_legacy_roots_are_read_only_candidates(self):
        legacy = self.root / "old_raw" / "seq_a"
        legacy.mkdir(parents=True)
        config = self.write_config(
            "schema_version = 1\n"
            "[paths]\nworkspace_root = 'new_workspace'\n"
            "[compat]\nenable_legacy_reads = true\n"
            "raw_roots = ['old_raw']\n")
        paths = ProjectPaths.load(
            repo_root=self.root, config_path=config, environ={})
        self.assertEqual(
            paths.legacy_candidates("raw", "seq_a"), (legacy,))
        self.assertEqual(paths.raw_sequence("real", "seq_a"),
                         self.root / "new_workspace/data/raw/real/seq_a")

    def test_disabled_legacy_returns_no_candidates(self):
        config = self.write_config(
            "schema_version = 1\n"
            "[paths]\nworkspace_root = 'workspace'\n"
            "[compat]\nenable_legacy_reads = false\n"
            "raw_roots = ['old_raw']\n")
        paths = ProjectPaths.load(
            repo_root=self.root, config_path=config, environ={})
        self.assertEqual(
            paths.legacy_candidates("raw", "seq_a", existing_only=False), ())

    def test_layout_creation_and_no_overwrite(self):
        paths = ProjectPaths.load(repo_root=self.root, environ={})
        created = paths.create_workspace_layout()
        self.assertTrue(all(path.is_dir() for path in created))
        run = paths.training_run("study_a", "run_001")
        self.assertEqual(paths.create_new_directory(run), run)
        with self.assertRaises(FileExistsError):
            paths.create_new_directory(run)
        with self.assertRaises(ValueError):
            paths.create_new_directory(self.root / "outside")

    def test_schema_and_config_types_fail_closed(self):
        bad_schema = self.write_config(
            "schema_version = 2\n[paths]\nworkspace_root = 'workspace'\n",
            "bad_schema.toml")
        with self.assertRaises(ValueError):
            ProjectPaths.load(
                repo_root=self.root, config_path=bad_schema, environ={})

        bad_roots = self.write_config(
            "schema_version = 1\n[compat]\nraw_roots = 'not-a-list'\n",
            "bad_roots.toml")
        with self.assertRaises(ValueError):
            ProjectPaths.load(
                repo_root=self.root, config_path=bad_roots, environ={})


if __name__ == "__main__":
    unittest.main()
