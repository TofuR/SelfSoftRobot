from datetime import datetime
from pathlib import Path
import tempfile
import unittest

from src.registry.paths import ProjectPaths
from src.registry.runs import allocate_numbered_run, create_analysis_run


class RunRegistryTest(unittest.TestCase):
    def test_analysis_run_is_workspace_owned_and_incrementing(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            paths = ProjectPaths.load(repo_root=repo, environ={})
            now = datetime(2026, 9, 1, 12, 0, 0)

            first = create_analysis_run(paths, "horizon", now=now)
            second = create_analysis_run(paths, "horizon", now=now)

            self.assertEqual(
                first, repo / "workspace/runs/analysis/horizon/run_20260901_000")
            self.assertEqual(
                second, repo / "workspace/runs/analysis/horizon/run_20260901_001")

    def test_allocator_rejects_unsafe_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "前缀"):
                allocate_numbered_run(Path(root), prefix="../escape")


if __name__ == "__main__":
    unittest.main()
