from pathlib import Path
import tempfile
import unittest

from scripts.maintenance.check_legacy_path_literals import find_violations


class LegacyPathLiteralTest(unittest.TestCase):
    def test_repository_does_not_exceed_legacy_baseline(self):
        self.assertEqual(find_violations(), [])

    def test_new_business_default_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            script = repo / "scripts/new_analysis.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                "def main():\n"
                "    output_dir = 'output/new_analysis'\n"
                "    return output_dir\n",
                encoding="utf-8")

            violations = find_violations(repo, baseline={})

            self.assertEqual(len(violations), 1)
            self.assertIn("scripts/new_analysis.py", violations[0])


if __name__ == "__main__":
    unittest.main()
