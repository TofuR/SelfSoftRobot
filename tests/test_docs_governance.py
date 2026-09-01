import tempfile
import unittest
from pathlib import Path

from scripts.maintenance.check_docs_governance import (
    PROJECT_ROOT,
    audit_repository,
    check_front_matter,
    check_relative_links,
)


class DocsGovernanceTests(unittest.TestCase):
    def test_current_governed_documents_pass(self):
        self.assertEqual(audit_repository(PROJECT_ROOT), [])

    def test_missing_front_matter_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("# no metadata\n", encoding="utf-8")
            self.assertIn("missing opening", check_front_matter(path)[0])

    def test_broken_relative_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("[missing](not-here.md)\n", encoding="utf-8")
            self.assertIn("broken relative link", check_relative_links(path)[0])


if __name__ == "__main__":
    unittest.main()
