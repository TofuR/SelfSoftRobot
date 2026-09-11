import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from real_validation.paths import default_results_root, resolve_results_root
from real_validation.tools.package_hereditary import build_package


class PortablePackageTests(unittest.TestCase):
    def test_portable_results_are_relative_to_installation_not_cwd(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {}, clear=True):
            app = Path(d)/'real_validation'; app.mkdir()
            (app/'PACKAGE_MANIFEST.json').write_text('{}')
            self.assertEqual(default_results_root(app), Path(d)/'results')
            self.assertEqual(resolve_results_root('recordings', app), Path(d)/'recordings')
            with patch.dict(os.environ, {'REAL_VALIDATION_RESULTS':'custom'}):
                self.assertEqual(default_results_root(app), Path(d)/'custom')
            with self.assertRaises(ValueError): resolve_results_root('', app)

    def test_source_checkout_uses_configured_workspace(self):
        app = Path(__file__).resolve().parents[1]/'real_validation'
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'SSR_WORKSPACE_ROOT':d}):
            with patch.dict(os.environ):
                os.environ.pop('REAL_VALIDATION_RESULTS', None)
                self.assertEqual(default_results_root(app), Path(d)/'runs/validation')

    def test_runtime_package_contract_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); weights = root/'control.npz'; weights.write_bytes(b'control')
            meta = dict(dt=.1, weights_sha256=hashlib.sha256(b'control').hexdigest(),
                        source_checkpoint='/server/private/training.pt')
            candidates = root/'candidates'; candidates.mkdir()
            (candidates/'duplicate.npz').write_bytes(b'control')
            with patch('real_validation.tools.package_hereditary.load_bundle', return_value=(None,meta.copy())):
                archive = build_package(weights, root/'app.zip', candidates_dir=candidates)
                with self.assertRaises(FileExistsError): build_package(weights, archive)
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                self.assertEqual({p.split('/')[0] for p in names}, {'README.md','run.bat','install.bat','real_validation'})
                self.assertEqual([p for p in names if p.endswith('.md')], ['README.md'])
                self.assertFalse(any('/tools/' in p or '/runs/' in p or p.endswith('.onnx') for p in names))
                self.assertIn('real_validation/paths.py', names)
                self.assertIn('real_validation/gui/main_window.py', names)
                m = json.loads(z.read('real_validation/PACKAGE_MANIFEST.json'))
                self.assertEqual(len(m['models']), 1)
                for name,digest in m['files'].items():
                    self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), digest)
                self.assertNotIn(b'/server/private', z.read(m['models'][0]['path'].replace('.npz','.json')))
                req = z.read('real_validation/requirements.txt').decode()
                self.assertNotIn('-r ', req)
                self.assertIn('pyserial', req)


if __name__ == '__main__': unittest.main()
