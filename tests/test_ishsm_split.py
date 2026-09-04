import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.experiments.prepare_ishsm_split import create_ishsm_split


class ISHSMSplitTests(unittest.TestCase):
    @staticmethod
    def _write_sequence(path: Path, frames: int):
        np.savez_compressed(
            path,
            actions=np.arange(frames * 6, dtype=np.float32).reshape(frames, 6),
            positions=np.zeros((frames, 3, 15), dtype=np.float32),
            model_action_channels=np.array([0, 1, 3, 5]),
            node_order=np.array("base_to_tip"),
            state_length_unit=np.array("mm"),
        )

    def test_fit_dev_test_are_derived_without_overwriting_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            (source / "train").mkdir(parents=True)
            (source / "val").mkdir()
            self._write_sequence(source / "train" / "a.npz", 100)
            self._write_sequence(source / "train" / "b.npz", 50)
            self._write_sequence(source / "val" / "c.npz", 30)
            output = root / "derived"

            manifest = create_ishsm_split(
                source, output, fit_fraction=0.8, context_frames=10)

            self.assertEqual(manifest["roles"]["fit"]["frames"], 120)
            self.assertEqual(manifest["roles"]["dev"]["evaluation_frames"], 30)
            with np.load(output / "dev" / "a.npz") as dev:
                self.assertEqual(len(dev["actions"]), 30)
                self.assertEqual(dev["evaluation_mask"].sum(), 20)
            with self.assertRaises(FileExistsError):
                create_ishsm_split(source, output)


if __name__ == "__main__":
    unittest.main()
