import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.real.masks_to_transition_npz import masks_to_positions, save_npz
from src.evaluation.diameter_scale import (
    estimate_diameter_px,
    resolve_diameter_scale,
)


class DiameterScaleTest(unittest.TestCase):
    def test_body_width_is_measured_from_centerline_interior(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask_dir = Path(temporary) / "masks"
            mask_dir.mkdir()
            mask = np.zeros((180, 180), np.uint8)
            cv2.rectangle(mask, (80, 20), (100, 160), 255, -1)
            cv2.imwrite(str(mask_dir / "00000.png"), mask)

            _, _, qc = masks_to_positions(
                str(mask_dir), n_points=15, skeleton_method="skeletonize",
                endpoint_fix=True, return_qc=True)

            self.assertAlmostEqual(qc[0]["body_width_px"], 22.0, delta=2.0)

    def test_qc_scale_uses_16_mm_and_robust_median(self):
        rows = [
            {"body_width_px": "20", "hard_invalid": "False"},
            {"body_width_px": "22", "hard_invalid": "False"},
            {"body_width_px": "200", "hard_invalid": "True"},
        ]
        diameter_px, source = estimate_diameter_px(rows)
        self.assertEqual(source, "body_width_px")
        self.assertEqual(diameter_px, 21.0)

    def test_resolve_scale_falls_back_to_sequence_qc(self):
        with tempfile.TemporaryDirectory() as temporary:
            sequence = Path(temporary) / "seq"
            data_dir = sequence / "val"
            qc_dir = sequence / "qc_skeleton"
            data_dir.mkdir(parents=True)
            qc_dir.mkdir()
            with (qc_dir / "skeleton_metrics.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["tip_width_px", "hard_invalid"])
                writer.writeheader()
                writer.writerow({"tip_width_px": 20, "hard_invalid": False})
                writer.writerow({"tip_width_px": 22, "hard_invalid": False})

            scale = resolve_diameter_scale({}, str(data_dir), diameter_mm=16)

            self.assertAlmostEqual(scale.diameter_px, 21.0)
            self.assertAlmostEqual(scale.mm_per_px, 16 / 21)
            self.assertEqual(scale.source, "qc_skeleton:tip_width_px")

    def test_npz_records_scale_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data" / "sample.npz"
            save_npz(
                str(path), np.zeros((2, 3, 15), np.float32),
                np.zeros((2, 6), np.float32),
                robot_diameter_mm=16, robot_diameter_px=20)
            with np.load(path) as saved:
                self.assertEqual(float(saved["robot_diameter_mm"]), 16.0)
                self.assertEqual(float(saved["robot_diameter_px"]), 20.0)
                self.assertAlmostEqual(float(saved["mm_per_px"]), 0.8)


if __name__ == "__main__":
    unittest.main()
