import unittest

import numpy as np

from scripts.real.benchmark_online_segmentation import mask_metrics, skeleton_metrics


class OnlineSegmentationBenchmarkTest(unittest.TestCase):
    def test_mask_metrics_have_expected_overlap(self):
        reference = np.zeros((4, 4), np.uint8)
        predicted = np.zeros((4, 4), np.uint8)
        reference[1:3, 1:3] = 1
        predicted[1:3, 2:4] = 1
        result = mask_metrics(predicted, reference)
        self.assertAlmostEqual(result["iou"], 2 / 6)
        self.assertAlmostEqual(result["dice"], 4 / 8)
        self.assertAlmostEqual(result["area_ratio"], 1.0)

    def test_skeleton_metrics_convert_fixed_scale(self):
        reference = np.zeros((3, 2), np.float32)
        predicted = np.asarray(((3, 4), (0, 5), (0, 0)), np.float32)
        result = skeleton_metrics(predicted, reference, mm_per_px=0.5)
        self.assertAlmostEqual(result["node_mean_px"], 10 / 3)
        self.assertAlmostEqual(result["tip_px"], 0.0)
        self.assertAlmostEqual(result["base_px"], 5.0)
        self.assertAlmostEqual(result["node_mean_mm"], 5 / 3)
        self.assertAlmostEqual(result["tip_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
