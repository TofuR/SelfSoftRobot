"""在线质量门控:每条判据一个测试。

在线没有未来帧 → 坏帧只能拒,不能像离线那样时间插值修复。
"""

import unittest

import numpy as np

from tests.test_perception_parity import synthetic_masks


def _skeleton(mask, n_points=15):
    from real_validation.perception.skeleton import extract_skeleton_2d
    return extract_skeleton_2d(mask, n_points, tip_fix=True, return_info=True)


def _thresholds(area_median_px=680.0, **overrides):
    from real_validation.perception.quality import QualityThresholds
    return QualityThresholds(area_median_px, **overrides)


class QualityTest(unittest.TestCase):
    def setUp(self):
        self.bent = dict(synthetic_masks())["bent_tube"]
        self.area = float(self.bent.sum())

    def _assess(self, mask, **kwargs):
        from real_validation.perception.quality import assess_frame
        skeleton, info = _skeleton(mask)
        thresholds = kwargs.pop("thresholds", _thresholds(self.area))
        return assess_frame(mask, skeleton, info, thresholds, **kwargs)

    def test_healthy_frame_is_ok(self):
        quality = self._assess(self.bent)
        self.assertEqual(quality.verdict, "ok", quality.reasons)
        self.assertEqual(quality.reasons, ())
        self.assertAlmostEqual(quality.flags["mask_area_ratio"], 1.0, places=6)

    def test_empty_mask_is_rejected(self):
        quality = self._assess(dict(synthetic_masks())["empty"])
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("empty_mask", quality.reasons)

    def test_area_too_small_is_rejected(self):
        quality = self._assess(self.bent, thresholds=_thresholds(self.area * 2.0))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("area_ratio_low", quality.reasons)

    def test_area_too_large_is_rejected(self):
        quality = self._assess(self.bent, thresholds=_thresholds(self.area * 0.5))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("area_ratio_high", quality.reasons)

    def test_arm_not_reaching_base_is_rejected(self):
        truncated = self.bent.copy()
        truncated[:40, :] = 0                       # 顶部 40 行清空 → top_row 变大
        quality = self._assess(truncated, thresholds=_thresholds(float(truncated.sum())))
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("top_row_high", quality.reasons)

    def test_second_blob_is_degraded_not_rejected(self):
        with_blob = self.bent.copy()
        blob_area = int(0.25 * self.bent.sum())
        side = max(2, int(np.sqrt(blob_area)))
        with_blob[5:5 + side, 65:65 + side] = 1     # 手/异物
        quality = self._assess(with_blob, thresholds=_thresholds(float(with_blob.sum())))
        self.assertEqual(quality.verdict, "degraded")
        self.assertIn("second_blob_present", quality.reasons)

    def test_silent_tip_fix_skip_is_degraded(self):
        thin = np.zeros((40, 20), np.uint8)
        thin[10:13, 8:11] = 1                       # 前景 < 10px → tip_fix 静默跳过
        quality = self._assess(thin, thresholds=_thresholds(float(thin.sum()),
                                                           min_height_frac=0.0,
                                                           max_top_row=40))
        self.assertIn("tip_fix_skipped", quality.reasons)
        self.assertEqual(quality.flags["tip_fix_reason"], "foreground_lt_10")

    def test_node_jump_is_rejected(self):
        skeleton, _ = _skeleton(self.bent)
        moved = skeleton + 50.0
        quality = self._assess(self.bent, prev_skeleton=moved)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("node_step_high", quality.reasons)
        self.assertGreater(quality.flags["max_node_step_px"], 4.0)

    def test_stale_frame_is_rejected(self):
        quality = self._assess(self.bent, frame_age_s=1.2)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("frame_stale", quality.reasons)

    def test_registration_displacement_is_rejected(self):
        quality = self._assess(self.bent, registration_displacement_px=7.5)
        self.assertEqual(quality.verdict, "reject")
        self.assertIn("registration_displaced", quality.reasons)

    def test_flags_are_json_safe(self):
        import json
        quality = self._assess(self.bent, frame_age_s=0.03,
                               registration_displacement_px=0.4)
        # io.py 的 atomic_write_json 是 allow_nan=False，且 json 不认 numpy 标量
        payload = json.dumps(quality.flags, allow_nan=False)
        self.assertIn("mask_area_ratio", payload)
        for value in quality.flags.values():
            self.assertIsInstance(value, (bool, int, float, str, type(None)))
            self.assertNotIsInstance(value, np.generic)

    def test_area_median_has_no_default(self):
        from real_validation.perception.quality import QualityThresholds
        with self.assertRaises(TypeError):
            QualityThresholds()          # 数据相关阈值必须显式提供


if __name__ == "__main__":
    unittest.main()
