"""位姿注册的离线验收。

三段测法缺一不可：
  (a) 同一帧 → 位移 ≈ 0            —— 证明不误报
  (b) 人工平移 3 px → 位移 ≈ 3      —— **唯一能证明数值有意义而不是恒返回 0 的测试**
  (c) 纯色图特征不足 → 必须显式失败 —— 否则"配准通过"变成默认值
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def textured_frame(seed: int = 11) -> np.ndarray:
    """带足量角点的合成灰度图（ORB 需要真实纹理，随机噪声不够稳）。"""
    import cv2
    rng = np.random.default_rng(seed)
    image = np.full((240, 320), 40, np.uint8)
    for _ in range(120):
        x, y = int(rng.integers(10, 300)), int(rng.integers(10, 220))
        size = int(rng.integers(4, 14))
        shade = int(rng.integers(120, 250))
        cv2.rectangle(image, (x, y), (x + size, y + size), shade, -1)
    return cv2.GaussianBlur(image, (3, 3), 0)


class RegistrationTest(unittest.TestCase):
    def test_identical_frame_reports_zero_displacement(self):
        from real_validation.perception.registration import estimate_registration
        frame = textured_frame()
        result = estimate_registration(frame, frame.copy())
        self.assertTrue(result.ok, result.reason)
        self.assertLess(result.displacement_px, 0.5)
        self.assertLess(result.fit_residual_px, 0.5)

    def test_known_translation_is_recovered(self):
        import cv2
        from real_validation.perception.registration import estimate_registration
        frame = textured_frame()
        shift = np.float32([[1, 0, 3], [0, 1, 0]])
        moved = cv2.warpAffine(frame, shift, (frame.shape[1], frame.shape[0]))
        result = estimate_registration(frame, moved, max_displacement_px=2.0)
        self.assertLess(abs(result.displacement_px - 3.0), 0.3,
                        f"位移={result.displacement_px}")
        self.assertFalse(result.ok)          # 3 px > 阈值 2 px → 必须阻断
        self.assertEqual(result.reason, "displaced")

    def test_featureless_frame_fails_loudly(self):
        from real_validation.perception.registration import estimate_registration
        blank = np.zeros((240, 320), np.uint8)
        result = estimate_registration(blank, blank.copy())
        self.assertFalse(result.ok)
        self.assertIn(result.reason, {"too_few_features", "too_few_matches"})
        # 关键：失败时绝不能报 0 位移，否则"配准通过"成为默认
        self.assertTrue(np.isnan(result.displacement_px))

    def test_round_trip_json(self):
        from real_validation.perception.registration import (
            estimate_registration, load_registration, save_registration)
        frame = textured_frame()
        result = estimate_registration(frame, frame.copy(), reference_sha256="deadbeef")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registration.json"
            save_registration(result, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["reference_sha256"], "deadbeef")
            restored = load_registration(path)
            self.assertEqual(restored.reference_sha256, "deadbeef")
            self.assertAlmostEqual(restored.displacement_px, result.displacement_px, places=9)


class BackgroundTest(unittest.TestCase):
    def test_median_background_ignores_moving_object(self):
        from real_validation.perception.background import build_median_background_from_frames
        base = textured_frame(seed=3)
        frames = []
        for index in range(9):
            frame = base.copy()
            frame[:, index * 30:index * 30 + 20] = 255      # 移动的亮块
            frames.append(frame)
        median = build_median_background_from_frames(np.stack(frames))
        # 每列被遮挡的时间 < 50% → 中值应回到 base
        self.assertLess(float(np.abs(median.astype(np.int16) -
                                     base.astype(np.int16)).mean()), 3.0)

    def test_drift_detects_shifted_background(self):
        import cv2
        from real_validation.perception.background import background_drift
        # 高密度随机纹理:平移后绝大多数像素变化 → 中位数 absdiff 显著非零。
        # (textured_frame 的稀疏纹理会被大面积常量暗底主导,中位数归 0,测不出位移)
        rng = np.random.default_rng(5)
        base = rng.integers(0, 256, (160, 240), dtype=np.uint8)
        moved = cv2.warpAffine(base, np.float32([[1, 0, 8], [0, 1, 0]]),
                               (base.shape[1], base.shape[0]))
        self.assertLess(background_drift(base, base.copy()), 1.0)
        self.assertGreater(background_drift(base, moved), 5.0)


REAL_BG = REPO / "real_capture/data/derived/seq_20260627_163921/bg_median.png"
REAL_CAM0 = REPO / "real_capture/data/raw/seq_20260627_163921/cam0"


@unittest.skipUnless(REAL_BG.is_file() and REAL_CAM0.is_dir(),
                     "真实采集数据不存在（已 gitignore；只能在服务器/本机验收）")
class RegistrationOnRealFramesTest(unittest.TestCase):
    def test_consecutive_real_frames_are_registered(self):
        import cv2
        from real_validation.perception.registration import estimate_registration
        frames = sorted(REAL_CAM0.glob("*.png"))[:2]
        first = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
        second = cv2.imread(str(frames[1]), cv2.IMREAD_GRAYSCALE)
        result = estimate_registration(first, second)
        self.assertTrue(result.ok, f"{result.reason} disp={result.displacement_px}")
        self.assertLess(result.displacement_px, 1.0)


if __name__ == "__main__":
    unittest.main()
