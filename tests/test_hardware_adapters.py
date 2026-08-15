"""hardware/ 适配层的纯逻辑测试(不依赖真机硬件/串口)。

覆盖:_bootstrap sys.path 桥接、valve 单位收口、camera 指纹断言、ndi 隐藏评价流。
真机 import(ValveController/RealSenseCam/NdiThread)在真机才有,本测试不触碰。
"""

import sys
import unittest

from real_validation.hardware._bootstrap import ensure_real_capture_importable
from real_validation.hardware import camera, ndi, valve


class BootstrapTest(unittest.TestCase):
    def test_ensure_real_capture_is_importable(self):
        path = ensure_real_capture_importable()
        self.assertTrue(path.is_dir())
        self.assertEqual(path.name, "real_capture")
        self.assertIn(str(path), sys.path)


class ValveAdapterTest(unittest.TestCase):
    def test_missing_com_raises_without_touching_hardware(self):
        with self.assertRaises(valve.ValveHardwareError):
            valve.create_valve_controller("", "")

    def test_kpa_conversion_scales_by_action_scale(self):
        self.assertEqual(valve.valve_to_kpa_requested((0.5,), (150.0,)), (75.0,))
        self.assertEqual(valve.valve_to_kpa_requested((1.0,), (150.0,)), (150.0,))

    def test_kpa_conversion_requires_scale(self):
        with self.assertRaises(valve.ValveHardwareError):
            valve.valve_to_kpa_requested((0.5,), None)

    def test_kpa_conversion_dimension_mismatch(self):
        with self.assertRaises(valve.ValveHardwareError):
            valve.valve_to_kpa_requested((0.5, 0.5), (150.0,))


class CameraAdapterTest(unittest.TestCase):
    def test_fingerprint_mismatch_blocks(self):
        with self.assertRaises(camera.CameraHardwareError):
            camera.assert_camera_fingerprint(
                {"width": 640, "height": 480, "fps": 30},
                width=1280, height=480, fps=30, serial=None)

    def test_no_fingerprint_is_permissive(self):
        # 无 manifest 指纹 → 不硬阻断(部署契约缺失时由 preflight 兜底)
        camera.assert_camera_fingerprint(
            None, width=640, height=480, fps=30, serial=None)


class NdiAdapterTest(unittest.TestCase):
    def test_hidden_evaluation_source_marker(self):
        self.assertEqual(ndi.HIDDEN_EVALUATION_SOURCE, "ndi_hidden_eval")

    def test_ndi_requires_com_port(self):
        with self.assertRaises(ndi.NdiHardwareError):
            ndi.create_ndi_thread("")


if __name__ == "__main__":
    unittest.main()
