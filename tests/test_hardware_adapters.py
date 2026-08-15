"""hardware/ 适配层的纯逻辑测试(不依赖真机硬件/串口)。

覆盖:valve 自包含与单位收口、camera 指纹断言、ndi 隐藏评价流。
真机 import(ValveController/RealSenseCam/NdiThread)在真机才有,本测试不触碰。
"""

import sys
import unittest

from real_validation.hardware import camera, ndi, valve


class ValveSelfContainedTest(unittest.TestCase):
    def test_valve_module_does_not_import_real_capture(self):
        # 移植后硬件模块必须完全自包含(不触碰 real_capture)
        self.assertNotIn("real_capture", sys.modules)
        self.assertNotIn("modbus_manager", sys.modules)   # 顶层模块名也不该出现


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
