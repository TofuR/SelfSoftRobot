"""HardwareProfile/HardwareSession 的模式与生命周期契约。"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from real_validation.hardware.profile import (
    BackendMode, DeviceState, HardwareProfile, required_groups_for_channels,
)
from real_validation.hardware.session import HardwareSession, HardwareSessionError


_app = None


def app():
    global _app
    _app = QApplication.instance() or QApplication([])
    return _app


class HardwareProfileTest(unittest.TestCase):
    def test_presets_are_explicit(self):
        mock = HardwareProfile.all_mock()
        self.assertEqual(mock.camera_backend, BackendMode.MOCK)
        self.assertEqual(mock.valve_backend, BackendMode.MOCK)
        self.assertEqual(mock.ndi_backend, BackendMode.MOCK)
        real = HardwareProfile.real()
        self.assertEqual(real.camera_backend, BackendMode.REAL)
        self.assertEqual(real.valve_backend, BackendMode.REAL)
        self.assertEqual(real.ndi_backend, BackendMode.REAL)

    def test_duplicate_camera_serials_rejected(self):
        with self.assertRaises(ValueError):
            HardwareProfile(camera_backend="real", camera_count=2,
                            camera_serials=("A", "A"))

    def test_required_groups_follow_channel_map(self):
        self.assertEqual(required_groups_for_channels((0,)), (1,))
        self.assertEqual(required_groups_for_channels((4,)), (2,))
        self.assertEqual(required_groups_for_channels((0, 5)), (1, 2))

    def test_round_trip(self):
        value = HardwareProfile.real(camera_count=2, camera_serials=("A", "B"))
        self.assertEqual(HardwareProfile.from_dict(value.to_dict()), value)


class HardwareSessionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app()

    def test_mock_valve_uses_controller_and_becomes_ready(self):
        session = HardwareSession()
        session.apply_profile(HardwareProfile.all_mock())
        controller = session.prepare_valves()
        self.assertIn("MockValveController", type(controller).__name__)
        result = session.connect_prepared_valves((1,))
        self.assertTrue(result[1][0])
        self.assertEqual(session.states["valve"], DeviceState.READY)
        session.require_valves_ready((1,))
        with self.assertRaises(HardwareSessionError):
            session.require_valves_ready((2,))
        session.shutdown()

    def test_disabled_valve_never_falls_back_mock(self):
        session = HardwareSession()
        session.apply_profile(HardwareProfile(valve_backend="disabled"))
        with self.assertRaises(HardwareSessionError):
            session.prepare_valves()
        with self.assertRaises(HardwareSessionError):
            session.create_transport((1,))

    def test_profile_cannot_change_while_hardware_exists(self):
        session = HardwareSession()
        session.prepare_valves()
        with self.assertRaises(HardwareSessionError):
            session.apply_profile(HardwareProfile.real())
        session.shutdown()

    def test_real_valve_failure_stays_real_and_enters_error(self):
        session = HardwareSession()
        session.apply_profile(HardwareProfile(valve_backend="real"))
        controller = session.prepare_valves()
        self.assertNotIn("Mock", type(controller).__name__)
        with patch("real_validation.hardware.valve.connect_valve_groups",
                   return_value={1: (False, "port unavailable")}):
            result = session.connect_prepared_valves((1,))
        self.assertFalse(result[1][0])
        self.assertEqual(session.states["valve"], DeviceState.ERROR)
        self.assertIs(session.valve_controller, controller)
        session.shutdown()

    def test_shutdown_releases_all_devices_and_preserves_backends(self):
        session = HardwareSession()
        session.apply_profile(HardwareProfile.all_mock())
        session.start_cameras()
        session.start_ndi()
        session.connect_prepared_valves((1, 2))
        session.shutdown()
        self.assertFalse(session.any_running)
        self.assertEqual(set(session.states.values()), {DeviceState.OFF})


if __name__ == "__main__":
    unittest.main()
