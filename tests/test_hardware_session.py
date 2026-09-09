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

    def test_disconnected_group_config_does_not_rebuild_connected_controller(self):
        from dataclasses import replace
        session=HardwareSession();controller=session.prepare_valves();session.connect_prepared_valves((1,))
        changed=replace(session.profile,group2_port='COM99')
        session.apply_disconnected_config(changed)
        self.assertIs(session.valve_controller,controller)
        self.assertEqual(session.profile.group2_port,'COM99')
        with self.assertRaises(HardwareSessionError):session.apply_disconnected_config(replace(changed,group1_port='COM98'))
        self.assertEqual(session.profile,changed);session.shutdown()

    def test_dispatch_uses_latest_filter_and_zero_bypasses_it(self):
        import threading,time
        session=HardwareSession();controller=session.prepare_valves();session.connect_prepared_valves((1,2))
        transport=session.create_transport((1,2));receipts=[]
        controller.configure_safety([0.]*6,[0.]*6)
        def transact(fn):
            worker=threading.Thread(target=lambda:receipts.append(fn()));worker.start()
            end=time.monotonic()+2
            while worker.is_alive() and time.monotonic()<end:app().processEvents();time.sleep(.001)
            worker.join(1);self.assertFalse(worker.is_alive())
            self.assertEqual(receipts[-1].status,'ack');return receipts[-1]
        transport.command_filter=lambda action:[min(value,3.) for value in action]
        receipt=transact(lambda:transport.send([10.]*6,(1,2),.5))
        self.assertEqual(receipt.requested6,(10.,)*6);self.assertEqual(receipt.applied6,(3.,)*6)
        transport.command_filter=lambda action:(_ for _ in ()).throw(ValueError('normal commands blocked'))
        zero=transact(lambda:transport.zero(.5));self.assertEqual(zero.applied6,(0.,)*6)
        transport.close();session.shutdown()

    def test_optional_ndi_buffer_has_monotonic_samples_and_rejects_old_epoch(self):
        session=HardwareSession();session._ndi_epoch=2
        session._buffer_ndi([1.]*11,10.,1);session._buffer_ndi([2.]*11,11.,2)
        samples=session.evaluation_samples(10.5,12.)
        self.assertEqual(samples['samples'],[(11.,[2.]*11)])
        self.assertFalse(samples['connected'])
        session._on_ndi_data([1.]*11,12.,1)
        self.assertEqual(session.states['ndi'],DeviceState.OFF)
        session._on_camera_error(0,'old camera',session.camera_epoch-1)
        self.assertEqual(session.states['camera'],DeviceState.OFF)
        samples['samples'][0][1][0]=99
        self.assertEqual(session.evaluation_samples(0,12.)['samples'][0][1][0],2.)

    def test_capture_buffer_owns_pixels_and_invalidates_old_generation(self):
        import numpy as np
        session = HardwareSession()
        frame = np.ones((4,5,3),dtype=np.uint8)
        epoch = session.camera_epoch
        session._buffer_frame(0,frame,1.,epoch)
        frame[:] = 9
        first,stamp = session.latest_camera_frame(0)
        self.assertEqual(stamp,1.)
        self.assertTrue(np.all(first==1))
        first[:] = 7
        self.assertTrue(np.all(session.latest_camera_frame(0)[0]==1))
        session.stop_cameras()
        session._buffer_frame(0,frame,2.,epoch)
        self.assertIsNone(session.latest_camera_frame(0))

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

    def test_last_applied_pressure_tracks_controller_ack_source(self):
        session = HardwareSession()
        session.apply_profile(HardwareProfile.all_mock())
        controller = session.prepare_valves()
        session.connect_prepared_valves((1, 2))
        controller.set_pressures((1, 2, 3, 4, 5, 6), bypass_rate=True)
        self.assertEqual(session.last_applied6, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.assertEqual(session.snapshot()["last_applied6"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
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
