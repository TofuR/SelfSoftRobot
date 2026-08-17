"""平面受约束六通道动作的采集、部署与执行回归测试。"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
_REAL_CAPTURE = _ROOT / "real_capture"
if str(_REAL_CAPTURE) not in sys.path:
    sys.path.insert(0, str(_REAL_CAPTURE))

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from recorder import ValveRecorder
from valve_control import (
    MockValveController,
    ValveDriver,
    apply_channel_equalities,
    channel_equality_residuals,
    normalize_channel_equalities,
)


_app: QApplication | None = None


def _ensure_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


class _CameraStub(QObject):
    frame_ready = pyqtSignal(object, float)

    def stop(self):
        pass


class _NdiStub(QObject):
    ndi_data = pyqtSignal(list, float)

    def stop(self):
        pass

    def wait(self, _timeout_ms):
        return True


class CaptureEqualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_invalid_self_and_overlapping_pairs_fail(self):
        for pairs in (((1, 1),), ((1, 2), (2, 3)), ((6, 1),)):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                normalize_channel_equalities(pairs)

    def test_manual_random_sweep_and_replay_projection_uses_six_columns(self):
        pairs = ((1, 2), (4, 5))
        manual = apply_channel_equalities([10, 20, 99, 30, 40, 88], pairs)
        self.assertEqual(manual, [10, 20, 20, 30, 40, 40])
        for mode in ("random", "sweep"):
            driver = ValveDriver([0] * 6, [100] * 6, mode, seed=7)
            for _ in range(20):
                action = apply_channel_equalities(driver.next_action(), pairs)
                self.assertEqual(len(action), 6)
                self.assertEqual(channel_equality_residuals(action, pairs), (0.0, 0.0))
        replay = apply_channel_equalities([1, 2, 3, 4, 5, 6], pairs)
        self.assertEqual(replay, [1, 2, 2, 4, 5, 5])

    def test_controller_projects_before_rate_limit_and_rejects_rate_mismatch(self):
        controller = MockValveController()
        controller.connect()
        controller.configure_channel_equalities(((1, 2), (4, 5)))
        controller.configure_safety([100] * 6, [100] * 6)
        _, applied, _ = controller.set_pressures(
            [10, 20, 99, 30, 40, 88], bypass_rate=True)
        self.assertEqual(applied, [10, 20, 20, 30, 40, 40])
        with self.assertRaises(ValueError):
            controller.configure_safety(
                [100, 100, 90, 100, 100, 100], [100] * 6)

    def test_recorder_writes_equality_contract_and_command_residuals(self):
        controller = MockValveController()
        controller.connect()
        recorder = ValveRecorder(_CameraStub(), _NdiStub(), controller)
        with tempfile.TemporaryDirectory(prefix="planar_capture_") as root:
            seq = Path(root) / "seq"
            try:
                recorder.set_manual_target([1, 20, 99, 3, 40, 88])
                self.assertTrue(recorder.start_recording(
                    str(seq), "manual", [0] * 6, [100] * 6,
                    1.0, 0.1, 6, "test", [100] * 6, [100] * 6,
                    required_groups={1, 2},
                    channel_equalities=((1, 2), (4, 5))))
                recorder._clock.stop()
                recorder._on_tick()
                recorder._clock.stop()
                _ensure_app().processEvents()
                for command_id in list(recorder._pending_commands):
                    recorder._finalize_command(command_id)
                recorder.stop_recording()

                meta = json.loads((seq / "meta.json").read_text())
                with (seq / "commands.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(meta["channel_equalities"], [[1, 2], [4, 5]])
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["action_command1"], rows[0]["action_command2"])
                self.assertEqual(rows[0]["action_command4"], rows[0]["action_command5"])
                self.assertEqual(float(rows[0]["pair_residual0"]), 0.0)
                self.assertEqual(float(rows[0]["pair_residual1"]), 0.0)
            finally:
                recorder.shutdown()

    def test_gui_mirrors_and_locks_follower_controls(self):
        from main_capture import CaptureWindow, N_CHAN

        class TestWindow(CaptureWindow):
            def _load_config(self):
                return None

            def _save_config(self):
                return None

        window = TestWindow(mock_cam=True, mock_valve=True, mock_ndi=True,
                            ndi_count=1)
        window.show()
        _ensure_app().processEvents()
        try:
            window._equality_enabled[0].setChecked(True)
            _ensure_app().processEvents()
            leader = window._equality_leader[0].currentIndex()
            follower = window._equality_follower[0].currentIndex()
            self.assertEqual(window.cb_active.currentIndex(), N_CHAN)
            for widgets, value in (
                    (window._min_sb, 12.0), (window._max_sb, 140.0),
                    (window._rise_sb, 25.0), (window._fall_sb, 18.0),
                    (window._target_sb, 70.0)):
                widgets[leader].setValue(value)
                _ensure_app().processEvents()
                self.assertEqual(widgets[follower].value(), value)
                self.assertFalse(widgets[follower].isEnabled())
        finally:
            window.close()
            _ensure_app().processEvents()


if __name__ == "__main__":
    unittest.main()
