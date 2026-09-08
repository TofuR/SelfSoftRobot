"""Replay regression tests: no hardware, synthetic camera frames only."""
import csv
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real_capture'))
import numpy as np
from PyQt5.QtCore import QObject, QCoreApplication, QEventLoop, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication
from valve_control import MockValveController, ReplayDriver, load_action_sequence
from recorder import ValveRecorder

class Camera(QObject):
    frame_ready = pyqtSignal(np.ndarray, float)
    def stop(self): pass

class Ndi(QObject):
    ndi_data = pyqtSignal(list, float)
    def stop(self): pass
    def wait(self, _): pass

class ReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.controller = MockValveController()
        self.controller.connect()
        self.rec = ValveRecorder(Camera(), Ndi(), self.controller)

    def tearDown(self):
        self.rec.shutdown()
        self.tmp.cleanup()

    def csv(self, rows):
        path = self.root / 'input.csv'
        with path.open('w', newline='') as stream:
            csv.writer(stream).writerows(rows)
        return str(path)

    def start(self, path, interval=.08, settle=.06, hi=10):
        return self.rec.start_recording(str(self.root / 'seq'), 'replay',
            [0]*6, [hi]*6, interval, settle, 6, 'test',
            rise_rates=[0]*6, fall_rates=[0]*6, replay_path=path,
            required_groups={1,2})

    def run_events(self, ms):
        loop = QEventLoop()
        QTimer.singleShot(ms, loop.quit)
        loop.exec_()

    def test_formats_ignore_time_and_preserve_pressure(self):
        actions = [[1,2,3,4,5,6], [2,3,4,5,6,7]]
        for rows in (actions, [['c'+str(i) for i in range(6)], *actions],
                     [['t_sec', *['c'+str(i) for i in range(6)]],
                      ['ignored', *actions[0]], [-99, *actions[1]]]):
            driver = ReplayDriver(self.csv(rows))
            self.assertEqual(driver.next_action(), actions[0])
            self.assertEqual(driver.next_delay(.12), .12)
            self.assertEqual(driver.next_action(), actions[1])
            self.assertIsNone(driver.next_action())

    def test_rejects_bad_pressure_schema_and_ranges_before_start(self):
        for row in ([float('nan')]*6, [float('inf')]*6, [501]*6,
                    [-1]*6, [1]*5, [1]*8, ['bad']*6):
            with self.subTest(row=row), self.assertRaises(ValueError):
                load_action_sequence(self.csv([row]))
        self.assertFalse(self.start(self.csv([[11]*6])))
        self.assertFalse((self.root/'seq').exists())
        self.assertEqual(self.controller.last_command, [0]*6)

    def test_projection_must_also_respect_limits(self):
        driver = ReplayDriver(self.csv([[5,0,0,0,0,0]]))
        with self.assertRaises(ValueError):
            driver.validate_ranges([0]*6, [10,1,10,10,10,10], [0,0,2,3,4,5])

    def test_timing_final_frame_and_automatic_completion_at_two_rates(self):
        for interval in (.08, .14):
            with self.subTest(interval=interval):
                if (self.root/'seq').exists():
                    (self.root/'seq').rename(self.root/'previous')
                path = self.csv([[0,1,0,0,0,0,0], [.001,2,0,0,0,0,0], [100,3,0,0,0,0,0]])
                timer = QTimer()
                timer.timeout.connect(lambda: self.rec._on_cam(0, np.zeros((8,8,3),np.uint8), time.monotonic()))
                timer.start(3)
                try:
                    self.assertTrue(self.start(path, interval, interval-.02))
                    self.run_events(int(interval*4000)+500)
                    self.assertFalse(self.rec.recording)
                    meta = json.loads((self.root/'seq/meta.json').read_text())
                    self.assertTrue(meta['replay_completed'])
                    with (self.root/'seq/commands.csv').open() as f:
                        commands = list(csv.DictReader(f))
                    with (self.root/'seq/actions6.csv').open() as f:
                        frames = list(csv.DictReader(f))
                    self.assertEqual([float(r['c0']) for r in frames], [1,2,3])
                    self.assertEqual(len(commands), 3)
                    delta = float(commands[1]['t_command'])-float(commands[0]['t_command'])
                    self.assertAlmostEqual(delta, interval, delta=.035)
                    self.assertEqual(self.controller.last_command[0],3)
                finally:
                    timer.stop()

    def test_stale_frame_rejected_and_live_limit_change_stops(self):
        self.assertTrue(self.start(self.csv([[1]*6,[2]*6])))
        self.rec._clock.stop()
        self.rec._on_cam(0,np.zeros((8,8,3),np.uint8),time.monotonic()-1)
        self.rec._on_tick()
        self.rec._clock.stop()
        self.run_events(70)
        self.assertEqual(self.rec._frame_idx,0)
        self.rec.update_ranges([0]*6,[.5]*6)
        self.assertFalse(self.rec.recording)
        self.assertEqual(self.controller.last_command,[0]*6)

    def test_old_grab_callback_cannot_write_into_restarted_run(self):
        path = self.csv([[1]*6])
        self.assertTrue(self.start(path, .3, .2))
        self.rec._clock.stop()
        self.rec._on_tick()
        self.rec.stop_recording()
        (self.root/'seq').rename(self.root/'old')
        self.assertTrue(self.start(path, .3, .2))
        self.rec._clock.stop()
        self.run_events(250)
        self.assertEqual(self.rec._frame_idx,0)
        self.assertTrue(self.rec.recording)

if __name__ == '__main__':
    unittest.main()
