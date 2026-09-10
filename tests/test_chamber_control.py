"""Manual GUI command ownership, rate limiting and emulated serial ACK checks."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from real_validation.gui.main_window import ValidationWindow
from real_validation.hardware.modbus import ModbusRTU


class EchoSerial:
    """Emulate the device's Modbus write reply; never open a hardware port."""
    instances=[]
    def __init__(self,**kwargs):
        self.is_open=True;self.frames=[];self.reply=b'';self.fail=False
        self.instances.append(self)
    def reset_input_buffer(self):self.reply=b''
    def reset_output_buffer(self):pass
    def write(self,data):
        self.frames.append(bytes(data))
        payload=bytes(data[:6]);crc=ModbusRTU.calculate_crc(payload)
        self.reply=payload+crc.to_bytes(2,'little')
        return len(data)
    def flush(self):pass
    def read(self,count):return b'' if self.fail else self.reply[:count]
    def close(self):self.is_open=False


class ChamberControlTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.w=ValidationWindow()
        self.w._save_hardware_config=lambda:None
        self.w._set_combo_data(self.w.hw_profile_preset,'all_mock')
        self.w._on_profile_preset_changed(self.w.hw_profile_preset.currentIndex())
        self.w.run_root.setText(self.tmp.name);self.p=self.w.hereditary_panel
        meta=dict(dt=.1,expansion6=[0,1,1,2,2,3],action_unit_to_kpa=[150.]*4,
                  lower_kpa=[0.]*4,upper_kpa=[150.]*4,rate_kpa_s=[50.]*4,
                  max_horizon=80,radius_mm=8.,checkpoint_sha256='test')
        self.p.loaded((SimpleNamespace(channels=4,n_nodes=15),meta))
        self.d=self.p.chambers
    def wait(self,predicate,timeout=3):
        end=time.monotonic()+timeout
        while not predicate():
            self.app.processEvents();time.sleep(.005)
            if time.monotonic()>end:self.fail(self.d.drive_status.text()+' / '+self.d.status.text())
        self.app.processEvents()
    def connect(self):
        self.w._require_ui_profile_applied()
        self.w.hardware.connect_prepared_valves((1,2));self.p._attach_controller();self.d.refresh()
    def tearDown(self):
        self.d.stop();self.wait(lambda:not self.p.busy)
        self.w.close();self.app.processEvents();self.tmp.cleanup()
    def test_one_button_ramps_and_updates_without_restarting_worker(self):
        self.connect();sent=[]
        self.w.hardware.valve_controller.command_issued.connect(lambda i,r,a,t:sent.append((np.array(a),t)))
        self.d.targets[1].setValue(12);self.d.start_button.click()
        self.wait(lambda:abs(self.p.ack6[1]-12)<.1)
        self.assertAlmostEqual(self.p.ack6[2],12,places=1)
        self.assertGreater(len(sent),1)
        for (a,t),(b,u) in zip(sent,sent[1:]):
            self.assertLessEqual(abs(b[1]-a[1]),50*(u-t)+1e-6)
        worker=self.p.job;self.d.refresh();self.d.targets[1].setValue(20)
        self.d.start_button.click();self.wait(lambda:abs(self.p.ack6[1]-20)<.1)
        self.assertIs(self.p.job,worker);self.assertFalse(self.d.takeover_pending)
        self.assertFalse(self.p.runtime.initialized);self.assertEqual(len(self.p.runtime.history),0)
        self.d.end_button.click();self.wait(lambda:not self.p.busy)
        self.assertAlmostEqual(self.w.hardware.valve_controller.last_command[1],20,places=1)
    def test_settings_only_does_not_send_and_invalid_target_is_rejected(self):
        self.connect();controller=self.w.hardware.valve_controller
        self.d.targets[0].setValue(10);self.d.apply_button.click()
        self.assertEqual(controller.last_command,[0.]*6)
        self.assertIn('MOCK',self.d.device_status.text())
        self.assertIn('尚未下发',self.d.status.text())
        self.d.targets[0].setValue(151);self.d.start_button.click()
        self.assertFalse(self.p.busy);self.assertIn('超出',self.d.status.text())
        self.assertEqual(controller.last_command,[0.]*6)
    def test_real_controller_serial_frames_ack_and_failure_are_visible(self):
        EchoSerial.instances=[]
        self.w._set_combo_data(self.w.hw_valve_backend,'real')
        with patch('real_validation.hardware.modbus.serial',SimpleNamespace(Serial=EchoSerial)):self.connect()
        self.d.targets[0].setValue(30);self.d.start_button.click()
        self.wait(lambda:abs(self.p.ack6[0]-30)<.01)
        port=EchoSerial.instances[0]
        self.assertEqual(port.frames[-1][1],0x10)
        self.assertEqual(int.from_bytes(port.frames[-1][7:9],'big'),4960)
        self.assertTrue(all(int.from_bytes(frame[7:9],'big')<=4960 for frame in port.frames))
        self.d.refresh();self.assertIn('REAL',self.d.device_status.text())
        diagnostic=json.loads(self.d.save_diagnostics().read_text(encoding='utf-8'))
        self.assertEqual(diagnostic['controller_type'],'ValveController')
        self.assertAlmostEqual(diagnostic['ack6'][0],30)
        tx=[row for row in diagnostic['recent_commands'] if row['event']=='serial' and row['direction']=='tx' and row['group']==1]
        self.assertTrue(tx);self.assertEqual(bytes.fromhex(tx[-1]['frame_hex']),port.frames[-1])
        # Same real controller/worker/Qt bridge, with an absent serial response.
        port.fail=True;self.wait(lambda:not self.p.busy)
        self.assertIn('timeout',self.d.drive_status.text())
        self.assertIn('调压失败',self.d.drive_status.text())
        self.assertFalse(self.d.driving)
