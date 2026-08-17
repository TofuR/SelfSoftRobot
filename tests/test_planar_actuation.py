"""平面受约束六通道动作的采集、部署与执行回归测试。"""

from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
_REAL_CAPTURE = _ROOT / "real_capture"

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

from scripts.real.masks_to_transition_npz import (
    load_planarity_qc,
    save_npz,
    validate_action_equalities,
    validate_equality_action_maxes,
)
from scripts.evaluation.eval_planarity import (
    evaluate_planarity,
    load_ndi_xyz,
)
from real_validation.contracts.deploy_manifest import DeployManifest
from real_validation.contracts.models import (
    Anchor,
    ModelDescriptor,
    SafetyPolicy,
    Scene,
    ScenePrimitive,
)
from real_validation.execution.executor import (
    CommandReceipt,
    ExecutionError,
    MockCommandTransport,
    PlanExecutor,
)
from real_validation.execution.preflight import validate_plan
from real_validation.planning.openloop_planner import (
    OpenLoopShootingPlanner,
    ShootingConfig,
)
from real_validation.planning.planner_service import build_plan
from real_validation.runtime.warmup import warmup_actions


_app: QApplication | None = None


def _ensure_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


def _capture_module(name: str):
    if str(_REAL_CAPTURE) not in sys.path:
        sys.path.insert(0, str(_REAL_CAPTURE))
    return importlib.import_module(name)


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

    @classmethod
    def tearDownClass(cls):
        # real_validation 的硬件适配层必须保持自包含；采集测试不得把顶层硬件模块
        # 泄漏给同进程中的 import-hygiene 测试。
        for name in ("main_capture", "recorder", "valve_control", "modbus_manager",
                     "realsense_cam", "hardware_threads", "nditracker"):
            sys.modules.pop(name, None)
        try:
            sys.path.remove(str(_REAL_CAPTURE))
        except ValueError:
            pass

    def test_invalid_self_and_overlapping_pairs_fail(self):
        normalize_channel_equalities = _capture_module(
            "valve_control").normalize_channel_equalities
        for pairs in (((1, 1),), ((1, 2), (2, 3)), ((6, 1),)):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                normalize_channel_equalities(pairs)

    def test_manual_random_sweep_and_replay_projection_uses_six_columns(self):
        valve = _capture_module("valve_control")
        ValveDriver = valve.ValveDriver
        apply_channel_equalities = valve.apply_channel_equalities
        channel_equality_residuals = valve.channel_equality_residuals
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
        MockValveController = _capture_module("valve_control").MockValveController
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
        ValveRecorder = _capture_module("recorder").ValveRecorder
        MockValveController = _capture_module("valve_control").MockValveController
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
        main_capture = _capture_module("main_capture")
        CaptureWindow, N_CHAN = main_capture.CaptureWindow, main_capture.N_CHAN

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


class PreprocessingEqualityTest(unittest.TestCase):
    def test_six_dimensional_equalities_are_validated_before_normalization(self):
        actions = [[1, 20, 20, 4, 50, 50], [2, 30, 30, 5, 60, 60]]
        residual = validate_action_equalities(
            actions, range(6), ((1, 2), (4, 5)))
        self.assertEqual(residual.tolist(), [0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "必须使用 --action-channels"):
            validate_action_equalities(actions, (0,), ((1, 2),))
        actions[1][2] = 31
        with self.assertRaisesRegex(ValueError, "违反 channel_equalities"):
            validate_action_equalities(actions, range(6), ((1, 2), (4, 5)))

    def test_linked_action_normalization_maxes_must_match(self):
        validate_equality_action_maxes(
            [100] * 6, range(6), ((1, 2), (4, 5)))
        with self.assertRaisesRegex(ValueError, "归一化上限必须相同"):
            validate_equality_action_maxes(
                [100, 100, 90, 100, 100, 100], range(6), ((1, 2),))

    def test_npz_keeps_six_actions_and_equality_metadata(self):
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="planar_npz_") as root:
            path = Path(root) / "train" / "seq.npz"
            save_npz(
                str(path), np.zeros((2, 3, 15)), np.zeros((2, 6)),
                n_points=15, tip_fix=True,
                channel_equalities=((1, 2), (4, 5)),
                pair_residual_max=[0.0, 0.0],
                planarity_qc={"planarity_pass": True})
            with np.load(path) as data:
                self.assertEqual(data["actions"].shape, (2, 6))
                self.assertEqual(json.loads(str(data["channel_equalities"])),
                                 [[1, 2], [4, 5]])
                self.assertEqual(data["pair_residual_max"].tolist(), [0.0, 0.0])
                self.assertTrue(json.loads(str(data["planarity_qc"]))["planarity_pass"])

    def test_failed_planarity_qc_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="planarity_qc_") as root:
            path = Path(root) / "planarity_qc.json"
            path.write_text(json.dumps({"planarity_pass": False}))
            with self.assertRaisesRegex(ValueError, "平面性质控未通过"):
                load_planarity_qc(root)


def _deployment_fixtures(*, history=None, safety=None):
    model = ModelDescriptor(
        "mock.pt", "planar", "state_transition", 6, 3, 2,
        k_train=4, k_safe=4,
        action_scale_kpa=(100.0,) * 6, channel_map=tuple(range(6)),
        channel_equalities=((1, 2), (4, 5)),
        train_dt_nominal_s=0.1, train_dt_measured_s=0.1,
        train_dt_std_s=0.0)
    anchor = Anchor(
        state=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
        action_history=history or ((0.0,) * 6, (0.0,) * 6), source="test")
    scene = Scene("planar", (ScenePrimitive(
        "target_point", "model_normalized", {"xy": [0.2, 0.0], "node": 0}),))
    safety = safety or SafetyPolicy(
        pressure_max6=(100.0,) * 6,
        rise_rate6=(100.0,) * 6, fall_rate6=(100.0,) * 6,
        ack_timeout_s=0.1)
    plan = build_plan(
        model_actions=((1, 2, 9, 3, 4, 8),),
        channel_map=tuple(range(6)), step_interval_s=0.1,
        model=model, anchor=anchor, scene=scene, safety=safety)
    return model, anchor, scene, safety, plan


class DeploymentEqualityTest(unittest.TestCase):
    def test_manifest_round_trip_keeps_equalities(self):
        manifest = DeployManifest(
            checkpoint_sha256="deadbeef", action_scale_kpa=(100.0,) * 6,
            channel_map=tuple(range(6)), channel_equalities=((1, 2), (4, 5)),
            train_dt_nominal_s=0.1, mask_source="white_on_blue",
            n_nodes=15, window_size=40, z_dim=16, episode_len=40,
            action_dim=6, encoder_type="fractional", hidden_dim=128, n_scales=4)
        restored = DeployManifest.from_dict(manifest.to_dict())
        self.assertEqual(restored.channel_equalities, ((1, 2), (4, 5)))
        with self.assertRaisesRegex(ValueError, "identity channel_map"):
            DeployManifest(
                checkpoint_sha256="deadbeef", action_scale_kpa=(100.0,) * 6,
                channel_map=(1, 0, 2, 3, 4, 5), channel_equalities=((1, 2),),
                train_dt_nominal_s=0.1, mask_source="white_on_blue",
                n_nodes=15, window_size=40, z_dim=16, episode_len=40,
                action_dim=6, encoder_type="fractional", hidden_dim=128, n_scales=4)

    def test_build_plan_projects_model_actions_and_records_contract(self):
        model, anchor, scene, safety, plan = _deployment_fixtures()
        self.assertEqual(plan.channel_equalities, model.channel_equalities)
        self.assertEqual(plan.actions6[0], (1.0, 2.0, 2.0, 3.0, 4.0, 4.0))
        self.assertTrue(validate_plan(plan, model, anchor, scene, safety).ok)

    def test_preflight_rejects_history_and_safety_outside_manifold(self):
        bad_history = ((0.0,) * 6, (0.0, 1.0, 2.0, 0.0, 0.0, 0.0))
        model, anchor, scene, safety, plan = _deployment_fixtures(history=bad_history)
        codes = {issue.code for issue in validate_plan(
            plan, model, anchor, scene, safety).issues}
        self.assertIn("history_equality", codes)

        bad_safety = SafetyPolicy(
            pressure_max6=(100, 100, 90, 100, 100, 100),
            rise_rate6=(100,) * 6, fall_rate6=(100,) * 6,
            ack_timeout_s=0.1)
        model, anchor, scene, _, plan = _deployment_fixtures(safety=bad_safety)
        codes = {issue.code for issue in validate_plan(
            plan, model, anchor, scene, bad_safety).issues}
        self.assertIn("safety_equality", codes)

    def test_seeded_warmup_is_projected_after_jitter(self):
        actions = warmup_actions(
            6, 20, seed=7, channel_equalities=((1, 2), (4, 5)))
        self.assertTrue((actions[:, 1] == actions[:, 2]).all())
        self.assertTrue((actions[:, 4] == actions[:, 5]).all())

    def test_executor_zeros_when_ack_breaks_equality(self):
        class BadAckTransport(MockCommandTransport):
            def send(self, action6, required_groups, timeout_s):
                receipt = super().send(action6, required_groups, timeout_s)
                if any(float(value) for value in action6):
                    applied = list(receipt.applied6)
                    applied[2] += 1.0
                    return CommandReceipt(
                        receipt.command_id, receipt.requested6, tuple(applied),
                        receipt.t_command, receipt.t_ack, receipt.status)
                return receipt

        _, _, _, safety, plan = _deployment_fixtures()
        transport = BadAckTransport()
        with self.assertRaisesRegex(ExecutionError, "applied6"):
            PlanExecutor(transport, safety).execute(plan)
        self.assertEqual(transport.commands[-1], (0.0,) * 6)

    def test_planner_optimizes_four_variables_and_outputs_equal_six_channels(self):
        import torch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))
                self.register_buffer("pc_center", torch.zeros(3))
                self.register_buffer("pc_scale", torch.ones(3))

            def init_z_from_action(self, action_window):
                return action_window.new_zeros((action_window.shape[0], 1))

            def forward(self, action_window, state, state_previous, latent):
                del state_previous
                delta = action_window[:, -1, 0].view(-1, 1, 1)
                direction = state.new_tensor((1.0, 0.0, 0.0)).view(1, 1, 3)
                return {"skeleton": state + delta * direction,
                        "latent_z": latent}

        model, anchor, scene, safety, _ = _deployment_fixtures()
        runtime = type("Runtime", (), {
            "descriptor": model, "model": TinyModel(), "info": {"norm_factor": 1.0},
        })()
        with tempfile.TemporaryDirectory(prefix="planar_planner_") as root:
            plan = OpenLoopShootingPlanner(runtime).plan(
                anchor=anchor, scene=scene, safety=safety,
                channel_map=tuple(range(6)), step_interval_s=0.1,
                output_dir=root,
                config=ShootingConfig(
                    horizon=2, n_iter=3, n_restarts=1, learning_rate=0.05))
        self.assertEqual(plan.metadata["optimizer_action_dim"], 4)
        for action in plan.actions6:
            self.assertEqual(action[1], action[2])
            self.assertEqual(action[4], action[5])
        self.assertTrue(validate_plan(plan, model, anchor, scene, safety).ok)


class PlanarityQcTest(unittest.TestCase):
    def test_baseline_plane_and_p95_pass_are_reported(self):
        import numpy as np

        points = np.array([
            [0, 0, 10], [1, 0, 10], [2, 0, 10],
            [3, 0, 10.2], [4, 0, 9.8],
        ], dtype=float)
        report = evaluate_planarity(
            points, (0, 0, 1), 0.25, baseline_samples=3, pass_stat="p95")
        self.assertTrue(report["planarity_pass"])
        self.assertEqual(report["valid_samples"], 5)
        self.assertAlmostEqual(report["plane_point_mm"][2], 10.0)
        self.assertAlmostEqual(report["planarity_tip_abs_mm_max"], 0.2)

    def test_failed_threshold_and_invalid_normal_are_rejected_or_flagged(self):
        points = [[0, 0, 0], [0, 0, 2]]
        report = evaluate_planarity(
            points, (0, 0, 1), 0.5, plane_point=(0, 0, 0), pass_stat="max")
        self.assertFalse(report["planarity_pass"])
        with self.assertRaisesRegex(ValueError, "零向量"):
            evaluate_planarity(points, (0, 0, 0), 1.0)

    def test_ndi_loader_filters_nonfinite_and_low_quality_rows(self):
        with tempfile.TemporaryDirectory(prefix="ndi_qc_") as root:
            path = Path(root) / "ndi.csv"
            path.write_text(
                "t_sec,ndi0_x,ndi0_y,ndi0_z,ndi0_quality\n"
                "0,1,2,3,0.9\n"
                "1,nan,2,3,0.9\n"
                "2,4,5,6,0.1\n", encoding="utf-8")
            points, total = load_ndi_xyz(path, min_quality=0.5)
            self.assertEqual(total, 3)
            self.assertEqual(points.tolist(), [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
