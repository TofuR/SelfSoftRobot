import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from real_validation.planning.coordinate_system import PlanarTransform
from real_validation.execution.executor import ExecutionError, MockCommandTransport, PlanExecutor
from real_validation.contracts.models import (
    ActionPlan, Anchor, ModelDescriptor, SafetyPolicy, Scene, ScenePrimitive,
)
from real_validation.execution.metrics import (
    evaluate_command_safety, evaluate_plan_scene, evaluate_prediction,
)
from real_validation.runtime.observation_policy import ObservationPolicy
from real_validation.runtime.observation_policy import ActionHistoryBuffer
from real_validation.runtime.anchors import anchor_from_npz
from real_validation.planning.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
from real_validation.planning.planner_service import build_plan, expand_model_actions
from real_validation.execution.preflight import validate_plan
from real_validation.runtime.loader import load_openloop_model
from real_validation.runtime.model import OpenLoopTransitionModel
from real_validation.core.session import ExperimentSession, SessionState


def fixtures():
    model = ModelDescriptor(
        "mock.pt", "abc", "state_transition", 1, 3, 2,
        k_train=4, k_safe=4,
        action_scale_kpa=(1.0,), channel_map=(0,),
        train_dt_nominal_s=0.1, train_dt_measured_s=0.1, train_dt_std_s=0.0,
        mask_source="white_on_blue", mask_source_provenance="test",
        segment_params={"val": 100.0},
        k_safe_table_px={"5px": 4}, registration_residual_max_px=2.0)
    anchor = Anchor(
        state=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
        action_history=((0.0,), (0.0,)), source="test")
    scene = Scene("test", (ScenePrimitive("target_point", "model", {"xy": [2, 3]}),))
    safety = SafetyPolicy(
        pressure_max6=(100.0,) * 6,
        rise_rate6=(100.0,) * 6,
        fall_rate6=(100.0,) * 6,
        ack_timeout_s=0.1,
    )
    plan = build_plan(model_actions=((10.0,), (20.0,)), channel_map=(0,),
                      step_interval_s=0.1, model=model, anchor=anchor,
                      scene=scene, safety=safety, random_seed=7)
    return model, anchor, scene, safety, plan


class ValidationCoreTest(unittest.TestCase):
    def test_expand_requires_explicit_mapping(self):
        self.assertEqual(expand_model_actions(((1, 2, 3),), (0, 2, 5))[0],
                         (1.0, 0.0, 2.0, 0.0, 0.0, 3.0))
        with self.assertRaises(ValueError):
            expand_model_actions(((1, 2),), (0,))

    def test_preflight_passes_and_detects_stale_scene(self):
        model, anchor, scene, safety, plan = fixtures()
        self.assertTrue(validate_plan(plan, model, anchor, scene, safety).ok)
        changed = scene.with_primitive(
            ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 2}))
        result = validate_plan(plan, model, anchor, changed, safety)
        self.assertIn("stale_scene", {issue.code for issue in result.issues})

    def test_preflight_blocks_slew_and_inactive_channel(self):
        model, anchor, scene, safety, plan = fixtures()
        bad = ActionPlan(
            actions6=((50, 1, 0, 0, 0, 0),), step_interval_s=0.1,
            model_action_dim=1, channel_map=(0,), model_hash=model.checkpoint_hash,
            scene_digest=scene.digest, anchor_id=anchor.anchor_id,
            safety_digest=safety.digest)
        codes = {issue.code for issue in validate_plan(bad, model, anchor, scene, safety).issues}
        self.assertIn("slew_rate", codes)
        self.assertIn("inactive_channel", codes)

    def test_preflight_blocks_predicted_collision(self):
        model, anchor, scene, safety, plan = fixtures()
        data = plan.to_dict(); data["metadata"] = {"predicted_min_obstacle_clearance": -0.2}
        result = validate_plan(ActionPlan.from_dict(data), model, anchor, scene, safety)
        self.assertIn("predicted_collision", {issue.code for issue in result.issues})

    def test_session_arm_and_plan_invalidation(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model)
            session.set_anchor(anchor)
            session.set_scene(scene)
            session.set_safety(safety)
            self.assertTrue(session.accept_plan(plan).ok)
            self.assertEqual(session.state, SessionState.READY)
            session.arm()
            self.assertEqual(session.state, SessionState.ARMED)
            snapshot = json.loads((session.run_dir / "experiment.json").read_text())
            self.assertEqual(snapshot["state"], "armed")

    def test_scene_without_and_replace_primitive(self):
        model, anchor, scene, safety, plan = fixtures()
        obstacle = ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 2})
        scene2 = scene.with_primitive(obstacle)
        self.assertEqual(len(scene2.primitives), 2)
        removed = scene2.without_primitive(obstacle.primitive_id)
        self.assertEqual(len(removed.primitives), 1)
        # 删→加回同一原语,digest 不同(新 primitive_id + 新 revision):任何编辑都让旧 plan stale
        added_back = removed.with_primitive(
            ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 2}))
        self.assertNotEqual(added_back.digest, scene2.digest)
        with self.assertRaises(KeyError):
            scene2.without_primitive("nonexistent")

    def test_scene_change_invalidates_ready_plan(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.set_scene(scene); session.set_safety(safety)
            self.assertTrue(session.accept_plan(plan).ok)
            session.set_scene(scene.with_primitive(
                ScenePrimitive("waypoint", "model", {"xy": [0, 0]})))
            self.assertIsNone(session.plan)
            self.assertEqual(session.state, SessionState.IDLE)

    def test_scene_set_in_executing_is_blocked(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.set_scene(scene); session.set_safety(safety)
            session.accept_plan(plan)
            session.arm()
            session.transition(SessionState.EXECUTING, "test")
            with self.assertRaises(RuntimeError):
                session.set_scene(scene)          # 执行中禁止改 scene(B16)
            with self.assertRaises(RuntimeError):
                session.set_anchor(anchor)

    def test_invalidate_model_clears_descriptor(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.invalidate_model("load failed")
            self.assertIsNone(session.model)
            self.assertIsNone(session.anchor)
            self.assertIsNone(session.plan)

    def test_replay_session_cannot_arm(self):
        model, anchor, scene, safety, plan = fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            session.configure_model(model); session.set_anchor(anchor)
            session.set_scene(scene); session.set_safety(safety)
            self.assertTrue(session.accept_plan(plan).ok)
            replay = ExperimentSession.load_for_replay(session.run_dir)
            self.assertTrue(replay.replay_only)
            with self.assertRaises(RuntimeError):
                replay.arm()

    def test_executor_records_ack(self):
        *_, safety, plan = fixtures()
        transport = MockCommandTransport()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "execution.csv"
            receipts = PlanExecutor(transport, safety).execute(plan, output)
            self.assertEqual(len(receipts), 2)
            self.assertTrue(output.is_file())
            self.assertTrue(all(item.status == "ack" for item in receipts))

    def test_executor_failure_zeros_all_channels(self):
        *_, safety, plan = fixtures()
        transport = MockCommandTransport(fail_at=2, status="timeout")
        executor = PlanExecutor(transport, safety)
        with self.assertRaises(ExecutionError):
            executor.execute(plan)
        self.assertEqual(transport.commands[-1], (0.0,) * 6)

    def test_execution_csv_records_jitter(self):
        import csv
        *_, safety, plan = fixtures()
        # 下发阻塞 → 后续命令错过 deadline → jitter_s 滞后
        transport = MockCommandTransport(send_delay_s=0.15)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "execution.csv"
            PlanExecutor(transport, safety).execute(plan, output)
            rows = list(csv.reader(output.open()))
            header = rows[0]
            self.assertIn("t_expected", header)
            self.assertIn("jitter_s", header)
            jitters = [float(r[header.index("jitter_s")]) for r in rows[1:]]
            self.assertGreater(jitters[-1], 0.05)   # 末步被前一步阻塞拖滞后 > 0.05s

    def test_zero_with_retry_hard_fails(self):
        *_, safety, plan = fixtures()
        transport = MockCommandTransport(fail_at=1, zero_always_fails=True)
        executor = PlanExecutor(transport, safety)
        with self.assertRaises(ExecutionError):
            executor.execute(plan)
        # 归零重试 3 次都失败 → 抛 ExecutionError(不能静默放过)

    def test_zero_pause_requires_replanning(self):
        *_, safety, _ = fixtures()
        executor = PlanExecutor(MockCommandTransport(), safety)
        executor.pause()
        with self.assertRaises(ExecutionError):
            executor.resume()

    def test_invalid_hardware_values_are_rejected(self):
        with self.assertRaises(ValueError):
            SafetyPolicy(pressure_max6=(600,) * 6)
        with self.assertRaises(ValueError):
            Anchor(state=((0, 0),), action_history=((float("nan"),),))

    def test_hidden_observation_is_rejected(self):
        policy = ObservationPolicy("anchor_only")
        first = policy.decide(step=0, timestamp=0.0, source="cam0")
        hidden = policy.decide(step=1, timestamp=1.0, source="cam0")
        self.assertTrue(first.allowed)
        self.assertFalse(hidden.allowed)
        with self.assertRaises(PermissionError):
            policy.require_allowed(hidden)

    def test_action_history_uses_applied_six_channel_commands(self):
        history = ActionHistoryBuffer(2, 2, (1, 4))
        history.append_applied6((0, 1, 2, 3, 4, 5))
        self.assertFalse(history.ready)
        history.append_applied6((5, 4, 3, 2, 1, 0))
        self.assertEqual(history.snapshot(), ((1.0, 4.0), (4.0, 1.0)))

    def test_executor_accumulates_applied_history_for_reanchor(self):
        # 功能①:执行后,本次实验的实际下发动作累积进 history_buffer(供滚动重锚定)
        _, _, _, safety, plan = fixtures()
        history = ActionHistoryBuffer(history_steps=2, action_dim=1, channel_map=(0,))
        executor = PlanExecutor(MockCommandTransport(), safety, history_buffer=history)
        executor.execute(plan)
        self.assertTrue(history.ready)
        snapshot = history.snapshot()
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(len(snapshot[0]), 1)   # action_dim=1
        self.assertNotEqual(snapshot[0], (0.0,))  # 累积的是真实下发值,不是零

    def test_offline_npz_anchor_requires_full_history(self):
        import torch
        model, _, _, _, _ = fixtures()
        runtime_model = type("Geometry", (), {
            "pc_center": torch.zeros(3), "pc_scale": torch.ones(3),
        })()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.npz"
            np.savez(path, positions=np.zeros((4, 3, 3), dtype=np.float32),
                     actions=np.arange(4, dtype=np.float32).reshape(4, 1))
            with self.assertRaises(ValueError):
                anchor_from_npz(path, 0, model, runtime_model)
            anchor = anchor_from_npz(path, 2, model, runtime_model)
            self.assertEqual(anchor.action_history, ((1.0,), (2.0,)))

    def test_planar_transform_roundtrip(self):
        transform = PlanarTransform([[2, 0, 3], [0, 2, 4], [0, 0, 1]],
                                    "pixel", "model")
        points = np.array([[0.0, 0.0], [1.0, 2.0]])
        self.assertLess(transform.roundtrip_error(points), 1e-12)

    def test_qt_valve_transport_keeps_controller_on_qt_thread(self):
        from PyQt5.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal
        from real_validation.execution.hardware_session import QtValveTransport

        class FakeController(QObject):
            communication_result = pyqtSignal(str, int, bool, float, str)

            @property
            def connected_groups(self):
                return {1, 2}

            def set_pressures(self, action, command_id, bypass_rate, required_groups):
                del bypass_rate
                timestamp = time.monotonic()
                for group in required_groups:
                    QTimer.singleShot(0, lambda current=group: self.communication_result.emit(
                        command_id, current, True, time.monotonic(), "ack"))
                return command_id, list(action), timestamp

        app = QCoreApplication.instance() or QCoreApplication([])
        controller = FakeController()
        transport = QtValveTransport(controller)
        result = []

        worker = threading.Thread(target=lambda: result.append(
            transport.send((1, 2, 3, 4, 5, 6), (1, 2), 1.0)))
        worker.start()
        deadline = time.monotonic() + 2.0
        while worker.is_alive() and time.monotonic() < deadline:
            app.processEvents(); time.sleep(0.001)
        worker.join(0.1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result[0].status, "ack")
        transport.close()

    def test_openloop_planner_produces_preflight_safe_plan(self):
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

        model, anchor, _, safety, _ = fixtures()
        scene = Scene("planner", (ScenePrimitive(
            "target_point", "model_normalized", {"xy": [0.2, 0.0], "node": 0}),))
        runtime = type("Runtime", (), {
            "descriptor": model, "model": TinyModel(), "info": {"norm_factor": 100.0},
        })()
        with tempfile.TemporaryDirectory() as temporary:
            plan = OpenLoopShootingPlanner(runtime).plan(
                anchor=anchor, scene=scene, safety=safety, channel_map=(0,),
                step_interval_s=0.1, output_dir=temporary,
                config=ShootingConfig(horizon=2, n_iter=8, n_restarts=1,
                                      learning_rate=0.1))
            self.assertTrue(validate_plan(plan, model, anchor, scene, safety).ok)
            self.assertTrue((Path(temporary) / plan.predicted_states_path).is_file())

    def test_openloop_planner_repeat_warmstart_is_kpa_space(self):
        """锁 I1:restart-0 repeat warm-start 必须落在 kPa 空间(_project_actions 用 kPa 投影)。

        action_scale_kpa=50, norm=1, 模型单位 history[-1]=0.8 → seed_last=40 kPa。
        常量输出模型使 raw 无梯度,physical = 从 initial=0 向 seed_last 的 slew 限制爬升。
        旧式 seed_last = history[-1]*norm = 0.8(名义 kPa) → 首步≈0.8;修复后首步 = rise*dt = 10。
        """
        import torch

        class ConstModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(()))
                self.register_buffer("pc_center", torch.zeros(3))
                self.register_buffer("pc_scale", torch.ones(3))

            def init_z_from_action(self, action_window):
                return action_window.new_zeros((action_window.shape[0], 1))

            def forward(self, action_window, state, state_previous, latent):
                del action_window, state_previous
                return {"skeleton": state.new_ones(state.shape) * 0.5,
                        "latent_z": latent}

        model = ModelDescriptor(
            "mock.pt", "abc", "state_transition", 1, 3, 2,
            k_train=4, k_safe=4,
            action_scale_kpa=(50.0,), channel_map=(0,),
            train_dt_nominal_s=0.1, train_dt_measured_s=0.1, train_dt_std_s=0.0)
        anchor = Anchor(state=((0.0, 0.0), (1.0, 1.0), (2.0, 2.0)),
                        action_history=((0.5,), (0.8,)), action_units="model_normalized")
        scene = Scene("planner", (ScenePrimitive(
            "target_point", "model_normalized", {"xy": [0.2, 0.0], "node": 0}),))
        safety = SafetyPolicy(
            pressure_max6=(150.0,) * 6,
            rise_rate6=(100.0,) * 6, fall_rate6=(100.0,) * 6,
            ack_timeout_s=0.1)
        runtime = type("Runtime", (), {
            "descriptor": model, "model": ConstModel(), "info": {"norm_factor": 1.0},
        })()
        with tempfile.TemporaryDirectory() as temporary:
            plan = OpenLoopShootingPlanner(runtime).plan(
                anchor=anchor, scene=scene, safety=safety, channel_map=(0,),
                step_interval_s=0.1, output_dir=temporary,
                config=ShootingConfig(horizon=2, n_iter=2, n_restarts=1,
                                      learning_rate=1e-6))
            # seed_last=40 kPa,从 initial=0 起步,slew=100 kPa/s × 0.1s = 10 → 首步≈10 kPa;
            # 旧 bug(0.8 名义 kPa)会让首步≈0.8 → 断言 5 kPa 以上区分两者
            self.assertGreater(plan.actions6[0][0], 5.0)
            self.assertGreater(plan.actions6[1][0], 5.0)

    def test_standalone_runtime_loads_local_checkpoint(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "model": "OpenLoopTransitionModel", "action_dim": 1,
                "n_nodes": 3, "window_size": 2, "hidden_dim": 8,
                "n_scales": 2, "encoder_type": "fractional", "z_dim": 4,
            }
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            source = OpenLoopTransitionModel(1, 3, 8, 2, 2, 4)
            checkpoint = root / "best_model.pt"
            torch.save(source.state_dict(), checkpoint)
            loaded = load_openloop_model(str(checkpoint))["model"]
            action = torch.randn(1, 2, 1)
            state = torch.randn(1, 3, 3)
            with torch.no_grad():
                source_z = source.init_z_from_action(action)
                loaded_z = loaded.init_z_from_action(action)
                source_output = source(action, state, state, source_z)["skeleton"]
                loaded_output = loaded(action, state, state, loaded_z)["skeleton"]
            self.assertTrue(torch.equal(source_output, loaded_output))

    def test_prediction_metrics_include_task_and_collision(self):
        predicted = np.zeros((2, 3, 2))
        observed = np.zeros((2, 3, 2))
        observed[-1, 0] = (2.0, 0.0)
        scene = Scene("metrics", (
            ScenePrimitive("target_circle", "model", {"center": [2, 0], "r": 0.1}),
            ScenePrimitive("obstacle_circle", "model", {"center": [5, 5], "r": 1}),
        ))
        metrics = evaluate_prediction(predicted, observed, scene)
        self.assertTrue(metrics["target_success"])
        self.assertFalse(metrics["collision"])
        self.assertEqual(len(metrics["error_by_k"]), 2)

    def test_plan_scene_metrics_use_predicted_states_only(self):
        # 打磨①:离线下用 predicted_states 即可算计划侧场景指标(无需 observed/gap)
        predicted = np.zeros((2, 3, 2))
        predicted[-1, 0] = (2.0, 0.0)              # 末端恰好到达目标中心
        scene = Scene("plan", (
            ScenePrimitive("target_circle", "model", {"center": [2, 0], "r": 0.1}),
            ScenePrimitive("obstacle_circle", "model", {"center": [5, 5], "r": 1}),
        ))
        metrics = evaluate_plan_scene(predicted, scene, tip_node=0)
        self.assertTrue(metrics["target_success"])
        self.assertFalse(metrics["collision"])
        self.assertEqual(metrics["steps"], 2)
        self.assertEqual(metrics["nodes"], 3)
        # 无 scene 时只返回步数/节点数,不崩
        bare = evaluate_plan_scene(predicted, None)
        self.assertEqual(bare["steps"], 2)
        self.assertNotIn("target_success", bare)

    def test_command_safety_metrics(self):
        *_, safety, _ = fixtures()
        metrics = evaluate_command_safety(((10, 0, 0, 0, 0, 0),
                                           (50, 0, 0, 0, 0, 0)), 0.1, safety)
        self.assertTrue(metrics["pressure_safe"])
        self.assertFalse(metrics["slew_safe"])


if __name__ == "__main__":
    unittest.main()
