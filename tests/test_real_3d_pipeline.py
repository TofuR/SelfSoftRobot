"""三维第一版：相机源、三角化质量、15节点合同与训练 loss。"""

import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CAPTURE = os.path.join(ROOT, "real_capture")
if CAPTURE not in sys.path:
    sys.path.insert(0, CAPTURE)

from camera_sources import create_camera_sources  # noqa: E402
from src.data.real.assemble_npz import build_real_npz  # noqa: E402
from src.data.real.triangulation import triangulate_skeletons_with_quality  # noqa: E402


def _projection_fixture(T=6, N=15):
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]])
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    t1 = np.array([[-0.12], [0.0], [0.0]])
    P1 = K @ np.hstack([np.eye(3), t1])
    Ps = np.stack([P0, P1]).astype(np.float32)
    states = []
    for t in range(T):
        s = np.linspace(0.0, 1.0, N)
        states.append(np.stack([
            0.03 * np.sin(s * np.pi + 0.1 * t),
            -0.15 + 0.30 * s,
            1.0 + 0.02 * np.cos(s * np.pi),
        ], axis=1))
    xyz = np.asarray(states, np.float32)
    homogeneous = np.concatenate([xyz, np.ones((T, N, 1), np.float32)], axis=2)
    pixels = []
    for P in Ps:
        q = np.einsum("ij,tnj->tni", P, homogeneous)
        pixels.append(q[..., :2] / q[..., 2:3])
    return xyz, np.asarray(pixels, np.float32), Ps


class CameraSourceTest(unittest.TestCase):
    def test_mixed_mock_rgbd_and_rgb_sources(self):
        cams = create_camera_sources("mock-depth,mock", width=64, height=48, fps=10)
        self.assertEqual(len(cams), 2)
        self.assertTrue(cams[0].has_depth)
        self.assertFalse(cams[1].has_depth)
        self.assertEqual(cams[0].source_metadata()["depth_scale_m"], 0.001)

    def test_opencv_source_does_not_claim_depth(self):
        cam = create_camera_sources("opencv:0")[0]
        self.assertEqual(cam.source_kind, "opencv")
        self.assertFalse(cam.has_depth)

    def test_multiple_realsense_sources_require_unique_serials(self):
        with self.assertRaisesRegex(ValueError, "唯一序列号"):
            create_camera_sources("realsense-depth:,realsense:")
        with self.assertRaisesRegex(ValueError, "不能重复"):
            create_camera_sources("realsense-depth:ABC,realsense:ABC")


class _FakeCam(QObject):
    frame_ready = pyqtSignal(np.ndarray, float)
    depth_ready = pyqtSignal(np.ndarray, float, float)

    def __init__(self, has_depth):
        super().__init__()
        self.has_depth = bool(has_depth)
        self.serial = None

    def source_metadata(self):
        return {"kind": "fake", "has_depth": self.has_depth,
                "depth_scale_m": 0.001 if self.has_depth else None}

    def stop(self):
        pass


class _FakeNdi(QObject):
    ndi_data = pyqtSignal(list, float)

    def stop(self):
        pass

    def wait(self, _timeout):
        pass


class RecorderSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_mixed_rgbd_rgb_raw_capture_is_separate_from_processing(self):
        from recorder import ValveRecorder
        from valve_control import MockValveController

        self.addCleanup(sys.modules.pop, "modbus_manager", None)
        self.addCleanup(sys.modules.pop, "valve_control", None)
        self.addCleanup(sys.modules.pop, "recorder", None)
        cams = [_FakeCam(True), _FakeCam(False)]
        controller = MockValveController()
        controller.connect_group(1); controller.connect_group(2)
        recorder = ValveRecorder(cams[0], _FakeNdi(), controller, cams=cams)
        with tempfile.TemporaryDirectory() as directory:
            recorder.start_recording(
                directory, "manual", [0] * 6, [0] * 6, 1.0, 0.1, 0,
                "schema-test", required_groups={1, 2})
            recorder._clock.stop()
            stamp = time.monotonic()
            recorder._on_depth(0, np.full((8, 8), 1000, np.uint16), stamp, 0.001)
            recorder._on_cam(0, np.zeros((8, 8, 3), np.uint8), stamp)
            recorder._on_cam(1, np.zeros((8, 8, 3), np.uint8), stamp)
            recorder._on_grab([0.0] * 6, "manual-0")
            recorder.stop_recording()
            recorder.save_thread.stop()
            self.assertTrue(os.path.isfile(os.path.join(directory, "cam0", "00000.png")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "cam1", "00000.png")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "depth0", "00000.png")))
            self.assertFalse(os.path.exists(os.path.join(directory, "depth1")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "camera_times.csv")))
            import json
            with open(os.path.join(directory, "meta.json"), encoding="utf-8") as stream:
                meta = json.load(stream)
            self.assertEqual(meta["schema_version"], 2)
            self.assertEqual([item["has_depth"] for item in meta["camera_sources"]],
                             [True, False])
        controller.close()


class TriangulationQualityTest(unittest.TestCase):
    def test_known_curve_round_trip(self):
        xyz, pixels, Ps = _projection_fixture(T=3, N=15)
        result = triangulate_skeletons_with_quality(
            pixels, np.zeros((2, 10)), 480, 640,
            max_reprojection_error_px=0.1, projection_matrices=Ps)
        np.testing.assert_allclose(result["positions_3d"], xyz, atol=2e-5)
        self.assertEqual(result["positions_2d"].shape, (3, 2, 15, 2))
        self.assertTrue(result["visibility"].all())
        self.assertTrue((result["position_confidence"] > 0).all())

    def test_bad_correspondence_is_rejected(self):
        _xyz, pixels, Ps = _projection_fixture(T=1, N=15)
        pixels[1, 0, 7] += np.array([80.0, 0.0])
        result = triangulate_skeletons_with_quality(
            pixels, np.zeros((2, 10)), 480, 640,
            max_reprojection_error_px=2.0, projection_matrices=Ps)
        self.assertEqual(result["source_mask"][0, 7], 0)
        self.assertEqual(result["position_confidence"][0, 7], 0.0)


class Real3DTrainingContractTest(unittest.TestCase):
    def _write_npz(self, directory):
        xyz, pixels_vt, Ps = _projection_fixture(T=6, N=15)
        pixels = np.transpose(pixels_vt, (1, 0, 2, 3))
        images = np.zeros((6, 2, 8, 8), np.float32)
        masks = np.zeros_like(images)
        data = build_real_npz(
            images, masks, xyz, np.zeros((6, 3), np.float32),
            np.zeros((2, 10), np.float32), 0.1, ["cam0", "cam1"],
            positions_2d=pixels, visibility=np.ones((6, 2, 15), bool),
            position_confidence=np.ones((6, 15), np.float32),
            source_mask=np.full((6, 15), 2, np.uint8),
            projection_matrices=Ps)
        path = os.path.join(directory, "sequence.npz")
        np.savez_compressed(path, **data)
        return path

    def test_assemble_uses_time_view_axis_and_15_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_npz(directory)
            with np.load(path) as data:
                self.assertEqual(data["positions"].shape, (6, 3, 15))
                self.assertEqual(data["images"].shape[:2], (6, 2))
                self.assertEqual(data["positions_2d"].shape, (6, 2, 15, 2))

    def test_masked_3d_and_reprojection_loss_backward(self):
        from src.data.dataset_spatial import StateTransitionDataset, spatial_collate_fn
        from src.models.model_open_loop_transition import OpenLoopTransitionModel
        from src.training.trainer_unified import UnifiedTrainer

        with tempfile.TemporaryDirectory() as directory:
            self._write_npz(directory)
            dataset = StateTransitionDataset(
                directory, seq_len=2, episode_mode=True, episode_len=2)
            batch = spatial_collate_fn([dataset[0]])
            model = OpenLoopTransitionModel(
                action_dim=3, n_nodes=15, hidden_dim=16, window_size=2,
                n_orders=2, encoder_type="ema", z_dim=4, episode_len=2)
            center, scale = dataset.get_normalization_params()
            model.set_normalization(center, scale, dataset.norm_factor)
            trainer = UnifiedTrainer(
                model, config={"loss_weights": {"skeleton_reprojection": 0.1}})
            trainer.device = torch.device("cpu")
            phase = SimpleNamespace(
                active_losses=["skeleton", "spatial_smooth", "skeleton_reprojection"],
                dense_step_weight="uniform", teacher_forcing_ratio=0.0,
                tf_anneal_epochs=0)
            losses = trainer._compute_sequence_losses(batch, phase)
            self.assertTrue(torch.isfinite(losses["total"]))
            self.assertIn("skeleton_reprojection", losses)

            masked = {key: (value.clone() if torch.is_tensor(value) else value)
                      for key, value in batch.items()}
            masked["position_confidence"][:, :, 7] = 0.0
            masked_losses = trainer._compute_sequence_losses(masked, phase)
            masked["gt_skeletons"][:, :, 7] += 100.0
            altered_losses = trainer._compute_sequence_losses(masked, phase)
            torch.testing.assert_close(
                masked_losses["skeleton"], altered_losses["skeleton"])
            torch.testing.assert_close(
                masked_losses["spatial_smooth"], altered_losses["spatial_smooth"])

            losses["total"].backward()
            self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
