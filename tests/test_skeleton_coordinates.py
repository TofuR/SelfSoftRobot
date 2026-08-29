import unittest
import tempfile
from pathlib import Path

import numpy as np
import torch

from real_validation.perception.coordinates import (
    ROBOT_PLANAR_FRAME,
    SkeletonFrameTransform,
    estimate_skeleton_frame,
    transform_positions_to_model,
)
from scripts.real.recommend_robot_roi import recommend_square_roi
from real_validation.runtime.live_anchor import anchor_from_camera_frame
from real_validation.runtime.model_runtime import certified_k_safe
from src.data.dataset_spatial import SpatialSequenceDataset


class SkeletonFrameTransformTests(unittest.TestCase):
    def test_roundtrip_is_subpixel_exact(self):
        transform = SkeletonFrameTransform(
            origin_camera_px=(321.4, 73.2),
            axial_axis_camera=(0.3, 0.953939201),
            pixels_per_mm=2.75,
            source="test")
        camera = np.array([[321.4, 73.2], [350.0, 180.0], [280.0, 250.0]])
        restored = transform.model_to_camera(transform.camera_to_model(camera))
        np.testing.assert_allclose(restored, camera, atol=2e-5)

    def test_camera_similarity_changes_produce_same_robot_state(self):
        robot = np.array([
            [0.0, 0.0], [-2.0, 35.0], [-6.0, 70.0],
            [-12.0, 100.0], [-18.0, 130.0],
        ], dtype=np.float32)
        views = [
            SkeletonFrameTransform((300.0, 60.0), (0.0, 1.0), 2.0, "view_a"),
            SkeletonFrameTransform((480.0, 220.0), (0.6, 0.8), 3.5, "view_b"),
        ]
        recovered = []
        for view in views:
            camera = view.model_to_camera(robot)
            estimated = estimate_skeleton_frame(
                camera, robot_diameter_mm=16.0,
                robot_diameter_px=16.0 * view.pixels_per_mm,
                source="estimated")
            recovered.append(estimated.camera_to_model(camera))
        np.testing.assert_allclose(recovered[0], recovered[1], atol=2e-4)

    def test_sequence_uses_one_fixed_frame_and_keeps_motion(self):
        transform = SkeletonFrameTransform(
            (250.0, 40.0), (0.0, 1.0), 2.0, "capture")
        first = np.array([
            [0.0, 0.0], [0.0, 30.0], [0.0, 60.0],
            [0.0, 90.0], [0.0, 120.0],
        ])
        second = first.copy()
        second[-1, 0] = 24.0
        second[-2, 0] = 12.0
        camera = np.stack((transform.model_to_camera(first),
                           transform.model_to_camera(second)))
        positions = np.zeros((2, 3, 5), dtype=np.float32)
        positions[:, :2, :] = camera.transpose(0, 2, 1)
        estimated = estimate_skeleton_frame(
            positions, robot_diameter_mm=16.0, robot_diameter_px=32.0)
        model = transform_positions_to_model(positions, estimated)
        lateral_motion = model[1, 0, -1] - model[0, 0, -1]
        self.assertAlmostEqual(float(lateral_motion), 24.0, places=4)
        self.assertEqual(estimated.frame_id, ROBOT_PLANAR_FRAME)

    def test_roi_uses_physical_padding_and_stays_in_source_image(self):
        bounds = [(280, 30, 410, 300), (260, 35, 420, 310)]
        roi = recommend_square_roi(
            bounds, (640, 480), diameter_px=20.0,
            padding_diameters=3.0, quantile=0.0)
        x, y, width, height = roi
        self.assertEqual(width, height)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 640)
        self.assertLessEqual(y + height, 480)
        self.assertLessEqual(x, 260)
        self.assertGreaterEqual(x + width, 420)

    def test_live_anchor_records_robot_frame_transform(self):
        image = np.full((160, 120, 3), 255, dtype=np.uint8)
        image[4:145, 50:70] = 0

        class Model:
            action_dim = 4
            history_steps = 3
            pc_center = torch.tensor([[[0.0, 70.0, 0.0]]])
            pc_scale = torch.tensor([[[50.0, 70.0, 1.0]]])

        anchor, quality, skeleton = anchor_from_camera_frame(
            image, background_gray=None, segment_params={"thresh": 60},
            n_nodes=15, model=Model(), action_history=[],
            area_median_px=20 * 141, zero_pad_history=True,
            segmentation_method="backlight",
            state_coordinate_frame=ROBOT_PLANAR_FRAME,
            robot_diameter_mm=16.0)
        self.assertIsNotNone(anchor, quality.reasons)
        self.assertEqual(anchor.quality["state_coordinate_frame"], ROBOT_PLANAR_FRAME)
        transform = SkeletonFrameTransform.from_dict(
            anchor.quality["skeleton_frame_transform"])
        model_nodes = transform.camera_to_model(skeleton)
        np.testing.assert_allclose(model_nodes[0], (0.0, 0.0), atol=1e-4)

    def test_mm_horizon_table_uses_tightest_tolerance(self):
        self.assertEqual(certified_k_safe({"2mm": 7, "4mm": 15, "8mm": 32}), 7)

    def test_dataset_rejects_mixed_state_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            positions = np.zeros((4, 3, 5), dtype=np.float32)
            actions = np.zeros((4, 2), dtype=np.float32)
            np.savez(root / "a.npz", positions=positions, actions=actions,
                     state_coordinate_frame=np.array("camera_pixel_v1"),
                     state_length_unit=np.array("px"))
            np.savez(root / "b.npz", positions=positions, actions=actions,
                     state_coordinate_frame=np.array(ROBOT_PLANAR_FRAME),
                     state_length_unit=np.array("mm"))
            with self.assertRaisesRegex(ValueError, "混用了多个状态坐标合同"):
                SpatialSequenceDataset(str(root), seq_len=2)


if __name__ == "__main__":
    unittest.main()
