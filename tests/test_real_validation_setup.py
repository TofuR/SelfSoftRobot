"""验证实验配置、ROI 坐标与在线 Anchor 的聚焦测试。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from real_validation.contracts.models import Anchor
from real_validation.contracts.validation_setup import ValidationSetup
from real_validation.core.session import ExperimentSession
from real_validation.perception.coordinates import ROBOT_PLANAR_FRAME
from real_validation.perception.roi import (
    camera_to_roi_local,
    clamp_roi_xywh,
    roi_local_to_camera,
    suggest_square_roi,
)


def _setup(**changes):
    values = dict(
        camera_backend="real", camera_index=0, camera_identity="camera-A",
        frame_size_wh=(640, 480), roi_xywh=(160, 40, 400, 400),
        segmentation_method="backlight", segment_params={"thresh": 60},
        model_checkpoint_hash="model-A",
        state_coordinate_frame=ROBOT_PLANAR_FRAME, state_length_unit="mm",
        robot_diameter_mm=16.0,
    )
    values.update(changes)
    return ValidationSetup(**values)


class RoiCoordinateTest(unittest.TestCase):
    def test_local_camera_roundtrip(self):
        roi = (137, 42, 320, 320)
        local = np.asarray([[0.0, 0.0], [20.5, 80.25], [319.0, 319.0]])
        camera = roi_local_to_camera(local, roi)
        np.testing.assert_allclose(camera_to_roi_local(camera, roi), local)

    def test_clamp_keeps_square_inside_frame(self):
        self.assertEqual(
            clamp_roi_xywh((-20, 390, 300, 200), (640, 480), square=True),
            (0, 180, 300, 300))

    def test_mask_candidate_restores_source_offset(self):
        mask = np.zeros((100, 120), np.uint8)
        mask[20:80, 40:70] = 1
        roi = suggest_square_roi(
            mask, source_offset_xy=(200, 50), source_size_wh=(640, 480),
            padding_px=20)
        x, y, width, height = roi
        self.assertEqual(width, height)
        self.assertLessEqual(x, 240)
        self.assertGreaterEqual(x + width, 270)
        self.assertLessEqual(y, 70)
        self.assertGreaterEqual(y + height, 130)

    def test_base_attachment_trim_keeps_robot_body(self):
        from real_validation.perception.segmentation import trim_wide_base_attachment
        mask = np.zeros((100, 120), np.uint8)
        mask[:6, 20:100] = 1
        mask[6:90, 55:65] = 1
        trimmed = trim_wide_base_attachment(mask, base_side="top")
        self.assertEqual(int(trimmed[:6].sum()), 0)
        self.assertGreater(int(trimmed[6:].sum()), 0)


class ValidationSetupSessionTest(unittest.TestCase):
    def test_setup_is_persisted_and_change_invalidates_live_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            setup = _setup()
            session.set_validation_setup(setup)
            anchor = Anchor(
                state=((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)),
                action_history=((0.0,),),
                quality={"kind": "camera_live",
                         "validation_setup_id": setup.setup_id})
            session.set_anchor(anchor)
            changed = setup.updated(roi_xywh=(120, 40, 400, 400))
            session.set_validation_setup(changed)
            self.assertIsNone(session.anchor)
            replay = ExperimentSession.load_for_replay(session.run_dir)
            self.assertEqual(replay.validation_setup.roi_xywh,
                             changed.roi_xywh)

    def test_live_planning_requires_matching_setup_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = ExperimentSession.create(temporary)
            setup = _setup()
            session.set_validation_setup(setup)
            session.model = type("Model", (), {})()  # begin_planning 只检查存在性
            session.anchor = Anchor(
                state=((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)),
                action_history=((0.0,),),
                quality={"kind": "camera_live",
                         "validation_setup_id": "another-setup"})
            with self.assertRaisesRegex(RuntimeError, "ROI/感知配置已变化"):
                session.begin_planning()


class LiveAnchorRoiTest(unittest.TestCase):
    def test_roi_skeleton_returns_source_camera_pixels(self):
        import torch
        from real_validation.runtime.live_anchor import anchor_from_camera_frame

        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        frame[34:205, 150:170] = 0

        class Model:
            action_dim = 4
            history_steps = 3
            pc_center = torch.tensor([[[0.0, 80.0, 0.0]]])
            pc_scale = torch.tensor([[[50.0, 80.0, 1.0]]])

        anchor, quality, skeleton = anchor_from_camera_frame(
            frame, background_gray=None, segment_params={"thresh": 60},
            n_nodes=15, model=Model(), action_history=[],
            area_median_px=None, zero_pad_history=True,
            segmentation_method="backlight",
            state_coordinate_frame=ROBOT_PLANAR_FRAME,
            robot_diameter_mm=16.0,
            roi_xywh=(120, 20, 100, 200), skeleton_method="skeletonize",
            segment_lengths=(1.0, 1.0),
            quality_params={"max_top_row": 40},
            validation_setup_id="setup-A")
        self.assertIsNotNone(anchor, quality.reasons)
        self.assertGreater(float(skeleton[:, 0].min()), 120.0)
        self.assertGreater(float(skeleton[:, 1].min()), 20.0)
        self.assertEqual(anchor.quality["roi_xywh"], [120, 20, 100, 200])
        self.assertEqual(anchor.quality["validation_setup_id"], "setup-A")

    def test_observation_artifacts_include_roi_and_segmentation_stage(self):
        from real_validation.runtime.online_perception import (
            extract_skeleton_with_setup, save_observation_artifacts)
        frame = np.full((240, 320, 3), 255, dtype=np.uint8)
        frame[20:220, 150:170] = 0
        setup = _setup(
            camera_backend="mock", camera_identity="mock:cam0",
            frame_size_wh=(320, 240), roi_xywh=(40, 0, 240, 240),
            state_coordinate_frame="camera_pixel_v1", state_length_unit="px")
        skeleton, _mask, _info = extract_skeleton_with_setup(
            frame, setup, n_nodes=15)
        with tempfile.TemporaryDirectory() as temporary:
            output = save_observation_artifacts(
                temporary, "anchor_test", frame, setup, skeleton,
                SimpleNamespace(verdict="ok", reasons=(), flags={}))
            self.assertTrue((output / "camera_roi_overlay.png").is_file())
            self.assertTrue(
                (output / "segmentation_stages" / "final.png").is_file())


if __name__ == "__main__":
    unittest.main()
