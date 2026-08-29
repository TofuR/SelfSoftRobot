import csv
import os
import tempfile
import unittest

import cv2
import numpy as np


class GeneralCenterlineTest(unittest.TestCase):
    def _s_mask(self):
        mask = np.zeros((180, 180), np.uint8)
        points = np.array([[90, 0], [90, 35], [65, 70], [110, 110], [90, 150]],
                          np.int32)
        cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, 19, cv2.LINE_AA)
        return (mask > 0).astype(np.uint8)

    def test_two_equal_segments_share_node7(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        skeleton, info = extract_centerline_2d(
            self._s_mask(), n_points=15, method="skeletonize",
            segment_lengths=(1, 1), return_info=True)
        self.assertEqual(skeleton.shape, (15, 2))
        self.assertTrue(info["success"])
        self.assertEqual(info["segment_intervals"], (7, 7))
        self.assertEqual(info["joint_node_indices"], (7,))
        self.assertLess(skeleton[0, 1], skeleton[-1, 1])

    def test_explicit_base_anchor_controls_direction(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        skeleton = extract_centerline_2d(
            self._s_mask(), n_points=15, method="skeletonize",
            segment_lengths=(1, 1), base_anchor_xy=(90, 175))
        self.assertGreater(skeleton[0, 1], skeleton[-1, 1])
        self.assertTrue(np.allclose(skeleton[0], [90, 175]))

    def test_base_anchor_excludes_attachment_branch_from_main_path(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        mask = np.zeros((180, 180), np.uint8)
        cv2.rectangle(mask, (80, 20), (100, 160), 1, -1)
        cv2.rectangle(mask, (100, 20), (155, 40), 1, -1)
        anchor = (90, 30)
        skeleton = extract_centerline_2d(
            mask, n_points=15, method="skeletonize",
            base_anchor_xy=anchor, endpoint_fix=True)
        self.assertLess(np.linalg.norm(skeleton[0] - anchor), 8.0)
        self.assertGreater(skeleton[-1, 1], 145)
        self.assertLess(
            np.linalg.norm(np.diff(skeleton, axis=0), axis=1).sum(), 150)

    def test_endpoint_fix_reaches_both_flat_cap_centers(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        mask = np.zeros((180, 180), np.uint8)
        cv2.rectangle(mask, (70, 20), (110, 160), 1, -1)
        raw = extract_centerline_2d(mask, n_points=15, endpoint_fix=False)
        fixed, info = extract_centerline_2d(
            mask, n_points=15, endpoint_fix=True, return_info=True)
        self.assertGreater(raw[0, 1] - 20, 10)
        self.assertGreater(160 - raw[-1, 1], 10)
        self.assertTrue(np.allclose(fixed[0], [90, 20], atol=1.0))
        self.assertTrue(np.allclose(fixed[-1], [90, 160], atol=1.0))
        self.assertTrue(info["tip_endpoint_fix_applied"])
        self.assertTrue(info["base_endpoint_fix_applied"])
        self.assertEqual(info["segment_intervals"], (7, 7))
        self.assertEqual(info["joint_node_indices"], (7,))

    def test_rotated_caps_use_wide_edge_centers_not_corners(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        box = cv2.boxPoints(((100, 100), (36, 130), 28)).astype(np.int32)
        mask = np.zeros((220, 220), np.uint8)
        cv2.fillConvexPoly(mask, box, 1)
        raw = extract_centerline_2d(mask, n_points=15, endpoint_fix=False)
        fixed = extract_centerline_2d(mask, n_points=15, endpoint_fix=True)
        cap_centers = np.asarray([(box[0] + box[3]) / 2,
                                  (box[1] + box[2]) / 2], dtype=float)
        for endpoint in fixed[[0, -1]]:
            self.assertLess(np.linalg.norm(cap_centers - endpoint, axis=1).min(), 2.0)
        raw_error = sum(np.linalg.norm(cap_centers - endpoint, axis=1).min()
                        for endpoint in raw[[0, -1]])
        fixed_error = sum(np.linalg.norm(cap_centers - endpoint, axis=1).min()
                          for endpoint in fixed[[0, -1]])
        self.assertLess(fixed_error, 0.15 * raw_error)

    def test_legacy_row_centroid_tip_fix_is_unchanged(self):
        from real_validation.perception.skeleton import extract_centerline_2d

        with_endpoint_flag, info = extract_centerline_2d(
            self._s_mask(), n_points=15, method="row_centroid", tip_fix=True,
            endpoint_fix=True, return_info=True)
        without_endpoint_flag = extract_centerline_2d(
            self._s_mask(), n_points=15, method="row_centroid", tip_fix=True,
            endpoint_fix=False)
        self.assertTrue(np.array_equal(with_endpoint_flag, without_endpoint_flag))
        self.assertEqual(info["reason"], "applied")

    def test_cropped_masks_restore_source_camera_coordinates(self):
        from scripts.real.masks_to_transition_npz import masks_to_positions

        with tempfile.TemporaryDirectory() as root:
            cv2.imwrite(os.path.join(root, "00000.png"), self._s_mask() * 255)
            local, _, local_qc = masks_to_positions(
                root, n_points=15, skeleton_method="skeletonize",
                segment_lengths=(1, 1), crop_offset_xy=(0, 0), return_qc=True)
            source, _, source_qc = masks_to_positions(
                root, n_points=15, skeleton_method="skeletonize",
                segment_lengths=(1, 1), crop_offset_xy=(300, 128), return_qc=True)
        self.assertTrue(np.allclose(source[:, 0, :], local[:, 0, :] + 300))
        self.assertTrue(np.allclose(source[:, 1, :], local[:, 1, :] + 128))
        self.assertTrue(np.allclose(
            source_qc[0]["fixed_tip_xy"],
            np.asarray(local_qc[0]["fixed_tip_xy"]) + [300, 128]))

    def test_explicit_repairs_use_real_frame_ids(self):
        from scripts.real.masks_to_transition_npz import frame_ids_to_mask

        qc = [{"frame": 101}, {"frame": 205}, {"frame": 999}]
        selected = frame_ids_to_mask(qc, (205, 205))
        self.assertEqual(selected.tolist(), [False, True, False])
        with self.assertRaisesRegex(ValueError, "不存在的frame ID"):
            frame_ids_to_mask(qc, (1,))

    def test_centerline_mask_close_fills_narrow_occlusion_slit(self):
        from scripts.real.masks_to_transition_npz import prepare_centerline_mask

        mask = np.zeros((80, 60), np.uint8)
        mask[5:75, 20:40] = 1
        mask[35:75, 29:32] = 0
        closed = prepare_centerline_mask(mask, close_kernel=5)
        self.assertTrue(closed[50, 30])
        self.assertGreater(int(closed.sum()), int(mask.sum()))
        self.assertTrue(np.array_equal(
            prepare_centerline_mask(mask, close_kernel=0), mask))
        with self.assertRaisesRegex(ValueError, "正奇数"):
            prepare_centerline_mask(mask, close_kernel=4)


class CandidateSegmentationTest(unittest.TestCase):
    def test_wide_base_attachment_is_trimmed_without_shortening_body(self):
        from scripts.real.prepare_sam2_anchors import trim_wide_base_attachment

        mask = np.zeros((100, 80), np.uint8)
        mask[10:80, 31:49] = 1
        mask[10:20, 20:60] = 1  # base处误粘的横向支架
        trimmed = trim_wide_base_attachment(
            mask, base_side="top", width_ratio=1.5, stable_span=5)
        self.assertFalse(trimmed[:20].any())
        self.assertTrue(trimmed[20:80, 31:49].all())
        self.assertEqual(int(trimmed.sum()), 60 * 18)

    def test_base_trim_keeps_already_narrow_cap(self):
        from scripts.real.prepare_sam2_anchors import trim_wide_base_attachment

        mask = np.zeros((100, 80), np.uint8)
        mask[10:80, 31:49] = 1
        trimmed = trim_wide_base_attachment(mask, base_side="top")
        self.assertTrue(np.array_equal(trimmed, mask))

    def test_staged_and_compatibility_outputs_match(self):
        from real_validation.perception.segmentation import (
            segment_white_on_blue, segment_white_on_blue_stages,
        )

        image = np.zeros((100, 100, 3), np.uint8)
        image[:] = (130, 70, 20)
        cv2.rectangle(image, (43, 0), (57, 80), (235, 235, 235), -1)
        bg = np.full((100, 100), 70, np.uint8)
        params = dict(sat=100, val=120, diff=25, dil=9, open_k=3, close_k=5,
                      min_area_frac=.001, min_h_frac=.1)
        stages = segment_white_on_blue_stages(image, bg, **params)
        self.assertEqual(set(stages), {"white", "moved", "gated", "morph", "final"})
        self.assertTrue(np.array_equal(
            stages["final"], segment_white_on_blue(image, bg, **params)))

    def test_manifest_preselection_wins(self):
        from sam2.segment_video_full import load_anchor_manifest, select_anchor

        with tempfile.TemporaryDirectory() as root:
            for frame, width in ((0, 12), (1, 16), (2, 14)):
                mask = np.zeros((80, 60), np.uint8)
                mask[:60, 20:20 + width] = 255
                cv2.imwrite(os.path.join(root, f"{frame:05d}.png"), mask)
            manifest = os.path.join(root, "anchor_manifest.csv")
            with open(manifest, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=("frame", "quality", "selected"))
                writer.writeheader()
                writer.writerows([
                    {"frame": 0, "quality": .9, "selected": 0},
                    {"frame": 1, "quality": .2, "selected": 1},
                    {"frame": 2, "quality": .8, "selected": 0},
                ])
            frame, _ = select_anchor(root, [0, 1, 2], 800,
                                     anchor_manifest=load_anchor_manifest(manifest))
            self.assertEqual(frame, 1)


class EvaluationContractTest(unittest.TestCase):
    def test_new_gui_ndi_schema_and_quality_filter(self):
        from scripts.evaluation.eval_real_quant import load_ndi_tip

        with tempfile.TemporaryDirectory() as root:
            ndi_path = os.path.join(root, "ndi.csv")
            frame_path = os.path.join(root, "frame_times.txt")
            with open(ndi_path, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t_sec", "ndi0_x", "ndi0_y", "ndi0_z", "ndi0_quality"])
                writer.writerow([0.0, 0.0, 1.0, 2.0, 0.9])
                writer.writerow([0.5, 99.0, 99.0, 99.0, 0.1])
                writer.writerow([1.0, 10.0, 11.0, 12.0, 0.9])
            np.savetxt(frame_path, np.array([0.0, 0.5, 1.0]))
            aligned = load_ndi_tip(ndi_path, frame_path, ndi_index=0, min_quality=0.5)
            self.assertTrue(np.allclose(aligned[1], [5.0, 6.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
