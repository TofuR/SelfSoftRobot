import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from scripts.real.manage_training_trial import (
    prepare_trial_layout,
    validate_dataset_manifest,
    validate_open_loop_start,
)
from scripts.real.masks_to_transition_npz import save_npz
from scripts.real.preprocess_capture import (
    build_parser,
    resolve_pipeline_args,
    validate_stage_dependencies,
)
from scripts.real.save_preprocess_stage_example import save_stage_example


class RealPipelineAutomationTest(unittest.TestCase):
    def test_json_config_and_cli_override(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "sequence.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({
                    "schema_version": 1,
                    "seq": "seq_demo",
                    "roi": [10, 20, 256, 256],
                    "gpus": [1, 3],
                    "segment_lengths": [1, 1],
                    "base_anchor": [128, 22],
                    "anchor": {"sat": 88, "close_k": 13},
                }, stream)
            parsed = build_parser().parse_args(
                ["--config", path, "--n-points", "17"])
            resolved = resolve_pipeline_args(parsed)

        self.assertEqual(resolved.seq, "seq_demo")
        self.assertEqual(resolved.roi, (10, 20, 256, 256))
        self.assertEqual(resolved.gpus, "1,3")
        self.assertEqual(resolved.n_points, 17)
        self.assertEqual(resolved.segment_lengths, "1.0,1.0")
        self.assertEqual(resolved.base_anchor, "128.0,22.0")
        self.assertEqual(resolved.anchor["sat"], 88)
        self.assertEqual(resolved.anchor["close_k"], 13)
        self.assertEqual(resolved.anchor["diff"], 25)

    def test_training_manifest_gate(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "dataset_manifest.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump({
                    "dataset_id": "seq_demo_n15",
                    "quality_control": {
                        "automated_checks_passed": True,
                        "training_ready": True,
                        "checks": [{"name": "frame_coverage", "passed": True}],
                    },
                }, stream)
            result = validate_dataset_manifest(path)
            self.assertTrue(result["training_ready"])

            with open(path, "w", encoding="utf-8") as stream:
                json.dump({
                    "quality_control": {
                        "automated_checks_passed": False,
                        "training_ready": False,
                        "checks": [{"name": "frame_coverage", "passed": False}],
                    },
                }, stream)
            with self.assertRaisesRegex(ValueError, "frame_coverage"):
                validate_dataset_manifest(path)

    def test_training_trial_layout_and_open_loop_start_contract(self):
        with tempfile.TemporaryDirectory() as root:
            trial = os.path.join(root, "trial_001")
            train_dir = os.path.join(root, "data", "train")
            val_dir = os.path.join(root, "data", "val")
            os.makedirs(os.path.join(
                trial, "stages", "gt", "phase_gt_transition", "model"),
                exist_ok=True)
            os.makedirs(train_dir)
            os.makedirs(val_dir)
            with open(os.path.join(trial, "config.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"data": {"train_dir": train_dir,
                                     "val_dir": val_dir}}, stream)
            with open(os.path.join(
                    trial, "stages", "gt", "phase_gt_transition", "model",
                    "best_model.pt"), "wb") as stream:
                stream.write(b"checkpoint")

            prepare_trial_layout(trial)
            result = validate_open_loop_start(trial, train_dir, val_dir)

            self.assertEqual(result["trial_dir"], trial)
            for relative in (
                    "evaluations/gt/best/quantitative",
                    "evaluations/gt/best/overlay",
                    "evaluations/open_loop/best/quantitative",
                    "evaluations/open_loop/best/overlay"):
                self.assertTrue(os.path.isdir(os.path.join(trial, relative)))
            with self.assertRaisesRegex(ValueError, "数据目录"):
                validate_open_loop_start(
                    trial, os.path.join(root, "other"), val_dir)

            open_loop = os.path.join(trial, "stages", "open_loop")
            os.makedirs(open_loop, exist_ok=True)
            with open(os.path.join(open_loop, "config.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({}, stream)
            with self.assertRaisesRegex(FileExistsError, "已有训练产物"):
                validate_open_loop_start(trial, train_dir, val_dir)

    def test_rebuilt_upstream_stage_requires_downstream_refresh(self):
        validate_stage_dependencies(("skeleton",))
        validate_stage_dependencies(("audit",))
        validate_stage_dependencies(("crop", "anchors", "sam2", "skeleton"))
        with self.assertRaisesRegex(ValueError, "下游阶段"):
            validate_stage_dependencies(("sam2",))

    def test_npz_records_raw_and_model_action_scales(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "train", "seq_train.npz")
            save_npz(
                path,
                np.zeros((2, 3, 15), dtype=np.float32),
                np.zeros((2, 6), dtype=np.float32),
                n_points=15,
                model_action_channels=(0, 1, 3, 5),
                raw_action_scale6_kpa=(150, 150, 150, 150, 150, 150),
                action_scale_kpa=(150, 150, 150, 150),
            )
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["raw_action_scale6_kpa"].shape, (6,))
                self.assertEqual(data["action_scale_kpa"].shape, (4,))

    def test_single_frame_stage_example_saves_all_pipeline_views(self):
        with tempfile.TemporaryDirectory() as root:
            seq = os.path.join(root, "raw", "seq_demo")
            derived = os.path.join(root, "derived", "seq_demo")
            crop_root = os.path.join(derived, "crop")
            crop_dir = os.path.join(crop_root, "cam0")
            candidate_dir = os.path.join(derived, "masks_candidate")
            masks_dir = os.path.join(root, "sam2_masks")
            dataset = os.path.join(root, "dataset")
            for path in (os.path.join(seq, "cam0"), crop_dir, candidate_dir,
                         masks_dir, os.path.join(dataset, "train"),
                         os.path.join(dataset, "qc_skeleton")):
                os.makedirs(path, exist_ok=True)

            crop = np.zeros((100, 100, 3), np.uint8)
            centerline = np.asarray(((50, 5), (51, 30), (55, 55), (62, 90)),
                                    dtype=np.int32)
            cv2.polylines(crop, [centerline.reshape(-1, 1, 2)], False,
                          (245, 245, 245), 15, cv2.LINE_AA)
            raw = np.zeros((120, 160, 3), np.uint8)
            raw[10:110, 30:130] = crop
            mask = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8)
            cv2.imwrite(os.path.join(seq, "cam0", "00000.png"), raw)
            cv2.imwrite(os.path.join(crop_dir, "00000.png"), crop)
            cv2.imwrite(os.path.join(candidate_dir, "00000.png"), mask * 255)
            cv2.imwrite(os.path.join(masks_dir, "00000.png"), mask * 255)
            cv2.imwrite(os.path.join(derived, "bg_median.png"),
                        np.zeros((100, 100), np.uint8))

            with open(os.path.join(crop_root, "crop_meta.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({"crop_xywh": [30, 10, 100, 100]}, stream)
            with open(os.path.join(derived, "candidate_summary.json"), "w",
                      encoding="utf-8") as stream:
                json.dump({
                    "chunk_size": 200,
                    "base_side": "top",
                    "base_attachment_trim": {"width_ratio": 1.5,
                                               "stable_span": 3},
                    "segmentation_params": {
                        "sat": 100, "val": 120, "diff": 25, "dil": 3,
                        "open_k": 3, "close_k": 3,
                        "min_area_frac": 0.001, "min_h_frac": 0.1,
                    },
                }, stream)
            with open(os.path.join(derived, "anchor_manifest.csv"), "w",
                      encoding="utf-8") as stream:
                stream.write("frame,quality,selected\n0,0.9,1\n")
            with open(os.path.join(dataset, "qc_skeleton", "skeleton_metrics.csv"),
                      "w", encoding="utf-8") as stream:
                stream.write("index,frame,success\n0,0,True\n")

            nodes_local = np.column_stack((np.linspace(62, 50, 15),
                                           np.linspace(90, 5, 15))).astype(np.float32)
            camera_nodes = nodes_local + np.asarray((30, 10), np.float32)
            positions_camera = np.zeros((1, 3, 15), np.float32)
            positions_camera[0, :2] = camera_nodes.T
            positions_mm = np.zeros((1, 3, 15), np.float32)
            positions_mm[0, 0] = np.linspace(2, 0, 15)
            positions_mm[0, 1] = np.linspace(90, 0, 15)
            np.savez_compressed(
                os.path.join(dataset, "train", "seq_demo_train.npz"),
                positions=positions_mm,
                positions_camera_px=positions_camera,
                actions=np.zeros((1, 6), np.float32),
                state_coordinate_frame=np.array("robot_planar_mm_v1"),
                state_length_unit=np.array("mm"),
                model_action_channels=np.asarray((0, 1, 3, 5), np.int64),
            )

            result = save_stage_example(
                seq=seq, camera="cam0", derived=derived,
                masks_dir=masks_dir, dataset_root=dataset,
                mask_close_k=5, n_points=15, segment_lengths=(1, 1),
                base_anchor_source=(80, 15))
            output = os.path.join(derived, "qc_pipeline_example")
            self.assertEqual(result["selection"], "best_selected_anchor")
            self.assertEqual(result["sam2_context"]["direction"], "anchor prompt")
            self.assertEqual(len(result["stages"]), 17)
            self.assertTrue(os.path.isfile(os.path.join(
                output, "00_pipeline_overview.png")))
            self.assertTrue(os.path.isfile(os.path.join(
                output, "17_model_training_sample.png")))
            self.assertTrue(os.path.isfile(os.path.join(output, "README.md")))


if __name__ == "__main__":
    unittest.main()
