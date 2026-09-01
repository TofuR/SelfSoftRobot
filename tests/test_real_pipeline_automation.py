import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.real.manage_training_trial import (
    infer_sequence_tag,
    prepare_trial_layout,
    resolve_real_pipeline_paths,
    validate_dataset_manifest,
    validate_open_loop_start,
)
from scripts.real.masks_to_transition_npz import save_npz
from scripts.real.preprocess_capture import (
    build_dataset_manifest,
    build_parser,
    resolve_capture_sequence,
    resolve_preprocess_layout,
    resolve_pipeline_args,
    validate_stage_dependencies,
)
from src.registry.manifests import (
    validate_dataset_manifest as validate_registry_dataset_manifest,
)
from src.registry.paths import ProjectPaths
from src.registry.real_assets import SAM2_VIDEO_RECIPE
from scripts.real.save_preprocess_stage_example import save_stage_example


class RealPipelineAutomationTest(unittest.TestCase):
    def test_preprocess_reads_legacy_raw_and_writes_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            (repo / "config").mkdir(parents=True)
            (repo / "config/paths.local.toml").write_text(
                "schema_version = 1\n"
                "[paths]\nworkspace_root = '../large_workspace'\n"
                "[compat]\nraw_roots = ['old_raw']\n",
                encoding="utf-8")
            legacy = repo / "old_raw/seq_demo/cam0"
            legacy.mkdir(parents=True)
            paths = ProjectPaths.load(repo_root=repo, environ={})
            args = build_parser().parse_args([
                "--seq", "seq_demo", "--roi", "0,0,10,10"])
            resolved = resolve_pipeline_args(args)

            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                layout = resolve_preprocess_layout(resolved, paths)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(layout["seq"], legacy.parent)
            expected_workspace = Path(root) / "large_workspace"
            self.assertEqual(
                layout["derived"],
                expected_workspace /
                "data/intermediate/real/seq_demo/seq_demo_n15_sam2_robot_mm")
            self.assertEqual(
                layout["out_root"],
                expected_workspace /
                "data/processed/real/seq_demo_n15_sam2_robot_mm")
            self.assertEqual(layout["mask_dir"],
                             layout["derived"] / "sam2_masks")

    def test_preprocess_prefers_canonical_raw_and_accepts_explicit_legacy(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            paths = ProjectPaths.load(repo_root=repo, environ={})
            canonical = paths.raw_sequence("real", "seq_demo")
            legacy = repo / "real_capture/data/raw/seq_demo"
            for sequence in (canonical, legacy):
                (sequence / "cam0").mkdir(parents=True)

            self.assertEqual(
                resolve_capture_sequence("seq_demo", "cam0", paths), canonical)
            self.assertEqual(
                resolve_capture_sequence(str(legacy), "cam0", paths), legacy)

    def test_preprocess_rejects_new_processed_output_in_legacy_root(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            (repo / "raw/seq_demo/cam0").mkdir(parents=True)
            paths = ProjectPaths.load(repo_root=repo, environ={})
            args = resolve_pipeline_args(build_parser().parse_args([
                "--seq", str(repo / "raw/seq_demo"),
                "--roi", "0,0,10,10",
                "--out-root", "data/real_seq/legacy_write",
            ]))
            with self.assertRaisesRegex(ValueError, "workspace"):
                resolve_preprocess_layout(args, paths)

    def test_dataset_manifest_is_portable_and_registry_valid(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            paths = ProjectPaths.load(repo_root=repo, environ={})
            seq = repo / "legacy_raw/seq_demo"
            derived = paths.intermediate_sequence(
                "real", "seq_demo", "recipe_001")
            crop_root = derived / "crop"
            mask_dir = derived / "sam2_masks"
            out_root = paths.processed_dataset("real", "dataset_demo")
            for directory in (
                    seq / "cam0", crop_root / "cam0", mask_dir,
                    derived / "qc_capture", out_root / "train",
                    out_root / "val", out_root / "qc_skeleton"):
                directory.mkdir(parents=True)

            (derived / "qc_capture/capture_audit.json").write_text(
                json.dumps({"ready_for_image_preprocessing": True,
                            "issues": [], "counts": {"images": 2}}))
            (crop_root / "crop_meta.json").write_text(json.dumps({
                "complete": True, "n_output_frames": 2,
                "n_source_frames": 2,
            }))
            (derived / "candidate_summary.json").write_text(json.dumps({
                "n_frames": 2, "n_empty": 0, "n_selected_anchors": 1,
            }))
            (out_root / "qc_skeleton/skeleton_metrics.csv").write_text(
                "success,hard_invalid,interpolated,suspicious,explicit_repair\n"
                "True,False,False,False,False\n"
                "True,False,False,False,False\n")
            np.savetxt(seq / "frame_times.txt", np.asarray((0.0, 0.2)))
            image = np.zeros((4, 4, 3), np.uint8)
            mask = np.zeros((4, 4), np.uint8)
            for frame in range(2):
                cv2.imwrite(str(crop_root / "cam0" / f"{frame:05d}.png"), image)
                cv2.imwrite(str(mask_dir / f"{frame:05d}.png"), mask)

            common = {
                "positions": np.zeros((1, 3, 15), np.float32),
                "positions_camera_px": np.zeros((1, 3, 15), np.float32),
                "actions": np.zeros((1, 6), np.float32),
                "node_order": np.array("base_to_tip"),
                "n_points": np.array(15),
                "state_coordinate_frame": np.array("robot_planar_mm_v1"),
                "state_length_unit": np.array("mm"),
                "raw_action_dim": np.array(6),
                "model_action_dim": np.array(6),
            }
            np.savez_compressed(out_root / "train/train.npz", **common)
            np.savez_compressed(out_root / "val/val.npz", **common)
            config = {
                "n_points": 15, "state_frame": "robot_planar_mm",
                "max_interpolated_fraction": 0.05,
                "seq": str(seq), "out_root": str(out_root),
            }
            manifest = build_dataset_manifest(
                seq=str(seq), camera="cam0", derived=str(derived),
                crop_root=str(crop_root), mask_dir=str(mask_dir),
                out_root=str(out_root), resolved_config=config,
                commands=[f"python {repo}/scripts/run.py --seq {seq}"],
                sam2_summary={
                    "frame_ids_match": True, "failures_empty": True,
                    "mask_count": 2,
                }, paths=paths, git_commit="f06c8c9")

            validate_registry_dataset_manifest(manifest)
            serialized = json.dumps(
                manifest, ensure_ascii=False, allow_nan=False)
            self.assertNotIn(str(repo), serialized)
            self.assertIn("artifact://data/processed/real/dataset_demo", serialized)
            self.assertEqual(manifest["status"], "draft")
            self.assertTrue(manifest["quality_control"]["training_ready"])

    def test_training_tag_reads_registry_sources_contract(self):
        with tempfile.TemporaryDirectory() as root:
            manifest_path = Path(root) / "dataset_manifest.json"
            manifest_path.write_text(json.dumps({
                "dataset_id": "dataset_demo",
                "sources": [{"sequence_id": "seq_20260819_172644"}],
            }))
            self.assertEqual(
                infer_sequence_tag(root, str(manifest_path)),
                "seq_20260819_172644")

    def test_real_training_paths_use_workspace_and_legacy_read_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            (repo / "config").mkdir(parents=True)
            (repo / "config/paths.local.toml").write_text(
                "schema_version = 1\n"
                "[paths]\nworkspace_root = 'artifacts'\n"
                "[compat]\nraw_roots = ['old_raw']\n"
                "intermediate_roots = ['old_derived', 'old_masks']\n",
                encoding="utf-8")
            raw = repo / "old_raw/seq_demo"
            masks = repo / "old_masks/seq_demo_full"
            raw.mkdir(parents=True)
            masks.mkdir(parents=True)
            paths = ProjectPaths.load(repo_root=repo, environ={})

            resolved = resolve_real_pipeline_paths(
                "seq_demo", "dataset_demo", "seq_demo", paths)
            self.assertEqual(resolved["raw_sequence"], raw)
            self.assertEqual(resolved["masks_dir"], masks)
            self.assertEqual(
                resolved["trial_base"],
                repo / "artifacts/runs/training/real_pipeline/seq_demo")

            canonical_raw = paths.raw_sequence("real", "seq_demo")
            canonical_masks = paths.intermediate_sequence(
                "real", "seq_demo", SAM2_VIDEO_RECIPE)
            canonical_raw.mkdir(parents=True)
            canonical_masks.mkdir(parents=True)
            resolved = resolve_real_pipeline_paths(
                "seq_demo", "dataset_demo", "seq_demo", paths)
            self.assertEqual(resolved["raw_sequence"], canonical_raw)
            self.assertEqual(resolved["masks_dir"], canonical_masks)

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
