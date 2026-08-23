import json
import os
import tempfile
import unittest

import numpy as np

from scripts.real.manage_training_trial import validate_dataset_manifest
from scripts.real.masks_to_transition_npz import save_npz
from scripts.real.preprocess_capture import (
    build_parser,
    resolve_pipeline_args,
    validate_stage_dependencies,
)


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


if __name__ == "__main__":
    unittest.main()
