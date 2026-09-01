import copy
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from src.registry.manifests import (
    ManifestError,
    ManifestStore,
    build_file_record,
    sha256_file,
    validate_dataset_manifest,
    validate_run_manifest,
)
from src.registry.paths import ProjectPaths


HASH_A = "a" * 64
HASH_B = "b" * 64


def dataset_manifest():
    train_uri = "artifact://data/processed/real/dataset_a/splits/train/a.npz"
    val_uri = "artifact://data/processed/real/dataset_a/splits/val/a.npz"
    return {
        "schema_version": 2,
        "kind": "dataset",
        "dataset_id": "dataset_a",
        "created_at": "2026-08-31T12:00:00+08:00",
        "status": "released",
        "sources": [{
            "sequence_id": "seq_20260819_172644",
            "raw_manifest_sha256": HASH_A,
        }],
        "recipe": {
            "name": "sam2_to_transition",
            "version": 1,
            "git_commit": "3157b12",
            "parameters": {"n_nodes": 15},
            "commands": ["python scripts/real/preprocess_capture.py --config recipe.toml"],
        },
        "contracts": {
            "state": {"node_order": "base_to_tip"},
            "action": {"model_action_dim": 4},
            "timing": {"median_dt_s": 0.2},
            "observation": {"camera": "cam0"},
        },
        "split_policy": {
            "name": "chronological_purged_v1",
            "group_key": "sequence_id",
            "seed": None,
            "embargo_frames": 40,
            "evidence_level": "within_sequence",
        },
        "splits": {
            "train": [{"uri": train_uri, "sha256": HASH_A, "frames": 80}],
            "val": [{"uri": val_uri, "sha256": HASH_B, "frames": 20}],
            "test": [],
        },
        "quality_control": {"training_ready": True},
        "files": [
            {"uri": train_uri, "sha256": HASH_A, "bytes": 100},
            {"uri": val_uri, "sha256": HASH_B, "bytes": 25},
        ],
    }


def run_manifest():
    root = "artifact://runs/training/study_a/run_001"
    return {
        "schema_version": 1,
        "kind": "training_run",
        "run_id": "run_001",
        "study_id": "study_a",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "run_kind": "formal",
        "dataset": {
            "manifest_uri": "artifact://data/processed/real/dataset_a/manifest.json",
            "manifest_sha256": HASH_A,
        },
        "source": {"git_commit": "3157b12", "dirty": False, "patch_uri": None},
        "commands": ["python scripts/training/train_transition.py --mode gt"],
        "resolved_config_uri": root + "/config.resolved.json",
        "environment_uri": root + "/environment.txt",
        "seed": 42,
        "stages": [{"name": "gt"}, {"name": "open_loop"}],
        "selection": {
            "metric": "validation.node_mean_mm",
            "mode": "min",
            "dataset_role": "val",
            "checkpoint_uri": root + "/stages/gt/checkpoints/best_val.pt",
        },
        "expected_artifacts": [root + "/stages/gt/checkpoints/best_val.pt"],
    }


def run_manifest_v2():
    root = "artifact://runs/training/real_pipeline.seq_a/run_001"
    return {
        "schema_version": 2,
        "kind": "training_run",
        "run_id": "run_001",
        "study_id": "real_pipeline.seq_a",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "run_kind": "formal",
        "dataset": {
            "dataset_id": "dataset_a",
            "manifest_uri": "artifact://data/processed/real/dataset_a/manifest.json",
        },
        "source": {"git_commit": "3157b12", "dirty": True},
        "commands_uri": root + "/commands.sh",
        "resolved_config_uri": root + "/config.json",
        "seed": 42,
        "stages": [{
            "name": "gt",
            "selection": {
                "metric": "validation.node_mean_mm",
                "mode": "min",
                "dataset_role": "val",
                "checkpoint_uri": root + "/stages/gt/model/best_eval_model.pt",
            },
        }],
        "expected_artifacts": [
            root + "/stages/gt/model/best_eval_model.pt"],
        "complete_marker_uri": root + "/COMPLETE",
        "final_evaluations": [{
            "stage": "gt",
            "dataset_role": "test",
            "quantitative_uri": root + "/evaluations/test/gt/quantitative/summary.txt",
            "overlay_uri": root + "/evaluations/test/gt/overlay/summary.txt",
        }],
    }


class TestDatasetManifest(unittest.TestCase):
    def test_valid_contract(self):
        validate_dataset_manifest(dataset_manifest())

    def test_split_reuse_and_unregistered_file_fail_closed(self):
        duplicate = dataset_manifest()
        duplicate["splits"]["test"] = [
            copy.deepcopy(duplicate["splits"]["val"][0])]
        with self.assertRaisesRegex(ManifestError, "复用了同一文件"):
            validate_dataset_manifest(duplicate)

        missing = dataset_manifest()
        missing["splits"]["test"] = [{
            "uri": "artifact://data/processed/real/dataset_a/splits/test/a.npz",
            "sha256": HASH_A,
            "frames": 10,
        }]
        with self.assertRaisesRegex(ManifestError, "未登记到 files"):
            validate_dataset_manifest(missing)

    def test_released_source_hash_and_portable_uri_are_required(self):
        no_hash = dataset_manifest()
        del no_hash["sources"][0]["raw_manifest_sha256"]
        with self.assertRaisesRegex(ManifestError, "raw_manifest_sha256"):
            validate_dataset_manifest(no_hash)

        absolute = dataset_manifest()
        absolute["files"][0]["uri"] = "/Data5/private/train.npz"
        with self.assertRaisesRegex(ManifestError, "artifact URI"):
            validate_dataset_manifest(absolute)


class TestRunManifest(unittest.TestCase):
    def test_valid_contract(self):
        validate_run_manifest(run_manifest())

    def test_test_selection_and_dirty_without_patch_fail(self):
        selected_on_test = run_manifest()
        selected_on_test["selection"]["dataset_role"] = "test"
        with self.assertRaisesRegex(ManifestError, "dataset_role"):
            validate_run_manifest(selected_on_test)

        dirty = run_manifest()
        dirty["source"]["dirty"] = True
        with self.assertRaisesRegex(ManifestError, "source.patch_uri"):
            validate_run_manifest(dirty)

    def test_complete_run_requires_marker(self):
        complete = run_manifest()
        complete["status"] = "complete"
        with self.assertRaisesRegex(ManifestError, "complete_marker_uri"):
            validate_run_manifest(complete)

    def test_lightweight_v2_accepts_dirty_source_without_payload_hashes(self):
        validate_run_manifest(run_manifest_v2())

    def test_v2_keeps_selection_on_validation(self):
        selected_on_test = run_manifest_v2()
        selected_on_test["stages"][0]["selection"]["dataset_role"] = "test"
        with self.assertRaisesRegex(ManifestError, "dataset_role"):
            validate_run_manifest(selected_on_test)

        evaluated_on_val = run_manifest_v2()
        evaluated_on_val["final_evaluations"][0]["dataset_role"] = "val"
        with self.assertRaisesRegex(ManifestError, "dataset_role"):
            validate_run_manifest(evaluated_on_val)


class TestManifestStore(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})
        self.store = ManifestStore(self.paths)

    def tearDown(self):
        self.temp.cleanup()

    def test_hash_record_atomic_write_and_no_overwrite(self):
        data_file = (self.paths.processed_dataset("real", "dataset_a") /
                     "splits/train/a.npz")
        data_file.parent.mkdir(parents=True)
        data_file.write_bytes(b"self-soft-robot")
        record = build_file_record(self.paths, data_file)
        self.assertEqual(record["sha256"], sha256_file(data_file))
        self.assertEqual(record["bytes"], 15)

        value = dataset_manifest()
        target = self.store.write_dataset(value)
        self.assertEqual(self.store.read_dataset(target), value)
        with self.assertRaises(FileExistsError):
            self.store.write_dataset(value)
        updated = copy.deepcopy(value)
        updated["status"] = "deprecated"
        self.assertEqual(
            self.store.write_dataset(updated, overwrite=True), target)
        self.assertEqual(self.store.read_dataset(target), updated)

    def test_store_rejects_target_outside_workspace(self):
        with self.assertRaises(ValueError):
            self.store.write_dataset(
                dataset_manifest(), target=self.repo / "manifest.json")


if __name__ == "__main__":
    unittest.main()
