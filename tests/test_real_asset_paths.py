from pathlib import Path
import tempfile
import unittest

from src.registry.paths import ProjectPaths
from src.registry.real_assets import (
    SAM2_VIDEO_RECIPE,
    canonical_output,
    resolve_candidate_masks,
    resolve_processed_dataset,
    resolve_raw_sequence,
    resolve_repaired_masks,
    resolve_sam2_masks,
)
from scripts.real.combine_transition_datasets import (
    build_parser as build_combine_parser, resolve_output,
)


class RealAssetPathTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name) / "repo"
        self.repo.mkdir()
        self.paths = ProjectPaths.load(repo_root=self.repo, environ={})

    def tearDown(self):
        self.temporary.cleanup()

    def test_canonical_assets_win_over_legacy(self):
        legacy_raw = self.repo / "real_capture/data/raw/seq_a/cam0"
        legacy_data = self.repo / "data/real_seq/dataset_a"
        legacy_masks = self.repo / "sam2/masks/seq_a_full"
        for path in (legacy_raw, legacy_data, legacy_masks):
            path.mkdir(parents=True)
        canonical_raw = self.paths.raw_sequence("real", "seq_a")
        canonical_data = self.paths.processed_dataset("real", "dataset_a")
        canonical_masks = self.paths.intermediate_sequence(
            "real", "seq_a", SAM2_VIDEO_RECIPE)
        for path in (canonical_raw / "cam0", canonical_data, canonical_masks):
            path.mkdir(parents=True)

        self.assertEqual(
            resolve_raw_sequence(self.paths, "seq_a", camera="cam0"),
            canonical_raw.resolve())
        self.assertEqual(
            resolve_processed_dataset(self.paths, "dataset_a"),
            canonical_data.resolve())
        self.assertEqual(
            resolve_sam2_masks(self.paths, "seq_a"),
            canonical_masks.resolve())

    def test_legacy_inputs_remain_readable(self):
        raw = self.repo / "real_capture/data/raw/seq_a/cam0"
        dataset = self.repo / "data/real_seq/dataset_a"
        masks = self.repo / "real_capture/data/derived/seq_a/masks"
        for path in (raw, dataset, masks):
            path.mkdir(parents=True)

        self.assertEqual(
            resolve_raw_sequence(self.paths, "seq_a", camera="cam0"),
            raw.parent.resolve())
        self.assertEqual(
            resolve_processed_dataset(self.paths, "dataset_a"),
            dataset.resolve())
        self.assertEqual(
            resolve_candidate_masks(self.paths, "seq_a"), masks.resolve())

    def test_canonical_repaired_masks_recipe_is_discoverable(self):
        repaired = self.paths.intermediate_sequence(
            "real", "seq_a", "legacy-mask-repair-v1")
        repaired.mkdir(parents=True)
        self.assertEqual(
            resolve_repaired_masks(self.paths, "seq_a"), repaired.resolve())

    def test_new_output_must_be_inside_workspace(self):
        target = self.paths.processed_dataset("real", "dataset_a")
        self.assertEqual(canonical_output(self.paths, target), target)
        with self.assertRaisesRegex(ValueError, "workspace"):
            canonical_output(self.paths, self.repo / "data/real_seq/new")

    def test_combined_dataset_defaults_to_workspace(self):
        args = build_combine_parser().parse_args([
            "--dataset-id", "combined_a", "--train", "train.npz",
            "--val", "val.npz",
        ])
        self.assertEqual(
            resolve_output(args, self.paths),
            self.paths.processed_dataset("real", "combined_a"))

        legacy = build_combine_parser().parse_args([
            "--out-root", "data/real_seq/combined_a",
            "--train", "train.npz", "--val", "val.npz",
        ])
        with self.assertRaisesRegex(ValueError, "workspace"):
            resolve_output(legacy, self.paths)


if __name__ == "__main__":
    unittest.main()
