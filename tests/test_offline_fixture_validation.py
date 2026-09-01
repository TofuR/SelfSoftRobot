import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from real_validation.contracts.models import ModelDescriptor
from scripts.real.validate_offline_fixture import validate_offline_fixture
from src.registry.manifests import sha256_file
from src.registry.paths import ProjectPaths


class _FakeRuntime:
    def __init__(self, checkpoint, data_dir=None, device="cpu"):
        del data_dir, device
        self.descriptor = ModelDescriptor(
            checkpoint=checkpoint,
            checkpoint_hash="c" * 64,
            model_type="state_transition",
            action_dim=4,
            n_nodes=15,
            history_steps=3,
            channel_map=(0, 1, 3, 5),
            state_coordinate_frame="robot_planar_mm_v1",
            state_length_unit="mm",
            node_order="base_to_tip",
        )
        self.model = type("Geometry", (), {
            "pc_center": torch.zeros(3),
            "pc_scale": torch.ones(3),
        })()
        self.cleared = False

    def clear(self):
        self.cleared = True


class OfflineFixtureValidationTest(unittest.TestCase):
    def test_registered_test_split_builds_offline_anchor(self):
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root) / "repo"
            repo.mkdir()
            paths = ProjectPaths.load(repo_root=repo, environ={})
            dataset = paths.processed_dataset("real", "reference_v1")
            npz = dataset / "splits/test/test.npz"
            npz.parent.mkdir(parents=True)
            np.savez_compressed(
                npz,
                positions=np.zeros((6, 3, 15), np.float32),
                actions=np.arange(36, dtype=np.float32).reshape(6, 6),
                state_coordinate_frame=np.array("robot_planar_mm_v1"),
                state_length_unit=np.array("mm"),
                node_order=np.array("base_to_tip"),
            )
            digest = sha256_file(npz)
            manifest = {
                "splits": {"train": [], "val": [], "test": [{
                    "uri": paths.artifact_uri(npz), "sha256": digest,
                }]},
            }
            (dataset / "dataset_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8")
            checkpoint = repo / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            out = paths.validation_run("fixture_001") / "offline_fixture.json"

            result = validate_offline_fixture(
                paths=paths,
                dataset_id="reference_v1",
                role="test",
                checkpoint=checkpoint,
                frame_index=3,
                out=out,
                runtime_factory=_FakeRuntime,
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["n_nodes"], 15)
            self.assertEqual(result["history_steps"], 3)
            self.assertEqual(result["action_dim"], 4)
            self.assertEqual(result["checkpoint_uri"], "repo://checkpoint.pt")
            self.assertNotIn(str(repo), json.dumps(result))
            self.assertEqual(json.loads(out.read_text()), result)
            with self.assertRaises(FileExistsError):
                validate_offline_fixture(
                    paths=paths,
                    dataset_id="reference_v1",
                    role="test",
                    checkpoint=checkpoint,
                    frame_index=3,
                    out=out,
                    runtime_factory=_FakeRuntime,
                )


if __name__ == "__main__":
    unittest.main()
