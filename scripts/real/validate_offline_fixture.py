#!/usr/bin/env python3
"""Validate that a registered dataset split can seed real_validation offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from real_validation.runtime.anchors import anchor_from_npz  # noqa: E402
from real_validation.runtime.model_runtime import ModelRuntime  # noqa: E402
from src.registry.datasets import DatasetSelector  # noqa: E402
from src.registry.manifests import atomic_write_json  # noqa: E402
from src.registry.paths import ProjectPaths  # noqa: E402


def _portable_uri(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.artifact_uri(path)
    except ValueError:
        return paths.repo_uri(path)


def validate_offline_fixture(
    *,
    paths: ProjectPaths,
    dataset_id: str,
    role: str,
    checkpoint: Path,
    frame_index: int,
    device: str = "cpu",
    out: Path | None = None,
    runtime_factory=ModelRuntime,
) -> dict:
    artifact = DatasetSelector(paths).resolve(dataset_id, role, verify_hash=True)
    runtime = runtime_factory(
        str(Path(checkpoint).resolve()), data_dir=str(artifact.path.parent),
        device=device)
    try:
        anchor = anchor_from_npz(
            artifact.path, frame_index, runtime.descriptor, runtime.model,
            padding="reject")
        result = {
            "schema_version": 1,
            "kind": "offline_fixture_acceptance",
            "status": "passed",
            "dataset_id": dataset_id,
            "dataset_role": role,
            "artifact_uri": paths.artifact_uri(artifact.path),
            "manifest_uri": (
                paths.artifact_uri(artifact.manifest_path)
                if artifact.manifest_path else None),
            "frame_index": int(frame_index),
            "checkpoint_uri": _portable_uri(paths, Path(checkpoint)),
            "checkpoint_sha256": runtime.descriptor.checkpoint_hash,
            "state_coordinate_frame": runtime.descriptor.state_coordinate_frame,
            "state_length_unit": runtime.descriptor.state_length_unit,
            "node_order": runtime.descriptor.node_order,
            "n_nodes": len(anchor.state),
            "history_steps": len(anchor.action_history),
            "action_dim": (
                len(anchor.action_history[0]) if anchor.action_history else 0),
            "anchor_source": (
                f"{paths.artifact_uri(artifact.path)}#frame={frame_index}"),
        }
        if out is not None:
            atomic_write_json(Path(out), result, overwrite=False)
        return result
    finally:
        clear = getattr(runtime, "clear", None)
        if callable(clear):
            clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--role", choices=("train", "val", "test"),
                        default="test")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workspace-root", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_offline_fixture(
        paths=ProjectPaths.load(
            repo_root=PROJECT_ROOT, workspace_root=args.workspace_root),
        dataset_id=args.dataset_id,
        role=args.role,
        checkpoint=args.checkpoint,
        frame_index=args.frame_index,
        device=args.device,
        out=args.out,
    )
    print(
        f"offline fixture: {result['status']} dataset={result['dataset_id']} "
        f"role={result['dataset_role']} frame={result['frame_index']} "
        f"nodes={result['n_nodes']} history={result['history_steps']}x"
        f"{result['action_dim']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
