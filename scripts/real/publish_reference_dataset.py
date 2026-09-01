#!/usr/bin/env python3
"""Publish a small immutable train/val/frozen-test reference release."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.manifests import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    validate_dataset_manifest,
)
from src.registry.paths import ProjectPaths  # noqa: E402


CONTRACT_KEYS = (
    "state_coordinate_frame",
    "state_length_unit",
    "node_order",
    "n_points",
    "raw_action_dim",
    "model_action_dim",
    "model_action_channels",
    "channel_source6",
    "action_expansion6",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()


def _portable_uri(paths: ProjectPaths, path: Path) -> str:
    try:
        return paths.artifact_uri(path)
    except ValueError:
        return paths.repo_uri(path)


def _scalar_or_list(value: np.ndarray):
    return value.item() if value.ndim == 0 else value.tolist()


def inspect_npz(path: Path) -> dict:
    source = path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as data:
        missing = {"positions", "actions", *CONTRACT_KEYS} - set(data.files)
        if missing:
            raise ValueError(f"{source} 缺少 NPZ 合同字段: {sorted(missing)}")
        positions = np.asarray(data["positions"])
        actions = np.asarray(data["actions"])
        if (positions.ndim != 3 or actions.ndim != 2 or
                len(positions) != len(actions)):
            raise ValueError(
                f"{source} 期望 positions=(T,3,N)/(T,N,3), actions=(T,D)")
        if not np.isfinite(positions).all() or not np.isfinite(actions).all():
            raise ValueError(f"{source} 含 NaN/Inf")
        contract = {
            key: _scalar_or_list(np.asarray(data[key]))
            for key in CONTRACT_KEYS
        }
        action_scales = {
            key: _scalar_or_list(np.asarray(data[key]))
            for key in ("raw_action_scale6_kpa", "action_scale_kpa")
            if key in data
        }
    return {
        "path": source,
        "frames": int(len(positions)),
        "positions_shape": list(positions.shape),
        "actions_shape": list(actions.shape),
        "contract": contract,
        "action_scales": action_scales,
    }


def _raw_manifest(paths: ProjectPaths, sequence_id: str) -> tuple[Path, str, dict]:
    raw = paths.raw_sequence("real", sequence_id)
    meta_path = raw / "meta.json"
    frame_times = raw / "frame_times.txt"
    camera = raw / "cam0"
    if not meta_path.is_file() or not frame_times.is_file() or not camera.is_dir():
        raise FileNotFoundError(f"raw sequence 不完整: {raw}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image_count = sum(1 for path in camera.glob("*.png") if path.is_file())
    frame_count = sum(
        1 for line in frame_times.read_text(encoding="utf-8").splitlines()
        if line.strip())
    if image_count != frame_count or image_count != int(meta.get("frames", -1)):
        raise ValueError(
            f"raw frame count 不一致: {sequence_id}; "
            f"images={image_count}, times={frame_count}, meta={meta.get('frames')}")
    manifest = {
        "schema_version": 1,
        "kind": "raw_sequence",
        "sequence_id": sequence_id,
        "created_at": _timestamp(),
        "status": "registered",
        "sequence_uri": paths.artifact_uri(raw),
        "capture": {
            "start_iso": meta.get("start_iso"),
            "stop_iso": meta.get("stop_iso"),
            "action_interval_s": meta.get("action_interval_s"),
            "camera_count": meta.get("camera_count"),
            "ndi_count": meta.get("ndi_count"),
        },
        "counts": {"cam0_images": image_count, "frame_times": frame_count},
        "metadata_files": [
            name for name in ("meta.json", "frame_times.txt", "actions6.csv", "ndi.csv")
            if (raw / name).is_file()
        ],
        "note": "Lightweight registration manifest; raw payload files are not re-hashed.",
    }
    target = raw / "raw_manifest.json"
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("sequence_id") != sequence_id:
            raise ValueError(f"raw manifest sequence_id 不一致: {target}")
    else:
        atomic_write_json(target, manifest, overwrite=False)
    return target, sha256_file(target), meta


def publish_reference_release(
    *,
    paths: ProjectPaths,
    dataset_id: str,
    train_npz: Path,
    val_npz: Path,
    test_npz: Path,
    train_sequence: str,
    test_sequence: str,
    command: str,
) -> Path:
    target = paths.processed_dataset("real", dataset_id)
    if target.exists():
        raise FileExistsError(f"拒绝覆盖 processed release: {target}")
    if train_sequence == test_sequence:
        raise ValueError("frozen test 必须来自独立 sequence")

    inspected = {
        "train": inspect_npz(Path(train_npz)),
        "val": inspect_npz(Path(val_npz)),
        "test": inspect_npz(Path(test_npz)),
    }
    reference_contract = inspected["train"]["contract"]
    for role, item in inspected.items():
        if item["contract"] != reference_contract:
            raise ValueError(f"{role} NPZ 与 train 的状态/动作视图合同不一致")

    raw_sources = {}
    raw_meta = {}
    for sequence_id in (train_sequence, test_sequence):
        manifest_path, manifest_hash, meta = _raw_manifest(paths, sequence_id)
        raw_sources[sequence_id] = {
            "sequence_id": sequence_id,
            "raw_manifest_uri": paths.artifact_uri(manifest_path),
            "raw_manifest_sha256": manifest_hash,
        }
        raw_meta[sequence_id] = meta
    intervals = {
        round(float(meta["action_interval_s"]), 9)
        for meta in raw_meta.values()
    }
    if len(intervals) != 1:
        raise ValueError(f"train/test action_interval_s 不一致: {sorted(intervals)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.", dir=target.parent))
    try:
        file_records = []
        splits = {"train": [], "val": [], "test": []}
        source_sequences = {
            "train": train_sequence,
            "val": train_sequence,
            "test": test_sequence,
        }
        for role, item in inspected.items():
            destination = temporary / "splits" / role / f"{role}.npz"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["path"], destination)
            digest = sha256_file(destination)
            uri = paths.artifact_uri(target / "splits" / role / destination.name)
            file_records.append({
                "uri": uri,
                "sha256": digest,
                "bytes": destination.stat().st_size,
            })
            splits[role].append({
                "uri": uri,
                "sha256": digest,
                "frames": item["frames"],
                "sequence_id": source_sequences[role],
            })

        manifest = {
            "schema_version": 2,
            "kind": "dataset",
            "dataset_id": dataset_id,
            "created_at": _timestamp(),
            "status": "released",
            "sources": [raw_sources[train_sequence], raw_sources[test_sequence]],
            "recipe": {
                "name": "reference_split_release",
                "version": 1,
                "git_commit": _git_commit(),
                "parameters": {
                    "train_sequence": train_sequence,
                    "test_sequence": test_sequence,
                    "source_roles": {
                        role: _portable_uri(paths, item["path"])
                        for role, item in inspected.items()
                    },
                    "copy_mode": "copy2_no_transform",
                },
                "commands": [command],
            },
            "contracts": {
                "state": {
                    key: reference_contract[key] for key in (
                        "state_coordinate_frame", "state_length_unit",
                        "node_order", "n_points")
                },
                "action": {
                    key: reference_contract[key] for key in (
                        "raw_action_dim", "model_action_dim",
                        "model_action_channels", "channel_source6",
                        "action_expansion6")
                } | {
                    "stored_units": "normalized_ratio",
                    "physical_scale_kpa_by_role": {
                        role: item["action_scales"]
                        for role, item in inspected.items()
                    },
                },
                "timing": {
                    "action_interval_s": intervals.pop(),
                    "sampling_policy": "native_sequence_rate",
                },
                "observation": {
                    "camera": "cam0",
                    "representation": "sam2_skeleton_robot_planar_mm",
                },
                "evaluation": {"test_policy": "frozen_final_only"},
            },
            "split_policy": {
                "name": "within_sequence_val_independent_sequence_test_v1",
                "group_key": "sequence_id",
                "embargo_frames": 0,
                "seed": None,
                "evidence_level": "within_sequence",
            },
            "files": file_records,
            "splits": splits,
            "quality_control": {
                "automated_checks_passed": True,
                "training_ready": True,
                "checks": [
                    {"name": "npz_contract_match", "passed": True,
                     "required_for_training": True},
                    {"name": "finite_state_action", "passed": True,
                     "required_for_training": True},
                    {"name": "independent_frozen_test_sequence", "passed": True,
                     "required_for_training": True},
                    {"name": "native_timing_match", "passed": True,
                     "required_for_training": True},
                ],
            },
        }
        validate_dataset_manifest(manifest)
        atomic_write_json(temporary / "dataset_manifest.json", manifest, overwrite=False)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "dataset_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--train-sequence", required=True)
    parser.add_argument("--test-sequence", required=True)
    parser.add_argument("--workspace-root", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(
        repo_root=PROJECT_ROOT, workspace_root=args.workspace_root)
    command = " ".join(["python", "scripts/real/publish_reference_dataset.py", *sys.argv[1:]])
    manifest = publish_reference_release(
        paths=paths,
        dataset_id=args.dataset_id,
        train_npz=args.train,
        val_npz=args.val,
        test_npz=args.test,
        train_sequence=args.train_sequence,
        test_sequence=args.test_sequence,
        command=command,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
