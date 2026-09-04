#!/usr/bin/env python3
"""Create a non-overwriting fit/dev/test split for the first ISHSM study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slice_npz(source: Path, target: Path, start: int, stop: int,
               evaluation_start: int | None) -> dict:
    with np.load(source, allow_pickle=False) as raw:
        if "actions" not in raw or "positions" not in raw:
            raise ValueError(f"缺少 actions/positions: {source}")
        total = len(raw["actions"])
        if len(raw["positions"]) != total:
            raise ValueError(f"actions/positions 帧数不一致: {source}")
        payload = {}
        for key in raw.files:
            value = raw[key]
            payload[key] = value[start:stop] if value.ndim > 0 and value.shape[0] == total else value
        if evaluation_start is not None:
            mask = np.zeros(stop - start, dtype=np.bool_)
            mask[evaluation_start:] = True
            payload["evaluation_mask"] = mask
        np.savez_compressed(target, **payload)
    return {
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "source_frames": total,
        "source_slice": [start, stop],
        "stored_frames": stop - start,
        "evaluation_start": evaluation_start,
        "evaluation_frames": (
            stop - start - evaluation_start
            if evaluation_start is not None else stop - start),
    }


def create_ishsm_split(source_root, output_root, *, fit_fraction=0.8,
                       context_frames=40) -> dict:
    source_root = Path(source_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有派生 split: {output_root}")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction 必须在 (0,1)")
    if context_frames < 1:
        raise ValueError("context_frames 必须为正整数")
    train_files = sorted((source_root / "train").glob("*.npz"))
    test_files = sorted((source_root / "val").glob("*.npz"))
    if not train_files or not test_files:
        raise FileNotFoundError("源目录必须同时包含 train/*.npz 和 val/*.npz")

    for role in ("fit", "dev", "test"):
        (output_root / role).mkdir(parents=True, exist_ok=False)
    roles = {role: {"files": [], "frames": 0, "evaluation_frames": 0}
             for role in ("fit", "dev", "test")}

    for source in train_files:
        with np.load(source, allow_pickle=False) as raw:
            total = len(raw["actions"])
        split = int(np.floor(total * fit_fraction))
        if split <= context_frames or total - split < 1:
            raise ValueError(f"序列太短，无法划分 fit/dev: {source}")
        fit_info = _slice_npz(
            source, output_root / "fit" / source.name, 0, split, None)
        dev_start = split - context_frames
        dev_info = _slice_npz(
            source, output_root / "dev" / source.name,
            dev_start, total, context_frames)
        for role, info in (("fit", fit_info), ("dev", dev_info)):
            roles[role]["files"].append(info)
            roles[role]["frames"] += info["stored_frames"]
            roles[role]["evaluation_frames"] += info["evaluation_frames"]

    for source in test_files:
        with np.load(source, allow_pickle=False) as raw:
            total = len(raw["actions"])
        info = _slice_npz(
            source, output_root / "test" / source.name,
            0, total, 1)
        roles["test"]["files"].append(info)
        roles["test"]["frames"] += info["stored_frames"]
        roles["test"]["evaluation_frames"] += info["evaluation_frames"]

    manifest = {
        "schema": "ishsm_split_v1",
        "source_root": str(source_root.resolve()),
        "fit_fraction": fit_fraction,
        "dev_context_frames": context_frames,
        "test_policy": "copied_but_reserved_until_configuration_freeze",
        "roles": roles,
    }
    with (output_root / "split_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--context-frames", type=int, default=40)
    args = parser.parse_args()
    result = create_ishsm_split(
        args.source, args.output, fit_fraction=args.fit_fraction,
        context_frames=args.context_frames)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
