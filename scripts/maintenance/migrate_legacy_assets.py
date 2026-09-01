"""Audit and atomically migrate registered legacy assets into workspace.

The tool never copies large trees.  On one filesystem it renames each source
to its canonical target, then leaves a relative compatibility symlink.  A
content ledger makes verification, alias removal and rollback explicit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.manifests import atomic_write_json  # noqa: E402
from src.registry.paths import ProjectPaths  # noqa: E402
from src.registry.real_assets import (  # noqa: E402
    LEGACY_DERIVED_RECIPE, SAM2_VIDEO_RECIPE,
)


LEDGER_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _sam2_target(paths: ProjectPaths, collection: Path) -> Path:
    name = collection.name
    match = re.match(r"^(seq_[0-9_]+)_full(?:_(.+))?$", name)
    if match:
        sequence_id, suffix = match.groups()
        recipe = SAM2_VIDEO_RECIPE if not suffix else (
            "legacy-sam2-" + re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-"))
        return paths.intermediate_sequence("real", sequence_id, recipe)
    return paths.intermediate_sequence(
        "real", "legacy-unassigned", f"sam2-{name}")


def discover_mappings(paths: ProjectPaths) -> list[dict]:
    mappings = []

    def add(kind: str, source: Path, target: Path, identity: str) -> None:
        if source.exists() and not source.is_symlink():
            mappings.append({
                "asset_id": f"{kind}:{identity}",
                "kind": kind,
                "source_uri": paths.repo_uri(source),
                "target_uri": paths.artifact_uri(target),
                "identity": identity,
            })

    raw_root = paths.repo_root / "real_capture/data/raw"
    for sequence in sorted(raw_root.iterdir()) if raw_root.is_dir() else ():
        if sequence.is_dir():
            add("raw", sequence, paths.raw_sequence("real", sequence.name), sequence.name)

    derived_root = paths.repo_root / "real_capture/data/derived"
    for sequence in sorted(derived_root.iterdir()) if derived_root.is_dir() else ():
        if sequence.is_dir():
            add("intermediate", sequence, paths.intermediate_sequence(
                "real", sequence.name, LEGACY_DERIVED_RECIPE), sequence.name)

    sam2_root = paths.repo_root / "sam2/masks"
    for collection in sorted(sam2_root.iterdir()) if sam2_root.is_dir() else ():
        if collection.is_dir():
            add("intermediate", collection, _sam2_target(paths, collection),
                collection.name)

    processed_root = paths.repo_root / "data/real_seq"
    for dataset in sorted(processed_root.iterdir()) if processed_root.is_dir() else ():
        if dataset.is_dir():
            add("processed", dataset, paths.processed_dataset(
                "real", dataset.name), dataset.name)
        elif dataset.name == "README.md":
            add("processed-index", dataset,
                paths.data_root / "processed/real/legacy_README.md", dataset.name)

    training_root = paths.repo_root / "train_log"
    for item in sorted(training_root.iterdir()) if training_root.is_dir() else ():
        target = paths.runs_root / "training" / item.name
        add("training", item, target, item.name)

    validation_root = paths.repo_root / "real_validation/runs"
    for run in sorted(validation_root.iterdir()) if validation_root.is_dir() else ():
        if run.is_dir():
            add("validation", run, paths.validation_run(run.name), run.name)

    analysis_root = paths.repo_root / "output"
    for collection in sorted(analysis_root.iterdir()) if analysis_root.is_dir() else ():
        if collection.is_dir():
            add("analysis", collection,
                paths.analysis_run(collection.name, "legacy-import"), collection.name)

    checkpoints = paths.repo_root / "sam2/checkpoints"
    add("pretrained-model", checkpoints,
        paths.pretrained_model_dir("sam2"), "sam2")

    sources = [item["source_uri"] for item in mappings]
    targets = [item["target_uri"] for item in mappings]
    if len(sources) != len(set(sources)) or len(targets) != len(set(targets)):
        raise ValueError("迁移映射含重复 source 或 target")
    return mappings


def build_ledger(paths: ProjectPaths) -> dict:
    mappings = discover_mappings(paths)
    entries = []
    for index, item in enumerate(mappings, 1):
        source = paths.resolve_repo_uri(item["source_uri"])
        print(f"[{index}] register {item['asset_id']}: {source}", flush=True)
        item["status"] = "planned"
        entries.append(item)
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "kind": "legacy_asset_migration",
        "created_at": _utc_now(),
        "git_commit": _git_commit(paths.repo_root),
        "workspace_uri": "artifact://",
        "method": "same_filesystem_atomic_rename_with_relative_compat_symlink",
        "entries": entries,
    }


def _validate_ledger(ledger: dict) -> None:
    if (ledger.get("schema_version") != LEDGER_SCHEMA_VERSION
            or ledger.get("kind") != "legacy_asset_migration"):
        raise ValueError("不是受支持的 legacy migration ledger")
    if not isinstance(ledger.get("entries"), list) or not ledger["entries"]:
        raise ValueError("migration ledger entries 不能为空")


def _load_ledger(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        ledger = json.load(stream)
    _validate_ledger(ledger)
    return ledger


def _source_path(paths: ProjectPaths, uri: str) -> Path:
    """Resolve repo URI lexically so a compatibility symlink is not followed."""
    prefix = "repo://"
    if not isinstance(uri, str) or not uri.startswith(prefix):
        raise ValueError(f"不是 repo URI: {uri!r}")
    relative = PurePosixPath(uri[len(prefix):])
    if (not relative.parts or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)):
        raise ValueError(f"非法 repo URI: {uri!r}")
    return paths.repo_root.joinpath(*relative.parts)


def _write_ledger(path: Path, ledger: dict) -> None:
    atomic_write_json(path, ledger, overwrite=path.exists())


def apply_migration(paths: ProjectPaths, ledger: dict) -> None:
    _validate_ledger(ledger)
    for item in ledger["entries"]:
        source = _source_path(paths, item["source_uri"])
        target = paths.resolve_artifact_uri(item["target_uri"])
        if source.is_symlink() and source.resolve() == target.resolve():
            item["status"] = "aliased"
            continue
        if not source.exists():
            raise FileNotFoundError(f"迁移 source 不存在: {source}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"迁移 target 已存在: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.stat().st_dev != target.parent.stat().st_dev:
            raise OSError(f"source/target 不在同一文件系统: {source} -> {target}")
        source.rename(target)
        relative_target = os.path.relpath(target, source.parent)
        source.symlink_to(relative_target, target_is_directory=target.is_dir())
        item["status"] = "aliased"
        item["migrated_at"] = _utc_now()
        print(f"migrated {source} -> {target}", flush=True)


def verify_migration(paths: ProjectPaths, ledger: dict, *, require_alias=True) -> None:
    _validate_ledger(ledger)
    for item in ledger["entries"]:
        source = _source_path(paths, item["source_uri"])
        target = paths.resolve_artifact_uri(item["target_uri"])
        if not target.exists():
            raise FileNotFoundError(f"canonical target 不存在: {target}")
        if require_alias and (not source.is_symlink() or source.resolve() != target.resolve()):
            raise ValueError(f"compat alias 无效: {source} -> {target}")
        print(f"verified {item['asset_id']}", flush=True)


def remove_aliases(paths: ProjectPaths, ledger: dict) -> None:
    verify_migration(paths, ledger, require_alias=True)
    for item in ledger["entries"]:
        source = _source_path(paths, item["source_uri"])
        target = paths.resolve_artifact_uri(item["target_uri"])
        if not source.is_symlink() or source.resolve() != target.resolve():
            raise ValueError(f"拒绝删除非预期 alias: {source}")
        source.unlink()
        item["status"] = "canonical-only"
        item["alias_removed_at"] = _utc_now()
        print(f"removed alias {source}", flush=True)


def rollback_migration(paths: ProjectPaths, ledger: dict) -> None:
    _validate_ledger(ledger)
    for item in reversed(ledger["entries"]):
        source = _source_path(paths, item["source_uri"])
        target = paths.resolve_artifact_uri(item["target_uri"])
        if source.is_symlink() and source.resolve() == target.resolve():
            source.unlink()
        elif source.exists():
            raise FileExistsError(f"rollback source 已被占用: {source}")
        if not target.exists():
            raise FileNotFoundError(f"rollback target 不存在: {target}")
        source.parent.mkdir(parents=True, exist_ok=True)
        target.rename(source)
        item["status"] = "rolled-back"
        item["rolled_back_at"] = _utc_now()
        print(f"rolled back {target} -> {source}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="历史资产可验证原子迁移")
    parser.add_argument("command", choices=(
        "snapshot", "apply", "verify", "remove-aliases", "rollback"))
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--workspace-root", default=None)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(workspace_root=args.workspace_root)
    ledger_path = Path(args.ledger).expanduser().resolve()
    if args.command == "snapshot":
        if ledger_path.exists():
            raise FileExistsError(f"拒绝覆盖 migration ledger: {ledger_path}")
        ledger = build_ledger(paths)
        _write_ledger(ledger_path, ledger)
        print(ledger_path)
        return
    ledger = _load_ledger(ledger_path)
    if args.command == "apply":
        apply_migration(paths, ledger)
    elif args.command == "verify":
        verify_migration(paths, ledger)
    elif args.command == "remove-aliases":
        remove_aliases(paths, ledger)
    elif args.command == "rollback":
        rollback_migration(paths, ledger)
    _write_ledger(ledger_path, ledger)


if __name__ == "__main__":
    main()
