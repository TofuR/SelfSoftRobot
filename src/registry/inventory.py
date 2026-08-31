"""Read-only inventory of legacy data and run roots.

The inventory intentionally avoids hashing or recursively sizing large image
trees. It records identities, contracts and completion markers needed before a
safe migration. Physical paths are represented as ``repo://`` or opaque legacy
root references, never copied into the snapshot as machine-specific absolutes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from .manifests import atomic_write_json
from .paths import ProjectPaths


def _legacy_uri(kind: str, root_index: int, *parts: str) -> str:
    suffix = "/".join(str(part).strip("/") for part in parts if part)
    base = f"legacy://{kind}/{root_index}"
    return f"{base}/{suffix}" if suffix else base


def _root_location(paths: ProjectPaths, kind: str, index: int, root: Path) -> str:
    try:
        return paths.repo_uri(root)
    except ValueError:
        return f"external-root://{kind}/{index}"


def _subdirectories(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name)


def _load_json_summary(path: Path) -> tuple[Optional[dict], Optional[str]]:
    if not path.is_file():
        return None, "missing"
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            return None, "top-level JSON is not an object"
        return value, None
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError: line={exc.lineno} column={exc.colno}"
    except OSError as exc:
        return None, f"{type(exc).__name__}: errno={exc.errno}"


def _source_sequence_ids(manifest: Mapping) -> list[str]:
    source = manifest.get("source", {})
    if not isinstance(source, dict):
        return []
    values = []
    sequence = source.get("sequence")
    if isinstance(sequence, str):
        values.append(sequence)
    datasets = source.get("datasets", [])
    if isinstance(datasets, list):
        for item in datasets:
            if isinstance(item, dict) and isinstance(item.get("sequence"), str):
                values.append(item["sequence"])
    return sorted(set(values))


def _split_summary(manifest: Mapping) -> dict:
    result = {}
    splits = manifest.get("splits", {})
    if not isinstance(splits, dict):
        return result
    for role, entries in sorted(splits.items()):
        if not isinstance(entries, list):
            continue
        frames = 0
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("frames"), int):
                frames += entry["frames"]
        result[role] = {"files": len(entries), "frames": frames}
    return result


class LegacyInventoryBuilder:
    def __init__(
        self,
        paths: ProjectPaths,
        *,
        now: Optional[Callable[[], datetime]] = None,
    ):
        self.paths = paths
        self._now = now or (lambda: datetime.now(timezone.utc))

    def build(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "legacy_inventory",
            "generated_at": self._now().isoformat(),
            "policy": {
                "read_only": True,
                "hash_large_files": False,
                "move_or_delete_assets": False,
            },
            "roots": self._roots(),
            "raw_sequences": self._raw_sequences(),
            "intermediate_collections": self._intermediate_collections(),
            "processed_datasets": self._processed_datasets(),
            "training_runs": self._training_runs(),
            "validation_runs": self._simple_runs("validation"),
            "analysis_collections": self._simple_runs("analysis"),
        }

    def _roots(self) -> dict:
        result = {}
        for kind in (
                "raw", "intermediate", "processed", "training",
                "validation", "analysis"):
            result[kind] = [
                {
                    "root_ref": _legacy_uri(kind, index),
                    "location": _root_location(self.paths, kind, index, root),
                    "exists": root.is_dir(),
                }
                for index, root in enumerate(self.paths.legacy.roots_for(kind))
            ]
        return result

    def _raw_sequences(self) -> list[dict]:
        result = []
        for root_index, root in enumerate(self.paths.legacy.roots_for("raw")):
            for sequence in _subdirectories(root):
                meta_path = sequence / "meta.json"
                cameras = sorted(
                    path.name for path in sequence.iterdir()
                    if path.is_dir() and path.name.startswith("cam"))
                result.append({
                    "sequence_id": sequence.name,
                    "uri": _legacy_uri("raw", root_index, sequence.name),
                    "meta_present": meta_path.is_file(),
                    "camera_dirs": cameras,
                    "capture_files": {
                        name: (sequence / name).is_file()
                        for name in (
                            "actions6.csv", "frame_times.txt", "ndi.csv",
                            "samples.csv", "commands.csv")
                    },
                })
        return sorted(result, key=lambda item: (item["sequence_id"], item["uri"]))

    def _intermediate_collections(self) -> list[dict]:
        result = []
        for root_index, root in enumerate(
                self.paths.legacy.roots_for("intermediate")):
            for collection in _subdirectories(root):
                result.append({
                    "collection_id": collection.name,
                    "uri": _legacy_uri(
                        "intermediate", root_index, collection.name),
                    "top_level_entries": len(list(collection.iterdir())),
                    "manifest_present": any(
                        (collection / name).is_file() for name in (
                            "preprocess_manifest.json", "stage_manifest.json")),
                })
        return sorted(result, key=lambda item: (item["collection_id"], item["uri"]))

    def _processed_datasets(self) -> list[dict]:
        result = []
        for root_index, root in enumerate(
                self.paths.legacy.roots_for("processed")):
            for dataset in _subdirectories(root):
                manifest_path = dataset / "dataset_manifest.json"
                manifest, error = _load_json_summary(manifest_path)
                item = {
                    "dataset_id": dataset.name,
                    "uri": _legacy_uri("processed", root_index, dataset.name),
                    "manifest_uri": (
                        _legacy_uri(
                            "processed", root_index, dataset.name,
                            "dataset_manifest.json")
                        if manifest_path.is_file() else None),
                    "manifest_schema_version": (
                        manifest.get("schema_version") if manifest else None),
                    "source_sequence_ids": (
                        _source_sequence_ids(manifest) if manifest else []),
                    "splits": _split_summary(manifest) if manifest else {},
                    "training_ready": (
                        manifest.get("quality_control", {}).get("training_ready")
                        if manifest and isinstance(
                            manifest.get("quality_control"), dict) else None),
                    "manifest_error": error,
                }
                result.append(item)
        return sorted(result, key=lambda item: (item["dataset_id"], item["uri"]))

    @staticmethod
    def _is_run_config(path: Path) -> bool:
        return (path.name == "config.json" and
                (path.parent.name.startswith("exp_") or
                 path.parent.name.startswith("trial_")))

    @staticmethod
    def _completion_markers(run: Path) -> list[str]:
        candidates = (
            "RUN_COMPLETE",
            "COMPLETE",
            "evaluations/COMPLETE",
        )
        return [name for name in candidates if (run / name).is_file()]

    def _training_runs(self) -> list[dict]:
        result = []
        for root_index, root in enumerate(
                self.paths.legacy.roots_for("training")):
            if not root.is_dir():
                continue
            for config_path in sorted(root.rglob("config.json")):
                if not self._is_run_config(config_path):
                    continue
                run = config_path.parent
                relative = run.relative_to(root)
                markers = self._completion_markers(run)
                result.append({
                    "run_id": run.name,
                    "study_path": "/".join(relative.parts[:-1]),
                    "uri": _legacy_uri("training", root_index, *relative.parts),
                    "config_present": True,
                    "completion_markers": markers,
                    "operational_status": "complete" if markers else "unverified",
                })
        return sorted(
            result, key=lambda item: (item["study_path"], item["run_id"], item["uri"]))

    def _simple_runs(self, kind: str) -> list[dict]:
        result = []
        for root_index, root in enumerate(self.paths.legacy.roots_for(kind)):
            for item in _subdirectories(root):
                result.append({
                    "id": item.name,
                    "uri": _legacy_uri(kind, root_index, item.name),
                    "top_level_entries": len(list(item.iterdir())),
                })
        return sorted(result, key=lambda item: (item["id"], item["uri"]))


def write_legacy_inventory(
    paths: ProjectPaths,
    inventory: Mapping,
    *,
    target: Optional[str | Path] = None,
) -> Path:
    destination = Path(target) if target is not None else (
        paths.registry_root / "legacy_inventory.json")
    paths.artifact_uri(destination)
    return atomic_write_json(destination, inventory, overwrite=True)
