"""Lightweight index of canonical datasets and historical training runs.

The index is intentionally observational: it reads directory names, small JSON
files and completion markers, but does not hash or open dataset/checkpoint
payloads.  It is therefore suitable for lineage discovery, not for claiming
that a historical artifact satisfies the current immutable manifest contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Optional

from .manifests import ManifestError, atomic_write_json, validate_dataset_manifest
from .paths import ProjectPaths


_DATASET_PATH_PATTERNS = (
    re.compile(r"(?:^|/)data/real_seq/([^/]+)"),
    re.compile(r"(?:^|/)data/processed/real/([^/]+)"),
)
_SEQUENCE_FILENAME = re.compile(r"^(seq_\d{8}_\d{6})(?:_|\.)")


def _read_json(path: Path) -> tuple[Optional[dict], Optional[str]]:
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


def _walk_values(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield None, item
            yield from _walk_values(item)


def _dataset_ids_from_config(config: Mapping[str, Any]) -> list[str]:
    result = set()
    for key, value in _walk_values(config):
        if key == "dataset_id" and isinstance(value, str) and value:
            result.add(value)
        if not isinstance(value, str):
            continue
        normalized = value.replace("\\", "/")
        for pattern in _DATASET_PATH_PATTERNS:
            match = pattern.search(normalized)
            if match:
                result.add(match.group(1))
    return sorted(result)


def _source_sequence_ids(manifest: Mapping[str, Any]) -> list[str]:
    result = set()
    source = manifest.get("source", {})
    if isinstance(source, dict):
        sequence = source.get("sequence")
        if isinstance(sequence, str) and sequence:
            result.add(sequence)
        for item in source.get("datasets", []):
            if isinstance(item, dict) and isinstance(item.get("sequence"), str):
                result.add(item["sequence"])
    sources = manifest.get("sources", [])
    if isinstance(sources, list):
        for item in sources:
            if isinstance(item, dict) and isinstance(
                    item.get("sequence_id"), str):
                result.add(item["sequence_id"])
    return sorted(result)


def _manifest_contract(manifest: Optional[dict], error: Optional[str]) -> str:
    if manifest is None:
        return "missing" if error == "missing" else "invalid_json"
    try:
        validate_dataset_manifest(manifest)
    except ManifestError:
        return "historical"
    return "strict_v2"


def _artifact_uri_if_file(paths: ProjectPaths, path: Path) -> Optional[str]:
    return paths.artifact_uri(path) if path.is_file() else None


def _split_root(dataset: Path, role: str) -> Path:
    canonical = dataset / "splits" / role
    return canonical if canonical.is_dir() else dataset / role


class WorkspaceIndexBuilder:
    """Build dataset-to-run lineage from the current canonical workspace."""

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        now: Optional[Callable[[], datetime]] = None,
    ):
        self.paths = paths
        self._now = now or (lambda: datetime.now(timezone.utc))

    def build(self) -> dict:
        datasets = self._datasets()
        runs = self._training_runs()
        known_ids = {item["dataset_id"] for item in datasets}
        dataset_to_runs = {dataset_id: [] for dataset_id in sorted(known_ids)}
        unresolved = []
        for run in runs:
            for dataset_id in run["dataset_ids"]:
                if dataset_id in dataset_to_runs:
                    dataset_to_runs[dataset_id].append(run["uri"])
                else:
                    unresolved.append({
                        "dataset_id": dataset_id,
                        "run_uri": run["uri"],
                    })
        return {
            "schema_version": 1,
            "kind": "workspace_asset_index",
            "generated_at": self._now().isoformat(),
            "policy": {
                "read_only_scan": True,
                "hash_payload_files": False,
                "move_or_delete_assets": False,
            },
            "datasets": datasets,
            "training_runs": runs,
            "reverse_index": {
                "dataset_to_runs": dataset_to_runs,
                "unresolved_dataset_references": unresolved,
            },
        }

    def _datasets(self) -> list[dict]:
        root = self.paths.data_root / "processed" / "real"
        if not root.is_dir():
            return []
        result = []
        for dataset in sorted(
                (item for item in root.iterdir() if item.is_dir()),
                key=lambda item: item.name):
            manifest_path = next(
                (dataset / name for name in (
                    "manifest.json", "dataset_manifest.json",
                    "legacy_dataset_manifest.json")
                 if (dataset / name).is_file()),
                dataset / "dataset_manifest.json",
            )
            manifest, error = _read_json(manifest_path)
            result.append({
                "dataset_id": dataset.name,
                "uri": self.paths.artifact_uri(dataset),
                "manifest_uri": _artifact_uri_if_file(
                    self.paths, manifest_path),
                "manifest_schema_version": (
                    manifest.get("schema_version") if manifest else None),
                "manifest_contract": _manifest_contract(manifest, error),
                "manifest_error": error,
                "source_sequence_ids": (
                    _source_sequence_ids(manifest) if manifest else []),
                "split_files": {
                    role: len(list(_split_root(dataset, role).glob("*.npz")))
                    if _split_root(dataset, role).is_dir() else 0
                    for role in ("train", "val", "test")
                },
            })
        return result

    @staticmethod
    def _is_run_config(path: Path) -> bool:
        return (path.name == "config.json" and
                (path.parent.name.startswith("exp_") or
                 path.parent.name.startswith("trial_")))

    @staticmethod
    def _completion_evidence(run: Path) -> list[str]:
        candidates = (
            "artifacts.json", "RUN_COMPLETE", "COMPLETE", "evaluations/COMPLETE")
        return [name for name in candidates if (run / name).is_file()]

    def _training_runs(self) -> list[dict]:
        root = self.paths.runs_root / "training"
        if not root.is_dir():
            return []
        result = []
        for config_path in sorted(root.rglob("config.json")):
            if not self._is_run_config(config_path):
                continue
            run = config_path.parent
            relative = run.relative_to(root)
            config, error = _read_json(config_path)
            evidence = self._completion_evidence(run)
            manifest_path = next(
                (run / name for name in
                 ("run_manifest.json", "legacy_run_manifest.json")
                 if (run / name).is_file()),
                run / "run_manifest.json",
            )
            result.append({
                "run_id": run.name,
                "study_path": "/".join(relative.parts[:-1]),
                "uri": self.paths.artifact_uri(run),
                "config_uri": self.paths.artifact_uri(config_path),
                "config_error": error,
                "manifest_uri": _artifact_uri_if_file(
                    self.paths, manifest_path),
                "dataset_ids": (
                    _dataset_ids_from_config(config) if config else []),
                "completion_evidence": evidence,
                "operational_status": "complete" if evidence else "unverified",
            })
        return sorted(
            result, key=lambda item: (item["study_path"], item["run_id"]))


def write_workspace_index(
    paths: ProjectPaths,
    index: Mapping[str, Any],
    *,
    target: Optional[str | Path] = None,
) -> Path:
    destination = Path(target) if target is not None else (
        paths.registry_root / "workspace_asset_index.json")
    paths.artifact_uri(destination)
    return atomic_write_json(destination, index, overwrite=True)


def _legacy_stage_records(paths: ProjectPaths, run: Path, config: dict) -> list[dict]:
    artifacts, _ = _read_json(run / "artifacts.json")
    artifact_stages = artifacts.get("stages", {}) if artifacts else {}
    configured = config.get("training", {}).get("stages", {})
    names = sorted(set(configured) | set(artifact_stages))
    result = []
    for name in names:
        item = artifact_stages.get(name, {})
        checkpoint = item.get("best_checkpoint") if isinstance(item, dict) else None
        checkpoint_path = run / checkpoint if isinstance(checkpoint, str) else None
        checkpoint_uri = (
            paths.artifact_uri(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.is_file() else None)
        result.append({
            "name": name,
            "configured": configured.get(name),
            "selected_checkpoint_uri": checkpoint_uri,
            "selection_basis": (
                "validation" if checkpoint and checkpoint.endswith("best_eval_model.pt")
                else "training_loss_or_unknown"),
        })
    return result


def _observed_sequence_ids(dataset: Path) -> list[str]:
    result = set()
    for role in ("train", "val", "test"):
        split = _split_root(dataset, role)
        if not split.is_dir():
            continue
        for path in split.glob("*.npz"):
            match = _SEQUENCE_FILENAME.match(path.name)
            if match:
                result.add(match.group(1))
    return sorted(result)


def write_missing_legacy_dataset_manifests(
    paths: ProjectPaths,
    index: Mapping[str, Any],
) -> list[Path]:
    """Describe manifest-less datasets from filenames without reading payloads."""
    written = []
    for item in index.get("datasets", []):
        if item.get("manifest_uri") is not None:
            continue
        dataset = paths.resolve_artifact_uri(item["uri"])
        target = dataset / "legacy_dataset_manifest.json"
        if target.exists():
            continue
        sequence_ids = _observed_sequence_ids(dataset)
        splits = {
            role: [
                {"uri": paths.artifact_uri(path)}
                for path in sorted((dataset / role).glob("*.npz"))
            ] if (dataset / role).is_dir() else []
            for role in ("train", "val", "test")
        }
        readme = dataset / "README.md"
        value = {
            "schema_version": 1,
            "kind": "legacy_processed_dataset",
            "dataset_id": item["dataset_id"],
            "provenance": {
                "mode": "observed_existing_files",
                "hash_payload_files": False,
                "open_payload_files": False,
            },
            "source_sequence_ids": sequence_ids,
            "lineage_basis": "npz_filename" if sequence_ids else "unknown",
            "splits": splits,
            "evidence": {
                "readme_uri": _artifact_uri_if_file(paths, readme),
            },
            "lifecycle": {
                "training_ready": "unknown",
                "immutable_contract": "unverified",
            },
        }
        written.append(atomic_write_json(target, value, overwrite=False))
    return written


def write_mainline_legacy_manifests(
    paths: ProjectPaths,
    index: Mapping[str, Any],
) -> list[Path]:
    """Backfill non-overwriting manifests for existing real-pipeline trials."""
    written = []
    for item in index.get("training_runs", []):
        if not item.get("study_path", "").startswith("real_pipeline/"):
            continue
        run = paths.resolve_artifact_uri(item["uri"])
        target = run / "legacy_run_manifest.json"
        if target.exists():
            continue
        config, error = _read_json(run / "config.json")
        if config is None:
            raise ValueError(f"无法读取主线试次 config: {run}; {error}")
        dataset_records = []
        for dataset_id in item.get("dataset_ids", []):
            dataset = paths.processed_dataset("real", dataset_id)
            manifest = next(
                (dataset / name for name in ("manifest.json", "dataset_manifest.json")
                 if (dataset / name).is_file()),
                None,
            )
            dataset_records.append({
                "dataset_id": dataset_id,
                "manifest_uri": (
                    paths.artifact_uri(manifest) if manifest else None),
            })
        trial = config.get("trial", {})
        value = {
            "schema_version": 1,
            "kind": "legacy_training_run",
            "run_id": item["run_id"],
            "study_path": item["study_path"],
            "created_at": trial.get("created_at"),
            "operational_status": item["operational_status"],
            "provenance": {
                "mode": "observed_existing_files",
                "hash_payload_files": False,
                "config_uri": item["config_uri"],
            },
            "datasets": dataset_records,
            "stages": _legacy_stage_records(paths, run, config),
            "evidence": {
                name: paths.artifact_uri(run / name)
                for name in item.get("completion_evidence", [])
            },
            "commands_uri": _artifact_uri_if_file(paths, run / "commands.sh"),
            "status_uri": _artifact_uri_if_file(paths, run / "status.txt"),
        }
        written.append(atomic_write_json(target, value, overwrite=False))
    return written
