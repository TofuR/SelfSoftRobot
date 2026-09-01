"""Resolve processed-dataset artifacts without application-owned copies.

Canonical workspace datasets are preferred.  During migration the selector
can also read registered legacy roots, but it never writes to those roots.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

from .manifests import sha256_file
from .paths import ARTIFACT_SCHEME, ProjectPaths


DATASET_ROLES = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetArtifact:
    dataset_id: str
    role: str
    path: Path
    source: str
    manifest_path: Optional[Path] = None
    sha256: Optional[str] = None


class DatasetSelector:
    """Select NPZ files by dataset identity and split role."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def dataset_ids(self) -> tuple[str, ...]:
        ids = set()
        canonical = self.paths.data_root / "processed" / "real"
        roots = (canonical,) + self.paths.legacy.roots_for("processed")
        for root in roots:
            if root.is_dir():
                ids.update(path.name for path in root.iterdir() if path.is_dir())
        return tuple(sorted(ids))

    def artifacts(self, dataset_id: str, role: str) -> tuple[DatasetArtifact, ...]:
        if role not in DATASET_ROLES:
            raise ValueError(f"dataset role 必须是 {DATASET_ROLES} 之一: {role!r}")
        canonical = self.paths.processed_dataset("real", dataset_id)
        roots = (("canonical", canonical),) + tuple(
            ("legacy", root / dataset_id)
            for root in self.paths.legacy.roots_for("processed"))
        for source, root in roots:
            values = self._artifacts_from_root(dataset_id, role, source, root)
            if values:
                return tuple(values)
        return ()

    def resolve(
        self,
        dataset_id: str,
        role: str,
        *,
        filename: Optional[str] = None,
        verify_hash: bool = True,
    ) -> DatasetArtifact:
        values = self.artifacts(dataset_id, role)
        if filename is not None:
            values = tuple(item for item in values if item.path.name == filename)
        if not values:
            suffix = f", filename={filename!r}" if filename else ""
            raise FileNotFoundError(
                f"找不到已注册 dataset artifact: dataset={dataset_id!r}, "
                f"role={role!r}{suffix}")
        artifact = values[0]
        if verify_hash and artifact.sha256 is not None:
            actual = sha256_file(artifact.path)
            if actual != artifact.sha256:
                raise ValueError(
                    f"dataset artifact hash 不匹配: {artifact.path}; "
                    f"expected={artifact.sha256}, actual={actual}")
        return artifact

    def _artifacts_from_root(
        self,
        dataset_id: str,
        role: str,
        source: str,
        root: Path,
    ) -> list[DatasetArtifact]:
        if not root.is_dir():
            return []
        for name in ("manifest.json", "dataset_manifest.json"):
            manifest_path = root / name
            if manifest_path.is_file():
                values = self._artifacts_from_manifest(
                    dataset_id, role, source, root, manifest_path)
                if values:
                    return values
        split_root = root / "splits" / role
        if not split_root.is_dir():
            split_root = root / role
        return [
            DatasetArtifact(dataset_id, role, path, source)
            for path in sorted(split_root.glob("*.npz")) if path.is_file()
        ]

    def _artifacts_from_manifest(
        self,
        dataset_id: str,
        role: str,
        source: str,
        root: Path,
        manifest_path: Path,
    ) -> list[DatasetArtifact]:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        splits = manifest.get("splits", {})
        entries = splits.get(role, []) if isinstance(splits, dict) else []
        if not isinstance(entries, list):
            return []
        result = []
        for entry in entries:
            if isinstance(entry, str):
                reference, expected_hash = entry, None
            elif isinstance(entry, dict):
                reference = entry.get("uri") or entry.get("path")
                expected_hash = entry.get("sha256")
            else:
                continue
            path = self._resolve_reference(reference, root, role)
            if path is None or not path.is_file() or path.suffix != ".npz":
                continue
            result.append(DatasetArtifact(
                dataset_id=dataset_id,
                role=role,
                path=path,
                source=source,
                manifest_path=manifest_path,
                sha256=expected_hash,
            ))
        return sorted(result, key=lambda item: item.path.name)

    def _resolve_reference(
        self, reference: object, root: Path, role: str,
    ) -> Optional[Path]:
        if not isinstance(reference, str) or not reference:
            return None
        if reference.startswith(ARTIFACT_SCHEME):
            candidate = self.paths.resolve_artifact_uri(reference)
        else:
            candidate = Path(reference).expanduser()
            if not candidate.is_absolute():
                candidate = self.paths.repo_root / candidate
        if candidate.is_file():
            return candidate.resolve()
        # Historical manifests often contain machine-specific absolute paths.
        # After an in-place migration the basename and role remain authoritative.
        basename = Path(reference).name
        for split_root in (root / "splits" / role, root / role):
            fallback = split_root / basename
            if fallback.is_file():
                return fallback.resolve()
        return candidate.resolve(strict=False)
