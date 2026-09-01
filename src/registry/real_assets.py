"""Canonical/legacy resolution helpers for real-data maintenance tools."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .paths import ProjectPaths


LEGACY_DERIVED_RECIPE = "legacy-derived"
SAM2_VIDEO_RECIPE = "sam2-video-v1"


def _explicit_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def resolve_raw_sequence(
    paths: ProjectPaths,
    value: str | Path,
    *,
    camera: Optional[str] = None,
) -> Path:
    """Resolve an explicit path or a sequence ID, preferring canonical raw."""
    explicit = _explicit_path(value, paths.repo_root)
    if explicit.is_dir():
        result = explicit
    else:
        sequence_id = Path(value).name
        canonical = paths.raw_sequence("real", sequence_id)
        candidates = (canonical,) + paths.legacy_candidates(
            "raw", sequence_id, existing_only=True)
        result = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if result is None:
            raise FileNotFoundError(f"找不到 raw sequence: {value}")
    if camera is not None and not (result / camera).is_dir():
        raise FileNotFoundError(f"raw sequence 缺少 {camera}: {result}")
    return result.resolve()


def resolve_processed_dataset(paths: ProjectPaths, value: str | Path) -> Path:
    """Resolve an explicit dataset root or dataset ID, canonical first."""
    explicit = _explicit_path(value, paths.repo_root)
    if explicit.is_dir():
        return explicit
    dataset_id = Path(value).name
    canonical = paths.processed_dataset("real", dataset_id)
    candidates = (canonical,) + paths.legacy_candidates(
        "processed", dataset_id, existing_only=True)
    result = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if result is None:
        raise FileNotFoundError(f"找不到 processed dataset: {value}")
    return result.resolve()


def canonical_output(paths: ProjectPaths, path: str | Path) -> Path:
    """Validate that a new output target is inside the configured workspace."""
    target = _explicit_path(path, paths.repo_root)
    paths.artifact_uri(target)
    return target


def canonical_intermediate(
    paths: ProjectPaths,
    sequence_id: str,
    recipe_id: str,
) -> Path:
    return paths.intermediate_sequence("real", sequence_id, recipe_id)


def _canonical_recipe_roots(paths: ProjectPaths, sequence_id: str) -> tuple[Path, ...]:
    root = paths.data_root / "intermediate" / "real" / sequence_id
    if not root.is_dir():
        return ()
    return tuple(sorted((path for path in root.iterdir() if path.is_dir()),
                        key=lambda path: path.name))


def _first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    return next((path.resolve() for path in candidates if path.is_dir()), None)


def resolve_candidate_masks(paths: ProjectPaths, sequence_id: str) -> Path:
    canonical = []
    for recipe in _canonical_recipe_roots(paths, sequence_id):
        canonical.extend((recipe / "masks", recipe / "masks_candidate"))
    legacy = []
    for root in paths.legacy.roots_for("intermediate"):
        legacy.append(root / sequence_id / "masks")
    result = _first_existing((*canonical, *legacy))
    if result is None:
        raise FileNotFoundError(f"找不到 candidate masks: {sequence_id}")
    return result


def resolve_repaired_masks(paths: ProjectPaths, sequence_id: str) -> Path:
    canonical = [
        recipe / "masks_repaired"
        for recipe in _canonical_recipe_roots(paths, sequence_id)
    ]
    legacy = [
        root / sequence_id / "masks_repaired"
        for root in paths.legacy.roots_for("intermediate")
    ]
    result = _first_existing((*canonical, *legacy))
    if result is None:
        raise FileNotFoundError(f"找不到 repaired masks: {sequence_id}")
    return result


def resolve_sam2_masks(paths: ProjectPaths, sequence_id: str) -> Path:
    canonical_sequence = (
        paths.data_root / "intermediate" / "real" / sequence_id)
    canonical = [
        canonical_sequence / SAM2_VIDEO_RECIPE,
        *(recipe / "sam2_masks"
          for recipe in _canonical_recipe_roots(paths, sequence_id)),
    ]
    legacy = []
    for root in paths.legacy.roots_for("intermediate"):
        legacy.extend((root / f"{sequence_id}_full", root / sequence_id / "sam2_masks"))
    result = _first_existing((*canonical, *legacy))
    if result is None:
        raise FileNotFoundError(f"找不到 SAM2 masks: {sequence_id}")
    return result
