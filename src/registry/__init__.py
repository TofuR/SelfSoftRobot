"""Artifact path and manifest registry primitives."""

from .datasets import DATASET_ROLES, DatasetArtifact, DatasetSelector

from .manifests import (
    ManifestError,
    ManifestStore,
    atomic_write_json,
    build_file_record,
    sha256_file,
    validate_dataset_manifest,
    validate_run_manifest,
)
from .paths import LegacyRoots, ProjectPaths
from .real_assets import (
    LEGACY_DERIVED_RECIPE,
    LEGACY_MASK_REPAIR_RECIPE,
    SAM2_VIDEO_RECIPE,
    canonical_intermediate,
    canonical_output,
    resolve_candidate_masks,
    resolve_processed_dataset,
    resolve_raw_sequence,
    resolve_repaired_masks,
    resolve_sam2_masks,
)
from .workspace_index import (
    WorkspaceIndexBuilder,
    write_mainline_legacy_manifests,
    write_workspace_index,
)

__all__ = [
    "DATASET_ROLES",
    "DatasetArtifact",
    "DatasetSelector",
    "LegacyRoots",
    "LEGACY_DERIVED_RECIPE",
    "LEGACY_MASK_REPAIR_RECIPE",
    "ManifestError",
    "ManifestStore",
    "ProjectPaths",
    "SAM2_VIDEO_RECIPE",
    "WorkspaceIndexBuilder",
    "atomic_write_json",
    "build_file_record",
    "canonical_intermediate",
    "canonical_output",
    "resolve_candidate_masks",
    "resolve_processed_dataset",
    "resolve_raw_sequence",
    "resolve_repaired_masks",
    "resolve_sam2_masks",
    "sha256_file",
    "validate_dataset_manifest",
    "validate_run_manifest",
    "write_mainline_legacy_manifests",
    "write_workspace_index",
]
