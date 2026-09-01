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

__all__ = [
    "DATASET_ROLES",
    "DatasetArtifact",
    "DatasetSelector",
    "LegacyRoots",
    "ManifestError",
    "ManifestStore",
    "ProjectPaths",
    "atomic_write_json",
    "build_file_record",
    "sha256_file",
    "validate_dataset_manifest",
    "validate_run_manifest",
]
