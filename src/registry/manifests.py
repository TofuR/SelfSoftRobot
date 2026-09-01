"""Dataset/run manifest contracts and atomic persistence.

The registry deliberately validates a small project-owned schema instead of
silently accepting arbitrary dictionaries. Historical schema-v1 manifests are
read by the inventory/import tooling; new published artifacts use these
contracts and portable ``artifact://`` references.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Optional

from .paths import ARTIFACT_SCHEME, ProjectPaths


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")


class ManifestError(ValueError):
    """Manifest violates the project-owned artifact contract."""


def _fail(message: str) -> None:
    raise ManifestError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{field} 必须是 object")
    return value


def _list(value: Any, field: str) -> list:
    if not isinstance(value, list):
        _fail(f"{field} 必须是 array")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        _fail(f"{field} 不是合法 ID: {value!r}")
    return value


def _iso_datetime(value: Any, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} 必须是 ISO datetime")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{field} 必须是 ISO datetime: {value!r}")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _fail(f"{field} 必须是小写 sha256: {value!r}")
    return value


def _artifact_uri(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.startswith(ARTIFACT_SCHEME):
        _fail(f"{field} 必须是 artifact URI: {value!r}")
    relative = PurePosixPath(value[len(ARTIFACT_SCHEME):])
    if (not relative.parts or relative.is_absolute() or
            any(part in ("", ".", "..") for part in relative.parts)):
        _fail(f"{field} 含非法 artifact URI: {value!r}")
    return value


def _choice(value: Any, field: str, choices) -> str:
    if value not in choices:
        _fail(f"{field} 必须是 {sorted(choices)} 之一: {value!r}")
    return value


def _string_list(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    items = _list(value, field)
    if nonempty and not items:
        _fail(f"{field} 不能为空")
    if not all(isinstance(item, str) and item for item in items):
        _fail(f"{field} 必须只含非空字符串")
    return items


def _file_record(value: Any, field: str) -> Mapping[str, Any]:
    record = _mapping(value, field)
    _artifact_uri(record.get("uri"), f"{field}.uri")
    _sha256(record.get("sha256"), f"{field}.sha256")
    size = record.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _fail(f"{field}.bytes 必须是非负整数")
    return record


def validate_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate an immutable processed-dataset manifest (schema v2)."""
    value = _mapping(manifest, "manifest")
    if value.get("schema_version") != 2:
        _fail("dataset manifest schema_version 必须为 2")
    if value.get("kind") != "dataset":
        _fail("dataset manifest kind 必须为 'dataset'")
    _identifier(value.get("dataset_id"), "dataset_id")
    _iso_datetime(value.get("created_at"), "created_at")
    status = _choice(
        value.get("status"), "status", {"draft", "released", "deprecated"})

    sources = _list(value.get("sources"), "sources")
    if not sources:
        _fail("sources 不能为空")
    for index, source_value in enumerate(sources):
        source = _mapping(source_value, f"sources[{index}]")
        _identifier(source.get("sequence_id"), f"sources[{index}].sequence_id")
        raw_hash = source.get("raw_manifest_sha256")
        if raw_hash is not None:
            _sha256(raw_hash, f"sources[{index}].raw_manifest_sha256")
        if status == "released" and raw_hash is None:
            _fail("released dataset 的每个 source 都必须有 raw_manifest_sha256")

    recipe = _mapping(value.get("recipe"), "recipe")
    _identifier(recipe.get("name"), "recipe.name")
    version = recipe.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        _fail("recipe.version 必须是正整数")
    commit = recipe.get("git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT.fullmatch(commit):
        _fail("recipe.git_commit 必须是 7-40 位小写十六进制 commit")
    _mapping(recipe.get("parameters"), "recipe.parameters")
    _string_list(recipe.get("commands"), "recipe.commands", nonempty=True)

    contracts = _mapping(value.get("contracts"), "contracts")
    for name in ("state", "action", "timing", "observation"):
        _mapping(contracts.get(name), f"contracts.{name}")

    split_policy = _mapping(value.get("split_policy"), "split_policy")
    _identifier(split_policy.get("name"), "split_policy.name")
    group_key = split_policy.get("group_key")
    if not isinstance(group_key, str) or not group_key:
        _fail("split_policy.group_key 必须是非空字符串")
    embargo = split_policy.get("embargo_frames")
    if not isinstance(embargo, int) or isinstance(embargo, bool) or embargo < 0:
        _fail("split_policy.embargo_frames 必须是非负整数")
    seed = split_policy.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        _fail("split_policy.seed 必须是 int 或 null")
    _choice(
        split_policy.get("evidence_level"), "split_policy.evidence_level",
        {"smoke", "within_sequence", "cross_sequence"})

    files = _list(value.get("files"), "files")
    file_uris = set()
    for index, record_value in enumerate(files):
        record = _file_record(record_value, f"files[{index}]")
        uri = record["uri"]
        if uri in file_uris:
            _fail(f"files 含重复 uri: {uri}")
        file_uris.add(uri)

    splits = _mapping(value.get("splits"), "splits")
    seen_split_uris = set()
    for role in ("train", "val", "test"):
        entries = _list(splits.get(role), f"splits.{role}")
        for index, entry_value in enumerate(entries):
            entry = _mapping(entry_value, f"splits.{role}[{index}]")
            uri = _artifact_uri(
                entry.get("uri"), f"splits.{role}[{index}].uri")
            _sha256(entry.get("sha256"), f"splits.{role}[{index}].sha256")
            frames = entry.get("frames")
            if (not isinstance(frames, int) or isinstance(frames, bool) or
                    frames < 0):
                _fail(f"splits.{role}[{index}].frames 必须是非负整数")
            if uri in seen_split_uris:
                _fail(f"split 之间复用了同一文件: {uri}")
            if uri not in file_uris:
                _fail(f"split 文件未登记到 files: {uri}")
            seen_split_uris.add(uri)

    _mapping(value.get("quality_control"), "quality_control")


def validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate a reproducible training-run manifest (schema v1 or v2)."""
    value = _mapping(manifest, "manifest")
    schema_version = value.get("schema_version")
    if schema_version == 1:
        _validate_run_manifest_v1(value)
        return
    if schema_version == 2:
        _validate_run_manifest_v2(value)
        return
    _fail("run manifest schema_version 必须为 1 或 2")


def _validate_run_manifest_v1(value: Mapping[str, Any]) -> None:
    """Validate the original strict run contract kept for compatibility."""
    if value.get("kind") != "training_run":
        _fail("run manifest kind 必须为 'training_run'")
    _identifier(value.get("run_id"), "run_id")
    _identifier(value.get("study_id"), "study_id")
    _iso_datetime(value.get("created_at"), "created_at")
    status = _choice(
        value.get("status"), "status",
        {"planned", "running", "complete", "failed"})
    _choice(value.get("run_kind"), "run_kind", {"exploratory", "formal"})

    dataset = _mapping(value.get("dataset"), "dataset")
    _artifact_uri(dataset.get("manifest_uri"), "dataset.manifest_uri")
    _sha256(dataset.get("manifest_sha256"), "dataset.manifest_sha256")

    source = _mapping(value.get("source"), "source")
    commit = source.get("git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT.fullmatch(commit):
        _fail("source.git_commit 必须是 7-40 位小写十六进制 commit")
    if not isinstance(source.get("dirty"), bool):
        _fail("source.dirty 必须是 bool")
    patch_uri = source.get("patch_uri")
    if source["dirty"] and patch_uri is None:
        _fail("dirty run 必须保存 source.patch_uri")
    if patch_uri is not None:
        _artifact_uri(patch_uri, "source.patch_uri")

    _string_list(value.get("commands"), "commands", nonempty=True)
    _artifact_uri(value.get("resolved_config_uri"), "resolved_config_uri")
    _artifact_uri(value.get("environment_uri"), "environment_uri")
    seed = value.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail("seed 必须是 int")

    stages = _list(value.get("stages"), "stages")
    if not stages:
        _fail("stages 不能为空")
    stage_names = set()
    for index, stage_value in enumerate(stages):
        stage = _mapping(stage_value, f"stages[{index}]")
        name = _identifier(stage.get("name"), f"stages[{index}].name")
        if name in stage_names:
            _fail(f"stages 含重复 name: {name}")
        stage_names.add(name)

    selection = _mapping(value.get("selection"), "selection")
    metric = selection.get("metric")
    if not isinstance(metric, str) or not metric:
        _fail("selection.metric 必须是非空字符串")
    _choice(selection.get("mode"), "selection.mode", {"min", "max"})
    if selection.get("dataset_role") != "val":
        _fail("selection.dataset_role 必须为 'val'")
    _artifact_uri(selection.get("checkpoint_uri"), "selection.checkpoint_uri")

    expected = _string_list(
        value.get("expected_artifacts"), "expected_artifacts", nonempty=True)
    for index, uri in enumerate(expected):
        _artifact_uri(uri, f"expected_artifacts[{index}]")
    if status == "complete":
        _artifact_uri(value.get("complete_marker_uri"), "complete_marker_uri")

def _validate_run_manifest_v2(value: Mapping[str, Any]) -> None:
    """Validate the lightweight stage-aware run contract used by new trials."""
    if value.get("kind") != "training_run":
        _fail("run manifest kind 必须为 'training_run'")
    _identifier(value.get("run_id"), "run_id")
    _identifier(value.get("study_id"), "study_id")
    _iso_datetime(value.get("created_at"), "created_at")
    status = _choice(
        value.get("status"), "status",
        {"planned", "running", "complete", "failed"})
    _choice(value.get("run_kind"), "run_kind", {"exploratory", "formal"})

    dataset = _mapping(value.get("dataset"), "dataset")
    _identifier(dataset.get("dataset_id"), "dataset.dataset_id")
    _artifact_uri(dataset.get("manifest_uri"), "dataset.manifest_uri")

    source = _mapping(value.get("source"), "source")
    commit = source.get("git_commit")
    if not isinstance(commit, str) or not _GIT_COMMIT.fullmatch(commit):
        _fail("source.git_commit 必须是 7-40 位小写十六进制 commit")
    if not isinstance(source.get("dirty"), bool):
        _fail("source.dirty 必须是 bool")

    _artifact_uri(value.get("commands_uri"), "commands_uri")
    _artifact_uri(value.get("resolved_config_uri"), "resolved_config_uri")
    seed = value.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        _fail("seed 必须是 int")

    stages = _list(value.get("stages"), "stages")
    if not stages:
        _fail("stages 不能为空")
    stage_names = set()
    for index, stage_value in enumerate(stages):
        stage = _mapping(stage_value, f"stages[{index}]")
        name = _identifier(stage.get("name"), f"stages[{index}].name")
        if name in stage_names:
            _fail(f"stages 含重复 name: {name}")
        stage_names.add(name)
        selection = _mapping(
            stage.get("selection"), f"stages[{index}].selection")
        metric = selection.get("metric")
        if not isinstance(metric, str) or not metric:
            _fail(f"stages[{index}].selection.metric 必须是非空字符串")
        _choice(
            selection.get("mode"), f"stages[{index}].selection.mode",
            {"min", "max"})
        if selection.get("dataset_role") != "val":
            _fail(f"stages[{index}].selection.dataset_role 必须为 'val'")
        _artifact_uri(
            selection.get("checkpoint_uri"),
            f"stages[{index}].selection.checkpoint_uri")

    expected = _string_list(
        value.get("expected_artifacts"), "expected_artifacts", nonempty=True)
    for index, uri in enumerate(expected):
        _artifact_uri(uri, f"expected_artifacts[{index}]")
    if status == "complete":
        _artifact_uri(value.get("complete_marker_uri"), "complete_marker_uri")

    final_evaluations = value.get("final_evaluations", [])
    if not isinstance(final_evaluations, list):
        _fail("final_evaluations 必须是 array")
    for index, evaluation_value in enumerate(final_evaluations):
        evaluation = _mapping(
            evaluation_value, f"final_evaluations[{index}]")
        _identifier(evaluation.get("stage"),
                    f"final_evaluations[{index}].stage")
        if evaluation.get("dataset_role") != "test":
            _fail(f"final_evaluations[{index}].dataset_role 必须为 'test'")
        _artifact_uri(
            evaluation.get("quantitative_uri"),
            f"final_evaluations[{index}].quantitative_uri")
        _artifact_uri(
            evaluation.get("overlay_uri"),
            f"final_evaluations[{index}].overlay_uri")
    if final_evaluations:
        _artifact_uri(value.get("offline_fixture_uri"), "offline_fixture_uri")
        _artifact_uri(value.get("deploy_manifest_uri"), "deploy_manifest_uri")


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正数")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_file_record(paths: ProjectPaths, path: str | os.PathLike[str]) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"manifest 文件不存在: {file_path}")
    return {
        "uri": paths.artifact_uri(file_path),
        "sha256": sha256_file(file_path),
        "bytes": file_path.stat().st_size,
    }


def atomic_write_json(
        target: Path, value: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    target = target.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖 manifest: {target}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # hard-link creation is the no-clobber commit point: unlike an
            # exists()+replace() pair, another writer cannot win between the
            # check and publication. The temporary file lives in the same dir.
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"拒绝覆盖 manifest: {target}") from exc
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


class ManifestStore:
    """Validate and atomically persist manifests under one workspace."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths

    def write_dataset(
        self,
        manifest: Mapping[str, Any],
        *,
        target: Optional[str | os.PathLike[str]] = None,
        overwrite: bool = False,
    ) -> Path:
        validate_dataset_manifest(manifest)
        dataset_id = manifest["dataset_id"]
        destination = Path(target) if target is not None else (
            self.paths.processed_dataset("real", dataset_id) / "manifest.json")
        self.paths.artifact_uri(destination)
        return atomic_write_json(destination, manifest, overwrite=overwrite)

    def write_run(
        self,
        manifest: Mapping[str, Any],
        *,
        target: Optional[str | os.PathLike[str]] = None,
        overwrite: bool = False,
    ) -> Path:
        validate_run_manifest(manifest)
        destination = Path(target) if target is not None else (
            self.paths.training_run(manifest["study_id"], manifest["run_id"]) /
            "run_manifest.json")
        self.paths.artifact_uri(destination)
        return atomic_write_json(destination, manifest, overwrite=overwrite)

    @staticmethod
    def read_dataset(path: str | os.PathLike[str]) -> dict:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        validate_dataset_manifest(value)
        return value

    @staticmethod
    def read_run(path: str | os.PathLike[str]) -> dict:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        validate_run_manifest(value)
        return value
