"""统一解析源码仓库、运行工作区和迁移期历史路径。

解析优先级固定为显式参数、环境变量、本机 TOML、仓库默认值。这个模块只
返回路径；除 ``create_workspace_layout`` 和 ``create_new_directory`` 外不会
隐式创建目录，避免一次只读查询意外改变仓库或外部磁盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Tuple

try:  # Python 3.11+
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib


WORKSPACE_ENV = "SSR_WORKSPACE_ROOT"
CONFIG_ENV = "SSR_PATHS_CONFIG"
ARTIFACT_SCHEME = "artifact://"
REPO_SCHEME = "repo://"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _absolute_from_repo(value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _validate_parts(parts: Iterable[str]) -> Tuple[str, ...]:
    result = tuple(str(part) for part in parts)
    for part in result:
        pure = PurePosixPath(part)
        if (not part or pure.is_absolute() or len(pure.parts) != 1 or
                part in (".", "..") or "\\" in part):
            raise ValueError(f"非法路径分段: {part!r}")
    return result


def _load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if value.get("schema_version") != 1:
        raise ValueError(
            f"路径配置 schema_version 必须为 1: {path}")
    return value


def _tuple_roots(values, repo_root: Path) -> Tuple[Path, ...]:
    if values is None:
        return ()
    if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values):
        raise ValueError("compat root 必须是字符串数组")
    return tuple(_absolute_from_repo(item, repo_root) for item in values)


@dataclass(frozen=True)
class LegacyRoots:
    """迁移期只读根。这里的路径永远不能作为新产物默认写入目标。"""

    enabled: bool = True
    raw: Tuple[Path, ...] = field(default_factory=tuple)
    intermediate: Tuple[Path, ...] = field(default_factory=tuple)
    processed: Tuple[Path, ...] = field(default_factory=tuple)
    training: Tuple[Path, ...] = field(default_factory=tuple)
    validation: Tuple[Path, ...] = field(default_factory=tuple)
    analysis: Tuple[Path, ...] = field(default_factory=tuple)

    def roots_for(self, kind: str) -> Tuple[Path, ...]:
        if kind not in {
                "raw", "intermediate", "processed", "training",
                "validation", "analysis"}:
            raise KeyError(f"未知 legacy kind: {kind}")
        return getattr(self, kind) if self.enabled else ()


@dataclass(frozen=True)
class ProjectPaths:
    """SelfSoftRobot 的唯一项目路径合同。"""

    repo_root: Path
    workspace_root: Path
    legacy: LegacyRoots = field(default_factory=LegacyRoots)

    @classmethod
    def load(
        cls,
        *,
        workspace_root: Optional[str | os.PathLike[str]] = None,
        config_path: Optional[str | os.PathLike[str]] = None,
        repo_root: Optional[str | os.PathLike[str]] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "ProjectPaths":
        repo = Path(repo_root or _default_repo_root()).expanduser().resolve()
        env = os.environ if environ is None else environ

        selected_config = config_path or env.get(CONFIG_ENV)
        if selected_config is None:
            local = repo / "config" / "paths.local.toml"
            selected_config = local if local.is_file() else None

        config = {}
        if selected_config is not None:
            config_file = _absolute_from_repo(selected_config, repo)
            if not config_file.is_file():
                raise FileNotFoundError(f"路径配置不存在: {config_file}")
            config = _load_toml(config_file)

        configured_root = config.get("paths", {}).get(
            "workspace_root", "workspace")
        if not isinstance(configured_root, str) or not configured_root:
            raise ValueError("paths.workspace_root 必须是非空字符串")
        selected_root = (
            workspace_root or env.get(WORKSPACE_ENV) or configured_root)
        workspace = _absolute_from_repo(selected_root, repo)

        compat = config.get("compat", {})
        if not isinstance(compat, dict):
            raise ValueError("compat 必须是 TOML table")
        enabled = compat.get("enable_legacy_reads", True)
        if not isinstance(enabled, bool):
            raise ValueError("compat.enable_legacy_reads 必须是 bool")

        defaults = {
            "raw_roots": ["real_capture/data/raw"],
            "intermediate_roots": ["real_capture/data/derived", "sam2/masks"],
            "processed_roots": ["data/real_seq"],
            "training_roots": ["train_log"],
            "validation_roots": ["real_validation/runs"],
            "analysis_roots": ["output"],
        }
        legacy = LegacyRoots(
            enabled=enabled,
            raw=_tuple_roots(compat.get("raw_roots", defaults["raw_roots"]), repo),
            intermediate=_tuple_roots(
                compat.get("intermediate_roots", defaults["intermediate_roots"]), repo),
            processed=_tuple_roots(
                compat.get("processed_roots", defaults["processed_roots"]), repo),
            training=_tuple_roots(
                compat.get("training_roots", defaults["training_roots"]), repo),
            validation=_tuple_roots(
                compat.get("validation_roots", defaults["validation_roots"]), repo),
            analysis=_tuple_roots(
                compat.get("analysis_roots", defaults["analysis_roots"]), repo),
        )
        return cls(repo_root=repo, workspace_root=workspace, legacy=legacy)

    @property
    def data_root(self) -> Path:
        return self.workspace_root / "data"

    @property
    def runs_root(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def registry_root(self) -> Path:
        return self.workspace_root / "registry"

    def raw_sequence(self, domain: str, sequence_id: str) -> Path:
        domain, sequence_id = _validate_parts((domain, sequence_id))
        return self.data_root / "raw" / domain / sequence_id

    def intermediate_sequence(
            self, domain: str, sequence_id: str, recipe_id: str) -> Path:
        domain, sequence_id, recipe_id = _validate_parts(
            (domain, sequence_id, recipe_id))
        return (self.data_root / "intermediate" / domain /
                sequence_id / recipe_id)

    def processed_dataset(self, domain: str, dataset_id: str) -> Path:
        domain, dataset_id = _validate_parts((domain, dataset_id))
        return self.data_root / "processed" / domain / dataset_id

    def fixture(self, fixture_id: str) -> Path:
        (fixture_id,) = _validate_parts((fixture_id,))
        return self.data_root / "fixtures" / fixture_id

    def training_run(self, study_id: str, run_id: str) -> Path:
        study_id, run_id = _validate_parts((study_id, run_id))
        return self.runs_root / "training" / study_id / run_id

    def training_study(self, study_id: str) -> Path:
        """返回模型/研究级训练根，新试次应在此目录下原子编号。"""
        (study_id,) = _validate_parts((study_id,))
        return self.runs_root / "training" / study_id

    def validation_run(self, run_id: str) -> Path:
        (run_id,) = _validate_parts((run_id,))
        return self.runs_root / "validation" / run_id

    def analysis_run(self, analysis_id: str, run_id: str) -> Path:
        analysis_id, run_id = _validate_parts((analysis_id, run_id))
        return self.runs_root / "analysis" / analysis_id / run_id

    def pretrained_model_dir(self, model_id: str) -> Path:
        (model_id,) = _validate_parts((model_id,))
        return self.workspace_root / "models" / "pretrained" / model_id

    def artifact_uri(self, path: str | os.PathLike[str]) -> str:
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            relative = resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"路径不在 workspace 内: {resolved}") from exc
        return ARTIFACT_SCHEME + relative.as_posix()

    def resolve_artifact_uri(self, uri: str) -> Path:
        if not isinstance(uri, str) or not uri.startswith(ARTIFACT_SCHEME):
            raise ValueError(f"不是 artifact URI: {uri!r}")
        relative = PurePosixPath(uri[len(ARTIFACT_SCHEME):])
        if (not relative.parts or relative.is_absolute() or
                any(part in ("", ".", "..") for part in relative.parts)):
            raise ValueError(f"非法 artifact URI: {uri!r}")
        resolved = self.workspace_root.joinpath(*relative.parts).resolve(
            strict=False)
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"artifact URI 越过 workspace: {uri!r}") from exc
        return resolved

    def repo_uri(self, path: str | os.PathLike[str]) -> str:
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            relative = resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"路径不在源码仓库内: {resolved}") from exc
        return REPO_SCHEME + relative.as_posix()

    def resolve_repo_uri(self, uri: str) -> Path:
        if not isinstance(uri, str) or not uri.startswith(REPO_SCHEME):
            raise ValueError(f"不是 repo URI: {uri!r}")
        relative = PurePosixPath(uri[len(REPO_SCHEME):])
        if (not relative.parts or relative.is_absolute() or
                any(part in ("", ".", "..") for part in relative.parts)):
            raise ValueError(f"非法 repo URI: {uri!r}")
        resolved = self.repo_root.joinpath(*relative.parts).resolve(strict=False)
        try:
            resolved.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"repo URI 越过源码仓库: {uri!r}") from exc
        return resolved

    def legacy_candidates(
        self,
        kind: str,
        *parts: str,
        existing_only: bool = True,
    ) -> Tuple[Path, ...]:
        safe_parts = _validate_parts(parts)
        candidates = tuple(
            root.joinpath(*safe_parts) for root in self.legacy.roots_for(kind))
        if existing_only:
            candidates = tuple(path for path in candidates if path.exists())
        return candidates

    def create_workspace_layout(self) -> Tuple[Path, ...]:
        """显式创建固定顶层目录；可重复调用，不创建任何数据集或 run。"""
        directories = (
            self.data_root / "raw",
            self.data_root / "intermediate",
            self.data_root / "processed",
            self.data_root / "fixtures",
            self.runs_root / "training",
            self.runs_root / "validation",
            self.runs_root / "analysis",
            self.workspace_root / "models" / "pretrained",
            self.workspace_root / "reports",
            self.workspace_root / "cache",
            self.registry_root,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        return directories

    def create_new_directory(self, path: str | os.PathLike[str]) -> Path:
        """在 workspace 内原子占用新目标；已存在或越界均拒绝。"""
        target = Path(path).expanduser().resolve(strict=False)
        try:
            target.relative_to(self.workspace_root)
        except ValueError as exc:
            raise ValueError(f"拒绝在 workspace 外创建运行目录: {target}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir()
        return target
