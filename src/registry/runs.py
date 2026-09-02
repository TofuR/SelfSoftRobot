"""Run directory allocation helpers for workspace-owned artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from .paths import ProjectPaths


def allocate_numbered_run(
    base_dir: Path,
    *,
    prefix: str = "run",
    now: datetime | None = None,
) -> Path:
    """Atomically allocate ``<prefix>_YYYYMMDD_NNN`` below ``base_dir``."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError(f"非法 run 前缀: {prefix!r}")
    current = now or datetime.now()
    date = current.strftime("%Y%m%d")
    pattern = re.compile(rf"^{re.escape(prefix)}_{date}_(\d+)$")
    base_dir.mkdir(parents=True, exist_ok=True)
    indices = [
        int(match.group(1))
        for path in base_dir.iterdir()
        if (match := pattern.fullmatch(path.name))
    ]
    index = max(indices, default=-1) + 1
    while True:
        target = base_dir / f"{prefix}_{date}_{index:03d}"
        try:
            target.mkdir()
            return target
        except FileExistsError:
            index += 1


def create_analysis_run(
    paths: ProjectPaths,
    analysis_id: str,
    *,
    now: datetime | None = None,
) -> Path:
    """Allocate a new non-overwriting analysis run in the configured workspace."""
    base = paths.analysis_run(analysis_id, "placeholder").parent
    return allocate_numbered_run(base, prefix="run", now=now)
