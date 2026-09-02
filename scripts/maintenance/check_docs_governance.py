#!/usr/bin/env python3
"""Lightweight checks for SelfSoftRobot's governed documentation surfaces."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_OWNERS = {
    "Constitution": Path("CLAUDE.md"),
    "Map": Path("docs/README.md"),
    "Status": Path("docs/overview/status.md"),
    "History": Path("docs/maintenance/README.md"),
}
REQUIRED_FRONT_MATTER = {"title", "kind", "status", "updated", "scope"}
ALLOWED_STATUS = {"draft", "active", "complete", "superseded", "archived"}
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def governed_docs(root: Path) -> list[Path]:
    """Return the intentionally small set whose metadata is enforced now."""
    fixed = [
        root / "docs/README.md",
        root / "docs/HANDOFF.md",
        root / "docs/overview/status.md",
        root / "docs/ref/README.md",
        root / "docs/paper/README.md",
        root / "docs/papers/README.md",
        root / "sam2/README.md",
    ]
    expanded = [
        *sorted((root / "docs/standards").glob("*.md")),
        *sorted((root / "docs/maintenance").glob("*.md")),
    ]
    return sorted(set(fixed + expanded))


def front_matter(path: Path) -> tuple[dict[str, str], str | None]:
    """Parse only top-level YAML-like keys without adding a YAML dependency."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, "missing opening front matter delimiter"
    try:
        end = next(i for i, line in enumerate(lines[1:], 1)
                   if line.strip() == "---")
    except StopIteration:
        return {}, "missing closing front matter delimiter"
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if line and not line[0].isspace() and ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values, None


def check_front_matter(path: Path) -> list[str]:
    values, error = front_matter(path)
    if error:
        return [f"{path}: {error}"]
    missing = sorted(REQUIRED_FRONT_MATTER - values.keys())
    errors = [f"{path}: missing front matter key {key}" for key in missing]
    status = values.get("status")
    if status and status not in ALLOWED_STATUS:
        errors.append(f"{path}: unsupported status {status!r}")
    return errors


def check_relative_links(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for raw_target in LINK_RE.findall(text):
        target = raw_target.strip().strip("<>").split(maxsplit=1)[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        relative = unquote(target.split("#", 1)[0])
        if not relative:
            continue
        resolved = (path.parent / relative).resolve()
        if not resolved.exists():
            errors.append(f"{path}: broken relative link {target!r}")
    return errors


def audit_repository(root: Path = PROJECT_ROOT) -> list[str]:
    errors: list[str] = []
    for role, relative in ROLE_OWNERS.items():
        owner = root / relative
        if not owner.is_file():
            errors.append(f"{role}: missing owner {relative}")

    map_path = root / ROLE_OWNERS["Map"]
    if map_path.is_file():
        map_text = map_path.read_text(encoding="utf-8")
        for role in ROLE_OWNERS:
            marker = f"| {role} |"
            count = map_text.count(marker)
            if count != 1:
                errors.append(
                    f"{map_path}: expected one {role} role row, found {count}")

    for path in governed_docs(root):
        if not path.is_file():
            errors.append(f"missing governed document {path}")
            continue
        errors.extend(check_front_matter(path))
        errors.extend(check_relative_links(path))

    handoff = root / "docs/HANDOFF.md"
    if handoff.is_file() and len(handoff.read_text(encoding="utf-8").splitlines()) > 80:
        errors.append(f"{handoff}: redirect grew beyond 80 lines")
    return errors


def main() -> int:
    errors = audit_repository()
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        return 1
    print(
        "docs governance: PASS "
        f"roles={len(ROLE_OWNERS)} governed_docs={len(governed_docs(PROJECT_ROOT))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
