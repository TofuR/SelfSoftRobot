#!/usr/bin/env python3
"""Prevent new executable references to retired repository-local roots.

The baseline is intentionally small and count based: existing compatibility
reads and reproducibility-only scripts may be reduced over time, while a new
file or an increased count fails the check.  Python comments/docstrings and
shell comments are ignored.
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = ("real_capture", "real_validation", "scripts", "src")
EXCLUDED_FILES = {"scripts/maintenance/check_legacy_path_literals.py"}
RETIRED_ROOTS = (
    "real_capture/data",
    "data/real_seq",
    "train_log",
    "real_validation/runs",
    "output/",
)

# Compatibility/migration code and older reproducibility scripts may still
# read these roots.  This is a non-increasing budget, not permission for new
# writes.  Remove entries as their consumers are migrated.
BASELINE = {
    "scripts/evaluation/eval_real_quant.py": {"real_capture/data": 1},
    "scripts/evaluation/visualize_3d_shape.py": {"train_log": 1},
    "scripts/experiments/evaluate_hereditary_v2_checkpoint.sh": {"data/real_seq": 1},
    "scripts/experiments/exp1_skeleton_from_2d.py": {"output/": 1},
    "scripts/experiments/exp1b_improved_2d_skeleton.py": {"output/": 1},
    "scripts/experiments/exp2_pure_2d_comparison.py": {"output/": 1},
    "scripts/experiments/exp3_multi_camera.py": {"train_log": 1, "output/": 1},
    "scripts/experiments/exp4_domain_randomization.py": {"output/": 1},
    "scripts/experiments/exp4b_fixed_dr_eval.py": {"output/": 1},
    "scripts/experiments/exp5_hysteresis_analysis.py": {"output/": 1},
    "scripts/experiments/exp5b_hysteresis_loop.py": {"output/": 1},
    "scripts/experiments/exp6_comprehensive_report.py": {"train_log": 5, "output/": 3},
    "scripts/experiments/exp7_3d_occupancy.py": {"output/": 2},
    "scripts/experiments/exp7_multiview_2d_skeleton.py": {"output/": 1},
    "scripts/experiments/run_hereditary_v2_experiment.sh": {"data/real_seq": 1},
    "scripts/maintenance/migrate_legacy_assets.py": {
        "real_capture/data": 2,
        "data/real_seq": 1,
        "train_log": 1,
        "real_validation/runs": 1,
    },
    "scripts/real/clean_transition_npz.py": {"real_capture/data": 1, "data/real_seq": 2},
    "scripts/real/compare_skeleton_methods.py": {"output/": 1},
    "scripts/real/composite_frames.py": {"real_capture/data": 1},
    "scripts/real/write_data_readme.py": {"real_capture/data": 5, "data/real_seq": 7},
    "scripts/training/train_search.py": {"train_log": 2},
    "scripts/training/train_transition.py": {"train_log": 1},
    "scripts/utils/build_control_report.py": {"data/real_seq": 1},
    "scripts/visualization/test_3d_seq.py": {"train_log": 2},
    "src/registry/paths.py": {
        "real_capture/data": 2,
        "data/real_seq": 1,
        "train_log": 1,
        "real_validation/runs": 1,
    },
    "src/registry/workspace_index.py": {"data/real_seq": 1},
}


def _count_text(value: str, counter: Counter) -> None:
    for retired in RETIRED_ROOTS:
        counter[retired] += value.count(retired)


def _python_counts(path: Path) -> Counter:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and
                    isinstance(first.value, ast.Constant) and
                    isinstance(first.value.value, str)):
                docstrings.add(id(first.value))

    counts = Counter()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str) and
                id(node) not in docstrings):
            _count_text(node.value, counts)
    return counts


def _shell_counts(path: Path) -> Counter:
    counts = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("#"):
            _count_text(line, counts)
    return counts


def collect_counts(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, int]]:
    found = defaultdict(Counter)
    for source_root in SOURCE_ROOTS:
        root = repo_root / source_root
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(repo_root).as_posix()
            if relative not in EXCLUDED_FILES:
                found[relative].update(_python_counts(path))
        for path in sorted(root.rglob("*.sh")):
            relative = path.relative_to(repo_root).as_posix()
            if relative not in EXCLUDED_FILES:
                found[relative].update(_shell_counts(path))
    return {
        path: {token: count for token, count in counts.items() if count}
        for path, counts in found.items() if any(counts.values())
    }


def find_violations(
        repo_root: Path = REPO_ROOT,
        baseline: dict[str, dict[str, int]] = BASELINE,
) -> list[str]:
    violations = []
    for path, counts in collect_counts(repo_root).items():
        allowed = baseline.get(path, {})
        for token, count in counts.items():
            if count > allowed.get(token, 0):
                violations.append(
                    f"{path}: {token!r} count={count}, allowed={allowed.get(token, 0)}")
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print("发现新增或增加的退役路径字面量:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("legacy path literal check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
