#!/usr/bin/env python3
"""Index canonical datasets and training runs without hashing payload files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.paths import ProjectPaths  # noqa: E402
from src.registry.workspace_index import (  # noqa: E402
    WorkspaceIndexBuilder,
    write_missing_legacy_dataset_manifests,
    write_mainline_legacy_manifests,
    write_workspace_index,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="轻量盘点 canonical dataset/run；不移动、不删除、不 hash 大文件")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None,
                        help="默认 workspace/registry/workspace_asset_index.json")
    parser.add_argument("--stdout", action="store_true",
                        help="只输出 JSON，不写索引或补 manifest")
    parser.add_argument(
        "--backfill-mainline-manifests", action="store_true",
        help="为已有 real_pipeline 试次补非覆盖 legacy_run_manifest.json")
    parser.add_argument(
        "--backfill-dataset-manifests", action="store_true",
        help="为缺清单的 processed dataset 补观察型 legacy manifest")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(
        repo_root=PROJECT_ROOT,
        workspace_root=args.workspace_root,
        config_path=args.config,
    )
    index = WorkspaceIndexBuilder(paths).build()
    if args.stdout:
        json.dump(index, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    paths.create_workspace_layout()
    written = []
    if args.backfill_dataset_manifests:
        written.extend(write_missing_legacy_dataset_manifests(paths, index))
    if args.backfill_mainline_manifests:
        written.extend(write_mainline_legacy_manifests(paths, index))
    if args.backfill_dataset_manifests or args.backfill_mainline_manifests:
        index = WorkspaceIndexBuilder(paths).build()
    target = write_workspace_index(paths, index, target=args.out)
    print(target)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
