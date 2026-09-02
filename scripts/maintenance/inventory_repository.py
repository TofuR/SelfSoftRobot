#!/usr/bin/env python3
"""Generate a read-only inventory before repository artifact migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.registry.inventory import (  # noqa: E402
    LegacyInventoryBuilder,
    write_legacy_inventory,
)
from src.registry.paths import ProjectPaths  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        description="盘点历史 raw/dataset/run；不移动、不删除、不 hash 大文件")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None,
                        help="默认 workspace/registry/legacy_inventory.json")
    parser.add_argument("--stdout", action="store_true",
                        help="只输出 JSON，不创建 workspace")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(
        repo_root=PROJECT_ROOT,
        workspace_root=args.workspace_root,
        config_path=args.config,
    )
    inventory = LegacyInventoryBuilder(paths).build()
    if args.stdout:
        json.dump(inventory, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    paths.create_workspace_layout()
    target = write_legacy_inventory(paths, inventory, target=args.out)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
