"""将多个已前处理 run 组合成一个可训练的 transition 数据集。"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.real.preprocess_capture import summarize_npz  # noqa: E402
from src.registry import ProjectPaths, canonical_output  # noqa: E402


STATE_KEYS = (
    "state_coordinate_frame",
    "state_length_unit",
    "node_order",
    "n_points",
    "segment_lengths",
    "segment_intervals",
    "joint_node_indices",
)
ACTION_KEYS = (
    "raw_action_dim",
    "model_action_dim",
    "model_action_channels",
    "channel_source6",
    "action_expansion6",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _shared_contract(items, keys, label):
    if not items:
        raise ValueError(f"组合数据集缺少 {label} NPZ")
    first = {key: items[0].get(key) for key in keys}
    mismatches = []
    for item in items[1:]:
        current = {key: item.get(key) for key in keys}
        if _canonical(current) != _canonical(first):
            mismatches.append({"path": item["path"], "contract": current})
    if mismatches:
        raise ValueError(
            f"{label} NPZ 合同不一致: expected={first}, mismatches={mismatches}")
    return first


def _source_manifest(npz_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(npz_path)))
    path = os.path.join(root, "dataset_manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"NPZ 缺少来源 dataset_manifest.json: {npz_path}")
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    quality = manifest.get("quality_control", {})
    if quality.get("training_ready") is not True:
        raise ValueError(f"来源数据集尚未通过训练准入: {path}")
    return path, manifest


def _copy_split(paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    seen = set()
    for source in paths:
        source = os.path.abspath(source)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        name = os.path.basename(source)
        if name in seen:
            raise ValueError(f"组合数据集出现重名 NPZ: {name}")
        seen.add(name)
        destination = os.path.join(out_dir, name)
        shutil.copy2(source, destination)
        summaries.append(summarize_npz(destination))
    return summaries


def combine(args):
    paths = ProjectPaths.load(workspace_root=args.workspace_root)
    out_root = str(resolve_output(args, paths))
    if os.path.exists(out_root) and os.listdir(out_root):
        raise FileExistsError(f"输出目录已包含产物: {out_root}")

    source_entries = {}
    for path in (*args.train, *args.val):
        manifest_path, manifest = _source_manifest(path)
        source_entries[manifest_path] = {
            "dataset_id": manifest.get("dataset_id"),
            "manifest": manifest_path,
            "sequence": manifest.get("source", {}).get("sequence"),
        }

    source_train = [summarize_npz(os.path.abspath(path)) for path in args.train]
    source_val = [summarize_npz(os.path.abspath(path)) for path in args.val]
    source_items = source_train + source_val
    state = _shared_contract(source_items, STATE_KEYS, "状态")
    action = _shared_contract(source_items, ACTION_KEYS, "动作")
    if state["node_order"] != "base_to_tip":
        raise ValueError(f"训练节点方向必须为 base_to_tip: {state['node_order']}")

    os.makedirs(out_root, exist_ok=True)
    train = _copy_split(args.train, os.path.join(out_root, "train"))
    val = _copy_split(args.val, os.path.join(out_root, "val"))

    checks = [
        {
            "name": "source_datasets_ready",
            "passed": True,
            "required_for_training": True,
            "detail": list(source_entries.values()),
        },
        {
            "name": "dataset_splits",
            "passed": bool(train and val),
            "required_for_training": True,
            "detail": {
                "train_files": len(train),
                "val_files": len(val),
                "train_frames": sum(item["frames"] for item in train),
                "val_frames": sum(item["frames"] for item in val),
            },
        },
        {
            "name": "state_contract",
            "passed": True,
            "required_for_training": True,
            "detail": state,
        },
        {
            "name": "action_contract",
            "passed": True,
            "required_for_training": True,
            "detail": action,
        },
    ]
    training_ready = all(
        item["passed"] for item in checks if item["required_for_training"])
    command = shlex.join([sys.executable, *sys.argv])
    manifest = {
        "schema_version": 2,
        "dataset_id": os.path.basename(out_root),
        "created_at": datetime.now().astimezone().isoformat(),
        "source": {"datasets": list(source_entries.values())},
        "state": {
            "coordinate_frame": state["state_coordinate_frame"],
            "length_unit": state["state_length_unit"],
            "n_nodes": state["n_points"],
            "node_order": state["node_order"],
            "segment_lengths": state["segment_lengths"],
            "segment_intervals": state["segment_intervals"],
            "joint_node_indices": state["joint_node_indices"],
        },
        "action": action,
        "splits": {"train": train, "val": val},
        "quality_control": {
            "checks": checks,
            "automated_checks_passed": training_ready,
            "training_ready": training_ready,
        },
        "reproducibility": {"command": command},
    }
    manifest_path = os.path.join(out_root, "dataset_manifest.json")
    temporary = f"{manifest_path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, manifest_path)
    print(manifest_path)
    return manifest_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="组合多个已通过准入的 transition NPZ 数据集")
    parser.add_argument("--dataset-id",
                        help="新组合数据集 ID；省略 --out-root 时必填")
    parser.add_argument("--out-root",
                        help="显式输出根；必须位于 workspace 内")
    parser.add_argument("--workspace-root",
                        help="覆盖 SSR_WORKSPACE_ROOT/config/默认 workspace")
    parser.add_argument("--train", nargs="+", required=True,
                        help="进入组合训练 split 的 NPZ")
    parser.add_argument("--val", nargs="+", required=True,
                        help="进入组合验证 split 的 NPZ")
    return parser


def resolve_output(args, paths):
    if args.out_root:
        return canonical_output(paths, args.out_root)
    if not args.dataset_id:
        raise ValueError("必须提供 --dataset-id 或 workspace 内的 --out-root")
    return paths.processed_dataset("real", args.dataset_id)


if __name__ == "__main__":
    combine(build_parser().parse_args())
