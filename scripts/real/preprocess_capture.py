"""一键执行通用真实序列的训练前图像处理（不启动训练）。

阶段固定为：采集审计 -> 固定ROI裁剪 -> 自动候选/SAM2锚点 -> SAM2传播
-> 15节点中心线/NPZ -> 自动QC。所有阶段均保持原始采集目录只读。

示例：
  python scripts/real/preprocess_capture.py \
    --seq real_capture/data/raw/seq_20260819_172644 \
    --roi 220,68,300,300 --gpus 1,3
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np


STAGES = ("audit", "crop", "anchors", "sam2", "skeleton")

CONFIG_DEFAULTS = {
    "camera": "cam0",
    "gpus": "0",
    "chunk_size": 200,
    "n_points": 15,
    "state_frame": "robot_planar_mm",
    "segment_lengths": "1,1",
    "base_anchor": None,
    "mask_close_k": 11,
    "max_interpolated_fraction": 0.05,
    "qc_frames": "",
    "repair_frames": "",
    "out_root": None,
}
ANCHOR_CONFIG_DEFAULTS = {
    "n_bg": 500,
    "background_image": None,
    "base_side": "top",
    "base_trim_width_ratio": 1.5,
    "base_trim_stable_span": 5,
    "sat": 100,
    "val": 120,
    "diff": 25,
    "dil": 35,
    "open_k": 5,
    "close_k": 15,
    "min_area_frac": 0.003,
    "min_h_frac": 0.15,
}


def parse_roi(text):
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI必须为整数 x,y,w,h") from error
    if len(values) != 4 or min(values[:2]) < 0 or min(values[2:]) <= 0:
        raise argparse.ArgumentTypeError("ROI必须为非负x,y和正数w,h")
    return values


def parse_stages(text):
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    invalid = [value for value in values if value not in STAGES]
    if invalid or not values:
        raise argparse.ArgumentTypeError(
            f"stages必须取自 {','.join(STAGES)}，收到 {invalid or values}")
    return values


def _csv_value(value, *, integer=False):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"配置值必须是字符串或数组，实际为 {type(value).__name__}")
    convert = int if integer else float
    return ",".join(str(convert(item)) for item in value)


def load_pipeline_config(path):
    """读取单序列配置；CLI 中显式给出的值在后续覆盖配置。"""
    if not path:
        return {}
    with open(path, encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("前处理配置顶层必须是 JSON 对象")
    allowed = {"schema_version", "description", "anchor"} | set(CONFIG_DEFAULTS) \
        | {"seq", "roi"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"前处理配置包含未知字段: {unknown}")
    anchor = config.get("anchor", {})
    if not isinstance(anchor, dict):
        raise ValueError("配置 anchor 必须是 JSON 对象")
    unknown_anchor = sorted(set(anchor) - set(ANCHOR_CONFIG_DEFAULTS))
    if unknown_anchor:
        raise ValueError(f"anchor 配置包含未知字段: {unknown_anchor}")
    return config


def resolve_pipeline_args(args):
    """合并默认值、JSON 配置和显式 CLI 参数，并统一为现有脚本参数格式。"""
    config = load_pipeline_config(args.config)
    resolved = vars(args).copy()
    for name, default in CONFIG_DEFAULTS.items():
        if resolved.get(name) is None:
            resolved[name] = config.get(name, default)
    for name in ("seq", "roi"):
        if resolved.get(name) is None:
            resolved[name] = config.get(name)
    if not resolved["seq"]:
        raise ValueError("需要在 --config 或 --seq 中指定原始序列")
    if resolved["roi"] is None:
        raise ValueError("需要在 --config 或 --roi 中指定固定 ROI")
    if not isinstance(resolved["roi"], tuple):
        resolved["roi"] = parse_roi(_csv_value(resolved["roi"], integer=True))
    resolved["gpus"] = _csv_value(resolved["gpus"], integer=True)
    resolved["segment_lengths"] = _csv_value(resolved["segment_lengths"])
    resolved["base_anchor"] = _csv_value(resolved["base_anchor"])
    resolved["qc_frames"] = _csv_value(resolved["qc_frames"], integer=True) or ""
    resolved["repair_frames"] = _csv_value(
        resolved["repair_frames"], integer=True) or ""
    if not 0.0 <= float(resolved["max_interpolated_fraction"]) <= 1.0:
        raise ValueError("max_interpolated_fraction 必须位于 [0,1]")
    resolved["anchor"] = {**ANCHOR_CONFIG_DEFAULTS, **config.get("anchor", {})}
    resolved["config_source"] = os.path.abspath(args.config) if args.config else None
    resolved["resolved_config"] = {
        "schema_version": 1,
        "seq": resolved["seq"],
        "camera": resolved["camera"],
        "roi": list(resolved["roi"]),
        "gpus": [int(value) for value in resolved["gpus"].split(",")],
        "chunk_size": int(resolved["chunk_size"]),
        "n_points": int(resolved["n_points"]),
        "state_frame": resolved["state_frame"],
        "segment_lengths": [float(value) for value in
                            resolved["segment_lengths"].split(",")],
        "base_anchor": ([float(value) for value in resolved["base_anchor"].split(",")]
                        if resolved["base_anchor"] else None),
        "mask_close_k": int(resolved["mask_close_k"]),
        "max_interpolated_fraction": float(
            resolved["max_interpolated_fraction"]),
        "qc_frames": [int(value) for value in resolved["qc_frames"].split(",")
                      if value],
        "repair_frames": [int(value) for value in
                          resolved["repair_frames"].split(",") if value],
        "out_root": resolved["out_root"],
        "anchor": resolved["anchor"],
    }
    return argparse.Namespace(**resolved)


def validate_stage_dependencies(stages):
    selected = set(stages)
    downstream = {
        "crop": {"anchors", "sam2", "skeleton"},
        "anchors": {"sam2", "skeleton"},
        "sam2": {"skeleton"},
    }
    for stage, required in downstream.items():
        if stage in selected and not required.issubset(selected):
            raise ValueError(
                f"重建 {stage} 时必须同步执行下游阶段: {sorted(required)}")


def display_command(command, env_update=None):
    prefix = []
    for key, value in (env_update or {}).items():
        if key in ("CUDA_VISIBLE_DEVICES", "MPLCONFIGDIR"):
            prefix.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(prefix + [shlex.join(command)])


def run(command, cwd, records, env_update=None):
    environment = os.environ.copy()
    environment.update(env_update or {})
    rendered = display_command(command, env_update)
    print(f"\n>>> {rendered}", flush=True)
    records.append(rendered)
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def frame_ids(directory):
    result = []
    for path in glob.glob(os.path.join(directory, "*.png")):
        try:
            result.append(int(os.path.splitext(os.path.basename(path))[0]))
        except ValueError:
            continue
    return sorted(result)


def validate_sam2(image_dir, mask_dir):
    images = frame_ids(image_dir)
    masks = frame_ids(mask_dir)
    failures = []
    failure_paths = sorted(glob.glob(os.path.join(mask_dir, "failures*.txt")))
    for failures_path in failure_paths:
        with open(failures_path, encoding="utf-8") as stream:
            failures = [line.strip() for line in stream
                        if line.strip() and not line.lstrip().startswith("#")] + failures
    missing = sorted(set(images) - set(masks))
    extra = sorted(set(masks) - set(images))
    if missing or extra or failures or len(images) != len(masks):
        raise RuntimeError(
            "SAM2完整性检查失败："
            f"images={len(images)} masks={len(masks)} "
            f"missing={missing[:20]} extra={extra[:20]} failures={failures[:5]}")

    areas = []
    for frame in masks:
        mask = cv2.imread(os.path.join(mask_dir, f"{frame:05d}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"SAM2 mask无法读取: frame={frame}")
        areas.append(int(np.count_nonzero(mask > 127)))
    values = np.asarray(areas, dtype=np.float64)
    summary = {
        "image_count": len(images),
        "mask_count": len(masks),
        "frame_range": [images[0], images[-1]] if images else [],
        "frame_ids_match": images == masks,
        "failures_empty": not failures,
        "failure_logs": [os.path.basename(path) for path in failure_paths],
        "area_px": {
            "min": float(values.min()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        } if len(values) else {},
    }
    print(f">>> SAM2完整性通过：{len(masks)}帧，area={summary['area_px']}")
    return summary


def _read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _npz_scalar(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    return value.item() if getattr(value, "ndim", 1) == 0 else value.tolist()


def _boolean_csv(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def summarize_skeleton_qc(path):
    rows = []
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    return {
        "frame_count": len(rows),
        "success_count": sum(_boolean_csv(row.get("success")) for row in rows),
        "hard_invalid_count": sum(_boolean_csv(row.get("hard_invalid"))
                                  for row in rows),
        "interpolated_count": sum(_boolean_csv(row.get("interpolated"))
                                  for row in rows),
        "suspicious_count": sum(_boolean_csv(row.get("suspicious"))
                                for row in rows),
        "explicit_repair_count": sum(_boolean_csv(row.get("explicit_repair"))
                                     for row in rows),
    }


def summarize_npz(path):
    with np.load(path, allow_pickle=False) as data:
        positions = np.asarray(data["positions"])
        actions = np.asarray(data["actions"])
        node_order = _npz_scalar(data, "node_order")
        if node_order != "base_to_tip":
            raise ValueError(
                f"{path} 的 node_order 必须为 base_to_tip，当前为 {node_order!r}")
        camera_positions = (np.asarray(data["positions_camera_px"])
                            if "positions_camera_px" in data else None)
        return {
            "path": os.path.abspath(path),
            "frames": int(positions.shape[0]),
            "positions_shape": list(positions.shape),
            "actions_shape": list(actions.shape),
            "finite_positions": bool(np.isfinite(positions).all()),
            "finite_actions": bool(np.isfinite(actions).all()),
            "finite_positions_camera_px": bool(
                camera_positions is not None and np.isfinite(camera_positions).all()),
            "action_range": [float(actions.min()), float(actions.max())],
            "n_points": int(_npz_scalar(data, "n_points", positions.shape[2])),
            "node_order": str(node_order),
            "segment_lengths": _npz_scalar(data, "segment_lengths", []),
            "segment_intervals": _npz_scalar(data, "segment_intervals", []),
            "joint_node_indices": _npz_scalar(data, "joint_node_indices", []),
            "state_coordinate_frame": str(_npz_scalar(
                data, "state_coordinate_frame", "camera_pixel_v1")),
            "state_length_unit": str(_npz_scalar(data, "state_length_unit", "px")),
            "raw_action_dim": int(_npz_scalar(data, "raw_action_dim", actions.shape[1])),
            "model_action_dim": int(_npz_scalar(
                data, "model_action_dim", actions.shape[1])),
            "model_action_channels": _npz_scalar(
                data, "model_action_channels", list(range(actions.shape[1]))),
            "channel_source6": _npz_scalar(data, "channel_source6", list(range(6))),
            "action_expansion6": _npz_scalar(data, "action_expansion6", []),
            "raw_action_scale6_kpa": _npz_scalar(
                data, "raw_action_scale6_kpa", []),
            "action_scale_kpa": _npz_scalar(data, "action_scale_kpa", []),
            "robot_diameter_mm": float(_npz_scalar(data, "robot_diameter_mm", 16.0)),
            "robot_diameter_px": float(_npz_scalar(data, "robot_diameter_px", np.nan)),
            "mm_per_px": float(_npz_scalar(data, "mm_per_px", np.nan)),
            "image_crop_xywh": _npz_scalar(data, "image_crop_xywh", []),
            "source_image_size_wh": _npz_scalar(data, "source_image_size_wh", []),
            "processed_image_size_wh": _npz_scalar(
                data, "processed_image_size_wh", []),
            "skeleton_frame_transform": (json.loads(str(_npz_scalar(
                data, "skeleton_frame_transform")))
                if "skeleton_frame_transform" in data else None),
        }


def _check(name, passed, detail, *, required=True):
    return {
        "name": name,
        "passed": bool(passed),
        "required_for_training": bool(required),
        "detail": detail,
    }


def build_dataset_manifest(*, seq, camera, derived, crop_root, mask_dir,
                           out_root, resolved_config, commands, sam2_summary):
    """汇总机器可判定的合同与统计，并写出训练入口使用的数据清单。"""
    capture_audit_path = os.path.join(derived, "qc_capture", "capture_audit.json")
    legacy_audit_path = os.path.join(derived, "audit", "capture_audit.json")
    if not os.path.isfile(capture_audit_path) and os.path.isfile(legacy_audit_path):
        capture_audit_path = legacy_audit_path
    crop_meta_path = os.path.join(crop_root, "crop_meta.json")
    candidate_summary_path = os.path.join(derived, "candidate_summary.json")
    skeleton_metrics_path = os.path.join(
        out_root, "qc_skeleton", "skeleton_metrics.csv")
    capture_audit = _read_json(capture_audit_path, {})
    crop_meta = _read_json(crop_meta_path, {})
    candidate = _read_json(candidate_summary_path, {})
    skeleton = summarize_skeleton_qc(skeleton_metrics_path)
    train_files = sorted(glob.glob(os.path.join(out_root, "train", "*.npz")))
    val_files = sorted(glob.glob(os.path.join(out_root, "val", "*.npz")))
    train = [summarize_npz(path) for path in train_files]
    val = [summarize_npz(path) for path in val_files]
    splits = train + val
    image_count = len(frame_ids(os.path.join(crop_root, camera)))
    mask_count = len(frame_ids(mask_dir))
    split_frames = sum(item["frames"] for item in splits)
    skeleton["interpolated_fraction"] = (
        float(skeleton["interpolated_count"] / image_count) if image_count else None)
    skeleton["max_interpolated_fraction"] = float(
        resolved_config.get("max_interpolated_fraction", 0.05))

    checks = [
        _check("capture_contract", capture_audit.get(
            "ready_for_image_preprocessing") is True,
            {"issues": capture_audit.get("issues", []),
             "counts": capture_audit.get("counts", {})}),
        _check("crop_complete", crop_meta.get("complete") is True and
               int(crop_meta.get("n_output_frames", -1)) ==
               int(crop_meta.get("n_source_frames", -2)) == image_count,
               {"source_frames": crop_meta.get("n_source_frames"),
                "output_frames": crop_meta.get("n_output_frames"),
                "files": image_count}),
        _check("candidate_coverage", int(candidate.get("n_frames", -1)) == image_count,
               {"candidate_frames": candidate.get("n_frames"),
                "empty_candidates": candidate.get("n_empty"),
                "selected_anchors": candidate.get("n_selected_anchors")}),
        _check("sam2_frame_coverage", bool(sam2_summary) and
               sam2_summary.get("frame_ids_match") is True and
               sam2_summary.get("failures_empty") is True and
               int(sam2_summary.get("mask_count", -1)) == image_count == mask_count,
               sam2_summary),
        _check("dataset_splits", len(train) == 1 and len(val) == 1 and
               split_frames == image_count,
               {"train_files": len(train), "val_files": len(val),
                "split_frames": split_frames, "image_frames": image_count}),
        _check("skeleton_contract", skeleton["frame_count"] == image_count and
               skeleton["success_count"] > 0 and
               skeleton["interpolated_count"] <= image_count * float(
                   resolved_config.get("max_interpolated_fraction", 0.05)) and
               all(item["n_points"] == int(resolved_config["n_points"])
                   for item in splits) and
               all(item["finite_positions"] and
                   item["finite_positions_camera_px"] for item in splits),
               skeleton),
        _check("state_contract", bool(splits) and
               len({(item["state_coordinate_frame"], item["state_length_unit"])
                    for item in splits}) == 1 and
               all(item["state_coordinate_frame"] ==
                   ("robot_planar_mm_v1" if resolved_config["state_frame"] ==
                    "robot_planar_mm" else "camera_pixel_v1") for item in splits),
               [{"frame": item["state_coordinate_frame"],
                 "unit": item["state_length_unit"]} for item in splits]),
        _check("action_contract", bool(splits) and
               all(item["raw_action_dim"] == 6 and item["finite_actions"] and
                   item["action_range"][0] >= -1e-6 and
                   item["action_range"][1] <= 1.0 + 1e-6 for item in splits) and
               len({json.dumps({
                   "channels": item["model_action_channels"],
                   "sources": item["channel_source6"],
                   "expansion": item["action_expansion6"],
               }, sort_keys=True) for item in splits}) == 1,
               [{"raw_action_dim": item["raw_action_dim"],
                 "model_action_dim": item["model_action_dim"],
                 "model_action_channels": item["model_action_channels"],
                 "channel_source6": item["channel_source6"],
                 "action_range": item["action_range"]} for item in splits]),
    ]
    training_ready = all(item["passed"] for item in checks
                         if item["required_for_training"])
    frame_times_path = os.path.join(seq, "frame_times.txt")
    timing = {}
    if os.path.isfile(frame_times_path):
        frame_times = np.atleast_1d(np.loadtxt(frame_times_path)).astype(float)
        delta = np.diff(frame_times)
        if len(delta):
            timing = {
                "frames": int(len(frame_times)),
                "median_dt_s": float(np.median(delta)),
                "mean_dt_s": float(np.mean(delta)),
                "std_dt_s": float(np.std(delta)),
                "median_hz": float(1.0 / np.median(delta)),
            }
    first = splits[0] if splits else {}
    manifest = {
        "schema_version": 2,
        "dataset_id": os.path.basename(out_root.rstrip("/")),
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "source": {
            "sequence": os.path.basename(seq.rstrip("/")),
            "sequence_path": os.path.abspath(seq),
            "camera": camera,
        },
        "preprocessing": {
            "resolved_config": resolved_config,
            "crop_meta": crop_meta,
            "candidate_segmentation": candidate,
            "sam2": sam2_summary,
            "skeleton": skeleton,
        },
        "state": {
            "coordinate_frame": first.get("state_coordinate_frame"),
            "length_unit": first.get("state_length_unit"),
            "n_nodes": first.get("n_points"),
            "node_order": first.get("node_order"),
            "segment_lengths": first.get("segment_lengths"),
            "segment_intervals": first.get("segment_intervals"),
            "joint_node_indices": first.get("joint_node_indices"),
            "robot_diameter_mm": first.get("robot_diameter_mm"),
            "robot_diameter_px": first.get("robot_diameter_px"),
            "mm_per_px": first.get("mm_per_px"),
            "skeleton_frame_transform": first.get("skeleton_frame_transform"),
        },
        "action": {
            "raw_action_dim": first.get("raw_action_dim"),
            "model_action_dim": first.get("model_action_dim"),
            "model_action_channels": first.get("model_action_channels"),
            "channel_source6": first.get("channel_source6"),
            "action_expansion6": first.get("action_expansion6"),
            "raw_action_scale6_kpa": first.get("raw_action_scale6_kpa"),
            "action_scale_kpa": first.get("action_scale_kpa"),
        },
        "timing": timing,
        "splits": {"train": train, "val": val},
        "quality_control": {
            "capture_audit": os.path.abspath(capture_audit_path),
            "skeleton_metrics": os.path.abspath(skeleton_metrics_path),
            "checks": checks,
            "automated_checks_passed": training_ready,
            "training_ready": training_ready,
            "qc_review": "optional",
        },
        "reproducibility": {
            "commands": commands,
            "preprocess_manifest": os.path.abspath(os.path.join(
                derived, "preprocess_manifest.json")),
        },
    }
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(
        description="真实采集序列一键前处理；只生成训练前产物，不启动训练")
    parser.add_argument("--config", default=None,
                        help="单序列JSON配置；显式CLI参数覆盖同名配置")
    parser.add_argument("--seq", default=None,
                        help="原始seq目录，或real_capture/data/raw下的序列名")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--roi", default=None, type=parse_roi,
                        help="固定源图像ROI：x,y,w,h")
    parser.add_argument("--gpus", default=None,
                        help="SAM2物理GPU编号，逗号分隔；每进程内部均使用cuda:0")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--n-points", type=int, default=None)
    parser.add_argument("--state-frame", choices=("robot_planar_mm", "camera_pixel"),
                        default=None,
                        help="模型状态坐标；默认机器人基座平面毫米坐标")
    parser.add_argument("--segment-lengths", default=None)
    parser.add_argument("--base-anchor", default=None,
                        help="可选基座像素x,y（源相机坐标），透传到骨架阶段")
    parser.add_argument("--mask-close-k", type=int, default=None,
                        help="骨架化前闭运算核；默认11，用于闭合线缆遮挡窄裂缝")
    parser.add_argument("--max-interpolated-fraction", type=float, default=None,
                        help="自动插值骨架帧比例上限，默认0.05")
    parser.add_argument("--qc-frames", default=None,
                        help="额外进入骨架叠加图的重点帧号，逗号分隔")
    parser.add_argument("--repair-frames", default=None,
                        help="传给骨架阶段：按真实frame ID选择性插值人工确认坏帧")
    parser.add_argument("--stages", type=parse_stages, default=STAGES,
                        help="默认全部；可传audit,crop,anchors,sam2,skeleton")
    parser.add_argument("--overwrite-crop", action="store_true",
                        help="确认ROI变化后重写全部裁剪帧")
    parser.add_argument("--reset-sam2", action="store_true",
                        help="显式删除本序列已有SAM2 mask后重算")
    parser.add_argument("--out-root", default=None,
                        help="骨架NPZ输出；默认按 state-frame 使用 _robot_mm/_camera_px 后缀")
    return parser


def main(argv=None):
    args = resolve_pipeline_args(build_parser().parse_args(argv))
    validate_stage_dependencies(args.stages)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    seq = os.path.abspath(args.seq)
    if not os.path.isdir(os.path.join(seq, args.camera)):
        seq = os.path.join(project_root, "real_capture", "data", "raw",
                           os.path.basename(args.seq.rstrip("/")))
    if not os.path.isdir(os.path.join(seq, args.camera)):
        raise FileNotFoundError(f"找不到采集视角目录: {seq}/{args.camera}")
    seq_name = os.path.basename(seq.rstrip("/"))
    derived = os.path.join(project_root, "real_capture", "data", "derived",
                           seq_name)
    crop_root = os.path.join(derived, "crop")
    crop_camera = os.path.join(crop_root, args.camera)
    mask_dir = os.path.join(project_root, "sam2", "masks",
                            f"{seq_name}_full")
    out_root = (os.path.abspath(args.out_root) if args.out_root else
                os.path.join(project_root, "data", "real_seq",
                             f"{seq_name}_n{args.n_points}_sam2" +
                             ("_robot_mm" if args.state_frame == "robot_planar_mm"
                              else "_camera_px")))
    python = sys.executable
    commands = []
    common_env = {"MPLCONFIGDIR": "/tmp/selfsoftrobot-mpl"}
    os.makedirs(derived, exist_ok=True)

    if args.reset_sam2 and os.path.isdir(mask_dir):
        allowed_parent = os.path.realpath(os.path.join(project_root, "sam2", "masks"))
        if os.path.dirname(os.path.realpath(mask_dir)) != allowed_parent:
            raise RuntimeError(f"拒绝删除非标准SAM2目录: {mask_dir}")
        shutil.rmtree(mask_dir)
        print(f">>> 已按 --reset-sam2 删除可重建产物: {mask_dir}")
    if args.overwrite_crop and frame_ids(mask_dir) and not args.reset_sam2:
        raise RuntimeError("重写ROI会使已有SAM2 mask失效；请同时显式传 --reset-sam2")

    if "audit" in args.stages:
        run([python, "scripts/real/audit_capture.py", "--seq", seq,
             "--camera", args.camera], project_root, commands, common_env)
        audit_path = os.path.join(derived, "qc_capture", "capture_audit.json")
        audit = _read_json(audit_path, {})
        if audit.get("ready_for_image_preprocessing") is not True:
            critical = [item.get("code") for item in audit.get("issues", [])
                        if item.get("severity") == "critical"]
            raise RuntimeError(
                f"采集关键合同检查失败: {critical}; audit={audit_path}")
    if "crop" in args.stages:
        command = [python, "scripts/real/crop_capture.py", "--seq", seq,
                   "--camera", args.camera, "--roi", ",".join(map(str, args.roi))]
        if args.overwrite_crop:
            command.append("--overwrite")
        run(command, project_root, commands)
    if "anchors" in args.stages:
        anchor = args.anchor
        command = [python, "scripts/real/prepare_sam2_anchors.py",
             "--seq", crop_root, "--camera", args.camera,
             "--out-root", derived, "--chunk-size", str(args.chunk_size),
             "--n-bg", str(anchor["n_bg"]),
             "--base-side", str(anchor["base_side"]),
             "--base-trim-width-ratio", str(anchor["base_trim_width_ratio"]),
             "--base-trim-stable-span", str(anchor["base_trim_stable_span"]),
             "--sat", str(anchor["sat"]), "--val", str(anchor["val"]),
             "--diff", str(anchor["diff"]), "--dil", str(anchor["dil"]),
             "--open-k", str(anchor["open_k"]),
             "--close-k", str(anchor["close_k"]),
             "--min-area-frac", str(anchor["min_area_frac"]),
             "--min-h-frac", str(anchor["min_h_frac"])]
        if anchor.get("background_image"):
            command.extend(("--background-image", str(anchor["background_image"])))
        run(command, project_root, commands, common_env)
        candidate_summary_path = os.path.join(derived, "candidate_summary.json")
        candidate_summary = _read_json(candidate_summary_path, {})
        expected_anchors = int(math.ceil(
            int(candidate_summary.get("n_frames", 0)) / args.chunk_size))
        if int(candidate_summary.get("n_selected_anchors", -1)) != expected_anchors:
            raise RuntimeError(
                "候选分割没有为每个SAM2 chunk生成锚点: "
                f"selected={candidate_summary.get('n_selected_anchors')} "
                f"expected={expected_anchors}; summary={candidate_summary_path}")
    if "sam2" in args.stages:
        gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
        if len(set(gpu_ids)) != len(gpu_ids) or not gpu_ids:
            raise ValueError("--gpus必须给出至少一个不重复的物理GPU编号")
        os.makedirs(mask_dir, exist_ok=True)
        for failure_log in glob.glob(os.path.join(mask_dir, "failures*.txt")):
            os.remove(failure_log)
        processes = []
        for shard, gpu in enumerate(gpu_ids):
            command = [python, "sam2/segment_video_full.py",
                       "--seq", crop_root, "--camera", args.camera,
                       "--anchor-mask-dir", os.path.join(derived, "masks_candidate"),
                       "--anchor-manifest", os.path.join(derived, "anchor_manifest.csv"),
                       "--out", mask_dir, "--chunk-size", str(args.chunk_size),
                       "--shards", str(len(gpu_ids)), "--shard", str(shard),
                       "--device", "cuda:0"]
            env_update = {**common_env, "CUDA_VISIBLE_DEVICES": gpu}
            rendered = display_command(command, env_update)
            print(f"\n>>> {rendered}", flush=True)
            commands.append(rendered)
            environment = os.environ.copy(); environment.update(env_update)
            processes.append(subprocess.Popen(
                command, cwd=project_root, env=environment))
        while True:
            return_codes = [process.poll() for process in processes]
            failure = next((code for code in return_codes
                            if code is not None and code != 0), None)
            if failure is not None:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
                raise subprocess.CalledProcessError(failure, "SAM2 shards")
            if all(code == 0 for code in return_codes):
                break
            time.sleep(0.5)

    sam2_summary = {}
    if "sam2" in args.stages or "skeleton" in args.stages:
        sam2_summary = validate_sam2(crop_camera, mask_dir)
    if "skeleton" in args.stages:
        command = [python, "scripts/real/masks_to_transition_npz.py",
             "--seq", seq, "--masks-dir", mask_dir,
             "--crop-meta", os.path.join(crop_root, "crop_meta.json"),
             "--skeleton-method", "skeletonize", "--endpoint-fix",
             "--mask-close-k", str(args.mask_close_k),
             "--n-points", str(args.n_points),
             "--segment-lengths", args.segment_lengths,
             "--action-channels", "auto", "--out-root", out_root,
             "--state-frame", args.state_frame]
        if args.qc_frames:
            command.extend(("--qc-frames", args.qc_frames))
        if args.base_anchor:
            command.extend(("--base-anchor", args.base_anchor))
        if args.repair_frames:
            command.extend(("--repair-frames", args.repair_frames))
        run(command, project_root, commands, common_env)

        stage_qc_command = [
            python, "scripts/real/save_preprocess_stage_example.py",
            "--seq", seq, "--camera", args.camera,
            "--derived", derived, "--masks-dir", mask_dir,
            "--dataset-root", out_root,
            "--mask-close-k", str(args.mask_close_k),
            "--n-points", str(args.n_points),
            "--segment-lengths", args.segment_lengths,
        ]
        if args.base_anchor:
            stage_qc_command.extend(("--base-anchor", args.base_anchor))
        run(stage_qc_command, project_root, commands, common_env)

    dataset_manifest = build_dataset_manifest(
        seq=seq, camera=args.camera, derived=derived, crop_root=crop_root,
        mask_dir=mask_dir, out_root=out_root,
        resolved_config=args.resolved_config, commands=commands,
        sam2_summary=sam2_summary)
    dataset_manifest_path = os.path.join(out_root, "dataset_manifest.json")
    os.makedirs(out_root, exist_ok=True)
    with open(dataset_manifest_path, "w", encoding="utf-8") as stream:
        json.dump(dataset_manifest, stream, indent=2, ensure_ascii=False)
    training_ready = bool(dataset_manifest["quality_control"]["training_ready"])
    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "sequence": seq,
        "camera": args.camera,
        "crop_xywh": list(args.roi),
        "processed_image_size_wh": [args.roi[2], args.roi[3]],
        "sam2": sam2_summary,
        "n_points": args.n_points,
        "state_frame": args.state_frame,
        "segment_lengths": [float(value) for value in args.segment_lengths.split(",")],
        "base_anchor_source_xy": (
            [float(value) for value in args.base_anchor.split(",")]
            if args.base_anchor else None),
        "mask_close_kernel": args.mask_close_k,
        "npz_out_root": out_root,
        "dataset_manifest": dataset_manifest_path,
        "config_source": args.config_source,
        "resolved_config": args.resolved_config,
        "commands": commands,
        "automated_checks": dataset_manifest["quality_control"]["checks"],
        "automated_checks_passed": training_ready,
        "training_ready": training_ready,
        "qc_review": "optional",
        "training_started": False,
    }
    manifest_path = os.path.join(derived, "preprocess_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    report_path = os.path.join(derived, "PREPROCESS_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as stream:
        stream.write(f"# {seq_name} 训练前处理报告\n\n")
        stream.write("状态：自动合同检查通过，数据已可用于训练。\n\n" if training_ready
                     else "状态：自动合同检查发现未满足的训练条件。\n\n")
        stream.write(f"- 原始序列：`{seq}`\n- ROI：`{args.roi}`\n")
        if sam2_summary:
            stream.write(
                f"- SAM2 mask：`{mask_dir}`（{sam2_summary['mask_count']}帧）\n")
        stream.write(f"- {args.n_points}节点数据：`{out_root}`\n")
        stream.write(f"- 数据清单：`{dataset_manifest_path}`\n\n")
        stream.write("## 自动检查\n\n")
        for item in dataset_manifest["quality_control"]["checks"]:
            stream.write(f"- {'PASS' if item['passed'] else 'FAIL'} "
                         f"`{item['name']}`\n")
        stream.write("\n## 复现命令\n\n```bash\n")
        stream.write("\n".join(commands)); stream.write("\n```\n\n")
        stream.write("## 抽样 QC（按需查看）\n\n")
        stream.write(f"- `{crop_root}/qc/`\n- `{derived}/qc_candidate/`\n")
        stream.write(f"- `{mask_dir}/qc/`\n- `{out_root}/qc_skeleton/`\n")
        stream.write(f"- `{derived}/qc_pipeline_example/`\n")
    if not training_ready:
        failed = [item["name"] for item in
                  dataset_manifest["quality_control"]["checks"]
                  if item["required_for_training"] and not item["passed"]]
        raise RuntimeError(f"自动合同检查失败: {failed}；报告={report_path}")
    print(f"\n>>> 前处理完成，数据已可训练。dataset={dataset_manifest_path}")
    print(f">>> 报告与抽样QC：{report_path}")


if __name__ == "__main__":
    main()
