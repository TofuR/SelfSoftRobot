"""保存同一帧从原始照片到训练样本的逐阶段可视化。

该脚本只读取前处理产物，不重新生成数据标签。默认选择质量最高的已选 SAM2
锚帧；也可以用 ``--frame`` 指定真实 frame ID。输出包含每阶段独立 PNG、总览图、
机器可读清单和中文说明。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_validation.perception.skeleton import (  # noqa: E402
    _fix_path_endcaps,
    _medial_longest_path,
    _resample_path_segmented,
)
from scripts.real.masks_to_transition_npz import prepare_centerline_mask  # noqa: E402
from scripts.real.prepare_sam2_anchors import candidate_stages  # noqa: E402


TILE_WIDTH = 480
TILE_HEIGHT = 480
HEADER_HEIGHT = 70

STAGE_TITLES_ZH = {
    "01_raw_camera_roi": "原始相机画面与固定 ROI",
    "02_fixed_roi_crop": "固定 ROI 裁剪图",
    "03_median_background": "序列中值背景",
    "04_white_appearance": "白色外观候选",
    "05_background_motion": "背景差运动区域",
    "06_white_motion_gate": "白色与运动交集",
    "07_morphology_fill": "形态学清理与孔洞填充",
    "08_main_component": "主体连通域",
    "09_candidate_prompt": "SAM2 候选提示",
    "10_sam2_video_mask": "SAM2 视频分割结果",
    "11_close_occlusion_gaps": "遮挡窄缝闭合",
    "12_medial_skeleton": "单像素中轴",
    "13_longest_main_path": "最长中心主路径",
    "14_endcap_corrected_path": "端帽中心修正路径",
    "15_resampled_nodes": "15 节点分段重采样",
    "16_camera_training_state": "源相机像素训练状态",
    "17_model_training_sample": "模型坐标训练样本与动作",
}


def _read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _image(path, mode=cv2.IMREAD_COLOR):
    value = cv2.imread(str(path), mode)
    if value is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return value


def _letterbox(image, width=TILE_WIDTH, height=TILE_HEIGHT - HEADER_HEIGHT):
    values = np.asarray(image)
    if values.ndim == 2:
        values = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
    h, w = values.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(values, (max(1, int(round(w * scale))),
                                  max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_NEAREST if values.dtype == np.uint8
                         and np.unique(values.reshape(-1, values.shape[-1]), axis=0).shape[0] <= 4
                         else cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 24, np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _tile(image, title, detail=""):
    body = _letterbox(image)
    canvas = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 24, np.uint8)
    canvas[HEADER_HEIGHT:] = body
    cv2.putText(canvas, title, (14, 29), cv2.FONT_HERSHEY_SIMPLEX,
                0.66, (245, 245, 245), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail, (14, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (180, 210, 255), 1, cv2.LINE_AA)
    return canvas


def _mask_image(mask):
    return np.where(np.asarray(mask)[..., None] > 0, 255, 0).astype(np.uint8).repeat(3, axis=2)


def _overlay(image, mask, color, alpha=0.38):
    result = image.copy()
    tint = result.copy()
    tint[np.asarray(mask) > 0] = color
    cv2.addWeighted(tint, alpha, result, 1.0 - alpha, 0, dst=result)
    return result


def _draw_path(image, path, color, width=2, points=False):
    result = image.copy()
    values = np.rint(np.asarray(path)).astype(np.int32)
    if len(values) > 1:
        cv2.polylines(result, [values.reshape(-1, 1, 2)], False,
                      color, width, cv2.LINE_AA)
    if points:
        for point in values:
            cv2.circle(result, tuple(point), 2, color, -1, cv2.LINE_AA)
    return result


def _draw_nodes(image, nodes):
    result = image.copy()
    values = np.rint(np.asarray(nodes)).astype(np.int32)
    if len(values) > 1:
        cv2.polylines(result, [values.reshape(-1, 1, 2)], False,
                      (0, 220, 255), 2, cv2.LINE_AA)
    joint = len(values) // 2
    for index, point in enumerate(values):
        color = (0, 255, 0) if index in (0, len(values) - 1) else (
            (255, 80, 255) if index == joint else (255, 255, 255))
        cv2.circle(result, tuple(point), 3, color, -1, cv2.LINE_AA)
    for index, label in ((0, "node0 base"), (joint, f"node{joint} joint"),
                         (len(values) - 1, f"node{len(values) - 1} tip")):
        point = values[index]
        cv2.putText(result, label, (int(point[0]) + 5, int(point[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(result, label, (int(point[0]) + 5, int(point[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
    return result


def _orient_base_to_tip(path, base_anchor_local=None):
    values = np.asarray(path, dtype=np.float64)
    if base_anchor_local is None:
        first_is_base = values[0, 1] <= values[-1, 1]
    else:
        anchor = np.asarray(base_anchor_local, dtype=np.float64)
        first_is_base = (np.linalg.norm(values[0] - anchor) <=
                         np.linalg.norm(values[-1] - anchor))
    if not first_is_base:
        values = values[::-1]
    if base_anchor_local is not None:
        anchor = np.asarray(base_anchor_local, dtype=np.float64)
        if np.linalg.norm(values[0] - anchor) > 1e-6:
            values = np.concatenate([anchor[None], values], axis=0)
    return values


def _choose_frame(rows, requested, required_dirs):
    def available(frame):
        return all(os.path.isfile(os.path.join(directory, f"{frame:05d}.png"))
                   for directory in required_dirs)

    if requested is not None:
        if not available(requested):
            raise FileNotFoundError(f"frame {requested} 的阶段产物不完整")
        return int(requested)
    selected = []
    for row in rows:
        try:
            frame = int(row["frame"])
            is_selected = int(row.get("selected", 0)) == 1
            quality = float(row.get("quality", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if is_selected and available(frame):
            selected.append((quality, frame))
    if not selected:
        raise RuntimeError("找不到阶段产物完整的已选 SAM2 锚帧")
    return max(selected)[1]


def _sam2_context(rows, frame, chunk_size):
    frame_rows = []
    for row in rows:
        try:
            frame_rows.append((int(row["frame"]), row))
        except (KeyError, ValueError):
            continue
    first = min(value for value, _ in frame_rows)
    last = max(value for value, _ in frame_rows)
    chunk_index = (frame - first) // chunk_size
    start = first + chunk_index * chunk_size
    end = min(last, start + chunk_size - 1)
    anchors = [value for value, row in frame_rows
               if start <= value <= end and int(row.get("selected", 0)) == 1]
    anchor = anchors[0] if anchors else None
    direction = "unknown"
    if anchor is not None:
        direction = "anchor prompt" if frame == anchor else (
            "forward propagation" if frame > anchor else "reverse propagation")
    return {"chunk_index": chunk_index, "chunk_range": [start, end],
            "anchor_frame": anchor, "direction": direction}


def _load_training_sample(dataset_root, frame, metrics_rows):
    row = next((item for item in metrics_rows
                if int(item.get("frame", -1)) == int(frame)), None)
    if row is None:
        return None
    sample_index = int(row["index"])
    files = (sorted(glob.glob(os.path.join(dataset_root, "train", "*.npz"))) +
             sorted(glob.glob(os.path.join(dataset_root, "val", "*.npz"))))
    offset = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            count = int(data["positions"].shape[0])
            if sample_index < offset + count:
                local = sample_index - offset
                scalar = lambda key, default=None: (data[key].item()
                    if key in data and data[key].ndim == 0 else
                    (data[key].tolist() if key in data else default))
                return {
                    "positions": np.asarray(data["positions"][local]),
                    "positions_camera_px": (np.asarray(data["positions_camera_px"][local])
                                            if "positions_camera_px" in data else None),
                    "actions": np.asarray(data["actions"][local]),
                    "state_coordinate_frame": str(scalar(
                        "state_coordinate_frame", "camera_pixel_v1")),
                    "state_length_unit": str(scalar("state_length_unit", "px")),
                    "model_action_channels": scalar("model_action_channels", []),
                    "source_file": os.path.abspath(path),
                    "sample_index": sample_index,
                }
            offset += count
    raise IndexError(f"骨架 sample index {sample_index} 超出 NPZ 总帧数 {offset}")


def _state_plot(sample):
    canvas = np.full((480, 480, 3), 250, np.uint8)
    values = np.asarray(sample["positions"], dtype=float)[:2].T
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if not len(values):
        return canvas
    low = np.minimum(values.min(axis=0), (0.0, 0.0))
    high = np.maximum(values.max(axis=0), (0.0, 0.0))
    center = 0.5 * (low + high)
    span = max(float((high - low).max()), 1.0) * 1.25
    left, right, top, bottom = 62, 455, 36, 390

    def map_xy(point):
        x = left + (point[0] - (center[0] - span / 2)) / span * (right - left)
        y = bottom - (point[1] - (center[1] - span / 2)) / span * (bottom - top)
        return int(round(x)), int(round(y))

    zero = map_xy((0.0, 0.0))
    cv2.line(canvas, (left, zero[1]), (right, zero[1]), (170, 170, 170), 1)
    cv2.line(canvas, (zero[0], top), (zero[0], bottom), (170, 170, 170), 1)
    mapped = np.asarray([map_xy(point) for point in values], dtype=np.int32)
    cv2.polylines(canvas, [mapped.reshape(-1, 1, 2)], False,
                  (30, 130, 220), 3, cv2.LINE_AA)
    for index, point in enumerate(mapped):
        color = (20, 170, 20) if index in (0, len(mapped) - 1) else (80, 80, 80)
        cv2.circle(canvas, tuple(point), 4, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "x: lateral", (right - 80, min(bottom + 25, 430)),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "y: axial", (max(8, zero[0] + 8), top + 12),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (60, 60, 60), 1, cv2.LINE_AA)
    actions = ", ".join(f"{value:.3f}" for value in sample["actions"])
    cv2.putText(canvas, f"normalized action6: [{actions}]", (16, 438),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (40, 40, 40), 1, cv2.LINE_AA)
    channels = sample.get("model_action_channels") or []
    cv2.putText(canvas, f"model channels: {channels}", (16, 462),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (40, 40, 40), 1, cv2.LINE_AA)
    return canvas


def save_stage_example(*, seq, camera, derived, masks_dir, dataset_root,
                       frame=None, mask_close_k=11, n_points=15,
                       segment_lengths=(1.0, 1.0), base_anchor_source=None,
                       out_dir=None):
    seq = os.path.abspath(seq)
    derived = os.path.abspath(derived)
    masks_dir = os.path.abspath(masks_dir)
    dataset_root = os.path.abspath(dataset_root)
    crop_root = os.path.join(derived, "crop")
    crop_dir = os.path.join(crop_root, camera)
    candidate_dir = os.path.join(derived, "masks_candidate")
    rows = _read_csv(os.path.join(derived, "anchor_manifest.csv"))
    requested_frame = frame
    frame = _choose_frame(rows, frame, (crop_dir, candidate_dir, masks_dir))
    context = _sam2_context(rows, frame, int(_read_json(
        os.path.join(derived, "candidate_summary.json"), {}).get("chunk_size", 200)))

    crop_meta = _read_json(os.path.join(crop_root, "crop_meta.json"), {})
    x, y, width, height = [int(value) for value in crop_meta["crop_xywh"]]
    raw = _image(os.path.join(seq, camera, f"{frame:05d}.png"))
    crop = _image(os.path.join(crop_dir, f"{frame:05d}.png"))
    background = _image(os.path.join(derived, "bg_median.png"), cv2.IMREAD_GRAYSCALE)
    summary = _read_json(os.path.join(derived, "candidate_summary.json"), {})
    params = summary.get("segmentation_params", {})
    trim = summary.get("base_attachment_trim", {})
    candidate = candidate_stages(
        crop, background, params, summary.get("base_side", "top"),
        float(trim.get("width_ratio", 1.5)), int(trim.get("stable_span", 5)))
    sam2_mask = (_image(os.path.join(masks_dir, f"{frame:05d}.png"),
                        cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    centerline_mask = prepare_centerline_mask(sam2_mask, int(mask_close_k))

    base_local = None
    if base_anchor_source is not None:
        base_local = (float(base_anchor_source[0]) - x,
                      float(base_anchor_source[1]) - y)
    main_path, _ = _medial_longest_path(
        centerline_mask, algorithm="skeletonize", anchor_xy=base_local)
    if main_path is None:
        raise RuntimeError(f"frame {frame} 无法提取最长中心线")
    main_path = _orient_base_to_tip(main_path, base_local)
    fixed_path, _, _, tip_reason, base_reason = _fix_path_endcaps(
        centerline_mask, main_path, enabled=True, fix_base=base_local is None)
    nodes_local, intervals, joints = _resample_path_segmented(
        fixed_path, int(n_points), tuple(float(value) for value in segment_lengths))

    from skimage.morphology import skeletonize
    medial = skeletonize(centerline_mask > 0)
    raw_roi = raw.copy()
    cv2.rectangle(raw_roi, (x, y), (x + width - 1, y + height - 1),
                  (0, 220, 0), 2)
    medial_overlay = crop.copy()
    medial_overlay[medial] = (255, 0, 255)
    main_overlay = _draw_path(_overlay(crop, centerline_mask, (0, 150, 0), .22),
                              main_path, (255, 255, 0), 1)
    fixed_overlay = _draw_path(_overlay(crop, centerline_mask, (0, 150, 0), .22),
                               main_path, (255, 255, 0), 1)
    fixed_overlay = _draw_path(fixed_overlay, fixed_path, (0, 220, 255), 2)
    nodes_overlay = _draw_nodes(
        _overlay(crop, centerline_mask, (0, 150, 0), .22), nodes_local)

    metrics_rows = _read_csv(os.path.join(dataset_root, "qc_skeleton",
                                          "skeleton_metrics.csv"))
    sample = _load_training_sample(dataset_root, frame, metrics_rows)
    camera_overlay = raw.copy()
    nodes_camera = nodes_local + np.asarray((x, y), dtype=float)
    if sample is not None and sample["positions_camera_px"] is not None:
        nodes_camera = sample["positions_camera_px"][:2].T
    camera_overlay = _draw_nodes(camera_overlay, nodes_camera)

    sam_detail = f"chunk {context['chunk_range'][0]}-{context['chunk_range'][1]}, " \
                 f"anchor f{context['anchor_frame']}, {context['direction']}"
    stages = [
        ("01_raw_camera_roi", "01 Raw camera + ROI", f"frame {frame}, source pixels", raw_roi),
        ("02_fixed_roi_crop", "02 Fixed ROI crop", f"x={x}, y={y}, w={width}, h={height}", crop),
        ("03_median_background", "03 Median background", "sequence-level gray reference", background),
        ("04_white_appearance", "04 White appearance", "HSV low saturation + high value", _mask_image(candidate["white"])),
        ("05_background_motion", "05 Background motion", "gray difference + dilation", _mask_image(candidate["moved"])),
        ("06_white_motion_gate", "06 White & motion gate", "appearance intersect motion", _mask_image(candidate["gated"])),
        ("07_morphology_fill", "07 Morphology + fill", "open, close, fill holes", _mask_image(candidate["morph"])),
        ("08_main_component", "08 Main component", "area/height filtered component", _mask_image(candidate["pretrim"])),
        ("09_candidate_prompt", "09 Candidate prompt", "base attachment trim", _overlay(crop, candidate["final"], (0, 220, 0))),
        ("10_sam2_video_mask", "10 SAM2 video mask", sam_detail, _overlay(crop, sam2_mask, (0, 220, 0))),
        ("11_close_occlusion_gaps", "11 Close occlusion gaps", f"ellipse kernel={mask_close_k}", _mask_image(centerline_mask)),
        ("12_medial_skeleton", "12 One-pixel skeleton", f"pixels={int(medial.sum())}", medial_overlay),
        ("13_longest_main_path", "13 Longest main path", "8-neighbor graph diameter", main_overlay),
        ("14_endcap_corrected_path", "14 Endcap-corrected path", f"tip={tip_reason}, base={base_reason}", fixed_overlay),
        ("15_resampled_nodes", "15 Resampled nodes", f"N={n_points}, intervals={intervals}", nodes_overlay),
        ("16_camera_training_state", "16 Camera-space state", "node0 base -> node14 tip", camera_overlay),
    ]
    if sample is not None:
        unit = sample["state_length_unit"]
        stages.append(("17_model_training_sample", "17 Model training sample",
                       f"{sample['state_coordinate_frame']} [{unit}] + action6",
                       _state_plot(sample)))

    output = os.path.abspath(out_dir or os.path.join(derived, "qc_pipeline_example"))
    os.makedirs(output, exist_ok=True)
    tiles = []
    stage_records = []
    for key, title, detail, image in stages:
        tile = _tile(image, title, detail)
        filename = f"{key}.png"
        cv2.imwrite(os.path.join(output, filename), tile)
        tiles.append(tile)
        stage_records.append({"id": key, "title_zh": STAGE_TITLES_ZH[key],
                              "title": title, "detail": detail, "file": filename})

    columns = 4
    rows_n = int(math.ceil(len(tiles) / columns))
    overview = np.full((rows_n * TILE_HEIGHT, columns * TILE_WIDTH, 3), 18, np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        overview[row * TILE_HEIGHT:(row + 1) * TILE_HEIGHT,
                 column * TILE_WIDTH:(column + 1) * TILE_WIDTH] = tile
    overview_path = os.path.join(output, "00_pipeline_overview.png")
    cv2.imwrite(overview_path, overview)

    manifest = {
        "schema_version": 1,
        "sequence": os.path.basename(seq.rstrip("/")),
        "camera": camera,
        "frame": frame,
        "selection": ("explicit" if requested_frame is not None
                      else "best_selected_anchor"),
        "sam2_context": context,
        "crop_xywh": [x, y, width, height],
        "mask_close_kernel": int(mask_close_k),
        "n_points": int(n_points),
        "segment_lengths": [float(value) for value in segment_lengths],
        "segment_intervals": list(intervals),
        "joint_node_indices": list(joints),
        "dataset_root": dataset_root,
        "training_sample": ({key: value for key, value in sample.items()
                             if key not in {"positions", "positions_camera_px", "actions"}}
                            if sample is not None else None),
        "stages": stage_records,
    }
    with open(os.path.join(output, "stage_manifest.json"), "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    with open(os.path.join(output, "README.md"), "w", encoding="utf-8") as stream:
        stream.write(f"# 单帧自动前处理阶段图\n\n")
        stream.write(f"- 序列：`{manifest['sequence']}`\n- 相机：`{camera}`\n")
        stream.write(f"- frame：`{frame}`\n- SAM2 chunk：`{context['chunk_range']}`\n")
        stream.write(f"- chunk 锚帧：`{context['anchor_frame']}`\n")
        stream.write(f"- 当前帧生成方式：`{context['direction']}`\n\n")
        stream.write("总览：`00_pipeline_overview.png`\n\n")
        stream.write("| 阶段 | 文件 | 说明 |\n|---|---|---|\n")
        for item in stage_records:
            stream.write(f"| {item['title_zh']}（{item['title']}） | "
                         f"`{item['file']}` | {item['detail']} |\n")
    print(f">>> 单帧逐阶段 QC: {overview_path}")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="保存同一帧的完整自动前处理阶段图")
    parser.add_argument("--seq", required=True, help="原始采集序列目录")
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--derived", required=True, help="该序列 derived 目录")
    parser.add_argument("--masks-dir", required=True, help="最终 SAM2 mask 目录")
    parser.add_argument("--dataset-root", required=True, help="转换后的数据根目录")
    parser.add_argument("--frame", type=int, default=None, help="真实 frame ID；默认自动选锚帧")
    parser.add_argument("--mask-close-k", type=int, default=11)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--segment-lengths", default="1,1")
    parser.add_argument("--base-anchor", default=None, help="源相机像素 x,y")
    parser.add_argument("--out", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    segment_lengths = tuple(float(value) for value in args.segment_lengths.split(",")
                            if value.strip())
    base_anchor = (tuple(float(value) for value in args.base_anchor.split(","))
                   if args.base_anchor else None)
    save_stage_example(
        seq=args.seq, camera=args.camera, derived=args.derived,
        masks_dir=args.masks_dir, dataset_root=args.dataset_root,
        frame=args.frame, mask_close_k=args.mask_close_k,
        n_points=args.n_points, segment_lengths=segment_lengths,
        base_anchor_source=base_anchor, out_dir=args.out)


if __name__ == "__main__":
    main()
