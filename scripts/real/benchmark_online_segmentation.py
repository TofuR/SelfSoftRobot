"""比较单帧分割、因果 SAM2 前向传播与离线双向 SAM2 标签。

离线双向 SAM2 mask 是本项目当前二维视觉伪 GT。因果 SAM2 每个块只在块首帧使用
当时可获得的候选 mask，并仅向未来传播；输出不读取未来帧。官方 video predictor
需要目录形式的视频 state，因此脚本把 JPEG 暂存、state 初始化和逐帧模型计算分别计时。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_validation.perception.skeleton import extract_centerline_2d  # noqa: E402
from scripts.real.masks_to_transition_npz import prepare_centerline_mask  # noqa: E402
from scripts.real.prepare_sam2_anchors import candidate_stages  # noqa: E402
from sam2.segment_video_full import build_predictor, prepare_jpeg_dir  # noqa: E402


def _read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def mask_metrics(predicted, reference):
    predicted = np.asarray(predicted) > 0
    reference = np.asarray(reference) > 0
    intersection = int(np.logical_and(predicted, reference).sum())
    union = int(np.logical_or(predicted, reference).sum())
    total = int(predicted.sum() + reference.sum())
    return {
        "iou": float(intersection / union) if union else 1.0,
        "dice": float(2 * intersection / total) if total else 1.0,
        "area_ratio": float(predicted.sum() / reference.sum())
        if reference.any() else math.nan,
    }


def skeleton_metrics(predicted, reference, mm_per_px):
    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape or not np.isfinite(predicted).all() \
            or not np.isfinite(reference).all():
        return {"node_mean_px": math.nan, "tip_px": math.nan,
                "base_px": math.nan, "node_mean_mm": math.nan,
                "tip_mm": math.nan}
    errors = np.linalg.norm(predicted - reference, axis=1)
    return {
        "node_mean_px": float(errors.mean()),
        "tip_px": float(errors[-1]),
        "base_px": float(errors[0]),
        "node_mean_mm": float(errors.mean() * mm_per_px),
        "tip_mm": float(errors[-1] * mm_per_px),
    }


def extract_nodes(mask, *, n_points, segment_lengths, base_anchor_local,
                  mask_close_k):
    prepared = prepare_centerline_mask(mask, mask_close_k)
    nodes, info = extract_centerline_2d(
        prepared, n_points=n_points, method="skeletonize",
        segment_lengths=segment_lengths, base_anchor_xy=base_anchor_local,
        endpoint_fix=True, return_info=True)
    valid = bool(info.get("success")) and np.isfinite(nodes).all()
    return np.asarray(nodes, dtype=np.float32), info, valid


def _load_mm_per_px(dataset_root):
    for split in ("train", "val"):
        paths = sorted(Path(dataset_root, split).glob("*.npz"))
        if paths:
            with np.load(paths[0], allow_pickle=False) as data:
                if "mm_per_px" in data:
                    return float(data["mm_per_px"].item())
                if "robot_diameter_px" in data and "robot_diameter_mm" in data:
                    return float(data["robot_diameter_mm"].item() /
                                 data["robot_diameter_px"].item())
    raise ValueError(f"数据目录缺少固定 mm/px 合同: {dataset_root}")


def _load_contract(seq, camera, derived, offline_masks, dataset_root):
    crop_meta = _read_json(os.path.join(derived, "crop", "crop_meta.json"), {})
    summary = _read_json(os.path.join(derived, "candidate_summary.json"), {})
    preprocess = _read_json(os.path.join(derived, "preprocess_manifest.json"), {})
    crop_xywh = tuple(int(value) for value in crop_meta["crop_xywh"])
    base_source = preprocess.get("base_anchor_source_xy")
    base_local = None
    if base_source is not None:
        base_local = (float(base_source[0]) - crop_xywh[0],
                      float(base_source[1]) - crop_xywh[1])
    return {
        "seq": os.path.abspath(seq),
        "camera": camera,
        "derived": os.path.abspath(derived),
        "crop_dir": os.path.join(derived, "crop", camera),
        "candidate_dir": os.path.join(derived, "masks_candidate"),
        "background_path": os.path.join(derived, "bg_median.png"),
        "offline_masks": os.path.abspath(offline_masks),
        "dataset_root": os.path.abspath(dataset_root),
        "crop_xywh": crop_xywh,
        "base_anchor_source_xy": base_source,
        "base_anchor_local_xy": base_local,
        "segmentation_params": summary.get("segmentation_params", {}),
        "base_side": summary.get("base_side", "top"),
        "base_attachment_trim": summary.get("base_attachment_trim", {}),
        "mm_per_px": _load_mm_per_px(dataset_root),
    }


def _frame_ids(directory):
    values = []
    for path in Path(directory).glob("*.png"):
        try:
            values.append(int(path.stem))
        except ValueError:
            continue
    return sorted(values)


def _traditional_mask(image, background, contract):
    trim = contract["base_attachment_trim"]
    stages = candidate_stages(
        image, background, contract["segmentation_params"],
        contract["base_side"], float(trim.get("width_ratio", 1.5)),
        int(trim.get("stable_span", 5)))
    return stages["final"]


def _draw_nodes(image, nodes, color):
    result = image.copy()
    values = np.rint(np.asarray(nodes)).astype(np.int32)
    if len(values) > 1:
        cv2.polylines(result, [values.reshape(-1, 1, 2)], False,
                      color, 2, cv2.LINE_AA)
    for point in values:
        cv2.circle(result, tuple(point), 2, color, -1, cv2.LINE_AA)
    return result


def _overlay(image, mask, color):
    result = image.copy()
    tint = result.copy()
    tint[np.asarray(mask) > 0] = color
    cv2.addWeighted(tint, .35, result, .65, 0, dst=result)
    return result


def _label(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 30), (20, 20, 20), -1)
    cv2.putText(result, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def _save_montage(output, records, contract, online_dir, frame_ids, n=8):
    finite = [row for row in records if np.isfinite(row["causal_mask_iou"])]
    worst = sorted(finite, key=lambda row: row["causal_mask_iou"])[:max(2, n // 2)]
    even_indices = np.linspace(0, len(frame_ids) - 1, min(n, len(frame_ids))).astype(int)
    chosen = list(dict.fromkeys([int(row["frame"]) for row in worst] +
                                [frame_ids[index] for index in even_indices]))[:n]
    background = cv2.imread(contract["background_path"], cv2.IMREAD_GRAYSCALE)
    rows = []
    by_frame = {int(row["frame"]): row for row in records}
    for frame in chosen:
        image = cv2.imread(os.path.join(contract["crop_dir"], f"{frame:05d}.png"))
        offline = (cv2.imread(os.path.join(contract["offline_masks"], f"{frame:05d}.png"),
                              cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        online = (cv2.imread(os.path.join(online_dir, f"{frame:05d}.png"),
                             cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        traditional = _traditional_mask(image, background, contract)
        offline_nodes, _, _ = extract_nodes(
            offline, n_points=15, segment_lengths=(1, 1),
            base_anchor_local=contract["base_anchor_local_xy"], mask_close_k=11)
        online_nodes, _, _ = extract_nodes(
            online, n_points=15, segment_lengths=(1, 1),
            base_anchor_local=contract["base_anchor_local_xy"], mask_close_k=11)
        traditional_nodes, _, _ = extract_nodes(
            traditional, n_points=15, segment_lengths=(1, 1),
            base_anchor_local=contract["base_anchor_local_xy"], mask_close_k=11)
        metric = by_frame[frame]
        cells = [
            _label(image, f"f{frame} RGB"),
            _label(_draw_nodes(_overlay(image, traditional, (0, 0, 255)),
                               traditional_nodes, (0, 255, 255)),
                   f"single IoU={metric['traditional_mask_iou']:.3f}"),
            _label(_draw_nodes(_overlay(image, online, (0, 180, 255)),
                               online_nodes, (255, 255, 0)),
                   f"causal IoU={metric['causal_mask_iou']:.3f}"),
            _label(_draw_nodes(_overlay(image, offline, (0, 220, 0)),
                               offline_nodes, (255, 255, 255)),
                   "offline bidirectional pseudo-GT"),
        ]
        rows.append(np.hstack(cells))
    if rows:
        cv2.imwrite(os.path.join(output, "method_comparison_montage.png"),
                    np.vstack(rows))


def _save_plots(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = np.asarray([row["frame"] for row in records])
    traditional_iou = np.asarray([row["traditional_mask_iou"] for row in records])
    causal_iou = np.asarray([row["causal_mask_iou"] for row in records])
    traditional_nodes = np.asarray([row["traditional_node_mean_mm"] for row in records])
    causal_nodes = np.asarray([row["causal_node_mean_mm"] for row in records])
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(frame, traditional_iou, lw=.65, label="single-frame")
    axes[0].plot(frame, causal_iou, lw=.65, label="causal SAM2")
    axes[0].set_ylabel("mask IoU vs offline SAM2")
    axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=.25); axes[0].legend()
    axes[1].plot(frame, traditional_nodes, lw=.65, label="single-frame")
    axes[1].plot(frame, causal_nodes, lw=.65, label="causal SAM2")
    axes[1].set_ylabel("15-node mean error [mm]")
    axes[1].set_xlabel("frame"); axes[1].grid(alpha=.25); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(output, "accuracy_over_time.png"), dpi=140)
    plt.close(fig)

    traditional_ms = np.asarray([row["traditional_total_ms"] for row in records])
    causal_ms = np.asarray([row["causal_forward_skeleton_ms"] for row in records])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([traditional_ms[np.isfinite(traditional_ms)],
                causal_ms[np.isfinite(causal_ms)]],
               tick_labels=["single-frame segmentation+skeleton",
                            "causal SAM2 forward+skeleton"], showfliers=False)
    ax.set_ylabel("latency [ms]"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(output, "latency_boxplot.png"), dpi=140)
    plt.close(fig)


def summarize_causal_by_k(records, bin_size=25):
    """汇总周期锚定后不同因果步距的精度，区分漂移与片段差异。"""
    result = []
    max_k = max(int(row["causal_k_from_anchor"]) for row in records)
    for start in range(0, max_k + 1, int(bin_size)):
        rows = [row for row in records
                if start <= int(row["causal_k_from_anchor"]) < start + bin_size]
        if not rows:
            continue
        result.append({
            "k_start": start,
            "k_end": start + bin_size - 1,
            "frames": len(rows),
            "mask_iou_mean": float(np.mean(
                [row["causal_mask_iou"] for row in rows])),
            "node_mean_mm": float(np.mean(
                [row["causal_node_mean_mm"] for row in rows])),
            "tip_mm": float(np.mean([row["causal_tip_mm"] for row in rows])),
        })
    return result


def _save_k_drift(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(output, "accuracy_by_k.csv"), "w", newline="",
              encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    center = np.asarray([(row["k_start"] + row["k_end"]) / 2 for row in rows])
    fig, left = plt.subplots(figsize=(10, 5))
    right = left.twinx()
    left.plot(center, [row["mask_iou_mean"] for row in rows], "o-",
              color="tab:blue", label="mask IoU")
    right.plot(center, [row["node_mean_mm"] for row in rows], "s-",
               color="tab:orange", label="15-node mean")
    left.set_xlabel("causal frames since current anchor")
    left.set_ylabel("mask IoU vs offline SAM2", color="tab:blue")
    right.set_ylabel("15-node mean error [mm]", color="tab:orange")
    left.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(output, "accuracy_by_k.png"), dpi=140)
    plt.close(fig)


def offline_anchor_lookahead(derived, source_hz, chunk_size=200):
    rows = []
    path = os.path.join(derived, "anchor_manifest.csv")
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    anchors = sorted(int(row["frame"]) for row in rows
                     if int(row.get("selected", 0)) == 1)
    first_frame = min((int(row["frame"]) for row in rows), default=0)
    offsets = np.asarray([(frame - first_frame) % int(chunk_size)
                          for frame in anchors], dtype=float)
    return {
        "selected_anchor_frames": anchors,
        "lookahead_frames": _percentiles(offsets),
        "lookahead_seconds": (_percentiles(offsets / float(source_hz))
                              if source_hz else None),
        "meaning": "frames before the selected anchor require this future-frame wait",
    }


def benchmark(args):
    seq = os.path.abspath(args.seq)
    seq_name = os.path.basename(seq.rstrip("/"))
    derived = os.path.abspath(args.derived or os.path.join(
        PROJECT_ROOT, "real_capture", "data", "derived", seq_name))
    offline_masks = os.path.abspath(args.offline_masks or os.path.join(
        PROJECT_ROOT, "sam2", "masks", f"{seq_name}_full"))
    dataset_root = os.path.abspath(args.dataset_root)
    output = os.path.abspath(args.output or os.path.join(
        PROJECT_ROOT, "output", "online_segmentation_benchmark", seq_name))
    online_dir = os.path.join(output, "causal_sam2_masks")
    os.makedirs(online_dir, exist_ok=True)
    contract = _load_contract(seq, args.camera, derived, offline_masks, dataset_root)
    frames = _frame_ids(contract["crop_dir"])
    reference_frames = _frame_ids(offline_masks)
    if frames != reference_frames:
        raise ValueError("裁剪图与离线SAM2 mask帧号不一致")
    if args.max_frames:
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("没有可评价帧")
    background = cv2.imread(contract["background_path"], cv2.IMREAD_GRAYSCALE)
    if background is None:
        raise FileNotFoundError(contract["background_path"])

    records = []
    offline_nodes = {}
    traditional_masks = {}
    print(f">>> 单帧基线与离线参考骨架: {len(frames)} 帧", flush=True)
    for index, frame in enumerate(frames):
        image = cv2.imread(os.path.join(contract["crop_dir"], f"{frame:05d}.png"))
        reference = (cv2.imread(os.path.join(offline_masks, f"{frame:05d}.png"),
                                cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        start = time.perf_counter()
        traditional = _traditional_mask(image, background, contract)
        segment_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        traditional_nodes, _, traditional_valid = extract_nodes(
            traditional, n_points=args.n_points,
            segment_lengths=args.segment_lengths,
            base_anchor_local=contract["base_anchor_local_xy"],
            mask_close_k=args.mask_close_k)
        skeleton_ms = (time.perf_counter() - start) * 1000
        reference_nodes, _, reference_valid = extract_nodes(
            reference, n_points=args.n_points,
            segment_lengths=args.segment_lengths,
            base_anchor_local=contract["base_anchor_local_xy"],
            mask_close_k=args.mask_close_k)
        masks = mask_metrics(traditional, reference)
        nodes = skeleton_metrics(traditional_nodes, reference_nodes,
                                 contract["mm_per_px"])
        records.append({
            "frame": frame,
            "traditional_mask_iou": masks["iou"],
            "traditional_mask_dice": masks["dice"],
            "traditional_area_ratio": masks["area_ratio"],
            "traditional_node_mean_px": nodes["node_mean_px"],
            "traditional_tip_px": nodes["tip_px"],
            "traditional_node_mean_mm": nodes["node_mean_mm"],
            "traditional_tip_mm": nodes["tip_mm"],
            "traditional_valid": int(traditional_valid),
            "offline_valid": int(reference_valid),
            "traditional_segment_ms": segment_ms,
            "traditional_skeleton_ms": skeleton_ms,
            "traditional_total_ms": segment_ms + skeleton_ms,
        })
        offline_nodes[frame] = reference_nodes
        traditional_masks[frame] = traditional
        if (index + 1) % 500 == 0:
            print(f"    {index + 1}/{len(frames)}", flush=True)

    import torch
    torch.cuda.set_device(args.device)
    torch.cuda.synchronize(args.device)
    start_model = time.perf_counter()
    predictor = build_predictor(args.device)
    torch.cuda.synchronize(args.device)
    model_load_s = time.perf_counter() - start_model
    print(f">>> SAM2 model loaded in {model_load_s:.3f}s on {args.device}", flush=True)

    chunk_records = []
    record_by_frame = {int(row["frame"]): row for row in records}
    with tempfile.TemporaryDirectory(prefix=f"{seq_name}_causal_", dir=args.temp_root) as temp:
        for chunk_index, offset in enumerate(range(0, len(frames), args.reanchor_interval)):
            chunk_frames = frames[offset:offset + args.reanchor_interval]
            jpeg_dir = os.path.join(temp, f"chunk_{chunk_index:04d}")
            start = time.perf_counter()
            if not prepare_jpeg_dir(contract["crop_dir"], chunk_frames, jpeg_dir):
                raise RuntimeError(f"chunk {chunk_index} JPEG准备失败")
            jpeg_staging_s = time.perf_counter() - start

            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            state = predictor.init_state(
                video_path=jpeg_dir, offload_video_to_cpu=True,
                offload_state_to_cpu=False, async_loading_frames=False)
            torch.cuda.synchronize(args.device)
            state_init_s = time.perf_counter() - start

            anchor_frame = chunk_frames[0]
            anchor_mask = traditional_masks[anchor_frame]
            torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=anchor_mask)
            torch.cuda.synchronize(args.device)
            prompt_s = time.perf_counter() - start

            generator = predictor.propagate_in_video(
                state, start_frame_idx=0,
                max_frame_num_to_track=len(chunk_frames) - 1, reverse=False)
            yielded = 0
            for local_index in range(len(chunk_frames)):
                torch.cuda.synchronize(args.device)
                start = time.perf_counter()
                frame_index, _, mask_logits = next(generator)
                mask = (mask_logits[0].detach().cpu().numpy() > 0).squeeze().astype(np.uint8)
                torch.cuda.synchronize(args.device)
                forward_ms = (time.perf_counter() - start) * 1000
                frame = chunk_frames[int(frame_index)]
                cv2.imwrite(os.path.join(online_dir, f"{frame:05d}.png"), mask * 255)
                start = time.perf_counter()
                nodes_online, _, online_valid = extract_nodes(
                    mask, n_points=args.n_points,
                    segment_lengths=args.segment_lengths,
                    base_anchor_local=contract["base_anchor_local_xy"],
                    mask_close_k=args.mask_close_k)
                skeleton_ms = (time.perf_counter() - start) * 1000
                reference = (cv2.imread(os.path.join(
                    offline_masks, f"{frame:05d}.png"), cv2.IMREAD_GRAYSCALE) > 127)
                masks = mask_metrics(mask, reference)
                nodes = skeleton_metrics(nodes_online, offline_nodes[frame],
                                         contract["mm_per_px"])
                row = record_by_frame[frame]
                row.update({
                    "causal_mask_iou": masks["iou"],
                    "causal_mask_dice": masks["dice"],
                    "causal_area_ratio": masks["area_ratio"],
                    "causal_node_mean_px": nodes["node_mean_px"],
                    "causal_tip_px": nodes["tip_px"],
                    "causal_node_mean_mm": nodes["node_mean_mm"],
                    "causal_tip_mm": nodes["tip_mm"],
                    "causal_valid": int(online_valid),
                    "causal_forward_ms": forward_ms,
                    "causal_skeleton_ms": skeleton_ms,
                    "causal_forward_skeleton_ms": forward_ms + skeleton_ms,
                    "causal_chunk": chunk_index,
                    "causal_anchor_frame": anchor_frame,
                    "causal_k_from_anchor": int(frame - anchor_frame),
                })
                yielded += 1
            if yielded != len(chunk_frames):
                raise RuntimeError(f"chunk {chunk_index} 输出不完整")
            chunk_records.append({
                "chunk": chunk_index,
                "frame_start": chunk_frames[0], "frame_end": chunk_frames[-1],
                "frames": len(chunk_frames), "anchor_frame": anchor_frame,
                "jpeg_staging_s": jpeg_staging_s,
                "state_init_s": state_init_s,
                "prompt_s": prompt_s,
                "state_init_amortized_ms": state_init_s * 1000 / len(chunk_frames),
                "prompt_amortized_ms": prompt_s * 1000 / len(chunk_frames),
            })
            del generator, state
            gc.collect(); torch.cuda.empty_cache()
            shutil.rmtree(jpeg_dir, ignore_errors=True)
            print(f"    causal chunk {chunk_index + 1}/"
                  f"{math.ceil(len(frames) / args.reanchor_interval)} "
                  f"frames={len(chunk_frames)} init={state_init_s:.2f}s", flush=True)

    fieldnames = sorted({key for row in records for key in row})
    with open(os.path.join(output, "per_frame.csv"), "w", newline="",
              encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records)

    def summarize_method(prefix):
        return {
            "mask_iou": _percentiles([row[f"{prefix}_mask_iou"] for row in records]),
            "mask_dice": _percentiles([row[f"{prefix}_mask_dice"] for row in records]),
            "node_mean_mm": _percentiles([row[f"{prefix}_node_mean_mm"] for row in records]),
            "tip_mm": _percentiles([row[f"{prefix}_tip_mm"] for row in records]),
            "valid_fraction": float(np.mean([row[f"{prefix}_valid"] for row in records])),
        }

    init_amortized = [row["state_init_amortized_ms"] for row in chunk_records]
    prompt_amortized = [row["prompt_amortized_ms"] for row in chunk_records]
    forward_skeleton = [row["causal_forward_skeleton_ms"] for row in records]
    causal_estimated = [value + float(np.mean(init_amortized)) +
                        float(np.mean(prompt_amortized)) for value in forward_skeleton]
    summary = {
        "schema_version": 1,
        "sequence": seq_name,
        "frames": len(frames),
        "source_hz": None,
        "reference": "offline_bidirectional_sam2_pseudo_gt",
        "causal_sam2": {
            "anchor_policy": "current single-frame candidate at each block start",
            "future_frames_used": False,
            "reanchor_interval_frames": args.reanchor_interval,
            "device": args.device,
            "model_load_s": model_load_s,
            "model_input_size": int(predictor.image_size),
            "state_init_amortized_ms": _percentiles(init_amortized),
            "prompt_amortized_ms": _percentiles(prompt_amortized),
            "forward_ms": _percentiles([row["causal_forward_ms"] for row in records]),
            "skeleton_ms": _percentiles([row["causal_skeleton_ms"] for row in records]),
            "forward_plus_skeleton_ms": _percentiles(forward_skeleton),
            "estimated_live_total_ms": _percentiles(causal_estimated),
            "jpeg_staging_s": _percentiles([row["jpeg_staging_s"]
                                              for row in chunk_records]),
            "timing_scope": (
                "estimated_live_total = synchronous state image preprocessing amortization "
                "+ prompt amortization + GPU forward/output transfer + CPU skeleton; "
                "camera exposure and offline PNG-to-JPEG staging excluded"),
            "api_scope": (
                "forward masks are causal; timing uses the official directory-based "
                "video state with synchronous image preprocessing. real_validation "
                "still needs an incremental frame adapter before deployment"),
        },
        "traditional_latency_ms": {
            "segmentation": _percentiles([row["traditional_segment_ms"] for row in records]),
            "skeleton": _percentiles([row["traditional_skeleton_ms"] for row in records]),
            "total": _percentiles([row["traditional_total_ms"] for row in records]),
        },
        "accuracy_vs_offline_bidirectional_sam2": {
            "traditional_single_frame": summarize_method("traditional"),
            "causal_sam2": summarize_method("causal"),
        },
        "coordinate_scale": {"mm_per_px": contract["mm_per_px"],
                             "robot_diameter_mm": 16.0},
        "chunk_timing": chunk_records,
        "contract": contract,
    }
    times_path = os.path.join(seq, "frame_times.txt")
    if os.path.isfile(times_path):
        timestamps = np.atleast_1d(np.loadtxt(times_path))[:len(frames)]
        if len(timestamps) > 1:
            summary["source_hz"] = float(1.0 / np.median(np.diff(timestamps)))
    k_values = np.asarray([row["causal_k_from_anchor"] for row in records], dtype=float)
    causal_iou = np.asarray([row["causal_mask_iou"] for row in records], dtype=float)
    causal_nodes = np.asarray([row["causal_node_mean_mm"] for row in records], dtype=float)
    summary["causal_sam2"]["accuracy_by_k"] = summarize_causal_by_k(records)
    summary["causal_sam2"]["k_correlation"] = {
        "mask_iou": float(np.corrcoef(k_values, causal_iou)[0, 1]),
        "node_mean_mm": float(np.corrcoef(k_values, causal_nodes)[0, 1]),
    }
    summary["offline_bidirectional_sam2"] = offline_anchor_lookahead(
        derived, summary["source_hz"], args.reanchor_interval)
    with open(os.path.join(output, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    _save_plots(output, records)
    _save_k_drift(output, summary["causal_sam2"]["accuracy_by_k"])
    _save_montage(output, records, contract, online_dir, frames)
    print(f">>> benchmark complete: {os.path.join(output, 'summary.json')}")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="在线分割与离线双向SAM2伪GT比较")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--derived", default=None)
    parser.add_argument("--offline-masks", default=None)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--reanchor-interval", type=int, default=200)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--segment-lengths", default="1,1")
    parser.add_argument("--mask-close-k", type=int, default=11)
    parser.add_argument("--temp-root", default="/tmp")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.segment_lengths = tuple(float(value) for value in
                                 args.segment_lengths.split(",") if value.strip())
    benchmark(args)


if __name__ == "__main__":
    main()
