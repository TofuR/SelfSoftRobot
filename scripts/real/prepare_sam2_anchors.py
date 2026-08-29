"""自动生成 SAM2 候选锚点，并保存逐阶段 QC。

主线不读取动作通道，也不假设某一段静止：RGB 经白色/背景差候选分割后，对每帧按
面积、主体宽度、跨度和基座连续性自动打分，每个 SAM2 chunk 选择一个高置信锚点。
候选 mask 不是训练 GT；SAM2 传播结果仍需经过后续 mask/中心线 QC。

输出 ``<out-root>/``：
  bg_median.png, masks_candidate/, anchors/, anchor_manifest.csv,
  candidate_metrics.csv, candidate_summary.json, qc/*.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from real_validation.perception.segmentation import (  # noqa: E402
    build_median_background,
    segment_white_on_blue_stages,
)


STAGE_NAMES = ("white", "moved", "gated", "morph", "pretrim", "final")


def trim_wide_base_attachment(mask, base_side="top", width_ratio=1.5,
                              stable_span=5):
    """去掉机器人与宽支架相接处的短横向分支。

    方形上下文裁剪会把固定支架一起保留在RGB中。候选分割偶尔在base处粘住一小段
    横梁；它虽然面积很小，却会让最长骨架主路径的base端沿横梁偏移。这里仅从已分割
    mask的base侧逐截面检查宽度，删除首个稳定“主体管径”截面之前的宽附件。算法不
    读取动作，也不假设任何机器人段静止。
    """
    result = np.asarray(mask, dtype=np.uint8).copy()
    if base_side == "none" or width_ratio <= 0 or stable_span <= 0 \
            or not np.any(result):
        return result
    if base_side in ("top", "bottom"):
        widths = result.sum(axis=1)
    elif base_side in ("left", "right"):
        widths = result.sum(axis=0)
    else:
        raise ValueError(f"未知base_side: {base_side}")
    occupied = widths[widths > 0].astype(float)
    if not len(occupied):
        return result
    body_width = float(np.median(occupied))
    threshold = max(width_ratio * body_width, body_width + 2.0)
    indices = range(len(widths)) if base_side in ("top", "left") \
        else range(len(widths) - 1, -1, -1)
    ordered = list(indices)
    cut = None
    for offset in range(0, len(ordered) - stable_span + 1):
        values = widths[ordered[offset:offset + stable_span]]
        if np.all((values > 0) & (values <= threshold)):
            cut = ordered[offset]
            break
    if cut is None:
        return result
    if base_side == "top":
        result[:cut] = 0
    elif base_side == "bottom":
        result[cut + 1:] = 0
    elif base_side == "left":
        result[:, :cut] = 0
    else:
        result[:, cut + 1:] = 0
    return result


def candidate_stages(bgr, bg, params, base_side, trim_width_ratio,
                     trim_stable_span):
    stages = segment_white_on_blue_stages(bgr, bg, **params)
    stages["pretrim"] = stages["final"].copy()
    stages["final"] = trim_wide_base_attachment(
        stages["final"], base_side=base_side, width_ratio=trim_width_ratio,
        stable_span=trim_stable_span)
    return stages


def mask_metrics(mask):
    """只描述当前 mask 的几何，不使用历史动作或静态段。"""
    mask = np.asarray(mask, dtype=np.uint8)
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return {
            "area": 0, "x": 0, "y": 0, "w": 0, "h": 0,
            "top": H, "bottom": -1, "center_x": math.nan, "center_y": math.nan,
            "width_median": 0.0, "width_p95": 0.0, "width_max": 0.0,
        }
    widths = mask.sum(axis=1)
    widths = widths[widths > 0].astype(float)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return {
        "area": int(len(xs)), "x": x0, "y": y0,
        "w": x1 - x0 + 1, "h": y1 - y0 + 1,
        "top": y0, "bottom": y1,
        "center_x": float(xs.mean()), "center_y": float(ys.mean()),
        "width_median": float(np.median(widths)),
        "width_p95": float(np.percentile(widths, 95)),
        "width_max": float(widths.max()),
    }


def _robust_center_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center, max(1.4826 * mad, 0.05 * abs(center), 1.0)


def add_quality_scores(rows, image_shape, base_side="top"):
    """按序列稳健统计打分；只用于选择 SAM2 锚点，不修改 mask。"""
    H, W = image_shape
    fields = ("area", "width_median", "width_p95")
    stats = {name: _robust_center_scale([r[name] for r in rows if r["area"] > 0])
             for name in fields}
    for row in rows:
        if row["area"] <= 0:
            row["quality"] = 0.0
            row["quality_reason"] = "empty"
            continue
        z = 0.0
        for name in fields:
            center, scale = stats[name]
            z += min(abs(float(row[name]) - center) / scale, 10.0)
        span = max(row["h"] / H, row["w"] / W)
        if span < 0.12:
            z += 10.0
        if base_side == "top":
            base_dist = row["top"] / H
        elif base_side == "bottom":
            base_dist = (H - 1 - row["bottom"]) / H
        elif base_side == "left":
            base_dist = row["x"] / W
        elif base_side == "right":
            base_dist = (W - row["x"] - row["w"]) / W
        else:
            base_dist = 0.0
        z += min(base_dist / 0.04, 10.0)
        row["quality"] = float(1.0 / (1.0 + z))
        row["quality_reason"] = "ok" if row["quality"] >= 0.12 else "atypical"
    return stats


def select_chunk_anchors(rows, chunk_size):
    """每个实际帧号 chunk 选择质量最高的一帧，避免按某个动作通道挑帧。"""
    selected = set()
    if not rows:
        return selected
    first = int(rows[0]["frame"])
    groups = {}
    for row in rows:
        chunk = (int(row["frame"]) - first) // chunk_size
        groups.setdefault(chunk, []).append(row)
    for group in groups.values():
        valid = [row for row in group if row["quality"] > 0]
        if valid:
            best = max(valid, key=lambda row: (row["quality"], -abs(
                int(row["frame"]) - int(group[len(group) // 2]["frame"]))))
            selected.add(int(best["frame"]))
    return selected


def _overlay(bgr, mask, color=(0, 0, 255), alpha=0.38):
    out = bgr.copy()
    tint = out.copy()
    tint[mask > 0] = color
    cv2.addWeighted(tint, alpha, out, 1.0 - alpha, 0, dst=out)
    return out


def save_stage_qc(frame_paths, bg, params, rows, selected, qc_dir,
                  base_side="top", trim_width_ratio=1.5,
                  trim_stable_span=5, n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frame_paths:
        return
    by_frame = {int(row["frame"]): row for row in rows}
    frame_ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in frame_paths]
    evenly = np.linspace(0, len(frame_paths) - 1, min(n, len(frame_paths))).astype(int)
    worst = sorted(rows, key=lambda row: row["quality"])[:min(4, len(rows))]
    wanted = list(dict.fromkeys([frame_ids[i] for i in evenly] +
                                [int(row["frame"]) for row in worst]))
    path_map = {frame: path for frame, path in zip(frame_ids, frame_paths)}

    cols = ("rgb",) + STAGE_NAMES
    fig, axes = plt.subplots(len(wanted), len(cols),
                             figsize=(2.5 * len(cols), 2.25 * len(wanted)), squeeze=False)
    for r, frame in enumerate(wanted):
        bgr = cv2.imread(path_map[frame])
        stages = candidate_stages(
            bgr, bg, params, base_side, trim_width_ratio, trim_stable_span)
        for c, name in enumerate(cols):
            ax = axes[r, c]
            if name == "rgb":
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            elif name == "final":
                ax.imshow(cv2.cvtColor(_overlay(bgr, stages[name]), cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(stages[name], cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title("candidate final\n(not SAM2)" if name == "final" else name)
            ax.set_xticks([]); ax.set_yticks([])
        row = by_frame[frame]
        axes[r, 0].set_ylabel(
            f"f{frame}\nq={row['quality']:.3f}\nA={row['area']}" +
            ("\nANCHOR" if frame in selected else ""), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "stage_pipeline.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)

    x = np.array([row["frame"] for row in rows])
    area = np.array([row["area"] for row in rows])
    quality = np.array([row["quality"] for row in rows])
    fig, ax = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax[0].plot(x, area, lw=0.7); ax[0].set_ylabel("candidate area [px]"); ax[0].grid(alpha=.25)
    ax[1].plot(x, quality, lw=0.7); ax[1].set_ylabel("anchor quality")
    ax[1].set_xlabel("frame"); ax[1].grid(alpha=.25)
    for frame in selected:
        for cur in ax:
            cur.axvline(frame, color="tab:green", alpha=.16, lw=.8)
    fig.tight_layout()
    fig.savefig(os.path.join(qc_dir, "candidate_metrics.png"), dpi=120)
    plt.close(fig)

    anchor_frames = sorted(selected)
    if anchor_frames:
        cols_n = 4
        rows_n = int(math.ceil(len(anchor_frames) / cols_n))
        fig, axes = plt.subplots(rows_n, cols_n, figsize=(3 * cols_n, 3 * rows_n), squeeze=False)
        for k, frame in enumerate(anchor_frames):
            bgr = cv2.imread(path_map[frame])
            mask = candidate_stages(
                bgr, bg, params, base_side, trim_width_ratio,
                trim_stable_span)["final"]
            axes.flat[k].imshow(cv2.cvtColor(_overlay(bgr, mask, (0, 255, 0)), cv2.COLOR_BGR2RGB))
            axes.flat[k].set_title(f"f{frame} q={by_frame[frame]['quality']:.3f}", fontsize=8)
            axes.flat[k].axis("off")
        for k in range(len(anchor_frames), rows_n * cols_n):
            axes.flat[k].axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(qc_dir, "selected_anchors.png"), dpi=110)
        plt.close(fig)


def build_parser():
    pa = argparse.ArgumentParser(description="自动候选分割 + SAM2 锚点选择（动作维度无关）")
    pa.add_argument("--seq", required=True, help="原始序列目录，含 camN/")
    pa.add_argument("--camera", default="cam0")
    pa.add_argument("--out-root", default=None,
                    help="默认 real_capture/data/derived/<seq名>")
    pa.add_argument("--chunk-size", type=int, default=200)
    pa.add_argument("--frame-step", type=int, default=1, help="诊断抽样步长；正式运行必须为1")
    pa.add_argument("--n-bg", type=int, default=500)
    pa.add_argument("--background-image", default=None,
                    help="可选无机器人参考背景；未提供则自动使用全序列中值背景")
    pa.add_argument("--base-side", choices=("top", "bottom", "left", "right", "none"),
                    default="top", help="固定基座靠近哪一侧；只用于锚点打分")
    pa.add_argument("--base-trim-width-ratio", type=float, default=1.5,
                    help="删除base处宽支架分支的截面宽度阈值；<=0禁用")
    pa.add_argument("--base-trim-stable-span", type=int, default=5,
                    help="判定进入机器人主体所需的连续窄截面数")
    pa.add_argument("--sat", type=int, default=100)
    pa.add_argument("--val", type=int, default=120)
    pa.add_argument("--diff", type=int, default=25)
    pa.add_argument("--dil", type=int, default=35)
    pa.add_argument("--open-k", type=int, default=5)
    pa.add_argument("--close-k", type=int, default=15)
    pa.add_argument("--min-area-frac", type=float, default=0.003)
    pa.add_argument("--min-h-frac", type=float, default=0.15)
    return pa


def main(argv=None):
    args = build_parser().parse_args(argv)
    seq = os.path.abspath(args.seq.rstrip("/"))
    seq_name = os.path.basename(seq)
    out_root = args.out_root or os.path.join(
        PROJECT_ROOT, "real_capture", "data", "derived", seq_name)
    mask_dir = os.path.join(out_root, "masks_candidate")
    anchor_dir = os.path.join(out_root, "anchors")
    qc_dir = os.path.join(out_root, "qc_candidate")
    for path in (out_root, mask_dir, anchor_dir, qc_dir):
        os.makedirs(path, exist_ok=True)

    cam_dir = os.path.join(seq, args.camera)
    if args.background_image:
        bg = cv2.imread(args.background_image, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            raise FileNotFoundError(f"无法读取背景图: {args.background_image}")
        all_paths = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
        if not all_paths:
            raise FileNotFoundError(f"无图像: {cam_dir}")
        background_source = os.path.abspath(args.background_image)
    else:
        bg, all_paths = build_median_background(cam_dir, args.n_bg)
        background_source = "sequence_median"
    frame_paths = all_paths[::max(1, args.frame_step)]
    if args.frame_step != 1:
        print("[diagnostic] frame-step != 1：只生成抽样QC，不能直接用于完整SAM2传播")
    cv2.imwrite(os.path.join(out_root, "bg_median.png"), bg)
    params = {
        "sat": args.sat, "val": args.val, "diff": args.diff, "dil": args.dil,
        "open_k": args.open_k, "close_k": args.close_k,
        "min_area_frac": args.min_area_frac, "min_h_frac": args.min_h_frac,
    }

    rows = []
    for i, path in enumerate(frame_paths):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        frame = int(os.path.splitext(os.path.basename(path))[0])
        final = candidate_stages(
            bgr, bg, params, args.base_side, args.base_trim_width_ratio,
            args.base_trim_stable_span)["final"]
        cv2.imwrite(os.path.join(mask_dir, f"{frame:05d}.png"), final * 255)
        row = {"frame": frame, "file": os.path.basename(path), **mask_metrics(final)}
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  candidate {i + 1}/{len(frame_paths)}")

    image_shape = bg.shape
    robust_stats = add_quality_scores(rows, image_shape, args.base_side)
    selected = select_chunk_anchors(rows, args.chunk_size)
    for row in rows:
        row["selected"] = int(row["frame"] in selected)
        if row["selected"]:
            mask = cv2.imread(os.path.join(mask_dir, f"{int(row['frame']):05d}.png"),
                              cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(anchor_dir, f"{int(row['frame']):05d}.png"), mask)

    fieldnames = list(rows[0].keys()) if rows else ["frame", "file", "quality", "selected"]
    for name in ("candidate_metrics.csv", "anchor_manifest.csv"):
        with open(os.path.join(out_root, name), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader(); writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "sequence": seq,
        "camera": args.camera,
        "method": "white_on_blue_candidate_then_sam2",
        "action_independent": True,
        "frame_step": args.frame_step,
        "n_frames": len(rows),
        "n_empty": int(sum(row["area"] == 0 for row in rows)),
        "n_selected_anchors": len(selected),
        "chunk_size": args.chunk_size,
        "base_side": args.base_side,
        "base_attachment_trim": {
            "width_ratio": args.base_trim_width_ratio,
            "stable_span": args.base_trim_stable_span,
        },
        "background_source": background_source,
        "segmentation_params": params,
        "robust_feature_stats": {k: {"median": v[0], "scale": v[1]}
                                 for k, v in robust_stats.items()},
    }
    with open(os.path.join(out_root, "candidate_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    save_stage_qc(
        frame_paths, bg, params, rows, selected, qc_dir,
        base_side=args.base_side,
        trim_width_ratio=args.base_trim_width_ratio,
        trim_stable_span=args.base_trim_stable_span)
    print(f"完成：candidate={mask_dir} anchors={anchor_dir}")
    print(f"      manifest={os.path.join(out_root, 'anchor_manifest.csv')} qc={qc_dir}")


if __name__ == "__main__":
    main()
