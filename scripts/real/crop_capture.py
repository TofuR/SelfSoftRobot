"""固定 ROI 裁剪真实采集图像，并保存可复现的空间合同与 QC。

原始采集目录保持只读；裁剪图写到
``real_capture/data/derived/<seq>/crop/<camera>/``。下游 SAM2 在裁剪图上运行，
骨架转换时再用同一 ROI 的 ``x,y`` 偏移恢复到原相机像素坐标。

示例：
  python scripts/real/crop_capture.py \
    --seq real_capture/data/raw/seq_20260819_172644 \
    --roi 220,68,300,300
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.registry import (  # noqa: E402
    ProjectPaths, canonical_intermediate, canonical_output,
    resolve_raw_sequence,
)


def parse_roi(text: str) -> tuple[int, int, int, int]:
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ROI 必须为整数 x,y,w,h") from error
    if len(values) != 4 or values[0] < 0 or values[1] < 0 \
            or values[2] <= 0 or values[3] <= 0:
        raise argparse.ArgumentTypeError("ROI 必须为非负 x,y 和正数 w,h")
    return values


def _label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, .45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_qc(frame_paths, roi, qc_dir, n=12):
    """保存一张全图 ROI 参考和覆盖全序列的裁剪概览。"""
    x, y, w, h = roi
    selected = np.linspace(0, len(frame_paths) - 1,
                           min(n, len(frame_paths))).astype(int)
    crops = []
    reference = None
    for index in selected:
        path = frame_paths[int(index)]
        image = cv2.imread(path)
        if image is None:
            continue
        frame = os.path.splitext(os.path.basename(path))[0]
        crop = image[y:y + h, x:x + w]
        crops.append(_label(crop, f"f{frame}"))
        if reference is None:
            reference = image.copy()
            cv2.rectangle(reference, (x, y), (x + w - 1, y + h - 1),
                          (0, 255, 255), 2)
            cv2.putText(reference, f"ROI x={x} y={y} w={w} h={h}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (0, 255, 255), 2, cv2.LINE_AA)
    os.makedirs(qc_dir, exist_ok=True)
    if reference is not None:
        cv2.imwrite(os.path.join(qc_dir, "roi_reference.png"), reference)
    if crops:
        columns = 4
        rows = int(np.ceil(len(crops) / columns))
        canvas = np.zeros((rows * h, columns * w, 3), dtype=np.uint8)
        for index, crop in enumerate(crops):
            row, column = divmod(index, columns)
            canvas[row * h:(row + 1) * h,
                   column * w:(column + 1) * w] = crop
        cv2.imwrite(os.path.join(qc_dir, "crop_overview.png"), canvas)


def build_parser():
    parser = argparse.ArgumentParser(
        description="固定 ROI 裁剪真实采集图像（原始数据不改动）")
    parser.add_argument("--seq", required=True, help="原始序列目录，含 camN/")
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--roi", required=True, type=parse_roi,
                        help="源图像坐标 x,y,w,h")
    parser.add_argument("--out-root", default=None,
                        help="显式输出根；必须位于 workspace")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--preview-only", action="store_true",
                        help="只保存 ROI 参考/抽样概览，不批量写裁剪帧")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有同名裁剪帧；默认对正确尺寸文件断点续跑")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = ProjectPaths.load(workspace_root=args.workspace_root)
    seq = str(resolve_raw_sequence(paths, args.seq, camera=args.camera))
    seq_name = os.path.basename(seq)
    camera_dir = os.path.join(seq, args.camera)
    frame_paths = sorted(glob.glob(os.path.join(camera_dir, "*.png")))
    if not frame_paths:
        raise FileNotFoundError(f"无 PNG 图像: {camera_dir}")
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise ValueError(f"无法读取首帧: {frame_paths[0]}")
    source_h, source_w = first.shape[:2]
    x, y, w, h = args.roi
    if x + w > source_w or y + h > source_h:
        raise ValueError(
            f"ROI {args.roi} 超出源图像 {source_w}x{source_h}")

    out_root = str(canonical_output(paths, args.out_root or (
        canonical_intermediate(paths, seq_name, "crop-v1") / "crop")))
    output_camera = os.path.join(out_root, args.camera)
    qc_dir = os.path.join(out_root, "qc")
    meta_path = os.path.join(out_root, "crop_meta.json")
    if os.path.isfile(meta_path) and not args.overwrite:
        with open(meta_path, encoding="utf-8") as stream:
            previous = json.load(stream)
        previous_contract = (
            tuple(previous.get("crop_xywh", ())),
            tuple(previous.get("source_image_size_wh", ())),
            previous.get("camera"),
            os.path.abspath(previous.get("source_sequence", "")),
        )
        requested_contract = (
            tuple(args.roi), (source_w, source_h), args.camera, seq)
        if previous_contract != requested_contract:
            raise ValueError(
                "已有裁剪产物的空间合同与本次请求不同；请检查序列/ROI，"
                "确认重建时显式传 --overwrite。\n"
                f"已有={previous_contract}\n本次={requested_contract}")
    os.makedirs(output_camera, exist_ok=True)
    save_qc(frame_paths, args.roi, qc_dir)

    written = skipped = 0
    started = time.monotonic()
    if not args.preview_only:
        for index, source_path in enumerate(frame_paths):
            output_path = os.path.join(output_camera, os.path.basename(source_path))
            if not args.overwrite and os.path.isfile(output_path):
                existing = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
                if existing is not None and existing.shape[:2] == (h, w):
                    skipped += 1
                    continue
            image = cv2.imread(source_path)
            if image is None or image.shape[:2] != (source_h, source_w):
                raise ValueError(
                    f"帧缺失或尺寸不一致: {source_path}")
            if not cv2.imwrite(output_path, image[y:y + h, x:x + w]):
                raise OSError(f"写入失败: {output_path}")
            written += 1
            if (index + 1) % 1000 == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(f"  crop {index + 1}/{len(frame_paths)} "
                      f"({(index + 1) / elapsed:.1f} fps)")

    metadata = {
        "schema_version": 1,
        "source_sequence": seq,
        "camera": args.camera,
        "source_image_size_wh": [source_w, source_h],
        "crop_xywh": [x, y, w, h],
        "processed_image_size_wh": [w, h],
        "n_source_frames": len(frame_paths),
        "n_output_frames": (0 if args.preview_only else
                            len(glob.glob(os.path.join(output_camera, "*.png")))),
        "complete": not args.preview_only,
        "coordinate_contract": (
            "SAM2 masks use crop-local pixels; add crop_xywh[:2] to skeleton x,y "
            "before saving model state in source-camera pixel coordinates."),
    }
    with open(meta_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(f"完成：ROI={args.roi} source={source_w}x{source_h} frames={len(frame_paths)}")
    print(f"      crop={output_camera} written={written} skipped={skipped}")
    print(f"      qc={qc_dir} meta={meta_path}")


if __name__ == "__main__":
    main()
