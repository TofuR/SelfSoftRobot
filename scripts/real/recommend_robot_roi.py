"""根据已有机器人 masks 推荐固定方形 ROI。

该工具用于一次采集序列的分割检查后自动收紧 ROI。输入 mask 可以是源图尺寸，也可以
是带 ``crop_meta.json`` 的裁剪尺寸。输出保留相同的毫米尺度策略：四周留出若干个
机器人直径，并把方形 ROI 平移到源图范围内。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def recommend_square_roi(bounds, source_size_wh, diameter_px, padding_diameters=3.0,
                         quantile=1.0):
    boxes = np.asarray(bounds, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes):
        raise ValueError("bounds 必须是非空 (N,4) x0,y0,x1,y1")
    width, height = (int(value) for value in source_size_wh)
    if width <= 0 or height <= 0:
        raise ValueError("source_size_wh 必须为正")
    q = float(quantile)
    if q < 0 or q >= 50:
        raise ValueError("quantile 必须位于 [0,50)")
    lo = np.percentile(boxes[:, :2], q, axis=0)
    hi = np.percentile(boxes[:, 2:], 100.0 - q, axis=0)
    padding = float(diameter_px) * float(padding_diameters)
    side = int(math.ceil(max(hi[0] - lo[0], hi[1] - lo[1]) + 2.0 * padding))
    side = max(16, int(math.ceil(side / 16.0) * 16))
    side = min(side, width, height)
    center = 0.5 * (lo + hi)
    x = int(round(center[0] - side / 2.0))
    y = int(round(center[1] - side / 2.0))
    x = min(max(0, x), width - side)
    y = min(max(0, y), height - side)
    return x, y, side, side


def main(argv=None):
    parser = argparse.ArgumentParser(description="从机器人 masks 推荐源图方形 ROI")
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--crop-meta", help="mask 为裁剪图时提供 crop_meta.json")
    parser.add_argument("--source-size", help="源图 W,H；无 crop-meta 时必填")
    parser.add_argument("--diameter-px", type=float,
                        help="当前相机下机器人直径像素；默认从 masks 距离变换估计")
    parser.add_argument("--padding-diameters", type=float, default=3.0)
    parser.add_argument("--quantile", type=float, default=1.0,
                        help="边界稳健分位数；默认忽略两侧各1%%极端框")
    parser.add_argument("--out", default="roi_recommendation.json")
    args = parser.parse_args(argv)

    offset = np.zeros(2, dtype=np.int64)
    source_size = None
    if args.crop_meta:
        with open(args.crop_meta, encoding="utf-8") as stream:
            meta = json.load(stream)
        offset = np.asarray(meta["crop_xywh"][:2], dtype=np.int64)
        source_size = tuple(int(v) for v in meta["source_image_size_wh"])
    if args.source_size:
        source_size = tuple(int(v) for v in args.source_size.split(","))
    if source_size is None or len(source_size) != 2:
        raise ValueError("需要 --crop-meta 或 --source-size W,H")

    paths = sorted(glob.glob(os.path.join(args.masks_dir, "*.png")))
    if not paths:
        raise FileNotFoundError(f"没有 mask: {args.masks_dir}")
    bounds = []
    diameter_samples = []
    for path in paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 127)
        if not len(xs):
            continue
        bounds.append((xs.min() + offset[0], ys.min() + offset[1],
                       xs.max() + 1 + offset[0], ys.max() + 1 + offset[1]))
        if args.diameter_px is None:
            distance = cv2.distanceTransform((mask > 127).astype(np.uint8),
                                             cv2.DIST_L2, 5)
            positive = distance[distance > 0]
            if len(positive):
                diameter_samples.append(2.0 * float(np.percentile(positive, 95)))
    if not bounds:
        raise ValueError("所有 masks 均为空")
    diameter_px = (float(args.diameter_px) if args.diameter_px is not None else
                   float(np.median(diameter_samples)))
    roi = recommend_square_roi(
        bounds, source_size, diameter_px,
        padding_diameters=args.padding_diameters, quantile=args.quantile)
    payload = {
        "schema_version": 1,
        "roi_xywh": list(roi),
        "source_image_size_wh": list(source_size),
        "robot_diameter_px": diameter_px,
        "padding_diameters": float(args.padding_diameters),
        "bounds_quantile_percent": float(args.quantile),
        "mask_count": len(bounds),
        "masks_dir": str(Path(args.masks_dir).resolve()),
    }
    with open(args.out, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    print("--roi " + ",".join(str(value) for value in roi))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
