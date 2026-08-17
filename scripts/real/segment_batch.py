"""segment_batch.py — 实物序列批量 RGB → 二值前景掩码 (Phase 1b)。

用 src.data.real.segmentation.segment_white_on_blue（diag 校准的白半透明硅胶臂
专用：白∩动过 → 形态学 → 最大连通区）把 cam0 全部帧批量分割，落地到
derived/<seq>/masks/NNNNN.png（与 cam0 同名，0/255 二值），并保存中值背景 +
参数 meta + 全序列概览 QC 大图。

掩码即 Phase 1 交付物（"从 RGB 提取机器人二值前景"）；下游 planar-lift/训练
在 Phase 1c 用 capture_to_npz 复用此方法（或直接读 masks/）。

用法:
  # 试跑 200 帧（均匀间隔），核对全序列稳定性
  python scripts/real/segment_batch.py \
      --seq real_capture/data/raw/seq_20260627_163921 --max-frames 200
  # 全量 10214 帧
  python scripts/real/segment_batch.py \
      --seq real_capture/data/raw/seq_20260627_163921
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.real.segmentation import (  # noqa: E402
    build_median_background, segment_white_on_blue)


def save_overview(fps, mask_dir, out_root, n=16):
    """采样 n 帧画 RGB+掩码红叠加 4×4 概览，看全序列一致性。"""
    n = min(n, len(fps))
    pick = np.linspace(0, len(fps) - 1, n).astype(int)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for k, fi in enumerate(pick):
        bgr = cv2.imread(fps[fi])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask_path = os.path.join(mask_dir, os.path.basename(fps[fi]))
        m = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
             if os.path.exists(mask_path) else np.zeros(bgr.shape[:2], bool))
        ax = axes[k // cols][k % cols]
        ax.imshow(rgb)
        ax.imshow(np.where(m[..., None], [255, 0, 0], [0, 0, 0]), alpha=0.35)
        ax.set_title(f"#{fi} A={int(m.sum())}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    path = os.path.join(out_root, "overview.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


def build_parser():
    pa = argparse.ArgumentParser(description="实物序列批量 RGB → 二值掩码")
    pa.add_argument("--seq", required=True, help="序列目录(含 cam0/)")
    pa.add_argument("--max-frames", type=int, default=None, help="只处理前 N 帧(试跑)")
    pa.add_argument("--n-bg", type=int, default=500, help="中值背景采样帧数")
    pa.add_argument("--out-root", default=None,
                    help="输出根目录(默认 <seq>/../derived/<seq名>)")
    # segment_white_on_blue 参数（默认=diag 校准值）
    pa.add_argument("--sat", type=int, default=100)
    pa.add_argument("--val", type=int, default=120)
    pa.add_argument("--diff", type=int, default=25)
    pa.add_argument("--dil", type=int, default=35)
    pa.add_argument("--open-k", type=int, default=5)
    pa.add_argument("--close-k", type=int, default=15)
    pa.add_argument("--min-area-frac", type=float, default=0.003)
    pa.add_argument("--min-h-frac", type=float, default=0.15)
    pa.add_argument("--jobs", type=int, default=1,
                    help="预留并行(目前单线程，10214帧≈5min)")
    return pa


def main():
    args = build_parser().parse_args()
    seq = args.seq.rstrip("/")
    seq_name = os.path.basename(seq)
    out_root = args.out_root or os.path.abspath(
        os.path.join(os.path.dirname(seq), "..", "derived", seq_name))
    mask_dir = os.path.join(out_root, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    cam_dir = os.path.join(seq, "cam0")

    print(f">>> 构建中值背景（采样 {args.n_bg} 帧）...")
    bg, fps = build_median_background(cam_dir, args.n_bg)
    cv2.imwrite(os.path.join(out_root, "bg_median.png"), bg)
    print(f"    bg: mean={bg.mean():.1f}  → {out_root}/bg_median.png")

    params = dict(sat=args.sat, val=args.val, diff=args.diff, dil=args.dil,
                  open_k=args.open_k, close_k=args.close_k,
                  min_area_frac=args.min_area_frac, min_h_frac=args.min_h_frac)

    # max-frames < 总数时 = 均匀采样（试跑要覆盖全序列压力/光照范围，而非前N帧）
    if args.max_frames and args.max_frames < len(fps):
        idx = np.linspace(0, len(fps) - 1, args.max_frames).astype(int)
        fps_run = [fps[i] for i in idx]
        print(f"    （试跑：在 {len(fps)} 帧中均匀采样 {len(fps_run)} 帧）")
    else:
        fps_run = fps
    n_total = len(fps_run)
    print(f">>> 批量分割 {n_total} 帧 → {mask_dir}")
    t0 = time.monotonic()
    areas = np.zeros(n_total, np.int64)
    n_empty = 0
    for i, fp in enumerate(fps_run):
        bgr = cv2.imread(fp)
        m = segment_white_on_blue(bgr, bg, **params)
        a = int(m.sum())
        areas[i] = a
        n_empty += (a == 0)
        cv2.imwrite(os.path.join(mask_dir, os.path.basename(fp)), m * 255)
        if (i + 1) % 1000 == 0 or i + 1 == n_total:
            dt = time.monotonic() - t0
            print(f"    [{i+1}/{n_total}]  {dt:.1f}s  {(i+1)/dt:.1f} fps  "
                  f"空掩码 {n_empty}  面积中位 {int(np.median(areas[:i+1]))}")
    dt = time.monotonic() - t0

    print(f"\n=== 统计（{n_total} 帧, {dt:.1f}s, {n_total/dt:.1f} fps）===")
    print(f"  空掩码: {n_empty} ({n_empty/n_total*100:.2f}%)")
    nz = areas[areas > 0]
    if len(nz):
        print(f"  非空面积  min/med/mean/max = "
              f"{nz.min()}/{int(np.median(nz))}/{int(nz.mean())}/{nz.max()}")
        # 离群: 面积 < 中位 0.3× 或 > 3× 中位 的帧（疑似分割异常）
        med = np.median(nz)
        lo, hi = med * 0.3, med * 3.0
        out_lo = np.where((areas > 0) & (areas < lo))[0]
        out_hi = np.where(areas > hi)[0]
        print(f"  面积离群(<{lo:.0f}): {len(out_lo)} 帧 {out_lo[:10].tolist()}"
              f"{' ...' if len(out_lo)>10 else ''}")
        print(f"  面积离群(>{hi:.0f}): {len(out_hi)} 帧 {out_hi[:10].tolist()}"
              f"{' ...' if len(out_hi)>10 else ''}")

    print(">>> 生成概览 QC 大图...")
    ov = save_overview(fps_run, mask_dir, out_root)
    print(f"    {ov}")

    meta = {"seq": seq, "method": "white_on_blue", "params": params,
            "n_bg": args.n_bg, "n_frames": n_total,
            "n_empty": int(n_empty),
            "area_median": int(np.median(nz)) if len(nz) else 0,
            "bg_median_path": os.path.relpath(
                os.path.join(out_root, "bg_median.png"), out_root)}
    with open(os.path.join(out_root, "segment_meta.json"), "w") as f:
        import json
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f">>> 完成。masks 在 {mask_dir}  meta 在 {out_root}/segment_meta.json")


if __name__ == "__main__":
    main()
