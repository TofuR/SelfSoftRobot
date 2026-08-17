"""sam2/compare_masks.py — 之前 mask(启发式 masks_repaired) vs SAM2 mask 对比图。

回答用户: "生成之前mask结果和sam mask结果的对比图"。三列并排(每帧):
  [RAW mask | PREV=repaired mask | SAM2 mask], 都叠在原图上, 标 area + 顶部行 + IoU(prev∩sam)。
另产 area 散点(全序列 prev_area vs sam_area) 量化 SAM2 在哪改了 mask。

输入目录:
  raw:   real_capture/data/derived/<seq>/masks            (原始分割, 含腐败)
  prev:  real_capture/data/derived/<seq>/masks_repaired   (启发式修复 = "之前"的结果)
  sam2:  sam2/masks/<seq>_full                            (SAM2 视频分割 = 新结果)
输出:
  sam2/masks/<seq>_full/qc/compare_raw_prev_sam.png       (三列对比, 含腐败帧+干净帧+抽样)
  sam2/masks/<seq>_full/qc/area_scatter_prev_vs_sam.png   (全序列 area 散点 + 1:1 线)

用法:
  python sam2/compare_masks.py --seq seq_20260627_163921
  python sam2/compare_masks.py --seq seq_20260627_163921 --frames 4080,4902,2330,100
"""
import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

# 典型腐败帧(静态截断/动作段半mask/手污染) + 干净 + 跨序列抽样
DEFAULT_FRAMES = [100, 4080, 4902, 2330, 2316, 1692, 4516, 1000, 3000, 7000, 9000]


def _load(d, f):
    p = os.path.join(d, f"{f:05d}.png")
    return (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if os.path.isfile(p) else None


def _photo(seq, f):
    p = os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "cam0", f"{f:05d}.png")
    return cv2.imread(p) if os.path.isfile(p) else None


def _stats(m):
    if m is None or not m.any():
        return "area=0"
    ys, _ = np.where(m > 0)
    return f"area={int(ys.size)} top={int(ys.min())}"


def _iou(a, b):
    if a is None or b is None or not a.any() or not b.any():
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(np.logical_or(a, b).sum())


def _mask_on_photo(photo, mask, alpha=0.45, color=(0, 0, 255)):
    base = photo.copy() if photo is not None else np.full((mask.shape[0], mask.shape[1], 3), 30, np.uint8)
    if mask is not None and mask.any():
        ov = base.copy()
        ov[mask > 0] = color
        cv2.addWeighted(ov, alpha, base, 1 - alpha, 0, dst=base)
    return base


def _label(img, text):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, img, 0.55, 0, dst=img)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def compare_panels(args):
    seq = args.seq
    raw_d = os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq, "masks")
    prev_d = os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq, "masks_repaired")
    sam_d = args.sam_dir or os.path.join(HERE, "masks", f"{seq}_full")
    out_qc = os.path.join(sam_d, "qc")
    os.makedirs(out_qc, exist_ok=True)
    frames = [int(x) for x in args.frames.split(",")] if args.frames else DEFAULT_FRAMES

    rows = []
    n_sam_missing = 0
    for f in frames:
        raw = _load(raw_d, f)
        prev = _load(prev_d, f)
        sam = _load(sam_d, f)
        photo = _photo(seq, f)
        if sam is None:
            n_sam_missing += 1
        iou = _iou(prev, sam)
        p1 = _mask_on_photo(photo, raw, color=(128, 128, 128))
        p2 = _mask_on_photo(photo, prev, color=(0, 0, 255))      # 红=prev
        p3 = _mask_on_photo(photo, sam, color=(0, 255, 0))       # 绿=sam2
        _label(p1, f"f{f} | RAW ({_stats(raw)})")
        _label(p2, f"f{f} | PREV=repaired ({_stats(prev)})")
        _label(p3, f"f{f} | SAM2 ({_stats(sam)}) IoU(prev,sam)={iou:.2f}")
        rows.append(np.hstack([p1, p2, p3]))
    out_png = os.path.join(out_qc, "compare_raw_prev_sam.png")
    cv2.imwrite(out_png, np.vstack(rows))
    print(f"→ {out_png}  ({len(rows)} 帧 ×3列: RAW | PREV(repaired) | SAM2; "
          f"SAM2 缺 {n_sam_missing} 帧)")


def area_scatter(args):
    """全序列 prev_area vs sam_area 散点 + 1:1 线。SAM2 area 来自 area_curve.txt;
    prev area 现算。"""
    seq = args.seq
    prev_d = os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", seq, "masks_repaired")
    sam_d = args.sam_dir or os.path.join(HERE, "masks", f"{seq}_full")
    out_qc = os.path.join(sam_d, "qc")
    os.makedirs(out_qc, exist_ok=True)
    area_txt = os.path.join(sam_d, "area_curve.txt")
    if not os.path.isfile(area_txt):
        print(f"[skip] 无 {area_txt}(SAM2 还没跑完?)")
        return
    sam_area = {}
    with open(area_txt) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                sam_area[int(parts[0])] = int(parts[1])
    xs, ys, diffs = [], [], []
    big_diff_frames = []
    for f, sa in sorted(sam_area.items()):
        p = os.path.join(prev_d, f"{f:05d}.png")
        if not os.path.isfile(p):
            continue
        m = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        ys_cur = np.where(m > 0)[0]
        pa = int(ys_cur.size) if len(ys_cur) else 0
        if pa == 0:
            continue
        xs.append(pa); ys.append(sa)
        diffs.append(sa - pa)
        if abs(sa - pa) > 0.25 * pa:        # SAM2 与 prev 差 >25% → 标记
            big_diff_frames.append((f, pa, sa))
    if not xs:
        print("[skip] 无可对比帧")
        return
    xs = np.array(xs); ys = np.array(ys)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        lo, hi = 0, max(xs.max(), ys.max()) * 1.05
        ax[0].scatter(xs, ys, s=4, alpha=0.3)
        ax[0].plot([lo, hi], [lo, hi], "r--", lw=1, label="1:1 (prev==sam)")
        ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
        ax[0].set_xlabel("prev (repaired) area [px]")
        ax[0].set_ylabel("SAM2 area [px]")
        ax[0].set_title(f"prev vs SAM2 area  (n={len(xs)}, "
                        f"mean|Δ|={np.mean(np.abs(diffs)):.0f}px, "
                        f"P95|Δ|={np.percentile(np.abs(diffs),95):.0f}px)")
        ax[0].legend(); ax[0].set_aspect("equal")
        ax[1].hist(diffs, bins=80, color="steelblue", edgecolor="k")
        ax[1].axvline(0, color="r", ls="--")
        ax[1].set_xlabel("SAM2_area − prev_area [px]")
        ax[1].set_ylabel("帧数")
        ax[1].set_title(f"delta dist (SAM2 - prev); |delta|>25%prev frames: {len(big_diff_frames)}")
        fig.tight_layout()
        out_png = os.path.join(out_qc, "area_scatter_prev_vs_sam.png")
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        print(f"→ {out_png}  (n={len(xs)}, mean|Δ|={np.mean(np.abs(diffs)):.0f}px, "
              f"|Δ|>25%prev 帧={len(big_diff_frames)})")
        if big_diff_frames:
            top = sorted(big_diff_frames, key=lambda t: -abs(t[2]-t[1]))[:15]
            print("  差异最大 15 帧 (frame, prev_area, sam_area):")
            for f, pa, sa in top:
                print(f"    f{f}: prev={pa} sam={sa} Δ={sa-pa:+d}")
    except Exception as e:
        print(f"[skip scatter] {e}")


def main():
    pa = argparse.ArgumentParser(description="之前 mask(repaired) vs SAM2 mask 对比图")
    pa.add_argument("--seq", default="seq_20260627_163921")
    pa.add_argument("--sam-dir", default=None, help="SAM2 mask 目录(默认 sam2/masks/<seq>_full)")
    pa.add_argument("--frames", default=None, help="逗号分隔帧(默认含腐败+抽样)")
    pa.add_argument("--scatter-only", action="store_true", help="只画 area 散点(全序列量化)")
    args = pa.parse_args()
    if not args.scatter_only:
        compare_panels(args)
    area_scatter(args)


if __name__ == "__main__":
    main()
