"""viz_qc.py — 清晰、明确标注的质量检查可视化(避免混淆: 明示 raw vs 处理过, mask vs node)。

动机: 之前 clean_qc_montage 用 raw mask 叠骨架, 对 rep_clean(骨架来自 repaired mask)造成
"看不出修复"的混淆。本脚本把每步处理的输入/输出**按名称说清**, 放到对应文件夹的 qc/ 下。

两种模式:

1. mask-compare  两个 mask 目录并排(如 raw vs repaired), 每帧 [RAW mask | {tag} mask]。
   用途: 看 mask 修复(repair_masks)前后。
   python scripts/real/viz_qc.py mask-compare \\
       --mask-a real_capture/data/derived/<seq>/masks --tag-a raw \\
       --mask-b real_capture/data/derived/<seq>/masks_repaired --tag-b repaired \\
       --out real_capture/data/derived/<seq>/masks_repaired/qc

2. dataset  某 npz(clean 后骨架) 的全链路: 每帧 [RAW mask | {src} mask(管线输入) |
   {src} mask + 从{src}提取的骨架 | {src} mask + npz清洗后骨架]。
   明示: 这个数据集的骨架来自哪个 mask、清洗前后差别。
   python scripts/real/viz_qc.py dataset \\
       --npz data/real_seq/<seq>_rep_clean/train/<seq>_train.npz \\
       --mask-src real_capture/data/derived/<seq>/masks_repaired --src-tag repaired \\
       --raw-mask real_capture/data/derived/<seq>/masks \\
       --out data/real_seq/<seq>_rep_clean/qc

默认 frames 含典型腐败帧(f4902 半mask / f4080 静态截断 / f2330 手 / f1692,f4516 半mask / f100 干净)。
输出文件名自描述: mask_raw_vs_repaired.png / full_chain_<src>.png。图内大字标注每列含义。
"""
import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from src.utils.skeleton_2d import extract_skeleton_2d  # noqa: E402

DEFAULT_FRAMES = [100, 4080, 4902, 2330, 1692, 4516]   # 含各类腐败 + 干净


def _load_mask(d, f):
    p = os.path.join(d, f"{f:05d}.png")
    return (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if os.path.isfile(p) else None


def _photo(seq, f):
    p = os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", seq, "cam0", f"{f:05d}.png")
    return cv2.imread(p) if os.path.isfile(p) else None


def _draw_skel(img, sk, color, r=3, lw=2):
    if sk is None or np.abs(sk).max() == 0:
        return
    pts = np.round(sk).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, lw, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)


def _mask_on_photo(photo, mask, alpha=0.45, color=(0, 0, 255)):
    """mask 半透明色叠在原图上(无原图则灰底)。"""
    base = photo.copy() if photo is not None else np.full((mask.shape[0], mask.shape[1], 3), 30, np.uint8)
    if mask is not None and mask.any():
        ov = base.copy()
        ov[mask > 0] = color
        cv2.addWeighted(ov, alpha, base, 1 - alpha, 0, dst=base)
    return base


def _label(img, text, top=True):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0 if top else h - 30), (w, 30 if top else h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, img, 0.55, 0, dst=img)
    cv2.putText(img, text, (8, 21 if top else h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return img


def mask_compare(args):
    frames = [int(x) for x in args.frames.split(",")] if args.frames else DEFAULT_FRAMES
    cells = []
    for f in frames:
        a = _load_mask(args.mask_a, f)
        b = _load_mask(args.mask_b, f)
        if a is None and b is None:
            continue
        photo = _photo(args.seq, f) if args.seq else None
        pa = _mask_on_photo(photo, a)
        pb = _mask_on_photo(photo, b)
        _label(pa, f"f{f} | {args.tag_a.upper()} mask (area={int(a.sum()) if a is not None else 0})")
        _label(pb, f"f{f} | {args.tag_b.upper()} mask (area={int(b.sum()) if b is not None else 0})")
        cells.append(np.hstack([pa, pb]))
    out_png = os.path.join(args.out, f"mask_{args.tag_a}_vs_{args.tag_b}.png")
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(out_png, np.vstack(cells))
    print(f"→ {out_png}  ({len(cells)} 帧, 左={args.tag_a} 右={args.tag_b})")


def dataset_qc(args):
    frames = [int(x) for x in args.frames.split(",")] if args.frames else DEFAULT_FRAMES
    d = np.load(args.npz)
    pos = d["positions"].astype(np.float32)          # (T,3,N) cleaned skeleton
    N = pos.shape[2]
    cells = []
    for f in frames:
        clean_sk = pos[f, :2, :].T if f < len(pos) else None     # (N,2)
        src_mask = _load_mask(args.mask_src, f)
        raw_mask = _load_mask(args.raw_mask, f) if args.raw_mask else None
        if src_mask is None:
            continue
        photo = _photo(args.seq, f) if args.seq else None
        p1 = _mask_on_photo(photo, raw_mask) if raw_mask is not None else _mask_on_photo(photo, src_mask)
        p2 = _mask_on_photo(photo, src_mask)
        p3 = _mask_on_photo(photo, src_mask)
        p4 = _mask_on_photo(photo, src_mask)
        src_sk = extract_skeleton_2d(src_mask, N, tip_fix=True)
        _draw_skel(p3, src_sk, (255, 255, 0))                         # 青
        _draw_skel(p4, clean_sk, (0, 255, 255))                       # 黄
        _label(p1, f"f{f} | RAW mask")
        _label(p2, f"f{f} | {args.src_tag.upper()} mask (=pipeline input)")
        _label(p3, f"f{f} | skeleton from {args.src_tag} mask (cyan)")
        _label(p4, f"f{f} | CLEANED skeleton from npz (yellow)")
        cells.append(np.hstack([p1, p2, p3, p4]))
    out_png = os.path.join(args.out, f"full_chain_{args.src_tag}.png")
    os.makedirs(args.out, exist_ok=True)
    cv2.imwrite(out_png, np.vstack(cells))
    print(f"→ {out_png}  ({len(cells)} 帧 ×4列: raw mask | {args.src_tag} mask | skel-from-{args.src_tag} | cleaned-skel)")


def main(argv=None):
    pa = argparse.ArgumentParser(description="清晰标注的 QC 可视化(raw vs 处理过, mask vs node)")
    sub = pa.add_subparsers(dest="mode", required=True)

    pm = sub.add_parser("mask-compare", help="两个 mask 目录并排")
    pm.add_argument("--mask-a", required=True); pm.add_argument("--tag-a", required=True)
    pm.add_argument("--mask-b", required=True); pm.add_argument("--tag-b", required=True)
    pm.add_argument("--seq", default="seq_20260627_163921", help="原图序列名(叠原图用)")
    pm.add_argument("--frames", default=None, help="逗号分隔帧(默认含腐败帧)")
    pm.add_argument("--out", required=True, help="输出目录(建议 <结果>/qc)")

    pd = sub.add_parser("dataset", help="某 npz 全链路(raw mask→输入mask→骨架→清洗骨架)")
    pd.add_argument("--npz", required=True)
    pd.add_argument("--mask-src", required=True, help="该 npz 骨架来源 mask 目录")
    pd.add_argument("--src-tag", required=True, help="来源 mask 标签(如 repaired/raw)")
    pd.add_argument("--raw-mask", default=None, help="原始 mask 目录(对比)")
    pd.add_argument("--seq", default="seq_20260627_163921")
    pd.add_argument("--frames", default=None)
    pd.add_argument("--out", required=True)

    args = pa.parse_args(argv)
    (mask_compare if args.mode == "mask-compare" else dataset_qc)(args)


if __name__ == "__main__":
    main()
