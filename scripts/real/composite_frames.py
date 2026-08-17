"""composite_frames.py — 实物帧批量合成：原图 + mask 半透明覆盖 + 2D 骨架。

目的：把"采集原图 / 分割 mask / 提取骨架"三者合成到一张图，逐帧可视化，直观看清：
  - 分割是否干净（mask 是否贴住硅胶臂、有无误并管茬）
  - 骨架提取是否合理（中心线是否落在臂上、有无离群跑偏）
  - 三者是否对齐（mask ↔ 骨架 ↔ 原图）

对所有帧批量处理，输出到 real_capture/data/derived/<seq>/overlay/。
骨架用与训练同一管线（src/utils/skeleton_2d.py 的 extract_skeleton_2d，即
masks_to_transition_npz.py 用的同一个函数）；可选 --clean 叠加训练用的清洗后骨架
（clean_outlier_skeletons 时间插值修复的离群帧）便于对比。

输入布局（采集程序产物）:
  real_capture/data/raw/<seq>/cam0/<frame>.png        原图 480x640 BGR
  real_capture/data/derived/<seq>/masks/<frame>.png   二值 mask 0/255（同名对齐）

用法:
  # 全量（10214 帧）→ derived/<seq>/overlay/
  python scripts/real/composite_frames.py --seq real_capture/data/raw/seq_20260627_163921

  # 只跑前 200 帧（快速预览）
  python scripts/real/composite_frames.py --seq real_capture/data/raw/seq_20260627_163921 --limit 200

  # 同时叠加清洗后骨架（黄）对比原始提取（青）
  python scripts/real/composite_frames.py --seq ... --clean
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.skeleton_2d import extract_skeleton_2d  # noqa: E402


def parse_bgr(s, default):
    """'0,0,255' -> (0,0,255) BGR tuple。"""
    try:
        t = tuple(int(v) for v in s.split(','))
        if len(t) == 3:
            return t
    except Exception:
        pass
    return default


def draw_skeleton(img, skel, color, point_radius=3, line_w=2):
    """在 img 上画骨架：连线 + 节点。skel: (N,2) [col,row]=[x,y]。全 0 则跳过。"""
    if skel is None or np.abs(skel).max() == 0:
        return
    pts = skel.astype(np.int32).reshape(-1, 1, 2)  # (N,1,2) cv2 polylines 要求
    cv2.polylines(img, [pts], False, color, line_w, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), point_radius, color, -1, cv2.LINE_AA)


def composite(img_bgr, mask01, skel_raw, skel_clean, mask_color, mask_alpha, raw_color, clean_color):
    """合成单帧：原图 + mask 半透明色块 + 骨架。返回 BGR 图。"""
    out = img_bgr.copy()
    if mask01.any():
        overlay = out.copy()
        overlay[mask01 > 0] = mask_color
        cv2.addWeighted(overlay, mask_alpha, out, 1.0 - mask_alpha, 0, dst=out)
    draw_skeleton(out, skel_raw, raw_color)
    if skel_clean is not None:
        draw_skeleton(out, skel_clean, clean_color, point_radius=2, line_w=2)
    return out


def main(argv=None):
    pa = argparse.ArgumentParser(description='实物帧批量合成：原图 + mask + 骨架')
    pa.add_argument('--seq', required=True,
                    help='raw 序列目录 real_capture/data/raw/seq_<id>（含 cam0/）')
    pa.add_argument('--masks-dir', default=None,
                    help='mask 目录（缺省 derived/<seq>/masks）')
    pa.add_argument('--out-dir', default=None,
                    help='输出目录（缺省 derived/<seq>/overlay）')
    pa.add_argument('--n-points', type=int, default=31, help='骨架节点数')
    pa.add_argument('--tip-fix', action=argparse.BooleanOptionalAction, default=True,
                    help='末端 node0 垂直切片修正(修弯管 cap 角落偏移, 与训练 npz 同源; --no-tip-fix 关闭)')
    pa.add_argument('--mask-color', default='0,0,255', help='mask 覆盖色 BGR（默认红）')
    pa.add_argument('--mask-alpha', type=float, default=0.35, help='mask 透明度')
    pa.add_argument('--raw-color', default='255,255,0', help='原始骨架色 BGR（默认青）')
    pa.add_argument('--clean', action='store_true',
                    help='同时画清洗后骨架（clean_outlier_skeletons，黄），对比原始提取')
    pa.add_argument('--skel-dev-thresh', type=float, default=80.0,
                    help='[--clean] 离群判定阈值 px（与 masks_to_transition_npz 一致）')
    pa.add_argument('--limit', type=int, default=None, help='只处理前 N 帧（预览）')
    pa.add_argument('--every', type=int, default=1, help='帧步长（抽样，1=全部）')
    pa.add_argument('--montage', action='store_true', default=True,
                    help='额外生成一张采样网格总览 overlay_montage.png')
    args = pa.parse_args(argv)

    seq = args.seq.rstrip('/')
    seq_name = os.path.basename(seq)
    base = os.path.abspath(os.path.join(seq, '..', '..'))  # .../real_capture/data
    cam0 = os.path.join(seq, 'cam0')
    masks_dir = args.masks_dir or os.path.join(base, 'derived', seq_name, 'masks')
    out_dir = args.out_dir or os.path.join(base, 'derived', seq_name, 'overlay')
    if not os.path.isdir(cam0):
        sys.exit(f"找不到 cam0: {cam0}")
    if not os.path.isdir(masks_dir):
        sys.exit(f"找不到 masks: {masks_dir}")

    img_paths = sorted(glob.glob(os.path.join(cam0, '*.png')))
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(os.path.join(masks_dir, '*.png'))}
    pairs = [(p, mask_map[os.path.splitext(os.path.basename(p))[0]])
             for p in img_paths if os.path.splitext(os.path.basename(p))[0] in mask_map]
    if not pairs:
        sys.exit(f"原图与 mask 无同名匹配（cam0={cam0}, masks={masks_dir}）")
    if args.limit:
        pairs = pairs[:args.limit]
    os.makedirs(out_dir, exist_ok=True)

    mask_color = parse_bgr(args.mask_color, (0, 0, 255))
    raw_color = parse_bgr(args.raw_color, (255, 255, 0))
    clean_color = (0, 255, 255)  # 黄

    # 可选：批量算清洗后骨架（与训练同管线），用于 --clean 对比
    cleaned = None
    if args.clean:
        from scripts.real.masks_to_transition_npz import clean_outlier_skeletons
        print("提取全部骨架后做离群清洗（与训练同管线）...")
        skels = np.zeros((len(pairs), args.n_points, 2), np.float32)
        for i, (_ip, mp) in enumerate(pairs):
            m = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            skels[i] = extract_skeleton_2d(m, args.n_points, tip_fix=args.tip_fix)
        pos = np.zeros((len(pairs), 3, args.n_points), np.float32)
        pos[:, 0, :] = skels[:, :, 0]
        pos[:, 1, :] = skels[:, :, 1]
        pos, n_out, _ = clean_outlier_skeletons(pos, deviation_px=args.skel_dev_thresh)
        cleaned = pos[:, :2, :].transpose(0, 2, 1)  # (T,N,2) [col,row]
        print(f"  离群修复 {n_out}/{len(pairs)} 帧")

    n_empty, n_done = 0, 0
    sampled = []  # (frame_idx, img) for montage
    montage_sample = max(1, len(pairs) // 16)
    for i, (ip, mp) in enumerate(pairs):
        if i % args.every:
            continue
        img = cv2.imread(ip)
        if img is None:
            continue
        m01 = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        sk_raw = extract_skeleton_2d(m01, args.n_points, tip_fix=args.tip_fix)
        if np.abs(sk_raw).max() == 0:
            n_empty += 1
        sk_clean = cleaned[i] if (args.clean and cleaned is not None) else None
        out = composite(img, m01, sk_raw, sk_clean, mask_color, args.mask_alpha,
                        raw_color, clean_color)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(ip)), out)
        if args.montage and (i % montage_sample == 0) and len(sampled) < 16:
            sampled.append(out)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done}/{len(pairs)} 帧...")

    # 采样网格总览
    if args.montage and sampled:
        h, w = sampled[0].shape[:2]
        cols = 4
        rows = int(np.ceil(len(sampled) / cols))
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, im in enumerate(sampled):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
        cv2.imwrite(os.path.join(out_dir, 'overlay_montage.png'), canvas)
        print(f"  总览: {os.path.relpath(os.path.join(out_dir, 'overlay_montage.png'))}")

    print(f"\n完成: {n_done} 帧 → {os.path.relpath(out_dir)}"
          f"  (空 mask/无骨架帧: {n_empty})")
    if args.clean:
        print("  青=原始提取骨架, 黄=清洗后骨架(训练用); 红=mask 半透明覆盖")


if __name__ == '__main__':
    main()
