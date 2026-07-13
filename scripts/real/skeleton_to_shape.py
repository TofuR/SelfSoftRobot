"""skeleton_to_shape.py — node→shape 映射 v0: 骨架**半径偏移**(纯几何基线, 无 NN 无 action)。

复用 sdf_utils 思想(SDF = dist_to_skeleton - radius): shape = {像素: dist_to_skeleton ≤ r}。
实现: 骨架画成粗折线(thickness=2r)+节点圆, 即半径 r 的管。

为什么先做这个(用户: "先在node形成的驱动扩大一个半径作为简单版"):
  量化"形态 = 骨架 + 管半径"能解释多少。
  - 若 IoU 已高 → 形态主要由骨架+常数半径决定, 神经 decoder 可能多余。
  - 残差(压力依赖的宽度变化、末端 cap 形、taper) 才是 NN 该学的; action 窗口仅在
    "形态有骨架未捕捉的压力依赖形变"时才需要(骨架已编码弯曲, 瞬时形态≈f(骨架))。

模式:
  uniform  : 全臂常数半径 r(自动拟合 max-IoU vs GT mask)。
  variable : per-node 半径(从 GT mask 估局部半宽 = dist_to_boundary), 处理 taper/末端。
             注: 这是用 GT 估半径 = offset 法的**上界**(预测时无 GT, 需 NN 预测半径)。

用法:
  python scripts/real/skeleton_to_shape.py \
      --npz data/real_seq/seq_20260627_163921_clean/train/seq_20260627_163921_train.npz \
      --masks-dir real_capture/data/derived/seq_20260627_163921/masks
  # 用修复后的 mask 当 GT / per-node 半径上界:
  ... --masks-dir .../masks_repaired --mode variable
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def skeleton_to_shape_uniform(skel_xy, H, W, radius):
    """骨架 + 常数半径 → 二值管 mask。skel_xy:(N,2)[col,row]。"""
    img = np.zeros((H, W), np.uint8)
    if skel_xy is None or np.abs(skel_xy).max() == 0:
        return img
    pts = np.round(skel_xy).astype(np.int32).reshape(-1, 1, 2)
    thick = int(round(2 * radius)) | 1                      # 奇数厚度
    cv2.polylines(img, [pts], False, 1, thick, cv2.LINE_AA)
    r = int(round(radius))
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), r, 1, -1, cv2.LINE_AA)
    return img


def estimate_node_radii(skel_xy, gt_mask):
    """从 GT mask 估每节点局部半宽(distance_transform_edt 在节点处取值)。"""
    if not gt_mask.any():
        return None
    edt = distance_transform_edt(gt_mask)                   # 管内到边界距离 = 半宽
    r = []
    for x, y in skel_xy:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < gt_mask.shape[0] and 0 <= xi < gt_mask.shape[1]:
            r.append(float(edt[yi, xi]))
        else:
            r.append(0.0)
    return np.array(r)


def skeleton_to_shape_variable(skel_xy, radii, H, W):
    """骨架 + per-node 半径 → 二值管(逐节点圆 + 段粗线, 段厚≈两端半径均)。"""
    img = np.zeros((H, W), np.uint8)
    if skel_xy is None or np.abs(skel_xy).max() == 0:
        return img
    pts = np.round(skel_xy).astype(np.int32)
    for i in range(len(pts)):
        cv2.circle(img, (int(pts[i, 0]), int(pts[i, 1])),
                   max(1, int(round(radii[i]))), 1, -1, cv2.LINE_AA)
    for i in range(len(pts) - 1):
        rseg = max(int(round((radii[i] + radii[i + 1]) / 2)), 1)
        cv2.line(img, (int(pts[i, 0]), int(pts[i, 1])),
                 (int(pts[i + 1, 0]), int(pts[i + 1, 1])), 1, 2 * rseg | 1, cv2.LINE_AA)
    return img


def iou_dice(pred, gt):
    p = pred > 0
    g = gt > 0
    inter = (p & g).sum()
    union = (p | g).sum()
    iou = inter / union if union else 0.0
    dice = 2 * inter / (p.sum() + g.sum()) if (p.sum() + g.sum()) else 0.0
    return iou, dice


def main(argv=None):
    pa = argparse.ArgumentParser(description="node→shape 半径偏移基线(骨架→管)")
    pa.add_argument("--npz", required=True, help="骨架 npz(positions (T,3,N))")
    pa.add_argument("--masks-dir", required=True, help="GT mask 目录")
    pa.add_argument("--mode", choices=["uniform", "variable"], default="uniform",
                    help="uniform=常数半径(自动拟合); variable=per-node半径(从GT估, 上界)")
    pa.add_argument("--radius", type=float, default=None, help="[uniform] 指定半径; 默认自动拟合")
    pa.add_argument("--frame-offset", type=int, default=0, help="npz索引→mask帧号偏移(train=0/val=Ntrain)")
    pa.add_argument("--n-sample", type=int, default=16)
    pa.add_argument("--out", default=None)
    args = pa.parse_args(argv)

    d = np.load(args.npz)
    pos = d['positions'].astype(np.float32)                  # (T,3,N)
    T, _, N = pos.shape
    print(f">>> T={T} N={N} mode={args.mode}")

    sample_idx = list(np.linspace(0, T - 1, min(args.n_sample, T)).astype(int))

    def load_mask(t):
        fi = t + args.frame_offset
        p = os.path.join(args.masks_dir, f"{fi:05d}.png")
        return (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) if os.path.isfile(p) else None

    H = W = None
    if args.mode == "uniform" and args.radius is None:
        best_r, best_iou = 14.0, -1.0
        for rr in np.arange(8, 24, 0.5):
            ious = []
            for t in sample_idx:
                gt = load_mask(t)
                if gt is None:
                    continue
                H, W = gt.shape
                sk = pos[t, :2, :].T
                if np.abs(sk).max() == 0:
                    continue
                ious.append(iou_dice(skeleton_to_shape_uniform(sk, H, W, rr), gt)[0])
            if ious and np.mean(ious) > best_iou:
                best_iou, best_r = np.mean(ious), float(rr)
        radius = best_r
        print(f"  自动拟合均匀半径 r={radius:.1f}px (sample mean IoU={best_iou:.3f})")
    else:
        radius = args.radius if args.radius is not None else 14.0

    ious, dices, cells = [], [], []
    for t in sample_idx:
        gt = load_mask(t)
        if gt is None:
            continue
        H, W = gt.shape
        sk = pos[t, :2, :].T
        if np.abs(sk).max() == 0:
            continue
        if args.mode == "uniform":
            pred = skeleton_to_shape_uniform(sk, H, W, radius)
        else:
            radii = estimate_node_radii(sk, gt)
            pred = skeleton_to_shape_variable(sk, radii, H, W) if radii is not None else \
                skeleton_to_shape_uniform(sk, H, W, radius)
        i, dvc = iou_dice(pred, gt)
        ious.append(i); dices.append(dvc)
        if len(cells) < 16:
            ov = np.repeat((gt * 80)[:, :, None], 3, 2)      # GT 灰底
            ov[:, :, 2] = np.maximum(ov[:, :, 2], (pred > 0) * 255)  # pred 红叠加
            cv2.putText(ov, f"f{t+args.frame_offset} IoU={i:.2f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cells.append(ov)

    ious = np.array(ious); dices = np.array(dices)
    print(f"\n=== {args.mode} (r={radius:.1f} 若 uniform) ===")
    print(f"  IoU:  mean={ious.mean():.3f} median={np.median(ious):.3f} "
          f"p10={np.percentile(ious,10):.3f} min={ious.min():.3f}")
    print(f"  Dice: mean={dices.mean():.3f}")
    print(f"  解读: IoU 高→骨架+半径已解释形态(NN可能多余); 低→残差大(宽度变化/末端cap)→NN(+action)有价值")

    ckpt = os.path.splitext(os.path.basename(args.npz))[0]
    out_dir = args.out or os.path.join(PROJECT_ROOT, "output", "skeleton_to_shape", ckpt, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    if cells:
        h, w = cells[0].shape[:2]
        cols = 4
        canvas = np.zeros((int(np.ceil(len(cells)/cols))*h, cols*w, 3), np.uint8)
        for k, im in enumerate(cells):
            rr, cc = divmod(k, cols)
            canvas[rr*h:(rr+1)*h, cc*w:(cc+1)*w] = im
        cv2.imwrite(os.path.join(out_dir, "overlay_montage.png"), canvas)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"npz: {args.npz}\nmasks: {args.masks_dir}\nmode: {args.mode}\n")
        f.write(f"radius: {radius}\nIoU mean={ious.mean():.4f} median={np.median(ious):.4f} "
                f"p10={np.percentile(ious,10):.4f} min={ious.min():.4f}\nDice mean={dices.mean():.4f}\n")
    print(f"  → {out_dir} (overlay_montage.png + summary.txt)")


if __name__ == "__main__":
    main()
