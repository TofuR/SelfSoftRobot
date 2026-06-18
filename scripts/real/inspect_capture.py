"""inspect_capture.py — 实物采集可视化核对（Step 4 QA）。

在跑完整 capture_to_npz 之前，先用几张图核对：分割干净？2D 骨架贴合？
三角化 3D 骨架合理？避免分割阈值/标定没调好就批量处理。

每行=一个采样帧；前 V 列=各视角（原图 + 分割掩码红叠加 + 2D 骨架青色），
最后一列=三角化 3D 中心线（31 节点，含相机位置参考）。

用法:
  python scripts/real/inspect_capture.py \\
      --view-dirs raw/seq1/cam0 raw/seq1/cam1 raw/seq1/cam2 \\
      --camera-params config/real_camera_params.npz \\
      --method backlight --gray-thresh 60 \\
      --n-frames 3 --out inspect_seq1.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.data.real.io_video import load_image_views, make_undistorter  # noqa: E402
from src.data.real.segmentation import (  # noqa: E402
    segment_views, masks_to_skeletons_2d)
from src.data.real.triangulation import triangulate_skeletons  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="实物采集可视化核对（QA）")
    p.add_argument("--view-dirs", nargs="+", required=True, help="每视角一个图像目录")
    p.add_argument("--camera-params", required=True)
    p.add_argument("--method", default="backlight",
                   choices=["backlight", "bg_subtract", "color"])
    p.add_argument("--gray-thresh", type=int, default=60)
    p.add_argument("--bg-thresh", type=int, default=25)
    p.add_argument("--color-bounds", type=int, nargs=6, default=None,
                   metavar=("Hl", "Sl", "Vl", "Hu", "Su", "Vu"))
    p.add_argument("--n-frames", type=int, default=3, help="采样核对帧数")
    p.add_argument("--out", default="inspect_capture.png")
    return p


def main():
    args = build_parser().parse_args()
    calib = np.load(args.camera_params, allow_pickle=True)
    cp = calib["camera_params"]
    H, W = int(calib["H"]), int(calib["W"])
    undistort = (make_undistorter(calib["K"], calib["dist"], H, W)
                 if "K" in calib and "dist" in calib else None)

    images, _ = load_image_views(args.view_dirs, undistort)     # (V,N,H,W,3)
    V, N = images.shape[:2]
    n = min(args.n_frames, N)
    idxs = np.linspace(0, N - 1, n, dtype=int)

    color_bounds = None
    if args.method == "color":
        cb = args.color_bounds
        color_bounds = (np.array(cb[:3], np.uint8), np.array(cb[3:], np.uint8))
    masks = segment_views(images, args.method, color_bounds=color_bounds,
                          gray_thresh=args.gray_thresh, bg_thresh=args.bg_thresh)
    sk2d = masks_to_skeletons_2d(masks, n_points=31)            # (V,N,31,2)
    sk3d = triangulate_skeletons(sk2d, cp, H, W)                # (N,31,3)

    fig, axes = plt.subplots(n, V + 1, figsize=(4 * (V + 1), 4 * n),
                             squeeze=False)
    eyes = cp[:, 0:3]
    for r, fi in enumerate(idxs):
        for v in range(V):
            ax = axes[r, v]
            img = images[v, fi]
            ax.imshow(img if img.ndim == 3 else img,
                      cmap="gray" if img.ndim == 2 else None)
            ax.imshow(masks[v, fi], cmap="Reds", alpha=0.35)
            sk = sk2d[v, fi]
            if np.abs(sk).max() > 0:
                ax.plot(sk[:, 0], sk[:, 1], "c.-", ms=3, lw=1.5)
            ax.set_title(f"cam{v} frame {fi}")
            ax.axis("off")
        ax3 = axes[r, V]
        ax3.remove()
        ax3 = fig.add_subplot(n, V + 1, r * (V + 1) + V + 1, projection="3d")
        pts = sk3d[fi]
        valid = np.isfinite(pts[:, 0])
        if valid.any():
            p = pts[valid]
            ax3.plot(p[:, 0], p[:, 1], p[:, 2], "o-", ms=3)
        ax3.scatter(eyes[:, 0], eyes[:, 1], eyes[:, 2], c="r", marker="^",
                    s=40, label="cameras")
        ax3.set_title(f"3D frame {fi}")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
        ratio = float(np.isfinite(sk3d[fi]).all(axis=-1).mean())
        print(f"  frame {fi}: 有效节点 {ratio:.0%}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f">>> 保存核对图: {args.out}  （红=分割掩码, 青=2D 骨架, 右列=3D 三角化）")


if __name__ == "__main__":
    main()
