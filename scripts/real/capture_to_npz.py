"""capture_to_npz.py — 实物图像/视频 → 仿真 schema .npz（主管线入口）。

串联（对应 docs/directions/11 §5）:
  io_video.load_image_views(+去畸变)  →  (V,N,H,W,3)
  segmentation.segment_views          →  masks (V,N,H,W)
  segmentation.masks_to_skeletons_2d  →  2D 骨架 (V,N,31,2)   [复用 skeleton_2d]
  triangulation.triangulate_skeletons →  3D 骨架 (N,31,3)     [DLT]
  assemble_npz.save_real_npz          →  data/*.npz           [仿真 schema]

动作同步（§7）：--actions 传入的 (N,A) 实测气压应已按相机帧对齐
（上游用 LED/时间戳对齐）；这里按帧数截断/补零。NDI 末端锚点可选
（--ndi-tip npz，字段 tip=(N,3)）作为独立动态验证。

用法:
  python scripts/real/capture_to_npz.py \\
      --view-dirs raw/seq1/cam0 raw/seq1/cam1 raw/seq1/cam2 \\
      --camera-params config/real_camera_params.npz \\
      --method backlight --gray-thresh 60 --dt 0.0333 \\
      --actions raw/seq1/actions.npz --out data/real_seq/seq1.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.real.io_video import load_image_views, make_undistorter  # noqa: E402
from src.data.real.segmentation import (  # noqa: E402
    segment_views, masks_to_skeletons_2d)
from src.data.real.triangulation import triangulate_skeletons  # noqa: E402
from src.data.real.assemble_npz import save_real_npz  # noqa: E402


def _load_actions(path, n_frames):
    """加载 (N,A) 实测气压，按帧数截断/补零。支持 .npz(actions) 或 .csv。"""
    if path is None:
        return np.zeros((n_frames, 2), np.float32)        # 单腔道占位 [0,0]
    if path.endswith(".npz"):
        a = np.load(path)["actions"]
    else:
        a = np.loadtxt(path, delimiter=",")
        if a.ndim == 1:
            a = a[:, None]
    A = a.shape[1]
    out = np.zeros((n_frames, A), np.float32)
    out[:min(n_frames, len(a))] = a[:min(n_frames, len(a))]
    return out


def build_parser():
    p = argparse.ArgumentParser(description="实物图像/视频 → 仿真 schema .npz")
    p.add_argument("--view-dirs", nargs="+", default=None,
                   help="每视角一个图像目录（与 --videos 二选一）")
    p.add_argument("--videos", nargs="+", default=None,
                   help="每视角一个视频文件")
    p.add_argument("--camera-params", required=True,
                   help="calibrate_cameras.py 输出的 npz")
    p.add_argument("--method", default="backlight",
                   choices=["backlight", "bg_subtract", "color"])
    p.add_argument("--gray-thresh", type=int, default=60)
    p.add_argument("--bg-thresh", type=int, default=25)
    p.add_argument("--color-bounds", type=int, nargs=6, default=None,
                   metavar=("Hl", "Sl", "Vl", "Hu", "Su", "Vu"))
    p.add_argument("--dt", type=float, default=0.0333, help="帧间隔(秒)")
    p.add_argument("--actions", default=None, help="实测气压 (N,A): .npz/.csv")
    p.add_argument("--ndi-tip", default=None, help="NDI 末端锚点 npz(字段 tip)")
    p.add_argument("--out", required=True, help="输出 .npz 路径")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def main():
    args = build_parser().parse_args()
    calib = np.load(args.camera_params, allow_pickle=True)
    cp = calib["camera_params"]                              # (V,10)
    H, W = int(calib["H"]), int(calib["W"])
    view_names = (calib["view_names"].tolist()
                  if "view_names" in calib else
                  [f"cam{i}" for i in range(cp.shape[0])])
    undistort = None
    if "K" in calib and "dist" in calib:
        undistort = make_undistorter(calib["K"], calib["dist"], H, W)

    print(">>> 加载多视角图像（+ 去畸变）...")
    if args.view_dirs:
        images, _ = load_image_views(args.view_dirs, undistort, args.max_frames)
    else:
        from src.data.real.io_video import load_video_views
        images = load_video_views(args.videos, undistort, args.max_frames)
    V, N = images.shape[:2]
    print(f"    {V} 视角 × {N} 帧  {images.shape[2]}x{images.shape[3]}")

    print(f">>> 分割 ({args.method}) → 2D 骨架（复用 skeleton_2d）...")
    color_bounds = None
    if args.method == "color":
        cb = args.color_bounds
        color_bounds = (np.array(cb[:3], np.uint8), np.array(cb[3:], np.uint8))
    masks = segment_views(images, args.method, color_bounds=color_bounds,
                          gray_thresh=args.gray_thresh,
                          bg_thresh=args.bg_thresh)
    sk2d = masks_to_skeletons_2d(masks, n_points=31)         # (V,N,31,2)

    print(">>> 多视角三角化 → 3D 骨架 (GT)...")
    sk3d = triangulate_skeletons(sk2d, cp, H, W)             # (N,31,3)
    valid = np.isfinite(sk3d).all(axis=-1).mean()
    print(f"    有效节点比例: {valid:.1%}")

    actions = _load_actions(args.actions, N)
    ndi_tip = (np.load(args.ndi_tip)["tip"] if args.ndi_tip else None)

    save_real_npz(args.out, images=images, masks=masks, skeletons_3d=sk3d,
                  actions=actions, camera_params=cp, dt=args.dt,
                  view_names=view_names, ndi_tip_anchor=ndi_tip)
    print(f">>> 保存: {args.out}")
    print("    可直接用 train_unified.py / evaluate 训练评估（仿真 schema）")


if __name__ == "__main__":
    main()
