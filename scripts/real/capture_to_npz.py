"""capture_to_npz.py — 实物图像/视频 → 仿真 schema .npz（主管线入口）。

串联（对应 docs/directions/11 §5）:
  io_video.load_image_views(+去畸变)  →  (V,N,H,W,3)
  segmentation.segment_views          →  masks (V,N,H,W)
  segmentation.masks_to_skeletons_2d  →  2D 骨架 (V,T,J,2)    [默认 J=15]
  triangulation + quality              →  3D 骨架 (T,J,3)      [DLT]
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
from src.data.real.triangulation import (  # noqa: E402
    triangulate_skeletons_with_quality, planar_lift_skeletons)
from src.data.real.assemble_npz import save_real_npz  # noqa: E402
from src.data.real.preprocess import (  # noqa: E402
    clean_nan_skeleton, align_actions_to_frames)


def _load_actions(path, n_frames, has_ts=False, rate=None, frame_times=None):
    """加载实测气压并对齐到相机帧。三种模式:

      - 默认: (N, A) 已按帧对齐 → 截断/补零。
      - has_ts: (M, 1+A)，第 0 列时间戳(秒) → 按帧时刻插值（高频气压对齐）。
      - rate:   (M, A) 无时间戳但以 rate Hz 采样 → 赋时后按帧时刻插值。
    """
    if path is None:
        return np.zeros((n_frames, 2), np.float32)        # 单腔道占位 [0,0]
    if path.endswith(".npz"):
        raw = np.load(path)["actions"]
    else:
        raw = np.atleast_2d(np.genfromtxt(path, delimiter=",", dtype=float))
        while raw.shape[0] and np.isnan(raw[0]).all():    # 跳过表头行（兼容新带表头/旧无表头）
            raw = raw[1:]
    if has_ts or rate:
        if frame_times is None:
            sys.exit("时间戳对齐需要 --fps 或 --frame-times 来定义相机帧时刻")
        if has_ts:
            return align_actions_to_frames(raw, frame_times)
        t = np.arange(raw.shape[0]) / float(rate)
        return align_actions_to_frames(np.hstack([t[:, None], raw]), frame_times)
    A = raw.shape[1]
    out = np.zeros((n_frames, A), np.float32)
    out[:min(n_frames, len(raw))] = raw[:min(n_frames, len(raw))]
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
    p.add_argument("--clean-nan", action="store_true",
                   help="三角化后沿节点轴插值清洗 NaN（推荐，否则训练遇 NaN 会崩）")
    p.add_argument("--actions-has-timestamps", action="store_true",
                   help="气压日志第 0 列为时间戳(秒)，按帧时刻插值（高频对齐）")
    p.add_argument("--actions-rate", type=float, default=None,
                   help="气压日志无时间戳时的采样率(Hz)，按帧时刻插值")
    p.add_argument("--fps", type=float, default=None,
                   help="相机 fps，生成均匀帧时刻 i/fps（须与气压同一时钟原点）")
    p.add_argument("--frame-times", default=None,
                   help="每帧时间戳文件(秒，每行一个)，优先于 --fps")
    p.add_argument("--ndi-tip", default=None, help="NDI 末端锚点 npz(字段 tip 或 tips；多探头保留 tips)")
    p.add_argument("--planar-lift", action="store_true",
                   help="单相机平面升维：射线-平面相交把 2D 骨架升成 3D（1-DOF 平面弯曲）")
    p.add_argument("--plane-point", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="弯曲平面上一点(世界系,米)，默认基座原点 [0,0,0]")
    p.add_argument("--plane-normal", type=float, nargs=3, default=None,
                   metavar=("NX", "NY", "NZ"),
                   help="弯曲平面法向(世界系)；默认=相机朝向(正对安装)")
    p.add_argument("--out", required=True, help="输出 .npz 路径")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--n-nodes", type=int, default=15,
                   help="每条骨架的规范节点数（默认 15）")
    p.add_argument("--max-reprojection-error-px", type=float, default=5.0,
                   help="三角化节点允许的平均重投影误差（像素）")
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
    if "Ks" in calib and "dists" in calib:
        undistort = [make_undistorter(K, dist, H, W)
                     for K, dist in zip(calib["Ks"], calib["dists"])]
    elif "K" in calib and "dist" in calib:
        undistort = make_undistorter(calib["K"], calib["dist"], H, W)
    projection_matrices = (calib["projection_matrices"]
                           if "projection_matrices" in calib else None)

    print(">>> 加载多视角图像（+ 去畸变）...")
    if args.view_dirs:
        images, _ = load_image_views(args.view_dirs, undistort, args.max_frames)
    else:
        from src.data.real.io_video import load_video_views
        images = load_video_views(args.videos, undistort, args.max_frames)
    V, N = images.shape[:2]
    print(f"    {V} 视角 × {N} 帧  {images.shape[2]}x{images.shape[3]}")
    if cp.shape[0] != V:
        sys.exit(f"相机标定数量与图像视角数量不一致：camera_params={cp.shape[0]}，images={V}；"
                 "请使用同一批相机生成的标定文件。")
    if len(view_names) != V:
        view_names = [f"cam{i}" for i in range(V)]

    print(f">>> 分割 ({args.method}) → 2D 骨架（复用 skeleton_2d）...")
    color_bounds = None
    if args.method == "color":
        cb = args.color_bounds
        color_bounds = (np.array(cb[:3], np.uint8), np.array(cb[3:], np.uint8))
    masks = segment_views(images, args.method, color_bounds=color_bounds,
                          gray_thresh=args.gray_thresh,
                          bg_thresh=args.bg_thresh)
    if args.n_nodes < 2:
        sys.exit("--n-nodes 必须至少为 2")
    sk2d = masks_to_skeletons_2d(masks, n_points=args.n_nodes)  # (V,T,J,2)
    visibility = np.isfinite(sk2d).all(axis=-1) & ~np.all(sk2d == 0.0, axis=-1)

    if args.planar_lift:
        if cp.shape[0] != 1:
            sys.exit("--planar-lift 仅支持单相机（V=1）")
        pn = args.plane_normal
        if pn is None:
            eye, center = cp[0, 0:3], cp[0, 3:6]
            pn = np.asarray(center, float) - np.asarray(eye, float)  # 默认=相机朝向(正对)
        print(">>> 平面升维（射线-平面相交，1-DOF 平面弯曲）→ 3D 骨架...")
        sk3d = planar_lift_skeletons(sk2d, cp, args.plane_point, pn, H, W)
        positions_2d = np.transpose(sk2d, (1, 0, 2, 3)).astype(np.float32)
        visibility_tvj = np.transpose(visibility, (1, 0, 2))
        reprojection_error = np.full(visibility_tvj.shape, np.nan, np.float32)
        position_confidence = np.isfinite(sk3d).all(axis=-1).astype(np.float32)
        source_mask = np.where(position_confidence > 0, 1, 0).astype(np.uint8)
    else:
        print(">>> 多视角三角化 + 重投影质控 → 3D 骨架监督...")
        quality = triangulate_skeletons_with_quality(
            sk2d, cp, H, W, args.max_reprojection_error_px,
            projection_matrices=projection_matrices)
        sk3d = quality["positions_3d"]
        positions_2d = quality["positions_2d"]
        visibility_tvj = quality["visibility"]
        reprojection_error = quality["reprojection_error"]
        position_confidence = quality["position_confidence"]
        source_mask = quality["source_mask"]
    valid = np.isfinite(sk3d).all(axis=-1).mean()
    print(f"    有效节点比例: {valid:.1%}")

    if args.clean_nan:
        missing_before_clean = ~np.isfinite(sk3d).all(axis=-1)
        sk3d = clean_nan_skeleton(sk3d)
        source_mask[missing_before_clean] = 3
        position_confidence[missing_before_clean] = 0.0
        print("    已清洗 NaN；插值节点 source_mask=3 且不进入 3D 监督")

    # 帧时刻：用于把高频气压对齐到相机帧（同一时钟原点）
    if args.frame_times:
        frame_times = np.atleast_1d(np.loadtxt(args.frame_times))   # 至少 1 维：单帧文件 loadtxt 返回 0-d 标量会让下游 len() 崩
    elif args.fps:
        frame_times = np.arange(N) / float(args.fps)
    else:
        frame_times = None
    if frame_times is not None and len(frame_times) != N:
        sys.exit(f"frame_times 数量 {len(frame_times)} 与图像帧数 {N} 不一致")
    actions = _load_actions(args.actions, N, args.actions_has_timestamps,
                            args.actions_rate, frame_times)
    if frame_times is not None:
        print(f"    气压按帧时刻插值对齐（{len(frame_times)} 帧）")
    if args.ndi_tip:
        tip_data = np.load(args.ndi_tip)
        ndi_tip = tip_data["tip"] if "tip" in tip_data else tip_data["tips"]
    else:
        ndi_tip = None

    # io/segmentation 使用 (V,T,...);训练 schema 明确使用 (T,V,...)
    images_tv = np.transpose(images, (1, 0, 2, 3, 4))
    masks_tv = np.transpose(masks, (1, 0, 2, 3))
    save_real_npz(args.out, images=images_tv, masks=masks_tv, skeletons_3d=sk3d,
                  actions=actions, camera_params=cp, dt=args.dt,
                  view_names=view_names, ndi_tip_anchor=ndi_tip,
                  positions_2d=positions_2d, visibility=visibility_tvj,
                  reprojection_error=reprojection_error,
                  position_confidence=position_confidence,
                  source_mask=source_mask, frame_times=frame_times,
                  projection_matrices=projection_matrices)
    print(f">>> 保存: {args.out}")
    print("    可用 train_real_3d_transition.py 训练，或由通用 loader 消费")


if __name__ == "__main__":
    main()
