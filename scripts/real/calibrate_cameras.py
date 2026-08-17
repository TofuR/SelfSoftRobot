"""calibrate_cameras.py — 实物多相机标定入口。

从棋盘格图像分别求每台相机的内参 + 外参，输出完整 P=K[R|t]，供
混合 RealSense/普通相机的 capture_to_npz.py 与训练消费。camera_params(V,10)
继续保存为旧工具兼容字段，但三角化优先使用 projection_matrices。

流程（对应 docs/directions/11 §3）:
  1) 内参：每视角一个【含多张不同姿态棋盘格】的目录 → calibrate_intrinsics
  2) 外参：每视角一张【贴在机器人基座（世界原点）】的棋盘格图 → solve_extrinsics
  3) 换算 → camera_params(V,10) + K/dist（保存供去畸变）

世界系 = 基座处棋盘格自身坐标系。唯一要用尺子量的是方格边长 --square。

用法:
  python scripts/real/calibrate_cameras.py \\
      --intrinsic-dirs cam0_calib cam1_calib cam2_calib \\
      --extrinsic-imgs cam0_world.jpg cam1_world.jpg cam2_world.jpg \\
      --pattern 9 6 --square 0.015 --view-names cam0 cam1 cam2 \\
      --H 480 --W 640 --out config/real_camera_params.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.calibration import calibrate_intrinsics, solve_extrinsics  # noqa: E402
from src.calibration.camera_params_format import extrinsics_to_camera_params  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description="实物多相机标定 → camera_params(V,10)")
    p.add_argument("--intrinsic-dirs", nargs="+", required=True,
                   help="每视角一个目录，内含多张不同姿态的棋盘格图")
    p.add_argument("--extrinsic-imgs", nargs="+", required=True,
                   help="每视角一张【世界原点(基座)处】的棋盘格图")
    p.add_argument("--pattern", nargs=2, type=int, default=[9, 6],
                   metavar=("COLS", "ROWS"), help="棋盘格内角点 列数 行数")
    p.add_argument("--square", type=float, required=True,
                   help="方格边长（米，量一次）")
    p.add_argument("--view-names", nargs="+", default=None,
                   help="视角名（默认 cam0 cam1 ...）")
    p.add_argument("--H", type=int, required=True, help="图像高（像素）")
    p.add_argument("--W", type=int, required=True, help="图像宽（像素）")
    p.add_argument("--out", type=str, default="config/real_camera_params.npz",
                   help="输出路径")
    return p


def main():
    args = build_parser().parse_args()
    V = len(args.intrinsic_dirs)
    if len(args.extrinsic_imgs) != V:
        sys.exit(f"视角数不一致: intrinsic={V} extrinsic="
                 f"{len(args.extrinsic_imgs)}")
    view_names = args.view_names or [f"cam{i}" for i in range(V)]
    pattern_size = tuple(args.pattern)

    print(f">>> 内参标定（{V} 视角，每台相机独立求 K/dist）...")
    intrinsics = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for i, directory in enumerate(args.intrinsic_dirs):
        paths = [path for path in sorted(glob.glob(os.path.join(directory, "*")))
                 if os.path.splitext(path)[1].lower() in extensions]
        if len(paths) < 3:
            sys.exit(f"{view_names[i]} 内参目录至少需要3张图: {directory}")
        intr = calibrate_intrinsics(paths, pattern_size, args.square)
        if tuple(intr["image_size"]) != (args.W, args.H):
            sys.exit(f"{view_names[i]} 标定图尺寸 {intr['image_size']} != {(args.W, args.H)}")
        intrinsics.append(intr)
        print(f"    [{view_names[i]}] fx={intr['fx']:.1f} fy={intr['fy']:.1f} "
              f"error={intr['reproj_error']:.3f}px")

    print(">>> 外参标定（每视角一张【机器人基座系】棋盘格图）...")
    rows, projections, rotations, translations = [], [], [], []
    for i, (intr, image) in enumerate(zip(intrinsics, args.extrinsic_imgs)):
        ex = solve_extrinsics(intr["K"], intr["dist"], image,
                              pattern_size, args.square)
        if not ex["found"]:
            sys.exit(f"外参求解失败: {view_names[i]} {image}")
        legacy = extrinsics_to_camera_params(ex["R"], ex["t"], intr["fx"])
        rows.append([*legacy["eye"], *legacy["center"], *legacy["up"],
                     legacy["focal"]])
        projections.append(intr["K"] @ np.hstack([ex["R"], ex["t"][:, None]]))
        rotations.append(ex["R"])
        translations.append(ex["t"])
        print(f"    [{view_names[i]}] eye={np.round(legacy['eye'], 3).tolist()}")
    cp = np.asarray(rows, np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, camera_params=cp,
                        Ks=np.stack([item["K"] for item in intrinsics]).astype(np.float32),
                        dists=np.stack([item["dist"].reshape(-1) for item in intrinsics]).astype(np.float32),
                        Rs=np.stack(rotations).astype(np.float32),
                        ts=np.stack(translations).astype(np.float32),
                        projection_matrices=np.stack(projections).astype(np.float32),
                        H=args.H, W=args.W,
                        view_names=np.array(view_names))
    print(f">>> 保存: {args.out}  camera_params{cp.shape}, P={(V, 3, 4)}")
    print("    下一步: capture_to_npz.py --camera-params", args.out)


if __name__ == "__main__":
    main()
