"""calibrate_cameras.py — 实物多相机标定入口。

从棋盘格图像求内参 + 外参，输出 camera_params(V,10)（项目格式），供
capture_to_npz.py 与训练消费。

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
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.calibration import (  # noqa: E402
    calibrate_intrinsics, calibrate_camera_params,
)


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

    print(f">>> 内参标定（{V} 视角，每视角用各自目录的棋盘格图）...")
    intr = calibrate_intrinsics(args.intrinsic_dirs, pattern_size, args.square)
    print(f"    fx={intr['fx']:.1f} fy={intr['fy']:.1f}  "
          f"reproj_error={intr['reproj_error']:.3f} px  "
          f"image_size={intr['image_size']}")

    print(">>> 外参标定（每视角一张【世界原点】棋盘格图）...")
    res = calibrate_camera_params(intr, args.extrinsic_imgs, pattern_size,
                                  args.square, args.H, args.W)
    cp = res["camera_params"]                                  # (V,10)
    for i, v in enumerate(res["views"]):
        print(f"    [{view_names[i]}] eye={np.round(v['eye'], 3).tolist()}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, camera_params=cp, K=intr["K"],
                        dist=intr["dist"], H=args.H, W=args.W,
                        fx=intr["fx"], fy=intr["fy"],
                        view_names=np.array(view_names))
    print(f">>> 保存: {args.out}  camera_params{cp.shape}")
    print("    下一步: capture_to_npz.py --camera-params", args.out)


if __name__ == "__main__":
    main()
