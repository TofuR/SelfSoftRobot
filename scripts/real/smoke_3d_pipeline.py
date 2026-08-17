"""无硬件运行双视角 RGB → 15节点3D NPZ 的端到端 smoke。

生成已知三维曲线、两台标定相机的背光图像和动作日志，再调用正式
``capture_to_npz.py``。输出目录默认保留在 /tmp，便于检查。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _fixture(frames, nodes):
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]], np.float32)
    Rs = np.stack([np.eye(3), np.eye(3)]).astype(np.float32)
    ts = np.asarray([[0.0, 0.0, 0.0], [-0.12, 0.0, 0.0]], np.float32)
    Ps = np.stack([K @ np.hstack([R, t[:, None]])
                   for R, t in zip(Rs, ts)]).astype(np.float32)
    curves = []
    for t in range(frames):
        s = np.linspace(0.0, 1.0, nodes)
        curves.append(np.stack([
            0.025 * np.sin(np.pi * s + 0.15 * t),
            -0.28 + 0.56 * s,
            1.0 + 0.015 * np.cos(np.pi * s),
        ], axis=1))
    return K, Rs, ts, Ps, np.asarray(curves, np.float32)


def _project(P, xyz):
    homogeneous = np.concatenate([xyz, np.ones((*xyz.shape[:2], 1), np.float32)], axis=2)
    q = np.einsum("ij,tnj->tni", P, homogeneous)
    return q[..., :2] / q[..., 2:3]


def build_raw_sequence(directory, frames=8, nodes=15):
    K, Rs, ts, Ps, xyz = _fixture(frames, nodes)
    os.makedirs(directory, exist_ok=True)
    for view, P in enumerate(Ps):
        cam_dir = os.path.join(directory, f"cam{view}")
        os.makedirs(cam_dir, exist_ok=True)
        pixels = _project(P, xyz)
        for t in range(frames):
            image = np.full((480, 640, 3), 245, np.uint8)
            polyline = np.rint(pixels[t]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(image, [polyline], False, (20, 20, 20), 13,
                          lineType=cv2.LINE_8)
            cv2.imwrite(os.path.join(cam_dir, f"{t:05d}.png"), image)

    frame_times = np.arange(frames, dtype=float) * 0.1
    np.savetxt(os.path.join(directory, "frame_times.txt"), frame_times, fmt="%.6f")
    with open(os.path.join(directory, "actions6.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_sec", "c0", "c1", "c2", "c3", "c4", "c5"])
        for t, stamp in enumerate(frame_times):
            writer.writerow([stamp, 10 + t, 5, 0, 0, 0, 0])

    calibration = os.path.join(directory, "calibration.npz")
    camera_params = np.zeros((2, 10), np.float32)
    camera_params[:, 9] = K[0, 0]
    np.savez_compressed(
        calibration, camera_params=camera_params,
        Ks=np.stack([K, K]), dists=np.zeros((2, 5), np.float32),
        Rs=Rs, ts=ts, projection_matrices=Ps,
        H=480, W=640, view_names=np.asarray(["cam0", "cam1"]))
    with open(os.path.join(directory, "fixture.json"), "w", encoding="utf-8") as stream:
        json.dump({"frames": frames, "nodes": nodes}, stream, indent=2)
    return calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description="双视角15节点三维管线 smoke")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args(argv)
    out_dir = os.path.abspath(args.out_dir or tempfile.mkdtemp(prefix="real3d_smoke_"))
    calibration = build_raw_sequence(out_dir, args.frames, 15)
    processed_dir = os.path.join(out_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_npz = os.path.join(processed_dir, "sequence_3d.npz")
    command = [
        sys.executable, os.path.join(ROOT, "scripts", "real", "capture_to_npz.py"),
        "--view-dirs", os.path.join(out_dir, "cam0"), os.path.join(out_dir, "cam1"),
        "--camera-params", calibration,
        "--method", "backlight", "--gray-thresh", "60",
        "--actions", os.path.join(out_dir, "actions6.csv"),
        "--actions-has-timestamps",
        "--frame-times", os.path.join(out_dir, "frame_times.txt"),
        "--n-nodes", "15", "--max-reprojection-error-px", "3.0",
        "--clean-nan", "--dt", "0.1", "--out", output_npz,
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    with np.load(output_npz, allow_pickle=False) as data:
        measured = float((data["position_confidence"] > 0).mean())
        summary = {
            "output": output_npz,
            "positions_shape": list(data["positions"].shape),
            "positions_2d_shape": list(data["positions_2d"].shape),
            "images_shape": list(data["images"].shape),
            "measured_node_ratio": measured,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["positions_shape"] != [args.frames, 3, 15]:
        raise SystemExit("positions shape 验收失败")
    if measured < 0.8:
        raise SystemExit(f"有效三角化节点比例过低: {measured:.1%}")


if __name__ == "__main__":
    main()
