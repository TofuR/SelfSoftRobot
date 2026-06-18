"""_smoke_triangulation.py — 自检：标定格式桥 + 三角化的数值往返正确性。

无硬件依赖：合成相机 + 合成 3D 点 → 投影 → 三角化 → 比对。
覆盖:
  1) OpenCV [R|t] ↔ camera_params 精确往返
  2) 多视角投影 → 三角化 往返
  3) 与项目 src/utils/skeleton_2d.project_3d_to_2d 的投影一致性

用法: python scripts/real/_smoke_triangulation.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.calibration.camera_params_format import (  # noqa: E402
    extrinsics_to_camera_params, camera_params_to_K_Rt,
    build_camera_params_array, projection_matrix,
)
from src.data.real.triangulation import triangulate_skeletons  # noqa: E402


def random_rotation(rng):
    """随机正交旋转矩阵（det=+1）。"""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def main():
    rng = np.random.default_rng(0)
    H, W, fx = 480, 640, 800.0

    print("== 1) [R|t] <-> camera_params 往返 ==")
    max_r = max_t = 0.0
    for _ in range(20):
        R = random_rotation(rng)
        t = rng.standard_normal(3) + np.array([0.0, 0.0, 2.0])
        p = extrinsics_to_camera_params(R, t, fx)
        _, R2, t2 = camera_params_to_K_Rt(p, H, W)
        max_r = max(max_r, float(np.max(np.abs(R - R2))))
        max_t = max(max_t, float(np.max(np.abs(t - t2))))
    print(f"   max|R-R'|={max_r:.2e}  max|t-t'|={max_t:.2e}")
    assert max_r < 1e-9 and max_t < 1e-6, "格式桥往返失败"
    print("   OK")

    print("== 2) 多视角 投影 → 三角化 往返 ==")
    look_at = [0.0, 0.0, 0.05]
    cam_defs = [
        dict(eye=[1.2, 0.0, 0.5], center=look_at, up=[0, 0, 1], focal=fx),
        dict(eye=[0.6, 1.0, 0.5], center=look_at, up=[0, 0, 1], focal=fx),
        dict(eye=[-0.6, 0.8, 0.5], center=look_at, up=[0, 0, 1], focal=fx),
    ]
    cp = build_camera_params_array(cam_defs, H, W)            # (3,10)
    Ps = [projection_matrix(r, H, W) for r in cp]
    pts3d = rng.uniform([-0.05, -0.05, 0.0], [0.05, 0.05, 0.15], size=(31, 3))
    sk2d = np.zeros((3, 1, 31, 2))
    for v, P in enumerate(Ps):
        x = np.hstack([pts3d, np.ones((31, 1))])
        proj = (P @ x.T).T
        sk2d[v, 0] = proj[:, :2] / proj[:, 2:3]
    tri = triangulate_skeletons(sk2d, cp, H, W)               # (1,31,3)
    err = float(np.max(np.abs(tri[0] - pts3d)))
    print(f"   max triangulation error = {err:.2e} m")
    assert err < 1e-6, "三角化往返失败"
    print("   OK")

    print("== 3) 与 skeleton_2d.project_3d_to_2d 一致性 ==")
    try:
        import torch  # noqa: F401
        from src.utils.skeleton_2d import project_3d_to_2d
        cam = cam_defs[0]
        p2 = project_3d_to_2d(
            torch.from_numpy(pts3d).float(),
            cam["eye"], cam["center"], cam["up"], fx, H, W)
        p2 = np.asarray(p2)
        x = np.hstack([pts3d, np.ones((31, 1))])
        proj = (Ps[0] @ x.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err2 = float(np.max(np.abs(p2 - proj)))
        print(f"   max|project_3d_to_2d - P| = {err2:.2e} px")
        assert err2 < 1e-3, "与项目投影不一致"  # float32 torch vs float64 numpy
        print("   OK")
    except ImportError:
        print("   (torch 不可用，跳过)")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
