"""camera_params_format.py — OpenCV 标定结果 ↔ 项目 camera_params(V,10) 格式互转。

项目用 [eye, center, up, focal] 描述每个相机（见 src/utils/camera_system.py 与
src/utils/skeleton_2d.py::project_3d_to_2d）。实物标定得到的是 OpenCV 约定的
内参 K 与外参 [R|t]（world→camera，相机看 +z，x 右 y 下）。本模块在两者间无损互转，
使实物数据能直接填入仿真 schema 的 camera_params(V,10)。

投影约定（与 skeleton_2d.project_3d_to_2d 完全一致，三角化据此重建 P）::

    view_dir = normalize(center - eye)          # 相机朝向 = OpenCV +z 在世界系方向
    right    = normalize(cross(view_dir, up))    # 图像右 = OpenCV +x
    true_up  = normalize(cross(right, view_dir)) # 图像上 = OpenCV -y
    col = focal * (p-eye)·right    / (p-eye)·view_dir + W/2
    row = -focal * (p-eye)·true_up / (p-eye)·view_dir + H/2
    K   = [[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]]   # fx=fy=focal, 主点=图像中心

注意：项目 schema 只存单一 focal、主点固定为图像中心。实物若 cx/cy 偏离中心或
有畸变，应在送入骨架提取前对图像 cv2.undistort + 必要时裁剪重定心。
"""

import numpy as np


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def extrinsics_to_camera_params(R, t, fx, fy=None):
    """OpenCV 外参 [R|t]（world→camera）→ 项目 {eye, center, up, focal}。

    Args:
        R: (3,3) world→camera 旋转（OpenCV: 相机看 +z, x 右, y 下）。
        t: (3,) world→camera 平移。
        fx, fy: 像素焦距（fy 默认 = fx；项目用单一 focal，取 fx）。

    Returns:
        dict(eye=(3,), center=(3,), up=(3,), focal=float)。
    """
    R = np.asarray(R, float).reshape(3, 3)
    t = np.asarray(t, float).reshape(3)
    eye = -R.T @ t                                      # 相机在世界系的位置
    view_dir = _unit(R.T @ np.array([0.0, 0.0, 1.0]))  # 朝向 = +z_cam 在世界系
    center = eye + view_dir
    up = R.T @ np.array([0.0, -1.0, 0.0])               # 图像上方向；使 cross(view_dir,up)∝right
    return {"eye": eye, "center": center, "up": up, "focal": float(fx)}


def camera_params_to_K_Rt(params, H, W):
    """项目 {eye, center, up, focal} → OpenCV K, R, t（world→camera）。

    用于三角化/投影，与 skeleton_2d.project_3d_to_2d 完全一致。

    Args:
        params: dict 或 (10,) 数组 [eye(3), center(3), up(3), focal]。
        H, W: 图像尺寸（主点 cx=W/2, cy=H/2）。

    Returns:
        K (3,3), R (3,3), t (3,)。
    """
    if not isinstance(params, dict):
        params = np.asarray(params, float).reshape(10)
        params = {"eye": params[0:3], "center": params[3:6],
                  "up": params[6:9], "focal": float(params[9])}
    eye = np.asarray(params["eye"], float).reshape(3)
    center = np.asarray(params["center"], float).reshape(3)
    up = np.asarray(params["up"], float).reshape(3)
    focal = float(params["focal"])

    view_dir = _unit(center - eye)
    right = _unit(np.cross(view_dir, up))
    true_up = _unit(np.cross(right, view_dir))
    # world→camera: x_cam=right, y_cam=-true_up(图像下), z_cam=view_dir(前)
    R = np.stack([right, -true_up, view_dir], axis=0)   # 行 = 相机轴在世界系方向
    t = -R @ eye
    K = np.array([[focal, 0.0, W / 2.0],
                  [0.0, focal, H / 2.0],
                  [0.0, 0.0, 1.0]], float)
    return K, R, t


def build_camera_params_array(views, H=None, W=None):
    """多视角标定结果 → (V,10) [eye, center, up, focal]。

    Args:
        views: list，每项为 {R, t, fx}（OpenCV 外参，推荐）或
               {eye, center, up, focal}（已是项目格式）。
        H, W: 仅用于记录，不影响数组（focal 已含尺度）。

    Returns:
        (V, 10) float32，与 MultiCameraSystem.get_camera_params_array() 同格式。
    """
    rows = []
    for v in views:
        if {"R", "t"} <= set(v):
            p = extrinsics_to_camera_params(v["R"], v["t"],
                                            v.get("fx", v.get("focal")))
        else:
            p = {"eye": np.asarray(v["eye"], float).reshape(3),
                 "center": np.asarray(v["center"], float).reshape(3),
                 "up": np.asarray(v["up"], float).reshape(3),
                 "focal": float(v["focal"])}
        rows.append([*p["eye"], *p["center"], *p["up"], p["focal"]])
    return np.array(rows, dtype=np.float32)


def projection_matrix(params, H, W):
    """单视角 → P (3,4) world→pixel（K @ [R|t]），供三角化用。"""
    K, R, t = camera_params_to_K_Rt(params, H, W)
    return K @ np.hstack([R, t.reshape(3, 1)])
