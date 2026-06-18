"""calibration.py — 实物相机标定（内参 + 外参），输出 camera_params(V,10)。

回答"如何得到 camera_params"：不是用尺子量 eye/center/up/focal，而是
  1) 内参标定：每相机拍多张棋盘格 → cv2.calibrateCamera → K, dist, fx
  2) 外参标定：每相机拍一张【放在世界原点】的棋盘格 → solvePnP → R, t
  3) 换算：R, t, fx → extrinsics_to_camera_params → eye, center, up, focal

世界系 = 标定靶自身坐标系（方格角点按边长定义）。把靶贴在机器人基座处，
靶系即 robot-base 系。相机与机器人的距离/高度不必作为输入——它们是标定输出，
仅用于 sanity check。唯一要用尺子量的是方格边长（square_size，米）。
"""

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _CV2_ERR = exc

from src.calibration.camera_params_format import extrinsics_to_camera_params


def _object_points(pattern_size, square_size):
    nx, ny = pattern_size
    obj = np.zeros((nx * ny, 3), float)
    obj[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    return (obj * square_size).astype(np.float32)


def _find_corners(gray, pattern_size):
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate_intrinsics(image_paths, pattern_size, square_size):
    """棋盘格内参标定。

    Args:
        image_paths: 该相机的棋盘格图像路径列表（≥3 张不同姿态）。
        pattern_size: (内角点列数, 行数)，如 (9, 6)。
        square_size: 方格边长（米，量一次）。

    Returns:
        dict(K=(3,3), dist, fx, fy, reproj_error, image_size=(W,H))。
    """
    objp = _object_points(pattern_size, square_size)
    objpoints, imgpoints, img_size = [], [], None
    for path in image_paths:
        if cv2 is None:
            raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        corners = _find_corners(gray, pattern_size)
        if corners is None:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
    if not objpoints:
        raise RuntimeError(f"未在 {len(image_paths)} 张图中检测到棋盘格角点")
    ret, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size,
                                             None, None)
    return {"K": K, "dist": dist, "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "reproj_error": float(ret), "image_size": img_size}


def solve_extrinsics(K, dist, image, pattern_size, square_size):
    """对一张【世界系原点】处的棋盘格图解外参 R, t（world→camera）。

    世界系 = 该棋盘格自身坐标系（靶放在机器人基座处即 base 系）。

    Args:
        image: 图像路径或 BGR 数组。
        K, dist: 内参（来自 calibrate_intrinsics）。
        pattern_size, square_size: 同 calibrate_intrinsics。

    Returns:
        dict(R=(3,3), t=(3,), found=bool)。
    """
    if cv2 is None:
        raise RuntimeError(f"需要 opencv：{_CV2_ERR}")
    if isinstance(image, str):
        image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = _find_corners(gray, pattern_size)
    if corners is None:
        return {"R": None, "t": None, "found": False}
    objp = _object_points(pattern_size, square_size)
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        return {"R": None, "t": None, "found": False}
    R, _ = cv2.Rodrigues(rvec)
    return {"R": R, "t": tvec.reshape(3), "found": True}


def calibrate_camera_params(intrinsics, extrinsics_images, pattern_size,
                            square_size, H, W):
    """汇总：内参 + 每视角外参图 → camera_params(V,10) + 各视角详情。

    Args:
        intrinsics: calibrate_intrinsics 输出（各视角共享同一内参）。
        extrinsics_images: list[V]，每视角一张【世界原点】棋盘格图路径。
        pattern_size, square_size: 同上。
        H, W: 输出 camera_params 对应图像尺寸（主点 cx=W/2, cy=H/2）。

    Returns:
        dict(camera_params=(V,10) float32, views=[{K,R,t,eye,center,up,focal}])。
    """
    K, dist, fx = intrinsics["K"], intrinsics["dist"], intrinsics["fx"]
    views = []
    for path in extrinsics_images:
        ex = solve_extrinsics(K, dist, path, pattern_size, square_size)
        if not ex["found"]:
            raise RuntimeError(f"外参求解失败（未检测到棋盘格）: {path}")
        p = extrinsics_to_camera_params(ex["R"], ex["t"], fx)
        views.append({"K": K, "R": ex["R"], "t": ex["t"], **p})
    rows = [[*v["eye"], *v["center"], *v["up"], v["focal"]] for v in views]
    return {"camera_params": np.array(rows, dtype=np.float32), "views": views}
