"""实物相机标定模块。

把 OpenCV 标定结果（内参 K + 外参 [R|t]）转换成项目 camera_params(V,10) 格式
[eye, center, up, focal]，使实物数据可直接填入仿真 schema。

回答"如何得到 camera_params"：不是用尺子量 eye/center/up/focal，而是
  1) 棋盘格内参标定 → K, dist, fx（见 calibration.calibrate_intrinsics）
  2) 世界原点棋盘格外参标定 → R, t（见 calibration.solve_extrinsics）
  3) 换算 → eye, center, up, focal（见 camera_params_format）
"""

from src.calibration.camera_params_format import (
    extrinsics_to_camera_params,
    camera_params_to_K_Rt,
    build_camera_params_array,
)
from src.calibration.calibration import (
    calibrate_intrinsics,
    solve_extrinsics,
    calibrate_camera_params,
)

__all__ = [
    "extrinsics_to_camera_params",
    "camera_params_to_K_Rt",
    "build_camera_params_array",
    "calibrate_intrinsics",
    "solve_extrinsics",
    "calibrate_camera_params",
]
