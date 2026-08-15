"""相机适配层:包装 real_capture 的 RealSenseCam,断言指纹与 manifest 一致。

设计 spec §3.2:相机是"数据来源",必须与训练采集位姿一致 —— 用
deploy_manifest.camera 指纹(serial/width/height/fps)校验,防用错设备导致
骨架坐标与训练分布漂移。真机依赖 pyrealsense2;延迟 import。

RealSenseCam 驱动移植在 Task 2(此处暂保留 real_capture 延迟 import 占位)。
"""

from __future__ import annotations

from typing import Any


class CameraHardwareError(RuntimeError):
    pass


def create_realsense_cam(width: int = 640, height: int = 480, fps: int = 30,
                         serial: str | None = None):
    """构造真机 RealSenseCam(QThread,start() 后 emit 帧)。"""
    from real_capture.realsense_cam import RealSenseCam  # type: ignore[import-not-found]
    return RealSenseCam(width=width, height=height, fps=fps, serial=serial)


def assert_camera_fingerprint(descriptor_fingerprint: dict[str, Any] | None,
                              *, width: int, height: int, fps: int,
                              serial: str | None) -> None:
    """校验实际相机与 manifest 指纹一致(不一致则阻断,防坐标漂移)。"""
    if not descriptor_fingerprint:
        return  # 无 manifest 指纹 → 不硬阻断(部署契约缺失时由 preflight 兜底)
    expected = descriptor_fingerprint
    mismatches = []
    if expected.get("width") is not None and int(expected["width"]) != int(width):
        mismatches.append(f"width {expected['width']} != {width}")
    if expected.get("height") is not None and int(expected["height"]) != int(height):
        mismatches.append(f"height {expected['height']} != {height}")
    if expected.get("fps") is not None and int(expected["fps"]) != int(fps):
        mismatches.append(f"fps {expected['fps']} != {fps}")
    if expected.get("serial") and serial and str(expected["serial"]) != str(serial):
        mismatches.append(f"serial {expected['serial']} != {serial}")
    if mismatches:
        raise CameraHardwareError("相机指纹不匹配(可能与训练采集位姿不一致): "
                                  + "; ".join(mismatches))
