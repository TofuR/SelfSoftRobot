"""相机适配层:内部移植 RealSenseCam,断言指纹与 manifest 一致。

设计 spec §3.2:相机是"数据来源",必须与训练采集位姿一致 —— 用
deploy_manifest.camera 指纹(serial/width/height/fps)校验,防用错设备导致
骨架坐标与训练分布漂移。真机依赖 pyrealsense2;延迟 import(与 realsense_cam 原样)。

`RealSenseCam` 驱动已自包含移植进本文件(不再经 real_capture 桥接),源码原样保持:
只取 color stream;每帧发 `frame_ready(np.ndarray BGR, float monotonic_time)`;
带 `mock` 模式合成背光剪影,无相机也能跑通 GUI + recorder + capture_to_npz 链路。
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


def _mock_frame(phase: float, w: int, h: int) -> np.ndarray:
    """合成背光剪影：近白背景 + 一条随 phase 弯曲的暗臂（悬臂梁形状）。

    让 capture_to_npz / inspect_capture 能从合成图里提出非平凡的 2D 骨架，
    从而无硬件也能验证整条管线。
    """
    img = np.full((h, w, 3), 245, dtype=np.uint8)            # 近白背景（背光）
    cx = w / 2.0
    amp = w * 0.20 * np.sin(phase * 0.8)                      # 末端偏移（px），随时间变
    thickness = max(6, int(w * 0.035))
    ys = np.arange(h)
    f = ys / max(1, h - 1)                                     # 0=末端, 1=基座
    x_center = cx + amp * (1.0 - f) ** 2                       # 悬臂梁：末端偏移最大
    x0 = np.clip((x_center - thickness / 2).astype(int), 0, w)
    x1 = np.clip((x_center + thickness / 2).astype(int), 0, w)
    for y in range(h):
        if x1[y] > x0[y]:
            img[y, x0[y]:x1[y], :] = 20                        # 暗剪影
    return img


class RealSenseCam(QThread):
    """Color-stream 捕获线程。

    Signals:
        frame_ready(np.ndarray img_bgr, float t_monotonic)
        error(str)
    """

    frame_ready = pyqtSignal(np.ndarray, float)
    error = pyqtSignal(str)

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 exposure_us: int = 0, gain: int = 0, mock: bool = False,
                 serial=None, parent=None):
        super().__init__(parent)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.exposure_us = int(exposure_us)
        self.gain = int(gain)
        self.mock = bool(mock)
        self.serial = str(serial) if serial else None
        self._running = True

    @staticmethod
    def list_devices():
        """返回当前 RealSense 设备序列号；无驱动/无设备时返回空列表。"""
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            return [str(dev.get_info(rs.camera_info.serial_number))
                    for dev in ctx.query_devices()]
        except Exception:
            return []

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)  # > wait_for_frames(1000)，避免相机阻塞时 wait 超时未退出

    def run(self):  # QThread entry point
        if self.mock:
            self._run_mock()
        else:
            self._run_real()

    # ---------------- 真机 ----------------
    def _run_real(self):
        try:
            import pyrealsense2 as rs
        except Exception as e:  # pragma: no cover - 环境依赖
            self.error.emit(f"pyrealsense2 导入失败: {e}（pip install pyrealsense2）")
            return

        pipe = None
        try:
            ctx = rs.context()
            cfg = rs.config()
            if self.serial:
                cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.color, self.width, self.height,
                              rs.format.bgr8, self.fps)
            pipe = rs.pipeline(ctx)
            prof = pipe.start(cfg)
            self._apply_exposure(prof, rs)
            while self._running:
                frames = pipe.wait_for_frames(1000)  # 短超时，stop() 能尽快响应
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.array(cf.get_data())  # 强制拷贝，脱离 RealSense 复用的帧 buffer（避免跨线程叠影）
                self.frame_ready.emit(img, time.monotonic())
        except Exception as e:  # pragma: no cover - 硬件异常
            self.error.emit(f"RealSense 采集异常: {e}")
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass

    def _apply_exposure(self, prof, rs):
        """背光剪影法：关自动曝光、压短曝光/低 gain 让臂成纯黑剪影。失败则忽略。"""
        try:
            for sensor in prof.get_device().query_sensors():
                opts = set(sensor.get_supported_options())
                if rs.option.enable_auto_exposure in opts:
                    sensor.set_option(rs.option.enable_auto_exposure, 0)
                if self.exposure_us > 0 and rs.option.exposure in opts:
                    sensor.set_option(rs.option.exposure, float(self.exposure_us))
                if rs.option.gain in opts:
                    sensor.set_option(rs.option.gain, float(self.gain))
        except Exception:
            pass  # 曝光控制可选；自动曝光也能采

    # ---------------- mock ----------------
    def _run_mock(self):
        period = 1.0 / max(1, self.fps)
        phase = 0.0
        while self._running:
            t = time.monotonic()
            self.frame_ready.emit(_mock_frame(phase, self.width, self.height), t)
            phase += period
            slack = period - (time.monotonic() - t)
            if slack > 0:
                time.sleep(slack)


class CameraHardwareError(RuntimeError):
    pass


def create_realsense_cam(width: int = 640, height: int = 480, fps: int = 30,
                         serial: str | None = None):
    """构造真机 RealSenseCam(QThread,start() 后 emit 帧)。"""
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
