# realsense_cam.py
"""RealSense RGB / 可选对齐深度捕获线程。

每帧发 `frame_ready(np.ndarray BGR, float monotonic_time)`。
启用深度时额外发 `depth_ready(np.ndarray uint16, float monotonic_time,
float depth_scale_m)`；RGB 与 depth 来自同一个 frameset，并先对齐到 color。
带 `mock` 模式：合成一条随时间弯曲的"剪影臂"，无需相机即可跑通
GUI + recorder + capture_to_npz 整条链路（验收 §8 冒烟清单）。

时间戳用 `time.monotonic()`（绝对值），由 recorder 减去各自的 t0 得相对秒——
这样相机在录制开始前就能开预览，t0 只在录制时才定。
"""
from __future__ import annotations

import time
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


def _mock_depth(color: np.ndarray) -> np.ndarray:
    """与 mock RGB 对齐的毫米深度：背景 1.5m，暗臂 1.0m。"""
    gray = color.mean(axis=2)
    return np.where(gray < 100, 1000, 1500).astype(np.uint16)


class RealSenseCam(QThread):
    """Color-stream + 可选 aligned-depth 捕获线程。

    Signals:
        frame_ready(np.ndarray img_bgr, float t_monotonic)
        error(str)
    """

    frame_ready = pyqtSignal(np.ndarray, float)
    depth_ready = pyqtSignal(np.ndarray, float, float)
    error = pyqtSignal(str)

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 exposure_us: int = 0, gain: int = 0, mock: bool = False,
                 serial=None, enable_depth: bool = False, parent=None):
        super().__init__(parent)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.exposure_us = int(exposure_us)
        self.gain = int(gain)
        self.mock = bool(mock)
        self.serial = str(serial) if serial else None
        self.enable_depth = bool(enable_depth)
        self.has_depth = self.enable_depth
        self.source_kind = "mock" if self.mock else "realsense"
        self.depth_scale_m = 0.001 if self.enable_depth else None
        self._running = True

    def source_metadata(self):
        return {
            "kind": self.source_kind,
            "serial": self.serial,
            "has_depth": self.has_depth,
            "depth_scale_m": self.depth_scale_m,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

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
            if self.enable_depth:
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
            pipe = rs.pipeline(ctx)
            prof = pipe.start(cfg)
            align = rs.align(rs.stream.color) if self.enable_depth else None
            if self.enable_depth:
                self.depth_scale_m = float(
                    prof.get_device().first_depth_sensor().get_depth_scale())
            self._apply_exposure(prof, rs)
            while self._running:
                frames = pipe.wait_for_frames(1000)  # 短超时，stop() 能尽快响应
                if align is not None:
                    frames = align.process(frames)
                cf = frames.get_color_frame()
                if not cf:
                    continue
                img = np.array(cf.get_data())  # 强制拷贝，脱离 RealSense 复用的帧 buffer（避免跨线程叠影）
                stamp = time.monotonic()
                if self.enable_depth:
                    df = frames.get_depth_frame()
                    if not df:
                        continue
                    depth = np.array(df.get_data()).copy()
                    self.depth_ready.emit(depth, stamp, float(self.depth_scale_m))
                self.frame_ready.emit(img, stamp)
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
            color = _mock_frame(phase, self.width, self.height)
            if self.enable_depth:
                self.depth_ready.emit(_mock_depth(color), t, 0.001)
            self.frame_ready.emit(color, t)
            phase += period
            slack = period - (time.monotonic() - t)
            if slack > 0:
                time.sleep(slack)
