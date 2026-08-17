"""可移植相机源：RealSense RGB(D)、普通 OpenCV RGB 与 Mock。

所有源只负责原始帧生产，统一暴露 ``frame_ready(BGR, monotonic_time)``；
后处理、标定和骨架重建不进入本模块。
"""

from __future__ import annotations

import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from realsense_cam import RealSenseCam


class OpenCVCam(QThread):
    """无需厂商 SDK 的 ``cv2.VideoCapture`` RGB 相机。"""

    frame_ready = pyqtSignal(np.ndarray, float)
    error = pyqtSignal(str)

    def __init__(self, device=0, width=640, height=480, fps=30, parent=None):
        super().__init__(parent)
        self.device = _parse_device(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.serial = None
        self.has_depth = False
        self.source_kind = "opencv"
        self._running = True

    def source_metadata(self):
        return {
            "kind": "opencv",
            "device": str(self.device),
            "has_depth": False,
            "depth_scale_m": None,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)

    def run(self):
        try:
            import cv2
        except Exception as error:  # pragma: no cover - 环境依赖
            self.error.emit(f"OpenCV 导入失败: {error}")
            return
        cap = cv2.VideoCapture(self.device)
        try:
            if not cap.isOpened():
                self.error.emit(f"无法打开普通相机/视频源: {self.device}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error.emit(f"普通相机读取失败: {self.device}")
                    return
                self.frame_ready.emit(np.ascontiguousarray(frame), time.monotonic())
        finally:
            cap.release()


def _parse_device(value):
    text = str(value).strip()
    if text and text.lstrip("+-").isdigit():
        return int(text)
    return text


def parse_source_descriptors(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def create_camera_source(descriptor: str, *, width=640, height=480, fps=30):
    """从稳定描述串创建相机，打开失败由线程 signal 明确报告。"""
    value = str(descriptor).strip()
    kind, sep, payload = value.partition(":")
    kind = kind.lower()
    if kind in {"mock", "mock-depth"} and not sep:
        return RealSenseCam(width, height, fps, mock=True,
                            enable_depth=(kind == "mock-depth"))
    if kind in {"realsense", "realsense-depth"}:
        serial = payload.strip() or None
        return RealSenseCam(width, height, fps, serial=serial,
                            enable_depth=(kind == "realsense-depth"))
    if kind == "opencv" and sep and payload.strip():
        return OpenCVCam(payload.strip(), width, height, fps)
    raise ValueError(
        f"未知相机源 {descriptor!r}; 支持 realsense[:SERIAL], "
        "realsense-depth[:SERIAL], opencv:DEVICE, mock, mock-depth")


def create_camera_sources(text: str, *, width=640, height=480, fps=30):
    descriptors = parse_source_descriptors(text)
    if not descriptors:
        raise ValueError("camera source 描述不能为空")
    realsense_serials = []
    for descriptor in descriptors:
        kind, _sep, payload = descriptor.partition(":")
        if kind.lower() in {"realsense", "realsense-depth"}:
            realsense_serials.append(payload.strip())
    if len(realsense_serials) > 1:
        if any(not serial for serial in realsense_serials):
            raise ValueError("多个 RealSense 来源必须逐台填写唯一序列号")
        if len(set(realsense_serials)) != len(realsense_serials):
            raise ValueError("多个 RealSense 来源的序列号不能重复")
    return [create_camera_source(item, width=width, height=height, fps=fps)
            for item in descriptors]
