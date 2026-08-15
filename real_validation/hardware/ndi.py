"""NDI 适配层:内部移植 NdiThread + MockNdiThread,标记为**隐藏评价流**。

设计 spec §3.3:ndi_mm 只进评价、**永不进控制/模型**。本模块把 NDI 封装成
只读的评价观测源 —— 执行期同步记录末端 mm 真值,但绝不喂给 planner 或模型。
真机依赖 scikit-surgerynditracker;延迟 import(与 nditracker 原样)。

`NdiThread`/`MockNdiThread` 驱动已自包含移植进本文件(不再经 real_capture 桥接),
源码原样保持:真机用同包 `nditracker.ndi_load(port)` 起跟踪,循环读取前
`ndi_count` 个 tracking object,发 `ndi_data(list[11*ndi_count], float monotonic)`;
每个探头的布局为 [x, y, z, Rx, Ry, Rz, qw, qx, qy, qz, quality];失锁(quality 为
NaN)时把 x..qz 置 NaN(下游 clean-nan 可剔),不写 10000 哨兵。mock 合成 XY 画圆
+ Z 微动的末端轨迹,无硬件也能跑通整条链路。
"""

from __future__ import annotations

import math
import time

from PyQt5.QtCore import QThread, pyqtSignal

# 11 维位姿布局常量(与 nditracker.get_ndi_value 的有效分支一致)
_NDI_N = 11


class NdiThread(QThread):
    """NDI Aurora 读取线程。

    Signals:
        ndi_data(list, float)  — [x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality] + monotonic 时间
        error(str)             — 初始化/读取异常
    """
    ndi_data = pyqtSignal(list, float)
    error = pyqtSignal(str)

    def __init__(self, port: str, rate_hz: float = 50.0, ndi_count: int = 1, parent=None):
        super().__init__(parent)
        self.port = port
        self.period = 1.0 / max(1.0, float(rate_hz))
        self.ndi_count = max(1, int(ndi_count))
        self._running = True
        self._tracker = None

    def run(self):  # QThread entry point
        try:
            from . import nditracker  # 同包内的 nditracker.py(真机才装 sksurgerynditracker)
        except Exception as e:  # pragma: no cover - 环境依赖
            self.error.emit(f"nditracker 导入失败: {e}")
            return
        try:
            self._tracker = nditracker.ndi_load(self.port)   # 内部已 start_tracking
        except Exception as e:  # pragma: no cover - 硬件异常
            self.error.emit(f"NDI 初始化失败（端口 {self.port}）: {e}")
            return
        try:
            while self._running:
                pose = nditracker.get_ndi_values(self._tracker, self.ndi_count)
                self.ndi_data.emit(self._normalize(pose), time.monotonic())
                # 短睡控采样率；sleep 对 stop() 响应足够快（period ~20ms）
                time.sleep(self.period)
        except Exception as e:  # pragma: no cover - 运行期异常
            self.error.emit(f"NDI 读取异常: {e}")
        finally:
            self._stop_tracking()

    def _normalize(self, pose: list) -> list:
        """按 ndi_count 对齐多个 11 维位姿，单个探头失锁不影响其他探头。"""
        out = []
        values = list(pose)
        for i in range(self.ndi_count):
            one = values[i * _NDI_N:(i + 1) * _NDI_N]
            if len(one) < _NDI_N:
                one += [float("nan")] * (_NDI_N - len(one))
            one = one[:_NDI_N]
            quality = one[10]
            try:
                bad = (not math.isfinite(float(quality))
                       or any(abs(float(v)) > 1e6 for v in one[:10]))
            except (TypeError, ValueError):
                bad = True
            out.extend(([float("nan")] * 10 + [float(quality)]) if bad
                       else [float(v) for v in one])
        return out

    def _stop_tracking(self):
        if self._tracker is None:
            return
        try:
            self._tracker.stop_tracking()
        except Exception:
            pass
        self._tracker = None

    def stop(self):
        self._running = False
        self.quit()
        self.wait(3000)


class MockNdiThread(QThread):
    """无 NDI 时的合成末端轨迹：XY 画圆 + Z 微动 + 微小姿态扰动。

    让 recorder/GUI 在无硬件时也能产出非平凡的 3D 末端序列，验证整条管线。
    """
    ndi_data = pyqtSignal(list, float)

    def __init__(self, rate_hz: float = 50.0, ndi_count: int = 1, parent=None):
        super().__init__(parent)
        self.period = 1.0 / max(1.0, float(rate_hz))
        self.ndi_count = max(1, int(ndi_count))
        self._running = True
        self._t0 = time.monotonic()

    def run(self):
        while self._running:
            t = time.monotonic() - self._t0
            # XY 圆周（半径随时间慢变）+ Z 正弦
            r = 40.0 + 10.0 * math.sin(t * 0.3)
            x = r * math.cos(t * 0.8)
            y = r * math.sin(t * 0.8)
            z = 8.0 * math.sin(t * 1.2)
            rx, ry, rz = 5.0 * math.sin(t * 0.5), 5.0 * math.cos(t * 0.4), 0.0
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            quality = 0.95
            poses = []
            for i in range(self.ndi_count):
                dx = 90.0 * i
                poses.extend([x + dx, y, z, rx, ry, rz, qw, qx, qy, qz, quality])
            self.ndi_data.emit(poses, time.monotonic())
            time.sleep(self.period)

    def stop(self):
        self._running = False
        self.quit()
        self.wait(2000)


class NdiHardwareError(RuntimeError):
    pass


# 隐藏评价流标记:任何模型/规划器消费观测前须断言 allowed(ObservationPolicy)。
HIDDEN_EVALUATION_SOURCE = "ndi_hidden_eval"


def create_ndi_thread(port: str, *, rate_hz: float = 50.0, ndi_count: int = 1):
    """构造真机 NdiThread(QThread,emit ndi_data 末端 mm 真值)。"""
    if not port:
        raise NdiHardwareError("NDI 需要串口(COM)")
    return NdiThread(port=port, rate_hz=rate_hz, ndi_count=ndi_count)


def require_hidden_evaluation_allowed(policy, *, timestamp: float, source: str) -> None:
    """NDI 观测必须经 ObservationPolicy 判为 allowed 才允许进入评价;模型侧禁读。

    设计 spec §3.3:ndi_mm 只进评价。任何把 NDI 数据送进模型的路径都必须
    先调 ObservationPolicy.require_allowed(否则 raise PermissionError)。
    """
    decision = policy.decide(timestamp=timestamp, source=source, force=False)
    policy.require_allowed(decision)
    return decision
