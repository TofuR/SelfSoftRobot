"""NDI 适配层:包装 real_capture 的 NdiThread,标记为**隐藏评价流**。

设计 spec §3.3:ndi_mm 只进评价、**永不进控制/模型**。本模块把 NDI 封装成
只读的评价观测源 —— 执行期同步记录末端 mm 真值,但绝不喂给 planner 或模型。
真机依赖 scikit-surgerynditracker;延迟 import。

NdiThread 驱动移植在 Task 3(此处暂保留 real_capture 延迟 import 占位)。
"""

from __future__ import annotations


class NdiHardwareError(RuntimeError):
    pass


# 隐藏评价流标记:任何模型/规划器消费观测前须断言 allowed(ObservationPolicy)。
HIDDEN_EVALUATION_SOURCE = "ndi_hidden_eval"


def create_ndi_thread(port: str, *, rate_hz: float = 50.0, ndi_count: int = 1):
    """构造真机 NdiThread(QThread,emit ndi_data 末端 mm 真值)。"""
    if not port:
        raise NdiHardwareError("NDI 需要串口(COM)")
    from real_capture.hardware_threads import NdiThread  # type: ignore[import-not-found]
    return NdiThread(port=port, rate_hz=rate_hz, ndi_count=ndi_count)


def require_hidden_evaluation_allowed(policy, *, timestamp: float, source: str) -> None:
    """NDI 观测必须经 ObservationPolicy 判为 allowed 才允许进入评价;模型侧禁读。

    设计 spec §3.3:ndi_mm 只进评价。任何把 NDI 数据送进模型的路径都必须
    先调 ObservationPolicy.require_allowed(否则 raise PermissionError)。
    """
    decision = policy.decide(timestamp=timestamp, source=source, force=False)
    policy.require_allowed(decision)
    return decision
