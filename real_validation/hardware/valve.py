"""阀适配层:包装 real_capture 的 ValveController,**kPa 单位在这里收口**。

设计 spec §4.6:kPa ↔ 模型归一化换算只允许出现在 hardware/valve.py(硬件边界)
与 openloop_planner(优化边界)。本模块保证发往硬件的命令是 **kPa**,且经
ValveController 的 slew limiter 约束;计划动作(已过 preflight)不经 bypass_rate。

真机依赖 pyserial;本模块延迟 import real_capture.valve_control(真机才装)。
"""

from __future__ import annotations

import logging
from typing import Any

from ._bootstrap import ensure_real_capture_importable

_logger = logging.getLogger(__name__)


class ValveHardwareError(RuntimeError):
    pass


def create_valve_controller(group1: str, group2: str, *,
                            baudrate: int = 9600, slave_addr: int = 1):
    """构造真机 ValveController(两个 Modbus 控制组,各 3 通道)。

    延迟 import real_capture(真机才有;Mock 流程不调用)。串口 open 在
    connect_group 时才发生(阻塞 → 应在后台线程调用)。
    """
    if not group1 and not group2:
        raise ValveHardwareError("至少需要一组串口(COM)")
    ports = {}
    if group1:
        ports[1] = str(group1).strip()
    if group2:
        ports[2] = str(group2).strip()

    ensure_real_capture_importable()
    from real_capture.valve_control import ValveController  # type: ignore[import-not-found]
    return ValveController(ports, baudrate=int(baudrate), slave_addr=int(slave_addr))


def connect_valve_groups(controller, groups: tuple[int, ...] = (1, 2)) -> dict[int, tuple[bool, str]]:
    """在后台线程连接指定控制组;返回 {gid: (ok, msg)}。串口 open 可能阻塞。"""
    results: dict[int, tuple[bool, str]] = {}
    for gid in groups:
        try:
            ok, err = controller.connect_group(gid)
            results[gid] = (bool(ok), str(err or ""))
        except Exception as error:  # 硬件异常不吞,标记失败
            results[gid] = (False, f"{type(error).__name__}: {error}")
    return results


def valve_to_kpa_requested(action6, action_scale_kpa) -> tuple[float, ...]:
    """模型归一化动作 → kPa(仅真机执行时,从模型 action_scale_kpa 换算)。

    训练域 [0,1] × action_scale_kpa[i] → kPa。这是硬件边界单位收口;
    Mock 链路不经此函数(plan.actions6 已是 kPa 六通道命令)。
    """
    if action_scale_kpa is None:
        raise ValveHardwareError("缺少 action_scale_kpa,无法换算 kPa(部署契约缺失)")
    if len(action6) != len(action_scale_kpa):
        raise ValveHardwareError(
            f"动作维度 {len(action6)} 与 action_scale_kpa {len(action_scale_kpa)} 不同")
    return tuple(float(v) * float(s) for v, s in zip(action6, action_scale_kpa))
