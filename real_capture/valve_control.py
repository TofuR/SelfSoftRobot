# valve_control.py
"""6 通道气压阀控制层 + 自动驱动。

- `ValveController`：把 6 维气压向量 `[c0..c5]` 拆成 modbus 两组
  （group1=`[c0,c1,c2]`、group2=`[c3,c4,c5]`）下发；每次下发 emit
  `action_logged(6vec, monotonic)`——下发值即动作（与仿真 action 语义一致）。
  `modbus_manager.py` 保持原样不动，所有"动作留痕"新逻辑都在这一层。
- `MockValveController`：同接口、无硬件；只回放命令（比例阀本就无反馈）。
- `ValveDriver`：纯计算下一拍 action 向量（random 有界游走 / sweep 往返扫描）。
  实际下发由 recorder 的采集时钟调 `controller.set_pressures(driver.next_action())`。

气压单位 kPa，范围 0–500（4–20mA → 0–500kPa，见 modbus_manager）。
"""
from __future__ import annotations

import random
import time
from typing import List

from PyQt5.QtCore import QObject, pyqtSignal

from modbus_manager import ModbusManager

N_CHAN = 6
P_MIN = 0.0
P_MAX = 500.0


def _clamp6(vec) -> List[float]:
    """规整成 6 维、钳到 [P_MIN, P_MAX]。"""
    v = [float(x) for x in list(vec)[:N_CHAN]]
    if len(v) < N_CHAN:
        v += [P_MIN] * (N_CHAN - len(v))
    return [max(P_MIN, min(P_MAX, x)) for x in v]


class ValveController(QObject):
    """真机：6 维向量 → 2 个 modbus 控制组。

    Args:
        group_ports: {1: "COMx", 2: "COMy"}（两组串口）
        baudrate, slave_addr: modbus 参数
    Signals:
        action_logged(list, float) — 6vec kPa + monotonic 时间（每次下发）
        connection_changed(bool, str) — 两组都连上才算连上
        log(str)
    """
    action_logged = pyqtSignal(list, float)
    connection_changed = pyqtSignal(bool, str)            # 整体状态（任一组连上即 True）+ 摘要
    group_connection_changed = pyqtSignal(int, bool)      # (group_id, connected) 每组独立
    log = pyqtSignal(str)

    def __init__(self, group_ports: dict, baudrate: int = 9600, slave_addr: int = 1, parent=None):
        super().__init__(parent)
        self.group_ports = dict(group_ports)
        self.baudrate = int(baudrate)
        self.slave_addr = int(slave_addr)
        self.mgr = ModbusManager()
        self._last = [P_MIN] * N_CHAN

    def connect_group(self, gid: int):
        """连接单个控制组（串口 open 可能阻塞 → 建议在后台线程调用）。"""
        port = self.group_ports.get(gid)
        if not port:
            msg = f"组{gid} 未配置串口"
            self.connection_changed.emit(self.connected, msg)
            self.log.emit("⚠ " + msg)
            return False, msg
        ok, err = self.mgr.connect_group(gid, port, self.baudrate, self.slave_addr)
        connected = self.mgr.is_group_connected(gid)
        self.group_connection_changed.emit(gid, connected)
        msg = f"g{gid}@{port}:{'OK' if ok else (err or 'FAIL')}"
        self.connection_changed.emit(self.connected, msg)
        self.log.emit((f"Modbus 组{gid} 已连接 " if ok else f"⚠ Modbus 组{gid} 连接失败 ") + msg)
        return ok, err

    def disconnect_group(self, gid: int):
        """断开单个控制组（释放该组串口 + 停通信线程）。"""
        if self.mgr.is_group_connected(gid):
            self.mgr.disconnect_group(gid)
        self.group_connection_changed.emit(gid, False)
        self.connection_changed.emit(self.connected, f"g{gid} 已断开")
        self.log.emit(f"Modbus 组{gid} 已断开")

    def connect(self) -> bool:
        """连接所有已配置组（便捷封装；逐组连接请用 connect_group）。向后兼容。"""
        ok_all = True
        for gid in list(self.group_ports.keys()):
            ok, _ = self.connect_group(gid)
            ok_all = ok_all and ok
        return ok_all

    def is_group_connected(self, gid: int) -> bool:
        return self.mgr.is_group_connected(gid)

    @property
    def connected_groups(self):
        """已连接的组 id 集合（组1→ch0-2，组2→ch3-5）。"""
        return {g for g in self.group_ports if self.mgr.is_group_connected(g)}

    @property
    def connected(self) -> bool:
        return bool(self.connected_groups)

    def set_pressures(self, pressures6):
        """下发 6 维气压（kPa）。**只写给已连接的组**；未连接的组不发命令（避免无效串口写）。"""
        p6 = _clamp6(pressures6)
        conn = self.connected_groups
        if 1 in conn:
            self.mgr.set_all_pressures(1, p6[0:3])
        if 2 in conn:
            self.mgr.set_all_pressures(2, p6[3:6])
        self._last = p6
        self.action_logged.emit(p6, time.monotonic())

    def set_channel(self, idx: int, kpa: float):
        v = list(self._last)
        v[idx] = kpa
        self.set_pressures(v)

    @property
    def last_command(self) -> List[float]:
        return list(self._last)

    def zero_all(self):
        self.set_pressures([P_MIN] * N_CHAN)

    def close(self):
        try:
            self.mgr.close_all()
        except Exception as e:
            self.log.emit(f"Modbus 关闭异常: {e}")


class MockValveController(QObject):
    """`ValveController` 的软件替身：比例阀无反馈，mock 只回放命令值。
    连接状态按组模拟（_mock_conn），与真机一样支持只连一组。"""
    action_logged = pyqtSignal(list, float)
    connection_changed = pyqtSignal(bool, str)
    group_connection_changed = pyqtSignal(int, bool)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last = [P_MIN] * N_CHAN
        self._mock_conn = {1: False, 2: False}
        self.group_ports = {1: "MOCK", 2: "MOCK"}

    def connect_group(self, gid: int):
        self._mock_conn[gid] = True
        self.group_connection_changed.emit(gid, True)
        self.connection_changed.emit(self.connected, f"MOCK 组{gid} 已连接")
        self.log.emit(f"MOCK 组{gid} 已连接（假阀）。")
        return True, ""

    def disconnect_group(self, gid: int):
        self._mock_conn[gid] = False
        self.group_connection_changed.emit(gid, False)
        self.connection_changed.emit(self.connected, f"MOCK 组{gid} 已断开")
        self.log.emit(f"MOCK 组{gid} 已断开。")

    def connect(self) -> bool:
        for gid in list(self._mock_conn.keys()):
            self.connect_group(gid)
        return True

    def is_group_connected(self, gid: int) -> bool:
        return self._mock_conn.get(gid, False)

    @property
    def connected_groups(self):
        return {g for g, c in self._mock_conn.items() if c}

    @property
    def connected(self) -> bool:
        return bool(self.connected_groups)

    def set_pressures(self, pressures6):
        p6 = _clamp6(pressures6)
        self._last = p6
        self.action_logged.emit(p6, time.monotonic())

    def set_channel(self, idx: int, kpa: float):
        v = list(self._last)
        v[idx] = kpa
        self.set_pressures(v)

    @property
    def last_command(self) -> List[float]:
        return list(self._last)

    def zero_all(self):
        self.set_pressures([P_MIN] * N_CHAN)

    def close(self):
        for gid in list(self._mock_conn.keys()):
            self._mock_conn[gid] = False


class ValveDriver(QObject):
    """6 通道自动驱动：random（有界随机游走，反射后强制钳位）/ sweep（往返扫描）。

    纯计算下一拍 action 向量；实际下发由 recorder 的采集时钟负责（调
    `controller.set_pressures(driver.next_action())`）。每通道在 `[lo_i, hi_i]` kPa 内；
    `lo_i==hi_i`（range=0）的通道恒定 → **单通道模式**就是把其余 5 通道 min=max=0。
    """

    def __init__(self, lows, highs, mode: str, step_frac: float = 0.35, parent=None):
        super().__init__(parent)
        lo = [float(x) for x in lows]
        hi = [float(x) for x in highs]
        # 每通道确保 lo<=hi，且钳到合法气压范围
        self.lo = [max(P_MIN, min(a, b)) for a, b in zip(lo, hi)]
        self.hi = [min(P_MAX, max(a, b)) for a, b in zip(lo, hi)]
        self.mode = mode
        self.step_frac = float(step_frac)
        self._cur = list(self.lo)
        self._dir = [1.0] * N_CHAN

    def reset(self):
        self._cur = list(self.lo)
        self._dir = [1.0] * N_CHAN

    def set_ranges(self, lows, highs):
        """运行中改每通道范围（钳到合法域），并把当前值拉进新 [lo,hi]（避免越界）。"""
        lo = [float(x) for x in lows]
        hi = [float(x) for x in highs]
        self.lo = [max(P_MIN, min(a, b)) for a, b in zip(lo, hi)]
        self.hi = [min(P_MAX, max(a, b)) for a, b in zip(lo, hi)]
        self._cur = [max(self.lo[i], min(self.hi[i], self._cur[i])) for i in range(N_CHAN)]

    def next_action(self) -> List[float]:
        out = []
        for i in range(N_CHAN):
            span = self.hi[i] - self.lo[i]
            if span <= 1e-6:
                out.append(self.lo[i])          # range=0 → 钉死（单通道模式的其余通道）
                continue
            if self.mode == "sweep":
                step = span * self.step_frac * self._dir[i]
                nxt = self._cur[i] + step
                if nxt >= self.hi[i]:
                    nxt, self._dir[i] = self.hi[i], -1.0
                elif nxt <= self.lo[i]:
                    nxt, self._dir[i] = self.lo[i], 1.0
            else:                                # random bounded walk
                maxstep = span * self.step_frac
                nxt = self._cur[i] + random.uniform(-maxstep, maxstep)
                if nxt < self.lo[i]:
                    nxt = self.lo[i] + (self.lo[i] - nxt)      # 反射
                if nxt > self.hi[i]:
                    nxt = self.hi[i] - (nxt - self.hi[i])
                nxt = max(self.lo[i], min(self.hi[i], nxt))    # 反射后再钳位
            self._cur[i] = nxt
            out.append(nxt)
        return out
