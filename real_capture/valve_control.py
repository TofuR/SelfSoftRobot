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

import csv
import itertools
import random
import time
from array import array
from typing import List, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from modbus_manager import ModbusManager

N_CHAN = 6
P_MIN = 0.0
P_MAX = 500.0
DEFAULT_RATE_KPA_S = 100.0
# 这是“最终命令”容差，不是真实腔压容差。Modbus 4–20mA 换算分辨率约
# 1/32 kPa；留 0.5 kPa 可容纳量化/浮点差异，同时仍能抓住映射或限速配置错误。
EQUALITY_TOLERANCE_KPA = 0.5


def _clamp6(vec) -> List[float]:
    """规整成 6 维、钳到 [P_MIN, P_MAX]。"""
    v = [float(x) for x in list(vec)[:N_CHAN]]
    if len(v) < N_CHAN:
        v += [P_MIN] * (N_CHAN - len(v))
    return [max(P_MIN, min(P_MAX, x)) for x in v]


def _vec6(values, fill=0.0) -> List[float]:
    v = [float(x) for x in list(values)[:N_CHAN]]
    if len(v) < N_CHAN:
        v += [float(fill)] * (N_CHAN - len(v))
    return v


def normalize_channel_equalities(pairs) -> tuple[tuple[int, int], ...]:
    """规范化互不重叠的 ``(leader, follower)`` 六通道等值约束。"""
    result = []
    used = set()
    for item in pairs or ():
        if len(item) != 2:
            raise ValueError("每个通道等值约束必须是 [leader, follower]")
        leader, follower = (int(item[0]), int(item[1]))
        if leader == follower:
            raise ValueError("通道不能跟随自身")
        if leader not in range(N_CHAN) or follower not in range(N_CHAN):
            raise ValueError("等值约束通道必须位于 0..5")
        if leader in used or follower in used:
            raise ValueError("等值约束必须是互不重叠的通道对")
        used.update((leader, follower))
        result.append((leader, follower))
    return tuple(result)


def apply_channel_equalities(values, pairs) -> List[float]:
    """把 follower 投影为 leader；输入输出均为六通道 kPa。"""
    result = _clamp6(values)
    for leader, follower in normalize_channel_equalities(pairs):
        result[follower] = result[leader]
    return result


def channel_equality_residuals(values, pairs) -> tuple[float, ...]:
    vector = _vec6(values)
    return tuple(abs(vector[leader] - vector[follower])
                 for leader, follower in normalize_channel_equalities(pairs))


class PressureSlewLimiter:
    """按实际命令间隔限制每通道的压力命令变化速率。"""

    def __init__(self, rise_rates=None, fall_rates=None, initial=None):
        self.rise = _vec6(rise_rates or [DEFAULT_RATE_KPA_S] * N_CHAN)
        self.fall = _vec6(fall_rates or [DEFAULT_RATE_KPA_S] * N_CHAN)
        self._last = _clamp6(initial or [P_MIN] * N_CHAN)
        self._last_t = time.monotonic()

    def configure(self, rise_rates, fall_rates, initial=None):
        self.rise = [max(0.0, float(x)) for x in _vec6(rise_rates)]
        self.fall = [max(0.0, float(x)) for x in _vec6(fall_rates)]
        if initial is not None:
            self._last = _clamp6(initial)
        self._last_t = time.monotonic()

    def apply(self, target, now=None, bypass=False):
        now = time.monotonic() if now is None else float(now)
        dt = max(0.0, now - self._last_t)
        requested = _clamp6(target)
        if bypass:
            applied = requested
        else:
            applied = []
            for i, value in enumerate(requested):
                delta = value - self._last[i]
                rate = self.rise[i] if delta >= 0.0 else self.fall[i]
                max_delta = float("inf") if rate <= 0.0 else rate * dt
                applied.append(self._last[i] + max(-max_delta, min(max_delta, delta)))
            applied = _clamp6(applied)
        self._last = applied
        self._last_t = now
        return applied


def load_action_sequence(path: str):
    """加载旧/新 actions6.csv，返回相对时间和六通道动作。"""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            try:
                values = [float(x) for x in row[:7]]
            except (TypeError, ValueError):
                continue
            if len(values) == 7:
                rows.append(values)
    if not rows:
        raise ValueError(f"actions6.csv 没有有效的 7 列数值记录: {path}")
    times = [max(0.0, float(r[0]) - float(rows[0][0])) for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("actions6.csv 时间戳必须严格递增")
    return times, [_clamp6(r[1:7]) for r in rows]


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
    command_issued = pyqtSignal(str, list, list, float)  # id, requested, applied, monotonic
    communication_result = pyqtSignal(str, int, bool, float, str)  # id, group, ok, t_ack, status
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
        self._command_ids = itertools.count(1)
        self._required_groups = None
        self._channel_equalities = ()
        self._slew = PressureSlewLimiter(initial=self._last)
        self.mgr.command_ack.connect(self._on_command_ack)
        self.mgr.command_error.connect(self._on_command_error)

    def allocate_command_id(self) -> str:
        return str(next(self._command_ids))

    def set_required_groups(self, groups=None):
        self._required_groups = None if groups is None else set(int(g) for g in groups)

    def configure_safety(self, rise_rates, fall_rates):
        rise = _vec6(rise_rates)
        fall = _vec6(fall_rates)
        for leader, follower in self._channel_equalities:
            if (abs(rise[leader] - rise[follower]) > EQUALITY_TOLERANCE_KPA or
                    abs(fall[leader] - fall[follower]) > EQUALITY_TOLERANCE_KPA):
                raise ValueError("等值通道必须使用相同 rise/fall 速率")
        self._slew.configure(rise_rates, fall_rates, initial=self._last)

    def configure_channel_equalities(self, pairs) -> None:
        normalized = normalize_channel_equalities(pairs)
        residuals = channel_equality_residuals(self._last, normalized)
        if any(value > EQUALITY_TOLERANCE_KPA for value in residuals):
            raise ValueError("启用等值约束前 linked 通道当前命令必须相等；请先全部归零")
        self._channel_equalities = normalized

    @property
    def channel_equalities(self):
        return tuple(self._channel_equalities)

    def _on_command_ack(self, command_id, group_id, t_ack):
        self.communication_result.emit(str(command_id), int(group_id), True,
                                       float(t_ack), "ack")

    def _on_command_error(self, command_id, group_id, t_ack, status):
        self.communication_result.emit(str(command_id), int(group_id), False,
                                       float(t_ack), str(status))

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

    def set_pressures(self, pressures6, command_id=None, bypass_rate=False,
                      required_groups=None):
        """下发 6 维气压；返回 ``(command_id, applied6)``。"""
        t_command = time.monotonic()
        command_id = str(command_id or self.allocate_command_id())
        requested = apply_channel_equalities(pressures6, self._channel_equalities)
        p6 = self._slew.apply(requested, now=t_command, bypass=bypass_rate)
        if any(value > EQUALITY_TOLERANCE_KPA for value in
               channel_equality_residuals(p6, self._channel_equalities)):
            raise RuntimeError("限速后的 applied6 破坏了通道等值约束")
        conn = self.connected_groups
        if required_groups is not None:
            required = set(int(g) for g in required_groups)
        elif self._required_groups is not None:
            required = self._required_groups
        else:
            required = conn
        for gid in (1, 2):
            if gid not in required:
                self.communication_result.emit(command_id, gid, True,
                                               t_command, "inactive")
            elif gid not in conn:
                self.communication_result.emit(command_id, gid, False,
                                               t_command, "not_connected")
            elif not self.mgr.set_all_pressures(
                    gid, p6[0:3] if gid == 1 else p6[3:6], command_id):
                self.communication_result.emit(command_id, gid, False,
                                               t_command, "queue_full")
        self._last = p6
        self.command_issued.emit(command_id, requested, p6, t_command)
        self.action_logged.emit(p6, t_command)
        return command_id, list(p6), t_command

    def set_channel(self, idx: int, kpa: float):
        v = list(self._last)
        v[idx] = kpa
        self.set_pressures(v)

    @property
    def last_command(self) -> List[float]:
        return list(self._last)

    def zero_all(self):
        # 归零是安全动作：不受当前采集模式的 required_groups 限制，
        # 对所有当前已连接组下发，避免单通道录制时另一组仍保留旧压力。
        return self.set_pressures([P_MIN] * N_CHAN, bypass_rate=True,
                                  required_groups=self.connected_groups)

    def close(self):
        try:
            self.mgr.close_all()
        except Exception as e:
            self.log.emit(f"Modbus 关闭异常: {e}")

    def wait_idle(self, timeout_s=1.0):
        self.mgr.wait_idle(timeout_s)


class MockValveController(QObject):
    """`ValveController` 的软件替身：比例阀无反馈，mock 只回放命令值。
    连接状态按组模拟（_mock_conn），与真机一样支持只连一组。"""
    action_logged = pyqtSignal(list, float)
    command_issued = pyqtSignal(str, list, list, float)
    communication_result = pyqtSignal(str, int, bool, float, str)
    connection_changed = pyqtSignal(bool, str)
    group_connection_changed = pyqtSignal(int, bool)
    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last = [P_MIN] * N_CHAN
        self._mock_conn = {1: False, 2: False}
        self.group_ports = {1: "MOCK", 2: "MOCK"}
        self._command_ids = itertools.count(1)
        self._required_groups = None
        self._channel_equalities = ()
        self._slew = PressureSlewLimiter(initial=self._last)

    def allocate_command_id(self) -> str:
        return str(next(self._command_ids))

    def set_required_groups(self, groups=None):
        self._required_groups = None if groups is None else set(int(g) for g in groups)

    def configure_safety(self, rise_rates, fall_rates):
        rise = _vec6(rise_rates)
        fall = _vec6(fall_rates)
        for leader, follower in self._channel_equalities:
            if (abs(rise[leader] - rise[follower]) > EQUALITY_TOLERANCE_KPA or
                    abs(fall[leader] - fall[follower]) > EQUALITY_TOLERANCE_KPA):
                raise ValueError("等值通道必须使用相同 rise/fall 速率")
        self._slew.configure(rise_rates, fall_rates, initial=self._last)

    def configure_channel_equalities(self, pairs) -> None:
        normalized = normalize_channel_equalities(pairs)
        residuals = channel_equality_residuals(self._last, normalized)
        if any(value > EQUALITY_TOLERANCE_KPA for value in residuals):
            raise ValueError("启用等值约束前 linked 通道当前命令必须相等；请先全部归零")
        self._channel_equalities = normalized

    @property
    def channel_equalities(self):
        return tuple(self._channel_equalities)

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

    def set_pressures(self, pressures6, command_id=None, bypass_rate=False,
                      required_groups=None):
        t_command = time.monotonic()
        command_id = str(command_id or self.allocate_command_id())
        requested = apply_channel_equalities(pressures6, self._channel_equalities)
        p6 = self._slew.apply(requested, now=t_command, bypass=bypass_rate)
        if any(value > EQUALITY_TOLERANCE_KPA for value in
               channel_equality_residuals(p6, self._channel_equalities)):
            raise RuntimeError("限速后的 applied6 破坏了通道等值约束")
        self._last = p6
        self.command_issued.emit(command_id, requested, p6, t_command)
        self.action_logged.emit(p6, t_command)
        if required_groups is not None:
            required = set(int(g) for g in required_groups)
        elif self._required_groups is not None:
            required = self._required_groups
        else:
            required = self.connected_groups
        for gid in (1, 2):
            if gid not in required:
                ok, status = True, "inactive"
            elif gid in self.connected_groups:
                ok, status = True, "ack"
            else:
                ok, status = False, "not_connected"
            # 异步回调，保持与真实 Modbus 的生命周期顺序一致。
            QTimer.singleShot(0, lambda g=gid, good=ok, s=status: self.communication_result.emit(
                command_id, g, good, time.monotonic(), s))
        return command_id, list(p6), t_command

    def set_channel(self, idx: int, kpa: float):
        v = list(self._last)
        v[idx] = kpa
        self.set_pressures(v)

    @property
    def last_command(self) -> List[float]:
        return list(self._last)

    def zero_all(self):
        return self.set_pressures([P_MIN] * N_CHAN, bypass_rate=True,
                                  required_groups=self.connected_groups)

    def close(self):
        for gid in list(self._mock_conn.keys()):
            self._mock_conn[gid] = False


class ValveDriver(QObject):
    """6 通道自动驱动：random（有界随机游走，反射后强制钳位）/ sweep（往返扫描）。

    纯计算下一拍 action 向量；实际下发由 recorder 的采集时钟负责（调
    `controller.set_pressures(driver.next_action())`）。每通道在 `[lo_i, hi_i]` kPa 内；
    `lo_i==hi_i`（range=0）的通道恒定 → **单通道模式**就是把其余 5 通道 min=max=0。
    """

    def __init__(self, lows, highs, mode: str, step_frac: float = 0.35,
                 seed: Optional[int] = None, parent=None):
        super().__init__(parent)
        lo = [float(x) for x in lows]
        hi = [float(x) for x in highs]
        # 每通道确保 lo<=hi，且钳到合法气压范围
        self.lo = [max(P_MIN, min(a, b)) for a, b in zip(lo, hi)]
        self.hi = [min(P_MAX, max(a, b)) for a, b in zip(lo, hi)]
        self.mode = mode
        self.step_frac = float(step_frac)
        self.seed = seed
        self.rng = random.Random(seed)
        self._cur = list(self.lo)
        self._dir = [1.0] * N_CHAN
        # 紧凑存储预生成动作：最多 1e6 步时约 24 MB，而不是百万个 Python
        # list/float 对象造成数百 MB 峰值。每步仍按 6 维 list 对外返回。
        self._sequence = None
        self._sequence_index = 0

    def reset(self):
        self._cur = list(self.lo)
        self._dir = [1.0] * N_CHAN
        self._sequence_index = 0

    def pre_generate(self, count: int):
        """预生成固定动作序列；count<=0 时恢复在线生成。"""
        count = min(1_000_000, max(0, int(count)))
        if count <= 0:
            self._sequence = None
            self._sequence_index = 0
            return
        self._sequence = None
        self._sequence_index = 0
        self.reset()
        sequence = array("f")
        for _ in range(count):
            sequence.extend(self._next_generated())
        self._sequence = sequence
        self._sequence_index = 0
        self.reset()

    def set_ranges(self, lows, highs):
        """运行中改每通道范围（钳到合法域），并把当前值拉进新 [lo,hi]（避免越界）。"""
        lo = [float(x) for x in lows]
        hi = [float(x) for x in highs]
        self.lo = [max(P_MIN, min(a, b)) for a, b in zip(lo, hi)]
        self.hi = [min(P_MAX, max(a, b)) for a, b in zip(lo, hi)]
        self._cur = [max(self.lo[i], min(self.hi[i], self._cur[i])) for i in range(N_CHAN)]
        # 范围变化后旧预生成动作不再适用，释放它并恢复在线生成。
        self._sequence = None
        self._sequence_index = 0

    def _next_generated(self) -> List[float]:
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
                nxt = self._cur[i] + self.rng.uniform(-maxstep, maxstep)
                if nxt < self.lo[i]:
                    nxt = self.lo[i] + (self.lo[i] - nxt)      # 反射
                if nxt > self.hi[i]:
                    nxt = self.hi[i] - (nxt - self.hi[i])
                nxt = max(self.lo[i], min(self.hi[i], nxt))    # 反射后再钳位
            self._cur[i] = nxt
            out.append(nxt)
        return out

    def next_action(self) -> List[float]:
        if self._sequence is not None:
            start = self._sequence_index * N_CHAN
            if start < len(self._sequence):
                self._sequence_index += 1
                return [float(v) for v in self._sequence[start:start + N_CHAN]]
            self._sequence = None
            self._sequence_index = 0
        return self._next_generated()


class ReplayDriver:
    """按 actions6.csv 的相对时间顺序回放六维动作。"""

    def __init__(self, path: str):
        self.path = path
        self.times, self.actions = load_action_sequence(path)
        self.index = 0

    def next_action(self):
        if self.index >= len(self.actions):
            return None
        action = self.actions[self.index]
        self.index += 1
        return list(action)

    def next_delay(self, default_s=0.2):
        if self.index == 0:
            return max(0.02, float(default_s))
        if self.index >= len(self.times):
            return None
        return max(0.02, self.times[self.index] - self.times[self.index - 1])
