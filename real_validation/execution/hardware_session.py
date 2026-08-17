"""验证工作台与现有 Qt 硬件对象之间的线程安全桥接。"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot

from .executor import CommandReceipt


@dataclass
class _PendingCommand:
    requested6: tuple[float, ...]
    required_groups: tuple[int, ...]
    event: threading.Event = field(default_factory=threading.Event)
    applied6: tuple[float, ...] | None = None
    t_command: float | None = None
    acknowledgements: dict[int, tuple[bool, float, str]] = field(default_factory=dict)
    issued: bool = False


class QtValveTransport(QObject):
    """把 worker 中的阻塞 ``send`` 安全转发到 controller 的 Qt 线程。

    本对象必须在 controller 所属线程创建。计划命令永不使用 ``bypass_rate``；只有
    ``zero`` 使用 controller 已有的安全归零语义。
    """

    _request = pyqtSignal(str, object, object, bool)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        if controller.thread() != self.thread():
            raise RuntimeError("QtValveTransport 必须与 ValveController 位于同一 Qt 线程")
        self._pending: dict[str, _PendingCommand] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._request.connect(self._issue_on_qt_thread, Qt.QueuedConnection)
        controller.communication_result.connect(self._on_communication_result)

    def send(self, action6: Sequence[float], required_groups: Sequence[int],
             timeout_s: float) -> CommandReceipt:
        return self._send(action6, tuple(int(g) for g in required_groups), timeout_s,
                          bypass_rate=False)

    def zero(self, timeout_s: float) -> CommandReceipt:
        return self._send((0.0,) * 6, None, timeout_s, bypass_rate=True)

    def _send(self, action6, required_groups, timeout_s, bypass_rate):
        if QThread.currentThread() == self.thread():
            raise RuntimeError("阻塞 send/zero 必须从 worker 调用，不能阻塞硬件 Qt 线程")
        if self._closed:
            raise RuntimeError("Valve transport 已关闭")
        requested = tuple(float(value) for value in action6)
        if len(requested) != 6:
            raise ValueError("阀命令必须为六通道")
        command_id = uuid.uuid4().hex
        groups = tuple(required_groups or ())
        pending = _PendingCommand(requested, groups)
        with self._lock:
            self._pending[command_id] = pending
        self._request.emit(command_id, list(requested),
                           None if required_groups is None else list(groups), bypass_rate)
        completed = pending.event.wait(float(timeout_s))
        with self._lock:
            self._pending.pop(command_id, None)
        if not completed:
            return CommandReceipt(command_id, requested,
                                  pending.applied6 or requested,
                                  pending.t_command or time.monotonic(), None, "timeout")
        failures = [value for value in pending.acknowledgements.values() if not value[0]]
        status = failures[0][2] if failures else "ack"
        ack_times = [value[1] for value in pending.acknowledgements.values() if value[0]]
        return CommandReceipt(command_id, requested, pending.applied6 or requested,
                              pending.t_command or time.monotonic(),
                              max(ack_times) if ack_times else None, status)

    @pyqtSlot(str, object, object, bool)
    def _issue_on_qt_thread(self, command_id, action6, groups, bypass_rate):
        with self._lock:
            pending = self._pending.get(command_id)
        if pending is None:
            return  # worker 已 timeout，禁止迟到命令继续下发
        required = (tuple(int(g) for g in groups) if groups is not None
                    else tuple(sorted(self.controller.connected_groups)))
        if not required:
            pending.acknowledgements[0] = (False, time.monotonic(), "not_connected")
            pending.event.set()
            return
        pending.required_groups = required
        try:
            _, applied, t_command = self.controller.set_pressures(
                action6, command_id=command_id, bypass_rate=bool(bypass_rate),
                required_groups=required)
            pending.applied6 = tuple(float(value) for value in applied)
            pending.t_command = float(t_command)
            pending.issued = True
            if (any(not value[0] for value in pending.acknowledgements.values()) or
                    all(group in pending.acknowledgements for group in required)):
                pending.event.set()
        except Exception as error:
            pending.acknowledgements[0] = (
                False, time.monotonic(), f"controller_error:{type(error).__name__}")
            pending.event.set()

    @pyqtSlot(str, int, bool, float, str)
    def _on_communication_result(self, command_id, group_id, ok, timestamp, status):
        with self._lock:
            pending = self._pending.get(str(command_id))
        if pending is None or int(group_id) not in pending.required_groups:
            return
        pending.acknowledgements[int(group_id)] = (
            bool(ok), float(timestamp), str(status))
        if pending.issued and (not ok or all(group in pending.acknowledgements
                                             for group in pending.required_groups)):
            pending.event.set()

    def close(self) -> None:
        self._closed = True
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for command in pending:
            command.acknowledgements[0] = (
                False, time.monotonic(), "transport_closed")
            command.event.set()
        try:
            self.controller.communication_result.disconnect(
                self._on_communication_result)
        except (TypeError, RuntimeError):
            pass
