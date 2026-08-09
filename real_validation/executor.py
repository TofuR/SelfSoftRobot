"""ACK 感知的动作计划执行器。

执行器依赖小型 transport 协议，不依赖 Qt。真阀适配器可在硬件线程中实现同一协议；
当前 ``MockCommandTransport`` 用于 Phase 0/1 全链路及错误注入。
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .models import ActionPlan, SafetyPolicy


@dataclass(frozen=True)
class CommandReceipt:
    command_id: str
    requested6: tuple[float, ...]
    applied6: tuple[float, ...]
    t_command: float
    t_ack: float | None
    status: str
    t_expected: float | None = None   # ★P4:期望下发时刻(execute 的绝对时基 deadline)

    @property
    def jitter_s(self) -> float | None:
        """实际下发时刻相对期望的偏差(负=提前,正=滞后)。ACK 等待超时会滞后。"""
        if self.t_expected is None:
            return None
        return self.t_command - self.t_expected


class CommandTransport(Protocol):
    def send(self, action6: Sequence[float], required_groups: Sequence[int],
             timeout_s: float) -> CommandReceipt: ...

    def zero(self, timeout_s: float) -> CommandReceipt: ...


class MockCommandTransport:
    def __init__(self, fail_at: int | None = None, status: str = "timeout",
                 latency_s: float = 0.0, send_delay_s: float = 0.0,
                 zero_always_fails: bool = False):
        self.fail_at = fail_at
        self.failure_status = status
        self.latency_s = latency_s          # ACK 延迟(t_ack = now + latency)
        self.send_delay_s = send_delay_s    # 下发本身阻塞(t_command 滞后 → jitter)
        self.zero_always_fails = zero_always_fails   # 模拟归零也失败 → zero_with_retry 测试
        self.commands: list[tuple[float, ...]] = []
        self._counter = 0

    def send(self, action6: Sequence[float], required_groups: Sequence[int],
             timeout_s: float) -> CommandReceipt:
        del required_groups, timeout_s
        if self.send_delay_s:
            time.sleep(self.send_delay_s)   # 模拟下发阻塞 → 后续命令错过 deadline
        self._counter += 1
        action = tuple(float(value) for value in action6)
        self.commands.append(action)
        now = time.monotonic()
        failed = self.fail_at == self._counter
        return CommandReceipt(str(self._counter), action, action, now,
                              None if failed else now + self.latency_s,
                              self.failure_status if failed else "ack")

    def zero(self, timeout_s: float) -> CommandReceipt:
        del timeout_s
        if self.zero_always_fails:
            now = time.monotonic()
            return CommandReceipt("zero", (0.0,) * 6, (0.0,) * 6, now, None, "timeout")
        return self.send((0.0,) * 6, (), 0.0)


class ExecutionError(RuntimeError):
    pass


class PlanExecutor:
    def __init__(self, transport: CommandTransport, safety: SafetyPolicy,
                 event_callback: Callable[[str, dict], None] | None = None):
        self.transport = transport
        self.safety = safety
        self.event_callback = event_callback
        self._abort = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self.receipts: list[CommandReceipt] = []

    def pause(self) -> None:
        self._resume.clear()
        if self.safety.pause_policy == "zero":
            receipt = self.transport.zero(self.safety.ack_timeout_s)
            self.receipts.append(receipt)
            self._emit("paused_zeroed", {"receipt": asdict(receipt)})
            # 归零改变了后续动作的真实初态，原计划的 slew preflight 不再成立。
            # 因此 zero-pause 是安全终止；必须重新锚定/规划后才能继续。
            self._abort.set()
            self._resume.set()
        else:
            self._emit("paused_hold", {})

    def resume(self) -> None:
        if self.safety.pause_policy == "zero" and self._abort.is_set():
            raise ExecutionError("zero-pause 后必须重新规划，不能恢复旧计划")
        self._resume.set()
        self._emit("resumed", {})

    def abort(self) -> None:
        self._abort.set()
        self._resume.set()

    def _zero_with_retry(self, retries: int = 3) -> CommandReceipt:
        """归零失败重试 N 次;全败保持 ERROR(不能静默放过)。"""
        last = None
        for _ in range(max(1, retries)):
            last = self.transport.zero(self.safety.ack_timeout_s)
            if last.status == "ack":
                return last
        raise ExecutionError(
            f"归零失败({retries} 次重试均未 ACK,末次 {last.status}):请人工介入/急停")

    def execute(self, plan: ActionPlan, output_csv: str | Path | None = None) -> list[CommandReceipt]:
        self._abort.clear()
        self._resume.set()
        self.receipts = []
        started = time.monotonic()
        try:
            for step, action in enumerate(plan.actions6):
                self._wait_until_resumed()
                if self._abort.is_set():
                    raise ExecutionError("operator_abort")
                deadline = started + step * plan.step_interval_s
                if not self._wait_until(deadline):
                    raise ExecutionError("operator_abort")
                receipt = self.transport.send(action, self.safety.required_groups,
                                              self.safety.ack_timeout_s)
                # 记录期望下发时刻(绝对时基)→ jitter 可归因
                receipt = replace(receipt, t_expected=deadline)
                self.receipts.append(receipt)
                self._emit("command", {"step": step, "receipt": asdict(receipt)})
                if receipt.status != "ack":
                    raise ExecutionError(f"command {receipt.command_id}: {receipt.status}")
            self._emit("completed", {"steps": len(plan.actions6)})
            return list(self.receipts)
        except Exception as error:
            zero_receipt = self._zero_with_retry()
            self.receipts.append(zero_receipt)
            self._emit("aborted_zeroed", {"error": str(error),
                                           "receipt": asdict(zero_receipt)})
            raise
        finally:
            if output_csv is not None:
                self.write_receipts(output_csv)

    def _wait_until_resumed(self) -> None:
        while not self._resume.wait(0.05):
            if self._abort.is_set():
                return

    def _wait_until(self, deadline: float) -> bool:
        while True:
            if self._abort.is_set():
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            self._abort.wait(min(remaining, 0.05))

    def _emit(self, event: str, payload: dict) -> None:
        if self.event_callback:
            self.event_callback(event, payload)

    def write_receipts(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["command_id", "t_command", "t_expected", "jitter_s",
                             "t_ack", "status",
                             *[f"requested_c{i}" for i in range(6)],
                             *[f"applied_c{i}" for i in range(6)]])
            for item in self.receipts:
                writer.writerow([
                    item.command_id, item.t_command,
                    "" if item.t_expected is None else item.t_expected,
                    "" if item.jitter_s is None else item.jitter_s,
                    "" if item.t_ack is None else item.t_ack, item.status,
                    *item.requested6, *item.applied6])
