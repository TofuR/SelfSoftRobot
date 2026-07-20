"""线程安全的验证事件与结果记录。"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .io import atomic_write_json


class ValidationRecorder:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.run_dir / "events.jsonl"
        self._lock = threading.Lock()

    def event(self, name: str, payload: dict | None = None) -> None:
        row = {"t_monotonic": time.monotonic(), "event": name, **(payload or {})}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
        with self._lock, self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()

    def write_metrics(self, metrics: dict) -> None:
        atomic_write_json(self.run_dir / "metrics.json", metrics)
