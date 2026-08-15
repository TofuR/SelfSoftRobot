"""计划的通用 CSV 导入导出。"""

from __future__ import annotations

import csv
from pathlib import Path

from .models import ActionPlan


def write_actions6_csv(plan: ActionPlan, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["t_sec", "c0", "c1", "c2", "c3", "c4", "c5"])
        for step, action in enumerate(plan.actions6):
            writer.writerow([step * plan.step_interval_s, *action])
