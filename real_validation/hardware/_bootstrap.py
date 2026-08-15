"""把兄弟目录 real_capture/ 加入 sys.path,以便 import 其硬件驱动。

部署方式:real_capture/ 与 real_validation/ 并排拷到 PC(设计 spec §3.1)。
本模块只在**真机接线**时被 import(Setup 页连接硬件),Mock 流程不碰。

用 append 不用 insert,避免遮蔽 stdlib 模块名。real_capture 与 real_validation
当前无同名模块(recorder vs validation_recorder),新增文件时须检查。
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_real_capture_importable() -> Path:
    """返回 real_capture 目录并把它加入 sys.path(幂等)。"""
    here = Path(__file__).resolve().parent          # .../real_validation/hardware
    sibling = here.parent.parent / "real_capture"  # repo 根 / real_capture
    candidate = sibling if sibling.is_dir() else here.parent.parent.parent / "real_capture"
    target = candidate.resolve()
    if target.is_dir() and str(target) not in sys.path:
        sys.path.append(str(target))                # 只 append,不 insert
    return target
