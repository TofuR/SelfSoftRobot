"""实机验证工作台 GUI 入口。

三种启动方式都支持:
  - 在 real_validation/ 目录内:  python main.py
  - 在父目录:                     python -m real_validation.main
  - Windows 双击 run_gui.bat(内部已 cd 到父目录)
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):   # 直接 `python main.py`(在包目录内)→ 把父目录加进 sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "real_validation"

from .gui.main_window import main

if __name__ == "__main__":
    raise SystemExit(main())
