"""实机模型验证工作台。

核心模块不依赖 Qt；GUI、CLI 和测试共用同一套 session、preflight 与 executor。

⚠️ **本文件的 import 闭包必须保持 stdlib-only**。
   src/ 与离线数据准备脚本反向依赖本包（src/utils/skeleton_2d.py 与
   src/data/real/segmentation.py 是 real_validation.perception 的薄壳），
   一旦这里 import 了 torch / PyQt5 / cv2 / scipy，仿真训练与 npz 准备就会被迫
   拉入部署侧依赖。新增重导出前先跑 tests/test_import_hygiene.py。
"""

from .models import (
    ActionPlan,
    Anchor,
    ModelDescriptor,
    SafetyPolicy,
    Scene,
    ScenePrimitive,
)
from .session import ExperimentSession, SessionState

__all__ = [
    "ActionPlan",
    "Anchor",
    "ExperimentSession",
    "ModelDescriptor",
    "SafetyPolicy",
    "Scene",
    "ScenePrimitive",
    "SessionState",
]
