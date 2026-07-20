"""实机模型验证工作台。

核心模块不依赖 Qt；GUI、CLI 和测试共用同一套 session、preflight 与 executor。
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
