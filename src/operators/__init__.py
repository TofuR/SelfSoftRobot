"""src/operators — 迟滞算子库（HereditaryOperatorModel 的物理构件层）。

四个独立构件，各自可单独复用与测试：

  - MonotoneSplineDrive: 逐通道单调样条驱动 e_c(a_c)（记忆无关的伪应变代理）
  - PlayBank:            Prandtl–Ishlinskii play 算子组（率无关 / 摩擦型迟滞）
  - MaxwellBank:         广义 Maxwell 元件组（率相关 / 粘弹型迟滞，精确 ZOH 离散）
  - LocalFrameModeBank:  局部标架模态位移场（平面切向/法向）

设计文档: docs/designs/2026-08-29_hereditary_operator_model.md（§2.2 Version B）
"""

from .static_drive import MonotoneSplineDrive
from .play_bank import PlayBank
from .maxwell_bank import MaxwellBank
from .local_modes import LocalFrameModeBank

__all__ = [
    "MonotoneSplineDrive",
    "PlayBank",
    "MaxwellBank",
    "LocalFrameModeBank",
]
