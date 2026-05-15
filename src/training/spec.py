"""spec.py — 模型训练需求声明。

模型通过类属性 training_spec 声明自己需要几个训练阶段、每个阶段冻结什么、
用什么 forward 方法、启用哪些 loss。
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PhaseSpec:
    """单个训练阶段配置。"""
    name: str
    freeze_modules: list[str] = field(default_factory=list)
    forward_attr: str = "forward"
    data_mode: str = "sequence"  # "canonical" | "sequence"
    lr: Optional[float] = None
    active_losses: list[str] = field(default_factory=lambda: ["recon", "smooth"])


@dataclass
class TrainingSpec:
    """模型的训练需求声明。"""
    phases: list[PhaseSpec]
    supports_smoothness: bool = True

    @property
    def is_two_phase(self) -> bool:
        return len(self.phases) > 1

    @property
    def needs_canonical_data(self) -> bool:
        return any(p.data_mode == "canonical" for p in self.phases)
