"""phase_strategy.py — 根据 TrainingSpec 管理模型训练阶段切换。"""

import torch
from src.training.spec import PhaseSpec, TrainingSpec


class PhaseStrategy:
    """管理模型的多阶段训练：冻结/解冻子模块、选择 forward 方法。"""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.spec: TrainingSpec = model.training_spec
        self.current_phase_idx = 0

    @property
    def current_phase(self) -> PhaseSpec:
        return self.spec.phases[self.current_phase_idx]

    @property
    def is_last_phase(self) -> bool:
        return self.current_phase_idx == len(self.spec.phases) - 1

    def enter_phase(self, phase_idx: int):
        """切换到指定阶段：解冻全部 → 冻结指定子模块。"""
        self.current_phase_idx = phase_idx
        for p in self.model.parameters():
            p.requires_grad = True
        for mod_name in self.current_phase.freeze_modules:
            module = getattr(self.model, mod_name)
            for p in module.parameters():
                p.requires_grad = False

    def get_forward_fn(self):
        """返回当前阶段应使用的 forward 函数。"""
        return getattr(self.model, self.current_phase.forward_attr)

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        """返回当前阶段可训练的参数。"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def iterate_phases(self):
        """迭代所有阶段，每个 yield (phase_idx, PhaseSpec)，自动切换。"""
        for i, phase in enumerate(self.spec.phases):
            self.enter_phase(i)
            yield i, phase
