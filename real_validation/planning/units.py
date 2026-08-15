"""kPa ↔ 模型动作单位的唯一换算(修 B1/B2)。

数据链:actions6.csv 原始 kPa → /action_scale_kpa → 训练域 [0,1] → /action_norm_factor → 模型输入。
反变换:模型输出 → ×action_norm_factor → ×action_scale_kpa → kPa。

action_scale_kpa 来自 meta.json 的 hi6[ch](操作上限,经 masks_to_transition_npz.action_max_per_channel
的 fallback 逻辑);action_norm_factor 是 checkpoint buffer(npz 已归一到 [0,1] 后训练时的二次归一化,
对本数据 ≈ 1.0,no-op)。

这个换算**只允许出现在两处**:hardware/valve.py(硬件边界)与 openloop_planner(优化边界)。
其余任何地方禁止手写 kPa ↔ 模型单位。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _as_numpy(value) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def kPa_to_model(actions_kpa, *, action_scale_kpa, action_norm_factor):
    """kPa → 模型输入。actions_kpa: (..., A) kPa;返回同形状(训练域 [0,1] 再 /norm)。

    支持 numpy 与 torch(张量时返回同 device 的张量,保梯度)。
    """
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    norm = float(action_norm_factor)
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError(f"action_norm_factor 必须为正有限值,收到 {norm}")
    if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
        raise ValueError(f"action_scale_kpa 必须全为正有限值,收到 {scale}")
    if torch is not None and isinstance(actions_kpa, torch.Tensor):
        scale_t = torch.as_tensor(scale, dtype=actions_kpa.dtype, device=actions_kpa.device)
        return actions_kpa / scale_t / norm
    values = np.asarray(actions_kpa, dtype=np.float64)
    return values / scale / norm


def model_to_kPa(actions_model, *, action_scale_kpa, action_norm_factor):
    """模型输出 → kPa(逆变换,仅报告/展示用;优化边界不调用)。"""
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    norm = float(action_norm_factor)
    if torch is not None and isinstance(actions_model, torch.Tensor):
        scale_t = torch.as_tensor(scale, dtype=actions_model.dtype, device=actions_model.device)
        return actions_model * norm * scale_t
    values = np.asarray(actions_model, dtype=np.float64)
    return values * norm * scale


def check_unit_consistency(action_scale_kpa, action_norm_factor, *, hi6=None) -> str:
    """判定归一化链路是否一致,返回诊断字符串(不 raise)。

    - 若 norm_factor ∈ (0.9, 1.1) → npz 侧已归一到 [0,1],链路是 /scale /norm(正确)。
    - 若 hi6 提供且 norm_factor ≈ max(hi6) → 旧式未归一化数据,链路应只 /norm(数据侧没 /scale)。
    两种链路不可混用;返回值供 preflight 记录与人工核对。
    """
    norm = float(action_norm_factor)
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    if 0.9 <= norm <= 1.1:
        return f"npz 已归一化:kPa→/action_scale_kpa={scale}→/norm={norm}→模型"
    if hi6 is not None and norm > 0 and abs(norm - float(np.max(hi6))) / norm < 0.1:
        return f"旧式未归一化:kPa→/norm={norm}(≈hi6) — 与 /action_scale_kpa 链路不可混用"
    return f"norm_factor={norm} 非典型,请人工核对训练侧归一化"
