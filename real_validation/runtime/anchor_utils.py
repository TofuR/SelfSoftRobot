"""Anchor 构建共享 helper:归一化参数读取 + 骨架归一化(offline/live 共用)。"""

from __future__ import annotations

import numpy as np


def model_normalization(model) -> tuple[np.ndarray, np.ndarray]:
    """pc_center / pc_scale → (3,) numpy;缺字段抛 ValueError(文案沿用)。"""
    try:
        center = model.pc_center.detach().cpu().numpy().reshape(3)
        scale = model.pc_scale.detach().cpu().numpy().reshape(3)
    except (AttributeError, ValueError) as error:
        raise ValueError("模型缺少可用 pc_center/pc_scale") from error
    return center, scale


def normalize_rows(nodes, center, scale, dims=slice(None)) -> np.ndarray:
    """(N,3|2) 像素 → 归一化。dims 限定维度(offline 全 3D;live 只 [:2])。"""
    c = np.asarray(center, dtype=np.float64)[dims]
    s = np.asarray(scale, dtype=np.float64)[dims]
    if np.any(np.abs(s) < 1e-12):
        raise ValueError("模型 pc_scale 含零")
    return (np.asarray(nodes, dtype=np.float64)[..., dims] - c) / s


def float_rows(nodes) -> tuple[tuple[float, ...], ...]:
    """numpy 行 → 嵌套 float 元组(JSON 安全,与 Anchor 契约一致)。"""
    return tuple(tuple(float(v) for v in node) for node in nodes)
