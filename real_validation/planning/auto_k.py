"""变长 K 选择:据首末差距选规划步数。

CLI inverse_plan.py 的 --auto_k 移植。关键修正(B17):step_budget_px 必须从
**学到的 delta_scale** 现算,不能基于 delta_scale_max=1.0 —— 前向真正的系数是
clamp(delta_scale, max=delta_scale_max),而 delta_scale 是可学参数(初值 0.1,
存在 checkpoint 里)。基于 1.0 会高估 ~10× → K 选小 10× → 步数不够到不了目标。
"""

from __future__ import annotations

import math

import numpy as np
import torch


def torch_clamp_delta(model) -> float:
    return float(torch.clamp(model.delta_scale, max=model.delta_scale_max).item())


def step_budget_px(model) -> float:
    """模型单步最大末端位移(px) = clamp(delta_scale, max=delta_scale_max) × pc_scale。"""
    scale = torch_clamp_delta(model)
    pc = model.pc_scale.detach().cpu().numpy().reshape(3)
    return scale * float(np.abs(pc[:2]).max())


def select_k_by_gap(gap_tip_px: float, step_budget_px: float,
                    k_min: int, k_max: int) -> int:
    """K = clamp(ceil(gap / step_budget), k_min, k_max)。"""
    if k_min > k_max:
        raise ValueError(f"k_min({k_min}) 不能大于 k_max({k_max})")
    k = int(math.ceil(gap_tip_px / max(step_budget_px, 1e-6)))
    return max(k_min, min(k_max, k))


def gap_px_point(tip_px, target_xy, radius: float = 0.0) -> float:
    """单节点目标:到圆边界的距离(圆内 → 0,无需额外行程)。"""
    distance = math.hypot(float(tip_px[0]) - float(target_xy[0]),
                          float(tip_px[1]) - float(target_xy[1]))
    return max(0.0, distance - radius)


def gap_px_skeleton(now_px, goal_px, tolerance: float = 0.0) -> float:
    """整形态目标:瓶颈是走得最远那个节点 → 取 max,不是 node0 也不是 mean。"""
    now = np.asarray(now_px, dtype=np.float64)
    goal = np.asarray(goal_px, dtype=np.float64)
    per_node = np.linalg.norm(now[:, :2] - goal[:, :2], axis=1)
    return max(0.0, float(per_node.max()) - tolerance)
