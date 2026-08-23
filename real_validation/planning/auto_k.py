"""变长 K 选择：据目标形态差距和训练数据实测位移表选规划步数。"""

from __future__ import annotations

import math

import numpy as np


def select_k_by_displacement(gap: float, displacement: dict[str, float],
                             k_min: int, k_max: int) -> int:
    """选取累积p95位移统计覆盖目标gap的最小K，超出表范围时使用K上限。"""
    if k_min > k_max:
        raise ValueError(f"k_min({k_min}) 不能大于 k_max({k_max})")
    values = sorted((int(k), float(v)) for k, v in displacement.items()
                    if k_min <= int(k) <= k_max and np.isfinite(v) and float(v) > 0)
    if not values:
        raise ValueError("部署合同在当前K范围内没有实测形态位移表")
    cumulative = 0.0
    for k, capacity in values:
        cumulative = max(cumulative, capacity)
        if cumulative >= float(gap):
            return k
    return k_max


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


# 公共行为与长度单位无关；保留旧函数名供已有脚本导入。
gap_point = gap_px_point
gap_skeleton = gap_px_skeleton
