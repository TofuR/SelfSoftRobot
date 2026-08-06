"""障碍几何解析 + 2D SDF 惩罚(CLI 与 GUI 共用的唯一实现,修 B4)。

口径:mean-over-(K,N)。CLI inverse_plan.py 原实现对 k 求和、对 N 求均值,与工作台的
(K,N) 全均值差 ≈K 倍。统一为 mean 的判据:auto_k 会让 K 随 gap 变化,sum-over-K 使
同一 w_obs 的避障压强随 K 线性漂移,与 auto_k 直接冲突。

坐标:preds 为 (K,N,3) 归一化,obstacles 为 model 坐标(px [col,row])。归一化 → px 用
pc_center/pc_scale;只取 [:2](col,row 平面)。pc_scale[2]=1e-6,z 通道污染会被放大 1e6,
故严禁把 z 计入。
"""

from __future__ import annotations

import math

import torch


def parse_obstacle_circles(scene):
    """从 Scene 的 obstacle_circle 原语解析 → [(cx, cy, r_px)];含 safety_margin。

    只接受 frame_id == "model" 的圆;其他障碍类型(AABB/多边形)由调用方另作解析。
    """
    supported = []
    for item in scene.primitives:
        if not item.kind.startswith("obstacle_"):
            continue
        if item.kind != "obstacle_circle":
            raise ValueError(f"当前 obstacle 解析尚不支持 {item.kind}")
        if item.frame_id != "model":
            raise ValueError("圆障碍必须先转换到 model 坐标")
        center = item.geometry.get("center", item.geometry.get("xy"))
        radius = float(item.geometry.get("radius", item.geometry.get("r", 0.0)))
        if not isinstance(center, (list, tuple)) or len(center) != 2 or radius <= 0:
            raise ValueError("obstacle_circle 需要 center=[x,y] 与正 radius")
        supported.append((float(center[0]), float(center[1]),
                          radius + float(item.safety_margin)))
    return supported


def obstacle_term(preds, pc_center, pc_scale, obstacles, reduce: str = "mean"):
    """障碍惩罚:preds (K,N,3) 归一化 → px,对每个 keep-out 圆罚穿透。标量。

    obstacles: [(cx, cy, r_px), ...] in model px。聚合 = mean-over-(K,N)(对 K 不变)。
    """
    if not obstacles:
        return preds.new_zeros(())
    physical = preds * pc_scale + pc_center          # (K,N,3) px
    total = preds.new_zeros(())
    for (cx, cy, radius) in obstacles:
        distance = torch.linalg.vector_norm(
            physical[:, :, :2] - physical.new_tensor((cx, cy)), dim=2)
        total = total + torch.relu(radius - distance).square().mean()
    if reduce == "mean":
        return total / preds.shape[0]
    if reduce == "sum":
        return total
    raise ValueError(f"未知 reduce: {reduce}")


def obstacle_term_ext(preds, pc_center, pc_scale, obstacles, reduce: str = "mean"):
    """扩展版:支持 circle 与 aabb。obstacles = [("circle",(cx,cy),r) | ("aabb",(x0,y0,x1,y1),0)]。

    只在需要 AABB 的场景用;纯 circle 用 obstacle_term 更快更简单。
    """
    if not obstacles:
        return preds.new_zeros(())
    physical = preds * pc_scale + pc_center
    total = preds.new_zeros(())
    for kind, geom, radius in obstacles:
        xy = physical[:, :, :2]
        if kind == "circle":
            distance = torch.linalg.vector_norm(
                xy - xy.new_tensor((geom[0], geom[1])), dim=2)
            total = total + torch.relu(radius - distance).square().mean()
        elif kind == "aabb":
            x0, y0, x1, y1 = geom
            qx = torch.abs(xy[..., 0] - xy.new_tensor((x0 + x1) / 2)) - xy.new_tensor((x1 - x0) / 2)
            qy = torch.abs(xy[..., 1] - xy.new_tensor((y0 + y1) / 2)) - xy.new_tensor((y1 - y0) / 2)
            outside = torch.sqrt(torch.relu(qx).square() + torch.relu(qy).square())
            inside = torch.minimum(torch.maximum(qx, qy), xy.new_zeros(()))
            sdf = outside + inside                     # 盒外正 / 盒内负
            total = total + torch.relu(-sdf).square().mean()
        else:
            raise ValueError(f"未知障碍类型: {kind}")
    if reduce == "mean":
        return total / preds.shape[0]
    if reduce == "sum":
        return total
    raise ValueError(f"未知 reduce: {reduce}")


def clearance_min(states_px, obstacles) -> float | None:
    """states_px (K,N,3) 或 (N,3) px → 到所有障碍的最小净距(distance - r,可为负)。

    供 preflight 碰撞门用。无障碍返回 None。
    """
    if not obstacles:
        return None
    values = []
    for (cx, cy, radius) in obstacles:
        xy = torch.as_tensor(states_px, dtype=torch.float64)[..., :2]
        distance = torch.linalg.vector_norm(xy - torch.as_tensor((cx, cy), dtype=torch.float64),
                                            dim=-1)
        values.append(float((distance - radius).min()))
    return min(values)


def _clearance_min_numpy(states_px, obstacles) -> float | None:
    """numpy 版(CLI 报告用,no_grad 场景)。states_px (K,N,3) 或 (N,3)。"""
    if not obstacles:
        return None
    import numpy as np
    xy = np.asarray(states_px, dtype=np.float64)[..., :2]
    values = []
    for (cx, cy, radius) in obstacles:
        distance = np.linalg.norm(xy - np.asarray((cx, cy), dtype=np.float64), axis=-1)
        values.append(float((distance - radius).min()))
    return min(values)


# ---------------- CLI 兼容层(inverse_plan.py 委托到共享核的落点) ----------------
def cli_obstacle_loss(preds_norm, pc_center, pc_scale, obs_list):
    """与 CLI inverse_plan.obstacle_loss 签名一致,但聚合改为 mean-over-(K,N)。

    obs_list: [(cx, cy, r_px)] in px。preds_norm (K,N,3) 归一化。
    注:聚合口径从"对 k 求和"改为"mean",同一 w_obs 的避障压强不再随 K 漂移(与 auto_k 兼容)。
    """
    return obstacle_term(preds_norm, pc_center, pc_scale, obs_list, reduce="mean")
