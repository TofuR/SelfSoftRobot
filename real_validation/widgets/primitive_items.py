"""ScenePrimitive → Qt 图形的共享绘制(plan_preview 与 camera_view 共用)。

统一障碍色语义:target=红 #E53E3E,obstacle=琥珀 #B7791F。坐标均为 model 像素
(col=x, row=y)。
"""

from __future__ import annotations

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsRectItem

from ..models import ScenePrimitive

TARGET_COLOR = "#E53E3E"
OBSTACLE_COLOR = "#B7791F"


def scene_primitive_item(p: ScenePrimitive, *,
                         target_color: str = TARGET_COLOR,
                         obstacle_color: str = OBSTACLE_COLOR):
    """ScenePrimitive → QGraphicsItem;不支持/缺几何返回 None。"""
    xy = p.geometry.get("xy", p.geometry.get("center"))
    if p.kind == "target_point" and xy:
        return pg.ScatterPlotItem([xy[0]], [xy[1]], symbol="x", size=16,
                                  pen=pg.mkPen(target_color, width=3))
    if p.kind in {"target_circle", "obstacle_circle"} and xy:
        radius = float(p.geometry.get("radius", p.geometry.get("r", 0.0)))
        if p.kind == "obstacle_circle":
            radius += float(p.safety_margin)
        item = QGraphicsEllipseItem(xy[0] - radius, xy[1] - radius, 2 * radius, 2 * radius)
        item.setPen(QPen(Qt.red if p.kind == "target_circle" else obstacle_color,
                         2, Qt.DashLine))
        return item
    if p.kind == "obstacle_aabb":
        lo, hi = p.geometry.get("min"), p.geometry.get("max")
        if not lo or not hi:
            return None
        margin = float(p.safety_margin)
        item = QGraphicsRectItem(lo[0] - margin, lo[1] - margin,
                                 (hi[0] - lo[0]) + 2 * margin, (hi[1] - lo[1]) + 2 * margin)
        item.setPen(QPen(obstacle_color, 2, Qt.DashLine))
        return item
    if p.kind == "target_skeleton":
        nodes = p.geometry.get("nodes")
        if not nodes:
            return None
        return pg.ScatterPlotItem([n[0] for n in nodes], [n[1] for n in nodes],
                                  symbol="x", size=8, pen=pg.mkPen(target_color, width=2))
    return None
