"""实时相机视图:图像层 + 骨架层 + 场景原语层 + 锚点层 + 鼠标点选。

坐标约定:col = x, row = y, 图像顶部 = row 0 → invertY(False)。
(与 plan_preview 的 model 坐标相反 —— 那是列 [col,row] 但 invertY(True);图像层直接像素。)

鼠标:view.scene().sigMouseClicked 的 scenePos 经 getViewBox().mapSceneToView() 换算到图像像素。
tool 模式:select(点选原语)/ add_target(点击加目标点)/ add_obstacle(点击加圆障碍)/ move(拖拽)。
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import (QGraphicsEllipseItem, QGraphicsRectItem, QLabel,
                             QVBoxLayout, QWidget)

from ..models import Scene, ScenePrimitive

_SKELETON_COLOR = "#2CB1BC"
_TARGET_COLOR = "#E53E3E"
_OBSTACLE_COLOR = "#B7791F"


class CameraViewWidget(QWidget):
    sig_image_clicked = pyqtSignal(int, int)      # (col, row) 图像像素
    target_picked = pyqtSignal(object)            # ScenePrimitive(target_*)
    obstacle_picked = pyqtSignal(object)          # ScenePrimitive(obstacle_*)
    selection_changed = pyqtSignal(str)           # primitive_id
    geometry_edited = pyqtSignal(object)          # 编辑后的 Scene

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.invertY(False)                  # 图像顶部 = row 0
        self.plot.setAspectLocked(True)
        root.addWidget(self.plot, 1)
        self._hint = QLabel("工具: select(点选) / add_target(点加目标) / add_obstacle(点加圆障碍)")
        root.addWidget(self._hint)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self.skeleton_curve = self.plot.plot([], [], pen=pg.mkPen(_SKELETON_COLOR, width=2),
                                             symbol="o", symbolSize=5, symbolBrush=_SKELETON_COLOR)
        self.anchor_scatter = pg.ScatterPlotItem(symbol="t", size=14, pen=pg.mkPen("#38A169", width=2))
        self.plot.addItem(self.anchor_scatter)

        self.tool = "select"
        self._scene: Scene | None = None
        self._scene_items: list[tuple[str, object]] = []   # [(primitive_id, item)]
        self.plot.scene().sigMouseClicked.connect(self._on_click)

    # ---- 图层更新 ----
    def set_frame(self, bgr) -> None:
        """显示一帧 BGR;并锁定 view 到图像范围。"""
        bgr = np.asarray(bgr)
        rgb = bgr[..., ::-1] if bgr.ndim == 3 and bgr.shape[2] == 3 else bgr
        self.image_item.setImage(rgb)
        self.plot.setXRange(0, rgb.shape[1])
        self.plot.setYRange(0, rgb.shape[0])

    def set_skeleton(self, skeleton) -> None:
        sk = np.asarray(skeleton, dtype=np.float64)
        if sk.size and sk.shape[1] >= 2:
            self.skeleton_curve.setData(sk[:, 0], sk[:, 1])
        else:
            self.skeleton_curve.setData([], [])

    def set_anchor(self, skeleton) -> None:
        sk = np.asarray(skeleton, dtype=np.float64)
        if sk.size and sk.shape[1] >= 2:
            self.anchor_scatter.setData(x=sk[:, 0], y=sk[:, 1])
        else:
            self.anchor_scatter.setData(x=[], y=[])

    def set_tool(self, tool: str) -> None:
        if tool not in {"select", "add_target", "add_obstacle"}:
            raise ValueError(f"未知工具: {tool}")
        self.tool = tool

    def set_scene(self, scene: Scene) -> None:
        """清旧原语、按 scene 重绘。"""
        for _pid, item in self._scene_items:
            self.plot.removeItem(item)
        self._scene_items = []
        self._scene = scene
        for primitive in scene.primitives:
            item = self._draw_primitive(primitive)
            if item is not None:
                item.setData(primitive_id=primitive.primitive_id)
                self.plot.addItem(item)
                self._scene_items.append((primitive.primitive_id, item))

    def _draw_primitive(self, p: ScenePrimitive):
        xy = p.geometry.get("xy", p.geometry.get("center"))
        if p.kind == "target_point" and xy:
            return pg.ScatterPlotItem([xy[0]], [xy[1]], symbol="x", size=16,
                                      pen=pg.mkPen(_TARGET_COLOR, width=3))
        if p.kind in {"target_circle", "obstacle_circle"} and xy:
            radius = float(p.geometry.get("radius", p.geometry.get("r", 0.0)))
            if p.kind == "obstacle_circle":
                radius += float(p.safety_margin)
            item = QGraphicsEllipseItem(xy[0] - radius, xy[1] - radius, 2 * radius, 2 * radius)
            item.setPen(QPen(Qt.red if p.kind == "target_circle" else _OBSTACLE_COLOR,
                             2, Qt.DashLine))
            return item
        if p.kind == "obstacle_aabb":
            lo = p.geometry.get("min"); hi = p.geometry.get("max")
            if not lo or not hi:
                return None
            margin = float(p.safety_margin)
            item = QGraphicsRectItem(lo[0] - margin, lo[1] - margin,
                                     (hi[0] - lo[0]) + 2 * margin, (hi[1] - lo[1]) + 2 * margin)
            item.setPen(QPen(_OBSTACLE_COLOR, 2, Qt.DashLine))
            return item
        if p.kind == "target_skeleton":
            nodes = p.geometry.get("nodes")
            if not nodes:
                return None
            xs = [n[0] for n in nodes]; ys = [n[1] for n in nodes]
            return pg.ScatterPlotItem(xs, ys, symbol="x", size=8, pen=pg.mkPen(_TARGET_COLOR, width=2))
        return None

    # ---- 鼠标 ----
    def _on_click(self, ev) -> None:
        if not self.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        view = self.plot.getViewBox()
        mapped = view.mapSceneToView(ev.scenePos())
        col, row = int(mapped.x()), int(mapped.y())
        self.sig_image_clicked.emit(col, row)
        if self.tool == "add_target":
            self.target_picked.emit(ScenePrimitive(
                "target_point", "model", {"xy": [col, row], "node": 0},
                name=f"target_{len(self._scene_items)}"))
        elif self.tool == "add_obstacle":
            self.obstacle_picked.emit(ScenePrimitive(
                "obstacle_circle", "model", {"center": [col, row], "radius": 10.0},
                name=f"obstacle_{len(self._scene_items)}"))
