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
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..contracts.models import Scene, ScenePrimitive
from .primitive_items import TARGET_COLOR, scene_primitive_item

_SKELETON_COLOR = "#2CB1BC"


class CameraViewWidget(QWidget):
    sig_image_clicked = pyqtSignal(int, int)      # (col, row) 图像像素
    target_picked = pyqtSignal(object)            # ScenePrimitive(target_*)
    obstacle_picked = pyqtSignal(object)          # ScenePrimitive(obstacle_*)
    target_skeleton_picked = pyqtSignal(object)   # ScenePrimitive(target_skeleton) 功能③
    skeleton_draft_changed = pyqtSignal(int)       # 当前未提交节点数
    selection_changed = pyqtSignal(str)           # primitive_id
    geometry_edited = pyqtSignal(object)          # 编辑后的 Scene

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.invertY(False)                  # 图像顶部 = row 0
        self.plot.setAspectLocked(True)
        root.addWidget(self.plot, 1)
        self._hint = QLabel("工具: select / 点加目标 / 点出目标骨架 / 点加障碍")
        root.addWidget(self._hint)

        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)
        self._skeleton_curve = self.plot.plot([], [], pen=pg.mkPen(_SKELETON_COLOR, width=2),
                                              symbol="o", symbolSize=5, symbolBrush=_SKELETON_COLOR)
        self.anchor_scatter = pg.ScatterPlotItem(symbol="t", size=14, pen=pg.mkPen("#38A169", width=2))
        self.plot.addItem(self.anchor_scatter)

        # 功能③:点出目标骨架 —— 已点节点预览(红 x,随点击累积)
        self._skeleton_preview = pg.ScatterPlotItem(symbol="x", size=12,
                                                    pen=pg.mkPen("#E53E3E", width=2))
        self.plot.addItem(self._skeleton_preview)
        self._skeleton_points: list[tuple[float, float]] = []

        self.tool = "select"
        self.read_only = False          # 只读模式(主显示区):禁用鼠标点选 + 隐藏工具提示
        self._scene: Scene | None = None
        self._scene_items: list[tuple[str, object]] = []   # [(primitive_id, item)]
        self.plot.scene().sigMouseClicked.connect(self._on_click)

        # ---- 主显示增强图层(规划预测轨迹 / 执行实际骨架 / NDI 末端) ----
        self._predicted_items: list[object] = []
        self._actual_scatter = pg.ScatterPlotItem(symbol="o", size=10,
                                                  pen=pg.mkPen("#EF4E4E", width=2),
                                                  brush=pg.mkBrush("#EF4E4E"))
        self.plot.addItem(self._actual_scatter)
        self._ndi_scatter = pg.ScatterPlotItem(symbol="star", size=14,
                                               pen=pg.mkPen("#805AD5", width=2))
        self.plot.addItem(self._ndi_scatter)
        # 图层可见性映射(_layer_items["predicted"] 与 self._predicted_items 指向同一
        # list 对象,set_predicted_states 用 .clear() 而非重绑定以保持引用)。
        self._layer_items: dict[str, list[object]] = {
            "skeleton": [self._skeleton_curve],
            "scene": [],            # set_scene 时登记
            "predicted": self._predicted_items,
            "actual": [self._actual_scatter],
            "ndi": [self._ndi_scatter],
        }

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
            self._skeleton_curve.setData(sk[:, 0], sk[:, 1])
        else:
            self._skeleton_curve.setData([], [])

    def set_anchor(self, skeleton) -> None:
        sk = np.asarray(skeleton, dtype=np.float64)
        if sk.size and sk.shape[1] >= 2:
            self.anchor_scatter.setData(x=sk[:, 0], y=sk[:, 1])
        else:
            self.anchor_scatter.setData(x=[], y=[])

    # ---- 主显示增强图层 ----
    def set_predicted_states(self, states) -> None:
        """规划预测轨迹:states(K,N,2) 图像像素,每条 K 画一条灰色虚线。"""
        for item in self._predicted_items:
            self.plot.removeItem(item)
        self._predicted_items.clear()   # 保持 _layer_items["predicted"] 引用同一 list
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 3 or states.shape[2] < 2:
            return
        for k in range(states.shape[0]):
            line = self.plot.plot(states[k, :, 0], states[k, :, 1],
                                  pen=pg.mkPen("#8B9BB4", width=1,
                                               style=Qt.DashLine))
            self._predicted_items.append(line)

    def set_actual_skeleton(self, skeleton) -> None:
        """执行实际骨架(N,2) 图像像素,红点叠加。"""
        sk = np.asarray(skeleton, dtype=np.float64)
        if sk.size and sk.shape[1] >= 2:
            self._actual_scatter.setData(x=sk[:, 0], y=sk[:, 1])
        else:
            self._actual_scatter.setData(x=[], y=[])

    def set_ndi_position(self, xy) -> None:
        """NDI 末端位置(图像像素),紫星;None 时清空。"""
        if xy is None:
            self._ndi_scatter.setData(x=[], y=[])
        else:
            self._ndi_scatter.setData(x=[float(xy[0])], y=[float(xy[1])])

    def set_layer_visible(self, layer: str, visible: bool) -> None:
        """图层可见性开关:layer ∈ {"skeleton","scene","predicted","actual","ndi"}。"""
        if layer not in self._layer_items:
            raise ValueError(f"未知图层: {layer}")
        for item in self._layer_items[layer]:
            item.setVisible(bool(visible))

    def set_tool(self, tool: str) -> None:
        if tool not in {"select", "add_target", "add_obstacle", "add_target_skeleton"}:
            raise ValueError(f"未知工具: {tool}")
        self.tool = tool
        if tool != "add_target_skeleton":
            self.clear_skeleton_points()   # 离开骨架工具清掉未提交的点

    def set_scene(self, scene: Scene) -> None:
        """清旧原语、按 scene 重绘。scene 为 None 时仅清空(图层开关测试用)。"""
        for _pid, item in self._scene_items:
            self.plot.removeItem(item)
        self._scene_items = []
        self._scene = scene
        if scene is None:
            self._layer_items["scene"] = []
            return
        for primitive in scene.primitives:
            item = self._draw_primitive(primitive)
            if item is not None:
                self.plot.addItem(item)
                self._scene_items.append((primitive.primitive_id, item))
        # 登记 scene 图层(每次重绘后刷新,供 set_layer_visible("scene") 开关)
        self._layer_items["scene"] = [item for _pid, item in self._scene_items]

    def _draw_primitive(self, p: ScenePrimitive):
        return scene_primitive_item(p, target_color=TARGET_COLOR)

    # ---- 点出目标骨架(功能③) ----
    def clear_skeleton_points(self) -> None:
        self._skeleton_points = []
        self._skeleton_preview.setData(x=[], y=[])
        self.skeleton_draft_changed.emit(0)

    def commit_skeleton_target(self) -> bool:
        """把已点节点提交为 target_skeleton(双击完成)。至少 2 点。"""
        if len(self._skeleton_points) < 2:
            return False
        nodes = [[x, y] for x, y in self._skeleton_points]
        self.target_skeleton_picked.emit(ScenePrimitive(
            "target_skeleton", "model", {"nodes": nodes, "tolerance_px": 4.0},
            name=f"目标骨架（{len(nodes)}节点）"))
        self.clear_skeleton_points()
        return True

    def set_read_only(self, read_only: bool) -> None:
        """只读模式:主显示区是纯显示,禁用鼠标点选 handler + 隐藏工具提示。"""
        self.read_only = bool(read_only)
        self._hint.setVisible(not self.read_only)

    # ---- 鼠标 ----
    def _on_click(self, ev) -> None:
        if self.read_only:
            return
        if not self.plot.sceneBoundingRect().contains(ev.scenePos()):
            return
        view = self.plot.getViewBox()
        mapped = view.mapSceneToView(ev.scenePos())
        col, row = int(mapped.x()), int(mapped.y())
        self.sig_image_clicked.emit(col, row)
        if self.tool == "add_target_skeleton":
            if ev.double():
                self.commit_skeleton_target()
                return
            self._skeleton_points.append((float(col), float(row)))
            xs = [p[0] for p in self._skeleton_points]
            ys = [p[1] for p in self._skeleton_points]
            self._skeleton_preview.setData(x=xs, y=ys)
            self.skeleton_draft_changed.emit(len(self._skeleton_points))
            return
        if self.tool == "add_target":
            self.target_picked.emit(ScenePrimitive(
                "target_point", "model", {"xy": [col, row], "node": 0},
                name=f"target_{len(self._scene_items)}"))
        elif self.tool == "add_obstacle":
            self.obstacle_picked.emit(ScenePrimitive(
                "obstacle_circle", "model", {"center": [col, row], "radius": 10.0},
                name=f"obstacle_{len(self._scene_items)}"))
