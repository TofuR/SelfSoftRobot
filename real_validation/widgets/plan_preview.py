"""有界内存的预测轨迹与六通道动作预览。"""

from __future__ import annotations

import numpy as np
from pathlib import Path
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen
from PyQt5.QtWidgets import QGraphicsEllipseItem, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class PlanPreviewWidget(QWidget):
    COLORS = ("#2CB1BC", "#667EEA", "#EF4E4E", "#F6AD55", "#38A169", "#805AD5")

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self); plots = QHBoxLayout()
        self.shape_plot = pg.PlotWidget(title="OpenLoop predicted whole-body state")
        self.shape_plot.invertY(True)
        self.shape_plot.showGrid(x=True, y=True, alpha=0.2)
        self.action_plot = pg.PlotWidget(title="planned actions6 (commanded kPa)")
        self.action_plot.showGrid(x=True, y=True, alpha=0.2)
        plots.addWidget(self.shape_plot, 1); plots.addWidget(self.action_plot, 1)
        root.addLayout(plots, 1)
        row = QHBoxLayout(); self.step_label = QLabel("k=-")
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._draw_step)
        row.addWidget(self.step_label); row.addWidget(self.slider, 1); root.addLayout(row)
        self._states = None
        self._shape_curve = self.shape_plot.plot([], [], pen=pg.mkPen("#1A202C", width=2),
                                                 symbol="o", symbolSize=6,
                                                 symbolBrush="#2CB1BC")
        self._scene_items = []

    def clear_plan(self) -> None:
        self._states = None; self.slider.setRange(0, 0); self.step_label.setText("k=-")
        self._shape_curve.setData([], []); self.action_plot.clear()
        for item in self._scene_items:
            self.shape_plot.removeItem(item)
        self._scene_items.clear()

    def set_plan(self, plan, scene, safety, run_dir=None) -> None:
        self.clear_plan()
        if not plan.predicted_states_path:
            raise ValueError("plan 没有 predicted_states_path")
        states_path = Path(plan.predicted_states_path)
        if not states_path.is_absolute() and run_dir is not None:
            states_path = Path(run_dir) / states_path
        with np.load(states_path) as data:
            key = "states_model" if "states_model" in data else "states_normalized"
            states = np.asarray(data[key], dtype=np.float32)
        if states.ndim != 3 or states.shape[0] != plan.horizon or states.shape[2] < 2:
            raise ValueError("predicted_states 与 plan horizon 不一致")
        self._states = states[:, :, :2]
        self.slider.setRange(0, plan.horizon - 1)
        steps = np.arange(plan.horizon)
        actions = np.asarray(plan.actions6)
        for channel, color in enumerate(self.COLORS):
            self.action_plot.plot(steps, actions[:, channel], pen=pg.mkPen(color, width=2),
                                  name=f"ch{channel}")
            self.action_plot.addItem(pg.InfiniteLine(
                safety.pressure_min6[channel], angle=0,
                pen=pg.mkPen(color, width=1, style=Qt.DotLine)))
            self.action_plot.addItem(pg.InfiniteLine(
                safety.pressure_max6[channel], angle=0,
                pen=pg.mkPen(color, width=1, style=Qt.DotLine)))
        self._draw_scene(scene)
        self.slider.setValue(0); self._draw_step(0)

    def _draw_scene(self, scene) -> None:
        for primitive in scene.primitives:
            if primitive.frame_id != "model":
                continue
            xy = primitive.geometry.get("xy", primitive.geometry.get("center"))
            if primitive.kind == "target_point" and xy:
                item = pg.ScatterPlotItem([xy[0]], [xy[1]], symbol="x", size=16,
                                          pen=pg.mkPen("#E53E3E", width=3))
            elif primitive.kind in {"target_circle", "obstacle_circle"} and xy:
                radius = float(primitive.geometry.get(
                    "radius", primitive.geometry.get("r", 0.0)))
                if primitive.kind == "obstacle_circle":
                    radius += float(primitive.safety_margin)
                item = QGraphicsEllipseItem(xy[0] - radius, xy[1] - radius,
                                            2 * radius, 2 * radius)
                item.setPen(QPen(Qt.red if primitive.kind == "target_circle" else Qt.darkYellow,
                                 2, Qt.DashLine))
            else:
                continue
            self.shape_plot.addItem(item); self._scene_items.append(item)

    def _draw_step(self, step: int) -> None:
        if self._states is None:
            return
        step = max(0, min(int(step), len(self._states) - 1))
        state = self._states[step]
        self._shape_curve.setData(state[:, 0], state[:, 1])
        self.step_label.setText(f"k={step + 1}/{len(self._states)}")
