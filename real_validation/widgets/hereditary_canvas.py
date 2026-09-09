"""Source-pixel camera canvas, editable initial curve and translucent tube views."""
from PyQt5.QtCore import Qt, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget
import numpy as np


class HereditaryCanvas(QWidget):
    curve_finished = pyqtSignal(object)
    draft_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.frame = None
        self.stroke = []
        self.prediction = None
        self.preview = None
        self.target = None
        self.draft = None
        self.mode = 'view'
        self.drawing = False
        self.locked = False
        self.frozen = False
        self.drag_node = None
        self.show_tube = True
        self.opacity = .28
        self.radius = 10.

    def set_frame(self, frame):
        if not self.frozen:
            self.frame = np.ascontiguousarray(frame)
            self.update()

    def _layout(self):
        h, w = self.frame.shape[:2] if self.frame is not None else (480, 640)
        scale = min(self.width()/w, self.height()/h)
        return scale, (self.width()-w*scale)/2, (self.height()-h*scale)/2

    def _point(self, event):
        s, x, y = self._layout()
        point = np.array(((event.x()-x)/s, (event.y()-y)/s))
        if self.frame is not None:
            point = np.clip(point, [0, 0], [self.frame.shape[1]-1, self.frame.shape[0]-1])
        return tuple(point)

    def mousePressEvent(self, event):
        if self.locked or self.mode == 'view' or event.button() != Qt.LeftButton:
            return
        point = self._point(event)
        if self.mode == 'edit' and self.draft is not None:
            distances = np.linalg.norm(self.draft-point, axis=1)
            index = int(np.argmin(distances))
            if distances[index]*self._layout()[0] <= 14:
                self.drag_node = index
            return
        self.stroke = [point]
        self.drawing = True
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_node is not None:
            self.draft[self.drag_node] = self._point(event)
            self.draft_changed.emit(self.draft.copy())
            self.update()
        elif self.drawing:
            point = self._point(event)
            if np.linalg.norm(np.asarray(point)-self.stroke[-1]) >= 2:
                self.stroke.append(point)
                self.update()

    def mouseReleaseEvent(self, event):
        if self.drag_node is not None:
            self.draft[self.drag_node] = self._point(event)
            self.drag_node = None
            self.draft_changed.emit(self.draft.copy())
            self.update()
        if self.drawing:
            self.drawing = False
            if len(self.stroke) >= 2:
                self.curve_finished.emit(np.asarray(self.stroke))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor('#17232f'))
        scale, x, y = self._layout()
        painter.translate(x, y)
        painter.scale(scale, scale)
        if self.frame is not None:
            rgb = np.ascontiguousarray(self.frame[..., ::-1])
            h, w = rgb.shape[:2]
            painter.drawImage(QPointF(0, 0), QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888))
        for points, color in ((self.target, '#ef5350'), (self.prediction, '#00c8c8'),
                              (self.preview, '#b388ff'), (self.draft, '#ffd54f'), (self.stroke, '#ffd54f')):
            if points is None or len(points) < 2:
                continue
            poly = QPolygonF([QPointF(float(a), float(b)) for a, b in points])
            if self.show_tube:
                fill = QColor(color)
                fill.setAlphaF(self.opacity)
                painter.setPen(QPen(fill, 2*self.radius, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawPolyline(poly)
            painter.setPen(QPen(QColor(color), 1.5/scale))
            painter.drawPolyline(poly)
            if points is self.draft:
                painter.setBrush(QColor('#ffd54f'))
                for point in poly:
                    painter.drawEllipse(point, 4/scale, 4/scale)
                painter.drawText(poly[0]+QPointF(10, 0), 'BASE')
                painter.drawText(poly[-1]+QPointF(10, 0), 'TIP')
                painter.setBrush(Qt.NoBrush)
        if self.frozen:
            painter.setPen(QColor('#ffd54f'))
            painter.drawText(QPointF(12, 52), 'FROZEN - edit draft, then confirm / cancel')
