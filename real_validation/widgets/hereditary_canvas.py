"""Source-pixel camera canvas, editable initial curve and translucent tube views."""
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPolygonF
from PyQt5.QtWidgets import QWidget
import numpy as np


class HereditaryCanvas(QWidget):
    curve_finished = pyqtSignal(object)
    draft_changed = pyqtSignal(object)
    roi_finished = pyqtSignal(object)
    prompt_added = pyqtSignal(object,int)

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
        self.highlight_indices = None
        self.roi = None
        self.mask = None
        self.show_mask = True
        self.prompts = []

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
        if self.mode in ('sam_positive','sam_negative'):
            self.prompt_added.emit(np.asarray(point),int(self.mode=='sam_positive'));return
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
        elif self.drawing and self.mode != 'goal_point':
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
            if self.mode == 'roi':
                first=np.asarray(self.stroke[0]);last=np.asarray(self._point(event))
                self.roi_finished.emit(np.r_[np.minimum(first,last),np.maximum(first,last)])
                self.stroke=[];self.update()
            elif self.mode == 'goal_point':
                self.curve_finished.emit(np.asarray([self._point(event)]))
            elif len(self.stroke) >= 2:
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
            if self.mask is not None and self.show_mask:
                rgba=np.zeros((h,w,4),np.uint8);rgba[self.mask>0]=[32,200,110,round(255*self.opacity)]
                painter.drawImage(QPointF(0,0),QImage(rgba.data,w,h,rgba.strides[0],QImage.Format_RGBA8888))
        region=self.roi
        if self.mode=='roi' and len(self.stroke)>1:
            region=np.r_[np.minimum(self.stroke[0],self.stroke[-1]),np.maximum(self.stroke[0],self.stroke[-1])]
        if region is not None:
            painter.setPen(QPen(QColor('#42a5f5'),2/scale,Qt.DashLine));painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(float(region[0]),float(region[1]),float(region[2]-region[0]),float(region[3]-region[1])))
        for point,label in self.prompts:
            painter.setPen(QPen(QColor('#40ef80' if label else '#ff5050'),2/scale))
            p=QPointF(float(point[0]),float(point[1]));painter.drawEllipse(p,4/scale,4/scale)
        for points, color in ((self.target, '#ef5350'), (self.prediction, '#00c8c8'),
                              (self.preview, '#b388ff'), (self.draft, '#ffd54f'), (self.stroke if self.mode!='roi' else [], '#ffd54f')):
            if points is None or len(points) == 0:
                continue
            if len(points) == 1:
                point = QPointF(float(points[0][0]), float(points[0][1]))
                painter.setPen(QPen(QColor(color), 2/scale))
                painter.drawEllipse(point, 6/scale, 6/scale)
                painter.drawLine(point-QPointF(10/scale, 0), point+QPointF(10/scale, 0))
                painter.drawLine(point-QPointF(0, 10/scale), point+QPointF(0, 10/scale))
                continue
            poly = QPolygonF([QPointF(float(a), float(b)) for a, b in points])
            if self.show_tube:
                fill = QColor(color)
                fill.setAlphaF(self.opacity)
                painter.setPen(QPen(fill, 2*self.radius, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
                painter.drawPolyline(poly)
            painter.setPen(QPen(QColor(color), 1.5/scale))
            painter.drawPolyline(poly)
            if points is self.prediction and self.highlight_indices is not None:
                for i in self.highlight_indices:
                    if i < len(poly):
                        painter.drawEllipse(poly[i], 3/scale, 3/scale)
                        painter.drawText(poly[i]+QPointF(5/scale, -5/scale), str(i))
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
