"""Scene 编辑器:原语列表 + 属性表单 + 增删改(底层用不可变 Scene API)。

任何编辑走 main_validation 的 session.set_scene(有 B16 守卫 + 落盘 scene.json)。
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QListWidget, QPushButton, QVBoxLayout,
                             QWidget)

from ..contracts.models import Scene, ScenePrimitive


class SceneEditorPanel(QWidget):
    scene_edited = pyqtSignal(object)      # 编辑后的 Scene
    primitive_selected = pyqtSignal(str)   # primitive_id

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        root.addWidget(self.list, 1)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.kind_label = QLabel("-")
        self.geometry_label = QLabel("-")
        form.addRow("名称", self.name_edit)
        form.addRow("kind", self.kind_label)
        form.addRow("geometry", self.geometry_label)
        root.addLayout(form)
        buttons = QHBoxLayout()
        self.delete_btn = QPushButton("删除")
        self.apply_btn = QPushButton("应用修改")
        buttons.addWidget(self.delete_btn); buttons.addWidget(self.apply_btn)
        buttons.addStretch()
        root.addLayout(buttons)

        self._scene: Scene | None = None
        self._current_id: str | None = None

        self.list.currentItemChanged.connect(self._on_select)
        self.delete_btn.clicked.connect(self._on_delete)
        self.apply_btn.clicked.connect(self._on_apply)

    def set_scene(self, scene: Scene) -> None:
        self._scene = scene
        self.list.clear()
        for p in scene.primitives:
            self.list.addItem(f"[{p.kind}] {p.name or p.primitive_id[:8]}")
        self._current_id = None

    def _on_select(self, current, _previous) -> None:
        if current is None or self._scene is None:
            return
        primitives = list(self._scene.primitives)
        if self.list.currentRow() >= len(primitives):
            return
        p = primitives[self.list.currentRow()]
        self._current_id = p.primitive_id
        self.name_edit.setText(p.name)
        self.kind_label.setText(p.kind)
        self.geometry_label.setText(str(p.geometry)[:60])
        self.primitive_selected.emit(p.primitive_id)

    def _on_delete(self) -> None:
        """批量删除选中项(Ctrl 多选/Shift 范围/Ctrl+A 全选)。"""
        if self._scene is None:
            return
        rows = sorted({idx.row() for idx in self.list.selectedIndexes()}, reverse=True)
        if not rows:
            return
        primitives = list(self._scene.primitives)
        to_delete = [primitives[row].primitive_id for row in rows if row < len(primitives)]
        if not to_delete:
            return
        updated = self._scene
        for pid in to_delete:
            try:
                updated = updated.without_primitive(pid)
            except KeyError:
                continue
        self._scene = updated
        self.scene_edited.emit(updated)
        self.set_scene(updated)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self._on_delete()
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_apply(self) -> None:
        if self._current_id is None or self._scene is None:
            return
        primitives = list(self._scene.primitives)
        target = next((p for p in primitives if p.primitive_id == self._current_id), None)
        if target is None:
            return
        renamed = ScenePrimitive(
            kind=target.kind, frame_id=target.frame_id, geometry=target.geometry,
            name=self.name_edit.text().strip() or target.name,
            safety_margin=target.safety_margin, primitive_id=target.primitive_id)
        updated = self._scene.replace_primitive(target.primitive_id, renamed)
        self._scene = updated
        self.scene_edited.emit(updated)
        self.set_scene(updated)
