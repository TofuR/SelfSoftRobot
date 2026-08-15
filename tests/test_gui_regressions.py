"""GUI 绘制/接线层回归测试(offscreen)。

这两个 bug 由真机操作"添加目标/障碍"暴露,已在 systematic-debugging 下修复,
本文件用离屏 Qt 锁住行为防止回归:
- A: primitive_items 用 QPen(str) 画障碍崩溃 → 改用 pg.mkPen(str)
- B: _scene_changed 引用从未创建的 scene_summary → 在 _observe_page 创建

与其它测试隔离:本文件 import PyQt5,且必须在 QT_QPA_PLATFORM=offscreen 下运行
(测试内部设置;无显示器环境也能跑)。不影响 test_import_hygiene(它只断言
real_validation 包根闭包 stdlib-only,本文件不在该闭包内)。
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QSplitter

from real_validation.contracts.models import Scene, ScenePrimitive
from real_validation.core.session import ExperimentSession
from real_validation.widgets.camera_view import CameraViewWidget
from real_validation.widgets.primitive_items import scene_primitive_item
from real_validation.widgets.scene_editor import SceneEditorPanel

# 惰性创建单个 QApplication,避免 unittest 多次 init
_app: QApplication | None = None


def _ensure_app() -> QApplication:
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication([])
    return _app


class PrimitivePenRegressionTest(unittest.TestCase):
    """bug A:障碍/目标绘制用 str 色值不再崩(修 QPen(str) → pg.mkPen(str))。"""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_obstacle_circle_pen_is_valid_color(self):
        item = scene_primitive_item(
            ScenePrimitive("obstacle_circle", "model", {"center": [10, 10], "r": 5}))
        self.assertIsNotNone(item)
        pen = item.pen()
        self.assertTrue(pen.color().isValid())
        self.assertIn(pen.color().name(), ("#b7791f", "#b7791F"))

    def test_obstacle_aabb_pen_is_valid_color(self):
        item = scene_primitive_item(
            ScenePrimitive("obstacle_aabb", "model", {"min": [0, 0], "max": [20, 20]}))
        self.assertIsNotNone(item)
        self.assertTrue(item.pen().color().isValid())

    def test_target_circle_pen_is_red(self):
        item = scene_primitive_item(
            ScenePrimitive("target_circle", "model", {"center": [50, 50], "r": 5}))
        self.assertIsNotNone(item)
        pen = item.pen()
        self.assertTrue(pen.color().isValid())
        self.assertIn(pen.color().name(), ("#e53e3e", "#E53E3E"))

    def test_target_point_and_skeleton_pen_valid(self):
        for primitive in (
            ScenePrimitive("target_point", "model", {"xy": [5, 5]}),
            ScenePrimitive("target_skeleton", "model", {"nodes": [[0, 0], [10, 10]]}),
        ):
            item = scene_primitive_item(primitive)
            self.assertIsNotNone(item)

    def test_target_skeleton_kind_is_in_whitelist(self):
        # 白名单缺 target_skeleton 会让 primitive_items 的该分支成死代码
        from real_validation.contracts.models import Scene
        scene = Scene("s", (ScenePrimitive("target_skeleton", "model",
                                           {"nodes": [[0, 0], [1, 1]]}),))
        self.assertEqual(len(scene.primitives), 1)


class SceneSummaryRegressionTest(unittest.TestCase):
    """bug B:添加目标/障碍触发 _scene_changed 不再因 scene_summary 缺失崩溃。"""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_add_target_and_obstacle_writes_scene_summary(self):
        from real_validation.gui.main_window import ValidationWindow
        window = ValidationWindow()
        window.session = ExperimentSession.create(
            tempfile.mkdtemp(prefix="gui_regress_"))
        try:
            # 必须先有 scene_summary(修 bug B 后在 _observe_page 创建)
            self.assertTrue(hasattr(window, "scene_summary"))
            window.target_x.setValue(100)
            window.target_y.setValue(200)
            window.target_radius.setValue(5)
            window._set_target()
            self.assertEqual(len(window.session.scene.primitives), 1)
            self.assertIn("primitives=1", window.scene_summary.toPlainText())
            window.obstacle_x.setValue(30)
            window.obstacle_y.setValue(40)
            window.obstacle_radius.setValue(8)
            window._add_obstacle()
            self.assertEqual(len(window.session.scene.primitives), 2)
            self.assertIn("primitives=2", window.scene_summary.toPlainText())
        finally:
            window.close()

    def test_numeric_add_syncs_visualization(self):
        # 打磨:数值添加(_set_target/_add_obstacle)后,右侧 camera_view 原语与
        # scene_editor 列表必须同步更新(原先只有工具点加才刷新)。
        from real_validation.gui.main_window import ValidationWindow
        window = ValidationWindow()
        window.session = ExperimentSession.create(
            tempfile.mkdtemp(prefix="gui_regress_sync_"))
        try:
            self.assertEqual(len(window.camera_view._scene_items), 0)
            self.assertEqual(window.scene_editor.list.count(), 0)
            window.target_x.setValue(100)
            window.target_y.setValue(200)
            window.target_radius.setValue(5)
            window._set_target()
            # 数值添加后 camera_view 重绘 + scene_editor 列表同步
            self.assertEqual(len(window.camera_view._scene_items), 1)
            self.assertEqual(window.scene_editor.list.count(), 1)
            window.obstacle_x.setValue(30)
            window.obstacle_y.setValue(40)
            window.obstacle_radius.setValue(8)
            window._add_obstacle()
            self.assertEqual(len(window.camera_view._scene_items), 2)
            self.assertEqual(window.scene_editor.list.count(), 2)
        finally:
            window.close()

    def test_click_skeleton_target_commits_on_double_click(self):
        # 功能③:『点出目标骨架』—— 依次点击累积节点,双击提交 target_skeleton
        import numpy as np
        from PyQt5.QtCore import QPointF
        from real_validation.gui.main_window import ValidationWindow
        window = ValidationWindow()
        window.session = ExperimentSession.create(
            tempfile.mkdtemp(prefix="gui_regress_skel_"))
        try:
            view = window.camera_view
            view.set_frame(np.zeros((240, 320, 3)))   # 设图像范围,mapViewToScene 才正确
            vb = view.plot.getViewBox()

            class _FakeEv:
                def __init__(self, pos, double):
                    self._pos, self._double = pos, double
                def scenePos(self):
                    return self._pos
                def double(self):
                    return self._double

            def click(px, py, double=False):
                view._on_click(_FakeEv(vb.mapViewToScene(QPointF(px, py)), double))

            window._set_tool("add_target_skeleton")
            click(100, 100)
            click(150, 120)
            click(180, 150)
            self.assertEqual(len(view._skeleton_points), 3)
            click(180, 150, double=True)   # 双击提交
            self.assertEqual(len(view._skeleton_points), 0)   # 提交后清空
            self.assertEqual(len(window.session.scene.primitives), 1)
            primitive = window.session.scene.primitives[0]
            self.assertEqual(primitive.kind, "target_skeleton")
            self.assertEqual(len(primitive.geometry["nodes"]), 3)
        finally:
            window.close()


class MainDisplayLayerTest(unittest.TestCase):
    """Task 1:主显示多层叠加(预测轨迹/实际骨架/NDI)+ 图层可见性开关。"""

    def setUp(self):
        _ensure_app()
        self.view = CameraViewWidget()
        self.view.set_frame(np.zeros((240, 320, 3)))

    def test_layer_api_accepts_predicted_and_actual(self):
        self.view.set_predicted_states(np.zeros((5, 15, 2)))
        self.view.set_actual_skeleton(np.zeros((15, 2)))
        self.view.set_ndi_position((10.0, 20.0))
        # 不应抛异常,且 predicted/actual item 已加入 plot
        self.assertEqual(len(self.view._predicted_items), 5)   # K 条骨架线
        self.assertIsNotNone(self.view._actual_scatter)

    def test_layer_visibility_toggles_items(self):
        self.view.set_scene(None)  # 不设场景,只测骨架层
        self.view.set_skeleton(np.zeros((15, 2)))
        self.assertTrue(self.view._skeleton_curve.isVisible())
        self.view.set_layer_visible("skeleton", False)
        self.assertFalse(self.view._skeleton_curve.isVisible())
        self.view.set_layer_visible("skeleton", True)
        self.assertTrue(self.view._skeleton_curve.isVisible())

    def test_predicted_layer_visibility_survives_restate(self):
        # 锁死实现细节:set_predicted_states 用 .clear() 而非重绑定,否则
        # _layer_items["predicted"] 会指向陈旧 list,二次 set 后开关失效。
        self.view.set_predicted_states(np.zeros((3, 15, 2)))
        self.view.set_predicted_states(np.zeros((4, 15, 2)))   # 二次 set(重绘)
        self.assertEqual(len(self.view._predicted_items), 4)
        self.view.set_layer_visible("predicted", False)
        self.assertTrue(all(not it.isVisible() for it in self.view._predicted_items))
        self.view.set_layer_visible("predicted", True)
        self.assertTrue(all(it.isVisible() for it in self.view._predicted_items))

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            self.view.set_layer_visible("no_such_layer", True)


class MainWindowLayoutTest(unittest.TestCase):
    """Task 3:主窗口左右两栏 —— 左主显示区 + 右 5 页 Tab。"""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_two_column_layout_with_main_display(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            splitters = w.findChildren(QSplitter)
            horizontal = [s for s in splitters if s.orientation() == 1]  # Qt.Horizontal
            self.assertTrue(horizontal, "应有一个水平 splitter 分左右两栏")
            self.assertTrue(hasattr(w, "main_display"), "应有主显示视图")
            self.assertIsNotNone(w.main_display)
        finally:
            w.close()


class CompactLayoutTest(unittest.TestCase):
    """Task 4:紧凑排版 —— 安全配置 6×5 一次显示全 + Setup/Plan 参数压缩 + 窗口 1400x860。"""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_safety_table_shows_all_six_rows(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            table = w.safety_table
            actual_content = sum(table.verticalHeader().sectionSize(i) for i in range(6)) \
                             + table.horizontalHeader().height()
            self.assertLessEqual(actual_content, table.maximumHeight())   # 6 行内容必须装进最大高度,无滚动
            self.assertEqual(table.rowCount(), 6)
        finally:
            w.close()

    def test_model_summary_height_capped(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            self.assertLessEqual(w.model_summary.maximumHeight(), 110)
        finally:
            w.close()

    def test_plan_summary_height_capped(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            self.assertLessEqual(w.plan_summary.maximumHeight(), 90)
        finally:
            w.close()

    def test_window_default_size(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            self.assertEqual(w.size().width(), 1400)
            self.assertEqual(w.size().height(), 860)
        finally:
            w.close()


class SceneEditorMultiSelectTest(unittest.TestCase):
    """Task 2:scene_editor 列表多选(ExtendedSelection)+ 批量删 + Del 快捷键。"""

    def setUp(self):
        _ensure_app()
        self.editor = SceneEditorPanel()
        self.edited = []
        self.editor.scene_edited.connect(self.edited.append)
        scene = Scene("s", (
            ScenePrimitive("target_point", "model", {"xy": [0, 0]}, name="t1"),
            ScenePrimitive("obstacle_circle", "model", {"center": [1, 1], "r": 1}, name="o1"),
            ScenePrimitive("target_point", "model", {"xy": [2, 2]}, name="t2"),
        ))
        self.editor.set_scene(scene)

    def test_extended_selection_enabled(self):
        self.assertEqual(self.editor.list.selectionMode(),
                         3)  # QAbstractItemView.ExtendedSelection

    def test_batch_delete_removes_selected(self):
        self.editor.list.setCurrentRow(0)
        self.editor.list.item(0).setSelected(True)
        self.editor.list.item(2).setSelected(True)   # 选 t1 + t2
        self.editor._on_delete()
        self.assertEqual(len(self.edited), 1)
        remaining = self.edited[-1].primitives
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].name, "o1")     # 只留障碍

    def test_delete_key_triggers_batch_delete(self):
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent
        self.editor.list.item(0).setSelected(True)
        self.editor.list.item(2).setSelected(True)   # 选 t1 + t2
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier)
        self.editor.keyPressEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(len(self.edited), 1)
        remaining = self.edited[-1].primitives
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].name, "o1")

    def test_delete_key_no_selection_is_noop(self):
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtGui import QKeyEvent
        self.editor.list.clearSelection()
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Backspace, Qt.NoModifier)
        self.editor.keyPressEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(len(self.edited), 0)
        self.assertEqual(self.editor.list.count(), 3)


if __name__ == "__main__":
    unittest.main()
