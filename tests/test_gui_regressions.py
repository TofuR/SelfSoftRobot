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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from real_validation.models import ScenePrimitive
from real_validation.session import ExperimentSession
from real_validation.widgets.primitive_items import scene_primitive_item

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
        from real_validation.models import Scene
        scene = Scene("s", (ScenePrimitive("target_skeleton", "model",
                                           {"nodes": [[0, 0], [1, 1]]}),))
        self.assertEqual(len(scene.primitives), 1)


class SceneSummaryRegressionTest(unittest.TestCase):
    """bug B:添加目标/障碍触发 _scene_changed 不再因 scene_summary 缺失崩溃。"""

    @classmethod
    def setUpClass(cls):
        _ensure_app()

    def test_add_target_and_obstacle_writes_scene_summary(self):
        from real_validation.main_validation import ValidationWindow
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


if __name__ == "__main__":
    unittest.main()
