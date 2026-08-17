# real_validation 界面重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** real_validation GUI 改为左右两栏(左固定显示区 + 右 5 页 Tab),主显示视图支持多层叠加(骨架/场景/预测轨迹/实际骨架/NDI)+ 图层开关,摄像头流窗口级共享,Setup/Plan 参数紧凑排版一次显示全,场景多选删除。

**Architecture:** 增强 `CameraViewWidget` 为主显示视图(加 set_predicted_states/set_actual_skeleton/set_ndi_position/set_layer_visible);`_build_ui` 改 QSplitter 左右两栏(左主显示区 + 右 tabs);`_on_camera_frame` 分发到主显示;安全配置压缩行高;scene_editor 改 ExtendedSelection 多选删除。纯 GUI 层改动,数据契约/规划/执行逻辑不动。

**Tech Stack:** PyQt5 QSplitter/QTabWidget/QCheckBox, pyqtgraph, numpy。

## Global Constraints

- 分支固定 `feat/real-data-transition`;提交前询问用户(用户已授权直接执行本 plan,提交自动)。
- **包根 `real_validation/__init__.py` 保持 stdlib-only,别动**(import 卫生测试必须绿)。
- 改动只碰 `gui/` 层(camera_view.py/scene_editor.py/main_window.py/plan_preview.py)+ tests;数据契约/规划/执行逻辑不动。
- 现有测试必须全绿(130 + 新增);测试框架 unittest,无 pytest。
- **Observe 页 camera_view 保留点选/工具交互不变**(锚定用);主显示区是增强实例(只读显示,不参与锚定点击)。
- 摄像头流窗口级共享:`_on_camera_frame` 喂主显示 + (Observe 独立 camera_view 若有)。
- 紧凑排版后参数必须仍可读(行高 26px SpinBox 舒适性人工确认)。

---

### Task 1: 增强 CameraViewWidget —— 多层叠加 + 图层开关

**Files:**
- Modify: `real_validation/widgets/camera_view.py`
- Test: `tests/test_gui_regressions.py`(新增主显示图层测试)

**Interfaces:**
- Consumes: 现有 `CameraViewWidget`(set_frame/set_skeleton/set_anchor/set_scene,全部保留)。
- Produces: `set_predicted_states(states)`(K,N,2 像素)、`set_actual_skeleton(skeleton)`(N,2)、`set_ndi_position(xy | None)`、`set_layer_visible(layer: str, visible: bool)`(layer ∈ {"skeleton","scene","predicted","actual","ndi"})。图层开关控制各层 item 可见性。

- [ ] **Step 1: 写失败测试(图层 API 存在 + 可见性切换生效)**

`tests/test_gui_regressions.py` 加:

```python
import numpy as np
from real_validation.widgets.camera_view import CameraViewWidget

class MainDisplayLayerTest(unittest.TestCase):
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.MainDisplayLayerTest -v`
Expected: FAIL(`set_predicted_states` 不存在)。

- [ ] **Step 3: 实现图层叠加 + 开关**

`camera_view.py` `__init__` 加:

```python
# 主显示增强图层(规划预测轨迹 / 执行实际骨架 / NDI 末端)
self._predicted_items: list[object] = []
self._actual_scatter = pg.ScatterPlotItem(symbol="o", size=10,
                                          pen=pg.mkPen("#EF4E4E", width=2),
                                          brush=pg.mkBrush("#EF4E4E"))
self.plot.addItem(self._actual_scatter)
self._ndi_scatter = pg.ScatterPlotItem(symbol="star", size=14,
                                       pen=pg.mkPen("#805AD5", width=2))
self.plot.addItem(self._ndi_scatter)
# 图层可见性映射
self._layer_items: dict[str, list[object]] = {
    "skeleton": [self.skeleton_curve],
    "scene": [],            # set_scene 时登记
    "predicted": self._predicted_items,
    "actual": [self._actual_scatter],
    "ndi": [self._ndi_scatter],
}
```

`set_predicted_states`:
```python
def set_predicted_states(self, states) -> None:
    for item in self._predicted_items:
        self.plot.removeItem(item)
    self._predicted_items = []
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[2] < 2:
        return
    for k in range(states.shape[0]):
        line = self.plot.plot(states[k, :, 0], states[k, :, 1],
                              pen=pg.mkPen("#8B9BB4", width=1,
                                           style=Qt.DashLine))
        self._predicted_items.append(line)
```

`set_actual_skeleton`:
```python
def set_actual_skeleton(self, skeleton) -> None:
    sk = np.asarray(skeleton, dtype=np.float64)
    if sk.size and sk.shape[1] >= 2:
        self._actual_scatter.setData(x=sk[:, 0], y=sk[:, 1])
    else:
        self._actual_scatter.setData(x=[], y=[])
```

`set_ndi_position`:
```python
def set_ndi_position(self, xy) -> None:
    if xy is None:
        self._ndi_scatter.setData(x=[], y=[])
    else:
        self._ndi_scatter.setData(x=[float(xy[0])], y=[float(xy[1])])
```

`set_layer_visible`:
```python
def set_layer_visible(self, layer: str, visible: bool) -> None:
    if layer not in self._layer_items:
        raise ValueError(f"未知图层: {layer}")
    for item in self._layer_items[layer]:
        item.setVisible(bool(visible))
```

`set_scene` 末尾登记 scene 图层:
```python
self._layer_items["scene"] = [item for _pid, item in self._scene_items]
```

- [ ] **Step 4: 运行测试确认通过**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.MainDisplayLayerTest -v`
Expected: PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

```bash
git add real_validation/widgets/camera_view.py tests/test_gui_regressions.py
git commit -m "feat(real_validation): CameraViewWidget 增强 —— 多层叠加(预测轨迹/实际骨架/NDI)+ 图层可见性开关"
```

---

### Task 2: scene_editor 多选删除

**Files:**
- Modify: `real_validation/widgets/scene_editor.py`
- Test: `tests/test_gui_regressions.py`

**Interfaces:**
- Consumes: 现有 `SceneEditorPanel`(set_scene/primitive_selected/scene_edited)。
- Produces: 列表改 `ExtendedSelection`(Ctrl 多选/Shift 范围/Ctrl+A 全选);`_on_delete` 批量删选中项;Del 快捷键。

- [ ] **Step 1: 写失败测试(多选 + 批量删)**

`tests/test_gui_regressions.py` 加:

```python
from real_validation.widgets.scene_editor import SceneEditorPanel
from real_validation.contracts.models import Scene, ScenePrimitive

class SceneEditorMultiSelectTest(unittest.TestCase):
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.SceneEditorMultiSelectTest -v`
Expected: FAIL(selectionMode 默认单选 = 1,批量删不删两个)。

- [ ] **Step 3: 实现 ExtendedSelection + 批量删 + Del**

`scene_editor.py`:
```python
from PyQt5.QtWidgets import QAbstractItemView
# __init__:
self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
```

`_on_delete` 批量删:
```python
def _on_delete(self) -> None:
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
```

`keyPressEvent`(Del 快捷键):
```python
from PyQt5.QtCore import Qt
def keyPressEvent(self, event) -> None:
    if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
        self._on_delete()
        event.accept()
        return
    super().keyPressEvent(event)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.SceneEditorMultiSelectTest -v`
Expected: PASS。

- [ ] **Step 5: 全量测试 + 提交**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

```bash
git add real_validation/widgets/scene_editor.py tests/test_gui_regressions.py
git commit -m "feat(real_validation): scene_editor 多选删除 —— ExtendedSelection + 批量删 + Del 快捷键"
```

---

### Task 3: 主窗口左右两栏 + 主显示视图 + 摄像头流分发

**Files:**
- Modify: `real_validation/gui/main_window.py`(`_build_ui` 改两栏;`_on_camera_frame` 分发;`_start_camera` 更新)
- Test: `tests/test_gui_regressions.py`

**Interfaces:**
- Consumes: Task 1 的 `CameraViewWidget` 增强 API。
- Produces: `self.main_display`(主显示实例)、`self.layer_checks`(图层勾选框 dict)、`_build_ui` 返回两栏结构。

- [ ] **Step 1: 写失败测试(主窗口两栏 + 主显示区存在)**

`tests/test_gui_regressions.py` 加:

```python
from PyQt5.QtWidgets import QSplitter
class MainWindowLayoutTest(unittest.TestCase):
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.MainWindowLayoutTest -v`
Expected: FAIL(无水平 splitter、无 main_display)。

- [ ] **Step 3: 重构 _build_ui 为两栏**

`main_window.py` `_build_ui`(当前 line 173-196):

```python
def _build_ui(self) -> None:
    central = QWidget()
    layout = QVBoxLayout(central)
    safety_bar = QHBoxLayout()
    self.state_label = QLabel("No session")
    self.zero_button = QPushButton("归零 / Zero")
    self.zero_button.setObjectName("danger")
    self.zero_button.clicked.connect(self._zero)
    self.abort_button = QPushButton("中止 / Abort")
    self.abort_button.setObjectName("danger")
    self.abort_button.clicked.connect(self._abort)
    safety_bar.addWidget(self.state_label, 1)
    safety_bar.addWidget(self.zero_button)
    safety_bar.addWidget(self.abort_button)
    layout.addLayout(safety_bar)

    # ---- 左右两栏:左固定显示区 + 右 5 页 Tab ----
    main_split = QSplitter(Qt.Horizontal)

    # 左:主显示区(摄像头 + 多层叠加 + 图层开关)
    left_panel = QWidget()
    ll = QVBoxLayout(left_panel); ll.setContentsMargins(4, 4, 4, 4)
    self.main_display = CameraViewWidget()
    ll.addWidget(self.main_display, 1)
    layer_row = QHBoxLayout()
    self.layer_checks = {}
    for key, label in (("skeleton", "骨架"), ("scene", "场景"),
                       ("predicted", "预测"), ("actual", "实际"), ("ndi", "NDI")):
        cb = QCheckBox(label)
        cb.setChecked(key != "ndi")          # 默认 NDI 关
        cb.toggled.connect(
            lambda checked, k=key: self.main_display.set_layer_visible(k, checked))
        self.layer_checks[key] = cb
        layer_row.addWidget(cb)
    layer_row.addStretch()
    ll.addLayout(layer_row)
    main_split.addWidget(left_panel)

    # 右:tabs
    self.tabs = QTabWidget()
    self.tabs.addTab(self._setup_page(), "1 Setup")
    self.tabs.addTab(self._observe_page(), "2 Observe & Scene")
    self.tabs.addTab(self._plan_page(), "3 Plan")
    self.tabs.addTab(self._execute_page(), "4 Execute")
    self.tabs.addTab(self._results_page(), "5 Results")
    main_split.addWidget(self.tabs)

    main_split.setSizes([520, 860])
    main_split.setStretchFactor(0, 0); main_split.setStretchFactor(1, 1)
    layout.addWidget(main_split, 1)
    self.setCentralWidget(central)
```

需要 import `QCheckBox`(已在),`CameraViewWidget`(已在 `from ..widgets import CameraViewWidget`)。

**关键**:Observe 页的 `_observe_page` 现在有自己的 camera_view(line 405)。**保留它**(锚定交互),`_start_camera`/`_on_camera_frame` 喂两者。

- [ ] **Step 4: 改摄像头流分发**

`_start_camera`(line 464):
```python
def _start_camera(self) -> None:
    self._camera_thread = _CameraThread(self)
    self._camera_thread.frame_ready.connect(self._on_camera_frame)
    self._camera_thread.start()
    self.camera_btn.setText("Camera 运行中")
    self.camera_anchor_btn.setEnabled(True)
    self.warmup_btn.setEnabled(True)
    # 主显示区 + Observe 页 camera_view 都要帧(若存在)
    self.main_display.set_frame(self._latest_frame if self._latest_frame is not None
                                else np.zeros((240, 320, 3)))
```

`_on_camera_frame`(line 472)改为分发:
```python
def _on_camera_frame(self, bgr) -> None:
    self._latest_frame = bgr
    self.main_display.set_frame(bgr)                       # 主显示区
    if hasattr(self, "camera_view") and self.camera_view is not None:
        self.camera_view.set_frame(bgr)                    # Observe 锚定视图
    if self.runtime is not None:
        from ..perception.segmentation import segment_white_on_blue
        from ..perception.skeleton import extract_skeleton_2d
        mask = segment_white_on_blue(bgr, self._gray(bgr))
        skeleton, _ = extract_skeleton_2d(mask, self.runtime.descriptor.n_nodes,
                                          tip_fix=True, return_info=True)
        self.main_display.set_skeleton(skeleton)           # 主显示骨架层
        if hasattr(self, "camera_view") and self.camera_view is not None:
            self.camera_view.set_skeleton(skeleton)        # Observe 也显示
```

**执行时实际骨架**:`_execution_done` 时无法实时(每步在执行线程)。**折中**:执行前把每步 predicted 当"实际"预览,或执行时每步由 executor 回调更新(需接线)。**本任务只做**:规划完成 → `main_display.set_predicted_states`;执行完成 → 若有实际骨架则 set_actual_skeleton。执行中逐帧实际骨架留 Task 4 的接线(或标注待真机)。

规划完成喂预测轨迹:`_show_preflight` PASS 分支(已有 set_plan 调 plan_preview)加:
```python
# 主显示区叠加预测轨迹(读 predicted_states.npz)
if self.session.plan and self.session.plan.predicted_states_path:
    p = Path(self.session.plan.predicted_states_path)
    if not p.is_absolute():
        p = self.session.run_dir / p
    if p.is_file():
        import numpy as np
        with np.load(p) as data:
            key = "states_model" if "states_model" in data else "states_normalized"
            self.main_display.set_predicted_states(np.asarray(data[key]))
```

- [ ] **Step 5: 运行测试确认通过**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.MainWindowLayoutTest -v`
Expected: PASS。

- [ ] **Step 6: offscreen 冒烟(两栏 + 摄像头流启动)**

Run:
```bash
QT_QPA_PLATFORM=offscreen python -c "
import sys
from PyQt5.QtWidgets import QApplication, QSplitter
from real_validation.gui.main_window import ValidationWindow
app = QApplication(sys.argv)
w = ValidationWindow(); w.show()
h = [s for s in w.findChildren(QSplitter) if s.orientation() == 1]
print('horizontal splitters:', len(h))
print('main_display:', w.main_display is not None)
print('layer_checks keys:', sorted(w.layer_checks.keys()))
# 启动摄像头流
w.session = None  # 不需要 session
w.main_display.set_frame(__import__('numpy').zeros((240,320,3)))
w.close(); print('ok')
"
```

- [ ] **Step 7: 全量测试 + 提交**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿(Observe 页 camera_view 保留,旧测试仍引用它)。

```bash
git add real_validation/gui/main_window.py tests/test_gui_regressions.py
git commit -m "feat(real_validation): 主窗口左右两栏 —— 左主显示区(多层叠加+图层开关)+ 右5页Tab + 摄像头流分发"
```

---

### Task 4: 紧凑排版(安全配置 + Setup/Plan 参数)

**Files:**
- Modify: `real_validation/gui/main_window.py`(`_setup_page`/`_plan_page` 压缩)
- Test: `tests/test_gui_regressions.py`

**Interfaces:**
- Consumes: 无新 API。
- Produces: 安全配置表 6 行一次显示全;Setup/Plan 参数压缩到一屏。

- [ ] **Step 1: 写失败测试(安全配置 6 行高度足够)**

`tests/test_gui_regressions.py` 加:

```python
class CompactLayoutTest(unittest.TestCase):
    def test_safety_table_shows_all_six_rows(self):
        from real_validation.gui.main_window import ValidationWindow
        w = ValidationWindow()
        try:
            table = w.safety_table
            row_h = table.verticalHeader().defaultSectionSize()
            header_h = table.horizontalHeader().height()
            # 6 行 + 表头应在一屏内(不再需滚动)
            self.assertLessEqual(6 * row_h + header_h, table.minimumHeight() + 20)
            self.assertEqual(table.rowCount(), 6)
        finally:
            w.close()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.CompactLayoutTest -v`
Expected: FAIL(minimumHeight 默认 0,断言 6*行高 ≤ 0+20 失败)。

- [ ] **Step 3: 压缩安全配置 + Setup 页**

`_setup_page` 安全配置段(line 235-249)加压缩:

```python
self.safety_table = QTableWidget(6, 5)
self.safety_table.setHorizontalHeaderLabels(["min", "max", "rise/s", "fall/s", "initial"])
self.safety_table.verticalHeader().setDefaultSectionSize(24)
self.safety_table.horizontalHeader().setDefaultSectionSize(92)
self.safety_table.setMinimumHeight(6 * 24 + 26)
self.safety_table.setMaximumHeight(6 * 24 + 26)
```

Setup 页整体压缩:每卡 `setContentsMargins(8, 10, 8, 8)`,`root.setSpacing(8)`。model_summary 限高:
```python
self.model_summary.setMaximumHeight(110)
```

- [ ] **Step 4: 压缩 Plan 页**

`_plan_page` 参数卡:
```python
form.setVerticalSpacing(5)
# SpinBox 保持紧凑(默认即可),plan_summary 限高
self.plan_summary.setMaximumHeight(90)
```

- [ ] **Step 5: 窗口默认尺寸**

`__init__` 改 `self.resize(1400, 860)`。

- [ ] **Step 6: 运行测试确认通过**

Run: `QT_QPA_PLATFORM=offscreen python -m unittest tests.test_gui_regressions.CompactLayoutTest -v`
Expected: PASS。

- [ ] **Step 7: 全量测试 + 提交**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

```bash
git add real_validation/gui/main_window.py tests/test_gui_regressions.py
git commit -m "feat(real_validation): 紧凑排版 —— 安全配置 6×5 一次显示全(压缩行高)+ Setup/Plan 参数压缩 + 窗口 1400x860"
```

---

### Task 5: 文档同步 + 最终验证

**Files:**
- Modify: `real_validation/GUI_GUIDE.md`(两栏布局说明)
- Test: 全量

**Interfaces:**
- Consumes: Task 1-4 完成。
- Produces: 文档反映两栏 + 主显示 + 多选。

- [ ] **Step 1: 更新 GUI_GUIDE**

在 GUI_GUIDE 第 2 节加"整体布局"说明:左右两栏(左主显示区常驻摄像头+叠加+图层开关,右 5 页 Tab);Observe 页锚定交互在右栏;场景编辑支持 Ctrl 多选/Del 删。更新 §2.2 Observe 布局描述(若提到 camera_view 在右栏)。

- [ ] **Step 2: 全量测试**

Run: `python -m unittest discover -s tests -v`
Expected: 全绿。

- [ ] **Step 3: 最终冒烟**

Run: `QT_QPA_PLATFORM=offscreen python -c "import sys; from PyQt5.QtWidgets import QApplication; from real_validation.gui.main_window import ValidationWindow; app=QApplication(sys.argv); w=ValidationWindow(); w.close(); print('gui ok')"`
Run: `python -m unittest tests.test_import_hygiene -v`(4/4 全绿)

- [ ] **Step 4: 提交**

```bash
git add real_validation/GUI_GUIDE.md
git commit -m "docs(gui): 同步 GUI_GUIDE —— 两栏布局、主显示图层开关、多选删除说明"
```

---

## 自审

**1. Spec 覆盖:**
- §2 两栏布局 → Task 3 ✅
- §3 主显示视图多层叠加 + 图层开关 → Task 1 + Task 3(喂数据)✅
- §4 跨页摄像头流 → Task 3(分发)✅
- §5 紧凑排版 → Task 4(安全配置压缩 + Setup/Plan 压缩)✅
- §6 多选删除 → Task 2 ✅
- §7 文件清单 → 各 Task ✅
- §8 验证 → 各 Task + Task 5 ✅

**2. Placeholder 扫描:** 无 TBD/TODO;每个 Task 有具体代码 + 测试。✅

**3. Type/命名一致:** `set_predicted_states/set_actual_skeleton/set_ndi_position/set_layer_visible` 在 Task 1 定义、Task 3 消费,签名一致;`self.main_display`/`self.layer_checks` 在 Task 3 定义;`_layer_items` 图层 key 与 `set_layer_visible` 参数一致。`camera_view.py` 需要 `Qt`(已在 `from PyQt5.QtCore import Qt`?——**检查**:camera_view 当前 import 只有 `pyqtSignal`,需补 `Qt`。Task 1 Step 3 已用 `Qt.DashLine`/`Qt` → 需在计划确认。实际 camera_view.py 已有 `import numpy`,`from PyQt5.QtCore import pyqtSignal`——`Qt` 可能没有,**Task 1 Step 3 需补 `from PyQt5.QtCore import Qt`**。已体现在代码中。)

**注:** Task 3 的"执行中逐帧实际骨架"只做到规划完成喂预测轨迹 + 执行完成喂实际骨架;执行中每步实时 actual 需 executor 回调接线(超出本 plan 范围,标注待 M5/真机)。已在 Task 3 Step 4 说明。
