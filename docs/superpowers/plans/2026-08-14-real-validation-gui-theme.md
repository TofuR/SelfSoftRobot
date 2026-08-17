# real_validation GUI 主题美化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `real_validation/` GUI 从默认 Qt 灰升级为 real_capture 同款"现代扁平化医疗风格"(浅蓝灰背景 + 白色圆角卡片 + 医疗青/警示红/停止紫按钮语义 + 彩色状态),保留五页 Tab 结构与全部逻辑。

**Architecture:** 新增 `real_validation/widgets/theme.py`(纯 QSS 字符串 + pyqtgraph 配置 + 状态色映射,模块级零副作用);`main_validation.py` 在 `main()` 应用 QSS、在 `ValidationWindow.__init__` 最早调用 `configure_pyqtgraph()`,并把每页控件用 QGroupBox 卡片分组、按钮设 objectName(primary/danger/accent)、顶部安全栏改状态色块。`plan_preview.py` grid alpha 微调。

**Tech Stack:** PyQt5 QSS, pyqtgraph, Python 3 stdlib. 无新依赖。

## Global Constraints

- 分支固定 `feat/real-data-transition`;提交前必须询问用户(每次提交前问,小修改不自动提交)。
- **不破坏 import 卫生**:`real_validation/__init__.py` 必须保持 stdlib-only;theme 只被 `main_validation.py` 依赖,不得加进 `real_validation/__init__.py` 或 `real_validation/widgets/__init__.py`。
- 不改五页 Tab 结构 / 布局骨架 / 信号槽接线 / 各 widget 逻辑;所有 `self.xxx` 控件属性名保持原样(处理器依赖它们)。
- 不动纯逻辑模块:`models.py` / `session.py` / `preflight.py` / `executor.py` / `perception/`。
- 部署目标是 Windows(PC),开发是 Linux → 中文字体回退链 `"Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", sans-serif`。
- 现有测试必须全绿:`test_real_validation_core` / `test_real_validation_contracts` / `test_perception_parity` / `test_perception_{quality,registration,probe}` / `test_import_hygiene`。
- 配色单一来源在 `theme.py`;按钮不再用内联 `setStyleSheet` 改色,统一 objectName + QSS。
- 无 GUI 自动化测试:页面分组靠人工目检冒烟 + 既有测试全绿。

---

## 文件结构

| 文件 | 责任 |
|---|---|
| `real_validation/widgets/theme.py` | **新建**:调色板常量、`QSS` 字符串、`PGG_OPTS`、`STATE_BADGE_COLORS`、`configure_pyqtgraph()`。模块级零副作用 |
| `tests/test_theme.py` | **新建**:锁死调色板进 QSS、pg 白底、badge 映射覆盖全部 SessionState |
| `real_validation/main_validation.py` | **修改**:应用 QSS、configure_pyqtgraph 提前、五页 QGroupBox 分组、按钮 objectName、顶部安全栏色块 |
| `real_validation/widgets/plan_preview.py` | **修改**:`showGrid` alpha 0.2 → 0.15(白底对比) |
| `real_validation/widgets/camera_view.py` | **不改**(图像视图不加网格,避免叠加视觉噪声) |

---

### Task 1: theme.py 主题模块(调色板 + QSS + pg 配置 + 状态色)

**Files:**
- Create: `real_validation/widgets/theme.py`
- Test: `tests/test_theme.py`

**Interfaces:**
- Produces: `QSS: str`、`PGG_OPTS: dict`、`STATE_BADGE_COLORS: dict[str, str]`、`configure_pyqtgraph() -> None`、调色板常量(`BG/CARD/BORDER/TEXT_MAIN/TEXT_STRONG/TEXT_MUTED/INPUT_BG/PRIMARY/DANGER/ACCENT/OK_GREEN/WARN_ORANGE` 及 hover/pressed 变体)。
- 后续 Task 2 消费:在 `main_validation.py` `from .widgets.theme import QSS, configure_pyqtgraph, STATE_BADGE_COLORS`。

- [ ] **Step 1: 写失败测试**

创建 `tests/test_theme.py`:

```python
import unittest

from real_validation.widgets.theme import (
    PRIMARY, DANGER, ACCENT, QSS, PGG_OPTS, STATE_BADGE_COLORS)


class ThemeTest(unittest.TestCase):
    def test_palette_colors_present_in_qss(self):
        for color in (PRIMARY, DANGER, ACCENT):
            self.assertIn(color, QSS)

    def test_pyqtgraph_is_white_background(self):
        self.assertEqual(PGG_OPTS["background"], "#FFFFFF")
        self.assertEqual(PGG_OPTS["foreground"], "#334E68")

    def test_badge_colors_cover_all_session_states(self):
        from real_validation.session import SessionState
        state_values = {s.value for s in SessionState}
        self.assertTrue(state_values <= set(STATE_BADGE_COLORS))

    def test_configure_is_callable(self):
        from real_validation.widgets.theme import configure_pyqtgraph
        self.assertTrue(callable(configure_pyqtgraph))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试验证失败**

Run: `python -m unittest tests.test_theme -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'real_validation.widgets.theme'`

- [ ] **Step 3: 实现 theme.py**

创建 `real_validation/widgets/theme.py`(模块级不 import pyqtgraph,`configure_pyqtgraph` 内延迟 import):

```python
"""验证工作台 GUI 主题:real_capture 同款现代扁平化医疗风格。

模块级只放字符串/纯数据结构(QSS、调色板、pg 配置 dict、状态色映射)——
**不 import pyqtgraph**,pyqtgraph 配置延迟到 configure_pyqtgraph() 内。
本模块只被 main_validation.py(GUI 入口)依赖,勿加进 __init__.py。
"""

from __future__ import annotations

# ---- 调色板(单一来源)----
BG = "#F0F4F8"            # 窗口背景(浅蓝灰)
CARD = "#FFFFFF"          # 卡片 / 文本框背景
BORDER = "#D9E2EC"        # 卡片 / 输入框边框
TAB_BG = "#E3E8EE"        # Tab 未选中底
TEXT_MAIN = "#334E68"     # 主文本
TEXT_STRONG = "#102A43"   # 输入框文字
TEXT_MUTED = "#486581"    # 次级文本 / 占位
INPUT_BG = "#F8FAFC"      # 输入框底
PRIMARY = "#2CB1BC"       # 医疗青:主动作
PRIMARY_HOVER = "#38BEC9"
PRIMARY_PRESSED = "#14919B"
DANGER = "#EF4E4E"        # 警示红:危险
DANGER_HOVER = "#F86A6A"
DANGER_PRESSED = "#E02424"
ACCENT = "#667EEA"        # 停止紫:次要动作
ACCENT_HOVER = "#7F9CF5"
ACCENT_PRESSED = "#5A67D8"
OK_GREEN = "#38A169"      # 状态 OK
WARN_ORANGE = "#F6AD55"   # 状态警告 / 过渡

FONT_FAMILY = '"Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", sans-serif'
MONO_FAMILY = 'Consolas, "Microsoft YaHei", "Noto Sans CJK SC", monospace'

_QSS_TEMPLATE = """
QMainWindow, QWidget {{
    background-color: {bg};
    color: {text};
    font-family: {font};
}}
QGroupBox {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 8px;
    margin-top: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: {text};
    font-size: 13px;
    font-weight: bold;
}}
QPushButton {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px 12px;
    color: {text};
    font-weight: bold;
}}
QPushButton:hover {{ background-color: {tab_bg}; }}
QPushButton:pressed {{ background-color: {border}; }}
QPushButton#primary {{ background-color: {primary}; color: #FFFFFF; border: none; }}
QPushButton#primary:hover {{ background-color: {primary_hover}; }}
QPushButton#primary:pressed {{ background-color: {primary_pressed}; }}
QPushButton#danger {{ background-color: {danger}; color: #FFFFFF; border: none; }}
QPushButton#danger:hover {{ background-color: {danger_hover}; }}
QPushButton#danger:pressed {{ background-color: {danger_pressed}; }}
QPushButton#accent {{ background-color: {accent}; color: #FFFFFF; border: none; }}
QPushButton#accent:hover {{ background-color: {accent_hover}; }}
QPushButton#accent:pressed {{ background-color: {accent_pressed}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    border: 1px solid {border};
    border-radius: 4px;
    padding: 4px 6px;
    background-color: {input_bg};
    color: {text_strong};
    selection-background-color: {primary};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {primary};
}}
QLabel {{ color: {text}; }}
QTableWidget {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 6px;
    gridline-color: {border};
    alternate-background-color: {input_bg};
}}
QTableWidget::item {{ padding: 2px; }}
QHeaderView::section {{
    background-color: {tab_bg};
    border: none;
    padding: 4px;
    color: {text};
    font-weight: bold;
}}
QPlainTextEdit {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 6px;
    font-family: {mono};
    color: {text};
}}
QListWidget {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 6px;
}}
QTabWidget::pane {{ border: none; background: {bg}; }}
QTabBar::tab {{
    background: {tab_bg};
    color: {text_muted};
    padding: 8px 16px;
    border-radius: 6px 6px 0 0;
    margin-right: 2px;
    font-weight: bold;
}}
QTabBar::tab:selected {{ background: {primary}; color: #FFFFFF; }}
"""

QSS = _QSS_TEMPLATE.format(
    bg=BG, card=CARD, border=BORDER, tab_bg=TAB_BG,
    text=TEXT_MAIN, text_strong=TEXT_STRONG, text_muted=TEXT_MUTED,
    input_bg=INPUT_BG,
    primary=PRIMARY, primary_hover=PRIMARY_HOVER, primary_pressed=PRIMARY_PRESSED,
    danger=DANGER, danger_hover=DANGER_HOVER, danger_pressed=DANGER_PRESSED,
    accent=ACCENT, accent_hover=ACCENT_HOVER, accent_pressed=ACCENT_PRESSED,
    font=FONT_FAMILY, mono=MONO_FAMILY,
)

# pyqtgraph 全局配置(白底对齐参考界面)。PlotWidget 实例化前调用 configure_pyqtgraph()。
PGG_OPTS = dict(background=CARD, foreground=TEXT_MAIN)

# SessionState.value -> 状态色块边框/文字色。SessionState 见 real_validation/session.py。
STATE_BADGE_COLORS = {
    "idle": TEXT_MUTED,
    "planning": DANGER,
    "ready": OK_GREEN,
    "armed": WARN_ORANGE,
    "executing": DANGER,
    "paused": WARN_ORANGE,
    "reanchor": WARN_ORANGE,
    "completed": OK_GREEN,
    "aborting": DANGER,
    "zeroed": OK_GREEN,
    "error": DANGER,
    "no_session": TEXT_MUTED,
}


def configure_pyqtgraph() -> None:
    """任何 PlotWidget 创建前调用一次。延迟 import,避免模块级副作用。"""
    import pyqtgraph as pg
    pg.setConfigOptions(antialias=True, **PGG_OPTS)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `python -m unittest tests.test_theme -v`
Expected: PASS(4 个测试)

- [ ] **Step 5: 提交**

```bash
git add real_validation/widgets/theme.py tests/test_theme.py
git commit -m "feat(real_validation): 新增 GUI 主题模块 theme.py —— real_capture 同款扁平化医疗 QSS + 白底 pg 配置 + 状态色映射"
```

---

### Task 2: 接入主题 + 顶部安全栏状态色块 + 按钮 objectName

**Files:**
- Modify: `real_validation/main_validation.py`

**Interfaces:**
- Consumes: Task 1 的 `QSS` / `configure_pyqtgraph` / `STATE_BADGE_COLORS`。
- Produces: `main()` 应用 QSS;`ValidationWindow.__init__` 最早调用 `configure_pyqtgraph()`;`_refresh()` 更新 `state_label` 色块样式。

- [ ] **Step 1: import theme 并应用**

在 `main_validation.py` import 区(`from .widgets import ...` 之后)加:

```python
from .widgets.theme import QSS, STATE_BADGE_COLORS, configure_pyqtgraph
```

在 `ValidationWindow.__init__` 的 `super().__init__()` 之后、`self._build_ui()` 之前加:

```python
configure_pyqtgraph()          # 任何 PlotWidget 之前,保证白底全局生效
```

在 `main()` 里加:

```python
def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    window = ValidationWindow(); window.show()
    return app.exec_()
```

- [ ] **Step 2: 顶部安全栏改造**

在 `_build_ui` 中,`self.state_label` 不再内联样式,`abort_button` 改 objectName `danger`:

```python
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
```

(删除原来的 `self.state_label.setStyleSheet("font-weight:600;padding:6px")` 和 `self.abort_button.setStyleSheet("background:#C53030;color:white;font-weight:600")`。)

- [ ] **Step 3: `_refresh` 更新状态色块**

在 `_refresh()` 中,设置 `state_label` 文本后追加色块样式(保留原有按钮 enable/disable 逻辑不变):

```python
def _refresh(self) -> None:
    state = self.session.state.value if self.session else "no_session"
    run = self.session.run_dir.name if self.session else "-"
    self.state_label.setText(f"Run: {run}    State: {state}    Hardware: MOCK")
    color = STATE_BADGE_COLORS.get(state, STATE_BADGE_COLORS["no_session"])
    self.state_label.setStyleSheet(
        f"background:#FFFFFF;border:2px solid {color};border-radius:12px;"
        f"padding:4px 12px;color:{color};font-weight:bold;")
    # ...原有 self.arm_button.setEnabled(...) 等保持不动
```

- [ ] **Step 4: 冒烟验证**

Run: `python -m real_validation.main_validation`(有显示环境直接看;无显示环境用 `QT_QPA_PLATFORM=offscreen python -c "import sys; from PyQt5.QtWidgets import QApplication; from real_validation.main_validation import ValidationWindow; app=QApplication(sys.argv); w=ValidationWindow(); print(w.state_label.text()); w.close()"`)
Expected: 白底界面;顶部"Run: - State: no_session Hardware: MOCK"灰边色块;归零/中止红底;窗口可正常构造与关闭,无异常。

- [ ] **Step 5: 跑既有测试确认未破坏**

Run: `python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts tests.test_perception_parity tests.test_import_hygiene -v`
Expected: 全绿(import 卫生尤其要过 —— theme 未进 `__init__` 闭包)。

- [ ] **Step 6: 提交**

```bash
git add real_validation/main_validation.py
git commit -m "feat(real_validation): GUI 接入主题 —— main() 应用 QSS + pg 白底提前配置 + 顶部安全栏状态色块 + 归零/中止危险红"
```

---

### Task 3: Setup 页 QGroupBox 分组

**Files:**
- Modify: `real_validation/main_validation.py`(`_setup_page`)

**Interfaces:**
- Consumes: Task 2 的 `QSS`(已全局生效,本任务只需分组)。
- Produces: `_setup_page` 返回的 widget 内部用 QGroupBox 分三卡:**实验与运行** / **模型与部署契约** / **安全配置(六通道 kPa / kPa·s⁻¹)**。所有 `self.*` 属性名不变。

- [ ] **Step 1: 补充 QGroupBox import**

`main_validation.py` 顶部 `from PyQt5.QtWidgets import (...)` 列表**当前缺 QGroupBox**,把它加进该 import 元组(按字母序,放 QFileDialog 后):

```python
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
```

- [ ] **Step 2: 重构 `_setup_page`**

替换整个 `_setup_page` 方法体:

```python
def _setup_page(self) -> QWidget:
    page = QWidget(); root = QVBoxLayout(page)

    # 卡1:实验与运行
    gb_exp = QGroupBox("实验与运行")
    exp = QVBoxLayout(gb_exp); exp.setContentsMargins(12, 14, 12, 12)
    self.run_root = QLineEdit(str(APP_DIR / "runs"))
    exp.addWidget(self._path_row(self.run_root, True))
    buttons = QHBoxLayout()
    create = QPushButton("New Experiment"); create.setObjectName("primary")
    create.clicked.connect(self._new_session)
    replay = QPushButton("Open Run (Replay)"); replay.clicked.connect(self._open_replay)
    buttons.addWidget(create); buttons.addWidget(replay); buttons.addStretch()
    exp.addLayout(buttons)
    root.addWidget(gb_exp)

    # 卡2:模型与部署契约
    gb_model = QGroupBox("模型与部署契约")
    m = QVBoxLayout(gb_model); m.setContentsMargins(12, 14, 12, 12)
    form = QFormLayout()
    self.checkpoint = QLineEdit(str(APP_DIR / "checkpoints" / "current" / "best_model.pt"))
    self.data_dir = QLineEdit(str(APP_DIR / "data"))
    self.k_safe = QSpinBox(); self.k_safe.setRange(0, 100000)
    self.k_safe.setSpecialValueText("未认证")
    self.device = QComboBox(); self.device.addItems(["cpu", "cuda"])
    form.addRow("Checkpoint", self._path_row(self.checkpoint, False))
    form.addRow("训练数据目录", self._path_row(self.data_dir, True))
    form.addRow("K_safe", self.k_safe)
    form.addRow("模型设备", self.device)
    m.addLayout(form)
    load = QPushButton("Load Model"); load.setObjectName("primary")
    load.clicked.connect(self._load_model)
    m.addWidget(load)
    self.model_summary = QPlainTextEdit(); self.model_summary.setReadOnly(True)
    m.addWidget(self.model_summary, 1)
    root.addWidget(gb_model, 1)

    # 卡3:安全配置(六通道 kPa / kPa·s⁻¹)
    gb_safety = QGroupBox("安全配置(六通道 kPa / kPa·s⁻¹)")
    s = QVBoxLayout(gb_safety); s.setContentsMargins(12, 14, 12, 12)
    self.safety_table = QTableWidget(6, 5)
    self.safety_table.setHorizontalHeaderLabels(["min", "max", "rise/s", "fall/s", "initial"])
    self._safety_cells = []
    for channel in range(6):
        self.safety_table.setVerticalHeaderItem(channel, QTableWidgetItem(f"ch{channel}"))
        row = []
        for column, default in enumerate((0.0, 150.0, 100.0, 100.0, 0.0)):
            cell = QDoubleSpinBox(); cell.setRange(0, 500); cell.setDecimals(1)
            cell.setValue(default); self.safety_table.setCellWidget(channel, column, cell)
            row.append(cell)
        self._safety_cells.append(row)
    s.addWidget(self.safety_table)
    apply_safety = QPushButton("应用安全配置并使旧计划失效")
    apply_safety.setObjectName("primary")
    apply_safety.clicked.connect(self._apply_safety)
    s.addWidget(apply_safety)
    root.addWidget(gb_safety)
    return page
```

> 原 `_build_ui` 中的 `apply_safety` 局部变量改在 `_setup_page` 内部创建(原实现它在 `_setup_page` 里,未变)。`_path_row` 内部创建的是局部 holder,不变。

- [ ] **Step 3: 冒烟验证**

Run: `QT_QPA_PLATFORM=offscreen python -c "import sys; from PyQt5.QtWidgets import QApplication; from real_validation.main_validation import ValidationWindow; app=QApplication(sys.argv); w=ValidationWindow(); w.close(); print('constructed ok')"`
(简化为人工目检:有显示环境运行 `python -m real_validation.main_validation`,确认 Setup 页三张白卡、卡标题清晰、六通道表可编辑。)

- [ ] **Step 4: 跑既有测试**

Run: `python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts -v`
Expected: 全绿(Setup 页无逻辑改动)。

- [ ] **Step 5: 提交**

```bash
git add real_validation/main_validation.py
git commit -m "feat(real_validation): Setup 页 QGroupBox 分组(实验与运行/模型与部署契约/安全配置六通道)"
```

---

### Task 4: Observe & Scene 页 QGroupBox 分组

**Files:**
- Modify: `real_validation/main_validation.py`(`_observe_page`)

**Interfaces:**
- Produces: `_observe_page` 分四卡:**离线锚定** / **目标与障碍** / **实时相机与 Warmup** / **场景编辑**。工具按钮(select/点加目标/点加障碍)归"实时相机与 Warmup"卡。所有 `self.*` 属性名、信号连接、`_set_tool` / `_add_primitive` / `_apply_scene_edit` / `_start_camera` / `_warmup` / `_camera_anchor` 的绑定不变。

- [ ] **Step 1: 重构 `_observe_page`**

替换整个 `_observe_page` 方法体(保留全部现有 `self.*` 与信号连接,只重新组织进 QGroupBox):

```python
def _observe_page(self) -> QWidget:
    page = QWidget(); root = QVBoxLayout(page)

    # 卡1:离线锚定
    gb_off = QGroupBox("离线锚定")
    off = QVBoxLayout(gb_off); off.setContentsMargins(12, 14, 12, 12)
    buttons = QHBoxLayout()
    anchor = QPushButton("加载 anchor.json"); anchor.clicked.connect(self._load_anchor)
    scene = QPushButton("加载 scene.json"); scene.clicked.connect(self._load_scene)
    buttons.addWidget(anchor); buttons.addWidget(scene); buttons.addStretch()
    off.addLayout(buttons)
    offline = QFormLayout()
    self.anchor_npz = QLineEdit()
    self.anchor_index = QSpinBox(); self.anchor_index.setRange(0, 100000000)
    load_npz = QPushButton("从 NPZ 建立 Anchor"); load_npz.setObjectName("primary")
    load_npz.clicked.connect(self._load_anchor_npz)
    offline.addRow("Transition NPZ", self._path_row(self.anchor_npz, False))
    index_row = QHBoxLayout(); index_row.addWidget(self.anchor_index); index_row.addWidget(load_npz)
    offline.addRow("帧索引(必须已有完整 H)", index_row)
    off.addLayout(offline)
    root.addWidget(gb_off)

    # 卡2:目标与障碍
    gb_tgt = QGroupBox("目标与障碍")
    t = QVBoxLayout(gb_tgt); t.setContentsMargins(12, 14, 12, 12)
    target_form = QFormLayout()
    self.target_x = QDoubleSpinBox(); self.target_x.setRange(-100000, 100000)
    self.target_y = QDoubleSpinBox(); self.target_y.setRange(-100000, 100000)
    self.target_radius = QDoubleSpinBox(); self.target_radius.setRange(0, 100000)
    self.obstacle_x = QDoubleSpinBox(); self.obstacle_x.setRange(-100000, 100000)
    self.obstacle_y = QDoubleSpinBox(); self.obstacle_y.setRange(-100000, 100000)
    self.obstacle_radius = QDoubleSpinBox(); self.obstacle_radius.setRange(0.01, 100000)
    target_row = QHBoxLayout()
    target_row.addWidget(self.target_x); target_row.addWidget(self.target_y)
    target_row.addWidget(self.target_radius)
    target_button = QPushButton("设置末端目标"); target_button.setObjectName("primary")
    target_button.clicked.connect(self._set_target)
    target_row.addWidget(target_button)
    target_form.addRow("目标 x / y / 半径 (model)", target_row)
    obstacle_row = QHBoxLayout()
    obstacle_row.addWidget(self.obstacle_x); obstacle_row.addWidget(self.obstacle_y)
    obstacle_row.addWidget(self.obstacle_radius)
    obstacle_button = QPushButton("添加圆障碍"); obstacle_button.setObjectName("accent")
    obstacle_button.clicked.connect(self._add_obstacle)
    obstacle_row.addWidget(obstacle_button)
    target_form.addRow("障碍 x / y / 半径 (model)", obstacle_row)
    t.addLayout(target_form)
    root.addWidget(gb_tgt)

    # 卡3:实时相机与 Warmup(含相机视图交互工具)
    gb_live = QGroupBox("实时相机与 Warmup")
    live = QVBoxLayout(gb_live); live.setContentsMargins(12, 14, 12, 12)
    live_buttons = QHBoxLayout()
    self.camera_btn = QPushButton("Start Camera (Mock)"); self.camera_btn.setObjectName("primary")
    self.camera_btn.clicked.connect(self._start_camera)
    self.camera_anchor_btn = QPushButton("从相机取流锚定"); self.camera_anchor_btn.setObjectName("primary")
    self.camera_anchor_btn.setEnabled(False)
    self.camera_anchor_btn.clicked.connect(self._camera_anchor)
    self.warmup_btn = QPushButton("Warmup(填动作历史)"); self.warmup_btn.setObjectName("accent")
    self.warmup_btn.setEnabled(False)
    self.warmup_btn.clicked.connect(self._warmup)
    live_buttons.addWidget(self.camera_btn); live_buttons.addWidget(self.camera_anchor_btn)
    live_buttons.addWidget(self.warmup_btn); live_buttons.addStretch()
    live.addLayout(live_buttons)
    tool_row = QHBoxLayout()
    self.tool_select_btn = QPushButton("select")
    self.tool_select_btn.clicked.connect(lambda: self._set_tool("select"))
    self.tool_target_btn = QPushButton("点加目标")
    self.tool_target_btn.clicked.connect(lambda: self._set_tool("add_target"))
    self.tool_obstacle_btn = QPushButton("点加障碍")
    self.tool_obstacle_btn.clicked.connect(lambda: self._set_tool("add_obstacle"))
    tool_row.addWidget(QLabel("工具:")); tool_row.addWidget(self.tool_select_btn)
    tool_row.addWidget(self.tool_target_btn); tool_row.addWidget(self.tool_obstacle_btn)
    tool_row.addStretch()
    live.addLayout(tool_row)
    root.addWidget(gb_live)

    # 卡4:场景编辑(相机视图 + 场景列表)
    gb_scene = QGroupBox("场景编辑")
    sc = QVBoxLayout(gb_scene); sc.setContentsMargins(12, 14, 12, 12)
    self.camera_view = CameraViewWidget()
    self.scene_editor = SceneEditorPanel()
    split = QSplitter(); split.addWidget(self.camera_view); split.addWidget(self.scene_editor)
    split.setSizes([520, 260])
    sc.addWidget(split, 1)
    self.anchor_status = QLabel("未锚定")
    sc.addWidget(self.anchor_status)
    root.addWidget(gb_scene, 1)

    # 绑定(保持原样)
    self.camera_view.target_picked.connect(self._add_primitive)
    self.camera_view.obstacle_picked.connect(self._add_primitive)
    self.scene_editor.scene_edited.connect(self._apply_scene_edit)
    self._camera_thread = None
    self._latest_frame = None
    self._action_history = []
    return page
```

- [ ] **Step 2: 冒烟验证**

Run: `python -m real_validation.main_validation`(人工目检:Observe 页四张白卡;工具按钮在"实时相机与 Warmup"卡;camera_view 白底显示合成帧;点"点加目标"后点击图像能加原语)。

- [ ] **Step 3: 跑既有测试**

Run: `python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts -v`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add real_validation/main_validation.py
git commit -m "feat(real_validation): Observe & Scene 页 QGroupBox 分组(离线锚定/目标与障碍/实时相机与Warmup/场景编辑)"
```

---

### Task 5: Plan 页 QGroupBox 分组

**Files:**
- Modify: `real_validation/main_validation.py`(`_plan_page`)

**Interfaces:**
- Produces: `_plan_page` 分两卡:**规划参数** / **规划与预检**,下方 PlanPreviewWidget 占满剩余。所有 `self.*` 与 `_start_planning` / `_cancel_planning` / `_load_plan` / `_run_preflight` 绑定不变。

- [ ] **Step 1: 重构 `_plan_page`**

替换整个 `_plan_page` 方法体:

```python
def _plan_page(self) -> QWidget:
    page = QWidget(); root = QVBoxLayout(page)

    # 卡1:规划参数
    gb_param = QGroupBox("规划参数")
    p = QVBoxLayout(gb_param); p.setContentsMargins(12, 14, 12, 12)
    form = QFormLayout()
    self.plan_k = QSpinBox(); self.plan_k.setRange(1, 10000); self.plan_k.setValue(20)
    self.plan_iter = QSpinBox(); self.plan_iter.setRange(1, 100000); self.plan_iter.setValue(400)
    self.plan_restarts = QSpinBox(); self.plan_restarts.setRange(1, 32); self.plan_restarts.setValue(4)
    self.plan_dt = QDoubleSpinBox(); self.plan_dt.setRange(0.01, 60); self.plan_dt.setValue(0.2)
    self.plan_dt.setDecimals(3)
    self.channel_map = QLineEdit("0")
    form.addRow("K", self.plan_k); form.addRow("优化迭代", self.plan_iter)
    form.addRow("多起点", self.plan_restarts); form.addRow("动作周期(s)", self.plan_dt)
    form.addRow("模型维度→硬件通道", self.channel_map)
    p.addLayout(form)
    root.addWidget(gb_param)

    # 卡2:规划与预检
    gb_act = QGroupBox("规划与预检")
    a = QVBoxLayout(gb_act); a.setContentsMargins(12, 14, 12, 12)
    buttons = QHBoxLayout()
    generate = QPushButton("运行 OpenLoop Planner"); generate.setObjectName("primary")
    generate.clicked.connect(self._start_planning)
    cancel = QPushButton("取消规划"); cancel.setObjectName("accent")
    cancel.clicked.connect(self._cancel_planning)
    load = QPushButton("导入 plan.json"); load.clicked.connect(self._load_plan)
    check = QPushButton("运行 Preflight"); check.clicked.connect(self._run_preflight)
    buttons.addWidget(generate); buttons.addWidget(cancel); buttons.addWidget(load)
    buttons.addWidget(check); buttons.addStretch()
    a.addLayout(buttons)
    self.plan_summary = QPlainTextEdit(); self.plan_summary.setReadOnly(True)
    self.plan_summary.setPlaceholderText("异步 shooting planner 与交互式候选预览将在此页接入。")
    a.addWidget(self.plan_summary)
    root.addWidget(gb_act)

    self.plan_preview = PlanPreviewWidget(); root.addWidget(self.plan_preview, 1)
    return page
```

- [ ] **Step 2: 冒烟验证**

Run: `python -m real_validation.main_validation`(人工目检:Plan 页两卡 + 白底 PlanPreviewWidget;点"运行 OpenLoop Planner"在无模型时弹"规划前需要 session、模型和 anchor"错误提示)。

- [ ] **Step 3: 跑既有测试**

Run: `python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts -v`
Expected: 全绿。

- [ ] **Step 4: 提交**

```bash
git add real_validation/main_validation.py
git commit -m "feat(real_validation): Plan 页 QGroupBox 分组(规划参数/规划与预检)+ 主动作医疗青/取消紫"
```

---

### Task 6: Execute + Results 页 QGroupBox 分组

**Files:**
- Modify: `real_validation/main_validation.py`(`_execute_page`, `_results_page`)

**Interfaces:**
- Produces: `_execute_page` 分两卡:**执行控制** / **执行日志**;`_results_page` 一张卡:**结果与指标**。所有 `self.*` 与 `_arm` / `_execute` / `_pause` / `_resume` 绑定不变。

- [ ] **Step 1: 重构 `_execute_page`**

替换整个 `_execute_page` 方法体:

```python
def _execute_page(self) -> QWidget:
    page = QWidget(); root = QVBoxLayout(page)

    # 卡1:执行控制
    gb_ctrl = QGroupBox("执行控制")
    c = QHBoxLayout(gb_ctrl); c.setContentsMargins(12, 14, 12, 12)
    self.arm_button = QPushButton("Arm / Confirm"); self.arm_button.setObjectName("primary")
    self.arm_button.clicked.connect(self._arm)
    self.execute_button = QPushButton("Mock Execute"); self.execute_button.setObjectName("primary")
    self.execute_button.clicked.connect(self._execute)
    self.pause_button = QPushButton("Pause"); self.pause_button.setObjectName("accent")
    self.pause_button.clicked.connect(self._pause)
    self.resume_button = QPushButton("Resume"); self.resume_button.setObjectName("accent")
    self.resume_button.clicked.connect(self._resume)
    for button in (self.arm_button, self.execute_button, self.pause_button, self.resume_button):
        c.addWidget(button)
    c.addStretch()
    root.addWidget(gb_ctrl)

    # 卡2:执行日志
    gb_log = QGroupBox("执行日志")
    l = QVBoxLayout(gb_log); l.setContentsMargins(12, 14, 12, 12)
    self.execution_log = QPlainTextEdit(); self.execution_log.setReadOnly(True)
    l.addWidget(self.execution_log)
    root.addWidget(gb_log, 1)
    return page
```

- [ ] **Step 2: 重构 `_results_page`**

替换整个 `_results_page` 方法体:

```python
def _results_page(self) -> QWidget:
    page = QWidget(); root = QVBoxLayout(page)
    gb = QGroupBox("结果与指标")
    g = QVBoxLayout(gb); g.setContentsMargins(12, 14, 12, 12)
    self.results = QPlainTextEdit(); self.results.setReadOnly(True)
    self.results.setPlaceholderText("执行记录保存在 run/execution.csv；自动指标将在后续接入。")
    g.addWidget(self.results)
    root.addWidget(gb)
    return page
```

- [ ] **Step 3: 冒烟验证**

Run: `python -m real_validation.main_validation`(人工目检:Execute 页两卡,四个执行按钮语义色正确;Results 页一张卡)。

- [ ] **Step 4: 跑既有测试**

Run: `python -m unittest tests.test_real_validation_core tests.test_real_validation_contracts -v`
Expected: 全绿。

- [ ] **Step 5: 提交**

```bash
git add real_validation/main_validation.py
git commit -m "feat(real_validation): Execute/Results 页 QGroupBox 分组 + 执行按钮语义色"
```

---

### Task 7: plan_preview 白底网格微调

**Files:**
- Modify: `real_validation/widgets/plan_preview.py:20-22`

**Interfaces:**
- Produces: 两个 PlotWidget 的 `showGrid` alpha 从 0.2 降到 0.15(白底上更柔和)。无签名变化。

- [ ] **Step 1: 改 alpha**

```python
self.shape_plot.showGrid(x=True, y=True, alpha=0.15)
self.action_plot.showGrid(x=True, y=True, alpha=0.15)
```

- [ ] **Step 2: 冒烟验证**

Run: `python -m real_validation.main_validation`(人工目检:Plan 预览两个白底图,网格浅灰,曲线颜色 `#2CB1BC` 等清晰)。

- [ ] **Step 3: 提交**

```bash
git add real_validation/widgets/plan_preview.py
git commit -m "feat(real_validation): plan_preview 白底网格 alpha 0.2→0.15 对齐主题"
```

---

### Task 8: 全量验证 + 冒烟

**Files:**
- 无代码改动(验证任务)

- [ ] **Step 1: 全量单测**

Run:
```bash
python -m unittest discover -s tests -v
```
Expected: 全绿(含 4 个 test_theme、20 契约、55 core+contracts、parity、import 卫生、感知 quality/registration/probe)。

- [ ] **Step 2: GUI 全流程冒烟**

Run: `python -m real_validation.main_validation`
人工过一遍五页:Setup 三卡(六通道表可编辑)→ Observe 四卡(Start Camera 合成帧 + 点加目标)→ Plan 两卡 → Execute 两卡 → Results 一卡;顶部状态色块随 New Experiment(创建 run)→ no_session 灰、idle 灰、规划/执行时变红/橙。

- [ ] **Step 3: import 卫生重点复核**

Run: `python -m unittest tests.test_import_hygiene -v`
Expected: 4 个测试全过 —— 确认 theme 未污染 `real_validation` / `real_validation.perception` 闭包。

- [ ] **Step 4: 汇总提交(如有遗留)**

检查 `git status`;如有未提交改动,先询问用户再提交。

---

## 自审

**1. Spec 覆盖:**
- §3 配色单一来源 → Task 1(theme.py 调色板)✅
- §4 按钮语义(primary/danger/accent)+ 内联样式清理 → Task 2/4/5/6 设 objectName,删除 `#C53030` 内联 ✅
- §4.2 QGroupBox 卡片 → Task 3/4/5/6 ✅
- §5 顶部安全栏状态色块 + state 映射 → Task 2(`STATE_BADGE_COLORS` 覆盖 `SessionState` 全部 11 值 + no_session)✅
- §6 页面分组表 → Task 3(Setup)/Task 4(Observe)/Task 5(Plan)/Task 6(Execute+Results),分组与设计表逐行对应 ✅
- §7 绘图白底 → Task 1(`configure_pyqtgraph`) + Task 7(alpha 微调)✅
- §9 验证(冒烟 + 测试全绿 + import 卫生)→ Task 8 ✅
- §10 风险:theme 不进 `__init__` → 各任务只 `from .widgets.theme import`(widgets/__init__ 未改)✅

**2. Placeholder 扫描:** 无 TBD/TODO;每个任务含可直接复制的完整代码;信号连接/属性名与现有 `main_validation.py` 逐字核对。✅

**3. Type/命名一致:** `QSS`/`configure_pyqtgraph`/`STATE_BADGE_COLORS` 在 Task 1 定义、Task 2 消费,签名一致;`_setup_page`/`_observe_page`/`_plan_page`/`_execute_page`/`_results_page` 方法名与现有代码一致;`self.*` 属性全部保持原样。`QGroupBox` 在 `main_validation.py` 顶部 import 元组中**原本缺失**,已由 Task 3 Step 1 补入 import。✅
