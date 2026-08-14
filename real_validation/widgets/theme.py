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
