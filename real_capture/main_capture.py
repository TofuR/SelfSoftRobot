# main_capture.py
"""六通道气压阀 + NDI 末端 + 相机 同步采集 GUI（动作门控）。

功能：
  - 连接 Modbus（2 组 × 3 通道 = 6 路电流型比例阀，0–500kPa）；
  - 6 通道目标/min/max 控制（**单通道模式**：把其余 5 路 min=max=0 即只动 1 路，向后兼容）；
  - 连接 NDI Aurora 电磁导航，实时末端 x/y/z + 轨迹；
  - 相机实时预览（RealSense，可 mock）；
  - 动作门控采集：动作每 `action_interval`（默认 0.2s）下发一次，等 `settle`（默认 0.19s）
    软臂稳定后抓一帧 + NDI 末端，三者同索引落盘；
  - 后处理：生成 tip.npz + 调 capture_to_npz 出训练用 .npz / 导出汇总 CSV；
  - 设置持久化（real_capture_config.ini，跨机器不踩绝对路径坑）；
  - **mock 任意组合**：--mock-cam / --mock-valve / --mock-ndi 单选或混选；--mock = 三个全开。

运行示例（在本文件所在目录执行）：
    # 1) 全 mock（无任何硬件，先验证 GUI 与整条链路）
    python main_capture.py --mock

    # 2) 真机全用（两组 Modbus 串口 + NDI 串口 + RealSense）
    python main_capture.py --group1 COM3 --group2 COM46 --ndi COM9

    # 3) 只调单通道（其余 5 路 min=max=0），假 NDI
    python main_capture.py --mock-ndi --group1 COM3 --group2 COM46

    # 任意组合皆可；--mock 等价于 --mock-cam --mock-valve --mock-ndi

依赖：PyQt5, pyqtgraph, numpy,（真机还需 pyrealsense2/pyserial/scipy/scikit-surgerynditracker）。
"""
from __future__ import annotations

import configparser
import os
import subprocess
import sys
import threading
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QPushButton, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget)

from recorder import ValveRecorder, build_ndi_tip_npz, export_summary_csv
from realsense_cam import RealSenseCam
from valve_control import N_CHAN, P_MAX, P_MIN

# 现代白底风格（对齐旧 main_capture.py）
pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", "#FFFFFF")
pg.setConfigOption("foreground", "#334E68")

PLOT_LEN = 300
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # py 文件所在目录；保存路径默认基于此

# 6 路曲线颜色
_CH_COLORS = ["#2CB1BC", "#667EEA", "#EF4E4E", "#F6AD55", "#68D391", "#B388FF"]

# 单通道模式下，选中通道的默认 max（保守值；inactive 通道 min=max=0）
DEFAULT_MAX = 200.0


class _CleanAxis(pg.AxisItem):
    """刻度按间距取整 + 限 6 位，去掉浮点噪声（轴很小时出现 0.30000000000000004）。"""

    def tickStrings(self, values, scale, spacing):
        if getattr(self, "logMode", False):       # 与上游对齐：对数轴走 10^n 格式（未来若启用 setLogMode，review #5）
            return self.logTickStrings(values, scale, spacing)
        if spacing <= 0 or not values:
            return [""] * len(values)
        places = min(6, max(0, int(np.ceil(-np.log10(spacing * scale)))))
        fmt = "{:.%df}" % places
        out = []
        for v in values:
            vs = v * scale
            if abs(vs) < 1e-4 or abs(vs) >= 1e5:
                out.append("%g" % vs)
            else:
                out.append(fmt.format(round(vs, places)))
        return out


def _detect_project_root() -> str:
    """向上查找包含 scripts/real/capture_to_npz.py 的目录（用于"生成 npz"）。"""
    d = SCRIPT_DIR
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "scripts", "real", "capture_to_npz.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.getcwd()


class CaptureWindow(QMainWindow):
    log_signal = pyqtSignal(str)          # 子进程(capture_to_npz)输出回 GUI 线程

    def __init__(self, mock_cam=False, mock_valve=False, mock_ndi=False,
                 group1="COM3", group2="COM46", ndi_port="COM9", baudrate=9600,
                 slave_addr=1, fps=30, ndi_count=2):
        super().__init__()
        self.mock_cam = bool(mock_cam)
        self.mock_valve = bool(mock_valve)
        self.mock_ndi = bool(mock_ndi)
        self.ndi_count = max(1, int(ndi_count))
        self.fps = int(fps)
        self.project_root = _detect_project_root()
        self._cfg_path = os.path.join(SCRIPT_DIR, "real_capture_config.ini")
        self._npz_proc = None

        self.setWindowTitle("六通道阀 · NDI · 相机  同步采集  ·  "
                            + ("MOCK" if (mock_cam or mock_valve or mock_ndi) else "HARDWARE"))
        self.resize(1380, 880)

        # ---- 硬件/核心（每个组件按需 mock，可任意组合）----
        self.cam = RealSenseCam(mock=mock_cam, fps=fps)

        if mock_ndi:
            from hardware_threads import MockNdiThread
            self.ndi = MockNdiThread(ndi_count=self.ndi_count)
        else:
            from hardware_threads import NdiThread
            self.ndi = NdiThread(port=ndi_port, ndi_count=self.ndi_count)

        if mock_valve:
            from valve_control import MockValveController
            self.controller = MockValveController()
        else:
            from valve_control import ValveController
            self.controller = ValveController({1: group1, 2: group2}, baudrate, slave_addr)

        self.core = ValveRecorder(self.cam, self.ndi, self.controller)
        self.core.set_ndi_count(self.ndi_count)

        # ---- 数据缓存（曲线）----
        self._p_buf = np.zeros((N_CHAN, PLOT_LEN))
        self._xy_trail = np.zeros((2, 0))        # NDI XY 轨迹（动态增长，截断到 PLOT_LEN）

        # 6 通道控件
        self._target_sb = []
        self._min_sb = []
        self._max_sb = []
        self._guard = False                     # 程序化 setValue 时抑制 valueChanged 回灌（防反馈循环）
        # 持久化缓存：用户真正配置的 6 路 lo/hi（独立于单通道模式的显示归零），
        # 避免切到单通道把 inactive 显示归零后 _save_config 把配置永久写成 0（review #2）。
        self._cfg_lo = [P_MIN] * N_CHAN
        self._cfg_hi = [DEFAULT_MAX] * N_CHAN
        self._cfg_rise = [100.0] * N_CHAN
        self._cfg_fall = [100.0] * N_CHAN
        self.max_frame_age = 0.5
        self.max_ndi_age = 0.5
        self._rise_sb = []
        self._fall_sb = []

        self._build_ui()
        self._connect_core()
        self.log_signal.connect(self._log)
        self._load_config()
        self._on_active_changed()                # 按 restored 主通道：缓存→显示→锁定→曲线显隐

        # CLI 覆盖（仅端口类参数）
        self.le_g1.setText(group1); self.le_g2.setText(group2)
        self.le_ndi.setText(ndi_port)
        self.sb_baud.setValue(int(baudrate)); self.sb_slave.setValue(int(slave_addr))

        # 相机立即开（预览）；mock 阀两组都连，方便直接验证 six-channel/all 模式
        self.cam.start()
        if mock_valve:
            self.controller.connect_group(1)
            self.controller.connect_group(2)
        if mock_ndi:
            self.ndi.start()
            self._sync_ndi_button()

        mocked = [n for n, m in (("相机", mock_cam), ("阀/Modbus", mock_valve), ("NDI", mock_ndi)) if m]
        if mocked:
            self._log(f"就绪。MOCK 组件：{' + '.join(mocked)}（这些用假数据，其余用真硬件）")
        else:
            self._log("就绪。（全部真硬件，需先连接 Modbus / NDI）")
        self._log("单通道用法：把不用的 5 路 min=max=0，只给目标通道设范围。")

    # ===================== UI 构建 =====================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        split = QSplitter(Qt.Horizontal)

        # ---------- 左：控制面板 ----------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)

        # ---- Modbus 连接（组1/组2 独立 连接/断开）----
        gb = QGroupBox("Modbus 连接（2 组 × 3 通道 = 6 路，4–20mA；组1=ch0-2，组2=ch3-5）")
        g = QGridLayout(gb)
        g.addWidget(QLabel("组1串口"), 0, 0); self.le_g1 = QLineEdit("COM3"); g.addWidget(self.le_g1, 0, 1)
        g.addWidget(QLabel("组2串口"), 0, 2); self.le_g2 = QLineEdit("COM46"); g.addWidget(self.le_g2, 0, 3)
        g.addWidget(QLabel("波特"), 1, 0); self.sb_baud = QDoubleSpinBox(); self.sb_baud.setRange(1200, 115200); self.sb_baud.setValue(9600); self.sb_baud.setDecimals(0); g.addWidget(self.sb_baud, 1, 1)
        g.addWidget(QLabel("从站"), 1, 2); self.sb_slave = QDoubleSpinBox(); self.sb_slave.setRange(1, 247); self.sb_slave.setValue(1); self.sb_slave.setDecimals(0); g.addWidget(self.sb_slave, 1, 3)
        row = QHBoxLayout()
        self.btn_g1 = QPushButton("组1 连接"); self.btn_g1.setStyleSheet("background:#2CB1BC;color:white")
        self.btn_g1.clicked.connect(lambda: self._toggle_group(1))
        self.btn_g2 = QPushButton("组2 连接"); self.btn_g2.setStyleSheet("background:#2CB1BC;color:white")
        self.btn_g2.clicked.connect(lambda: self._toggle_group(2))
        self.lbl_conn = QLabel("未连接"); self.lbl_conn.setStyleSheet("color:#888")
        row.addWidget(self.btn_g1); row.addWidget(self.btn_g2); row.addWidget(self.lbl_conn, 1)
        g.addLayout(row, 2, 0, 1, 4)
        ll.addWidget(gb)

        # ---- 6 通道阀控制 ----
        gb = QGroupBox("阀控制（目标/min/max，kPa；rise/fall 为命令速率上限 kPa/s，0=不限速）")
        g = QGridLayout(gb)
        g.addWidget(QLabel("通道"), 0, 0); g.addWidget(QLabel("目标"), 0, 1)
        g.addWidget(QLabel("min"), 0, 2); g.addWidget(QLabel("max"), 0, 3)
        g.addWidget(QLabel("rise/s"), 0, 4); g.addWidget(QLabel("fall/s"), 0, 5)
        for i in range(N_CHAN):
            g.addWidget(QLabel(f"ch{i}"), i + 1, 0)
            t = QDoubleSpinBox(); t.setRange(P_MIN, P_MAX); t.setDecimals(1); t.setSingleStep(5.0)
            mn = QDoubleSpinBox(); mn.setRange(P_MIN, P_MAX); mn.setDecimals(1); mn.setSingleStep(5.0)
            mx = QDoubleSpinBox(); mx.setRange(P_MIN, P_MAX); mx.setDecimals(1); mx.setSingleStep(5.0)
            rise = QDoubleSpinBox(); rise.setRange(0.0, 5000.0); rise.setDecimals(1); rise.setSingleStep(10.0); rise.setValue(100.0)
            fall = QDoubleSpinBox(); fall.setRange(0.0, 5000.0); fall.setDecimals(1); fall.setSingleStep(10.0); fall.setValue(100.0)
            if i == 0:                                   # 默认 ch0 有范围，其余钉 0（单通道）
                t.setValue(0.0); mn.setValue(0.0); mx.setValue(DEFAULT_MAX)
            else:
                t.setValue(0.0); mn.setValue(0.0); mx.setValue(0.0)
            g.addWidget(t, i + 1, 1); g.addWidget(mn, i + 1, 2); g.addWidget(mx, i + 1, 3)
            g.addWidget(rise, i + 1, 4); g.addWidget(fall, i + 1, 5)
            # 先设值再接线：初始化 setValue 不触发 _on_range_changed（此时 cb_active 尚未创建）
            t.valueChanged.connect(self._on_target_changed)
            mn.valueChanged.connect(self._on_range_changed)   # min/max 改动实时同步驱动（录制中也生效）
            mx.valueChanged.connect(self._on_range_changed)
            rise.valueChanged.connect(self._on_rate_changed)
            fall.valueChanged.connect(self._on_rate_changed)
            self._target_sb.append(t); self._min_sb.append(mn); self._max_sb.append(mx)
            self._rise_sb.append(rise); self._fall_sb.append(fall)
        row = QHBoxLayout()
        row.addWidget(QLabel("主通道")); self.cb_active = QComboBox()
        self.cb_active.addItems([f"ch{i}" for i in range(N_CHAN)] + ["全部 (all)"])
        self.cb_active.setToolTip(
            "单通道 chN：其余 5 路 min/max/target 自动归零并锁定（防误改），气压图只画主通道 1 条线。\n"
            "『全部(all)』：放开 6 路范围可改，气压图画 6 条线。\n"
            "采集动作统一写 actions6.csv；命令 ACK/质量写 commands.csv、samples.csv。")
        self.cb_active.currentIndexChanged.connect(self._on_active_changed)
        row.addWidget(self.cb_active)
        self.btn_send = QPushButton("立即下发目标"); self.btn_send.clicked.connect(self._on_send)
        self.btn_zero = QPushButton("全部归零"); self.btn_zero.clicked.connect(self._on_zero)
        row.addWidget(self.btn_send); row.addWidget(self.btn_zero)
        g.addLayout(row, N_CHAN + 1, 0, 1, 6)
        ll.addWidget(gb)

        # ---- NDI ----
        gb = QGroupBox("NDI 末端（Aurora 电磁导航）")
        g = QGridLayout(gb)
        g.addWidget(QLabel("串口"), 0, 0); self.le_ndi = QLineEdit("COM1"); g.addWidget(self.le_ndi, 0, 1)
        g.addWidget(QLabel("探头数"), 0, 2); self.sb_ndi_count = QSpinBox(); self.sb_ndi_count.setRange(1, 8); self.sb_ndi_count.setValue(self.ndi_count); self.sb_ndi_count.valueChanged.connect(self._on_ndi_count_changed); g.addWidget(self.sb_ndi_count, 0, 3)
        self.btn_ndi = QPushButton("连接 NDI"); self.btn_ndi.setStyleSheet("background:#2CB1BC;color:white")
        self.btn_ndi.clicked.connect(self._toggle_ndi); g.addWidget(self.btn_ndi, 0, 4)
        self.lbl_ndi = QLabel("末端: x=--  y=--  z=--  (失锁时 NaN)"); self.lbl_ndi.setStyleSheet("color:#334E68")
        g.addWidget(self.lbl_ndi, 1, 0, 1, 5)
        ll.addWidget(gb)

        # ---- 采集 ----
        gb = QGroupBox("数据采集（动作门控）")
        g = QGridLayout(gb)
        g.addWidget(QLabel("保存目录"), 0, 0); self.le_seq = QLineEdit("data/raw"); g.addWidget(self.le_seq, 0, 1, 1, 2)
        self.btn_browse = QPushButton("…"); self.btn_browse.setFixedWidth(34); self.btn_browse.clicked.connect(self._on_browse); g.addWidget(self.btn_browse, 0, 3)
        g.addWidget(QLabel("模式"), 1, 0); self.cb_mode = QComboBox()
        self.cb_mode.addItems(["手动录制 (Manual)", "自动随机游走 (Random)",
                               "自动往返扫描 (Sweep)", "actions6.csv 回放 (Replay)"])
        g.addWidget(self.cb_mode, 1, 1, 1, 2)
        self.cb_ts = QCheckBox("自动时间戳命名"); self.cb_ts.setChecked(True); g.addWidget(self.cb_ts, 1, 3)
        g.addWidget(QLabel("动作间隔(s)"), 2, 0); self.sb_interval = QDoubleSpinBox(); self.sb_interval.setRange(0.05, 5.0); self.sb_interval.setSingleStep(0.05); self.sb_interval.setValue(0.20); g.addWidget(self.sb_interval, 2, 1)
        g.addWidget(QLabel("稳定等待(s)"), 2, 2); self.sb_settle = QDoubleSpinBox(); self.sb_settle.setRange(0.0, 4.0); self.sb_settle.setSingleStep(0.01); self.sb_settle.setValue(0.19); g.addWidget(self.sb_settle, 2, 3)
        g.addWidget(QLabel("random seed"), 3, 0); self.sb_seed = QSpinBox(); self.sb_seed.setRange(0, 2147483647); self.sb_seed.setValue(0); self.sb_seed.setSpecialValueText("自动"); g.addWidget(self.sb_seed, 3, 1)
        g.addWidget(QLabel("预生成步数"), 3, 2); self.sb_steps = QSpinBox(); self.sb_steps.setRange(0, 1000000); self.sb_steps.setValue(0); self.sb_steps.setSpecialValueText("在线"); g.addWidget(self.sb_steps, 3, 3)
        g.addWidget(QLabel("Replay 文件"), 4, 0); self.le_replay = QLineEdit(""); g.addWidget(self.le_replay, 4, 1, 1, 2)
        self.btn_replay = QPushButton("…"); self.btn_replay.setFixedWidth(34); self.btn_replay.clicked.connect(self._on_browse_replay); g.addWidget(self.btn_replay, 4, 3)
        g.addWidget(QLabel("最大 frame age(s)"), 5, 0)
        self.sb_max_frame_age = QDoubleSpinBox(); self.sb_max_frame_age.setRange(0.0, 60.0)
        self.sb_max_frame_age.setDecimals(3); self.sb_max_frame_age.setSingleStep(0.05)
        self.sb_max_frame_age.setValue(0.5); g.addWidget(self.sb_max_frame_age, 5, 1)
        g.addWidget(QLabel("最大 NDI age(s)"), 5, 2)
        self.sb_max_ndi_age = QDoubleSpinBox(); self.sb_max_ndi_age.setRange(0.0, 60.0)
        self.sb_max_ndi_age.setDecimals(3); self.sb_max_ndi_age.setSingleStep(0.05)
        self.sb_max_ndi_age.setValue(0.5); g.addWidget(self.sb_max_ndi_age, 5, 3)
        g.addWidget(QLabel("备注"), 6, 0); self.le_note = QLineEdit(""); g.addWidget(self.le_note, 6, 1, 1, 3)
        row = QHBoxLayout()
        self.btn_start = QPushButton("▶ 开始采集"); self.btn_start.setStyleSheet("background:#2CB1BC;color:white"); self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("■ 停止采集"); self.btn_stop.setStyleSheet("background:#667EEA;color:white"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._on_stop)
        row.addWidget(self.btn_start); row.addWidget(self.btn_stop)
        g.addLayout(row, 7, 0, 1, 4)
        self.lbl_rec = QLabel("未录制"); self.lbl_rec.setStyleSheet("color:#888"); g.addWidget(self.lbl_rec, 8, 0, 1, 4)
        ll.addWidget(gb)

        # ---- 后处理 ----
        gb = QGroupBox("后处理")
        g = QGridLayout(gb)
        g.addWidget(QLabel("camera_params(npz)"), 0, 0); self.le_camparam = QLineEdit("config/real_camera_params.npz"); g.addWidget(self.le_camparam, 0, 1, 1, 2)
        g.addWidget(QLabel("灰度阈值"), 1, 0); self.sb_thresh = QDoubleSpinBox(); self.sb_thresh.setRange(0, 255); self.sb_thresh.setDecimals(0); self.sb_thresh.setValue(60); g.addWidget(self.sb_thresh, 1, 1)
        self.cb_planar = QCheckBox("单相机--planar-lift"); self.cb_planar.setChecked(True); g.addWidget(self.cb_planar, 1, 2)
        self.btn_npz = QPushButton("⚡ 生成 npz（tip.npz + capture_to_npz）"); self.btn_npz.clicked.connect(self._on_gen_npz); g.addWidget(self.btn_npz, 2, 0, 1, 3)
        self.btn_summary = QPushButton("📋 导出汇总 CSV（按帧对齐 6路气压+NDI）"); self.btn_summary.clicked.connect(self._on_summary); g.addWidget(self.btn_summary, 3, 0, 1, 3)
        ll.addWidget(gb)
        ll.addStretch(1)

        # ---------- 右：可视化 ----------
        right = QWidget()
        rl = QVBoxLayout(right)
        self.preview = QLabel("相机预览"); self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220); self.preview.setStyleSheet("background:#222;color:#aaa")
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self.preview, 3)
        self.p_plot = pg.PlotWidget(
            title="6 路气压 (kPa)",
            axisItems={"left": _CleanAxis(orientation="left"),
                       "bottom": _CleanAxis(orientation="bottom")})
        self.p_plot.showGrid(x=True, y=True, alpha=0.3)
        self.p_plot.addLegend(offset=(-10, 10))
        self.p_curves = [self.p_plot.plot(pen=pg.mkPen(color=c, width=2), name=f"ch{i}")
                         for i, c in enumerate(_CH_COLORS)]
        rl.addWidget(self.p_plot, 2)
        self.ndi_plot = pg.PlotWidget(
            title="NDI 末端 XY 轨迹",
            axisItems={"left": _CleanAxis(orientation="left"),
                       "bottom": _CleanAxis(orientation="bottom")})
        self.ndi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.ndi_plot.setLabel("bottom", "x (mm)"); self.ndi_plot.setLabel("left", "y (mm)")
        self.ndi_curve = self.ndi_plot.plot(pen=None, symbol="o", symbolSize=3, symbolPen=None, symbolBrush="#EF4E4E")
        rl.addWidget(self.ndi_plot, 2)

        split.addWidget(left); split.addWidget(right)
        split.setStretchFactor(0, 0); split.setStretchFactor(1, 1); split.setSizes([470, 910])
        root.addWidget(split, 1)

        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True); self.log_box.setMaximumHeight(140)
        root.addWidget(self.log_box)
        self.setCentralWidget(central)

    # ===================== 信号接线 =====================
    def _connect_core(self):
        self.core.log.connect(self._log)
        self.core.preview_frame.connect(self._on_preview)
        self.core.pressure_status.connect(self._on_pressure)
        self.core.ndi_status.connect(self._on_ndi)
        self.core.connection_changed.connect(self._on_conn)
        self.core.group_connection_changed.connect(self._on_grp_conn)
        self.core.recording_started.connect(self._on_rec_started)
        self.core.recording_status.connect(self._on_rec_status)
        self.core.recording_stopped.connect(self._on_rec_stopped)

    # ===================== 配置持久化（跨机器安全）=====================
    def _load_config(self):
        try:
            cp = configparser.ConfigParser()
            if not cp.read(self._cfg_path, encoding="utf-8") or not cp.has_section("capture"):
                return
            c = cp["capture"]
            self.le_g1.setText(c.get("group1", self.le_g1.text()))
            self.le_g2.setText(c.get("group2", self.le_g2.text()))
            self.le_ndi.setText(c.get("ndi_port", self.le_ndi.text()))
            self.ndi_count = max(1, int(c.get("ndi_count", self.ndi_count)))
            self.sb_ndi_count.setValue(self.ndi_count)
            self.core.set_ndi_count(self.ndi_count)
            if hasattr(self.ndi, "ndi_count"):
                self.ndi.ndi_count = self.ndi_count
            self.sb_baud.setValue(float(c.get("baudrate", self.sb_baud.value())))
            self.sb_slave.setValue(float(c.get("slave_addr", self.sb_slave.value())))
            saved = c.get("seq_dir", None)
            if saved:
                # 相对路径总恢复；绝对路径仅当其父目录在本机存在才恢复（否则视为别机失效路径）
                if not os.path.isabs(saved) or os.path.isdir(os.path.dirname(saved) or "."):
                    self.le_seq.setText(saved)
            self.cb_mode.setCurrentIndex(int(c.get("mode", self.cb_mode.currentIndex())))
            self.sb_interval.setValue(float(c.get("action_interval", self.sb_interval.value())))
            self.sb_settle.setValue(float(c.get("settle", self.sb_settle.value())))
            self.sb_seed.setValue(int(c.get("random_seed", self.sb_seed.value())))
            self.sb_steps.setValue(int(c.get("pre_generate_steps", self.sb_steps.value())))
            self.le_replay.setText(c.get("replay_file", self.le_replay.text()))
            self.max_frame_age = max(0.0, float(c.get("max_frame_age", self.max_frame_age)))
            self.max_ndi_age = max(0.0, float(c.get("max_ndi_age", self.max_ndi_age)))
            self.sb_max_frame_age.setValue(self.max_frame_age)
            self.sb_max_ndi_age.setValue(self.max_ndi_age)
            self._guard = True                  # 恢复 active/cfg 期间禁止 _on_active_changed/_on_range_changed 回灌
            try:
                self.cb_active.setCurrentIndex(int(c.get("active_channel", self.cb_active.currentIndex())))
                for i in range(N_CHAN):          # 只填持久化缓存；spinbox 显示由后续 _on_active_changed 按 mode 刷
                    self._cfg_lo[i] = float(c.get(f"lo{i}", self._cfg_lo[i]))
                    self._cfg_hi[i] = float(c.get(f"hi{i}", self._cfg_hi[i]))
                    self._cfg_rise[i] = float(c.get(f"rise{i}", self._cfg_rise[i]))
                    self._cfg_fall[i] = float(c.get(f"fall{i}", self._cfg_fall[i]))
                    self._rise_sb[i].setValue(self._cfg_rise[i])
                    self._fall_sb[i].setValue(self._cfg_fall[i])
            finally:
                self._guard = False
            self.le_camparam.setText(c.get("cam_param", self.le_camparam.text()))
            self.sb_thresh.setValue(float(c.get("threshold", self.sb_thresh.value())))
            self.cb_ts.setChecked(c.get("auto_timestamp", "1") == "1")
            self.cb_planar.setChecked(c.get("planar_lift", "1") == "1")
        except Exception as e:
            print(f"[cfg] load: {e}")

    def _save_config(self):
        try:
            cp = configparser.ConfigParser()
            cp["capture"] = {
                "group1": self.le_g1.text(), "group2": self.le_g2.text(),
                "ndi_port": self.le_ndi.text(),
                "ndi_count": str(self.sb_ndi_count.value()),
                "baudrate": str(int(self.sb_baud.value())), "slave_addr": str(int(self.sb_slave.value())),
                "seq_dir": self.le_seq.text(), "mode": str(self.cb_mode.currentIndex()),
                "action_interval": str(self.sb_interval.value()), "settle": str(self.sb_settle.value()),
                "random_seed": str(self.sb_seed.value()),
                "pre_generate_steps": str(self.sb_steps.value()),
                "replay_file": self.le_replay.text(),
                "max_frame_age": str(self.sb_max_frame_age.value()),
                "max_ndi_age": str(self.sb_max_ndi_age.value()),
                "active_channel": str(self.cb_active.currentIndex()),
                "cam_param": self.le_camparam.text(), "threshold": str(int(self.sb_thresh.value())),
                "auto_timestamp": "1" if self.cb_ts.isChecked() else "0",
                "planar_lift": "1" if self.cb_planar.isChecked() else "0",
            }
            for i in range(N_CHAN):              # 从持久化缓存写（不被单通道显示归零污染，review #2）
                cp["capture"][f"lo{i}"] = str(self._cfg_lo[i])
                cp["capture"][f"hi{i}"] = str(self._cfg_hi[i])
                cp["capture"][f"rise{i}"] = str(self._cfg_rise[i])
                cp["capture"][f"fall{i}"] = str(self._cfg_fall[i])
            with open(self._cfg_path, "w", encoding="utf-8") as f:
                cp.write(f)
        except Exception as e:
            print(f"[cfg] save: {e}")

    # ===================== 槽 =====================
    def _log(self, text: str):
        self.log_box.appendPlainText(text)

    def _on_preview(self, img: np.ndarray):
        if not self.preview.isVisible():
            return
        rgb = np.ascontiguousarray(img[:, :, ::-1])   # BGR -> RGB
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self.preview.width() - 4, self.preview.height() - 4,
                                             Qt.KeepAspectRatio, Qt.FastTransformation)
        self.preview.setPixmap(pix)

    def _on_pressure(self, p6: list):
        p6 = list(p6)[:N_CHAN] + [0.0] * (N_CHAN - len(p6))
        self._p_buf[:, :-1] = self._p_buf[:, 1:]
        self._p_buf[:, -1] = p6
        for i, curve in enumerate(self.p_curves):
            curve.setData(self._p_buf[i])

    def _on_ndi(self, pose: list):
        x, y, z = (pose[0], pose[1], pose[2]) if len(pose) >= 3 else (float("nan"),) * 3
        q = pose[10] if len(pose) > 10 else float("nan")
        parts = [f"ndi0: ({x:.1f},{y:.1f},{z:.1f}) q={q:.2f}"]
        for i in range(1, self.ndi_count):
            base = i * 11
            if len(pose) >= base + 11:
                parts.append(f"ndi{i}: ({pose[base]:.1f},{pose[base+1]:.1f},{pose[base+2]:.1f}) q={pose[base+10]:.2f}")
        self.lbl_ndi.setText(" | ".join(parts))
        if np.isfinite(x) and np.isfinite(y):
            pt = np.array([[float(x)], [float(y)]])
            self._xy_trail = np.hstack([self._xy_trail, pt])[:, -PLOT_LEN:]
            self.ndi_curve.setData(self._xy_trail[0], self._xy_trail[1])

    def _on_conn(self, ok: bool, msg: str):
        self._log(msg)

    def _on_rec_started(self, seq_dir: str):
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)
        self._log(f"录制中 -> {seq_dir}")

    def _on_rec_status(self, frames, elapsed, action6, x, y, z):
        p_str = " ".join(f"{v:.0f}" for v in action6)
        self.lbl_rec.setText(f"录制中：{frames} 帧 | {elapsed:.1f}s | [{p_str}] | tip({x:.0f},{y:.0f},{z:.0f})")
        self.lbl_rec.setStyleSheet("color:#2CB1BC")

    def _on_rec_stopped(self, seq_dir, frames):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.lbl_rec.setText(f"已停止：{frames} 帧 -> {seq_dir}"); self.lbl_rec.setStyleSheet("color:#888")

    # ===================== 按钮回调 =====================
    def _active_idx(self):
        """cb_active 当前索引：0..5 = 单通道；6 = 全部(all)。"""
        return self.cb_active.currentIndex()

    def _current_targets(self):
        """manual 每拍重发的目标向量。单通道→仅主通道非 0；all→仅已连接组的通道。
        未连接组的通道恒 0（不会被驱动）。"""
        idx = self._active_idx(); avail = self._available_channels()
        if idx < N_CHAN:
            return [self._target_sb[i].value() if (i == idx and i in avail) else 0.0 for i in range(N_CHAN)]
        return [self._target_sb[i].value() if i in avail else 0.0 for i in range(N_CHAN)]

    def _current_lo(self):
        idx = self._active_idx(); avail = self._available_channels()
        if idx < N_CHAN:
            return [self._min_sb[i].value() if (i == idx and i in avail) else 0.0 for i in range(N_CHAN)]
        return [self._min_sb[i].value() if i in avail else 0.0 for i in range(N_CHAN)]

    def _current_hi(self):
        idx = self._active_idx(); avail = self._available_channels()
        if idx < N_CHAN:
            return [self._max_sb[i].value() if (i == idx and i in avail) else 0.0 for i in range(N_CHAN)]
        return [self._max_sb[i].value() if i in avail else 0.0 for i in range(N_CHAN)]

    def _on_target_changed(self):
        if self._guard:
            return
        # manual 模式每拍重发最新目标；即时下发也用最新值
        self.core.set_manual_target(self._current_targets())

    def _on_range_changed(self):
        """min/max 改动：实时同步驱动 + 仅把 enabled 通道的编辑记进持久化缓存。"""
        if self._guard:
            return
        for i in range(N_CHAN):
            if self._min_sb[i].isEnabled():
                self._cfg_lo[i] = self._min_sb[i].value()
            if self._max_sb[i].isEnabled():
                self._cfg_hi[i] = self._max_sb[i].value()
        self.core.update_ranges(self._current_lo(), self._current_hi())

    def _on_rate_changed(self):
        if self._guard:
            return
        for i in range(N_CHAN):
            if self._rise_sb[i].isEnabled():
                self._cfg_rise[i] = self._rise_sb[i].value()
            if self._fall_sb[i].isEnabled():
                self._cfg_fall[i] = self._fall_sb[i].value()

    def _on_active_changed(self, idx=None):
        """主通道切换（一个功能两种实现）：
        - 单通道 chN：chN 设默认范围(0..DEFAULT_MAX)、target=0 并同步其缓存；其余 5 路 min/max/target
          **显示归零并锁定**，但各自的 _cfg_lo/hi 缓存保留 → 切回 all 可恢复、退出也不丢配置（review #2）；
        - 全部(all)：从 _cfg_lo/hi 缓存恢复 6 路显示，放开可改；
        同时刷新气压曲线显隐（单通道 1 条 / all 6 条）并把最新目标/范围同步给 recorder。"""
        if self._guard:
            return
        if idx is None:
            idx = self._active_idx()
        self._guard = True
        try:
            if idx < N_CHAN:
                self._cfg_lo[idx] = 0.0
                self._cfg_hi[idx] = DEFAULT_MAX
                for i in range(N_CHAN):
                    if i == idx:
                        self._min_sb[i].setValue(0.0)
                        self._max_sb[i].setValue(DEFAULT_MAX)
                    else:
                        self._min_sb[i].setValue(0.0)     # 仅显示归零；_cfg_lo/hi[i] 保留
                        self._max_sb[i].setValue(0.0)
                    self._target_sb[i].setValue(0.0)
            else:
                for i in range(N_CHAN):                   # all：从缓存恢复 6 路显示
                    self._min_sb[i].setValue(self._cfg_lo[i])
                    self._max_sb[i].setValue(self._cfg_hi[i])
                    self._target_sb[i].setValue(0.0)
        finally:
            self._guard = False
        self._apply_channel_lock(idx)
        self.core.set_manual_target(self._current_targets())
        self.core.update_ranges(self._current_lo(), self._current_hi())
        self._log(f"主通道 → {'all' if idx >= N_CHAN else f'ch{idx}'}（目标/范围已同步）。")

    def _available_channels(self):
        """已连接 modbus 组对应的可驱动通道：组1→ch0-2，组2→ch3-5。都没连→空集。"""
        conn = (self.controller.connected_groups
                if hasattr(self.controller, "connected_groups") else {1, 2})
        avail = set()
        if 1 in conn:
            avail.update(range(0, 3))
        if 2 in conn:
            avail.update(range(3, 6))
        return avail

    def _apply_channel_lock(self, idx=None):
        """按 主通道模式 × 已连接组 启停 spinbox + 曲线显隐（不改值，纯 UX）。
        未连接组的通道恒 disabled/隐藏；all 模式也只开放已连接组的通道。"""
        if idx is None:
            idx = self._active_idx()
        avail = self._available_channels()
        single = idx < N_CHAN
        for i in range(N_CHAN):
            if i not in avail:
                enabled = False
            elif single:
                enabled = (i == idx)
            else:
                enabled = True
            for sb in (self._min_sb[i], self._max_sb[i], self._target_sb[i],
                       self._rise_sb[i], self._fall_sb[i]):
                sb.setEnabled(enabled)
        for i, curve in enumerate(self.p_curves):
            if i not in avail:
                curve.setVisible(False)
            elif single:
                curve.setVisible(i == idx)
            else:
                curve.setVisible(True)

    def _update_active_dropdown_availability(self):
        """主通道下拉里禁用未连接组的通道项；当前选中项若被禁用→切到首个可用通道或 all。"""
        avail = self._available_channels()
        model = self.cb_active.model()
        if model is not None:
            for i in range(N_CHAN):
                item = model.item(i)
                if item is not None:
                    item.setEnabled(i in avail)
        cur = self.cb_active.currentIndex()
        if cur < N_CHAN and cur not in avail:
            new = next((i for i in range(N_CHAN) if i in avail), N_CHAN)  # 首个可用单通道，否则 all
            if new != cur:
                self.cb_active.setCurrentIndex(new)   # 自然触发 _on_active_changed

    def _toggle_group(self, gid: int):
        """组1/组2 切换按钮：未连→连，已连→断。串口操作放后台线程（open/close 可能阻塞）。"""
        btn = self.btn_g1 if gid == 1 else self.btn_g2
        if self.controller.is_group_connected(gid):
            btn.setEnabled(False); btn.setText(f"组{gid} 断开中…")
            threading.Thread(target=lambda: self.controller.disconnect_group(gid), daemon=True).start()
            return
        port = (self.le_g1 if gid == 1 else self.le_g2).text().strip()
        if not port:
            self._log(f"⚠ 请先填组{gid}串口。"); return
        if not self.mock_valve:
            from valve_control import ValveController
            if isinstance(self.controller, ValveController):
                self.controller.group_ports[gid] = port
                self.controller.baudrate = int(self.sb_baud.value())
                self.controller.slave_addr = int(self.sb_slave.value())
        btn.setEnabled(False); btn.setText(f"组{gid} 连接中…")
        threading.Thread(target=lambda: self.controller.connect_group(gid), daemon=True).start()

    def _on_grp_conn(self, gid: int, ok: bool):
        """某组连接状态变化：更新按钮文案/颜色 + 状态标签 + 刷新 ch 联动。"""
        btn = self.btn_g1 if gid == 1 else self.btn_g2
        btn.setEnabled(True)
        if ok:
            btn.setText(f"组{gid} 断开"); btn.setStyleSheet("background:#EF4E4E;color:white")
        else:
            btn.setText(f"组{gid} 连接"); btn.setStyleSheet("background:#2CB1BC;color:white")
        conn = sorted(self.controller.connected_groups) if hasattr(self.controller, "connected_groups") else []
        self.lbl_conn.setText("已连接: " + (",".join(f"组{g}" for g in conn) if conn else "无"))
        self.lbl_conn.setStyleSheet("color:#2CB1BC" if conn else "color:#888")
        self._update_active_dropdown_availability()   # 下拉禁用未连组的通道
        self._apply_channel_lock()                    # spinbox/曲线按可用通道刷新

    def _toggle_ndi(self):
        """NDI 连接/断开 切换。断开时 ndi.stop() 的 finally 会 stop_tracking 释放 Aurora 串口。"""
        if self.ndi.isRunning():
            self._log("NDI 断开中…")
            try:
                self.ndi.stop()
            except Exception as e:
                self._log(f"NDI 断开异常: {e}")
            self._log("NDI 已断开。")
            self._sync_ndi_button()
        else:
            self._on_connect_ndi()
            self._sync_ndi_button()

    def _on_ndi_count_changed(self, value):
        if self.ndi.isRunning():
            self._log("⚠ NDI 运行中不能修改探头数，请先断开 NDI。")
            self.sb_ndi_count.blockSignals(True)
            self.sb_ndi_count.setValue(self.ndi_count)
            self.sb_ndi_count.blockSignals(False)
            return
        self.ndi_count = max(1, int(value))
        self.core.set_ndi_count(self.ndi_count)

    def _sync_ndi_button(self):
        running = self.ndi.isRunning()
        self.btn_ndi.setText("断开 NDI" if running else "连接 NDI")
        self.btn_ndi.setStyleSheet("background:#EF4E4E;color:white" if running else "background:#2CB1BC;color:white")
        if not running:
            self.lbl_ndi.setText("NDI 已断开"); self.lbl_ndi.setStyleSheet("color:#888")

    def _on_connect_ndi(self):
        if self.ndi.isRunning():
            self._log("NDI 已在运行。"); return
        self.ndi_count = max(1, int(self.sb_ndi_count.value()))
        self.core.set_ndi_count(self.ndi_count)
        old = self.ndi
        # 重建全新线程（mock 也重建）：取最新端口 + 保证断开后能重连（旧线程 stop 后 _running=False 不可复用）
        if not self.mock_ndi:
            from hardware_threads import NdiThread
            self.ndi = NdiThread(port=self.le_ndi.text().strip(), ndi_count=self.ndi_count)
        else:
            from hardware_threads import MockNdiThread
            self.ndi = MockNdiThread(ndi_count=self.ndi_count)
        self.ndi.ndi_data.connect(self.core._on_ndi)
        # ⚠ 同步更新 recorder 引用，否则 shutdown() 停到旧线程 → 真 Aurora _tracker 永不关闭
        self.core.ndi = self.ndi
        self.ndi.start()
        # 停掉旧线程（在跑→关其 Aurora _tracker；未启动→stop() 无害 no-op；去旧死连接防双发）
        if old is not self.ndi:
            try:
                old.ndi_data.disconnect(self.core._on_ndi)
            except (TypeError, RuntimeError):
                pass
            try:
                old.stop()
            except Exception:
                pass
        self._log(f"NDI 启动（端口 {self.le_ndi.text().strip()}）。")

    def _on_send(self):
        if not self.controller.connected:
            self._log("⚠ 先连接 Modbus。"); return
        idx = self.cb_active.currentIndex()
        self.controller.set_required_groups({1} if idx < 3 else {2} if idx < N_CHAN else {1, 2})
        self.controller.set_pressures(self._current_targets())

    def _on_zero(self):
        if not self.controller.connected:
            self._log("⚠ 先连接 Modbus。"); return
        idx = self.cb_active.currentIndex()
        self.controller.set_required_groups({1} if idx < 3 else {2} if idx < N_CHAN else {1, 2})
        for sb in self._target_sb:
            sb.setValue(0.0)
        self.controller.zero_all()

    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "选择保存目录", ".")
        if d:
            self.le_seq.setText(d)

    def _on_start(self):
        mode = ["manual", "random", "sweep", "replay"][self.cb_mode.currentIndex()]
        active_idx = self.cb_active.currentIndex()
        required_groups = ({1} if active_idx < 3 else {2} if active_idx < N_CHAN else {1, 2})
        connected = set(getattr(self.controller, "connected_groups", set()))
        if not required_groups.issubset(connected):
            self._log(f"⚠ 当前模式需要控制组 {sorted(required_groups)}，已连接 {sorted(connected)}。")
            return
        if mode == "replay" and not self.le_replay.text().strip():
            self._log("⚠ Replay 模式请先选择 actions6.csv。"); return
        base = self.le_seq.text().strip()
        if not base:
            self._log("请填写保存目录。"); return
        if not os.path.isabs(base):
            base = os.path.join(SCRIPT_DIR, base)         # 相对路径按 py 文件目录解析
        seq = base
        if self.cb_ts.isChecked():
            seq = os.path.join(base, "seq_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.ndi_count = max(1, int(self.sb_ndi_count.value()))
        self.core.set_ndi_count(self.ndi_count)
        self.controller.set_required_groups(required_groups)
        self.core.set_manual_target(self._current_targets())
        self.core.start_recording(seq, mode, self._current_lo(), self._current_hi(),
                                  self.sb_interval.value(), self.sb_settle.value(),
                                  active_idx if active_idx < N_CHAN else 0,
                                  self.le_note.text().strip(),
                                  [sb.value() for sb in self._rise_sb],
                                  [sb.value() for sb in self._fall_sb],
                                  (self.sb_seed.value() or None), self.sb_steps.value(),
                                  self.le_replay.text().strip() or None,
                                  required_groups, self.sb_max_frame_age.value(),
                                  self.sb_max_ndi_age.value())

    def _on_browse_replay(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 actions6.csv", self.le_seq.text(), "CSV (*.csv);;All files (*)")
        if path:
            self.le_replay.setText(path)

    def _on_stop(self):
        self.core.stop_recording()

    def _seq_abspath(self) -> str:
        """当前保存目录解析成绝对（优先用最近一次实际录制目录）。"""
        seq = self.core.seq_dir if self.core.seq_dir else self.le_seq.text().strip()
        if not os.path.isabs(seq):
            seq = os.path.join(SCRIPT_DIR, seq)
        return os.path.abspath(seq)

    def _on_summary(self):
        seq = self._seq_abspath()
        if not (os.path.isdir(os.path.join(seq, "cam0")) or os.path.isfile(os.path.join(seq, "actions6.csv"))):
            self._log(f"找不到序列目录 {seq}（要有 actions6.csv/）。"); return
        try:
            out = export_summary_csv(seq)
            self._log(f"汇总已导出 -> {out}")
        except Exception as e:
            self._log(f"导出汇总失败: {e}")

    def _on_gen_npz(self):
        seq = self._seq_abspath()
        cam0 = os.path.join(seq, "cam0")
        act = os.path.join(seq, "actions6.csv")
        ft = os.path.join(seq, "frame_times.txt")
        if not os.path.isdir(cam0):
            self._log(f"找不到帧目录 {cam0}，先采集或改保存目录。"); return
        for need, path in (("actions6.csv", act), ("frame_times.txt", ft)):
            if not os.path.isfile(path):
                self._log(f"找不到 {need}（{path}）。"); return
        # 1) tip.npz
        tip = None
        try:
            tip = build_ndi_tip_npz(seq)
            self._log(f"已生成 NDI 末端锚点 -> {tip}")
        except Exception as e:
            self._log(f"⚠ tip.npz 生成失败（{e}）；将不传 --ndi-tip。")
        # 2) capture_to_npz
        script = os.path.join(self.project_root, "scripts", "real", "capture_to_npz.py")
        if not os.path.isfile(script):
            self._log(f"未找到 {script}（project_root/scripts/real/）。"); return
        cam_param = self.le_camparam.text().strip()
        if not os.path.isabs(cam_param):
            cam_param = os.path.join(self.project_root, cam_param)
        out_npz = os.path.join(os.path.dirname(seq), os.path.basename(seq) + ".npz")
        cmd = [sys.executable, script, "--view-dirs", cam0, "--camera-params", cam_param,
               "--method", "backlight", "--gray-thresh", str(int(self.sb_thresh.value())),
               "--dt", f"{1.0 / self.fps:.4f}",
               "--actions", act, "--actions-has-timestamps", "--frame-times", ft,
               "--clean-nan", "--out", out_npz]
        if self.cb_planar.isChecked():
            cmd.append("--planar-lift")
        if tip:
            cmd += ["--ndi-tip", tip]
        self._log("生成 npz：" + " ".join(cmd))

        def worker():
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, cwd=self.project_root)
                self._npz_proc = proc
                for line in proc.stdout:
                    self.log_signal.emit(line.rstrip())
                proc.wait()
                self.log_signal.emit(f"[npz] 退出码 {proc.returncode} -> {out_npz}")
            except Exception as e:
                self.log_signal.emit(f"[npz] 失败: {e}")
            finally:
                self._npz_proc = None

        threading.Thread(target=worker, daemon=True).start()

    # ===================== 退出 =====================
    def closeEvent(self, event):
        self._save_config()
        if self._npz_proc is not None and self._npz_proc.poll() is None:
            try:
                self._npz_proc.terminate()
            except Exception:
                pass
        try:
            self.core.shutdown()
        except Exception as e:
            print(f"shutdown error: {e}")
        event.accept()


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="六通道阀 · NDI · 相机 同步采集 GUI（动作门控）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="运行示例见文件顶部 docstring。mock 任意组合：--mock-cam/--mock-valve/--mock-ndi；--mock=全开。")
    p.add_argument("--mock", action="store_true", help="全 mock（= --mock-cam --mock-valve --mock-ndi）")
    p.add_argument("--mock-cam", action="store_true", help="假相机")
    p.add_argument("--mock-valve", action="store_true", help="假阀/Modbus")
    p.add_argument("--mock-ndi", action="store_true", help="假 NDI 末端")
    p.add_argument("--group1", default="COM55", help="Modbus 组1 串口")
    p.add_argument("--group2", default="COM56", help="Modbus 组2 串口")
    p.add_argument("--ndi", default="COM1", dest="ndi_port", help="NDI 串口")
    p.add_argument("--ndi-count", type=int, default=2, help="NDI 探头数量")
    p.add_argument("--baudrate", type=int, default=9600)
    p.add_argument("--slave", type=int, default=1, dest="slave_addr")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    mock_cam = args.mock or args.mock_cam
    mock_valve = args.mock or args.mock_valve
    mock_ndi = args.mock or args.mock_ndi

    app = QApplication(sys.argv)
    win = CaptureWindow(mock_cam=mock_cam, mock_valve=mock_valve, mock_ndi=mock_ndi,
                        group1=args.group1, group2=args.group2, ndi_port=args.ndi_port,
                        baudrate=args.baudrate, slave_addr=args.slave_addr, fps=args.fps,
                        ndi_count=args.ndi_count)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
