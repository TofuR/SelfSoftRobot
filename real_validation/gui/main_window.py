"""OpenLoop 实机验验证工作台：显式硬件 profile、安全执行与回放。"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

if __package__ in (None, ""):  # 支持复制目录后直接 ``python gui/main_window.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "real_validation.gui"

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ..execution.executor import PlanExecutor
from ..contracts.io import atomic_write_json, read_json
from ..runtime.model_runtime import ModelRuntime
from ..contracts.models import ActionPlan, Anchor, SafetyPolicy, Scene, ScenePrimitive
from ..planning.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
from ..runtime.anchors import anchor_from_npz
from ..contracts.plan_io import write_actions6_csv
from ..core.session import ExperimentSession, SessionState
from ..hardware.profile import (BackendMode, DeviceState, HardwareProfile,
                                required_groups_for_channels)
from ..hardware.session import HardwareSession
from ..widgets import CameraViewWidget, PlanPreviewWidget, SceneEditorPanel
from .theme import QSS, CARD, STATE_BADGE_COLORS, configure_pyqtgraph

import pyqtgraph as pg   # 可视化面板的气压/NDI 实时曲线(real_capture 右栏同款)

APP_DIR = Path(__file__).resolve().parent.parent  # real_validation/ 包根(数据目录 config/checkpoints/data/runs 不变)


class SafetyPolicyDialog(QDialog):
    """六通道安全配置独立对话框，避免长期挤占 Setup 页面。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("六通道安全配置")
        self.setMinimumWidth(650)
        root = QVBoxLayout(self)
        help_text = QLabel(
            "单位：压力 kPa，变化率 kPa/s。0 变化率表示不限速；真实执行前仍需 Preflight。")
        help_text.setWordWrap(True)
        root.addWidget(help_text)
        grid = QGridLayout(); grid.setHorizontalSpacing(8); grid.setVerticalSpacing(5)
        headers = ["min", "max", "rise/s", "fall/s", "initial"]
        grid.addWidget(QLabel("通道"), 0, 0)
        for column, header in enumerate(headers, 1):
            grid.addWidget(QLabel(header), 0, column)
        self.cells = []
        for channel in range(6):
            grid.addWidget(QLabel(f"ch{channel}"), channel + 1, 0)
            row = []
            for column, default in enumerate((0.0, 150.0, 100.0, 100.0, 0.0), 1):
                cell = QDoubleSpinBox(); cell.setRange(0, 500); cell.setDecimals(1)
                cell.setValue(default); cell.setMinimumWidth(96)
                grid.addWidget(cell, channel + 1, column)
                row.append(cell)
            self.cells.append(row)
        root.addLayout(grid)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)


class _ModelLoadThread(QThread):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, checkpoint: str, data_dir: str, device: str,
                 k_safe: int | None):
        super().__init__()
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.device = device
        self.k_safe = k_safe

    def run(self) -> None:
        from ..runtime.model_runtime import ModelLoadError
        try:
            runtime = ModelRuntime(self.checkpoint, self.data_dir or None, self.device,
                                   k_safe=self.k_safe)
            self.loaded.emit(runtime)
        except (ModelLoadError, FileNotFoundError, ValueError) as error:
            self.failed.emit(str(error))               # 可操作提示,不弹 traceback
        except Exception:
            self.failed.emit(traceback.format_exc())   # 真 bug 才给 traceback


class _ValveConnectThread(QThread):
    """后台执行串口连接(open 阻塞,不能卡 GUI)。controller 在 GUI 线程创建并传入。

    ValveController 是 QObject,须与 QtValveTransport 同线程(GUI 线程,有事件循环);
    只有 connect_group(串口 open)阻塞 → 放后台线程。
    """
    connected = pyqtSignal(object, str)      # (controller, 摘要)
    failed = pyqtSignal(str)

    def __init__(self, hardware: HardwareSession, groups: tuple[int, ...]):
        super().__init__()
        self.hardware = hardware
        self.groups = groups

    def run(self) -> None:
        try:
            results = self.hardware.connect_prepared_valves(self.groups)
            ok_groups = [gid for gid, (ok, _) in results.items() if ok]
            failed_groups = [gid for gid, (ok, _) in results.items() if not ok]
            summary = (f"已连接组: {sorted(ok_groups) or '无'}"
                       + (f" | 失败组: {sorted(failed_groups)}" if failed_groups else ""))
            if not ok_groups:
                self.failed.emit(f"阀连接失败: {summary}")
                return
            self.connected.emit(self.hardware.valve_controller, summary)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class _ExecutionThread(QThread):
    event = pyqtSignal(str, object)
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, executor: PlanExecutor, plan: ActionPlan, output_csv: Path):
        super().__init__()
        self.executor = executor
        self.plan = plan
        self.output_csv = output_csv
        self.executor.event_callback = lambda name, payload: self.event.emit(name, payload)

    def run(self) -> None:
        try:
            receipts = self.executor.execute(self.plan, self.output_csv)
            self.finished_ok.emit(receipts)
        except Exception as error:
            self.failed.emit(str(error))


class _ZeroThread(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, transport, timeout_s: float):
        super().__init__()
        self.transport = transport
        self.timeout_s = float(timeout_s)

    def run(self) -> None:
        try:
            receipt = self.transport.zero(self.timeout_s)
            if receipt.status != "ack":
                raise RuntimeError(receipt.status)
            self.finished_ok.emit(receipt)
        except Exception as error:
            self.failed.emit(str(error))


class _PlanningThread(QThread):
    planned = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, planner: OpenLoopShootingPlanner, kwargs: dict):
        super().__init__()
        self.planner = planner
        self.kwargs = kwargs
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            plan = self.planner.plan(cancel_event=self.cancel_event, **self.kwargs)
            self.planned.emit(plan)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class ValidationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SelfSoftRobot · OpenLoop 实机验证工作台")
        self.resize(1400, 860)
        self.session: ExperimentSession | None = None
        self.runtime: ModelRuntime | None = None
        self.executor: PlanExecutor | None = None
        self._model_thread: _ModelLoadThread | None = None
        self._planning_thread: _PlanningThread | None = None
        self._execution_thread: _ExecutionThread | None = None
        self._zero_thread: _ZeroThread | None = None
        self._zero_target: SessionState | None = None
        self.hardware = HardwareSession(self)
        self.hardware.device_state_changed.connect(self._on_device_state)
        self.hardware.camera_frame.connect(self._on_camera_frame_cam)
        self.hardware.ndi_data.connect(self._on_ndi_data)
        self.hardware.valve_command.connect(self._push_pressure)
        self.hardware.log.connect(self._log)
        self.valve_controller = None   # 兼容旧槽函数；真实来源为 self.hardware
        self.ndi_thread = None
        self._camera_frames: dict[int, object] = {}   # 多相机最新帧(cam_index → bgr)
        self._current_cam_index = 0    # 主显示当前相机
        self._valve_connect_thread: _ValveConnectThread | None = None
        configure_pyqtgraph()          # 任何 PlotWidget 之前,保证白底全局生效
        self._build_ui()
        self._init_viz_buffers()
        self._load_hardware_config()   # 回填上次保存的串口配置(若有)
        self._apply_profile_from_ui(log=False)
        self._refresh()
        self._log("SelfSoftRobot 实机验证工作台已启动。")
        self._log("请先选择运行配置并显式连接所需硬件；连接失败不会回退 Mock。")

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central); layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(7)
        safety_bar = QHBoxLayout()
        self.state_label = QLabel("No session")
        self.model_badge = QLabel("模型: 未加载")
        self.device_badges = {}
        for device, label in (("camera", "相机"), ("valve", "阀"), ("ndi", "NDI")):
            badge = QLabel(f"{label}: OFF")
            badge.setStyleSheet(
                "padding:4px 9px;border-radius:9px;background:#E3E8EE;color:#486581;")
            self.device_badges[device] = badge
        self.zero_button = QPushButton("全部归零")
        self.zero_button.setObjectName("danger")
        self.zero_button.clicked.connect(self._zero)
        self.abort_button = QPushButton("中止执行")
        self.abort_button.setObjectName("danger")
        self.abort_button.clicked.connect(self._abort)
        safety_bar.addWidget(self.state_label, 1)
        safety_bar.addWidget(self.model_badge)
        for badge in self.device_badges.values():
            safety_bar.addWidget(badge)
        safety_bar.addWidget(self.zero_button)
        safety_bar.addWidget(self.abort_button)
        layout.addLayout(safety_bar)

        # ---- 左右两栏:左 5 页 Tab(控制台) + 右 可视化面板(参考 real_capture)----
        self.main_split = QSplitter(Qt.Horizontal)

        # 主显示摄像头:先创建(tab 页构建会引用它),放右上面板
        self.main_display = CameraViewWidget()
        self.main_display.set_read_only(True)   # 默认纯显示;Observe 页激活时可交互锚定
        self.main_display.setMinimumHeight(360)  # 大横屏画面(不再小框)

        # 左:5 页 Tab —— 控制台(实验/锚点/规划/执行/结果)
        # Setup/Observe 内容较多,包 QScrollArea 防止把窗口撑高(超高时页内滚动)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll_page(self._setup_page()), "1 Setup")
        self.tabs.addTab(self._scroll_page(self._observe_page()), "2 Observe & Scene")
        self.tabs.addTab(self._plan_page(), "3 Plan")
        self.tabs.addTab(self._execute_page(), "4 Execute")
        self.tabs.addTab(self._results_page(), "5 Results")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.main_split.addWidget(self.tabs)

        # 右:主工作区。相机占主要面积，曲线和日志按需切换。
        viz = QWidget()
        vl = QVBoxLayout(viz); vl.setContentsMargins(6, 6, 6, 6); vl.setSpacing(6)
        vl.addWidget(self.main_display, 5)
        cam_bar = QHBoxLayout(); cam_bar.setSpacing(6)
        cam_bar.addWidget(QLabel("显示图层")); cam_bar.addSpacing(4)
        self.layer_checks = {}
        for key, label in (("skeleton", "骨架"), ("scene", "场景"),
                           ("predicted", "预测"), ("actual", "实际"), ("ndi", "NDI")):
            cb = QCheckBox(label)
            cb.setChecked(key != "ndi")          # 默认 NDI 关
            cb.toggled.connect(
                lambda checked, k=key: self.main_display.set_layer_visible(k, checked))
            self.layer_checks[key] = cb
            cam_bar.addWidget(cb)
        cam_bar.addStretch()
        vl.addLayout(cam_bar)
        self.press_plot = pg.PlotWidget(title="气压命令 (kPa · 6 通道)")
        self.press_plot.showGrid(x=True, y=True, alpha=0.3)
        self._p_curves = []
        _p_colors = ["#2CB1BC", "#667EEA", "#3182CE", "#805AD5", "#38B2AC", "#F6AD55"]
        for _ch in range(6):
            self._p_curves.append(self.press_plot.plot(pen=pg.mkPen(_p_colors[_ch], width=1.4)))
        self.ndi_plot = pg.PlotWidget(title="NDI 末端坐标 (mm · 隐藏评价流)")
        self.ndi_plot.showGrid(x=True, y=True, alpha=0.3)
        self._ndi_curves = [
            self.ndi_plot.plot(pen=pg.mkPen("#EF4E4E", width=1.4)),   # X
            self.ndi_plot.plot(pen=pg.mkPen("#38A169", width=1.4)),   # Y
            self.ndi_plot.plot(pen=pg.mkPen("#3182CE", width=1.4)),   # Z
        ]
        self.log_box = QPlainTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("运行日志")
        self.viz_tabs = QTabWidget()
        self.viz_tabs.addTab(self.press_plot, "动作曲线")
        self.viz_tabs.addTab(self.ndi_plot, "NDI 评价")
        self.viz_tabs.addTab(self.log_box, "运行日志")
        self.viz_tabs.setMaximumHeight(230)
        vl.addWidget(self.viz_tabs, 2)
        self.main_info = QLabel("相机: OFF | 骨架: - | NDI: OFF")
        self.main_info.setStyleSheet("color:#486581;font-size:11px;padding:2px 4px;")
        vl.addWidget(self.main_info)
        self.main_split.addWidget(viz)

        self.tabs.setMinimumWidth(535)
        self.main_split.setSizes([560, 840])
        self.main_split.setStretchFactor(0, 0); self.main_split.setStretchFactor(1, 1)
        layout.addWidget(self.main_split, 1)

        self.setCentralWidget(central)

    def _scroll_page(self, page: QWidget) -> QScrollArea:
        """把 tab 页包进滚动区:窗口高度不随内容被撑大;内容超高时页内滚动。"""
        sa = QScrollArea(); sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setWidget(page)
        return sa

    def _setup_page(self) -> QWidget:
        """按真实实验前后依赖排列：建立实验→应用模式→连设备→加载模型。"""
        page = QWidget(); root = QVBoxLayout(page); root.setSpacing(10)

        gb_exp = QGroupBox("实验与运行")
        exp = QVBoxLayout(gb_exp); exp.setContentsMargins(10, 12, 10, 10); exp.setSpacing(6)
        row = QHBoxLayout(); row.addWidget(QLabel("Run 根目录"))
        self.run_root = QLineEdit(str(APP_DIR / "runs"))
        row.addWidget(self.run_root, 1); row.addWidget(self._browse_button(self.run_root, True))
        exp.addLayout(row)
        row = QHBoxLayout()
        create = QPushButton("新建实验"); create.setObjectName("primary")
        create.clicked.connect(self._new_session)
        replay = QPushButton("打开 Run（只读回放）"); replay.clicked.connect(self._open_replay)
        row.addWidget(create); row.addWidget(replay); row.addStretch(); exp.addLayout(row)
        root.addWidget(gb_exp)

        gb_profile = QGroupBox("运行配置（须先应用）")
        profile_layout = QVBoxLayout(gb_profile)
        row = QHBoxLayout(); row.addWidget(QLabel("预设"))
        self.hw_profile_preset = QComboBox()
        self.hw_profile_preset.addItem("全 Mock", "all_mock")
        self.hw_profile_preset.addItem("真机验证", "real")
        self.hw_profile_preset.addItem("自定义混合", "custom")
        self.hw_profile_preset.currentIndexChanged.connect(self._on_profile_preset_changed)
        self.hw_apply_profile_btn = QPushButton("应用配置")
        self.hw_apply_profile_btn.setObjectName("primary")
        self.hw_apply_profile_btn.clicked.connect(self._apply_profile_from_ui)
        row.addWidget(self.hw_profile_preset, 1); row.addWidget(self.hw_apply_profile_btn)
        profile_layout.addLayout(row)
        self.hw_profile_hint = QLabel("配置只在设备全部断开时可更改；任何真机失败都不会回退 Mock。")
        self.hw_profile_hint.setWordWrap(True); profile_layout.addWidget(self.hw_profile_hint)
        root.addWidget(gb_profile)

        def backend_combo(include_disabled=True):
            combo = QComboBox()
            if include_disabled:
                combo.addItem("Disabled", BackendMode.DISABLED.value)
            combo.addItem("Mock", BackendMode.MOCK.value)
            combo.addItem("Real", BackendMode.REAL.value)
            combo.currentIndexChanged.connect(self._on_profile_control_edited)
            return combo

        gb_camera = QGroupBox("相机（Anchor / 场景观察）")
        camera = QVBoxLayout(gb_camera); camera.setSpacing(6)
        self.hw_camera_backend = backend_combo()
        self.hw_camera_count = QSpinBox(); self.hw_camera_count.setRange(1, 8); self.hw_camera_count.setValue(1)
        self.hw_camera_count.valueChanged.connect(self._on_profile_control_edited)
        self.hw_camera_serials = QLineEdit(); self.hw_camera_serials.setPlaceholderText("可留空自动枚举；多台用逗号分隔唯一 serial")
        self.hw_camera_serials.textChanged.connect(self._on_profile_control_edited)
        self.hw_camera_view = QComboBox(); self.hw_camera_view.setEnabled(False)
        self.hw_camera_view.currentIndexChanged.connect(self._on_camera_view_changed)
        self.camera_btn = QPushButton("连接相机"); self.camera_btn.setObjectName("primary")
        self.camera_btn.clicked.connect(self._toggle_camera)
        row = QHBoxLayout(); row.addWidget(QLabel("Backend")); row.addWidget(self.hw_camera_backend)
        row.addWidget(QLabel("数量")); row.addWidget(self.hw_camera_count)
        row.addWidget(QLabel("主显示")); row.addWidget(self.hw_camera_view, 1); camera.addLayout(row)
        row = QHBoxLayout(); row.addWidget(QLabel("Serials")); row.addWidget(self.hw_camera_serials, 1); camera.addLayout(row)
        row = QHBoxLayout(); row.addWidget(self.camera_btn); row.addStretch(); camera.addLayout(row)
        root.addWidget(gb_camera)

        gb_valve = QGroupBox("六通道比例阀（动作执行）")
        valve = QVBoxLayout(gb_valve); valve.setSpacing(6)
        self.hw_valve_backend = backend_combo(include_disabled=False)
        self.hw_g1 = QLineEdit("COM3"); self.hw_g1.setFixedWidth(78)
        self.hw_g2 = QLineEdit("COM46"); self.hw_g2.setFixedWidth(78)
        self.hw_baud = QSpinBox(); self.hw_baud.setRange(1200, 115200); self.hw_baud.setValue(9600)
        self.hw_baud.setFixedWidth(88)
        self.hw_slave = QSpinBox(); self.hw_slave.setRange(1, 247); self.hw_slave.setValue(1)
        row = QHBoxLayout(); row.addWidget(QLabel("Backend")); row.addWidget(self.hw_valve_backend)
        row.addWidget(QLabel("baud")); row.addWidget(self.hw_baud)
        row.addWidget(QLabel("slave")); row.addWidget(self.hw_slave); valve.addLayout(row)
        row = QHBoxLayout(); row.addWidget(QLabel("组1")); row.addWidget(self.hw_g1)
        row.addWidget(QLabel("组2")); row.addWidget(self.hw_g2); row.addStretch(); valve.addLayout(row)
        row = QHBoxLayout()
        self.hw_conn1_btn = QPushButton("连接组1"); self.hw_conn1_btn.clicked.connect(lambda: self._connect_valve_group(1))
        self.hw_conn2_btn = QPushButton("连接组2"); self.hw_conn2_btn.clicked.connect(lambda: self._connect_valve_group(2))
        self.hw_g1_status = QLabel("未连"); self.hw_g2_status = QLabel("未连")
        self.hw_disconn_btn = QPushButton("断开全部"); self.hw_disconn_btn.setObjectName("danger")
        self.hw_disconn_btn.clicked.connect(self._disconnect_valve)
        row.addWidget(self.hw_conn1_btn); row.addWidget(self.hw_g1_status)
        row.addWidget(self.hw_conn2_btn); row.addWidget(self.hw_g2_status)
        row.addWidget(self.hw_disconn_btn); row.addStretch(); valve.addLayout(row)
        root.addWidget(gb_valve)

        gb_ndi = QGroupBox("NDI（隐藏评价流，不输入 Planner）")
        ndi_layout = QVBoxLayout(gb_ndi); ndi_layout.setSpacing(6)
        self.hw_ndi_backend = backend_combo()
        self.hw_ndi_port = QLineEdit("COM9"); self.hw_ndi_port.setFixedWidth(70)
        self.hw_ndi_count = QSpinBox(); self.hw_ndi_count.setRange(1, 8); self.hw_ndi_count.setValue(1)
        self.hw_ndi_btn = QPushButton("连接 NDI"); self.hw_ndi_btn.setObjectName("primary")
        self.hw_ndi_btn.clicked.connect(self._connect_ndi)
        row = QHBoxLayout(); row.addWidget(QLabel("Backend")); row.addWidget(self.hw_ndi_backend)
        row.addWidget(QLabel("串口")); row.addWidget(self.hw_ndi_port)
        row.addWidget(QLabel("探头")); row.addWidget(self.hw_ndi_count)
        row.addWidget(self.hw_ndi_btn); row.addStretch(); ndi_layout.addLayout(row)
        root.addWidget(gb_ndi)

        # ---- 卡3:模型与部署契约(紧凑行)----
        gb_model = QGroupBox("模型与部署契约")
        m = QVBoxLayout(gb_model); m.setContentsMargins(10, 12, 10, 10); m.setSpacing(6)
        self.checkpoint = QLineEdit(str(APP_DIR / "checkpoints" / "current" / "best_model.pt"))
        self.data_dir = QLineEdit(str(APP_DIR / "data"))
        self.k_safe = QSpinBox(); self.k_safe.setRange(0, 100000)
        self.k_safe.setSpecialValueText("未认证")
        self.device = QComboBox(); self.device.addItems(["cpu", "cuda"])
        load = QPushButton("加载部署模型"); load.setObjectName("primary")
        load.clicked.connect(self._load_model)
        row = QHBoxLayout()
        row.addWidget(QLabel("Checkpoint")); row.addWidget(self.checkpoint, 1)
        row.addWidget(self._browse_button(self.checkpoint, False))
        row.addWidget(QLabel("设备")); row.addWidget(self.device)
        m.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("训练数据")); row.addWidget(self.data_dir, 1)
        row.addWidget(self._browse_button(self.data_dir, True))
        row.addWidget(QLabel("K_safe")); row.addWidget(self.k_safe)
        m.addLayout(row)
        m.addWidget(load)
        self.model_summary = QPlainTextEdit(); self.model_summary.setReadOnly(True)
        self.model_summary.setMaximumHeight(80)
        m.addWidget(self.model_summary)
        root.addWidget(gb_model)

        self.safety_dialog = SafetyPolicyDialog(self)
        self._safety_cells = self.safety_dialog.cells
        self.safety_dialog.buttons.button(QDialogButtonBox.Apply).clicked.connect(self._apply_safety)
        safety_button = QPushButton("安全配置…（六通道 kPa / kPa·s⁻¹）")
        safety_button.clicked.connect(self.safety_dialog.show)
        root.addWidget(safety_button)
        root.addStretch()
        self._sync_profile_controls()
        return page

    def _observe_page(self) -> QWidget:
        """Observe 页:纯控制面板(紧凑满宽卡堆叠)。

        锚定/点选交互在右上面板主摄像头 —— 本页激活时主摄像头可交互(工具点亮)。
        不再在页内复制一个小摄像头(那会把窗口最小宽度撑爆,也逻辑重复)。
        """
        page = QWidget(); root = QVBoxLayout(page); root.setSpacing(10)

        # 卡1:离线锚定
        gb_off = QGroupBox("离线锚定")
        off = QVBoxLayout(gb_off); off.setContentsMargins(10, 12, 10, 10); off.setSpacing(6)
        buttons = QHBoxLayout()
        anchor = QPushButton("加载 anchor.json"); anchor.clicked.connect(self._load_anchor)
        scene = QPushButton("加载 scene.json"); scene.clicked.connect(self._load_scene)
        buttons.addWidget(anchor); buttons.addWidget(scene); buttons.addStretch()
        off.addLayout(buttons)
        self.anchor_npz = QLineEdit(str(APP_DIR / "data" / "npz" / "seq_20260627_163921_train.npz"))
        self.anchor_index = QSpinBox(); self.anchor_index.setRange(0, 100000000)
        self.anchor_index.setValue(39)
        self.anchor_index.setToolTip(
            "从该帧建立 anchor。帧索引往前必须凑满模型历史长度 H(此模型 40 步)——\n"
            "选太靠前的帧会报『缺少 N 步历史』。示例数据共 8172 帧,建议从 39 开始。")
        load_npz = QPushButton("从 NPZ 建立 Anchor"); load_npz.setObjectName("primary")
        load_npz.clicked.connect(self._load_anchor_npz)
        row = QHBoxLayout()
        row.addWidget(self.anchor_npz, 1)
        row.addWidget(self._browse_button(self.anchor_npz, False))
        row.addWidget(QLabel("帧")); row.addWidget(self.anchor_index)
        row.addWidget(load_npz)
        off.addLayout(row)
        self.anchor_help = QLabel(
            "Anchor = 规划起点(当前形状 + 最近 H 步动作),从 transition NPZ 选一帧提取;\n"
            "示例数据已内置 data/npz/(15 节点,8172 帧),帧索引用 39+。")
        self.anchor_help.setWordWrap(True)
        self.anchor_help.setStyleSheet("color:#486581;font-size:11px;")
        off.addWidget(self.anchor_help)
        root.addWidget(gb_off)

        # 卡2:目标与障碍(两行紧凑)
        gb_tgt = QGroupBox("目标与障碍")
        t = QVBoxLayout(gb_tgt); t.setContentsMargins(10, 12, 10, 10); t.setSpacing(6)
        self.target_x = QDoubleSpinBox(); self.target_x.setRange(-100000, 100000); self.target_x.setFixedWidth(78)
        self.target_y = QDoubleSpinBox(); self.target_y.setRange(-100000, 100000); self.target_y.setFixedWidth(78)
        self.target_radius = QDoubleSpinBox(); self.target_radius.setRange(0, 100000); self.target_radius.setFixedWidth(78)
        self.obstacle_x = QDoubleSpinBox(); self.obstacle_x.setRange(-100000, 100000); self.obstacle_x.setFixedWidth(78)
        self.obstacle_y = QDoubleSpinBox(); self.obstacle_y.setRange(-100000, 100000); self.obstacle_y.setFixedWidth(78)
        self.obstacle_radius = QDoubleSpinBox(); self.obstacle_radius.setRange(0.01, 100000); self.obstacle_radius.setValue(10); self.obstacle_radius.setFixedWidth(78)
        target_button = QPushButton("设置目标"); target_button.setObjectName("primary")
        target_button.clicked.connect(self._set_target)
        obstacle_button = QPushButton("添加障碍"); obstacle_button.setObjectName("accent")
        obstacle_button.clicked.connect(self._add_obstacle)
        row = QHBoxLayout()
        row.addWidget(QLabel("目标")); row.addWidget(self.target_x); row.addWidget(self.target_y); row.addWidget(self.target_radius)
        row.addWidget(target_button); row.addStretch()
        t.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("障碍")); row.addWidget(self.obstacle_x); row.addWidget(self.obstacle_y); row.addWidget(self.obstacle_radius)
        row.addWidget(obstacle_button); row.addStretch()
        t.addLayout(row)
        root.addWidget(gb_tgt)

        # 卡3:相机锚定与工具(交互发生在右上面板主摄像头)
        gb_live = QGroupBox("相机锚定与工具(在右上面板画面上操作)")
        live = QVBoxLayout(gb_live); live.setContentsMargins(10, 12, 10, 10); live.setSpacing(6)
        live_buttons = QHBoxLayout()
        self.camera_anchor_btn = QPushButton("从相机取流锚定"); self.camera_anchor_btn.setObjectName("primary")
        self.camera_anchor_btn.setEnabled(False)
        self.camera_anchor_btn.clicked.connect(self._camera_anchor)
        self.warmup_btn = QPushButton("Warmup(填动作历史)"); self.warmup_btn.setObjectName("accent")
        self.warmup_btn.setEnabled(False)
        self.warmup_btn.clicked.connect(self._warmup)
        live_buttons.addWidget(self.camera_anchor_btn); live_buttons.addWidget(self.warmup_btn)
        live_buttons.addStretch()
        live.addLayout(live_buttons)
        self.anchor_prereq = QLabel("相机锚定需要：已建实验 + 已加载模型 + 相机 READY + 已收到帧")
        self.anchor_prereq.setWordWrap(True)
        self.anchor_prereq.setStyleSheet("color:#486581;font-size:11px;")
        live.addWidget(self.anchor_prereq)
        self.zero_history_cb = QCheckBox("零历史起步(免 warmup,首窗口 OOD)")
        self.zero_history_cb.setChecked(True)
        self.zero_history_cb.setToolTip(
            "勾选:用全 0 动作历史直接锚定,不需先 Warmup。\n"
            "⚠️ 模型训练从没见过零填充窗口,首窗口预测可能明显不准;\n"
            "运行几步后自动用本次真实动作累积历史,误差会收敛。")
        self.zero_history_cb.toggled.connect(self._on_zero_history_toggled)
        self.zero_hist_hint = QLabel("已启用:零历史起步(OOD)")
        self.zero_hist_hint.setStyleSheet("color:#F6AD55;font-size:11px;")
        live.addWidget(self.zero_history_cb)
        live.addWidget(self.zero_hist_hint)
        tool_row = QHBoxLayout()
        self.tool_select_btn = QPushButton("select"); self.tool_select_btn.setCheckable(True)
        self.tool_select_btn.clicked.connect(lambda: self._set_tool("select"))
        self.tool_target_btn = QPushButton("点加目标"); self.tool_target_btn.setCheckable(True)
        self.tool_target_btn.clicked.connect(lambda: self._set_tool("add_target"))
        self.tool_skeleton_btn = QPushButton("点出目标骨架"); self.tool_skeleton_btn.setCheckable(True)
        self.tool_skeleton_btn.setToolTip(
            "点出目标骨架:按末端 node0 到基座 nodeN-1 依次点击 N 个点,\n"
            "然后点击『完成目标骨架』。"
            "规划让机器人拟合这个目标骨架(全身目标)。")
        self.tool_skeleton_btn.clicked.connect(lambda: self._set_tool("add_target_skeleton"))
        self.tool_obstacle_btn = QPushButton("点加障碍"); self.tool_obstacle_btn.setCheckable(True)
        self.tool_obstacle_btn.clicked.connect(lambda: self._set_tool("add_obstacle"))
        tool_row.addWidget(QLabel("工具:")); tool_row.addWidget(self.tool_select_btn)
        tool_row.addWidget(self.tool_target_btn); tool_row.addWidget(self.tool_skeleton_btn)
        tool_row.addWidget(self.tool_obstacle_btn); tool_row.addStretch()
        live.addLayout(tool_row)
        skeleton_row = QHBoxLayout()
        self.finish_skeleton_btn = QPushButton("完成目标骨架")
        self.finish_skeleton_btn.setEnabled(False)
        self.finish_skeleton_btn.clicked.connect(self._finish_skeleton_target)
        self.cancel_skeleton_btn = QPushButton("取消绘制")
        self.cancel_skeleton_btn.setEnabled(False)
        self.cancel_skeleton_btn.clicked.connect(self._cancel_skeleton_target)
        self.skeleton_draft_status = QLabel("未开始绘制")
        skeleton_row.addWidget(self.finish_skeleton_btn)
        skeleton_row.addWidget(self.cancel_skeleton_btn)
        skeleton_row.addWidget(self.skeleton_draft_status, 1)
        live.addLayout(skeleton_row)
        root.addWidget(gb_live)

        # 卡4:场景编辑(原语列表 + 状态)
        gb_scene = QGroupBox("场景编辑")
        sc = QVBoxLayout(gb_scene); sc.setContentsMargins(10, 12, 10, 10); sc.setSpacing(6)
        self.scene_editor = SceneEditorPanel()
        self.scene_editor.list.setMinimumHeight(180)   # 压最小高度,防把窗口撑高
        sc.addWidget(self.scene_editor, 1)
        self.anchor_status = QLabel("未锚定")
        sc.addWidget(self.anchor_status)
        self.skeleton_hint = QLabel("青线+圆点 = 实时骨架(15 节点,与训练同源);加载模型 + 开相机后自动显示")
        self.skeleton_hint.setWordWrap(True)
        self.skeleton_hint.setStyleSheet("color:#486581;font-size:11px;")
        sc.addWidget(self.skeleton_hint)
        self.scene_summary = QPlainTextEdit()
        self.scene_summary.setReadOnly(True)
        self.scene_summary.setMaximumHeight(80)
        self.scene_summary.setPlaceholderText("场景摘要:anchor / scene / primitives / digest")
        sc.addWidget(self.scene_summary)
        root.addWidget(gb_scene)

        # 主摄像头(右上面板)承担锚定点选;信号接到场景编辑
        self.main_display.target_picked.connect(self._add_primitive)
        self.main_display.obstacle_picked.connect(self._add_primitive)
        self.main_display.target_skeleton_picked.connect(self._add_primitive)
        self.main_display.skeleton_draft_changed.connect(self._on_skeleton_draft_changed)
        self.scene_editor.scene_edited.connect(self._apply_scene_edit)
        self.scene_editor.redraw_requested.connect(self._redraw_target_skeleton)
        self._latest_frame = None
        self._action_history = []  # warmup 填充(H×action_dim 模型单位)
        self._history_buffer = None  # 功能①:执行累积的实际动作历史(滚动重锚定用)
        return page

    def _on_tab_changed(self, index: int) -> None:
        """Observe 页(index 1)激活 → 右上面板主摄像头可交互(锚定点选);其它页纯显示。"""
        is_observe = (index == 1)
        self.main_display.set_read_only(not is_observe)
        if not is_observe:
            self.main_display.set_tool("select")

    def _set_tool(self, tool: str) -> None:
        if tool != "select" and not self.session:
            self._error("请先新建实验，再在画面上编辑目标/障碍")
            return
        self.main_display.set_read_only(False)   # 选工具即进入可交互(Observe 锚定)
        self.main_display.set_tool(tool)         # 锚定交互在右上面板主摄像头
        for btn, name in ((self.tool_select_btn, "select"),
                          (self.tool_target_btn, "add_target"),
                          (self.tool_skeleton_btn, "add_target_skeleton"),
                          (self.tool_obstacle_btn, "add_obstacle")):
            btn.setChecked(name == tool)

    def _on_skeleton_draft_changed(self, count: int) -> None:
        expected = (self.runtime.descriptor.n_nodes if self.runtime is not None else None)
        if count:
            suffix = f" / 模型需要 {expected} 节点" if expected else " / 加载模型后校验节点数"
            self.skeleton_draft_status.setText(f"已点 {count} 个节点{suffix}")
        else:
            self.skeleton_draft_status.setText("未开始绘制")
        self.finish_skeleton_btn.setEnabled(count >= 2)
        self.cancel_skeleton_btn.setEnabled(count > 0)

    def _finish_skeleton_target(self) -> None:
        count = len(self.main_display._skeleton_points)
        if count < 2:
            self._error("目标骨架至少需要 2 个节点")
            return
        if self.runtime is not None and count != self.runtime.descriptor.n_nodes:
            self._error(f"目标骨架已点 {count} 个节点，但当前模型需要 "
                        f"{self.runtime.descriptor.n_nodes} 个；请按末端 node0 到基座 nodeN-1 顺序重画")
            return
        self.main_display.commit_skeleton_target()
        self._set_tool("select")

    def _cancel_skeleton_target(self) -> None:
        self.main_display.clear_skeleton_points()
        self._set_tool("select")

    def _redraw_target_skeleton(self, primitive_id: str) -> None:
        if not self.session:
            return
        primitive = next((item for item in self.session.scene.primitives
                          if item.primitive_id == primitive_id), None)
        if primitive is None or primitive.kind != "target_skeleton":
            return
        self.main_display.clear_skeleton_points()
        self._set_tool("add_target_skeleton")
        self.skeleton_draft_status.setText(
            "正在重画：旧对象会保留到新骨架点击“完成”")

    def _add_primitive(self, primitive) -> None:
        if not self.session:
            return
        existing = self.session.scene.primitives
        if primitive.kind.startswith("target_"):
            retained = tuple(item for item in existing
                             if not item.kind.startswith("target_"))
            replaced = len(existing) - len(retained)
            scene = Scene(self.session.scene.name, retained + (primitive,),
                          self.session.scene.dimension)
            if replaced:
                self._log("已用新目标替换旧目标（Planner 只允许一个活动目标）")
        else:
            scene = self.session.scene.with_primitive(primitive)
        self.session.set_scene(scene)
        self._scene_changed()
        self.scene_editor.select_primitive(primitive.primitive_id)

    def _apply_scene_edit(self, scene) -> None:
        if not self.session:
            return
        self.session.set_scene(scene)
        self._scene_changed()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def _profile_from_ui(self) -> HardwareProfile:
        serials = tuple(value.strip() for value in
                        self.hw_camera_serials.text().split(",") if value.strip())
        return HardwareProfile(
            name=str(self.hw_profile_preset.currentData() or "custom"),
            camera_backend=str(self.hw_camera_backend.currentData()),
            camera_count=self.hw_camera_count.value(), camera_serials=serials,
            valve_backend=str(self.hw_valve_backend.currentData()),
            group1_port=self.hw_g1.text().strip(), group2_port=self.hw_g2.text().strip(),
            baudrate=self.hw_baud.value(), slave_addr=self.hw_slave.value(),
            ndi_backend=str(self.hw_ndi_backend.currentData()),
            ndi_port=self.hw_ndi_port.text().strip(), ndi_count=self.hw_ndi_count.value())

    def _apply_profile_from_ui(self, _checked=False, *, log=True) -> None:
        try:
            profile = self._profile_from_ui()
            self.hardware.apply_profile(profile)
            self.valve_controller = self.hardware.valve_controller
            self._sync_profile_controls()
            if self.session is not None and not self.session.replay_only:
                atomic_write_json(self.session.run_dir / "hardware_profile.json",
                                  profile.to_dict())
            if log:
                self._log(f"已应用运行配置: {profile.name} | "
                          f"camera={profile.camera_backend.value}, "
                          f"valve={profile.valve_backend.value}, "
                          f"ndi={profile.ndi_backend.value}")
        except Exception as error:
            if log:
                self._error(f"应用运行配置失败: {error}")
            else:
                self._log(f"WARN: 应用运行配置失败 {error}")

    def _require_ui_profile_applied(self) -> None:
        if self._profile_from_ui() != self.hardware.profile:
            raise RuntimeError("硬件参数已更改，请先点击“应用配置”")

    def _on_profile_preset_changed(self, _index: int) -> None:
        if getattr(self, "_syncing_profile", False):
            return
        preset = self.hw_profile_preset.currentData()
        if preset == "all_mock":
            values = ("mock", "mock", "mock")
        elif preset == "real":
            values = ("real", "real", "real")
        else:
            self._sync_profile_controls()
            return
        self._syncing_profile = True
        try:
            for combo, value in zip((self.hw_camera_backend, self.hw_valve_backend,
                                     self.hw_ndi_backend), values):
                self._set_combo_data(combo, value)
        finally:
            self._syncing_profile = False
        self._sync_profile_controls()

    def _on_profile_control_edited(self, *_args) -> None:
        if getattr(self, "_syncing_profile", False):
            return
        self._syncing_profile = True
        try:
            self._set_combo_data(self.hw_profile_preset, "custom")
        finally:
            self._syncing_profile = False
        self._sync_profile_controls()

    def _sync_profile_controls(self) -> None:
        camera_backend = BackendMode(str(self.hw_camera_backend.currentData()))
        valve_backend = BackendMode(str(self.hw_valve_backend.currentData()))
        ndi_backend = BackendMode(str(self.hw_ndi_backend.currentData()))
        unlocked = not self.hardware.any_running
        for widget in (self.hw_profile_preset, self.hw_camera_backend,
                       self.hw_valve_backend, self.hw_ndi_backend,
                       self.hw_camera_count, self.hw_camera_serials,
                       self.hw_g1, self.hw_g2, self.hw_baud, self.hw_slave,
                       self.hw_ndi_port, self.hw_ndi_count):
            widget.setEnabled(unlocked)
        self.hw_camera_serials.setEnabled(unlocked and camera_backend == BackendMode.REAL)
        self.hw_g1.setEnabled(unlocked and valve_backend == BackendMode.REAL)
        self.hw_g2.setEnabled(unlocked and valve_backend == BackendMode.REAL)
        self.hw_baud.setEnabled(unlocked and valve_backend == BackendMode.REAL)
        self.hw_slave.setEnabled(unlocked and valve_backend == BackendMode.REAL)
        self.hw_ndi_port.setEnabled(unlocked and ndi_backend == BackendMode.REAL)
        self.hw_apply_profile_btn.setEnabled(unlocked)
        self.camera_btn.setEnabled(camera_backend != BackendMode.DISABLED)
        self.hw_ndi_btn.setEnabled(ndi_backend != BackendMode.DISABLED)
        count = self.hw_camera_count.value()
        selected = self.hw_camera_view.currentIndex()
        self.hw_camera_view.clear()
        self.hw_camera_view.addItems([f"cam{i}" for i in range(count)])
        self.hw_camera_view.setCurrentIndex(max(0, min(selected, count - 1)))
        self.hw_camera_view.setEnabled(count > 1)

    def _on_device_state(self, device: str, state: str, message: str) -> None:
        labels = {"camera": "相机", "valve": "阀", "ndi": "NDI"}
        colors = {
            DeviceState.READY.value: ("#D3F9D8", "#18794E"),
            DeviceState.CONNECTING.value: ("#FFF3BF", "#8D5A00"),
            DeviceState.ERROR.value: ("#FFE3E3", "#C92A2A"),
            DeviceState.DISABLED.value: ("#E9ECEF", "#6C757D"),
            DeviceState.OFF.value: ("#E3E8EE", "#486581"),
        }
        badge = self.device_badges.get(device)
        if badge is not None:
            background, foreground = colors.get(state, colors[DeviceState.OFF.value])
            backend = getattr(self.hardware.profile, f"{device}_backend").value.upper()
            badge.setText(f"{labels.get(device, device)}: {backend} / {state.upper()}")
            badge.setToolTip(message)
            badge.setStyleSheet(
                f"padding:4px 9px;border-radius:9px;background:{background};color:{foreground};")
        if device == "camera":
            running = state in {DeviceState.CONNECTING.value, DeviceState.READY.value,
                                DeviceState.ERROR.value} and bool(self.hardware.cameras)
            self.camera_btn.setText("断开相机" if running else "连接相机")
        elif device == "ndi":
            self.hw_ndi_btn.setText("断开 NDI" if self.hardware.ndi_thread else "连接 NDI")
        elif device == "valve":
            self.valve_controller = self.hardware.valve_controller
            self._refresh_valve_status()
        self._sync_profile_controls()
        self._refresh_anchor_controls()
        self._update_main_info()

    def _toggle_camera(self) -> None:
        """相机的唯一 GUI 入口；backend 只来自已应用 profile。"""
        if self._camera_is_running():
            self._stop_camera()
            return
        self._start_camera()

    def _camera_is_running(self) -> bool:
        return bool(self.hardware.cameras)

    def _stop_camera(self) -> None:
        self.hardware.stop_cameras()
        self._camera_frames.clear()

    def _start_camera(self) -> None:
        import numpy as np
        try:
            self._require_ui_profile_applied()
            self.hardware.start_cameras()
            self.main_display.set_frame(self._latest_frame if self._latest_frame is not None
                                        else np.zeros((240, 320, 3)))
        except Exception as error:
            self._error(f"相机连接失败: {error}")

    def _on_camera_frame(self, bgr) -> None:
        if bgr is None:   # 真实相机 error → 显示提示,不崩
            self._log("ERROR: 相机错误(检查 RealSense 连接)")
            return
        self._latest_frame = bgr
        self._refresh_anchor_controls()
        self.main_display.set_frame(bgr)                       # 主显示区(唯一画面)
        if self.runtime is not None:
            from ..perception.segmentation import segment_white_on_blue
            from ..perception.skeleton import extract_skeleton_2d
            # Mock 场景:背景 = 帧自身灰度近似(真机用 manifest.segment_params + 无臂静态背景)
            mask = segment_white_on_blue(bgr, self._gray(bgr))
            skeleton, _ = extract_skeleton_2d(mask, self.runtime.descriptor.n_nodes,
                                              tip_fix=True, return_info=True)
            self.main_display.set_skeleton(skeleton)           # 主显示骨架层

    def _on_camera_frame_cam(self, cam_index: int, bgr, _timestamp=None) -> None:
        """多相机按索引存帧，只将所选视图送入主显示和感知。"""
        if cam_index not in self._camera_frames:
            self._camera_frames[cam_index] = None
        if bgr is None:
            return
        self._camera_frames[cam_index] = bgr
        if cam_index == self._current_cam_index:
            self._on_camera_frame(bgr)

    def _on_camera_error(self, cam_index: int, message: str) -> None:
        """兼容旧信号的错误日志；不改 backend，不自动降级。"""
        self._log(f"ERROR: 相机 {cam_index + 1}: {message}")

    def _on_camera_view_changed(self, index: int) -> None:
        """切换主显示区显示哪台相机(多相机时)。"""
        self._current_cam_index = int(index)
        frames = getattr(self, "_camera_frames", {})
        frame = frames.get(self._current_cam_index)
        if frame is not None:
            self._on_camera_frame(frame)
        self._update_main_info()

    def _update_main_info(self) -> None:
        """刷新可视化面板状态栏:相机来源 / 骨架节点 / NDI。"""
        if not hasattr(self, "main_info"):       # 构建期(tab 页先于状态栏)调用时跳过
            return
        profile = self.hardware.profile
        cam_src = f"{profile.camera_backend.value.upper()}×{profile.camera_count}"
        cam_show = f"#{self._current_cam_index + 1}" if self._current_cam_index > 0 else ""
        nodes = "?"
        if self.runtime is not None:
            nodes = str(self.runtime.descriptor.n_nodes)
        ndi = (f"{profile.ndi_backend.value.upper()} / "
               f"{self.hardware.states['ndi'].value.upper()}")
        self.main_info.setText(
            f"相机: {cam_src}{cam_show} | 骨架: {nodes}节点 | NDI: {ndi}")

    def _gray(self, bgr):
        import numpy as np
        return np.mean(np.asarray(bgr, dtype=np.float64), axis=2).astype(np.uint8)

    def _refresh_anchor_controls(self) -> None:
        if not hasattr(self, "camera_anchor_btn"):
            return
        scene_editable = bool(self.session is not None and not self.session.replay_only)
        for button in (self.tool_target_btn, self.tool_skeleton_btn, self.tool_obstacle_btn):
            button.setEnabled(scene_editable)
        missing = []
        if self.session is None:
            missing.append("新建实验")
        if self.runtime is None:
            missing.append("加载模型")
        if self.hardware.states["camera"] != DeviceState.READY:
            missing.append("相机 READY")
        if self._latest_frame is None:
            missing.append("收到相机帧")
        if self.hardware.profile.camera_backend == BackendMode.REAL and self.runtime is not None:
            reference = getattr(self.runtime, "reference_frame_path", None)
            if reference is None or not Path(reference).is_file():
                missing.append("部署包参考背景")
        self.camera_anchor_btn.setEnabled(not missing)
        if missing:
            self.anchor_prereq.setText("相机锚定尚缺：" + "、".join(missing))
            self.anchor_prereq.setStyleSheet("color:#B7791F;font-size:11px;")
        else:
            self.anchor_prereq.setText("相机锚定已就绪：当前帧 + 最近 H 步动作历史将成为 Planner 起点")
            self.anchor_prereq.setStyleSheet("color:#18794E;font-size:11px;")

        mock_warmup = (self.hardware.profile.valve_backend == BackendMode.MOCK)
        warmup_ready = bool(self.session is not None and self.runtime is not None and mock_warmup)
        self.warmup_btn.setEnabled(warmup_ready)
        if mock_warmup:
            self.warmup_btn.setText("生成 Mock Warmup 历史")
            self.warmup_btn.setToolTip("仅用于 Mock/算法调试：生成训练分布内的 H 步动作历史，不控制真阀。")
            self.zero_history_cb.setEnabled(True)
            self.zero_history_cb.setToolTip(
                "勾选后用全 0 动作历史起步，可跳过 Mock Warmup。\n"
                "初始窗口是 OOD，后续用执行动作逐步替换。")
        else:
            self.warmup_btn.setText("真机 Warmup 尚未接入")
            self.warmup_btn.setToolTip("真机 Warmup 必须实际下发并用 ACK applied6 建立历史；当前 fail-closed。")
            self.zero_history_cb.setEnabled(True)
            self.zero_history_cb.setToolTip(
                "Real 模式允许操作员显式接受零历史起步。\n"
                "前 H 步历史与训练分布不一致，初始预测可能偏差较大；\n"
                "后续执行会用 ACK applied6 逐步替换零填充历史。")

    def _warmup(self) -> None:
        if not self.runtime or self.runtime.descriptor.action_scale_kpa is None:
            self._error("warmup 需要已加载带 manifest 的模型")
            return
        from ..runtime.warmup import warmup_actions
        descriptor = self.runtime.descriptor
        seq = warmup_actions(
            descriptor.action_dim, descriptor.history_steps, kind="ramp",
            channel_equalities=descriptor.channel_equalities)
        self._action_history = [tuple(float(v) for v in row) for row in seq]
        # 简化:用 mock 传输"下发"填历史(真机用 QtValveTransport)
        self.warmup_btn.setText(f"Mock Warmup 已生成:{len(seq)} 步")
        self._log(f"Mock warmup: {len(seq)} 步动作历史已就绪（未下发真阀）")

    def _on_zero_history_toggled(self, checked: bool) -> None:
        self.zero_hist_hint.setText(
            "已启用:零历史起步(OOD,首窗口可能不准)" if checked
            else "已禁用:锚定需先 Warmup 填真实历史")
        self.zero_hist_hint.setStyleSheet(
            "color:#F6AD55;font-size:11px;" if checked else "color:#486581;font-size:11px;")

    def _camera_anchor(self) -> None:
        import numpy as np   # 冒烟分支 np.asarray 用(本模块 numpy 均为方法局部 import)
        if self.session is None or self._latest_frame is None or not self.runtime:
            self._error("相机锚定需要：已建实验、已加载模型、相机 READY 并收到帧")
            return
        if not self._action_history and not self.zero_history_cb.isChecked():
            self._error("无动作历史:勾选『零历史起步』可免 warmup 直接锚定,或先点 Warmup")
            return
        from ..runtime.anchors import anchor_from_camera_frame
        descriptor = self.runtime.descriptor
        manifest = self.runtime.manifest
        if self.hardware.profile.camera_backend == BackendMode.MOCK:
            bg = None
            segmentation_method = "backlight"
            segment_params = {"thresh": 60}
        else:
            reference = getattr(self.runtime, "reference_frame_path", None)
            if reference is None or not Path(reference).is_file():
                self._error("真实相机锚定需要 deploy_manifest.reference_frame 参考背景")
                return
            from ..perception._compat import cv2
            if cv2 is None:
                self._error("读取参考背景需要 OpenCV")
                return
            bg = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
            if bg is None or bg.shape != self._latest_frame.shape[:2]:
                self._error("参考背景无法读取或尺寸与当前相机帧不一致")
                return
            segmentation_method = "white_on_blue"
            segment_params = manifest.segment_params if manifest else {}
        area_median = manifest.mask_area_median_px if manifest else None
        if area_median is None:
            if segmentation_method != "backlight":
                self._error("真实相机锚定需要 deploy_manifest.mask_area_median_px")
                return
            area_median = float(self._latest_frame.shape[0] * self._latest_frame.shape[1] * 0.035)
        anchor, quality, skeleton = anchor_from_camera_frame(
            self._latest_frame, background_gray=bg,
            segment_params=segment_params,
            n_nodes=descriptor.n_nodes, model=self.runtime.model,
            action_history=self._action_history, area_median_px=float(area_median),
            frame_ref=f"camera_live#{self.hardware.profile.camera_backend.value}",
            zero_pad_history=self.zero_history_cb.isChecked(),
            segmentation_method=segmentation_method)
        if anchor is None:
            self._error(f"帧质量 reject:{quality.reasons};请重试或调场景")
            return
        self.session.set_anchor(anchor)
        # 打磨③:页间引导 —— 锚定成功提示下一步
        zero_note = " · 零历史起步(OOD)" if self.zero_history_cb.isChecked() else ""
        self.anchor_status.setText(
            f"已锚定 {anchor.anchor_id[:8]} verdict={quality.verdict}{zero_note} → 可前往 3 Plan 规划")
        self.main_display.set_anchor(skeleton)
        self._refresh()

    def _plan_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)

        # 卡1:规划参数(紧凑行,不再每字段独占一行)
        gb_param = QGroupBox("规划参数")
        p = QVBoxLayout(gb_param); p.setContentsMargins(10, 12, 10, 10); p.setSpacing(6)
        self.plan_k = QSpinBox(); self.plan_k.setRange(1, 10000); self.plan_k.setValue(20)
        self.plan_iter = QSpinBox(); self.plan_iter.setRange(1, 100000); self.plan_iter.setValue(400)
        self.plan_restarts = QSpinBox(); self.plan_restarts.setRange(1, 32); self.plan_restarts.setValue(4)
        self.plan_dt = QDoubleSpinBox(); self.plan_dt.setRange(0.01, 60); self.plan_dt.setValue(0.2)
        self.plan_dt.setDecimals(3)
        self.channel_map = QLineEdit("0")
        row = QHBoxLayout()
        row.addWidget(QLabel("K")); row.addWidget(self.plan_k)
        row.addWidget(QLabel("迭代")); row.addWidget(self.plan_iter)
        row.addWidget(QLabel("多起点")); row.addWidget(self.plan_restarts)
        p.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("周期(s)")); row.addWidget(self.plan_dt)
        row.addWidget(QLabel("通道映射")); row.addWidget(self.channel_map)
        row.addStretch()
        p.addLayout(row)
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
        self.plan_summary.setMaximumHeight(90)
        self.plan_summary.setPlaceholderText("异步 shooting planner 与交互式候选预览将在此页接入。")
        a.addWidget(self.plan_summary)
        root.addWidget(gb_act)

        self.plan_preview = PlanPreviewWidget(); root.addWidget(self.plan_preview, 1)
        return page

    def _execute_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)

        # 卡1:执行控制
        gb_ctrl = QGroupBox("执行控制")
        c = QHBoxLayout(gb_ctrl); c.setContentsMargins(12, 14, 12, 12)
        self.arm_button = QPushButton("Arm / Confirm"); self.arm_button.setObjectName("primary")
        self.arm_button.clicked.connect(self._arm)
        self.execute_button = QPushButton("运行 Mock 计划"); self.execute_button.setObjectName("primary")
        self.execute_button.clicked.connect(self._execute)
        self.pause_button = QPushButton("归零并重新锚定"); self.pause_button.setObjectName("accent")
        self.pause_button.clicked.connect(self._pause)
        self.resume_button = QPushButton("Resume"); self.resume_button.setObjectName("accent")
        self.resume_button.clicked.connect(self._resume)
        self.resume_button.hide()  # pause_policy=zero 时不允许恢复旧计划
        for button in (self.arm_button, self.execute_button, self.pause_button):
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

    def _results_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)
        gb = QGroupBox("结果与指标")
        g = QVBoxLayout(gb); g.setContentsMargins(12, 14, 12, 12)
        self.results = QPlainTextEdit(); self.results.setReadOnly(True)
        self.results.setPlaceholderText("执行记录保存在 run/execution.csv；自动指标将在后续接入。")
        g.addWidget(self.results)
        root.addWidget(gb)
        return page

    def _browse_button(self, edit: QLineEdit, directory: bool) -> QPushButton:
        """返回一个"…"按钮,点击弹文件/目录选择并把结果写回 edit(供紧凑行内联使用)。"""
        button = QPushButton("…")
        def browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text())
            if path:
                edit.setText(path)
        button.clicked.connect(browse)
        return button

    def _path_row(self, edit: QLineEdit, directory: bool) -> QWidget:
        holder = QWidget(); row = QHBoxLayout(holder); row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1); row.addWidget(self._browse_button(edit, directory))
        return holder

    def _new_session(self) -> None:
        try:
            self.session = ExperimentSession.create(self.run_root.text().strip())
            atomic_write_json(self.session.run_dir / "hardware_profile.json",
                              self.hardware.profile.to_dict())
            self._log(f"创建 {self.session.run_dir}")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _open_replay(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择 run 目录", self.run_root.text())
        if not path:
            return
        try:
            self.session = ExperimentSession.load_for_replay(path)
            self.runtime = None
            self.model_summary.setPlainText(
                "Replay-only\n" + (f"checkpoint={self.session.model.checkpoint}\n"
                if self.session.model else "model=None\n"))
            self._scene_changed(display_only=True)
            if self.session.plan:
                self.plan_summary.setPlainText("历史计划（只读，不能 Arm）")
                if self.session.plan.predicted_states_path:
                    self.plan_preview.set_plan(self.session.plan, self.session.scene,
                                               self.session.safety, self.session.run_dir)
            self._refresh()
        except Exception:
            self._error(traceback.format_exc())

    def _load_model(self) -> None:
        if self.session is None:
            self._error("请先 New Experiment")
            return
        if self._model_thread and self._model_thread.isRunning():
            return
        checkpoint = self.checkpoint.text().strip()
        if not checkpoint:
            self._error("请选择 checkpoint")
            return
        self.model_summary.setPlainText("正在后台加载模型……")
        k_safe = self.k_safe.value() or None
        self._model_thread = _ModelLoadThread(
            checkpoint, self.data_dir.text().strip(), self.device.currentText(), k_safe)
        self._model_thread.loaded.connect(self._model_loaded)
        self._model_thread.failed.connect(self._model_load_failed)
        self._model_thread.start()

    def _model_load_failed(self, message: str) -> None:
        # B15:加载失败必须清 runtime,否则操作员看到新 checkpoint 路径、以为换了模型,
        # 实际后续 Plan 用旧 runtime;preflight 比对的两个 hash 都是旧的照样放行。
        self.runtime = None
        self.model_summary.setPlainText("模型未加载")
        if self.session is not None and self.session.model is not None:
            try:
                self.session.invalidate_model("model reload failed")
            except RuntimeError as error:
                self._log(f"WARN: 无法清除旧模型 descriptor: {error}")
        self._error(message)
        self._refresh()

    def _model_loaded(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        assert self.session is not None
        self.session.configure_model(runtime.descriptor)
        descriptor = runtime.descriptor
        if descriptor.channel_map is not None:
            from dataclasses import replace
            groups = required_groups_for_channels(descriptor.channel_map)
            if self.session.safety.required_groups != groups:
                self.session.set_safety(replace(self.session.safety, required_groups=groups))
                self._log(f"安全所需阀组已按 channel_map 设为 {groups}")
        self.model_summary.setPlainText(
            f"type={descriptor.model_type}\nclass={descriptor.model_class}\n"
            f"action_dim={descriptor.action_dim}\n"
            f"nodes={descriptor.n_nodes}\nH={descriptor.history_steps}\n"
            f"K_train={descriptor.k_train}\nK_safe={descriptor.k_safe}\n"
            f"train_dt={descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s}\n"
            f"action_scale_kpa={descriptor.action_scale_kpa}\n"
            f"sha256={descriptor.checkpoint_hash}")
        # B5:plan_dt 默认取训练实测 Δt(不再硬编码 0.2)
        ref_dt = descriptor.train_dt_measured_s or descriptor.train_dt_nominal_s
        if ref_dt:
            self.plan_dt.setValue(float(ref_dt))
        # B9:K_safe 从 k_safe_table_px 自动读(按 10px 容差),不再手填
        k_safe_source = "手动"
        if descriptor.k_safe_table_px:
            k = (descriptor.k_safe_table_px.get("10px")
                 or descriptor.k_safe_table_px.get("5px"))
            if k:
                self.k_safe.setValue(int(k))
                k_safe_source = "认证表(10px 容差)" if "10px" in descriptor.k_safe_table_px else "认证表"
        # 打磨③:K_safe 来源标注(唯一安全门,操作员需知它是自动还是手动)
        if descriptor.k_safe_table_px:
            self.k_safe.setToolTip(f"K_safe 来源: {k_safe_source}(视野认证表)。"
                                   f"这是规划视野上限,修改后 Preflight 按新值门禁。")
        else:
            self.k_safe.setToolTip("K_safe 来源: 手动。该模型无视野认证表,规划由 preflight 的 k_safe_uncertified 门保护。")
        self.model_summary.appendPlainText(f"K_safe 来源: {k_safe_source}")
        self._refresh()
        self._update_main_info()

    def _apply_safety(self) -> None:
        if not self.session:
            self._error("请先 New Experiment")
            return
        columns = list(zip(*[[cell.value() for cell in row]
                             for row in self._safety_cells]))
        try:
            mapping = (self.runtime.descriptor.channel_map if self.runtime is not None
                       else tuple(int(value.strip()) for value in
                                  self.channel_map.text().split(",") if value.strip()))
            groups = required_groups_for_channels(mapping or (0,))
            safety = SafetyPolicy(
                pressure_min6=tuple(columns[0]), pressure_max6=tuple(columns[1]),
                rise_rate6=tuple(columns[2]), fall_rate6=tuple(columns[3]),
                initial_action6=tuple(columns[4]), required_groups=groups)
            self.session.set_safety(safety)
            atomic_write_json(self.session.run_dir / "safety.json", safety.to_dict())
            self._log("安全配置已应用；旧计划已失效")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _connect_valve_group(self, gid: int) -> None:
        """按已应用 backend 连接单组阀；Mock/Real 共用同一控制器路径。"""
        if self._valve_connect_thread and self._valve_connect_thread.isRunning():
            return
        try:
            self._require_ui_profile_applied()
            if self.hardware.profile.valve_backend == BackendMode.REAL:
                port = {1: self.hardware.profile.group1_port,
                        2: self.hardware.profile.group2_port}[gid].strip()
                if not port:
                    raise RuntimeError(f"组{gid} 未配置串口")
            self.hardware.prepare_valves()
        except Exception as error:
            self._error(f"阀连接准备失败: {error}")
            return
        self._valve_connect_thread = _ValveConnectThread(self.hardware, groups=(gid,))
        self._valve_connect_thread.connected.connect(self._valve_connected)
        self._valve_connect_thread.failed.connect(self._valve_connect_failed)
        self._valve_connect_thread.start()

    def _valve_connected(self, controller, summary: str) -> None:
        self.valve_controller = self.hardware.valve_controller
        backend = self.hardware.profile.valve_backend.value.upper()
        self._log(f"{backend} 阀: {summary}")
        self._refresh_valve_status()

    def _valve_connect_failed(self, message: str) -> None:
        self._log(f"ERROR: 阀连接失败 {message}")
        self._refresh_valve_status()

    def _refresh_valve_status(self) -> None:
        """按 controller.connected_groups 刷新每组状态 + 整体执行通道(真阀/Mock)。"""
        controller = self.hardware.valve_controller
        g1 = bool(controller and 1 in controller.connected_groups)
        g2 = bool(controller and 2 in controller.connected_groups)
        for lbl, ok in ((self.hw_g1_status, g1), (self.hw_g2_status, g2)):
            lbl.setText("已连" if ok else "未连")
            lbl.setStyleSheet("color:#38A169;font-size:11px;" if ok else "color:#888;font-size:11px;")
        self.valve_controller = controller
        self._refresh()

    def _disconnect_valve(self) -> None:
        self.hardware.disconnect_valves(zero=True)
        self.valve_controller = self.hardware.valve_controller
        self._refresh_valve_status()

    def _connect_ndi(self) -> None:
        """显式连接/断开 NDI；Mock 也必须由操作员启动。"""
        if self.hardware.ndi_thread is not None:
            self.hardware.stop_ndi()
            self.ndi_thread = None
            return
        try:
            self._require_ui_profile_applied()
            self._ndi_confirmed = False
            self.hardware.start_ndi()
            self.ndi_thread = self.hardware.ndi_thread
            self._log(f"NDI 已启动: {self.hardware.profile.ndi_backend.value.upper()} "
                      f"×{self.hardware.profile.ndi_count}（隐藏评价流）")
        except Exception as error:
            self._error(f"NDI 连接失败: {error}")

    def _on_mock_ndi_data(self, data, _t) -> None:
        """Mock NDI 仅喂曲线(显示用),不改连接状态 —— 避免把 Mock 误报成已连接。"""
        self._push_ndi(data, _t)

    def _on_ndi_error(self, message: str) -> None:
        """兼容旧 NDI 信号的日志入口。"""
        self._log(f"ERROR: NDI 错误: {message}")
        self._update_main_info()

    def _on_ndi_data(self, data: list, _t: float) -> None:
        """真实 NDI 末端 → 曲线 + 主显示 NDI 图层(紫星,第 1 探头);只显示,不进模型。

        数据布局(对齐 real_capture):每探头 11 维 [x,y,z,Rx,Ry,Rz,qw,qx,qy,qz,quality]。
        首次收到数据才把状态转绿(真实连接成功),否则保持"连接中/失败"。
        """
        self._push_ndi(data, _t)
        if data and not getattr(self, "_ndi_confirmed", False):
            self._ndi_confirmed = True
            self._update_main_info()
        try:
            if data and len(data) >= 3:
                # 第 1 探头 x,y(平面);多探头数据更全但主显示只取 probe0 平面
                self.main_display.set_ndi_position((float(data[0]), float(data[1])))
        except Exception:
            pass

    # ---- 实时曲线(可视化面板,参考 real_capture 右栏 pyqtgraph)----
    def _init_viz_buffers(self) -> None:
        """气压(6ch)与 NDI(X/Y/Z)滚动缓冲。在 Mock NDI 线程启动前初始化。"""
        import numpy as np
        self._p_buf = np.zeros((6, 300))
        self._ndi_buf = np.full((3, 300), np.nan)

    def _push_pressure(self, applied6) -> None:
        """气压命令曲线:applied6(6 通道 kPa)推进滚动缓冲,6 条曲线同步更新。"""
        buf = getattr(self, "_p_buf", None)
        curves = getattr(self, "_p_curves", None)
        if buf is None or curves is None:
            return
        for ch, value in enumerate(list(applied6)[:6]):
            buf[ch, :-1] = buf[ch, 1:]
            buf[ch, -1] = float(value)
        for ch, curve in enumerate(curves):
            curve.setData(buf[ch])

    def _push_ndi(self, data, _t) -> None:
        """NDI 曲线:第 1 探头 X/Y/Z 推进滚动缓冲(仅显示,不进模型)。"""
        buf = getattr(self, "_ndi_buf", None)
        curves = getattr(self, "_ndi_curves", None)
        if buf is None or curves is None:
            return
        values = [float("nan")] * 3
        if data and len(data) >= 3:
            for i in range(3):
                v = float(data[i])
                values[i] = v if abs(v) < 1e6 else float("nan")   # 失锁置 NaN 不画
        for i in range(3):
            buf[i, :-1] = buf[i, 1:]
            buf[i, -1] = values[i]
        for i, curve in enumerate(curves):
            curve.setData(buf[i])

    def _load_anchor(self) -> None:
        self._load_session_json("anchor", Anchor.from_dict)

    def _load_anchor_npz(self) -> None:
        if not self.session or not self.runtime:
            self._error("请先创建 session 并加载模型")
            return
        try:
            anchor = anchor_from_npz(
                self.anchor_npz.text().strip(), self.anchor_index.value(),
                self.runtime.descriptor, self.runtime.model, padding="reject")
            self.session.set_anchor(anchor)
            atomic_write_json(self.session.run_dir / "anchor.json", anchor.to_dict())
            self._scene_changed()
        except FileNotFoundError as error:
            self._error(f"找不到 NPZ 文件:\n{error}\n\n请把 transition NPZ 拷入 "
                        f"real_validation/data/npz/,或点击 … 选择现有文件。")
        except IndexError as error:
            self._error(f"帧索引越界:\n{error}\n\n请把『帧索引』改到数据帧数范围内"
                        f"(示例数据 0~8171)。")
        except ValueError as error:
            # anchor_from_npz 的格式/节点/历史/NaN 错误;转成可读引导
            message = str(error)
            hint = ""
            if "缺少" in message and "步历史" in message:
                hint = ("\n\n该帧往前凑不满模型历史 H。把『帧索引』调大"
                        "(如示例数据用 39 及以上),或换更长序列。")
            elif "节点形状" in message or "N=" in message:
                hint = ("\n\nNPZ 的节点数与模型不匹配(模型 "
                        f"n_nodes={self.runtime.descriptor.n_nodes})。"
                        "请选同节点数的 npz。")
            elif "action_dim" in message:
                hint = (f"\n\nNPZ 动作维度与模型 action_dim="
                        f"{self.runtime.descriptor.action_dim} 不匹配。")
            elif "positions 和 actions" in message:
                hint = ("\n\n该文件不是 transition NPZ,或格式不符。需要 "
                        "positions(T,3,N) + actions(T,D)。")
            self._error(message + hint)

    def _load_scene(self) -> None:
        self._load_session_json("scene", Scene.from_dict)

    def _set_target(self) -> None:
        if not self.session:
            self._error("请先 New Experiment")
            return
        retained = tuple(item for item in self.session.scene.primitives
                         if not item.kind.startswith("target_"))
        kind = "target_circle" if self.target_radius.value() > 0 else "target_point"
        primitive = ScenePrimitive(kind, "model", {
            "xy": [self.target_x.value(), self.target_y.value()],
            "radius": self.target_radius.value(), "node": 0,
        }, name="tip_target")
        self.session.set_scene(Scene(self.session.scene.name, retained + (primitive,),
                                     self.session.scene.dimension))
        self._scene_changed()

    def _add_obstacle(self) -> None:
        if not self.session:
            self._error("请先 New Experiment")
            return
        primitive = ScenePrimitive("obstacle_circle", "model", {
            "center": [self.obstacle_x.value(), self.obstacle_y.value()],
            "radius": self.obstacle_radius.value(),
        }, name=f"obstacle_{len(self.session.scene.primitives)}")
        self.session.set_scene(self.session.scene.with_primitive(primitive))
        self._scene_changed()

    def _scene_changed(self, display_only: bool = False) -> None:
        assert self.session is not None
        if not display_only:
            atomic_write_json(self.session.run_dir / "scene.json", self.session.scene.to_dict())
        # 同步可视化:数值添加(_set_target/_add_obstacle)与加载 scene.json 也走这里,
        # 同步可视化:主显示(右上面板)+ 原语列表
        self.main_display.set_scene(self.session.scene)
        self.scene_editor.set_scene(self.session.scene)
        self.scene_summary.setPlainText(
            f"anchor={self.session.anchor.anchor_id if self.session.anchor else 'None'}\n"
            f"scene={self.session.scene.name}\nprimitives={len(self.session.scene.primitives)}\n"
            f"scene_digest={self.session.scene.digest}")
        self._refresh()

    def _load_session_json(self, kind: str, factory) -> None:
        if self.session is None:
            self._error("请先 New Experiment")
            return
        path, _ = QFileDialog.getOpenFileName(self, f"加载 {kind}", "", "JSON (*.json)")
        if not path:
            return
        try:
            value = factory(read_json(path))
            if kind == "anchor":
                self.session.set_anchor(value)
            else:
                self.session.set_scene(value)
            self._scene_changed()
        except Exception:
            self._error(traceback.format_exc())

    def _load_plan(self) -> None:
        if self.session is None:
            self._error("请先 New Experiment")
            return
        path, _ = QFileDialog.getOpenFileName(self, "加载 plan", "", "JSON (*.json)")
        if not path:
            return
        try:
            plan = ActionPlan.from_dict(read_json(path))
            result = self.session.accept_plan(plan)
            if result.ok:
                atomic_write_json(self.session.run_dir / "plan.json", plan.to_dict())
                write_actions6_csv(plan, self.session.run_dir / "planned_actions6.csv")
            self._show_preflight(result)
        except Exception:
            self._error(traceback.format_exc())

    def _start_planning(self) -> None:
        if not self.session or not self.runtime or not self.session.anchor:
            self._error("规划前需要 session、模型和 anchor")
            return
        if self._planning_thread and self._planning_thread.isRunning():
            return
        try:
            mapping = tuple(int(value.strip()) for value in self.channel_map.text().split(",")
                            if value.strip())
            config = ShootingConfig(
                horizon=self.plan_k.value(), n_iter=self.plan_iter.value(),
                n_restarts=self.plan_restarts.value(), random_seed=0)
            self.session.begin_planning()
            kwargs = dict(
                anchor=self.session.anchor, scene=self.session.scene,
                safety=self.session.safety, channel_map=mapping,
                step_interval_s=self.plan_dt.value(), output_dir=self.session.run_dir,
                config=config)
            self._planning_thread = _PlanningThread(OpenLoopShootingPlanner(self.runtime), kwargs)
            self._planning_thread.planned.connect(self._planning_done)
            self._planning_thread.failed.connect(self._planning_failed)
            self._planning_thread.start()
            self.plan_summary.setPlainText("规划中……")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _cancel_planning(self) -> None:
        if self._planning_thread and self._planning_thread.isRunning():
            self._planning_thread.cancel()

    def _planning_done(self, plan: ActionPlan) -> None:
        assert self.session is not None
        result = self.session.accept_plan(plan)
        atomic_write_json(self.session.run_dir / "plan.json", plan.to_dict())
        write_actions6_csv(plan, self.session.run_dir / "planned_actions6.csv")
        self._show_preflight(result)

    def _planning_failed(self, error: str) -> None:
        if self.session and self.session.state == SessionState.PLANNING:
            self.session.transition(SessionState.IDLE, error)
        self.plan_summary.setPlainText("规划失败/取消：" + error)
        self._refresh()

    def _run_preflight(self) -> None:
        if not self.session or not self.session.plan:
            self._error("尚未导入有效计划")
            return
        result = self.session.accept_plan(self.session.plan)
        self._show_preflight(result)

    def _show_preflight(self, result) -> None:
        import numpy as np
        if result.ok:
            # 打磨③:页间引导 —— preflight 通过提示可去 Execute
            self.plan_summary.setPlainText("Preflight: PASS → 可前往 4 Execute 进行 Arm")
            if self.session and self.session.plan and self.session.plan.predicted_states_path:
                try:
                    self.plan_preview.set_plan(self.session.plan, self.session.scene,
                                               self.session.safety, self.session.run_dir)
                except Exception as error:
                    self.plan_summary.appendPlainText(f"\nPreview unavailable: {error}")
            # 主显示区叠加预测轨迹(读 predicted_states.npz)
            # 失败/缺失一律清空,防历史 run/replay 的 corrupt npz 抛未捕获、旧轨迹残留。
            states = np.zeros((0, 0, 0))
            try:
                if self.session and self.session.plan and self.session.plan.predicted_states_path:
                    p = Path(self.session.plan.predicted_states_path)
                    if not p.is_absolute():
                        p = self.session.run_dir / p
                    if p.is_file():
                        with np.load(p) as data:
                            key = "states_model" if "states_model" in data else "states_normalized"
                            states = np.asarray(data[key])
            except Exception as error:
                self.plan_summary.appendPlainText(f"\n预测轨迹叠加失败: {error}")
                states = np.zeros((0, 0, 0))
            self.main_display.set_predicted_states(states)
        else:
            self.plan_summary.setPlainText("Preflight: BLOCKED\n" + "\n".join(
                f"[{item.code}] {item.message}" for item in result.issues))
            self.plan_preview.clear_plan()
            self.main_display.set_predicted_states(np.zeros((0, 0, 0)))
        self._refresh()

    def _arm(self) -> None:
        try:
            if not self.session:
                raise RuntimeError("没有 session")
            if not self.session.plan:
                raise RuntimeError("没有已通过 Preflight 的计划")
            groups = required_groups_for_channels(self.session.plan.channel_map)
            self.hardware.require_valves_ready(groups)
            self.session.arm(); self._log("计划已由操作员 Arm")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _make_transport(self):
        """只从 HardwareSession 创建 transport，不存在隐式 Mock fallback。"""
        if not self.session or not self.session.plan:
            raise RuntimeError("尚无执行计划")
        groups = required_groups_for_channels(self.session.plan.channel_map)
        return self.hardware.create_transport(groups)

    def _execute(self) -> None:
        if not self.session or self.session.state != SessionState.ARMED or not self.session.plan:
            self._error("计划必须先通过 Preflight 并 Arm")
            return
        # 功能①:执行时累积本次实验的真实动作历史,供后续滚动重锚定/重规划
        from ..runtime.observation_policy import ActionHistoryBuffer
        descriptor = self.runtime.descriptor if self.runtime else None
        if descriptor is not None and descriptor.channel_map is not None:
            if (getattr(self, "_history_buffer", None) is None
                    or self._history_buffer.history_steps != descriptor.history_steps
                    or self._history_buffer.channel_map != descriptor.channel_map):
                self._history_buffer = ActionHistoryBuffer(
                    descriptor.history_steps, descriptor.action_dim,
                    descriptor.channel_map)
        try:
            transport = self._make_transport()
        except Exception as error:
            self._error(f"禁止执行: {error}")
            return
        self.executor = PlanExecutor(
            transport, self.session.safety,
            history_buffer=getattr(self, "_history_buffer", None))
        backend = self.hardware.profile.valve_backend.value
        self.session.transition(SessionState.EXECUTING, f"{backend} execution")
        self._execution_thread = _ExecutionThread(
            self.executor, self.session.plan, self.session.run_dir / "execution.csv")
        self._execution_thread.event.connect(self._on_execution_event)
        self._execution_thread.finished_ok.connect(self._execution_done)
        self._execution_thread.failed.connect(self._execution_failed)
        self._execution_thread.finished.connect(self._refresh)
        self._execution_thread.start(); self._refresh()

    def _on_execution_event(self, name: str, payload) -> None:
        """执行线程事件:记日志;command 事件把该步 applied6 喂气压曲线。"""
        self._log(f"{name}: {payload}")
        if name == "command":
            try:
                self._push_pressure(payload["receipt"]["applied6"])
            except Exception:
                pass

    def _execution_done(self, receipts) -> None:
        assert self.session is not None
        self.session.transition(SessionState.COMPLETED, "all commands acked")
        # P4:执行摘要 —— 命令安全 + jitter 统计(替代占位符)
        from ..execution.metrics import evaluate_command_safety, evaluate_plan_scene
        plan = self.session.plan
        actions6 = [tuple(r.applied6) for r in receipts]
        safety_metrics = evaluate_command_safety(
            actions6, plan.step_interval_s if plan else 0.1,
            self.session.safety)
        jitters = [getattr(r, "jitter_s", None) for r in receipts]
        jitters = [j for j in jitters if j is not None]
        jitter_summary = (f"jitter mean={sum(jitters) / len(jitters) * 1e3:.1f}ms "
                          f"max={max(jitters) * 1e3:.1f}ms" if jitters else "jitter 无记录")

        # 打磨①:计划侧场景指标(predicted_states + scene,离线下即可算)
        plan_scene_summary = ""
        if plan and plan.predicted_states_path:
            states_path = Path(plan.predicted_states_path)
            if not states_path.is_absolute():
                states_path = self.session.run_dir / states_path
            if states_path.is_file():
                try:
                    import numpy as np
                    with np.load(states_path) as data:
                        key = "states_model" if "states_model" in data else "states_normalized"
                        states = np.asarray(data[key], dtype=np.float32)
                    scene_metrics = evaluate_plan_scene(
                        states, self.session.scene, tip_node=0)
                    plan_scene_summary = (
                        f"末端目标距离: {scene_metrics.get('terminal_target_distance', float('nan')):.2f} px  "
                        f"目标达成: {'✓' if scene_metrics.get('target_success') else '✗'}\n"
                        f"最小障碍间距: {scene_metrics.get('minimum_obstacle_clearance', float('nan')):.2f} px  "
                        f"碰撞: {'是' if scene_metrics.get('collision') else '否'}\n")
                except Exception as error:
                    plan_scene_summary = f"(计划场景指标不可用: {error})\n"

        self.results.setPlainText(
            f"执行完成:{len(receipts)} 条命令\n"
            f"{plan_scene_summary}"
            f"压力越界:{safety_metrics['pressure_violation_count']}  "
            f"速率越界:{safety_metrics['slew_violation_count']}\n"
            f"{jitter_summary}\n"
            f"prediction-to-execution gap: 待真机闭环(M5)\n"
            f"{self.session.run_dir / 'execution.csv'}")
        self._refresh()

    def _execution_failed(self, error: str) -> None:
        if self.session and self.session.state == SessionState.REANCHOR \
                and "operator_abort" in error:
            self._log("已安全归零；等待重新 Anchor / Plan")
            self._refresh()
            return
        if self.session and self.session.state in {SessionState.EXECUTING, SessionState.ABORTING}:
            target = SessionState.ZEROED if self.session.state == SessionState.ABORTING else SessionState.ERROR
            self.session.transition(target, error)
        if "operator_abort" in error and self.session and self.session.state == SessionState.ZEROED:
            self._log("执行已中止并安全归零")
        else:
            self._error(error)
        self._refresh()

    def _pause(self) -> None:
        if self.executor and self.session and self.session.state == SessionState.EXECUTING:
            # 默认 zero-pause 会改变真实初态；中止 worker 后由其完成归零，
            # 直接进入 REANCHOR，绝不暴露 Resume 旧计划。
            self.session.transition(SessionState.REANCHOR, "operator zero-pause")
            self.executor.abort()
            self._log("执行已中止并归零；必须重新 Anchor / Plan")
            self._refresh()

    def _resume(self) -> None:
        if self.executor and self.session and self.session.state == SessionState.PAUSED:
            self.executor.resume(); self.session.transition(SessionState.EXECUTING, "operator resume")
            self._refresh()

    def _abort(self) -> None:
        if not self.session or self.session.state not in {
                SessionState.EXECUTING, SessionState.PAUSED, SessionState.ARMED}:
            return
        if self.session.state == SessionState.ARMED and self.executor is None:
            self.session.transition(SessionState.ABORTING, "operator abort")
            self._start_zero(SessionState.ZEROED)
        elif self.executor:
            self.session.transition(SessionState.ABORTING, "operator abort")
            self.executor.abort(); self._refresh()
        self._refresh()

    def _zero(self) -> None:
        if not self.session:
            return
        if self.session.state in {SessionState.ARMED, SessionState.EXECUTING,
                                  SessionState.PAUSED}:
            self._abort()
            return
        try:
            target = (SessionState.ZEROED if self.session.state in {
                SessionState.COMPLETED, SessionState.ERROR, SessionState.ABORTING,
                SessionState.PAUSED} else None)
            self._start_zero(target)
        except Exception as error:
            self._error(str(error))

    def _start_zero(self, target: SessionState | None) -> None:
        if self._zero_thread and self._zero_thread.isRunning():
            return
        transport = self.executor.transport if self.executor else self._make_transport()
        self._zero_target = target
        self._zero_thread = _ZeroThread(transport, self.session.safety.ack_timeout_s)
        self._zero_thread.finished_ok.connect(self._zero_done)
        self._zero_thread.failed.connect(self._zero_failed)
        self._zero_thread.start()

    def _zero_done(self, receipt) -> None:
        if self.session and self._zero_target is not None and self.session.state != self._zero_target:
            self.session.transition(self._zero_target, "operator zero")
        backend = self.hardware.profile.valve_backend.value.upper()
        self._log(f"六通道已归零（{backend}，{receipt.status}）")
        self._zero_target = None
        self._refresh()

    def _zero_failed(self, message: str) -> None:
        self._zero_target = None
        self._error(f"归零未确认: {message}")

    def _load_hardware_config(self) -> None:
        """启动时加载硬件连接配置(config/hardware.json,gitignore 不入库)。"""
        path = APP_DIR / "config" / "hardware.json"
        if not path.is_file():
            return
        try:
            value = read_json(path)
            if isinstance(value.get("profile"), dict):
                profile = HardwareProfile.from_dict(value["profile"])
            else:
                # 旧文件只有重载 camera_input；在迁移时一次性解析，
                # 新 GUI 不再使用“0/1/serial 暗示 backend”。
                camera = str(value.get("camera_input", value.get("camera_src", "0"))).strip()
                if camera and camera != "0":
                    if camera.isdigit():
                        count, serials = max(1, min(8, int(camera))), ()
                    else:
                        serials = tuple(item.strip() for item in camera.split(",") if item.strip())
                        count = len(serials) or 1
                    camera_backend = BackendMode.REAL
                else:
                    count, serials, camera_backend = 1, (), BackendMode.MOCK
                profile = HardwareProfile(
                    name="custom", camera_backend=camera_backend,
                    camera_count=count, camera_serials=serials,
                    valve_backend=BackendMode.MOCK,
                    group1_port=str(value.get("group1", "COM3")),
                    group2_port=str(value.get("group2", "COM46")),
                    baudrate=int(value.get("baudrate", 9600)),
                    ndi_backend=BackendMode.MOCK,
                    ndi_port=str(value.get("ndi_port", "COM9")),
                    ndi_count=int(value.get("ndi_count", 1)))
            self._syncing_profile = True
            try:
                preset = profile.name if profile.name in {"all_mock", "real"} else "custom"
                self._set_combo_data(self.hw_profile_preset, preset)
                self._set_combo_data(self.hw_camera_backend, profile.camera_backend.value)
                self._set_combo_data(self.hw_valve_backend, profile.valve_backend.value)
                self._set_combo_data(self.hw_ndi_backend, profile.ndi_backend.value)
                self.hw_camera_count.setValue(profile.camera_count)
                self.hw_camera_serials.setText(",".join(profile.camera_serials))
                self.hw_g1.setText(profile.group1_port); self.hw_g2.setText(profile.group2_port)
                self.hw_baud.setValue(profile.baudrate); self.hw_slave.setValue(profile.slave_addr)
                self.hw_ndi_port.setText(profile.ndi_port); self.hw_ndi_count.setValue(profile.ndi_count)
            finally:
                self._syncing_profile = False
            self._sync_profile_controls()
        except Exception as error:
            self._log(f"WARN: 加载硬件配置失败 {error}")

    def _save_hardware_config(self) -> None:
        path = APP_DIR / "config" / "hardware.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"profile": self._profile_from_ui().to_dict()})
        except Exception as error:
            self._log(f"WARN: 保存硬件配置失败 {error}")

    def _refresh(self) -> None:
        state = self.session.state.value if self.session else "no_session"
        run = self.session.run_dir.name if self.session else "-"
        self.state_label.setText(f"Run: {run}    State: {state}")
        color = STATE_BADGE_COLORS.get(state, STATE_BADGE_COLORS["no_session"])
        self.state_label.setStyleSheet(
            f"background:{CARD};border:2px solid {color};border-radius:12px;"
            f"padding:4px 12px;color:{color};font-weight:bold;")
        valves_ready = self.hardware.states["valve"] == DeviceState.READY
        self.arm_button.setEnabled(bool(self.session and not self.session.replay_only and
                                        self.session.state == SessionState.READY and valves_ready))
        self.execute_button.setEnabled(bool(self.session and self.session.state == SessionState.ARMED))
        backend = self.hardware.profile.valve_backend
        self.execute_button.setText("执行真机计划" if backend == BackendMode.REAL
                                    else "运行 Mock 计划")
        self.pause_button.setEnabled(bool(self.session and self.session.state == SessionState.EXECUTING))
        self.resume_button.setEnabled(False)
        # B8:执行中锁页 1/2/3(否则执行中改 scene 会清空 experiment.json 的 plan,
        # 执行记录与实际下发计划脱钩 = 溯源腐败)
        worker_active = bool(self._execution_thread and self._execution_thread.isRunning())
        executing = bool((self.session and self.session.state in {
            SessionState.EXECUTING, SessionState.PAUSED, SessionState.ARMED})
            or worker_active)
        self.tabs.setTabEnabled(0, not executing)
        self.tabs.setTabEnabled(1, not executing)
        self.tabs.setTabEnabled(2, not executing)
        self._refresh_anchor_controls()

    def _log(self, message: str) -> None:
        self.execution_log.appendPlainText(message)
        if hasattr(self, "log_box"):            # 底部常驻日志(对齐 real_capture)
            self.log_box.appendPlainText(message)

    def _error(self, message: str) -> None:
        self._log("ERROR: " + message)
        QMessageBox.critical(self, "实机验证工作台", message)

    def closeEvent(self, event) -> None:
        if self.session and self.session.state in {
                SessionState.ARMED, SessionState.EXECUTING, SessionState.PAUSED}:
            self._abort()
        if self.executor:
            self.executor.abort()
        if self._execution_thread and self._execution_thread.isRunning():
            deadline = time.monotonic() + 4.0
            while self._execution_thread.isRunning() and time.monotonic() < deadline:
                QApplication.processEvents(); self._execution_thread.wait(20)
        if self._zero_thread and self._zero_thread.isRunning():
            deadline = time.monotonic() + 2.0
            while self._zero_thread.isRunning() and time.monotonic() < deadline:
                QApplication.processEvents(); self._zero_thread.wait(20)
        if self._planning_thread and self._planning_thread.isRunning():
            self._planning_thread.cancel(); self._planning_thread.wait(3000)
        self.hardware.shutdown()                  # 停相机/NDI，归零并关阀
        if self.runtime:
            self.runtime.clear()
        self._save_hardware_config()
        event.accept()


def main() -> int:
    # 相机/阀/NDI 全在 GUI 里选(Setup 页硬件连接卡,自由填写框)。
    # 默认真实相机 ×1(真实是正常系统);无硬件调试时在 Setup 页把『相机』框填 0 用 Mock。
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    window = ValidationWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
