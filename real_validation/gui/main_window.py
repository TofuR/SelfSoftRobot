"""实机验证工作台 GUI 第一版。

运行：``python -m real_validation.main``(入口壳)。
当前完成离线/Mock 会话、模型元数据加载、scene/anchor/plan 导入、preflight 与
Mock ACK 执行；真硬件连接与交互式 scene view 按 TODO 后续阶段接入。
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

if __package__ in (None, ""):  # 支持复制目录后直接 ``python gui/main_window.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    __package__ = "real_validation.gui"

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..execution.executor import MockCommandTransport, PlanExecutor
from ..contracts.io import atomic_write_json, read_json
from ..runtime.model_runtime import ModelRuntime
from ..contracts.models import ActionPlan, Anchor, SafetyPolicy, Scene, ScenePrimitive
from ..planning.openloop_planner import OpenLoopShootingPlanner, ShootingConfig
from ..runtime.anchors import anchor_from_npz
from ..contracts.plan_io import write_actions6_csv
from ..core.session import ExperimentSession, SessionState
from ..widgets import CameraViewWidget, PlanPreviewWidget, SceneEditorPanel
from .theme import QSS, CARD, STATE_BADGE_COLORS, configure_pyqtgraph

APP_DIR = Path(__file__).resolve().parent.parent  # real_validation/ 包根(数据目录 config/checkpoints/data/runs 不变)


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

    def __init__(self, controller, groups: tuple[int, ...]):
        super().__init__()
        self.controller = controller
        self.groups = groups

    def run(self) -> None:
        try:
            from ..hardware.valve import connect_valve_groups
            results = connect_valve_groups(self.controller, groups=self.groups)
            ok_groups = [gid for gid, (ok, _) in results.items() if ok]
            failed_groups = [gid for gid, (ok, _) in results.items() if not ok]
            summary = (f"已连接组: {sorted(ok_groups) or '无'}"
                       + (f" | 失败组: {sorted(failed_groups)}" if failed_groups else ""))
            if not ok_groups:
                self.failed.emit(f"阀连接失败: {summary}")
                return
            self.connected.emit(self.controller, summary)
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


class _CameraThread(QThread):
    frame_ready = pyqtSignal(object)

    def run(self) -> None:
        # 合成弯曲剪影臂(离线演示;真机换 RealSenseCam)
        import time

        import numpy as np
        while not self.isInterruptionRequested():
            frame = np.zeros((240, 320, 3), np.uint8)
            frame[:, :, 0] = 180; frame[:, :, 1] = 70; frame[:, :, 2] = 40
            phase = int(time.time() * 2) % 12
            for row in range(30, 220):
                left = 150 + int(8 * np.sin(phase / 2.0)) + int((row - 30) ** 1.2 / 12.0)
                frame[row, left:left + 11] = (235, 235, 238)
            self.frame_ready.emit(frame)
            self.msleep(200)


class ValidationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SelfSoftRobot 实机验证工作台 · Mock/离线基线")
        self.resize(1400, 860)
        self.session: ExperimentSession | None = None
        self.runtime: ModelRuntime | None = None
        self.executor: PlanExecutor | None = None
        self._model_thread: _ModelLoadThread | None = None
        self._planning_thread: _PlanningThread | None = None
        self._execution_thread: _ExecutionThread | None = None
        self.valve_controller = None   # 真机阀(连接成功后有值;否则 Mock)
        self._valve_connect_thread: _ValveConnectThread | None = None
        configure_pyqtgraph()          # 任何 PlotWidget 之前,保证白底全局生效
        self._build_ui()
        self._load_hardware_config()   # 回填上次保存的串口配置(若有)
        self._refresh()

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

    def _setup_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page); root.setSpacing(8)

        # 卡1:实验与运行
        gb_exp = QGroupBox("实验与运行")
        exp = QVBoxLayout(gb_exp); exp.setContentsMargins(8, 10, 8, 8)
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
        m = QVBoxLayout(gb_model); m.setContentsMargins(8, 10, 8, 8)
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
        self.model_summary.setMaximumHeight(110)
        m.addWidget(self.model_summary, 1)
        root.addWidget(gb_model, 1)

        # 卡3:安全配置(六通道 kPa / kPa·s⁻¹)
        gb_safety = QGroupBox("安全配置(六通道 kPa / kPa·s⁻¹)")
        s = QVBoxLayout(gb_safety); s.setContentsMargins(8, 10, 8, 8)
        self.safety_table = QTableWidget(6, 5)
        self.safety_table.setHorizontalHeaderLabels(["min", "max", "rise/s", "fall/s", "initial"])
        self.safety_table.verticalHeader().setDefaultSectionSize(24)
        self.safety_table.horizontalHeader().setDefaultSectionSize(92)
        self.safety_table.setMinimumHeight(6 * 24 + 26)
        self.safety_table.setMaximumHeight(6 * 24 + 26)
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

        # 卡4:硬件连接(真机阀/NDI —— 设计 spec §3.2 + §5 [Setup])
        gb_hw = QGroupBox("硬件连接(真机)")
        hw = QVBoxLayout(gb_hw); hw.setContentsMargins(8, 10, 8, 8)
        hw_form = QFormLayout()
        self.hw_g1 = QLineEdit("")          # 组1 串口(COM),默认空 = 不接
        self.hw_g1.setPlaceholderText("如 COM3")
        self.hw_g2 = QLineEdit("")          # 组2 串口
        self.hw_g2.setPlaceholderText("如 COM46")
        self.hw_baud = QSpinBox(); self.hw_baud.setRange(4800, 115200)
        self.hw_baud.setValue(9600)
        hw_form.addRow("组1 串口(阀,ch0-2)", self.hw_g1)
        hw_form.addRow("组2 串口(阀,ch3-5)", self.hw_g2)
        hw_form.addRow("baudrate", self.hw_baud)
        hw.addLayout(hw_form)
        hw_buttons = QHBoxLayout()
        self.hw_connect_btn = QPushButton("连接阀"); self.hw_connect_btn.setObjectName("primary")
        self.hw_connect_btn.clicked.connect(self._connect_valve)
        self.hw_disconnect_btn = QPushButton("断开阀"); self.hw_disconnect_btn.setObjectName("danger")
        self.hw_disconnect_btn.setEnabled(False)
        self.hw_disconnect_btn.clicked.connect(self._disconnect_valve)
        hw_buttons.addWidget(self.hw_connect_btn); hw_buttons.addWidget(self.hw_disconnect_btn)
        hw_buttons.addStretch()
        hw.addLayout(hw_buttons)
        self.hw_status = QLabel("未连接(Mock 执行仍可用)")
        self.hw_status.setWordWrap(True)
        self.hw_status.setStyleSheet("color:#486581;font-size:11px;")
        hw.addWidget(self.hw_status)
        root.addWidget(gb_hw)
        return page

    def _observe_page(self) -> QWidget:
        # 打磨②:左右两栏 —— 左控制面板(锚定/目标障碍/相机) + 右可视化(场景编辑)
        page = QWidget(); root = QVBoxLayout(page)
        outer = QSplitter(Qt.Horizontal)

        # ---- 左栏:控制面板 ----
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)

        # 卡1:离线锚定
        gb_off = QGroupBox("离线锚定")
        off = QVBoxLayout(gb_off); off.setContentsMargins(12, 14, 12, 12)
        buttons = QHBoxLayout()
        anchor = QPushButton("加载 anchor.json"); anchor.clicked.connect(self._load_anchor)
        scene = QPushButton("加载 scene.json"); scene.clicked.connect(self._load_scene)
        buttons.addWidget(anchor); buttons.addWidget(scene); buttons.addStretch()
        off.addLayout(buttons)
        offline = QFormLayout()
        # 打磨:默认指向随目录携带的示例数据(15 节点 clean npz),让 Mock 流程可立即跑通
        self.anchor_npz = QLineEdit(str(APP_DIR / "data" / "npz" / "seq_20260627_163921_train.npz"))
        self.anchor_index = QSpinBox(); self.anchor_index.setRange(0, 100000000)
        self.anchor_index.setValue(39)
        self.anchor_index.setToolTip(
            "从该帧建立 anchor。帧索引往前必须凑满模型历史长度 H(此模型 40 步)——\n"
            "选太靠前的帧会报『缺少 N 步历史』。示例数据共 8172 帧,建议从 39 开始。")
        load_npz = QPushButton("从 NPZ 建立 Anchor"); load_npz.setObjectName("primary")
        load_npz.clicked.connect(self._load_anchor_npz)
        offline.addRow("Transition NPZ", self._path_row(self.anchor_npz, False))
        index_row = QHBoxLayout(); index_row.addWidget(self.anchor_index); index_row.addWidget(load_npz)
        offline.addRow("帧索引(≥H-1,悬停看说明)", index_row)
        off.addLayout(offline)
        # 打磨:anchor 规则提示 —— 数据从哪来、格式是什么、帧怎么选
        self.anchor_help = QLabel(
            "Anchor = 模型规划的起点(当前软臂形状 + 最近 H 步动作)。\n"
            "数据来自 transition NPZ(positions(T,3,N) + actions(T,D),动作已归一 [0,1])。\n"
            "示例数据已内置在 data/npz/(15 节点,8172 帧);其它序列按 GUI_GUIDE 拷入。")
        self.anchor_help.setWordWrap(True)
        self.anchor_help.setStyleSheet("color:#486581;font-size:11px;")
        off.addWidget(self.anchor_help)
        ll.addWidget(gb_off)

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
        ll.addWidget(gb_tgt)

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
        # 功能①:零历史起步 —— 勾选后不强制 warmup,用全 0 历史锚定(⚠️ 训练分布外,首窗口可能不准)
        zero_row = QHBoxLayout()
        self.zero_history_cb = QCheckBox("零历史起步(免 warmup,首窗口 OOD)")
        self.zero_history_cb.setChecked(True)
        self.zero_history_cb.setToolTip(
            "勾选:用全 0 动作历史直接锚定,不需先 Warmup。\n"
            "⚠️ 模型训练从没见过零填充窗口,首窗口预测可能明显不准;\n"
            "运行几步后自动用本次真实动作累积历史,误差会收敛。")
        self.zero_history_cb.toggled.connect(self._on_zero_history_toggled)
        self.zero_hist_hint = QLabel("已启用:零历史起步(OOD)")
        self.zero_hist_hint.setStyleSheet("color:#F6AD55;font-size:11px;")
        zero_row.addWidget(self.zero_history_cb); zero_row.addStretch()
        live.addLayout(zero_row)
        live.addWidget(self.zero_hist_hint)
        tool_row = QHBoxLayout()
        self.tool_select_btn = QPushButton("select"); self.tool_select_btn.setCheckable(True)
        self.tool_select_btn.clicked.connect(lambda: self._set_tool("select"))
        self.tool_target_btn = QPushButton("点加目标"); self.tool_target_btn.setCheckable(True)
        self.tool_target_btn.clicked.connect(lambda: self._set_tool("add_target"))
        self.tool_skeleton_btn = QPushButton("点出目标骨架"); self.tool_skeleton_btn.setCheckable(True)
        self.tool_skeleton_btn.setToolTip(
            "点出目标骨架:依次点击 N 个点连成期望软臂形状,双击完成。\n"
            "规划让机器人拟合这个目标骨架(全身目标)。")
        self.tool_skeleton_btn.clicked.connect(lambda: self._set_tool("add_target_skeleton"))
        self.tool_obstacle_btn = QPushButton("点加障碍"); self.tool_obstacle_btn.setCheckable(True)
        self.tool_obstacle_btn.clicked.connect(lambda: self._set_tool("add_obstacle"))
        tool_row.addWidget(QLabel("工具:")); tool_row.addWidget(self.tool_select_btn)
        tool_row.addWidget(self.tool_target_btn); tool_row.addWidget(self.tool_skeleton_btn)
        tool_row.addWidget(self.tool_obstacle_btn)
        tool_row.addStretch()
        live.addLayout(tool_row)
        ll.addWidget(gb_live)
        ll.addStretch(1)

        # ---- 右栏:场景编辑(相机视图 + 场景列表) ----
        gb_scene = QGroupBox("场景编辑")
        sc = QVBoxLayout(gb_scene); sc.setContentsMargins(12, 14, 12, 12)
        self.camera_view = CameraViewWidget()
        self.scene_editor = SceneEditorPanel()
        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self.camera_view); right_split.addWidget(self.scene_editor)
        right_split.setSizes([560, 220])
        sc.addWidget(right_split, 1)
        self.anchor_status = QLabel("未锚定")
        sc.addWidget(self.anchor_status)
        # 功能②:骨架叠加说明 —— 青线+圆点是实时骨架(与训练同源骨架化方法)
        self.skeleton_hint = QLabel("青线+圆点 = 实时骨架(15 节点,与训练同源);加载模型 + 开相机后自动显示")
        self.skeleton_hint.setWordWrap(True)
        self.skeleton_hint.setStyleSheet("color:#486581;font-size:11px;")
        sc.addWidget(self.skeleton_hint)
        # 打磨补:B 修复 —— _scene_changed 引用的 scene_summary 从未被创建(悬空引用 bug)
        self.scene_summary = QPlainTextEdit()
        self.scene_summary.setReadOnly(True)
        self.scene_summary.setMaximumHeight(88)
        self.scene_summary.setPlaceholderText("场景摘要:anchor / scene / primitives / digest")
        sc.addWidget(self.scene_summary)

        outer.addWidget(left); outer.addWidget(gb_scene)
        outer.setSizes([360, 720])
        outer.setStretchFactor(0, 0); outer.setStretchFactor(1, 1)
        root.addWidget(outer, 1)

        # 绑定(保持原样)
        self.camera_view.target_picked.connect(self._add_primitive)
        self.camera_view.obstacle_picked.connect(self._add_primitive)
        self.camera_view.target_skeleton_picked.connect(self._add_primitive)  # 功能③
        self.scene_editor.scene_edited.connect(self._apply_scene_edit)
        self._camera_thread = None
        self._latest_frame = None
        self._action_history = []  # warmup 填充(H×action_dim 模型单位)
        self._history_buffer = None  # 功能①:执行累积的实际动作历史(滚动重锚定用)
        return page

    def _set_tool(self, tool: str) -> None:
        self.camera_view.set_tool(tool)
        for btn, name in ((self.tool_select_btn, "select"),
                          (self.tool_target_btn, "add_target"),
                          (self.tool_skeleton_btn, "add_target_skeleton"),
                          (self.tool_obstacle_btn, "add_obstacle")):
            btn.setChecked(name == tool)

    def _add_primitive(self, primitive) -> None:
        if not self.session:
            return
        self.session.set_scene(self.session.scene.with_primitive(primitive))
        self.camera_view.set_scene(self.session.scene)
        self.scene_editor.set_scene(self.session.scene)
        self._refresh()

    def _apply_scene_edit(self, scene) -> None:
        if not self.session:
            return
        self.session.set_scene(scene)
        self.camera_view.set_scene(scene)
        self._refresh()

    def _start_camera(self) -> None:
        import numpy as np
        self._camera_thread = _CameraThread(self)
        self._camera_thread.frame_ready.connect(self._on_camera_frame)
        self._camera_thread.start()
        self.camera_btn.setText("Camera 运行中")
        self.camera_anchor_btn.setEnabled(True)
        self.warmup_btn.setEnabled(True)
        # 主显示区 + Observe 页 camera_view 都要帧(若存在)
        self.main_display.set_frame(self._latest_frame if self._latest_frame is not None
                                    else np.zeros((240, 320, 3)))

    def _on_camera_frame(self, bgr) -> None:
        self._latest_frame = bgr
        self.main_display.set_frame(bgr)                       # 主显示区
        if hasattr(self, "camera_view") and self.camera_view is not None:
            self.camera_view.set_frame(bgr)                    # Observe 锚定视图
        if self.runtime is not None:
            from ..perception.segmentation import segment_white_on_blue
            from ..perception.skeleton import extract_skeleton_2d
            # Mock 场景:背景 = 帧自身灰度近似(真机用 manifest.segment_params + 无臂静态背景)
            mask = segment_white_on_blue(bgr, self._gray(bgr))
            skeleton, _ = extract_skeleton_2d(mask, self.runtime.descriptor.n_nodes,
                                              tip_fix=True, return_info=True)
            self.main_display.set_skeleton(skeleton)           # 主显示骨架层
            if hasattr(self, "camera_view") and self.camera_view is not None:
                self.camera_view.set_skeleton(skeleton)        # Observe 也显示

    def _gray(self, bgr):
        import numpy as np
        return np.mean(np.asarray(bgr, dtype=np.float64), axis=2).astype(np.uint8)

    def _warmup(self) -> None:
        if not self.runtime or self.runtime.descriptor.action_scale_kpa is None:
            self._error("warmup 需要已加载带 manifest 的模型")
            return
        from ..runtime.warmup import warmup_actions
        descriptor = self.runtime.descriptor
        seq = warmup_actions(descriptor.action_dim, descriptor.history_steps, kind="ramp")
        self._action_history = [tuple(float(v) for v in row) for row in seq]
        # 简化:用 mock 传输"下发"填历史(真机用 QtValveTransport)
        self.warmup_btn.setText(f"Warmup 完成:{len(seq)} 步")
        self.camera_anchor_btn.setEnabled(True)
        self._log(f"warmup: {len(seq)} 步动作历史已就绪(模型单位,可锚定)")

    def _on_zero_history_toggled(self, checked: bool) -> None:
        self.zero_hist_hint.setText(
            "已启用:零历史起步(OOD,首窗口可能不准)" if checked
            else "已禁用:锚定需先 Warmup 填真实历史")
        self.zero_hist_hint.setStyleSheet(
            "color:#F6AD55;font-size:11px;" if checked else "color:#486581;font-size:11px;")

    def _camera_anchor(self) -> None:
        import numpy as np   # 冒烟分支 np.asarray 用(本模块 numpy 均为方法局部 import)
        if self._latest_frame is None or not self.runtime:
            self._error("先 Start Camera")
            return
        if not self._action_history and not self.zero_history_cb.isChecked():
            self._error("无动作历史:勾选『零历史起步』可免 warmup 直接锚定,或先点 Warmup")
            return
        from ..runtime.anchors import anchor_from_camera_frame
        descriptor = self.runtime.descriptor
        bg = self._gray(self._latest_frame)   # mock 场景:背景即自身灰度近似
        manifest = self.runtime.manifest
        area_median = manifest.mask_area_median_px if manifest else None
        if area_median is None:
            area_median = float(np.asarray(self._latest_frame).sum())  # 冒烟
        anchor, quality, skeleton = anchor_from_camera_frame(
            self._latest_frame, background_gray=bg,
            segment_params=(manifest.segment_params if manifest else {}),
            n_nodes=descriptor.n_nodes, model=self.runtime.model,
            action_history=self._action_history, area_median_px=float(area_median),
            frame_ref="camera_live#mock",
            zero_pad_history=self.zero_history_cb.isChecked())
        if anchor is None:
            self._error(f"帧质量 reject:{quality.reasons};请重试或调场景")
            return
        self.session.set_anchor(anchor)
        # 打磨③:页间引导 —— 锚定成功提示下一步
        zero_note = " · 零历史起步(OOD)" if self.zero_history_cb.isChecked() else ""
        self.anchor_status.setText(
            f"已锚定 {anchor.anchor_id[:8]} verdict={quality.verdict}{zero_note} → 可前往 3 Plan 规划")
        self.camera_view.set_anchor(skeleton)
        self._refresh()

    def _plan_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)

        # 卡1:规划参数
        gb_param = QGroupBox("规划参数")
        p = QVBoxLayout(gb_param); p.setContentsMargins(12, 14, 12, 12)
        form = QFormLayout(); form.setVerticalSpacing(5)
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

    def _results_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)
        gb = QGroupBox("结果与指标")
        g = QVBoxLayout(gb); g.setContentsMargins(12, 14, 12, 12)
        self.results = QPlainTextEdit(); self.results.setReadOnly(True)
        self.results.setPlaceholderText("执行记录保存在 run/execution.csv；自动指标将在后续接入。")
        g.addWidget(self.results)
        root.addWidget(gb)
        return page

    def _path_row(self, edit: QLineEdit, directory: bool) -> QWidget:
        holder = QWidget(); row = QHBoxLayout(holder); row.setContentsMargins(0, 0, 0, 0)
        button = QPushButton("…")
        def browse() -> None:
            if directory:
                path = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text())
            if path:
                edit.setText(path)
        button.clicked.connect(browse); row.addWidget(edit, 1); row.addWidget(button)
        return holder

    def _new_session(self) -> None:
        try:
            self.session = ExperimentSession.create(self.run_root.text().strip())
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

    def _apply_safety(self) -> None:
        if not self.session:
            self._error("请先 New Experiment")
            return
        columns = list(zip(*[[cell.value() for cell in row]
                             for row in self._safety_cells]))
        try:
            safety = SafetyPolicy(
                pressure_min6=tuple(columns[0]), pressure_max6=tuple(columns[1]),
                rise_rate6=tuple(columns[2]), fall_rate6=tuple(columns[3]),
                initial_action6=tuple(columns[4]))
            self.session.set_safety(safety)
            atomic_write_json(self.session.run_dir / "safety.json", safety.to_dict())
            self._log("安全配置已应用；旧计划已失效")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _connect_valve(self) -> None:
        """连接真机阀:controller 在 GUI 线程构造(QObject 须与 transport 同线程),
        串口 open 放后台线程。连接成功 → 执行走真阀,失败回退 Mock。"""
        g1, g2 = self.hw_g1.text().strip(), self.hw_g2.text().strip()
        if not g1 and not g2:
            self._error("请至少填一组串口(COM)。真机需要:组1 COMx,组2 COMy。")
            return
        if self._valve_connect_thread and self._valve_connect_thread.isRunning():
            return
        try:
            from ..hardware.valve import create_valve_controller
            controller = create_valve_controller(g1, g2, baudrate=self.hw_baud.value())
        except Exception as error:
            self._error(f"构造阀控制器失败: {error}")
            return
        self.hw_connect_btn.setEnabled(False)
        self.hw_status.setText("正在连接阀(后台线程)……")
        self.hw_status.setStyleSheet("color:#F6AD55;font-size:11px;")
        self._valve_connect_thread = _ValveConnectThread(controller, groups=(1, 2))
        self._valve_connect_thread.connected.connect(self._valve_connected)
        self._valve_connect_thread.failed.connect(self._valve_connect_failed)
        self._valve_connect_thread.start()

    def _valve_connected(self, controller, summary: str) -> None:
        self.valve_controller = controller
        self.hw_status.setText(f"已连接真机阀: {summary}。执行将走真阀(非 Mock)。")
        self.hw_status.setStyleSheet("color:#38A169;font-size:11px;")
        self.hw_connect_btn.setEnabled(False)
        self.hw_disconnect_btn.setEnabled(True)
        self._log(f"真机阀已连接: {summary}")
        self._refresh()

    def _valve_connect_failed(self, message: str) -> None:
        self.valve_controller = None
        self.hw_status.setText(f"阀连接失败: {message}。执行继续用 Mock。")
        self.hw_status.setStyleSheet("color:#EF4E4E;font-size:11px;")
        self.hw_connect_btn.setEnabled(True)
        self.hw_disconnect_btn.setEnabled(False)
        self._log(f"ERROR: 阀连接失败 {message}")
        self._refresh()

    def _disconnect_valve(self) -> None:
        if self.valve_controller is not None:
            try:
                self.valve_controller.disconnect_group(1)
                self.valve_controller.disconnect_group(2)
            except Exception as error:
                self._log(f"WARN: 断开阀异常 {error}")
        self.valve_controller = None
        self.hw_status.setText("已断开,执行回退 Mock。")
        self.hw_status.setStyleSheet("color:#486581;font-size:11px;")
        self.hw_connect_btn.setEnabled(True)
        self.hw_disconnect_btn.setEnabled(False)
        self._refresh()

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
        # 若不刷 camera_view/scene_editor,右边可视化和原语列表不会更新(工具点加才更新)。
        self.camera_view.set_scene(self.session.scene)
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
            if self.session and self.session.plan and self.session.plan.predicted_states_path:
                p = Path(self.session.plan.predicted_states_path)
                if not p.is_absolute():
                    p = self.session.run_dir / p
                if p.is_file():
                    import numpy as np
                    with np.load(p) as data:
                        key = "states_model" if "states_model" in data else "states_normalized"
                        self.main_display.set_predicted_states(np.asarray(data[key]))
        else:
            self.plan_summary.setPlainText("Preflight: BLOCKED\n" + "\n".join(
                f"[{item.code}] {item.message}" for item in result.issues))
            self.plan_preview.clear_plan()
        self._refresh()

    def _arm(self) -> None:
        try:
            if not self.session:
                raise RuntimeError("没有 session")
            self.session.arm(); self._log("计划已由操作员 Arm")
            self._refresh()
        except Exception as error:
            self._error(str(error))

    def _make_transport(self):
        """执行 transport 工厂:真机阀已连接 → QtValveTransport(线程安全桥接);否则 Mock。

        QtValveTransport 须与 controller 同线程创建(GUI 线程);执行器在 worker 线程
        调它的 send,内部经 QueuedConnection 转发到 controller 的 Qt 线程。
        """
        if self.valve_controller is not None:
            from ..execution.hardware_session import QtValveTransport
            return QtValveTransport(self.valve_controller)
        from ..execution.executor import MockCommandTransport
        return MockCommandTransport()

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
        self.executor = PlanExecutor(
            self._make_transport(), self.session.safety,
            history_buffer=getattr(self, "_history_buffer", None))
        self.session.transition(SessionState.EXECUTING, "mock execution")
        self._execution_thread = _ExecutionThread(
            self.executor, self.session.plan, self.session.run_dir / "execution.csv")
        self._execution_thread.event.connect(
            lambda name, payload: self._log(f"{name}: {payload}"))
        self._execution_thread.finished_ok.connect(self._execution_done)
        self._execution_thread.failed.connect(self._execution_failed)
        self._execution_thread.start(); self._refresh()

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
        if self.session and self.session.state in {SessionState.EXECUTING, SessionState.ABORTING}:
            target = SessionState.ZEROED if self.session.state == SessionState.ABORTING else SessionState.ERROR
            self.session.transition(target, error)
        self._error(error); self._refresh()

    def _pause(self) -> None:
        if self.executor and self.session and self.session.state == SessionState.EXECUTING:
            self.executor.pause(); self.session.transition(SessionState.PAUSED, "operator pause")
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
            receipt = self._make_transport().zero(self.session.safety.ack_timeout_s)
            self.session.transition(SessionState.ZEROED, receipt.status)
        elif self.executor:
            self.session.transition(SessionState.ABORTING, "operator abort")
            self.executor.abort(); self._refresh()
        self._refresh()

    def _zero(self) -> None:
        if not self.session:
            return
        try:
            transport = self.executor.transport if self.executor else self._make_transport()
            receipt = transport.zero(self.session.safety.ack_timeout_s)
            if receipt.status != "ack":
                raise RuntimeError(receipt.status)
            if self.session.state in {SessionState.COMPLETED, SessionState.ERROR,
                                      SessionState.ABORTING, SessionState.PAUSED}:
                self.session.transition(SessionState.ZEROED, "operator zero")
            self._log("六通道已归零（Mock）"); self._refresh()
        except Exception as error:
            self._error(str(error))

    def _load_hardware_config(self) -> None:
        """启动时加载硬件连接配置(config/hardware.json,gitignore 不入库)。"""
        path = APP_DIR / "config" / "hardware.json"
        if not path.is_file():
            return
        try:
            value = read_json(path)
            self.hw_g1.setText(str(value.get("group1", "")))
            self.hw_g2.setText(str(value.get("group2", "")))
            self.hw_baud.setValue(int(value.get("baudrate", 9600)))
        except Exception as error:
            self._log(f"WARN: 加载硬件配置失败 {error}")

    def _save_hardware_config(self) -> None:
        path = APP_DIR / "config" / "hardware.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {
                "group1": self.hw_g1.text().strip(),
                "group2": self.hw_g2.text().strip(),
                "baudrate": self.hw_baud.value(),
            })
        except Exception as error:
            self._log(f"WARN: 保存硬件配置失败 {error}")

    def _refresh(self) -> None:
        state = self.session.state.value if self.session else "no_session"
        run = self.session.run_dir.name if self.session else "-"
        hardware = "REAL VALVE" if self.valve_controller is not None else "MOCK"
        self.state_label.setText(f"Run: {run}    State: {state}    Hardware: {hardware}")
        color = STATE_BADGE_COLORS.get(state, STATE_BADGE_COLORS["no_session"])
        self.state_label.setStyleSheet(
            f"background:{CARD};border:2px solid {color};border-radius:12px;"
            f"padding:4px 12px;color:{color};font-weight:bold;")
        self.arm_button.setEnabled(bool(self.session and not self.session.replay_only and
                                        self.session.state == SessionState.READY))
        self.execute_button.setEnabled(bool(self.session and self.session.state == SessionState.ARMED))
        self.pause_button.setEnabled(bool(self.session and self.session.state == SessionState.EXECUTING))
        self.resume_button.setEnabled(bool(self.session and self.session.state == SessionState.PAUSED))
        # B8:执行中锁页 1/2/3(否则执行中改 scene 会清空 experiment.json 的 plan,
        # 执行记录与实际下发计划脱钩 = 溯源腐败)
        executing = bool(self.session and self.session.state in {
            SessionState.EXECUTING, SessionState.PAUSED, SessionState.ARMED})
        self.tabs.setTabEnabled(0, not executing)
        self.tabs.setTabEnabled(1, not executing)
        self.tabs.setTabEnabled(2, not executing)

    def _log(self, message: str) -> None:
        self.execution_log.appendPlainText(message)

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
            self._execution_thread.wait(3000)
        if self._planning_thread and self._planning_thread.isRunning():
            self._planning_thread.cancel(); self._planning_thread.wait(3000)
        if self.runtime:
            self.runtime.clear()
        self._save_hardware_config()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    window = ValidationWindow(); window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
