"""实机验证工作台 GUI 第一版。

运行：``python -m real_validation.main_validation``。
当前完成离线/Mock 会话、模型元数据加载、scene/anchor/plan 导入、preflight 与
Mock ACK 执行；真硬件连接与交互式 scene view 按 TODO 后续阶段接入。
"""

from __future__ import annotations

import sys
import threading
import traceback
from pathlib import Path

if __package__ in (None, ""):  # 支持复制目录后直接 ``python main_validation.py``
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "real_validation"

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .executor import MockCommandTransport, PlanExecutor
from .io import atomic_write_json, read_json
from .model_runtime import ModelRuntime
from .models import ActionPlan, Anchor, SafetyPolicy, Scene, ScenePrimitive
from .openloop_planner import OpenLoopShootingPlanner, ShootingConfig
from .offline_anchor import anchor_from_npz
from .plan_io import write_actions6_csv
from .session import ExperimentSession, SessionState
from .widgets import CameraViewWidget, PlanPreviewWidget, SceneEditorPanel

APP_DIR = Path(__file__).resolve().parent


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
        from .model_runtime import ModelLoadError
        try:
            runtime = ModelRuntime(self.checkpoint, self.data_dir or None, self.device,
                                   k_safe=self.k_safe)
            self.loaded.emit(runtime)
        except (ModelLoadError, FileNotFoundError, ValueError) as error:
            self.failed.emit(str(error))               # 可操作提示,不弹 traceback
        except Exception:
            self.failed.emit(traceback.format_exc())   # 真 bug 才给 traceback


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
        self.resize(1280, 820)
        self.session: ExperimentSession | None = None
        self.runtime: ModelRuntime | None = None
        self.executor: PlanExecutor | None = None
        self._model_thread: _ModelLoadThread | None = None
        self._planning_thread: _PlanningThread | None = None
        self._execution_thread: _ExecutionThread | None = None
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        safety_bar = QHBoxLayout()
        self.state_label = QLabel("No session")
        self.state_label.setStyleSheet("font-weight:600;padding:6px")
        self.zero_button = QPushButton("归零 / Zero")
        self.zero_button.clicked.connect(self._zero)
        self.abort_button = QPushButton("中止 / Abort")
        self.abort_button.setStyleSheet("background:#C53030;color:white;font-weight:600")
        self.abort_button.clicked.connect(self._abort)
        safety_bar.addWidget(self.state_label, 1)
        safety_bar.addWidget(self.zero_button)
        safety_bar.addWidget(self.abort_button)
        layout.addLayout(safety_bar)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._setup_page(), "1 Setup")
        self.tabs.addTab(self._observe_page(), "2 Observe & Scene")
        self.tabs.addTab(self._plan_page(), "3 Plan")
        self.tabs.addTab(self._execute_page(), "4 Execute")
        self.tabs.addTab(self._results_page(), "5 Results")
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

    def _setup_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page); form = QFormLayout()
        self.run_root = QLineEdit(str(APP_DIR / "runs"))
        self.checkpoint = QLineEdit(str(APP_DIR / "checkpoints" / "current" / "best_model.pt"))
        self.data_dir = QLineEdit(str(APP_DIR / "data"))
        self.k_safe = QSpinBox(); self.k_safe.setRange(0, 100000)
        self.k_safe.setSpecialValueText("未认证")
        self.device = QComboBox(); self.device.addItems(["cpu", "cuda"])
        form.addRow("Run 根目录", self._path_row(self.run_root, True))
        form.addRow("Checkpoint", self._path_row(self.checkpoint, False))
        form.addRow("训练数据目录", self._path_row(self.data_dir, True))
        form.addRow("K_safe", self.k_safe)
        form.addRow("模型设备", self.device)
        root.addLayout(form)
        buttons = QHBoxLayout()
        create = QPushButton("New Experiment"); create.clicked.connect(self._new_session)
        replay = QPushButton("Open Run (Replay)"); replay.clicked.connect(self._open_replay)
        load = QPushButton("Load Model"); load.clicked.connect(self._load_model)
        buttons.addWidget(create); buttons.addWidget(replay); buttons.addWidget(load); buttons.addStretch()
        root.addLayout(buttons)
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
        root.addWidget(QLabel("六通道安全配置（kPa / kPa·s⁻¹）"))
        root.addWidget(self.safety_table)
        apply_safety = QPushButton("应用安全配置并使旧计划失效")
        apply_safety.clicked.connect(self._apply_safety)
        root.addWidget(apply_safety)
        self.model_summary = QPlainTextEdit(); self.model_summary.setReadOnly(True)
        root.addWidget(self.model_summary, 1)
        return page

    def _observe_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page); buttons = QHBoxLayout()
        anchor = QPushButton("加载 anchor.json"); anchor.clicked.connect(self._load_anchor)
        scene = QPushButton("加载 scene.json"); scene.clicked.connect(self._load_scene)
        buttons.addWidget(anchor); buttons.addWidget(scene); buttons.addStretch()
        root.addLayout(buttons)
        offline = QFormLayout()
        self.anchor_npz = QLineEdit()
        self.anchor_index = QSpinBox(); self.anchor_index.setRange(0, 100000000)
        load_npz = QPushButton("从 NPZ 建立 Anchor"); load_npz.clicked.connect(self._load_anchor_npz)
        offline.addRow("Transition NPZ", self._path_row(self.anchor_npz, False))
        index_row = QHBoxLayout(); index_row.addWidget(self.anchor_index); index_row.addWidget(load_npz)
        offline.addRow("帧索引（必须已有完整 H）", index_row)
        root.addLayout(offline)
        target_form = QFormLayout()
        self.target_x = QDoubleSpinBox(); self.target_x.setRange(-100000, 100000)
        self.target_y = QDoubleSpinBox(); self.target_y.setRange(-100000, 100000)
        self.target_radius = QDoubleSpinBox(); self.target_radius.setRange(0, 100000)
        self.obstacle_x = QDoubleSpinBox(); self.obstacle_x.setRange(-100000, 100000)
        self.obstacle_y = QDoubleSpinBox(); self.obstacle_y.setRange(-100000, 100000)
        self.obstacle_radius = QDoubleSpinBox(); self.obstacle_radius.setRange(0.01, 100000)
        target_row = QHBoxLayout(); target_row.addWidget(self.target_x); target_row.addWidget(self.target_y)
        target_row.addWidget(self.target_radius)
        target_button = QPushButton("设置末端目标"); target_button.clicked.connect(self._set_target)
        target_row.addWidget(target_button); target_form.addRow("目标 x / y / 半径 (model)", target_row)
        obstacle_row = QHBoxLayout(); obstacle_row.addWidget(self.obstacle_x); obstacle_row.addWidget(self.obstacle_y)
        obstacle_row.addWidget(self.obstacle_radius)
        obstacle_button = QPushButton("添加圆障碍"); obstacle_button.clicked.connect(self._add_obstacle)
        obstacle_row.addWidget(obstacle_button); target_form.addRow("障碍 x / y / 半径 (model)", obstacle_row)
        root.addLayout(target_form)
        # ---- P3:实时相机视图 + 场景编辑器 + 锚定/warmup ----
        live_buttons = QHBoxLayout()
        self.camera_btn = QPushButton("Start Camera (Mock)"); self.camera_btn.clicked.connect(self._start_camera)
        self.camera_anchor_btn = QPushButton("从相机取流锚定"); self.camera_anchor_btn.clicked.connect(self._camera_anchor)
        self.camera_anchor_btn.setEnabled(False)
        self.warmup_btn = QPushButton("Warmup(填动作历史)"); self.warmup_btn.clicked.connect(self._warmup)
        self.warmup_btn.setEnabled(False)
        live_buttons.addWidget(self.camera_btn); live_buttons.addWidget(self.camera_anchor_btn)
        live_buttons.addWidget(self.warmup_btn); live_buttons.addStretch()
        root.addLayout(live_buttons)

        tool_row = QHBoxLayout()
        self.tool_select_btn = QPushButton("select"); self.tool_select_btn.clicked.connect(lambda: self._set_tool("select"))
        self.tool_target_btn = QPushButton("点加目标"); self.tool_target_btn.clicked.connect(lambda: self._set_tool("add_target"))
        self.tool_obstacle_btn = QPushButton("点加障碍"); self.tool_obstacle_btn.clicked.connect(lambda: self._set_tool("add_obstacle"))
        tool_row.addWidget(QLabel("工具:")); tool_row.addWidget(self.tool_select_btn)
        tool_row.addWidget(self.tool_target_btn); tool_row.addWidget(self.tool_obstacle_btn); tool_row.addStretch()
        root.addLayout(tool_row)

        self.camera_view = CameraViewWidget()
        self.scene_editor = SceneEditorPanel()
        split = QSplitter(); split.addWidget(self.camera_view); split.addWidget(self.scene_editor)
        split.setSizes([520, 260])
        root.addWidget(split, 1)
        self.anchor_status = QLabel("未锚定")
        root.addWidget(self.anchor_status)

        # 绑定:camera_view 点选 → 加原语;scene_editor 编辑 → session.set_scene
        self.camera_view.target_picked.connect(self._add_primitive)
        self.camera_view.obstacle_picked.connect(self._add_primitive)
        self.scene_editor.scene_edited.connect(self._apply_scene_edit)

        self._camera_thread = None
        self._latest_frame = None
        self._action_history = []          # warmup 填充(H×action_dim 模型单位)
        return page

    def _set_tool(self, tool: str) -> None:
        self.camera_view.set_tool(tool)
        for btn, name in ((self.tool_select_btn, "select"), (self.tool_target_btn, "add_target"),
                          (self.tool_obstacle_btn, "add_obstacle")):
            btn.setStyleSheet("font-weight:bold" if name == tool else "")

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
        self._camera_thread = _CameraThread(self)
        self._camera_thread.frame_ready.connect(self._on_camera_frame)
        self._camera_thread.start()
        self.camera_btn.setText("Camera 运行中")
        self.camera_anchor_btn.setEnabled(True)
        self.warmup_btn.setEnabled(True)

    def _on_camera_frame(self, bgr) -> None:
        self._latest_frame = bgr
        self.camera_view.set_frame(bgr)
        if self.runtime is not None:
            from .perception.segmentation import segment_white_on_blue
            from .perception.skeleton import extract_skeleton_2d
            # Mock 场景:背景 = 帧自身灰度近似(真机用 manifest.segment_params + 无臂静态背景)
            mask = segment_white_on_blue(bgr, self._gray(bgr))
            skeleton, _ = extract_skeleton_2d(mask, self.runtime.descriptor.n_nodes,
                                              tip_fix=True, return_info=True)
            self.camera_view.set_skeleton(skeleton)

    def _gray(self, bgr):
        import numpy as np
        return np.mean(np.asarray(bgr, dtype=np.float64), axis=2).astype(np.uint8)

    def _warmup(self) -> None:
        if not self.runtime or self.runtime.descriptor.action_scale_kpa is None:
            self._error("warmup 需要已加载带 manifest 的模型")
            return
        from .warmup import warmup_actions
        descriptor = self.runtime.descriptor
        seq = warmup_actions(descriptor.action_dim, descriptor.history_steps, kind="ramp")
        self._action_history = [tuple(float(v) for v in row) for row in seq]
        # 简化:用 mock 传输"下发"填历史(真机用 QtValveTransport)
        self.warmup_btn.setText(f"Warmup 完成:{len(seq)} 步")
        self.camera_anchor_btn.setEnabled(True)
        self._log(f"warmup: {len(seq)} 步动作历史已就绪(模型单位,可锚定)")

    def _camera_anchor(self) -> None:
        if self._latest_frame is None or not self.runtime:
            self._error("先 Start Camera")
            return
        if not self._action_history:
            self._error("warmup 未完成 —— 需先填充动作历史")
            return
        from .live_anchor import anchor_from_camera_frame
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
            frame_ref="camera_live#mock")
        if anchor is None:
            self._error(f"帧质量 reject:{quality.reasons};请重试或调场景")
            return
        self.session.set_anchor(anchor)
        self.anchor_status.setText(f"已锚定 {anchor.anchor_id[:8]} verdict={quality.verdict}")
        self.camera_view.set_anchor(skeleton)
        self._refresh()

    def _plan_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)
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
        root.addLayout(form)
        buttons = QHBoxLayout()
        generate = QPushButton("运行 OpenLoop Planner"); generate.clicked.connect(self._start_planning)
        cancel = QPushButton("取消规划"); cancel.clicked.connect(self._cancel_planning)
        load = QPushButton("导入 plan.json"); load.clicked.connect(self._load_plan)
        check = QPushButton("运行 Preflight"); check.clicked.connect(self._run_preflight)
        buttons.addWidget(generate); buttons.addWidget(cancel); buttons.addWidget(load)
        buttons.addWidget(check); buttons.addStretch()
        root.addLayout(buttons)
        self.plan_summary = QPlainTextEdit(); self.plan_summary.setReadOnly(True)
        self.plan_summary.setPlaceholderText("异步 shooting planner 与交互式候选预览将在此页接入。")
        root.addWidget(self.plan_summary)
        self.plan_preview = PlanPreviewWidget(); root.addWidget(self.plan_preview, 1)
        return page

    def _execute_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page); buttons = QHBoxLayout()
        self.arm_button = QPushButton("Arm / Confirm"); self.arm_button.clicked.connect(self._arm)
        self.execute_button = QPushButton("Mock Execute"); self.execute_button.clicked.connect(self._execute)
        self.pause_button = QPushButton("Pause"); self.pause_button.clicked.connect(self._pause)
        self.resume_button = QPushButton("Resume"); self.resume_button.clicked.connect(self._resume)
        for button in (self.arm_button, self.execute_button, self.pause_button, self.resume_button):
            buttons.addWidget(button)
        buttons.addStretch(); root.addLayout(buttons)
        self.execution_log = QPlainTextEdit(); self.execution_log.setReadOnly(True)
        root.addWidget(self.execution_log, 1)
        return page

    def _results_page(self) -> QWidget:
        page = QWidget(); root = QVBoxLayout(page)
        self.results = QPlainTextEdit(); self.results.setReadOnly(True)
        self.results.setPlaceholderText("执行记录保存在 run/execution.csv；自动指标将在后续接入。")
        root.addWidget(self.results)
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
        if descriptor.k_safe_table_px:
            k = (descriptor.k_safe_table_px.get("10px")
                 or descriptor.k_safe_table_px.get("5px"))
            if k:
                self.k_safe.setValue(int(k))
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
        except Exception:
            self._error(traceback.format_exc())

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
            self.plan_summary.setPlainText("Preflight: PASS")
            if self.session and self.session.plan and self.session.plan.predicted_states_path:
                try:
                    self.plan_preview.set_plan(self.session.plan, self.session.scene,
                                               self.session.safety, self.session.run_dir)
                except Exception as error:
                    self.plan_summary.appendPlainText(f"\nPreview unavailable: {error}")
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
        """执行 transport 工厂:离线 Mock;接实机时改返回 QtValveTransport(需活 controller)。"""
        from .executor import MockCommandTransport
        return MockCommandTransport()

    def _execute(self) -> None:
        if not self.session or self.session.state != SessionState.ARMED or not self.session.plan:
            self._error("计划必须先通过 Preflight 并 Arm")
            return
        self.executor = PlanExecutor(self._make_transport(), self.session.safety)
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
        from .metrics import evaluate_command_safety
        actions6 = [tuple(r.applied6) for r in receipts]
        safety_metrics = evaluate_command_safety(
            actions6, self.session.plan.step_interval_s if self.session.plan else 0.1,
            self.session.safety)
        jitters = [getattr(r, "jitter_s", None) for r in receipts]
        jitters = [j for j in jitters if j is not None]
        jitter_summary = (f"jitter mean={sum(jitters) / len(jitters) * 1e3:.1f}ms "
                          f"max={max(jitters) * 1e3:.1f}ms" if jitters else "jitter 无记录")
        self.results.setPlainText(
            f"执行完成:{len(receipts)} 条命令\n"
            f"压力越界:{safety_metrics['pressure_violation_count']}  "
            f"速率越界:{safety_metrics['slew_violation_count']}\n"
            f"{jitter_summary}\n{self.session.run_dir / 'execution.csv'}")
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

    def _refresh(self) -> None:
        state = self.session.state.value if self.session else "no_session"
        run = self.session.run_dir.name if self.session else "-"
        self.state_label.setText(f"Run: {run}    State: {state}    Hardware: MOCK")
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
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = ValidationWindow(); window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
