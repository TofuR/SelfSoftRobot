"""Hereditary experiment page, sharing the workbench camera and valve bridge."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import threading
import time
import uuid
import json

import numpy as np
from PyQt5.QtCore import QThread,QTimer,pyqtSignal,Qt
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QPushButton,QLineEdit,QFileDialog,QComboBox,QSpinBox,QCheckBox,QPlainTextEdit,QSlider,QDoubleSpinBox,QGridLayout,QGroupBox,QDialog

from ..runtime.hereditary_deployment import load_bundle,HereditaryDeployment,transform,resample_curve
from ..execution.hereditary_executor import HereditaryExecutor
from ..runtime.shape_target import target_indices, target_distances
from ..execution.executor import CommandReceipt
from ..widgets.hereditary_canvas import HereditaryCanvas


class Job(QThread):
    succeeded=pyqtSignal(object)
    failed=pyqtSignal(str)
    progress=pyqtSignal(object)
    def __init__(self,fn,parent=None):super().__init__(parent);self.fn=fn
    def run(self):
        try:self.succeeded.emit(self.fn())
        except Exception as error:self.failed.emit(f'{type(error).__name__}: {error}')


class HereditaryPanel(QWidget):
    def __init__(self,host):
        super().__init__(host);self.host=host;self.runtime=None;self.job=None;self.executor=None
        self.plan=None;self.target=None;self.target_ids=None;self.auto_curve=None;self.target_matrix=None;self.target_samples=None;self.armed=False;self.busy=False;self.frame=None
        self.frame_lock=threading.Lock();self.transport=None;self.controller=None;self.commands={};self.camera_identity=None
        self.alignment_error=None;self.cancel_event=threading.Event();self.ack6=np.full(6,np.nan);self.zero_pending=False
        self.pages=[QWidget() for _ in range(4)]
        layouts=[QVBoxLayout(page) for page in self.pages]
        prepare,initial,goal,run=layouts
        self.state_label=QLabel('准备：加载模型并连接设备；尚未开始估计历史')
        self.state_label.setWordWrap(True)
        self.canvas=HereditaryCanvas()
        self.canvas.curve_finished.connect(self.curve)
        self.canvas.draft_changed.connect(self.draft_changed)
        self.canvas.roi_finished.connect(self.roi_selected)
        self.canvas.prompt_added.connect(self.add_sam_prompt)
        self.initial_roi=None;self.roi_camera_key=None;self.sam_points=[];self.sam_labels=[];self.draft_info={}
        from ..perception.sam_initial import SamInitialSegmenter,default_checkpoint
        self.sam_segmenter=SamInitialSegmenter()
        self.draft_stamp=None;self.draft_version=None;self.auto_prompt_pending=False;self.evidence_stamp=-float('inf')
        self.display_controls=QWidget();display=QHBoxLayout(self.display_controls)
        self.display_mode=QComboBox();self.display_mode.addItems(['整体形状 + 中心线','仅骨架中心线'])
        self.display_mode.currentIndexChanged.connect(self.update_display)
        self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(5,70);self.opacity.setValue(28)
        self.opacity.valueChanged.connect(self.update_display)
        display.addWidget(QLabel('叠图'));display.addWidget(self.display_mode)
        display.addWidget(QLabel('不透明度'));display.addWidget(self.opacity)
        legend=QLabel('青：模型估计  白点：可见边缘\n紫：预览  红：目标  绿：冻结分割');legend.setWordWrap(True)
        display.addWidget(legend);self.opacity.setMinimumWidth(80)
        row=QHBoxLayout();self.path=QLineEdit();self.path.setPlaceholderText('hereditary.npz（同名 JSON 放旁边）')
        browse=QPushButton('选择模型');browse.clicked.connect(self.browse)
        row.addWidget(self.path);row.addWidget(browse);prepare.addLayout(row)
        default_bundle=Path(__file__).resolve().parents[1]/'checkpoints'/'hereditary_current'/'hereditary.npz'
        if default_bundle.is_file():self.path.setText(str(default_bundle))
        self.load_button=QPushButton('加载模型');self.load_button.clicked.connect(self.load);prepare.addWidget(self.load_button)
        self.mapping=[];grid=QGridLayout()
        for i in range(6):
            combo=QComboBox()
            for j in range(4):combo.addItem(f'u{j}',j)
            combo.setCurrentIndex([0,1,1,2,2,3][i]);self.mapping.append(combo)
            grid.addWidget(QLabel(f'c{i} ←'),i//3,(i%3)*2);grid.addWidget(combo,i//3,(i%3)*2+1)
            combo.currentIndexChanged.connect(self.invalidate_plan)
        prepare.addLayout(grid)
        self.mapping_button=QPushButton('应用六腔映射');self.mapping_button.clicked.connect(self.apply_mapping);prepare.addWidget(self.mapping_button)
        self.probe_button=QPushButton('设备自检：新图像 / 阀组状态');self.probe_button.clicked.connect(self.probe);prepare.addWidget(self.probe_button)
        self.probe_result=QPlainTextEdit();self.probe_result.setReadOnly(True);self.probe_result.setMaximumHeight(120);prepare.addWidget(self.probe_result);self.probe_result.hide()
        note=QLabel('先用全局「六腔控制」调到初始压力并结束调压。提取完整形状、检查黄点，再确认部署；系统保持当前压力估计状态和预热。')
        note.setWordWrap(True);initial.addWidget(note)
        self.auto_button=QPushButton('自动提取当前完整形状');self.auto_button.clicked.connect(self.auto_shape);initial.addWidget(self.auto_button)
        self.initial_tools_toggle=QPushButton('修正形状 / 高级设置');self.initial_tools_toggle.setCheckable(True);initial.addWidget(self.initial_tools_toggle)
        self.initial_tools=QWidget();detail=QVBoxLayout(self.initial_tools);detail.setContentsMargins(0,0,0,0)
        self.initial_tools_toggle.toggled.connect(self.initial_tools.setVisible)
        self.polarity=QComboBox();self.polarity.addItem('亮色臂身 / 较暗背景','bright');self.polarity.addItem('黑色剪影 / 明亮背景','dark');detail.addWidget(self.polarity)
        self.polarity.currentIndexChanged.connect(self.invalidate_plan)
        row=QHBoxLayout();self.initial_method=QComboBox();self.initial_method.addItem('SAM2 初始化（推荐，可较慢）','sam2');self.initial_method.addItem('传统分割（免 SAM 依赖）','classical');row.addWidget(self.initial_method)
        self.sam_settings=QDialog(self);self.sam_settings.setWindowTitle('SAM2 初始化设置');sf=QFormLayout(self.sam_settings)
        self.sam_path=QLineEdit(str(default_checkpoint()));sf.addRow('SAM2.1 Tiny 权重',self.sam_path)
        browse_sam=QPushButton('选择分割权重');browse_sam.clicked.connect(self.browse_sam);sf.addRow(browse_sam)
        self.sam_device=QComboBox();self.sam_device.addItems(['auto','cpu','cuda']);sf.addRow('推理设备',self.sam_device)
        sf.addRow(QLabel('仅在初始化使用。CPU 可运行，首次加载较慢；不会改变气压。'))
        settings_button=QPushButton('SAM 设置');settings_button.clicked.connect(self.sam_settings.show);row.addWidget(settings_button);detail.addLayout(row)
        row=QHBoxLayout();self.roi_button=QPushButton('框选完整臂身（排除支架）');self.roi_button.clicked.connect(lambda:self.begin_initial('roi'));row.addWidget(self.roi_button)
        self.clear_roi_button=QPushButton('清除框选');self.clear_roi_button.clicked.connect(self.clear_initial_roi);row.addWidget(self.clear_roi_button);detail.addLayout(row)
        self.align_button=QPushButton('手动重画当前中心线（base → tip）');self.align_button.clicked.connect(lambda:self.set_mode('align'));detail.addWidget(self.align_button)
        row=QHBoxLayout()
        self.refine_button=QPushButton('结合图像微调草稿');self.refine_button.clicked.connect(self.refine_shape);row.addWidget(self.refine_button)
        self.refine_radius=QSpinBox();self.refine_radius.setRange(0,240);self.refine_radius.setValue(0);self.refine_radius.setSpecialValueText('自动（随尺度）');self.refine_radius.setSuffix(' px');row.addWidget(QLabel('搜索半径'));row.addWidget(self.refine_radius);detail.addLayout(row)
        self.keep_endpoints=QCheckBox('微调时保留我标定的 BASE / TIP（无需算法识别短边）');self.keep_endpoints.setChecked(True);detail.addWidget(self.keep_endpoints)
        row=QHBoxLayout();self.positive_button=QPushButton('点选臂身 +');self.positive_button.clicked.connect(lambda:self.prompt_mode(True));row.addWidget(self.positive_button)
        self.negative_button=QPushButton('排除支架/背景 −');self.negative_button.clicked.connect(lambda:self.prompt_mode(False));row.addWidget(self.negative_button)
        self.rerun_sam_button=QPushButton('按提示重新分割');self.rerun_sam_button.clicked.connect(self.resegment_initial);row.addWidget(self.rerun_sam_button);detail.addLayout(row)
        self.mask_check=QCheckBox('叠加分割掩膜（绿色）');self.mask_check.setChecked(True);self.mask_check.toggled.connect(self.show_initial_mask);detail.addWidget(self.mask_check)
        row=QHBoxLayout()
        self.flip_button=QPushButton('交换 base / tip');self.flip_button.clicked.connect(self.flip_draft);row.addWidget(self.flip_button)
        self.cancel_button=QPushButton('取消草稿 / 恢复实时图像');self.cancel_button.clicked.connect(self.cancel_draft);row.addWidget(self.cancel_button);detail.addLayout(row)
        self.warmup_options=QGroupBox('高级：预热质量阈值');self.warmup_options.setCheckable(True);self.warmup_options.setChecked(False)
        warm_form=QFormLayout(self.warmup_options);self.warmup_fields=[]
        for label,default,lo,hi in [('最短保持 s',2.,0,60),('超时 s',20.,1,120),('连续新帧',8,2,100),('边缘覆盖',.75,.1,1),('边缘残差 px',3.,.1,20),('帧间变化 px',1.,.1,10)]:
            box=QDoubleSpinBox();box.setRange(lo,hi);box.setValue(default);self.warmup_fields.append(box);warm_form.addRow(label,box);box.setVisible(False)
        self.warmup_options.toggled.connect(lambda visible:[warm_form.itemAt(i).widget().setVisible(visible) for i in range(warm_form.count())])
        for i in range(warm_form.count()):warm_form.itemAt(i).widget().hide()
        detail.addWidget(self.warmup_options)
        initial.addWidget(self.initial_tools);self.initial_tools.hide()
        self.confirm_button=QPushButton('确认形状并部署预热');self.confirm_button.clicked.connect(self.confirm_alignment);initial.addWidget(self.confirm_button)
        self.alignment_label=QLabel('自动提取后检查 BASE/TIP；有歧义时在臂身中间点一下。修正工具按需展开。');self.alignment_label.setWordWrap(True);initial.addWidget(self.alignment_label)
        self.goal_mode=QComboBox();self.goal_mode.addItems(['完整形状','局部目标（末端 / 一段）']);goal.addWidget(self.goal_mode)
        self.local_options=QWidget();local=QHBoxLayout(self.local_options);local.setContentsMargins(0,0,0,0)
        self.local_kind=QComboBox();self.local_kind.addItems(['末端点','指定节点的一段','任意臂段自动匹配']);local.addWidget(self.local_kind)
        self.node_start=QComboBox();self.node_end=QComboBox()
        for box in (self.node_start,self.node_end):
            for i in range(1,15):box.addItem(f'节点 {i}'+(' (TIP)' if i==14 else ''),i)
        self.node_start.setCurrentIndex(10);self.node_end.setCurrentIndex(13)
        self.node_arrow=QLabel('→');local.addWidget(self.node_start);local.addWidget(self.node_arrow);local.addWidget(self.node_end);goal.addWidget(self.local_options)
        self.goal_button=QPushButton('连续画目标中心线（base → tip）');self.goal_button.clicked.connect(lambda:self.set_mode('goal'));goal.addWidget(self.goal_button)
        self.goal_note=QLabel();self.goal_note.setWordWrap(True);goal.addWidget(self.goal_note)
        for box in (self.goal_mode,self.local_kind,self.node_start,self.node_end):box.currentIndexChanged.connect(self.change_goal_mode)
        self.change_goal_mode()
        self.horizon=QSpinBox();self.horizon.setRange(2,80);self.horizon.setValue(80)
        self.tolerance=QDoubleSpinBox();self.tolerance.setRange(.05,20);self.tolerance.setValue(2.);self.tolerance.setSuffix(' mm')
        self.max_node=QDoubleSpinBox();self.max_node.setRange(.1,50);self.max_node.setValue(4.);self.max_node.setSuffix(' mm')
        self.planning_dialog=QDialog(self);self.planning_dialog.setWindowTitle('初始规划参数');self.planning_dialog.setModal(False)
        pf=QVBoxLayout(self.planning_dialog);form=QFormLayout();pf.addLayout(form)
        self.reserve_steps=QSpinBox();self.reserve_steps.setRange(0,100);self.reserve_steps.setValue(10)
        form.addRow('末端调整余量（步，0 为关闭）',self.reserve_steps)
        self.planning_rate=QDoubleSpinBox();self.planning_rate.setRange(10,100);self.planning_rate.setValue(80);self.planning_rate.setSuffix(' %')
        self.planning_rate.setToolTip('初始规划只用当前有效升/降速率的一部分；在线矫正仍可用完整上限。不是降低压力上限。')
        form.addRow('初始规划使用的速度上限比例',self.planning_rate)
        form.addRow('受约束节点平均容限',self.tolerance);form.addRow('最大目标点偏差',self.max_node);form.addRow('搜索上限步数（自动选择长度）',self.horizon)
        self.planning_budget=QDoubleSpinBox();self.planning_budget.setRange(.1,120);self.planning_budget.setValue(15);self.planning_budget.setSuffix(' s')
        self.planning_iterations=QSpinBox();self.planning_iterations.setRange(1,100);self.planning_iterations.setValue(24)
        self.planning_shooting=QSpinBox();self.planning_shooting.setRange(1,200);self.planning_shooting.setValue(30)
        self.planning_stride=QSpinBox();self.planning_stride.setRange(1,80);self.planning_stride.setValue(20)
        for label,box in [('总计算预算（迭代间检查）',self.planning_budget),('每个长度 B 迭代上限',self.planning_iterations),('终态优化函数评估上限',self.planning_shooting),('长度搜索步长',self.planning_stride)]:form.addRow(label,box)
        explanation=QLabel('参数修改立即生效，已有计划失效。自动尝试较短长度，达标即停；步数上限不是固定执行长度。总预算在优化迭代间检查，单次迭代可能略超预算。余量已计入预览和执行总时长，先保持末条压力供 B 调整；用尽后保持，不无限续行。发令周期由模型 dt 决定，矫正间隔在第4页设置。')
        explanation.setWordWrap(True);pf.addWidget(explanation)
        close=QPushButton('完成');close.clicked.connect(self.planning_dialog.hide);pf.addWidget(close)
        self.planning_settings=QPushButton('初始规划参数…');self.planning_settings.clicked.connect(self.show_planning_settings);goal.addWidget(self.planning_settings)
        self.planning_fields=[self.planning_rate,self.reserve_steps,self.horizon,self.tolerance,self.max_node,self.planning_budget,self.planning_iterations,self.planning_shooting,self.planning_stride]
        for box in self.planning_fields:box.valueChanged.connect(self.invalidate_plan)
        self.plan_button=QPushButton('规划当前目标 / 预览');self.plan_button.clicked.connect(self.make_plan);goal.addWidget(self.plan_button)
        self.preview=QPlainTextEdit();self.preview.setReadOnly(True);self.preview.setMaximumHeight(180);goal.addWidget(self.preview)
        row=QHBoxLayout();self.play_button=QPushButton('播放模型预测');self.play_button.clicked.connect(self.play_preview);row.addWidget(self.play_button)
        self.scrubber=QSlider(Qt.Horizontal);self.scrubber.setRange(0,0);self.scrubber.valueChanged.connect(self.show_preview);row.addWidget(self.scrubber);goal.addLayout(row)
        self.pressure_preview=QLabel('拖动时间轴查看预测形状与六腔压力');self.pressure_preview.setWordWrap(True);goal.addWidget(self.pressure_preview)
        import pyqtgraph as pg
        self.preview_plot=pg.PlotWidget();self.preview_plot.setMaximumHeight(150);self.preview_plot.setLabel('left','预测压力',units='kPa');self.preview_plot.setLabel('bottom','时间',units='s');goal.addWidget(self.preview_plot)
        self.play_timer=QTimer(self);self.play_timer.timeout.connect(self.next_preview)
        from .occlusion_controls import OcclusionControls
        self.occlusion_controls=OcclusionControls(self);self.occlusion=self.occlusion_controls.enabled
        run.addWidget(self.occlusion_controls)
        self.max_missing=QSpinBox();self.max_missing.setRange(1,10);self.max_missing.setValue(3)
        self.max_skipped=QSpinBox();self.max_skipped.setRange(1,100);self.max_skipped.setValue(10)
        self.feedback_interval=QSpinBox();self.feedback_interval.setRange(1,10);self.feedback_interval.setValue(2);self.feedback_interval.setSuffix(' 步')
        self.feedback_interval.setToolTip('1：原逐步同步反馈；2：第1步计算、第2步继续旧计划、第3步前采用。中间步不会计作超时；仍逐步记录压力、图像和 NDI。')
        self.feedback_frequency=QLabel();self.feedback_interval.valueChanged.connect(self.update_feedback_frequency)
        self.update_feedback_frequency()
        self.feedback_settle=QDoubleSpinBox();self.feedback_settle.setRange(0,90);self.feedback_settle.setSuffix(' ms');self.feedback_settle.setToolTip('始终要求 ACK 后的新图像；只有需要额外等待时才增大，会减少本周期反馈预算')
        missing_form=QFormLayout();missing_form.addRow('连续无有效反馈几次后停止归零',self.max_missing);missing_form.addRow('连续跳过反馈几次后停止归零',self.max_skipped);missing_form.addRow('发令后额外等图',self.feedback_settle);run.addLayout(missing_form)
        missing_form.insertRow(0,'矫正间隔',self.feedback_interval);missing_form.insertRow(1,'发令 / 矫正频率',self.feedback_frequency)
        self.open_loop_check=QCheckBox('不使用矫正，仅执行规划（对照实验）');self.open_loop_check.setToolTip('执行时不校正图像状态、不优化剩余动作；仍记录图像/压力/NDI，保留压力接续与停止控制');run.insertWidget(0,self.open_loop_check)
        self.open_loop_check.toggled.connect(self.execution_mode_changed)
        self.trial_check=QCheckBox('允许试运行未达标计划（已检查预览，仅探索）');self.trial_check.setVisible(False);run.addWidget(self.trial_check)
        self.trial_note=QLabel();self.trial_note.setWordWrap(True);run.addWidget(self.trial_note)
        self.arm_check=QCheckBox('已检查目标、映射、对齐和压力范围，允许执行');self.arm_check.toggled.connect(self.arm);run.addWidget(self.arm_check)
        self.execute_button=QPushButton('开始实验运动：执行 Analytic B 图像反馈');self.execute_button.clicked.connect(self.execute);run.addWidget(self.execute_button)
        self.hold_stop_button=QPushButton('停止运动（保持当前压力）');self.hold_stop_button.clicked.connect(self.request_hold);run.addWidget(self.hold_stop_button)
        self.stop_button=QPushButton('中止并全部归零');self.stop_button.clicked.connect(self.stop);run.addWidget(self.stop_button)
        self.feedback_label=QLabel('反馈自动筛选可信边缘，不需要手画遮挡区。边缘缺失可能来自遮挡、光照或对齐误差。');self.feedback_label.setWordWrap(True);run.addWidget(self.feedback_label)
        self.results=QPlainTextEdit();self.results.setReadOnly(True);run.addWidget(self.results)
        note=QLabel('完成后保持最终压力。此模型仅支持正压；不支持实际负压。虚拟相机与 ACK 仅用于软件流程验证。')
        note.setWordWrap(True);run.addWidget(note)
        for layout in layouts:layout.addStretch()
        self.controls=[self.open_loop_check,self.trial_check,self.refine_button,self.refine_radius,self.goal_mode,self.local_options,self.path,browse,self.load_button,self.polarity,*self.mapping,self.mapping_button,self.auto_button,self.align_button,self.flip_button,self.cancel_button,self.confirm_button,self.goal_button,self.horizon,self.tolerance,self.max_node,self.plan_button,self.planning_settings,*self.planning_fields,self.arm_check,self.execute_button,self.probe_button,self.occlusion_controls,self.max_missing,self.max_skipped,self.feedback_settle,*self.warmup_fields]
        self.controls.extend([self.initial_tools_toggle,self.initial_method,self.roi_button,self.clear_roi_button,self.positive_button,self.negative_button,self.rerun_sam_button,self.keep_endpoints,self.sam_path,self.sam_device,browse_sam])
        self.controls.append(self.feedback_interval)
        from .chamber_control import ChamberControl
        self.chambers=ChamberControl(self)
        self.chamber_button=QPushButton('六腔控制');self.chamber_button.clicked.connect(self.chambers.show)
        display.insertWidget(0,self.chamber_button)
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)

    @property
    def active(self):return self.busy or self.armed

    def update_feedback_frequency(self):
        if self.runtime is None:
            self.feedback_frequency.setText('加载模型后根据 dt 显示');return
        dt=self.runtime.dt;n=self.feedback_interval.value()
        self.feedback_frequency.setText(f'{1/dt:.1f} Hz / {1/(dt*n):.1f} Hz'+('（逐步反馈）' if n==1 else f'（跨 {n} 步异步计算）'))

    def message(self,text):self.state_label.setText(str(text));self.host._log('Hereditary: '+str(text))
    def invalidate_plan(self,*_):
        self.plan=None;self.armed=False
        if hasattr(self,'trial_check'):
            self.trial_check.setChecked(False);self.trial_check.hide();self.trial_note.clear()
        if hasattr(self,'play_timer'):self.play_timer.stop();self.canvas.preview=None;self.canvas.update()
        if hasattr(self,'arm_check'):self.arm_check.blockSignals(True);self.arm_check.setChecked(False);self.arm_check.blockSignals(False)
    def fail(self,error):
        self.invalidate_plan();self.message(error)
    def browse(self):
        path,_=QFileDialog.getOpenFileName(self,'选择冻结 hereditary 部署包',self.path.text(),'Hereditary (*.npz)')
        if path:self.path.setText(path)
    def _job(self,fn,done):
        if self.busy:raise ValueError('已有操作正在进行')
        if self.host.session and self.host.session.state.value in ('armed','executing','paused'):
            raise ValueError('请先停止原 OpenLoop 执行')
        self.cancel_event.clear();self.busy=True;self.canvas.locked=True
        for c in self.controls:c.setEnabled(False)
        self.job=Job(fn,self);self.job.succeeded.connect(done);self.job.failed.connect(self.fail)
        self.job.finished.connect(self._finished);self.host._refresh();self.job.start()
    def _finished(self):
        self.busy=False;self.canvas.locked=False
        for c in self.controls:c.setEnabled(True)
        self.host._refresh()
        if self.zero_pending:self.zero_pending=False;QTimer.singleShot(0,self.stop)
    def load(self):
        try:
            if any(hasattr(camera,'engine') for camera in self.host.hardware.cameras):raise ValueError('请先断开模型驱动 Mock 相机，再更换模型')
            path=self.path.text().strip()
            self._job(lambda:load_bundle(path),self.loaded)
        except Exception as e:self.fail(e)
    def loaded(self,value):
        if self.runtime:self.runtime.close()
        self.canvas.draft=None;self.canvas.frozen=False;self.canvas.mode='view';self.alignment_error=None
        engine,meta=value
        self.clear_initial_roi();self.canvas.mask=None;self.canvas.prompts=[];self.draft_info={}
        root=Path(self.host.run_root.text())/'hereditary'
        folder=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.runtime=HereditaryDeployment(engine,meta,folder)
        self.host.model_badge.setText('Hereditary · 4 输入 / Analytic B')
        for c,v in zip(self.mapping,meta['expansion6']):c.setCurrentIndex(v)
        if hasattr(self,'saved_chambers'):
            self.runtime.set_mapping(self.saved_chambers['mapping'])
            for c,v in zip(self.mapping,self.saved_chambers['mapping']):c.setCurrentIndex(v)
        self.feedback_settle.setMaximum(max(0,meta['dt']*1000-5));self.horizon.setMaximum(meta['max_horizon']);self.invalidate_plan();self.target=None
        self.update_feedback_frequency()
        self.canvas.target=None;self.canvas.prediction=None
        for box in (self.node_start,self.node_end):
            box.blockSignals(True);box.clear()
            for i in range(1,engine.n_nodes):box.addItem(f'节点 {i}'+(' (TIP)' if i==engine.n_nodes-1 else ''),i)
            box.blockSignals(False)
        self.node_start.setCurrentIndex(max(0,engine.n_nodes-5));self.node_end.setCurrentIndex(engine.n_nodes-2);self.change_goal_mode()
        self.chambers.apply();self._attach_controller();self.host._update_main_info()
        self.message(f'已加载 {engine.channels} 输入 / {engine.n_nodes} 节点，dt={meta["dt"]} s\n压力上限 {meta["upper_kpa"]} kPa；历史尚未初始化\n日志 {folder}')
    def _attach_controller(self):
        controller=self.host.hardware.valve_controller
        if controller is None or controller is self.controller:return
        if self.controller is not None:
            self.controller.command_issued.disconnect(self.issued);self.controller.communication_result.disconnect(self.ack)
        self.controller=controller;controller.command_issued.connect(self.issued);controller.communication_result.connect(self.ack)
    def start_mock_camera(self):
        from ..hardware.profile import DeviceState
        from ..hardware.hereditary_mock import HereditaryMockCamera
        h=self.host.hardware
        if self.runtime is None:raise ValueError('动作驱动 Mock 相机需要先加载模型')
        h.prepare_valves();self._attach_controller()
        meta=dict(self.runtime.meta,expansion6=list(self.runtime.mapping.expansion))
        camera=HereditaryMockCamera(self.runtime.engine,meta,h.valve_controller,h)
        h.bind_frame_buffer(camera,0)
        camera.frame_ready.connect(lambda frame,stamp:h._on_camera_frame(0,frame,stamp))
        camera.error.connect(lambda message:h._on_camera_error(0,message))
        h.cameras=[camera];camera.start();h._set_state('camera',DeviceState.READY,'MODEL MOCK · 动作驱动图像')
        self.set_occlusion(self.occlusion.isChecked());self.invalidate_plan()

    def set_occlusion(self,checked):
        # All cameras use the same image-only perturbation layer. The virtual
        # plant itself continues producing unmodified raw images.
        for camera in self.host.hardware.cameras:
            if hasattr(camera,'occluded'):camera.occluded=False
        self.occlusion_controls.apply()

    def issued(self,ident,requested,applied,stamp):
        if self.runtime is None:return
        row=self.commands.setdefault(str(ident),{'acks':{}})
        row.update(requested=tuple(requested),applied=tuple(applied),stamp=stamp)
        if self.runtime.initialized:
            with self.runtime.lock:
                self.runtime.pending.add(str(ident));self.runtime.record('command_issued',command_id=str(ident),requested6=requested,applied6=applied,t_command=stamp)
        self._settle(str(ident))
    def ack(self,ident,group,ok,stamp,status):
        if self.runtime is None:return
        row=self.commands.setdefault(str(ident),{'acks':{}});row['acks'][group]=(ok,stamp,status);self._settle(str(ident))
    def _settle(self,ident):
        row=self.commands[ident]
        if 'stamp' not in row or not all(g in row['acks'] for g in (1,2)):return
        for group,(ok,stamp,status) in row['acks'].items():
            if ok and status!='inactive':self.ack6[(group-1)*3:group*3]=row['applied'][(group-1)*3:group*3]
        status=next((x[2] for x in row['acks'].values() if not x[0] or x[2]=='inactive'),'ack')
        receipt=CommandReceipt(ident,row['requested'],row['applied'],row['stamp'],max(x[1] for x in row['acks'].values()),status)
        try:
            if self.runtime.initialized:self.runtime.acknowledge(receipt)
        except Exception as error:self.runtime.fault=str(error);self.fail(error)
        self.commands.pop(ident,None)
    def _transport(self,groups=(1,2)):
        self.host._require_ui_profile_applied();self._attach_controller()
        hardware=self.host.hardware;hardware.require_valves_ready(groups)
        if self.transport is not None:self.transport.close()
        self.transport=hardware.create_transport(groups)
        def guard(action6):
            r=self.runtime
            with r.lock:
                previous=r.mapping.reduce(hardware.valve_controller.last_command)
                action=r.mapping.reduce(action6)
                return r.mapping.expand(r.bounds.project(action[None],previous)[0]).tolist()
        self.transport.command_filter=guard
        rise=self.runtime.bounds.rise/self.runtime.dt*self.runtime.mapping.scale
        fall=self.runtime.bounds.fall/self.runtime.dt*self.runtime.mapping.scale
        hardware.valve_controller.configure_safety(rise[list(self.runtime.mapping.expansion)].tolist(),fall[list(self.runtime.mapping.expansion)].tolist())
        if self.runtime.initialized:self.runtime.record('hardware',profile=hardware.profile.to_dict())
        return self.transport
    def apply_mapping(self):
        try:
            if self.runtime is None:raise ValueError('请先加载模型')
            if self.host.hardware.cameras:raise ValueError('请先断开相机再更改映射')
            self.runtime.set_mapping([c.currentData() for c in self.mapping]);self.chambers.apply();self.invalidate_plan();self.target=None
            self.canvas.target=None;self.canvas.prediction=None;self.cancel_draft();self.alignment_error=None;self.message('映射已保存，连接设备后到第 2 页提取当前形状并部署')
        except Exception as e:self.fail(e)
    def update_display(self,*_):
        self.canvas.show_tube=self.display_mode.currentIndex()==0
        self.canvas.opacity=self.opacity.value()/100
        self.canvas.update()

    def leave_drawing(self):
        if self.canvas.mode in ('goal','goal_point'):
            self.canvas.mode='view';self.canvas.stroke=[];self.canvas.update()

    def cancel_draft(self):
        self.canvas.frozen=False;self.canvas.draft=None;self.canvas.stroke=[]
        self.canvas.mode='view';self.canvas.drag_node=None;self.canvas.drawing=False
        self.draft_stamp=None;self.draft_version=None;self.canvas.update()
        self.canvas.mask=None;self.canvas.prompts=[];self.canvas.roi=None;self.auto_prompt_pending=False
        self.sam_points=[];self.sam_labels=[]

    def _freeze(self,frame=None):
        r=self.runtime
        if r is None:raise ValueError('请先加载模型')
        if r.pending or r.fault:raise ValueError(r.fault or '等待指令 ACK')
        if frame is None:
            frame=self.frame_provider()
            if frame is None or time.monotonic()-frame[1]>.3:raise ValueError('需要新相机图像')
        self.cancel_draft();self.invalidate_plan()
        r.alignment_confirmed=False;r.ready=False;self.alignment_error=None;self.target=None;self.canvas.target=None
        self.canvas.set_frame(frame[0].copy());self.canvas.frozen=True
        self.canvas.roi=self.initial_roi
        self.draft_stamp=frame[1];self.draft_version=r.version;self.camera_identity=self._camera_key()
        return frame

    def browse_sam(self):
        path,_=QFileDialog.getOpenFileName(self,'选择 SAM2.1 Tiny 权重',self.sam_path.text(),'SAM2 (*.pt)')
        if path:self.sam_path.setText(path)

    def show_initial_mask(self,visible):
        self.canvas.show_mask=visible;self.canvas.update()

    def clear_initial_roi(self):
        self.initial_roi=None;self.roi_camera_key=None;self.canvas.roi=None;self.canvas.update()

    def roi_selected(self,region):
        region=np.rint(region).astype(int)
        if np.any(region[2:]-region[:2]<8):self.message('框选太小，请包含完整臂身');return
        self.initial_roi=region.tolist();self.roi_camera_key=self._camera_key()
        self.canvas.roi=region;self.canvas.mode='view';self.canvas.update()
        self.message('框选已保存。点击“自动提取当前完整形状”；框内尽量排除支架，两端留少量背景。')

    def prompt_mode(self,positive):
        if not self.canvas.frozen:self.message('请先框选或分割，冻结当前图像');return
        self.canvas.mode='sam_positive' if positive else 'sam_negative'
        self.message('在臂身内部点选正提示' if positive else '在误选的支架或背景上点负提示；随后按提示重新分割')

    def add_sam_prompt(self,point,label):
        self.sam_points.append(np.asarray(point).tolist());self.sam_labels.append(int(label))
        self.canvas.prompts=list(zip(self.sam_points,self.sam_labels));self.canvas.update()
        if self.auto_prompt_pending and label:
            self.auto_prompt_pending=False;self.resegment_initial()

    def initial_config(self):
        if self.initial_roi is not None and self.roi_camera_key!=self._camera_key():
            raise ValueError('框选后相机配置已改变，请清除框选或重新框选')
        return dict(method=self.initial_method.currentData(),polarity=self.polarity.currentData(),
                    roi=self.initial_roi,checkpoint=self.sam_path.text(),device=self.sam_device.currentText(),
                    points=list(self.sam_points),labels=list(self.sam_labels))

    def segment_initial(self,image,config,guide=None):
        if config['method']!='sam2':return None,{}
        mask,info=self.sam_segmenter.segment(image,config['checkpoint'],config['device'],
                                              config['roi'],guide,config['points'],config['labels'],polarity=config['polarity'])
        return mask,info

    def initial_result(self,image,config,guide=None,preserve=True):
        from ..perception.initial_shape import extract_initial_shape,refine_initial_shape
        mask,sam_info=self.segment_initial(image,config,guide)
        try:
            if guide is None:
                curve,mask,info=extract_initial_shape(image,self.runtime.engine.n_nodes,config['polarity'],config['roi'],mask)
            else:
                curve,mask,info=refine_initial_shape(image,guide,config['polarity'],config.get('search_px',0),preserve,mask)
        except ValueError as error:
            if mask is None:raise
            return None,mask,dict(sam=sam_info,extraction_error=str(error))
        info['sam']=sam_info
        return curve,mask,info

    def publish_initial(self,image,value,source):
        curve,mask,info=value
        import cv2
        name=source+'_'+uuid.uuid4().hex[:8]
        cv2.imwrite(str(self.runtime.run_dir/(name+'.png')),image)
        cv2.imwrite(str(self.runtime.run_dir/(name+'_mask.png')),mask)
        self.runtime.record('initial_shape_draft',source=source,frame=name+'.png',timestamp=self.draft_stamp,curve=curve,diagnostics=info)
        if curve is None:
            self.canvas.mask=mask;self.canvas.update()
            self.message('SAM 掩膜已显示，但中心线仍需修正：'+info['extraction_error']+'。可补正/负提示重分割或手绘；已有草稿保留。')
            return
        self.draft_info=info;self.canvas.draft=curve.astype(float);self.canvas.mask=mask
        self.canvas.radius=info['radius_px'];self.canvas.mode='edit';self.canvas.update()
        self.draft_changed(curve,operator=False)
        warning='；'.join(info.get('warnings',[]))
        if info.get('endpoints_preserved'):warning+=' 两端采用人工位置，未由算法认证短边。'
        self.alignment_label.setText('绿色是分割，黄线是待确认中心线。可拖动黄点；SAM 误选可加正/负提示重分割。'+warning)
        timing=info.get('sam',{});elapsed=timing.get('elapsed_ms')
        duration='' if elapsed is None else f' 分割 {elapsed/1000:.2f}s。'
        self.message('草稿已生成，请检查完整臂身与 BASE/TIP，再确认部署。'+duration+warning)

    def auto_shape(self):
        self.begin_initial('auto')

    def begin_initial(self,mode):
        try:
            if self.busy:raise ValueError('请等待当前操作结束或先停止')
            r=self.runtime
            if r is None:raise ValueError('请先加载模型')
            if self.chambers.driving:raise ValueError('请先结束手动调压并保持')
            if self.occlusion_controls.config.enabled:raise ValueError('初始化前请关闭软件遮挡测试')
            if tuple(c.currentData() for c in self.mapping)!=r.mapping.expansion:raise ValueError('请先应用映射')
            config=self.initial_config();config['points']=[];config['labels']=[]
            transport=self._transport()
            def work():
                receipt=transport.send(self.host.hardware.valve_controller.last_command,(1,2),.5)
                if receipt.status!='ack':raise ValueError('当前保持压力未确认：'+receipt.status)
                if r.initialized and not r.fault:r.acknowledge(receipt)
                else:r.initialize(receipt.applied6,receipt.t_command)
                until=time.monotonic()+.8
                while time.monotonic()<until:
                    if self.cancel_event.wait(.01):raise ValueError('提取已取消')
                    frame=self.frame_provider()
                    if frame is not None and frame[1]>max(receipt.t_ack,receipt.t_command+.05):break
                else:raise ValueError('确认保持后没有新图像')
                if mode!='auto':return frame,None,None
                try:
                    from threadpoolctl import threadpool_limits
                    with threadpool_limits(4):value=self.initial_result(frame[0],config)
                    if self.cancel_event.is_set():raise ValueError('提取已取消')
                    return frame,value,None
                except Exception as error:return frame,None,str(error)
            def done(value):
                frame,result,error=value
                if self.cancel_event.is_set():self.message('已取消提取');return
                self._freeze(frame);self.sam_points=[];self.sam_labels=[];self.canvas.prompts=[]
                if error:
                    self.auto_prompt_pending=config['method']=='sam2'
                    if self.auto_prompt_pending:self.canvas.mode='sam_positive'
                    self.message(error+'；图像已冻结。可在臂身中间点一下重试，或展开修正工具。');return
                if mode=='auto':self.publish_initial(frame[0],result,'initial')
                else:
                    self.canvas.mode=mode
                    self.message('拖动框选完整臂身，尽量排除支架' if mode=='roi' else '从实际 BASE 短边中心连续画到 TIP 短边中心；松开后可拖动黄点')
            self.message('确认当前保持压力并获取图像；SAM 首次加载可能较慢，后台处理不改变目标压力。')
            self._job(work,done)
        except Exception as e:self.fail(e)

    def resegment_initial(self):
        try:
            if not self.canvas.frozen:raise ValueError('请先冻结当前图像')
            if self.camera_identity!=self._camera_key() or self.draft_version!=self.runtime.version:raise ValueError('相机或动作历史变化，请重新提取')
            config=self.initial_config();config['method']='sam2';image=self.canvas.frame.copy()
            def done(value):
                if not self.cancel_event.is_set():self.publish_initial(image,value,'sam_prompt')
            self._job(lambda:self.initial_result(image,config),done)
        except Exception as error:self.fail(error)

    def refine_shape(self):
        try:
            if self.runtime is None or self.canvas.draft is None or not self.canvas.frozen:
                raise ValueError('请先自动提取或手动绘制当前完整形状草稿')
            if self.camera_identity!=self._camera_key() or self.draft_version!=self.runtime.version:
                raise ValueError('相机或动作历史变化，请重新提取')
            image=self.canvas.frame.copy();draft=self.canvas.draft.copy()
            config=self.initial_config();config['search_px']=self.refine_radius.value();preserve=self.keep_endpoints.isChecked()
            def done(value):
                if not self.cancel_event.is_set():self.publish_initial(image,value,'refined')
            self._job(lambda:self.initial_result(image,config,draft,preserve),done)
        except Exception as e:self.fail(e)

    def change_goal_mode(self,*_):
        self.invalidate_plan();self.target=None;self.target_ids=None;self.auto_curve=None;self.target_matrix=None;self.target_samples=None
        self.canvas.target=None;self.leave_drawing()
        if self.runtime:
            self.runtime.target_shape=None;self.runtime.target_node_indices=None
        partial=self.goal_mode.currentIndex()==1;segment=self.local_kind.currentIndex()==1
        self.canvas.highlight_indices=(np.arange(self.node_start.currentData(),self.node_end.currentData()+1) if partial and segment and self.node_start.currentData() is not None and self.node_end.currentData() is not None else None)
        self.local_options.setVisible(partial)
        self.node_start.setVisible(segment);self.node_end.setVisible(segment);self.node_arrow.setVisible(segment)
        self.goal_button.setText('画一段目标中心线（近 base 端 → 近 tip 端）' if partial and segment else
                                 ('单击设置末端目标' if partial else '连续画目标中心线（base → tip）'))
        self.goal_note.setText('局部目标只拟合所选节点，其余臂身由模型决定。节点从 base=0 到 TIP 递增；一段曲线按所选节点等弧长对应。切换模式/区段清除旧目标。当前不含避障。' if partial else
                              '从已对齐 base 连续画到 tip；全部活动节点参与拟合。切换到局部目标会清除完整目标。当前不含避障。')
        if partial and self.local_kind.currentIndex()==2:
            self.goal_button.setText('画一段目标（自动选择匹配臂段）')
            self.goal_note.setText('只画需要到达的一段，不选择节点。规划在预算内搜索不同臂段和两个绘制方向，按整段曲线采样误差选取；预览标出匹配区段，执行时固定对应关系。当前不含避障。')
        self.canvas.update()

    def draft_changed(self,points,operator=True):
        if operator:self.draft_info=dict(self.draft_info,operator_edited=True)
        self.invalidate_plan()
        self.runtime.alignment_confirmed=False
        self.alignment_error=None
        self.alignment_label.setText('草稿已修改，尚未计算对齐；确认后计算尺度、旋转和 base。')

    def flip_draft(self):
        if self.canvas.draft is not None:
            self.canvas.draft=self.canvas.draft[::-1].copy()
            self.draft_changed(self.canvas.draft);self.canvas.update()

    def set_mode(self,mode):
        try:
            if mode=='align':
                if (self.canvas.frozen and self.runtime is not None and self.draft_version==self.runtime.version
                        and self.camera_identity==self._camera_key()):
                    self.canvas.mode='align';self.canvas.stroke=[]
                    self.message('在冻结图像上从 BASE 画到 TIP；手绘可直接确认，图像微调可选')
                else:self.begin_initial('align')
                return
            if self.runtime is None or not self.runtime.initialized:raise ValueError('请先提取当前形状，建立保持压力历史')
            if mode=='align':self._freeze()
            elif not self.runtime.ready:raise ValueError('请先完成部署预热')
            else:self.cancel_draft()
            if mode=='goal' and self.goal_mode.currentIndex()==1 and self.local_kind.currentIndex()==0:mode='goal_point'
            self.invalidate_plan();self.canvas.mode=mode;self.canvas.stroke=[]
            if mode in ('goal','goal_point'):
                self.message(self.goal_button.text()+'；新目标将替换旧目标');return
            self.message('按住左键，从 base 连续画到 tip；松开完成 '+('当前形状草稿' if mode=='align' else '目标全形状'))
        except Exception as e:self.fail(e)

    def curve(self,points):
        try:
            if self.canvas.mode=='align':
                self.canvas.draft=resample_curve(points,self.runtime.engine.n_nodes)
                self.canvas.mode='edit';self.draft_changed(self.canvas.draft)
                self.draft_info={'method':'operator_drawn','endpoints_preserved':True}
            else:
                self.auto_curve=None;self.target_matrix=None;self.target_samples=None
                if self.goal_mode.currentIndex()==0:
                    self.target=self.runtime.goal(points);self.target_ids=None
                    display=self.target
                elif self.local_kind.currentIndex()==2:
                    self.auto_curve=transform(resample_curve(points,32),np.linalg.inv(self.runtime.matrix))
                    self.target=self.auto_curve.copy();self.target_ids=None;display=self.target
                else:
                    last=self.runtime.engine.n_nodes-1
                    if self.local_kind.currentIndex()==0:ids=np.array([last])
                    else:
                        start,end=self.node_start.currentData(),self.node_end.currentData()
                        if start>=end:raise ValueError('一段中心线的起始节点必须小于结束节点')
                        ids=np.arange(start,end+1)
                    self.target,self.target_ids=self.runtime.partial_goal(points,ids)
                    display=self.target[self.target_ids]
                self.runtime.target_shape=self.target.copy() if self.auto_curve is None else None;self.runtime.target_node_indices=self.target_ids
                self.canvas.target=transform(display,self.runtime.matrix)
                self.runtime.record('target',camera_curve=points,model_goal=self.target,
                                    node_indices=target_indices(self.runtime.engine.n_nodes,self.target_ids) if self.auto_curve is None else None,automatic_curve=self.auto_curve,mode=self.goal_mode.currentText())
                self.message('目标已记录，可规划并预览；仅受约束节点参与目标误差')
                self.canvas.mode='view'
            self.invalidate_plan();self.canvas.stroke=[];self.canvas.update()
        except Exception as e:self.fail(e)

    def confirm_alignment(self):
        try:
            if self.canvas.draft is None:
                if self.camera_identity!=self._camera_key():raise ValueError('相机已改变，请重新提取并配准')
                if self.runtime is None:raise ValueError('请先加载模型')
                self.runtime.prepare_rewarmup();self.invalidate_plan();self.start_warmup();return
            if self.camera_identity!=self._camera_key() or self.draft_version!=self.runtime.version:
                raise ValueError('相机或动作历史在提取后变化，请重新提取')
            curve=self.canvas.draft.copy();stamp=self.draft_stamp;source=dict(self.draft_info);ready={'ok':False}
            def work():
                from threadpoolctl import threadpool_limits
                with threadpool_limits(1):return self.runtime.calibrate_full_shape(curve,stamp,self.cancel_event)
            def done(info):
                if self.cancel_event.is_set():return
                scale=info['scale_px_per_mm'];base=info['base_px'];self.alignment_error=info['rms_px']
                self.alignment_label.setText(f'尺度 {scale:.3f} px/mm · 旋转 {info["angle_deg"]:.1f}°\nbase ({base[0]:.1f}, {base[1]:.1f}) px · RMS {info["rms_px"]:.2f} px / 臂长 {info["rms_fraction"]:.2%}')
                self.runtime.confirm_alignment();self.runtime.record('initial_shape_confirmed',curve=curve,timestamp=stamp,source=source)
                self.canvas.radius=self.runtime.meta['radius_mm']*scale
                self.cancel_draft();ready['ok']=True
                self.message('尺度、坐标与有界状态拟合完成；保持当前压力，准备连续图像预热')
            self._job(work,done)
            def warmup_after_finished():
                if ready['ok'] and not self.cancel_event.is_set() and not self.zero_pending:self.start_warmup()
            self.job.finished.connect(warmup_after_finished)
        except Exception as e:self.fail(e)

    def start_warmup(self):
        from ..runtime.deployment_quality import WarmupCriteria,WarmupGate
        values=[box.value() for box in self.warmup_fields];values[2]=int(values[2])
        criteria=WarmupCriteria(*values);r=self.runtime;r.ready=False;r.edge_polarity=self.polarity.currentData()
        gate=WarmupGate(criteria,time.monotonic())
        def work():
            from threadpoolctl import threadpool_limits
            with threadpool_limits(1):
                while time.monotonic()-gate.started<=criteria.timeout_s:
                    if self.cancel_event.wait(r.dt):raise ValueError('部署预热已取消')
                    frame=self.frame_provider()
                    if frame is None:continue
                    _,info=r.feedback(frame[0],frame[1],np.empty((0,4)),np.empty((0,r.engine.n_nodes,2)))
                    ready=gate.update(info,frame[1],time.monotonic());r.record('deployment_warmup',**info)
                    self.job.progress.emit(info)
                    if ready:
                        r.ready=True;r.deployment_id=uuid.uuid4().hex
                        r.record('deployment_snapshot',deployment_id=r.deployment_id,checkpoint=r.meta['checkpoint_sha256'],mapping=r.mapping.expansion,bounds=vars(r.bounds),matrix=r.matrix,state=r.state,applied6=r.mapping.expand(r.action),timestamp=r.at,quality=info['warmup'])
                        return info
            raise ValueError('预热超时；保持压力、恢复无遮挡视野后可点击重新预热。仅配准错误或相机移动时重新提取。')
        self._job(work,lambda _:self.message('模型就绪：对齐与状态估计已完成，保持时继续观测。进入第 3 页画目标。'))
        self.job.progress.connect(self.progress)

    def progress(self,info):
        from .feedback_timing import computation_text,STATUS_TEXT,timing_line
        self.canvas.evidence=np.asarray(info.get('evidence_pixels_px',[]));self.evidence_stamp=info.get('source_frame_timestamp',info.get('frame_timestamp',time.monotonic()))
        if 'prediction_px' in info:
            self.canvas.prediction=np.asarray(info['prediction_px']);self.canvas.update()
        if info.get('observation_mode')=='manual_prediction':
            self.feedback_label.setText('手动调压：青色为 ACK 动作历史的模型预测；结束调压并保持后恢复视觉校正，实验前重新预热。');return
        if info.get('control_mode')=='open_loop':
            self.feedback_label.setText('对照：不使用矫正，按规划执行；图像仅记录。'+('本步未获得新图像' if info.get('revision_status')=='frame_missing' else ''))
            return
        visibility=info.get('visibility',{})
        text=(f'{computation_text(info)} · 可信边缘 {info.get("edges",0)} · '
              f'可见证据覆盖 {visibility.get("coverage",0):.0%} · {visibility.get("status","图像不可用")}')
        if info.get('visible_target_error_mm') is not None:text+=f' · 可见边缘目标差 {info["visible_target_error_mm"]:.2f} mm'
        if 'warmup' in info:text+=f' · 预热连续合格 {info["warmup"]["good"]}/{info["warmup"]["required"]}'
        if info.get('consecutive_missing'):text+=f' · 连续无反馈 {info["consecutive_missing"]} 次，仅模型预测'
        if 'revision_status' in info:
            status=STATUS_TEXT.get(info['revision_status'],info['revision_status'])
            text+=f' · {status}'
            self.results.appendPlainText(timing_line(info))
        self.feedback_label.setText(text)
        self.state_label.setText(text)

    def show_planning_settings(self):
        self.planning_dialog.show();self.planning_dialog.raise_()

    def make_plan(self):
        try:
            if self.target is None:raise ValueError('请先绘制当前模式的目标')
            self.safety_ui=np.asarray([[c.value() for c in row] for row in self.host._safety_cells])[:,:4]
            self.runtime.configure_limits(*self.safety_ui.T)
            self.runtime.edge_polarity=self.polarity.currentData()
            self.runtime.record('image_frontend',polarity=self.runtime.edge_polarity)
            self.invalidate_plan();h=self.horizon.value();goal=self.target.copy()
            tolerance=self.tolerance.value();maximum=self.max_node.value()
            settings=dict(planning_rate_fraction=self.planning_rate.value()/100,reserve_steps=self.reserve_steps.value(),budget_s=self.planning_budget.value(),iterations=self.planning_iterations.value(),shooting_nfev=self.planning_shooting.value(),horizon_step=self.planning_stride.value())
            if self.auto_curve is not None:
                curve=self.auto_curve.copy()
                self._job(lambda:self.runtime.plan_any_segment(curve,tolerance,maximum,h,cancel=self.cancel_event,**settings),self.planned)
            else:
                settings['node_indices']=None if self.target_ids is None else self.target_ids.copy()
                self._job(lambda:self.runtime.plan_to_tolerance(goal,tolerance,maximum,h,cancel=self.cancel_event,**settings),self.planned)
        except Exception as e:self.fail(e)
    def planned(self,plan):
        self.plan=plan
        self.target_matrix=plan.get('target_matrix');self.target_samples=plan.get('target_samples')
        if plan.get('matching_mode')=='any_segment':
            self.target=plan['goal'].copy();self.target_ids=plan['node_indices'].copy()
            self.runtime.target_shape=self.target.copy();self.runtime.target_node_indices=self.target_ids
            self.canvas.highlight_indices=self.target_ids.copy()
        self.trial_check.setChecked(False);self.trial_check.setVisible(not plan['qualified'])
        self.trial_note.setText('' if plan['qualified'] else f'未达到目标：平均误差 {plan["mean_error"]:.2f} mm，最大误差 {plan["max_error"]:.2f} mm。可检查紫色预览后勾选试运行；不表示已到位。')
        p=self.runtime.mapping.expand(plan['actions'])
        predicted=self.runtime.engine.rollout(plan['state'],plan['actions'])
        error=float(target_distances(predicted[-1],plan['goal'],plan.get('node_indices'),plan.get('target_matrix'),plan.get('target_samples')).mean())
        np.savez_compressed(self.runtime.run_dir/('plan_'+uuid.uuid4().hex[:8]+'.npz'),**{k:v for k,v in plan.items() if k not in ('trace','prediction','attempts','matching_attempts')},prediction=predicted)
        self.preview.setPlainText(f'模型内终态目标残差（受约束节点） {error:.3f} mm\n每腔最小 {p.min(0).round(2)}\n每腔最大 {p.max(0).round(2)}\n步数 {plan.get("primary_steps",len(p))} + 余量 {plan.get("reserve_steps",0)} = {len(p)}，dt={self.runtime.dt:.3f}s\n初始规划耗时 {plan.get("planning_ms",0)/1000:.3f} s')
        self.preview.appendPlainText('搜索：'+', '.join(f'{a["horizon"]}步/{a.get("total_ms",0):.0f}ms' for a in plan['attempts']))
        self.preview.appendPlainText(f'规划速度预算 {plan.get("planning_config",{}).get("planning_rate_fraction",1)*100:.0f}%；在线矫正可用完整有效速率上限')
        self.preview.appendPlainText(('预测达标' if plan['qualified'] else '当前搜索未达标，可确认后试运行')+f' · 最大目标点 {plan["max_error"]:.3f} mm')
        if plan.get('matching_mode')=='any_segment':
            self.preview.appendPlainText(f'自动匹配节点 {self.target_ids[0]} → {self.target_ids[-1]}，'+('反向' if plan['matching_reversed'] else '正向')+f'绘制；尝试 {len(plan["matching_attempts"])}/{plan["matching_candidates_total"]} 个对应。误差覆盖32个曲线采样点。')
        self.preview_plot.clear()
        for i in range(6):self.preview_plot.plot(np.arange(1,len(p)+1)*self.runtime.dt,p[:,i],pen=(i,6),name=f'c{i}')
        self.scrubber.setRange(0,len(p)-1);self.show_preview(0)
        self.message('找到预测达标计划，请播放预览后确认执行' if plan['qualified'] else '当前搜索未达标；可重新规划，或到第4页明确确认后试运行')
    def arm(self,checked):
        if checked and (self.plan is None or (not self.plan.get('qualified',False) and not self.trial_check.isChecked())):
            self.arm_check.setChecked(False);self.message('请先生成计划；未达标计划需检查预览并勾选允许试运行');return
        self.armed=bool(checked);self.canvas.locked=self.armed
        for c in self.controls:
            if c not in (self.arm_check,self.execute_button):c.setEnabled(not self.armed)
        self.host._refresh()
    def execution_mode_changed(self,checked):
        self.arm_check.setChecked(False)
        self.trial_check.setChecked(False)
        self.execute_button.setText('开始实验运动：仅按规划执行（不矫正）' if checked else '开始实验运动：执行 Analytic B 图像反馈')
        self.message('对照模式：仅按规划运行，图像仅记录；请检查计划后重新确认' if checked else '已启用 Analytic B 图像矫正；请检查计划后重新确认')

    def frame_provider(self):
        return self.host.hardware.latest_camera_frame(self.host._current_cam_index)
    def _camera_key(self):
        profile=self.host.hardware.profile
        return (self.host._current_cam_index,self.host.hardware.camera_epoch,profile.camera_backend,profile.camera_count,profile.camera_serials,profile.camera_driver,profile.camera_sources)
    def execute(self):
        try:
            if not self.armed or self.plan is None:raise ValueError('请先确认当前计划')
            current_safety=np.asarray([[c.value() for c in row] for row in self.host._safety_cells])[:,:4]
            if not np.array_equal(current_safety,self.safety_ui):raise ValueError('六腔安全配置已更改，请重新规划')
            if not self.runtime.alignment_confirmed or self.camera_identity!=self._camera_key():raise ValueError('相机配置已变化，需重新对齐')
            if tuple(c.currentData() for c in self.mapping)!=self.runtime.mapping.expansion:raise ValueError('腔道映射尚未应用')
            frame=self.frame_provider()
            from ..hardware.profile import DeviceState
            if self.host.hardware.states['camera']!=DeviceState.READY:raise ValueError('相机未 READY')
            if frame is None or time.monotonic()-frame[1]>.3:raise ValueError('没有新相机图像')
            self.results.clear()
            transport=self._transport();plan=self.plan;allow_trial=self.trial_check.isChecked()
            config=self.occlusion_controls.config
            self.executor=HereditaryExecutor(self.runtime,transport,self.frame_provider,
                image_transform=config.apply,camera_provider=self.host.hardware.camera_frames,
                evaluation_provider=self.host.hardware.evaluation_samples,selected_camera=self.host._current_cam_index,
                metadata=dict(hardware=self.host.hardware.snapshot(),software_occlusion=dict(enabled=config.enabled,rectangles=config.rectangles)),
                max_missing=self.max_missing.value(),max_skipped=self.max_skipped.value(),settle_s=self.feedback_settle.value()/1000,use_correction=not self.open_loop_check.isChecked(),feedback_interval_steps=self.feedback_interval.value())
            self.arm_check.blockSignals(True);self.arm_check.setChecked(False);self.arm_check.blockSignals(False);self.armed=False
            self._job(lambda:self.executor.execute(plan,allow_unqualified=allow_trial),lambda rows:self.completed(rows))
            self.executor.callback=self.job.progress.emit;self.job.progress.connect(self.progress)
        except Exception as e:self.fail(e)
    def completed(self,rows):
        status='已停止运动' if self.executor and self.executor.hold_requested else '计划指令完成'
        self.invalidate_plan();self.results.appendPlainText(f'{status}，{len(rows)} 条 ACK 指令；当前保持最终压力。\n动作 CSV / NDI / 原图 / 反馈图：{getattr(self.executor,"last_execution_dir",self.runtime.run_dir)}\nACK 不是实测腔压；模型残差不是实机到位误差。')
        assessment=getattr(self.executor,'completion_assessment',None)
        if assessment and assessment.get('estimated_mean_error_mm') is not None:
            self.results.appendPlainText(f'最终模型估计：平均 {assessment["estimated_mean_error_mm"]:.2f} mm / 最大 {assessment["estimated_max_error_mm"]:.2f} mm；'+('估计达标' if assessment['estimate_within_tolerance'] else '估计未达标')+'。这不是实测到位判定。')
        next_step='可继续规划' if self.runtime.ready else '回第二页点击重新预热，配准仍有效时无需重新提取'
        self.message(f'{status}，{len(rows)} 条 ACK 指令，保留末条压力；{next_step}。')
    def play_preview(self):
        if self.plan is None:return
        if self.play_timer.isActive():self.play_timer.stop()
        else:self.play_timer.start(max(20,int(self.runtime.dt*1000)))
    def next_preview(self):
        self.scrubber.setValue((self.scrubber.value()+1)%(self.scrubber.maximum()+1))
    def show_preview(self,index):
        if self.plan is None:return
        self.canvas.preview=transform(self.plan['prediction'][index],self.runtime.matrix);self.canvas.update()
        pressure=self.runtime.mapping.expand(self.plan['actions'][index])
        self.pressure_preview.setText(f'模型预测 t={(index+1)*self.runtime.dt:.2f}s / 第 {index+1} 步\n六腔 kPa：'+str(pressure.round(1)))
    def request_hold(self):
        self.chambers.stop();self.cancel_event.set();self.invalidate_plan()
        if self.busy and self.executor:self.executor.hold()
    def stop(self):
        self.chambers.stop();self.cancel_event.set()
        if self.busy:
            self.zero_pending=True
            if self.executor:self.executor.abort()
            return
        self.invalidate_plan()
        if self.runtime and self.host.hardware.valve_controller:
            try:
                transport=self._transport(tuple(sorted(self.host.hardware.valve_controller.connected_groups)));executor=HereditaryExecutor(self.runtime,transport,self.frame_provider)
                self._job(executor.zero,lambda _:self.message('六腔归零已 ACK；保留配准与已知历史，第二页重新预热后再规划'))
            except Exception as e:self.fail(e)
    def probe(self):
        frame=self.frame_provider();hardware=self.host.hardware
        info=dict(profile=hardware.profile.to_dict(),frame_age_ms=None if frame is None else (time.monotonic()-frame[1])*1000,
                  frame_size=None if frame is None else list(frame[0].shape),valve_groups=[] if hardware.valve_controller is None else sorted(hardware.valve_controller.connected_groups),
                  pressure_feedback='命令及 ACK；设备没有实测腔压接口')
        if self.runtime:self.runtime.record('device_probe',**info)
        self.probe_result.show();self.probe_result.setPlainText(json.dumps(info,ensure_ascii=False,indent=2))
    def tick(self):
        self._attach_controller()
        if time.monotonic()-self.evidence_stamp>.3:self.canvas.evidence=None
        if not self.busy:
            reusable=self.runtime is not None and self.runtime.matrix is not None and self.runtime.alignment_confirmed and not self.canvas.frozen
            self.confirm_button.setText('保持当前压力，重新预热（复用配准）' if reusable else '确认形状并部署预热')
        index=self.host._current_cam_index
        frame=self.host._camera_frames.get(index);stamp=self.host._camera_frame_times.get(index)
        if frame is not None and stamp is not None:
            with self.frame_lock:self.frame=(frame,stamp)
            self.canvas.set_frame(self.occlusion_controls.config.apply(frame)[0])
        r=self.runtime
        if r and r.initialized and r.matrix is not None:
            if self.camera_identity is not None and self.camera_identity!=self._camera_key():
                r.alignment_confirmed=False;r.ready=False;self.invalidate_plan()
                if self.busy and self.executor:self.executor.abort()
            if not self.busy and not self.canvas.frozen:
                try:
                    fresh=self.frame_provider()
                    if r.alignment_confirmed and self.plan is None and fresh is not None and fresh[1]>r.last_frame:
                        from threadpoolctl import threadpool_limits
                        with threadpool_limits(1):
                            _,info=r.feedback(self.occlusion_controls.config.apply(fresh[0])[0],fresh[1],np.empty((0,4)),np.empty((0,r.engine.n_nodes,2)))
                        r.record('hold_observation',**info,state=r.state)
                        if info.get('edges',0) or 'visibility' in info:self.progress(info)
                        if self.target is not None and (self.auto_curve is None or self.target_ids is not None) and info.get('prediction_px') is not None:
                            model=transform(np.asarray(info['prediction_px']),np.linalg.inv(r.matrix))
                            error=float(target_distances(model,self.target,self.target_ids,self.target_matrix,self.target_samples).mean())
                            self.feedback_label.setText(self.feedback_label.text()+f' · 模型目标差 {error:.2f} mm（非实测全形状）')
                    with r.lock:z,u=r.state_at(time.monotonic())
                    self.canvas.prediction=transform(r.engine.observe(z,u),r.matrix);self.canvas.radius=r.meta["radius_mm"]*np.linalg.norm(r.matrix[:2,0]);self.canvas.update()
                except Exception as e:self.fail(e)
    def shutdown(self):
        self.timer.stop();self.play_timer.stop();self.chambers.stop();self.cancel_event.set()
        if self.executor:self.executor.abort()
        return self.job if self.job and self.job.isRunning() else None
