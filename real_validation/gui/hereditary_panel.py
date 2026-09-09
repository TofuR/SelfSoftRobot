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
        self.plan=None;self.target=None;self.armed=False;self.busy=False;self.frame=None
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
        self.draft_stamp=None;self.draft_version=None
        self.display_controls=QWidget();display=QHBoxLayout(self.display_controls)
        self.display_mode=QComboBox();self.display_mode.addItems(['整体形状 + 中心线','仅骨架中心线'])
        self.display_mode.currentIndexChanged.connect(self.update_display)
        self.opacity=QSlider(Qt.Horizontal);self.opacity.setRange(5,70);self.opacity.setValue(28)
        self.opacity.valueChanged.connect(self.update_display)
        display.addWidget(QLabel('叠图'));display.addWidget(self.display_mode)
        display.addWidget(QLabel('不透明度'));display.addWidget(self.opacity)
        display.addWidget(QLabel('青：当前估计  紫：轨迹预览  红：目标  黄：草稿'))
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
        self.polarity=QComboBox();self.polarity.addItem('亮色臂身 / 较暗背景','bright');self.polarity.addItem('黑色剪影 / 明亮背景','dark');initial.addWidget(self.polarity)
        self.polarity.currentIndexChanged.connect(self.invalidate_plan)
        self.auto_button=QPushButton('自动提取当前完整形状（冻结图像，可拖动黄点）');self.auto_button.clicked.connect(self.auto_shape);initial.addWidget(self.auto_button)
        self.align_button=QPushButton('手动重画当前中心线（base → tip）');self.align_button.clicked.connect(lambda:self.set_mode('align'));initial.addWidget(self.align_button)
        row=QHBoxLayout()
        self.flip_button=QPushButton('交换 base / tip');self.flip_button.clicked.connect(self.flip_draft);row.addWidget(self.flip_button)
        self.cancel_button=QPushButton('取消草稿 / 恢复实时图像');self.cancel_button.clicked.connect(self.cancel_draft);row.addWidget(self.cancel_button);initial.addLayout(row)
        self.confirm_button=QPushButton('确认形状并部署预热');self.confirm_button.clicked.connect(self.confirm_alignment);initial.addWidget(self.confirm_button)
        self.alignment_label=QLabel('等待完整形状：初始化应无遮挡；自动端点方向需检查。');self.alignment_label.setWordWrap(True);initial.addWidget(self.alignment_label)
        self.warmup_options=QGroupBox('高级：预热质量阈值');self.warmup_options.setCheckable(True);self.warmup_options.setChecked(False)
        warm_form=QFormLayout(self.warmup_options);self.warmup_fields=[]
        for label,default,lo,hi in [('最短保持 s',2.,0,60),('超时 s',20.,1,120),('连续新帧',8,2,100),('边缘覆盖',.75,.1,1),('边缘残差 px',3.,.1,20),('帧间变化 px',1.,.1,10)]:
            box=QDoubleSpinBox();box.setRange(lo,hi);box.setValue(default);self.warmup_fields.append(box);warm_form.addRow(label,box);box.setVisible(False)
        self.warmup_options.toggled.connect(lambda visible:[warm_form.itemAt(i).widget().setVisible(visible) for i in range(warm_form.count())])
        for i in range(warm_form.count()):warm_form.itemAt(i).widget().hide()
        initial.addWidget(self.warmup_options)
        self.goal_button=QPushButton('连续画目标中心线（base → tip）');self.goal_button.clicked.connect(lambda:self.set_mode('goal'));goal.addWidget(self.goal_button)
        note=QLabel('从已对齐 base 开始连续描画。拟合全部节点；整体形状显示由中心线和模型半径扩展，不是分割测量。当前路线未加入避障。')
        note.setWordWrap(True);goal.addWidget(note)
        self.horizon=QSpinBox();self.horizon.setRange(2,80);self.horizon.setValue(80)
        self.tolerance=QDoubleSpinBox();self.tolerance.setRange(.05,20);self.tolerance.setValue(2.);self.tolerance.setSuffix(' mm')
        self.max_node=QDoubleSpinBox();self.max_node.setRange(.1,50);self.max_node.setValue(4.);self.max_node.setSuffix(' mm')
        self.planning_dialog=QDialog(self);self.planning_dialog.setWindowTitle('初始规划参数');self.planning_dialog.setModal(False)
        pf=QVBoxLayout(self.planning_dialog);form=QFormLayout();pf.addLayout(form)
        form.addRow('全形状平均容限（不含 base）',self.tolerance);form.addRow('最大节点偏差',self.max_node);form.addRow('搜索上限步数（自动选择长度）',self.horizon)
        self.planning_budget=QDoubleSpinBox();self.planning_budget.setRange(.1,120);self.planning_budget.setValue(15);self.planning_budget.setSuffix(' s')
        self.planning_iterations=QSpinBox();self.planning_iterations.setRange(1,100);self.planning_iterations.setValue(24)
        self.planning_shooting=QSpinBox();self.planning_shooting.setRange(1,200);self.planning_shooting.setValue(30)
        self.planning_stride=QSpinBox();self.planning_stride.setRange(1,80);self.planning_stride.setValue(20)
        for label,box in [('总计算预算（迭代间检查）',self.planning_budget),('每个长度 B 迭代上限',self.planning_iterations),('终态优化函数评估上限',self.planning_shooting),('长度搜索步长',self.planning_stride)]:form.addRow(label,box)
        explanation=QLabel('参数修改立即生效，已有计划失效。自动尝试较短长度，达标即停；步数上限不是固定执行长度。总预算在优化迭代间检查，单次迭代可能略超预算。在线反馈期限由模型 dt 决定。')
        explanation.setWordWrap(True);pf.addWidget(explanation)
        close=QPushButton('完成');close.clicked.connect(self.planning_dialog.hide);pf.addWidget(close)
        self.planning_settings=QPushButton('初始规划参数…');self.planning_settings.clicked.connect(self.show_planning_settings);goal.addWidget(self.planning_settings)
        self.planning_fields=[self.horizon,self.tolerance,self.max_node,self.planning_budget,self.planning_iterations,self.planning_shooting,self.planning_stride]
        for box in self.planning_fields:box.valueChanged.connect(self.invalidate_plan)
        self.plan_button=QPushButton('规划全形状 / 预览');self.plan_button.clicked.connect(self.make_plan);goal.addWidget(self.plan_button)
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
        self.feedback_settle=QDoubleSpinBox();self.feedback_settle.setRange(0,90);self.feedback_settle.setSuffix(' ms');self.feedback_settle.setToolTip('始终要求 ACK 后的新图像；只有需要额外等待时才增大，会减少本周期反馈预算')
        missing_form=QFormLayout();missing_form.addRow('连续无有效反馈几次后停止归零',self.max_missing);missing_form.addRow('连续跳过反馈几次后停止归零',self.max_skipped);missing_form.addRow('发令后额外等图',self.feedback_settle);run.addLayout(missing_form)
        self.arm_check=QCheckBox('已检查目标、映射、对齐和压力范围，允许执行');self.arm_check.toggled.connect(self.arm);run.addWidget(self.arm_check)
        self.execute_button=QPushButton('开始实验运动：执行 Analytic B 图像反馈');self.execute_button.clicked.connect(self.execute);run.addWidget(self.execute_button)
        self.hold_stop_button=QPushButton('停止运动（保持当前压力）');self.hold_stop_button.clicked.connect(self.request_hold);run.addWidget(self.hold_stop_button)
        self.stop_button=QPushButton('中止并全部归零');self.stop_button.clicked.connect(self.stop);run.addWidget(self.stop_button)
        self.feedback_label=QLabel('反馈自动筛选可信边缘，不需要手画遮挡区。边缘缺失可能来自遮挡、光照或对齐误差。');self.feedback_label.setWordWrap(True);run.addWidget(self.feedback_label)
        self.results=QPlainTextEdit();self.results.setReadOnly(True);run.addWidget(self.results)
        note=QLabel('完成后保持最终压力。此模型仅支持正压；不支持实际负压。虚拟相机与 ACK 仅用于软件流程验证。')
        note.setWordWrap(True);run.addWidget(note)
        for layout in layouts:layout.addStretch()
        self.controls=[self.path,browse,self.load_button,self.polarity,*self.mapping,self.mapping_button,self.auto_button,self.align_button,self.flip_button,self.cancel_button,self.confirm_button,self.goal_button,self.horizon,self.tolerance,self.max_node,self.plan_button,self.planning_settings,*self.planning_fields,self.arm_check,self.execute_button,self.probe_button,self.occlusion_controls,self.max_missing,self.max_skipped,self.feedback_settle,*self.warmup_fields]
        from .chamber_control import ChamberControl
        self.chambers=ChamberControl(self)
        self.chamber_button=QPushButton('六腔控制');self.chamber_button.clicked.connect(self.chambers.show)
        display.insertWidget(0,self.chamber_button)
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(100)

    @property
    def active(self):return self.busy or self.armed

    def message(self,text):self.state_label.setText(str(text));self.host._log('Hereditary: '+str(text))
    def invalidate_plan(self,*_):
        self.plan=None;self.armed=False
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
        root=Path(self.host.run_root.text())/'hereditary'
        folder=root/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
        self.runtime=HereditaryDeployment(engine,meta,folder)
        self.host.model_badge.setText('Hereditary · 4 输入 / Analytic B')
        for c,v in zip(self.mapping,meta['expansion6']):c.setCurrentIndex(v)
        if hasattr(self,'saved_chambers'):
            self.runtime.set_mapping(self.saved_chambers['mapping'])
            for c,v in zip(self.mapping,self.saved_chambers['mapping']):c.setCurrentIndex(v)
        self.feedback_settle.setMaximum(max(0,meta['dt']*1000-5));self.horizon.setMaximum(meta['max_horizon']);self.invalidate_plan();self.target=None
        self.canvas.target=None;self.canvas.prediction=None
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
        if self.canvas.mode=='goal':
            self.canvas.mode='view';self.canvas.stroke=[];self.canvas.update()

    def cancel_draft(self):
        self.canvas.frozen=False;self.canvas.draft=None;self.canvas.stroke=[]
        self.canvas.mode='view';self.canvas.drag_node=None;self.canvas.drawing=False
        self.draft_stamp=None;self.draft_version=None;self.canvas.update()

    def _freeze(self):
        r=self.runtime
        if r is None:raise ValueError('请先加载模型')
        if r.pending or r.fault:raise ValueError(r.fault or '等待指令 ACK')
        frame=self.frame_provider()
        if frame is None or time.monotonic()-frame[1]>.3:raise ValueError('需要新相机图像')
        self.cancel_draft();self.invalidate_plan()
        r.alignment_confirmed=False;r.ready=False;self.alignment_error=None;self.target=None;self.canvas.target=None
        self.canvas.set_frame(frame[0].copy());self.canvas.frozen=True
        self.draft_stamp=frame[1];self.draft_version=r.version;self.camera_identity=self._camera_key()
        return frame

    def auto_shape(self):
        try:
            from ..perception.initial_shape import extract_initial_shape
            r=self.runtime
            if r is None:raise ValueError('请先加载模型')
            if self.chambers.driving:raise ValueError('请先结束手动调压并保持')
            if self.occlusion_controls.config.enabled:raise ValueError('初始化提取完整形状前，请关闭软件遮挡测试')
            if tuple(c.currentData() for c in self.mapping)!=r.mapping.expansion:raise ValueError('请先应用映射')
            transport=self._transport();count=r.engine.n_nodes;polarity=self.polarity.currentData();frames=[]
            def work():
                receipt=transport.send(self.host.hardware.valve_controller.last_command,(1,2),.5)
                if receipt.status!='ack':raise ValueError('当前保持压力未确认')
                if r.initialized and not r.fault:r.acknowledge(receipt)
                else:r.initialize(receipt.applied6,receipt.t_command)
                until=time.monotonic()+.8
                while time.monotonic()<until:
                    if self.cancel_event.wait(.01):raise ValueError('提取已取消')
                    frame=self.frame_provider()
                    if frame is not None and frame[1]>max(receipt.t_ack,receipt.t_command+.05):
                        frames.append(frame);return extract_initial_shape(frame[0],count,polarity)
                raise ValueError('确认保持后没有新图像')
            def done(value):
                curve,mask,info=value;frame=frames[0]
                self._freeze();self.canvas.set_frame(frame[0]);self.canvas.frame=frame[0].copy();self.draft_stamp=frame[1]
                import cv2
                name='initial_'+uuid.uuid4().hex[:8]
                cv2.imwrite(str(self.runtime.run_dir/(name+'.png')),frame[0])
                cv2.imwrite(str(self.runtime.run_dir/(name+'_mask.png')),mask)
                self.runtime.record('initial_shape_draft',source='pixels',frame=name+'.png',timestamp=frame[1],curve=curve,diagnostics=info)
                self.canvas.draft=curve.astype(float);self.canvas.radius=info['radius_px'];self.canvas.mode='edit';self.canvas.update()
                self.alignment_label.setText('图像已冻结。拖动黄点修正；检查 BASE / TIP，可交换方向。确认按钮才计算并提交对齐。')
                self.message('完整形状草稿已提取，等待人工检查与确认')
            self._job(work,done)
        except Exception as e:self.fail(e)

    def draft_changed(self,points):
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
            if self.runtime is None or not self.runtime.initialized:raise ValueError('请先提取当前形状，建立保持压力历史')
            if mode=='align':self._freeze()
            elif not self.runtime.ready:raise ValueError('请先完成部署预热')
            else:self.cancel_draft()
            self.invalidate_plan();self.canvas.mode=mode;self.canvas.stroke=[]
            self.message('按住左键，从 base 连续画到 tip；松开完成 '+('当前形状草稿' if mode=='align' else '目标全形状'))
        except Exception as e:self.fail(e)

    def curve(self,points):
        try:
            if self.canvas.mode=='align':
                self.canvas.draft=resample_curve(points,self.runtime.engine.n_nodes)
                self.canvas.mode='edit';self.draft_changed(self.canvas.draft)
            else:
                self.target=self.runtime.goal(points);self.runtime.target_shape=self.target.copy();self.canvas.target=transform(self.target,self.runtime.matrix)
                self.runtime.record('target',camera_curve=points,model_goal=self.target);self.message('目标已记录，可规划全形状接近过程')
                self.canvas.mode='view'
            self.invalidate_plan();self.canvas.stroke=[];self.canvas.update()
        except Exception as e:self.fail(e)

    def confirm_alignment(self):
        try:
            if self.canvas.draft is None:raise ValueError('请先自动提取或描画当前完整中心线')
            if self.camera_identity!=self._camera_key() or self.draft_version!=self.runtime.version:
                raise ValueError('相机或动作历史在提取后变化，请重新提取')
            matrix,error=self.runtime.align(self.canvas.draft,timestamp=self.draft_stamp)
            self.draft_version=self.runtime.version;self.alignment_error=error
            scale=np.linalg.norm(matrix[:2,0]);angle=np.degrees(np.arctan2(matrix[1,0],matrix[0,0]))
            base=transform(self.runtime.engine.base[None],matrix)[0]
            self.alignment_label.setText(f'尺度 {scale:.3f} px/mm · 旋转 {angle:.1f}°\nbase ({base[0]:.1f}, {base[1]:.1f}) px · 拟合 RMS {error:.2f} px')
            if error>8:raise ValueError('对齐 RMS 超过 8 px；检查草稿、方向或初始姿态。尚未确认。')
            state_error=self.runtime.fit_full_state(self.canvas.draft,self.draft_stamp)
            self.runtime.confirm_alignment();self.runtime.record('initial_shape_confirmed',curve=self.canvas.draft,timestamp=self.draft_stamp)
            self.canvas.radius=self.runtime.meta['radius_mm']*scale
            self.cancel_draft();self.message(f'坐标已固定，初始状态拟合 {state_error:.2f} mm；保持当前压力预热中')
            self.start_warmup()
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
            raise ValueError('预热超时，模型尚未就绪；检查图像和对齐后重新提取')
        self._job(work,lambda _:self.message('模型就绪：对齐与状态估计已完成，保持时继续观测。进入第 3 页画目标。'))
        self.job.progress.connect(self.progress)

    def progress(self,info):
        if 'prediction_px' in info:
            self.canvas.prediction=np.asarray(info['prediction_px']);self.canvas.update()
        visibility=info.get('visibility',{})
        latency=info.get('compute_ms',info.get('feedback_wait_ms',0))
        timing_label='计算' if 'compute_ms' in info else '等待反馈'
        text=(f'{timing_label} {latency:.1f} ms · 可信边缘 {info.get("edges",0)} · '
              f'可见证据覆盖 {visibility.get("coverage",0):.0%} · {visibility.get("status","图像不可用")}')
        if info.get('visible_target_error_mm') is not None:text+=f' · 可见边缘目标差 {info["visible_target_error_mm"]:.2f} mm'
        if 'warmup' in info:text+=f' · 预热连续合格 {info["warmup"]["good"]}/{info["warmup"]["required"]}'
        if info.get('consecutive_missing'):text+=f' · 连续无反馈 {info["consecutive_missing"]} 次，仅模型预测'
        if 'revision_status' in info:
            status={'committed':'已提交','deadline_expired':'超时跳过','worker_busy':'上次仍在计算，跳过','no_budget':'本周期无预算，跳过','snapshot_changed':'状态变化，丢弃','frame_missing':'无新图像','operator_abort':'已停止'}.get(info['revision_status'],info['revision_status'])
            text+=f' · {status}'
            interval=info.get('command_interval_ms')
            self.results.appendPlainText(f'步骤 {info.get("step")} | 发令间隔 {interval:.1f} ms | 等待反馈 {info.get("feedback_wait_ms",0):.1f}/{info.get("feedback_budget_ms",0):.1f} ms | {status}' if interval is not None else f'步骤 {info.get("step")} | {info["revision_status"]}')
        self.feedback_label.setText(text)
        self.state_label.setText(text)

    def show_planning_settings(self):
        self.planning_dialog.show();self.planning_dialog.raise_()

    def make_plan(self):
        try:
            if self.target is None:raise ValueError('请先画完整目标')
            self.safety_ui=np.asarray([[c.value() for c in row] for row in self.host._safety_cells])[:,:4]
            self.runtime.configure_limits(*self.safety_ui.T)
            self.runtime.edge_polarity=self.polarity.currentData()
            self.runtime.record('image_frontend',polarity=self.runtime.edge_polarity)
            self.invalidate_plan();h=self.horizon.value();goal=self.target.copy()
            tolerance=self.tolerance.value();maximum=self.max_node.value()
            settings=dict(budget_s=self.planning_budget.value(),iterations=self.planning_iterations.value(),shooting_nfev=self.planning_shooting.value(),horizon_step=self.planning_stride.value())
            self._job(lambda:self.runtime.plan_to_tolerance(goal,tolerance,maximum,h,cancel=self.cancel_event,**settings),self.planned)
        except Exception as e:self.fail(e)
    def planned(self,plan):
        self.plan=plan
        p=self.runtime.mapping.expand(plan['actions'])
        predicted=self.runtime.engine.rollout(plan['state'],plan['actions'])
        error=float(np.linalg.norm(predicted[-1]-plan['goal'],axis=1)[1:].mean())
        np.savez_compressed(self.runtime.run_dir/('plan_'+uuid.uuid4().hex[:8]+'.npz'),**{k:v for k,v in plan.items() if k not in ('trace','prediction','attempts')},prediction=predicted)
        self.preview.setPlainText(f'模型内终态全形状残差 {error:.3f} mm\n每腔最小 {p.min(0).round(2)}\n每腔最大 {p.max(0).round(2)}\n步数 {len(p)}，dt={self.runtime.dt:.3f}s\n初始规划耗时 {plan.get("planning_ms",0)/1000:.3f} s')
        self.preview.appendPlainText('搜索：'+', '.join(f'{a["horizon"]}步/{a.get("total_ms",0):.0f}ms' for a in plan['attempts']))
        self.preview.appendPlainText(('预测达标' if plan['qualified'] else '当前搜索未达标，不能执行')+f' · 最大节点 {plan["max_error"]:.3f} mm')
        self.preview_plot.clear()
        for i in range(6):self.preview_plot.plot(np.arange(1,len(p)+1)*self.runtime.dt,p[:,i],pen=(i,6),name=f'c{i}')
        self.scrubber.setRange(0,len(p)-1);self.show_preview(0)
        self.message('找到预测达标计划，请播放预览后确认执行' if plan['qualified'] else '当前搜索未达标；调整目标或容限后重新规划')
    def arm(self,checked):
        if checked and (self.plan is None or not self.plan.get('qualified',False)):self.arm_check.setChecked(False);self.fail('请先规划');return
        self.armed=bool(checked);self.canvas.locked=self.armed
        for c in self.controls:
            if c not in (self.arm_check,self.execute_button):c.setEnabled(not self.armed)
        self.host._refresh()
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
            transport=self._transport();plan=self.plan
            config=self.occlusion_controls.config
            self.executor=HereditaryExecutor(self.runtime,transport,self.frame_provider,
                image_transform=config.apply,camera_provider=self.host.hardware.camera_frames,
                evaluation_provider=self.host.hardware.evaluation_samples,selected_camera=self.host._current_cam_index,
                metadata=dict(hardware=self.host.hardware.snapshot(),software_occlusion=dict(enabled=config.enabled,rectangles=config.rectangles)),
                max_missing=self.max_missing.value(),max_skipped=self.max_skipped.value(),settle_s=self.feedback_settle.value()/1000)
            self.arm_check.blockSignals(True);self.arm_check.setChecked(False);self.arm_check.blockSignals(False);self.armed=False
            self._job(lambda:self.executor.execute(plan),lambda rows:self.completed(rows))
            self.executor.callback=self.job.progress.emit;self.job.progress.connect(self.progress)
        except Exception as e:self.fail(e)
    def completed(self,rows):
        status='已停止运动' if self.executor and self.executor.hold_requested else '计划指令完成'
        self.invalidate_plan();self.results.appendPlainText(f'{status}，{len(rows)} 条 ACK 指令；当前保持最终压力。\n动作 CSV / NDI / 原图 / 反馈图：{getattr(self.executor,"last_execution_dir",self.runtime.run_dir)}\nACK 不是实测腔压；模型残差不是实机到位误差。')
        self.message(f'{status}，{len(rows)} 条 ACK 指令，保留末条压力；继续观测当前形状。')
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
                self._job(executor.zero,lambda _:self.message('六腔归零已 ACK；后续需重新规划'))
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
                        if info.get('edges',0):self.progress(info)
                        if self.target is not None and info.get('prediction_px') is not None:
                            model=transform(np.asarray(info['prediction_px']),np.linalg.inv(r.matrix))
                            error=float(np.linalg.norm(model[1:]-self.target[1:],axis=1).mean())
                            self.feedback_label.setText(self.feedback_label.text()+f' · 模型目标差 {error:.2f} mm（非实测全形状）')
                    with r.lock:z,u=r.state_at(time.monotonic())
                    self.canvas.prediction=transform(r.engine.observe(z,u),r.matrix);self.canvas.radius=r.meta["radius_mm"]*np.linalg.norm(r.matrix[:2,0]);self.canvas.update()
                except Exception as e:self.fail(e)
    def shutdown(self):
        self.timer.stop();self.play_timer.stop();self.chambers.stop();self.cancel_event.set()
        if self.executor:self.executor.abort()
        return self.job if self.job and self.job.isRunning() else None
