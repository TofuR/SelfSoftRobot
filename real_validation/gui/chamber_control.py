"""Nonmodal chamber controls, with one worker owning the shared actuator."""
import threading
import time
import json
import uuid
from collections import deque
import numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QPushButton,QDoubleSpinBox

class ChamberControl(QDialog):
    def __init__(self,panel):
        super().__init__(panel.host);self.panel=panel;self.setWindowTitle('六腔控制 · kPa / kPa·s⁻¹');self.setModal(False)
        self.setWindowFlags(self.windowFlags()|Qt.Tool)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.desired=np.zeros(6);self.driving=False;self.takeover_pending=False;self.drive_error=None
        self.recent_commands=deque(maxlen=1000);self.audit_controller=None
        root=QVBoxLayout(self)
        self.device_status=QLabel();self.device_status.setWordWrap(True);root.addWidget(self.device_status)
        grid=QGridLayout();root.addLayout(grid)
        for j,text in enumerate(['腔','输入','编辑目标','生效目标','ACK 下发','min','max','上升/s','下降/s']):grid.addWidget(QLabel(text),0,j)
        self.targets=[];self.mapping_labels=[];self.acks=[];self.active_targets=[];self.limits=[]
        for i in range(6):
            grid.addWidget(QLabel(f'c{i}'),i+1,0);label=QLabel('—');self.mapping_labels.append(label);grid.addWidget(label,i+1,1)
            target=QDoubleSpinBox();target.setRange(0,500);target.setDecimals(1);target.setMaximumWidth(85);self.targets.append(target);grid.addWidget(target,i+1,2)
            target.valueChanged.connect(lambda value,k=i:self.link(k,value))
            active=QLabel('—');self.active_targets.append(active);grid.addWidget(active,i+1,3)
            ack=QLabel('未知');self.acks.append(ack);grid.addWidget(ack,i+1,4)
            cells=[]
            for j,value in enumerate([0,150,50,50]):
                cell=QDoubleSpinBox();cell.setRange(0,500);cell.setDecimals(1);cell.setValue(value);cell.setMaximumWidth(85);grid.addWidget(cell,i+1,j+5);cells.append(cell)
            self.limits.append(cells)
        self.effective=QLabel('加载模型后显示实际生效交集');self.effective.setWordWrap(True);root.addWidget(self.effective)
        self.status=QLabel('输入目标后点击“下发目标（按限速）”。ACK 为下发确认，不是实测腔压。');self.status.setWordWrap(True);root.addWidget(self.status)
        self.drive_status=QLabel('尚未启动手动下发');self.drive_status.setWordWrap(True);root.addWidget(self.drive_status)
        row=QHBoxLayout();root.addLayout(row)
        self.apply_button=QPushButton('仅应用设置（不启动）');self.apply_button.clicked.connect(self.apply)
        self.start_button=QPushButton('下发目标（按限速）');self.start_button.clicked.connect(self.start)
        self.start_button.setToolTip('校验当前目标和限制，然后连续下发直到目标；运行中再次点击更新目标。自动控制中先接管。')
        self.end_button=QPushButton('结束调压（保持）');self.end_button.clicked.connect(self.stop)
        self.zero_button=QPushButton('全部归零');self.zero_button.clicked.connect(panel.stop)
        self.diagnostic_button=QPushButton('保存调压诊断');self.diagnostic_button.clicked.connect(self.save_diagnostics)
        for w in (self.apply_button,self.start_button,self.end_button,self.zero_button):row.addWidget(w)
        for w in (self.apply_button,self.start_button,self.end_button,self.zero_button):w.setAutoDefault(False)
        self.start_button.setDefault(True)
        root.addWidget(self.diagnostic_button)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(100)

    def link(self,index,value):
        r=self.panel.runtime
        if r is None:return
        for i,box in enumerate(self.targets):
            if r.mapping.expansion[i]==r.mapping.expansion[index]:
                box.blockSignals(True);box.setValue(value);box.blockSignals(False)
        if hasattr(self,'status'):
            self.status.setText('目标已编辑，尚未下发；点击“下发目标（按限速）”使其生效。')

    def refresh(self):
        p=self.panel;r=p.runtime;c=p.host.hardware.valve_controller;groups=set() if c is None else set(c.connected_groups)
        if c is not self.audit_controller:
            if self.audit_controller is not None:
                self.audit_controller.command_issued.disconnect(self.record_command)
                self.audit_controller.communication_result.disconnect(self.record_ack)
                if hasattr(self.audit_controller,'mgr'):self.audit_controller.mgr.wire_event.disconnect(self.record_wire)
            self.audit_controller=c;self.recent_commands.clear()
            if c is not None:
                c.command_issued.connect(self.record_command);c.communication_result.connect(self.record_ack)
                if hasattr(c,'mgr'):c.mgr.wire_event.connect(self.record_wire)
        from ..hardware.valve import MockValveController
        if c is None:self.device_status.setText('阀未连接')
        elif isinstance(c,MockValveController):
            self.device_status.setText('MOCK 虚拟阀：ACK 为模拟应答，不会发送串口或驱动机器人。')
            self.device_status.setStyleSheet('color:#b45309;font-weight:bold')
        else:
            ports=' / '.join(f'组{g}: {c.group_ports.get(g,"未配置")}'+(' 已连接' if g in groups else ' 未连接') for g in (1,2))
            self.device_status.setText(f'REAL 串口阀 · {ports} · {c.baudrate} baud / 从站 {c.slave_addr} · 两组 4–20 mA')
            self.device_status.setStyleSheet('font-weight:bold')
        for i in range(6):
            input_id=None if r is None else r.mapping.expansion[i]
            needed=set() if r is None else {j//3+1 for j,v in enumerate(r.mapping.expansion) if v==input_id}
            available=bool(needed) and needed<=groups
            self.mapping_labels[i].setText('—' if input_id is None else f'u{input_id}')
            self.targets[i].setEnabled(available and (not p.busy or self.driving))
            self.acks[i].setText(f'{p.ack6[i]:.1f}' if i//3+1 in groups and np.isfinite(p.ack6[i]) else '未知')
            self.active_targets[i].setText(f'{self.desired[i]:.1f}' if self.driving else '—')
            for cell in self.limits[i]:cell.setEnabled(not p.busy or self.driving)
        self.apply_button.setEnabled(not p.busy or self.driving)
        self.start_button.setEnabled(r is not None and bool(groups) and not self.takeover_pending)
        self.start_button.setText('更新并下发目标' if self.driving else ('接管并下发目标' if p.busy else '下发目标（按限速）'))
        if self.driving and self.drive_error is None:
            active=[i for i,v in enumerate(r.mapping.expansion)
                    if {j//3+1 for j,x in enumerate(r.mapping.expansion) if x==v}<=groups]
            if not active or not np.isfinite(p.ack6[active]).all():
                self.drive_status.setText('手动调压中：等待阀组 ACK')
            else:
                with self.lock:error=float(np.max(abs(p.ack6[active]-self.desired[active])))
                self.drive_status.setText('目标指令已 ACK，持续保持（非实测腔压）' if error<.1
                                          else f'按限速调压中：ACK 指令距目标最大 {error:.1f} kPa')
        if self.takeover_pending and not p.busy:
            self.takeover_pending=False
            if r and r.fault:self.status.setText('历史或通信故障，未自动接管：'+r.fault)
            else:self.start()

    def apply(self):
        try:
            p=self.panel;r=p.runtime
            if r is None:raise ValueError('请先加载模型')
            if p.busy and not self.driving:raise ValueError('自动控制中，请先手动接管')
            values=np.array([[c.value() for c in row] for row in self.limits])
            controller=p.host.hardware.valve_controller
            current=np.zeros(6) if controller is None else np.array(controller.last_command)
            bounds=r.configure_limits(*values.T,commit=False,current=r.mapping.reduce(current))
            desired=np.array([box.value() for box in self.targets]);u=r.mapping.reduce(desired)
            if np.any(u<bounds.lower) or np.any(u>bounds.upper):raise ValueError('目标超出共享腔与模型范围的交集')
            with self.lock:
                r.configure_limits(*values.T,current=r.mapping.reduce(current));self.desired=desired.copy()
                if controller is not None:
                    controller.configure_safety((bounds.rise/r.dt*r.mapping.scale)[list(r.mapping.expansion)].tolist(),(bounds.fall/r.dt*r.mapping.scale)[list(r.mapping.expansion)].tolist())
            for row,source in zip(p.host._safety_cells,values):
                for cell,value in zip(row,source):cell.setValue(float(value))
            p.invalidate_plan();p.host._save_hardware_config()
            self.effective.setText('生效 u0…u3：min '+str((bounds.lower*r.mapping.scale).round(1))+' / max '+str((bounds.upper*r.mapping.scale).round(1))+' kPa')
            self.status.setText('目标与限制已应用，下一拍下发。' if self.driving else '设置已保存，尚未下发；点击“下发目标（按限速）”开始调压。');return True
        except Exception as error:
            suffix='；新设置未生效，仍按“生效目标”调压。可点击“结束调压（保持）”。' if self.driving else '；未下发。'
            self.status.setText(str(error)+suffix);return False

    def start(self):
        p=self.panel
        if self.driving:
            self.apply();return
        if p.busy:
            p.request_hold();self.takeover_pending=True;self.status.setText('等待当前操作及在途 ACK 结束，再接管');return
        if not self.apply():return
        try:
            r=p.runtime;c=p.host.hardware.valve_controller;groups=tuple(sorted(c.connected_groups))
            if r.initialized and groups!=(1,2):raise ValueError('在线模型需要两组阀；断组后请重新部署')
            if not groups:raise ValueError('请先连接所需阀组')
            transport=p._transport(groups);p.invalidate_plan();p.cancel_draft();self.stop_event.clear();self.driving=True;self.drive_error=None
            self.status.setText('已启动连续调压；修改目标后点击“更新并下发目标”。')
            self.drive_status.setText('手动调压中：等待阀组 ACK')
            def work():
                try:
                    while not self.stop_event.is_set():
                        with self.lock:
                            current=np.array(c.last_command);u=r.mapping.reduce(current)
                            target=self.desired.copy()
                            for i,input_id in enumerate(r.mapping.expansion):
                                required={j//3+1 for j,v in enumerate(r.mapping.expansion) if v==input_id}
                                if not required<=set(groups):target[i]=current[i]
                            step=r.bounds.project(r.mapping.reduce(target)[None],u)[0]
                        receipt=transport.send(r.mapping.expand(step),groups,.5)
                        if receipt.status!='ack':raise ValueError(f'手动命令未 ACK：{receipt.status}（阀组 {groups}，命令 {receipt.command_id}）')
                        if r.initialized:r.acknowledge(receipt)
                        if self.stop_event.wait(r.dt):break
                    return None
                except Exception:
                    transport.zero(.5);raise
            def done(_):self.drive_status.setText('已结束调压，保持最后 ACK 压力')
            p._job(work,done)
            p.job.failed.connect(self.failed)
            p.job.finished.connect(self.finished)
        except Exception as error:self.driving=False;self.status.setText(str(error))
    def failed(self,error):
        self.drive_error=str(error);self.drive_status.setText('调压失败：'+str(error));self.status.setText('已停止手动调压并尝试归零；请检查阀组连接及主窗口通信日志。')
    def record_command(self,ident,requested,applied,stamp):
        self.recent_commands.append(dict(event='command',command_id=ident,t=stamp,requested6=requested,applied6=applied))
    def record_ack(self,ident,group,ok,stamp,status):
        self.recent_commands.append(dict(event='ack',command_id=ident,group=group,ok=ok,t=stamp,status=status))
    def record_wire(self,ident,group,direction,frame,stamp):
        self.recent_commands.append(dict(event='serial',command_id=ident,group=group,direction=direction,frame_hex=frame,t=stamp))
    def save_diagnostics(self):
        try:
            p=self.panel;r=p.runtime;c=p.host.hardware.valve_controller
            if r is None:raise ValueError('请先加载模型')
            folder=r.run_dir/'manual_diagnostics';folder.mkdir(exist_ok=True)
            path=folder/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]+'.json')
            payload=dict(hardware=p.host.hardware.snapshot(),controller_type=None if c is None else type(c).__name__,
                         device_status=self.device_status.text(),driving=self.driving,status=self.drive_status.text(),
                         edited_target6=[box.value() for box in self.targets],active_target6=self.desired.tolist(),
                         last_command6=None if c is None else c.last_command,
                         ack6=[float(x) if np.isfinite(x) else None for x in p.ack6],
                         mapping=list(r.mapping.expansion),dt=r.dt,limits6=[[box.value() for box in row] for row in self.limits],
                         recent_commands=list(self.recent_commands),note='ACK 是指令写入应答，不是实测压力；环形通信诊断不参与模型历史。')
            path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            self.status.setText('调压诊断已保存：'+str(path));p.host._log('调压诊断：'+str(path));return path
        except Exception as error:self.status.setText(str(error));return None
    def finished(self):self.driving=False
    def stop(self):self.takeover_pending=False;self.stop_event.set()
    def closeEvent(self,event):self.stop();event.accept()
    def reject(self):self.stop();super().reject()
