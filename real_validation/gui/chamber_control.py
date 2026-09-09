"""Nonmodal chamber controls, with one worker owning the shared actuator."""
import threading
import time
import numpy as np
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,QPushButton,QDoubleSpinBox

class ChamberControl(QDialog):
    def __init__(self,panel):
        super().__init__(panel.host);self.panel=panel;self.setWindowTitle('六腔控制 · kPa / kPa·s⁻¹');self.setModal(False)
        self.setWindowFlags(self.windowFlags()|Qt.Tool)
        self.stop_event=threading.Event();self.lock=threading.Lock();self.desired=np.zeros(6);self.driving=False;self.takeover_pending=False
        root=QVBoxLayout(self);grid=QGridLayout();root.addLayout(grid)
        for j,text in enumerate(['腔','输入','目标','ACK 下发','min','max','上升/s','下降/s']):grid.addWidget(QLabel(text),0,j)
        self.targets=[];self.mapping_labels=[];self.acks=[];self.limits=[]
        for i in range(6):
            grid.addWidget(QLabel(f'c{i}'),i+1,0);label=QLabel('—');self.mapping_labels.append(label);grid.addWidget(label,i+1,1)
            target=QDoubleSpinBox();target.setRange(0,500);target.setDecimals(1);target.setMaximumWidth(85);self.targets.append(target);grid.addWidget(target,i+1,2)
            target.valueChanged.connect(lambda value,k=i:self.link(k,value))
            ack=QLabel('未知');self.acks.append(ack);grid.addWidget(ack,i+1,3)
            cells=[]
            for j,value in enumerate([0,150,50,50]):
                cell=QDoubleSpinBox();cell.setRange(0,500);cell.setDecimals(1);cell.setValue(value);cell.setMaximumWidth(85);grid.addWidget(cell,i+1,j+4);cells.append(cell)
            self.limits.append(cells)
        self.effective=QLabel('加载模型后显示实际生效交集');self.effective.setWordWrap(True);root.addWidget(self.effective)
        self.status=QLabel('编辑不会自动发送。ACK 为下发确认，不是实测腔压。');self.status.setWordWrap(True);root.addWidget(self.status)
        row=QHBoxLayout();root.addLayout(row)
        self.apply_button=QPushButton('应用目标与限制');self.apply_button.clicked.connect(self.apply)
        self.start_button=QPushButton('手动调压 / 接管');self.start_button.clicked.connect(self.start)
        self.end_button=QPushButton('结束调压（保持）');self.end_button.clicked.connect(self.stop)
        self.zero_button=QPushButton('全部归零');self.zero_button.clicked.connect(panel.stop)
        for w in (self.apply_button,self.start_button,self.end_button,self.zero_button):row.addWidget(w)
        self.timer=QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(100)

    def link(self,index,value):
        r=self.panel.runtime
        if r is None:return
        for i,box in enumerate(self.targets):
            if r.mapping.expansion[i]==r.mapping.expansion[index]:
                box.blockSignals(True);box.setValue(value);box.blockSignals(False)

    def refresh(self):
        p=self.panel;r=p.runtime;c=p.host.hardware.valve_controller;groups=set() if c is None else set(c.connected_groups)
        for i in range(6):
            input_id=None if r is None else r.mapping.expansion[i]
            needed=set() if r is None else {j//3+1 for j,v in enumerate(r.mapping.expansion) if v==input_id}
            available=bool(needed) and needed<=groups
            self.mapping_labels[i].setText('—' if input_id is None else f'u{input_id}')
            self.targets[i].setEnabled(available and (not p.busy or self.driving))
            self.acks[i].setText(f'{p.ack6[i]:.1f}' if i//3+1 in groups and np.isfinite(p.ack6[i]) else '未知')
            for cell in self.limits[i]:cell.setEnabled(not p.busy or self.driving)
        self.apply_button.setEnabled(not p.busy or self.driving)
        self.start_button.setEnabled(r is not None and bool(groups) and not self.driving)
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
            self.status.setText('目标与限制已应用；手动调压运行时下一拍生效。');return True
        except Exception as error:self.status.setText(str(error));return False

    def start(self):
        p=self.panel
        if p.busy:
            p.request_hold();self.takeover_pending=True;self.status.setText('等待当前操作及在途 ACK 结束，再接管');return
        if not self.apply():return
        try:
            r=p.runtime;c=p.host.hardware.valve_controller;groups=tuple(sorted(c.connected_groups))
            if r.initialized and groups!=(1,2):raise ValueError('在线模型需要两组阀；断组后请重新部署')
            transport=p._transport(groups);p.invalidate_plan();p.cancel_draft();self.stop_event.clear();self.driving=True
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
                        if receipt.status!='ack':raise ValueError('手动命令未 ACK')
                        if r.initialized:r.acknowledge(receipt)
                        if self.stop_event.wait(r.dt):break
                    return None
                except Exception:
                    transport.zero(.5);raise
            def done(_):self.status.setText('已结束调压，保持最后 ACK 压力')
            p._job(work,done)
            p.job.finished.connect(self.finished)
        except Exception as error:self.driving=False;self.status.setText(str(error))
    def finished(self):self.driving=False
    def stop(self):self.takeover_pending=False;self.stop_event.set()
    def closeEvent(self,event):self.stop();event.accept()
    def reject(self):self.stop();super().reject()
