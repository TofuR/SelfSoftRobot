"""Action-driven virtual camera for plumbing checks, never physical evidence."""
import time
import cv2
import numpy as np
from PyQt5.QtCore import QObject,QTimer,pyqtSignal
from ..runtime.hereditary_deployment import advance,ChannelMapping,transform


class HereditaryMockCamera(QObject):
    frame_ready=pyqtSignal(object,float)
    error=pyqtSignal(str)
    def __init__(self,engine,meta,controller,parent=None):
        super().__init__(parent);self.engine=engine;self.meta=meta;self.controller=controller
        self.mapping=ChannelMapping(tuple(meta['expansion6']),tuple(meta['action_unit_to_kpa']))
        self.action=np.zeros(engine.channels);drive,_=engine.drive(self.action)
        self.state=np.r_[np.repeat(drive,engine.n_play),np.repeat(drive,engine.n_maxwell)]
        self.at=time.monotonic();self.matrix=np.eye(3);self.matrix[:2,:2]*=1.25
        shape=engine.observe(self.state,self.action)
        self.matrix[:2,2]=[320,95]-1.25*shape[0]
        self.occluded=False
        self.timer=QTimer(self);self.timer.setInterval(33);self.timer.timeout.connect(self.emit_frame)
        controller.action_logged.connect(self.command)
    def command(self,pressures,stamp):
        try:
            u=self.mapping.reduce(pressures)
            self.state=advance(self.engine,self.state,self.action,max(0.,stamp-self.at),self.meta['dt'])
            self.state=advance(self.engine,self.state,u,0.,self.meta['dt']);self.at=stamp;self.action=u
        except Exception as error:self.error.emit(str(error))
    def current_curve(self):
        z=advance(self.engine,self.state,self.action,time.monotonic()-self.at,self.meta['dt'])
        return transform(self.engine.observe(z,self.action),self.matrix)
    def emit_frame(self):
        image=np.full((480,640,3),(45,35,25),np.uint8)
        curve=self.current_curve();radius=self.meta['radius_mm']*np.linalg.norm(self.matrix[:2,0])
        # Model nodes represent short-edge centers. A thick cv2 polyline adds
        # round caps beyond both nodes and biases pixel-derived scale/base.
        tangent=np.gradient(curve,axis=0)
        normal=np.column_stack([-tangent[:,1],tangent[:,0]])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
        polygon=np.vstack([curve+radius*normal,(curve-radius*normal)[::-1]])
        cv2.fillPoly(image,[np.rint(polygon).astype(np.int32)],(225,225,225),cv2.LINE_AA)
        if self.occluded:image[222:278,295:351]=35
        cv2.putText(image,'MODEL MOCK - NOT PHYSICAL',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(80,180,240),1)
        self.frame_ready.emit(image,time.monotonic())
    def start(self):self.timer.start();self.emit_frame()
    def stop(self):
        self.timer.stop()
        try:self.controller.action_logged.disconnect(self.command)
        except (TypeError,RuntimeError):pass
