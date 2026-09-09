"""Readiness requires distinct fresh observations, quality and stable shape."""
from dataclasses import dataclass
import numpy as np

@dataclass
class WarmupCriteria:
    minimum_s: float = 2.
    timeout_s: float = 20.
    consecutive: int = 8
    coverage: float = .75
    residual_px: float = 3.
    motion_px: float = 1.

    def __post_init__(self):
        values=[self.minimum_s,self.timeout_s,self.consecutive,self.coverage,self.residual_px,self.motion_px]
        if not np.isfinite(values).all() or not 0<=self.minimum_s<self.timeout_s or self.consecutive<2 or not 0<self.coverage<=1 or min(self.residual_px,self.motion_px)<=0:
            raise ValueError('预热阈值无效')

class WarmupGate:
    def __init__(self,criteria,started):
        self.criteria=criteria;self.started=started;self.last=-float('inf');self.previous=None;self.good=0
    def update(self,info,stamp,now):
        c=self.criteria
        if now-self.started>c.timeout_s:raise ValueError('预热超时，图像质量或稳定性未达标')
        if stamp<=self.last or now-stamp>.3:
            self.good=0;return False
        self.last=stamp
        shape=np.asarray(info.get('prediction_px',[]))
        observer=info.get('observer',{});count=observer.get('count',0)
        residual=2*np.sqrt(observer.get('after',float('inf'))/max(1,count))
        motion=float('inf') if self.previous is None or shape.shape!=self.previous.shape or shape.ndim!=2 or len(shape)==0 else float(np.max(np.linalg.norm(shape-self.previous,axis=1)))
        quality=count>0 and info.get('visibility',{}).get('coverage',0)>=c.coverage and residual<=c.residual_px and motion<=c.motion_px
        self.good=self.good+1 if quality else 0;self.previous=shape.copy()
        info['warmup']=dict(good=self.good,required=c.consecutive,residual_px=float(residual) if np.isfinite(residual) else None,motion_px=float(motion) if np.isfinite(motion) else None,elapsed_s=now-self.started)
        return now-self.started>=c.minimum_s and self.good>=c.consecutive
