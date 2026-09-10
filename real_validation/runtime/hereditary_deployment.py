"""Portable Analytic B deployment: explicit mapping, alignment and timed history.

No training modules or hardware imports. Pressures are commanded kPa, not sensed
chamber pressure. Coordinates are planar model mm and host camera pixels.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from .hereditary_math import FrozenHereditary, fast_suffix_b
from .hereditary_bounds import ActionBounds
from .shape_target import target_indices, target_distances, segment_projection, validate_projection
from ..perception.partial_edges import extract_edges_vectorized


def resample_curve(points, count):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError('请从 base 到 tip 画一条完整曲线')
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]]
    if len(p) < 2:
        raise ValueError('曲线长度必须为正')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return np.column_stack([np.interp(np.linspace(0, distance[-1], count), distance, p[:, j]) for j in range(2)])


@dataclass(frozen=True)
class ChannelMapping:
    expansion: tuple[int, ...]
    scale: tuple[float, ...]  # kPa per checkpoint-normalized input, includes norm_factor

    def __post_init__(self):
        if len(self.expansion) != 6 or set(self.expansion) != set(range(len(self.scale))):
            raise ValueError('6 个腔必须覆盖所有模型输入；同一输入可分配给多个腔')
        if any(not np.isfinite(v) or v <= 0 for v in self.scale):
            raise ValueError('模型压力尺度必须为正有限值')

    def expand(self, action):
        p = np.asarray(action)*self.scale
        return p[..., list(self.expansion)]

    def reduce(self, pressures):
        p = np.asarray(pressures, dtype=float)
        if p.shape != (6,) or not np.isfinite(p).all():
            raise ValueError('需要六个有限压力值')
        result = []
        for i, scale in enumerate(self.scale):
            values = p[np.asarray(self.expansion) == i]
            if np.ptp(values) > .05:
                raise ValueError(f'u{i} 对应腔压力不一致，不能恢复四维动作')
            result.append(float(values.mean())/scale)
        return np.asarray(result)


def similarity_alignment(model_points, camera_points):
    """Base-anchored rotation + uniform scale; never fit a deforming affine warp."""
    x = np.asarray(model_points, dtype=float)
    y = resample_curve(camera_points, len(x))
    a, b = x-x[0], y-y[0]
    u, _, vt = np.linalg.svd(a.T@b)
    rotation = u@np.diag([1., np.linalg.det(u@vt)])@vt
    scale = float(np.sum((a@rotation)*b)/np.sum(a*a))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('对齐失败：请检查 base→tip 方向')
    matrix = np.eye(3)
    matrix[:2,:2] = scale*rotation.T
    matrix[:2,2] = y[0]-matrix[:2,:2]@x[0]
    residual = float(np.sqrt(np.mean(np.sum((transform(x,matrix)-y)**2, axis=1))))
    return matrix, residual


def transform(points, matrix):
    return np.asarray(points)@np.asarray(matrix)[:2,:2].T + np.asarray(matrix)[:2,2]


def load_bundle(path):
    """Read a frozen numeric export with an integrity-bound deployment contract."""
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema') != 'hereditary_deployment_v1':
        raise ValueError('不是 hereditary 部署包')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['weights_sha256']:
        raise ValueError('部署参数哈希不匹配')
    engine = FrozenHereditary.__new__(FrozenHereditary)
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            v = data[key]
            setattr(engine, key, v.item() if v.ndim == 0 else v.copy())
    if engine.channels != len(meta['action_unit_to_kpa']) or engine.channels != 4:
        raise ValueError('本实验入口要求 4 维 hereditary 模型')
    limits=np.asarray([meta['lower_kpa'],meta['upper_kpa'],meta['rate_kpa_s']],dtype=float)
    if (limits.shape!=(3,4) or not np.isfinite(limits).all() or np.any(limits[0]!=0)
            or np.any(limits[1]<=0) or np.any(limits[1]>500) or np.any(limits[2]<=0)):
        raise ValueError('部署包正压/速率范围无效')
    if (not np.isfinite(meta['dt']) or not 0<meta['dt']<=1
            or not np.isfinite(meta['radius_mm']) or meta['radius_mm']<=0
            or not 2<=meta['max_horizon']<=200):
        raise ValueError('部署包时间/半径/规划长度无效')
    return engine, meta


def advance(engine, state, action, duration, dt):
    if duration < -1e-8:
        raise ValueError('时间倒退')
    # PI reacts to the new input immediately; Maxwell integrates actual hold time.
    drive, _ = engine.drive(action)
    p = np.clip(state[:engine.n_p].reshape(engine.channels,engine.n_play),
                drive[:,None]-engine.thresholds, drive[:,None]+engine.thresholds)
    h = state[engine.n_p:].reshape(engine.channels,engine.n_maxwell)
    alpha = engine.alpha**(max(0.,duration)/dt)
    return np.r_[p.ravel(), (alpha*h+(1-alpha)*drive[:,None]).ravel()]


def edge_state_update(engine, state, action, evidence, matrix, radius, bounds,
                      prior_std=.08, max_delta=.12):
    """Analytic point-to-segment residual derivative, with fixed associations."""
    if not len(evidence.pixels):
        return state.copy(), {'accepted':False, 'count':0, 'reason':'no_evidence'}
    def evaluate(z, derivative=False):
        if derivative:
            shape, cz, _ = engine.observe(z, action, derivatives=True)
        else:
            shape = engine.observe(z,action)
        curve = transform(shape,matrix)
        ids = evidence.segments
        a,b = curve[ids],curve[ids+1]
        v = b-a
        ratio = np.clip(np.sum((evidence.pixels-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-8),0,1)
        offset = a+ratio[:,None]*v-evidence.pixels
        length = np.linalg.norm(offset,axis=1)
        residual = (length-radius)/2.
        if not derivative:
            return residual
        camera_jac = np.einsum('ij,njz->niz',matrix[:2,:2],cz)
        closest_jac = (1-ratio[:,None,None])*camera_jac[ids]+ratio[:,None,None]*camera_jac[ids+1]
        jac = np.einsum('ni,niz->nz',offset/np.maximum(length[:,None],1e-8),closest_jac)/2.
        return residual,jac
    value,jac = evaluate(state,True)
    weight = 2./np.maximum(abs(value),2.)
    delta = np.linalg.solve(jac.T@(weight[:,None]*jac)+np.eye(engine.z_dim)/prior_std**2,-jac.T@(weight*value))
    delta *= min(1.,max_delta/max(float(abs(delta).max()),1e-12))
    cost = lambda v: np.where(abs(v)<=2.,v*v,4*abs(v)-4).sum()
    before = float(cost(value))
    drive,_ = engine.drive(action)
    lo,_ = engine.drive(bounds.lower);hi,_ = engine.drive(bounds.upper)
    for fraction in (1.,.5,.25,.125):
        candidate = state+fraction*delta
        candidate[:engine.n_p] = np.clip(candidate[:engine.n_p].reshape(engine.channels,engine.n_play),drive[:,None]-engine.thresholds,drive[:,None]+engine.thresholds).ravel()
        candidate[engine.n_p:] = np.clip(candidate[engine.n_p:].reshape(engine.channels,engine.n_maxwell),np.minimum(lo,0)[:,None],np.maximum(hi,0)[:,None]).ravel()
        after = float(cost(evaluate(candidate)))
        if after+np.sum(((candidate-state)/prior_std)**2) < before-1e-9:
            return candidate,dict(accepted=True,count=len(value),before=before,after=after)
    return state.copy(),dict(accepted=False,count=len(value),before=before,after=before)


class HereditaryDeployment:
    def __init__(self, engine, meta, run_dir, clock=time.monotonic):
        self.engine,self.meta,self.clock = engine,meta,clock
        self.dt = float(meta['dt'])
        self.mapping = ChannelMapping(tuple(meta['expansion6']),tuple(meta['action_unit_to_kpa']))
        self.bounds = ActionBounds(np.asarray(meta['lower_kpa'])/self.mapping.scale,
                                   np.asarray(meta['upper_kpa'])/self.mapping.scale,
                                   np.asarray(meta['rate_kpa_s'])*self.dt/self.mapping.scale,
                                   np.asarray(meta['rate_kpa_s'])*self.dt/self.mapping.scale)
        self.run_dir = Path(run_dir);self.run_dir.mkdir(parents=True,exist_ok=False)
        self.lock = threading.RLock()
        self.log = (self.run_dir/'events.jsonl').open('a',buffering=1)
        self.matrix = None;self.alignment_confirmed = False;self.initialized = False
        self.state = None;self.action = np.zeros(engine.channels);self.at = self.clock()
        self.history = deque(maxlen=4096);self.seen = set();self.version = 0
        self.last_frame = -float('inf');self.fault = None
        self.ready=False;self.deployment_id=None
        self.edge_polarity='bright'
        self.pending = set()
        self.phase = "preparation"; self.history_epoch = 0
        self.record('loaded',checkpoint=meta['checkpoint_sha256'],initial_history='unknown',metadata=meta)

    def record(self,event,**payload):
        self.log.write(json.dumps(dict(event=event,t=self.clock(),phase=self.phase,history_epoch=self.history_epoch,**payload),default=lambda a: np.asarray(a).tolist(),allow_nan=False)+'\n')

    def set_phase(self, phase, **context):
        if phase not in ('preparation','initial_hold','planning','control','final_hold','stopped'):
            raise ValueError('unknown experiment phase')
        with self.lock:
            previous=self.phase;self.phase=phase
            self.record('phase_changed',previous_phase=previous,**context)

    def begin_history_epoch(self):
        """Start a new experimental segment without erasing known physical memory."""
        with self.lock:
            if not self.initialized or self.pending or self.fault:
                raise ValueError(self.fault or '需要已初始化且全部 ACK 的当前状态')
            now=self.clock();state,action=self.state_at(now)
            self.at,self.state,self.action=now,state,action
            self.history.clear();self._snapshot();self.history_epoch+=1;self.version+=1
            self.last_frame=-float('inf')
            self.set_phase('initial_hold',reason='new_history_epoch_preserving_known_state')
            self.record('history_epoch_started',state=self.state,applied6=self.mapping.expand(self.action),
                        assumption='carry known p/h and actual hold; earlier audit log retained')

    def set_mapping(self, expansion):
        with self.lock:
            self.mapping = ChannelMapping(tuple(expansion),self.mapping.scale)
            self.initialized=False;self.ready=False;self.alignment_confirmed=False;self.matrix=None;self.version+=1
            self.set_phase('preparation',reason='mapping_changed')
            self.record('mapping',expansion=list(expansion))

    def configure_limits(self, lower6, upper6, rise6, fall6, *, commit=True, current=None):
        values=np.asarray([lower6,upper6,rise6,fall6],dtype=float)
        if values.shape!=(4,6) or not np.isfinite(values).all() or np.any(values<0) or np.any(values[0]>values[1]):
            raise ValueError('六腔压力/速率配置无效')
        lo=[];hi=[];rise=[];fall=[]
        for i in range(self.engine.channels):
            ids=np.asarray(self.mapping.expansion)==i
            lo.append(max(self.meta['lower_kpa'][i],float(values[0,ids].max())))
            hi.append(min(self.meta['upper_kpa'][i],float(values[1,ids].min())))
            rise.append(min(self.meta['rate_kpa_s'][i],float(np.where(values[2,ids]>0,values[2,ids],np.inf).min())))
            fall.append(min(self.meta['rate_kpa_s'][i],float(np.where(values[3,ids]>0,values[3,ids],np.inf).min())))
        scale=np.asarray(self.mapping.scale)
        bounds=ActionBounds(np.asarray(lo)/scale,np.asarray(hi)/scale,np.asarray(rise)*self.dt/scale,np.asarray(fall)*self.dt/scale)
        check=self.action if current is None else np.asarray(current)
        if np.any(check<bounds.lower-1e-7) or np.any(check>bounds.upper+1e-7):
            raise ValueError('新范围排除当前压力，请先调压再收紧范围')
        if not commit:return bounds
        with self.lock:
            if all(np.array_equal(getattr(bounds,k),getattr(self.bounds,k)) for k in ('lower','upper','rise','fall')):return bounds
            self.bounds=bounds;self.version+=1
            self.record('effective_limits',lower_kpa=lo,upper_kpa=hi,rise_kpa_s=rise,fall_kpa_s=fall)

    def initialize(self, pressures, timestamp=None):
        with self.lock:
            u=self.mapping.reduce(pressures)
            if np.any(u<self.bounds.lower) or np.any(u>self.bounds.upper):
                raise ValueError('初始压力超出当前模型范围')
            drive,_=self.engine.drive(u)
            self.state=np.r_[np.repeat(drive,self.engine.n_play),np.repeat(drive,self.engine.n_maxwell)]
            self.action=u;self.at=self.clock() if timestamp is None else float(timestamp)
            self.initialized=True;self.ready=False;self.fault=None;self.history.clear();self.version+=1
            self.matrix=None;self.alignment_confirmed=False;self.last_frame=-float('inf');self.history_epoch+=1
            self.set_phase('initial_hold',reason='confirmed_hold_equilibrium_prior')
            self._snapshot();self.record('initialize',assumption='training-consistent equilibrium prior at ACK input; physical memory remains uncertain',applied6=list(pressures))

    def _snapshot(self):
        self.history.append((self.at,self.state.copy(),self.action.copy()))

    def state_at(self,timestamp):
        if not self.initialized or timestamp < self.history[0][0]:
            raise ValueError('图像早于有效动作历史')
        t,z,u=next(row for row in reversed(self.history) if row[0]<=timestamp)
        return advance(self.engine,z,u,timestamp-t,self.dt),u.copy()

    def acknowledge(self,receipt):
        with self.lock:
            if receipt.command_id in self.seen:return
            self.seen.add(receipt.command_id);self.pending.discard(receipt.command_id)
            if self.initialized:self.record('command_receipt',**vars(receipt))
            if receipt.status!='ack':
                self.fault='指令未确认，状态历史失效';raise ValueError(self.fault)
            if not self.initialized:return
            u=self.mapping.reduce(receipt.applied6)
            if (np.any(u<self.bounds.lower-1e-6) or np.any(u>self.bounds.upper+1e-6)) and not np.all(abs(u)<1e-8):
                self.fault='已下发压力超出模型范围';raise ValueError(self.fault)
            # Only commands create knots; idle holding is integrated analytically.
            if receipt.t_command < self.at-1e-6:
                self.fault='指令时间早于已校正状态';raise ValueError(self.fault)
            self.state=advance(self.engine,self.state,self.action,receipt.t_command-self.at,self.dt)
            self.state=advance(self.engine,self.state,u,0,self.dt)
            self.action=u;self.at=receipt.t_command;self.version+=1;self._snapshot()

    def align(self,camera_curve, timestamp=None):
        with self.lock:
            z,u=self.state_at(self.clock() if timestamp is None else timestamp)
            matrix,error=similarity_alignment(self.engine.observe(z,u),camera_curve)
            self.matrix=matrix;self.alignment_confirmed=False;self.version+=1
            self.record('alignment_candidate',matrix=matrix,residual_px=error)
            return matrix,error

    def confirm_alignment(self):
        if self.matrix is None:raise ValueError('请先绘制当前完整形状建立对齐')
        self.alignment_confirmed=True;self.record('alignment_confirmed',matrix=self.matrix)

    def goal(self,camera_curve):
        if not self.alignment_confirmed:raise ValueError('请先确认对齐叠图')
        curve=resample_curve(camera_curve,self.engine.n_nodes)
        result=transform(curve,np.linalg.inv(self.matrix))
        if np.linalg.norm(result[0]-self.engine.base)>5:
            raise ValueError('目标必须从已对齐 base 开始，起点偏差超过 5 mm')
        result[0]=self.engine.base
        return result

    def partial_goal(self, camera_curve, node_indices):
        if not self.alignment_confirmed:raise ValueError('请先确认对齐叠图')
        ids=target_indices(self.engine.n_nodes,node_indices)
        points=np.asarray(camera_curve,dtype=float)
        if len(ids)==1:
            if points.shape!=(1,2) or not np.isfinite(points).all():raise ValueError('请单击指定一个目标点')
        else:points=resample_curve(points,len(ids))
        with self.lock:
            z,u=self.state_at(self.clock())
            result=self.engine.observe(z,u).copy()
        result[ids]=transform(points,np.linalg.inv(self.matrix))
        return result,ids

    def plan(self,goal,horizon, *, snapshot=None, seed_actions=None, deadline=None, cancel=None, iterations=24, node_indices=None, target_matrix=None, target_samples=None):
        if not 2<=horizon<=self.meta['max_horizon']:raise ValueError('规划长度超出部署包实验上限')
        with self.lock:
            if self.fault or self.pending:raise ValueError(self.fault or '等待指令 ACK')
            if snapshot is None:z,u=self.state_at(self.clock());version=self.version
            else:z,u,version=snapshot
        ids=target_indices(self.engine.n_nodes,node_indices)
        target_distances(self.engine.observe(z,u),goal,ids,target_matrix,target_samples)
        self.set_phase('planning',reason='goal_plan_requested')
        baseline=np.tile(u,(horizon,1))
        # ReLU spline autograd is zero exactly at the lower knot. A feasible
        # interior seed avoids a spurious stationary zero-pressure initial plan.
        epsilon=np.minimum(.001,(self.bounds.upper-self.bounds.lower)/100)
        seed=np.maximum(u,self.bounds.lower+epsilon)
        old=self.bounds.project(np.tile(seed,(horizon,1)),u)
        if seed_actions is not None:
            initial=np.vstack([seed_actions,np.tile(seed_actions[-1],(max(0,horizon-len(seed_actions)),1))])[:horizon]
            old=self.bounds.project(initial,u)
        # Only explicitly mapped nodes contribute to approach and terminal costs.
        current=self.engine.observe(z,u)
        reference=current[None]+np.minimum(np.arange(1,horizon+1)/(horizon*.7),1)[:,None,None]*(goal-current)
        if target_matrix is not None:
            target_matrix,target_samples=validate_projection(target_matrix,target_samples,self.engine.n_nodes,ids)
            sampled=target_matrix@current
            reference=sampled[None]+np.minimum(np.arange(1,horizon+1)/(horizon*.7),1)[:,None,None]*(target_samples-sampled)
        def cost(actions):
            shapes=self.engine.rollout(z,actions)
            residual=shapes[:,ids]-reference[:,ids] if target_matrix is None else np.einsum('sn,tnc->tsc',target_matrix,shapes)-reference
            return np.mean(residual**2)
        trace=[]
        for _ in range(iterations):
            if cancel is not None and cancel.is_set():raise ValueError('规划已取消')
            if deadline is not None and self.clock()>deadline:break
            old,info=fast_suffix_b(self.engine,z,old,u,reference,self.bounds,node_indices=ids,target_matrix=target_matrix);trace.append(info)
            if not info['accepted']:break
        if cost(old)>cost(baseline):
            old=baseline
        result=dict(actions=old,reference=reference,state=z,previous=u,version=version,created=self.clock(),trace=trace,goal=goal,node_indices=ids)
        if target_matrix is not None:result.update(target_matrix=target_matrix.copy(),target_samples=target_samples.copy())
        return result

    def fit_full_state(self, curve, timestamp):
        """Fit bounded memory with the camera transform fixed, retaining a prior."""
        with self.lock:
            z,u=self.state_at(timestamp);prior=z.copy()
            target=transform(resample_curve(curve,self.engine.n_nodes),np.linalg.inv(self.matrix))
            drive,_=self.engine.drive(u)
            low,_=self.engine.drive(self.bounds.lower);high,_=self.engine.drive(self.bounds.upper)
            def project(v):
                v[:self.engine.n_p]=np.clip(v[:self.engine.n_p].reshape(self.engine.channels,self.engine.n_play),drive[:,None]-self.engine.thresholds,drive[:,None]+self.engine.thresholds).ravel()
                v[self.engine.n_p:]=np.clip(v[self.engine.n_p:].reshape(self.engine.channels,self.engine.n_maxwell),np.minimum(low,0)[:,None],np.maximum(high,0)[:,None]).ravel()
                return v
            def cost(v):return np.sum((self.engine.observe(v,u)[1:]-target[1:])**2)+np.sum((v-prior)**2)/.08**2
            for _ in range(12):
                shape,jac,_=self.engine.observe(z,u,derivatives=True)
                j=jac[1:].reshape(-1,self.engine.z_dim);res=(shape[1:]-target[1:]).ravel()
                delta=np.linalg.solve(j.T@j+np.eye(self.engine.z_dim)/.08**2,-j.T@res-(z-prior)/.08**2)
                delta*=min(1.,.12/max(abs(delta).max(),1e-12))
                candidate=project(z+delta)
                if cost(candidate)>=cost(z)-1e-9:break
                z=candidate
            self.at=timestamp;self.state=z;self.action=u;self.history.clear();self._snapshot();self.version+=1
            error=float(np.linalg.norm(self.engine.observe(z,u)[1:]-target[1:],axis=1).mean())
            self.record('full_shape_state_fit',mean_mm=error,state=z,prior=prior,matrix=self.matrix)
            return error

    def calibrate_full_shape(self,curve,timestamp,cancel=None,max_nfev=60):
        """Fit camera similarity and bounded memory together before readiness checks.

        The state prior prevents claiming unique physical memory from one image.
        Residual acceptance is relative to arm length, independent of camera scale.
        """
        from scipy.optimize import least_squares
        target=resample_curve(curve,self.engine.n_nodes)
        length=float(np.linalg.norm(np.diff(target,axis=0),axis=1).sum())
        if length<20:raise ValueError('草稿太小，无法可靠标定；请检查完整中心线')
        with self.lock:
            if self.pending or self.fault:raise ValueError(self.fault or '等待压力 ACK')
            prior,u=self.state_at(timestamp);version=self.version
            prior=prior.copy();u=u.copy()
        matrix,before=similarity_alignment(self.engine.observe(prior,u),target)
        scale0=float(np.linalg.norm(matrix[:2,0]));theta0=float(np.arctan2(matrix[1,0],matrix[0,0]))
        drive,_=self.engine.drive(u);low,_=self.engine.drive(self.bounds.lower);high,_=self.engine.drive(self.bounds.upper)
        lo=np.r_[(drive[:,None]-self.engine.thresholds).ravel(),np.repeat(np.minimum(low,0),self.engine.n_maxwell)]
        hi=np.r_[(drive[:,None]+self.engine.thresholds).ravel(),np.repeat(np.maximum(high,0),self.engine.n_maxwell)]
        # Fixed play operators remain fixed, rather than giving the solver equal bounds.
        active=np.flatnonzero(hi-lo>1e-10);prior=np.clip(prior,lo,hi)
        def evaluate(v,derivatives=False):
            if cancel is not None and cancel.is_set():raise ValueError('部署标定已取消')
            z=prior.copy();z[active]=v[:-2];theta,logscale=v[-2:];scale=np.exp(logscale)
            rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
            if derivatives:shape,jac,_=self.engine.observe(z,u,derivatives=True)
            else:shape=self.engine.observe(z,u)
            relative=shape-self.engine.base
            pixels=scale*relative@rotation.T+target[0]
            residual=np.r_[((pixels[1:]-target[1:])/scale0).ravel(),(z[active]-prior[active])/.08]
            if not derivatives:return residual
            jstate=scale*np.einsum('ij,njz->niz',rotation,jac[:,:,active])[1:].reshape(-1,len(active))/scale0
            rotated=relative@rotation.T
            jtheta=scale*np.column_stack([-rotated[:,1],rotated[:,0]])[1:].ravel()/scale0
            jscale=(scale*rotated)[1:].ravel()/scale0
            return np.vstack([np.column_stack([jstate,jtheta,jscale]),np.column_stack([np.eye(len(active))/.08,np.zeros((len(active),2))])])
        start=np.r_[prior[active],theta0,np.log(scale0)]
        result=least_squares(evaluate,start,jac=lambda v:evaluate(v,True),
                             bounds=(np.r_[lo[active],theta0-np.pi,np.log(scale0/2)],np.r_[hi[active],theta0+np.pi,np.log(scale0*2)]),max_nfev=max_nfev)
        z=prior.copy();z[active]=result.x[:-2];theta,logscale=result.x[-2:];scale=float(np.exp(logscale))
        matrix=np.eye(3);matrix[:2,:2]=scale*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        matrix[:2,2]=target[0]-matrix[:2,:2]@self.engine.base
        pixels=transform(self.engine.observe(z,u),matrix)
        error=float(np.sqrt(np.mean(np.sum((pixels-target)**2,axis=1))))
        info=dict(rms_px=error,rms_mm=error/scale,rms_fraction=error/length,scale_px_per_mm=scale,
                  angle_deg=float(np.degrees(theta)),base_px=target[0].tolist(),initial_rms_px=before,nfev=result.nfev,
                  accepted=error/length<=.03,method='joint_similarity_bounded_state_prior')
        with self.lock:
            if self.version!=version or self.pending or self.fault:raise ValueError('标定期间压力或状态变化，请重新提取')
            if cancel is not None and cancel.is_set():raise ValueError('部署标定已取消')
            self.record('joint_initial_calibration',curve=target,matrix=matrix,state=z,diagnostics=info)
            if not info['accepted']:raise ValueError(f'状态拟合后形状 RMS 仍为臂长的 {error/length:.1%}（上限 3%）；请检查完整草稿、BASE/TIP、腔道映射或姿态')
            self.matrix=matrix;self.at=timestamp;self.state=z;self.action=u;self.history.clear();self._snapshot();self.version+=1
            self.alignment_confirmed=False;self.ready=False
        return info

    def plan_to_tolerance(self, goal, tolerance=1., max_node=3., max_horizon=None, budget_s=15., cancel=None, iterations=24, shooting_nfev=30, horizon_step=20, node_indices=None, target_matrix=None, target_samples=None, snapshot=None):
        started=self.clock()
        ids=target_indices(self.engine.n_nodes,node_indices)
        if not self.ready:raise ValueError('模型尚未通过部署预热')
        if any(isinstance(v,bool) or int(v)!=v or not 1<=v<=limit for v,limit in ((iterations,100),(shooting_nfev,200),(horizon_step,80))):raise ValueError('规划迭代/评估次数和搜索步长超出范围')
        if not np.isfinite([tolerance,max_node,budget_s]).all() or min(tolerance,max_node,budget_s)<=0:raise ValueError('规划容限和预算必须为有限正数')
        if max_horizon is not None and not 2<=max_horizon<=self.meta['max_horizon']:raise ValueError('搜索长度超出模型包上限')
        with self.lock:
            if snapshot is None:
                z,u=self.state_at(self.clock());snapshot=(z.copy(),u.copy(),self.version)
            else:z,u,_=snapshot
        if target_matrix is not None:target_matrix,target_samples=validate_projection(target_matrix,target_samples,self.engine.n_nodes,ids)
        def residual_shape(shape):
            return shape[ids]-goal[ids] if target_matrix is None else target_matrix@shape-target_samples
        def distances(shape):return np.linalg.norm(residual_shape(shape),axis=1)
        end=started+budget_s;best=None;seed=None;attempts=[]
        maximum=int(max_horizon or self.meta['max_horizon'])
        horizons=sorted(set([min(10,maximum),maximum]+list(range(horizon_step,maximum+1,horizon_step))))
        horizons=[h for h in horizons if h>=2]
        config=dict(tolerance=tolerance,max_node=max_node,max_horizon=maximum,budget_s=budget_s,iterations=iterations,shooting_nfev=shooting_nfev,horizon_step=horizon_step,node_indices=ids.tolist())
        for h in horizons:
            if cancel is not None and cancel.is_set():raise ValueError('规划已取消')
            if attempts and self.clock()>end:break
            attempt_start=self.clock()
            candidate=self.plan(goal,h,snapshot=snapshot,seed_actions=seed,deadline=end,cancel=cancel,iterations=iterations,node_indices=ids,target_matrix=target_matrix,target_samples=target_samples)
            b_done=self.clock();shooting_start=b_done
            prediction=self.engine.rollout(z,candidate['actions'])
            # Bounded terminal shooting uses the same selected nodes as B.
            # For partial targets keep the requested terminal geometry in the
            # feedback reference; free nodes never become tracking constraints.
            if distances(prediction[-1]).mean()>tolerance and self.clock()<end:
                from scipy.optimize import least_squares
                active=np.flatnonzero(self.bounds.upper-self.bounds.lower>1e-8)
                def actions_for(parameters):
                    target=self.bounds.lower.copy();target[active]=parameters
                    return self.bounds.project(np.tile(target,(h,1)),u)
                def residual(parameters):
                    if cancel is not None and cancel.is_set():raise ValueError('规划已取消')
                    if self.clock()>end:raise TimeoutError('initial planning budget')
                    return residual_shape(self.engine.rollout(z,actions_for(parameters))[-1]).ravel()
                if len(active):
                    try:
                        seed_pressure=np.clip(candidate['actions'][-1,active],self.bounds.lower[active]+1e-7,self.bounds.upper[active]-1e-7)
                        solved=least_squares(residual,seed_pressure,bounds=(self.bounds.lower[active],self.bounds.upper[active]),max_nfev=shooting_nfev,ftol=1e-5,xtol=1e-5,gtol=1e-5)
                        actions=actions_for(solved.x);shooting=self.engine.rollout(z,actions)
                        if distances(shooting[-1]).mean()<distances(prediction[-1]).mean():
                            candidate['actions']=actions;prediction=shooting
                            if node_indices is None and target_matrix is None:candidate['reference']=shooting.copy()
                    except TimeoutError:pass
            errors=distances(prediction[-1])
            candidate.update(prediction=prediction,mean_error=float(errors.mean()),max_error=float(errors.max()),tolerance=float(tolerance),max_node=float(max_node))
            candidate['qualified']=candidate['mean_error']<=tolerance and candidate['max_error']<=max_node
            attempts.append(dict(horizon=h,mean_mm=candidate['mean_error'],max_mm=candidate['max_error'],b_ms=(b_done-attempt_start)*1000,shooting_and_rollout_ms=(self.clock()-shooting_start)*1000,total_ms=(self.clock()-attempt_start)*1000,b_iterations=len(candidate['trace'])))
            if best is None or candidate['mean_error']<best['mean_error']:best=candidate
            seed=candidate['actions']
            if candidate['qualified']:best=candidate;break
        best['attempts']=attempts
        best.update(planning_ms=(self.clock()-started)*1000,planning_config=config,planning_exit='qualified' if best['qualified'] else ('budget_exhausted' if self.clock()>=end else 'search_exhausted'))
        self.record('horizon_search',attempts=attempts,qualified=best['qualified'],planning_ms=best['planning_ms'],config=config,exit_reason=best['planning_exit'])
        return best

    def plan_any_segment(self, model_curve, tolerance=1., max_node=3., max_horizon=None,
                         budget_s=15., cancel=None, iterations=24, shooting_nfev=30, horizon_step=20):
        """Budgeted search of material intervals and both drawing directions.

        Every candidate fits 32 samples along the whole curve. The chosen
        correspondence is fixed in the returned plan for preview and feedback.
        """
        if not self.ready:raise ValueError('模型尚未通过部署预热')
        if not np.isfinite([budget_s,tolerance,max_node]).all() or min(budget_s,tolerance,max_node)<=0:
            raise ValueError('规划容限和预算必须为有限正数')
        if any(isinstance(v,bool) or int(v)!=v or not 1<=v<=limit for v,limit in ((iterations,100),(shooting_nfev,200),(horizon_step,80))):
            raise ValueError('规划迭代参数无效')
        curve=resample_curve(model_curve,32)
        started=self.clock();end=started+budget_s;screen_end=started+.55*budget_s
        with self.lock:
            z,u=self.state_at(started);snapshot=(z.copy(),u.copy(),self.version)
        current=self.engine.observe(z,u)
        length=np.linalg.norm(np.diff(curve,axis=0),axis=1).sum()
        candidates=[]
        for first in range(1,self.engine.n_nodes-1):
            for last in range(first+1,self.engine.n_nodes):
                ids=np.arange(first,last+1);matrix=segment_projection(self.engine.n_nodes,ids)
                sampled=matrix@current
                span=np.linalg.norm(np.diff(current[ids],axis=0),axis=1).sum()
                for reverse in (False,True):
                    samples=curve[::-1].copy() if reverse else curve.copy()
                    rank=float(np.linalg.norm(sampled-samples,axis=1).mean()+abs(span-length))
                    goal=current.copy();goal[ids]=resample_curve(samples,len(ids))
                    candidates.append((rank,ids,matrix,samples,goal,reverse))
        if not candidates:raise ValueError('模型活动节点不足以匹配一段曲线')
        candidates.sort(key=lambda c:c[0]);attempts=[];best=None
        settings=dict(tolerance=tolerance,max_node=max_node,max_horizon=max_horizon,cancel=cancel,horizon_step=horizon_step,snapshot=snapshot)
        for _,ids,matrix,samples,goal,reverse in candidates:
            if cancel is not None and cancel.is_set():raise ValueError('规划已取消')
            remaining=screen_end-self.clock()
            if best is not None and remaining<=0:break
            plan=self.plan_to_tolerance(goal,**settings,budget_s=max(.001,min(.35,remaining)),
                iterations=min(iterations,2),shooting_nfev=min(shooting_nfev,8),node_indices=ids,target_matrix=matrix,target_samples=samples)
            plan['matching_reversed']=reverse
            attempts.append(dict(node_indices=ids.tolist(),reversed=reverse,mean_mm=plan['mean_error'],max_mm=plan['max_error'],qualified=plan['qualified'],planning_ms=plan['planning_ms']))
            if best is None or plan['mean_error']<best['mean_error']:best=plan
            if plan['qualified']:best=plan;break
        if not best['qualified'] and end-self.clock()>.01:
            refined=self.plan_to_tolerance(best['goal'],**settings,budget_s=end-self.clock(),iterations=iterations,
                shooting_nfev=shooting_nfev,node_indices=best['node_indices'],target_matrix=best['target_matrix'],target_samples=best['target_samples'])
            if refined['qualified'] or refined['mean_error']<best['mean_error']:
                refined['matching_reversed']=best['matching_reversed'];best=refined
        best.update(matching_mode='any_segment',goal_curve=curve,matching_attempts=attempts,
                    matching_candidates_total=len(candidates),planning_ms=(self.clock()-started)*1000)
        best['planning_config'].update(matching_mode='any_segment',budget_s=budget_s,iterations=iterations,shooting_nfev=shooting_nfev)
        best['planning_exit']='qualified' if best['qualified'] else ('budget_exhausted' if self.clock()>=end else 'search_exhausted')
        self.record('segment_matching',node_indices=best['node_indices'],reversed=best['matching_reversed'],
                    attempts=attempts,candidates_total=len(candidates),qualified=best['qualified'],planning_ms=best['planning_ms'])
        return best

    def feedback_snapshot(self):
        """Independent mutable state; the worker never owns the live runtime lock."""
        with self.lock:
            snapshot=copy.copy(self)
            snapshot.__dict__.pop('_feedback_job',None)
            snapshot.lock=threading.RLock()
            snapshot.state=self.state.copy();snapshot.action=self.action.copy()
            snapshot.history=deque([(t,z.copy(),u.copy()) for t,z,u in self.history],maxlen=4096)
            snapshot.pending=set(self.pending);snapshot.matrix=None if self.matrix is None else self.matrix.copy()
            snapshot.bounds=copy.deepcopy(self.bounds)
            snapshot.record=lambda *args,**kwargs:None
            return snapshot,self.version

    def commit_feedback(self,snapshot,version,deadline):
        with self.lock:
            if self.clock()>=deadline:return False,'deadline_expired'
            if self.version!=version or self.pending or self.fault:return False,'snapshot_changed'
            for name in ('at','state','action','history','last_frame','version'):
                setattr(self,name,getattr(snapshot,name))
            return True,'committed'

    def feedback(self,image,timestamp,old,reference):
        started=self.clock()
        with self.lock:
            if self.pending or self.fault:raise ValueError(self.fault or '存在未确认指令')
            if timestamp<=self.last_frame or started-timestamp>.3 or timestamp>started+.01:
                return old.copy(),dict(accepted=False,reason='stale_or_duplicate_frame')
            z,u=self.state_at(timestamp)
            predicted=transform(self.engine.observe(z,u),self.matrix)
            radius=float(self.meta['radius_mm']*np.linalg.norm(self.matrix[:2,0]))
            detection_image=255-image if self.edge_polarity=='dark' else image
            evidence=extract_edges_vectorized(detection_image,predicted,radius=radius)
            eligible=[i for i in range(1,len(predicted)-2) if np.linalg.norm(predicted[i+1]-predicted[i])>=2]
            present=set(evidence.segments.tolist())
            missing=[i for i in eligible if i not in present]
            coverage=(len(eligible)-len(missing))/max(1,len(eligible))
            visibility=dict(coverage=coverage,missing_segments=missing,
                            status='证据不足' if not len(evidence.pixels) else ('局部证据缺失（可能遮挡）' if missing else '侧边缘覆盖完整'),
                            occlusion_confirmed=False)
            visible_target_error=None
            if getattr(self,'target_shape',None) is not None and len(evidence.pixels):
                target_px=transform(self.target_shape,self.matrix)
                selected=target_indices(self.engine.n_nodes,getattr(self,'target_node_indices',None))
                valid=np.isin(evidence.segments,selected)&np.isin(evidence.segments+1,selected)
                ids=evidence.segments[valid];pixels=evidence.pixels[valid]
                a=target_px[ids];v=target_px[ids+1]-a
                fraction=np.clip(np.sum((pixels-a)*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-8),0,1)
                residual=np.abs(np.linalg.norm(a+fraction[:,None]*v-pixels,axis=1)-radius)
                if len(residual):visible_target_error=float(residual.mean()/np.linalg.norm(self.matrix[:2,0]))
            edges_done=self.clock()
            corrected,observer=edge_state_update(self.engine,z,u,evidence,self.matrix,radius,self.bounds)
            observer_done=self.clock()
            # Replay acknowledged command changes after image time, including a
            # new command already issued while the image was in transit.
            at=timestamp;last=u
            subsequent=[row for row in self.history if row[0]>timestamp]
            self.history=deque([row for row in self.history if row[0]<timestamp],maxlen=4096)
            self.history.append((at,corrected.copy(),last.copy()))
            for t,_,action in subsequent:
                corrected=advance(self.engine,corrected,last,t-at,self.dt)
                corrected=advance(self.engine,corrected,action,0,self.dt)
                at,last=t,action
                self.history.append((at,corrected.copy(),last.copy()))
            self.at,self.state,self.action=at,corrected,last
            self.last_frame=timestamp;self.version+=1
            boundary=self.clock()
            state=advance(self.engine,corrected,last,boundary-at,self.dt)
            control={'accepted':False,'reason':'no_evidence_or_suffix'};new=old.copy()
            if len(old) and observer['count']:
                new,control=fast_suffix_b(self.engine,state,old,last,reference,self.bounds,node_indices=getattr(self,'control_node_indices',None),target_matrix=getattr(self,'control_target_matrix',None))
            return new,dict(edge_ms=(edges_done-started)*1000,observer_ms=(observer_done-edges_done)*1000,visible_target_error_mm=visible_target_error,prediction_px=transform(self.engine.observe(state,last),self.matrix),observer=observer,control=control,visibility=visibility,edges=len(evidence.pixels),frame_age_ms=(started-timestamp)*1000,compute_ms=(self.clock()-started)*1000)

    def close(self):
        self.log.close()
