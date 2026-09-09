"""Serial visual-feedback executor over the existing ACK transport.

Missed slots are skipped, never sent in a burst. Planning and setup are frozen
by the GUI while this worker owns the actuator. Real camera time remains host
receipt time, not certified exposure time.
"""
from __future__ import annotations
import threading
import time
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from .deadline_feedback import deadline_feedback


class HereditaryExecutor:
    def __init__(self,runtime,transport,frame_provider,*,callback=None,timeout=.5,clock=time.monotonic,
                 image_transform=None,camera_provider=None,evaluation_provider=None,metadata=None,selected_camera=0,max_missing=3,max_skipped=10,settle_s=0.):
        self.runtime,self.transport,self.frame_provider=runtime,transport,frame_provider
        self.callback,self.timeout,self.clock=callback,timeout,clock
        if not 1<=max_missing<=10:raise ValueError('无反馈上限必须在 1..10')
        self.image_transform=image_transform;self.camera_provider=camera_provider;self.evaluation_provider=evaluation_provider
        self.metadata=metadata or {};self.selected_camera=selected_camera;self.max_missing=max_missing;self.archive=None
        if not 1<=max_skipped<=100:raise ValueError('连续跳过上限必须在 1..100')
        if not np.isfinite(settle_s) or not 0<=settle_s<runtime.dt:raise ValueError('额外等图时间必须小于控制周期')
        self.settle_s=settle_s
        self.max_skipped=max_skipped
        self.abort_event=threading.Event();self.hold_requested=False

    def abort(self):self.hold_requested=False;self.abort_event.set()

    def hold(self):self.hold_requested=True;self.abort_event.set()

    def zero(self):
        last=None
        for _ in range(3):
            last=self.transport.zero(self.timeout)
            if self.archive:self.archive.command('zero',last)
            if last.status=='ack':
                self.runtime.acknowledge(last)
                self.runtime.set_phase('stopped',reason='zero_ack')
                return last
        self.runtime.fault='归零未获得 ACK，需要人工检查'
        raise RuntimeError(self.runtime.fault)

    def execute(self,plan):
        r=self.runtime
        if plan['version']!=r.version or r.pending or r.fault:
            raise ValueError('计划已过期或动作历史无效，请重新规划')
        old=np.array(plan['actions'],copy=True);reference=np.array(plan['reference'],copy=True)
        if not r.bounds.valid(old,r.action):raise ValueError('计划压力接续不合法')
        if 'qualified' in plan:
            if not plan['qualified'] or not r.ready:raise ValueError('计划未达标或部署未就绪')
            with r.lock:state,current=r.state_at(self.clock())
            prediction=r.engine.rollout(state,old)
            distances=np.linalg.norm(prediction[-1,1:]-plan['goal'][1:],axis=1)
            drift=float(np.max(np.linalg.norm(prediction-plan['prediction'],axis=2)))
            if distances.mean()>plan['tolerance'] or distances.max()>plan['max_node'] or drift>max(.5,plan['tolerance']/2):
                raise ValueError('保持期间状态发生变化，原预览已失效；请重新规划并预览')
        started=self.clock();receipts=[];misses=0;skipped=0
        execution_id=uuid.uuid4().hex
        frames=r.run_dir/'executions'/execution_id/'frames'
        frames.mkdir(parents=True,exist_ok=False)
        from .experiment_archive import ExperimentArchive
        self.last_execution_dir=frames.parent
        self.archive=ExperimentArchive(frames.parent,started,dict(self.metadata,execution_id=execution_id,deployment_id=r.deployment_id,max_missing=self.max_missing,max_skipped=self.max_skipped,feedback_reserve_ms=3,control_dt=r.dt,settle_s=self.settle_s,selected_camera=self.selected_camera),self.evaluation_provider)
        outcome='failed';timing=None
        r.set_phase('control',reason='execute_pressed')
        r.record('execute_begin',execution_id=execution_id,deployment_id=r.deployment_id,goal=plan['goal'],plan=old,reference=reference,planning_ms=plan.get('planning_ms'),planning_config=plan.get('planning_config'))
        np.savez_compressed(frames.parent/'initial_plan.npz',actions_model=old,actions_kpa=r.mapping.expand(old),reference_mm=reference,state=plan.get('state',r.state),previous_model=plan.get('previous',r.action),goal_mm=plan['goal'],version=plan['version'],execution_state=r.state,execution_action=r.action,execution_state_time=r.at,camera_matrix=r.matrix,expansion6=r.mapping.expansion,action_unit_to_kpa=r.mapping.scale,dt=r.dt)
        try:
            with threadpool_limits(1):
                for k in range(len(old)):
                    if self.abort_event.is_set():raise RuntimeError('operator_abort')
                    # Enforce the minimum command interval without rounding a
                    # completed feedback cycle up to the next whole grid slot.
                    # Slow cycles dispatch when ready; no queued catch-up burst.
                    now=self.clock()
                    deadline=started if not receipts else receipts[-1].t_command+r.dt
                    slot=max(k,int(max(0.,now-started)/r.dt))
                    if self.abort_event.wait(max(0,deadline-now)):raise RuntimeError('operator_abort')
                    with r.lock:
                        if r.fault or r.pending:raise RuntimeError(r.fault or '外部指令未确认')
                        if not r.bounds.valid(old[k:],r.action):raise ValueError('剩余压力接续不合法')
                        action=r.mapping.expand(old[k])
                    cycle_start=self.clock();before=old[k+1:].copy();version_before=r.version
                    timing=None
                    try:
                        timing=dict(step=k,execution_id=execution_id,revision_status='command_failed',state_committed=False,version_before=version_before,remaining_before_model=before)
                        receipt=self.transport.send(action,(1,2),self.timeout)
                        next_deadline=receipt.t_command+r.dt
                        timing.update(command_id=receipt.command_id,t_command=receipt.t_command,t_ack=receipt.t_ack,feedback_deadline=next_deadline,command_jitter_ms=(receipt.t_command-deadline)*1000,ack_ms=(receipt.t_ack-receipt.t_command)*1000 if receipt.t_ack is not None else None,command_interval_ms=(receipt.t_command-receipts[-1].t_command)*1000 if receipts else None)
                        receipts.append(receipt);self.archive.command(k,receipt);r.acknowledge(receipt)
                        if receipt.status!='ack':raise RuntimeError(f'command {receipt.command_id}: {receipt.status}')
                        # Valve slew uses actual command intervals, so applied6 may
                        # differ from the requested model-dt command. Make the tail
                        # feasible from that ACK before observation or optimization;
                        # even a missed image must leave a legal fallback tail.
                        with r.lock:
                            if len(old[k+1:]):
                                repaired=r.bounds.project(old[k+1:],r.action)
                                shift=float(np.max(abs(repaired-old[k+1:])))
                                if shift>1e-10:
                                    r.record('applied_suffix_rebase',execution_id=execution_id,step=k,
                                             requested6=receipt.requested6,applied6=receipt.applied6,
                                             max_change_model_units=shift,remaining_actions=repaired)
                                old[k+1:]=repaired
                        # The image must be newly received after this command and a
                        # optional settling interval. Never use a cached pre-ACK frame.
                        earliest=max(receipt.t_command+self.settle_s,receipt.t_ack or receipt.t_command)
                        wait_start=self.clock();wait_end=next_deadline-.003;frame=None
                        timing['revision_status']='frame_missing'
                        while self.clock()<wait_end:
                            if self.abort_event.wait(.005):raise RuntimeError('operator_abort')
                            frame=self.frame_provider()
                            if frame is not None and frame[1]>=earliest and frame[1]>r.last_frame:break
                            frame=None
                        timing['frame_wait_ms']=(self.clock()-wait_start)*1000
                        timing['fallback_model']=old[k+1:].copy()
                        if frame is None:
                            if self.camera_provider:
                                for camera,other in self.camera_provider().items():self.archive.enqueue_image(k,camera,other,receipt)
                            misses+=1;r.record('feedback_missing',step=k,execution_id=execution_id,consecutive_missing=misses)
                            if self.callback:self.callback(dict(edges=0,consecutive_missing=misses,visibility=dict(status='没有新图像；仅模型预测')))
                            if misses>=self.max_missing:raise RuntimeError('连续三次没有新图像，停止并归零' if self.max_missing==3 else f'连续 {misses} 次没有新图像，停止并归零')
                            continue
                        # Full old suffix is consumed exactly once at the command boundary.
                        preprocess_start=self.clock()
                        feedback,occlusion=self.image_transform(frame[0]) if self.image_transform else (frame[0],{'enabled':False})
                        timing['preprocess_ms']=(self.clock()-preprocess_start)*1000
                        save_start=self.clock()
                        # Queue owned copies, including the last blind/failing image.
                        self.archive.enqueue_image(k,self.selected_camera,frame,receipt,feedback,occlusion)
                        if self.camera_provider:
                            for camera,other in self.camera_provider().items():
                                if camera!=self.selected_camera:self.archive.enqueue_image(k,camera,other,receipt)
                        timing['archive_enqueue_ms']=(self.clock()-save_start)*1000
                        # Reserve 3 ms for state commit, audit and dispatch preparation.
                        feedback_deadline=next_deadline-.003
                        timing['feedback_budget_ms']=max(0.,(feedback_deadline-self.clock())*1000)
                        timing['revision_status']='feedback_error'
                        candidate,info=deadline_feedback(r,feedback,frame[1],old[k+1:],reference[k+1:],feedback_deadline,self.abort_event,frames.parent/'feedback_jobs'/f'{k:05d}.json')
                        info['software_occlusion']=occlusion
                        if info.get('revision_status')=='committed' and info.get('observer',{}).get('count',0)==0:
                            misses+=1

                        elif info.get('state_committed'):
                            misses=0
                        old[k+1:]=candidate
                        skipped=0 if info.get('state_committed') else skipped+1
                        info['consecutive_skipped']=skipped
                        info.update(consecutive_missing=misses,observation_mode='image_supported' if info.get('state_committed') and not misses else 'model_only',step=k,slot=slot,execution_id=execution_id,frame_timestamp=frame[1],command_jitter_ms=(receipt.t_command-deadline)*1000)
                        timing.update(info,control_ms=info.get('control',{}).get('time_ms'))
                        r.record('feedback',**info,state=r.state,remaining_actions=old[k+1:])
                        if skipped>=self.max_skipped:raise RuntimeError(f'连续 {skipped} 次未能及时提交反馈，停止并归零')
                        if misses>=self.max_missing:raise RuntimeError('连续三次没有有效图像边缘，停止并归零' if self.max_missing==3 else f'连续 {misses} 次没有有效图像边缘，停止并归零')
                    finally:
                        if timing is not None:
                            timing.update(version_after=r.version,state_after=r.state.copy(),remaining_applied_model=old[k+1:].copy(),remaining_applied_kpa=r.mapping.expand(old[k+1:]),suffix_changed=not np.array_equal(before,old[k+1:]),cycle_ms=(self.clock()-cycle_start)*1000)
                            self.archive.step(timing)
                            if self.callback:self.callback(timing)
            if misses or skipped:r.ready=False
            r.set_phase('final_hold',reason='plan_completed')
            r.record('execute_completed',execution_id=execution_id,commands=len(receipts),final_pressure=r.mapping.expand(r.action))
            outcome='completed_unobserved' if misses or skipped else 'completed';return receipts
        except Exception as error:
            if self.hold_requested and str(error)=='operator_abort':
                r.set_phase('final_hold',reason='operator_hold');r.record('execute_held',execution_id=execution_id,commands=len(receipts));outcome='held';return receipts
            r.ready=False;r.record('execute_failed',execution_id=execution_id,error=str(error))
            try:self.zero()
            finally:r.record('execute_stopped',fault=r.fault)
            raise
        finally:
            archive,self.archive=self.archive,None
            if archive:archive.close(outcome)
