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
from ..runtime.shape_target import target_indices, target_distances


class HereditaryExecutor:
    def __init__(self,runtime,transport,frame_provider,*,callback=None,timeout=.5,clock=time.monotonic,
                 image_transform=None,camera_provider=None,evaluation_provider=None,metadata=None,selected_camera=0,max_missing=3,max_skipped=10,settle_s=0.,use_correction=True,feedback_interval_steps=1):
        self.runtime,self.transport,self.frame_provider=runtime,transport,frame_provider
        self.callback,self.timeout,self.clock=callback,timeout,clock
        if not 1<=max_missing<=10:raise ValueError('无反馈上限必须在 1..10')
        self.image_transform=image_transform;self.camera_provider=camera_provider;self.evaluation_provider=evaluation_provider
        self.metadata=metadata or {};self.selected_camera=selected_camera;self.max_missing=max_missing;self.archive=None
        self.use_correction=bool(use_correction)
        if isinstance(feedback_interval_steps,(bool,np.bool_)) or not np.isfinite(feedback_interval_steps) or int(feedback_interval_steps)!=feedback_interval_steps or not 1<=feedback_interval_steps<=10:
            raise ValueError('矫正间隔必须为 1..10 步整数')
        self.feedback_interval_steps=int(feedback_interval_steps)
        self.control_mode='analytic_b' if self.use_correction else 'open_loop'
        if not 1<=max_skipped<=100:raise ValueError('连续跳过上限必须在 1..100')
        if not np.isfinite(settle_s) or not 0<=settle_s<runtime.dt:raise ValueError('额外等图时间必须小于控制周期')
        self.settle_s=settle_s
        self.max_skipped=max_skipped
        self.abort_event=threading.Event();self.hold_requested=False
        self.completion_assessment=None

    def abort(self):self.hold_requested=False;self.abort_event.set()

    def hold(self):self.hold_requested=True;self.abort_event.set()

    def assess_completion(self,plan,info):
        """Describe the current estimate without certifying physical arrival."""
        r=self.runtime
        with r.lock:
            stamp=self.clock();state,action=r.state_at(stamp)
            shape=r.engine.observe(state,action)
            errors=target_distances(shape,plan['goal'],plan.get('node_indices'),plan.get('target_matrix'),plan.get('target_samples'))
            age=stamp-r.last_frame
        supported=bool(self.use_correction and info.get('state_committed') and info.get('observer',{}).get('count',0)>0 and 0<=age<=.3)
        finite=bool(np.isfinite(errors).all())
        within=bool(finite and 'tolerance' in plan and 'max_node' in plan and errors.mean()<=plan['tolerance'] and errors.max()<=plan['max_node'])
        return dict(estimated_mean_error_mm=float(errors.mean()) if finite else None,
                    estimated_max_error_mm=float(errors.max()) if finite else None,
                    estimate_within_tolerance=within,latest_image_supported=supported,
                    last_frame_age_ms=float(age*1000) if np.isfinite(age) else None,
                    coverage=info.get('visibility',{}).get('coverage'),
                    physical_arrival_verified=False,
                    reason='estimated_within_tolerance' if within else 'finite_plan_exhausted',
                    semantics='Model estimate, possibly corrected by partial image evidence; not measured full-shape arrival. Final pressure is held; no automatic global replan.')

    def zero(self):
        last=None
        for _ in range(3):
            last=self.transport.zero(self.timeout)
            if self.archive:self.archive.command('zero',last)
            if last.status=='ack':
                self.runtime.acknowledge(last)
                self.runtime.ready=False
                self.runtime.set_phase('stopped',reason='zero_ack')
                return last
        self.runtime.fault='归零未获得 ACK，需要人工检查'
        raise RuntimeError(self.runtime.fault)

    def collect_delayed(self,pipeline,step,tail,reference,command_time):
        if not pipeline.due(step):return tail,None
        if self.abort_event.wait(max(0.,command_time-pipeline.commit_reserve_s-self.clock())):
            raise RuntimeError('operator_abort')
        candidate,info=pipeline.take(step,tail,reference,command_time,command_time-.003)
        self.runtime.record('delayed_feedback_application',execution_id=self.last_execution_dir.name,step=step,remaining_before_command_model=tail,
                            remaining_after_command_model=candidate,state=self.runtime.state,state_time=self.runtime.at,**info)
        if pipeline.skipped>=self.max_skipped:raise RuntimeError(f'连续 {pipeline.skipped} 个矫正周期未提交，停止并归零')
        if pipeline.missing>=self.max_missing:raise RuntimeError(f'连续 {pipeline.missing} 个矫正周期没有有效边缘，停止并归零')
        return candidate,info

    def execute(self,plan,*,allow_unqualified=False):
        r=self.runtime
        if plan['version']!=r.version or r.pending or r.fault:
            raise ValueError('计划已过期或动作历史无效，请重新规划')
        ids=target_indices(r.engine.n_nodes,plan.get('node_indices'))
        old=np.array(plan['actions'],copy=True);reference=np.array(plan['reference'],copy=True)
        if not r.bounds.valid(old,r.action):raise ValueError('计划压力接续不合法')
        trial=bool(not plan.get('qualified',True) and allow_unqualified)
        if 'qualified' in plan:
            if not r.ready:raise ValueError('部署未就绪')
            if not plan['qualified'] and not trial:raise ValueError('计划未达标，请检查预览并明确允许试运行')
            with r.lock:state,current=r.state_at(self.clock())
            prediction=r.engine.rollout(state,old)
            distances=target_distances(prediction[-1],plan['goal'],ids,plan.get('target_matrix'),plan.get('target_samples'))
            drift=float(np.max(np.linalg.norm(prediction-plan['prediction'],axis=2)))
            if not np.isfinite(distances).all() or not np.isfinite(drift):raise ValueError('模型预测非有限，不能执行')
            if (not trial and (distances.mean()>plan['tolerance'] or distances.max()>plan['max_node'])) or drift>max(.5,plan['tolerance']/2):
                raise ValueError('保持期间状态发生变化，原预览已失效；请重新规划并预览')
        r.control_target_matrix=None if 'target_matrix' not in plan else plan['target_matrix'].copy()
        r.control_node_indices=ids.copy();r.target_node_indices=ids.copy();r.target_shape=np.array(plan['goal'],copy=True)
        started=self.clock();receipts=[];misses=0;skipped=0
        primary_steps=int(plan.get('primary_steps',len(old)))
        reserve_steps=int(plan.get('reserve_steps',0))
        if primary_steps<1 or reserve_steps<0 or primary_steps+reserve_steps!=len(old) or len(old)>200:
            raise ValueError('主规划与末端余量长度不一致或超过 200 步')
        execution_id=uuid.uuid4().hex
        frames=r.run_dir/'executions'/execution_id/'frames'
        frames.mkdir(parents=True,exist_ok=False)
        from .experiment_archive import ExperimentArchive
        self.last_execution_dir=frames.parent
        self.archive=ExperimentArchive(frames.parent,started,dict(self.metadata,control_mode=self.control_mode,correction_enabled=self.use_correction,unqualified_trial=trial,planned_qualified=plan.get('qualified'),planned_mean_error=plan.get('mean_error'),planned_max_error=plan.get('max_error'),matching_mode=plan.get('matching_mode','fixed'),execution_id=execution_id,deployment_id=r.deployment_id,max_missing=self.max_missing,max_skipped=self.max_skipped,feedback_reserve_ms=3,control_dt=r.dt,settle_s=self.settle_s,selected_camera=self.selected_camera),self.evaluation_provider)
        self.archive.metadata.update(primary_steps=primary_steps,reserve_steps=reserve_steps,total_steps=len(old))
        self.archive.metadata.update(feedback_interval_steps=self.feedback_interval_steps,
                                     correction_period_s=r.dt*self.feedback_interval_steps,
                                     planning_rate_fraction=plan.get('planning_config',{}).get('planning_rate_fraction',1.),
                                     feedback_mode=('delayed' if self.feedback_interval_steps>1 else 'per_step') if self.use_correction else 'disabled')
        pipeline=None
        if self.use_correction and self.feedback_interval_steps>1:
            from .multirate_feedback import MultirateFeedback
            pipeline=MultirateFeedback(r,self.feedback_interval_steps,frames.parent/'feedback_jobs')
        outcome='failed';timing=None
        r.set_phase('control',reason='execute_pressed')
        r.record('execute_begin',control_mode=self.control_mode,correction_enabled=self.use_correction,unqualified_trial=trial,planned_qualified=plan.get('qualified'),planned_mean_error=plan.get('mean_error'),planned_max_error=plan.get('max_error'),execution_id=execution_id,deployment_id=r.deployment_id,goal=plan['goal'],node_indices=ids,plan=old,reference=reference,planning_ms=plan.get('planning_ms'),planning_config=plan.get('planning_config'))
        np.savez_compressed(frames.parent/'initial_plan.npz',actions_model=old,actions_kpa=r.mapping.expand(old),primary_steps=primary_steps,reserve_steps=reserve_steps,reference_mm=reference,state=plan.get('state',r.state),previous_model=plan.get('previous',r.action),goal_mm=plan['goal'],node_indices=ids,unqualified_trial=trial,control_mode=self.control_mode,correction_enabled=self.use_correction,planned_qualified=plan.get('qualified',True),**{key:plan[key] for key in ('target_matrix','target_samples','matching_mode','matching_reversed','goal_curve') if key in plan},version=plan['version'],execution_state=r.state,execution_action=r.action,execution_state_time=r.at,camera_matrix=r.matrix,expansion6=r.mapping.expansion,action_unit_to_kpa=r.mapping.scale,dt=r.dt)
        try:
            with threadpool_limits(1):
                for k in range(len(old)):
                    if self.abort_event.is_set():raise RuntimeError('operator_abort')
                    # Enforce the minimum command interval without rounding a
                    # completed feedback cycle up to the next whole grid slot.
                    # Slow cycles dispatch when ready; no queued catch-up burst.
                    now=self.clock()
                    deadline=started if not receipts else receipts[-1].t_command+r.dt
                    application=None
                    if pipeline is not None:
                        old[k:],application=self.collect_delayed(pipeline,k,old[k:],reference[k:],deadline)
                        now=self.clock()
                    slot=max(k,int(max(0.,now-started)/r.dt))
                    if self.abort_event.wait(max(0,deadline-now)):raise RuntimeError('operator_abort')
                    with r.lock:
                        if r.fault or r.pending:raise RuntimeError(r.fault or '外部指令未确认')
                        if not r.bounds.valid(old[k:],r.action):raise ValueError('剩余压力接续不合法')
                        action=r.mapping.expand(old[k])
                    cycle_start=self.clock();before=old[k+1:].copy();version_before=r.version
                    timing=None
                    try:
                        timing=dict(step=k,plan_phase='reserve' if k>=primary_steps else 'primary',primary_steps=primary_steps,reserve_steps=reserve_steps,control_mode=self.control_mode,execution_id=execution_id,revision_status='command_failed',state_committed=False,version_before=version_before,remaining_before_model=before)
                        if application is not None:timing['feedback_application']=application
                        send_started=self.clock()
                        receipt=self.transport.send(action,(1,2),self.timeout)
                        send_returned=self.clock()
                        next_deadline=receipt.t_command+r.dt
                        timing.update(command_id=receipt.command_id,t_command=receipt.t_command,t_ack=receipt.t_ack,feedback_deadline=next_deadline,command_jitter_ms=(receipt.t_command-deadline)*1000,ack_ms=(receipt.t_ack-receipt.t_command)*1000 if receipt.t_ack is not None else None,command_interval_ms=(receipt.t_command-receipts[-1].t_command)*1000 if receipts else None)
                        timing.update(send_wait_ms=(send_returned-send_started)*1000,
                                      dispatch_wait_ms=max(0.,(receipt.t_command-send_started)*1000),
                                      ack_delivery_ms=max(0.,(send_returned-receipt.t_ack)*1000) if receipt.t_ack is not None else None)
                        receipts.append(receipt);self.archive.command(k,receipt);r.acknowledge(receipt)
                        if receipt.status!='ack':raise RuntimeError(f'command {receipt.command_id}: {receipt.status}')
                        if pipeline is not None:pipeline.acknowledge(k,receipt)
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
                        polls=0
                        while self.clock()<wait_end:
                            if self.abort_event.is_set():raise RuntimeError('operator_abort')
                            frame=self.frame_provider()
                            polls+=1
                            if frame is not None and frame[1]>=earliest and frame[1]>r.last_frame:break
                            frame=None
                            if self.abort_event.wait(min(.005,max(0.,wait_end-self.clock()))):raise RuntimeError('operator_abort')
                        timing['frame_wait_ms']=(self.clock()-wait_start)*1000
                        timing.update(camera_poll_count=polls,feedback_budget_ms=max(0.,(wait_end-self.clock())*1000))
                        timing['fallback_model']=old[k+1:].copy()
                        if frame is None:
                            if pipeline is not None:
                                pipeline.missed_sample(k);skipped=pipeline.skipped
                            if self.camera_provider:
                                for camera,other in self.camera_provider().items():self.archive.enqueue_image(k,camera,other,receipt)
                            misses+=1;r.record('feedback_missing',step=k,execution_id=execution_id,consecutive_missing=misses)
                            timing.update(edges=0,consecutive_missing=misses,control_mode=self.control_mode,
                                          visibility=dict(status='没有 ACK 后的新图像；仅模型预测'))
                            if skipped>=self.max_skipped:raise RuntimeError(f'连续 {skipped} 个矫正周期未提交，停止并归零')
                            if misses>=self.max_missing:raise RuntimeError('连续三次没有新图像，停止并归零' if self.max_missing==3 else f'连续 {misses} 次没有新图像，停止并归零')
                            continue
                        timing.update(frame_age_at_selection_ms=(self.clock()-frame[1])*1000,
                                      frame_after_ack_ms=(frame[1]-receipt.t_ack)*1000 if receipt.t_ack is not None else None)
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
                        if pipeline is not None:
                            launch=pipeline.start(k,receipt,feedback,frame[1],old[k+1:],reference[k+1:],len(old))
                            if launch.get('proposal_path'):self.archive.proposals.append(launch['proposal_path'])
                            info=dict(launch if application is None else application,feedback_launch=launch)
                            candidate=old[k+1:].copy();misses=0
                            # Cyan follows every acknowledged command, even on
                            # steps that intentionally do not run an observer.
                            from ..runtime.hereditary_deployment import transform
                            with r.lock:
                                state,current=r.state_at(self.clock())
                                info['prediction_px']=transform(r.engine.observe(state,current),r.matrix)
                        elif self.use_correction:
                            candidate,info=deadline_feedback(r,feedback,frame[1],old[k+1:],reference[k+1:],feedback_deadline,self.abort_event,frames.parent/'feedback_jobs'/f'{k:05d}.json')
                        else:
                            # Same acquisition/archive path, but no image state
                            # correction, no suffix optimizer and no worker job.
                            candidate=old[k+1:].copy();misses=0;skipped=0
                            with r.lock:
                                z,u=r.state_at(self.clock())
                                predicted=r.engine.observe(z,u)
                            from ..runtime.hereditary_deployment import transform
                            info=dict(revision_status='disabled',state_committed=False,frame_age_ms=(self.clock()-frame[1])*1000,
                                      control=dict(accepted=False,reason='correction_disabled'),
                                      visibility=dict(status='矫正关闭；图像仅记录'),
                                      prediction_px=transform(predicted,r.matrix) if r.matrix is not None else predicted)
                        info['software_occlusion']=occlusion
                        if pipeline is None and info.get('revision_status')=='committed' and info.get('observer',{}).get('count',0)==0:
                            misses+=1

                        elif info.get('state_committed'):
                            misses=0
                        old[k+1:]=candidate
                        skipped=pipeline.skipped if pipeline is not None else (0 if not self.use_correction or info.get('state_committed') else skipped+1)
                        info['consecutive_skipped']=skipped
                        info.update(control_mode=self.control_mode,consecutive_missing=max(misses,pipeline.missing) if pipeline is not None else misses,observation_mode=('record_only' if not self.use_correction else ('image_supported' if info.get('state_committed') and not misses else 'model_only')),step=k,slot=slot,execution_id=execution_id,frame_timestamp=frame[1],command_jitter_ms=(receipt.t_command-deadline)*1000)
                        timing.update(info,control_ms=info.get('control',{}).get('time_ms'))
                        r.record('feedback',**info,state=r.state,remaining_actions=old[k+1:])
                        if skipped>=self.max_skipped:raise RuntimeError(f'连续 {skipped} 次未能及时提交反馈，停止并归零')
                        if misses>=self.max_missing:raise RuntimeError('连续三次没有有效图像边缘，停止并归零' if self.max_missing==3 else f'连续 {misses} 次没有有效图像边缘，停止并归零')
                    finally:
                        if timing is not None:
                            timing.update(version_after=r.version,state_after=r.state.copy(),remaining_applied_model=old[k+1:].copy(),remaining_applied_kpa=r.mapping.expand(old[k+1:]),suffix_changed=not np.array_equal(before,old[k+1:]),cycle_ms=(self.clock()-cycle_start)*1000)
                            self.archive.step(timing)
                            if self.callback:self.callback(timing)
                if pipeline is not None:
                    _,terminal=self.collect_delayed(pipeline,len(old),old[len(old):],reference[len(old):],receipts[-1].t_command+r.dt)
                    if terminal is not None:
                        r.record('terminal_feedback',execution_id=execution_id,**terminal)
                        self.archive.metadata['terminal_feedback_application']={key:terminal.get(key) for key in ('source_step','apply_step','revision_status','state_committed','compute_ms','commit_ms')}
                        if self.callback:self.callback(dict(terminal,step='末压保持',control_mode=self.control_mode))
                    timing=pipeline.last_info or timing
                    skipped=pipeline.skipped;misses=max(misses,pipeline.missing)
            if misses or skipped:r.ready=False
            r.set_phase('final_hold',reason='plan_completed')
            self.completion_assessment=self.assess_completion(plan,timing or {})
            self.archive.metadata['completion_assessment']=self.completion_assessment
            r.record('execute_completed',control_mode=self.control_mode,execution_id=execution_id,commands=len(receipts),primary_steps=primary_steps,reserve_steps=reserve_steps,completion_assessment=self.completion_assessment,final_pressure=r.mapping.expand(r.action))
            outcome='completed_unobserved' if misses or skipped else 'completed';return receipts
        except Exception as error:
            if self.hold_requested and str(error)=='operator_abort':
                r.set_phase('final_hold',reason='operator_hold');r.record('execute_held',execution_id=execution_id,commands=len(receipts));outcome='held';return receipts
            r.ready=False;r.record('execute_failed',execution_id=execution_id,error=str(error))
            try:self.zero()
            finally:r.record('execute_stopped',fault=r.fault)
            raise
        finally:
            if pipeline is not None:pipeline.cancel()
            archive,self.archive=self.archive,None
            if archive:archive.close(outcome)
