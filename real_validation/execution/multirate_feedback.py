"""One speculative correction, a frozen command prefix, and a future commit.

The worker never modifies live state. The executor keeps sending the old plan;
at the chosen boundary it replays real ACKs and revalidates only the unissued
suffix. Waiting for a scheduled command is not waiting for optimizer completion.
"""
from collections import deque
import time
import numpy as np

from .deadline_feedback import FeedbackJob
from ..runtime.hereditary_deployment import advance, transform
from ..runtime.hereditary_math import fast_suffix_b
from ..runtime.shape_target import target_indices


class DelayedFeedbackJob(FeedbackJob):
    def __init__(self,runtime,image,stamp,tail,reference,path,source_step,apply_step,command_time):
        self.source_step=source_step;self.apply_step=apply_step;self.stamp=stamp
        self.command_time=command_time;self.frozen_count=apply_step-source_step-1
        self.target_time=command_time+(apply_step-source_step)*runtime.dt
        self.deadline=self.target_time-.003
        self.nominal=np.array(tail[self.frozen_count:],copy=True)
        self.receipts=[];self.consumed=False
        super().__init__(runtime,image,stamp,tail,reference,path)

    def _compute(self,image,stamp,tail,reference):
        r=self.snapshot;started=time.perf_counter()
        _,info=r.feedback(image,stamp,np.empty((0,r.engine.channels)),np.empty((0,r.engine.n_nodes,2)))
        info.update(source_step=self.source_step,apply_step=self.apply_step,
                    source_frame_timestamp=self.stamp,frozen_prefix_steps=self.frozen_count,
                    frozen_prefix_actions_model=tail[:self.frozen_count].copy(),scheduled_apply_time=self.target_time)
        if not info.get('observer',{}).get('count',0):return self.nominal.copy(),info
        z=r.state.copy();u=r.action.copy();at=r.at
        # Commands before apply_step are immutable in this proposal. Actual
        # pressures/times will be replayed again by the executor before commit.
        for j,action in enumerate(tail[:self.frozen_count]):
            stamp=self.command_time+(j+1)*r.dt
            z=advance(r.engine,z,u,stamp-at,r.dt)
            z=advance(r.engine,z,action,0,r.dt);at=stamp;u=action
        z=advance(r.engine,z,u,self.target_time-at,r.dt)
        candidate=self.nominal.copy()
        if len(candidate):
            candidate,info['control']=fast_suffix_b(r.engine,z,candidate,u,reference[self.frozen_count:],r.bounds,
                node_indices=getattr(r,'control_node_indices',None),target_matrix=getattr(r,'control_target_matrix',None))
        info['compute_ms']=(time.perf_counter()-started)*1000
        return candidate,info


class MultirateFeedback:
    # Reserve a short interval for actual-ACK replay and nonlinear acceptance.
    # This remains soft real time; unusually slow validation is discarded.
    commit_reserve_s=.012

    def __init__(self,runtime,interval,folder):
        self.runtime=runtime;self.interval=interval;self.folder=folder
        self.job=None;self.skipped=0;self.missing=0;self.last_info=None;self.last_failure_step=None

    def start(self,step,receipt,image,stamp,tail,reference,total_steps):
        r=self.runtime
        if step%self.interval:
            return dict(revision_status='feedback_interval',state_committed=False,feedback_wait_ms=0.)
        previous=getattr(r,'_feedback_job',None)
        if previous is not None and previous.thread.is_alive():
            if self.last_failure_step!=step:self.skipped+=1
            self.last_failure_step=step
            return dict(revision_status='worker_busy',state_committed=False,feedback_wait_ms=0.)
        apply_step=min(step+self.interval,total_steps)
        path=self.folder/f'{step:05d}.json'
        self.job=DelayedFeedbackJob(r,image.copy(),stamp,tail,reference,path,step,apply_step,receipt.t_command)
        r._feedback_job=self.job
        return dict(revision_status='feedback_started',state_committed=False,feedback_wait_ms=0.,
                    feedback_budget_ms=max(0.,(self.job.deadline-r.clock())*1000),
                    source_step=step,apply_step=apply_step,proposal_path=path.name)

    def acknowledge(self,step,receipt):
        if self.job is not None and not self.job.consumed and step>self.job.source_step:
            self.job.receipts.append(receipt)

    def missed_sample(self,step):
        if step%self.interval==0:
            if self.last_failure_step!=step:self.skipped+=1
            self.last_failure_step=step
            self.last_info=dict(revision_status='frame_missing',state_committed=False,source_step=step)

    def due(self,step):
        return self.job is not None and not self.job.consumed and self.job.apply_step==step

    def take(self,step,tail,reference,command_time,deadline):
        """Nonblocking collection. No edits if late, stale, invalid, or blind."""
        job=self.job;r=self.runtime
        if not self.due(step):return tail.copy(),None
        job.consumed=True;started=r.clock()
        info=dict(source_step=job.source_step,apply_step=step,proposal_path=job.path.name,
                  source_frame_timestamp=job.stamp,state_committed=False,feedback_wait_ms=0.,
                  feedback_budget_ms=max(0.,(job.deadline-job.started)*1000),
                  feedback_age_ms=(started-job.stamp)*1000)
        def reject(reason):
            info.update(revision_status=reason,commit_ms=(r.clock()-started)*1000)
            if reason=='no_evidence':self.missing+=1
            else:
                self.skipped+=1;self.last_failure_step=step
            self.last_info=info
            return tail.copy(),info
        if not job.done.is_set() or job.finished>=job.deadline or r.clock()>=deadline:
            return reject('deadline_expired')
        if job.error:
            info['error']=job.error
            return reject('feedback_error')
        proposal,diagnostics=job.result
        info.update(diagnostics)
        if not diagnostics.get('observer',{}).get('count',0):
            return reject('no_evidence')
        with r.lock:
            # ACKs from this executor are the only permitted intervening changes.
            if (r.pending or r.fault or r.version!=job.version+len(job.receipts)
                    or not np.array_equal(r.matrix,job.snapshot.matrix)):
                return reject('snapshot_changed')
            version=r.version
            z=job.snapshot.state.copy();u=job.snapshot.action.copy();at=job.snapshot.at
            history=deque(job.snapshot.history,maxlen=4096)
            for receipt in job.receipts:
                if receipt.status!='ack' or receipt.t_command<at:
                    return reject('snapshot_changed')
                action=r.mapping.reduce(receipt.applied6)
                z=advance(r.engine,z,u,receipt.t_command-at,r.dt)
                z=advance(r.engine,z,action,0,r.dt);at=receipt.t_command;u=action
                history.append((at,z.copy(),u.copy()))
            expected_at=job.receipts[-1].t_command if job.receipts else job.command_time
            if not np.allclose(u,r.action,rtol=0,atol=1e-8) or abs(expected_at-r.at)>1e-8:
                return reject('snapshot_changed')
            bounds=r.bounds
        # Full state is replayed from real ACKs; it never receives speculative
        # future actions. Only the validation readout advances to dispatch time.
        state=advance(r.engine,z,u,max(command_time,r.clock())-at,r.dt)
        if proposal.shape!=tail.shape:return reject('snapshot_changed')
        candidate=bounds.project(tail+(proposal-job.nominal),u)
        accepted=False;before=after=None
        if len(tail):
            ids=target_indices(r.engine.n_nodes,getattr(r,'control_node_indices',None))
            matrix=getattr(r,'control_target_matrix',None)
            def cost(actions):
                shapes=r.engine.rollout(state,actions)
                residual=shapes[:,ids]-reference[:,ids] if matrix is None else np.einsum('sn,tnc->tsc',matrix,shapes)-reference
                return float(np.mean(residual**2))
            before=cost(tail);after=cost(candidate)
            accepted=bool(bounds.valid(candidate,u) and after<before-1e-8)
            if not accepted:candidate=tail.copy();after=before
        with r.lock:
            if r.clock()>=deadline:return reject('deadline_expired')
            if r.version!=version or r.pending or r.fault:return reject('snapshot_changed')
            r.state=z;r.action=u;r.at=at;r.history=history;r.last_frame=job.stamp;r.version+=1
        self.skipped=0;self.missing=0
        info.update(revision_status='committed',state_committed=True,
                    prediction_px=transform(r.engine.observe(state,u),r.matrix),
                    commit_ms=(r.clock()-started)*1000,
                    applied_control=dict(accepted=accepted,mse_before_mm2=before,mse_after_mm2=after),
                    replayed_commands=[receipt.command_id for receipt in job.receipts])
        self.last_info=info
        return candidate,info

    def cancel(self):
        if self.job is not None:self.job.consumed=True
