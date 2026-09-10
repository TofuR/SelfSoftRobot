"""One bounded, speculative feedback job per runtime; only caller may commit.

Threads avoid serialization overhead. This is a soft deadline, not an OS real-time
scheduler: Python/GIL, camera, transport and filesystem jitter remain measurable.
A stuck solver occupies this one slot, never the live runtime lock or a queue.
"""
import json
import threading
import numpy as np


class FeedbackJob:
    def __init__(self,runtime,image,stamp,tail,reference,path):
        self.snapshot,self.version=runtime.feedback_snapshot()
        self.done=threading.Event();self.result=None;self.error=None
        self.started=runtime.clock();self.finished=None
        self.path=path
        self.thread=threading.Thread(target=self._run,args=(image,stamp,tail.copy(),reference.copy()),daemon=True,name='hereditary-feedback')
        self.thread.start()

    def _run(self,image,stamp,tail,reference):
        try:self.result=self._compute(image,stamp,tail,reference)
        except Exception as error:self.error=f'{type(error).__name__}: {error}'
        finally:
            self.finished=self.snapshot.clock();self.done.set()
        # Independent proposal artifact, even if executor has already discarded
        # this job and closed its archive. No live runtime or archive handle.
        value=dict(version=self.version,started=self.started,finished=self.finished,
                   wall_ms=(self.finished-self.started)*1000,error=self.error,
                   semantics='proposal only; consult steps.jsonl for commit decision')
        if self.result is not None:value.update(proposed_model_actions=self.result[0],diagnostics=self.result[1])
        try:
            temporary=self.path.with_suffix('.pending')
            temporary.write_text(json.dumps(value,default=lambda a:np.asarray(a).tolist(),allow_nan=False)+'\n')
            temporary.replace(self.path)
        except Exception as error:
            # Preserve a discoverable failure without touching the live logger.
            self.archive_error=str(error)
            if self.path.parent.exists():
                try:self.path.with_suffix('.error.txt').write_text(str(error))
                except OSError:pass

    def _compute(self,image,stamp,tail,reference):
        return self.snapshot.feedback(image,stamp,tail,reference)


def deadline_feedback(runtime,image,stamp,tail,reference,deadline,abort,path):
    """Return a revision only after the entire state+suffix transaction commits."""
    job=getattr(runtime,'_feedback_job',None)
    if job is not None and job.thread.is_alive():
        return tail.copy(),dict(revision_status='worker_busy',feedback_wait_ms=0.,state_committed=False)
    if runtime.clock()>=deadline:
        return tail.copy(),dict(revision_status='no_budget',feedback_wait_ms=0.,state_committed=False)
    job=FeedbackJob(runtime,image,stamp,tail,reference,path);runtime._feedback_job=job
    while not job.done.is_set():
        remaining=deadline-runtime.clock()
        if remaining<=0 or abort.is_set():break
        job.done.wait(min(.002,remaining))
    info=dict(feedback_wait_ms=(runtime.clock()-job.started)*1000,state_committed=False,proposal_path=str(path.name))
    if abort.is_set():info['revision_status']='operator_abort'
    elif not job.done.is_set() or job.finished>=deadline:info['revision_status']='deadline_expired'
    elif job.error:raise RuntimeError(job.error)
    else:
        candidate,diagnostics=job.result;info.update(diagnostics)
        if diagnostics.get('reason')=='stale_or_duplicate_frame':
            info['revision_status']='stale_or_duplicate_frame';return tail.copy(),info
        committed,reason=runtime.commit_feedback(job.snapshot,job.version,deadline)
        info.update(state_committed=committed,revision_status=reason)
        if committed:return candidate,info
    info.pop('prediction_px',None)
    info['candidate_discarded']=info['revision_status']
    return tail.copy(),info
