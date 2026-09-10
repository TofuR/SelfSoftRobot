"""Analysis records with explicit command, camera and optional NDI timestamps."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np

TIMING_FIELDS=['step','revision_status','command_interval_ms','command_jitter_ms','ack_ms','frame_wait_ms','preprocess_ms','feedback_budget_ms','feedback_wait_ms','compute_ms','edge_ms','observer_ms','control_ms','archive_enqueue_ms','cycle_ms','state_committed','suffix_changed',
               'send_wait_ms','dispatch_wait_ms','ack_delivery_ms','camera_poll_count','frame_age_at_selection_ms','frame_after_ack_ms',
               'source_step','apply_step','feedback_age_ms','commit_ms']

class ExperimentArchive:
    def __init__(self,folder,started,metadata,evaluation_provider=None):
        self.folder=Path(folder);self.started=started;self.provider=evaluation_provider
        self.metadata=metadata;self.samples=[];self.proposals=[];self.handles=[]
        self.commands=self.csv('commands.csv',['step','command_id','t_command','t_ack','status']+[f'requested_c{i}_kpa' for i in range(6)]+[f'applied_c{i}_kpa' for i in range(6)])
        self.frames=self.csv('samples.csv',['step','camera','frame_timestamp','command_id','t_command','after_command_ms','fresh_after_command','raw_path','feedback_path','software_occlusion','image_write_ms'])
        self.image_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='experiment-images');self.image_jobs=deque()
        self.steps=(self.folder/'steps.jsonl').open('x',buffering=1);self.handles.append(self.steps)
        (self.folder/'feedback_jobs').mkdir()
        self.timings=self.csv('timings.csv',TIMING_FIELDS)
        self.write_status('running')
    def csv(self,name,columns):
        f=(self.folder/name).open('x',newline='',buffering=1);self.handles.append(f);writer=csv.writer(f);writer.writerow(columns);return writer
    def write_status(self,status,**extra):
        (self.folder/'metadata.json').write_text(json.dumps(dict(self.metadata,status=status,started=self.started,**extra),ensure_ascii=False,indent=2))
    def command(self,step,receipt):
        self.commands.writerow([step,receipt.command_id,receipt.t_command,receipt.t_ack,receipt.status,*receipt.requested6,*receipt.applied6])
    def step(self,value):
        if value.get('proposal_path'):self.proposals.append(value['proposal_path'])
        self.steps.write(json.dumps(value,default=lambda a:np.asarray(a).tolist(),allow_nan=False)+'\n')
        self.timings.writerow([value.get(k) for k in TIMING_FIELDS])

    def enqueue_image(self,*args):
        while self.image_jobs and self.image_jobs[0].done():self.image_jobs.popleft().result()
        if len(self.image_jobs)>=16:raise RuntimeError('图像存储积压，停止实验以保留完整记录')
        # Producers may reuse camera buffers; the archive owns these copies.
        step,camera,frame,receipt,*rest=args
        frame=(frame[0].copy(),frame[1])
        if rest and rest[0] is not None:rest[0]=rest[0].copy()
        self.image_jobs.append(self.image_pool.submit(self.image,step,camera,frame,receipt,*rest))

    def image(self,step,camera,frame,receipt,feedback=None,occlusion=None):
        write_started=time.monotonic()
        image,stamp=frame;relative=f'raw/cam{camera}/{step:05d}.png';path=self.folder/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if not cv2.imwrite(str(path),image):raise OSError('原图保存失败')
        processed=''
        if feedback is not None:
            processed=f'frames/{step:05d}.png'
            if not cv2.imwrite(str(self.folder/processed),feedback):raise OSError('反馈图保存失败')
        self.frames.writerow([step,camera,stamp,receipt.command_id,receipt.t_command,(stamp-receipt.t_command)*1000,stamp>=max(receipt.t_command+self.metadata.get('settle_s',.05),receipt.t_ack or receipt.t_command),relative,processed,json.dumps(occlusion or {}),(time.monotonic()-write_started)*1000])
        self.samples.append((step,camera,stamp))
    def close(self,status):
        ended=time.monotonic()
        try:
            self.image_pool.shutdown(wait=True)
            for job in self.image_jobs:job.result()
            try:
                value=self.provider(self.started,ended) if self.provider else dict(backend='disabled',state='off',connected=False,probe_count=0,samples=[])
            except Exception as error:
                value=dict(backend='unknown',state='error',connected=False,probe_count=0,samples=[],error=str(error))
            rows=value.pop('samples');poses=self.csv('ndi.csv',['timestamp','probe','x_mm','y_mm','z_mm','rx','ry','rz','qw','qx','qy','qz','quality','valid'])
            for stamp,data in rows:
                for i in range(max(value['probe_count'],len(data)//11)):
                    pose=list(data[i*11:(i+1)*11]);pose+= [float('nan')]*(11-len(pose))
                    valid=bool(np.isfinite(pose).all())
                    poses.writerow([stamp,i,*pose,valid])
            links=self.csv('frame_ndi.csv',['step','camera','frame_timestamp','ndi_timestamp','ndi_age_ms','fresh_200ms'])
            stamps=np.array([row[0] for row in rows])
            for step,camera,stamp in self.samples:
                index=int(np.searchsorted(stamps,stamp,side='right'))-1
                t=stamps[index] if index>=0 else None;age=None if t is None else (stamp-t)*1000
                links.writerow([step,camera,stamp,t,age,age is not None and 0<=age<=200])
            self.write_status(status,ended=ended,ndi=value,ndi_samples=len(rows),feedback_pending_at_close=[name for name in self.proposals if not (self.folder/'feedback_jobs'/name).is_file()],synchronization='host monotonic receive times; nearest preceding NDI; cameras not exposure synchronized')
        except Exception as error:
            self.write_status('archive_failed',execution_outcome=status,error=str(error),ended=ended)
            raise
        finally:
            for handle in self.handles:handle.close()
