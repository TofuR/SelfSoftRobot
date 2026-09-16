#!/usr/bin/env python3
"""Validation-led modeling experiments on a fixed three-recording subset."""
from pathlib import Path
import argparse,concurrent.futures,json,os,queue,subprocess,sys,threading,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.benchmarks.modeling_data import write_json


def run_jobs(study,jobs,phase,gpus=(2,3),workers_per_gpu=1):
    if workers_per_gpu<1 or not gpus:raise ValueError('At least one GPU and worker are required')
    write_json(study/'status.json',dict(phase=phase,status='running',complete=0,failed=[],total=len(jobs),updated=time.time()))
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    done=[];failed=[];lock=threading.Lock()
    def worker(gpu,slot):
        while True:
            try:job=pending.get_nowait()
            except queue.Empty:return
            file=study/'jobs'/f"{job['id']}.json"
            command=[sys.executable,str(Path(__file__).resolve()),'task','--job',str(file)]
            job['config']=dict(job.get('config',{}),command=command)
            job['gpu']=gpu;job['worker_slot']=slot;write_json(file,job)
            log=study/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with log.open('w') as stream:
                p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                write_json(study/'workers'/f'gpu{gpu}_slot{slot}.json',dict(job=job['id'],pid=p.pid,phase=phase,log=str(log)))
                code=p.wait()
            with lock:
                (done if code==0 else failed).append(job['id'])
                write_json(study/'status.json',dict(phase=phase,status='running',complete=len(done),failed=failed,total=len(jobs),updated=time.time()))
            print(phase,job['id'],'exit',code,flush=True)
    with concurrent.futures.ThreadPoolExecutor(len(gpus)*workers_per_gpu) as pool:
        for f in [pool.submit(worker,g,slot) for g in gpus for slot in range(workers_per_gpu)]:f.result()
    write_json(study/'status.json',dict(phase=phase,status='failed' if failed else 'complete',complete=len(done),failed=failed,total=len(jobs),updated=time.time()))
    if failed:raise RuntimeError(f'Failed jobs: {failed}')


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('run');q.add_argument('--study',type=Path,required=True);q.add_argument('--plan',type=Path,required=True);q.add_argument('--phase',required=True)
    q.add_argument('--gpus',type=int,nargs='+',default=[2,3]);q.add_argument('--workers-per-gpu',type=int,default=1)
    q=s.add_parser('pipeline');q.add_argument('--study',type=Path,required=True)
    q.add_argument('--gpus',type=int,nargs='+',default=[2,3]);q.add_argument('--workers-per-gpu',type=int,default=2)
    q=s.add_parser('task');q.add_argument('--job',type=Path,required=True)
    a=p.parse_args()
    if a.command=='run':run_jobs(a.study.resolve(),json.loads(a.plan.read_text()),a.phase,a.gpus,a.workers_per_gpu)
    elif a.command=='pipeline':
        study=a.study.resolve()
        for phase,plan in [('formal_training','formal_plan.json'),('test_evaluation','evaluation_plan.json')]:
            run_jobs(study,json.loads((study/plan).read_text()),phase,a.gpus,a.workers_per_gpu)
        run_jobs(study,json.loads((study/'latency_plan.json').read_text()),'latency',(a.gpus[0],),1)
        subprocess.run([sys.executable,str(ROOT/'scripts/experiments/report_modeling_improvement.py'),
                        '--study',str(study)],cwd=ROOT,check=True)
        write_json(study/'status.json',dict(phase='results',status='complete',updated=time.time()))
    else:
        j=json.loads(a.job.read_text())
        if j['task']=='train':
            from src.benchmarks.modeling_fast_training import train_fast
            train_fast(j['manifest'],j['output'],j['config'])
        elif j['task']=='evaluate':
            from src.benchmarks.modeling_runner import evaluate_run
            evaluate_run(j['run'],j['output'],role=j.get('role','test'),device='cuda:0',verify_mask_hashes=False)
        elif j['task']=='latency':
            from src.benchmarks.modeling_seed_summary import benchmark_latency
            benchmark_latency(j['run'],'cuda:0',j['output'])
        else:raise ValueError(j['task'])

if __name__=='__main__':main()
