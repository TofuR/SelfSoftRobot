#!/usr/bin/env python3
"""Four-GPU single-holdout study: screen, freeze, repeat, fixed test scoring."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.benchmarks.modeling_data import prepare_temporal_pool,sha256,write_json

COMPARISONS=['chen_direction','bezier_gru','park_tcn','oscillator','koopman','pcc','mlp','window_mlp','linear','polynomial2']
ABLATIONS=['hov','hov_no_play','hov_no_maxwell','hov_no_memory']
SCREEN_MODELS=['hov',*COMPARISONS]


def now():return datetime.datetime.now().astimezone().isoformat()


def create_study(study,source):
    study=Path(study).resolve();study.mkdir(parents=True,exist_ok=False)
    manifest=prepare_temporal_pool(source,study/'data',history=20)
    frozen=study/'frozen_code'
    shutil.copytree(ROOT/'src',frozen/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    (frozen/'scripts/experiments').mkdir(parents=True)
    for name in ('modeling_study.py','modeling_benchmark.py'):
        shutil.copy2(ROOT/'scripts/experiments'/name,frozen/'scripts/experiments'/name)
    files={str(p.relative_to(frozen)):sha256(p) for p in frozen.rglob('*') if p.is_file()}
    base=dict(history=20,hidden=64,chen_hidden=128,park_channels=4,latent=8,force_hidden=32,
              n_play=2,n_maxwell=6,batch_size=256,lr=.001,ridge=1e-4,endpoint_weight=.25,
              prior_steps=500,train_stride=1,eval_stride=1,threads=2,device='cuda:0',
              validation_interval=5,min_delta_mm=.001,study_id=study.name)
    plan=dict(schema='modeling_single_holdout_study_v1',created=now(),study_id=study.name,
              dataset_manifest=str(manifest),dataset_sha256=sha256(manifest),base_config=base,
              gpus=[0,1,2,3],seeds=[0,1,2,3,4],comparison_models=COMPARISONS,ablations=ABLATIONS,
              screening=dict(seed=101,epochs=15,candidates='2 learning rates x 2 capacities; ridge models 3 regularization values',
                             validation_only=True,selection='minimum sequence-macro validation node mean',schedule_lr=False),
              formal=dict(epochs=300,minimum_epochs=100,early_stop_checks=12,schedule_lr=True,
                          scheduler='val plateau factor .5, patience4 validation checks, min_lr1e-5'),
              execution_order=['full-data quality checks','parallel train/val screening','freeze all configurations',
                               '70 independent full-data training runs','fixed test scoring for all selected checkpoints'],
              analysis='User will request interpretation and significance aggregation after training; analyze subcommand is prepared',
              statistical_unit='Original sequence held-out temporal slice; average five seeds within each of seven sequences',
              source_hashes=files,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(study/'study_plan.json',plan)
    write_json(study/'status.json',dict(status='planned',phase='ready',created=now(),formal_runs=70))
    return study


def candidates(model,base):
    if model in ('linear','polynomial2'):
        return [dict(base,model=model,ridge=r) for r in (1e-5,1e-4,1e-3)]
    result=[]
    for lr in (.001,.003):
        for capacity in ('base','wide'):
            cfg=dict(base,model=model,lr=lr,capacity=capacity)
            if capacity=='wide':
                if model=='hov':cfg.update(n_play=3,n_maxwell=8)
                elif model=='chen_direction':cfg.update(chen_hidden=256)
                elif model=='park_tcn':cfg.update(park_channels=8)
                elif model=='oscillator':cfg.update(latent=16,force_hidden=64)
                elif model=='koopman':cfg.update(hidden=128,latent=16)
                else:cfg['hidden']=128
            result.append(cfg)
    return result


def run_jobs(study,jobs,phase):
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    lock=threading.Lock();finished=[];failed=[]
    interpreter=sys.executable
    script=study/'frozen_code/scripts/experiments/modeling_study.py'
    def worker(gpu):
        while True:
            try:job=pending.get_nowait()
            except queue.Empty:return
            job['gpu']=gpu
            jobfile=study/'jobs'/f"{job['id']}.json"
            command=[interpreter,str(script),'task','--job',str(jobfile)]
            job.setdefault('config',{})['command']=command
            write_json(jobfile,job)
            log=study/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            with log.open('w') as stream:
                process=subprocess.Popen(command,cwd=study/'frozen_code',env=env,stdout=stream,stderr=subprocess.STDOUT)
                write_json(study/'workers'/f'gpu{gpu}.json',dict(phase=phase,job=job['id'],pid=process.pid,status='running',started=now(),log=str(log)))
                code=process.wait()
            with lock:
                (finished if code==0 else failed).append(job['id'])
                write_json(study/'status.json',dict(status='running',phase=phase,completed=len(finished),failed=failed,total=len(jobs),updated=now()))
            write_json(study/'workers'/f'gpu{gpu}.json',dict(phase=phase,job=job['id'],status='complete' if code==0 else 'failed',exit_code=code,updated=now()))
            print(f'{phase}: {job["id"]} exit={code}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu) for gpu in (0,1,2,3)]
        for future in futures:future.result()
    if failed:raise RuntimeError(f'{phase} jobs failed: {failed}; logs retained, next phase blocked')


def run_study(study):
    study=Path(study).resolve();plan=json.loads((study/'study_plan.json').read_text())
    status=json.loads((study/'status.json').read_text())
    if status['status']!='planned':raise ValueError('Study already started; create a new study or review failed jobs explicitly')
    with (study/'ORCHESTRATOR_STARTED').open('x') as stream:stream.write(now()+'\n')
    write_json(study/'status.json',dict(status='running',phase='screening',completed=0,updated=now()))
    try:
        for relative,digest in plan['source_hashes'].items():
            if sha256(study/'frozen_code'/relative)!=digest:raise ValueError('Frozen source changed')
        if sha256(plan['dataset_manifest'])!=plan['dataset_sha256']:raise ValueError('Dataset changed')
        write_json(study/'environment.json',dict(created=now(),python=sys.executable,
                   gpu_query=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv'],text=True)))
        screening=[]
        for model in SCREEN_MODELS:
            for i,cfg in enumerate(candidates(model,plan['base_config'])):
                cfg.update(seed=101,run_kind='screening',epochs=15,minimum_epochs=15,early_stop_checks=10000,schedule_lr=False)
                screening.append(dict(id=f'screen_{model}_c{i}',task='train',manifest=plan['dataset_manifest'],
                                      output=str(study/'screening'/model/f'candidate{i}'),config=cfg))
        write_json(study/'screening_plan.json',screening)
        run_jobs(study,screening,'screening')
        selected={};selection=[]
        for model in SCREEN_MODELS:
            rows=[]
            for job in [j for j in screening if j['config']['model']==model]:
                state=json.loads((Path(job['output'])/'run_manifest.json').read_text())
                rows.append(dict(candidate=job['id'],score=state['best_validation_node_mean_mm'],
                                 convergence=state['convergence'],source_run=job['output'],config=job['config']))
            winner=min(rows,key=lambda r:r['score'])
            selected[model]=dict(winner['config'])
            selection.append(dict(model=model,winner=winner['candidate'],all_candidates=rows))
        # All memory ablations inherit the full-model setting and static-prior budget.
        for model in ABLATIONS[1:]:selected[model]=dict(selected['hov'],model=model)
        frozen=dict(created=now(),selection_role='val',selection=selection,configs=selected,
                    ablation_policy='reuse selected full-HOV capacity/lr/prior settings; independently refit and retrain each seed')
        write_json(study/'frozen_configs.json',frozen)
        formal=[]
        order=['hov','chen_direction','bezier_gru','park_tcn','oscillator','koopman','pcc','mlp','window_mlp',
               'hov_no_play','hov_no_maxwell','hov_no_memory','linear','polynomial2']
        for seed in plan['seeds']:
            for model in order:
                cfg=dict(selected[model],**plan['formal'])
                cfg.update(model=model,seed=seed,run_kind='formal',frozen_config_sha256=sha256(study/'frozen_configs.json'))
                category='ablation' if model in ABLATIONS else 'comparison'
                formal.append(dict(id=f'formal_{model}_seed{seed}',task='train',manifest=plan['dataset_manifest'],
                                   output=str(study/'formal'/category/model/f'seed{seed}'),config=cfg))
        write_json(study/'formal_plan.json',formal)
        write_json(study/'status.json',dict(status='running',phase='formal',completed=0,total=len(formal),updated=now()))
        run_jobs(study,formal,'formal')
        (study/'TRAINING_COMPLETE').write_text(now()+'\n')
        tests=[dict(id=f'test_{job["config"]["model"]}_seed{job["config"]["seed"]}',task='evaluate',
                    run=job['output'],output=str(Path(job['output'])/'evaluation_test')) for job in formal]
        write_json(study/'test_plan.json',tests)
        run_jobs(study,tests,'test_scoring')
        (study/'TEST_SCORING_COMPLETE').write_text(now()+'\n')
        write_json(study/'status.json',dict(status='complete',phase='awaiting_user_analysis',training_runs=len(formal),test_evaluations=len(tests),updated=now()))
    except Exception:
        import traceback
        write_json(study/'status.json',dict(status='failed',updated=now(),error=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--study',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p=sub.add_parser('run');p.add_argument('--study',type=Path,required=True)
    p=sub.add_parser('task');p.add_argument('--job',type=Path,required=True)
    p=sub.add_parser('analyze');p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':print(create_study(args.study,args.source))
    elif args.command=='run':run_study(args.study)
    elif args.command=='task':
        job=json.loads(args.job.read_text())
        if job['task']=='train':
            from src.benchmarks.modeling_fast_training import train_fast
            train_fast(job['manifest'],job['output'],job['config'])
        else:
            from src.benchmarks.modeling_runner import evaluate_run
            evaluate_run(job['run'],job['output'],role='test',device='cuda:0')
    else:
        from src.benchmarks.modeling_runner import aggregate
        if not (args.study/'TEST_SCORING_COMPLETE').exists():raise ValueError('Test scoring is incomplete')
        plan=json.loads((args.study/'formal_plan.json').read_text())
        for category,models in [('comparison',['hov',*COMPARISONS]),('ablation',ABLATIONS)]:
            evaluations=[Path(j['output'])/'evaluation_test' for j in plan if j['config']['model'] in models]
            aggregate(evaluations,args.out/category,reference='hov',metrics=['mean_node_mm','endpoint_mm','mask_iou','mask_dice'])


if __name__=='__main__':main()
