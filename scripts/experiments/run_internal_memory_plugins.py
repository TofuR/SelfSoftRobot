#!/usr/bin/env python3
"""Val-selected internal memory branches on the frozen unified20 dataset.

All new fits and tests live in their own run. Test loading occurs only after
learning-rate selection and all 160 formal fits have completed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
SOURCE_RUN=ROOT/'workspace/runs/training/modeling_unified20_20260913_004'
DEFAULT_RUN=ROOT/'workspace/runs/training/modeling_internal_plugins_20260913_006'
ANALYSIS=ROOT/'workspace/runs/analysis/modeling_extensions_20260913_006/internal_plugins'
CONFIGS=[('mlp','base'),('mlp','both'),('mlp','static_capacity'),
         ('koopman','base'),('koopman','path'),('koopman','time'),('koopman','both'),('koopman','static_capacity')]
SEEDS=list(range(100,120))
EXPECTED={('mlp','base'):7405,('mlp','both'):9473,('mlp','static_capacity'):9473,
          ('koopman','base'):3981,('koopman','path'):4361,('koopman','time'):5081,
          ('koopman','both'):5441,('koopman','static_capacity'):5441}
L=None
DATA=None
RUN=None


def stamp():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)
def csv_write(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def name(family,variant):return f'{family}_{variant}'


def runtime(run):
    global L,np,torch,InternalMemoryModel,fit_plugin_normalization
    if L is not None:return
    path=run/'source/scripts/experiments/run_modeling_unified_repetitions.py'
    spec=importlib.util.spec_from_file_location('_frozen_unified20_loader',path)
    L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
    L.runtime(run,1)
    np,torch=L.np,L.torch
    from src.benchmarks.modeling_memory_plugin import InternalMemoryModel,fit_plugin_normalization


def prepare(run,workers):
    if (run/'PREPARED').exists():
        protocol=read(run/'protocol.json');assert protocol['workers']==workers
        runtime(run);return protocol
    run.mkdir(parents=True,exist_ok=True)
    assert not (run/'formal').exists(),'Existing fits require the completed preparation marker.'
    shutil.copytree(SOURCE_RUN/'source',run/'source',dirs_exist_ok=True)
    for source in [ROOT/'src/benchmarks/modeling_memory_plugin.py',Path(__file__).resolve()]:
        dest=run/'source'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    old=read(SOURCE_RUN/'protocol.json')
    protocol=dict(schema='internal_memory_plugin_protocol_v1',created_at=stamp(),source_run=str(SOURCE_RUN),
        dataset_manifest=old['dataset_manifest'],history=20,dt=.2,epochs=100,batch_size=256,
        seeds=SEEDS,tuning_seed=900001,learning_rates=[.001,.003],workers=workers,threads=1,
        configurations=[dict(family=f,variant=v,parameters=EXPECTED[(f,v)]) for f,v in CONFIGS],
        formal_fits=160,tuning_fits=16,optimizer='Adam',loss='normalized node MSE + 0.25 endpoint MSE',
        validation='epoch 1 and each multiple of 5; strict best; ReduceLROnPlateau factor .5 patience 4 min_lr 1e-5',
        initialization='Matched backbone weights per seed; zero memory projection; independent shared minibatch generator',
        memory='Learned monotone unit-range drive; fixed play thresholds .02/.5 and six taus .6..2s; q=e-p,d=h-e; split-local q0=d0=0',
        mlp_fusion='Second hidden preactivation L2(tanh(L1(u))) + D m; 64/64 hidden widths',
        koopman_fusion='Same lifted linear latent recurrence; output C z + D m; hidden128 latent16',
        capacity_control='32 current-drive powers (orders 1..8), same learned 20-coefficient drive and same D as both memory',
        feature_scaling='Training-only mean/std at initial drive; kept fixed during joint optimization',
        statistics=dict(primary='pooled test mean_node_mm',families={'mlp_internal':2,'koopman_internal':4},
                        contrasts=[dict(family=f+'_internal',reference=name(f,'base'),alternative=name(f,v)) for f,v in CONFIGS if v!='base'],
                        method='Paired exact signed-rank enumeration, Holm by family; paired-seed percentile bootstrap 20000; exact sign sensitivity',
                        direction='base error minus branch error; positive means improvement'),
        interpretation='Supplementary architecture evaluation on previously inspected fixed data; no fresh-data generalization claim',
        mask_evaluation='Not included in this plugin extension; skeleton and endpoint evaluated on all common targets')
    write(run/'protocol.json',protocol)
    runtime(run)
    data=L.load_roles(Path(protocol['dataset_manifest']),('train','val'))
    norm=fit_plugin_normalization(data['train']['x'])
    write(run/'plugin_normalization.json',norm)
    old_norm=read(SOURCE_RUN/'normalization.json')
    write(run/'output_normalization.json',{k:old_norm[k] for k in ['center','scale']})
    counts={}
    for family,variant in CONFIGS:
        torch.manual_seed(100);model=InternalMemoryModel(family,variant,norm)
        n=sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n==EXPECTED[(family,variant)],(family,variant,n)
        counts[name(family,variant)]=n
        y=model(data['val']['x'][:3]);assert tuple(y.shape)==(3,15,3) and torch.isfinite(y).all()
    write(run/'preflight.json',dict(counts=counts,train_windows=len(data['train']['x']),val_windows=len(data['val']['x']),test_opened=False,at=stamp()))
    (run/'PREPARED').write_text(stamp()+'\n');return protocol


def worker_init(run_string):
    global RUN,DATA,NORM,CENTER,SCALE
    RUN=Path(run_string);runtime(RUN)
    DATA=L.load_roles(Path(read(RUN/'protocol.json')['dataset_manifest']),('train','val'))
    NORM=read(RUN/'plugin_normalization.json');out=read(RUN/'output_normalization.json')
    CENTER=torch.tensor(out['center'],dtype=torch.float32);SCALE=out['scale']


def predict(model,x,center,scale,batch=512):
    model.eval()
    with torch.inference_mode():return torch.cat([model(b)*scale+center for b in x.split(batch)])


def fit_job(job):
    dest=RUN/job['id']
    if (dest/'COMPLETE').exists():
        result=read(dest/'run_manifest.json');assert result['config']==job and result['current_epoch']==100
        return result
    if dest.exists():
        retained=RUN/'interrupted_attempts'/f'{time.time_ns()}'/job['id'];retained.parent.mkdir(parents=True,exist_ok=True);dest.rename(retained)
    dest.mkdir(parents=True)
    with (dest/'train.log').open('w',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:return fit_impl(job,dest)
        except BaseException:
            write(dest/'failure.json',dict(at=stamp(),error=traceback.format_exc()));raise


def fit_impl(job,dest):
    torch.manual_seed(job['seed']);np.random.seed(job['seed'])
    start=time.perf_counter();model=InternalMemoryModel(job['family'],job['variant'],NORM)
    parameters=list(model.parameters());optimizer=torch.optim.Adam(parameters,lr=job['lr'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4,min_lr=1e-5)
    generator=torch.Generator().manual_seed(job['seed']+170001)
    x,y=DATA['train']['x'],DATA['train']['y'];vx,vy=DATA['val']['x'],DATA['val']['y'];target=(y-CENTER)/SCALE
    val0=float(torch.linalg.vector_norm(predict(model,vx,CENTER,SCALE)-vy,dim=-1).mean())
    state=dict(status='running',model=name(job['family'],job['variant']),family=job['family'],variant=job['variant'],
               seed=job['seed'],config=job,parameter_count=sum(p.numel() for p in parameters),
               epoch0_validation_node_mean_mm=val0,current_epoch=0,started_at=stamp(),train_windows=len(x),val_windows=len(vx))
    write(dest/'run_manifest.json',state)
    history=[];best=float('inf');best_epoch=None
    for epoch in range(1,101):
        model.train();loss_sum=0.;ep_start=time.perf_counter()
        for indices in torch.randperm(len(x),generator=generator).split(256):
            optimizer.zero_grad(set_to_none=True)
            pred=model(x[indices]);goal=target[indices]
            loss=(pred-goal).square().mean()+.25*(pred[:,-1]-goal[:,-1]).square().mean()
            assert torch.isfinite(loss),job
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10.);optimizer.step()
            loss_sum+=float(loss.detach())*len(indices)
        if epoch!=1 and epoch%5:continue
        val=float(torch.linalg.vector_norm(predict(model,vx,CENTER,SCALE)-vy,dim=-1).mean())
        if val<best:
            best,best_epoch=val,epoch
            torch.save(dict(schema='internal_memory_plugin_checkpoint_v1',family=job['family'],variant=job['variant'],
                state_dict=model.state_dict(),normalization=NORM,center=CENTER.tolist(),scale=SCALE,
                selected_epoch=epoch,validation_node_mean_mm=val,config=job),dest/'best_eval_model.pt')
        history.append(dict(epoch=epoch,validation_node_mean_mm=val,best_validation_node_mean_mm=best,
            train_loss=loss_sum/len(x),lr=optimizer.param_groups[0]['lr'],epoch_seconds=time.perf_counter()-ep_start,
            elapsed_seconds=time.perf_counter()-start))
        scheduler.step(val)
        state.update(current_epoch=epoch,best_epoch=best_epoch,best_validation_node_mean_mm=best,wall_seconds=time.perf_counter()-start)
        write(dest/'history.json',history);write(dest/'run_manifest.json',state)
        print(f'epoch={epoch} val={val:.6f} best={best:.6f}',flush=True)
    best80=min(h['validation_node_mean_mm'] for h in history if h['epoch']<=80)
    state.update(status='complete',completed_at=stamp(),wall_seconds=time.perf_counter()-start,
                 after80_best_gain_mm=best80-best,after80_best_gain_pct=100*(best80-best)/best80)
    torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    generator=generator.get_state(),epoch=100,config=job),dest/'training_state.pt')
    write(dest/'run_manifest.json',state);(dest/'COMPLETE').write_text(stamp()+'\n');return state


def execute(run,jobs,phase,workers):
    rows=[];write(run/'status.json',dict(phase=phase,status='running',completed=0,total=len(jobs),at=stamp()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(str(run),)) as pool:
        futures={pool.submit(fit_job,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            row=future.result();rows.append(row)
            write(run/'status.json',dict(phase=phase,status='running',completed=len(rows),total=len(jobs),last=futures[future]['id'],at=stamp()))
            print(f'{phase} {len(rows)}/{len(jobs)} {row["model"]} seed={row["seed"]} val={row["best_validation_node_mean_mm"]:.6f}',flush=True)
    return rows


def paired_statistics(rows):
    from scipy import stats
    lookup={(r['model'],r['seed']):r['mean_node_mm'] for r in rows}
    indices=np.random.default_rng(20260913).integers(0,20,size=(20000,20));contrasts=[]
    for family,variant in CONFIGS:
        if variant=='base':continue
        base,other=name(family,'base'),name(family,variant)
        delta=np.array([lookup[(base,s)]-lookup[(other,s)] for s in SEEDS]);active=delta[delta!=0]
        ranks=np.rint(2*stats.rankdata(np.abs(active))).astype(int);counts=np.zeros(int(ranks.sum())+1,dtype=np.int64);counts[0]=1
        for rank in ranks:
            previous=counts.copy();counts[rank:]+=previous[:-rank]
        observed=int(ranks[active>0].sum());p=min(1.,2*min(counts[:observed+1].sum(),counts[observed:].sum())/2**len(active))
        ci=np.quantile(delta[indices].mean(1),[.025,.975])
        contrasts.append(dict(family=family+'_internal',reference=base,alternative=other,n_pairs=20,
            mean_reference_minus_alternative_mm=float(delta.mean()),bootstrap95_lower_mm=float(ci[0]),bootstrap95_upper_mm=float(ci[1]),
            positive_pairs=int((delta>0).sum()),negative_pairs=int((delta<0).sum()),zero_pairs=int((delta==0).sum()),
            wilcoxon_exact_p=float(p),sign_test_exact_p=float(stats.binomtest(int((active>0).sum()),len(active)).pvalue) if len(active) else 1.,
            seed_differences_mm=delta.tolist()))
    for family in ['mlp_internal','koopman_internal']:
        group=[r for r in contrasts if r['family']==family]
        for key in ['wilcoxon_exact_p','sign_test_exact_p']:
            for row,value in zip(group,L.holm([r[key] for r in group])):row[key.replace('exact','holm')]=value
    return contrasts


def evaluate(run,jobs):
    assert (run/'TRAIN_VAL_COMPLETE.json').exists()
    runtime(run);protocol=read(run/'protocol.json');test=L.load_roles(Path(protocol['dataset_manifest']),('test',))['test']
    rows=[];sequence_rows=[]
    for i,job in enumerate(jobs):
        model_name=name(job['family'],job['variant']);dest=run/'evaluation'/model_name/f'seed_{job["seed"]}'
        if (dest/'COMPLETE').exists():
            rows.append(read(dest/'metrics.json'));sequence_rows.extend(read(dest/'sequence_metrics.json'));continue
        dest.mkdir(parents=True,exist_ok=True)
        ck=torch.load(run/job['id']/'best_eval_model.pt',map_location='cpu',weights_only=False)
        model=InternalMemoryModel(ck['family'],ck['variant'],ck['normalization']);model.load_state_dict(ck['state_dict'],strict=True)
        pred=predict(model,test['x'],torch.tensor(ck['center']),ck['scale']).numpy();target=test['y'].numpy()
        errors=np.linalg.norm(pred.astype(np.float64)-target.astype(np.float64),axis=-1)
        def metrics(mask):
            e=errors[mask];return dict(mean_node_mm=float(e.mean()),endpoint_mm=float(e[:,-1].mean()),
                node_rmse_mm=float(np.sqrt(np.mean(e**2,axis=1)).mean()),node_global_rmse_mm=float(np.sqrt(np.mean(e**2))),
                endpoint_rmse_mm=float(np.sqrt(np.mean(e[:,-1]**2))))
        row=dict(model=model_name,family=job['family'],variant=job['variant'],seed=job['seed'],test_frames=len(errors),
                 parameter_count=sum(p.numel() for p in model.parameters()),selected_epoch=ck['selected_epoch'],lr=job['lr'],**metrics(slice(None)))
        groups=[dict(model=model_name,seed=job['seed'],group=s['record']['group'],test_frames=int((test['groups']==g).sum()),**metrics(test['groups']==g)) for g,s in enumerate(test['sequences'])]
        np.savez_compressed(dest/'predictions.npz',prediction_mm=pred,target_mm=target,groups=test['groups'],frame_ids=test['frame_ids'])
        write(dest/'metrics.json',row);write(dest/'sequence_metrics.json',groups);(dest/'COMPLETE').write_text(stamp()+'\n')
        rows.append(row);sequence_rows.extend(groups)
        write(run/'status.json',dict(phase='test',completed=i+1,total=len(jobs),status='running',at=stamp()))
    csv_write(run/'raw_test.csv',rows);csv_write(run/'raw_test_by_sequence.csv',sequence_rows)
    contrasts=paired_statistics(rows);write(run/'paired_statistics.json',dict(primary='pooled mean_node_mm',contrasts=contrasts))
    summary=[]
    for family,variant in CONFIGS:
        group=[r for r in rows if r['family']==family and r['variant']==variant];assert sorted(r['seed'] for r in group)==SEEDS
        item=dict(model=name(family,variant),family=family,variant=variant,n=20,parameters=group[0]['parameter_count'],test_frames=2958)
        for metric in ['mean_node_mm','endpoint_mm','node_rmse_mm','node_global_rmse_mm']:
            values=[r[metric] for r in group];item[metric]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)))
        summary.append(item)
    ANALYSIS.mkdir(parents=True,exist_ok=True)
    result=dict(schema='internal_memory_plugin_results_v1',source_run=str(run),models=summary,statistics=contrasts,protocol=protocol)
    write(ANALYSIS/'summary.json',result)
    csv_write(ANALYSIS/'model_summary.csv',[dict(model=x['model'],n=x['n'],parameters=x['parameters'],**{k+'_'+stat:x[k][stat] for k in ['mean_node_mm','endpoint_mm'] for stat in ['mean','sd']}) for x in summary])
    lines=['# 网络内部记忆分支：20次重复结果','','同一三记录时间622划分；H20；100 epoch；仅val选学习率，再评价2958测试目标。全部20 seeds保留。','',
           '|配置|参数|骨架/mm|末端/mm|','|---|---:|---:|---:|']
    for r in summary:lines.append(f'|{r["model"]}|{r["parameters"]}|{r["mean_node_mm"]["mean"]:.6f}±{r["mean_node_mm"]["sd"]:.6f}|{r["endpoint_mm"]["mean"]:.6f}±{r["endpoint_mm"]["sd"]:.6f}|')
    lines+=['','差值=同一骨干base误差−分支误差；正数为改善。区间为20个配对seed的bootstrap95%区间；MLP2项、Koopman4项分别Holm。','',
            '|配置|差值/mm|95%CI|Wilcoxon校正p|符号校正p|','|---|---:|---|---:|---:|']
    for r in contrasts:lines.append(f'|{r["alternative"]}|{r["mean_reference_minus_alternative_mm"]:.6f}|[{r["bootstrap95_lower_mm"]:.6f}, {r["bootstrap95_upper_mm"]:.6f}]|{r["wilcoxon_holm_p"]:.6g}|{r["sign_test_holm_p"]:.6g}|')
    lines+=['','驱动变换与网络/分支读出联合学习，阈值和时间常数固定。该实验不同于此前固定编码后拼输入的插件实验，需分别报告。数据与划分此前已被分析，结果为当前固定数据上的补充结构评价。']
    (ANALYSIS/'summary.md').write_text('\n'.join(lines)+'\n')
    write(run/'status.json',dict(phase='complete',status='complete',formal_fits=160,test_fits=160,at=stamp()))
    write(ANALYSIS/'COMPLETE.json',dict(status='complete',at=stamp(),models=8,seeds=20,test_frames=2958))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=DEFAULT_RUN);parser.add_argument('--workers',type=int,default=8);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    run=args.run.resolve();assert run.parent==ROOT/'workspace/runs/training' and run!=SOURCE_RUN
    run.mkdir(parents=True,exist_ok=True)
    with (run/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=prepare(run,args.workers)
        if args.prepare_only:return
        tuning=[dict(id=f'tuning/{name(f,v)}/lr_{lr:g}',family=f,variant=v,seed=900001,lr=lr,phase='tuning') for f,v in CONFIGS for lr in [.001,.003]]
        if not (run/'selected_configuration.json').exists():
            tuned=execute(run,tuning,'tuning',args.workers);chosen={}
            for f,v in CONFIGS:
                best=min([r for r in tuned if r['family']==f and r['variant']==v],key=lambda r:(r['best_validation_node_mean_mm'],r['config']['lr']))
                chosen[name(f,v)]=dict(family=f,variant=v,lr=best['config']['lr'],tuning_val_mm=best['best_validation_node_mean_mm'])
            write(run/'selected_configuration.json',dict(frozen_at=stamp(),configurations=chosen,test_opened=False))
        chosen=read(run/'selected_configuration.json')['configurations']
        jobs=[dict(id=f'formal/{name(f,v)}/seed_{seed}',family=f,variant=v,seed=seed,lr=chosen[name(f,v)]['lr'],phase='formal') for seed in SEEDS for f,v in CONFIGS]
        if not (run/'TRAIN_VAL_COMPLETE.json').exists():
            formal=execute(run,jobs,'formal',args.workers);assert len(formal)==160
            csv_write(run/'raw_validation.csv',[dict(model=r['model'],seed=r['seed'],best_epoch=r['best_epoch'],best_val_mm=r['best_validation_node_mean_mm'],epoch0_val_mm=r['epoch0_validation_node_mean_mm'],wall_seconds=r['wall_seconds'],after80_best_gain_mm=r['after80_best_gain_mm'],after80_best_gain_pct=r['after80_best_gain_pct']) for r in formal])
            write(run/'TRAIN_VAL_COMPLETE.json',dict(at=stamp(),fits=160,test_opened=False))
        evaluate(run,jobs)
        print('Complete: '+str(ANALYSIS/'summary.md'),flush=True)


if __name__=='__main__':main()
