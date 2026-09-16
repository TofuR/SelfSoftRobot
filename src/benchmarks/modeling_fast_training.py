"""Full-data GPU training with cached causal windows and validation selection."""
from __future__ import annotations
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from src.benchmarks.modeling_data import CausalWindows, ROOT, load_sequences, sha256, write_json
from src.benchmarks.modeling_models import Polynomial, fit_normalization, make_model
from src.benchmarks.modeling_runner import _archive_code, _seed


def cache_windows(sequences, history, device):
    windows=CausalWindows(sequences,history)
    x=np.stack([sequences[i]['actions'][t-history+1:t+1] for i,t in windows.indices])
    y=np.stack([sequences[i]['positions'][t] for i,t in windows.indices])
    groups=np.asarray([i for i,t in windows.indices])
    return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),groups


def validation_mean(errors, groups, aggregation='sequence_macro'):
    if aggregation=='pooled_frames':return float(np.mean(errors))
    if aggregation=='sequence_macro':return float(np.mean([errors[groups==i].mean() for i in np.unique(groups)]))
    raise ValueError(f'Unknown validation aggregation: {aggregation}')


def train_fast(manifest, output, config):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=Path(manifest).resolve()
    cfg=dict(config,dataset_manifest=str(manifest),dataset_manifest_sha256=sha256(manifest),
             H=config['history'],K_train=1,K_eval=1,train_stride=1,eval_stride=1,
             max_train_windows=None,max_eval_windows=None,
             history_protocol='independent causal window; current target; split-local history',
             optimizer='Adam',scheduler='ReduceLROnPlateau',selection_metric=(
                 'pooled_frame_node_mean_mm' if config.get('aggregation')=='pooled_frames' else 'sequence_macro_node_mean_mm'))
    state=dict(schema_version=2,study_id=config['study_id'],run_id=output.name,
               status='running',run_kind=cfg['run_kind'],dataset_manifest=str(manifest),
               dataset_manifest_sha256=sha256(manifest),seed=cfg['seed'],config='resolved_config.json',
               physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),command=config['command'],
               git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               expected_outputs=['best_eval_model.pt','history.json','COMPLETE'],
               selection={'role':'val','metric':cfg['selection_metric'],'direction':'min'})
    write_json(output/'run_manifest.json',state)
    write_json(output/'resolved_config.json',cfg)
    start=time.perf_counter()
    try:
        state['source_hashes']=_archive_code(output)
        meta,sequences=load_sequences(manifest,roles=('train','val'))
        train=[s for s in sequences if s['record']['role']=='train']
        val=[s for s in sequences if s['record']['role']=='val']
        cfg['dt']=meta['dt']
        _seed(cfg['seed']);torch.set_num_threads(cfg.get('threads',2))
        device=torch.device(cfg['device'])
        center_np,scale=fit_normalization(train)
        center=torch.tensor(center_np,device=device)
        fit_start=time.perf_counter()
        model,geometry=make_model(cfg['model'],cfg,train,(center_np,scale))
        state['model_initialization_and_prior_seconds']=time.perf_counter()-fit_start
        model.to(device)
        x,y,_=cache_windows(train,cfg['history'],device)
        vx,vy,vgroups=cache_windows(val,cfg['history'],device)
        if cfg.get('memory_readout_init',False):
            from src.benchmarks.modeling_memory_initialization import initialize_memory_readout
            init_start=time.perf_counter()
            state['memory_initialization']=initialize_memory_readout(
                model,x,y,ridge=cfg.get('memory_ridge',1e-3))
            state['memory_initialization_seconds']=time.perf_counter()-init_start
        y=(y-center)/scale
        parameters=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=cfg['lr']) if parameters else None
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=4,min_lr=1e-5) if optimizer else None
        closed_coefficients=model.coefficients.numel() if isinstance(model,Polynomial) else 0
        state.update(parameter_count=sum(p.numel() for p in model.parameters())+closed_coefficients,
                     trainable_parameter_count=sum(p.numel() for p in parameters),
                     closed_form_coefficients=closed_coefficients,prior_fit_frames=len(x) if geometry else 0,
                     supervised_windows=len(x),validation_windows=len(vx),fit_frames=sum(len(s['actions']) for s in train))
        fitted_reference_names={'reference_bend_bias','reference_bend_dirs','reference_length_bias',
                                'reference_length_dirs','reference_drive_weights'}
        state['fitted_reference_buffer_count']=sum(
            value.numel() for name,value in model.named_buffers()
            if name.rsplit('.',1)[-1] in fitted_reference_names)
        state['parameter_count_including_fitted_reference']=state['parameter_count']+state['fitted_reference_buffer_count']
        if isinstance(model,Polynomial):model.fit(x,y,cfg.get('ridge',1e-4))
        write_json(output/'resolved_config.json',cfg)
        write_json(output/'run_manifest.json',state)
        history=[];best=float('inf');best_epoch=0;stale=0;training_seconds=0.
        epochs=cfg['epochs'] if optimizer else 1
        for epoch in range(1,epochs+1):
            epoch_start=time.perf_counter();loss_sum=0.
            if optimizer:
                model.train();order=torch.randperm(len(x),device=device)
                for indices in order.split(cfg['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    prediction=model(x[indices]);target=y[indices]
                    loss=(prediction-target).square().mean()+cfg['endpoint_weight']*(prediction[:,-1]-target[:,-1]).square().mean()
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(parameters,10.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(indices)
            training_seconds+=time.perf_counter()-epoch_start
            if epoch % cfg['validation_interval'] and epoch not in (1,epochs):continue
            model.eval();errors=[]
            with torch.inference_mode():
                for lo in range(0,len(vx),cfg['batch_size']):
                    estimate=model(vx[lo:lo+cfg['batch_size']])*scale+center
                    error=torch.linalg.vector_norm(estimate-vy[lo:lo+cfg['batch_size']],dim=-1).mean(-1)
                    if not torch.isfinite(error).all():raise FloatingPointError('Nonfinite validation')
                    errors.append(error.cpu().numpy())
            errors=np.concatenate(errors)
            metric=validation_mean(errors,vgroups,cfg.get('aggregation','sequence_macro'))
            old_best=best
            if metric<best:
                best,best_epoch=metric,epoch
                torch.save(dict(schema='shape_modeling_checkpoint_v1',model=cfg['model'],state_dict=model.state_dict(),
                                config=cfg,geometry_config=geometry,center=center_np.tolist(),scale=scale,
                                selected_epoch=epoch,validation_node_mean_mm=best),output/'best_eval_model.pt')
            stale=0 if metric<old_best-cfg.get('min_delta_mm',.001) else stale+1
            history.append(dict(epoch=epoch,train_loss=loss_sum/len(x) if optimizer else None,
                                validation_node_mean_mm=metric,best_validation_node_mean_mm=best,
                                lr=optimizer.param_groups[0]['lr'] if optimizer else None,
                                epoch_seconds=time.perf_counter()-epoch_start,
                                cumulative_training_seconds=training_seconds,
                                elapsed_seconds=time.perf_counter()-start))
            if scheduler and cfg.get('schedule_lr',True):scheduler.step(metric)
            write_json(output/'history.json',history)
            state.update(current_epoch=epoch,best_epoch=best_epoch,best_validation_node_mean_mm=best,
                         wall_seconds=time.perf_counter()-start)
            write_json(output/'run_manifest.json',state)
            print(f"{output.name} epoch={epoch}/{epochs} val={metric:.5f} best={best:.5f}",flush=True)
            if optimizer and epoch%25==0:
                torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),
                                scheduler=scheduler.state_dict(),config=cfg),output/'training_state.pt')
            if epoch>=cfg.get('minimum_epochs',epochs) and stale>=cfg.get('early_stop_checks',100000):
                state['stop_reason']='validation_plateau';break
        recent=[r['validation_node_mean_mm'] for r in history[-3:]]
        trend=(recent[0]-min(recent))/max(abs(recent[0]),1e-8) if len(recent)>1 else 0.
        state.update(status='complete',wall_seconds=time.perf_counter()-start,
                     convergence={'recent_relative_improvement':trend,'checks':len(recent),
                                  'assessment':'still_improving' if trend>.02 else 'short_budget_flat_or_noisy'},
                     stop_reason=state.get('stop_reason','fixed_budget'),checkpoint_sha256=sha256(output/'best_eval_model.pt'))
        write_json(output/'run_manifest.json',state)
        (output/'COMPLETE').write_text('Training and validation selection completed.\n')
    except Exception:
        state.update(status='failed',error=traceback.format_exc());write_json(output/'run_manifest.json',state);raise
    return state
