#!/usr/bin/env python3
"""Prepare same-platform native-rate data, train HOV2.2, export GUI bundles.

Each invocation owns a new study directory. Existing source datasets stay read-only.
A failed preprocessing/QC step blocks that rate's release; no silent partial corpus.
"""
from __future__ import annotations
import argparse, concurrent.futures, datetime, hashlib, json, os, shlex, subprocess, sys, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'workspace/data/raw/real'
PROCESSED=ROOT/'workspace/data/processed/real'
SERIES={5:['172644','181044','181548','183351','183526','183547','183740','184036'],10:['182253','182519']}

def stamp():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def run(cmd,log,env=None):
    with Path(log).open('a') as f:
        f.write('\n'+stamp()+' '+shlex.join(map(str,cmd))+'\n');f.flush()
        subprocess.run(list(map(str,cmd)),cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def prepare(seq,gpu,study):
    raw_actions=np.loadtxt(RAW/seq/'actions6.csv',delimiter=',',skiprows=1)
    if len(raw_actions)<80 and np.max(np.abs(raw_actions[:,1:7]))<.01:
        return RAW/seq  # zero-only calibration, no trainable history/rollout split
    source=PROCESSED/(seq+'_n15_sam2_robot_mm')
    if (source/'dataset_manifest.json').is_file():return source
    target=study/'processed'/seq
    if (target/'dataset_manifest.json').exists():raise FileExistsError(target)
    config={'schema_version':1,'seq':str(RAW/seq),'roi':[220,68,300,300],
            'gpus':[gpu],'base_anchor':[370,110],'out_root':str(target)}
    cfg=study/'preprocessing'/f'{seq}.json';write(cfg,config)
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    # Parent training visibility must not renumber preprocessing GPUs.
    env.pop('CUDA_VISIBLE_DEVICES',None)
    run([sys.executable,'scripts/real/preprocess_capture.py','--config',cfg],cfg.with_suffix('.log'),env)
    return target

def materialize(rate,sources,study):
    target=study/f'dataset_{rate}hz';target.mkdir()
    for role in ['train','val']:(target/role).mkdir()
    records=[];excluded=[];window=20 if rate==5 else 40
    for source in sources:
        if source.parent==RAW:
            raw_actions=np.loadtxt(source/'actions6.csv',delimiter=',',skiprows=1)
            if len(raw_actions)>=80 or np.max(np.abs(raw_actions[:,1:7]))>=.01:raise ValueError('Unexpected calibration exclusion')
            excluded.append(dict(sequence=source.name,frames=len(raw_actions),reason='39-frame zero-only calibration: too short for disjoint history/rollout train and validation; raw images retained',source_actions=str(source/'actions6.csv'),source_sha256=sha(source/'actions6.csv')))
            continue
        manifest=source/'dataset_manifest.json';meta=json.loads(manifest.read_text())
        qc=meta.get('quality_control',{})
        ready=qc.get('training_ready',qc.get('automated_checks_passed',False))
        if not ready:raise ValueError(f'QC not ready: {manifest}')
        files=[*sorted((source/'train').glob('*.npz')),*sorted((source/'val').glob('*.npz'))]
        if len(files)!=2:raise ValueError(f'Expected complete original train+val: {source}')
        arrays=[]
        for p in files:
            with np.load(p,allow_pickle=False) as d:arrays.append({k:d[k].copy() for k in d.files})
        seq=source.name.split('_n15')[0]
        if not seq.startswith('seq_20260819_'):raise ValueError(seq)
        n=sum(len(d['positions']) for d in arrays);first=arrays[0]
        physical=np.concatenate([d['actions']*d['raw_action_scale6_kpa'] for d in arrays])
        camera=np.concatenate([d['positions_camera_px'] for d in arrays])
        expected=np.loadtxt(RAW/seq/'actions6.csv',delimiter=',',skiprows=1)[:,1:7]
        native=float(json.loads((RAW/seq/'meta.json').read_text())['action_interval_s'])
        if not np.isclose(native,1/rate):raise ValueError('Native sampling rate mismatch')
        if len(expected)!=n or not np.allclose(physical,expected,atol=.03):raise ValueError(f'Action/source mismatch {seq}')
        if not np.isfinite(camera).all() or not np.isfinite(physical).all():raise ValueError('Nonfinite arrays')
        if np.max(physical)>150.01 or np.min(physical)<-.01:raise ValueError('Outside current deployment pressure domain')
        if not np.allclose(physical[:,1],physical[:,2],atol=.5) or not np.allclose(physical[:,3],physical[:,4],atol=.5):raise ValueError('Mapping changed')
        # Keep one source-camera calibration for this fixed-camera collection day.
        positions=camera.copy();positions[:,0]=(camera[:,0]-370)*.8;positions[:,1]=(camera[:,1]-110)*.8;positions[:,2]=0
        frame=dict(origin_camera_px=[370.,110.],axial_axis_camera=[0.,1.],pixels_per_mm=1.25,
                   source='fixed_capture_day_calibration_370_110_1.25px_per_mm',frame_id='robot_planar_mm_v1',schema_version=1,
                   camera_to_model_matrix=[[.8,0,-296],[0,.8,-88],[0,0,1]],model_to_camera_matrix=[[1.25,0,370],[0,1.25,110],[0,0,1]],length_unit='mm')
        split=int(.8*n);val_start=split+window
        if split<2*window+1 or n-val_start<2*window+1:
            excluded.append(dict(sequence=seq,frames=n,reason='Too short for disjoint history and rollout train/validation; calibration-only sequence',source_manifest=str(manifest),source_manifest_sha256=sha(manifest)))
            if np.max(physical)>.01:raise ValueError(f'Nontrivial sequence cannot be silently excluded: {seq}')
            continue
        for role,start,stop in [('train',0,split),('val',val_start,n)]:
            d={k:v for k,v in first.items() if k not in ['positions','actions','positions_camera_px']}
            d.update(positions=positions[start:stop].astype('float32'),positions_camera_px=camera[start:stop],actions=(physical[start:stop]/150).astype('float32'),
                     raw_action_scale6_kpa=np.full(6,150,dtype='float32'),action_scale_kpa=np.full(4,150,dtype='float32'),skeleton_frame_transform=np.array(json.dumps(frame)),
                     source_frame_ids=np.arange(start,stop),source_sequence=np.array(seq),dt_nominal_s=np.array(1/rate))
            p=target/role/(seq+'.npz');np.savez_compressed(p,**d)
            records.append(dict(role=role,sequence=seq,start=start,stop=stop,frames=stop-start,path=str(p),sha256=sha(p),source_manifest=str(manifest),source_manifest_sha256=sha(manifest),source_files=[dict(path=str(x),sha256=sha(x)) for x in files]))
    write(target/'dataset_manifest.json',dict(schema='native_rate_hov_deployment_dataset_v1',created=stamp(),rate_hz=rate,dt=1/rate,
        evidence_level='within_sequence',purpose='deployment development; new physical trials required for independent test',
        split_policy='Each independent original sequence kept separate; first80% train, embargo=history window, final remainder validation; no cross-boundary context',
        grouping_key='original sequence; contiguous within-sequence development split',embargo_frames=window,
        action_scale_kpa=[150]*4,expansion6=[0,1,1,2,2,3],state_calibration=frame,
        source_raw_frames=sum(r['stop'] for r in records if r['role']=='val')+sum(x['frames'] for x in excluded),
        counts={role:sum(r['frames'] for r in records if r['role']==role) for role in ['train','val']},excluded=excluded,files=records))
    return target

def train(rate,study,gpu,prep_gpus,datasets_from=None):
    status=study/f'status_{rate}hz.json'
    try:
        write(status,dict(status='preprocessing',rate_hz=rate,started=stamp()))
        if datasets_from is not None:
            dataset=datasets_from/f'dataset_{rate}hz'
            manifest=json.loads((dataset/'dataset_manifest.json').read_text())
            if manifest['rate_hz']!=rate or not np.isclose(manifest['dt'],1/rate):raise ValueError('Dataset frequency mismatch')
            for record in manifest['files']:
                if sha(record['path'])!=record['sha256']:raise ValueError('Dataset hash mismatch')
        else:
            seqs=['seq_20260819_'+s for s in SERIES[rate]]
            def lane(items,g):return [prepare(seq,g,study) for seq in items]
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(prep_gpus)) as pool:
                futures=[pool.submit(lane,seqs[i::len(prep_gpus)],g) for i,g in enumerate(prep_gpus)]
                sources=[p for f in futures for p in f.result()]
            dataset=materialize(rate,sorted(sources),study)
        run_dir=study/f'training_{rate}hz';window=20 if rate==5 else 40
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(study/f'mpl_{rate}hz'))
        cmd=[sys.executable,'scripts/training/train_transition.py','--mode','hereditary_geo','--data_dir',dataset/'train','--val_dir',dataset/'val','--experiment-dir',run_dir,
             '--n_epochs','300','--batch_size','256','--num_workers','0','--episode_len',str(window),'--window_size',str(window),'--dt',str(1/rate),
             '--n_play','2','--n_maxwell','6','--tau_max','2.0','--burnin_mode','equilibrium','--n_bend_modes','14','--section_intervals','7,7',
             '--h0_reference','monotone_spline','--h0_knots','5','--h0_fit_steps','500','--h0_fit_objective','geometry','--h0_geometry_weight','1.0','--h0_endpoint_weight','.25',
             '--bend_basis_kind','local','--operator_drive_normalization','unit_range','--bend_loss_weight','.005','--length_loss_weight','.01','--endpoint_loss_weight','.25',
             '--hov21_residual','none','--validation_interval','5','--validation_max_steps','1000000','--validation_warmup','2','--scheduler_patience','4',
             '--save_interval','10','--eval_interval','0','--seed','42']
        (study/f'command_{rate}hz.sh').write_text(shlex.join(map(str,cmd))+'\n')
        write(status,dict(status='training',rate_hz=rate,started=stamp(),gpu=gpu,dataset=str(dataset),run=str(run_dir),command=list(map(str,cmd))))
        run(cmd,study/f'training_{rate}hz.log',env)
        checkpoint=run_dir/'phase_hereditary_geometry/model/best_eval_model.pt'
        if not checkpoint.exists():raise FileNotFoundError(checkpoint)
        bundle=study/f'deploy_{rate}hz'
        run([sys.executable,'scripts/evaluation/export_hereditary_deployment.py','--checkpoint',checkpoint,'--out',bundle,'--action-scale-kpa','150','150','150','150','--upper-kpa','150','150','150','150','--rate-kpa-s','50','50','50','50','--radius-mm','8','--max-horizon','80'],study/f'export_{rate}hz.log',env)
        # Self-contained immutable pair plus provenance. GUI reads npz + same-stem json.
        meta=json.loads((bundle/'hereditary.json').read_text());meta.update(initialization='Operator sets current pressure manually; align current full shape, estimate prior and warm model from ACK history; loading does not send pressure',
            dataset_manifest=str(dataset/'dataset_manifest.json'),dataset_manifest_sha256=sha(dataset/'dataset_manifest.json'),training_rate_hz=rate,evidence='full eligible native-rate corpus, within-sequence validation selection; not physical control certification')
        write(bundle/'hereditary.json',meta)
        cfg=json.loads((run_dir/'config.json').read_text());sel=cfg['phases'][0]['validation_selection']
        sys.path.insert(0,str(ROOT));from real_validation.runtime.hereditary_deployment import load_bundle
        engine,loaded=load_bundle(bundle/'hereditary.npz');assert np.isclose(loaded['dt'],1/rate)
        run([sys.executable,'-m','real_validation.tools.package_hereditary','--bundle',bundle/'hereditary.npz','--out',study/f'real_validation_{rate}hz.zip'],study/f'export_{rate}hz.log',env)
        write(status,dict(status='complete',rate_hz=rate,finished=stamp(),checkpoint=str(checkpoint),bundle=str(bundle/'hereditary.npz'),package=str(study/f'real_validation_{rate}hz.zip'),selection=sel,dataset=str(dataset),weights_sha256=sha(bundle/'hereditary.npz')))
    except Exception:
        write(status,dict(status='failed',rate_hz=rate,at=stamp(),error=traceback.format_exc()));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--rates',type=int,nargs='+',default=[5,10],choices=[5,10]);p.add_argument('--datasets-from',type=Path,help='Reuse immutable checked native-rate datasets from a previous study');a=p.parse_args()
    study=a.study.resolve();study.mkdir(parents=True,exist_ok=False)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (study/'code_diff.patch').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
    (study/'code_status.txt').write_bytes(subprocess.check_output(['git','status','--short'],cwd=ROOT))
    # The running orchestration source may be untracked; archive exact bytes.
    (study/'orchestrator.py').write_bytes(Path(__file__).read_bytes())
    write(study/'study_manifest.json',dict(study_id=study.name,run_kind='deployment_development',created=stamp(),git_commit=commit,git_dirty=bool((study/'code_status.txt').read_text().strip()),seed=42,rates=a.rates,epochs_max=300,batch_size=256,optimizer='Adam',lr=.001,selection='val node_mean_mm min',early_stopping='disabled; run all300epochs',datasets_from=str(a.datasets_from) if a.datasets_from else None,source_series=SERIES,checkpoint_export='validation-selected; no test consumed',expected_outputs=['status_5hz.json','status_10hz.json','deploy_5hz/hereditary.npz','deploy_10hz/hereditary.npz']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.rates)) as pool:
        fs=[pool.submit(train,rate,study,0 if rate==10 else 1,[2,3] if rate==5 else [0],a.datasets_from.resolve() if a.datasets_from else None) for rate in a.rates]
        outcomes=[]
        for f in fs:
            try:f.result();outcomes.append(True)
            except Exception:traceback.print_exc();outcomes.append(False)
    if not all(outcomes):raise SystemExit(1)
    (study/'COMPLETE').write_text(stamp()+'\n')
if __name__=='__main__':main()
