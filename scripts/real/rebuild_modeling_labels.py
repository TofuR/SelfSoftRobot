"""Rebuild seven 5-Hz recordings in a new directory; raw captures stay read-only.

Two persistent SAM2 workers share a queue of disjoint blocks. This avoids model
reload overhead and balances the long recording across GPUs. No model training.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
NAMES=['seq_20260819_'+n for n in ['172644','181044','181548','183351','183547','183740','184036']]

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def now():return datetime.now().astimezone().isoformat()

def run(command,log):
    with Path(log).open('a') as stream:
        stream.write('\nCOMMAND '+json.dumps(command)+'\n');stream.flush()
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def prepare(root,name):
    raw=ROOT/'workspace/data/raw/real'/name
    base=root/'sequences'/name;derived=base/'intermediate';log=root/'logs'/f'{name}_prepare.log'
    # Check all files, not only image counts; repeat at final acceptance.
    files={str(p.relative_to(raw)):digest(p) for p in sorted(raw.rglob('*')) if p.is_file()}
    write(base/'raw_hashes_before.json',files)
    config={'seq':str(raw),'roi':[220,68,300,300],'gpus':[2,3],
            'base_anchor':[370,110],'intermediate_root':str(derived),'out_root':str(base/'processed')}
    write(base/'config.json',config)
    commands=[
      [sys.executable,'scripts/real/audit_capture.py','--seq',str(raw),'--camera','cam0','--out',str(derived/'qc_capture')],
      [sys.executable,'scripts/real/crop_capture.py','--seq',str(raw),'--camera','cam0','--roi','220,68,300,300','--out-root',str(derived/'crop')],
      [sys.executable,'scripts/real/prepare_sam2_anchors.py','--seq',str(derived/'crop'),'--camera','cam0','--out-root',str(derived),'--chunk-size','200']]
    for command in commands:run(command,log)
    audit=json.loads((derived/'qc_capture/capture_audit.json').read_text())
    if not audit['ready_for_image_preprocessing']:raise RuntimeError(f'Capture audit failed: {name}')
    frames=sorted(int(p.stem) for p in (derived/'crop/cam0').glob('[0-9]*.png'))
    if frames!=list(range(len(frames))):raise ValueError('Noncontiguous original frames')
    (derived/'sam2_masks').mkdir()
    write(base/'preparation.json',{'status':'complete','frames':len(frames),'commands':commands,'completed_at':now()})
    return name,len(frames)

def gpu_worker(gpu,root,work,progress):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    log=(Path(root)/'logs'/f'gpu{gpu}.log').open('w',buffering=1)
    sys.stdout=log;sys.stderr=log
    try:
        spec=importlib.util.spec_from_file_location('sam2_full_rebuild',ROOT/'sam2/segment_video_full.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from src.registry import ProjectPaths
        paths=ProjectPaths.load()
        ckpt=paths.pretrained_model_dir('sam2')/'sam2.1_hiera_tiny.pt'
        if ckpt.is_file():module.CKPT=str(ckpt)
        predictor=module.build_predictor('cuda:0')
        jpeg=module.create_jpeg_workspace(paths.workspace_root/'cache/sam2_jpeg','rebuild',gpu)
        config={'checkpoint_sha256':digest(module.CKPT),'config':module.CONFIG_FILE,
                'script_sha256':digest(module.__file__),'orchestrator_sha256':digest(__file__),
                'base_trim_sha256':digest(ROOT/'scripts/real/prepare_sam2_anchors.py')}
        manifests={};medians={}
        while True:
            task=work.get()
            if task is None:break
            name,start,stop=task
            base=Path(root)/'sequences'/name/'intermediate'
            if name not in manifests:
                manifests[name]=module.load_anchor_manifest(base/'anchor_manifest.csv')
                medians[name]=module.global_median_area(str(base/'masks_candidate'),stop+1)
            areas=module.process_chunk(predictor,str(base/'crop/cam0'),str(base/'masks_candidate'),
                str(base/'sam2_masks'),jpeg,start,stop,medians[name],anchor_manifest=manifests[name],
                inference_config=config,trim_base_attachment=True)
            if areas is None:raise RuntimeError(f'Failed block: {task}')
            progress.put({'status':'block_complete','sequence':name,'start':start,'stop':stop,'gpu':gpu,'time':now()})
        import shutil
        shutil.rmtree(jpeg)
        progress.put({'status':'worker_complete','gpu':gpu})
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        progress.put({'status':'failed','gpu':gpu,'error':error});raise
    finally:log.close()

def skeleton(root,name):
    base=root/'sequences'/name
    run([sys.executable,'scripts/real/preprocess_capture.py','--config',str(base/'config.json'),
         '--stages','skeleton'],root/'logs'/f'{name}_skeleton.log')
    return name

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--gpus',default='2,3')
    args=parser.parse_args();root=args.root.resolve();gpus=[int(g) for g in args.gpus.split(',')]
    if (root/'REBUILD_STARTED').exists():raise FileExistsError('Use a new root; partial artifacts retained for audit')
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    (root/'REBUILD_STARTED').write_text(now()+'\n')
    os.environ.update(MPLCONFIGDIR='/tmp/selfsr-relabel-mpl',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    write(root/'plan.json',{'sequences':NAMES,'gpus':gpus,'created_at':now(),'raw_read_only':True,
          'roi':[220,68,300,300],'node_order':'base_to_tip','nodes':15,'source_hashes':{
          p:digest(ROOT/p) for p in ['sam2/segment_video_full.py','scripts/real/preprocess_capture.py',
          'scripts/real/prepare_sam2_anchors.py','scripts/real/masks_to_transition_npz.py',
          'scripts/real/rebuild_modeling_labels.py']}})
    try:
        write(root/'status.json',{'status':'running','phase':'prepare_images_anchors','updated_at':now()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            counts=dict(pool.map(lambda name:prepare(root,name),NAMES))
        tasks=[]
        # Round robin distributes short and long recordings over the same two workers.
        for start in range(0,max(counts.values()),200):
            for name in NAMES:
                if start<counts[name]:tasks.append((name,start,min(start+199,counts[name]-1)))
        ctx=mp.get_context('spawn');work=ctx.Queue();progress=ctx.Queue()
        for task in tasks:work.put(task)
        for _ in gpus:work.put(None)
        processes=[ctx.Process(target=gpu_worker,args=(gpu,str(root),work,progress)) for gpu in gpus]
        for p in processes:p.start()
        done=[]
        while len(done)<len(tasks):
            try:event=progress.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode not in (None,0) for p in processes):raise RuntimeError('GPU worker died; see logs')
                continue
            if event['status']=='failed':raise RuntimeError(event['error'])
            if event['status']=='block_complete':
                done.append(event)
                write(root/'status.json',{'status':'running','phase':'sam2','completed_blocks':len(done),'total_blocks':len(tasks),'updated_at':now(),'last_completed':event})
                write(root/'blocks.json',done)
                print(f'SAM2 {len(done)}/{len(tasks)} {event["sequence"]} {event["start"]}',flush=True)
        for p in processes:
            p.join()
            if p.exitcode!=0:raise RuntimeError('GPU worker failed')
        for name in NAMES:
            write(root/'sequences'/name/'intermediate/sam2_masks/run_meta_rebuild.json',{
                  'schema_version':2,'sequence':name,'input_sequence_path':str(root/'sequences'/name/'intermediate/crop'),
                  'input_frame_count':counts[name],'input_image_size_wh':[300,300],'camera':'cam0',
                  'gpus':gpus,'temporary_jpeg_isolation':'unique_directory_per_worker','block_provenance':'provenance/*.json',
                  'trim_base_attachment':True,'completed_at':now()})
        write(root/'status.json',{'status':'running','phase':'skeleton','updated_at':now()})
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name in pool.map(lambda name:skeleton(root,name),NAMES):print('Skeleton complete:',name,flush=True)
        write(root/'status.json',{'status':'awaiting_image_review','phase':'generated','frames':sum(counts.values()),'counts':counts,'updated_at':now()})
    except Exception:
        write(root/'status.json',{'status':'failed','updated_at':now(),'error':traceback.format_exc()});raise

if __name__=='__main__':main()
