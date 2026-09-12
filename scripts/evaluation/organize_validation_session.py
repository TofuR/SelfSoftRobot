#!/usr/bin/env python3
"""Audit and copy complete physical validation trials into a named workspace archive."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

def portable_refs(value):
    if isinstance(value, dict):return {k:portable_refs(v) for k,v in value.items()}
    if isinstance(value, list):return [portable_refs(v) for v in value]
    if isinstance(value, str) and value.startswith(str(REPO)+"/"):return value[len(str(REPO))+1:]
    return value


def read_text(path):
    data = path.read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f'Unsupported text encoding: {path}')


def read_json(path): return json.loads(read_text(path))
def read_csv(path): return list(csv.DictReader(io.StringIO(read_text(path))))
def read_jsonl(path): return [json.loads(s) for s in read_text(path).splitlines() if s.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path, value): path.write_text(json.dumps(portable_refs(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stats(values):
    a = np.array([float(v) for v in values if v not in (None, '')], dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), median=float(np.median(a)), p95=float(np.percentile(a,95)), maximum=float(a.max())) if len(a) else dict(n=0,median=None,p95=None,maximum=None)


def number(value):
    try:
        v=float(value)
        return v if np.isfinite(v) else None
    except (ValueError,TypeError): return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--analysis-out',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve();analysis=a.analysis_out.resolve()
    if out.exists():raise FileExistsError(out)
    annotations=read_json(a.annotations)
    records=sorted([(read_json(d/'metadata.json')['started'],d,read_json(d/'metadata.json')) for d in (source/'executions').iterdir() if d.is_dir()])
    clear=set(annotations['clear_trials']);occluded=set(annotations['physical_occlusion_trials'])
    if annotations.get('source_session_id')!=source.name or clear & occluded or clear | occluded != set(range(1,len(records)+1)):
        raise ValueError('Visual annotations must cover this session exactly once')
    events=read_jsonl(source/'events.jsonl');begins={r['execution_id']:r for r in events if r['event']=='execute_begin'}
    failures={r['execution_id']:r['error'] for r in events if r['event']=='execute_failed'}
    modelmeta=next(r['metadata'] for r in events if r['event']=='loaded')
    rows=[];target_groups={};details={}
    for ordinal,(_,d,m) in enumerate(records,1):
        allcommands=read_csv(d/'commands.csv');commands=[r for r in allcommands if r['step'].isdigit()];samples=read_csv(d/'samples.csv');ndi=read_csv(d/'ndi.csv');timings=read_csv(d/'timings.csv');steps=read_jsonl(d/'steps.jsonl')
        jobs=[read_json(f) for f in sorted((d/'feedback_jobs').glob('*.json'))]
        with np.load(d/'initial_plan.npz',allow_pickle=False) as plan:
            arrays={k:plan[k].copy() for k in plan.files}
        goal=arrays['goal_mm'];ids=arrays['node_indices'];matrix=arrays['camera_matrix']
        goalpx=goal@matrix[:2,:2].T+matrix[:2,2];total=len(arrays['actions_model'])
        kind='tip' if ids.tolist()==[len(goal)-1] else ('full' if set(ids)==set(range(1,len(goal))) or set(ids)==set(range(len(goal))) else 'segment')
        dx=float(goalpx[-1,0]-goalpx[0,0]);direction='left' if dx < -2 else ('right' if dx > 2 else 'center')
        signature=hashlib.sha256(np.round(goalpx[ids],3).tobytes()+ids.tobytes()).hexdigest()
        group=target_groups.setdefault(signature,f'G{len(target_groups)+1:02d}')
        occlusion='occluded' if ordinal in annotations['physical_occlusion_trials'] else ('clear' if ordinal in annotations['clear_trials'] else 'unreviewed')
        loop='closed' if m['correction_enabled'] else 'open';interval=m['feedback_interval_steps'];hz=round(1/m['control_dt'])
        name=f't{ordinal:02d}_{kind}_{direction}_{occlusion}_{loop}_{hz}hz_i{interval}'
        problems=[]
        if [int(r['step']) for r in commands]!=list(range(total)):problems.append('command_count_or_sequence')
        if any(r['status']!='ack' for r in commands):problems.append('non_ack_command')
        if [int(r['step']) for r in steps]!=list(range(total)):problems.append('step_count_or_sequence')
        if len(timings)!=total:problems.append('timing_count')
        selected=[r for r in samples if int(r['camera'])==m['selected_camera']]
        if [int(r['step']) for r in selected]!=list(range(total)):problems.append('image_count_or_sequence')
        cids={int(r['step']):r['command_id'] for r in commands}
        shapes=set();images=[]
        for s in samples:
            if s['command_id']!=cids.get(int(s['step'])):problems.append('image_command_link')
            for field in ('raw_path','feedback_path'):
                if not s[field]:problems.append('image_path_empty');continue
                f=(d/s[field]).resolve()
                if not f.is_relative_to(d.resolve()) or not f.is_file():problems.append('missing_image');continue
                image=cv2.imread(str(f))
                if image is None:problems.append('unreadable_image')
                else:shapes.add(image.shape)
            if s in selected:images.append(d/s['raw_path'])
        times=np.array([float(r['t_command']) for r in commands])
        if len(times)>1 and np.any(np.diff(times)<=0):problems.append('nonmonotonic_commands')
        frame_times=np.array([float(s['frame_timestamp']) for s in selected])
        if len(frame_times)>1 and np.any(np.diff(frame_times)<=0):problems.append('nonmonotonic_images')
        if any(s['fresh_after_command']!='True' for s in selected):problems.append('stale_image')
        if len(shapes)!=1:problems.append('changing_image_size')
        if m['hardware']['profile']['valve_backend']!='real' or m['hardware']['profile']['camera_backend']!='real':problems.append('not_real_hardware')
        selected_for_archive=m['status']=='completed' and not problems
        completion=m.get('completion_assessment',{})
        probe_summary={}
        for probe in sorted({r['probe'] for r in ndi}):
            values=[r for r in ndi if r['probe']==probe];valid=[r for r in values if r['valid']=='True' and all(number(r[k]) is not None for k in ('x_mm','y_mm','z_mm'))]
            start=[r for r in valid if float(r['timestamp'])<m['started']+.3];end=[r for r in valid if float(r['timestamp'])>m['ended']-.3]
            positions=lambda x:np.median([[float(r[k]) for k in ('x_mm','y_mm','z_mm')] for r in x],axis=0)
            delta=positions(end)-positions(start) if start and end else None
            probe_summary[probe]=dict(rows=len(values),valid_rows=len(valid),valid_fraction=len(valid)/len(values),start_window_count=len(start),end_window_count=len(end),delta_mm=None if delta is None else delta.tolist(),displacement_mm=None if delta is None else float(np.linalg.norm(delta)))
        stages={k:stats(t.get(k) for t in timings) for k in ('send_wait_ms','ack_ms','frame_wait_ms','dispatch_wait_ms','ack_delivery_ms','image_write_ms')}
        stages['image_write_ms']=stats(r.get('image_write_ms') for r in samples)
        stages['job_wall_ms']=stats(j.get('wall_ms') for j in jobs)
        for k in ('edge_ms','observer_ms','compute_ms'):
            stages[k]=stats(j.get('diagnostics',{}).get(k) for j in jobs)
        for k in ('derivative_ms','solver_ms','validation_ms','time_ms'):
            stages['suffix_'+k]=stats(j.get('diagnostics',{}).get('control',{}).get(k) for j in jobs)
        stages['command_interval_ms']=stats(np.diff(times)*1000)
        applications=[r for r in events if r['event']=='delayed_feedback_application' and r.get('execution_id')==d.name]
        commits=sum(s.get('state_committed') is True for s in steps)
        begin=begins.get(d.name,{})
        row=dict(trial=ordinal,name=name,execution_id=d.name,status=m['status'],archived=selected_for_archive,target_group=group,target_kind=kind,direction=direction,occlusion=occlusion,correction=loop,control_hz=hz,feedback_interval=interval,correction_hz=hz/interval if m['correction_enabled'] else 0.,
                 started=m['started'],local_time=datetime.fromtimestamp(m['hardware']['timestamp'],ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),planned_steps=total,commands=len(commands),zero_commands=sum(r['step']=='zero' for r in allcommands),frames=len(selected),primary_steps=m['primary_steps'],reserve_steps=m['reserve_steps'],
                 state_commits_in_steps=commits,revision_status_counts=dict(Counter(s['revision_status'] for s in steps)),feedback_jobs=len(jobs),job_errors=sum(j.get('error') is not None for j in jobs),
                 planned_mean_error_mm=m.get('planned_mean_error'),estimated_final_mean_error_mm=completion.get('estimated_mean_error_mm'),estimated_within_tolerance=completion.get('estimate_within_tolerance'),latest_image_supported=completion.get('latest_image_supported'),
                 physical_arrival_verified=False,planning_ms=begin.get('planning_ms'),target_tip_relative_base_px=dx,pressure_mapping=arrays['expansion6'].tolist(),initial_action_model=arrays['execution_action'].tolist(),initial_memory_l2=float(np.linalg.norm(arrays['execution_state'])),
                 command_interval_median_ms=stages['command_interval_ms']['median'],command_interval_p95_ms=stages['command_interval_ms']['p95'],ack_median_ms=stages['ack_ms']['median'],job_median_ms=stages['job_wall_ms']['median'],job_p95_ms=stages['job_wall_ms']['p95'],
                 ndi_probe0_valid=probe_summary.get('0',{}).get('valid_fraction'),ndi_probe1_valid=probe_summary.get('1',{}).get('valid_fraction'),ndi_probe0_displacement_mm=probe_summary.get('0',{}).get('displacement_mm'),ndi_probe1_displacement_mm=probe_summary.get('1',{}).get('displacement_mm'),
                 quality_problems=sorted(set(problems)),failure_reason=failures.get(d.name),source_execution=str(d),archive_relative='trials/'+name if selected_for_archive else None)
        details[d.name]=dict(metadata=m,timing_stages=stages,ndi=probe_summary,scheduled_application_status=dict(Counter(r.get('revision_status') for r in applications)),frame_times_span=[float(frame_times[0]),float(frame_times[-1])] if len(frame_times) else [],annotations=annotations.get('notes',{}).get(str(ordinal)))
        rows.append(row)
        # Compact visual evidence: early / final frame with only the specified goal overlaid.
        analysis.mkdir(parents=True,exist_ok=True);(analysis/'previews').mkdir(exist_ok=True)
        parts=[]
        for f in (images[0],images[-1]):
            im=cv2.imread(str(f));cv2.polylines(im,[goalpx[ids].astype('int32')],False,(0,0,255),2)
            for v in goalpx[ids]:cv2.circle(im,tuple(v.astype(int)),3,(0,0,255),-1)
            parts.append(cv2.resize(im,(400,300)))
        cv2.imwrite(str(analysis/'previews'/f't{ordinal:02d}.jpg'),np.concatenate(parts,axis=1))
    # Selection is based on complete execution evidence, never the model's reported target error.
    out.mkdir(parents=True,exist_ok=False);(out/'trials').mkdir();(out/'session').mkdir()
    checks=[]
    def copy_checked(src,dst):
        before=sha(src);size=src.stat().st_size
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        if sha(dst)!=before or sha(src)!=before:raise RuntimeError('Copy/source stability check failed: '+str(src))
        checks.append(dict(source=str(src.relative_to(source)),destination=str(dst.relative_to(out)),bytes=size,sha256=before))
    for f in source.iterdir():
        if f.is_file():copy_checked(f,out/'session'/f.name)
    for row in rows:
        if not row['archived']:continue
        d=source/'executions'/row['execution_id'];dest=out/row['archive_relative']
        for f in d.rglob('*'):
            if f.is_file():copy_checked(f,dest/f.relative_to(d))
        write_json(dest/'trial_info.json',dict(row,details=details[row['execution_id']]))
    manifest=dict(schema='real_validation_archive_v1',source_session=str(source),study_id=out.name,
                  source_session_id=source.name,operation='verified_copy_originals_retained',selection='status completed AND complete ordered ACK/step/image records; not target-arrival selection',
                  counts=dict(total=len(rows),archived=sum(r['archived'] for r in rows)),controller_metadata=modelmeta,annotations=annotations,
                  trials=rows,copied_files=checks)
    write_json(out/'manifest.json',manifest);write_json(analysis/'inventory.json',dict(trials=rows,details=details))
    columns=[k for k in rows[0] if k not in ('revision_status_counts','initial_action_model')]
    with (out/'trial_index.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in portable_refs(row).items()})
    write_json(out/'copy_verification.json',dict(files=len(checks),bytes=sum(c['bytes'] for c in checks),all_sha256_match=True,originals_retained=True))
    (out/'COMPLETE').write_text('Audit and verified copy complete; physical arrival evaluation remains separate.\n')
    print(json.dumps(dict(out=str(out),trials=len(rows),archived=sum(r['archived'] for r in rows),files=len(checks),bytes=sum(c['bytes'] for c in checks)),indent=2))


if __name__=='__main__':main()
