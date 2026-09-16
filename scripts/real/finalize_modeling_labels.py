"""Image QA, explicit base-boundary normalization and release of rebuilt labels.

SAM2 outputs remain in intermediate/sam2_masks. The reviewed mask version is
written separately; all edits have pixel counts and parent fingerprints.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/selfsr-relabel-mpl')
import cv2
import numpy as np
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.real.rebuild_modeling_labels import NAMES, digest, write


def normalize_base(mask, image, base_xy=(150,42), max_gap=30):
    """Limit annotation to the known base section; fill visible short base gaps.

    Only the fixed attachment neighborhood is modified. No pressure, temporal
    label or prediction is used. Image appearance must support every added pixel.
    """
    m=(mask>0).astype(np.uint8);before=m.copy();x,y=map(int,base_xy)
    m[:y]=0
    rows=np.flatnonzero(m.any(1))
    if not len(rows):raise ValueError('Empty body below base')
    top=int(rows[0]);gap=top-y
    if gap>max_gap:raise ValueError(f'Base gap {gap}px exceeds inspected repair scope')
    width=[];centers=[]
    sample_top=max(top,y+10)
    for yy in range(sample_top,min(sample_top+15,len(m))):
        cols=np.flatnonzero(m[yy])
        if len(cols)>8:width.append(cols[-1]-cols[0]+1);centers.append((cols[0]+cols[-1])/2)
    if not width:raise ValueError('No reliable base-body cross section')
    half=float(np.median(width))/2
    if not 6<=half<=15:raise ValueError(f'Unreliable body half-width {half}')
    join=min(max(top+8,y+15),len(m)-1);center=float(np.median(centers))
    polygon=np.round([[x-half,y],[x+half,y],[center+half,join],[center-half,join]]).astype(np.int32)
    patch=np.zeros_like(m);cv2.fillConvexPoly(patch,polygon,1)
    # A thin bracket rim can remain below the base row. Limit its first 10 rows
    # to the arm cross section measured farther down the visible body.
    m[y:y+10]*=patch[y:y+10]
    if gap>0 or not m[y,x]:
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        visible=(hsv[:,:,1]<=110)&(hsv[:,:,2]>=100)
        patch[~visible]=0
        m=np.maximum(m,patch)
    return m, {'original_top_row':top,'base_gap_px':gap,
               'added_pixels':int(((m>0)&(before==0)).sum()),
               'removed_pixels':int(((m==0)&(before>0)).sum())}


def remove_remote_thin_components(mask):
    """Discard detached thin cable speckles far from any arm-sized cross section.

    Keep narrow visible arm strips near the main body and both sides of a cable
    occlusion. This changes no connected arm component and fills no mask pixels.
    """
    m=(mask>0).astype(np.uint8)
    thick=cv2.distanceTransform(m,cv2.DIST_L2,5)>=4
    if not thick.any():raise ValueError('No arm-width foreground support')
    distance=cv2.distanceTransform((~thick).astype(np.uint8),cv2.DIST_L2,5)
    count,labels=cv2.connectedComponents(m,connectivity=8)
    result=m.copy()
    for label in range(1,count):
        region=labels==label
        if float(distance[region].min())>12:result[region]=0
    return result,int(m.sum()-result.sum())


def checked_sequence(root,name):
    base=root/'sequences'/name;source=base/'intermediate';final=base/'intermediate_verified'
    raw=ROOT/'workspace/data/raw/real'/name
    before=json.loads((base/'raw_hashes_before.json').read_text())
    current={str(p.relative_to(raw)):digest(p) for p in sorted(raw.rglob('*')) if p.is_file()}
    if before!=current:raise RuntimeError(f'Raw data changed: {name}')
    receipts=sorted((source/'sam2_masks/provenance').glob('chunk_*.json'))
    expected_images={};expected_masks={}
    for path in receipts:
        r=json.loads(path.read_text());expected_images.update(r['source']['image_hashes']);expected_masks.update(r['outputs'])
    count=len(list((raw/'cam0').glob('[0-9]*.png')))
    if set(expected_images)!=set(map(str,range(count))) or set(expected_masks)!=set(expected_images):
        raise RuntimeError('Incomplete inference provenance')
    final.mkdir(exist_ok=False);(final/'sam2_masks').mkdir()
    # These links reference this NEW version's independently rebuilt assets.
    for key in ['crop','qc_capture','masks_candidate','anchor_manifest.csv','candidate_summary.json','bg_median.png']:
        (final/key).symlink_to(source/key,target_is_directory=(source/key).is_dir())
    override_path=base/'image_repairs/overrides.json'
    overrides=json.loads(override_path.read_text()) if override_path.exists() else {}
    records=[]
    for f in range(count):
        raw_file=raw/'cam0'/f'{f:05d}.png';crop_file=source/'crop/cam0'/f'{f:05d}.png'
        sam_file=source/'sam2_masks'/f'{f:05d}.png'
        if digest(crop_file)!=expected_images[str(f)] or digest(sam_file)!=expected_masks[str(f)]:
            raise RuntimeError(f'Inference source/output changed: {name} {f}')
        original=cv2.imread(str(raw_file));crop=cv2.imread(str(crop_file));sam=cv2.imread(str(sam_file),0)
        if not np.array_equal(original[68:368,220:520],crop):raise RuntimeError(f'Crop differs from raw frame: {name} {f}')
        annotation=sam;override=overrides.get(str(f))
        if override:
            replacement=base/override['path']
            if digest(replacement)!=override['sha256'] or before[f'cam0/{f:05d}.png']!=override['raw_sha256']:
                raise ValueError('Image-reviewed override source mismatch')
            annotation=cv2.imread(str(replacement),0)
        m,detail=normalize_base(annotation,crop)
        m,removed=remove_remote_thin_components(m)
        detail.update(remote_thin_pixels_removed=removed,
                      override_mask_sha256=override['sha256'] if override else '',
                      override_reason=override['reason'] if override else '')
        target=final/'sam2_masks'/f'{f:05d}.png'
        if not cv2.imwrite(str(target),m*255):raise OSError(str(target))
        core=cv2.erode(m,np.ones((5,5),np.uint8))>0
        hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);fg=(hsv[:,:,1]<=110)&(hsv[:,:,2]>=100)
        support=float(fg[core].mean()) if core.any() else 0.
        candidate=cv2.imread(str(source/'masks_candidate'/f'{f:05d}.png'),0)>0
        a=m[65:270,90:240]>0;b=candidate[65:270,90:240]
        iou=float((a&b).sum()/max(1,(a|b).sum()))
        components,labels,stats,centroids=cv2.connectedComponentsWithStats(m,connectivity=8)
        largest=float(stats[1:,cv2.CC_STAT_AREA].max()/max(1,m.sum())) if components>1 else 0.
        records.append(dict(frame=f,raw_sha256=before[f'cam0/{f:05d}.png'],sam2_sha256=expected_masks[str(f)],
            mask_sha256=digest(target),raw_foreground_support=support,candidate_body_iou=iou,
            largest_component_fraction=largest,area_px=int(m.sum()),**detail))
    write(final/'sam2_masks/base_boundary_provenance.json',{'operation':'fixed_base_section_with_image_supported_short_gap_fill',
          'base_xy_crop':[150,42],'maximum_gap_px':30,'source':str(source/'sam2_masks'),
          'code_sha256':digest(__file__),'frames':records})
    with (base/'image_qa.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    flagged=[r['frame'] for r in records if r['raw_foreground_support']<.95 or r['largest_component_fraction']<.99]
    write(base/'image_qa.json',{'frames':count,'raw_files_checked':len(current),'raw_unchanged':True,
          'all_crops_match_raw':True,'all_inference_provenance_verified':True,
          'foreground_support_min':min(r['raw_foreground_support'] for r in records),
          'foreground_support_mean':float(np.mean([r['raw_foreground_support'] for r in records])),
          'flagged_frames':flagged,'threshold_note':'Appearance/connectedness are screening flags, followed by raw image review; they are not independent segmentation ground truth.'})
    config=json.loads((base/'config.json').read_text());config.update(intermediate_root=str(final),out_root=str(base/'processed_verified'))
    write(base/'verified_config.json',config)
    with (root/'logs'/f'{name}_verified_skeleton.log').open('w') as stream:
        subprocess.run([sys.executable,'scripts/real/preprocess_capture.py','--config',str(base/'verified_config.json'),
                        '--stages','skeleton'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,OPENCV_FOR_THREADS_NUM='1'))
    geometry_qa(base)
    return name


def geometry_qa(base):
    """Check saved node geometry against its own image mask, retaining flags."""
    processed=base/'processed_verified'
    parts=[]
    for role in ('train','val'):
        with np.load(next((processed/role).glob('*.npz'))) as z:
            if str(z['node_order'])!='base_to_tip':raise ValueError('Wrong node order')
            parts.append(z['positions_camera_px'].transpose(0,2,1))
    camera=np.concatenate(parts)
    if camera.shape[1:]!=(15,3) or not np.isfinite(camera).all():
        raise ValueError('Invalid node geometry')
    np.testing.assert_allclose(camera[:,0,:2],np.broadcast_to([370,110],camera[:,0,:2].shape),atol=.01)
    with (processed/'qc_skeleton/skeleton_metrics.csv').open() as stream:
        extraction=list(csv.DictReader(stream))
    if len(extraction)!=len(camera):raise ValueError('Skeleton/frame mismatch')
    repairs=[i for i,r in enumerate(extraction) if r.get('interpolated','').lower() in ('true','1')]
    suspicious=[i for i,r in enumerate(extraction) if r.get('suspicious','').lower() in ('true','1')]
    records=[]
    for f,xy in enumerate(camera[:,:,:2]-[220,68]):
        m=cv2.imread(str(base/'intermediate_verified/sam2_masks'/f'{f:05d}.png'),0)>0
        distance=cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5)
        pixels=np.rint(xy).astype(int)
        if ((pixels<0)|(pixels>=300)).any():raise ValueError(f'Nodes outside crop: {f}')
        outside=distance[pixels[:,1],pixels[:,0]]
        lengths=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        records.append(dict(frame=f,max_node_outside_mask_px=float(outside.max()),
                            tip_outside_mask_px=float(outside[-1]),arc_length_px=float(lengths.sum())))
    with (base/'geometry_qa.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(base/'geometry_qa.json',dict(frames=len(camera),nodes=15,finite=True,base_anchor_verified=True,
          interpolated_frames=repairs,temporal_suspicious_frames=suspicious,
          flagged_frames=[r['frame'] for r in records if r['max_node_outside_mask_px']>2],
          max_node_outside_mask_px=max(r['max_node_outside_mask_px'] for r in records)))


def package(root):
    from src.benchmarks.modeling_data import prepare_temporal_pool
    # Inherit only the fixed capture-day calibration and schema, never old labels.
    old=ROOT/'workspace/runs/training/modeling_5hz_622_20260912_000/data/dataset_manifest.json'
    template=json.loads(old.read_text());out=root/'full_sequences';out.mkdir(exist_ok=False);records=[]
    for name in NAMES:
        base=root/'sequences'/name;processed=base/'processed_verified';parts=[];sources=[]
        for role in ('train','val'):
            path=next((processed/role).glob('*.npz'))
            with np.load(path) as z:parts.append({k:z[k] for k in z.files})
            sources.append({'path':str(path),'sha256':digest(path)})
        physical=np.concatenate([p['actions']*p['raw_action_scale6_kpa'] for p in parts])
        camera=np.concatenate([p['positions_camera_px'] for p in parts]).transpose(0,2,1)
        raw=ROOT/'workspace/data/raw/real'/name;commands=np.loadtxt(raw/'actions6.csv',delimiter=',',skiprows=1)
        np.testing.assert_allclose(physical,commands[:,1:],atol=.03)
        calibration=template['calibration'];mat=np.asarray(calibration['camera_to_model_matrix'])
        xy1=np.concatenate([camera[:,:,:2],np.ones((*camera.shape[:2],1))],axis=2)@mat.T
        positions=np.concatenate([xy1[:,:,:2]/xy1[:,:,2:],np.zeros((*camera.shape[:2],1))],axis=2).astype('float32')
        transform=np.array([[1,0,-220],[0,1,-68],[0,0,1]])@np.asarray(calibration['model_to_camera_matrix'])
        p=out/f'{name}.npz'
        np.savez_compressed(p,actions=(physical[:,[0,1,3,5]]/150).astype('float32'),positions=positions,
                            frame_ids=np.arange(len(positions)),timestamps=commands[:,0],model_to_mask=transform)
        masks=base/'intermediate_verified/sam2_masks';inventory=out/f'{name}_masks.json'
        write(inventory,[{'frame':f,'sha256':digest(masks/f'{f:05d}.png')} for f in range(len(positions))])
        crop_meta=base/'intermediate_verified/crop/crop_meta.json'
        records.append({'group':name,'role':'train','path':p.name,'sha256':digest(p),'frames':len(positions),
            'masks':str(masks),'mask_shape':[300,300],'mask_inventory':inventory.name,'mask_inventory_sha256':digest(inventory),
            'crop_meta':str(crop_meta),'crop_meta_sha256':digest(crop_meta),'source_files':sources,
            'source_manifest':str(processed/'dataset_manifest.json'),'timing_jitter_std_s':float(np.diff(commands[:,0]).std())})
    for key in ['counts','scored_counts','split_ratio','history_context']:template.pop(key,None)
    template.update(schema='shape_modeling_grouped_v1',dataset_id=root.name,evidence_level='within_sequence',
        study_stage='rebuilt_labels_pending_visual_acceptance',split_policy='full original recordings before temporal split',
        source_manifest=str(root/'plan.json'),source_manifest_sha256=digest(root/'plan.json'),files=records)
    write(out/'dataset_manifest.json',template)
    split=prepare_temporal_pool(out/'dataset_manifest.json',root/'data',history=20)
    # Prove that relabeling preserved the previously frozen frame partition.
    new=json.loads(split.read_text());old_data=json.loads(old.read_text())
    new.update(dataset_id=root.name,study_stage='rebuilt_labels_pending_visual_acceptance')
    write(split,new)
    for entry in new['files']:
        prior=next(r for r in old_data['files'] if r['group']==entry['group'] and r['role']==entry['role'])
        assert (entry['start'],entry['stop'])==(prior['start'],prior['stop'])
    write(root/'package_qa.json',{'status':'awaiting_visual_acceptance','manifest':str(split),
          'counts':new['counts'],'scored_counts':new['scored_counts'],'frame_partition_matches_previous_protocol':True})


def contact_sheets(root,names=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selection_file=root/'qc/selected_frames.json'
    selection_file.parent.mkdir(parents=True,exist_ok=True)
    all_selected=json.loads(selection_file.read_text()) if selection_file.exists() else {}
    for name in names or NAMES:
        base=root/'sequences'/name
        with (base/'image_qa.csv').open() as stream:rows=list(csv.DictReader(stream))
        camera=[]
        for role in ('train','val'):
            with np.load(next((base/'processed_verified'/role).glob('*.npz'))) as z:
                camera.append(z['positions_camera_px'][:,:2,:].transpose(0,2,1))
        xy=np.concatenate(camera)-[220,68];n=len(rows)
        selected=set([0,n-1,int(n*.6)-1,int(n*.6),int(n*.8)-1,int(n*.8)])
        # A raw-image sample in EVERY inference block, all appearance flags, and difficult tails.
        selected.update(min(start+100,n-1) for start in range(0,n,200))
        selected.update(int(r['frame']) for r in sorted(rows,key=lambda r:float(r['raw_foreground_support']))[:8])
        selected.update(json.loads((base/'image_qa.json').read_text())['flagged_frames'])
        geometry=json.loads((base/'geometry_qa.json').read_text())
        selected.update(geometry['flagged_frames']);selected.update(geometry['interpolated_frames'])
        selected.update(geometry.get('temporal_suspicious_frames',[]))
        if name.endswith(('181044','181548')):selected.update(f for f in [100,300,500,700,900,1100,1250] if f<n)
        selected=sorted(selected);all_selected[name]=selected
        for page,lo in enumerate(range(0,len(selected),24)):
            ids=selected[lo:lo+24];fig,axs=plt.subplots((len(ids)+5)//6,6,figsize=(15,3.2*((len(ids)+5)//6)),squeeze=False,layout='constrained')
            for ax in axs.flat:ax.set_axis_off()
            for ax,f in zip(axs.flat,ids):
                im=cv2.cvtColor(cv2.imread(str(ROOT/'workspace/data/raw/real'/name/'cam0'/f'{f:05d}.png')),cv2.COLOR_BGR2RGB)[68:368,220:520]
                m=cv2.imread(str(base/'intermediate_verified/sam2_masks'/f'{f:05d}.png'),0)>0
                ax.imshow(im);ax.contour(m,levels=[.5],colors=['#00d7ff'],linewidths=.6)
                ax.plot(xy[f,:,0],xy[f,:,1],'.-',color='#ffbd35',ms=2,lw=.65)
                ax.set(xlim=(75,240),ylim=(280,25),title=f'f{f}: support {float(rows[f]["raw_foreground_support"]):.3f}')
            fig.suptitle(name+' / cyan mask; orange skeleton',fontsize=12)
            fig.savefig(root/'qc'/f'{name}_page{page}.png',dpi=125);plt.close(fig)
    write(root/'qc/selected_frames.json',all_selected)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root.resolve()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name in pool.map(lambda n:checked_sequence(root,n),NAMES):print('Checked and generated:',name,flush=True)
    package(root);contact_sheets(root)
    print('Packaged; inspect contact sheets and flags before final acceptance.',flush=True)

if __name__=='__main__':main()
