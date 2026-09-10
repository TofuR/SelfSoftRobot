#!/usr/bin/env python3
"""Offline pretrained robot-mask student; does not modify deployment or controls."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def mask_polygon(mask,crop,image_size):
    """Restore crop-local pixels before normalizing a single complete-arm polygon."""
    x,y,w,h=map(int,crop);iw,ih=map(int,image_size)
    if mask.shape!=(h,w) or x<0 or y<0 or x+w>iw or y+h>ih:raise ValueError('mask/crop/image geometry mismatch')
    binary=np.uint8(mask>0)
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('empty mask')
    contour=max(contours,key=cv2.contourArea)
    filled=np.zeros_like(binary);cv2.drawContours(filled,[contour],-1,1,cv2.FILLED)
    iou=float(np.count_nonzero(filled&binary)/max(1,np.count_nonzero(filled|binary)))
    if iou<.98:raise ValueError('mask has significant disconnected pieces or holes')
    polygon=cv2.approxPolyDP(contour,.35,True).reshape(-1,2)
    if len(polygon)<3 or cv2.contourArea(polygon)<20:raise ValueError('degenerate mask polygon')
    reconstructed=np.zeros_like(binary);cv2.fillPoly(reconstructed,[polygon],1)
    polygon_iou=float(np.count_nonzero(reconstructed&binary)/max(1,np.count_nonzero(reconstructed|binary)))
    if polygon_iou<.98:raise ValueError('polygon conversion loses mask boundary')
    return (polygon+np.array([x,y]))/np.array([iw,ih]),polygon_iou

def find_pair(sequence):
    parent=ROOT/'workspace/data/intermediate/real'/sequence
    for folder in [parent/'legacy-derived',parent/(sequence+'_n15_sam2_robot_mm')]:
        crop=folder/'crop/crop_meta.json';masks=folder/'sam2_masks'
        if not masks.is_dir():masks=parent/'sam2-video-v1'
        if crop.is_file() and masks.is_dir():return crop,masks
    raise FileNotFoundError('No final SAM mask/crop metadata for '+sequence)

def prepare(study,cap):
    if not 1<=cap<=100000:raise ValueError('invalid per-sequence cap')
    study.mkdir(parents=True,exist_ok=False);data=study/'dataset';records=[];rejected=[];sources=[]
    splits={'182253':'val','182519':'test'}
    sequences=['172644','181044','181548','182253','182519','183351','183547','183740','184036']
    for role in ('train','val','test'):
        for kind in ('images','labels'):(data/kind/role).mkdir(parents=True)
    for suffix in sequences:
        sequence='seq_20260819_'+suffix;role=splits.get(suffix,'train')
        crop_file,masks=find_pair(sequence);meta=json.loads(crop_file.read_text());crop=meta['crop_xywh'];size=meta['source_image_size_wh']
        images=ROOT/'workspace/data/raw/real'/sequence/'cam0'
        names=sorted(p.name for p in masks.glob('*.png'))
        if set(names)!=set(p.name for p in images.glob('*.png')):raise ValueError('source/mask frame pairing mismatch: '+sequence)
        selected=np.linspace(0,len(names)-1,min(len(names),cap)).astype(int)
        sources.append(dict(sequence=sequence,split=role,available=len(names),selected=len(selected),crop_xywh=crop,
                            crop_metadata=str(crop_file.relative_to(ROOT)),masks=str(masks.relative_to(ROOT)),
                            crop_sha256=hashlib.sha256(crop_file.read_bytes()).hexdigest()))
        for i in selected:
            name=names[i];source=images/name;maskfile=masks/name;mask=cv2.imread(str(maskfile),0)
            if mask is None:raise ValueError('unreadable mask '+str(maskfile))
            frame=cv2.imread(str(source));
            if frame is None or list(frame.shape[1::-1])!=size:raise ValueError('source image size mismatch')
            try:poly,iou=mask_polygon(mask,crop,size)
            except ValueError as error:
                rejected.append(dict(sequence=sequence,frame=name,reason=str(error)));continue
            stem=sequence+'_'+source.stem
            (data/'images'/role/(stem+'.png')).symlink_to(source.resolve())
            (data/'labels'/role/(stem+'.txt')).write_text('0 '+' '.join(f'{v:.8f}' for v in poly.ravel())+'\n')
            records.append(dict(split=role,sequence=sequence,frame=name,polygon_iou=iou,
                                source=str(source.relative_to(ROOT)),mask=str(maskfile.relative_to(ROOT)),mask_sha256=hashlib.sha256(maskfile.read_bytes()).hexdigest()))
    write(study/'dataset_manifest.json',dict(schema='robot_segmentation_student_v1',created=time.time(),sources=sources,
        label_origin='SAM2 pseudo-labels, not independently hand-annotated truth',
        split_policy='whole sequences: 182253 validation, 182519 test, all other listed sequences training; same camera/background domain',
        sample_policy=f'deterministic uniform sampling, at most {cap} per sequence to limit long-hold dominance',
        excluded_sequences={'seq_20260819_183526':'no paired final masks; short zero-only calibration'},
        counts={role:sum(r['split']==role for r in records) for role in ('train','val','test')},rejected=rejected,files=records))
    if len(rejected)>max(5,.02*(len(records)+len(rejected))):raise ValueError('Over 2% mask conversions rejected; inspect labels before training')
    for role in ('train','val','test'):
        if not any(r['split']==role for r in records):raise ValueError('empty split '+role)
    (study/'data.yaml').write_text('path: '+json.dumps(str(data.resolve()))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: soft_arm\n')
    (study/'DATA_READY').write_text('paired, restored, sequence split\n')

def benchmark(weights,onnx_path,study,device):
    """Image-array -> reconstructed masks; first load and warmup excluded."""
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    paths=sorted((study/'dataset/images/test').glob('*.png'))
    images=[cv2.imread(str(paths[i])) for i in np.linspace(0,len(paths)-1,20).astype(int)]
    rows=[]
    for backend,path,selected in [('pytorch_cuda',weights,device),('pytorch_cpu',weights,'cpu'),('onnx_cpu',onnx_path,'cpu')]:
        net=YOLO(str(path),task='segment')
        def synchronize():
            if selected!='cpu':torch.cuda.synchronize()
        for _ in range(5):net.predict(images[0],imgsz=640,device=selected,retina_masks=True,verbose=False)
        elapsed=[];stages=[];missing=0
        for frame in images:
            synchronize();start=time.perf_counter()
            result=net.predict(frame,imgsz=640,device=selected,retina_masks=True,verbose=False)[0]
            if result.masks is not None:result.masks.data.cpu().numpy()
            else:missing+=1
            synchronize();elapsed.append((time.perf_counter()-start)*1000);stages.append(result.speed)
        rows.append(dict(backend=backend,median_ms=float(np.median(elapsed)),p95_ms=float(np.percentile(elapsed,95)),max_ms=max(elapsed),
                         images=len(images),no_detection=missing,stages=stages))
    write(study/(Path(weights).parents[1].name+'_latency.json'),dict(rows=rows,
        scope='server warm image-array to masks batch=1; excludes camera/GUI/control, not deployment-PC timing'))

def train(study,epochs,device):
    if not (study/'DATA_READY').is_file():raise ValueError('dataset is not validated')
    from ultralytics import YOLO
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; not silently launching long CPU training')
    models=['yolo26n-seg','yolo26s-seg'];status=dict(phase='training',epochs=epochs,models=models)
    write(study/'status.json',status)
    for model in models:
        if (study/model).exists():raise FileExistsError('refuse overwriting run '+model)
        pretrained=ROOT/'workspace/models/pretrained/yolo26'/(model+'.pt')
        if not pretrained.is_file():raise FileNotFoundError(pretrained)
        status.update(model=model,phase='training');write(study/'status.json',status)
        net=YOLO(str(pretrained))
        net.train(data=str(study/'data.yaml'),epochs=epochs,patience=25,imgsz=640,batch=16,device=device,workers=4,
                  project=str(study),name=model,exist_ok=False,seed=42,deterministic=True,amp=False,
                  degrees=20.,translate=.15,scale=.5,fliplr=.5,flipud=0.,mosaic=0.,mixup=0.,copy_paste=0.,
                  hsv_h=.03,hsv_s=.4,hsv_v=.4,mask_ratio=2,plots=True,save=True)
        best=study/model/'weights/best.pt'
        net=YOLO(str(best));metrics=net.val(data=str(study/'data.yaml'),split='test',imgsz=640,device=device,
                                         project=str(study),name=model+'_heldout',exist_ok=False,plots=True)
        write(study/(model+'_test.json'),dict(metrics=metrics.results_dict,selection='Ultralytics validation fitness = mask fitness + box fitness; test never used for selection',
              pseudo_label_evaluation=True,weights_sha256=hashlib.sha256(best.read_bytes()).hexdigest()))
        exported=net.export(format='onnx',imgsz=640,opset=17,dynamic=False,simplify=False,device='cpu')
        benchmark(best,exported,study,device)
        status.update(phase='model_complete',best=str(best),onnx=str(exported));write(study/'status.json',status)
    status['phase']='complete';write(study/'status.json',status);(study/'COMPLETE').write_text('training, heldout evaluation and ONNX export complete\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--train',action='store_true')
    parser.add_argument('--cap-per-sequence',type=int,default=1500);parser.add_argument('--epochs',type=int,default=100);parser.add_argument('--device',default='0')
    args=parser.parse_args();study=args.study.resolve()
    if args.epochs<1 or args.epochs>1000:parser.error('epochs must be 1..1000')
    try:
        if args.prepare:prepare(study,args.cap_per_sequence)
        if args.train:train(study,args.epochs,args.device)
    except Exception as error:
        if study.is_dir():write(study/'failure.json',dict(error=repr(error),time=time.time()))
        raise
