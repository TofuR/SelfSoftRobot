"""Lazy, offline SAM2 image segmentation for reviewed deployment initialization."""
from pathlib import Path
import hashlib
import sys
import time
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]


def default_checkpoint():
    choices = [APP_DIR/'checkpoints/sam2/sam2.1_hiera_tiny.pt',
               APP_DIR.parent/'workspace/models/pretrained/sam2/sam2.1_hiera_tiny.pt']
    return next((p for p in choices if p.is_file()), choices[0])


def automatic_foreground_prompt(image, polarity='bright'):
    """Suggest one interior pixel using contrast and a central-object prior.

    This is an initialization hint, not a trained robot detector. Ambiguous
    separate foreground regions require a user click instead of a blind guess.
    The returned point never certifies a complete arm or bypasses draft review.
    """
    import cv2
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3:raise ValueError('SAM 需要 BGR 彩色图像')
    if polarity not in ('bright','dark'):raise ValueError('未知明暗方向')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    retry='自动定位不确定，请在臂身中间点一下，再重新提取'
    if np.ptp(gray)<20:raise ValueError(retry+'（图像对比不足）')
    flag=cv2.THRESH_BINARY if polarity=='bright' else cv2.THRESH_BINARY_INV
    threshold,mask=cv2.threshold(gray,0,255,flag|cv2.THRESH_OTSU)
    if polarity=='bright':
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask[hsv[...,1]>160]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    h,w=mask.shape
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    depth=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    yy,xx=np.indices(mask.shape)
    distance=np.hypot((xx-w*.5)/w,(yy-h*.5)/h)
    candidates=[]
    for i in range(1,count):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if not max(30,mask.size*.0003)<=area<=mask.size*.65:continue
        eligible=(labels==i)&(depth>=max(3,min(h,w)*.007))&(distance<.32)
        if not eligible.any():continue
        # Prefer an interior point close to image center; cap the width bonus
        # so a broad fixture does not win merely by being thicker.
        cost=np.where(eligible,distance-.02*np.minimum(depth/8,1),np.inf)
        y,x=np.unravel_index(np.argmin(cost),mask.shape)
        candidates.append((float(cost[y,x]),[int(x),int(y)],area))
    candidates.sort(key=lambda item:item[0])
    if not candidates:raise ValueError(retry+'（没有可信中心前景）')
    if len(candidates)>1 and candidates[1][0]-candidates[0][0]<.055:
        raise ValueError(retry+'（多个相近前景候选）')
    best=candidates[0]
    return best[1],dict(method='contrast_central_foreground',polarity=polarity,
                        threshold=float(threshold),center_cost=best[0],
                        component_area_px=best[2],candidate_count=len(candidates),
                        semantic_detection=False,review_required=True)


def _select_mask(masks,scores,coords,kinds,automatic):
    # SAM score is a candidate score, not a guarantee of complete observation.
    ranked=[]
    for i,mask in enumerate(masks):
        if not np.any(mask):continue
        if automatic is not None:
            # Avoid accepting the support/background selected by a central
            # seed. A rejected hint can be repaired with one manual point.
            ys,xs=np.nonzero(mask)
            if mask.mean()>.55 or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():continue
            extent=np.array([np.ptp(xs)+1,np.ptp(ys)+1])
            if extent.max()/max(1,extent.min())<1.4:continue
        mismatch=0.
        if len(coords):
            xy=np.rint(coords).astype(int);mismatch=float(np.mean(mask[xy[:,1],xy[:,0]]!=kinds))
        ranked.append((float(scores[i])-2*mismatch,i))
    if not ranked:raise ValueError('SAM 未得到可信臂身候选，请在臂身中间点一下，再重新提取')
    best_score,index=max(ranked)
    selection='highest_prompt_score'
    # SAM often offers the distal actuator and the complete multi-section
    # arm for one click. Prefer a containing, similarly confident candidate
    # to silently calibrating only one actuator as the whole robot.
    best=np.asarray(masks[index],dtype=bool);best_area=int(best.sum())
    containers=[]
    for score,i in ranked:
        candidate=np.asarray(masks[i],dtype=bool);area=int(candidate.sum())
        if score>=best_score-.08 and best_area<=area<=3*best_area:
            containment=np.count_nonzero(candidate&best)/max(1,best_area)
            if containment>=.95:containers.append((area,i))
    if containers:
        selected=max(containers)[1]
        if selected!=index:selection='containing_mask_similar_score'
        index=selected
    return index,selection


class SamInitialSegmenter:
    """One GUI worker owns the predictor; frozen image embeddings are reusable."""
    def __init__(self):
        self.predictor=None;self.key=None;self.image_key=None;self.weights_hash=None

    def segment(self,image,checkpoint,device='auto',roi=None,guide=None,points=(),labels=(),polarity='bright'):
        # All entrypoints (auto, point correction, hand-guide refinement) share
        # the same bounded CPU thread pool. The context restores prior limits.
        started=time.perf_counter()
        import torch  # Load OpenMP before threadpoolctl enumerates its libraries.
        from threadpoolctl import threadpool_limits
        import_ms=(time.perf_counter()-started)*1000
        with threadpool_limits(limits=4):
            mask,info=self._segment(image,checkpoint,device,roi,guide,points,labels,polarity)
        info['import_ms']=import_ms;info['elapsed_ms']=(time.perf_counter()-started)*1000
        return mask,info

    def _segment(self,image,checkpoint,device,roi,guide,points,labels,polarity):
        started=time.perf_counter()
        import torch
        path=Path(checkpoint).expanduser().resolve()
        image=np.asarray(image)
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('SAM 需要 BGR 彩色图像')
        automatic=None;prompt_source='operator'
        if roi is None and guide is None and not len(points):
            point,automatic=automatic_foreground_prompt(image,polarity)
            points=[point];labels=[1];prompt_source='automatic_image_hint'
        load_started=time.perf_counter()
        if not path.is_file():raise ValueError('找不到 SAM2 权重；请在 SAM 设置选择 sam2.1_hiera_tiny.pt，或选择传统分割')
        if device not in ('auto','cpu','cuda'):raise ValueError('SAM 设备必须为 auto/cpu/cuda')
        device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA 不可用，请在 SAM 设置选择 CPU 或自动')
        key=(str(path),path.stat().st_mtime_ns,device)
        if self.key!=key:
            sources=[APP_DIR/'vendor/sam2',APP_DIR.parent/'sam2/sam2_src']
            source=next((p for p in sources if (p/'sam2/build_sam.py').is_file()),None)
            if source is not None and str(source) not in sys.path:sys.path.insert(0,str(source))
            existing=sys.modules.get('sam2')
            if existing is not None and not getattr(existing,'__file__',None):sys.modules.pop('sam2')
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as error:
                raise RuntimeError('SAM2 依赖未安装，请运行部署包的安装脚本或 pip install -r real_validation/requirements-sam2.txt；'+str(error)) from error
            model=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(path),device=device,apply_postprocessing=False)
            self.predictor=SAM2ImagePredictor(model);self.key=key;self.image_key=None
            self.weights_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        load_ms=(time.perf_counter()-load_started)*1000
        h,w=image.shape[:2];image_key=hashlib.sha256(image.tobytes()).hexdigest()
        box=None if roi is None else np.asarray(roi,dtype=np.float32)
        if box is not None and (box.shape!=(4,) or not np.isfinite(box).all() or np.any(box[:2]<0) or np.any(box[2:]<=box[:2]) or np.any(box[2:]>[w,h])):raise ValueError('SAM 框选范围无效')
        coords=list(points);kinds=list(labels)
        if guide is not None:
            guide=np.asarray(guide,dtype=float)
            # Interior positive points, excluding fittings at the two short edges.
            ids=np.linspace(1,len(guide)-2,min(7,len(guide)-2)).astype(int)
            coords.extend(guide[ids].tolist());kinds.extend([1]*len(ids))
        if len(coords)!=len(kinds):raise ValueError('SAM 提示点与标签数量不一致')
        coords=np.asarray(coords,dtype=np.float32).reshape(-1,2);kinds=np.asarray(kinds,dtype=np.int32)
        if (not np.isfinite(coords).all() or np.any(coords<0) or np.any(coords>[w-1,h-1]) or not set(kinds.tolist())<={0,1}):raise ValueError('SAM 提示点无效')
        cache_hit=self.image_key==image_key
        def synchronize():
            if device=='cuda':torch.cuda.synchronize()
        synchronize();encode_started=time.perf_counter()
        with torch.inference_mode():
            if not cache_hit:
                self.predictor.set_image(np.ascontiguousarray(image[...,::-1]));self.image_key=image_key
            synchronize();encode_ms=(time.perf_counter()-encode_started)*1000
            decode_started=time.perf_counter()
            masks,scores,_=self.predictor.predict(point_coords=coords if len(coords) else None,
                                                 point_labels=kinds if len(coords) else None,box=box,multimask_output=True)
        synchronize();decode_ms=(time.perf_counter()-decode_started)*1000
        shape_hint=automatic if automatic is not None else ({} if roi is None and guide is None and np.any(kinds==1) else None)
        index,selection=_select_mask(masks,scores,coords,kinds,shape_hint)
        seed_scores=np.asarray(scores).tolist();automatic_box=None
        if shape_hint is not None:
            # One cached decoder pass with a mask-derived box usually removes
            # tiny holes and ambiguous actuator boundaries. No new image encode.
            ys,xs=np.nonzero(masks[index]);short=min(np.ptp(xs)+1,np.ptp(ys)+1)
            padding=max(4,int(round(short*.18)))
            automatic_box=np.array([max(0,int(xs.min())-padding),max(0,int(ys.min())-padding),
                                    min(w,int(xs.max())+padding+1),min(h,int(ys.max())+padding+1)],np.float32)
            refine_started=time.perf_counter()
            with torch.inference_mode():
                refined,refined_scores,_=self.predictor.predict(point_coords=None,point_labels=None,
                                                               box=automatic_box,multimask_output=True)
            synchronize();decode_ms+=(time.perf_counter()-refine_started)*1000
            try:
                original=np.asarray(masks[index],bool)
                compatible=[]
                for i,candidate in enumerate(np.asarray(refined,bool)):
                    overlap=np.count_nonzero(original&candidate)/max(1,np.count_nonzero(original|candidate))
                    if overlap>=.75:compatible.append(i)
                if compatible:
                    refined=np.asarray(refined)[compatible].copy();refined_scores=np.asarray(refined_scores)[compatible]
                    # Keep the first mask's extent: cleanup may repair internal
                    # holes but must not grow the base into the attached fixture.
                    extent=np.zeros_like(original)
                    extent[max(0,ys.min()-2):min(h,ys.max()+3),max(0,xs.min()-2):min(w,xs.max()+3)]=True
                    refined=np.asarray(refined,bool)&extent
                    refined_index,_=_select_mask(refined,refined_scores,coords,kinds,shape_hint)
                    masks,scores,index=refined,refined_scores,refined_index
                    selection='automatic_box_refinement';box=automatic_box
            except ValueError:pass  # Keep the valid first candidate for review.
        mask=np.uint8(masks[index])*255
        return mask,dict(method='sam2.1_hiera_tiny',device=device,checkpoint_sha256=self.weights_hash,
                         image_sha256=image_key,score=float(scores[index]),candidate_scores=np.asarray(scores).tolist(),
                         box=None if box is None else box.tolist(),points=coords.tolist(),labels=kinds.tolist(),
                         prompt_source=prompt_source,automatic_prompt=automatic,cache_hit=cache_hit,selection=selection,
                         seed_candidate_scores=seed_scores,automatic_box=None if automatic_box is None else automatic_box.tolist(),
                         load_ms=load_ms,encode_ms=encode_ms,decode_ms=decode_ms,elapsed_ms=(time.perf_counter()-started)*1000)
