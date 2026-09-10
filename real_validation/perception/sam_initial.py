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


class SamInitialSegmenter:
    """One GUI worker owns the predictor; frozen image embeddings are reusable."""
    def __init__(self):
        self.predictor=None;self.key=None;self.image_key=None;self.weights_hash=None

    def segment(self,image,checkpoint,device='auto',roi=None,guide=None,points=(),labels=()):
        import torch
        started=time.perf_counter();path=Path(checkpoint).expanduser().resolve()
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
        image=np.asarray(image)
        if image.ndim!=3 or image.shape[2]!=3:raise ValueError('SAM 需要 BGR 彩色图像')
        h,w=image.shape[:2];image_key=hashlib.sha256(image.tobytes()).hexdigest()
        box=np.array([0,0,w-1,h-1] if roi is None else roi,dtype=np.float32)
        if box.shape!=(4,) or np.any(box[:2]<0) or np.any(box[2:]<=box[:2]) or np.any(box[2:]>[w,h]):raise ValueError('SAM 框选范围无效')
        coords=list(points);kinds=list(labels)
        if guide is not None:
            guide=np.asarray(guide,dtype=float)
            # Interior positive points, excluding fittings at the two short edges.
            ids=np.linspace(1,len(guide)-2,min(7,len(guide)-2)).astype(int)
            coords.extend(guide[ids].tolist());kinds.extend([1]*len(ids))
        if len(coords)!=len(kinds):raise ValueError('SAM 提示点与标签数量不一致')
        coords=np.asarray(coords,dtype=np.float32).reshape(-1,2);kinds=np.asarray(kinds,dtype=np.int32)
        if (not np.isfinite(coords).all() or np.any(coords<0) or np.any(coords>[w-1,h-1]) or not set(kinds.tolist())<={0,1}):raise ValueError('SAM 提示点无效')
        with torch.inference_mode():
            if self.image_key!=image_key:
                self.predictor.set_image(np.ascontiguousarray(image[...,::-1]));self.image_key=image_key
            masks,scores,_=self.predictor.predict(point_coords=coords if len(coords) else None,
                                                 point_labels=kinds if len(coords) else None,box=box,multimask_output=True)
        # SAM score is a candidate score, not a guarantee of complete observation.
        ranked=[]
        for i,mask in enumerate(masks):
            if not np.any(mask):continue
            mismatch=0.
            if len(coords):
                xy=np.rint(coords).astype(int);mismatch=float(np.mean(mask[xy[:,1],xy[:,0]]!=kinds))
            ranked.append((float(scores[i])-2*mismatch,i))
        if not ranked:raise ValueError('SAM 未得到臂身区域，请框选或添加正/负提示点')
        index=max(ranked)[1];mask=np.uint8(masks[index])*255
        return mask,dict(method='sam2.1_hiera_tiny',device=device,checkpoint_sha256=self.weights_hash,
                         image_sha256=image_key,score=float(scores[index]),candidate_scores=np.asarray(scores).tolist(),
                         box=box.tolist(),points=coords.tolist(),labels=kinds.tolist(),elapsed_ms=(time.perf_counter()-started)*1000)
