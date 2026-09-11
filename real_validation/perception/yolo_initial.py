"""Lazy YOLO full-image segmentation for reviewed deployment initialization."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / 'checkpoints/perception'


def load_contract(directory):
    directory = Path(directory)
    config = json.loads((directory / 'inference.json').read_text(encoding='utf-8'))
    if (config.get('schema') != 'robot_yolo_initial_v1' or config.get('class_id') != 0
            or config.get('imgsz') != 640 or config.get('weights') != 'best.pt'
            or config.get('retina_masks') is not True):
        raise ValueError('不支持的 YOLO 分割模型配置')
    for name, digest in config['files'].items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('YOLO 模型文件必须在候选目录内')
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('YOLO 模型校验失败：' + name)
    if 'best.pt' not in config['files']:
        raise ValueError('YOLO 配置缺少权重校验')
    return config


def candidates():
    return sorted(MODEL_DIR.glob('*/inference.json'))


def select_mask(result, shape):
    """Do not pick a largest fragment when multiple robot instances are found."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        raise ValueError('YOLO 未检测到臂身；请检查视野，或切换 SAM2 / 手工修正')
    ids = np.flatnonzero(result.boxes.cls.cpu().numpy() == 0)
    if len(ids) != 1:
        raise ValueError('YOLO 检测到多个臂身候选；请排除干扰或切换 SAM2 提示分割')
    masks = result.masks.data.cpu().numpy()
    if masks.shape[1:] != tuple(shape):
        raise ValueError('YOLO mask 未返回原图尺寸，无法用于配准')
    mask = np.uint8(masks[ids[0]] > .5) * 255
    if not mask.any():
        raise ValueError('YOLO 返回空掩膜')
    return mask, float(result.boxes.conf.cpu().numpy()[ids[0]])


class YoloInitialSegmenter:
    def __init__(self):
        self.key = None
        self.model = None

    def segment(self, image, directory, device='auto'):
        started = time.perf_counter()
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('YOLO 需要 uint8 BGR 相机图像')
        directory = Path(directory).resolve()
        if not (directory / 'inference.json').is_file():
            raise ValueError('未找到 YOLO 分割候选，请使用包含感知权重的完整部署包')
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError('请安装 real_validation/requirements-yolo.txt 后使用 YOLO') from error
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('未知 YOLO 推理设备')
        selected = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if selected == 'cuda' and not torch.cuda.is_available():
            raise ValueError('已选择 CUDA，但当前 PyTorch 无可用 GPU')
        key = (str(directory), selected)
        load_ms = 0.
        if key != self.key:
            before = time.perf_counter()
            config = load_contract(directory)
            model = YOLO(str(directory / config['weights']), task='segment')
            self.model, self.config, self.key = model, config, key
            load_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        result = self.model.predict(image, imgsz=640, device=selected, conf=.25,
                                    classes=[0], retina_masks=True, verbose=False)[0]
        mask, confidence = select_mask(result, image.shape[:2])
        return mask, dict(method='yolo26_seg', model=directory.name, device=selected,
                          checkpoint_sha256=self.config['files']['best.pt'], confidence=confidence,
                          load_ms=load_ms, image_to_mask_ms=(time.perf_counter()-before)*1000,
                          elapsed_ms=(time.perf_counter()-started)*1000,
                          input_shape=list(image.shape), imgsz=640, retina_masks=True,
                          mask_coordinates='original_image', api_stages_ms=dict(result.speed))
