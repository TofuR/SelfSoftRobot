import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from real_validation.perception.yolo_initial import load_contract, select_mask


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def cpu(self): return self
    def numpy(self): return self.value


class Boxes:
    def __init__(self, n):
        self.cls = Tensor(np.zeros(n))
        self.conf = Tensor(np.full(n, .9))
        self.n = n
    def __len__(self): return self.n


def result(masks):
    return SimpleNamespace(boxes=Boxes(len(masks)), masks=SimpleNamespace(data=Tensor(masks)))


class YoloInitialTests(unittest.TestCase):
    def test_original_pixel_mask_is_preserved(self):
        mask = np.zeros((41, 73)); mask[5:35, 20:26] = 1
        actual, confidence = select_mask(result([mask]), mask.shape)
        np.testing.assert_array_equal(actual, mask*255)
        self.assertAlmostEqual(confidence, .9)

    def test_empty_detection_and_multiple_instances_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '未检测'):
            select_mask(SimpleNamespace(masks=None, boxes=None), (41, 73))
        with self.assertRaisesRegex(ValueError, '多个'):
            select_mask(result(np.ones((2, 41, 73))), (41, 73))

    def test_wrong_coordinates_and_empty_mask_are_rejected(self):
        with self.assertRaisesRegex(ValueError, '原图尺寸'):
            select_mask(result(np.ones((1, 640, 640))), (41, 73))
        with self.assertRaisesRegex(ValueError, '空掩膜'):
            select_mask(result(np.zeros((1, 41, 73))), (41, 73))

    def test_model_hash_and_relative_paths_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'best.pt').write_bytes(b'candidate')
            config = dict(schema='robot_yolo_initial_v1', class_id=0, imgsz=640,
                          weights='best.pt', retina_masks=True,
                          files={'best.pt':hashlib.sha256(b'candidate').hexdigest()})
            (p/'inference.json').write_text(json.dumps(config))
            load_contract(p)
            (p/'best.pt').write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, '校验失败'): load_contract(p)
            config['files'] = {'../best.pt':'ignored'}
            (p/'inference.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, '候选目录内'): load_contract(p)


if __name__ == '__main__': unittest.main()
