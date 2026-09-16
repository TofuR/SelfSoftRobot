"""Concurrent cropped sequences must not overwrite/delete each other's SAM2 input."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import json
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


class MaskPredictor:
    """Echo the anchor mask through both propagation directions, without a GPU."""
    def __init__(self):
        self.anchors = []

    def init_state(self, **kwargs):
        return {}

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        self.anchors.append(frame_idx)
        state['mask'] = mask

    def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track, reverse):
        step = -1 if reverse else 1
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = state['mask'][None]
        for offset in range(max_frame_num_to_track + 1):
            yield start_frame_idx + step * offset, [1], [tensor]


def load_full_script():
    script = Path(__file__).resolve().parents[1] / 'sam2/segment_video_full.py'
    spec = importlib.util.spec_from_file_location('sam2_regression_test', script)
    module = importlib.util.module_from_spec(spec)
    # The CLI adds its vendored SAM2 directory to sys.path; isolate that from
    # tests importing the repository's sam2.segment_video_full module later.
    with patch.object(sys, 'path', sys.path.copy()):
        spec.loader.exec_module(module)
    return module


def check_parallel_crops(tmp_path):
    module = load_full_script()
    cameras = []
    for name, value in [('sequence_a', 30), ('sequence_b', 220)]:
        camera = tmp_path / name / 'crop/cam0'
        camera.mkdir(parents=True)
        assert cv2.imwrite(str(camera/'00000.png'), np.full((8, 8, 3), value, np.uint8))
        cameras.append(camera)

    def prepare(camera):
        workspace = Path(module.create_jpeg_workspace(tmp_path/'cache', camera.parent.name, 0))
        chunk = workspace/'chunk_00000'
        assert module.prepare_jpeg_dir(str(camera), [0], str(chunk))
        return workspace, chunk/'000000.jpg'

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(prepare, cameras))
    assert first[0] != second[0]
    assert cv2.imread(str(first[1])).mean() == 30
    assert cv2.imread(str(second[1])).mean() == 220
    shutil.rmtree(first[0])
    assert cv2.imread(str(second[1])).mean() == 220


class JpegWorkspaceTest(unittest.TestCase):
    def test_changed_median_recomputes_anchor_instead_of_resuming(self):
        module = load_full_script()
        predictor = MaskPredictor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, anchors, out = [root/n for n in ('images', 'anchors', 'out')]
            for d in (images, anchors, out): d.mkdir()
            for f, width in enumerate((10, 20)):
                cv2.imwrite(str(images/f'{f:05d}.png'), np.zeros((30, 30, 3), np.uint8))
                mask = np.zeros((30, 30), np.uint8); mask[:10, :width] = 255
                cv2.imwrite(str(anchors/f'{f:05d}.png'), mask)
            args = (predictor, str(images), str(anchors), str(out), str(root/'jpeg'), 0, 1)
            self.assertTrue(module.process_chunk(*args, 100))
            self.assertEqual(module.process_chunk(*args, 100), [])
            self.assertTrue(module.process_chunk(*args, 200))
            self.assertEqual(predictor.anchors, [0, 1])
            self.assertEqual(cv2.countNonZero(cv2.imread(str(out/'00000.png'), 0)), 200)
            receipt = json.loads(module.block_receipt(out, [0, 1]).read_text())
            self.assertEqual(receipt['source']['inference']['med_area'], 200)

    def test_preprocess_anchor_trim_config_reaches_both_propagation_directions(self):
        from scripts.real import preprocess_capture as pipeline
        from src.registry import ProjectPaths
        module = load_full_script()
        predictor = MaskPredictor()
        class StopAfterSam2(Exception): pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); paths = ProjectPaths.load(repo_root=root, environ={})
            raw = paths.raw_sequence('real', 'seq_demo'); (raw/'cam0').mkdir(parents=True)
            derived = paths.workspace_root/'derived'
            images, anchors = derived/'crop/cam0', derived/'masks_candidate'
            for d in (images, anchors): d.mkdir(parents=True)
            mask = np.zeros((100, 100), np.uint8); mask[10:90, 40:60] = 255
            mask[10:15, 20:80] = 255; mask[85:90, 20:80] = 255
            for f in range(2):
                cv2.imwrite(str(images/f'{f:05d}.png'), np.zeros((100, 100, 3), np.uint8))
                cv2.imwrite(str(anchors/f'{f:05d}.png'), mask)
            checkpoint = paths.pretrained_model_dir('sam2')/'sam2.1_hiera_tiny.pt'
            checkpoint.parent.mkdir(parents=True); checkpoint.write_bytes(b'test checkpoint')
            config = root/'config.json'
            def launch(command, **kwargs):
                with patch('sys.argv', command[1:]): module.main()
                return Mock(poll=Mock(return_value=0))
            cases = [('top', 1.5, 5), ('bottom', 1.5, 5),
                     ('none', 1.5, 5), ('bottom', 4., 5), ('bottom', 1.5, 76)]
            for side, ratio, span in cases:
                with self.subTest(side=side, ratio=ratio, span=span):
                    config.write_text(json.dumps(dict(seq=str(raw), roi=[0, 0, 100, 100],
                        intermediate_root=str(derived), anchor=dict(base_side=side,
                        base_trim_width_ratio=ratio, base_trim_stable_span=span))))
                    with patch.object(ProjectPaths, 'load', return_value=paths), \
                         patch.object(module, 'build_predictor', return_value=predictor), \
                         patch.object(module, 'save_sam2_qc'), \
                         patch.object(pipeline.subprocess, 'Popen', side_effect=launch), \
                         patch.object(pipeline, 'validate_sam2', side_effect=StopAfterSam2):
                        with self.assertRaises(StopAfterSam2):
                            pipeline.main(['--config', str(config), '--stages', 'sam2,skeleton'])
                    expected = mask.copy()
                    if ratio == 1.5 and span == 5:
                        if side == 'top': expected[:15] = 0
                        if side == 'bottom': expected[85:] = 0
                    for f in range(2):
                        actual = cv2.imread(str(derived/'sam2_masks'/f'{f:05d}.png'), 0)
                        np.testing.assert_array_equal(actual, expected)
                    receipt = json.loads(module.block_receipt(derived/'sam2_masks', [0, 1]).read_text())
                    self.assertEqual(receipt['source']['inference']['base_trim'],
                                     dict(base_side=side, width_ratio=ratio, stable_span=span))
            self.assertEqual(predictor.anchors, [1] * len(cases))

    def test_parallel_crops_keep_independent_frames_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            check_parallel_crops(Path(directory))

    def test_resume_requires_matching_source_and_intact_masks(self):
        module = load_full_script()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir, anchor_dir, out = [root/n for n in ('images','anchors','out')]
            for d in (image_dir, anchor_dir, out): d.mkdir()
            for d in (image_dir, anchor_dir, out):
                cv2.imwrite(str(d/'00000.png'),np.full((10,10),255,np.uint8))
            source = module.block_source(image_dir,anchor_dir,[0],{}, {'model':'a'})
            self.assertFalse(module.chunk_done(out,[0],source))
            receipt = module.block_receipt(out,[0]);receipt.parent.mkdir()
            receipt.write_text(json.dumps({'source':source,'outputs':{'0':module.file_digest(out/'00000.png')}}))
            self.assertTrue(module.chunk_done(out,[0],source))
            changed = module.block_source(image_dir,anchor_dir,[0],{}, {'model':'b'})
            self.assertFalse(module.chunk_done(out,[0],changed))
            cv2.imwrite(str(image_dir/'00000.png'),np.zeros((10,10),np.uint8))
            changed = module.block_source(image_dir,anchor_dir,[0],{}, {'model':'a'})
            self.assertFalse(module.chunk_done(out,[0],changed))
            (out/'00000.png').write_bytes(b'broken')
            self.assertFalse(module.chunk_done(out,[0],source))

    def test_mask_validation_rejects_empty_and_wrong_size(self):
        from scripts.real.preprocess_capture import validate_sam2
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);im=root/'images';m=root/'masks';im.mkdir();m.mkdir()
            cv2.imwrite(str(im/'00000.png'),np.full((10,10,3),255,np.uint8))
            cv2.imwrite(str(m/'00000.png'),np.zeros((10,10),np.uint8))
            with self.assertRaisesRegex(RuntimeError,'empty mask'):validate_sam2(im,m)
            cv2.imwrite(str(m/'00000.png'),np.full((5,5),255,np.uint8))
            with self.assertRaisesRegex(RuntimeError,'shape mismatch'):validate_sam2(im,m)

    def test_output_paths_cannot_alias_raw_capture(self):
        from src.registry import ProjectPaths
        from scripts.real.preprocess_capture import build_parser, resolve_pipeline_args, resolve_preprocess_layout
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory);paths=ProjectPaths.load(repo_root=repo,environ={})
            raw=paths.raw_sequence('real','seq_demo');(raw/'cam0').mkdir(parents=True)
            alias=paths.workspace_root/'raw_alias';alias.symlink_to(raw,target_is_directory=True)
            for flag in ('--intermediate-root','--out-root'):
                for target in (raw,raw/'derived',alias):
                    args=resolve_pipeline_args(build_parser().parse_args([
                        '--seq',str(raw),'--roi','0,0,10,10',flag,str(target)]))
                    with self.assertRaisesRegex(ValueError,'disjoint from raw'):
                        resolve_preprocess_layout(args,paths)


if __name__ == '__main__':
    unittest.main()
