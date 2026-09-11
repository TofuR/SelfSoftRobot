#!/usr/bin/env python3
"""Publish completed YOLO study candidates into the validation App model folder."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='new versioned perception candidate root')
    args = parser.parse_args()
    study, out = args.study, args.out
    status = json.loads((study/'status.json').read_text())
    if status['phase'] != 'complete' or not (study/'COMPLETE').is_file():
        raise ValueError('Training/evaluation/export has not completed')
    if out.exists():
        raise FileExistsError(out)
    # Validate all sources before creating the new publication.
    sources = {}
    for model in status['models']:
        weights = study/model/'weights'
        metrics = json.loads((study/(model+'_test.json')).read_text())
        if hashlib.sha256((weights/'best.pt').read_bytes()).hexdigest() != metrics['weights_sha256']:
            raise ValueError('Test metrics refer to different weights: '+model)
        sources[model] = weights
        if not (weights/'best.onnx').is_file():
            raise FileNotFoundError(weights/'best.onnx')
    out.mkdir(parents=True)
    for model, weights in sources.items():
        dest = out/(model+'_'+study.name.removeprefix('robot_yolo26seg_'))
        dest.mkdir()
        for name in ('best.pt', 'best.onnx'):
            shutil.copy2(weights/name, dest/name)
        # Actual official checkpoint criterion; old run text said mask-only.
        selection = 'Ultralytics SegmentMetrics.fitness = mask fitness + box fitness; validation only'
        metrics = json.loads((study/(model+'_test.json')).read_text())
        metrics['selection'] = selection
        write(dest/'metrics.json', metrics)
        shutil.copy2(study/(model+'_latency.json'), dest/'latency.json')
        epochs = list(csv.DictReader((study/model/'results.csv').open()))
        write(dest/'provenance.json', dict(study_id=study.name, model=model,
              selection=selection, completed_epochs=len(epochs),
              dataset_manifest_sha256=hashlib.sha256((study/'dataset_manifest.json').read_bytes()).hexdigest(),
              supervision='same-platform sequence-heldout pseudo masks', physical_D415_tested=False,
              run_manifest=json.loads((study/'run_manifest.json').read_text())))
        write(dest/'inference.json', dict(schema='robot_yolo_initial_v1', weights='best.pt',
              files={name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in ('best.pt','best.onnx')},
              class_id=0, labels=['soft_arm'], imgsz=640, confidence=.25, retina_masks=True,
              backend='ultralytics', ultralytics_version='8.4.146', input='uint8 BGR image',
              output='binary mask at original image size',
              preprocessing='Ultralytics letterbox/color/normalization and original-size mask reconstruction',
              onnx='archival export; App uses best.pt', mask_threshold=.5,
              multiple_instances='reject and request review'))
    print(out)


if __name__ == '__main__':
    main()
