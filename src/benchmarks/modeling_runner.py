"""Train, select on validation, evaluate frozen checkpoints, and summarize."""
from __future__ import annotations

import csv
import json
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.benchmarks.modeling_data import CausalWindows, ROOT, load_sequences, resolve, sha256, write_json
from src.benchmarks.modeling_models import Polynomial, fit_normalization, make_model
from src.evaluation.modeling_benchmark_metrics import skeleton_metrics, mask_metrics, render_tube, compare_models


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _archive_code(output):
    archive = output / 'source'
    archive.mkdir()
    paths = list((ROOT / 'src/benchmarks').rglob('*.py')) + list((ROOT / 'src/benchmarks/vendor').glob('*.txt')) + [
        ROOT / 'src/evaluation/modeling_benchmark_metrics.py',
        ROOT / 'scripts/experiments/modeling_benchmark.py',
        ROOT / 'src/models/model_hereditary_geometry.py',
        ROOT / 'src/models/model_hereditary_operator.py',
        ROOT / 'src/models/model_ishsm.py'] + list((ROOT / 'src/operators').glob('*.py'))
    hashes = {}
    for path in paths:
        target = archive / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    (output / 'source.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT))
    return hashes


def _predict(model, sequences, config, center, scale, device, max_windows=None):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for sequence in sequences:
            windows = CausalWindows([sequence], config['history'], config['eval_stride'], max_windows)
            predicted, targets, frame_ids = [], [], []
            loader = DataLoader(windows, batch_size=config['batch_size'], shuffle=False)
            for action, target, _, frame in loader:
                estimate = model(action.to(device)) * scale + center
                if not torch.isfinite(estimate).all():
                    raise FloatingPointError('Nonfinite prediction')
                predicted.append(estimate.cpu().numpy())
                targets.append(target.numpy())
                frame_ids.append(sequence['frame_ids'][frame.numpy()])
            predictions.append((sequence, np.concatenate(frame_ids), np.concatenate(predicted), np.concatenate(targets)))
    return predictions


def _node_mean(predictions):
    # Macro-average sequences so a long random trajectory cannot dominate selection.
    return float(np.mean([np.linalg.norm(p - y, axis=-1).mean() for _, _, p, y in predictions]))


def train_run(manifest, output, config):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    resolved = dict(config)
    resolved.update(dataset_manifest=str(manifest), dataset_manifest_sha256=sha256(manifest),
                    study_id=output.parent.name, run_id=output.name,
                    optimizer='Adam', scheduler='none', early_stopping='disabled',
                    selection={'role': 'val', 'metric': 'sequence_macro_node_mean_mm', 'direction': 'min'},
                    history_protocol='causal window reset; current action at window end; target frame only',
                    H=config['history'], K_train=1, K_eval=1, seed_aggregation='within original sequence')
    write_json(output / 'resolved_config.json', resolved)
    run_manifest = dict(schema_version=2, status='running', run_kind=config['run_kind'],
                        dataset_manifest=str(manifest), dataset_manifest_sha256=sha256(manifest),
                        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        git_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                        seed=config['seed'], command=sys.argv, resolved_config='resolved_config.json',
                        selection=resolved['selection'], expected_outputs=['best_eval_model.pt', 'history.json', 'COMPLETE'])
    write_json(output / 'run_manifest.json', run_manifest)
    try:
        run_manifest['source_hashes'] = _archive_code(output)
        # Test arrays are intentionally unopened during training/model selection.
        meta, sequences = load_sequences(manifest, roles=('train', 'val'))
        train = [s for s in sequences if s['record']['role'] == 'train']
        val = [s for s in sequences if s['record']['role'] == 'val']
        if not train or not val:
            raise ValueError('Training requires disjoint train and validation groups')
        resolved['dt'] = meta['dt']
        write_json(output / 'resolved_config.json', resolved)
        _seed(config['seed'])
        torch.set_num_threads(config.get('threads', 2))
        device = torch.device(config['device'])
        center_np, scale = fit_normalization(train)
        center = torch.tensor(center_np, device=device)
        windows = CausalWindows(train, config['history'], config['train_stride'], config.get('max_train_windows'))
        model, geometry_config = make_model(config['model'], resolved, train, (center_np, scale))
        model.to(device)
        parameters = [p for p in model.parameters() if p.requires_grad]
        run_manifest.update(parameter_count=sum(p.numel() for p in model.parameters()),
                            trainable_parameter_count=sum(p.numel() for p in parameters),
                            fit_frames=sum(len(s['actions']) for s in train),
                            prior_fit_frames=len(windows) if geometry_config else 0)
        loader = DataLoader(windows, batch_size=config['batch_size'], shuffle=True,
                            generator=torch.Generator().manual_seed(config['seed']))
        optimizer = torch.optim.Adam(parameters, lr=config['lr']) if parameters else None
        if isinstance(model, Polynomial):
            # Same supervised window endpoints as neural methods.
            pairs = [windows[i] for i in range(len(windows))]
            x = torch.as_tensor(np.stack([p[0] for p in pairs]), device=device)
            y = (torch.as_tensor(np.stack([p[1] for p in pairs]), device=device) - center) / scale
            model.fit(x, y, config['ridge'])
        history, best = [], float('inf')
        epochs = config['epochs'] if optimizer else 1
        for epoch in range(1, epochs + 1):
            losses = []
            if optimizer:
                model.train()
                for actions, target, _, _ in loader:
                    actions, target = actions.to(device), (target.to(device)-center)/scale
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(actions)
                    loss = (prediction - target).square().mean()
                    loss = loss + config['endpoint_weight'] * (prediction[:, -1] - target[:, -1]).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite training loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 10.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            validation = _predict(model, val, resolved, center, scale, device, config.get('max_eval_windows'))
            metric = _node_mean(validation)
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None,
                                validation_node_mean_mm=metric))
            if metric < best:
                best = metric
                torch.save(dict(schema='shape_modeling_checkpoint_v1', model=config['model'],
                                state_dict=model.state_dict(), config=resolved, geometry_config=geometry_config,
                                center=center_np.tolist(), scale=scale, selected_epoch=epoch,
                                validation_node_mean_mm=best), output / 'best_eval_model.pt')
            write_json(output / 'history.json', history)
            print(f"{output.name}: epoch {epoch}/{epochs}, val node mean {metric:.4f} mm", flush=True)
        run_manifest.update(status='complete', wall_seconds=time.perf_counter()-start,
                            best_validation_node_mean_mm=best, supervised_windows=len(windows),
                            checkpoint_sha256=sha256(output / 'best_eval_model.pt'))
        write_json(output / 'run_manifest.json', run_manifest)
        (output / 'COMPLETE').write_text('Training and validation selection complete.\n')
    except Exception:
        run_manifest.update(status='failed', error=traceback.format_exc())
        write_json(output / 'run_manifest.json', run_manifest)
        raise
    return output


def evaluate_run(run, output, *, role='val', masks=True, radius_mm=8., mask_stride=1,
                 boundary_tolerance_px=2., device='cpu',verify_mask_hashes=True):
    """Evaluate an immutable validation-selected checkpoint into a fresh directory."""
    if role not in ('val', 'test') or radius_mm <= 0 or mask_stride < 1:
        raise ValueError('Invalid evaluation settings')
    run, output = Path(run).resolve(), Path(output).resolve()
    if not (run / 'COMPLETE').exists():
        raise ValueError('Training run is incomplete')
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', role=role, run=str(run), mask_adapter='fixed_radius_planar_tube',
                  masks_enabled=masks, radius_mm=radius_mm, radius_source='predeclared platform nominal diameter 16 mm',
                  mask_stride=mask_stride, boundary_tolerance_px=boundary_tolerance_px)
    status['verify_mask_hashes']=verify_mask_hashes
    status['evaluation_code_hashes'] = {str(p.relative_to(ROOT)): sha256(p) for p in [
        ROOT / 'src/benchmarks/modeling_runner.py', ROOT / 'src/evaluation/modeling_benchmark_metrics.py']}
    write_json(output / 'evaluation_manifest.json', status)
    try:
        checkpoint_path = run / 'best_eval_model.pt'
        training_manifest = json.loads((run / 'run_manifest.json').read_text())
        if sha256(checkpoint_path) != training_manifest['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed after training')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if sha256(config['dataset_manifest']) != config['dataset_manifest_sha256']:
            raise ValueError('Manifest changed since training')
        meta, sequences = load_sequences(config['dataset_manifest'], roles=(role,))
        if not sequences:
            raise ValueError('No sequences in requested evaluation role')
        torch.set_num_threads(config.get('threads', 2))
        model, _ = make_model(checkpoint['model'], config,
                              normalization=(checkpoint['center'], checkpoint['scale']),
                              geometry_config=checkpoint['geometry_config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        center = torch.tensor(checkpoint['center'], device=device)
        start = time.perf_counter()
        predictions = _predict(model, sequences, config, center, checkpoint['scale'], device,
                               config.get('max_eval_windows'))
        prediction_seconds = time.perf_counter() - start
        records = []
        for seq, frame_ids, prediction, target in predictions:
            row = seq['record']
            values = skeleton_metrics(prediction, target)
            summary = {key: float(np.mean(value)) for key, value in values.items()}
            summary['node_rmse_mm'] = float(np.sqrt(np.mean(values['node_rmse_mm']**2)))
            summary['node_p95_mm'] = float(np.quantile(np.linalg.norm(prediction-target, axis=-1), .95))
            summary['endpoint_p95_mm'] = float(np.quantile(values['endpoint_mm'], .95))
            mask_values, mask_frame_ids, annotation_adapter = [], [], []
            if masks:
                inventory_path = resolve(row['mask_inventory'], seq['manifest_dir'])
                if verify_mask_hashes and sha256(inventory_path) != row['mask_inventory_sha256']:
                    raise ValueError('Mask inventory changed')
                inventory = {r['frame']: r['sha256'] for r in json.loads(inventory_path.read_text())}
                matrix = seq['model_to_mask']
                # A circular projected tube requires an affine similarity transform.
                linear = matrix[:2, :2]
                if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose(linear.T @ linear, np.eye(2)*np.sum(linear[:, 0]**2)):
                    raise ValueError('Fixed tube radius requires similarity projection')
                radius_px = radius_mm * np.linalg.norm(linear[:, 0])
                for j in range(0, len(frame_ids), mask_stride):
                    frame = int(frame_ids[j])
                    path = resolve(row['masks']) / f'{frame:05d}.png'
                    if verify_mask_hashes and sha256(path) != inventory[frame]:
                        raise ValueError('Annotation hash changed')
                    target_mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if target_mask is None or list(target_mask.shape) != row['mask_shape']:
                        raise ValueError('Annotation dimensions disagree with crop')
                    prediction_mask = render_tube(prediction[j], matrix, target_mask.shape, radius_px)
                    scores = mask_metrics(prediction_mask, target_mask > 0, boundary_tolerance_px)
                    scores.pop('boundary_tolerance_px', None)
                    mask_values.append(scores)
                    label_tube = render_tube(target[j], matrix, target_mask.shape, radius_px)
                    annotation_adapter.append(mask_metrics(label_tube, target_mask > 0, boundary_tolerance_px))
                    mask_frame_ids.append(frame)
                summary.update({f'mask_{k}': float(np.mean([v[k] for v in mask_values])) for k in mask_values[0]})
            arrays = dict(prediction_mm=prediction, target_mm=target, frame_ids=frame_ids, **values,
                          mask_frame_ids=np.asarray(mask_frame_ids, dtype=int))
            if mask_values:
                arrays.update({f'mask_{k}': np.asarray([v[k] for v in mask_values]) for k in mask_values[0]})
            np.savez_compressed(output / f"{row['group']}_predictions.npz", **arrays)
            record = dict(model=config['model'], seed=config['seed'], group=row['group'], metrics=summary,
                          role=role, frames=len(frame_ids), mask_frames=len(mask_frame_ids), fold=meta['fold'],
                          run_kind=config['run_kind'], dataset_manifest_sha256=config['dataset_manifest_sha256'],
                          history=config['history'], eval_stride=config['eval_stride'],
                          label_tube_consistency={k: float(np.mean([v[k] for v in annotation_adapter]))
                                                  for k in ('iou', 'dice', 'boundary_f1')} if annotation_adapter else None)
            records.append(record)
        write_json(output / 'records.json', records)
        status.update(status='complete', run_kind=config['run_kind'],
                      checkpoint_sha256=sha256(checkpoint_path), selected_epoch=checkpoint['selected_epoch'],
                      dataset_manifest_sha256=config['dataset_manifest_sha256'],
                      annotation_source=meta['label_source'], evidence_level=meta['evidence_level'],
                      inference_seconds_including_loader=prediction_seconds,
                      prediction_frames=sum(len(item[1]) for item in predictions))
        write_json(output / 'evaluation_manifest.json', status)
        (output / 'COMPLETE').write_text('Evaluation complete.\n')
    except Exception:
        status.update(status='failed', error=traceback.format_exc())
        write_json(output / 'evaluation_manifest.json', status)
        raise
    return records


def aggregate(evaluations, output, reference='hov', metrics=None, allow_smoke=False):
    records, contracts = [], []
    for directory in map(Path, evaluations):
        if not (directory / 'COMPLETE').exists():
            raise ValueError(f'Incomplete evaluation: {directory}')
        status = json.loads((directory / 'evaluation_manifest.json').read_text())
        rows = json.loads((directory / 'records.json').read_text())
        if not allow_smoke and any(r['run_kind'] == 'smoke' for r in rows):
            raise ValueError('Smoke results require --allow-smoke and remain diagnostic')
        contracts.append((status['role'], status['masks_enabled'], status['radius_mm'], status['mask_stride'], status['boundary_tolerance_px'],
                          json.dumps(status['evaluation_code_hashes'], sort_keys=True)))
        records.extend(rows)
    if len(set(contracts)) != 1 or len({(r['history'], r['eval_stride']) for r in records}) != 1:
        raise ValueError('Cannot pool incompatible evaluation protocols')
    # A given held-out group must refer to the same split across every model/seed.
    for group in {r['group'] for r in records}:
        if len({r['dataset_manifest_sha256'] for r in records if r['group'] == group}) != 1:
            raise ValueError('Models used different training/validation splits')
    if metrics is None:
        metrics = sorted(set.intersection(*(set(r['metrics']) for r in records)))
    directions = {k: 'higher' if k.startswith('mask_') else 'lower' for k in metrics}
    report = compare_models(records, reference, directions)
    report['interpretation'] = 'Diagnostic smoke only' if any(r['run_kind'] == 'smoke' for r in records) else (
        'Paired sequence-level exploratory inference; LOSO training folds overlap, '
        'and collection-day effects limit generalization. Confirm on fresh held-out sequences.')
    report['evaluation_role'] = contracts[0][0]
    if contracts[0][0] == 'test' and status.get('evidence_level') == 'within_sequence':
        report['interpretation'] = ('Paired held-out temporal slices grouped by original sequence; seeds averaged within sequence. '
                                    'Training and test share acquisition sequences/day, so evidence concerns temporal generalization.')
    report['inference_eligible'] = contracts[0][0] == 'test' and not any(r['run_kind'] == 'smoke' for r in records)
    if not report['inference_eligible']:
        for comparison in report['comparisons']:
            comparison['significant'] = False
            if not any(r['run_kind'] == 'smoke' for r in records):
                comparison['status'] = 'diagnostic_validation'
        if contracts[0][0] == 'val' and not any(r['run_kind'] == 'smoke' for r in records):
            report['interpretation'] = ('Validation data selected the checkpoints. Differences and nominal p values '
                                        'are development diagnostics; they do not support significance claims.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'statistics.json', report)
    columns = ['model', 'seed', 'group', 'role', *metrics]
    with (output / 'per_sequence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({**{k: row[k] for k in columns[:4]}, **{k: row['metrics'].get(k) for k in metrics}})
    summaries = []
    for model in sorted({r['model'] for r in records}):
        selected = [r for r in records if r['model'] == model]
        seeds = sorted({r['seed'] for r in selected})
        groups = sorted({r['group'] for r in selected})
        for metric in metrics:
            seed_means = [np.mean([r['metrics'][metric] for r in selected if r['seed'] == seed]) for seed in seeds]
            group_means = [np.mean([r['metrics'][metric] for r in selected if r['group'] == group]) for group in groups]
            summaries.append(dict(model=model, metric=metric, mean=float(np.mean(group_means)),
                                  seed_std=float(np.std(seed_means, ddof=1)) if len(seeds)>1 else None,
                                  group_std=float(np.std(group_means, ddof=1)) if len(groups)>1 else None,
                                  n_seeds=len(seeds), n_groups=len(groups)))
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return report
