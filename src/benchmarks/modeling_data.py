"""Immutable sequence-grouped datasets and causal windows for shape modeling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def resolve(path, base=ROOT):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def prepare_grouped(source_manifest, output, *, fold=0, val_groups=1):
    """Recover full original sequences from deployment provenance, then split.

    Original train+val source files contain the embargo frames that deployment
    datasets omit. Recovery is verified against raw physical commands. Existing
    historical labels support an exploratory cross-sequence benchmark.
    """
    source_manifest = Path(source_manifest).resolve()
    source = json.loads(source_manifest.read_text())
    if source.get('schema') != 'native_rate_hov_deployment_dataset_v1':
        raise ValueError('Expected native-rate deployment provenance manifest')
    rows = {row['sequence']: row for row in source['files']}
    groups = sorted(rows)
    if not 0 <= fold < len(groups) or not 1 <= val_groups <= len(groups) - 2:
        raise ValueError('Need disjoint train/val/test sequences and valid fold')
    test = groups[fold]
    val = {groups[(fold + 1 + j) % len(groups)] for j in range(val_groups)}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for group in groups:
        row = rows[group]
        files = row['source_files']
        if len(files) != 2:
            raise ValueError('Expected original chronological train + val pair')
        parts = []
        for item in files:
            path = resolve(item['path'])
            if sha256(path) != item['sha256']:
                raise ValueError(f'Source hash mismatch: {path}')
            with np.load(path, allow_pickle=False) as data:
                parts.append({k: data[k].copy() for k in data.files})
        for part in parts:
            if str(part['node_order']) != 'base_to_tip' or str(part['state_length_unit']) != 'mm':
                raise ValueError('Require base_to_tip skeletons in mm')
            if not np.array_equal(part['model_action_channels'], [0, 1, 3, 5]):
                raise ValueError('Action channel contract mismatch')
        physical = np.concatenate([p['actions'] * p['raw_action_scale6_kpa'] for p in parts])
        raw = ROOT / 'workspace/data/raw/real' / group
        expected = np.loadtxt(raw / 'actions6.csv', delimiter=',', skiprows=1)
        dt = float(json.loads((raw / 'meta.json').read_text())['action_interval_s'])
        if not np.isclose(dt, source['dt']):
            raise ValueError('Mixed native rates are not supported')
        if expected[:, 1:7].shape != physical.shape or not np.allclose(expected[:, 1:7], physical, atol=.03):
            raise ValueError(f'Original chronological frame mapping failed: {group}')
        if not np.allclose(physical[:, [1, 3]], physical[:, [2, 4]], atol=.5):
            raise ValueError('Coupled pressure channels disagree')
        camera = np.concatenate([p['positions_camera_px'] for p in parts]).transpose(0, 2, 1)
        transform = source['state_calibration']
        to_robot = np.asarray(transform['camera_to_model_matrix'])
        xy1 = np.concatenate([camera[..., :2], np.ones((*camera.shape[:2], 1))], -1) @ to_robot.T
        positions = np.concatenate([xy1[..., :2] / xy1[..., 2:], np.zeros((*camera.shape[:2], 1))], -1)
        crop = np.asarray(parts[0]['image_crop_xywh'], dtype=int)
        if not all(np.array_equal(p['image_crop_xywh'], crop) for p in parts):
            raise ValueError('Crop changed within sequence')
        crop_shift = np.array([[1, 0, -crop[0]], [0, 1, -crop[1]], [0, 0, 1]])
        model_to_mask = crop_shift @ np.asarray(transform['model_to_camera_matrix'])
        # Each mask is a real SAM2 annotation in crop-local coordinates.
        intermediate = ROOT / 'workspace/data/intermediate/real' / group
        candidates = sorted(intermediate.rglob('sam2_masks'))
        legacy = intermediate / 'sam2-video-v1'
        if legacy.exists():
            candidates.append(legacy)
        mask_dirs = [p for p in candidates if (p / '00000.png').exists()]
        if len(mask_dirs) != 1:
            raise ValueError(f'Expected exactly one annotation directory: {group}: {mask_dirs}')
        mask_dir = mask_dirs[0]
        crop_meta_path = (intermediate / 'legacy-derived/crop/crop_meta.json'
                          if mask_dir == legacy else mask_dir.parent / 'crop/crop_meta.json')
        crop_meta = json.loads(crop_meta_path.read_text())
        if not np.array_equal(crop_meta['crop_xywh'], crop) or crop_meta['n_output_frames'] != len(physical):
            raise ValueError('Annotation crop/frame provenance mismatch')
        mask_inventory = []
        for i in range(len(physical)):
            path = mask_dir / f'{i:05d}.png'
            if not path.exists():
                raise FileNotFoundError(path)
            mask_inventory.append({'frame': i, 'sha256': sha256(path)})
        inventory_path = output / f'{group}_masks.json'
        write_json(inventory_path, mask_inventory)
        path = output / f'{group}.npz'
        np.savez_compressed(path, actions=(physical[:, [0, 1, 3, 5]] / 150).astype('float32'),
                            positions=positions.astype('float32'), frame_ids=np.arange(len(physical)),
                            timestamps=expected[:, 0], model_to_mask=model_to_mask)
        role = 'test' if group == test else 'val' if group in val else 'train'
        records.append(dict(group=group, role=role, path=path.name, sha256=sha256(path),
                            frames=len(physical), masks=str(mask_dir), mask_shape=[int(crop[3]), int(crop[2])],
                            mask_inventory=inventory_path.name, mask_inventory_sha256=sha256(inventory_path),
                            crop_meta=str(crop_meta_path), crop_meta_sha256=sha256(crop_meta_path),
                            source_files=files, source_manifest=row['source_manifest'],
                            timing_jitter_std_s=float(np.diff(expected[:, 0]).std())))
    manifest = dict(schema='shape_modeling_grouped_v1', dataset_id=output.name,
                    evidence_level='cross_sequence', study_stage='exploratory_historical_corpus',
                    grouping_key='independent_original_sequence', split_seed=None, fold=fold,
                    split_policy='sorted leave-one-sequence-out; next sequence(s) validation',
                    embargo_frames=0, H='resolved in run', K_train=1, K_eval=1,
                    dt=source['dt'], action_scale_kpa=[150] * 4, node_order='base_to_tip',
                    length_unit='mm', coordinate_frame='robot_planar_mm_v1',
                    calibration=source['state_calibration'],
                    calibration_evidence='inherited fixed capture-day calibration; validate independently for final paper',
                    label_source='SAM2 masks and derived centerline; correlated annotations, not independent ground truth',
                    source_manifest=str(source_manifest), source_manifest_sha256=sha256(source_manifest), files=records)
    write_json(output / 'dataset_manifest.json', manifest)
    return output / 'dataset_manifest.json'


def load_sequences(manifest_path, roles=('train', 'val', 'test')):
    manifest_path = Path(manifest_path).resolve()
    meta = json.loads(manifest_path.read_text())
    if meta.get('schema') not in ('shape_modeling_grouped_v1', 'shape_modeling_temporal_pool_v1') or meta.get('length_unit') != 'mm' or meta.get('node_order') != 'base_to_tip':
        raise ValueError('Unsupported modeling dataset contract')
    if not np.isfinite(meta['dt']) or meta['dt'] <= 0:
        raise ValueError('Invalid sampling interval')
    temporal = meta['schema'] == 'shape_modeling_temporal_pool_v1'
    if temporal:
        for group in {r['group'] for r in meta['files']}:
            parts = sorted((r for r in meta['files'] if r['group']==group), key=lambda r:r['start'])
            if [r['role'] for r in parts] != ['train','val','test']:
                raise ValueError('Temporal split requires three ordered disjoint slices')
            if parts[0]['start'] != 0 or parts[-1]['stop'] != parts[0]['original_frames']:
                raise ValueError('Temporal split does not cover original sequence')
            if any(a['stop'] != b['start'] for a,b in zip(parts,parts[1:])):
                raise ValueError('Overlapping or missing temporal split frames')
            if any(r['frames'] != r['stop'] - r['start'] for r in parts):
                raise ValueError('Temporal slice frames must equal stop-start')
            if len({r['parent_sha256'] for r in parts}) != 1:
                raise ValueError('Temporal slices have different parent sequences')
    seen = set()
    sequences = []
    for row in meta['files']:
        identity = (row['group'], row['role']) if temporal else row['group']
        if identity in seen:
            raise ValueError('A sequence may appear in exactly one split record')
        seen.add(identity)
        if row['role'] not in ('train', 'val', 'test'):
            raise ValueError('Unknown split role')
        if row['role'] not in roles:
            continue
        path = resolve(row['path'], manifest_path.parent)
        if sha256(path) != row['sha256']:
            raise ValueError(f'Dataset hash mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            seq = {key: data[key].copy() for key in data.files}
        x, y = seq['actions'], seq['positions']
        if x.shape != (len(y), 4) or y.shape != (row['frames'], 15, 3):
            raise ValueError('Expected (T,4) actions and (T,15,3) skeletons')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite modeling data')
        if not np.array_equal(seq['frame_ids'], np.arange(row.get('start', 0), row.get('start', 0)+len(y))):
            raise ValueError('Require full contiguous original sequence')
        if np.any(np.diff(seq['timestamps']) <= 0):
            raise ValueError('Nonmonotonic timestamps')
        seq['record'] = row
        seq['manifest_dir'] = manifest_path.parent
        sequences.append(seq)
    return meta, sequences


class CausalWindows(Dataset):
    """Windows end at the target, stay within one sequence, and exclude startup."""
    def __init__(self, sequences, history=20, stride=1, max_windows=None):
        if history < 2 or stride < 1 or (max_windows is not None and max_windows < 1):
            raise ValueError('Invalid causal window settings')
        self.sequences = sequences
        self.history = history
        self.indices = [(i, t) for i, s in enumerate(sequences)
                        for t in range(history - 1, len(s['actions']), stride)]
        if max_windows and len(self.indices) > max_windows:
            indices = np.linspace(0, len(self.indices) - 1, max_windows, dtype=int)
            self.indices = [self.indices[i] for i in indices]
        if not self.indices:
            raise ValueError('No complete causal windows')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i, t = self.indices[index]
        seq = self.sequences[i]
        return seq['actions'][t-self.history+1:t+1], seq['positions'][t], i, t


def prepare_temporal_pool(source_manifest, output, history=20):
    """Pool chronological 60/20/20 slices of all original sequences.

    Largest-remainder allocation preserves the exact global frame proportions;
    causal windows are built separately within each sequence/split slice.
    """
    source_manifest = Path(source_manifest).resolve()
    meta, sequences = load_sequences(source_manifest)
    if meta['schema'] != 'shape_modeling_grouped_v1' or not np.isclose(meta['dt'], .2):
        raise ValueError('Require full original 5 Hz sequences')
    sequences.sort(key=lambda s: s['record']['group'])
    lengths = np.array([len(s['actions']) for s in sequences])
    if len(lengths) != 7:
        raise ValueError('This study requires all seven original sequences')
    def allocate(fraction):
        exact = lengths * fraction
        counts = np.floor(exact).astype(int)
        remainder = round(int(lengths.sum()) * fraction) - counts.sum()
        for i in np.argsort(-(exact-counts), kind='stable')[:remainder]:
            counts[i] += 1
        return counts
    train_counts, val_counts = allocate(.6), allocate(.2)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, seq in enumerate(sequences):
        source_row = seq['record']
        boundaries = [0, int(train_counts[i]), int(train_counts[i]+val_counts[i]), int(lengths[i])]
        for role, start, stop in zip(('train', 'val', 'test'), boundaries[:-1], boundaries[1:]):
            if stop-start < history:
                raise ValueError('Slice too short for a complete history window')
            path = output / role / f"{source_row['group']}.npz"
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, actions=seq['actions'][start:stop], positions=seq['positions'][start:stop],
                                frame_ids=seq['frame_ids'][start:stop], timestamps=seq['timestamps'][start:stop],
                                model_to_mask=seq['model_to_mask'])
            row = dict(source_row, role=role, path=str(path), sha256=sha256(path), frames=stop-start,
                       start=start, stop=stop, original_frames=int(lengths[i]),
                       parent_file=str(resolve(source_row['path'], source_manifest.parent)),
                       parent_sha256=source_row['sha256'],
                       mask_inventory=str(resolve(source_row['mask_inventory'], source_manifest.parent)))
            records.append(row)
    result = dict(meta, schema='shape_modeling_temporal_pool_v1', dataset_id=output.name,
                  evidence_level='within_sequence', study_stage='formal_prespecified_historical_holdout',
                  fold=None, split_seed=None, split_ratio=[.6,.2,.2],
                  split_policy='chronological 60/20/20 per original sequence, pooled by role; largest remainder frame allocation',
                  embargo_frames=0, history_context='each split owns its first H-1 context frames; no cross-split history',
                  H=history, source_manifest=str(source_manifest), source_manifest_sha256=sha256(source_manifest),
                  counts={role:sum(r['frames'] for r in records if r['role']==role) for role in ('train','val','test')},
                  scored_counts={role:sum(r['frames']-history+1 for r in records if r['role']==role) for role in ('train','val','test')},
                  files=records)
    write_json(output/'dataset_manifest.json', result)
    load_sequences(output/'dataset_manifest.json')
    return output/'dataset_manifest.json'
