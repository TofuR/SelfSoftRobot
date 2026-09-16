#!/usr/bin/env python3
"""Sequence-grouped whole-body modeling benchmark (ICRA 2027)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmarks.modeling_data import prepare_grouped, write_json
from src.benchmarks.modeling_models import MODEL_NAMES

DEFAULTS = dict(model='hov', seed=0, run_kind='exploratory', history=20, hidden=64,
                epochs=100, batch_size=128, lr=.001, ridge=.0001,
                endpoint_weight=.25, prior_steps=500, train_stride=1, eval_stride=1,
                max_train_windows=None, max_eval_windows=None, threads=2, device='cpu')


def training_options(parser):
    parser.add_argument('--config', type=Path, help='JSON overrides for training defaults')
    parser.add_argument('--model', choices=MODEL_NAMES)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--run-kind', choices=['smoke', 'exploratory'])
    for name in ['history', 'hidden', 'epochs', 'batch_size', 'prior_steps', 'train_stride',
                 'eval_stride', 'max_train_windows', 'max_eval_windows', 'threads']:
        parser.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ['lr', 'ridge', 'endpoint_weight']:
        parser.add_argument('--' + name.replace('_', '-'), type=float)
    parser.add_argument('--device')


def configuration(args):
    config = dict(DEFAULTS)
    if args.config:
        supplied = json.loads(args.config.read_text())
        if set(supplied) - set(config):
            raise ValueError(f'Unknown config fields: {sorted(set(supplied)-set(config))}')
        config.update(supplied)
    for key in config:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if config['model'] not in MODEL_NAMES or config['run_kind'] not in ('smoke', 'exploratory'):
        raise ValueError('Invalid model/run kind')
    for key in ['history', 'hidden', 'epochs', 'batch_size', 'prior_steps', 'train_stride', 'eval_stride', 'threads']:
        if config[key] < (2 if key == 'history' else 1):
            raise ValueError(f'Invalid {key}')
    if config['lr'] <= 0 or config['ridge'] <= 0 or config['endpoint_weight'] < 0:
        raise ValueError('Invalid objective/optimizer parameters')
    if config['run_kind'] != 'smoke' and (config['max_train_windows'] or config['max_eval_windows']):
        raise ValueError('Window truncation is reserved for explicitly labeled smoke runs')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare', help='Recover original sequences and create immutable grouped split')
    prepare.add_argument('--source-manifest', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--fold', type=int, default=0)
    prepare.add_argument('--val-groups', type=int, default=1)
    train = sub.add_parser('train', help='Train and select checkpoint using validation only')
    train.add_argument('--manifest', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    training_options(train)
    evaluate = sub.add_parser('evaluate', help='Evaluate a frozen selected checkpoint')
    evaluate.add_argument('--run', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, required=True)
    evaluate.add_argument('--role', choices=['val', 'test'], default='val')
    evaluate.add_argument('--skeleton-only', action='store_true')
    evaluate.add_argument('--radius-mm', type=float, default=8.)
    evaluate.add_argument('--mask-stride', type=int, default=1)
    evaluate.add_argument('--boundary-tolerance-px', type=float, default=2.)
    evaluate.add_argument('--device', default='cpu')
    summary = sub.add_parser('aggregate', help='Paired group-level statistics and CSV')
    summary.add_argument('--evaluations', type=Path, nargs='+', required=True)
    summary.add_argument('--out', type=Path, required=True)
    summary.add_argument('--reference', default='hov')
    summary.add_argument('--metrics', nargs='+')
    summary.add_argument('--allow-smoke', action='store_true')
    sweep = sub.add_parser('sweep', help='Write a study plan; --execute runs the resolved plan')
    sweep.add_argument('--manifests', type=Path, nargs='+', required=True)
    sweep.add_argument('--models', choices=MODEL_NAMES, nargs='+', default=list(MODEL_NAMES))
    sweep.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    sweep.add_argument('--out', type=Path, required=True)
    sweep.add_argument('--execute', action='store_true')
    sweep.add_argument('--evaluate-role', choices=['val', 'test'], default='val')
    sweep.add_argument('--skeleton-only', action='store_true')
    training_options(sweep)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare_grouped(args.source_manifest, args.out, fold=args.fold, val_groups=args.val_groups))
        return
    from src.benchmarks.modeling_runner import train_run, evaluate_run, aggregate
    if args.command == 'train':
        train_run(args.manifest, args.out, configuration(args))
    elif args.command == 'evaluate':
        evaluate_run(args.run, args.out, role=args.role, masks=not args.skeleton_only,
                     radius_mm=args.radius_mm, mask_stride=args.mask_stride,
                     boundary_tolerance_px=args.boundary_tolerance_px, device=args.device)
    elif args.command == 'aggregate':
        aggregate(args.evaluations, args.out, args.reference, args.metrics, args.allow_smoke)
    elif args.command == 'sweep':
        config = configuration(args)
        args.out.mkdir(parents=True, exist_ok=False)
        plan = []
        if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Duplicate model/seed requested')
        for fold, manifest in enumerate(args.manifests):
            for model in args.models:
                for seed in args.seeds:
                    plan.append(dict(manifest=str(manifest.resolve()),
                                     run=str((args.out / f'fold{fold}_{model}_seed{seed}').resolve()),
                                     config={**config, 'model': model, 'seed': seed}))
        write_json(args.out / 'sweep_plan.json', dict(jobs=plan, role=args.evaluate_role,
                                                     masks=not args.skeleton_only, execute=args.execute))
        print(f'{len(plan)} runs planned in {args.out}', flush=True)
        if args.execute:
            for job in plan:
                run = train_run(job['manifest'], job['run'], job['config'])
                evaluate_run(run, run / f'evaluation_{args.evaluate_role}', role=args.evaluate_role,
                             masks=not args.skeleton_only, device=config['device'])
            (args.out / 'COMPLETE').write_text('All planned training and evaluation jobs complete.\n')


if __name__ == '__main__':
    main()
