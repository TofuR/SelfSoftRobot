#!/usr/bin/env python3
"""Complete the internal MLP comparison using the byte-identical 006 trainer.

Run from the repository root with ``python scripts/experiments/
complete_internal_mlp_memory_comparison.py --workers 8``. Interrupted fits are
retained and restarted by the original runner; completed fits are reused.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback


def repository_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / '.git').exists():
            return parent
    raise RuntimeError('Run this script inside the SelfSoftRobot repository')


ROOT = repository_root()
SOURCE = ROOT / 'workspace/runs/training/modeling_internal_plugins_20260913_006'
DEFAULT_RUN = ROOT / 'workspace/runs/training/modeling_internal_mlp_single_memory_20260913_007'
DEFAULT_ANALYSIS = ROOT / 'workspace/runs/analysis/modeling_internal_mlp_single_memory_20260913_007'
SCRIPT_REL = Path('scripts/experiments/complete_internal_mlp_memory_comparison.py')
RUNNER_REL = Path('scripts/experiments/run_internal_memory_plugins.py')
SEEDS = list(range(100, 120))
NEW = ['path', 'time']
VARIANTS = ['base', 'path', 'time', 'both', 'static_capacity']
COUNTS = dict(base=7405, path=7937, time=8961, both=9473, static_capacity=9473)
METRICS = ['mean_node_mm', 'endpoint_mm', 'node_rmse_mm', 'node_global_rmse_mm', 'endpoint_rmse_mm']
CONTRASTS = ([dict(family='mlp_base_extensions', reference='mlp_base', alternative='mlp_' + v)
              for v in VARIANTS if v != 'base'] +
             [dict(family='mlp_both_single_mechanism', reference='mlp_' + v, alternative='mlp_both')
              for v in NEW])
R = None


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage(run, phase, **details):
    event = dict(at=stamp(), phase=phase, training_run=str(run),
                 analysis=str(read(run / 'protocol.json')['analysis']), **details)
    write(run / 'stage_status.json', event)
    with (run / 'stage_events.jsonl').open('a') as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def runtime(run):
    global R
    if R is not None:
        return R
    sys.dont_write_bytecode = True
    path = run / 'source' / RUNNER_REL
    spec = importlib.util.spec_from_file_location('_frozen_internal_006', path)
    R = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = R
    spec.loader.exec_module(R)
    R.runtime(run)
    assert R.torch.get_num_threads() == R.torch.get_num_interop_threads() == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert Path(sys.modules[R.InternalMemoryModel.__module__].__file__).resolve().is_relative_to(run / 'source')
    return R


def prepare(run, analysis, workers):
    if (run / 'PREPARED').exists():
        protocol = read(run / 'protocol.json')
        assert protocol['workers'] == workers and protocol['analysis'] == str(analysis)
        assert digest(run / 'source' / SCRIPT_REL) == digest(Path(__file__))
        for relative, sha in read(run / 'source_sha256.json').items():
            assert digest(run / 'source' / relative) == sha
        for path, sha in read(run / 'input_provenance.json')['files_sha256'].items():
            assert digest(path) == sha, f'Source changed: {path}'
        runtime(run)
        return protocol
    assert not (run / 'formal').exists()
    old = read(SOURCE / 'protocol.json')
    assert read(SOURCE / 'status.json')['status'] == 'complete'
    for key, value in dict(history=20, dt=.2, epochs=100, batch_size=256,
                           seeds=SEEDS, tuning_seed=900001, learning_rates=[.001, .003], threads=1).items():
        assert old[key] == value, key
    shutil.copytree(SOURCE / 'source', run / 'source', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(Path(__file__), run / 'source' / SCRIPT_REL)
    frozen = {str(p.relative_to(run / 'source')): digest(p)
              for p in sorted((run / 'source').rglob('*.py'))}
    for relative, sha in frozen.items():
        if relative != str(SCRIPT_REL):
            assert sha == digest(SOURCE / 'source' / relative)
    write(run / 'source_sha256.json', frozen)
    protocol = dict(old, schema='internal_mlp_single_memory_protocol_v1', created_at=stamp(),
                    source_run=str(SOURCE), analysis=str(analysis), workers=workers, device='cpu',
                    configurations=[dict(family='mlp', variant=v, parameters=COUNTS[v]) for v in NEW],
                    formal_fits=40, tuning_fits=4,
                    reused_configurations=['mlp_base', 'mlp_both', 'mlp_static_capacity'],
                    trainer='Byte-identical 006 fit_job/fit_impl/predict; frozen source import',
                    evaluation='Unchanged frozen evaluate body through raw CSV export; new joint statistics',
                    statistics=dict(primary='pooled test mean_node_mm',
                        families=dict(mlp_base_extensions=4, mlp_both_single_mechanism=2),
                        contrasts=CONTRASTS, alternative='two-sided',
                        method='Exact signed-rank conditional enumeration; drop zero differences; average tied ranks; Holm within each family',
                        bootstrap=dict(replicates=20000, seed=20260913, unit='paired training seed', interval='percentile 95%', estimand='mean paired difference'),
                        direction='reference error minus alternative error; positive favors alternative'))
    write(run / 'protocol.json', protocol)
    for filename in ['plugin_normalization.json', 'output_normalization.json']:
        shutil.copyfile(SOURCE / filename, run / filename)
    R = runtime(run)
    data = R.L.load_roles(Path(old['dataset_manifest']), ('train', 'val'))
    norm = read(run / 'plugin_normalization.json')
    assert R.fit_plugin_normalization(data['train']['x']) == norm
    center, scale = R.L.fit_normalization(data['train']['sequences'])
    assert dict(center=center.tolist(), scale=float(scale)) == read(run / 'output_normalization.json')
    counts = {}
    initial = {}
    for variant in VARIANTS:
        R.torch.manual_seed(100)
        model = R.InternalMemoryModel('mlp', variant, norm)
        counts['mlp_' + variant] = model.parameter_count()
        assert counts['mlp_' + variant] == COUNTS[variant]
        with R.torch.inference_mode():
            prediction = model(data['val']['x'][:3])
        assert tuple(prediction.shape) == (3, 15, 3) and R.torch.isfinite(prediction).all()
        initial[variant] = prediction
        assert R.torch.equal(prediction, initial['base'])
    # Hash only the named protocol/norm files and six train/val NPZ slices.
    paths = [SOURCE / f for f in ['protocol.json', 'selected_configuration.json',
                                 'plugin_normalization.json', 'output_normalization.json']]
    manifest = Path(old['dataset_manifest'])
    paths.append(manifest)
    for row in read(manifest)['files']:
        if row['role'] in ('train', 'val'):
            path = Path(row['path'])
            paths.append(path if path.is_absolute() else manifest.parent / path)
    write(run / 'input_provenance.json', dict(at=stamp(), files_sha256={str(p): digest(p) for p in paths},
          source_python_files=len(frozen) - 1, frozen_sources_equal_006=True,
          plugin_normalization_recomputed_from_train_exact=True,
          output_normalization_recomputed_from_train_exact=True, test_opened=False))
    import scipy
    write(run / 'preflight.json', dict(at=stamp(), counts=counts,
          train_windows=len(data['train']['x']), val_windows=len(data['val']['x']), test_opened=False,
          identical_initial_predictions=True, python=sys.version, executable=sys.executable,
          torch=R.torch.__version__, numpy=R.np.__version__, scipy=scipy.__version__,
          platform=platform.platform(), device='cpu', threads=1, workers=workers,
          deterministic_algorithms=R.torch.are_deterministic_algorithms_enabled()))
    (run / 'PREPARED').write_text(stamp() + '\n')
    stage(run, 'prepared', tuning_fits=4, formal_fits=40, test_opened=False)
    return protocol


def worker_init(run_string):
    runtime(Path(run_string)).worker_init(run_string)


def fit_job(job):
    return R.fit_job(job)


def execute(run, jobs, phase, workers):
    rows = []
    write(run / 'status.json', dict(phase=phase, status='running', completed=0, total=len(jobs), at=stamp()))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=worker_init, initargs=(str(run),)) as pool:
        futures = {pool.submit(fit_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            write(run / 'status.json', dict(phase=phase, status='running', completed=len(rows),
                  total=len(jobs), last=futures[future]['id'], at=stamp()))
            print(f'{phase} {len(rows)}/{len(jobs)} {row["model"]} seed={row["seed"]} '
                  f'val={row["best_validation_node_mean_mm"]:.6f}', flush=True)
    return sorted(rows, key=lambda row: (row['model'], row['seed'], row['config']['lr']))


def fit_audit(run, jobs):
    R = runtime(run)
    norm = read(run / 'plugin_normalization.json')
    out = read(run / 'output_normalization.json')
    checks = []
    new_selection = read(run / 'selected_configuration.json')
    old_selection = read(SOURCE / 'selected_configuration.json')
    for variant in VARIANTS:
        source = run if variant in NEW else SOURCE
        selection = new_selection if variant in NEW else old_selection
        selected_lr = selection['configurations']['mlp_' + variant]['lr']
        for seed in SEEDS:
            directory = source / f'formal/mlp_{variant}/seed_{seed}'
            assert (directory / 'COMPLETE').exists()
            state = read(directory / 'run_manifest.json')
            history = read(directory / 'history.json')
            checkpoint = directory / 'best_eval_model.pt'
            ck = R.torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['status'] == 'complete' and state['current_epoch'] == 100
            assert state['config'] == ck['config']
            assert ck['config']['seed'] == seed and ck['config']['lr'] == selected_lr
            assert ck['family'] == 'mlp' and ck['variant'] == variant
            assert ck['normalization'] == norm and dict(center=ck['center'], scale=ck['scale']) == out
            for key in ['input_mean', 'input_std']:
                assert R.torch.equal(ck['state_dict'][key], R.torch.tensor(norm[key]))
            for key in ['memory_mean', 'memory_std']:
                assert R.torch.equal(ck['state_dict'][key], R.torch.tensor(norm[key][variant]))
            assert [h['epoch'] for h in history] == [1] + list(range(5, 101, 5))
            best = min(history, key=lambda h: h['validation_node_mean_mm'])
            assert best['epoch'] == ck['selected_epoch'] == state['best_epoch']
            assert best['validation_node_mean_mm'] == ck['validation_node_mean_mm'] == state['best_validation_node_mean_mm']
            assert state['train_windows'] == 8988 and state['val_windows'] == 2958
            assert state['parameter_count'] == COUNTS[variant]
            baseline = read(SOURCE / f'formal/mlp_base/seed_{seed}/run_manifest.json')
            assert state['epoch0_validation_node_mean_mm'] == baseline['epoch0_validation_node_mean_mm']
            if variant in NEW:
                assert state['started_at'] >= new_selection['frozen_at']
            checks.append(dict(model='mlp_' + variant, seed=seed, source_run=str(source),
                               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                               normalization_matches=True, selected_epoch=ck['selected_epoch'], lr=selected_lr))
    assert len(jobs) == 40 and len(checks) == 100
    write(run / 'fit_normalization_audit.json', dict(at=stamp(), status='pass', fits=100,
          matched_epoch0_validation=True, frozen_training_source_matches_006=True,
          normalization_matches=True, checks=checks))


def evaluate_new(run, jobs):
    """Reuse the frozen evaluator verbatim up to its old summary/statistics tail."""
    R = runtime(run)
    tree = ast.parse((run / 'source' / RUNNER_REL).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    boundary = next(i for i, node in enumerate(function.body)
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'contrasts' for t in node.targets))
    assert isinstance(function.body[boundary].value, ast.Call)
    assert function.body[boundary].value.func.id == 'paired_statistics'
    function.body = function.body[:boundary] + [ast.parse('return rows, sequence_rows').body[0]]
    function.name = '_evaluate_supplement_only'
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(run / 'source' / RUNNER_REL), 'exec'), R.__dict__)
    return R._evaluate_supplement_only(run, jobs)


def paired_statistics(rows):
    from scipy import stats
    np = R.np
    lookup = {(row['model'], row['seed']): row['mean_node_mm'] for row in rows}
    indices = np.random.default_rng(20260913).integers(0, 20, size=(20000, 20))
    results = []
    for contrast in CONTRASTS:
        delta = np.array([lookup[(contrast['reference'], seed)] - lookup[(contrast['alternative'], seed)] for seed in SEEDS])
        active = delta[delta != 0]
        ranks = np.rint(2 * stats.rankdata(np.abs(active))).astype(int)
        counts = np.zeros(int(ranks.sum()) + 1, dtype=np.int64)
        counts[0] = 1
        for rank in ranks:
            previous = counts.copy()
            counts[rank:] += previous[:-rank]
        observed = int(ranks[active > 0].sum())
        p = min(1., 2 * min(counts[:observed + 1].sum(), counts[observed:].sum()) / 2 ** len(active))
        ci = np.quantile(delta[indices].mean(1), [.025, .975])
        results.append(dict(contrast, n_pairs=20, seeds=SEEDS,
            mean_reference_minus_alternative_mm=float(delta.mean()),
            bootstrap95_lower_mm=float(ci[0]), bootstrap95_upper_mm=float(ci[1]),
            positive_pairs=int((delta > 0).sum()), negative_pairs=int((delta < 0).sum()), zero_pairs=int((delta == 0).sum()),
            wilcoxon_exact_p=float(p),
            sign_test_exact_p=float(stats.binomtest(int((active > 0).sum()), len(active)).pvalue) if len(active) else 1.,
            seed_differences_mm=delta.tolist()))
    for family, size in read(RUN_FOR_STATS / 'protocol.json')['statistics']['families'].items():
        group = [r for r in results if r['family'] == family]
        assert len(group) == size
        for key in ['wilcoxon_exact_p', 'sign_test_exact_p']:
            for row, adjusted in zip(group, R.L.holm([r[key] for r in group])):
                row[key.replace('exact', 'holm')] = adjusted
    return results


def combine(run, analysis, new_rows, new_sequences):
    global RUN_FOR_STATS
    RUN_FOR_STATS = run
    R = runtime(run)
    np = R.np
    rows, sequence_rows, validation = [], [], []
    target_reference = None
    metric_checks = []
    for variant in VARIANTS:
        source = run if variant in NEW else SOURCE
        for seed in SEEDS:
            directory = source / f'evaluation/mlp_{variant}/seed_{seed}'
            assert (directory / 'COMPLETE').exists()
            row = read(directory / 'metrics.json')
            groups = read(directory / 'sequence_metrics.json')
            assert row['model'] == 'mlp_' + variant and row['seed'] == seed and row['test_frames'] == 2958
            with np.load(directory / 'predictions.npz', allow_pickle=False) as arrays:
                target = arrays['target_mm']
                identity = [target, arrays['groups'], arrays['frame_ids']]
                if target_reference is None:
                    target_reference = [x.copy() for x in identity]
                assert all(np.array_equal(a, b) for a, b in zip(identity, target_reference))
                errors = np.linalg.norm(arrays['prediction_mm'].astype(np.float64) - target.astype(np.float64), axis=-1)
                recomputed = dict(mean_node_mm=float(errors.mean()), endpoint_mm=float(errors[:, -1].mean()),
                    node_rmse_mm=float(np.sqrt(np.mean(errors ** 2, axis=1)).mean()),
                    node_global_rmse_mm=float(np.sqrt(np.mean(errors ** 2))),
                    endpoint_rmse_mm=float(np.sqrt(np.mean(errors[:, -1] ** 2))))
                assert all(recomputed[key] == row[key] for key in METRICS)
                assert len(groups) == 3 and sum(g['test_frames'] for g in groups) == row['test_frames']
                assert np.isclose(sum(g['mean_node_mm'] * g['test_frames'] for g in groups) / 2958, row['mean_node_mm'], rtol=0, atol=1e-12)
            rows.append(dict(row, source_run=str(source), origin_run=str(source), metrics_source=str(directory / 'metrics.json')))
            sequence_rows.extend(dict(g, source_run=str(source), origin_run=str(source)) for g in groups)
            state = read(source / f'formal/mlp_{variant}/seed_{seed}/run_manifest.json')
            validation.append(dict(model=row['model'], seed=seed, best_epoch=state['best_epoch'],
                best_val_mm=state['best_validation_node_mean_mm'], epoch0_val_mm=state['epoch0_validation_node_mean_mm'],
                wall_seconds=state['wall_seconds'], after80_best_gain_mm=state['after80_best_gain_mm'],
                after80_best_gain_pct=state['after80_best_gain_pct'], source_run=str(source), origin_run=str(source)))
            metric_checks.append(dict(model=row['model'], seed=seed, metrics_match_saved_predictions=True,
                common_targets_groups_frame_ids=True, metrics_source=str(directory / 'metrics.json'),
                metrics_sha256=digest(directory / 'metrics.json')))
    assert len(rows) == 100 and len({(r['model'], r['seed']) for r in rows}) == 100
    assert len(new_rows) == 40 and len(new_sequences) == 120
    for variant in VARIANTS:
        assert sorted(r['seed'] for r in rows if r['variant'] == variant) == SEEDS
    stats = paired_statistics(rows)
    summaries = []
    for variant in VARIANTS:
        group = [row for row in rows if row['variant'] == variant]
        summaries.append(dict(model='mlp_' + variant, family='mlp', variant=variant, n=20,
            parameters=COUNTS[variant], test_frames=2958, source_run=group[0]['source_run'], origin_run=group[0]['source_run'],
            lr=group[0]['lr'], **{metric: dict(mean=float(np.mean([r[metric] for r in group])),
                 sd=float(np.std([r[metric] for r in group], ddof=1))) for metric in METRICS}))
    analysis.mkdir(parents=True, exist_ok=True)
    csv_write(analysis / 'raw_test.csv', rows)
    csv_write(analysis / 'raw_test_by_sequence.csv', sequence_rows)
    csv_write(analysis / 'raw_validation.csv', validation)
    csv_write(analysis / 'model_summary.csv', [dict(model=s['model'], n=s['n'], parameters=s['parameters'],
        lr=s['lr'], source_run=s['source_run'], origin_run=s['source_run'], **{metric + '_' + stat: s[metric][stat] for metric in METRICS for stat in ['mean', 'sd']}) for s in summaries])
    csv_write(analysis / 'paired_statistics.csv', [{k: v for k, v in row.items() if k not in ['seeds', 'seed_differences_mm']} for row in stats])
    csv_write(analysis / 'paired_seed_differences.csv', [dict(family=r['family'], reference=r['reference'],
        alternative=r['alternative'], seed=seed, reference_minus_alternative_mm=delta)
        for r in stats for seed, delta in zip(SEEDS, r['seed_differences_mm'])])
    statistics = dict(primary='pooled test mean_node_mm', protocol=read(run / 'protocol.json')['statistics'], contrasts=stats)
    write(run / 'paired_statistics.json', statistics)
    write(analysis / 'paired_statistics.json', statistics)
    result = dict(schema='internal_mlp_single_memory_results_v1', at=stamp(), source_run=str(run),
                  reused_source_run=str(SOURCE), models=summaries, statistics=stats, protocol=read(run / 'protocol.json'))
    write(analysis / 'summary.json', result)
    write(analysis / 'validation.json', dict(status='pass', at=stamp(), compared_fits=100,
        seed_coverage=SEEDS, metric_checks=metric_checks,
        fit_normalization_audit=str(run / 'fit_normalization_audit.json'),
        assessment='Share with caveats: paired seeds measure training randomness on previously inspected fixed data; mechanism comparisons are exploratory and parameter counts differ.'))
    lines = ['# MLP内部记忆：补齐path/time后的20次配对比较', '',
        '原三记录6:2:2时间划分，H=20、dt=0.2 s；train/val/test目标数8988/2958/2958。',
        'CPU、每任务1线程、8任务并行；Adam，100 epochs，batch=256。损失、梯度裁剪、调度、val频率和checkpoint选择直接复用006冻结训练函数。',
        '每配置以seed 900001比较lr=0.001/0.003并冻结；正式seed为100..119。40次补训全部完成后统一test。', '',
        '|配置|参数|学习率|骨架均值±SD / mm|末端均值±SD / mm|来源|',
        '|---|---:|---:|---:|---:|---|']
    for s in summaries:
        lines.append(f'|{s["model"]}|{s["parameters"]}|{s["lr"]}|{s["mean_node_mm"]["mean"]:.6f} ± {s["mean_node_mm"]["sd"]:.6f}|'
                     f'{s["endpoint_mm"]["mean"]:.6f} ± {s["endpoint_mm"]["sd"]:.6f}|{"007补训" if s["variant"] in NEW else "006复用"}|')
    lines += ['', '统计主指标为2958个共同test目标的pooled mean_node_mm；每个seed为一个配对单位，20个seed全部纳入。',
        '双侧Wilcoxon采用符号分配精确分布，零差值剔除、同绝对差平均秩；20000次配对seed重采样给出均值差的percentile 95% CI（RNG seed=20260913）。',
        'Holm分两族：mlp_base_extensions包含base与path/time/both/static_capacity四项；mlp_both_single_mechanism包含both与path/time两项，独立校正。',
        '差值=reference误差−alternative误差，正数支持alternative；机制族用single−both，正数支持both。CI为逐比较区间，未作同时覆盖校正。', '',
        '|族|reference|alternative|差值 / mm|95% CI|精确p|族内Holm p|',
        '|---|---|---|---:|---|---:|---:|']
    for r in stats:
        lines.append(f'|{r["family"]}|{r["reference"]}|{r["alternative"]}|{r["mean_reference_minus_alternative_mm"]:.6f}|'
                     f'[{r["bootstrap95_lower_mm"]:.6f}, {r["bootstrap95_upper_mm"]:.6f}]|{r["wilcoxon_exact_p"]:.8g}|{r["wilcoxon_holm_p"]:.8g}|')
    lines += ['', '训练源码逐字节同006；plugin归一化从train窗口重算、输出归一化从train记录重算，均与006完全一致。100个checkpoint的归一化、val最佳epoch、同seed初始val误差和公共test目标均已核查。',
        '该结果为此前已查看的固定数据上的补充结构实验；seed区间反映训练随机性，不能替代新记录上的泛化评价。both与single的机制比较属于探索性结构比较，参数容量同时变化。', '',
        f'训练与阶段日志：`{run}`；已有配置来源：`{SOURCE}`。',
        '每行来源见raw_test.csv和raw_validation.csv；逐seed差值见paired_seed_differences.csv；验收见validation.json和训练目录fit_normalization_audit.json。', '',
        '复现/恢复（仓库根目录）：', '', '```bash',
        f'{sys.executable} scripts/experiments/complete_internal_mlp_memory_comparison.py --workers 8',
        '```', '',
        '可用--run和--analysis指定新的隔离输出目录；已完成目录恢复时校验冻结源码和输入指纹，并保留完整训练。']
    (analysis / 'README.md').write_text('\n'.join(lines) + '\n')
    write(analysis / 'COMPLETE.json', dict(status='complete', at=stamp(), models=5, seeds=20,
          new_formal_fits=40, reused_formal_fits=60, test_frames=2958))
    write(run / 'status.json', dict(status='complete', phase='complete', at=stamp(),
          tuning_fits=4, formal_fits=40, test_fits=40, compared_fits=100))
    stage(run, 'complete', new_formal_fits=40, reused_formal_fits=60,
          summary=str(analysis / 'summary.json'), readme=str(analysis / 'README.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--analysis', type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run, analysis = args.run.resolve(), args.analysis.resolve()
    assert run.parent == ROOT / 'workspace/runs/training' and run != SOURCE
    assert analysis.parent == ROOT / 'workspace/runs/analysis'
    assert 1 <= args.workers <= 8
    run.mkdir(parents=True, exist_ok=True)
    with (run / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            prepare(run, analysis, args.workers)
            if args.prepare_only:
                return
            tuning = [dict(id=f'tuning/mlp_{variant}/lr_{lr:g}', family='mlp', variant=variant,
                           seed=900001, lr=lr, phase='tuning') for variant in NEW for lr in [.001, .003]]
            if not (run / 'selected_configuration.json').exists():
                tuned = execute(run, tuning, 'tuning', args.workers)
                chosen = {}
                for variant in NEW:
                    best = min([r for r in tuned if r['variant'] == variant],
                               key=lambda r: (r['best_validation_node_mean_mm'], r['config']['lr']))
                    chosen['mlp_' + variant] = dict(family='mlp', variant=variant, lr=best['config']['lr'],
                                                    tuning_val_mm=best['best_validation_node_mean_mm'])
                write(run / 'selected_configuration.json', dict(frozen_at=stamp(), configurations=chosen, test_opened=False))
                csv_write(run / 'raw_tuning.csv', [dict(model=r['model'], seed=r['seed'], lr=r['config']['lr'],
                          best_epoch=r['best_epoch'], best_val_mm=r['best_validation_node_mean_mm']) for r in tuned])
                stage(run, 'learning_rates_frozen', configurations=chosen, test_opened=False)
            chosen = read(run / 'selected_configuration.json')['configurations']
            jobs = [dict(id=f'formal/mlp_{variant}/seed_{seed}', family='mlp', variant=variant,
                         seed=seed, lr=chosen['mlp_' + variant]['lr'], phase='formal') for seed in SEEDS for variant in NEW]
            if not (run / 'TRAIN_VAL_COMPLETE.json').exists():
                formal = execute(run, jobs, 'formal', args.workers)
                assert len(formal) == 40
                csv_write(run / 'raw_validation.csv', [dict(model=r['model'], seed=r['seed'], best_epoch=r['best_epoch'],
                    best_val_mm=r['best_validation_node_mean_mm'], epoch0_val_mm=r['epoch0_validation_node_mean_mm'],
                    wall_seconds=r['wall_seconds'], after80_best_gain_mm=r['after80_best_gain_mm'],
                    after80_best_gain_pct=r['after80_best_gain_pct']) for r in formal])
                fit_audit(run, jobs)
                write(run / 'TRAIN_VAL_COMPLETE.json', dict(at=stamp(), fits=40, test_opened=False,
                      selected_configuration_sha256=digest(run / 'selected_configuration.json')))
                stage(run, 'train_val_complete', formal_fits=40, fit_normalization_audit='pass', test_opened=False)
            else:
                fit_audit(run, jobs)
            if not (run / 'TEST_STARTED.json').exists():
                write(run / 'TEST_STARTED.json', dict(at=stamp(), formal_fits=40,
                      selected_configuration_sha256=digest(run / 'selected_configuration.json')))
            rows, sequences = evaluate_new(run, jobs)
            stage(run, 'test_complete', test_fits=len(rows), test_frames=2958)
            combine(run, analysis, rows, sequences)
        except BaseException:
            write(run / 'failure.json', dict(at=stamp(), error=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
