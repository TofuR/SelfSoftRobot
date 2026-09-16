#!/usr/bin/env python3
"""Frozen path-memory evidence for section 3.6. No training or outcome selection.

Run: python -B scripts/experiments/analyze_path_memory_section36_v3.py
Only writes this task's report directory and the three requested figure files.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'workspace/reports/section36_rewrite_20260915/path'
FIG = ROOT / 'docs/icra2027/figures/section36_v3'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GEOM = ROOT / 'workspace/runs/analysis/modeling_unified20_20260913_005/geometry'
REV = ROOT / 'workspace/runs/analysis/draft2_experiment_extensions_20260914_009/spatial'
SEEDS = list(range(100, 120))
COLORS = {'full': '#2864a5', 'time': '#cf7935', 'path': '#268b82',
          'joint': '#7958a6', 'reference': '#777777'}
KINDS = ['reference', 'path', 'time', 'joint', 'full']


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def stats(x):
    a = np.asarray(x, float)
    assert a.size and np.isfinite(a).all()
    return {'mean': float(a.mean()), 'sd': float(a.std(ddof=1)) if a.size > 1 else 0.,
            'n': int(a.size)}


def cos(a, b):
    return np.sum(a*b, axis=(-2, -1))/np.maximum(
        np.linalg.norm(a, axis=(-2, -1))*np.linalg.norm(b, axis=(-2, -1)), 1e-15)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    inputs = np.load(GEOM/'test_inputs_targets.npz')
    kernel = np.load(GEOM/'kernels.npz')
    pairs = pd.read_csv(GEOM/'matched_pairs.csv')
    events = pd.read_csv(REV/'reversal_events.csv')
    reversal = np.load(REV/'reversal_per_seed.npz')
    protocol_old = json.loads((REV/'protocol.json').read_text())
    assert protocol_old['seeds'] == SEEDS
    assert kernel['seeds'].tolist() == SEEDS
    assert np.array_equal(inputs['pair_id'], pairs.pair_id)
    assert np.array_equal(inputs['pair_pooled_i'], pairs.pooled_i)
    assert np.array_equal(inputs['pair_pooled_j'], pairs.pooled_j)
    windows = inputs['windows'].astype(float)
    groups, times = inputs['group_index'], inputs['timestamps']
    lags = reversal['lags']
    i, j = pairs.pooled_i.to_numpy(), pairs.pooled_j.to_numpy()
    observed = inputs['target_xyz_mm'][i, :, :2]-inputs['target_xyz_mm'][j, :, :2]
    # Cache convention is j-i; this analysis uses i-j consistently.
    cached = inputs['pair_observed_delta_xy_mm']
    assert min(abs(cached-observed).max(), abs(cached+observed).max()) < 1e-12
    pairs['observed_difference_mm'] = np.linalg.norm(observed, axis=-1).mean(-1)
    pairs['observed_tip_difference_mm'] = np.linalg.norm(observed[:, -1], axis=-1)
    assert np.all(groups[i] == groups[j])
    assert np.all(np.abs(i-j) >= 20)
    assert np.all(pairs.history_rms_kpa >= 20)
    current_gap = np.max(abs(windows[i, -1]-windows[j, -1]), axis=-1)*150
    assert np.max(current_gap-pairs.tolerance_kpa.to_numpy()) < 1e-4
    event_meta = []
    separated = []
    last_time = {}
    for e in events.itertuples():
        center = e.pooled_index
        idx = np.arange(center-6, center+7)
        assert np.all(groups[idx] == groups[center])
        assert inputs['frame_ids'][center] == e.frame_id
        t = times[idx]-times[center]
        assert t[0] <= -1 and t[-1] >= 1
        channels = np.array([int(x) for x in e.channels.split(',')])
        signs = np.array([int(x) for x in e.preceding_direction.split(',')])
        # Independently check the command-only reversal criterion.
        delta = np.diff(windows[center+2, -5:, :], axis=0)*150
        for c, s in zip(channels, signs):
            assert np.all(delta[:2, c]*s > .5)
            assert np.all(delta[2:, c]*s < -.5)
        event_meta.append((idx, t, channels, signs))
        if times[center]-last_time.get(int(groups[center]), -1e30) >= 2.:
            separated.append(e.event_id)
            last_time[int(groups[center])] = times[center]
    pairs.to_csv(OUT/'matched_pairs.csv', index=False)
    events.to_csv(OUT/'reversal_events.csv', index=False)
    write_json('protocol.json', {
        'seeds': SEEDS, 'test_windows': len(windows), 'window_steps': 20,
        'sources': {'geometry': str(GEOM.relative_to(ROOT)),
                    'reversal': str(REV.relative_to(ROOT)),
                    'original_middle_panel': 'scripts/experiments/analyze_draft2_spatial_reversal.py::main, reversal_response (b)'},
        'pair_rule': 'Existing pairs only: same source recording; >=20 frames separation; current active-channel maximum gap <= tolerance; previous 19-step active-channel history RMS difference >=20 kPa. Greedy disjoint matching within each recording/category/tolerance, sorted by current-input gap then chronology; no shapes or predictions used for selection.',
        'pair_categories': {'opposite': 'At least one active channel has opposite final increment signs (0.3 kPa deadband).',
                            'same_direction': 'All active-channel final increment signs agree.',
                            'same_recent_two': 'Both latest complete four-channel commands differ by at most tolerance.'},
        'display_tolerance_kpa': 5, 'sensitivity_tolerances_kpa': [1, 2, 5, 10],
        'display_rationale': 'Existing 5 kPa tolerance retains coverage in all three source recordings; report all existing tolerances, no selection by prediction outcome.',
        'reversal_rule': protocol_old['reversal'], 'reversal_alignment': protocol_old['alignment'],
        'state_alignment': 'Multiply q/r by preceding loading direction, average reversing channels within each event, then average events. States are the frozen 20-frame-window states, not a continuous unreset replay.',
        'primary_pair_metric': 'Mean across pairs and all 15 nodes of ||(prediction_i-prediction_j)-(observation_i-observation_j)||_2 in mm. Fixed base contributes zero; 14 nonbase-node metrics also exported.',
        'readout_diagnostics': 'Reference+path and reference+time are analytic local contributions from the frozen full model, not retrained variants. Joint is their sum. Full uses cached nonlinear predictions.',
        'statistical_unit': '20 trained parameter sets on the same fixed test data. SD quantifies training variation only. Reversal windows overlap; pair sets overlap across categories/tolerances; adjacent windows remain correlated even for disjoint frame pairs.',
        'inference': 'Descriptive estimates only; no p values, independence assumption, physical identification or causal branch-effect claim.',
        'plot_contract': {'panels': ['Path states around reversals', 'Original skeleton-error reversal panel', 'Matched-pair shape-difference error'],
                          'renderer': 'Matplotlib SVG/PDF/PNG', 'size_inches': [10.6, 3.75],
                          'uncertainty': 'Mean +/- sample SD over 20 fits, after event/pair aggregation',
                          'palette': COLORS, 'noncolor': 'line styles and point markers'},
    })
    pair_rows, pair_summary_rows = [], []
    state_rows, geometry_rows, direction_rows, node_rows = [], [], [], []
    state_curves, quality = [], []
    for si, seed in enumerate(SEEDS):
        with np.load(GEOM/f'seed_{seed}_test_geometry.npz') as z:
            assert int(z['seed']) == seed
            assert np.array_equal(z['frame_ids'], inputs['frame_ids'])
            assert np.array_equal(z['group_index'], groups)
            drive = z['drive']
            r = kernel['play_thresholds'][si]
            q = np.zeros((len(windows), 4, 2))
            for ti in range(1, 20):
                q = np.clip(q+np.diff(drive[:, ti-1:ti+1], axis=1)[:, 0, :, None], -r, r)
            qerr = float(abs(q-z['q']).max())
            mem = np.einsum('bcj,cjg->bg', q, kernel['W_path'][si])
            merr = float(abs(mem-z['memory_path']).max())
            path = z['path_displacement_xy_mm']
            temporal = z['time_displacement_xy_mm']
            jerr = float(abs(np.einsum('bndg,bg->bnd', z['J_reference_xy'], mem)-path).max())
            assert qerr < 2e-7 and merr < 1e-6 and jerr < 1e-4
            quality.append({'seed': seed, 'q_recurrence_max_error': qerr,
                            'path_readout_max_error': merr, 'J_path_max_error_mm': jerr})
            ref = z['reference_xy_mm']
            predictions = {'reference': ref, 'path': ref+path, 'time': ref+temporal,
                           'joint': z['joint_linear_xy_mm'], 'full': z['full_xyz_mm'][:, :, :2]}
            delta_path, delta_time = path[i]-path[j], temporal[i]-temporal[j]
            frame = pairs.copy()
            frame.insert(0, 'seed', seed)
            for kind, pred in predictions.items():
                diff = pred[i]-pred[j]
                err = np.linalg.norm(diff-observed, axis=-1)
                frame[f'{kind}_difference_error_mm'] = err.mean(-1)
                frame[f'{kind}_nonbase_difference_error_mm'] = err[:, 1:].mean(-1)
                frame[f'{kind}_tip_difference_error_mm'] = err[:, -1]
                frame[f'{kind}_predicted_difference_mm'] = np.linalg.norm(diff, axis=-1).mean(-1)
            frame['path_contribution_difference_mm'] = np.linalg.norm(delta_path, axis=-1).mean(-1)
            frame['time_contribution_difference_mm'] = np.linalg.norm(delta_time, axis=-1).mean(-1)
            frame['path_time_difference_cosine'] = cos(delta_path, delta_time)
            frame['path_observed_difference_cosine'] = cos(delta_path, observed)
            frame['time_observed_difference_cosine'] = cos(delta_time, observed)
            frame['q_difference_rms'] = np.sqrt(np.mean((q[i]-q[j])**2, axis=(1, 2)))
            frame['d_difference_rms'] = np.sqrt(np.mean((z['d'][i]-z['d'][j])**2, axis=(1, 2)))
            pair_rows.append(frame)
            for (tol, cat), f in frame.groupby(['tolerance_kpa', 'category']):
                row = {'seed': seed, 'tolerance_kpa': tol, 'category': cat, 'pairs': len(f)}
                row.update({k: float(f[k].mean()) for k in f.columns if k.endswith(('_mm', '_cosine', '_rms'))})
                pair_summary_rows.append(row)
            # Channel direction/state checks use all windows. No response-based filtering.
            last_increment = (windows[:, -1]-windows[:, -2])*150
            for c in range(4):
                directions = np.where(last_increment[:, c] >= .3, 1,
                                      np.where(last_increment[:, c] <= -.3, -1, 0))
                for label, dsign in [('loading', 1), ('unloading', -1), ('near_hold', 0)]:
                    keep = directions == dsign
                    if not keep.any():
                        continue
                    for threshold in range(2):
                        direction_rows.append({'seed': seed, 'channel': c, 'direction': label,
                            'threshold_index': threshold, 'threshold': float(r[threshold]),
                            'frames': int(keep.sum()), 'mean_pressure_kpa': float(windows[keep, -1, c].mean()*150),
                            'mean_q_over_r': float((q[keep, c, threshold]/r[threshold]).mean()),
                            'boundary_fraction': float((abs(q[keep, c, threshold]) >= r[threshold]-1e-6).mean())})
            ev_state, ev_geom = [], []
            for idx, t, channels, signs in event_meta:
                # Each event has equal weight, including multichannel reversals.
                oriented = (q[idx][:, channels, :]/r*signs[None, :, None]).mean(1)
                ev_state.append(np.stack([np.interp(lags, t, oriented[:, a]) for a in range(2)], -1))
                # Contributions of all channels: reversal does not isolate its channel.
                vals = np.column_stack([np.linalg.norm(path[idx], axis=-1).mean(-1),
                    np.linalg.norm(temporal[idx], axis=-1).mean(-1),
                    np.sqrt(np.mean(mem[idx, :14]**2, axis=-1))*180/np.pi,
                    np.sqrt(np.mean(mem[idx, 14:]**2, axis=-1))])
                ev_geom.append(np.stack([np.interp(lags, t, vals[:, a]) for a in range(4)], -1))
            ev_state, ev_geom = np.stack(ev_state), np.stack(ev_geom)
            state_curves.append(ev_state.mean(0))
            for scope, keep in [('all_events', np.arange(len(events))), ('centers_2s_apart', separated)]:
                for li, lag in enumerate(lags):
                    for ri in range(2):
                        state_rows.append({'seed': seed, 'scope': scope, 'events': len(keep),
                            'lag_s': lag, 'threshold_index': ri, 'threshold': float(r[ri]),
                            'oriented_q_over_r': float(ev_state[keep, li, ri].mean())})
                    geometry_rows.append({'seed': seed, 'scope': scope, 'events': len(keep), 'lag_s': lag,
                        **dict(zip(['path_mean_node_contribution_mm', 'time_mean_node_contribution_mm',
                                    'path_local_bend_rms_deg', 'path_log_length_rms'],
                                   map(float, ev_geom[keep, li].mean(0))))})
            # Whole-test readout profile, including every channel and all 20 seeds.
            for c in range(4):
                cmem = np.einsum('bj,jg->bg', q[:, c], kernel['W_path'][si, c])
                disp = np.einsum('bndg,bg->bnd', z['J_reference_xy'], cmem)
                for node in range(15):
                    node_rows.append({'seed': seed, 'channel': c, 'node': node, 'test_frames': len(windows),
                        'path_x_rms_mm': float(np.sqrt(np.mean(disp[:, node, 0]**2))),
                        'path_y_rms_mm': float(np.sqrt(np.mean(disp[:, node, 1]**2))),
                        'local_bend_rms_deg': float(np.sqrt(np.mean(cmem[:, node-1]**2))*180/np.pi) if node else 0.})
            print(f'seed {seed}: verified path recurrence/readout; paired and reversal diagnostics done', flush=True)
    all_pairs = pd.concat(pair_rows, ignore_index=True)
    seed_pairs = pd.DataFrame(pair_summary_rows)
    state = pd.DataFrame(state_rows)
    geometry = pd.DataFrame(geometry_rows)
    for name, df in [('pair_metrics_per_seed.csv', all_pairs), ('pair_summary_per_seed.csv', seed_pairs),
                     ('path_state_reversal_per_seed.csv', state), ('path_geometry_reversal_per_seed.csv', geometry),
                     ('path_direction_per_seed.csv', pd.DataFrame(direction_rows)),
                     ('path_node_contributions_per_seed.csv', pd.DataFrame(node_rows))]:
        df.to_csv(OUT/name, index=False)
    summaries = []
    for (tol, cat), f in seed_pairs.groupby(['tolerance_kpa', 'category']):
        row = {'tolerance_kpa': tol, 'category': cat, 'pairs': int(f.pairs.iloc[0])}
        for col in f.columns:
            if col.endswith(('_mm', '_cosine', '_rms')):
                if col.startswith('observed_'):
                    fixed_pairs = pairs[(pairs.tolerance_kpa == tol) & (pairs.category == cat)]
                    row[col] = {**stats(fixed_pairs[col]), 'unit': 'fixed test pairs; descriptive spread, not uncertainty'}
                else:
                    row[col] = stats(f[col])
        row['full_reduction_vs_reference_pct'] = stats(100*(f.reference_difference_error_mm-f.full_difference_error_mm)/f.reference_difference_error_mm)
        row['path_increment_over_time_mm'] = stats(f.time_difference_error_mm-f.joint_difference_error_mm)
        row['models_with_positive_path_increment'] = int((f.time_difference_error_mm > f.joint_difference_error_mm).sum())
        summaries.append(row)
    reverse_rows, reverse_summary = [], {}
    model_map = {'reference': 'hov_no_memory', 'time': 'hov_no_play', 'path': 'hov_no_maxwell', 'full': 'hov'}
    for name, model in model_map.items():
        a = reversal[model][:, :, 0]
        assert a.shape == (20, len(lags))
        reverse_summary[name] = {label: stats(a[:, mask].mean(1)) for label, mask in
                                [('before_mm', lags < 0), ('after_mm', lags >= 0)]}
        for si, seed in enumerate(SEEDS):
            for li, lag in enumerate(lags):
                reverse_rows.append({'model': model, 'label': name, 'seed': seed,
                                     'time_s': lag, 'events': len(events), 'skeleton_error_mm': a[si, li]})
    pd.DataFrame(reverse_rows).to_csv(OUT/'reversal_error_per_seed.csv', index=False)
    events.groupby('source_group').size().rename('events').to_csv(OUT/'event_coverage.csv')
    pairs.groupby(['tolerance_kpa', 'category', 'group']).size().rename('pairs').to_csv(OUT/'pair_coverage.csv')
    write_json('summary.json', {'pair_results': summaries, 'reversal': reverse_summary,
        'event_count': len(events), 'event_centers_2s_apart': len(separated), 'seeds': SEEDS,
        'path_thresholds': kernel['play_thresholds'].tolist(),
        'q_over_r_reversal': [{'threshold': float(kernel['play_thresholds'][0, ri]),
                              'lag_s': float(lags[li]), **stats(np.array(state_curves)[:, li, ri])}
                             for ri in range(2) for li in range(len(lags))],
        'statistical_note': 'No hypothesis tests. Training-repeat SD does not represent uncertainty across independent robots or physical experiments.'})
    write_json('validation.json', {'checks': quality, 'passed': True,
        'pair_current_gap_verified_all_four_channels': True, 'event_command_criterion_verified': True,
        'no_model_fitting': True, 'no_outcome_based_selection': True,
        'scope': 'All 20 cached HOV fits; all 1597 original category/tolerance pair records; all 1121 original reversal events; 5 kPa shown in figure.'})
    # Figure: compact three-panel comparison. No per-event inferential bands.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
                         'axes.labelsize': 8.5, 'legend.fontsize': 7.2,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#888888', 'text.color': '#252525',
                         'axes.labelcolor': '#252525'})
    fig, axs = plt.subplots(1, 3, figsize=(10.6, 3.75), gridspec_kw={'width_ratios': [1, 1, 1.12]})
    fig.subplots_adjust(left=.061, right=.992, bottom=.26, top=.82, wspace=.37)
    a = np.array(state_curves)
    for ri, color, style in [(0, COLORS['full'], '-'), (1, COLORS['time'], '--')]:
        mu, sd = a[:, :, ri].mean(0), a[:, :, ri].std(0, ddof=1)
        axs[0].plot(lags, mu, color=color, ls=style, lw=1.7,
                    label=f'$r={kernel["play_thresholds"][0, ri]:g}$')
        axs[0].fill_between(lags, mu-sd, mu+sd, color=color, alpha=.16, lw=0)
    axs[0].set(title='(a) Path memory at reversals', ylabel=r'Oriented path output $s q/r$', ylim=(-1.08, 1.08))
    axs[0].axhline(0, color='#b8b8b8', lw=.7)
    axs[0].legend(frameon=False, loc='lower left', ncol=2)
    labels = {'reference': 'No memory', 'time': 'Time only', 'path': 'Path only', 'full': 'Full model'}
    styles = {'reference': ':', 'time': '--', 'path': '-.', 'full': '-'}
    for name, model in model_map.items():
        a = reversal[model][:, :, 0]
        mu, sd = a.mean(0), a.std(0, ddof=1)
        axs[1].plot(lags, mu, color=COLORS[name], ls=styles[name], lw=1.7, label=labels[name])
        axs[1].fill_between(lags, mu-sd, mu+sd, color=COLORS[name], alpha=.12, lw=0)
    axs[1].set(title='(b) Prediction at reversals', ylabel='Skeleton error (mm)', ylim=(1.43, 2.20))
    axs[1].legend(frameon=False, loc='upper right', ncol=2, columnspacing=.8, handlelength=2)
    for ax in axs[:2]:
        ax.axvline(0, color='#777777', ls=':', lw=.9)
        ax.set(xlabel='Time from command extremum (s)', xticks=[-1, -.5, 0, .5, 1])
    categories = ['opposite', 'same_direction', 'same_recent_two']
    labels_pair = {'reference': 'Reference', 'path': 'Reference + path',
                   'time': 'Reference + time', 'full': 'Full model'}
    # Keep the two branch readouts separate from the retrained variants in (b).
    for ki, name in enumerate(['reference', 'time', 'path', 'full']):
        vals = [seed_pairs[(seed_pairs.tolerance_kpa == 5) & (seed_pairs.category == c)]
                [f'{name}_difference_error_mm'].to_numpy() for c in categories]
        x = np.arange(3)+(ki-1.5)*.11
        axs[2].errorbar(x, [v.mean() for v in vals], yerr=[v.std(ddof=1) for v in vals],
                        color=COLORS[name], marker=['s', '^', 'D', 'o'][ki],
                        ls='none', ms=4, capsize=2, lw=1.1, label=labels_pair[name])
    counts = [len(pairs[(pairs.tolerance_kpa == 5) & (pairs.category == c)]) for c in categories]
    axs[2].set(title='(c) Matched-history differences', ylabel='Shape-difference error (mm)',
               xticks=range(3), xticklabels=[f'Opposite\n(n={counts[0]})', f'Same direction\n(n={counts[1]})',
                                            f'Same last two\n(n={counts[2]})'], xlim=(-.4, 2.4))
    axs[2].legend(frameon=False, loc='upper right', ncol=1, handletextpad=.35)
    for ax in axs:
        ax.grid(axis='y', color='#e6e6e6', lw=.6)
        ax.tick_params(labelsize=7.8, width=.6)
    fig.text(.061, .96, 'Path memory, pressure reversals and matched histories', fontsize=10, weight='bold')
    fig.text(.061, .895, '20 fits on one fixed test set; bands / bars: SD across fits', fontsize=8.2)
    fig.text(.061, .06, '(a,b) 1,121 overlapping reversal events.  (c) Current-command gap ≤5 kPa; history RMS gap ≥20 kPa.\n'
             '(b) Retrained variants.  (c) Frozen-model local readouts; full model uses nonlinear geometry.', fontsize=7.7)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(FIG/f'path_memory.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    selected = {r['category']: r for r in summaries if r['tolerance_kpa'] == 5}
    op, recent = selected['opposite'], selected['same_recent_two']
    def val(row, key):
        return f"{row[key]['mean']:.3f} ± {row[key]['sd']:.3f}"
    after = reverse_summary
    def state_mean(ri, lag):
        return state[(state.scope == 'all_events') & (state.threshold_index == ri) & (abs(state.lag_s-lag) < 1e-8)].oriented_q_over_r.mean()
    reversal_geometry = geometry[(geometry.scope == 'all_events') & (abs(geometry.lag_s) < 1e-8)].mean(numeric_only=True)
    sensitivity_text = '\n'.join(f"- {a['tolerance_kpa']:g} kPa，{a['pairs']}对：reference {a['reference_difference_error_mm']['mean']:.3f} → full {a['full_difference_error_mm']['mean']:.3f} mm。" for a in summaries if a['category'] == 'opposite')
    findings = f'''# 3.6 路径记忆分析

仅使用已冻结的20个模型及其缓存，未训练或选择模型。图为 `docs/icra2027/figures/section36_v3/path_memory.svg`（另有PDF和300 dpi PNG）。

## 可以写入论文的结果

1. 在固定测试集的1,121个输入定义的反转事件上，反转后0–1 s（含0时刻，6个采样位置）的骨架误差为：无记忆 {after['reference']['after_mm']['mean']:.3f} mm，时间记忆 {after['time']['after_mm']['mean']:.3f} mm，路径记忆 {after['path']['after_mm']['mean']:.3f} mm，完整模型 {after['full']['after_mm']['mean']:.3f} mm。此处是重新训练的各消融模型，与以下固定完整模型的读出分解不同。
2. 在当前压力差不超过5 kPa、历史输入RMS差至少20 kPa的216对反向运动样本中，真实形状差异均值为 {op['observed_difference_mm']['mean']:.3f} mm。参考形态预测的差异仅为 {op['reference_predicted_difference_mm']['mean']:.3f} mm，完整模型为 {op['full_predicted_difference_mm']['mean']:.3f} mm，仍低估真实差异。对应的形状差异预测误差由 {val(op, 'reference_difference_error_mm')} mm降至 {val(op, 'full_difference_error_mm')} mm；这里比较的是向量差异的预测误差，不是两个差异幅值之差。
3. 同一216对样本中，参考形态加时间记忆的差异预测误差为 {val(op, 'time_difference_error_mm')} mm，加入路径记忆后的一阶联合读出为 {val(op, 'joint_difference_error_mm')} mm。20次拟合中有 {op['models_with_positive_path_increment']}/20 次的平均误差下降，平均下降 {op['path_increment_over_time_mm']['mean']:.3f} mm。这说明固定模型中路径读出提供了时间读出之外的有用形状信息，不能据此将两者视为已独立辨识的物理机制。
4. 在最近两次输入均相近、较早历史不同的38对样本中，参考形态、参考加路径、参考加时间、完整模型的差异预测误差依次为 {recent['reference_difference_error_mm']['mean']:.3f}、{recent['path_difference_error_mm']['mean']:.3f}、{recent['time_difference_error_mm']['mean']:.3f}、{recent['full_difference_error_mm']['mean']:.3f} mm。该子集的完整模型未优于参考形态，不能据此主张较早历史提高了差异预测精度；同方向58对中完整模型也未优于参考形态（1.208 vs 1.185 mm）。这限制了结论范围：现有配对证据支持反向加载历史的区分，而不支持所有类型历史差异都得到改善。两个子集同时改变路径状态和时间状态，也不能称为路径记忆的隔离实验。

## 路径状态与几何贡献如何解释

对于每个通道和阈值，路径输出满足 `q_t = clip(q_(t-1) + e_t - e_(t-1), -r, r)`。在固定前态下，加载会使q增大，卸载会使q减小；到达边界后饱和，反转后进入内部区间。给定初始值后，这个关系由结构保证，不是从图中发现的新物理规律。图(a)用反转前的方向s统一加载/卸载朝向，绘制`s*q/r`；曲线先按同一事件的反转通道平均，再按全部事件平均。两个阈值为0.02和0.5，单位是模型变换输入，不能直接写成kPa。小阈值单元的平均归一化输出从反转处{state_mean(0, 0):.3f}变为0.2 s后的{state_mean(0, .2):.3f}；大阈值单元同期从{state_mean(1, 0):.3f}变为{state_mean(1, .2):.3f}。这反映本批输入中小阈值单元较快改变符号、大阈值单元仍保留先前方向的贡献；不能把0.2 s解释成路径算子的固有时间常数。

`memory_path = W_p q`已由缓存独立重算并核验。其前14个分量为局部转角修正（rad），最后2个为分段对数长度修正（无量纲）；`J_ref W_p q`给出各节点二维位移的一阶贡献（mm）。`path_geometry_reversal_per_seed.csv`记录反转附近局部弯曲RMS、长度修正RMS和节点位移贡献；`path_node_contributions_per_seed.csv`给出四通道沿臂分布，能够将“保留路径差异”与“产生几何差异”联系起来。反转处路径几何读出的局部转角RMS为{reversal_geometry['path_local_bend_rms_deg']:.3f}度，对数长度修正RMS为{reversal_geometry['path_log_length_rms']:.5f}，映射后的平均节点位移贡献为{reversal_geometry['path_mean_node_contribution_mm']:.3f} mm（先按每帧节点统计，再平均事件和模型）。位移贡献的大小本身不是精度提升，应结合配对向量误差下降评价。

图(a)使用缓存中每个20帧窗口初始化后得到的终态，不能解释成无重置的连续状态回放。固定输入下路径状态保持、时间状态衰减，是更新式的性质；目前测试集中没有满足原规则的全通道保持区间，不能把这一性质报告为保持实验的实测结论。

## 图注建议

路径记忆的表示与预测作用。(a) 以反转前方向统一符号的归一化路径输出，分别对应两个阈值；曲线按反转通道及事件依次平均。(b) 不同记忆配置重新训练后，在压力指令反转附近的骨架预测误差，取自原反转分析的中间面板。(c) 当前压力相近、历史不同的样本对中，真实形状差异与模型预测差异之间的平均节点距离；参考加路径、参考加时间使用固定完整模型的一阶几何贡献。曲线或点为20次训练的均值，阴影或误差棒为训练间标准差。(a,b)使用同一测试集全部1,121个反转事件，事件窗口重叠；(c)容差为5 kPa，历史RMS差至少20 kPa，三类配对数量依次为216、58、38。

## 反向配对的容差敏感性

{ sensitivity_text }

上述容差来自原分析，所有结果均保留。容差越严格样本越少，不能把四行当成独立重复实验；1 kPa的11对只来自两个记录，2 kPa起覆盖三个记录。

## 统计范围与限制

- 20个seed是同一数据集上的训练重复，不是20次独立物理实验。未计算或复用p值。
- 1,121个事件按帧计数（多通道同时反转只计一次），相邻事件及窗口相关；其中{len(separated)}个事件中心按时间顺序至少间隔2 s，状态及几何贡献的对应敏感性统计也已导出。
- 共1,597条原配对记录涵盖1/2/5/10 kPa容差。不同容差/类别可能重复使用样本，不能全部合并作为独立样本。单一容差和类别内部采用互不重用帧的贪心匹配，但窗口及序列仍相关。
- 主图5 kPa保证三个记录均有反向配对，1/2/10 kPa结果完整保留于summary.json及CSV，未依据模型优劣调筛选规则。
- 实际指令并非严格相同，参考形态本身有差异；比较使用完整预测差减真实差，而非将所有形状差异归于迟滞。路径与时间贡献相互混杂，真实形状差异也包含未建模因素及标注误差。
- 面板(a)反映既定路径算子的行为，不能作为模型学得材料规律的证据。面板(b,c)提供其在已采集数据上的预测证据，仍不能证明物理机制唯一性。

## 文件

- `protocol.json`：来源、规则、统计单位和作图设计。
- `summary.json`：全部容差/类别的20模型统计及反转结果。
- `pair_metrics_per_seed.csv`：逐seed逐配对原始指标，保留原pair_id以便追溯。
- `pair_summary_per_seed.csv`：逐seed配对汇总。
- `path_state_reversal_per_seed.csv`、`path_direction_per_seed.csv`：路径状态与反转/加载方向。
- `path_geometry_reversal_per_seed.csv`、`path_node_contributions_per_seed.csv`：W_p q及几何映射的贡献。
- `reversal_error_per_seed.csv`：原中间面板对应的全部20模型曲线。
- `validation.json`：索引对齐、反转条件、路径递推及几何读出的数值核验。
'''
    (OUT/'findings.md').write_text(findings, encoding='utf-8')
    print(json.dumps({'report': str(OUT), 'figure': str(FIG/'path_memory.png'),
                      'primary_opposite_pairs': op}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
