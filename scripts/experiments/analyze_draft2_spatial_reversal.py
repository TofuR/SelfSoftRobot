#!/usr/bin/env python3
"""Post-hoc spatial and reversal evaluation of frozen predictions; no fitting."""
from pathlib import Path
import csv
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/selfsr-draft2-spatial-mpl')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/training/modeling_unified20_20260913_004'
OUT = ROOT / 'workspace/runs/analysis/draft2_experiment_extensions_20260914_009/spatial'
MODELS = ['base', 'chen_direction', 'oscillator', 'koopman',
          'hov_no_memory', 'hov_no_play', 'hov_no_maxwell', 'hov']
SEEDS = list(range(100, 120))
LABEL = dict(base='MLP', chen_direction='Chen', oscillator='Krauss',
             koopman='Koopman', hov_no_memory='No memory',
             hov_no_play='Time only', hov_no_maxwell='Path only', hov='HOV')
COLORS = dict(base='#92989F', chen_direction='#D97732', oscillator='#59616A',
              koopman='#A98C35', hov_no_memory='#92989F', hov_no_play='#D97732',
              hov_no_maxwell='#59616A', hov='#2563A6')
STYLES = dict(base=':', chen_direction='--', oscillator='-.', koopman=':',
              hov_no_memory=':', hov_no_play='--', hov_no_maxwell='-.', hov='-')
LAGS = np.round(np.arange(-1., 1.01, .2), 8)


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n')


def stats(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), n=int(len(x)))


def paired(x, y):
    """Positive contrast means reference x has larger error than alternative y."""
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(20260914)
    boot = d[rng.integers(0, len(d), (20000, len(d)))].mean(axis=1)
    nz = d[d != 0]
    return dict(mean_difference_mm=float(d.mean()), sd_difference_mm=float(d.std(ddof=1)),
                ci95_low_mm=float(np.quantile(boot, .025)),
                ci95_high_mm=float(np.quantile(boot, .975)),
                positive=int((d > 0).sum()), negative=int((d < 0).sum()),
                zero=int((d == 0).sum()),
                exact_wilcoxon_p=float(wilcoxon(nz, method='exact').pvalue) if len(nz) else 1.)


def holm(rows):
    order = np.argsort([r['exact_wilcoxon_p'] for r in rows])
    prev = 0.
    for j, idx in enumerate(order):
        prev = max(prev, min(1., (len(rows)-j)*rows[idx]['exact_wilcoxon_p']))
        rows[idx]['holm_p'] = prev


def save(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=230, bbox_inches='tight', pad_inches=.1)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((RUN / 'protocol.json').read_text())
    mp = Path(protocol['dataset_manifest'])
    manifest = json.loads(mp.read_text())
    target = np.load(RUN / 'test_targets.npz')
    gt = target['target_mm'].astype(np.float64)
    assert gt.shape == (2958, 15, 3) and np.max(abs(gt[..., 2])) == 0
    event_rows, events, hold_rows, inventory = [], [], [], []
    offset = 0
    for group_idx, row in enumerate(r for r in manifest['files'] if r['role'] == 'test'):
        path = Path(row['path'])
        path = path if path.is_absolute() else mp.parent / path
        z = np.load(path)
        a, t, ids = z['actions'].astype(float)*150, z['timestamps'], z['frame_ids']
        n = len(a)-19
        assert np.array_equal(target['groups'][offset:offset+n], np.full(n, group_idx))
        assert np.array_equal(target['frame_ids'][offset:offset+n], ids[19:])
        assert np.array_equal(gt[offset:offset+n], z['positions'][19:])
        delta = np.diff(a, axis=0)
        sign = np.where(delta > .5, 1, np.where(delta < -.5, -1, 0))
        start_count = len(events)
        # e is the pressure extremum: two same-sign increments arrive at e,
        # followed by two opposite-sign increments. Each exceeds 0.5 kPa.
        for e in range(25, len(a)-6):
            channels = np.flatnonzero((sign[e-2] == sign[e-1]) &
                                     (sign[e-1]*sign[e] == -1) &
                                     (sign[e] == sign[e+1]))
            if not len(channels):
                continue
            native_idx = np.arange(e-6, e+7)
            native_time = t[native_idx]-t[e]
            if native_time[0] > -1 or native_time[-1] < 1:
                continue
            # One event per frame, even if multiple channels reverse together.
            # Pressure illustration averages only reversing channels, oriented
            # so loading before the extremum is positive.
            pressures = np.stack([np.interp(LAGS, native_time,
                (a[native_idx,c]-a[e,c])*sign[e-1,c]) for c in channels]).mean(axis=0)
            idx = offset+native_idx-19
            evt = dict(indices=idx, native_time=native_time, pressure=pressures)
            events.append(evt)
            event_rows.append(dict(event_id=len(events)-1, source_group=row['group'],
                frame_id=int(ids[e]), pooled_index=int(offset+e-19), timestamp_s=float(t[e]),
                channels=','.join(map(str, channels.tolist())),
                preceding_direction=','.join(map(str, sign[e-1,channels].tolist())),
                following_direction=','.join(map(str, sign[e,channels].tolist()))))
        quiet = np.max(abs(delta), axis=1) <= .1
        edge = np.diff(np.r_[False, quiet, False].astype(int))
        starts, stops = np.where(edge == 1)[0], np.where(edge == -1)[0]
        for s, stop in zip(starts, stops):
            if s >= 19 and stop-s >= 4 and t[stop]-t[s] >= .8:
                hold_rows.append(dict(group=row['group'], start=int(ids[s]), stop=int(ids[stop]),
                                      duration_s=float(t[stop]-t[s])))
        inventory.append(dict(group=row['group'], input_file=str(path), scored_frames=n,
                              reversal_events=len(events)-start_count))
        offset += n
    assert offset == len(gt)
    pd.DataFrame(event_rows).to_csv(OUT/'reversal_events.csv', index=False)
    spec = dict(question='Where does memory improve prediction, and does it remain useful around pressure reversals?',
        data_source=str(RUN), dataset_manifest=str(mp), seeds=SEEDS,
        population='All 2958 saved test predictions; fixed test split, no fitting or selection',
        spatial_regions={'proximal': 'nodes 1..7', 'distal': 'nodes 8..14', 'tip': 'node 14'},
        base_node='Fixed HOV base versus predicted baseline bases; exclude node 0 for spatial attribution',
        reversal='At least one channel: two preceding command increments share sign and two following increments share opposite sign; every increment >0.5 kPa in absolute value. One event per frame.',
        alignment='Actual timestamps; linear interpolation to -1..1 seconds at 0.2 second spacing. Requires bracketing saved predictions. Interpolation is analysis only.',
        weighting='Events pooled over dataset; same events for all models. Overlapping event windows are retained; they are not independent experimental replicates.',
        hold='All four command increments <=0.1 kPa for at least four steps and >=0.8 s; does not establish measured pressure constancy',
        hold_count=len(hold_rows), holds=hold_rows, inventory=inventory,
        uncertainty='Mean +/- SD over 20 training repetitions; paired CIs and tests conditional on this fixed data and event set',
        figures=[dict(name='node_profiles',family='line',palette='two roots blue and orange plus neutrals',
            question='How does prediction error vary along the arm?',footprint='two panels 10.6 x 3.8 inches',
            distinction='line style and markers, node 0 omitted; shared y axis'),
          dict(name='reversal_response',family='line',palette='two roots blue and orange plus neutrals',
            question='How do frozen variants predict around command reversals?',footprint='three panels 11.5 x 3.5 inches',
            distinction='line styles, exact timestamp alignment, SD bands')])
    write_json('protocol.json', spec)
    errors, nodes, profiles, regrows, event_curves, valid = {}, {}, [], [], {}, []
    first_step_changes = {}
    separated_changes = {}
    separated = []
    last_time = {}
    for i, row in enumerate(event_rows):
        if row['timestamp_s']-last_time.get(row['source_group'], -1e30) >= 2.:
            separated.append(i)
            last_time[row['source_group']] = row['timestamp_s']
    sensitivity = []
    raw = pd.read_csv(RUN/'raw_test.csv')
    for model in MODELS:
        arr = []
        for seed in SEEDS:
            z = np.load(RUN/f'evaluation/{model}/seed_{seed}/predictions.npz')
            assert np.array_equal(z['groups'], target['groups'])
            assert np.array_equal(z['frame_ids'], target['frame_ids'])
            e = np.linalg.norm(z['prediction_mm'].astype(float)-gt, axis=-1)
            reference = raw[(raw.model == model) & (raw.seed == seed)].iloc[0]
            diff = abs(float(e.mean())-float(reference.mean_node_mm))
            assert diff < 1e-6 and np.isfinite(e).all()
            valid.append(dict(model=model,seed=seed,absolute_recomputed_difference_mm=diff))
            arr.append(e)
        errors[model] = np.stack(arr)
        nodes[model] = errors[model].mean(axis=1)
        for node in range(15):
            profiles.append(dict(model=model,node=node,**stats(nodes[model][:,node])))
        for j, seed in enumerate(SEEDS):
            ee = errors[model][j]
            regrows.append(dict(model=model,seed=seed,all_nodes_mm=float(ee.mean()),
                nonbase_mm=float(ee[:,1:].mean()),proximal_mm=float(ee[:,1:8].mean()),
                distal_mm=float(ee[:,8:].mean()),tip_mm=float(ee[:,-1].mean())))
        metrics = np.stack([errors[model].mean(axis=-1), errors[model][:,:,-1]], axis=-1)
        extrema = np.array([e['indices'][6] for e in events])
        next_frames = np.array([e['indices'][7] for e in events])
        first_step_changes[model] = (metrics[:,next_frames,:]-metrics[:,extrema,:]).mean(axis=1)
        increment = metrics[:,next_frames,:]-metrics[:,extrema,:]
        separated_changes[model] = increment[:,separated,:].mean(axis=1)
        for mi, metric in enumerate(['shape','tip']):
            sensitivity.append(dict(model=model, metric=metric,
                event_selection='chronological centers at least 2 s apart',
                events=len(separated), **stats(increment[:,separated,mi].mean(axis=1))))
        curve = np.zeros((20,len(LAGS),2))
        for event in events:
            for si in range(20):
                for mi in range(2):
                    curve[si,:,mi] += np.interp(LAGS,event['native_time'],metrics[si,event['indices'],mi])
        event_curves[model] = curve/len(events)
        print(model, 'loaded; skeleton', errors[model].mean(), flush=True)
    pd.DataFrame(profiles).to_csv(OUT/'node_profiles.csv', index=False)
    regions = pd.DataFrame(regrows)
    regions.to_csv(OUT/'region_seed_metrics.csv', index=False)
    region_summary = {model:{key:stats(regions[regions.model==model][key])
        for key in ['all_nodes_mm','nonbase_mm','proximal_mm','distal_mm','tip_mm']} for model in MODELS}
    region_comparisons = []
    hov = regions[regions.model=='hov']
    for ref in ['base','chen_direction','hov_no_memory','hov_no_play','hov_no_maxwell']:
        rr = regions[regions.model==ref]
        for metric in ['all_nodes_mm','nonbase_mm','proximal_mm','distal_mm','tip_mm']:
            x,y=rr[metric].to_numpy(),hov[metric].to_numpy()
            region_comparisons.append(dict(reference=ref,alternative='hov',metric=metric,
                percent_reduction=float((x.mean()-y.mean())/x.mean()*100), **paired(x,y)))
    # All spatial contrasts are one exploratory family (25 comparisons).
    holm(region_comparisons)
    pd.DataFrame(region_comparisons).to_csv(OUT/'region_paired_statistics.csv', index=False)
    curve_rows, event_stats, event_comparisons = [], {}, []
    for model, c in event_curves.items():
        event_stats[model] = {}
        for mi, metric in enumerate(['shape','tip']):
            for li, lag in enumerate(LAGS):
                curve_rows.append(dict(model=model,metric=metric,time_s=float(lag),events=len(events),
                                       **stats(c[:,li,mi])))
            for name, keep in [('before', LAGS<0), ('after', LAGS>=0)]:
                event_stats[model][f'{name}_{metric}'] = stats(c[:,keep,mi].mean(axis=1))
    for ref in ['hov_no_memory','hov_no_play','hov_no_maxwell']:
        for mi, metric in enumerate(['shape','tip']):
            a=event_curves[ref][:,LAGS>=0,mi].mean(axis=1)
            b=event_curves['hov'][:,LAGS>=0,mi].mean(axis=1)
            event_comparisons.append(dict(reference=ref,alternative='hov',metric=metric,
                percent_reduction=float((a.mean()-b.mean())/a.mean()*100), **paired(a,b)))
    holm(event_comparisons)
    pd.DataFrame(curve_rows).to_csv(OUT/'reversal_curves.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'separated_event_sensitivity.csv',index=False)
    pd.DataFrame(event_comparisons).to_csv(OUT/'reversal_paired_statistics.csv',index=False)
    np.savez_compressed(OUT/'reversal_per_seed.npz',lags=LAGS,**event_curves)
    transient_comparisons = []
    for ref,alt in [('hov_no_play','hov'), ('hov_no_play','hov_no_maxwell')]:
        for mi,metric in enumerate(['shape','tip']):
            transient_comparisons.append(dict(reference=ref,alternative=alt,metric=metric,
                **paired(first_step_changes[ref][:,mi],first_step_changes[alt][:,mi])))
    holm(transient_comparisons)
    separated_comparisons = []
    for ref,alt in [('hov_no_play','hov'), ('hov_no_play','hov_no_maxwell')]:
        for mi,metric in enumerate(['shape','tip']):
            separated_comparisons.append(dict(reference=ref,alternative=alt,metric=metric,
                events=len(separated), **paired(separated_changes[ref][:,mi],separated_changes[alt][:,mi])))
    holm(separated_comparisons)
    pd.DataFrame(separated_comparisons).to_csv(OUT/'separated_event_paired_statistics.csv',index=False)
    pd.DataFrame(transient_comparisons).to_csv(OUT/'first_step_paired_statistics.csv',index=False)
    pd.DataFrame([dict(model=m,seed=seed,shape_increase_mm=float(v[si,0]),tip_increase_mm=float(v[si,1]))
        for m,v in first_step_changes.items() for si,seed in enumerate(SEEDS)]).to_csv(
            OUT/'first_step_seed_changes.csv',index=False)
    summary = dict(spatial=region_summary, spatial_comparisons=region_comparisons,
        reversal_events=len(events), event_stats=event_stats,event_comparisons=event_comparisons,
        first_step_changes={m:{metric:stats(v[:,mi]) for mi,metric in enumerate(['shape','tip'])}
                            for m,v in first_step_changes.items()},
        first_step_comparisons=transient_comparisons,
        first_step_time_s=dict(mean=float(np.mean([e['native_time'][7] for e in events])),
                              median=float(np.median([e['native_time'][7] for e in events]))),
        separated_event_sensitivity=sensitivity,
        separated_event_comparisons=separated_comparisons,
        holds=len(hold_rows),reversal_generalization='Post-hoc input-defined subgroup analysis; not a new test set or an isolated physical identification experiment',
        max_recompute_difference_mm=max(x['absolute_recomputed_difference_mm'] for x in valid))
    write_json('summary.json',summary)
    write_json('validation.json',dict(status='passed',checks=['2958 test frames exactly aligned',
        'all 20 seeds included for all eight models','Euclidean norms recomputed from saved coordinates',
        'global errors agree with source csv to <1e-6 mm', 'all coordinates finite',
        'reversal timestamps do not cross recording or test boundaries',
        'event selection uses pressure commands only', 'fixed base excluded from spatial interpretation'],
        max_recompute_difference_mm=summary['max_recompute_difference_mm'],
        limits=['Repeated-seed uncertainty is conditional on fixed data',
                'Event windows overlap and multi-channel reversals co-occur',
                'No qualifying all-channel holding interval in this test set']))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9BA3AB',
        'axes.labelcolor':'#26323E','text.color':'#26323E','svg.fonttype':'none',
        'pdf.fonttype':42,'legend.fontsize':9,'figure.facecolor':'white'})
    fig,axs=plt.subplots(1,2,figsize=(10.6,3.9),sharey=True,layout='constrained')
    for ax,names,title in zip(axs,[['base','chen_direction','oscillator','hov'],
        ['hov_no_memory','hov_no_play','hov_no_maxwell','hov']],
        ['(a) Comparison methods','(b) Retrained memory variants']):
        for model in names:
            a=nodes[model][:,1:]; x=np.arange(1,15); mu=a.mean(0);sd=a.std(0,ddof=1)
            ax.plot(x,mu,ls=STYLES[model],color=COLORS[model],lw=2 if model=='hov' else 1.5,
                    marker='o' if model=='hov' else None,ms=3,label=LABEL[model])
            ax.fill_between(x,mu-sd,mu+sd,color=COLORS[model],alpha=.10,linewidth=0)
        ax.axvline(7.5,color='#B9BEC4',ls=':',lw=1)
        ax.set(title=title,xlabel='Node index (base 0 omitted; tip 14)',xticks=[1,3,5,7,9,11,14],ylim=(0,4.8))
        ax.grid(axis='y',color='#E5E8EB',lw=.6);ax.legend(frameon=False,loc='upper left')
    axs[0].set_ylabel('Mean node distance (mm)')
    fig.suptitle('Spatial prediction error | 2,958 test frames; mean ± SD over 20 fits',fontsize=11)
    save(fig,'node_profiles')
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.8),layout='constrained')
    ps=np.stack([e['pressure'] for e in events]);mu=ps.mean(0)
    axs[0].plot(LAGS,mu,color='#59616A',lw=1.8)
    axs[0].fill_between(LAGS,np.quantile(ps,.25,axis=0),np.quantile(ps,.75,axis=0),color='#A9AFB5',alpha=.25)
    axs[0].set(title='(a) Reversing pressure commands',ylabel='Signed change from extremum (kPa)')
    for mi,ax in enumerate(axs[1:]):
        for model in ['hov_no_memory','hov_no_play','hov_no_maxwell','hov']:
            c=event_curves[model][:,:,mi];mu=c.mean(0);sd=c.std(0,ddof=1)
            ax.plot(LAGS,mu,ls=STYLES[model],color=COLORS[model],label=LABEL[model],lw=1.7)
            ax.fill_between(LAGS,mu-sd,mu+sd,color=COLORS[model],alpha=.12,linewidth=0)
        ax.set(title=['(b) Skeleton error','(c) Endpoint error'][mi],ylabel='Distance (mm)',
               ylim=[(1.4,2.1),(2.5,4.85)][mi])
    axs[2].legend(frameon=False,fontsize=8,loc='center left',bbox_to_anchor=(.02,.58))
    for ax in axs:
        ax.axvline(0,color='#26323E',ls=':',lw=1)
        ax.set(xlabel='Time from command extremum (s)',xticks=[-1,-.5,0,.5,1])
        ax.grid(axis='y',color='#E5E8EB',lw=.6)
    fig.suptitle(f'Prediction around pressure reversals | {len(events):,} overlapping events; 20 fits',fontsize=11)
    save(fig,'reversal_response')
    print(json.dumps(dict(output=str(OUT),events=len(events),holds=len(hold_rows),
        spatial_hov=region_summary['hov'],reversal=event_stats['hov']),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
