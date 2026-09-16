#!/usr/bin/env python3
"""Rebuild paper figures and independently check completed MLP/control records."""
from pathlib import Path
import argparse
import json
import os
import sys
sys.dont_write_bytecode = True
os.environ.setdefault('MPLCONFIGDIR', '/tmp/selfsr-completion007-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
MLP = ROOT/'workspace/runs/analysis/modeling_internal_mlp_single_memory_20260913_007'
REAL = ROOT/'workspace/runs/validation/real_robot_20260912_001'
OUT = ROOT/'workspace/runs/analysis/real_control_paper_20260913_007'
FIG = ROOT/'docs/icra2027/figures/completion007'
ORDER = ['base', 'path', 'time', 'both', 'static_capacity']
LABEL = dict(base='Base MLP', path='+ Path', time='+ Time', both='+ Both', static_capacity='Polynomial control')
COLOR = dict(base='#7D8793', path='#8B6BB1', time='#D88B39', both='#2F70AD', static_capacity='#A9AFB8')
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.spines.left':False, 'axes.edgecolor':'#C2C9D0',
    'svg.fonttype':'none', 'pdf.fonttype':42, 'figure.facecolor':'white',
    'savefig.facecolor':'white', 'axes.titlepad':16})

def read(path):
    return json.loads(path.read_text())

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def save(fig, name, directory=FIG):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ['png','svg','pdf']:
        fig.savefig(directory/f'{name}.{ext}', dpi=230)
    plt.close(fig)

def mlp_figure():
    data = read(MLP/'summary.json')
    raw = pd.read_csv(MLP/'raw_test.csv')
    assert len(raw)==100 and not raw.duplicated(['model','seed']).any()
    arrays = {}
    for variant in ORDER:
        rows = raw[raw.model=='mlp_'+variant].sort_values('seed')
        assert rows.seed.tolist()==list(range(100,120)) and (rows.test_frames==2958).all()
        arrays[variant] = rows
        record = next(s for s in data['models'] if s['variant']==variant)
        for metric in ['mean_node_mm','endpoint_mm']:
            np.testing.assert_allclose([rows[metric].mean(),rows[metric].std(ddof=1)],
                [record[metric]['mean'],record[metric]['sd']],rtol=0,atol=1e-12)
    contrasts=data['statistics']
    checked=[]
    for c in contrasts:
        delta = (arrays[c['reference'].removeprefix('mlp_')].mean_node_mm.to_numpy()
                 - arrays[c['alternative'].removeprefix('mlp_')].mean_node_mm.to_numpy())
        assert np.all(delta!=0) and len(np.unique(np.abs(delta)))==20
        p=float(stats.wilcoxon(delta, alternative='two-sided',method='exact').pvalue)
        np.testing.assert_allclose(p,c['wilcoxon_exact_p'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(delta,c['seed_differences_mm'],rtol=0,atol=1e-14)
        checked.append({'reference':c['reference'],'alternative':c['alternative'],'scipy_exact_p':p})
    for family in {c['family'] for c in contrasts}:
        group=[c for c in contrasts if c['family']==family]
        rank=np.argsort([c['wilcoxon_exact_p'] for c in group], kind='stable')
        adjusted=np.minimum(1,np.maximum.accumulate([(len(rank)-j)*group[i]['wilcoxon_exact_p'] for j,i in enumerate(rank)]))
        np.testing.assert_allclose(adjusted,[group[i]['wilcoxon_holm_p'] for i in rank],atol=1e-14,rtol=0)
    fig,axes=plt.subplots(1,3,figsize=(14.4,5.8),gridspec_kw={'width_ratios':[1.05,1.05,1.3]})
    fig.subplots_adjust(left=.105,right=.975,top=.78,bottom=.22,wspace=.74)
    fig.suptitle('Memory fusion in an MLP hidden layer',x=.035,y=.975,ha='left',fontsize=17,weight='bold')
    fig.text(.035,.905,'Same test set and 20 paired training repetitions per configuration',fontsize=11)
    for ax,metric,heading in zip(axes[:2],['mean_node_mm','endpoint_mm'],['(a) Skeleton error','(b) Endpoint error']):
        for i,v in enumerate(ORDER):
            vals=arrays[v][metric].to_numpy()
            offsets=np.random.default_rng(72).uniform(-.16,.16,20)
            ax.scatter(vals,i+offsets,s=15,color=COLOR[v],alpha=.45,edgecolor='none')
            ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='D',ms=6,color=COLOR[v],capsize=4,lw=2)
        ax.set(yticks=range(5),yticklabels=[LABEL[v] for v in ORDER],ylim=(4.7,-.6),xlabel='Error (mm)')
        ax.set_title(heading,loc='left',fontweight='bold')
        ax.grid(axis='x',color='#E4E8ED',lw=.7);ax.set_axisbelow(True)
    ax=axes[2]
    for i,c in enumerate(contrasts):
        mean=c['mean_reference_minus_alternative_mm'];lo=c['bootstrap95_lower_mm'];hi=c['bootstrap95_upper_mm']
        color=COLOR[c['alternative'].removeprefix('mlp_')]
        ax.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=color,capsize=4,ms=6,lw=2)
    labels=['Base → Path','Base → Time','Base → Both','Base → Polynomial','Path → Both','Time → Both']
    ax.set(yticks=range(6),yticklabels=labels,ylim=(5.6,-.65),xlabel='Skeleton error reduction (mm)',xlim=(-.065,.58))
    ax.axvline(0,color='#89939E',lw=1,ls='--');ax.axhline(3.5,color='#D4D9DF',lw=.8)
    ax.grid(axis='x',color='#E4E8ED',lw=.7);ax.set_axisbelow(True)
    ax.set_title('(c) Paired improvement',loc='left',fontweight='bold')
    fig.text(.035,.085,'(a–b) Dots: individual fits; diamonds: mean ± SD.   (c) Mean paired differences and 95% bootstrap intervals.',fontsize=10)
    fig.text(.035,.038,'Both vs. Path: 0.041 mm, Holm p = 0.000168.   Both vs. Time: 0.141 mm, Holm p = 0.00000381.',fontsize=10)
    save(fig,'internal_memory_mlp')
    write(OUT/'mlp_independent_validation.json',dict(status='pass',fits=100,test_frames=2958,exact_statistics=checked,
        checks=['all 20 repetitions per model','CSV means and SD vs summary','scipy exact Wilcoxon','Holm families'],
        source=str((MLP/'raw_test.csv').relative_to(ROOT))))

def real_inventory():
    index=pd.read_csv(REAL/'trial_index.csv')
    complete=index[index.status=='completed'].copy()
    assert len(index)==22 and len(complete)==15 and complete.archived.all()
    OUT.mkdir(parents=True,exist_ok=True)
    complete.to_csv(OUT/'completed_trials.csv',index=False)
    counts=index.status.value_counts().to_dict()
    fields=['planning_ms','command_interval_median_ms','command_interval_p95_ms','job_median_ms','job_p95_ms']
    ranges={f:dict(n=int(complete[f].notna().sum()),minimum=float(complete[f].min()),maximum=float(complete[f].max())) for f in fields}
    group=[]
    for (target,occ,corr),rows in complete.groupby(['target_group','occlusion','correction']):
        group.append(dict(target=target,occlusion=occ,correction=corr,n=len(rows),trials=rows.trial.astype(int).tolist()))
    info=dict(attempts=len(index),completed=len(complete),statuses=counts,
        clear=int((complete.occlusion=='clear').sum()),occluded=int((complete.occlusion=='occluded').sum()),
        closed=int((complete.correction=='closed').sum()),open=int((complete.correction=='open').sum()),
        full_shape=int((complete.target_kind=='full').sum()),tip=int((complete.target_kind=='tip').sum()),
        timing_ranges=ranges,groups=group,
        endpoint_source='visual/endpoint_measurements.csv',
        interpretation='Completed executions do not imply a predefined arrival success threshold.')
    write(OUT/'execution_summary.json',info)
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.7),gridspec_kw={'width_ratios':[1,1.4]})
    fig.subplots_adjust(left=.18,right=.98,top=.79,bottom=.2,wspace=.36)
    fig.suptitle('Physical execution and feedback timing',x=.055,y=.97,ha='left',fontsize=16,weight='bold')
    fig.text(.055,.90,'15 completed physical trials; command and feedback timing are measured separately',fontsize=10.5)
    ax=axes[0]
    names=['Clear / open loop','Clear / feedback','Occluded / open loop','Occluded / feedback']
    amounts=[int(((complete.occlusion==o)&(complete.correction==c)).sum()) for o,c in [('clear','open'),('clear','closed'),('occluded','open'),('occluded','closed')]]
    ax.barh(range(4),amounts,color=['#A1A9B2','#2F70AD','#A1A9B2','#2F70AD'],height=.55)
    for i,v in enumerate(amounts):ax.text(v+.1,i,str(v),va='center')
    ax.set(yticks=range(4),yticklabels=names,xlabel='Completed trials',xlim=(0,6),ylim=(3.6,-.6))
    ax.set_title('(a) Execution coverage',loc='left',weight='bold');ax.grid(axis='x',color='#E4E8ED');ax.set_axisbelow(True)
    ax=axes[1]
    for i,row in enumerate(complete.itertuples()):
        ax.plot([i,i],[row.command_interval_median_ms,row.command_interval_p95_ms],color='#4E5965',lw=1)
        ax.scatter(i,row.command_interval_median_ms,color='#4E5965',marker='o',s=22,label='Command interval: p50–p95' if i==0 else None)
        if row.correction=='closed':
            ax.plot([i+.2,i+.2],[row.job_median_ms,row.job_p95_ms],color='#2F70AD',lw=1.5)
            ax.scatter(i+.2,row.job_median_ms,color='#2F70AD',marker='s',s=22,label='Feedback job: p50–p95' if i==0 else None)
    ax.axhline(200,ls='--',color='#AAB1B8',lw=1)
    ax.set(xticks=range(15),xticklabels=[f'T{i:02d}' for i in complete.trial],ylabel='Time (ms)',ylim=(0,360))
    ax.tick_params(axis='x',rotation=65,labelsize=8)
    ax.set_title('(b) Observed timing per trial',loc='left',weight='bold')
    ax.legend(loc='upper left',frameon=False,fontsize=8)
    fig.text(.055,.065,'Across all 22 attempts: 15 complete, 6 protective stops, 1 with unconfirmed terminal feedback.',fontsize=10)
    save(fig,'real_control_timing')

def endpoint_figure(unit='px'):
    from PIL import Image
    source=OUT/'visual/endpoint_measurements.csv' if unit=='px' else ROOT/'workspace/runs/analysis/real_control_registered_mm_20260914_008/endpoint_measurements.csv'
    rows=pd.read_csv(source)
    metric='tip_error_px' if unit=='px' else 'tip_error_registered_mm'
    lower='tip_error_sensitivity_min_px' if unit=='px' else 'sensitivity_min_registered_mm'
    upper='tip_error_sensitivity_max_px' if unit=='px' else 'sensitivity_max_registered_mm'
    assert len(rows)==15 and not rows.status.str.contains('provisional|pending').any()
    np.testing.assert_allclose(np.hypot(rows.measured_tip_x_px-rows.target_tip_x_px,
        rows.measured_tip_y_px-rows.target_tip_y_px),rows.tip_error_px,atol=1e-12,rtol=0)
    fig,axes=plt.subplots(1,3,figsize=(14.6,6.7),gridspec_kw={'width_ratios':[.9,.9,1.8]})
    fig.subplots_adjust(left=.035,right=.98,top=.80,bottom=.21,wspace=.4)
    for ax,position in zip(axes,[[.035,.22,.205,.57],[.27,.22,.205,.57],[.67,.22,.31,.57]]):
        ax.set_position(position)
    fig.suptitle('Physical occlusion: final images and measured endpoint error',x=.035,y=.96,
                 ha='left',fontsize=16,weight='bold')
    fig.text(.035,.892,'Measurements use visible tip cap centers; each point represents one completed physical execution.',fontsize=10.5)
    for ax,trial,heading in zip(axes[:2],[12,21],['(a) Right-shape target','(b) Left-tip target']):
        row=rows[rows.trial==trial].iloc[0]
        raw=ROOT/row.raw_frame
        ax.imshow(Image.open(raw))
        plan=raw.parents[2]/'initial_plan.npz'
        with np.load(plan,allow_pickle=False) as data:
            target=data['goal_mm'] @ data['camera_matrix'][:2,:2].T + data['camera_matrix'][:2,2]
        if row.target_kind=='full':ax.plot(target[:,0],target[:,1],ls='--',lw=1.2,color='#37D5C5',label='Target shape')
        ax.scatter(row.target_tip_x_px,row.target_tip_y_px,c='#ED8AB7',marker='x',s=80,lw=2)
        ax.scatter(row.measured_tip_x_px,row.measured_tip_y_px,facecolors='none',edgecolors='#FFD45A',s=85,lw=1.7)
        ax.set(xlim=(225,440),ylim=(405,90))
        ax.set_title(heading+f'\nFeedback · {row[metric]:.1f} {unit}',loc='left',fontsize=10.5)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    ax=axes[2]
    conditions=[('G01','clear'),('G02','clear'),('G02','occluded'),('G03','clear'),('G03','occluded'),('G06','clear'),('G06','occluded')]
    labels=['Left shape A · clear','Left shape B · clear','Left shape B · occluded',
            'Right shape · clear','Right shape · occluded','Left tip · clear','Left tip · occluded']
    for i,(target,occ) in enumerate(conditions):
        subset=rows[(rows.target_group==target)&(rows.occlusion==occ)]
        for corr,offset,color,marker in [('open',-.12,'#818B98','o'),('closed',.12,'#2F70AD','D')]:
            selected=subset[subset.correction==corr]
            for j,row in enumerate(selected.itertuples()):
                y=i+offset+(j-(len(selected)-1)/2)*.10
                ax.scatter(getattr(row,metric),y,c=color,marker=marker,s=38,
                    label=('Open loop' if corr=='open' else 'Feedback') if i==0 and j==0 else None,zorder=3)
                ax.plot([getattr(row,lower),getattr(row,upper)],[y,y],color=color,lw=1.2)
    ax.set(yticks=range(7),yticklabels=labels,xlim=(0,max(25 if unit=='px' else 14,rows[metric].max()+1)),ylim=(6.7,-.8),xlabel=f'Final endpoint error ({unit})')
    ax.set_title('(c) Final-frame endpoint errors',loc='left',fontsize=10.5)
    ax.grid(axis='x',color='#E4E8ED',lw=.8);ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper right',fontsize=9)
    fig.text(.035,.115,'Images: dashed cyan = target shape; pink × = target endpoint; yellow circle = independently measured visible endpoint.',fontsize=10)
    note='Whiskers: extraction-setting ranges, not confidence intervals.'
    note+=(' Final command-associated frames; T21 is still moving.' if unit=='px' else ' Millimeter scale: nominal diameter and per-trial registration.')
    fig.text(.035,.07,note,fontsize=10)
    save(fig,'real_control_endpoints',FIG if unit=='px' else ROOT/'docs/icra2027/figures/control008')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['mlp','inventory','endpoints','all'],default='all')
    parser.add_argument('--endpoint-unit',choices=['px','mm'],default='px')
    args=parser.parse_args()
    if args.only in ['mlp','all']:mlp_figure()
    if args.only in ['inventory','all']:real_inventory()
    if args.only in ['endpoints','all']:endpoint_figure(args.endpoint_unit)
