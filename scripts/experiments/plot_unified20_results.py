#!/usr/bin/env python3
"""Summarize completed unified repetitions and export publication figures."""
from pathlib import Path
import argparse
import json
import os
import shutil
os.environ.setdefault('MPLCONFIGDIR', '/tmp/selfsr-unified20-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'workspace/runs/training/modeling_unified20_20260913_004'
OUT = ROOT/'workspace/runs/analysis/modeling_unified20_20260913_005'
FIG = ROOT/'docs/icra2027/figures/unified20'
REPORT = ROOT/'workspace/reports/modeling_unified20_20260913_005'
MAIN = ['linear','pcc','base','koopman','oscillator','chen_direction','hov','window']
VARIANTS = ['base','path','time','both','static_capacity','window']
EN = dict(linear='Linear',pcc='PCC',base='MLP',koopman='Koopman',oscillator='Krauss',chen_direction='Chen',hov='HOV',window='Window MLP',hov_no_memory='Reference only',hov_no_play='Time only',hov_no_maxwell='Path only',path='+ Path',time='+ Time',both='+ Both',static_capacity='Static expansion')
CN = dict(linear='线性回归',pcc='PCC',base='MLP',koopman='Koopman型模型',oscillator='Krauss潜振子',chen_direction='Chen方向网络',hov='本文模型',window='窗口MLP',hov_no_memory='参考形态（重训）',hov_no_play='仅时间记忆（重训）',hov_no_maxwell='仅路径记忆（重训）',path='当前输入＋路径记忆',time='当前输入＋时间记忆',both='当前输入＋双记忆',static_capacity='当前输入的静态扩展')
BLUE, ORANGE, GRAY, INK = '#2563A6','#D97732','#8A94A4','#27313D'
COLOR = {name: (BLUE if name=='hov' else ORANGE if name=='window' else GRAY) for name in MAIN}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A8AFB8',
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'axes.grid':False,'figure.facecolor':'white','savefig.facecolor':'white'})


def read(p): return json.loads(Path(p).read_text())
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean(v):
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def summary():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(RUN/'raw_test.csv'); v=pd.read_csv(RUN/'raw_validation.csv')
    rows=[]
    for model,group in d.groupby('model',sort=False):
        n=len(group); assert n in (1,20)
        if n==20: assert sorted(group.seed.astype(int))==list(range(100,120))
        item=dict(model=model,label=CN.get(model,model),n=n,test_frames=int(group.test_frames.iloc[0]),stored_parameters=int(group.parameter_count.iloc[0]))
        metrics=['mean_node_mm','node_rmse_mm','node_global_rmse_mm','endpoint_mm','mask_iou','mask_dice']
        for metric in metrics:
            x=group[metric].dropna()
            item[metric]=dict(mean=float(x.mean()),sd=float(x.std(ddof=1)) if len(x)>1 else None) if len(x) else None
        if n==20:
            counts=[]
            for seed in range(100,120):
                fit=read(RUN/f'formal/{model}/seed_{seed}/run_manifest.json')
                counts.append(fit['trainable_parameter_count']+fit['fitted_reference_buffer_count'])
            assert len(set(counts))==1
            item['active_fitted_parameters']=counts[0]
        else: item['active_fitted_parameters']=item['stored_parameters']
        rows.append(item)
    result=dict(run=str(RUN.relative_to(ROOT)),formal_fits=len(d),seeds=list(range(100,120)),models=rows,
                statistics=read(RUN/'paired_statistics.json'),data=read(RUN/'data_check.json'))
    write(OUT/'summary.json',clean(result))
    pd.DataFrame([dict(model=x['model'],label=x['label'],n=x['n'],active_fitted_parameters=x['active_fitted_parameters'],
        **{k+'_'+stat:x[k][stat] if x[k] else None for k in metrics for stat in ['mean','sd']}) for x in rows]).to_csv(OUT/'model_summary.csv',index=False)
    return d,{x['model']:x for x in rows}


def finish(fig,name,description):
    FIG.mkdir(parents=True,exist_ok=True); (REPORT/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/f'{name}.png',dpi=250,bbox_inches='tight',pad_inches=.08)
    for extension in ('svg','pdf','png'): shutil.copyfile(FIG/f'{name}.{extension}',REPORT/f'figures/{name}.{extension}')
    plt.close(fig)
    return dict(id=name,caption=description,formats=['svg','pdf','png'],path=str((FIG/f'{name}.png').relative_to(ROOT)))


def prediction_figure(d,s):
    fig,axes=plt.subplots(1,3,figsize=(11.8,4.1),sharey=True,layout='constrained')
    metrics=[('mean_node_mm','(a) Skeleton error','Mean node distance (mm)'),('endpoint_mm','(b) Endpoint error','Endpoint distance (mm)'),('mask_iou','(c) Silhouette overlap','Mask IoU')]
    for ax,(metric,title,xlabel) in zip(axes,metrics):
        for i,m in enumerate(MAIN):
            x=d.loc[d.model==m,metric].dropna().to_numpy(); y=i+np.linspace(-.14,.14,len(x))
            ax.scatter(x,y,color=COLOR[m],alpha=.27,s=10,zorder=2)
            ax.errorbar(x.mean(),i,xerr=x.std(ddof=1) if len(x)>1 else 0,fmt='D' if m=='window' else 'o',color=COLOR[m],mec=INK,mew=.5,ms=5,capsize=3,elinewidth=1.4,zorder=3)
        ax.set(xlabel=xlabel,title=title);ax.grid(axis='x',color='#E3E7EB',linewidth=.7);ax.set_axisbelow(True)
    axes[0].set(yticks=range(len(MAIN)),yticklabels=[EN[m] for m in MAIN]);axes[0].invert_yaxis()
    axes[0].set_xlim(1.28,2.34);axes[1].set_xlim(1.9,5.5);axes[2].set_xlim(.74,.845)
    return finish(fig,'prediction_summary','All test frames pooled. Dots: 20 training seeds; central marks and whiskers: mean ± sample SD. Linear: one deterministic fit. Focused point-plot axes; lower errors / higher IoU are better.')


def ablation_figure(d,s):
    names=['hov_no_memory','hov_no_play','hov_no_maxwell','hov'];fig,axes=plt.subplots(1,2,figsize=(9.4,3.2),layout='constrained')
    for i,m in enumerate(names):
        vals=d.loc[d.model==m].sort_values('seed').mean_node_mm.to_numpy()
        axes[0].scatter(i+np.linspace(-.1,.1,20),vals,color=BLUE if m=='hov' else GRAY,alpha=.4,s=10)
        axes[0].errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='o',ms=6,color=BLUE if m=='hov' else GRAY,capsize=3)
    axes[0].set(xticks=range(4),xticklabels=['Reference\nonly','Time\nonly','Path\nonly','Full'],ylabel='Skeleton error (mm)',title='(a) Retrained memory variants',ylim=(1.43,1.98))
    contrasts=[r for r in read(RUN/'paired_statistics.json')['contrasts'] if r['family']=='ablation']
    for i,r in enumerate(contrasts):
        effect=-r['mean_reference_minus_alternative_mm'];lo=-r['bootstrap95_upper_mm'];hi=-r['bootstrap95_lower_mm']
        axes[1].errorbar(effect,i,xerr=[[effect-lo],[hi-effect]],fmt='o',color=BLUE,capsize=4);axes[1].text(effect+.014,i,f'{effect:.3f}',va='center',fontsize=8)
    axes[1].set(yticks=range(3),yticklabels=['Remove path','Remove time','Remove both'],xlabel='Error increase relative to full HOV (mm)',title='(b) Paired effects and 95% CIs',xlim=(-.015,.53));axes[1].invert_yaxis();axes[1].axvline(0,color=INK,linewidth=.8)
    for ax in axes:ax.grid(axis='x' if ax==axes[1] else 'y',color='#E3E7EB',linewidth=.7);ax.set_axisbelow(True)
    return finish(fig,'memory_ablation','20 paired training seeds. Panel (a): mean ± sample SD, showing all seeds. Panel (b): paired-bootstrap 95% CIs. Each variant is retrained; this is distinct from removing a branch after fitting.')


def plugin_figure(d,s):
    labels=['Current','+ Path','+ Time','+ Both','Static\nexpansion','Window'];fig,axes=plt.subplots(1,2,figsize=(10.2,3.5),layout='constrained')
    linear_names=['linear','linear_path','linear_time','linear_both','linear_static_capacity','linear_window']
    lv=np.array([s[m]['mean_node_mm']['mean'] for m in linear_names]);mv=np.array([s[m]['mean_node_mm']['mean'] for m in VARIANTS]);sd=[s[m]['mean_node_mm']['sd'] for m in VARIANTS]
    axes[0].plot(np.arange(6)-.10,lv,marker='s',mfc='white',color=GRAY,ls='none',label='Linear (one fit)')
    axes[0].errorbar(np.arange(6)+.10,mv,yerr=sd,fmt='o',color=BLUE,capsize=3,lw=1.3,label='MLP (20 seeds)')
    axes[0].set(xticks=range(6),xticklabels=labels,ylabel='Skeleton error (mm)',ylim=(1.3,2.32),title='(a) Memory features across two readouts');axes[0].legend(frameon=False)
    contrasts=[r for r in read(RUN/'paired_statistics.json')['contrasts'] if r['family']=='plugin']
    for i,r in enumerate(contrasts):
        e=r['mean_reference_minus_alternative_mm'];lo=r['bootstrap95_lower_mm'];hi=r['bootstrap95_upper_mm']
        axes[1].errorbar(e,i,xerr=[[e-lo],[hi-e]],fmt='o',mfc='white' if e<0 else BLUE,color=BLUE,capsize=3)
    axes[1].axvline(0,color=INK,lw=.8);axes[1].set(yticks=range(5),yticklabels=['+ Path','+ Time','+ Both','Static expansion','Window'],xlabel='MLP error − variant error (mm)',title='(b) MLP paired effects and 95% CIs',xlim=(-.08,.64));axes[1].invert_yaxis()
    for ax in axes:ax.grid(axis='y' if ax==axes[0] else 'x',color='#E3E7EB',linewidth=.7);ax.set_axisbelow(True)
    return finish(fig,'memory_plugin','Same 64/64 MLP hidden layers, 100-epoch budget, fixed train/validation/test split. Panel (a) whiskers: sample SD. Panel (b): paired bootstrap 95% CI; positive values mean lower error than MLP.')


def sequence_figure(d,s):
    seq=pd.read_csv(RUN/'raw_test_by_sequence.csv');groups=sorted(seq.group.unique());vals=np.array([[seq.loc[(seq.model==m)&(seq.group==g),'mean_node_mm'].mean() for g in groups] for m in MAIN]);fig,axes=plt.subplots(1,2,figsize=(9.5,4.1),layout='constrained',gridspec_kw={'width_ratios':[1.25,1]})
    im=axes[0].imshow(vals,cmap='Blues',vmin=1.2,vmax=2.35,aspect='auto')
    for i in range(8):
        for j in range(3):axes[0].text(j,i,f'{vals[i,j]:.3f}',ha='center',va='center',color='white' if vals[i,j]>1.88 else INK,fontsize=8)
    axes[0].set(yticks=range(8),yticklabels=[EN[m] for m in MAIN],xticks=range(3),xticklabels=[g[-6:] for g in groups],title='(a) Error by held-out recording');fig.colorbar(im,ax=axes[0],label='Skeleton error (mm)',shrink=.8)
    for m in MAIN:
        x=s[m]['stored_parameters'];y=s[m]['mean_node_mm']['mean'];axes[1].scatter(x,y,color=COLOR[m],marker='D' if m=='window' else 'o',s=40)
        offsets={'linear':(7,-4),'pcc':(7,-4),'base':(7,-4),'koopman':(-8,10),'oscillator':(-7,-14),'chen_direction':(-38,9),'hov':(7,4),'window':(7,-3)}
        axes[1].annotate(EN[m],(x,y),xytext=offsets[m],textcoords='offset points',fontsize=8,ha='right' if m in ('koopman','oscillator') else 'left')
    axes[1].set_xscale('log');axes[1].set(xlabel='Parameters / fitted coefficients (log scale)',ylabel='Pooled skeleton error (mm)',title='(b) Representation size and accuracy',xlim=(140,100000),ylim=(1.32,2.32));axes[1].grid(color='#E3E7EB',lw=.7);axes[1].set_axisbelow(True)
    return finish(fig,'sequence_and_capacity','Descriptive recording-level results use the same models and split, not new independent repeats. The three recordings contribute 2457, 238, 263 targets; the main metric pools frames. Parameters refer to the fitted representation.')


def examples_figure():
    source=np.load(RUN/'test_targets.npz');gt=source['target_mm'];groups=source['groups'];ids=source['frame_ids'];p={}
    for m in ['hov','window']:
        path=RUN/f'evaluation/{m}/seed_100/predictions.npz'
        if not path.exists():path=next(RUN.glob(f'**/{m}/seed_100/predictions.npz'))
        p[m]=np.load(path)['prediction_mm']
    fig,axes=plt.subplots(2,3,figsize=(9.2,5.9),layout='constrained',gridspec_kw={'height_ratios':[2,1]});picked=[]
    sequence_ids=['172644','181044','181548'];max_error=0
    for j,ax in enumerate(axes[0]):
        candidates=np.flatnonzero(groups==j);k=int(candidates[np.argmax(np.abs(gt[candidates,-1,0]))]);picked.append(dict(group=int(j),sequence=sequence_ids[j],target_index=k,frame_id=int(ids[k]),rule='maximum absolute target endpoint x within each held-out recording; selected using target geometry alone'))
        for name,array,color,style in [('Visual target',gt,INK,'--'),('HOV',p['hov'],BLUE,'-'),('Window MLP',p['window'],ORANGE,':')]:ax.plot(array[k,:,0],array[k,:,1],color=color,ls=style,lw=1.5,marker='o' if name=='HOV' else None,ms=2,label=name)
        ax.set_aspect('equal',adjustable='box');ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'({chr(97+j)}) {sequence_ids[j]} / frame {int(ids[k])}',xlim=(-65,55),ylim=(185,-5))
        if j==0:ax.legend(frameon=False,fontsize=7,loc='upper left')
        for name,color,style in [('hov',BLUE,'o-'),('window',ORANGE,'s:')]:
            err=np.linalg.norm(p[name][k]-gt[k],axis=1);max_error=max(max_error,float(err.max()))
            axes[1,j].plot(range(15),err,style,color=color,lw=1.2,ms=3,label=EN[name])
        axes[1,j].set(xlabel='Node index (base = 0)',ylabel='Node error (mm)',xticks=[0,4,7,10,14],title=f'({chr(100+j)}) Local prediction error');axes[1,j].grid(axis='y',color='#E3E7EB',lw=.7)
    for ax in axes[1]:ax.set_ylim(0,max_error*1.12)
    write(OUT/'qualitative_selection.json',picked)
    return finish(fig,'shape_examples','Seed 100; one target per recording with the largest absolute lateral endpoint displacement, chosen from target geometry without model errors. Upper panels: shared equal-aspect image-plane axes; lower panels: node distances with a shared scale. These high-displacement examples illustrate spatial errors, not average performance.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--aggregate-only',action='store_true');args=parser.parse_args()
    d,s=summary()
    if args.aggregate_only:return
    records=[prediction_figure(d,s),ablation_figure(d,s),plugin_figure(d,s),sequence_figure(d,s),examples_figure()]
    write(OUT/'figure_catalog.json',records)
    write(OUT/'figure_design.json',dict(policy='Two accent roots (blue, orange) plus neutrals; direct labels and shape/line distinctions. One question per panel.',
        uncertainty='SD for per-model distributions; paired-bootstrap 95% CI for contrast effects. Definitions are given separately.',
        audience='Technical research; paper exports plus portable HTML report.',export='SVG/PDF vectors and 250-dpi PNG',figures=records))
    print(json.dumps(dict(summary=str(OUT/'summary.json'),figures=len(records)),ensure_ascii=False))


if __name__=='__main__':main()
