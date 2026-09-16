#!/usr/bin/env python3
"""Publication figures from completed training, actual timings and aligned targets."""
import numpy as np
import pandas as pd
from plot_unified20_results import OUT, BLUE, ORANGE, GRAY, INK, MAIN, EN, plt, finish, write


def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#E3E7EB', linewidth=.7)
    ax.set_axisbelow(True)


def learning_figure():
    history=pd.read_csv(OUT/'efficiency/training_history_raw.csv')
    train=pd.read_csv(OUT/'efficiency/training_summary.csv').set_index('model')
    latency=pd.read_csv(OUT/'efficiency/inference_summary.csv').set_index('mode')
    fig,axes=plt.subplots(2,2,figsize=(11.2,7.1),layout='constrained')
    styles={'hov':(BLUE,'-','o'),'window':(ORANGE,'--','D'),'chen_direction':(GRAY,'-.','s'),'base':(INK,':','^')}
    for model,(color,ls,marker) in styles.items():
        g=history[history.model==model].groupby('epoch').validation_node_mean_mm.agg(['mean','std'])
        for ax,early in [(axes[0,0],True),(axes[0,1],False)]:
            values=g[g.index<=10] if early else g[g.index>=10]
            x=values.index.to_numpy();mu=values['mean'].to_numpy();sd=values['std'].to_numpy()
            ax.plot(x,mu,color=color,ls=ls,marker=marker,ms=3 if early else 2,lw=1.3,label=EN[model])
            ax.fill_between(x,mu-sd,mu+sd,color=color,alpha=.10,linewidth=0)
    axes[0,0].set(yscale='log',xlabel='Epoch',ylabel='Validation skeleton error (mm; log)',title='(a) Early validation trajectory',xticks=[1,5,10],ylim=(1.3,9.5))
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2,loc='upper right')
    axes[0,1].set(xlabel='Epoch',ylabel='Validation skeleton error (mm)',title='(b) Validation trajectory, epochs 10–100',ylim=(1.30,2.75),xticks=[10,25,50,75,100])
    for ax in axes[0]:grid(ax)
    for i,model in enumerate(MAIN):
        row=train.loc[model];mu=row.task_wall_to_COMPLETE_seconds_mean;sd=row.task_wall_to_COMPLETE_seconds_sd
        color=BLUE if model=='hov' else ORANGE if model=='window' else GRAY
        axes[1,0].errorbar(mu,i,xerr=0 if pd.isna(sd) else sd,fmt='D' if model=='window' else 'o',color=color,capsize=3,ms=5)
        axes[1,0].annotate(f'{mu:.2f}',(mu,i),xytext=(8,0),textcoords='offset points',fontsize=8,va='center')
    axes[1,0].set(xscale='log',xlim=(.025,150),yticks=range(8),yticklabels=[EN[m] for m in MAIN],xlabel='Task wall time (s; log scale)',title='(c) Original concurrent fitting records');axes[1,0].invert_yaxis();grid(axes[1,0],'x')
    modes=['linear_h20','pcc_h20','base_h20','koopman_h20','oscillator_h20','chen_direction_h20','hov_h20_full','hov_cached_step','window_h20']
    labels=['Linear','PCC','Static MLP','Koopman','Krauss','Chen','HOV: full H20','HOV: cached step','Window MLP']
    for i,(mode,label) in enumerate(zip(modes,labels)):
        row=latency.loc[mode];color=BLUE if mode.startswith('hov') else ORANGE if mode.startswith('window') else GRAY
        p50,p95=row.p50_ms_mean,row.p95_ms_mean
        axes[1,1].plot([p50,p95],[i,i],color=color,lw=1.2)
        axes[1,1].scatter(p50,i,s=25,color=color,marker='o',label='p50' if i==0 else None)
        axes[1,1].scatter(p95,i,s=23,edgecolor=color,facecolor='white',marker='s',label='p95' if i==0 else None)
    axes[1,1].set(xscale='log',xlim=(.05,3.3),yticks=range(9),yticklabels=labels,xlabel='Per-call latency (ms; log scale)',title='(d) CPU, one thread, batch size 1');axes[1,1].invert_yaxis();grid(axes[1,1],'x');axes[1,1].legend(frameon=False,loc='upper right',ncol=2)
    return finish(fig,'learning_and_inference','Panels (a,b): current validation errors, 20-seed mean ± sample SD; HOV had reference and memory initialization before epoch 1. Four methods displayed for legibility; all eight appear in the report. (c): task wall time through result completion, including initialization, under original eight-worker CPU fitting; mean ± SD, not exclusive compute time. (d): mean across per-model p50/p95 from 500 measured calls after 50 warmups. The cached state is prepared from the same H20 prefix; its preparation is excluded from one-step latency. Model evaluation excludes image processing, communication and actuation.')


def sampling_figure():
    d=pd.read_csv(OUT/'sampling/pooled_seed.csv');stats=pd.read_csv(OUT/'sampling/paired_statistics.csv')
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.8),layout='constrained',gridspec_kw={'width_ratios':[1,1,1.15]})
    protocols=['nominal_10Hz_H39','decimated_5Hz_H20','wrong_dt_10Hz_H39_dt0.2']
    models=[('hov',BLUE,'HOV',-.09),('hov_no_maxwell',GRAY,'Path only',.09)]
    for ax,metric,title in zip(axes[:2],['node_mean_mm','endpoint_mean_mm'],['(a) Aligned skeleton error','(b) Aligned endpoint error']):
        for model,color,label,offset in models:
            dd=d[d.model==model]
            first=dd[dd.protocol==protocols[0]].sort_values('seed')[metric].to_numpy()
            second=dd[dd.protocol==protocols[1]].sort_values('seed')[metric].to_numpy()
            for v,w in zip(first,second):ax.plot([offset,1+offset],[v,w],color=color,alpha=.12,lw=.6)
            for j,protocol in enumerate(protocols):
                values=dd[dd.protocol==protocol][metric].to_numpy()
                ax.errorbar(j+offset,values.mean(),yerr=values.std(ddof=1),fmt='o' if j<2 else 'D',color=color,ms=5,capsize=3,label=label if j==0 else None)
        ax.axvline(1.5,color='#C3C8CF',ls=':',lw=.8)
        ax.set(xticks=range(3),xticklabels=['10 Hz\nH39 / 0.1 s','Stride 2\nH20 / 0.2 s','10 Hz\nH39 / 0.2 s'],ylabel='Error (mm)',xlabel='History length / nominal time step',title=title,xlim=(-.35,2.4));grid(ax)
        ax.text(2,.99,'dt diagnostic',ha='center',va='top',transform=ax.get_xaxis_transform(),fontsize=7,color=INK)
    axes[0].set_ylim(1.62,1.87);axes[1].set_ylim(2.78,3.56);axes[0].legend(frameon=False,loc='upper left',ncol=2,fontsize=7)
    comparisons=[('hov_sampling_5Hz_minus_10Hz','node_mean_mm','HOV / skeleton',BLUE),('no_maxwell_sampling_5Hz_minus_10Hz','node_mean_mm','Path only / skeleton',GRAY),('hov_sampling_5Hz_minus_10Hz','endpoint_mean_mm','HOV / endpoint',BLUE),('no_maxwell_sampling_5Hz_minus_10Hz','endpoint_mean_mm','Path only / endpoint',GRAY)]
    for i,(comparison,metric,label,color) in enumerate(comparisons):
        row=stats[(stats.scope=='pooled')&(stats.comparison==comparison)&(stats.metric==metric)].iloc[0]
        value=row.mean_delta_mm;lo=row.ci95_bootstrap_low_mm;hi=row.ci95_bootstrap_high_mm
        axes[2].errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o' if metric=='node_mean_mm' else 's',color=color,capsize=4,ms=5)
    axes[2].set(yticks=range(4),yticklabels=[x[2] for x in comparisons],xlabel='Stride-2 error − 10 Hz error (mm)',title='(c) Paired differences and 95% CIs',xlim=(-.06,.09));axes[2].invert_yaxis();axes[2].axvline(0,color=INK,lw=.8);grid(axes[2],'x')
    return finish(fig,'sampling_alignment','Frozen models, seeds 100–119; 3937 unique common targets from two older RGB label records, pooled once per target. Panels (a,b): mean ± sample SD with paired seed trajectories; both correct protocols span 3.8 s nominally. Third category keeps H39 but intentionally uses the wrong 0.2 s update. Panel (c): paired percentile-bootstrap 95% intervals across 20 seeds, not across frames. Recorded acquisition was approximately 9.05–9.08 Hz; stride-2 removes intermediate commands from the same execution and does not recreate a 5 Hz execution.')


if __name__=='__main__':
    records=[learning_figure(),sampling_figure()]
    write(OUT/'figure_efficiency_sampling.json',records)
    print('Exported learning_and_inference and sampling_alignment (SVG/PDF/PNG).')
