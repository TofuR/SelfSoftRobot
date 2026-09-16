#!/usr/bin/env python3
"""Publication layout for validated section-3.6 path analysis.

Two panels: window-state response around input-defined reversals and retrained
ablation prediction errors at the same events. 20-fit mean +/- SD.
Line/dot (not bar) axes focus on observed ranges. Original 1121 events retained;
no event-independence inference. Pair selectors never use model/shape outcomes.
SVG/PDF plus 320-dpi PNG at 174-mm width, 7--8pt labels. Caption carries protocols.
"""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/selfsr-section36-v3-mpl')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'workspace/reports/section36_rewrite_20260915/path'
OUT=ROOT/'docs/icra2027/figures/section36_v3'
s=pd.read_csv(SRC/'path_state_reversal_per_seed.csv');s=s[s.scope=='all_events']
e=pd.read_csv(SRC/'reversal_error_per_seed.csv')
assert s.seed.nunique()==e.seed.nunique()==20
colors={'full':'#2864a5','time':'#cf7935','path':'#268b82','reference':'#777777'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,
 'axes.labelsize':7.5,'legend.fontsize':6.8,'axes.titlelocation':'left','axes.titlepad':7,
 'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#888888','text.color':'#25313D','axes.labelcolor':'#25313D','axes.linewidth':.6})
fig,axs=plt.subplots(1,2,figsize=(6.85,2.8))
fig.subplots_adjust(left=.085,right=.99,bottom=.20,top=.89,wspace=.32)
for ri,color,style in [(0,colors['full'],'-'),(1,colors['time'],'--')]:
 z=s[s.threshold_index==ri].groupby('lag_s').oriented_q_over_r.agg(['mean','std'])
 axs[0].plot(z.index,z['mean'],color=color,ls=style,lw=1.3,label=f'$r={s[s.threshold_index==ri].threshold.iloc[0]:g}$')
 axs[0].fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],color=color,alpha=.16,lw=0)
axs[0].set(title='(a) Path outputs',ylabel=r'Path output $s_{\mathrm{rev}} q/r$',ylim=(-1.07,1.07),yticks=[-1,-.5,0,.5,1])
axs[0].legend(frameon=False,loc='lower left',ncol=1,handlelength=2)
axs[0].axhline(0,color='#999999',lw=.6)
for name,label,style in [('reference','No memory',':'),('time','Time only','--'),('path','Path only','-.'),('full','Full','-')]:
 z=e[e.label==name].groupby('time_s').skeleton_error_mm.agg(['mean','std'])
 axs[1].plot(z.index,z['mean'],color=colors[name],ls=style,lw=1.3,label=label)
 axs[1].fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],color=colors[name],alpha=.12,lw=0)
axs[1].set(title='(b) Reversal errors',ylabel='Skeleton error (mm)',ylim=(1.43,2.22))
axs[1].legend(frameon=False,loc='upper right',ncol=2,columnspacing=.5,handlelength=1.6,handletextpad=.35)
for ax in axs[:2]:
 ax.axvline(0,color='#777777',ls=':',lw=.7)
 ax.set(xlabel='Time from reversal (s)',xticks=[-1,-.5,0,.5,1],xticklabels=['−1','−0.5','0','0.5','1'])
for ax in axs:
 ax.grid(axis='y',color='#e6e6e6',lw=.5);ax.tick_params(labelsize=7.0,width=.6,length=2)
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'path_memory.{ext}',dpi=320,facecolor='white')
plt.close(fig)
(SRC/'publication_figure_contract.json').write_text(json.dumps({'contract':__doc__,
 'size_inches':[6.85,2.8],'sources':['path_state_reversal_per_seed.csv','reversal_error_per_seed.csv'],
 'palette':colors,'font_pt':7.5,'caption_protocol_required':True},indent=2)+'\n')
print(OUT/'path_memory.png')
