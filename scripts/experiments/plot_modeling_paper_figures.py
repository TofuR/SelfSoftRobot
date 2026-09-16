#!/usr/bin/env python3
"""Export manuscript figures from the completed representation and repeat analyses."""
from pathlib import Path
import json
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/selfsoftrobot-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'workspace/reports/modeling_paper_revision_20260913_002'
DEST=ROOT/'docs/icra2027/figures'
DEST.mkdir(parents=True,exist_ok=True)
read=lambda name:json.loads((SRC/(name+'.json')).read_text())
R,P,E=(read(n) for n in ['representation','plugin','efficiency'])


def chart(module,id):
    return next(c['rows'] for c in module['charts'] if c['id']==id)


def table(module,id):
    return module['tables'][id] if isinstance(module['tables'],dict) else next(t['rows'] for t in module['tables'] if t['id']==id)


def finish(fig,name):
    fig.tight_layout(pad=1.1)
    fig.savefig(DEST/(name+'.svg'),bbox_inches='tight')
    fig.savefig(DEST/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close(fig)


plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                     'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white'})
colors=['#2563a6','#d97732','#31886b','#8b5bab','#667085','#ac6179']
fig,axes=plt.subplots(1,3,figsize=(10.4,3.1))
reg=[r for r in table(R,'play_regimes') if r['split']=='test']
reg.sort(key=lambda r:r['threshold'])
x=np.arange(2)
axes[0].bar(x-.18,[r['interior_fraction']*100 for r in reg],.36,label='Interior accumulation',color=colors[0])
axes[0].bar(x+.18,[r['clipped_fraction']*100 for r in reg],.36,label='Boundary clipping',color=colors[1])
axes[0].set(xticks=x,xticklabels=['r = 0.02','r = 0.50'],ylabel='Active updates (%)',ylim=(0,112),title='(a) Path-memory regimes')
axes[0].legend(loc='upper left',fontsize=7)
kernel=chart(R,'representation_time_kernel')
for c,node in zip(colors,sorted({r['node'] for r in kernel})):
    rows=sorted([r for r in kernel if r['node']==node],key=lambda r:r['lag_s'])
    axes[1].plot([r['lag_s'] for r in rows],[r['lateral_gain_mm'] for r in rows],label=f'Node {node}',color=c)
axes[1].set(xlabel='Lag (s)',ylabel='Lateral gain (mm / drive unit)',title='(b) Learned local time kernel')
axes[1].legend(fontsize=7)
profiles=chart(R,'representation_geometry_profile')
labels={'实际模型位移':'Exact model displacement','一阶近似':'Jacobian approximation','一阶预测':'Jacobian approximation','余项':'First-order remainder'}
for c,m in zip(colors,dict.fromkeys(r['method'] for r in profiles)):
    rows=[r for r in profiles if r['method']==m]
    y=np.array([r['displacement_mm'] for r in rows]);xx=np.array([r['node'] for r in rows])
    translated=labels.get(m,'First-order remainder' if '余' in m else 'Jacobian approximation')
    axes[2].plot(xx,y,label=translated,color=c,linestyle='--' if 'Jacobian' in translated else '-',linewidth=1.7)
axes[2].set(xlabel='Node index (base to tip)',ylabel='Displacement (mm)',title='(c) Geometry transmission')
axes[2].legend(fontsize=7)
finish(fig,'memory_representation')

variants=['base','path','time','both','static_capacity','window']
names=['Current input','+ Path memory','+ Time memory','+ Both memories','Static features','Full window']
fig,axes=plt.subplots(1,2,figsize=(8.2,3.0),sharey=True)
lin={r['variant']:r for r in table(P,'linear_descriptive')}
mlp={r['variant']:r for r in P['plugin_summary']}
for ax,lookup,title,metric,sd in [(axes[0],lin,'(a) Linear readout (deterministic)','mean_node_mm',None),
                                (axes[1],mlp,'(b) MLP (20 seeds)','mean_node_mm_mean','mean_node_mm_sd')]:
    vals=[lookup[v][metric] for v in variants]
    errors=[lookup[v][sd] for v in variants] if sd else None
    ax.barh(np.arange(6),vals,color=colors,xerr=errors,capsize=3,error_kw={'linewidth':1})
    ax.set(yticks=np.arange(6),yticklabels=names,xlabel='Mean skeleton error (mm)',xlim=(0,2.6),title=title)
    for i,value in enumerate(vals):ax.text(value+.065,i,f'{value:.3f}',va='center',fontsize=8)
axes[0].invert_yaxis()
finish(fig,'memory_plugin')

fig,axes=plt.subplots(1,3,figsize=(10.4,3.1))
rows=chart(E,'efficiency_validation_early')
models={'hov':'HOV','mlp':'Static MLP','chen_direction':'Chen','oscillator':'Krauss','window_mlp':'Window MLP'}
for c,(model,name) in zip(colors,models.items()):
    rr=sorted([r for r in rows if r['model']==model],key=lambda r:r['epoch'])
    axes[0].plot([r['epoch'] for r in rr],[r['val_mm_mean'] for r in rr],label=name,color=c,marker='o',markersize=3)
axes[0].set(xlabel='Joint-optimization epoch',ylabel='Validation error (mm)',title='(a) Early prediction accuracy')
axes[0].legend(fontsize=7)
stages=chart(E,'efficiency_hov_stages')
axes[1].bar(np.arange(4),[r['val_mm_mean'] for r in stages],color=[colors[4],colors[2],colors[0],colors[3]])
axes[1].set(xticks=np.arange(4),xticklabels=['Reference','Memory\ninit.','Epoch 1','Best'],ylabel='Validation error (mm)',title='(b) HOV fitting stages',ylim=(0,2.45))
for i,r in enumerate(stages):axes[1].text(i,r['val_mm_mean']+.04,f"{r['val_mm_mean']:.3f}",ha='center',fontsize=8)
times=table(E,'efficiency_cpu_summary')
order=['hov','hov_cached','mlp','chen_direction','oscillator','window_mlp']
titles={'hov':'HOV window','hov_cached':'HOV cached','mlp':'Static MLP','chen_direction':'Chen','oscillator':'Krauss','window_mlp':'Window MLP'}
lookup={r['mode']:r for r in times}
order=[m for m in order if m in lookup]
axes[2].barh(np.arange(len(order)),[lookup[m]['p95_ms_mean'] for m in order],color=colors,
             xerr=[lookup[m]['p95_ms_sd'] for m in order],capsize=2)
axes[2].set(yticks=np.arange(len(order)),yticklabels=[titles[m] for m in order],xlabel='p95 model latency (ms)',title='(c) CPU, one thread, batch 1')
axes[2].invert_yaxis()
finish(fig,'learning_and_inference')
print(json.dumps({'directory':str(DEST),'figures':['memory_representation','memory_plugin','learning_and_inference']}))
