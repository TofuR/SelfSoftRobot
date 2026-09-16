#!/usr/bin/env python3
"""Four-channel temporal kernels and factors for v3 section 3.6.

Chart contract: compare lateral node-lag kernels at one training-mean reference
and show their rank-one factors. Four heatmaps have a shared signed linear
color scale; time/gain lines distinguish four channels with color and dash.
Heatmaps display 0--2 s; ALL SVDs and scalar metrics retain all 19 lags, 0--3.6 s.
The waveform panel explicitly retains the complete domain. No seed selection.
Display factors come from SVD of the 20-model mean; numerical claims average
individual-model results over the six existing training-selected references.
Static publication export: 174-mm width; 8--9pt fonts; SVG/PDF + Word PNG.
"""
from pathlib import Path
import os, json
os.environ.setdefault('MPLCONFIGDIR','/tmp/selfsr-section36-v3-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'workspace/runs/analysis/modeling_unified20_20260913_005/geometry'
OUT=ROOT/'docs/icra2027/figures/section36_v3'
REPORT=ROOT/'workspace/reports/section36_rewrite_20260915'
OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
data=np.load(SRC/'kernels.npz');seeds=data['seeds'];lag=data['lag_seconds'];nodes=data['node_ids']
assert seeds.tolist()==list(range(100,120))
full=data['kernel_xy_mm_per_delta_e'][...,0]
mean=full[:,0].mean(0)
cutoff=2.0;sel=lag<=cutoff+1e-9
colors=['#2864a5','#cf7935','#268b82','#7958a6']
styles=['-','--','-.',':'];INK='#25313D';GRAY='#7F8790'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
 'axes.labelsize':8,'axes.titlelocation':'left','axes.titlepad':5,'axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.edgecolor':GRAY,'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.dpi':320,'savefig.facecolor':'white'})
wave=[];gain=[];display_metrics=[];rows=[]
for c in range(4):
 u,s,vt=np.linalg.svd(mean[c],full_matrices=False)
 sign=1 if vt[0,np.argmax(abs(vt[0]))]>=0 else -1
 g=u[:,0]*s[0]*sign;w=vt[0]*sign;gain.append(g);wave.append(w)
 residual=mean[c]-np.outer(g,w)
 rho=1-np.sum(residual**2)/np.sum(mean[c]**2)
 assert np.isclose(rho,s[0]**2/np.sum(s*s))
 display_metrics.append({'channel':c,'rank1_fraction_of_mean_kernel':float(rho),
  'displayed_squared_response_fraction':float(np.sum(mean[c,:,sel]**2)/np.sum(mean[c]**2)),
  'tail_squared_response_fraction':float(np.sum(mean[c,:,~sel]**2)/np.sum(mean[c]**2))})
 for ni,n in enumerate(nodes):
  for li,t in enumerate(lag):rows.append({'channel':c,'node':int(n),'lag_s':float(t),
   'mean_kernel_mm_per_delta_e':float(mean[c,ni,li]),'gain':float(g[ni]),'wave':float(w[li]),
   'rank1_kernel':float(g[ni]*w[li]),'visible_in_heatmap':bool(sel[li])})
wave=np.array(wave);gain=np.array(gain)
metric_rows=[]
for si,seed in enumerate(seeds):
 for ref in range(full.shape[1]):
  for c in range(4):
   a=full[si,ref,c];s=np.linalg.svd(a,compute_uv=False)
   norms=np.linalg.norm(a,axis=1);z=a[norms>1e-12]/norms[norms>1e-12,None]
   sn=np.linalg.svd(z,compute_uv=False)
   metric_rows.append({'seed':int(seed),'reference':ref,'channel':c,
    'rank1_fraction':float(s[0]**2/np.sum(s*s)),
    'node_normalized_rank1_fraction':float(sn[0]**2/np.sum(sn*sn))})
metrics=pd.DataFrame(metric_rows)
# Top row four comparable heatmaps, bottom row two full-width factor comparisons.
fig=plt.figure(figsize=(6.85,4.10))
gs=fig.add_gridspec(2,4,left=.085,right=.935,bottom=.12,top=.88,
 hspace=.72,wspace=.20,height_ratios=[1.05,1.0])
cmap=LinearSegmentedColormap.from_list('signed_response',['#2864a5','#ffffff','#cf7935'])
limit=20.0
for c in range(4):
 ax=fig.add_subplot(gs[0,c])
 im=ax.imshow(mean[c][:,sel],origin='lower',aspect='auto',interpolation='nearest',
  extent=[-.1,cutoff+.1,.5,14.5],cmap=cmap,vmin=-limit,vmax=limit)
 ax.set_title(f'Channel {c}',color=colors[c])
 ax.set_xticks([0,1,2]);ax.set_xlabel('Lag (s)')
 ax.set_yticks([1,7,14]);ax.tick_params(length=2)
 ax.axhline(7,color=GRAY,lw=.5,ls=':')
 if c==0:ax.set_ylabel('Node (base to tip)')
 else:ax.set_yticklabels([])
 for sp in ax.spines.values():sp.set_visible(False)
cax=fig.add_axes([.95,.55,.012,.33]);cb=fig.colorbar(im,cax=cax,ticks=[-20,0,20])
cb.ax.tick_params(labelsize=6.5,length=2)
fig.text(.085,.965,'(a) Lateral response kernels',fontsize=9,weight='bold')
fig.text(.085,.925,r'Training-mean reference; 20-model mean; kernel unit: mm / unit $\Delta e_c$',fontsize=7)
axw=fig.add_axes([.085,.12,.365,.285]);axg=fig.add_axes([.565,.12,.370,.285])
for c in range(4):
 axw.plot(lag,wave[c],styles[c],color=colors[c],lw=1.55,label=f'Ch. {c}')
 axg.plot(nodes,gain[c],styles[c],color=colors[c],lw=1.55,
   marker=['o','s','^','D'][c],markersize=2.5,markevery=2)
axw.set(xlabel='Lag (s)',ylabel='Waveform (unit norm)',xlim=(0,3.6),xticks=[0,1,2,3,3.6])
axw.set_title('(b) Shared temporal waveforms',weight='bold')
axw.legend(frameon=False,ncol=2,fontsize=6.5,handlelength=2,loc='upper right',columnspacing=.7)
axg.set(xlabel='Node (base to tip)',ylabel=r'Gain (mm / unit $\Delta e_c$)',xticks=[1,4,7,10,14],xlim=(1,14))
axg.set_title('(c) Signed node gains',weight='bold')
axg.axvline(7,color=GRAY,lw=.6,ls=':')
for ax in [axw,axg]:ax.axhline(0,color=GRAY,lw=.6);ax.tick_params(length=2)
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'temporal_memory.{ext}')
plt.close(fig)
pd.DataFrame(rows).to_csv(REPORT/'temporal_figure_data.csv',index=False)
metrics.to_csv(REPORT/'temporal_individual_rank_metrics.csv',index=False)
np.savez(REPORT/'temporal_display_factors.npz',lag_seconds=lag,nodes=nodes,
 mean_kernel=mean,wave=wave,gain=gain,seeds=seeds)
summary={'source':str(SRC.relative_to(ROOT)/'kernels.npz'),
 'reference':0,'reference_actions':data['reference_actions'][0].tolist(),
 'display_aggregation':'SVD of mean kernel over all 20 models; separate SVD per channel',
 'svd_lags_s':[float(lag.min()),float(lag.max())],'heatmap_lags_s':[0,cutoff],
 'waveform_lags_s':[0,float(lag.max())],
 'display_metrics':display_metrics,'rank1_fraction_all_models_references_channels':float(metrics.rank1_fraction.mean()),
 'normalized_rank1_fraction_all_models_references_channels':float(metrics.node_normalized_rank1_fraction.mean()),
 'palette':colors,'heatmap_limits':[-limit,limit],
 'chart_contract':__doc__,'qa':{'shared_linear_color_scale':True,'svd_uses_full_horizon':True,
 'all_20_models_included':True,'vector_exports':['svg','pdf'],'no_interpolated_data':True}}
(REPORT/'temporal_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['display_metrics','rank1_fraction_all_models_references_channels','normalized_rank1_fraction_all_models_references_channels']},indent=2))
