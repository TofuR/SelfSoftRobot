#!/usr/bin/env python3
"""Post-training spatial and temporal response analysis; no fitting of robot data.

Uses all 20 frozen models. Rank-1/2 readout projections use the six existing
training-input reference geometries and are evaluated once on the saved test set.
Random directions are structural controls, not robot trials or a p-value null.
"""
from pathlib import Path
import os
import json

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/selfsr-memory-physics-mpl')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'workspace/runs/analysis/modeling_unified20_20260913_005/geometry'
OUT = ROOT / 'workspace/reports/memory_physics_20260914'
OUT.mkdir(parents=True, exist_ok=True)
K = np.load(SOURCE / 'kernels.npz')
SEEDS = K['seeds']
assert SEEDS.tolist() == list(range(100, 120))
G = K['generalized_kernel']
J = K['J_reference']
kernel = K['kernel_xy_mm_per_delta_e']
wh = K['W_time']
phi = K['phi']


def cosine(a, b):
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def rank_fraction(a, normalized=False):
    if normalized:
        norm = np.linalg.norm(a, axis=-1)
        a = a[norm > 1e-12] / norm[norm > 1e-12, None]
    vals = np.linalg.svd(a, compute_uv=False)
    return float(vals[0]**2 / np.sum(vals**2))


def decode(si, bend, loglength):
    theta = np.cumsum(bend, axis=1)
    ell = K['reference_segment_lengths_mm'][si][None] * np.exp(
        np.clip(loglength[:, np.repeat(np.arange(2), 7)], -.25, .25))
    inc = ell[..., None] * np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return K['base_position_mm'][si, :2] + np.concatenate(
        [np.zeros((len(bend), 1, 2)), np.cumsum(inc, axis=1)], axis=1)


def stats(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)),
                min=float(a.min()), max=float(a.max()))


# Contributions grouped by the location of the bending parameter, not by
# the location of the resulting displacement. Length contributions are separate.
component_masks = [np.arange(7), np.arange(7, 14), np.arange(14, 16)]
components = np.stack([
    np.einsum('srndg,scgl->srcnld', J[..., ids], G[:, :, ids, :])
    for ids in component_masks], axis=-1)
qa = {'max_component_sum_error': float(abs(components.sum(-1)-kernel).max())}
assert qa['max_component_sum_error'] < 1e-10

rows, pairs, gains, random_rows = [], [], [], []
for si, seed in enumerate(SEEDS):
    for ri in range(6):
        for c in range(4):
            a = kernel[si, ri, c, :, :, 0]
            u, vals, vt = np.linalg.svd(a, full_matrices=False)
            sign = np.sign(vt[0, np.argmax(abs(vt[0]))])
            gain = u[:, 0]*vals[0]*sign
            if ri == 0:
                gains.append(gain)
            for ni in range(14):
                curve = a[ni]
                contributions = components[si, ri, c, ni, :, 0]
                peak = int(np.argmax(abs(curve)))
                rows.append(dict(seed=int(seed), reference=ri, channel=c, node=ni+1,
                    dominant_gain=float(gain[ni]), peak_lag_s=float(K['lag_seconds'][peak]),
                    lag0=float(curve[0]), proximal_bend_lag0=float(contributions[0, 0]),
                    distal_bend_lag0=float(contributions[0, 1]), length_lag0=float(contributions[0, 2]),
                    kernel_l2=float(np.linalg.norm(curve)),
                    component_norm_sum=float(np.linalg.norm(contributions, axis=0).sum()),
                    cancellation_fraction=float(1-np.linalg.norm(curve)/
                        max(np.linalg.norm(contributions, axis=0).sum(), 1e-15))))
        for a, b in [(0, 1), (2, 3)]:
            pairs.append(dict(seed=int(seed), reference=ri, channel_pair=f'{a}-{b}',
                lateral_kernel_cosine=cosine(kernel[si,ri,a,:,:,0],kernel[si,ri,b,:,:,0])))
    # Norm-preserving random directions in standardized geometric coordinates.
    rng = np.random.default_rng(20260914+int(seed))
    scale = K['generalized_coordinate_scale'][si]
    for c in range(4):
        norms = np.linalg.norm(wh[si,c]/scale, axis=-1)
        for draw in range(100):
            direction = rng.normal(size=(6,16))
            direction /= np.linalg.norm(direction,axis=-1,keepdims=True)
            w = direction * norms[:,None] * scale[None]
            a = -J[si,0,:,0] @ w.T @ phi[si]
            generalized = -w.T @ phi[si]
            bend_square = np.sum(generalized[:14]**2, axis=-1)
            random_rows.append(dict(seed=int(seed), channel=c, draw=draw,
                rank1=rank_fraction(a), normalized_rank1=rank_fraction(a, True),
                proximal_bend_fraction=float(bend_square[:7].sum()/bend_square.sum())))

profiles = pd.DataFrame(rows)
profiles.to_csv(OUT/'spatial_components.csv',index=False)
pair_table = pd.DataFrame(pairs)
pair_table.to_csv(OUT/'opposing_channel_similarity.csv',index=False)
random_table = pd.DataFrame(random_rows)
random_table.to_csv(OUT/'random_direction_structure.csv',index=False)
gain_profiles = np.array(gains).reshape(20,4,14)

localization=[]
for si, seed in enumerate(SEEDS):
    for c in range(4):
        energy = np.sum(G[si,c,:14]**2,axis=-1)
        localization.append(dict(seed=int(seed),channel=c,
            proximal_bend_fraction=float(energy[:7].sum()/energy.sum()),
            distal_bend_fraction=float(energy[7:].sum()/energy.sum())))
localization=pd.DataFrame(localization)
localization.to_csv(OUT/'local_bend_distribution.csv',index=False)

# Project the time readout to rank 1 or 2 using all six training reference J's.
# Pole values, path branch, reference model and nonlinear decoder stay fixed.
compressed=np.empty((2,)+wh.shape)
projection_rows=[]
single_exp_rows=[]
tau_grid=np.geomspace(.02,20.,1000)
exp_grid=np.exp(-(K['lag_seconds'][:,None]+.2)/tau_grid[None])
exp_grid/=np.linalg.norm(exp_grid,axis=0,keepdims=True)
for si, seed in enumerate(SEEDS):
    left,sval,right=np.linalg.svd(phi[si],full_matrices=False)
    a=J[si].reshape(-1,16)
    for c in range(4):
        transformed=wh[si,c].T @ (left*sval[None])
        _,_,vt=np.linalg.svd(a @ transformed,full_matrices=False)
        response=a@wh[si,c].T@phi[si]
        energy=np.sum(response**2)
        scores=np.sum((response@exp_grid)**2,axis=0)/energy
        best=int(np.argmax(scores))
        single_exp_rows.append(dict(seed=int(seed),channel=c,
            best_tau_s=float(tau_grid[best]),
            single_exponential_energy_retained=float(scores[best]),
            arbitrary_rank1_energy_retained=rank_fraction(response)))
        for rank in (1,2):
            proj=vt[:rank].T@vt[:rank]
            reconstructed=transformed@proj@np.diag(1/sval)@left.T
            compressed[rank-1,si,c]=reconstructed.T
            old=a@wh[si,c].T@phi[si]
            new=a@reconstructed@phi[si]
            projection_rows.append(dict(seed=int(seed),channel=c,rank=rank,
                reference_kernel_energy_retained=float(1-np.sum((old-new)**2)/np.sum(old**2))))
pd.DataFrame(projection_rows).to_csv(OUT/'projection_reference_error.csv',index=False)
single_exp_table=pd.DataFrame(single_exp_rows)
single_exp_table.to_csv(OUT/'single_exponential_approximation.csv',index=False)
target=np.load(SOURCE/'test_inputs_targets.npz')['target_xyz_mm'][...,:2]
evaluation=[]
max_decode_diff=0.
for si,seed in enumerate(SEEDS):
    d=np.load(SOURCE/f'seed_{seed}_test_geometry.npz')
    for rank in (6,1,2):
        w=wh[si] if rank==6 else compressed[rank-1,si]
        memory=d['memory_path']+np.einsum('bck,ckg->bg',d['d'],w)
        pred=decode(si,d['reference_bend']+memory[:,:14],
                    d['reference_log_length']+memory[:,14:])
        if rank==6:
            max_decode_diff=max(max_decode_diff,float(abs(pred-d['full_xyz_mm'][...,:2]).max()))
        error=np.linalg.norm(pred-target,axis=-1)
        distance=np.linalg.norm(pred-d['full_xyz_mm'][...,:2],axis=-1)
        evaluation.append(dict(seed=int(seed),rank=rank,frames=len(target),
            mean_node_mm=float(error.mean()),endpoint_mm=float(error[:,-1].mean()),
            mean_distance_to_original_mm=float(distance.mean())))
qa['max_reconstructed_full_prediction_difference_mm']=max_decode_diff
assert max_decode_diff<2e-4
eval_table=pd.DataFrame(evaluation)
eval_table.to_csv(OUT/'frozen_projection_test.csv',index=False)

summary={'seeds':SEEDS.tolist(),'references':6,'test_frames':len(target),
         'basis_rank1':rank_fraction(phi[0]),'basis_condition_number':float(np.linalg.cond(phi[0])),
         'channel_localization':{},'opposing_channels':{},'spatial_reversal':{},
         'projection_test':{},'random_structural_reference':{},'single_exponential_approximation':{}}
for c in range(4):
    z=localization[localization.channel==c]
    summary['channel_localization'][str(c)]={
        key:stats(z[key]) for key in ['proximal_bend_fraction','distal_bend_fraction']}
    z=single_exp_table[single_exp_table.channel==c]
    summary['single_exponential_approximation'][str(c)]={key:stats(z[key]) for key in
        ['best_tau_s','single_exponential_energy_retained','arbitrary_rank1_energy_retained']}
    z=profiles[(profiles.reference==0)&(profiles.channel==c)]
    g7=z[z.node==7].dominant_gain.to_numpy();g14=z[z.node==14].dominant_gain.to_numpy()
    summary['spatial_reversal'][str(c)]=dict(
        reference0_n7=stats(g7),reference0_n14=stats(g14),
        reversal_count_reference0=int(np.sum(g7*g14<0)),
        tip_peak_lag_s_reference0=stats(z[z.node==14].peak_lag_s))
    for ni in (7,14):
        subset=z[z.node==ni]
        summary['spatial_reversal'][str(c)][f'node{ni}_components']={
            key:stats(subset[key]) for key in ['lag0','proximal_bend_lag0','distal_bend_lag0','length_lag0','cancellation_fraction']}
for pair,z in pair_table.groupby('channel_pair'):
    summary['opposing_channels'][pair]={
        'reference0':stats(z[z.reference==0].lateral_kernel_cosine),
        'seed_means_all_references':stats(z.groupby('seed').lateral_kernel_cosine.mean())}
for rank,z in eval_table.groupby('rank'):
    summary['projection_test'][str(rank)]={key:stats(z[key]) for key in ['mean_node_mm','endpoint_mm','mean_distance_to_original_mm']}
    original=eval_table[eval_table['rank']==6].set_index('seed')
    summary['projection_test'][str(rank)]['node_change_vs_original']=stats(z.set_index('seed').mean_node_mm-original.mean_node_mm)
for field in ['rank1','normalized_rank1','proximal_bend_fraction']:
    vals=random_table[field].to_numpy()
    summary['random_structural_reference'][field]=dict(mean=float(vals.mean()),
        q05=float(np.quantile(vals,.05)),q95=float(np.quantile(vals,.95)))
summary['relative_between_seed_dispersion']={}
for name, values in [('time_readout', wh), ('generalized_time_kernel', G),
                     ('position_time_kernel', kernel)]:
    mean=values.mean(0)
    summary['relative_between_seed_dispersion'][name]=float(
        np.sqrt(np.mean(np.sum((values-mean).reshape(20,-1)**2,axis=1)))/np.linalg.norm(mean))
max_fd=0.
for si in range(20):
    for ri in [0,3,5]:
        for c in range(4):
            v=G[si,c,:,0];eps=1e-6
            b=K['reference_bend'][si,ri][None];l=K['reference_log_length'][si,ri][None]
            plus=decode(si,b+eps*v[:14],l+eps*v[14:])
            minus=decode(si,b-eps*v[:14],l-eps*v[14:])
            fd=(plus-minus)[0,1:]/(2*eps)
            max_fd=max(max_fd,float(abs(fd-kernel[si,ri,c,:,0]).max()))
qa['max_kernel_vs_nonlinear_integration_finite_difference']=max_fd
assert max_fd<1e-6
summary['qa']=qa
summary['interpretation_limits']=[
    'Seeds are repeated optimization on one dataset, not independent physical experiments.',
    'Kernels are the time-memory term per transformed pressure increment at fixed reference geometry.',
    'Local bend fractions are squared-response concentration, not mechanical energy.',
    'Rank projection uses only six already selected training-input reference geometries; both fixed ranks are reported.',
    'Rank-1 readout still uses six exponential poles; it does not establish a single first-order physical mode.',
    'Random directions describe structural context, not a formal inferential null.',
    'Physical section association follows user-described recordings; proximal nodes1..7, distal8..14.',
]
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

BLUE='#2767A5';ORANGE='#CD742D';GRAY='#7F8790';INK='#25313D'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'svg.fonttype':'none',
    'savefig.facecolor':'white','figure.facecolor':'white','savefig.dpi':170})
fig,axes=plt.subplots(2,2,figsize=(13,8.8))
fig.subplots_adjust(top=.85,bottom=.12,hspace=.5,wspace=.27)
fig.suptitle('Time-memory response and robot geometry',x=.075,ha='left',fontsize=18,fontweight='bold',y=.97)
fig.text(.075,.925,'20 frozen fits | four channels | mean +/- sample SD | fixed training-mean reference',fontsize=11)
x=np.arange(4);prox=localization.groupby('channel').proximal_bend_fraction.agg(['mean','std'])
axes[0,0].bar(x,prox['mean']*100,color=BLUE,label='Proximal bends')
axes[0,0].bar(x,100-prox['mean']*100,bottom=prox['mean']*100,color=ORANGE,label='Distal bends')
axes[0,0].errorbar(x,prox['mean']*100,yerr=prox['std']*100,fmt='none',color=INK,capsize=3)
axes[0,0].set(xticks=x,xticklabels=['0 (distal)','1 (distal)','2 (proximal)','3 (proximal)'],
    ylim=(0,108),ylabel='Squared bend-kernel response (%)',title='(a) Location of learned bending response')
axes[0,0].legend(frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2)
for c,color,style in [(0,BLUE,'-'),(1,ORANGE,'--')]:
    a=gain_profiles[:,c];mean=a.mean(0);sd=a.std(0,ddof=1)
    axes[0,1].plot(np.arange(1,15),mean,style,color=color,label=f'Channel {c}',lw=2)
    axes[0,1].fill_between(np.arange(1,15),mean-sd,mean+sd,color=color,alpha=.15)
axes[0,1].axhline(0,color=GRAY,lw=.8);axes[0,1].axvline(7.5,color=GRAY,ls=':',lw=.8)
axes[0,1].set(xlabel='Node, base to tip',ylabel='SVD gain (mm / unit drive increment)',title='(b) Gain reversal along the arm',xticks=[1,4,7,10,14])
axes[0,1].legend(frameon=False)
for part,color,style,label in [(0,BLUE,'--','Proximal bending'),(1,ORANGE,':','Distal bending'),(2,GRAY,'-.','Length'),(-1,INK,'-','Total')]:
    a=components[:,0,0,:,:,0,part][:,:,0] if part>=0 else kernel[:,0,0,:,0,0]
    axes[1,0].plot(np.arange(1,15),a.mean(0),style,color=color,label=label,lw=2 if part==-1 else 1.5)
axes[1,0].axhline(0,color=GRAY,lw=.8);axes[1,0].axvline(7.5,color=GRAY,ls=':',lw=.8)
axes[1,0].set(xlabel='Node, base to tip',ylabel='Lateral kernel (mm / unit increment)',title='(c) Channel 0 at zero increment lag',xticks=[1,4,7,10,14])
axes[1,0].legend(frameon=False,fontsize=9)
for c,color,style in [(2,BLUE,'-'),(3,ORANGE,'--')]:
    a=kernel[:,0,c,-1,:,0];m=a.mean(0);sd=a.std(0,ddof=1)
    axes[1,1].plot(K['lag_seconds'],m,style,color=color,lw=2,label=f'Channel {c}')
    axes[1,1].fill_between(K['lag_seconds'],m-sd,m+sd,color=color,alpha=.15)
axes[1,1].axhline(0,color=GRAY,lw=.8)
axes[1,1].set(xlabel='Increment lag (s)',ylabel='Tip kernel (mm / unit increment)',title='(d) Time of maximum tip response')
axes[1,1].legend(frameon=False)
fig.text(.075,.035,'Local time-memory corrections; panel (a) is response concentration, not mechanical energy.\nSource: saved unified20 kernels; these curves are model responses, not independently measured step responses.',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'geometry_physics.{ext}')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,4.8));fig.subplots_adjust(top=.78,bottom=.22,wspace=.33)
fig.suptitle('Response structure and frozen readout approximation',x=.075,ha='left',fontsize=17,fontweight='bold',y=.98)
fig.text(.075,.9,'20 frozen models; ranks 1 and 2 fixed before test evaluation; no retraining',fontsize=11)
learned=pd.read_csv(SOURCE/'kernel_rank.csv')
z=learned[(learned.reference_index==0)&(learned.coordinate=='x')]
for offset,field,color,label in [(-.16,'rank1',BLUE,'Raw kernel'),(.16,'normalized_rank1',ORANGE,'Node-normalized')]:
    col='rank1_energy' if field=='rank1' else 'normalized_rank1_energy'
    grouped=z.groupby('channel')[col].agg(['mean','std'])
    axes[0].errorbar(x+offset,grouped['mean']*100,yerr=grouped['std']*100,fmt='o',color=color,capsize=3,label=label+' (learned)')
    for c in range(4):
        a=random_table[random_table.channel==c][field]*100
        lo,med,hi=np.quantile(a,[.05,.5,.95])
        axes[0].plot([c+offset]*2,[lo,hi],color=color,alpha=.3,lw=5)
        axes[0].plot(c+offset,med,'_',color=color,ms=13)
axes[0].set(xticks=x,xlabel='Channel',ylabel='First singular component (%)',ylim=(30,102),title='(a) Learned and random-direction kernels')
axes[0].legend(frameon=False,fontsize=9,loc='lower left')
for ix,rank in enumerate((6,1,2)):
    a=eval_table[eval_table['rank']==rank].mean_node_mm.to_numpy()
    axes[1].scatter(ix+np.linspace(-.065,.065,len(a)),a,s=16,color=GRAY,alpha=.55)
    axes[1].errorbar(ix,a.mean(),yerr=a.std(ddof=1),fmt='o',color=BLUE,capsize=4)
    axes[1].annotate(f'{a.mean():.3f} +/- {a.std(ddof=1):.3f}',(ix,a.mean()),xytext=(0,16),textcoords='offset points',ha='center',fontsize=9)
axes[1].set(xticks=[0,1,2],xticklabels=['Original','Rank 1','Rank 2'],ylabel='Test mean node error (mm)',title='(b) Six-reference geometry-weighted projection')
axes[1].margins(x=.18,y=.35)
fig.text(.075,.045,'(a) Faint bars: 5th-95th percentiles of 100 random directions per model/channel; not confidence intervals.\n(b) All six temporal poles remain. Only the spatial readout is projected; test data are used solely for evaluation.',fontsize=9)
for ext in ['png','svg']:fig.savefig(OUT/f'response_structure.{ext}')
plt.close(fig)
print(json.dumps({'localization':summary['channel_localization'],
    'projection_test':summary['projection_test'],'qa':qa},ensure_ascii=False))
