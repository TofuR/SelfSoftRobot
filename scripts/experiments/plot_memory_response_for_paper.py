#!/usr/bin/env python3
"""Paper figures: additive geometry and a spatial/temporal kernel factorization.

Static chart contract: 14 nodes, 19 lags, all 20 frozen models. Figure 3 uses
two line panels for contributions and tip waveforms. Figure 4 combines a
signed heatmap, two factor curves, and reconstructed node curves; the SVD
of the mean kernel illustrates the factorization, while manuscript summary
statistics are computed separately for each model. Use blue/orange and
neutrals, explicit zero lines, and open markers for approximation.
"""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/selfsr-memory-response-paper-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'workspace/reports/memory_physics_20260914'
OUT = ROOT / 'docs/icra2027/figures/memory_response'
OUT.mkdir(parents=True, exist_ok=True)
profiles = pd.read_csv(SOURCE / 'spatial_components.csv')
data = np.load(ROOT / 'workspace/runs/analysis/modeling_unified20_20260913_005/geometry/kernels.npz')
assert data['seeds'].tolist() == list(range(100, 120))
assert len(profiles) == 6720
BLUE = '#2767A5'
ORANGE = '#CD742D'
GRAY = '#7F8790'
INK = '#25313D'
WHITE = '#FFFFFF'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.titlelocation': 'left', 'axes.titlepad': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': GRAY, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK, 'ytick.color': INK, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'savefig.dpi': 200, 'savefig.facecolor': WHITE,
    'figure.facecolor': WHITE,
})


def save(fig, name):
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}')
    plt.close(fig)


lag = data['lag_seconds']
node = data['node_ids']
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
fig.subplots_adjust(top=.87, bottom=.16, left=.075, right=.98, wspace=.28)
ref = profiles[(profiles.reference == 0) & (profiles.channel == 0)]
for column, color, style, label in [
    ('proximal_bend_lag0', BLUE, '--', 'Proximal bending'),
    ('distal_bend_lag0', ORANGE, ':', 'Distal bending'),
    ('length_lag0', GRAY, '-.', 'Length'),
    ('lag0', INK, '-', 'Total'),
]:
    curve = ref.groupby('node')[column].mean()
    axes[0].plot(curve.index, curve.values, style, color=color, label=label,
                 lw=2 if label == 'Total' else 1.7)
axes[0].axvline(7, color=GRAY, ls=':', lw=.8)
axes[0].set(xlabel='Node index (base to tip)',
            ylabel=r'Kernel (mm / unit $\Delta e_0$)',
            title='(a) Geometric contributions, channel 0', xticks=[1, 4, 7, 10, 14])
axes[0].legend(frameon=False, fontsize=9, loc='upper left')
for channel, color, style in [(2, BLUE, '-'), (3, ORANGE, '--')]:
    array = data['kernel_xy_mm_per_delta_e'][:, 0, channel, -1, :, 0]
    mean = array.mean(0)
    sd = array.std(0, ddof=1)
    axes[1].plot(lag, mean, style, color=color, lw=2, label=f'Channel {channel}')
    axes[1].fill_between(lag, mean-sd, mean+sd, color=color, alpha=.15)
axes[1].set(xlabel='Increment lag (s)',
            ylabel=r'Tip kernel (mm / unit $\Delta e_c$)',
            title='(b) Temporal response at the tip')
axes[1].legend(frameon=False)
for ax in axes:
    ax.axhline(0, color=GRAY, lw=.8)
    ax.tick_params(labelsize=9)
save(fig, 'geometry_response')

# Illustration uses the mean kernel; no individual seed is selected.
matrix = data['kernel_xy_mm_per_delta_e'][:, 0, 0, :, :, 0].mean(0)
u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
sign = np.sign(vt[0, np.argmax(abs(vt[0]))])
wave = vt[0] * sign
gain = u[:, 0] * singular[0] * sign
approximation = np.outer(gain, wave)
residual = matrix - approximation
retained = 1 - np.sum(residual**2) / np.sum(matrix**2)
assert np.isclose(np.linalg.norm(wave), 1)
assert np.allclose(retained, singular[0]**2 / np.sum(singular**2))
assert np.max(abs(matrix - (approximation + residual))) < 1e-12

fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6))
fig.subplots_adjust(top=.85, bottom=.09, left=.085, right=.98,
                    hspace=.48, wspace=.30)
fig.suptitle(r'$\mathbf{K}_0^x \approx \mathbf{g}_0\boldsymbol{\psi}_0^{\mathsf{T}}$'
             '  |  Spatial and temporal factors',
             x=.085, y=.985, ha='left', fontsize=16)
fig.text(.085, .931,
         'Channel 0; training-mean reference; decomposition of the mean kernel over 20 models',
         fontsize=10)

diverging = LinearSegmentedColormap.from_list('signed_kernel', [BLUE, WHITE, ORANGE])
limit = float(abs(matrix).max())
heat = axes[0, 0].imshow(matrix, origin='lower', aspect='auto',
                         extent=[lag[0]-.1, lag[-1]+.1, .5, 14.5],
                         cmap=diverging, vmin=-limit, vmax=limit, interpolation='nearest')
axes[0, 0].set(xlabel='Increment lag (s)', ylabel='Node index (base to tip)',
               yticks=[1, 4, 7, 10, 14], title=r'(a) Node–lag kernel $\mathbf{K}_0^x$')
bar = fig.colorbar(heat, ax=axes[0, 0], fraction=.046, pad=.035)
bar.set_label(r'mm / unit $\Delta e_0$', fontsize=9)
bar.ax.tick_params(labelsize=8)

axes[0, 1].plot(node, gain, color=BLUE, lw=2, marker='o', markersize=3.8)
axes[0, 1].axvline(7, color=GRAY, ls=':', lw=.8)
axes[0, 1].axhline(0, color=GRAY, lw=.8)
axes[0, 1].set(xlabel='Node index (base to tip)',
               ylabel=r'Gain (mm / unit $\Delta e_0$)',
               xticks=[1, 4, 7, 10, 14], title=r'(b) Spatial gains $\mathbf{g}_0$')

axes[1, 0].plot(lag, wave, color=BLUE, marker='o', markersize=3.4, lw=2)
axes[1, 0].axhline(0, color=GRAY, lw=.8)
axes[1, 0].set(xlabel='Increment lag (s)', ylabel='Weight (unit norm)',
               title=r'(c) Shared time waveform $\boldsymbol{\psi}_0$')

handles = []
for n, color in [(7, BLUE), (10, GRAY), (14, ORANGE)]:
    axes[1, 1].plot(lag, matrix[n-1], color=color, lw=1.8)
    axes[1, 1].plot(lag[::2], approximation[n-1, ::2], 'o', color=color,
                    markerfacecolor=WHITE, markersize=4, markeredgewidth=1.1)
    handles.append(Line2D([0], [0], color=color, lw=1.8, label=f'Node {n}'))
handles += [Line2D([0], [0], color=INK, lw=1.8, label='Original kernel'),
            Line2D([0], [0], color=INK, marker='o', markerfacecolor=WHITE,
                   linestyle='none', markersize=4, label=r'$g_{n,0}\psi_0(\ell)$')]
axes[1, 1].legend(handles=handles, frameon=False, fontsize=8.5, loc='upper right')
axes[1, 1].axhline(0, color=GRAY, lw=.8)
axes[1, 1].set(xlabel='Increment lag (s)',
               ylabel=r'Kernel (mm / unit $\Delta e_0$)',
               title='(d) Node response reconstruction')
for ax in axes.flat:
    ax.tick_params(labelsize=9)
save(fig, 'kernel_factorization')

np.savez(OUT / 'factorization_display.npz', nodes=node, lag_seconds=lag,
         mean_kernel=matrix, gain=gain, wave=wave, approximation=approximation,
         residual=residual)
(OUT / 'factorization_display.json').write_text(json.dumps({
    'source': 'workspace/runs/analysis/modeling_unified20_20260913_005/geometry/kernels.npz',
    'channel': 0, 'reference_index': 0, 'seeds': data['seeds'].tolist(),
    'aggregation': 'SVD of the mean model kernel, for figure illustration',
    'rank1_retained_squared_response': float(retained),
    'reconstruction_max_abs_residual': float(abs(residual).max()),
    'units': 'mm per unit transformed-drive increment',
}, indent=2) + '\n')
print(OUT / 'geometry_response.svg')
print(OUT / 'kernel_factorization.svg')
