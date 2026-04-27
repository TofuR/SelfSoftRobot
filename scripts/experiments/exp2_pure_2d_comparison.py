#!/usr/bin/env python3
"""
方向2: 纯 2D 自建模对比 — 3D GT 监督 vs 纯 2D 渲染监督

训练两个模型:
  Model A: 使用 3D GT 骨架 loss (标准 MS-SCNF Phase 1)
  Model B: 仅使用 2D 渲染 loss + 物理先验
对比骨架预测精度，量化 3D GT 的贡献。

Usage:
    python scripts/experiments/exp2_pure_2d_comparison.py --gpu 0 --epochs 40
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='data/seq_rr_3d')
parser.add_argument('--output_dir', type=str, default='output/exp2_pure_2d')
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_render_samples', type=int, default=32)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向2: 3D GT vs 纯 2D 对比 ===')
print(f'Device: {device}, Epochs: {args.epochs}')

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

from src.data.dataset import SoftSequenceDataset
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.models.model_ms_scnf import MSSCNFModel

ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
H, W = ds.H, ds.W
focal = ds.focal if hasattr(ds, 'focal') and ds.focal > 0 else 136.42
near, far = 0.5, 2.5
n_samples = args.n_render_samples

cam_params = ds.get_camera_params()
eye = tuple(cam_params['eye']) if cam_params else (1.5, 0.0, 0.5)
center = tuple(cam_params['center']) if cam_params else (0.0, 0.0, 0.25)
up = tuple(cam_params['up']) if cam_params else (0.0, 0.0, 1.0)
rays_o, rays_d = get_rays(H, W, focal, eye, center, up, device=device)

n_total = len(ds)
n_train = int(0.8 * n_total)
train_ds = torch.utils.data.Subset(ds, range(n_train))
test_ds = torch.utils.data.Subset(ds, range(n_train, n_total))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

print(f'数据: 训练={n_train}, 测试={n_total - n_train}, Focal={focal:.1f}')

# ═══════════════════════════════════════════════════════════════
# 公共函数
# ═══════════════════════════════════════════════════════════════

def make_model():
    return MSSCNFModel(
        action_dim=ds.action_dim, window_size=20, n_scales=4,
        hidden_dim=128, d_filter=128, n_freqs=10,
        n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6,
    ).to(device)

def skeleton_smoothness(skel):
    d2 = skel[:, 2:] - 2 * skel[:, 1:-1] + skel[:, :-2]
    return (d2 ** 2).sum(-1).mean()

def length_preservation(skel, rest=0.5):
    segs = skel[:, 1:] - skel[:, :-1]
    total = segs.norm(dim=-1).sum(dim=-1)
    return ((total - rest) ** 2).mean()

def render_batch(model, aw, chunk_size=2048):
    B = aw.shape[0]
    pts, _ = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True)
    parts = []
    for s in range(0, H * W, chunk_size):
        e = min(s + chunk_size, H * W)
        raw = model(pts[s:e], aw).reshape(B, e - s, n_samples, 2)
        parts.append(torch.stack([OM_rendering(raw[b])[0] for b in range(B)]))
    return torch.cat(parts, dim=1)

def evaluate_skeleton(model, loader):
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            aw = batch[0].to(device)
            gt = batch[-1].permute(0, 2, 1).to(device)
            pred = model.predict_skeleton(aw)['fine']
            preds.append(pred.cpu().numpy())
            gts.append(gt.cpu().numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    errors = np.linalg.norm(preds - gts, axis=-1)
    return {
        'mne': errors.mean(),
        'tip': errors[:, -1].mean(),
        'max': errors.max(),
        'per_axis': {
            'x': np.abs(preds[:,:,0] - gts[:,:,0]).mean(),
            'y': np.abs(preds[:,:,1] - gts[:,:,1]).mean(),
            'z': np.abs(preds[:,:,2] - gts[:,:,2]).mean(),
        },
        'preds': preds,
        'gts': gts,
    }

# ═══════════════════════════════════════════════════════════════
# Model A: 3D GT 骨架监督 (标准 MS-SCNF Phase 1)
# ═══════════════════════════════════════════════════════════════

print('\n=== Model A: 3D GT 骨架监督 ===')
model_a = make_model()
opt_a = torch.optim.Adam(model_a.parameters(), lr=args.lr)
sched_a = torch.optim.lr_scheduler.CosineAnnealingLR(opt_a, T_max=args.epochs)

hist_a = {'loss': [], 'mne': []}

for epoch in range(args.epochs):
    model_a.train()
    total_loss = 0
    n_b = 0
    for batch in train_loader:
        aw = batch[0].to(device)
        gt_pos = batch[-1].permute(0, 2, 1).to(device)  # (B, 31, 3)

        pred_dict = model_a.predict_skeleton(aw)
        loss_skel = model_a.compute_skeleton_loss(pred_dict, gt_pos)

        # 平滑正则化
        skel_fine = pred_dict['fine']
        loss_reg = 0.01 * skeleton_smoothness(skel_fine)

        loss = loss_skel + loss_reg
        opt_a.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        opt_a.step()
        total_loss += loss.item()
        n_b += 1
    sched_a.step()

    avg_loss = total_loss / n_b
    hist_a['loss'].append(avg_loss)

    if (epoch + 1) % 10 == 0:
        metrics = evaluate_skeleton(model_a, test_loader)
        hist_a['mne'].append((epoch, metrics['mne']))
        print(f'  Epoch {epoch+1}: loss={avg_loss:.6f}, MNE={metrics["mne"]:.6f}m')
    else:
        print(f'  Epoch {epoch+1}: loss={avg_loss:.6f}')

metrics_a = evaluate_skeleton(model_a, test_loader)
print(f'  最终: MNE={metrics_a["mne"]:.6f}m, Tip={metrics_a["tip"]:.6f}m')

# ═══════════════════════════════════════════════════════════════
# Model B: 纯 2D 渲染监督
# ═══════════════════════════════════════════════════════════════

print('\n=== Model B: 纯 2D 渲染监督 ===')
model_b = make_model()
opt_b = torch.optim.Adam(model_b.parameters(), lr=args.lr)
sched_b = torch.optim.lr_scheduler.CosineAnnealingLR(opt_b, T_max=args.epochs)

hist_b = {'loss': [], 'mne': []}

for epoch in range(args.epochs):
    model_b.train()
    total_loss = 0
    n_b = 0
    for batch in train_loader:
        aw = batch[0].to(device)
        gt_img = batch[1].to(device)

        # 渲染 loss
        rendered = render_batch(model_b, aw)
        loss_recon = F.mse_loss(rendered, gt_img)

        # 骨架正则化
        physics_state = model_b.encode(aw)
        skel_dict = model_b.skeleton_head(physics_state)
        skel = skel_dict['fine']

        loss_reg = (0.1 * skeleton_smoothness(skel)
                    + 0.05 * length_preservation(skel))

        loss = loss_recon + loss_reg
        opt_b.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()
        total_loss += loss.item()
        n_b += 1
    sched_b.step()

    avg_loss = total_loss / n_b
    hist_b['loss'].append(avg_loss)

    if (epoch + 1) % 10 == 0:
        metrics = evaluate_skeleton(model_b, test_loader)
        hist_b['mne'].append((epoch, metrics['mne']))
        print(f'  Epoch {epoch+1}: loss={avg_loss:.6f}, MNE={metrics["mne"]:.6f}m')
    else:
        print(f'  Epoch {epoch+1}: loss={avg_loss:.6f}')

metrics_b = evaluate_skeleton(model_b, test_loader)
print(f'  最终: MNE={metrics_b["mne"]:.6f}m, Tip={metrics_b["tip"]:.6f}m')

# ═══════════════════════════════════════════════════════════════
# 对比可视化
# ═══════════════════════════════════════════════════════════════

print('\n--- 对比可视化 ---')

# 1. 训练曲线对比
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist_a['loss'], label='Model A: 3D GT Loss')
axes[0].plot(hist_b['loss'], label='Model B: 2D Render Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# MNE 对比
if hist_a['mne'] and hist_b['mne']:
    e_a, m_a = zip(*hist_a['mne'])
    e_b, m_b = zip(*hist_b['mne'])
    axes[1].plot(e_a, m_a, 'b-o', markersize=3, label='3D GT Supervised')
    axes[1].plot(e_b, m_b, 'r-o', markersize=3, label='2D Rendering Only')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MNE (m)')
axes[1].set_title('Skeleton Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# 指标柱状图
labels = ['MNE', 'Tip Error', 'X Error', 'Y Error', 'Z Error']
vals_a = [metrics_a['mne'], metrics_a['tip'],
          metrics_a['per_axis']['x'], metrics_a['per_axis']['y'], metrics_a['per_axis']['z']]
vals_b = [metrics_b['mne'], metrics_b['tip'],
          metrics_b['per_axis']['x'], metrics_b['per_axis']['y'], metrics_b['per_axis']['z']]
x = np.arange(len(labels))
axes[2].bar(x - 0.15, vals_a, 0.3, label='3D GT', color='steelblue', alpha=0.8)
axes[2].bar(x + 0.15, vals_b, 0.3, label='2D Only', color='salmon', alpha=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=15)
axes[2].set_ylabel('Error (m)'); axes[2].set_title('Metrics Comparison')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('3D GT Supervision vs Pure 2D Rendering', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'comparison.png'), dpi=150)
plt.close()

# 2. 骨架对比
n_show = min(4, len(metrics_a['preds']))
fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 10), subplot_kw={'projection': '3d'})
for i in range(n_show):
    gt = metrics_a['gts'][i]
    # Model A
    ax = axes[0, i] if n_show > 1 else axes[0]
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'b-o', lw=3, ms=4, label='GT')
    ax.plot(metrics_a['preds'][i, :, 0], metrics_a['preds'][i, :, 1], metrics_a['preds'][i, :, 2],
            'r-o', lw=2, ms=3, label='Pred')
    mne_a = np.linalg.norm(metrics_a['preds'][i] - gt, axis=1).mean()
    ax.set_title(f'A (3D GT): {mne_a:.4f}m')
    ax.set_xlim(-0.1, 0.3); ax.set_ylim(-0.15, 0.15); ax.set_zlim(0, 0.55)
    if i == 0: ax.legend(fontsize=8)

    # Model B
    ax = axes[1, i] if n_show > 1 else axes[1]
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], 'b-o', lw=3, ms=4, label='GT')
    ax.plot(metrics_b['preds'][i, :, 0], metrics_b['preds'][i, :, 1], metrics_b['preds'][i, :, 2],
            'r-o', lw=2, ms=3, label='Pred')
    mne_b = np.linalg.norm(metrics_b['preds'][i] - gt, axis=1).mean()
    ax.set_title(f'B (2D Only): {mne_b:.4f}m')
    ax.set_xlim(-0.1, 0.3); ax.set_ylim(-0.15, 0.15); ax.set_zlim(0, 0.55)
    if i == 0: ax.legend(fontsize=8)

plt.suptitle('Skeleton: GT (blue) vs Pred (red)\nTop: 3D GT Supervised | Bottom: 2D Only', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'skeleton_comparison.png'), dpi=150)
plt.close()

# 3. 逐轴误差分布
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, dim, label in zip(axes, range(3), ['X', 'Y', 'Z']):
    err_a = (metrics_a['preds'][:,:,dim] - metrics_a['gts'][:,:,dim]).flatten()
    err_b = (metrics_b['preds'][:,:,dim] - metrics_b['gts'][:,:,dim]).flatten()
    ax.hist(err_a, bins=50, alpha=0.5, label=f'3D GT (std={err_a.std():.4f})', color='steelblue')
    ax.hist(err_b, bins=50, alpha=0.5, label=f'2D Only (std={err_b.std():.4f})', color='salmon')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel(f'{label} Error (m)'); ax.set_ylabel('Count')
    ax.set_title(f'{label}-axis Error'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle('Per-Axis Error Distribution', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'axis_errors.png'), dpi=150)
plt.close()

# 保存摘要
with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== 方向2: 3D GT vs 纯 2D 对比 ===\n\n')
    f.write('Model A (3D GT Supervised):\n')
    f.write(f'  MNE:   {metrics_a["mne"]:.6f} m\n')
    f.write(f'  Tip:   {metrics_a["tip"]:.6f} m\n')
    f.write(f'  X err: {metrics_a["per_axis"]["x"]:.6f} m\n')
    f.write(f'  Y err: {metrics_a["per_axis"]["y"]:.6f} m\n')
    f.write(f'  Z err: {metrics_a["per_axis"]["z"]:.6f} m\n\n')
    f.write('Model B (2D Rendering Only):\n')
    f.write(f'  MNE:   {metrics_b["mne"]:.6f} m\n')
    f.write(f'  Tip:   {metrics_b["tip"]:.6f} m\n')
    f.write(f'  X err: {metrics_b["per_axis"]["x"]:.6f} m\n')
    f.write(f'  Y err: {metrics_b["per_axis"]["y"]:.6f} m\n')
    f.write(f'  Z err: {metrics_b["per_axis"]["z"]:.6f} m\n\n')
    ratio = metrics_b['mne'] / max(metrics_a['mne'], 1e-8)
    f.write(f'2D/3D MNE ratio: {ratio:.2f}x\n')
    if ratio > 2:
        f.write('结论: 3D GT 监督对骨架精度至关重要\n')
    elif ratio > 1.2:
        f.write('结论: 2D 监督可以学到合理的骨架，但精度不如 3D GT\n')
    else:
        f.write('结论: 2D 监督已接近 3D GT 效果\n')

print(f'\n=== 完成 ===')
print(f'结果: {args.output_dir}/')
