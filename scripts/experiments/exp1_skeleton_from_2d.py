#!/usr/bin/env python3
"""
方向1: 形态发现 — 从 2D 渲染 loss 学习骨架 (无 3D GT)

用 MS-SCNF 架构，但仅使用 2D 渲染 MSE loss + 物理先验正则化训练骨架。
训练后与 GT 3D 骨架对比，量化纯 2D 监督的能力边界。

Usage:
    python scripts/experiments/exp1_skeleton_from_2d.py --gpu 0 --epochs 50
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
parser.add_argument('--output_dir', type=str, default='output/exp1_skeleton_2d')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_samples_render', type=int, default=32)
parser.add_argument('--lambda_smooth', type=float, default=0.1)
parser.add_argument('--lambda_length', type=float, default=0.05)
parser.add_argument('--lambda_gravity', type=float, default=0.02)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向1: 从 2D 渲染学习骨架 ===')
print(f'Device: {device}, Epochs: {args.epochs}')

# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

from src.data.dataset import SoftSequenceDataset
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.models.model_ms_scnf import MSSCNFModel

ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
print(f'数据集: {len(ds)} 样本, action_dim={ds.action_dim}, H={ds.H}, W={ds.W}')

H, W = ds.H, ds.W
focal = ds.focal if hasattr(ds, 'focal') and ds.focal > 0 else 136.42
near, far = 0.5, 2.5
n_samples = args.n_samples_render

# 相机射线（只计算一次）
cam_params = ds.get_camera_params()
eye = tuple(cam_params['eye']) if cam_params else (1.5, 0.0, 0.5)
center = tuple(cam_params['center']) if cam_params else (0.0, 0.0, 0.25)
up = tuple(cam_params['up']) if cam_params else (0.0, 0.0, 1.0)

rays_o, rays_d = get_rays(H, W, focal, eye, center, up, device=device)
print(f'射线: {rays_o.shape}, Focal: {focal:.1f}')

# 数据分割
n_total = len(ds)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
train_ds = torch.utils.data.Subset(ds, range(n_train))
test_ds = torch.utils.data.Subset(ds, range(n_train, n_total))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

print(f'训练: {n_train}, 测试: {n_test}')

# ═══════════════════════════════════════════════════════════════
# 模型创建
# ═══════════════════════════════════════════════════════════════

model = MSSCNFModel(
    action_dim=ds.action_dim,
    window_size=20,
    n_scales=4,
    hidden_dim=128,
    d_filter=128,
    n_freqs=10,
    n_coarse=4,
    n_medium=10,
    n_fine=31,
    deform_n_freqs=6,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'模型参数: {n_params:,}')

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ═══════════════════════════════════════════════════════════════
# 正则化函数
# ═══════════════════════════════════════════════════════════════

def skeleton_smoothness(skeleton):
    """骨架二阶差分 (曲率正则化)。 skeleton: (B, N, 3)"""
    diff2 = skeleton[:, 2:] - 2 * skeleton[:, 1:-1] + skeleton[:, :-2]
    return (diff2 ** 2).sum(-1).mean()

def length_preservation(skeleton, rest_length=0.5):
    """骨架总长度约束。 skeleton: (B, N, 3)"""
    segs = skeleton[:, 1:] - skeleton[:, :-1]
    total_len = segs.norm(dim=-1).sum(dim=-1)
    return ((total_len - rest_length) ** 2).mean()

def gravity_prior(skeleton):
    """base 在 z≈0, 整体向上。skeleton: (B, N, 3)"""
    base_z = skeleton[:, 0, 2]
    return (base_z ** 2).mean()

# ═══════════════════════════════════════════════════════════════
# 渲染函数
# ═══════════════════════════════════════════════════════════════

def render_batch(model, rays_o, rays_d, action_window, near, far, n_samples, chunk_size=2048):
    """批量体渲染，分块避免 OOM。"""
    B = action_window.shape[0]
    N_rays = rays_o.shape[0]
    pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True)

    all_rendered = []
    for start in range(0, N_rays, chunk_size):
        end = min(start + chunk_size, N_rays)
        pts_chunk = pts[start:end]
        raw = model(pts_chunk, action_window)  # (B*chunk, n_samples, 2)
        raw = raw.reshape(B, end - start, n_samples, 2)
        chunk_imgs = []
        for b in range(B):
            img, _ = OM_rendering(raw[b])
            chunk_imgs.append(img)
        all_rendered.append(torch.stack(chunk_imgs))

    return torch.cat(all_rendered, dim=1)  # (B, N_rays)

# ═══════════════════════════════════════════════════════════════
# 训练循环
# ═══════════════════════════════════════════════════════════════

print('\n--- 开始训练 ---')
history = {'train_recon': [], 'train_smooth': [], 'train_length': [],
           'train_total': [], 'test_mne': []}

for epoch in range(args.epochs):
    model.train()
    epoch_losses = {'recon': 0, 'smooth': 0, 'length': 0, 'total': 0}
    n_batches = 0

    for batch in train_loader:
        aw = batch[0].to(device)      # (B, 20, 2)
        gt_img = batch[1].to(device)  # (B, H*W)

        # 渲染
        rendered = render_batch(model, rays_o, rays_d, aw, near, far, n_samples)

        # 渲染 loss
        loss_recon = F.mse_loss(rendered, gt_img)

        # 骨架正则化
        with torch.no_grad():
            pred_dict = model.predict_skeleton(aw)
        skeleton = pred_dict['fine']  # (B, 31, 3)

        # 注意：skeleton 是 no_grad 的，正则化 loss 需要重新计算
        # 所以我们重新 forward 一次获取有梯度的 skeleton
        physics_state = model.encode(aw)
        skeleton_dict = model.skeleton_head(physics_state)
        skel_fine = skeleton_dict['fine']  # (B, 31, 3)

        loss_smooth = skeleton_smoothness(skel_fine)
        loss_length = length_preservation(skel_fine)
        loss_gravity = gravity_prior(skel_fine)

        loss = (loss_recon
                + args.lambda_smooth * loss_smooth
                + args.lambda_length * loss_length
                + args.lambda_gravity * loss_gravity)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses['recon'] += loss_recon.item()
        epoch_losses['smooth'] += loss_smooth.item()
        epoch_losses['length'] += loss_length.item()
        epoch_losses['total'] += loss.item()
        n_batches += 1

    scheduler.step()

    # 平均 loss
    for k in epoch_losses:
        epoch_losses[k] /= n_batches
    history['train_recon'].append(epoch_losses['recon'])
    history['train_smooth'].append(epoch_losses['smooth'])
    history['train_length'].append(epoch_losses['length'])
    history['train_total'].append(epoch_losses['total'])

    # 测试评估
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        model.eval()
        mnes = []
        with torch.no_grad():
            for batch in test_loader:
                aw = batch[0].to(device)
                gt_pos = batch[-1].permute(0, 2, 1).to(device)  # (B, 31, 3)
                pred_dict = model.predict_skeleton(aw)
                pred = pred_dict['fine']  # (B, 31, 3)
                mne = (pred - gt_pos).norm(dim=-1).mean().item()
                mnes.append(mne)
        test_mne = np.mean(mnes)
        history['test_mne'].append((epoch, test_mne))
        print(f'  Epoch {epoch+1}/{args.epochs}: '
              f'recon={epoch_losses["recon"]:.6f} '
              f'smooth={epoch_losses["smooth"]:.6f} '
              f'total={epoch_losses["total"]:.6f} '
              f'test_MNE={test_mne:.6f}m')
    else:
        print(f'  Epoch {epoch+1}/{args.epochs}: total={epoch_losses["total"]:.6f}')

# ═══════════════════════════════════════════════════════════════
# 评估与可视化
# ═══════════════════════════════════════════════════════════════

print('\n--- 最终评估 ---')
model.eval()

# 收集测试集预测
all_pred, all_gt = [], []
all_pred_imgs, all_gt_imgs = [], []
with torch.no_grad():
    for batch in test_loader:
        aw = batch[0].to(device)
        gt_pos = batch[-1].permute(0, 2, 1)  # (B, 31, 3)
        gt_img = batch[1]

        pred_dict = model.predict_skeleton(aw)
        pred = pred_dict['fine'].cpu().numpy()

        all_pred.append(pred)
        all_gt.append(gt_pos.numpy())

        # 渲染对比（只取前几帧）
        if len(all_pred_imgs) < 5:
            rendered = render_batch(model, rays_o, rays_d, aw, near, far, n_samples, chunk_size=4096)
            for b in range(min(aw.shape[0], 5 - len(all_pred_imgs))):
                all_pred_imgs.append(rendered[b].cpu().numpy().reshape(H, W))
                all_gt_imgs.append(gt_img[b].numpy().reshape(H, W))

all_pred = np.concatenate(all_pred)
all_gt = np.concatenate(all_gt)

# 计算指标
errors = np.linalg.norm(all_pred - all_gt, axis=-1)  # (N, 31)
mean_mne = errors.mean()
max_mne = errors.max()
tip_err = errors[:, -1].mean()
base_err = errors[:, 0].mean()

# 逐轴误差
err_xyz = all_pred - all_gt
err_x = np.abs(err_xyz[:, :, 0]).mean()
err_y = np.abs(err_xyz[:, :, 1]).mean()
err_z = np.abs(err_xyz[:, :, 2]).mean()

print(f'  MNE:     {mean_mne:.6f} m')
print(f'  Max MNE: {max_mne:.6f} m (node {errors.mean(axis=0).argmax()})')
print(f'  Tip:     {tip_err:.6f} m')
print(f'  Per-axis: X={err_x:.5f}, Y={err_y:.5f}, Z={err_z:.5f}')

# ═══════════════════════════════════════════════════════════════
# 保存图表
# ═══════════════════════════════════════════════════════════════

# 1. 训练曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['train_total'], label='Total')
axes[0].plot(history['train_recon'], label='Recon')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].plot(history['train_smooth'], label='Smooth')
axes[1].plot(history['train_length'], label='Length')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('Regularization'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

if history['test_mne']:
    epochs_mne, mnes = zip(*history['test_mne'])
    axes[2].plot(epochs_mne, mnes, 'r-o', markersize=3)
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('MNE (m)')
axes[2].set_title('Test Skeleton MNE'); axes[2].grid(True, alpha=0.3)

plt.suptitle('Training: Skeleton from 2D Rendering Loss', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
plt.close()

# 2. 骨架对比
n_show = min(6, len(all_pred))
fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
axes = axes.flatten()
for i in range(n_show):
    ax = axes[i]
    ax.plot(all_gt[i, :, 0], all_gt[i, :, 1], all_gt[i, :, 2],
            'b-o', linewidth=3, markersize=4, label='GT', alpha=0.8)
    ax.plot(all_pred[i, :, 0], all_pred[i, :, 1], all_pred[i, :, 2],
            'r-o', linewidth=2, markersize=3, label='Pred', alpha=0.9)
    mne_i = errors[i].mean()
    ax.set_xlim(-0.15, 0.3); ax.set_ylim(-0.15, 0.15); ax.set_zlim(0, 0.55)
    ax.set_title(f'Sample {i}: MNE={mne_i:.4f}m')
    if i == 0:
        ax.legend(fontsize=8)
for i in range(n_show, 6):
    axes[i].axis('off')
plt.suptitle('Skeleton from 2D: GT (blue) vs Pred (red)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'skeleton_comparison.png'), dpi=150)
plt.close()

# 3. 逐节点误差
mean_node_err = errors.mean(axis=0)
std_node_err = errors.std(axis=0)
fig, ax = plt.subplots(figsize=(10, 4))
nodes = np.arange(31)
ax.bar(nodes, mean_node_err, yerr=std_node_err, color='salmon', alpha=0.8, capsize=2)
ax.axhline(mean_mne, color='red', linestyle='--', label=f'Mean: {mean_mne:.4f}m')
ax.set_xlabel('Node Index (base → tip)')
ax.set_ylabel('L2 Error (m)')
ax.set_title('Node-wise Error (2D-trained Skeleton)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'node_error.png'), dpi=150)
plt.close()

# 4. 渲染对比
if all_pred_imgs:
    n_render = min(4, len(all_pred_imgs))
    fig, axes = plt.subplots(2, n_render, figsize=(4 * n_render, 8))
    for i in range(n_render):
        axes[0, i].imshow(all_gt_imgs[i], cmap='gray')
        axes[0, i].set_title(f'GT {i}'); axes[0, i].axis('off')
        axes[1, i].imshow(all_pred_imgs[i], cmap='gray')
        axes[1, i].set_title(f'Pred {i}'); axes[1, i].axis('off')
    plt.suptitle('Rendering Quality', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'rendering_comparison.png'), dpi=150)
    plt.close()

# 保存摘要
with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== 方向1: 从 2D 渲染学习骨架 ===\n\n')
    f.write(f'训练配置:\n')
    f.write(f'  Epochs: {args.epochs}\n')
    f.write(f'  Batch size: {args.batch_size}\n')
    f.write(f'  Learning rate: {args.lr}\n')
    f.write(f'  Render samples: {n_samples}\n')
    f.write(f'  λ_smooth: {args.lambda_smooth}\n')
    f.write(f'  λ_length: {args.lambda_length}\n')
    f.write(f'  λ_gravity: {args.lambda_gravity}\n\n')
    f.write(f'结果:\n')
    f.write(f'  MNE:     {mean_mne:.6f} m\n')
    f.write(f'  Max MNE: {max_mne:.6f} m (node {errors.mean(axis=0).argmax()})\n')
    f.write(f'  Tip:     {tip_err:.6f} m\n')
    f.write(f'  Base:    {base_err:.6f} m\n')
    f.write(f'  Per-axis: X={err_x:.5f}, Y={err_y:.5f}, Z={err_z:.5f}\n')

# 保存模型
torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_2d_skeleton.pt'))
print(f'\n=== 完成 ===')
print(f'结果保存: {args.output_dir}/')
print(f'  training_curves.png, skeleton_comparison.png, node_error.png')
print(f'  rendering_comparison.png, summary.txt, model_2d_skeleton.pt')
