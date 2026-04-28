#!/usr/bin/env python3
"""
改进 exp1: 纯 2D 渲染 loss 学习骨架 — 渐进式训练

改进策略:
  1. 骨架初始化为垂直直线 (物理先验), 非随机
  2. 渐进式训练:
     Stage 1 (10 epochs): 仅 3D 先验 loss (垂直线 + 光滑 + 长度), 冻结 density
     Stage 2 (20 epochs): 渲染 loss + 3D 先验, 3D 先验权重逐步降低
     Stage 3 (20 epochs): 仅渲染 loss + 物理正则化

Usage:
    python scripts/experiments/exp1b_improved_2d_skeleton.py --gpu 0
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
parser.add_argument('--output_dir', type=str, default='output/exp1b_improved_2d')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_render_samples', type=int, default=48)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 改进 exp1: 渐进式 2D 骨架学习 ===')

from src.data.dataset import SoftSequenceDataset
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.models.model_ms_scnf import MSSCNFModel

ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
H, W = ds.H, ds.W
focal = ds.focal if hasattr(ds, 'focal') and ds.focal > 0 else 136.42
near, far, n_samples = 0.5, 2.5, args.n_render_samples

cam_params = ds.get_camera_params()
eye = tuple(cam_params['eye']) if cam_params else (1.5, 0.0, 0.5)
center = tuple(cam_params['center']) if cam_params else (0.0, 0.0, 0.25)
up = tuple(cam_params['up']) if cam_params else (0.0, 0.0, 1.0)
rays_o, rays_d = get_rays(H, W, focal, eye, center, up, device=device)

n_total = len(ds)
n_train = int(0.8 * n_total)
train_ds = torch.utils.data.Subset(ds, range(n_train))
test_ds = torch.utils.data.Subset(ds, range(n_train, n_total))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
print(f'数据: 训练={n_train}, 测试={n_total - n_train}')

# ── 模型创建 ──
model = MSSCNFModel(
    action_dim=ds.action_dim, window_size=20, n_scales=4,
    hidden_dim=128, d_filter=128, n_freqs=10,
    n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6,
).to(device)

# ── 初始化骨架为垂直直线 ──
with torch.no_grad():
    z_vals = np.linspace(0, 0.5, 31)
    vertical_line = np.stack([np.zeros(31), np.zeros(31), z_vals], axis=-1)  # (31, 3)
    target = torch.tensor(vertical_line, dtype=torch.float32, device=device)

    # 通过修改 skeleton_head 的最后一层 bias 来初始化
    for name, param in model.skeleton_head.named_parameters():
        if 'fine_head' in name and 'bias' in name:
            # fine_head 输出 (31*3), reshape 为 (31, 3)
            param.copy_(target.reshape(-1))
            print(f'  初始化 {name} → 垂直线')

# ── 渲染函数 ──
def render_batch(model, aw, chunk_size=2048):
    B = aw.shape[0]
    pts, _ = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True)
    parts = []
    for s in range(0, H * W, chunk_size):
        e = min(s + chunk_size, H * W)
        raw = model(pts[s:e], aw).reshape(B, e - s, n_samples, 2)
        parts.append(torch.stack([OM_rendering(raw[b])[0] for b in range(B)]))
    return torch.cat(parts, dim=1)

# ── 正则化 ──
def skeleton_smoothness(s):
    d2 = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
    return (d2 ** 2).sum(-1).mean()

def length_preservation(s, rest=0.5):
    segs = s[:, 1:] - s[:, :-1]
    return ((segs.norm(dim=-1).sum(dim=-1) - rest) ** 2).mean()

def vertical_prior(s):
    """骨架应接近垂直, x/y 偏移小。"""
    return (s[:, :, :2] ** 2).mean()

# ── 渐进式训练 ──
STAGE_1_EPOCHS = 10   # 仅 3D 先验
STAGE_2_EPOCHS = 20   # 渲染 + 3D 先验 (混合)
STAGE_3_EPOCHS = args.epochs - STAGE_1_EPOCHS - STAGE_2_EPOCHS  # 纯渲染

print(f'\n训练: Stage1={STAGE_1_EPOCHS}, Stage2={STAGE_2_EPOCHS}, Stage3={STAGE_3_EPOCHS}')

optimizer = torch.optim.Adam([
    {'params': model.temporal.parameters(), 'lr': args.lr},
    {'params': model.skeleton_head.parameters(), 'lr': args.lr},
    {'params': model.density.parameters(), 'lr': args.lr * 0.1},  # density 慢一点
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

history = {'total': [], 'render': [], 'prior': [], 'smooth': [], 'test_mne': []}

for epoch in range(args.epochs):
    model.train()

    # 确定训练阶段和 loss 权重
    if epoch < STAGE_1_EPOCHS:
        stage = 1
        w_render, w_prior, w_smooth = 0.0, 1.0, 0.1
        # 冻结 density
        for p in model.density.parameters():
            p.requires_grad = False
    elif epoch < STAGE_1_EPOCHS + STAGE_2_EPOCHS:
        stage = 2
        progress = (epoch - STAGE_1_EPOCHS) / STAGE_2_EPOCHS
        w_render = progress  # 0→1
        w_prior = 1.0 - 0.8 * progress  # 1→0.2
        w_smooth = 0.1
        # 解冻 density
        for p in model.density.parameters():
            p.requires_grad = True
    else:
        stage = 3
        w_render, w_prior, w_smooth = 1.0, 0.2, 0.1

    epoch_losses = {'total': 0, 'render': 0, 'prior': 0, 'smooth': 0}
    n_batches = 0

    for batch in train_loader:
        aw = batch[0].to(device)
        gt_img = batch[1].to(device)

        # 骨架预测
        physics_state = model.encode(aw)
        skel_dict = model.skeleton_head(physics_state)
        skel = skel_dict['fine']

        loss = torch.tensor(0.0, device=device)

        # 渲染 loss
        if w_render > 0:
            rendered = render_batch(model, aw)
            loss_render = F.mse_loss(rendered, gt_img)
            loss = loss + w_render * loss_render
            epoch_losses['render'] += loss_render.item()

        # 垂直线先验 loss
        loss_prior = vertical_prior(skel) + length_preservation(skel) * 0.5
        loss = loss + w_prior * loss_prior
        epoch_losses['prior'] += loss_prior.item()

        # 光滑性
        loss_smooth = skeleton_smoothness(skel)
        loss = loss + w_smooth * loss_smooth
        epoch_losses['smooth'] += loss_smooth.item()

        epoch_losses['total'] += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        n_batches += 1

    scheduler.step()

    for k in epoch_losses:
        epoch_losses[k] /= n_batches
    history['total'].append(epoch_losses['total'])
    history['render'].append(epoch_losses['render'])
    history['prior'].append(epoch_losses['prior'])
    history['smooth'].append(epoch_losses['smooth'])

    # 测试评估
    if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
        model.eval()
        mnes = []
        with torch.no_grad():
            for batch in test_loader:
                aw = batch[0].to(device)
                gt = batch[-1].permute(0, 2, 1).to(device)
                pred = model.predict_skeleton(aw)['fine']
                mnes.append((pred - gt).norm(dim=-1).mean().item())
        test_mne = np.mean(mnes)
        history['test_mne'].append((epoch, test_mne))
        print(f'  Epoch {epoch+1}/{args.epochs} [S{stage}]: '
              f'total={epoch_losses["total"]:.5f} '
              f'render={epoch_losses["render"]:.5f} '
              f'prior={epoch_losses["prior"]:.5f} '
              f'MNE={test_mne:.5f}m')
    else:
        print(f'  Epoch {epoch+1}/{args.epochs} [S{stage}]: total={epoch_losses["total"]:.5f}')

# ── 最终评估 ──
print('\n--- 最终评估 ---')
model.eval()
all_pred, all_gt = [], []
with torch.no_grad():
    for batch in test_loader:
        aw = batch[0].to(device)
        gt = batch[-1].permute(0, 2, 1).to(device)
        pred = model.predict_skeleton(aw)['fine']
        all_pred.append(pred.cpu().numpy())
        all_gt.append(gt.cpu().numpy())
all_pred = np.concatenate(all_pred)
all_gt = np.concatenate(all_gt)
errors = np.linalg.norm(all_pred - all_gt, axis=-1)

mne = errors.mean()
tip = errors[:, -1].mean()
err_x = np.abs(all_pred[:,:,0] - all_gt[:,:,0]).mean()
err_y = np.abs(all_pred[:,:,1] - all_gt[:,:,1]).mean()
err_z = np.abs(all_pred[:,:,2] - all_gt[:,:,2]).mean()

print(f'  MNE: {mne:.6f} m (exp1 原始: 0.3129m)')
print(f'  Tip: {tip:.6f} m')
print(f'  Per-axis: X={err_x:.5f}, Y={err_y:.5f}, Z={err_z:.5f}')

# ── 可视化 ──
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# 训练曲线
for k, label in [('total', 'Total'), ('render', 'Render'), ('prior', 'Prior'), ('smooth', 'Smooth')]:
    vals = history[k]
    if any(v > 0 for v in vals):
        axes[0].plot(vals, label=label)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss (Progressive)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')
# 标注训练阶段
axes[0].axvline(STAGE_1_EPOCHS, color='gray', linestyle='--', alpha=0.5)
axes[0].axvline(STAGE_1_EPOCHS + STAGE_2_EPOCHS, color='gray', linestyle='--', alpha=0.5)

# MNE 曲线
if history['test_mne']:
    e, m = zip(*history['test_mne'])
    axes[1].plot(e, m, 'r-o', markersize=3)
    axes[1].axhline(0.0593, color='green', linestyle='--', alpha=0.5, label='MS-SCNF baseline')
    axes[1].axhline(0.3129, color='orange', linestyle='--', alpha=0.5, label='exp1 original')
    axes[1].legend(fontsize=8)
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MNE (m)')
axes[1].set_title('Skeleton Accuracy'); axes[1].grid(True, alpha=0.3)

# 骨架对比
n_show = min(4, len(all_pred))
colors = plt.cm.tab10(np.linspace(0, 1, n_show))
for i in range(n_show):
    axes[2].plot(all_gt[i,:,2], all_gt[i,:,0], 'b-', alpha=0.4, linewidth=1)
    axes[2].plot(all_pred[i,:,2], all_pred[i,:,0], 'r-', alpha=0.6, linewidth=1.5)
axes[2].set_xlabel('Z'); axes[2].set_ylabel('X')
axes[2].set_title('GT (blue) vs Pred (red)\nFront View (ZX)')
axes[2].grid(True, alpha=0.3)

# 逐节点误差
mean_err = errors.mean(axis=0)
axes[3].bar(np.arange(31), mean_err, color='salmon', alpha=0.8)
axes[3].axhline(mne, color='red', linestyle='--', label=f'Mean: {mne:.4f}m')
axes[3].set_xlabel('Node'); axes[3].set_ylabel('Error (m)')
axes[3].set_title('Node-wise Error'); axes[3].legend(); axes[3].grid(True, alpha=0.3)

plt.suptitle('Improved 2D Skeleton Learning (Progressive Training)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'results.png'), dpi=150)
plt.close()

# 保存
torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_improved_2d.pt'))
with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== 改进 exp1: 渐进式 2D 骨架学习 ===\n\n')
    f.write(f'MNE:       {mne:.6f} m  (原 exp1: 0.3129m, MS-SCNF: 0.0593m)\n')
    f.write(f'Tip:       {tip:.6f} m\n')
    f.write(f'X err:     {err_x:.6f} m\n')
    f.write(f'Y err:     {err_y:.6f} m\n')
    f.write(f'Z err:     {err_z:.6f} m\n')
    f.write(f'\n对比:\n')
    f.write(f'  vs exp1 原始:   {"改善" if mne < 0.3129 else "未改善"} ({mne/0.3129:.1f}x)\n')
    f.write(f'  vs MS-SCNF 3D:  {mne/0.0593:.1f}x 差距\n')

print(f'\n结果保存: {args.output_dir}/')
