#!/usr/bin/env python3
"""
exp7: 多视角 + 2D 骨架替代 3D GT — 一键实验

完整流程:
  1. 采集多视角数据（正面+侧面）
  2. Phase 1: 2D 骨架投影 loss 学习骨架（不需要 3D GT）
  3. Phase 2: 多视角渲染 loss 联合训练
  4. 评估与可视化

核心创新:
  - Phase 1 监督信号来自从图像提取的 2D 骨架，不是仿真器 3D GT
  - 推理时仅通过驱动参数直接输出 3D 骨架，不需要图像

Usage:
    python scripts/experiments/exp7_multiview_2d_skeleton.py --gpu 0
    python scripts/experiments/exp7_multiview_2d_skeleton.py --gpu 0 --skip-collection
    python scripts/experiments/exp7_multiview_2d_skeleton.py --gpu 0 --data-dir data/my_multiview
"""

import os
import sys
import argparse
import time
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

parser = argparse.ArgumentParser(description="Exp7: Multi-view + 2D Skeleton")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data-dir', type=str, default='data/exp7_multiview')
parser.add_argument('--output-dir', type=str, default='output/exp7_multiview_2d')
parser.add_argument('--skip-collection', action='store_true', help='跳过数据采集')
parser.add_argument('--sequences', type=int, default=5, help='采集序列数')
parser.add_argument('--actions-per-seq', type=int, default=50)
parser.add_argument('--phase1-epochs', type=int, default=30)
parser.add_argument('--phase2-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n-render-samples', type=int, default=48)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print('=' * 60)
print('Exp7: 多视角 + 2D 骨架替代 3D GT')
print('=' * 60)


# ============================================================================
# Step 1: 数据采集
# ============================================================================
if not args.skip_collection:
    print('\n>>> Step 1: 采集多视角数据...')
    cmd = [
        sys.executable,
        'scripts/data_collection/collect.py',
        '--action-x', 'random', '--action-y', 'random',
        '--sequences', str(args.sequences),
        '--actions-per-seq', str(args.actions_per_seq),
        '--save-dir', args.data_dir,
    ]
    print(f'    命令: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)
else:
    print(f'\n>>> Step 1: 跳过采集，使用已有数据: {args.data_dir}')


# ============================================================================
# Step 2: 加载数据
# ============================================================================
print('\n>>> Step 2: 加载多视角数据集...')
from src.data.dataset_multiview import MultiViewDataset
from src.utils.skeleton_2d import compute_2d_skeleton_loss, project_3d_to_2d
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.models.model_ms_scnf import MSSCNFModel

ds = MultiViewDataset(args.data_dir, seq_len=20, return_3d=True)
H, W = ds.H, ds.W
focal = ds.focal
cameras = ds.cameras

# 为两个视角生成射线
rays_list = []
for cam in cameras:
    ro, rd = get_rays(H, W, cam['focal'], cam['eye'], cam['center'], cam['up'], device=device)
    rays_list.append((ro, rd))

near, far, n_samples = 0.5, 2.5, args.n_render_samples

# 划分训练/测试
n_total = len(ds)
n_train = int(0.8 * n_total)
train_ds = Subset(ds, range(n_train))
test_ds = Subset(ds, range(n_train, n_total))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
print(f'    训练: {n_train}, 测试: {n_total - n_train}')
print(f'    相机: front eye={cameras[0]["eye"]}, side eye={cameras[1]["eye"]}')


# ============================================================================
# Step 3: 创建模型
# ============================================================================
print('\n>>> Step 3: 创建模型...')
model = MSSCNFModel(
    action_dim=ds.action_dim, window_size=20, n_scales=4,
    hidden_dim=128, d_filter=128, n_freqs=10,
    n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6,
).to(device)

# 初始化骨架为垂直直线
with torch.no_grad():
    z_vals = np.linspace(0, 0.5, 31)
    vertical_line = np.stack([np.zeros(31), np.zeros(31), z_vals], axis=-1)
    target = torch.tensor(vertical_line, dtype=torch.float32, device=device)
    for name, param in model.skeleton_head.named_parameters():
        if 'fine_head' in name and 'bias' in name:
            param.copy_(target.reshape(-1))

n_params = sum(p.numel() for p in model.parameters())
print(f'    参数量: {n_params:,}')


# ============================================================================
# 工具函数
# ============================================================================
def render_single_view(model, aw, rays_o, rays_d, chunk_size=2048):
    """从单个视角渲染图像。"""
    B = aw.shape[0]
    pts, _ = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True)
    parts = []
    for s in range(0, H * W, chunk_size):
        e = min(s + chunk_size, H * W)
        raw = model(pts[s:e], aw).reshape(B, e - s, n_samples, 2)
        parts.append(torch.stack([OM_rendering(raw[b])[0] for b in range(B)]))
    return torch.cat(parts, dim=1)


def skeleton_smoothness(s):
    d2 = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
    return (d2 ** 2).sum(-1).mean()


def length_preservation(s, rest=0.5):
    segs = s[:, 1:] - s[:, :-1]
    return ((segs.norm(dim=-1).sum(dim=-1) - rest) ** 2).mean()


def vertical_prior(s):
    return (s[:, :, :2] ** 2).mean()


# ============================================================================
# Step 4: Phase 1 — 2D 骨架投影 loss 学习
# ============================================================================
print(f'\n>>> Step 4: Phase 1 训练（2D 骨架投影 loss，{args.phase1_epochs} epochs）...')
print('    监督信号: 从图像提取的 2D 骨架 → 投影 3D 骨架到 2D → L2 loss')

optimizer = torch.optim.Adam([
    {'params': model.temporal.parameters(), 'lr': args.lr},
    {'params': model.skeleton_head.parameters(), 'lr': args.lr},
    {'params': model.density.parameters(), 'lr': args.lr * 0.01},
])

# Phase 1: 冻结 density
for p in model.density.parameters():
    p.requires_grad = False

history = {'p1_total': [], 'p1_skel2d': [], 'p1_prior': [], 'p1_smooth': [], 'p1_mne': []}

for epoch in range(args.phase1_epochs):
    model.train()
    epoch_losses = {'total': 0, 'skel2d': 0, 'prior': 0, 'smooth': 0}
    n_batches = 0

    for batch in train_loader:
        aw = batch[0].to(device)
        skel_2d_front = batch[3].to(device)  # (B, 31, 2)
        skel_2d_side = batch[4].to(device)   # (B, 31, 2)

        physics_state = model.encode(aw)
        skel_dict = model.skeleton_head(physics_state)
        skel_3d = skel_dict['fine']  # (B, 31, 3)

        # 2D 骨架投影 loss
        skel_2d_list = [skel_2d_front, skel_2d_side]
        loss_skel2d = compute_2d_skeleton_loss(skel_3d, skel_2d_list, cameras)

        # 物理先验
        loss_prior = vertical_prior(skel_3d) + 0.5 * length_preservation(skel_3d)
        loss_smooth = skeleton_smoothness(skel_3d)

        loss = loss_skel2d + 0.1 * loss_prior + 0.1 * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses['total'] += loss.item()
        epoch_losses['skel2d'] += loss_skel2d.item()
        epoch_losses['prior'] += loss_prior.item()
        epoch_losses['smooth'] += loss_smooth.item()
        n_batches += 1

    for k in epoch_losses:
        epoch_losses[k] /= max(n_batches, 1)
    history['p1_total'].append(epoch_losses['total'])
    history['p1_skel2d'].append(epoch_losses['skel2d'])
    history['p1_prior'].append(epoch_losses['prior'])
    history['p1_smooth'].append(epoch_losses['smooth'])

    # 评估
    if (epoch + 1) % 5 == 0 or epoch == args.phase1_epochs - 1:
        model.eval()
        mnes = []
        with torch.no_grad():
            for batch in test_loader:
                aw = batch[0].to(device)
                gt = batch[-1].permute(0, 2, 1).to(device)  # (B, 3, 31) → (B, 31, 3)
                pred = model.predict_skeleton(aw)['fine']
                mnes.append((pred - gt).norm(dim=-1).mean().item())
        test_mne = np.mean(mnes)
        history['p1_mne'].append((epoch, test_mne))
        print(f'  P1 Epoch {epoch+1}/{args.phase1_epochs}: '
              f'skel2d={epoch_losses["skel2d"]:.2f} '
              f'prior={epoch_losses["prior"]:.5f} '
              f'MNE={test_mne:.5f}m')
    else:
        print(f'  P1 Epoch {epoch+1}/{args.phase1_epochs}: '
              f'skel2d={epoch_losses["skel2d"]:.2f}')

# 保存 Phase 1 checkpoint
torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_phase1.pt'))
print(f'    Phase 1 完成，checkpoint 已保存')


# ============================================================================
# Step 5: Phase 2 — 多视角渲染 loss 联合训练
# ============================================================================
print(f'\n>>> Step 5: Phase 2 训练（多视角渲染 loss，{args.phase2_epochs} epochs）...')
print('    监督信号: 双视角完整图像渲染 loss + 2D 骨架投影 loss')

# 解冻 density
for p in model.density.parameters():
    p.requires_grad = True

# 重新创建 optimizer（调整学习率）
optimizer = torch.optim.Adam([
    {'params': model.temporal.parameters(), 'lr': args.lr * 0.5},
    {'params': model.skeleton_head.parameters(), 'lr': args.lr * 0.5},
    {'params': model.density.parameters(), 'lr': args.lr * 0.1},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

history_p2 = {'p2_total': [], 'p2_render': [], 'p2_skel2d': [], 'p2_mne': []}

for epoch in range(args.phase2_epochs):
    model.train()
    epoch_losses = {'total': 0, 'render': 0, 'skel2d': 0}
    n_batches = 0

    for batch in train_loader:
        aw = batch[0].to(device)
        img_front = batch[1].to(device)  # (B, H*W)
        img_side = batch[2].to(device)   # (B, H*W)
        skel_2d_front = batch[3].to(device)
        skel_2d_side = batch[4].to(device)

        # 渲染两个视角
        rendered_front = render_single_view(model, aw, rays_list[0][0], rays_list[0][1])
        rendered_side = render_single_view(model, aw, rays_list[1][0], rays_list[1][1])

        loss_render = (F.mse_loss(rendered_front, img_front) +
                       F.mse_loss(rendered_side, img_side))

        # 2D 骨架投影 loss（保持，但权重降低）
        skel_dict = model.skeleton_head(model.encode(aw))
        skel_3d = skel_dict['fine']
        loss_skel2d = compute_2d_skeleton_loss(
            skel_3d, [skel_2d_front, skel_2d_side], cameras)

        # 物理正则化
        loss_phys = 0.1 * skeleton_smoothness(skel_3d) + 0.05 * length_preservation(skel_3d)

        loss = loss_render + 0.5 * loss_skel2d + loss_phys

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_losses['total'] += loss.item()
        epoch_losses['render'] += loss_render.item()
        epoch_losses['skel2d'] += loss_skel2d.item()
        n_batches += 1

    scheduler.step()

    for k in epoch_losses:
        epoch_losses[k] /= max(n_batches, 1)
    history_p2['p2_total'].append(epoch_losses['total'])
    history_p2['p2_render'].append(epoch_losses['render'])
    history_p2['p2_skel2d'].append(epoch_losses['skel2d'])

    if (epoch + 1) % 10 == 0 or epoch == args.phase2_epochs - 1:
        model.eval()
        mnes = []
        with torch.no_grad():
            for batch in test_loader:
                aw = batch[0].to(device)
                gt = batch[-1].permute(0, 2, 1).to(device)
                pred = model.predict_skeleton(aw)['fine']
                mnes.append((pred - gt).norm(dim=-1).mean().item())
        test_mne = np.mean(mnes)
        history_p2['p2_mne'].append((epoch, test_mne))
        print(f'  P2 Epoch {epoch+1}/{args.phase2_epochs}: '
              f'render={epoch_losses["render"]:.5f} '
              f'skel2d={epoch_losses["skel2d"]:.2f} '
              f'MNE={test_mne:.5f}m')
    else:
        print(f'  P2 Epoch {epoch+1}/{args.phase2_epochs}: '
              f'render={epoch_losses["render"]:.5f}')


# ============================================================================
# Step 6: 验证（使用 3D GT 作为真值）
# ============================================================================
print('\n>>> Step 6: 验证（3D GT 真值对比）...')
assert ds.has_3d, "数据中缺少 3D positions，无法进行验证。请确保采集时使用 --3d 参数。"

model.eval()
all_pred, all_gt = [], []
all_render_front, all_render_side = [], []
all_gt_front, all_gt_side = [], []
all_proj_err_front, all_proj_err_side = [], []

with torch.no_grad():
    for batch in test_loader:
        aw = batch[0].to(device)
        skel_2d_front = batch[3].to(device)
        skel_2d_side = batch[4].to(device)
        gt_3d = batch[-1].permute(0, 2, 1).to(device)  # (B, 31, 3)

        # 3D 骨架预测
        pred_3d = model.predict_skeleton(aw)['fine']
        all_pred.append(pred_3d.cpu().numpy())
        all_gt.append(gt_3d.cpu().numpy())

        # 2D 投影误差
        proj_front = project_3d_to_2d(pred_3d, cameras[0]['eye'], cameras[0]['center'],
                                       cameras[0]['up'], cameras[0]['focal'], H, W)
        proj_side = project_3d_to_2d(pred_3d, cameras[1]['eye'], cameras[1]['center'],
                                      cameras[1]['up'], cameras[1]['focal'], H, W)
        mask_f = (skel_2d_front.sum(dim=-1) > 0.1)
        mask_s = (skel_2d_side.sum(dim=-1) > 0.1)
        if mask_f.any():
            all_proj_err_front.append((proj_front[mask_f] - skel_2d_front[mask_f]).norm(dim=-1).mean().item())
        if mask_s.any():
            all_proj_err_side.append((proj_side[mask_s] - skel_2d_side[mask_s]).norm(dim=-1).mean().item())

        # 渲染对比
        rf = render_single_view(model, aw, rays_list[0][0], rays_list[0][1])
        rs = render_single_view(model, aw, rays_list[1][0], rays_list[1][1])
        all_render_front.append(rf.cpu().numpy())
        all_render_side.append(rs.cpu().numpy())
        all_gt_front.append(batch[1].numpy())
        all_gt_side.append(batch[2].numpy())

all_pred = np.concatenate(all_pred)
all_gt = np.concatenate(all_gt)
errors = np.linalg.norm(all_pred - all_gt, axis=-1)

# --- 3D 骨架精度 ---
mne = errors.mean()
tip = errors[:, -1].mean()
err_x = np.abs(all_pred[:, :, 0] - all_gt[:, :, 0]).mean()
err_y = np.abs(all_pred[:, :, 1] - all_gt[:, :, 1]).mean()
err_z = np.abs(all_pred[:, :, 2] - all_gt[:, :, 2]).mean()
total_3d = np.sqrt(err_x**2 + err_y**2 + err_z**2)
ratio_x = err_x / total_3d * 100
ratio_y = err_y / total_3d * 100
ratio_z = err_z / total_3d * 100

# --- 2D 投影精度 ---
proj_err_front = np.mean(all_proj_err_front) if all_proj_err_front else 0
proj_err_side = np.mean(all_proj_err_side) if all_proj_err_side else 0

# --- 渲染质量 ---
gt_front_all = np.concatenate(all_gt_front)
gt_side_all = np.concatenate(all_gt_side)
render_front_all = np.concatenate(all_render_front)
render_side_all = np.concatenate(all_render_side)
mse_front = np.mean((gt_front_all - render_front_all) ** 2)
mse_side = np.mean((gt_side_all - render_side_all) ** 2)
psnr_front = 10 * np.log10(1.0 / max(mse_front, 1e-10))
psnr_side = 10 * np.log10(1.0 / max(mse_side, 1e-10))

# --- 逐节点误差 ---
node_err = errors.mean(axis=0)  # (31,)
max_node_err = errors.max(axis=0).mean()

print(f'\n{"="*60}')
print(f'  验证报告（3D GT 真值对比）')
print(f'{"="*60}')
print(f'\n  --- 3D 骨架精度（vs 3D GT）---')
print(f'  MNE:           {mne:.6f} m')
print(f'  Tip Error:     {tip:.6f} m')
print(f'  Max Node Err:  {max_node_err:.6f} m (平均最大)')
print(f'  X err:         {err_x:.6f} m  ({ratio_x:.1f}%)')
print(f'  Y err:         {err_y:.6f} m  ({ratio_y:.1f}%)')
print(f'  Z err:         {err_z:.6f} m  ({ratio_z:.1f}%)')
print(f'\n  --- 2D 投影精度（vs 提取的2D骨架）---')
print(f'  Front proj err: {proj_err_front:.2f} px')
print(f'  Side proj err:  {proj_err_side:.2f} px')
print(f'\n  --- 渲染质量 ---')
print(f'  PSNR front:    {psnr_front:.2f} dB')
print(f'  PSNR side:     {psnr_side:.2f} dB')
print(f'\n  --- 与基线对比 ---')
print(f'  {"方法":<25} {"MNE(m)":<12} {"Tip(m)":<12} {"说明"}')
print(f'  {"-"*65}')
print(f'  {"exp1 (纯2D)":<25} {"0.313":<12} {"0.588":<12} {"单视角, 无先验"}')
print(f'  {"exp1b (2D+先验)":<25} {"0.154":<12} {"0.207":<12} {"单视角, 渐进+先验"}')
print(f'  {"MS-SCNF (3D GT)":<25} {"0.065":<12} {"0.108":<12} {"有3D GT监督"}')
print(f'  {"exp7 (ours)":<25} {f"{mne:.3f}":<12} {f"{tip:.3f}":<12} {"多视角+2D骨架, 无3D GT训练"}')


# ============================================================================
# Step 7: 可视化
# ============================================================================
print(f'\n>>> Step 7: 生成可视化...')

# --- 图1: 训练曲线 + 骨架对比 (8 panels) ---
fig, axes = plt.subplots(2, 4, figsize=(24, 10))

# Row 1: 训练曲线
if any(v > 0 for v in history['p1_skel2d']):
    axes[0, 0].plot(history['p1_skel2d'], label='2D Skeleton Loss')
    axes[0, 0].plot(history['p1_prior'], label='Physics Prior')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Phase 1: 2D Skeleton Projection Loss')
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_yscale('log')

if history['p1_mne']:
    e, m = zip(*history['p1_mne'])
    axes[0, 1].plot(e, m, 'b-o', markersize=3, label='Phase 1')
axes[0, 1].axhline(0.065, color='green', linestyle='--', alpha=0.5, label='MS-SCNF (3D GT)')
axes[0, 1].axhline(0.154, color='orange', linestyle='--', alpha=0.5, label='Exp1b')
axes[0, 1].axhline(0.313, color='red', linestyle='--', alpha=0.5, label='Exp1')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('MNE (m)')
axes[0, 1].set_title('Phase 1: Skeleton Accuracy (3D GT)')
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(history_p2['p2_render'], label='Render Loss')
axes[0, 2].plot(history_p2['p2_skel2d'], label='2D Skel Loss')
axes[0, 2].set_xlabel('Epoch'); axes[0, 2].set_ylabel('Loss')
axes[0, 2].set_title('Phase 2: Multi-View Rendering Loss')
axes[0, 2].legend(fontsize=8); axes[0, 2].grid(True, alpha=0.3); axes[0, 2].set_yscale('log')

if history_p2['p2_mne']:
    e, m = zip(*history_p2['p2_mne'])
    axes[0, 3].plot(e, m, 'r-o', markersize=3, label='Phase 2')
axes[0, 3].axhline(0.065, color='green', linestyle='--', alpha=0.5, label='MS-SCNF (3D GT)')
axes[0, 3].axhline(0.154, color='orange', linestyle='--', alpha=0.5, label='Exp1b')
axes[0, 3].set_xlabel('Epoch'); axes[0, 3].set_ylabel('MNE (m)')
axes[0, 3].set_title('Phase 2: Skeleton Accuracy (3D GT)')
axes[0, 3].legend(fontsize=8); axes[0, 3].grid(True, alpha=0.3)

# Row 2: 骨架/渲染对比
n_show = min(4, len(all_pred))
for i in range(n_show):
    axes[1, 0].plot(all_gt[i, :, 2], all_gt[i, :, 0], 'b-', alpha=0.4, linewidth=1)
    axes[1, 0].plot(all_pred[i, :, 2], all_pred[i, :, 0], 'r-', alpha=0.6, linewidth=1.5)
axes[1, 0].set_xlabel('Z'); axes[1, 0].set_ylabel('X')
axes[1, 0].set_title('3D Skeleton: GT (blue) vs Pred (red)\nFront View (ZX)')
axes[1, 0].grid(True, alpha=0.3)

for i in range(n_show):
    axes[1, 1].plot(all_gt[i, :, 2], all_gt[i, :, 1], 'b-', alpha=0.4, linewidth=1)
    axes[1, 1].plot(all_pred[i, :, 2], all_pred[i, :, 1], 'r-', alpha=0.6, linewidth=1.5)
axes[1, 1].set_xlabel('Z'); axes[1, 1].set_ylabel('Y')
axes[1, 1].set_title('3D Skeleton: GT (blue) vs Pred (red)\nSide View (ZY)')
axes[1, 1].grid(True, alpha=0.3)

if len(render_front_all) > 0:
    composite = np.zeros((H, W, 3))
    composite[:, :, 0] = gt_front_all[0].reshape(H, W)
    composite[:, :, 2] = render_front_all[0].reshape(H, W).clip(0, 1)
    axes[1, 2].imshow(composite, vmin=0, vmax=1)
    axes[1, 2].set_title('Front Render (R=GT, B=Pred)')
    axes[1, 2].axis('off')

if len(render_side_all) > 0:
    composite = np.zeros((H, W, 3))
    composite[:, :, 0] = gt_side_all[0].reshape(H, W)
    composite[:, :, 2] = render_side_all[0].reshape(H, W).clip(0, 1)
    axes[1, 3].imshow(composite, vmin=0, vmax=1)
    axes[1, 3].set_title('Side Render (R=GT, B=Pred)')
    axes[1, 3].axis('off')

plt.suptitle('Exp7: Multi-View + 2D Skeleton (No 3D GT Training)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'results.png'), dpi=150)
plt.close()

# --- 图2: 验证细节 (逐节点误差 + 逐轴误差比例) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 逐节点误差
axes[0].bar(np.arange(31), node_err, color='salmon', alpha=0.8)
axes[0].axhline(mne, color='red', linestyle='--', label=f'Mean: {mne:.4f}m')
axes[0].set_xlabel('Node Index'); axes[0].set_ylabel('Error (m)')
axes[0].set_title('Node-wise 3D Error (vs GT)')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 逐轴误差比例
axes[1].bar(['X (front depth)', 'Y (side depth)', 'Z (vertical)'],
            [ratio_x, ratio_y, ratio_z],
            color=['#e74c3c', '#3498db', '#2ecc71'], alpha=0.8)
axes[1].set_ylabel('Error Ratio (%)')
axes[1].set_title('Per-Axis Error Distribution')
for i, v in enumerate([ratio_x, ratio_y, ratio_z]):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# 2D 投影误差 + 渲染 PSNR 汇总
metrics = ['MNE\n(m)', 'Tip\n(m)', 'Proj Front\n(px)', 'Proj Side\n(px)', 'PSNR Front\n(dB)', 'PSNR Side\n(dB)']
values = [mne, tip, proj_err_front, proj_err_side, psnr_front, psnr_side]
colors = ['#e74c3c' if i < 2 else '#3498db' if i < 4 else '#2ecc71' for i in range(6)]
axes[2].bar(metrics, values, color=colors, alpha=0.8)
for i, v in enumerate(values):
    axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
axes[2].set_title('Verification Metrics Summary')
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Exp7: Verification (3D GT Ground Truth)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'verification.png'), dpi=150)
plt.close()

# --- 图3: 2D 骨架投影叠加可视化 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# 取一个测试样本展示
sample_idx = 0
sample_batch = next(iter(test_loader))
with torch.no_grad():
    aw = sample_batch[0][:1].to(device)
    pred_3d = model.predict_skeleton(aw)['fine']
    gt_img_front = sample_batch[1][0].numpy().reshape(H, W)
    gt_img_side = sample_batch[2][0].numpy().reshape(H, W)
    skel_2d_f = sample_batch[3][0].numpy()
    skel_2d_s = sample_batch[4][0].numpy()
    proj_f = project_3d_to_2d(pred_3d, cameras[0]['eye'], cameras[0]['center'],
                               cameras[0]['up'], cameras[0]['focal'], H, W)[0].cpu().numpy()
    proj_s = project_3d_to_2d(pred_3d, cameras[1]['eye'], cameras[1]['center'],
                               cameras[1]['up'], cameras[1]['focal'], H, W)[0].cpu().numpy()

for ax, img, skel_gt, proj, title in [
    (axes[0], gt_img_front, skel_2d_f, proj_f, 'Front View'),
    (axes[1], gt_img_side, skel_2d_s, proj_s, 'Side View'),
]:
    ax.imshow(img, cmap='gray', alpha=0.5)
    ax.plot(skel_gt[:, 0], skel_gt[:, 1], 'g.-', markersize=3, linewidth=1, label='2D extracted')
    ax.plot(proj[:, 0], proj[:, 1], 'r.-', markersize=3, linewidth=1.5, label='3D projected')
    ax.set_title(f'{title}: 2D Skeleton Overlay')
    ax.legend(fontsize=8)
    ax.axis('off')

plt.suptitle('2D Skeleton: Extracted (green) vs 3D Projected (red)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'skeleton_2d_overlay.png'), dpi=150)
plt.close()


# ============================================================================
# Step 8: 保存结果
# ============================================================================
torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_final.pt'))

with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== Exp7: 多视角 + 2D 骨架替代 3D GT ===\n\n')
    f.write(f'数据: {args.data_dir}\n')
    f.write(f'训练样本: {n_train}, 测试样本: {n_total - n_train}\n')
    f.write(f'Phase 1 epochs: {args.phase1_epochs}\n')
    f.write(f'Phase 2 epochs: {args.phase2_epochs}\n')
    f.write(f'注意: 训练过程完全不使用 3D GT，仅用 2D 骨架投影 loss + 渲染 loss\n')
    f.write(f'      3D GT 仅在验证阶段用于精度评估\n\n')

    f.write(f'--- 3D 骨架精度 (vs 3D GT 真值) ---\n')
    f.write(f'MNE:           {mne:.6f} m\n')
    f.write(f'Tip Error:     {tip:.6f} m\n')
    f.write(f'Max Node Err:  {max_node_err:.6f} m\n')
    f.write(f'X err:         {err_x:.6f} m  ({ratio_x:.1f}%)\n')
    f.write(f'Y err:         {err_y:.6f} m  ({ratio_y:.1f}%)\n')
    f.write(f'Z err:         {err_z:.6f} m  ({ratio_z:.1f}%)\n\n')

    f.write(f'--- 2D 投影精度 ---\n')
    f.write(f'Front proj err: {proj_err_front:.2f} px\n')
    f.write(f'Side proj err:  {proj_err_side:.2f} px\n\n')

    f.write(f'--- 渲染质量 ---\n')
    f.write(f'PSNR front:    {psnr_front:.2f} dB\n')
    f.write(f'PSNR side:     {psnr_side:.2f} dB\n\n')

    f.write(f'--- 与基线对比 (3D MNE) ---\n')
    f.write(f'  {"方法":<25} {"MNE(m)":<10} {"Tip(m)":<10} {"说明"}\n')
    f.write(f'  {"-"*60}\n')
    f.write(f'  {"exp1 (纯2D)":<25} {"0.313":<10} {"0.588":<10} {"单视角, 无先验"}\n')
    f.write(f'  {"exp1b (2D+先验)":<25} {"0.154":<10} {"0.207":<10} {"单视角, 渐进+先验"}\n')
    f.write(f'  {"MS-SCNF (3D GT)":<25} {"0.065":<10} {"0.108":<10} {"有3D GT监督"}\n')
    f.write(f'  {"exp7 (ours)":<25} {f"{mne:.3f}":<10} {f"{tip:.3f}":<10} {"多视角+2D骨架, 无3D GT训练"}\n')

    if mne < 0.154:
        f.write(f'\n  !! 超越 exp1b 单视角方案 ({mne/0.154:.2f}x)\n')
    if mne < 0.065:
        f.write(f'  !! 达到 3D GT 监督水平 ({mne/0.065:.2f}x)\n')

print(f'\n>>> 完成！结果保存至: {args.output_dir}/')
print(f'    model_final.pt  — 模型权重')
print(f'    results.png     — 训练曲线 + 骨架对比 + 渲染对比')
print(f'    verification.png — 逐节点误差 + 逐轴比例 + 指标汇总')
print(f'    skeleton_2d_overlay.png — 2D骨架叠加可视化')
print(f'    summary.txt     — 完整验证报告')
