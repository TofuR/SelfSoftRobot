#!/usr/bin/env python3
"""
方向4: 仿真到真实迁移 — 域随机化训练与鲁棒性评估

在训练图像上添加随机扰动（噪声、亮度、对比度、背景），
评估模型对输入扰动的鲁棒性，为 sim-to-real 做准备。

Usage:
    python scripts/experiments/exp4_domain_randomization.py --gpu 0 --epochs 50
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
from torch.utils.data import DataLoader, Dataset
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='data/seq_rr_3d')
parser.add_argument('--output_dir', type=str, default='output/exp4_sim2real')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_render_samples', type=int, default=32)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向4: 域随机化训练 ===')
print(f'Device: {device}, Epochs: {args.epochs}')

# ═══════════════════════════════════════════════════════════════
# 域随机化增强
# ═══════════════════════════════════════════════════════════════

class DomainRandomizer:
    """图像域随机化：噪声、亮度、对比度、背景。"""
    def __init__(self, noise_std=0.05, brightness_range=(0.7, 1.3),
                 contrast_range=(0.7, 1.3), bg_range=(0, 0.15)):
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.bg_range = bg_range

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: (H*W,) float tensor in [0, 1]
        Returns:
            augmented: (H*W,) float tensor
        """
        img = img_tensor.clone()

        # 随机背景
        bg_val = np.random.uniform(*self.bg_range)
        mask = img < 0.5
        img[mask] = bg_val

        # 高斯噪声
        noise = torch.randn_like(img) * self.noise_std
        img = img + noise

        # 亮度
        brightness = np.random.uniform(*self.brightness_range)
        img = img * brightness

        # 对比度
        contrast = np.random.uniform(*self.contrast_range)
        mean_val = img.mean()
        img = (img - mean_val) * contrast + mean_val

        return img.clamp(0, 1)

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

randomizer = DomainRandomizer()
print(f'数据: 训练={n_train}, 测试={n_total - n_train}')

# ═══════════════════════════════════════════════════════════════
# 公共函数
# ═══════════════════════════════════════════════════════════════

def make_model():
    return MSSCNFModel(
        action_dim=ds.action_dim, window_size=20, n_scales=4,
        hidden_dim=128, d_filter=128, n_freqs=10,
        n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6,
    ).to(device)

def render_batch(model, aw, chunk_size=2048):
    B = aw.shape[0]
    pts, _ = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True)
    parts = []
    for s in range(0, H * W, chunk_size):
        e = min(s + chunk_size, H * W)
        raw = model(pts[s:e], aw).reshape(B, e - s, n_samples, 2)
        parts.append(torch.stack([OM_rendering(raw[b])[0] for b in range(B)]))
    return torch.cat(parts, dim=1)

def evaluate(model, loader, augment_fn=None):
    model.eval()
    mnes, render_errs = [], []
    with torch.no_grad():
        for batch in loader:
            aw = batch[0].to(device)
            gt_pos = batch[-1].permute(0, 2, 1).to(device)
            gt_img = batch[1].to(device)

            # 骨架精度
            pred = model.predict_skeleton(aw)['fine']
            mne = (pred - gt_pos).norm(dim=-1).mean().item()
            mnes.append(mne)

            # 渲染质量（少量样本）
            if len(render_errs) < 3:
                rendered = render_batch(model, aw, chunk_size=4096)
                gt_aug = augment_fn(gt_img[0]).unsqueeze(0) if augment_fn else gt_img
                render_err = F.mse_loss(rendered, gt_aug[:rendered.shape[0]]).item()
                render_errs.append(render_err)

    return {
        'mne': np.mean(mnes),
        'render_mse': np.mean(render_errs) if render_errs else 0,
    }

# ═══════════════════════════════════════════════════════════════
# 训练 Model A: 无域随机化 (baseline)
# ═══════════════════════════════════════════════════════════════

print('\n=== Model A: 无域随机化 (Baseline) ===')
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
        gt_img = batch[1].to(device)

        rendered = render_batch(model_a, aw)
        loss = F.mse_loss(rendered, gt_img)

        opt_a.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        opt_a.step()
        total_loss += loss.item()
        n_b += 1
    sched_a.step()

    hist_a['loss'].append(total_loss / n_b)
    if (epoch + 1) % 10 == 0:
        m = evaluate(model_a, test_loader)
        hist_a['mne'].append((epoch, m['mne']))
        print(f'  Epoch {epoch+1}: loss={hist_a["loss"][-1]:.6f}, MNE={m["mne"]:.6f}m')
    else:
        print(f'  Epoch {epoch+1}: loss={hist_a["loss"][-1]:.6f}')

metrics_a_clean = evaluate(model_a, test_loader)
metrics_a_noisy = evaluate(model_a, test_loader, augment_fn=lambda x: randomizer(x))
print(f'  Clean MNE: {metrics_a_clean["mne"]:.6f}m')
print(f'  Noisy MNE: {metrics_a_noisy["mne"]:.6f}m')

# ═══════════════════════════════════════════════════════════════
# 训练 Model B: 域随机化
# ═══════════════════════════════════════════════════════════════

print('\n=== Model B: 域随机化训练 ===')
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

        # 域随机化增强
        aug_img = torch.stack([randomizer(gt_img[b]) for b in range(gt_img.shape[0])])

        rendered = render_batch(model_b, aw)
        loss = F.mse_loss(rendered, aug_img)

        opt_b.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()
        total_loss += loss.item()
        n_b += 1
    sched_b.step()

    hist_b['loss'].append(total_loss / n_b)
    if (epoch + 1) % 10 == 0:
        m = evaluate(model_b, test_loader)
        hist_b['mne'].append((epoch, m['mne']))
        print(f'  Epoch {epoch+1}: loss={hist_b["loss"][-1]:.6f}, MNE={m["mne"]:.6f}m')
    else:
        print(f'  Epoch {epoch+1}: loss={hist_b["loss"][-1]:.6f}')

metrics_b_clean = evaluate(model_b, test_loader)
metrics_b_noisy = evaluate(model_b, test_loader, augment_fn=lambda x: randomizer(x))
print(f'  Clean MNE: {metrics_b_clean["mne"]:.6f}m')
print(f'  Noisy MNE: {metrics_b_noisy["mne"]:.6f}m')

# ═══════════════════════════════════════════════════════════════
# 鲁棒性测试：逐步增加噪声
# ═══════════════════════════════════════════════════════════════

print('\n--- 鲁棒性测试 ---')
noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2]
robust_a, robust_b = [], []

for noise_std in noise_levels:
    rand = DomainRandomizer(noise_std=noise_std, brightness_range=(1.0, 1.0),
                            contrast_range=(1.0, 1.0), bg_range=(0, 0))
    ma = evaluate(model_a, test_loader, augment_fn=lambda x: rand(x))
    mb = evaluate(model_b, test_loader, augment_fn=lambda x: rand(x))
    robust_a.append(ma['mne'])
    robust_b.append(mb['mne'])
    print(f'  Noise={noise_std:.2f}: Baseline={ma["mne"]:.6f}m, DR={mb["mne"]:.6f}m')

# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

# 1. 增强示例
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
sample = ds[0]
gt_img = sample[1].reshape(H, W)

axes[0, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

aug_configs = [
    ('+ Noise', DomainRandomizer(noise_std=0.1, brightness_range=(1,1), contrast_range=(1,1), bg_range=(0,0))),
    ('+ Brightness', DomainRandomizer(noise_std=0, brightness_range=(0.7,1.3), contrast_range=(1,1), bg_range=(0,0))),
    ('+ Background', DomainRandomizer(noise_std=0, brightness_range=(1,1), contrast_range=(1,1), bg_range=(0,0.2))),
    ('All Combined', DomainRandomizer()),
]

for i, (label, aug) in enumerate(aug_configs):
    aug_img = aug(sample[1]).numpy().reshape(H, W)
    axes[0, i + 1 if i < 3 else 0].imshow(aug_img, cmap='gray', vmin=0, vmax=1)

# Row 2: more samples
np.random.seed(42)
for i in range(4):
    aug_img = randomizer(sample[1]).numpy().reshape(H, W)
    axes[1, i].imshow(aug_img, cmap='gray', vmin=0, vmax=1)
    axes[1, i].set_title(f'Random Aug {i+1}'); axes[1, i].axis('off')

# Fix row 0 labels
for i, (label, _) in enumerate(aug_configs):
    col = i + 1 if i < 3 else 0
    axes[0, col].set_title(label); axes[0, col].axis('off')
axes[0, 0].set_title('Original'); axes[0, 0].axis('off')
axes[0, 3].imshow(randomizer(sample[1]).numpy().reshape(H, W), cmap='gray', vmin=0, vmax=1)
axes[0, 3].set_title('All Combined'); axes[0, 3].axis('off')

plt.suptitle('Domain Randomization Examples', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'augmentation_examples.png'), dpi=150)
plt.close()

# 2. 鲁棒性曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(noise_levels, robust_a, 'b-o', label='Baseline (no DR)', markersize=5)
axes[0].plot(noise_levels, robust_b, 'r-o', label='Domain Randomized', markersize=5)
axes[0].set_xlabel('Noise Level (σ)'); axes[0].set_ylabel('MNE (m)')
axes[0].set_title('Robustness to Gaussian Noise'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# 训练曲线
axes[1].plot(hist_a['loss'], label='Baseline')
axes[1].plot(hist_b['loss'], label='DR Trained')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].set_title('Training Loss'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# 柱状图: clean vs noisy
x = np.arange(2)
width = 0.3
axes[2].bar(x - 0.15, [metrics_a_clean['mne'], metrics_a_noisy['mne']],
            width, label='Baseline', color='steelblue', alpha=0.8)
axes[2].bar(x + 0.15, [metrics_b_clean['mne'], metrics_b_noisy['mne']],
            width, label='DR Trained', color='salmon', alpha=0.8)
axes[2].set_xticks(x); axes[2].set_xticklabels(['Clean', 'Noisy'])
axes[2].set_ylabel('MNE (m)'); axes[2].set_title('Clean vs Noisy Test')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('Domain Randomization: Robustness Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'robustness.png'), dpi=150)
plt.close()

# 保存摘要
with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== 方向4: 域随机化 ===\n\n')
    f.write('Baseline (无 DR):\n')
    f.write(f'  Clean MNE: {metrics_a_clean["mne"]:.6f} m\n')
    f.write(f'  Noisy MNE: {metrics_a_noisy["mne"]:.6f} m\n')
    f.write(f'  Degradation: {(metrics_a_noisy["mne"]/metrics_a_clean["mne"]-1)*100:.1f}%\n\n')
    f.write('Domain Randomized:\n')
    f.write(f'  Clean MNE: {metrics_b_clean["mne"]:.6f} m\n')
    f.write(f'  Noisy MNE: {metrics_b_noisy["mne"]:.6f} m\n')
    f.write(f'  Degradation: {(metrics_b_noisy["mne"]/max(metrics_b_clean["mne"],1e-8)-1)*100:.1f}%\n\n')
    f.write('鲁棒性 (不同噪声水平):\n')
    for nl, ra, rb in zip(noise_levels, robust_a, robust_b):
        f.write(f'  σ={nl:.2f}: Baseline={ra:.6f}m, DR={rb:.6f}m\n')

print(f'\n=== 完成 ===')
print(f'结果: {args.output_dir}/')
