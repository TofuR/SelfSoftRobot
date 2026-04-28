#!/usr/bin/env python3
"""
改进 exp4: 修复域随机化评估 + 更严格的鲁棒性测试

修复:
  1. 原 exp4 的噪声评估只测 MNE (与图像无关), 所以所有噪声水平结果相同
  2. 改为评估渲染质量 (PSNR/MSE), 直接测噪声对渲染的影响
  3. 增加: 不同强度的 DR 训练对比

Usage:
    python scripts/experiments/exp4b_fixed_dr_eval.py --gpu 0
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
parser.add_argument('--output_dir', type=str, default='output/exp4b_fixed_dr')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--n_render_samples', type=int, default=32)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 改进 exp4: 修复域随机化评估 ===')

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

# ── 域随机化函数 ──
class DomainRandomizer:
    def __init__(self, noise_std=0.05, brightness=(0.7, 1.3), contrast=(0.7, 1.3), bg=(0, 0.15)):
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast
        self.bg = bg

    def __call__(self, img):
        out = img.clone()
        bg_val = np.random.uniform(*self.bg)
        out[out < 0.5] = bg_val
        out = out + torch.randn_like(out) * self.noise_std
        out = out * np.random.uniform(*self.brightness)
        c = np.random.uniform(*self.contrast)
        out = (out - out.mean()) * c + out.mean()
        return out.clamp(0, 1)

# ── 公共函数 ──
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

def train_model(model, loader, epochs, randomizer=None):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    for epoch in range(epochs):
        model.train()
        total = 0
        n = 0
        for batch in loader:
            aw = batch[0].to(device)
            gt_img = batch[1].to(device)
            target = torch.stack([randomizer(gt_img[b]) for b in range(gt_img.shape[0])]) if randomizer else gt_img
            rendered = render_batch(model, aw)
            loss = F.mse_loss(rendered, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        losses.append(total / n)
        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}: loss={losses[-1]:.6f}')
    return losses

# ── 关键修复: 评估渲染质量 (而非 MNE) ──
def evaluate_rendering_quality(model, loader, randomizer=None, n_eval=20):
    """评估渲染质量: PSNR 和 MSE, 可选加噪。"""
    model.eval()
    mses, psnrs = [], []
    count = 0
    with torch.no_grad():
        for batch in loader:
            if count >= n_eval:
                break
            aw = batch[0].to(device)
            gt_img = batch[1].to(device)
            # 对 GT 加噪
            target = torch.stack([randomizer(gt_img[b]) for b in range(gt_img.shape[0])]) if randomizer else gt_img
            rendered = render_batch(model, aw, chunk_size=4096)
            for b in range(aw.shape[0]):
                if count >= n_eval:
                    break
                pred = rendered[b].cpu().numpy()
                gt = target[b].cpu().numpy()
                mse = np.mean((pred - gt) ** 2)
                psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
                mses.append(mse)
                psnrs.append(psnr)
                count += 1
    return {'mse': np.mean(mses), 'psnr': np.mean(psnrs)}

def evaluate_skeleton(model, loader):
    model.eval()
    mnes = []
    with torch.no_grad():
        for batch in loader:
            aw = batch[0].to(device)
            gt = batch[-1].permute(0, 2, 1).to(device)
            pred = model.predict_skeleton(aw)['fine']
            mnes.append((pred - gt).norm(dim=-1).mean().item())
    return np.mean(mnes)

# ═══════════════════════════════════════════════════════════════
# 训练: 3 种配置
# ═══════════════════════════════════════════════════════════════

configs = {
    'no_dr': None,
    'dr_light': DomainRandomizer(noise_std=0.03, brightness=(0.9, 1.1), contrast=(0.9, 1.1), bg=(0, 0.05)),
    'dr_heavy': DomainRandomizer(noise_std=0.1, brightness=(0.7, 1.3), contrast=(0.7, 1.3), bg=(0, 0.2)),
}

trained_models = {}
train_histories = {}
for name, rand in configs.items():
    print(f'\n  训练 {name}...')
    m = make_model()
    losses = train_model(m, train_loader, args.epochs, randomizer=rand)
    trained_models[name] = m
    train_histories[name] = losses

# ═══════════════════════════════════════════════════════════════
# 鲁棒性评估: 对渲染图像加不同噪声 → 测 PSNR
# ═══════════════════════════════════════════════════════════════

print('\n--- 鲁棒性评估 (渲染质量) ---')
noise_levels = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
results = {name: {'psnr': [], 'mse': [], 'mne': []} for name in configs}

for noise_std in noise_levels:
    eval_rand = DomainRandomizer(noise_std=noise_std, brightness=(1,1), contrast=(1,1), bg=(0,0))
    for name, model in trained_models.items():
        q = evaluate_rendering_quality(model, test_loader, randomizer=eval_rand if noise_std > 0 else None)
        results[name]['psnr'].append(q['psnr'])
        results[name]['mse'].append(q['mse'])
        mne = evaluate_skeleton(model, test_loader)
        results[name]['mne'].append(mne)
    print(f'  Noise σ={noise_std:.2f}: ' +
          ' | '.join(f'{n} PSNR={results[n]["psnr"][-1]:.2f}dB' for n in configs))

# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. PSNR vs 噪声水平
for name in configs:
    axes[0, 0].plot(noise_levels, results[name]['psnr'], '-o', markersize=4, label=name)
axes[0, 0].set_xlabel('Noise σ'); axes[0, 0].set_ylabel('PSNR (dB)')
axes[0, 0].set_title('Rendering Quality vs Input Noise'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

# 2. MSE vs 噪声
for name in configs:
    axes[0, 1].plot(noise_levels, results[name]['mse'], '-o', markersize=4, label=name)
axes[0, 1].set_xlabel('Noise σ'); axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Render MSE vs Noise'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

# 3. MNE 对比 (骨架精度, 应该不受噪声影响)
for name in configs:
    axes[0, 2].plot(noise_levels, results[name]['mne'], '-o', markersize=4, label=name)
axes[0, 2].set_xlabel('Noise σ'); axes[0, 2].set_ylabel('MNE (m)')
axes[0, 2].set_title('Skeleton MNE (should be flat)'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

# 4. 训练曲线
for name, losses in train_histories.items():
    axes[1, 0].plot(losses, label=name)
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('Training Loss'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 5. 柱状图: Clean vs Heavy noise PSNR
x = np.arange(len(configs))
width = 0.3
clean_psnr = [results[n]['psnr'][0] for n in configs]
noisy_psnr = [results[n]['psnr'][4] for n in configs]  # σ=0.1
axes[1, 1].bar(x - width/2, clean_psnr, width, label='Clean', color='steelblue', alpha=0.8)
axes[1, 1].bar(x + width/2, noisy_psnr, width, label='Noisy (σ=0.1)', color='salmon', alpha=0.8)
axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(list(configs.keys()))
axes[1, 1].set_ylabel('PSNR (dB)'); axes[1, 1].set_title('Clean vs Noisy Rendering')
axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

# 6. PSNR 下降率
for name in configs:
    drop = [(results[name]['psnr'][0] - p) for p in results[name]['psnr']]
    axes[1, 2].plot(noise_levels, drop, '-o', markersize=4, label=name)
axes[1, 2].set_xlabel('Noise σ'); axes[1, 2].set_ylabel('PSNR Drop (dB)')
axes[1, 2].set_title('PSNR Degradation'); axes[1, 2].legend(); axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Domain Randomization: Fixed Evaluation (Rendering Quality)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'results.png'), dpi=150)
plt.close()

# 保存摘要
with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
    f.write('=== 改进 exp4: 域随机化 (修复评估) ===\n\n')
    f.write('评估指标: 渲染 PSNR (dB), 而非骨架 MNE\n\n')
    f.write(f'{"Config":<10} {"Clean PSNR":>12} {"PSNR@σ=0.1":>12} {"Drop":>8} {"MNE":>10}\n')
    f.write('-' * 55 + '\n')
    for name in configs:
        clean = results[name]['psnr'][0]
        noisy = results[name]['psnr'][4]
        drop = clean - noisy
        mne = results[name]['mne'][0]
        f.write(f'{name:<10} {clean:>12.2f} {noisy:>12.2f} {drop:>8.2f} {mne:>10.6f}\n')

print(f'\n结果保存: {args.output_dir}/')
