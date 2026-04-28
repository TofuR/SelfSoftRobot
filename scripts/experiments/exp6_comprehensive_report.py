#!/usr/bin/env python3
"""
exp6: 全部实验结果汇总 + 自动生成对比报告

自动运行所有评估，生成综合对比图和报告文件。
包括: MS-SCNF baseline, exp1, exp1b, exp3, exp4b 等。

Usage:
    python scripts/experiments/exp6_comprehensive_report.py --gpu 0
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='data/seq_rr_3d')
parser.add_argument('--output_dir', type=str, default='output/exp6_report')
parser.add_argument('--checkpoint', type=str, default='train_log/train_ms_scnf/exp_20260428_1/phase2/model/best_model.pt')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== exp6: 综合实验报告 ===')

from src.data.dataset import SoftSequenceDataset
from src.utils.model_loader import load_model
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.models.model_ms_scnf import MSSCNFModel

ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
H, W = ds.H, ds.W
focal = ds.focal if hasattr(ds, 'focal') and ds.focal > 0 else 136.42
near, far, n_samples = 0.5, 2.5, 48

cam_params = ds.get_camera_params()
eye = tuple(cam_params['eye']) if cam_params else (1.5, 0.0, 0.5)
center = tuple(cam_params['center']) if cam_params else (0.0, 0.0, 0.25)
up = tuple(cam_params['up']) if cam_params else (0.0, 0.0, 1.0)
rays_o, rays_d = get_rays(H, W, focal, eye, center, up, device=device)

n_total = len(ds)
n_test_start = int(0.8 * n_total)
test_ds = torch.utils.data.Subset(ds, range(n_test_start, n_total))
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

def render_batch(model, aw, chunk_size=4096):
    B = aw.shape[0]
    pts, _ = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=False)
    parts = []
    for s in range(0, H * W, chunk_size):
        e = min(s + chunk_size, H * W)
        raw = model(pts[s:e], aw).reshape(B, e - s, n_samples, 2)
        parts.append(torch.stack([OM_rendering(raw[b])[0] for b in range(B)]))
    return torch.cat(parts, dim=1)

def evaluate_full(model, loader, name=''):
    model.eval()
    mnes, tips, psnrs = [], [], []
    n_eval = 0
    with torch.no_grad():
        for batch in loader:
            aw = batch[0].to(device)
            gt_pos = batch[-1].permute(0, 2, 1).to(device)
            gt_img = batch[1].to(device)

            pred = model.predict_skeleton(aw)['fine']
            err = (pred - gt_pos).norm(dim=-1)
            mnes.append(err.mean().item())
            tips.append(err[:, -1].mean().item())

            # 渲染质量 (前 10 批)
            if n_eval < 10:
                rendered = render_batch(model, aw)
                for b in range(aw.shape[0]):
                    mse = F.mse_loss(rendered[b], gt_img[b]).item()
                    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
                    psnrs.append(psnr)
            n_eval += 1

    result = {
        'mne': np.mean(mnes),
        'tip': np.mean(tips),
        'psnr': np.mean(psnrs) if psnrs else 0,
    }

    # 逐轴误差
    with torch.no_grad():
        batch = next(iter(loader))
        aw = batch[0].to(device)
        gt = batch[-1].permute(0, 2, 1).cpu().numpy()
        pred = model.predict_skeleton(aw)['fine'].cpu().numpy()
        result['err_x'] = np.abs(pred[:,:,0] - gt[:,:,0]).mean()
        result['err_y'] = np.abs(pred[:,:,1] - gt[:,:,1]).mean()
        result['err_z'] = np.abs(pred[:,:,2] - gt[:,:,2]).mean()

    if name:
        print(f'  {name}: MNE={result["mne"]:.4f}m, Tip={result["tip"]:.4f}m, PSNR={result["psnr"]:.2f}dB')
    return result

# ═══════════════════════════════════════════════════════════════
# 评估所有可用模型
# ═══════════════════════════════════════════════════════════════

all_results = {}

# 1. MS-SCNF baseline (3D GT supervised)
if os.path.exists(args.checkpoint):
    print('\n--- MS-SCNF Baseline ---')
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    all_results['MS-SCNF\n(3D GT)'] = evaluate_full(info['model'], test_loader, 'MS-SCNF')

# 2. Exp1: 纯 2D 骨架 (原版)
ckpt1 = 'output/exp1_skeleton_2d/model_2d_skeleton.pt'
if os.path.exists(ckpt1):
    print('\n--- Exp1: 纯 2D 骨架 ---')
    m1 = MSSCNFModel(action_dim=2, window_size=20, n_scales=4, hidden_dim=128,
                      d_filter=128, n_freqs=10, n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6).to(device)
    m1.load_state_dict(torch.load(ckpt1, map_location=device, weights_only=True))
    all_results['Exp1\n(2D only)'] = evaluate_full(m1, test_loader, 'Exp1 2D')

# 3. Exp1b: 改进 2D 骨架
ckpt1b = 'output/exp1b_improved_2d/model_improved_2d.pt'
if os.path.exists(ckpt1b):
    print('\n--- Exp1b: 改进 2D 骨架 ---')
    m1b = MSSCNFModel(action_dim=2, window_size=20, n_scales=4, hidden_dim=128,
                       d_filter=128, n_freqs=10, n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6).to(device)
    m1b.load_state_dict(torch.load(ckpt1b, map_location=device, weights_only=True))
    all_results['Exp1b\n(Improved)'] = evaluate_full(m1b, test_loader, 'Exp1b')

# 4. 早期模型 (C-MSTNF, ODE-CMSTNF 等)
early_models = {
    'train_log/train_cmstnf/exp_20260425_4/phase2/model/best_model.pt': 'C-MSTNF',
    'train_log/train_ode_cmstnf/exp_20260425_1/phase2/model/best_model.pt': 'ODE-CMSTNF',
    'train_log/train_smooth_cmstnf/exp_20260425_0/phase2/model/best_model.pt': 'Smooth',
    'train_log/train_mstnf/exp_20260424_4/model/best_model.pt': 'MSTNF',
}
for ckpt_path, label in early_models.items():
    if os.path.exists(ckpt_path):
        try:
            print(f'\n--- {label} ---')
            info = load_model(ckpt_path, data_dir=args.data_dir, device=device)
            all_results[label] = evaluate_full(info['model'], test_loader, label)
        except Exception as e:
            print(f'  跳过 {label}: {e}')

# ═══════════════════════════════════════════════════════════════
# 综合对比图
# ═══════════════════════════════════════════════════════════════

if not all_results:
    print('无可用模型结果，退出')
    sys.exit(0)

names = list(all_results.keys())
n_models = len(names)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. MNE 对比
mnes = [all_results[n]['mne'] for n in names]
colors = ['green' if m < 0.1 else 'orange' if m < 0.5 else 'red' for m in mnes]
bars = axes[0, 0].bar(range(n_models), mnes, color=colors, alpha=0.8)
axes[0, 0].set_xticks(range(n_models))
axes[0, 0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
axes[0, 0].set_ylabel('MNE (m)')
axes[0, 0].set_title('Mean Node Error (Lower = Better)')
axes[0, 0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mnes):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# 2. Tip Error
tips = [all_results[n]['tip'] for n in names]
axes[0, 1].bar(range(n_models), tips, color='steelblue', alpha=0.8)
axes[0, 1].set_xticks(range(n_models))
axes[0, 1].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
axes[0, 1].set_ylabel('Tip Error (m)')
axes[0, 1].set_title('End-Effector Error')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. PSNR
psnrs = [all_results[n]['psnr'] for n in names]
axes[0, 2].bar(range(n_models), psnrs, color='coral', alpha=0.8)
axes[0, 2].set_xticks(range(n_models))
axes[0, 2].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
axes[0, 2].set_ylabel('PSNR (dB)')
axes[0, 2].set_title('Rendering Quality (Higher = Better)')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# 4. 逐轴误差对比
x = np.arange(n_models)
width = 0.25
err_x = [all_results[n]['err_x'] for n in names]
err_y = [all_results[n]['err_y'] for n in names]
err_z = [all_results[n]['err_z'] for n in names]
axes[1, 0].bar(x - width, err_x, width, label='X', color='red', alpha=0.7)
axes[1, 0].bar(x, err_y, width, label='Y', color='green', alpha=0.7)
axes[1, 0].bar(x + width, err_z, width, label='Z', color='blue', alpha=0.7)
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(names, rotation=30, ha='right', fontsize=8)
axes[1, 0].set_ylabel('Error (m)')
axes[1, 0].set_title('Per-Axis Error')
axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3, axis='y')

# 5. 误差占比雷达图 (如果有 MS-SCNF baseline)
if 'MS-SCNF\n(3D GT)' in all_results:
    baseline = all_results['MS-SCNF\n(3D GT)']
    ratios = {}
    for name in names:
        r = all_results[name]
        ratios[name] = {
            'mne': r['mne'] / max(baseline['mne'], 1e-8),
            'tip': r['tip'] / max(baseline['tip'], 1e-8),
        }
    for name, ratio in ratios.items():
        axes[1, 1].barh(name, ratio['mne'], alpha=0.6, label=name if ratio['mne'] <= 2 else '')
    axes[1, 1].axvline(1.0, color='red', linestyle='--', alpha=0.5, label='MS-SCNF baseline')
    axes[1, 1].set_xlabel('MNE Ratio vs Baseline')
    axes[1, 1].set_title('Relative to MS-SCNF (Lower = Better)')
    axes[1, 1].grid(True, alpha=0.3)

# 6. 综合排名
metrics_for_rank = []
for name in names:
    r = all_results[name]
    metrics_for_rank.append({'name': name, 'mne': r['mne'], 'tip': r['tip'], 'psnr': r['psnr']})

# 按 MNE 排序
ranked = sorted(metrics_for_rank, key=lambda x: x['mne'])
table_data = [[r['name'].replace('\n', ' '), f'{r["mne"]:.4f}', f'{r["tip"]:.4f}', f'{r["psnr"]:.1f}'] for r in ranked]
axes[1, 2].axis('off')
table = axes[1, 2].table(cellText=table_data, colLabels=['Model', 'MNE (m)', 'Tip (m)', 'PSNR (dB)'],
                          loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.auto_set_column_width(range(4))
# 高亮第一行 (最佳)
for j in range(4):
    table[1, j].set_facecolor('#90EE90')
axes[1, 2].set_title('Model Rankings (by MNE)')

plt.suptitle('Comprehensive Experiment Report', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'comprehensive_report.png'), dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════
# 文本报告
# ═══════════════════════════════════════════════════════════════

with open(os.path.join(args.output_dir, 'report.txt'), 'w') as f:
    f.write('=' * 70 + '\n')
    f.write('         综合实验报告 (Auto-generated)\n')
    f.write('=' * 70 + '\n\n')

    f.write(f'{"Model":<18} {"MNE(m)":>10} {"Tip(m)":>10} {"PSNR(dB)":>10} {"X_err":>10} {"Y_err":>10} {"Z_err":>10}\n')
    f.write('-' * 70 + '\n')

    for name in [r['name'] for r in ranked]:
        r = all_results[name]
        short_name = name.replace('\n', ' ')
        f.write(f'{short_name:<18} {r["mne"]:>10.4f} {r["tip"]:>10.4f} {r["psnr"]:>10.2f} '
                f'{r["err_x"]:>10.5f} {r["err_y"]:>10.5f} {r["err_z"]:>10.5f}\n')

    f.write('\n' + '=' * 70 + '\n')
    f.write('关键发现:\n\n')

    # 自动生成发现
    best = ranked[0]
    worst = ranked[-1]
    f.write(f'1. 最佳模型: {best["name"].replace(chr(10), " ")} (MNE={best["mne"]:.4f}m)\n')
    f.write(f'2. 最差模型: {worst["name"].replace(chr(10), " ")} (MNE={worst["mne"]:.4f}m)\n')
    f.write(f'3. 最佳/最差差距: {worst["mne"]/max(best["mne"],1e-8):.1f}x\n\n')

    # 检查 3D GT vs 2D
    gt_models = [n for n in names if '3D GT' in n or 'MS-SCNF' in n]
    models_2d = [n for n in names if '2D' in n or 'Exp1' in n]
    if gt_models and models_2d:
        gt_mne = all_results[gt_models[0]]['mne']
        d2_mne = min(all_results[n]['mne'] for n in models_2d)
        f.write(f'4. 3D GT vs 最佳 2D: {gt_mne:.4f}m vs {d2_mne:.4f}m ({d2_mne/gt_mne:.1f}x 差距)\n')
        f.write(f'   → 3D GT 监督对骨架精度至关重要\n\n')

    # 检查深度歧义
    f.write('5. 逐轴误差分析:\n')
    for name in names[:5]:
        r = all_results[name]
        total = r['err_x'] + r['err_y'] + r['err_z']
        if total > 0:
            f.write(f'   {name.replace(chr(10), " "):>16}: '
                    f'X={r["err_x"]/total*100:.0f}% Y={r["err_y"]/total*100:.0f}% Z={r["err_z"]/total*100:.0f}%\n')

    f.write('\n改进方向:\n')
    f.write('  1. 增加多视角约束缓解深度歧义\n')
    f.write('  2. 渐进式训练策略改善 2D-only 收敛\n')
    f.write('  3. 时序编码改进 (HA-EMA) 捕捉迟滞\n')
    f.write('  4. 域随机化提升鲁棒性用于 sim-to-real\n')

print(f'\n=== 报告完成 ===')
print(f'结果: {args.output_dir}/')
print(f'  comprehensive_report.png')
print(f'  report.txt')
