#!/usr/bin/env python3
"""
方向3: 多相机分析 — 评估不同视角的渲染和骨架预测

分析单视角的深度歧义问题，模拟双视角渲染，量化多视角约束的价值。

Usage:
    python scripts/experiments/exp3_multi_camera.py --gpu 0
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='data/seq_rr_3d')
parser.add_argument('--checkpoint', type=str, default='train_log/train_ms_scnf/exp_20260428_1/phase2/model/best_model.pt')
parser.add_argument('--output_dir', type=str, default='output/exp3_multi_camera')
parser.add_argument('--n_samples', type=int, default=30)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向3: 多相机分析 ===')
print(f'Device: {device}')

# ═══════════════════════════════════════════════════════════════
# Part 1: 加载模型和数据
# ═══════════════════════════════════════════════════════════════

from src.utils.model_loader import load_model
from src.data.dataset import SoftSequenceDataset
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays
from src.training.metrics_3d import mean_node_error, endpoint_error

# 加载数据
ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
loader = DataLoader(ds, batch_size=args.n_samples, shuffle=True)
batch = next(iter(loader))
action_window = batch[0].to(device)
gt_pos = batch[-1]  # (B, 3, 31)

# 加载模型（如果 checkpoint 存在）
model_loaded = False
if os.path.exists(args.checkpoint):
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']
    model_loaded = True
    print(f'模型已加载: {info["model_type"]} phase {info["phase"]}')
else:
    print(f'Checkpoint 不存在: {args.checkpoint}')
    print('将仅分析数据，不使用模型')

# ═══════════════════════════════════════════════════════════════
# Part 2: 深度歧义分析
# ═══════════════════════════════════════════════════════════════

print('\n--- Part 2: 深度歧义分析 ---')

# 从数据中提取 GT 骨架的统计信息
gt_skeletons = gt_pos.permute(0, 2, 1).numpy()  # (B, 31, 3)

# 分析 xy vs z 方向的变形量
base_pos = gt_skeletons[:, 0:1, :]  # (B, 1, 3)
displacement = gt_skeletons - base_pos  # 相对于 base 的位移
disp_x = displacement[:, :, 0]  # (B, 31)
disp_y = displacement[:, :, 1]
disp_z = displacement[:, :, 2]

# 计算各方向的变形幅度
range_x = disp_x.max(axis=1) - disp_x.min(axis=1)  # (B,)
range_y = disp_y.max(axis=1) - disp_y.min(axis=1)
range_z = disp_z.max(axis=1) - disp_z.min(axis=1)

print(f'  X 方向变形范围: mean={range_x.mean():.4f}, max={range_x.max():.4f} m')
print(f'  Y 方向变形范围: mean={range_y.mean():.4f}, max={range_y.max():.4f} m')
print(f'  Z 方向变形范围: mean={range_z.mean():.4f}, max={range_z.max():.4f} m')

# 正面相机看到的投影 (yz 平面)
front_view_x = gt_skeletons[:, :, 0]  # 相机看到的是 x 投影
front_view_z = gt_skeletons[:, :, 2]  # z 方向

# ═══════════════════════════════════════════════════════════════
# Part 3: 双视角渲染
# ═══════════════════════════════════════════════════════════════

print('\n--- Part 3: 双视角渲染 ---')

# 相机配置
cam_front = {
    'eye': np.array([1.5, 0.0, 0.5]),
    'center': np.array([0.0, 0.0, 0.25]),
    'up': np.array([0.0, 0.0, 1.0]),
    'label': 'Front (X-axis)',
}

cam_side = {
    'eye': np.array([0.0, 1.5, 0.5]),
    'center': np.array([0.0, 0.0, 0.25]),
    'up': np.array([0.0, 0.0, 1.0]),
    'label': 'Side (Y-axis)',
}

cam_top = {
    'eye': np.array([0.0, 0.0, 2.0]),
    'center': np.array([0.0, 0.0, 0.25]),
    'up': np.array([0.0, 1.0, 0.0]),
    'label': 'Top (Z-axis)',
}

cameras = [cam_front, cam_side, cam_top]

# 获取焦距
focal = ds.focal if hasattr(ds, 'focal') and ds.focal > 0 else 136.42
H, W = ds.H, ds.W
near, far = 0.5, 2.5
n_samples = 64

print(f'  Focal: {focal}, Image: {H}x{W}, Near/Far: {near}/{far}')

# 为每个相机生成射线
rays_dict = {}
for cam in cameras:
    rays_o, rays_d = get_rays(H, W, focal,
                               tuple(cam['eye']),
                               tuple(cam['center']),
                               tuple(cam['up']),
                               device=device)
    rays_dict[cam['label']] = (rays_o, rays_d)
    print(f'  {cam["label"]}: eye={cam["eye"]}, rays={rays_o.shape}')

if model_loaded:
    # 用模型从每个视角渲染
    print('\n  渲染中...')
    model.eval()

    n_render = min(5, action_window.shape[0])
    rendered_views = {}

    for cam_label, (rays_o, rays_d) in rays_dict.items():
        view_renders = []
        for i in range(n_render):
            aw = action_window[i:i+1]  # (1, 20, 2)
            pts, z_vals = sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=False)
            with torch.no_grad():
                # 分块渲染避免 OOM
                chunk_size = 2048
                chunks = []
                for start in range(0, H * W, chunk_size):
                    end = min(start + chunk_size, H * W)
                    raw = model(pts[start:end], aw)
                    raw = raw.reshape(1, end - start, n_samples, 2)
                    img_chunk, _ = OM_rendering(raw[0])
                    chunks.append(img_chunk)
                img = torch.cat(chunks)
            view_renders.append(img.cpu().numpy().reshape(H, W))
        rendered_views[cam_label] = view_renders

    # 预测骨架
    with torch.no_grad():
        pred_dict = model.predict_skeleton(action_window[:n_render])
    pred_skeletons = pred_dict['fine'].cpu().numpy()  # (n_render, 31, 3)
    gt_skel = gt_pos[:n_render].permute(0, 2, 1).numpy()

    # ═══════════════════════════════════════════════════════════
    # Part 4: 可视化
    # ═══════════════════════════════════════════════════════════

    # 1. 多视角渲染对比
    fig, axes = plt.subplots(n_render, 4, figsize=(16, 4 * n_render))
    if n_render == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_render):
        # GT 图像
        gt_img = batch[1][i].numpy().reshape(H, W)
        axes[i, 0].imshow(gt_img, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Sample {i}: GT (Front)')
        axes[i, 0].axis('off')

        for j, (cam_label, renders) in enumerate(rendered_views.items()):
            axes[i, j + 1].imshow(renders[i], cmap='gray')
            axes[i, j + 1].set_title(f'Pred {cam_label}')
            axes[i, j + 1].axis('off')

    plt.suptitle('Multi-View Rendering', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'multi_view_rendering.png'), dpi=150)
    plt.close()
    print(f'  保存: multi_view_rendering.png')

    # 2. 双视角骨架验证
    fig = plt.figure(figsize=(15, 5 * n_render))
    for i in range(min(n_render, 5)):
        # 正面投影 (xz)
        ax1 = fig.add_subplot(n_render, 3, i * 3 + 1)
        ax1.plot(gt_skel[i, :, 0], gt_skel[i, :, 2], 'b-o', markersize=3, label='GT')
        ax1.plot(pred_skeletons[i, :, 0], pred_skeletons[i, :, 2], 'r-o', markersize=2, label='Pred')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Z')
        ax1.set_title(f'Front View (XZ) - Sample {i}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 侧面投影 (yz)
        ax2 = fig.add_subplot(n_render, 3, i * 3 + 2)
        ax2.plot(gt_skel[i, :, 1], gt_skel[i, :, 2], 'b-o', markersize=3, label='GT')
        ax2.plot(pred_skeletons[i, :, 1], pred_skeletons[i, :, 2], 'r-o', markersize=2, label='Pred')
        ax2.set_xlabel('Y')
        ax2.set_ylabel('Z')
        ax2.set_title(f'Side View (YZ) - Sample {i}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3D 骨架
        ax3 = fig.add_subplot(n_render, 3, i * 3 + 3, projection='3d')
        ax3.plot(gt_skel[i, :, 0], gt_skel[i, :, 1], gt_skel[i, :, 2],
                 'b-o', linewidth=3, markersize=4, label='GT')
        ax3.plot(pred_skeletons[i, :, 0], pred_skeletons[i, :, 1], pred_skeletons[i, :, 2],
                 'r-o', linewidth=2, markersize=3, label='Pred')
        ax3.set_xlim(-0.1, 0.3)
        ax3.set_ylim(-0.15, 0.15)
        ax3.set_zlim(0, 0.55)
        mne_i = np.linalg.norm(pred_skeletons[i] - gt_skel[i], axis=1).mean()
        ax3.set_title(f'3D (MNE={mne_i:.4f}m)')
        ax3.legend()

    plt.suptitle('Skeleton: Front View / Side View / 3D', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'dual_view_skeleton.png'), dpi=150)
    plt.close()
    print(f'  保存: dual_view_skeleton.png')

    # 3. 深度歧义量化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    errors = pred_skeletons - gt_skel  # (N, 31, 3)
    err_x = errors[:, :, 0].flatten()
    err_y = errors[:, :, 1].flatten()
    err_z = errors[:, :, 2].flatten()

    for ax, err, label in zip(axes, [err_x, err_y, err_z], ['X Error', 'Y Error', 'Z Error']):
        ax.hist(err, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Count')
        ax.set_title(f'{label}\nmean={err.mean():.5f}, std={err.std():.5f}')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Axis Error Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'depth_ambiguity.png'), dpi=150)
    plt.close()
    print(f'  保存: depth_ambiguity.png')

    # ═══════════════════════════════════════════════════════════
    # Part 5: 统计分析
    # ═══════════════════════════════════════════════════════════

    # 逐轴误差统计
    axis_errors = {
        'X (front cam depth)': np.abs(errors[:, :, 0]),
        'Y (side cam depth)': np.abs(errors[:, :, 1]),
        'Z (vertical)': np.abs(errors[:, :, 2]),
    }

    print('\n  逐轴误差统计:')
    with open(os.path.join(args.output_dir, 'analysis.txt'), 'w') as f:
        f.write('=== 多相机分析结果 ===\n\n')
        f.write('相机配置:\n')
        for cam in cameras:
            f.write(f'  {cam["label"]}: eye={cam["eye"].tolist()}\n')
        f.write('\n')

        for axis_name, errs in axis_errors.items():
            mean_e = errs.mean()
            std_e = errs.std()
            max_e = errs.max()
            print(f'    {axis_name}: mean={mean_e:.5f}, std={std_e:.5f}, max={max_e:.5f} m')
            f.write(f'{axis_name}: mean={mean_e:.5f}, std={std_e:.5f}, max={max_e:.5f} m\n')

        f.write('\n逐轴误差占比:\n')
        total_err = sum(e.mean() for e in axis_errors.values())
        for axis_name, errs in axis_errors.items():
            ratio = errs.mean() / total_err * 100
            print(f'    {axis_name}: {ratio:.1f}%')
            f.write(f'  {axis_name}: {ratio:.1f}%\n')

        # 深度歧义结论
        f.write('\n结论:\n')
        x_ratio = axis_errors['X (front cam depth)'].mean() / total_err * 100
        y_ratio = axis_errors['Y (side cam depth)'].mean() / total_err * 100
        if x_ratio > 40 or y_ratio > 40:
            f.write('  存在明显的深度方向误差 → 多视角约束有价值\n')
        else:
            f.write('  深度方向误差较小 → 单视角可能已足够\n')

        # GT 骨架的各轴变形量
        f.write('\nGT 骨架各轴变形量:\n')
        f.write(f'  X range: mean={range_x.mean():.4f}m, max={range_x.max():.4f}m\n')
        f.write(f'  Y range: mean={range_y.mean():.4f}m, max={range_y.max():.4f}m\n')
        f.write(f'  Z range: mean={range_z.mean():.4f}m, max={range_z.max():.4f}m\n')

else:
    print('\n无模型，仅分析数据中的 3D 骨架变形特征...')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(min(5, len(gt_skeletons))):
        skel = gt_skeletons[i]
        axes[0].plot(skel[:, 0], skel[:, 2], '-o', markersize=2, alpha=0.5)
        axes[1].plot(skel[:, 1], skel[:, 2], '-o', markersize=2, alpha=0.5)
        axes[2] = fig.add_subplot(1, 3, 3, projection='3d')
        axes[2].plot(skel[:, 0], skel[:, 1], skel[:, 2], '-o', markersize=2, alpha=0.5)

    axes[0].set_xlabel('X'); axes[0].set_ylabel('Z')
    axes[0].set_title('Front View (XZ)')
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Y'); axes[1].set_ylabel('Z')
    axes[1].set_title('Side View (YZ)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('GT Skeleton from Multiple Views', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'gt_multi_view.png'), dpi=150)
    plt.close()
    print(f'  保存: gt_multi_view.png')

print(f'\n=== 完成 ===')
print(f'结果: {args.output_dir}/')
