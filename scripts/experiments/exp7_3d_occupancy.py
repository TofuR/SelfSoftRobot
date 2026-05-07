#!/usr/bin/env python3
"""
exp7_3d_occupancy.py — 查询模型密度场，生成 3D 占据点云 + 2D 投影对比 + GIF 动画。

占据判断模式 (--mode):
  1: combined = alpha * max(visibility, 0) > threshold    (乘积组合)
  2: alpha > threshold  OR  visibility > vis-threshold    (任一超阈值)
  3: alpha > threshold  AND visibility > vis-threshold    (两者都超阈值)

GT 对比: 从 positions + radii 生成管表面点云

Usage:
    # 首次运行：查询空间并缓存结果
    python scripts/experiments/exp7_3d_occupancy.py --gpu 0

    # Mode 1: 乘积组合 (默认)
    python scripts/experiments/exp7_3d_occupancy.py --skip-query --threshold 0.3

    # Mode 2: 任一超阈值
    python scripts/experiments/exp7_3d_occupancy.py --skip-query --mode 2 --threshold 0.9 --vis-threshold 0.1

    # Mode 3: 两者都超阈值
    python scripts/experiments/exp7_3d_occupancy.py --skip-query --mode 3 --threshold 0.9 --vis-threshold 0.1

    # 生成 GIF
    python scripts/experiments/exp7_3d_occupancy.py --skip-query --gif
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description="Exp7: 3D Occupancy Visualization")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data-dir', type=str, default='data/exp7_multiview')
parser.add_argument('--model-path', type=str, default='output/exp7_multiview_2d/model_final.pt')
parser.add_argument('--output-dir', type=str, default='output/exp7_3d_occupancy')
parser.add_argument('--resolution', type=int, default=40, help='3D 网格每轴分辨率')
parser.add_argument('--mode', type=int, default=1, choices=[1, 2, 3],
                    help='占据判断模式: 1=乘积组合, 2=任一超阈值, 3=两者都超阈值')
parser.add_argument('--threshold', type=float, default=0.3,
                    help='mode1: combined score阈值; mode2/3: alpha阈值')
parser.add_argument('--vis-threshold', type=float, default=0.5,
                    help='mode2/3: visibility阈值')
parser.add_argument('--skip-query', action='store_true', help='跳过模型查询，使用缓存结果')
parser.add_argument('--n-frames', type=int, default=8, help='GIF 动画帧数')
parser.add_argument('--seq-idx', type=int, default=0, help='使用哪个序列的数据')
parser.add_argument('--gif', action='store_true', help='是否生成 GIF 动画')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

# ============================================================================
# 加载数据和模型
# ============================================================================
print('=' * 60)
print('Exp7: 3D Occupancy Visualization')
print('=' * 60)

from src.data.dataset_multiview import MultiViewDataset
from src.models.model_ms_scnf import MSSCNFModel

print('\n>>> 加载数据集...')
ds = MultiViewDataset(args.data_dir, seq_len=20, return_3d=True)
H, W = ds.H, ds.W
cameras = ds.cameras
print(f'    H={H}, W={W}, cameras: front={cameras[0]["eye"]}, side={cameras[1]["eye"]}')

# 加载原始 npz 以获取 radii
npz_files = sorted(__import__('glob').glob(os.path.join(args.data_dir, '*.npz')))
raw_npz = np.load(npz_files[args.seq_idx])
gt_radii_all = raw_npz['radii']  # (T, 30)
gt_positions_all = raw_npz['positions']  # (T, 3, 31)

print('\n>>> 加载模型...')
model = MSSCNFModel(
    action_dim=ds.action_dim, window_size=20, n_scales=4,
    hidden_dim=128, d_filter=128, n_freqs=10,
    n_coarse=4, n_medium=10, n_fine=31, deform_n_freqs=6,
).to(device)
state_dict = torch.load(args.model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f'    模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# ============================================================================
# 选取测试帧
# ============================================================================
seq_samples = [(i, s) for i, s in enumerate(ds.samples) if s[0] == args.seq_idx]
if not seq_samples:
    args.seq_idx = 0
    seq_samples = [(i, s) for i, s in enumerate(ds.samples) if s[0] == args.seq_idx]

n_avail = len(seq_samples)
step = max(1, n_avail // args.n_frames)
selected_indices = [seq_samples[i][0] for i in range(0, n_avail, step)][:args.n_frames]
print(f'    序列 {args.seq_idx} 共 {n_avail} 帧，选取 {len(selected_indices)} 帧')


# ============================================================================
# 定义 3D 空间网格
# ============================================================================
x_range = (-0.15, 0.15)
y_range = (-0.15, 0.15)
z_range = (-0.02, 0.55)

res = args.resolution
xs = np.linspace(*x_range, res)
ys = np.linspace(*y_range, res)
zs = np.linspace(*z_range, res)

grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
grid_flat = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
n_points = len(grid_flat)
print(f'\n>>> 3D 网格: {res}^3 = {n_points} 个查询点')


# ============================================================================
# 查询模型密度场（带缓存）
# ============================================================================
cache_dir = os.path.join(args.output_dir, 'cache')
os.makedirs(cache_dir, exist_ok=True)

def get_cache_path(frame_idx):
    return os.path.join(cache_dir, f'occupancy_frame{frame_idx:04d}_res{res}.npz')

def query_frame(frame_idx, sample_idx):
    cache_path = get_cache_path(frame_idx)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['visibility'].reshape(res, res, res), data['density'].reshape(res, res, res)

    print(f'    帧 {frame_idx}: 查询模型...')
    batch = ds[sample_idx]
    aw = batch[0].unsqueeze(0).to(device)

    visibility_all = np.zeros(n_points, dtype=np.float32)
    density_all = np.zeros(n_points, dtype=np.float32)

    chunk_size = 4096
    with torch.no_grad():
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            pts = torch.tensor(grid_flat[start:end], dtype=torch.float32, device=device)
            pts = pts.unsqueeze(1)  # (chunk, 1, 3)
            raw = model(pts, aw).squeeze(1).cpu().numpy()  # (chunk, 2)
            visibility_all[start:end] = raw[:, 0]
            density_all[start:end] = raw[:, 1]

    vis_3d = visibility_all.reshape(res, res, res)
    den_3d = density_all.reshape(res, res, res)

    np.savez_compressed(cache_path, visibility=vis_3d, density=den_3d)
    return vis_3d, den_3d

print(f'\n>>> 查询密度场 (skip_query={args.skip_query})...')
frame_data = []
for fi, sample_idx in enumerate(selected_indices):
    if args.skip_query:
        cache_path = get_cache_path(fi)
        if not os.path.exists(cache_path):
            print(f'    缓存不存在: {cache_path}，请先不带 --skip-query 运行')
            continue
        data = np.load(cache_path)
        vis_3d = data['visibility'].reshape(res, res, res)
        den_3d = data['density'].reshape(res, res, res)
    else:
        vis_3d, den_3d = query_frame(fi, sample_idx)
    frame_data.append({'vis': vis_3d, 'den': den_3d, 'sample_idx': sample_idx})

if not frame_data:
    print('没有可用帧数据，退出')
    sys.exit(1)
print(f'    共加载 {len(frame_data)} 帧数据')


# ============================================================================
# 工具函数
# ============================================================================

MODE_NAMES = {
    1: 'combined = alpha * max(vis, 0)',
    2: 'alpha > T AND/OR visibility > Tv',
    3: 'alpha > T AND visibility > Tv',
}

def compute_alpha(den_3d):
    """density raw → alpha = 1 - exp(-softplus(density))"""
    return 1.0 - np.exp(-F.softplus(torch.tensor(den_3d)).numpy())

def compute_occupancy_score(vis_3d, den_3d, mode, threshold, vis_threshold):
    """根据 mode 计算占据分数和 mask。

    Returns:
        score: (res, res, res) 逐点分数（mode1 为乘积值, mode2/3 为 alpha）
        mask:  (res, res, res) bool，占据判定
    """
    alpha = compute_alpha(den_3d)

    if mode == 1:
        score = alpha * np.maximum(vis_3d, 0)
        mask = score > threshold
    elif mode == 2:
        mask_alpha = alpha > threshold
        mask_vis = vis_3d > vis_threshold
        mask = mask_alpha | mask_vis
        score = alpha
    elif mode == 3:
        mask_alpha = alpha > threshold
        mask_vis = vis_3d > vis_threshold
        mask = mask_alpha & mask_vis
        score = alpha

    return score, mask

def generate_gt_surface(positions, radii, n_circ=8):
    """从中心线节点 + 半径生成管表面点云。

    Args:
        positions: (3, 31) 中心线节点坐标
        radii: (30,) 每段半径
        n_circ: 每段圆周采样点数

    Returns:
        pts: (N, 3) 表面点云
    """
    centers = positions.T  # (31, 3)
    pts_list = []

    for i in range(len(radii)):
        seg = centers[i + 1] - centers[i]
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-8:
            continue

        seg_dir = seg / seg_len

        # 构建局部坐标系
        if abs(seg_dir[2]) < 0.99:
            perp1 = np.cross(seg_dir, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(seg_dir, np.array([1, 0, 0]))
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(seg_dir, perp1)

        r = radii[i]
        # 在段的两个端点附近各采一圈
        for t_offset in [0.0, 0.5, 1.0]:
            center = centers[i] + t_offset * seg
            for j in range(n_circ):
                theta = 2 * np.pi * j / n_circ
                pt = center + r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                pts_list.append(pt)

    return np.array(pts_list)

def extract_occupancy(vis_3d, den_3d, mode, threshold, vis_threshold):
    """用指定模式提取占据点。"""
    score, mask = compute_occupancy_score(vis_3d, den_3d, mode, threshold, vis_threshold)
    if not mask.any():
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)
    coords = np.stack([grid_x[mask], grid_y[mask], grid_z[mask]], axis=-1)
    values = score[mask]
    alphas = compute_alpha(den_3d)[mask]
    return coords, values, alphas


# ============================================================================
# Step 1: 分布图
# ============================================================================
print('\n>>> 生成分布图...')

all_vis = np.concatenate([f['vis'].ravel() for f in frame_data])
all_den = np.concatenate([f['den'].ravel() for f in frame_data])
all_alpha = compute_alpha(all_den.reshape(1, 1, -1)).ravel()

# 根据当前 mode 计算分数和 mask（用于分布图）
_all_score, all_mask = compute_occupancy_score(
    all_vis.reshape(1, 1, -1), all_den.reshape(1, 1, -1),
    args.mode, args.threshold, args.vis_threshold)
all_score = _all_score.ravel()
all_mask = all_mask.ravel()
n_occupied = all_mask.sum()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# visibility
axes[0, 0].hist(all_vis, bins=100, color='#5DADE2', alpha=0.85)
axes[0, 0].set_title('Visibility Distribution')
axes[0, 0].set_xlabel('Visibility'); axes[0, 0].set_ylabel('Count')

# alpha (density)
axes[0, 1].hist(all_alpha, bins=100, color='#F0B27A', alpha=0.85)
axes[0, 1].set_title('Alpha Distribution (from density)')
axes[0, 1].set_xlabel('Alpha'); axes[0, 1].set_ylabel('Count')

# score distribution
mode_label = {1: 'alpha * max(vis, 0)', 2: 'alpha (OR vis)', 3: 'alpha (AND vis)'}
axes[0, 2].hist(all_score, bins=100, color='#82E0AA', alpha=0.85)
axes[0, 2].set_title(f'Score Distribution (mode {args.mode}: {mode_label[args.mode]})')
axes[0, 2].set_xlabel('Score')
axes[0, 2].set_ylabel('Count')
axes[0, 2].axvline(args.threshold, color='red', linestyle='--', linewidth=2,
                    label=f'threshold={args.threshold}')
if args.mode in (2, 3):
    axes[0, 2].axvline(args.vis_threshold, color='blue', linestyle=':', linewidth=1.5,
                        label=f'vis_threshold={args.vis_threshold}')
axes[0, 2].legend(fontsize=9)

# visibility vs alpha 散点
idx_s = np.random.choice(len(all_vis), min(10000, len(all_vis)), replace=False)
sc = axes[1, 0].scatter(all_alpha[idx_s], all_vis[idx_s], s=1, alpha=0.3,
                         c=all_score[idx_s], cmap='YlOrRd')
axes[1, 0].set_title('Alpha vs Visibility (color=score)')
axes[1, 0].set_xlabel('Alpha')
axes[1, 0].set_ylabel('Visibility')
if args.mode in (2, 3):
    axes[1, 0].axhline(args.vis_threshold, color='blue', linestyle=':', alpha=0.7,
                        label=f'vis_T={args.vis_threshold}')
    axes[1, 0].axvline(args.threshold, color='red', linestyle='--', alpha=0.7,
                        label=f'alpha_T={args.threshold}')
    axes[1, 0].legend(fontsize=7)
plt.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04, label='score')

# score CDF
sorted_score = np.sort(all_score)
cdf = np.arange(1, len(sorted_score) + 1) / len(sorted_score)
axes[1, 1].plot(sorted_score, cdf, color='#2ECC71', linewidth=2)
axes[1, 1].axvline(args.threshold, color='red', linestyle='--', linewidth=2,
                    label=f'threshold={args.threshold}')
pct = all_mask.mean() * 100
axes[1, 1].set_title(f'Score CDF ({pct:.1f}% occupied)')
axes[1, 1].set_xlabel('Score')
axes[1, 1].set_ylabel('CDF')
axes[1, 1].legend(fontsize=9)

# 各阈值下的占据点数量
score_max = all_score.max() if all_score.max() > 0 else 1.0
thresholds = np.linspace(0, score_max, 50)
n_occ = [np.sum(all_score > t) for t in thresholds]
axes[1, 2].plot(thresholds, n_occ, color='#2ECC71', linewidth=2)
axes[1, 2].axvline(args.threshold, color='red', linestyle='--', linewidth=2,
                    label=f'current={args.threshold}')
axes[1, 2].set_title('Occupied Points vs Threshold')
axes[1, 2].set_xlabel('Threshold')
axes[1, 2].set_ylabel('N Occupied')
axes[1, 2].legend(fontsize=9)

plt.suptitle(f'Density Field Distribution — mode {args.mode}: {mode_label[args.mode]}', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'distribution.png'), dpi=150)
plt.close()
print(f'    分布图已保存')


# ============================================================================
# Step 2: 3D 点云 + GT 管表面可视化
# ============================================================================
print('\n>>> 生成 3D 点云可视化...')

from src.utils.skeleton_2d import project_3d_to_2d

# 配色方案
PRED_CMAP = cm.get_cmap('YlOrRd')  # 预测点云：浅黄→橙红
PRED_PT_SIZE = 1.5
PRED_ALPHA = 0.35
GT_PT_SIZE = 1.0
GT_PT_ALPHA = 0.25

def plot_3d_frame(ax, pred_coords, pred_scores, gt_surface_pts, gt_skel, pred_skel,
                  title='', elev=25, azim=45):
    ax.cla()

    # GT 管表面点云（浅蓝色）
    if gt_surface_pts is not None and len(gt_surface_pts) > 0:
        ax.scatter(gt_surface_pts[:, 0], gt_surface_pts[:, 1], gt_surface_pts[:, 2],
                   c='#85C1E9', s=GT_PT_SIZE, alpha=GT_PT_ALPHA, depthshade=True,
                   label='GT surface')

    # 预测占据点（浅黄→橙红，按分数着色）
    if len(pred_coords) > 0:
        vmin, vmax = np.percentile(pred_scores, [10, 95])
        norm = Normalize(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin + 1e-6))
        colors = PRED_CMAP(norm(pred_scores))
        ax.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                   c=colors, s=PRED_PT_SIZE, alpha=PRED_ALPHA, depthshade=True,
                   label='Predicted')

    # GT 中心线
    if gt_skel is not None:
        ax.plot(gt_skel[:, 0], gt_skel[:, 1], gt_skel[:, 2],
                color='#2471A3', linewidth=2, marker='o', markersize=2,
                label='GT center', alpha=0.9)

    # 预测中心线
    if pred_skel is not None:
        ax.plot(pred_skel[:, 0], pred_skel[:, 1], pred_skel[:, 2],
                color='#E74C3C', linewidth=2, marker='^', markersize=2,
                label='Pred center', alpha=0.9)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_xlim(*x_range); ax.set_ylim(*y_range); ax.set_zlim(*z_range)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(fontsize=7, loc='upper left', markerscale=3)

def plot_2d_projection(ax, pred_coords, gt_img, gt_skel_2d, pred_skel_2d,
                       camera, title='', gt_surface_pts=None):
    ax.cla()
    ax.imshow(gt_img.reshape(H, W), cmap='gray', alpha=0.4, vmin=0, vmax=1)

    # GT 管表面投影（浅蓝色）
    if gt_surface_pts is not None and len(gt_surface_pts) > 0:
        pts_t = torch.tensor(gt_surface_pts, dtype=torch.float32)
        proj = project_3d_to_2d(pts_t, camera['eye'], camera['center'],
                                camera['up'], camera['focal'], H, W).numpy()
        valid = (proj[:, 0] >= 0) & (proj[:, 0] < W) & \
                (proj[:, 1] >= 0) & (proj[:, 1] < H)
        if valid.any():
            ax.scatter(proj[valid, 0], proj[valid, 1], c='#85C1E9', s=0.3, alpha=0.15)

    # 预测占据点投影（橙色）
    if pred_coords is not None and len(pred_coords) > 0:
        pts_t = torch.tensor(pred_coords, dtype=torch.float32)
        proj = project_3d_to_2d(pts_t, camera['eye'], camera['center'],
                                camera['up'], camera['focal'], H, W).numpy()
        valid = (proj[:, 0] >= 0) & (proj[:, 0] < W) & \
                (proj[:, 1] >= 0) & (proj[:, 1] < H)
        if valid.any():
            ax.scatter(proj[valid, 0], proj[valid, 1], c='#F39C12', s=0.3, alpha=0.15)

    if gt_skel_2d is not None:
        ax.plot(gt_skel_2d[:, 0], gt_skel_2d[:, 1], color='#2471A3',
                linewidth=1.5, label='GT')
    if pred_skel_2d is not None:
        ax.plot(pred_skel_2d[:, 0], pred_skel_2d[:, 1], color='#E74C3C',
                linewidth=1.5, label='Pred')

    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.axis('off')


# ---- 第一帧详细图 ----
fi = 0
fd = frame_data[fi]
batch = ds[fd['sample_idx']]
gt_3d = batch[-1].numpy().T  # (31, 3)

# GT 管表面
_, t_in_seq = ds.samples[fd['sample_idx']]
gt_radii = gt_radii_all[t_in_seq]
gt_surface = generate_gt_surface(batch[-1].numpy(), gt_radii)

with torch.no_grad():
    aw = batch[0].unsqueeze(0).to(device)
    pred_skel = model.predict_skeleton(aw)['fine'][0].cpu().numpy()

pred_coords, pred_scores, pred_alphas = extract_occupancy(
    fd['vis'], fd['den'], args.mode, args.threshold, args.vis_threshold)
print(f'    帧 0: mode={args.mode}, threshold={args.threshold}, 占据点={len(pred_coords)}/{n_points}')

fig = plt.figure(figsize=(28, 16))

# Row 1: 3D 视角
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
plot_3d_frame(ax1, pred_coords, pred_scores, gt_surface, gt_3d, pred_skel,
              'Front View', elev=20, azim=-90)

ax2 = fig.add_subplot(3, 4, 2, projection='3d')
plot_3d_frame(ax2, pred_coords, pred_scores, gt_surface, gt_3d, pred_skel,
              'Side View', elev=20, azim=0)

ax3 = fig.add_subplot(3, 4, 3, projection='3d')
plot_3d_frame(ax3, pred_coords, pred_scores, gt_surface, gt_3d, pred_skel,
              'Top View', elev=80, azim=-90)

# MIP (XZ 平面)
ax4 = fig.add_subplot(3, 4, 4)
combined_3d, _ = compute_occupancy_score(fd['vis'], fd['den'], args.mode, args.threshold, args.vis_threshold)
mip_xz = combined_3d.max(axis=1)  # (res_x, res_z) squeeze y
im = ax4.imshow(mip_xz.T, extent=[*x_range, *z_range], origin='lower',
                cmap='YlOrRd', aspect='auto', vmin=0, vmax=mip_xz.max())
ax4.plot(gt_3d[:, 0], gt_3d[:, 2], color='#2471A3', linewidth=1.5,
         marker='o', markersize=2, label='GT')
ax4.plot(pred_skel[:, 0], pred_skel[:, 2], color='#E74C3C', linewidth=1.5,
         marker='^', markersize=2, label='Pred')
ax4.set_xlabel('X'); ax4.set_ylabel('Z')
ax4.set_title('MIP (XZ, squash Y)', fontsize=10)
ax4.legend(fontsize=8)
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

# Row 2: 2D 投影对比
gt_img_front = batch[1].numpy()
gt_img_side = batch[2].numpy()
gt_skel_2d_front = batch[3].numpy()
gt_skel_2d_side = batch[4].numpy()

pred_skel_t = torch.tensor(pred_skel, dtype=torch.float32)
pred_2d_front = project_3d_to_2d(pred_skel_t, cameras[0]['eye'], cameras[0]['center'],
                                  cameras[0]['up'], cameras[0]['focal'], H, W).numpy()
pred_2d_side = project_3d_to_2d(pred_skel_t, cameras[1]['eye'], cameras[1]['center'],
                                 cameras[1]['up'], cameras[1]['focal'], H, W).numpy()

ax5 = fig.add_subplot(3, 4, 5)
plot_2d_projection(ax5, pred_coords, gt_img_front, gt_skel_2d_front,
                   pred_2d_front, cameras[0], 'Front: GT surface (blue) + Pred (orange)',
                   gt_surface_pts=gt_surface)

ax6 = fig.add_subplot(3, 4, 6)
plot_2d_projection(ax6, pred_coords, gt_img_side, gt_skel_2d_side,
                   pred_2d_side, cameras[1], 'Side: GT surface (blue) + Pred (orange)',
                   gt_surface_pts=gt_surface)

# Row 2 右: 渲染对比
from src.utils.rendering import OM_rendering, sample_stratified
from src.utils.camera import get_rays

rays_list = []
for cam in cameras:
    ro, rd = get_rays(H, W, cam['focal'], cam['eye'], cam['center'], cam['up'], device=device)
    rays_list.append((ro, rd))

near, far, n_render_samples = 0.5, 2.5, 48
with torch.no_grad():
    rendered_imgs = []
    for ro, rd in rays_list:
        pts_render, _ = sample_stratified(ro, rd, near, far, n_render_samples, perturb=False)
        parts = []
        for s in range(0, H * W, 2048):
            e = min(s + 2048, H * W)
            raw = model(pts_render[s:e], aw).reshape(1, e - s, n_render_samples, 2)
            parts.append(OM_rendering(raw[0])[0])
        rendered_imgs.append(torch.cat(parts).cpu().numpy())

for vi, (vname, gt_img) in enumerate([('Front', gt_img_front), ('Side', gt_img_side)]):
    ax = fig.add_subplot(3, 4, 7 + vi)
    composite = np.zeros((H, W, 3))
    composite[:, :, 0] = gt_img.reshape(H, W)
    composite[:, :, 2] = rendered_imgs[vi].reshape(H, W).clip(0, 1)
    ax.imshow(composite, vmin=0, vmax=1)
    ax.set_title(f'{vname}: GT (R) vs Rendered (B)', fontsize=10)
    ax.axis('off')

# Row 3: 3D 只看预测 / 只看GT / 重叠对比
ax_gt = fig.add_subplot(3, 4, 9, projection='3d')
plot_3d_frame(ax_gt, np.zeros((0, 3)), np.zeros(0), gt_surface, gt_3d, None,
              'GT Only', elev=25, azim=-60)

ax_pred = fig.add_subplot(3, 4, 10, projection='3d')
plot_3d_frame(ax_pred, pred_coords, pred_scores, None, None, pred_skel,
              'Predicted Only', elev=25, azim=-60)

ax_overlay = fig.add_subplot(3, 4, 11, projection='3d')
ax_overlay.cla()
if len(gt_surface) > 0:
    ax_overlay.scatter(gt_surface[:, 0], gt_surface[:, 1], gt_surface[:, 2],
                       c='#85C1E9', s=0.8, alpha=0.2, depthshade=True)
if len(pred_coords) > 0:
    ax_overlay.scatter(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                       c='#F5B041', s=1.0, alpha=0.2, depthshade=True)
ax_overlay.plot(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], '#2471A3', linewidth=2, label='GT')
ax_overlay.plot(pred_skel[:, 0], pred_skel[:, 1], pred_skel[:, 2], '#E74C3C', linewidth=2, label='Pred')
ax_overlay.set_xlabel('X'); ax_overlay.set_ylabel('Y'); ax_overlay.set_zlabel('Z')
ax_overlay.set_xlim(*x_range); ax_overlay.set_ylim(*y_range); ax_overlay.set_zlim(*z_range)
ax_overlay.set_title('Overlay (blue=GT, orange=Pred)', fontsize=10)
ax_overlay.view_init(elev=25, azim=-60)
ax_overlay.legend(fontsize=7, loc='upper left', markerscale=3)

# 统计信息
ax_stats = fig.add_subplot(3, 4, 12)
ax_stats.axis('off')
stats_text = (
    f"Threshold: {args.threshold}\n"
    f"Combined = alpha * max(vis, 0)\n\n"
    f"Predicted points: {len(pred_coords)} / {n_points}\n"
    f"GT surface points: {len(gt_surface)}\n"
    f"GT radius: {gt_radii.mean():.4f} m\n\n"
    f"Predicted score range:\n"
    f"  min={pred_scores.min():.4f}\n" if len(pred_scores) > 0 else ""
)
if len(pred_scores) > 0:
    stats_text += (
        f"  median={np.median(pred_scores):.4f}\n"
        f"  max={pred_scores.max():.4f}\n\n"
    )
    # 计算预测点到最近GT表面点的距离
    from scipy.spatial import cKDTree
    gt_tree = cKDTree(gt_surface)
    dists, _ = gt_tree.query(pred_coords)
    stats_text += (
        f"Pred → GT surface dist:\n"
        f"  mean={dists.mean():.4f} m\n"
        f"  median={np.median(dists):.4f} m\n"
        f"  <5mm: {(dists < 0.005).mean()*100:.1f}%\n"
        f"  <10mm: {(dists < 0.01).mean()*100:.1f}%\n"
    )
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_stats.set_title('Statistics', fontsize=10)

plt.suptitle(f'Exp7: 3D Occupancy (mode {args.mode}: {MODE_NAMES[args.mode]}, T={args.threshold})',
             fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'occupancy_detail.png'), dpi=150)
plt.close()
print(f'    详细图已保存')


# ============================================================================
# Step 3: 多帧对比
# ============================================================================
print('\n>>> 生成多帧对比图...')

n_show = min(len(frame_data), 8)
fig = plt.figure(figsize=(6 * n_show, 18))

for fi in range(n_show):
    fd = frame_data[fi]
    batch = ds[fd['sample_idx']]
    gt_3d_i = batch[-1].numpy().T
    _, t_in_seq = ds.samples[fd['sample_idx']]
    gt_surface_i = generate_gt_surface(batch[-1].numpy(), gt_radii_all[t_in_seq])

    with torch.no_grad():
        aw_i = batch[0].unsqueeze(0).to(device)
        pred_skel_i = model.predict_skeleton(aw_i)['fine'][0].cpu().numpy()

    coords_i, scores_i, _ = extract_occupancy(fd['vis'], fd['den'], args.mode, args.threshold, args.vis_threshold)

    # Row 1: 3D
    ax = fig.add_subplot(3, n_show, fi + 1, projection='3d')
    plot_3d_frame(ax, coords_i, scores_i, gt_surface_i, gt_3d_i, pred_skel_i,
                  f'Frame {fi}', elev=25, azim=-60)

    # Row 2: MIP XZ
    ax2 = fig.add_subplot(3, n_show, n_show + fi + 1)
    combined_i, _ = compute_occupancy_score(fd['vis'], fd['den'], args.mode, args.threshold, args.vis_threshold)
    mip_i = combined_i.max(axis=1)  # squash Y
    ax2.imshow(mip_i.T, extent=[*x_range, *z_range], origin='lower',
               cmap='YlOrRd', aspect='auto', vmin=0, vmax=mip_i.max())
    ax2.plot(gt_3d_i[:, 0], gt_3d_i[:, 2], '#2471A3', linewidth=1, marker='.', markersize=1)
    ax2.plot(pred_skel_i[:, 0], pred_skel_i[:, 2], '#E74C3C', linewidth=1, marker='.', markersize=1)
    ax2.set_title(f'MIP {fi}', fontsize=9)
    ax2.axis('off')

    # Row 3: 正面投影
    ax3 = fig.add_subplot(3, n_show, 2 * n_show + fi + 1)
    gt_img_f = batch[1].numpy().reshape(H, W)
    plot_2d_projection(ax3, coords_i, gt_img_f, batch[3].numpy(), None,
                       cameras[0], f'Front {fi}', gt_surface_pts=gt_surface_i)

plt.suptitle(f'Multi-Frame (threshold={args.threshold})', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'multi_frame_comparison.png'), dpi=120)
plt.close()
print(f'    多帧对比已保存')


# ============================================================================
# Step 4: GIF 动画
# ============================================================================
if args.gif:
    print('\n>>> 生成 GIF 动画...')

    try:
        from PIL import Image as PILImage
    except ImportError:
        print('    需要 Pillow: pip install Pillow')
        sys.exit(0)

    gif_frames = []
    for fi in range(len(frame_data)):
        fd = frame_data[fi]
        batch = ds[fd['sample_idx']]
        gt_3d_i = batch[-1].numpy().T
        _, t_in_seq = ds.samples[fd['sample_idx']]
        gt_surface_i = generate_gt_surface(batch[-1].numpy(), gt_radii_all[t_in_seq])

        with torch.no_grad():
            aw_i = batch[0].unsqueeze(0).to(device)
            pred_skel_i = model.predict_skeleton(aw_i)['fine'][0].cpu().numpy()

        coords_i, scores_i, _ = extract_occupancy(fd['vis'], fd['den'], args.mode, args.threshold, args.vis_threshold)

        fig = plt.figure(figsize=(18, 6))

        # 3D 旋转
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        azim = -90 + fi * (360 / len(frame_data))
        plot_3d_frame(ax1, coords_i, scores_i, gt_surface_i, gt_3d_i, pred_skel_i,
                      f'Frame {fi}', elev=25, azim=azim)

        # 正面
        pred_2d_f = project_3d_to_2d(
            torch.tensor(pred_skel_i, dtype=torch.float32),
            cameras[0]['eye'], cameras[0]['center'], cameras[0]['up'],
            cameras[0]['focal'], H, W).numpy()
        ax2 = fig.add_subplot(1, 3, 2)
        plot_2d_projection(ax2, coords_i, batch[1].numpy(), batch[3].numpy(),
                           pred_2d_f, cameras[0], f'Front (frame {fi})',
                           gt_surface_pts=gt_surface_i)

        # 侧面
        pred_2d_s = project_3d_to_2d(
            torch.tensor(pred_skel_i, dtype=torch.float32),
            cameras[1]['eye'], cameras[1]['center'], cameras[1]['up'],
            cameras[1]['focal'], H, W).numpy()
        ax3 = fig.add_subplot(1, 3, 3)
        plot_2d_projection(ax3, coords_i, batch[2].numpy(), batch[4].numpy(),
                           pred_2d_s, cameras[1], f'Side (frame {fi})',
                           gt_surface_pts=gt_surface_i)

        plt.suptitle(f'3D Occupancy (mode {args.mode}, T={args.threshold})', fontsize=12)
        plt.tight_layout()

        tmp_path = os.path.join(args.output_dir, f'_gif_frame_{fi:03d}.png')
        plt.savefig(tmp_path, dpi=100)
        plt.close()
        gif_frames.append(PILImage.open(tmp_path))

    if gif_frames:
        gif_path = os.path.join(args.output_dir, 'occupancy_animation.gif')
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                           duration=500, loop=0)
        for f in gif_frames:
            f.close()
        for fi in range(len(frame_data)):
            tmp = os.path.join(args.output_dir, f'_gif_frame_{fi:03d}.png')
            if os.path.exists(tmp):
                os.remove(tmp)
        print(f'    GIF: {gif_path}')


# ============================================================================
# 汇总
# ============================================================================
print(f'\n{"="*60}')
print(f'  完成！输出目录: {args.output_dir}/')
print(f'{"="*60}')
print(f'  distribution.png          — 分布图（选阈值用）')
print(f'  occupancy_detail.png      — 详细 3D + 2D + 统计')
print(f'  multi_frame_comparison.png — 多帧对比')
if args.gif:
    print(f'  occupancy_animation.gif   — 旋转动画')
print(f'  cache/                    — 缓存（换阈值不重查）')
print(f'\n  占据判断: mode {args.mode} — {MODE_NAMES[args.mode]}')
print(f'  当前阈值: threshold={args.threshold}', end='')
if args.mode in (2, 3):
    print(f', vis_threshold={args.vis_threshold}')
else:
    print()
print(f'\n  调参示例:')
print(f'    # mode 1: 乘积组合')
print(f'    python scripts/experiments/exp7_3d_occupancy.py --skip-query --mode 1 --threshold 0.1')
print(f'    # mode 2: 任一超阈值')
print(f'    python scripts/experiments/exp7_3d_occupancy.py --skip-query --mode 2 --threshold 0.9 --vis-threshold 0.5')
print(f'    # mode 3: 两者都超阈值')
print(f'    python scripts/experiments/exp7_3d_occupancy.py --skip-query --mode 3 --threshold 0.9 --vis-threshold 0.5')
