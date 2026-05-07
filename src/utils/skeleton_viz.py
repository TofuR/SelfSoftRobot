"""skeleton_viz.py — 骨架可视化工具函数。

纯可视化函数，不涉及模型加载或数据处理。
所有函数接受 numpy 数组，基于 matplotlib 渲染。
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.animation as animation


# 默认 3D 视图范围（与仿真环境一致）
BOUNDS = {
    'x': (-0.3, 0.3),
    'y': (-0.3, 0.3),
    'z': (0.0, 0.6),
}


def _setup_ax_3d(ax, title="", bounds=None):
    """配置 3D 坐标轴。"""
    b = bounds or BOUNDS
    ax.set_xlim(b['x'])
    ax.set_ylim(b['y'])
    ax.set_zlim(b['z'])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    if title:
        ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])


def plot_skeleton_3d(pred, gt=None, title="", ax=None, show=True, save_path=None,
                     bounds=None, pred_color='red', gt_color='blue'):
    """单帧 3D 骨架可视化。

    Args:
        pred: (N, 3) numpy array — 预测骨架节点。
        gt:   (N, 3) 或 None — GT 骨架节点。
        title: 图标题。
        ax:   matplotlib 3D axes（可选，用于 subplot）。
        show: 是否调用 plt.show()。
        save_path: 保存路径。
        bounds: 坐标轴范围 dict。
        pred_color: 预测骨架颜色。
        gt_color: GT 骨架颜色。
    """
    created_ax = ax is None
    if created_ax:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

    if gt is not None:
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], '-o', color=gt_color,
                linewidth=3, markersize=4, label='GT', alpha=0.8)

    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], '-o', color=pred_color,
            linewidth=2, markersize=3, label='Pred', alpha=0.9)

    _setup_ax_3d(ax, title, bounds)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show and created_ax:
        plt.show()
    return ax


def plot_multi_scale(pred_dict, gt=None, save_path=None, show=True, bounds=None):
    """多尺度骨架对比 (coarse / medium / fine)。

    Args:
        pred_dict: dict with 'coarse', 'medium', 'fine' — each (N, 3)。
        gt: (31, 3) 或 None。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})

    scales = ['coarse', 'medium', 'fine']
    colors = ['green', 'orange', 'red']

    for i, (scale, color) in enumerate(zip(scales, colors)):
        ax = axes[i]
        skel = pred_dict[scale]

        if gt is not None and scale == 'fine':
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], '-o', color='blue',
                    linewidth=3, markersize=4, label='GT', alpha=0.6)

        ax.plot(skel[:, 0], skel[:, 1], skel[:, 2], '-o', color=color,
                linewidth=2, markersize=3, label=f'{scale} ({len(skel)})')
        _setup_ax_3d(ax, f'{scale.capitalize()} Scale', bounds)
        ax.legend(fontsize=8)

    plt.suptitle('Multi-Scale Skeleton Prediction', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_error_along_arm(errors, title="Node-wise Error", save_path=None, show=True):
    """沿杆体的逐节点误差分布。

    Args:
        errors: (N,) 每个节点的 L2 误差。
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    nodes = np.arange(len(errors))
    ax.bar(nodes, errors, color='salmon', alpha=0.8)
    ax.set_xlabel('Node Index (base → tip)')
    ax.set_ylabel('L2 Error (m)')
    ax.set_title(title)

    mean_err = np.mean(errors)
    ax.axhline(mean_err, color='red', linestyle='--', label=f'Mean: {mean_err:.4f}m')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_comparison_grid(preds, gts, n_cols=4, save_path=None, show=True, bounds=None):
    """多帧 GT vs Pred 对比网格。

    Args:
        preds: list of (N, 3) — 预测骨架。
        gts:   list of (N, 3) — GT 骨架。
        n_cols: 列数。
    """
    n_samples = len(preds)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows),
                              subplot_kw={'projection': '3d'})
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_samples):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        pred = preds[idx]
        gt = gts[idx] if idx < len(gts) else None

        if gt is not None:
            ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], '-o', color='blue',
                    linewidth=3, markersize=3, alpha=0.6)
        ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], '-o', color='red',
                linewidth=2, markersize=2)
        _setup_ax_3d(ax, f'Frame {idx}', bounds)

    for idx in range(n_samples, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    plt.suptitle('GT (blue) vs Prediction (red)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def animate_skeleton_sequence(pred_seq, gt_seq=None, actions=None, save_path=None,
                               fps=10, bounds=None, figsize=(15, 5)):
    """骨架序列动画 → GIF。

    Args:
        pred_seq: (T, N, 3) 预测序列。
        gt_seq:   (T, N, 3) GT 序列（可选）。
        actions:  (T, D) 动作序列（可选，显示在子图中）。
        save_path: GIF 保存路径。
        fps: 帧率。
        bounds: 坐标范围 dict。
    """
    T = len(pred_seq)
    has_gt = gt_seq is not None
    has_act = actions is not None

    n_plots = 1 + int(has_gt) + int(has_act)
    fig = plt.figure(figsize=figsize)

    axes = []
    ax_idx = 1
    # GT + Pred 3D 图
    ax3d = fig.add_subplot(1, n_plots, ax_idx, projection='3d')
    axes.append(ax3d)
    ax_idx += 1

    if has_gt:
        ax3d_gt = fig.add_subplot(1, n_plots, ax_idx, projection='3d')
        axes.append(ax3d_gt)
        ax_idx += 1

    if has_act:
        ax_act = fig.add_subplot(1, n_plots, ax_idx)
        axes.append(ax_act)

    b = bounds or BOUNDS

    # 初始化 3D 图元素
    line_pred, = axes[0].plot([], [], [], 'r-o', linewidth=2, markersize=3, label='Pred')
    _setup_ax_3d(axes[0], 'Prediction', b)

    if has_gt:
        line_gt, = axes[1].plot([], [], [], 'b-o', linewidth=3, markersize=4, label='GT')
        _setup_ax_3d(axes[1], 'Ground Truth', b)

    if has_act:
        colors = ['red', 'green']
        act_lines = []
        for d in range(actions.shape[1]):
            l, = ax_act.plot([], [], color=colors[d % len(colors)],
                             label=f'torque_{d}')
            act_lines.append(l)
        ax_act.set_xlim(0, T)
        ax_act.set_ylim(actions.min() * 1.1, actions.max() * 1.1)
        ax_act.set_xlabel('Frame')
        ax_act.set_ylabel('Torque')
        ax_act.legend()
        ax_act.set_title('Driving Actions')
        vline = ax_act.axvline(x=0, color='k', linestyle='--')

    def update(frame):
        pred = pred_seq[frame]
        line_pred.set_data(pred[:, 0], pred[:, 1])
        line_pred.set_3d_properties(pred[:, 2])

        if has_gt:
            gt = gt_seq[frame]
            line_gt.set_data(gt[:, 0], gt[:, 1])
            line_gt.set_3d_properties(gt[:, 2])

        if has_act:
            for d, l in enumerate(act_lines):
                l.set_data(np.arange(T), actions[:, d])
            vline.set_xdata([frame, frame])

        fig.suptitle(f'Frame {frame}/{T-1}', fontsize=12)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 // fps, blit=False)

    if save_path:
        ani.save(save_path, writer='pillow', fps=fps)
        print(f"Animation saved to {save_path}")
        plt.close()
    else:
        plt.show()

    return ani


def render_density_field(model, action_window, bounds, grid_res=30,
                         threshold=0.5, device='cpu', batch_size=4096):
    """从 MS-SCNF 密度场提取高密度点云。

    Args:
        model: MS-SCNF 模型（eval 模式）。
        action_window: (1, K, D) 动作窗口 tensor。
        bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax))。
        grid_res: 每轴采样点数。
        threshold: 密度阈值。
        device: 计算设备。
        batch_size: 批量处理点数。

    Returns:
        (N, 3) numpy array — 高密度点云。
    """
    xs = np.linspace(bounds[0][0], bounds[0][1], grid_res)
    ys = np.linspace(bounds[1][0], bounds[1][1], grid_res)
    zs = np.linspace(bounds[2][0], bounds[2][1], grid_res)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

    all_densities = []
    n_pts = grid_tensor.shape[0]
    # 将点 reshape 为 (N_rays, 1, 3) 模拟单采样点射线
    points = grid_tensor.unsqueeze(1)  # (N, 1, 3)

    with torch.no_grad():
        for start in range(0, n_pts, batch_size):
            end = min(start + batch_size, n_pts)
            pts_batch = points[start:end]
            output = model(pts_batch, action_window)
            density = torch.nn.functional.softplus(output[:, :, 1])
            all_densities.append(density.cpu().numpy().squeeze())

    densities = np.concatenate(all_densities)
    mask = densities > threshold
    return grid_points[mask]


def print_metrics(pred, gt, label=""):
    """计算并打印 3D 评估指标。

    Args:
        pred: (N, 3) numpy array。
        gt:   (N, 3) numpy array。
    """
    import torch
    pred_t = torch.from_numpy(pred).float()
    gt_t = torch.from_numpy(gt).float()

    from src.training.metrics_3d import mean_node_error, endpoint_error, curve_smoothness, chamfer_distance

    mne = mean_node_error(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).item()
    epe = endpoint_error(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).item()
    smooth = curve_smoothness(pred_t.unsqueeze(0)).item()
    cd = chamfer_distance(pred_t.unsqueeze(0), gt_t.unsqueeze(0)).item()

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MNE={mne:.6f}m  EPE={epe:.6f}m  Smooth={smooth:.6f}  CD={cd:.6f}")
    return {'mne': mne, 'epe': epe, 'smoothness': smooth, 'cd': cd}
