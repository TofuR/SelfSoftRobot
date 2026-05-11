"""visualize_3d_shape.py — 交互式 3D 形状可视化工具。

用法:
    python scripts/evaluation/visualize_3d_shape.py
    python scripts/evaluation/visualize_3d_shape.py --device cuda:0
    python scripts/evaluation/visualize_3d_shape.py --output output/visualize

交互式选择模型 checkpoint → 数据文件 → 帧 → 查询并可视化。
支持所有模型类型: SDF (marching cubes) / NeRF 系列 (密度阈值)。
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch

# 项目根目录加入 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


# ──────────────────────────── 扫描工具 ────────────────────────────

def scan_checkpoints():
    """扫描 train_log/ 下所有 checkpoint 文件。"""
    patterns = [
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'best_model.pt'),
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'skeleton_best.pt'),
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'canonical_best.pt'),
    ]
    ckpts = []
    for pat in patterns:
        ckpts.extend(glob.glob(pat, recursive=True))
    return sorted(set(ckpts))


def scan_data_dirs():
    """扫描 data/ 下包含 npz 文件的目录。"""
    data_root = os.path.join(PROJECT_ROOT, 'data')
    dirs = []
    for d in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, d)
        if os.path.isdir(full) and glob.glob(os.path.join(full, '*.npz')):
            dirs.append(full)
    return dirs


def scan_npz_files(data_dir):
    """列出目录下所有 npz 文件。"""
    return sorted(glob.glob(os.path.join(data_dir, '*.npz')))


# ──────────────────────────── 交互式选择 ────────────────────────────

def select_from_list(items, prompt, allow_custom=False):
    """终端交互式选择。"""
    if not items:
        print(f"  无可用选项 ({prompt})")
        return None
    print(f"\n{prompt}:")
    for i, item in enumerate(items, 1):
        display = os.path.relpath(item, PROJECT_ROOT) if isinstance(item, str) else str(item)
        print(f"  {i}. {display}")
    if allow_custom:
        print(f"  {len(items)+1}. [手动输入路径]")

    while True:
        try:
            raw = input(f"  > ").strip()
            if not raw:
                continue
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(items):
                    return items[idx]
                if allow_custom and idx == len(items):
                    path = input("  路径: ").strip()
                    if os.path.exists(path):
                        return path
                    print(f"  文件不存在: {path}")
                    continue
            print(f"  请输入 1-{len(items)}")
        except (EOFError, KeyboardInterrupt):
            return None


def input_int(prompt, default):
    """输入整数，回车使用默认值。"""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def input_float(prompt, default):
    """输入浮点数，回车使用默认值。"""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ──────────────────────────── 模型查询 ────────────────────────────

@torch.no_grad()
def query_sdf_model(model, action_window, bounds, grid_res, device,
                    coord_center=None, coord_scale=1.0, batch_size=100000):
    """SDF 模型: 在 3D 网格上查询 SDF 值，用 marching cubes 提取零等值面。

    模型在归一化坐标 [-1,1]^3 上训练，查询前需要先归一化。
    """
    from skimage import measure

    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[2], bounds[3], grid_res)
    z = np.linspace(bounds[4], bounds[5], grid_res)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords_np = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    # 归一化到 [-1, 1]^3（与训练时一致）
    if coord_center is not None:
        coords_np = (coords_np - coord_center[None, :]) / coord_scale

    sdf_all = np.zeros(len(coords_np), dtype=np.float32)
    n_batches = (len(coords_np) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(coords_np))
        pts = torch.from_numpy(coords_np[start:end]).to(device)
        sdf = model(pts, action_window)
        sdf_all[start:end] = sdf.cpu().numpy().ravel()

    sdf_grid = sdf_all.reshape(grid_res, grid_res, grid_res)

    # 返回 SDF 网格 + 坐标轴，用于体积渲染
    x_vals = np.linspace(bounds[0], bounds[1], grid_res)
    y_vals = np.linspace(bounds[2], bounds[3], grid_res)
    z_vals = np.linspace(bounds[4], bounds[5], grid_res)

    result = {
        'vertices': None, 'faces': None,
        'sdf_grid': sdf_grid,
        'x': x_vals, 'y': y_vals, 'z': z_vals,
    }

    # 尝试 marching cubes 提取零等值面
    if sdf_all.min() <= 0 <= sdf_all.max():
        # spacing 基于归一化空间的网格
        norm_min = (np.array([bounds[0], bounds[2], bounds[4]]) - coord_center) / coord_scale
        norm_max = (np.array([bounds[1], bounds[3], bounds[5]]) - coord_center) / coord_scale
        spacing = tuple((norm_max - norm_min) / (grid_res - 1))
        verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0, spacing=spacing)
        # 顶点在归一化空间，反归一化回世界坐标
        verts[:, 0] = verts[:, 0] + norm_min[0]
        verts[:, 1] = verts[:, 1] + norm_min[1]
        verts[:, 2] = verts[:, 2] + norm_min[2]
        # 反归一化到世界坐标
        if coord_center is not None:
            verts = verts * coord_scale + coord_center[None, :]
        result['vertices'] = verts
        result['faces'] = faces
    else:
        print(f"  无零等值面 (SDF: [{sdf_all.min():.6f}, {sdf_all.max():.6f}])，使用体积渲染")

    return result


@torch.no_grad()
def query_nerf_model(model, action_window, bounds, grid_res, device, n_samples=1,
                     batch_size=50000):
    """NeRF 系列模型: 在 3D 网格上查询 density，阈值过滤。"""
    import torch.nn.functional as F

    x = np.linspace(bounds[0], bounds[1], grid_res)
    y = np.linspace(bounds[2], bounds[3], grid_res)
    z = np.linspace(bounds[4], bounds[5], grid_res)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    coords_np = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1).astype(np.float32)

    n_pts = len(coords_np)
    all_vis = np.zeros(n_pts, dtype=np.float32)
    all_dens = np.zeros(n_pts, dtype=np.float32)
    n_batches = (n_pts + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_pts)
        pts = torch.from_numpy(coords_np[start:end]).to(device)
        pts = pts.unsqueeze(1).expand(-1, n_samples, -1)  # (N, 1, 3)

        raw = model(pts, action_window)  # (B*N, 1, 2) 或 (N, 1, 2)
        if raw.dim() == 3 and raw.shape[0] == action_window.shape[0] * (end - start):
            raw = raw.reshape(end - start, n_samples, 2)

        vis = torch.sigmoid(raw[..., 0]).mean(dim=-1).cpu().numpy()
        dens = F.softplus(raw[..., 1]).mean(dim=-1).cpu().numpy()

        all_vis[start:end] = vis
        all_dens[start:end] = dens

    return {'points': coords_np, 'density': all_dens, 'visibility': all_vis}


# ──────────────────────────── 可视化输出 ────────────────────────────

def export_html(result, output_path, model_type, threshold=None, gt_skeleton=None, pred_skeleton=None):
    """Plotly 交互式 HTML。"""
    import plotly.graph_objects as go

    fig = go.Figure()

    if model_type == 'sdf':
        if result.get('vertices') is not None:
            # 有零等值面 → mesh
            verts, faces = result['vertices'], result['faces']
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightblue', opacity=0.8, name='SDF surface',
            ))
        elif result.get('sdf_grid') is not None:
            # 无零等值面 → 彩色体积渲染
            grid = result['sdf_grid']
            x, y, z = result['x'], result['y'], result['z']
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            values = grid.ravel()
            # 归一化到 [0, 1] 用于颜色映射
            vmin, vmax = values.min(), values.max()
            norm = (values - vmin) / (vmax - vmin + 1e-8)
            fig.add_trace(go.Volume(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                value=norm,
                isomin=0.0, isomax=1.0,
                opacity=0.3, surface_count=10,
                colorscale='RdBu_r',
                colorbar=dict(title='SDF (normalized)'),
                name='SDF field',
            ))

    elif model_type != 'sdf' and result.get('points') is not None:
        pts = result['points']
        dens = result['density']
        mask = dens > threshold
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                mode='markers',
                marker=dict(size=1.5, color=dens[mask], colorscale='Viridis',
                            opacity=0.6, colorbar=dict(title='density')),
                name='density field',
            ))

    if gt_skeleton is not None:
        fig.add_trace(go.Scatter3d(
            x=gt_skeleton[0], y=gt_skeleton[1], z=gt_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=3),
            name='GT skeleton',
        ))

    if pred_skeleton is not None:
        fig.add_trace(go.Scatter3d(
            x=pred_skeleton[0], y=pred_skeleton[1], z=pred_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=3),
            name='Pred skeleton',
        ))

    fig.update_layout(
        title=f'{model_type.upper()} — 3D Shape (threshold={threshold})',
        scene=dict(aspectmode='data'),
        width=900, height=700,
    )
    fig.write_html(output_path)
    print(f"  HTML: {os.path.relpath(output_path, PROJECT_ROOT)}")


def export_png(result, output_path, model_type, threshold=None, gt_skeleton=None, pred_skeleton=None):
    """PyVista offscreen PNG。"""
    try:
        import pyvista as pv
        pv.OFF_SCREEN = True
    except ImportError:
        print("  跳过 PNG: pyvista 未安装")
        return

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))

    if model_type == 'sdf':
        if result.get('vertices') is not None:
            verts, faces = result['vertices'], result['faces']
            faces_pv = np.column_stack([np.full(len(faces), 3), faces]).ravel()
            mesh = pv.PolyData(verts, faces_pv)
            plotter.add_mesh(mesh, color='lightblue', opacity=0.9, show_edges=True)
        elif result.get('sdf_grid') is not None:
            # 无零等值面 → 用 PyVista 的 structured grid + contour 可视化
            x, y, z = result['x'], result['y'], result['z']
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            grid = pv.StructuredGrid(X, Y, Z)
            grid['sdf'] = result['sdf_grid'].ravel()
            # 用多个等值面展示 SDF 分布
            sdf_min, sdf_max = result['sdf_grid'].min(), result['sdf_grid'].max()
            levels = np.linspace(sdf_min, sdf_max, 6)[1:-1]
            contours = grid.contour(isosurfaces=levels)
            if contours.n_points > 0:
                plotter.add_mesh(contours, scalars='sdf', opacity=0.5,
                                 show_scalar_bar=True, cmap='coolwarm')

    elif model_type != 'sdf' and result.get('points') is not None:
        pts = result['points']
        dens = result['density']
        mask = dens > threshold
        if mask.any():
            cloud = pv.PolyData(pts[mask])
            cloud['density'] = dens[mask]
            plotter.add_mesh(cloud, scalars='density', point_size=3,
                             render_points_as_spheres=True, opacity=0.6)

    if gt_skeleton is not None:
        skel = pv.Spline(gt_skeleton.T, n_points=len(gt_skeleton[0]))
        plotter.add_mesh(skel, color='red', line_width=5)
        plotter.add_mesh(pv.PolyData(gt_skeleton.T), color='red', point_size=8)

    if pred_skeleton is not None:
        pred_skel = pv.Spline(pred_skeleton.T, n_points=len(pred_skeleton[0]))
        plotter.add_mesh(pred_skel, color='blue', line_width=5)
        plotter.add_mesh(pv.PolyData(pred_skeleton.T), color='blue', point_size=8)

    plotter.set_background('white')
    plotter.screenshot(output_path)
    plotter.close()
    print(f"  PNG:  {os.path.relpath(output_path, PROJECT_ROOT)}")


def _make_frame_traces(result, model_type, threshold, gt_skeleton, pred_skeleton):
    """为单帧生成 Plotly traces 列表。"""
    import plotly.graph_objects as go
    traces = []

    if model_type == 'sdf':
        if result.get('vertices') is not None:
            verts, faces = result['vertices'], result['faces']
            traces.append(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='lightblue', opacity=0.8,
            ))
        elif result.get('sdf_grid') is not None:
            grid = result['sdf_grid']
            x, y, z = result['x'], result['y'], result['z']
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            values = grid.ravel()
            vmin, vmax = values.min(), values.max()
            norm = (values - vmin) / (vmax - vmin + 1e-8)
            traces.append(go.Volume(
                x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                value=norm, isomin=0.0, isomax=1.0,
                opacity=0.3, surface_count=10, colorscale='RdBu_r',
            ))
    elif result.get('points') is not None:
        pts = result['points']
        dens = result['density']
        mask = dens > threshold
        if mask.any():
            traces.append(go.Scatter3d(
                x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                mode='markers',
                marker=dict(size=1.5, color=dens[mask], colorscale='Viridis', opacity=0.6),
            ))

    if gt_skeleton is not None:
        traces.append(go.Scatter3d(
            x=gt_skeleton[0], y=gt_skeleton[1], z=gt_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=3),
        ))

    if pred_skeleton is not None:
        traces.append(go.Scatter3d(
            x=pred_skeleton[0], y=pred_skeleton[1], z=pred_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=3),
        ))

    return traces


def export_html_animation(all_results, output_path, model_type, threshold,
                          all_gt=None, all_pred=None, frame_indices=None):
    """Plotly 动画 HTML — 带拖动条切换帧。"""
    import plotly.graph_objects as go

    n_frames = len(all_results)
    if frame_indices is None:
        frame_indices = list(range(n_frames))

    # 第一帧作为初始
    init_traces = _make_frame_traces(
        all_results[0], model_type, threshold,
        all_gt[0] if all_gt else None,
        all_pred[0] if all_pred else None,
    )

    fig = go.Figure(data=init_traces)

    # 添加所有帧
    frames = []
    for i in range(n_frames):
        ft = _make_frame_traces(
            all_results[i], model_type, threshold,
            all_gt[i] if all_gt else None,
            all_pred[i] if all_pred else None,
        )
        frames.append(go.Frame(data=ft, name=f'frame_{i}'))
    fig.frames = frames

    # 滑动条
    sliders = [dict(
        active=0,
        y=-0.05,
        xanchor='left',
        len=1.0,
        currentvalue=dict(font=dict(size=14), prefix='帧: ', visible=True, xanchor='center'),
        steps=[dict(
            label=f'{frame_indices[i]}',
            method='animate',
            args=[[f'frame_{i}'], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
        ) for i in range(n_frames)],
    )]

    # 播放/暂停按钮
    updatemenus = [dict(
        type='buttons',
        showactive=False,
        y=-0.05,
        x=0.0,
        buttons=[
            dict(label='▶', method='animate',
                 args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
            dict(label='⏸', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
        ],
    )]

    fig.update_layout(
        title=f'{model_type.upper()} — 3D Shape Animation ({n_frames} frames)',
        scene=dict(aspectmode='data'),
        width=900, height=700,
        sliders=sliders,
        updatemenus=updatemenus,
    )
    fig.write_html(output_path)
    print(f"  HTML动画: {os.path.relpath(output_path, PROJECT_ROOT)}")


def export_gif(all_results, output_path, model_type, threshold,
               all_gt=None, all_pred=None, frame_indices=None, fps=8):
    """将多帧渲染为 GIF 动画（需要 kaleido + Chrome）。"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  跳过 GIF: 需要 plotly")
        return
    from PIL import Image
    import io

    n_frames = len(all_results)
    if frame_indices is None:
        frame_indices = list(range(n_frames))

    # 统一 axis range（用第一帧的 bounds）
    # 先渲染所有帧为图片
    images = []
    for i in range(n_frames):
        traces = _make_frame_traces(
            all_results[i], model_type, threshold,
            all_gt[i] if all_gt else None,
            all_pred[i] if all_pred else None,
        )
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f'Frame {frame_indices[i]}',
            scene=dict(aspectmode='data'),
            width=600, height=500,
            margin=dict(l=0, r=0, t=30, b=0),
        )
        try:
            img_bytes = fig.to_image(format='png', scale=1)
        except Exception as e:
            print(f"  跳过 GIF: 渲染失败 ({type(e).__name__}: {e})")
            return
        images.append(Image.open(io.BytesIO(img_bytes)))

    # 保存 GIF
    images[0].save(
        output_path,
        save_all=True, append_images=images[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f"  GIF: {os.path.relpath(output_path, PROJECT_ROOT)} ({n_frames} frames, {fps} fps)")


# ──────────────────────────── 主流程 ────────────────────────────

def compute_bounds(data_path, frame_idx=None):
    """从数据文件计算查询空间的边界。frame_idx=None 时用所有帧。"""
    d = np.load(data_path)
    pos = d['positions']
    if pos.ndim == 3:
        if frame_idx is not None:
            pos = pos[frame_idx]
        else:
            pos = pos.reshape(3, -1)

    center = pos.mean(axis=-1)
    extent = np.abs(pos - center[:, None]).max(axis=-1)
    margin = max(extent.max() * 1.5, 0.05)

    return (
        center[0] - margin, center[0] + margin,
        center[1] - margin, center[1] + margin,
        center[2] - margin, center[2] + margin,
    )


def get_action_window(data_path, frame_idx, seq_len, norm_factor):
    """从数据文件提取指定帧的 action window。"""
    d = np.load(data_path)
    actions = d['actions'] / norm_factor
    action_dim = actions.shape[1]

    start = frame_idx - seq_len + 1
    if start >= 0:
        window = actions[start:frame_idx + 1].copy()
    else:
        pad = np.zeros((-start, action_dim), dtype=actions.dtype)
        window = np.concatenate([pad, actions[0:frame_idx + 1]], axis=0)

    return torch.from_numpy(window).float().unsqueeze(0)  # (1, K, D)


def get_gt_skeleton(data_path, frame_idx):
    """提取 GT skeleton（如果有 positions 字段）。"""
    d = np.load(data_path)
    if 'positions' not in d:
        return None
    pos = d['positions']
    if pos.ndim == 3:
        return pos[frame_idx]  # (3, N)
    return pos  # (3, N)


def main():
    parser = argparse.ArgumentParser(description='3D Shape Visualizer')
    parser.add_argument('--device', default='cuda:0', help='计算设备')
    parser.add_argument('--output', default=None, help='输出目录')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = args.output or os.path.join(PROJECT_ROOT, 'output', 'visualize')
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 3D Shape Visualizer ===\n")

    # ── Step 1: 选择 checkpoint ──
    ckpts = scan_checkpoints()
    ckpt_path = select_from_list(ckpts, "[1] 选择模型 checkpoint", allow_custom=True)
    if ckpt_path is None:
        return

    # ── Step 2: 加载模型 ──
    print(f"\n加载模型...")
    from src.utils.model_loader import load_model
    try:
        info = load_model(ckpt_path, data_dir=None, device=device)
    except Exception as e:
        print(f"  加载失败: {e}")
        print("  该模型类型可能尚未被 model_loader 支持")
        return
    model = info['model']
    model_type = info['model_type']
    window_size = info['window_size']
    norm_factor = info['norm_factor']
    print(f"  模型类型: {model_type}, window_size={window_size}")

    # ── Step 3: 选择数据目录 ──
    data_dirs = scan_data_dirs()
    data_dir = select_from_list(data_dirs, "[2] 选择数据目录", allow_custom=True)
    if data_dir is None:
        return

    # ── Step 4: 选择 npz 文件 ──
    npz_files = scan_npz_files(data_dir)
    npz_path = select_from_list(npz_files, "[3] 选择数据文件", allow_custom=True)
    if npz_path is None:
        return

    # ── Step 5: 选择帧范围 ──
    d = np.load(npz_path)
    n_frames = d['actions'].shape[0]
    print(f"\n[4] 帧范围 (共 {n_frames} 帧)")
    start_frame = input_int("  起始帧", 0)
    end_frame = input_int(f"  结束帧 (含, max {n_frames-1})", min(start_frame + 49, n_frames - 1))
    step = input_int("  帧间隔 (skip)", max(1, (end_frame - start_frame) // 50))
    frame_indices = list(range(start_frame, end_frame + 1, step))
    n_vis = len(frame_indices)
    print(f"  将可视化 {n_vis} 帧: {frame_indices[0]}-{frame_indices[-1]} (step={step})")

    # ── Step 6: 查询参数 ──
    grid_res = input_int("[5] 网格分辨率", 40)
    threshold = 0.5
    if model_type != 'sdf':
        threshold = input_float("[6] 密度阈值 (NeRF)", 0.5)

    # ── Step 7: 准备统一 bounds（覆盖所有帧） ──
    bounds = compute_bounds(npz_path, frame_idx=None)

    # ── Step 8: 逐帧查询模型 ──
    exp_name = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    base_name = f"{model_type}_{exp_name}_frames{start_frame}-{end_frame}"

    # SDF 归一化参数（只需算一次）
    norm_params = None
    if model_type == 'sdf':
        from src.data.dataset_sdf import SDFDataset
        norm_params = SDFDataset.compute_normalization(os.path.dirname(npz_path))

    print(f"\n查询模型 ({model_type}), {n_vis} 帧...")
    print(f"  空间: [{bounds[0]:.3f}, {bounds[1]:.3f}] x "
          f"[{bounds[2]:.3f}, {bounds[3]:.3f}] x [{bounds[4]:.3f}, {bounds[5]:.3f}]")
    print(f"  网格: {grid_res}^3 = {grid_res**3:,} 点/帧")

    all_results = []
    all_gt = []
    all_pred = []

    for vis_i, fidx in enumerate(frame_indices):
        action_window = get_action_window(npz_path, fidx, window_size, norm_factor)
        action_window = action_window.to(device)
        gt_skeleton = get_gt_skeleton(npz_path, fidx)
        pred_skeleton = None

        if model_type == 'sdf':
            result = query_sdf_model(model, action_window, bounds, grid_res, device,
                                      coord_center=norm_params['coord_center'],
                                      coord_scale=norm_params['coord_scale'])
        else:
            result = query_nerf_model(model, action_window, bounds, grid_res, device)

        if model_type == 'ms_scnf' and hasattr(model, 'predict_skeleton'):
            with torch.no_grad():
                skel_dict = model.predict_skeleton(action_window)
                pred_skeleton = skel_dict['fine'][0].cpu().numpy().T

        all_results.append(result)
        all_gt.append(gt_skeleton)
        all_pred.append(pred_skeleton)

        # 进度
        n_verts = len(result['vertices']) if result.get('vertices') is not None else 0
        print(f"  [{vis_i+1}/{n_vis}] frame {fidx}: {n_verts} vertices", end='\r')

    print(f"\n  完成 {n_vis} 帧查询")

    # ── Step 9: 输出 ──
    print(f"\n生成可视化...")

    # 动画 HTML（带拖动条）
    html_path = os.path.join(output_dir, f"{base_name}.html")
    export_html_animation(all_results, html_path, model_type, threshold,
                          all_gt, all_pred, frame_indices)

    # 单帧 PNG（中间帧）
    mid = n_vis // 2
    png_path = os.path.join(output_dir, f"{base_name}_mid.png")
    export_png(all_results[mid], png_path, model_type, threshold,
               all_gt[mid], all_pred[mid])

    # GIF 动画
    gif_path = os.path.join(output_dir, f"{base_name}.gif")
    export_gif(all_results, gif_path, model_type, threshold,
               all_gt, all_pred, frame_indices)

    print(f"\n完成!")


if __name__ == '__main__':
    main()
