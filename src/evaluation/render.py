"""render.py — 可视化渲染模块（Plotly HTML/PNG/GIF）。

供 visualize_3d_shape.py 和训练 validation 共用的渲染函数。
"""

import io
import os
import numpy as np


def _add_skeleton_traces(fig, gt_skeleton=None, pred_skeleton=None):
    """向 Plotly figure 添加骨架曲线。"""
    if gt_skeleton is not None:
        fig.add_trace(dict(
            type='scatter3d',
            x=gt_skeleton[0], y=gt_skeleton[1], z=gt_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='red', width=3),
            name='GT skeleton',
        ))
    if pred_skeleton is not None:
        fig.add_trace(dict(
            type='scatter3d',
            x=pred_skeleton[0], y=pred_skeleton[1], z=pred_skeleton[2],
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            line=dict(color='blue', width=3),
            name='Pred skeleton',
        ))


def render_pointcloud_html(result, gt_skeleton=None, pred_skeleton=None, title=""):
    """Flow Matching 点云 → Plotly Figure。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    pts = result.get('points')
    if pts is not None and len(pts) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=1.5, color='deepskyblue', opacity=0.6),
            name='predicted pointcloud',
        ))
    _add_skeleton_traces(fig, gt_skeleton, pred_skeleton)
    fig.update_layout(
        title=title or 'Flow Matching — Point Cloud',
        scene=dict(aspectmode='data'),
        width=700, height=600, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def render_density_html(result, threshold, gt_skeleton=None,
                        pred_skeleton=None, title=""):
    """density 点云 → Plotly Figure。"""
    import plotly.graph_objects as go

    fig = go.Figure()
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
    _add_skeleton_traces(fig, gt_skeleton, pred_skeleton)
    fig.update_layout(
        title=title or f'Density — threshold={threshold:.4f}',
        scene=dict(aspectmode='data'),
        width=700, height=600, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def render_sdf_html(result, sdf_mode='mesh', gt_skeleton=None,
                    pred_skeleton=None, title=""):
    """SDF mesh/pointcloud → Plotly Figure。"""
    import plotly.graph_objects as go

    fig = go.Figure()

    if sdf_mode == 'pointcloud' and result.get('sdf_grid') is not None:
        grid = result['sdf_grid']
        x, y, z = result['x'], result['y'], result['z']
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        mask = grid.ravel() <= 0
        if mask.any():
            fig.add_trace(go.Scatter3d(
                x=X.ravel()[mask], y=Y.ravel()[mask], z=Z.ravel()[mask],
                mode='markers',
                marker=dict(size=1.5, color='lightblue', opacity=0.6),
                name='SDF pointcloud',
            ))
    elif result.get('vertices') is not None:
        verts, faces = result['vertices'], result['faces']
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color='lightblue', opacity=0.8, name='SDF surface',
        ))
    elif result.get('sdf_grid') is not None:
        grid = result['sdf_grid']
        x, y, z = result['x'], result['y'], result['z']
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        values = grid.ravel()
        vmin, vmax = values.min(), values.max()
        norm = (values - vmin) / (vmax - vmin + 1e-8)
        fig.add_trace(go.Volume(
            x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
            value=norm, isomin=0.0, isomax=1.0,
            opacity=0.3, surface_count=10,
            colorscale='RdBu_r', colorbar=dict(title='SDF (normalized)'),
            name='SDF field',
        ))

    _add_skeleton_traces(fig, gt_skeleton, pred_skeleton)
    fig.update_layout(
        title=title or 'SDF — 3D Shape',
        scene=dict(aspectmode='data'),
        width=700, height=600, margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def render_animation(results, model_type, threshold, gt_skeletons,
                     pred_skeletons, frame_indices, sdf_mode='mesh',
                     output_path=None):
    """多帧动画 HTML（带滑块）。"""
    import plotly.graph_objects as go

    n_frames = len(results)
    is_sdf = model_type in ('sdf', 'skeleton_sdf')
    is_pc = model_type == 'flowmatch'
    fig = go.Figure()

    for i in range(n_frames):
        visible = (i == 0)
        r = results[i]
        gt = gt_skeletons[i] if gt_skeletons else None
        pred = pred_skeletons[i] if pred_skeletons else None

        if is_sdf:
            if r.get('vertices') is not None:
                v, f = r['vertices'], r['faces']
                fig.add_trace(go.Mesh3d(
                    x=v[:, 0], y=v[:, 1], z=v[:, 2],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    color='lightblue', opacity=0.8,
                    visible=visible, name=f'Frame {frame_indices[i]}',
                ))
        elif is_pc:
            pts = r.get('points')
            if pts is not None and len(pts) > 0:
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers',
                    marker=dict(size=1.5, color='deepskyblue', opacity=0.6),
                    visible=visible, name=f'Frame {frame_indices[i]}',
                ))
        else:
            pts = r.get('points')
            dens = r.get('density')
            if pts is not None and dens is not None:
                mask = dens > threshold
                if mask.any():
                    fig.add_trace(go.Scatter3d(
                        x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                        mode='markers',
                        marker=dict(size=1.5, color=dens[mask], colorscale='Viridis', opacity=0.6),
                        visible=visible, name=f'Frame {frame_indices[i]}',
                    ))

        for skel, color, name_prefix in [(gt, 'red', 'GT'), (pred, 'blue', 'Pred')]:
            if skel is not None:
                fig.add_trace(go.Scatter3d(
                    x=skel[0], y=skel[1], z=skel[2],
                    mode='lines+markers',
                    marker=dict(size=3, color=color),
                    line=dict(color=color, width=2),
                    visible=visible,
                    name=f'{name_prefix} {frame_indices[i]}',
                ))

    # 滑块
    steps = []
    n_traces_per_frame = len(fig.data) // n_frames if n_frames > 0 else 1
    for i in range(n_frames):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}],
            label=str(frame_indices[i]),
        )
        for j in range(n_traces_per_frame):
            step['args'][0]['visible'][i * n_traces_per_frame + j] = True
        steps.append(step)

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix='Frame '))],
        title=f'{model_type.upper()} — Animation ({n_frames} frames)',
        scene=dict(aspectmode='data'),
        width=800, height=700, margin=dict(l=0, r=0, t=50, b=50),
    )

    if output_path:
        fig.write_html(output_path)
        print(f"  HTML: {os.path.relpath(output_path)}")
    return fig


def render_png(fig, output_path, scale=1):
    """导出 PNG。"""
    try:
        img_bytes = fig.to_image(format='png', scale=scale)
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        print(f"  PNG: {os.path.relpath(output_path)}")
    except Exception as e:
        print(f"  PNG export failed: {e}")


def render_gif(results, model_type, threshold, gt_skeletons,
               pred_skeletons, frame_indices, sdf_mode='mesh',
               fps=3, output_path=None):
    """导出 GIF 动画。"""
    from PIL import Image

    images = []
    for i in range(len(results)):
        gt = gt_skeletons[i] if gt_skeletons else None
        pred = pred_skeletons[i] if pred_skeletons else None

        if model_type in ('sdf', 'skeleton_sdf'):
            fig = render_sdf_html(results[i], sdf_mode, gt, pred,
                                  title=f'Frame {frame_indices[i]}')
        elif model_type == 'flowmatch':
            fig = render_pointcloud_html(results[i], gt, pred,
                                          title=f'Frame {frame_indices[i]}')
        else:
            fig = render_density_html(results[i], threshold, gt, pred,
                                      title=f'Frame {frame_indices[i]}')
        fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, t=30, b=0))
        try:
            img_bytes = fig.to_image(format='png', scale=1)
            images.append(Image.open(io.BytesIO(img_bytes)))
        except Exception as e:
            print(f"  GIF export failed: {e}")
            return

    if output_path and images:
        images[0].save(
            output_path, save_all=True, append_images=images[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f"  GIF: {os.path.relpath(output_path)} ({len(images)} frames, {fps} fps)")
