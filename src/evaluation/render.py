"""render.py — 可视化渲染模块（Plotly HTML/PNG/GIF）。

供 visualize_3d_shape.py 和训练 validation 共用的渲染函数。
"""

import io
import os
import warnings
import numpy as np


# 直接输出骨架坐标的模型类型（lines+markers 渲染，固定相机视角）。
# 单一事实源：render_animation / render_gif 的分支判断都引用它，
# 避免新增骨架模型时遗漏（state_transition 之前漏在这里导致动画视角不固定）。
_SKELETON_MODEL_TYPES = ('spatial_sequence', 'pc_spatial', 'state_transition')


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


def _to_Nx3(p):
    """把骨架点统一成 (N,3)。兼容 result['points'] 的 (N,3) 与 GT/Pred 的 (3,N)。"""
    a = np.asarray(p, dtype=float)
    if a.ndim != 2:
        a = a.reshape(-1, 3)
    if a.shape[0] == 3 and a.shape[1] != 3:
        a = a.T  # (3,N) -> (N,3)
    return a


def _skeleton_scene_ranges(point_arrays, margin=0.05):
    """从一组骨架/点云计算 3D 场景配置（自适应范围 + 固定相机，帧间不跳变）。

    按数据形态分两种渲染（自动判别 z 是否退化）：
    - 实物 2D 骨架（z 恒 0，单相机平面弯曲）：data 填满 + 面对 x-y 平面的相机(沿 -z 看)
      + y 轴反转。实物骨架是近一维曲线(宽~50×高~293)，cube 会把它缩成细线(占 17% 宽)，
      视觉上"挤成一团"；data 模式让骨架填满画面、清晰可见（轻微比例失真换可见性，
      且让 col 方向的弯曲更醒目——这正是动作的效果）。y 反转使 row 小的固定端(base)
      显示在上=与相机原图一致(图顶固定/图底随动作)。沿 -z 看才能露出 col 弯曲(从 +x 看
      会把弯曲压进深度方向→只见一条直线)。
    - 仿真 3D 骨架（三轴都变）：cube 等比例 + up=最长轴(臂主轴竖直) + 侧向相机。
    render_animation 传全部帧进来→全局范围(帧间不跳)。
    """
    pts = [_to_Nx3(p) for p in point_arrays
           if p is not None and np.asarray(p).size >= 3]
    if not pts:
        return None
    allp = np.concatenate(pts, axis=0)
    mins = allp.min(axis=0)
    maxs = allp.max(axis=0)
    spans = (maxs - mins).astype(float)
    spans_safe = np.where(spans < 1e-6, 1.0, spans)  # 零跨度轴给单位 1 防退化
    pad = spans_safe * margin

    # 实物 2D 骨架判别：z 跨度相对 x/y 可忽略（单相机平面，第 3 维恒 0）
    is_2d = spans[2] < 1e-3 * max(spans[0], spans[1], 1e-6)
    if is_2d:
        return dict(
            xaxis=dict(range=[float(mins[0] - pad[0]), float(maxs[0] + pad[0])]),
            # y 反转: range=[max,min] → row 小(base/图顶)在上 = 图像方向
            yaxis=dict(range=[float(maxs[1] + pad[1]), float(mins[1] - pad[1])]),
            zaxis=dict(range=[-1.0, 1.0]),  # 给小范围避免退化(数据 z≡0)
            # cube：三轴等长 → 各向异性的实物骨架(col~36 / row~295)被各自拉满 → 填满画面。
            # data 模式会保比例→骨架缩成 6.7% 的细条(实测)；cube 才填满(实测 66%)。
            aspectmode='cube',
            # 相机沿 -z 看 x-y 平面(面对骨架)：露出 col 方向的弯曲(从 +x 看=侧视，
            # 会把平面内的弯曲压进深度→只见一条直线="挤在一起")。up=+y 配合 y 反转→base 在上。
            camera=dict(eye=dict(x=0.0, y=0.0, z=1.5),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=1.0, z=0.0)),
        )
    # 仿真 3D：cube 等比例 + up=最长轴(臂主轴竖直)
    _longest = int(np.argmax(spans_safe))
    _up = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}[_longest]
    return dict(
        xaxis=dict(range=[float(mins[0] - pad[0]), float(maxs[0] + pad[0])]),
        yaxis=dict(range=[float(mins[1] - pad[1]), float(maxs[1] + pad[1])]),
        zaxis=dict(range=[float(mins[2] - pad[2]), float(maxs[2] + pad[2])]),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=0.0, z=0.5),
                    center=dict(x=0.0, y=0.0, z=0.0),
                    up=dict(x=_up[0], y=_up[1], z=_up[2])),
    )


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


def render_skeleton_html(result, gt_skeleton=None, pred_skeleton=None, title=""):
    """骨架预测 → Plotly Figure（lines+markers）。

    用于 SpatialSequence/PCSpatial 等直接输出骨架坐标的模型。
    result 格式同 query_skeleton_direct 返回值。
    坐标轴固定为仿真全局范围，避免帧间跳变。
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    pts = result.get('points')
    if pts is not None:
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='lines+markers',
            marker=dict(size=5, color='blue'),
            line=dict(color='blue', width=4),
            name='Predicted skeleton',
        ))
    _add_skeleton_traces(fig, gt_skeleton, pred_skeleton)

    # 自适应坐标轴：按预测点 + GT + Pred 的实际范围推导（兼容仿真米/实物像素），
    # 替代原先硬编码的仿真米制视口——否则实物预测(~300px)落在视口外，整图全空。
    scene = _skeleton_scene_ranges([pts, gt_skeleton, pred_skeleton]) or dict(
        aspectmode='data',
        camera=dict(eye=dict(x=1.5, y=0.0, z=0.5),
                    center=dict(x=0.0, y=0.0, z=-0.1),
                    up=dict(x=0, y=0, z=1)))
    fig.update_layout(scene=scene)
    fig.update_layout(
        title=title or 'Skeleton Prediction',
        width=700, height=700, margin=dict(l=0, r=0, t=30, b=0),
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


def _frame_traces_3d(result, gt, pred, frame_idx, model_type, threshold, sdf_mode):
    """构建单帧的 3D plotly traces 列表（供 render_animation 的 frames 使用）。"""
    import plotly.graph_objects as go
    is_sdf = model_type in ('sdf', 'skeleton_sdf')
    is_pc = model_type == 'flowmatch'
    is_skeleton = model_type in _SKELETON_MODEL_TYPES
    tr = []
    if is_sdf:
        if result.get('vertices') is not None:
            v, f = result['vertices'], result['faces']
            tr.append(go.Mesh3d(
                x=v[:, 0], y=v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color='lightblue', opacity=0.8, name=f'Frame {frame_idx}'))
    elif is_skeleton:
        pts = result.get('points')
        if pts is not None:
            tr.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='lines+markers',
                marker=dict(size=4, color='blue'), line=dict(color='blue', width=3),
                name=f'Pred {frame_idx}'))
    elif is_pc:
        pts = result.get('points')
        if pts is not None and len(pts) > 0:
            tr.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                marker=dict(size=1.5, color='deepskyblue', opacity=0.6),
                name=f'Frame {frame_idx}'))
    else:
        pts = result.get('points')
        dens = result.get('density')
        if pts is not None and dens is not None:
            mask = dens > threshold
            if mask.any():
                tr.append(go.Scatter3d(
                    x=pts[mask, 0], y=pts[mask, 1], z=pts[mask, 2],
                    mode='markers',
                    marker=dict(size=1.5, color=dens[mask], colorscale='Viridis', opacity=0.6),
                    name=f'Frame {frame_idx}'))
    for skel, color, name_prefix in [(gt, 'red', 'GT'), (pred, 'blue', 'Pred')]:
        if skel is not None:
            tr.append(go.Scatter3d(
                x=skel[0], y=skel[1], z=skel[2],
                mode='lines+markers',
                marker=dict(size=3, color=color), line=dict(color=color, width=2),
                name=f'{name_prefix} {frame_idx}'))
    return tr


def render_animation(results, model_type, threshold, gt_skeletons,
                     pred_skeletons, frame_indices, sdf_mode='mesh',
                     output_path=None):
    """多帧动画 HTML（visibility-toggle 滑块；帧间不跳变）。

    滑块用 method='update' 切换每帧 trace 的 visible——这是仿真沿用至今、稳定可用的写法
    （拖动滑块即跳帧，从不卡顿）。⚠️ 不用 plotly frames+animate：在 WebGL scatter3d 上
    实测会卡顿（拖动/播放一次后画面不再变化），即便修了 mode='immediate' 也复现，故放弃
    frames 与 ▶播放按钮，回归此稳定实现。骨架模型坐标轴按全部帧求全局范围 + 固定相机，
    兼容仿真 3D(米) / 实物 2D(像素)。
    """
    import plotly.graph_objects as go

    n_frames = len(results)
    is_skeleton = model_type in _SKELETON_MODEL_TYPES

    # 全部帧 traces 平铺进一个 figure，初始只显示第 0 帧（visible 逐帧切换）
    fig = go.Figure()
    traces_per_frame = []
    for i in range(n_frames):
        gt = gt_skeletons[i] if gt_skeletons else None
        pred = pred_skeletons[i] if pred_skeletons else None
        tr = _frame_traces_3d(results[i], gt, pred, frame_indices[i],
                              model_type, threshold, sdf_mode)
        traces_per_frame.append(len(tr))
        for t in tr:
            t.visible = (i == 0)
            fig.add_trace(t)

    # 滑块：每步 method='update'，只把对应帧的 traces 置 visible（visibility-toggle，稳定）
    n_total = len(fig.data)
    steps, base = [], 0
    for i in range(n_frames):
        vis = [False] * n_total
        for j in range(traces_per_frame[i]):
            vis[base + j] = True
        base += traces_per_frame[i]
        steps.append(dict(method='update',
                          args=[{'visible': vis}],
                          label=str(frame_indices[i])))

    # 坐标轴：骨架按全部帧求全局范围(帧间不跳变) + 固定相机；其他自适应
    if is_skeleton:
        _collect = []
        for i in range(n_frames):
            _r = results[i]
            if _r.get('points') is not None:
                _collect.append(_r['points'])
            _g = gt_skeletons[i] if gt_skeletons else None
            if _g is not None:
                _collect.append(_g)
            _p = pred_skeletons[i] if pred_skeletons else None
            if _p is not None:
                _collect.append(_p)
        scene_cfg = _skeleton_scene_ranges(_collect) or dict(
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=0.0, z=0.5),
                        center=dict(x=0.0, y=0.0, z=-0.1),
                        up=dict(x=0, y=0, z=1)))
    else:
        scene_cfg = dict(aspectmode='data')

    fig.update_layout(
        sliders=[dict(active=0, steps=steps, currentvalue=dict(prefix='Frame '))],
        title=f'{model_type.upper()} — Animation ({n_frames} frames)',
        scene=scene_cfg,
        width=800, height=700, margin=dict(l=0, r=0, t=50, b=50),
    )

    if output_path:
        fig.write_html(output_path)
        print(f"  HTML: {os.path.relpath(output_path)}")
    return fig


def render_png(fig, output_path, scale=1):
    """导出 PNG。"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        elif model_type in _SKELETON_MODEL_TYPES:
            fig = render_skeleton_html(results[i], gt, pred,
                                        title=f'Frame {frame_indices[i]}')
        elif model_type == 'flowmatch':
            fig = render_pointcloud_html(results[i], gt, pred,
                                          title=f'Frame {frame_indices[i]}')
        else:
            fig = render_density_html(results[i], threshold, gt, pred,
                                      title=f'Frame {frame_indices[i]}')
        fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, t=30, b=0))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
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
