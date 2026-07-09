"""inspect_real_data.py — 实物 transition 数据诊断（免模型，只读 npz）。

目的：在训练/可视化之前，先**看清实物骨架数据本身**长什么样、是否正常。
之前反复出现"空图/挤在一起/比例不对"，根因多在对实物数据坐标的无依据猜测。
本脚本直接加载 positions(actions) npz，把骨架按**图像坐标**画出来（col=横、row 纵、
row 反转→与相机原图一致：固定端 base 在上、tip 在下），并打印逐通道范围/节点顺序/动作范围。

实物数据格式（masks_to_transition_npz.py 产物，免标定）：
  positions: (T, 3, N) = [col, row, 0]   像素坐标，第 3 维恒 0（单相机平面弯曲）
  actions:   (T, A)                       已归一化到 [0,1]（按通道操作上限）
  节点顺序:  node0 = tip（图底，大 row），node_{N-1} = base（图顶，row≈0）

注意：原始拍摄帧/mask 不在磁盘上（derived/ 已清），故此处展示"提取出的骨架"本身。

用法:
  python scripts/evaluation/inspect_real_data.py                 # 自动找 data/real_seq/*/train/*.npz
  python scripts/evaluation/inspect_real_data.py --npz data/real_seq/seq_20260627_163921/train/xxx.npz
  python scripts/evaluation/inspect_real_data.py --n-samples 12
"""

import argparse
import glob
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def auto_find_npz():
    """自动找实物 transition npz（优先 train）。"""
    cands = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'data', 'real_seq', '*', 'train', '*.npz')))
    if not cands:
        cands = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'data', 'real_seq', '*', '*', '*.npz')))
    return cands[0] if cands else None


def print_stats(pos, actions):
    """打印逐通道范围 + 节点顺序 + 动作范围。"""
    T, _, N = pos.shape
    print(f"\n=== 数据统计 ===")
    print(f"  positions: {pos.shape} (T, 3, N)   actions: {actions.shape} (T, A)")
    names = ['col(x)', 'row(y)', 'z']
    for ch, name in enumerate(names):
        vals = pos[:, ch, :]
        print(f"  {name:8s}: min={vals.min():8.2f}  max={vals.max():8.2f}  "
              f"span={vals.max()-vals.min():8.2f}  std={vals.std():7.2f}")
    z_zero = np.allclose(pos[:, 2, :], 0.0)
    print(f"  z 全 0 (平面 2D 骨架): {z_zero}")
    # 节点顺序：node0 vs node_{N-1} 的 row（确认 base/tip 端）
    mid_t = T // 2
    n0_row, nN_row = pos[mid_t, 1, 0], pos[mid_t, 1, N - 1]
    print(f"  节点顺序 (frame {mid_t}): node0 row={n0_row:.1f}  node{N-1} row={nN_row:.1f}")
    if nN_row < n0_row:
        print(f"    → node{N-1}=base(row小,图顶)  node0=tip(row大,图底)  [底→顶排列, 与 extract_skeleton_2d 一致]")
    else:
        print(f"    → node0=base  node{N-1}=tip  [顶→底排列]")
    if actions.size:
        for a in range(actions.shape[1]):
            print(f"  action[{a}]: min={actions[:,a].min():.4f}  max={actions[:,a].max():.4f}  "
                  f"mean={actions[:,a].mean():.4f}  (应∈[0,1])")


def render_2d_frames(pos, actions, frame_ids, output_png, output_html):
    """每个采样帧一张 2D 子图：col(x) vs -row(y, 反转→base在上=图像方向)。

    按节点序着色(base→tip)，连线。横轴=col(像素列)，纵轴=-row(图像行，反转)。
    横纵 1:1 锚定→保真比例（col 与 row 同为像素单位，不该拉伸）。
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    N = pos.shape[2]
    n = len(frame_ids)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    titles = [f"f{fid}  a={actions[fid,0]:.2f}" for fid in frame_ids]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=titles,
                        horizontal_spacing=0.04, vertical_spacing=0.06)

    col_min, col_max = pos[:, 0, :].min(), pos[:, 0, :].max()
    row_min, row_max = pos[:, 1, :].min(), pos[:, 1, :].max()
    pad_c = (col_max - col_min) * 0.08 or 4
    pad_r = (row_max - row_min) * 0.05 or 4

    for i, fid in enumerate(frame_ids):
        r, c = divmod(i, ncols)
        pts = pos[fid].T  # (N,3) [col,row,0]
        x = pts[:, 0]
        y = -pts[:, 1]    # 反转 row → base(row小)在上 = 图像方向
        node_idx = np.arange(N)
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines+markers',
            marker=dict(size=5, color=node_idx, colorscale='Viridis', showscale=(i == 0)),
            line=dict(color='royalblue', width=2),
            showlegend=False, hovertext=[f'node{k}' for k in range(N)],
        ), row=r + 1, col=c + 1)
        # base 端高亮（红圈）：node_{N-1}（row 小=base）
        fig.add_trace(go.Scatter(
            x=[pts[N - 1, 0]], y=[-pts[N - 1, 1]], mode='markers',
            marker=dict(size=12, color='red', symbol='circle-open'),
            showlegend=False,
        ), row=r + 1, col=c + 1)
        fig.update_xaxes(range=[col_min - pad_c, col_max + pad_c],
                         scaleanchor=f'y{i+1}', scaleratio=1,
                         row=r + 1, col=c + 1)
        fig.update_yaxes(range=[-(row_max + pad_r), -(row_min - pad_r)],
                         row=r + 1, col=c + 1)

    fig.update_layout(
        title=f'实物骨架 2D（图像方向：base 红圈在上，tip 在下）| {n} sampled frames',
        width=320 * ncols, height=300 * nrows,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.write_html(output_html)
    print(f"  HTML: {os.path.relpath(output_html)}")
    try:
        fig.write_image(output_png, scale=1)
        print(f"  PNG : {os.path.relpath(output_png)}")
    except Exception as e:
        print(f"  PNG 失败(kaleido?): {e}")


def render_3d_sample(pos, fid, output_html):
    """单帧 3D 骨架——用『即将用于 render.py 的修复后配置』预览：
    面对相机(沿 -z 看 x-y 平面) + 比例保真(cube) + y 反转(base 在上=图像方向)。
    """
    import plotly.graph_objects as go
    pts = pos[fid].T  # (N,3)
    col_min, col_max = pos[:, 0, :].min(), pos[:, 0, :].max()
    row_min, row_max = pos[:, 1, :].min(), pos[:, 1, :].max()
    pad_c = (col_max - col_min) * 0.08 or 4
    pad_r = (row_max - row_min) * 0.05 or 4
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='lines+markers',
        marker=dict(size=4, color=np.arange(pts.shape[0]), colorscale='Viridis'),
        line=dict(color='royalblue', width=4), name='skeleton'))
    fig.update_layout(
        title=f'实物骨架 3D 预览 frame {fid}（y 反转→base 在上；面对 x-y 平面）',
        scene=dict(
            xaxis=dict(title='col', range=[col_min - pad_c, col_max + pad_c]),
            # y range 反转: [max, min] → row 小(base)在上 = 图像方向
            yaxis=dict(title='row', range=[row_max + pad_r, row_min - pad_r]),
            zaxis=dict(title='z', range=[-1, 1]),
            aspectmode='cube',
            camera=dict(eye=dict(x=0, y=0, z=1.5),   # 沿 -z 看 x-y 平面(面对骨架)
                        center=dict(x=0, y=0, z=0), up=dict(x=0, y=1, z=0)),
        ),
        width=700, height=700, margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(output_html)
    print(f"  HTML: {os.path.relpath(output_html)}")


def main(argv=None):
    pa = argparse.ArgumentParser(description='实物 transition 数据诊断（免模型）')
    pa.add_argument('--npz', default=None, help='npz 路径（缺省自动找 data/real_seq/*/train/*.npz）')
    pa.add_argument('--n-samples', type=int, default=9, help='采样帧数')
    pa.add_argument('--output', default=None, help='输出目录（缺省 output/inspect_real）')
    args = pa.parse_args(argv)

    npz = args.npz or auto_find_npz()
    if not npz or not os.path.isfile(npz):
        print(f"找不到 npz: {npz}\n用 --npz 指定，或确认 data/real_seq/*/train/*.npz 存在。")
        sys.exit(1)

    out_dir = args.output or os.path.join(PROJECT_ROOT, 'output', 'inspect_real')
    os.makedirs(out_dir, exist_ok=True)

    print(f"加载: {os.path.relpath(npz)}")
    d = np.load(npz)
    pos = d['positions'].astype(np.float32)      # (T,3,N)
    actions = d['actions'].astype(np.float32)    # (T,A)
    print_stats(pos, actions)

    T = pos.shape[0]
    n = min(args.n_samples, T)
    frame_ids = np.linspace(0, T - 1, n).astype(int)
    print(f"\n采样 {len(frame_ids)} 帧: {frame_ids.tolist()}")

    print("\n生成 2D 骨架诊断（图像方向）...")
    render_2d_frames(pos, actions, frame_ids,
                     os.path.join(out_dir, 'skeleton_2d_frames.png'),
                     os.path.join(out_dir, 'skeleton_2d_frames.html'))
    print("\n生成 3D 单帧预览（修复后配置）...")
    render_3d_sample(pos, frame_ids[len(frame_ids) // 2],
                     os.path.join(out_dir, 'skeleton_3d_sample.html'))

    # 数据健康检查
    print(f"\n=== 健康检查 ===")
    bad = np.where(np.all(pos == 0, axis=(1, 2)))[0]   # 全 0 帧(空 mask→跳过)
    print(f"  全 0 帧(空 mask): {len(bad)}/{T}" + (f"  例 {bad[:10]}" if len(bad) else ""))
    nan_cnt = int(np.isnan(pos).sum() + np.isnan(actions).sum())
    print(f"  NaN 数: {nan_cnt}")
    if actions.size:
        out_of_range = int(((actions < 0) | (actions > 1)).sum())
        print(f"  action 超 [0,1]: {out_of_range}")
    print(f"\n完成 → {out_dir}")


if __name__ == '__main__':
    main()
