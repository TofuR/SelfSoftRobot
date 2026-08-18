"""旧单段驱动实验：固定关节与静态近端段共识。"""

import numpy as np


def stabilize_static_region(positions, joint_xy, n_static=None):
    """用跨帧中位曲线替换关节到base的静态段；仅限旧实验。"""
    T, _, N = positions.shape
    if n_static is None:
        n_static = max(4, int(0.4 * N))
    xy = positions[:, :2, :]
    out = positions.copy()
    anchor = np.asarray(joint_xy, np.float64)
    dist = np.sqrt(((xy - anchor[None, :, None]) ** 2).sum(1))
    joint_nodes = dist.argmin(1).astype(int)
    grid = np.linspace(0, 1, n_static)
    cols = np.full((T, n_static), np.nan)
    rows = np.full((T, n_static), np.nan)
    for t in range(T):
        x = xy[t, 0, joint_nodes[t]:N]
        y = xy[t, 1, joint_nodes[t]:N]
        if len(x) < 2:
            continue
        arc = np.concatenate([[0.0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]).cumsum()
        if arc[-1] < 1e-6:
            continue
        u = arc / arc[-1]
        cols[t] = np.interp(grid, u, x)
        rows[t] = np.interp(grid, u, y)
    consensus_col = np.nanmedian(cols, axis=0)
    consensus_row = np.nanmedian(rows, axis=0)
    for t in range(T):
        section = slice(int(joint_nodes[t]), N)
        x, y = xy[t, 0, section], xy[t, 1, section]
        if len(x) < 2:
            continue
        arc = np.concatenate([[0.0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]).cumsum()
        if arc[-1] < 1e-6:
            continue
        u = arc / arc[-1]
        out[t, 0, section] = np.interp(u, grid, consensus_col)
        out[t, 1, section] = np.interp(u, grid, consensus_row)
    return out, joint_nodes, consensus_col, consensus_row


def detect_joint_xy(positions, node_lo=None, node_hi=None):
    """从旧静态近端序列估计固定关节绝对位置。"""
    _, _, N = positions.shape
    if node_lo is None:
        node_lo = max(4, int(0.25 * N))
    if node_hi is None:
        node_hi = min(N - 3, int(0.85 * N))
    xy = positions[:, :2, :]
    mean_d2 = np.abs(np.diff(xy[:, 0, :], n=2, axis=1)).mean(axis=0)
    subset = mean_d2[node_lo - 1:node_hi]
    peak_node = (node_lo - 1) + int(subset.argmax()) + 1
    joint_xy = np.median(xy[:, :, peak_node], axis=0)
    return joint_xy, peak_node
