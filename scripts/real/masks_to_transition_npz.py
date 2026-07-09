"""masks_to_transition_npz.py — 实物 mask + actions → transition 训练 npz（免标定）。

为什么不需要标定（核心）:
  state_transition 模型的 state 是 3D 中心线骨架 s∈R^{N×3}（每个节点 3 坐标），
  模型只消费 (prev_skeleton, action) → next_skeleton，**不碰图像、不需要相机参数**。
  本序列是单相机 1-DOF 平面弯曲，直接用 mask 的 2D 图像骨架作 state（第 3 维 z=0）：
    positions[t,:,i] = [col_i, row_i, 0]
  模型在归一化图像坐标空间学动力学；GT-transition vs open-loop 的"预测方法"对比
  在该空间同样有效（对比的是框架，不是度量 3D 精度）。
  → 不标定、不 planar-lift、不 NeRF。需要度量 3D 时再标定（NDI 末端可作独立验证）。

action 归一化（**[0,1]，不到负数**）:
  气动单向 + 半自由度：ch0 只能充气(0→150)把臂往一个方向弯；负值 = 反向驱动(拮抗
  通道)，ch0 产生不了。映到 [-1,1] 会把"静止(c0=0)"和"全速反向"混到同一点，并诱导
  模型预测 OOD 负值。故每通道按**操作上限(hi6)*固定归一到 [0,1]：rest=0、full=1、
  零输入→零增量。骨架坐标归一化(dataset 的 pc_center/scale 到 [-1,1])是空间几何，
  与此无关，保留。

输入:
  --masks-dir  derived/<seq>/masks/      (segment_batch 产物，0/255 PNG)
  --actions    raw/<seq>/actions6.csv    (表头 t_sec,c0..c5)
  --action-channels 0                    (本序列只驱动 ch0 → action_dim=1)
输出:
  <out-root>/train/<seq>_train.npz  +  <out-root>/val/<seq>_val.npz
  每个 npz: positions:(T,3,31) float32, actions:(T,A) float32 (已归一化到 [0,1])
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.skeleton_2d import batch_extract_skeleton_2d  # noqa: E402


def load_actions(csv_path, channels):
    """actions6.csv → (T, A) 取指定通道列(原始 kPa)。跳表头。"""
    raw = np.atleast_2d(np.genfromtxt(csv_path, delimiter=",", dtype=float))
    while raw.shape[0] and np.isnan(raw[0]).all():        # 跳表头
        raw = raw[1:]
    cols = [int(c) + 1 for c in channels]                  # +1：第 0 列是 t_sec
    return raw[:, cols].astype(np.float32)


def action_max_per_channel(seq_dir, channels, actions):
    """每通道归一化上限：优先 meta.json 的 hi6[ch](操作上限)；hi6=0/缺失则用数据 max。

    气动单向 → 每通道固定 [0,1] 上限（rest=0, full=hi6）；跨序列一致（c0=0.5 永远=75kPa）。
    """
    hi6 = None
    meta = os.path.join(seq_dir, "meta.json")
    if os.path.isfile(meta):
        try:
            with open(meta) as f:
                hi6 = json.load(f).get("hi6")
        except Exception:
            hi6 = None
    maxes = []
    for i, c in enumerate(channels):
        c = int(c)
        if hi6 is not None and c < len(hi6) and hi6[c] > 0:
            m = float(hi6[c])                               # 操作上限（首选）
        else:
            col = actions[:, i] if actions.shape[0] else np.array([1.0])
            m = float(col.max())
            m = m if m > 0 else 1.0                         # 兜底
        maxes.append(m)
    return np.array(maxes, np.float32)


def masks_to_positions(mask_dir, n_points=31, tip_fix=True):
    """mask PNG → (T,3,N) positions [col,row,0]。空 mask → 全 0 骨架(下游跳过)。

    tip_fix=True(默认): 末端 node0 做"垂直于局部轴切片"修正, 修弯管 cap 倾斜导致的
    node0 落角落 + node0-1-2 折角(实物 34% 帧受益, 末端误差 -71%)。详见
    src/utils/skeleton_2d.extract_skeleton_2d 的 tip_fix 参数。
    """
    fs = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not fs:
        sys.exit(f"无 mask: {mask_dir}")
    masks = np.stack([(cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
                      for f in fs])                         # (T,H,W)
    sk2d = batch_extract_skeleton_2d(masks, n_points, tip_fix=tip_fix)  # (T,N,2) [col,row]
    T, N, _ = sk2d.shape
    positions = np.zeros((T, 3, N), np.float32)
    positions[:, 0, :] = sk2d[:, :, 0]                      # col → x
    positions[:, 1, :] = sk2d[:, :, 1]                      # row → y
    # z=0（图像平面；1-DOF 平面弯曲在单相机下投影即主体形变）
    return positions, fs


def clean_outlier_skeletons(positions, deviation_px=80):
    """检测并修复离群骨架帧（管-臂合并/管茬使骨架中心线跑偏到画面边缘）。

    判据：某帧任一节点偏离该节点的**时间中位** > deviation_px → 判离群。
    正常臂最大偏离 ≤~66px(@p95)；离群帧 col 跑到 [1,636] 远离臂体 [296,346]。
    80px 落在两者间隙，干净分离。离群帧用前后最近有效帧线性插值替换（保时序连贯）。
    返回 (cleaned_positions, n_outlier, bad_mask)。
    """
    T = positions.shape[0]
    xy = positions[:, :2, :]
    med = np.median(xy, axis=0, keepdims=True)              # (1,2,N) 每节点时间中位
    dev = np.abs(xy - med).max(axis=(1, 2))                 # (T,) 每帧最大节点偏离
    bad = dev > deviation_px
    good_idx = np.where(~bad)[0]
    out = positions.copy()
    if len(good_idx) > 0:
        for i in np.where(bad)[0]:
            before = good_idx[good_idx < i]
            after = good_idx[good_idx > i]
            if len(before) and len(after):
                b, a = before[-1], after[0]
                t = (i - b) / max(1, (a - b))
                out[i] = positions[b] * (1 - t) + positions[a] * t
            elif len(before):
                out[i] = positions[before[-1]]
            elif len(after):
                out[i] = positions[after[0]]
    return out, int(bad.sum()), bad


def stabilize_static_region(positions, joint_xy, n_static=11):
    """绝对位置锚定的静态段共识稳定（用户选定：均值/共识方案）。

    双段臂: node0(图底/末端)..关节..node30(图顶/base)。只驱动末端 1-DOF → 动作段=
    node0..关节(保留每帧真实弯曲)；关节及以上(近端段)静止。但骨架按弧长重采样到 31 点,
    提取臂长帧间变化(分割差异/管茬)→ 同一物理关节落到不同 node id（实测 19-27, 中位 20；
    用户提示"不能用相对 node id"）。故用关节**绝对位置**每帧定位:
      1. 每帧关节 node = 离 joint_xy 最近的 node（handles id 漂移）。
      2. 静态段 = nodes[joint_node..N-1]，按弧长重采样到 n_static 点 → 跨帧中位 = 共识曲线。
      3. 每帧静态段 nodes ← 共识按每帧弧长映射回（保持该帧 node 数与序，col/row 都修）。
    动作段(node0..joint_node-1)原值不动。关节的连接处偏移(node~20 突偏右)、上方 mask
    缺块致 node30 col 抖动 等，都由共识修复。joint_xy 由调用方 robust 估计(见 detect_joint_xy)。

    Args:
        positions: (T,3,N) [col,row,0]（建议先过 clean_outlier_skeletons）。
        joint_xy: (2,) 关节绝对位置 [col,row]（固定；相机/近端不动）。
        n_static: 静态段弧长重采样点数。
    Returns:
        stabilized: (T,3,N)。
        joint_node: (T,) int 每帧关节 node id（QC 用）。
        cons_col, cons_row: (n_static,) 静态段共识曲线。
    """
    T, _, N = positions.shape
    xy = positions[:, :2, :]
    out = positions.copy()
    anchor = np.asarray(joint_xy, np.float64)
    # 1) 每帧关节 node = 离绝对位置最近
    dist = np.sqrt(((xy - anchor[None, :, None]) ** 2).sum(1))   # (T,N)
    jn = dist.argmin(1).astype(int)
    # 2) 每帧静态段弧长重采样到 n_static 点
    u_grid = np.linspace(0, 1, n_static)
    cols = np.full((T, n_static), np.nan)
    rows = np.full((T, n_static), np.nan)
    for t in range(T):
        x = xy[t, 0, jn[t]:N]
        y = xy[t, 1, jn[t]:N]
        if len(x) < 2:
            continue
        seg = np.concatenate([[0.0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]).cumsum()
        if seg[-1] < 1e-6:
            continue
        u = seg / seg[-1]
        cols[t] = np.interp(u_grid, u, x)
        rows[t] = np.interp(u_grid, u, y)
    cons_col = np.nanmedian(cols, axis=0)
    cons_row = np.nanmedian(rows, axis=0)
    # 3) 共识按每帧弧长映射回该帧静态段
    for t in range(T):
        js = slice(int(jn[t]), N)
        x = xy[t, 0, js]
        y = xy[t, 1, js]
        if len(x) < 2:
            continue
        seg = np.concatenate([[0.0], np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)]).cumsum()
        if seg[-1] < 1e-6:
            continue
        u = seg / seg[-1]
        out[t, 0, js] = np.interp(u, u_grid, cons_col)
        out[t, 1, js] = np.interp(u, u_grid, cons_row)
    return out, jn, cons_col, cons_row


def detect_joint_xy(positions, node_lo=8, node_hi=25):
    """robust 估计关节绝对位置 [col,row]。

    关节处管-臂合并 → 局部 col 突偏(2nd-diff)；跨帧该位置稳定突偏。排除末端真实弯曲
    (node0-7 高曲率)与上方 mask 缺块噪声区(node26-30)，在 node_lo..node_hi 范围取跨帧
    mean|Δ²col| 峰值 node，返回其中位 (col,row)。默认搜 node8-25(关节实测落在 node19-21)。
    """
    T, _, N = positions.shape
    if node_hi is None:
        node_hi = N - 3
    xy = positions[:, :2, :]
    mean_d2 = np.abs(np.diff(xy[:, 0, :], n=2, axis=1)).mean(axis=0)   # idx0..N-3 ↔ node1..N-2
    sub = mean_d2[node_lo - 1:node_hi]                                  # node_lo..node_hi+1
    peak_node = (node_lo - 1) + int(sub.argmax()) + 1                   # node id (1..N-2)
    joint_xy = np.median(xy[:, :, peak_node], axis=0)
    return joint_xy, peak_node


def save_npz(path, positions, actions):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, positions=positions.astype(np.float32),
                        actions=actions.astype(np.float32))
    print(f"    {path}  positions={positions.shape} actions={actions.shape}")


def build_parser():
    pa = argparse.ArgumentParser(description="实物 mask+actions → transition npz（免标定）")
    pa.add_argument("--seq", required=True, help="raw 序列目录(取 actions6.csv + 默认 masks 路径)")
    pa.add_argument("--masks-dir", default=None,
                    help="mask 目录(默认 derived/<seq名>/masks)")
    pa.add_argument("--actions", default=None,
                    help="actions6.csv(默认 <seq>/actions6.csv)")
    pa.add_argument("--action-channels", default="0",
                    help="逗号分隔的通道下标(默认 0=ch0；本序列单通道)")
    pa.add_argument("--action-max", default=None,
                    help="每通道归一化上限(逗号分隔, kPa)；默认读 meta.json hi6[ch]")
    pa.add_argument("--n-points", type=int, default=31)
    pa.add_argument("--tip-fix", action=argparse.BooleanOptionalAction, default=True,
                    help="末端 node0 垂直切片修正(修弯管 cap 角落偏移, 实物默认开; --no-tip-fix 关闭)")
    pa.add_argument("--skel-dev-thresh", type=float, default=80.0,
                    help="骨架离群判据(px)：偏离时间中位>此值→插值修复(默认80，落正常66与离群>100间隙)")
    pa.add_argument("--val-frac", type=float, default=0.2,
                    help="末尾连续 val 比例(时序连续切分，避免乱序泄漏)")
    pa.add_argument("--out-root", default=None,
                    help="输出根(默认 data/real_seq/<seq名>)")
    return pa


def main():
    args = build_parser().parse_args()
    seq = args.seq.rstrip("/")
    seq_name = os.path.basename(seq)
    masks_dir = args.masks_dir or os.path.abspath(
        os.path.join(os.path.dirname(seq), "..", "derived", seq_name, "masks"))
    actions_csv = args.actions or os.path.join(seq, "actions6.csv")
    out_root = args.out_root or os.path.abspath(
        os.path.join("data", "real_seq", seq_name))
    channels = [c.strip() for c in args.action_channels.split(",") if c.strip() != ""]

    print(f">>> 读 mask → 2D 骨架: {masks_dir}  (tip_fix={args.tip_fix})")
    positions, fs = masks_to_positions(masks_dir, args.n_points, tip_fix=args.tip_fix)
    T = positions.shape[0]
    valid = int((positions[:, :2, :].sum(axis=(1, 2)) > 0).sum())   # 非空骨架帧
    print(f"    {T} 帧, 非空骨架 {valid} ({valid/T*100:.1f}%)")

    # 清理管-臂合并/管茬导致的离群骨架（防止 col 跑到 [1,636] 污染归一化与监督）
    positions, n_out, bad = clean_outlier_skeletons(positions, args.skel_dev_thresh)
    print(f"    离群骨架修复: {n_out} 帧 ({n_out/T*100:.2f}%) → 时间插值替换"
          f" (阈值 {args.skel_dev_thresh}px)")
    outlier_path = os.path.join(out_root, "skeleton_outlier_frames.txt")
    os.makedirs(out_root, exist_ok=True)
    with open(outlier_path, "w") as f:
        f.write(f"# 离群骨架帧(管-臂合并等)，已时间插值修复。判据: 偏离时间中位>{args.skel_dev_thresh}px\n")
        f.write(" ".join(str(int(i)) for i in np.where(bad)[0]) + "\n")
        f.write(f"# 总计 {n_out}/{T} 帧\n")

    print(f">>> 读 actions: {actions_csv} 通道 {channels}")
    actions = load_actions(actions_csv, channels)
    assert len(actions) == T, f"帧数不匹配: positions {T} vs actions {len(actions)}"
    print(f"    actions(原始 kPa) {actions.shape} 范围 [{actions.min():.1f}, {actions.max():.1f}]")

    # 每通道固定归一到 [0,1]（气动单向半DOF：rest=0, full=操作上限；负值=反向驱动不合法）
    if args.action_max:
        maxes = np.array([float(x) for x in args.action_max.split(",")], np.float32)
        assert len(maxes) == len(channels), "--action-max 通道数与 --action-channels 不符"
    else:
        maxes = action_max_per_channel(seq, channels, actions)
    actions = actions / maxes                                  # (T,A) ∈ [0,1]
    print(f"    归一化上限 {maxes.tolist()} → [0,1]（rest=0, full=1, 半DOF）")

    # 连续时序切分（首 (1-v) 训练 / 末 v 验证）
    n_val = int(T * args.val_frac)
    n_train = T - n_val
    pos_tr, pos_va = positions[:n_train], positions[n_train:]
    act_tr, act_va = actions[:n_train], actions[n_train:]
    print(f">>> 切分: train {n_train} 帧 / val {n_val} 帧  → {out_root}")
    save_npz(os.path.join(out_root, "train", f"{seq_name}_train.npz"), pos_tr, act_tr)
    save_npz(os.path.join(out_root, "val", f"{seq_name}_val.npz"), pos_va, act_va)

    print(f"\n>>> 完成。训练: --data_dir {os.path.join(out_root,'train')}")
    print(f"           验证: {os.path.join(out_root,'val')}")
    print(f"    action_dim={len(channels)} n_nodes={args.n_points}（train_transition.py 自动探测）")


if __name__ == "__main__":
    main()
