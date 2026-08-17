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
  --action-channels auto                 (约束序列自动得到 0,1,3,4 模型视图)
输出:
  <out-root>/train/<seq>_train.npz  +  <out-root>/val/<seq>_val.npz
  每个 npz: positions:(T,3,15) float32, actions:(T,6) float32 (已归一化到 [0,1])
  Dataset 再按 model_action_channels 投影为模型使用的四维动作。
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


EQUALITY_TOLERANCE_KPA = 0.5


def load_capture_metadata(seq_dir):
    """读取采集合同；旧序列没有 ``meta.json`` 时保持向后兼容。"""
    path = os.path.join(seq_dir, "meta.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"meta.json 顶层必须是对象: {path}")
    return value


def normalize_channel_equalities(pairs):
    """本地校验采集元数据，避免前处理依赖 GUI/Qt 硬件模块。"""
    result = []
    used = set()
    for item in pairs or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("channel_equalities 每项必须是 [leader, follower]")
        leader, follower = int(item[0]), int(item[1])
        if leader == follower or leader not in range(6) or follower not in range(6):
            raise ValueError("channel_equalities 必须引用两个不同的 0..5 通道")
        if leader in used or follower in used:
            raise ValueError("channel_equalities 通道对不能重叠")
        used.update((leader, follower))
        result.append((leader, follower))
    return tuple(result)


def independent_channels_for_equalities(equalities):
    followers = {follower for _leader, follower in normalize_channel_equalities(equalities)}
    return tuple(channel for channel in range(6) if channel not in followers)


def action_expansion6(channel_map, equalities):
    """返回每个硬件通道应读取的模型动作列；受约束四维例为 (0,1,1,2,3,3)。"""
    mapping = tuple(int(channel) for channel in channel_map)
    lookup = {channel: index for index, channel in enumerate(mapping)}
    follower_sources = {follower: leader for leader, follower in
                        normalize_channel_equalities(equalities)}
    result = []
    for hardware_channel in range(6):
        source = follower_sources.get(hardware_channel, hardware_channel)
        if source not in lookup:
            raise ValueError(f"硬件 ch{hardware_channel} 无法从 model action 展开")
        result.append(lookup[source])
    return tuple(result)


def validate_action_equalities(actions, channels, equalities,
                               tolerance=EQUALITY_TOLERANCE_KPA):
    """验证未归一化 kPa 动作；返回每个等值对在全序列的最大残差。"""
    pairs = normalize_channel_equalities(equalities)
    if not pairs:
        return np.empty((0,), dtype=np.float32)
    if not np.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("channel_equality_tolerance_kpa 必须是非负有限数")
    channel_ids = tuple(int(channel) for channel in channels)
    if channel_ids != tuple(range(6)):
        raise ValueError(
            "验证 channel_equalities 必须传入原始六通道动作")
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"等值约束动作必须是 (T,6)，实际为 {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("actions6.csv 含 NaN/Inf，不能验证等值约束")
    residual_max = np.array(
        [np.max(np.abs(values[:, leader] - values[:, follower]), initial=0.0)
         for leader, follower in pairs],
        dtype=np.float32)
    bad = np.where(residual_max > float(tolerance))[0]
    if bad.size:
        details = ", ".join(
            f"ch{pairs[index][1]}=ch{pairs[index][0]} residual="
            f"{float(residual_max[index]):.6g}kPa"
            for index in bad)
        raise ValueError(
            f"actions6.csv 违反 channel_equalities（tolerance={tolerance:g}kPa）：{details}")
    return residual_max


def validate_equality_action_maxes(maxes, channels, equalities,
                                   tolerance=EQUALITY_TOLERANCE_KPA):
    """归一化尺度也必须保持等值列相同，否则会把同一 kPa 投到流形外。"""
    values = np.asarray(maxes, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(channels):
        raise ValueError("动作归一化上限维数与 action channels 不一致")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("动作归一化上限必须全部为正有限数")
    pairs = normalize_channel_equalities(equalities)
    if not pairs:
        return
    channel_ids = tuple(int(channel) for channel in channels)
    index = {channel: i for i, channel in enumerate(channel_ids)}
    for leader, follower in pairs:
        if leader not in index or follower not in index:
            raise ValueError("等值通道缺少动作归一化上限")
        if abs(values[index[leader]] - values[index[follower]]) > float(tolerance):
            raise ValueError(
                f"等值通道 ch{leader}/ch{follower} 的动作归一化上限必须相同")


def load_planarity_qc(seq_dir, explicit_path=None):
    """读取可选离面 QC；明确为失败的序列不进入训练集。"""
    path = explicit_path or os.path.join(seq_dir, "planarity_qc.json")
    if not os.path.isfile(path):
        if explicit_path:
            raise FileNotFoundError(f"找不到指定的 planarity_qc: {path}")
        return None
    with open(path, encoding="utf-8") as stream:
        qc = json.load(stream)
    if not isinstance(qc, dict):
        raise ValueError(f"planarity_qc 顶层必须是对象: {path}")
    if qc.get("planarity_pass") is False:
        raise ValueError(
            f"平面性质控未通过，拒绝写入训练集: {path}；失败序列应保留作诊断")
    return qc


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


def stabilize_static_region(positions, joint_xy, n_static=None):
    """绝对位置锚定的静态段共识稳定（用户选定：均值/共识方案）。

    n_static 为 None 时按 N 自适应(max(4, 0.4·N)), 任意 n_points 不需手调。

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
    if n_static is None:
        n_static = max(4, int(0.4 * N))            # 静态段弧长重采样点数(随 N 缩放)
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


def detect_joint_xy(positions, node_lo=None, node_hi=None):
    """robust 估计关节绝对位置 [col,row]。node_lo/node_hi 为 None 时按 N 的分数自适应
    (排除末端弯曲段~前25% 与 上方缺块噪声~后15%), 使任意 n_points 都不需手调。"""
    T, _, N = positions.shape
    if node_lo is None:
        node_lo = max(4, int(0.25 * N))
    if node_hi is None:
        node_hi = min(N - 3, int(0.85 * N))
    xy = positions[:, :2, :]
    mean_d2 = np.abs(np.diff(xy[:, 0, :], n=2, axis=1)).mean(axis=0)   # idx0..N-3 ↔ node1..N-2
    sub = mean_d2[node_lo - 1:node_hi]                                  # node_lo..node_hi+1
    peak_node = (node_lo - 1) + int(sub.argmax()) + 1                   # node id (1..N-2)
    joint_xy = np.median(xy[:, :, peak_node], axis=0)
    return joint_xy, peak_node


def save_npz(path, positions, actions, n_points=None, tip_fix=None,
             channel_equalities=(), pair_residual_max=None, planarity_qc=None,
             model_action_channels=(), action_expansion=None):
    """存 npz。n_points/tip_fix 作元数据存入(供训练 config.json 记录数据配置, 辨识模型用)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(positions=positions.astype(np.float32), actions=actions.astype(np.float32))
    if n_points is not None:
        kw['n_points'] = np.array(n_points)
    if tip_fix is not None:
        kw['tip_fix'] = np.array(bool(tip_fix))
    kw['channel_equalities'] = np.array(json.dumps(
        [list(pair) for pair in channel_equalities], separators=(",", ":")))
    kw['pair_residual_max'] = np.asarray(
        pair_residual_max if pair_residual_max is not None else [], dtype=np.float32)
    kw['raw_action_dim'] = np.array(actions.shape[1])
    kw['model_action_dim'] = np.array(len(model_action_channels) or actions.shape[1])
    kw['model_action_channels'] = np.asarray(
        model_action_channels or tuple(range(actions.shape[1])), dtype=np.int64)
    kw['action_expansion6'] = np.asarray(
        action_expansion if action_expansion is not None else [], dtype=np.int64)
    if planarity_qc is not None:
        kw['planarity_qc'] = np.array(json.dumps(
            planarity_qc, ensure_ascii=False, separators=(",", ":")))
    np.savez_compressed(path, **kw)
    print(f"    {path}  positions={positions.shape} actions={actions.shape}")


def build_parser():
    pa = argparse.ArgumentParser(description="实物 mask+actions → transition npz（免标定）")
    pa.add_argument("--seq", required=True, help="raw 序列目录(取 actions6.csv + 默认 masks 路径)")
    pa.add_argument("--masks-dir", default=None,
                    help="mask 目录(默认 derived/<seq名>/masks)")
    pa.add_argument("--actions", default=None,
                    help="actions6.csv(默认 <seq>/actions6.csv)")
    pa.add_argument("--action-channels", default="auto",
                    help="模型动作对应的独立硬件通道；auto:有等值约束时删除 follower，"
                         "否则保持旧版单通道 ch0")
    pa.add_argument("--action-max", default=None,
                    help="每通道归一化上限(逗号分隔, kPa)；默认读 meta.json hi6[ch]")
    pa.add_argument("--n-points", type=int, default=15,
                    help="骨架节点数(默认 15; 实测降节点误差不大, 全管线按 N 分数自适应)")
    pa.add_argument("--tip-fix", action=argparse.BooleanOptionalAction, default=True,
                    help="末端 node0 垂直切片修正(修弯管 cap 角落偏移, 实物默认开; --no-tip-fix 关闭)")
    pa.add_argument("--skel-dev-thresh", type=float, default=80.0,
                    help="骨架离群判据(px)：偏离时间中位>此值→插值修复(默认80，落正常66与离群>100间隙)")
    pa.add_argument("--val-frac", type=float, default=0.2,
                    help="末尾连续 val 比例(时序连续切分，避免乱序泄漏)")
    pa.add_argument("--out-root", default=None,
                    help="输出根(默认 data/real_seq/<seq名>)")
    pa.add_argument("--planarity-qc", default=None,
                    help="可选 planarity_qc.json；默认读取 <seq>/planarity_qc.json")
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
    meta = load_capture_metadata(seq)
    equalities = normalize_channel_equalities(meta.get("channel_equalities", ()))
    equality_tolerance = float(meta.get(
        "channel_equality_tolerance_kpa", EQUALITY_TOLERANCE_KPA))
    planarity_qc = load_planarity_qc(seq, args.planarity_qc)
    independent_channels = independent_channels_for_equalities(equalities)
    if args.action_channels == "auto":
        channels = independent_channels if equalities else (0,)
    else:
        channels = tuple(int(c.strip()) for c in args.action_channels.split(",")
                         if c.strip() != "")
    if equalities and channels != independent_channels:
        raise ValueError(
            "带 channel_equalities 的序列必须按 follower 删除后的固定顺序使用 "
            f"--action-channels {','.join(map(str, independent_channels))}")
    expansion = action_expansion6(independent_channels, equalities) if equalities else None

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

    print(f">>> 读 actions: {actions_csv} 原始六维；模型动作视图 {channels}")
    raw_actions6 = load_actions(actions_csv, range(6))
    assert len(raw_actions6) == T, f"帧数不匹配: positions {T} vs actions {len(raw_actions6)}"
    pair_residual_max = validate_action_equalities(
        raw_actions6, range(6), equalities, equality_tolerance)
    print(f"    actions(原始 kPa) {raw_actions6.shape} 范围 "
          f"[{raw_actions6.min():.1f}, {raw_actions6.max():.1f}]")
    if equalities:
        print(f"    等值约束 {equalities} 已验证，最大残差 {pair_residual_max.tolist()} kPa")

    # 每通道固定归一到 [0,1]（气动单向半DOF：rest=0, full=操作上限；负值=反向驱动不合法）
    if args.action_max:
        maxes = np.array([float(x) for x in args.action_max.split(",")], np.float32)
        if equalities and len(maxes) == len(channels):
            maxes = maxes[np.asarray(expansion, dtype=np.int64)]
        assert len(maxes) == 6, "六维 NPZ 的 --action-max 必须为六列，或为可展开的四个独立列"
    else:
        maxes = action_max_per_channel(seq, range(6), raw_actions6)
    if equalities:
        raw_maxes6 = action_max_per_channel(seq, range(6), raw_actions6)
        validate_equality_action_maxes(
            raw_maxes6, range(6), equalities, equality_tolerance)
    actions = raw_actions6 / maxes                             # (T,6) ∈ [0,1]
    print(f"    归一化上限 {maxes.tolist()} → [0,1]（rest=0, full=1, 半DOF）")

    # 连续时序切分（首 (1-v) 训练 / 末 v 验证）
    n_val = int(T * args.val_frac)
    n_train = T - n_val
    pos_tr, pos_va = positions[:n_train], positions[n_train:]
    act_tr, act_va = actions[:n_train], actions[n_train:]
    print(f">>> 切分: train {n_train} 帧 / val {n_val} 帧  → {out_root}")
    save_npz(os.path.join(out_root, "train", f"{seq_name}_train.npz"), pos_tr, act_tr,
             n_points=args.n_points, tip_fix=args.tip_fix,
             channel_equalities=equalities, pair_residual_max=pair_residual_max,
             planarity_qc=planarity_qc,
             model_action_channels=channels,
             action_expansion=expansion)
    save_npz(os.path.join(out_root, "val", f"{seq_name}_val.npz"), pos_va, act_va,
             n_points=args.n_points, tip_fix=args.tip_fix,
             channel_equalities=equalities, pair_residual_max=pair_residual_max,
             planarity_qc=planarity_qc,
             model_action_channels=channels,
             action_expansion=expansion)

    print(f"\n>>> 完成。训练: --data_dir {os.path.join(out_root,'train')}")
    print(f"           验证: {os.path.join(out_root,'val')}")
    print(f"    raw_action_dim=6 model_action_dim={len(channels)} n_nodes={args.n_points}"
          "（Dataset 按合同投影）")


if __name__ == "__main__":
    main()
