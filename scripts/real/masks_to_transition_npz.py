"""masks_to_transition_npz.py — 实物 mask + actions → transition 训练 npz（免标定）。

为什么不需要标定（核心）:
  state_transition 模型的 state 是 3D 中心线骨架 s∈R^{N×3}（每个节点 3 坐标），
  模型只消费 (prev_skeleton, action) → next_skeleton，**不碰图像、不需要相机参数**。
  对满足平面约束的单相机实验，直接用 mask 的 2D 图像骨架作 state（第 3 维 z=0）：
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
  --masks-dir  sam2/masks/<seq>_full/    (SAM2 产物，0/255 PNG)
  --actions    raw/<seq>/actions6.csv    (表头 t_sec,c0..c5)
  --action-channels auto                 (约束序列自动得到 0,1,3,4 模型视图)
输出:
  <out-root>/train/<seq>_train.npz  +  <out-root>/val/<seq>_val.npz
  每个 npz: positions:(T,3,15) float32, actions:(T,6) float32 (已归一化到 [0,1])
  Dataset 再按 model_action_channels 投影为模型使用的 1–6 维动作。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from real_validation.perception.skeleton import (  # noqa: E402
    extract_centerline_2d,
)


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


def normalize_channel_sources(sources=None, equalities=()):
    """本地解析权威 channel_source6；旧 channel_equalities 自动迁移。"""
    if sources is None:
        values = list(range(6))
        for item in equalities or ():
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("channel_equalities 每项必须是 [leader, follower]")
            leader, follower = int(item[0]), int(item[1])
            if leader == follower or leader not in range(6) or follower not in range(6):
                raise ValueError("channel_equalities 必须引用两个不同的 0..5 通道")
            values[follower] = leader
    else:
        values = tuple(int(value) for value in sources)
        if len(values) != 6 or any(value not in range(6) for value in values):
            raise ValueError("channel_source6 必须是 6 个 0..5 通道下标")

    def root(start):
        seen = set()
        current = start
        while values[current] != current:
            if current in seen:
                raise ValueError("channel_source6 不能包含循环")
            seen.add(current)
            current = values[current]
        return current

    return tuple(root(channel) for channel in range(6))


def channel_equalities_from_sources(sources):
    normalized = normalize_channel_sources(sources)
    return tuple((source, channel) for channel, source in enumerate(normalized)
                 if channel != source)


def normalize_channel_equalities(pairs):
    """旧 pair 合同兼容入口。"""
    return channel_equalities_from_sources(
        normalize_channel_sources(equalities=pairs))


def independent_channels_for_sources(sources):
    normalized = normalize_channel_sources(sources)
    return tuple(channel for channel, source in enumerate(normalized)
                 if channel == source)


def independent_channels_for_equalities(equalities):
    return independent_channels_for_sources(
        normalize_channel_sources(equalities=equalities))


def action_expansion6(channel_map, equalities=(), channel_sources=None):
    """返回每个硬件通道读取的模型动作列。"""
    sources = normalize_channel_sources(channel_sources, equalities)
    mapping = tuple(int(channel) for channel in channel_map)
    lookup = {channel: index for index, channel in enumerate(mapping)}
    try:
        return tuple(lookup[source] for source in sources)
    except KeyError as error:
        raise ValueError(f"根通道 ch{int(error.args[0])} 未进入 model action") from error


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


def masks_to_positions(mask_dir, n_points=31, tip_fix=True, endpoint_fix=True,
                       skeleton_method="row_centroid", segment_lengths=(1.0,),
                       base_anchor_xy=None, return_qc=False):
    """mask PNG → (T,3,N) positions [col,row,0]。空 mask → 全 0 骨架(下游跳过)。

    skeletonize/medial_axis 默认使用 endpoint_fix 同时把 tip/base 延伸到端帽宽边中心；
    tip_fix 只保留给旧 row_centroid 的单端垂直切片兼容逻辑。
    """
    fs = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    if not fs:
        sys.exit(f"无 mask: {mask_dir}")
    skels = []
    qc = []
    for index, path in enumerate(fs):
        mask = (cv2.imread(path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        skel, info = extract_centerline_2d(
            mask, n_points=n_points, method=skeleton_method,
            segment_lengths=segment_lengths, base_anchor_xy=base_anchor_xy,
            tip_fix=tip_fix, endpoint_fix=endpoint_fix, return_info=True)
        skels.append(skel)
        qc.append({
            "index": index,
            "frame": int(os.path.splitext(os.path.basename(path))[0]),
            "mask_area": int(mask.sum()),
            **info,
        })
    sk2d = np.asarray(skels, dtype=np.float32)
    T, N, _ = sk2d.shape
    positions = np.zeros((T, 3, N), np.float32)
    positions[:, 0, :] = sk2d[:, :, 0]                      # col → x
    positions[:, 1, :] = sk2d[:, :, 1]                      # row → y
    # z=0（图像平面；动作维度与该几何表示相互独立）
    result = (positions, fs, qc) if return_qc else (positions, fs)
    return result


def temporal_skeleton_qc(positions, extraction_qc):
    """通用时间QC：只标记，不假设某一段静止，也不自动删除极端合法形态。"""
    T = positions.shape[0]
    xy = positions[:, :2, :].transpose(0, 2, 1)
    residual = np.zeros(T, dtype=np.float32)
    if T >= 3:
        midpoint = 0.5 * (xy[:-2] + xy[2:])
        residual[1:-1] = np.linalg.norm(xy[1:-1] - midpoint, axis=-1).max(axis=1)
    lengths = np.linalg.norm(np.diff(xy, axis=1), axis=-1).sum(axis=1)
    hard_invalid = np.array([
        not bool(item.get("success")) or not np.isfinite(xy[i]).all()
        for i, item in enumerate(extraction_qc)
    ], dtype=bool)
    positive_lengths = lengths[lengths > 0]
    median_length = float(np.median(positive_lengths)) if len(positive_lengths) else 0.0
    if median_length > 0:
        # 软体段会弯曲但物理弧长不会瞬间减半；这是明确的主路径提取失败，可自动修复。
        hard_invalid |= lengths < 0.5 * median_length

    def robust_flag(values, z=8.0):
        values = np.asarray(values, dtype=float)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        scale = max(1.4826 * mad, 1.0)
        return np.abs(values - med) > z * scale

    suspicious = robust_flag(lengths) | robust_flag(residual)
    for index, item in enumerate(extraction_qc):
        item["resampled_length_px"] = float(lengths[index])
        item["temporal_midpoint_residual_px"] = float(residual[index])
        item["hard_invalid"] = bool(hard_invalid[index])
        item["suspicious"] = bool(suspicious[index] and not hard_invalid[index])
    return hard_invalid, suspicious


def interpolate_flagged_frames(positions, flagged):
    """只用于明确指定的坏帧；完整六通道形态不会按全局中位被误删。"""
    out = positions.copy()
    good = np.where(~np.asarray(flagged, dtype=bool))[0]
    if not len(good):
        return out
    for index in np.where(flagged)[0]:
        before = good[good < index]
        after = good[good > index]
        if len(before) and len(after):
            lo, hi = before[-1], after[0]
            alpha = (index - lo) / max(1, hi - lo)
            out[index] = positions[lo] * (1 - alpha) + positions[hi] * alpha
        elif len(before):
            out[index] = positions[before[-1]]
        elif len(after):
            out[index] = positions[after[0]]
    return out


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


def save_npz(path, positions, actions, n_points=None, tip_fix=None,
             endpoint_fix=None,
             channel_equalities=(), channel_sources=None,
             pair_residual_max=None, planarity_qc=None,
             model_action_channels=(), action_expansion=None,
             skeleton_method=None, segment_lengths=(), segment_intervals=(),
             joint_node_indices=()):
    """存 npz。n_points/tip_fix 作元数据存入(供训练 config.json 记录数据配置, 辨识模型用)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw = dict(positions=positions.astype(np.float32), actions=actions.astype(np.float32))
    if n_points is not None:
        kw['n_points'] = np.array(n_points)
    if tip_fix is not None:
        kw['tip_fix'] = np.array(bool(tip_fix))
    if endpoint_fix is not None:
        kw['endpoint_fix'] = np.array(bool(endpoint_fix))
    if skeleton_method is not None:
        kw['skeleton_method'] = np.array(str(skeleton_method))
    kw['node_order'] = np.array('tip_to_base')
    kw['segment_lengths'] = np.asarray(segment_lengths, dtype=np.float32)
    kw['segment_intervals'] = np.asarray(segment_intervals, dtype=np.int64)
    kw['joint_node_indices'] = np.asarray(joint_node_indices, dtype=np.int64)
    has_source_contract = channel_sources is not None or bool(channel_equalities)
    sources = (normalize_channel_sources(channel_sources, channel_equalities)
               if has_source_contract else ())
    equalities = channel_equalities_from_sources(sources) if sources else ()
    if sources:
        kw['channel_source6'] = np.asarray(sources, dtype=np.int64)
    kw['channel_equalities'] = np.array(json.dumps(
        [list(pair) for pair in equalities], separators=(",", ":")))
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


def save_conversion_qc(out_root, seq, mask_dir, frame_paths, extraction_qc,
                       raw_positions, final_positions, joint_node_indices):
    """自动保存骨架阶段的逐帧表、曲线和原图叠加对比。"""
    qc_dir = os.path.join(out_root, "qc_skeleton")
    os.makedirs(qc_dir, exist_ok=True)
    fields = sorted({key for row in extraction_qc for key in row})
    with open(os.path.join(qc_dir, "skeleton_metrics.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in extraction_qc:
            writer.writerow({key: (json.dumps(value) if isinstance(value, (tuple, list)) else value)
                             for key, value in row.items()})

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(extraction_qc))
        length = np.array([row.get("resampled_length_px", np.nan) for row in extraction_qc])
        residual = np.array([row.get("temporal_midpoint_residual_px", np.nan)
                             for row in extraction_qc])
        area = np.array([row.get("mask_area", 0) for row in extraction_qc])
        fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        for ax, values, label in zip(
                axes, (area, length, residual),
                ("mask area [px]", "centerline length [px]", "temporal residual [px]")):
            ax.plot(x, values, lw=.65); ax.set_ylabel(label); ax.grid(alpha=.25)
        bad = [i for i, row in enumerate(extraction_qc)
               if row.get("hard_invalid") or row.get("suspicious")]
        for index in bad:
            for ax in axes:
                ax.axvline(index, color="tab:red", alpha=.12, lw=.8)
        axes[-1].set_xlabel("sample index")
        fig.tight_layout(); fig.savefig(os.path.join(qc_dir, "skeleton_metrics.png"), dpi=120)
        plt.close(fig)
    except Exception as error:
        print(f"    [QC] 曲线跳过: {error}")

    count = len(frame_paths)
    chosen = np.linspace(0, count - 1, min(12, count)).astype(int).tolist() if count else []
    flagged = [i for i, row in enumerate(extraction_qc)
               if row.get("hard_invalid") or row.get("suspicious")][:8]
    chosen = list(dict.fromkeys(chosen + flagged))
    cam0 = os.path.join(seq, "cam0")
    cells = []
    palette = [(0, 220, 0), (255, 80, 0), (220, 0, 220), (0, 180, 255)]
    boundaries = (0,) + tuple(int(v) for v in joint_node_indices) + (raw_positions.shape[2] - 1,)
    for index in chosen:
        frame = int(extraction_qc[index]["frame"])
        image_path = os.path.join(cam0, f"{frame:05d}.png")
        image = cv2.imread(image_path)
        mask = cv2.imread(frame_paths[index], cv2.IMREAD_GRAYSCALE)
        if image is None and mask is not None:
            image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if image is None:
            continue
        if mask is not None:
            tint = image.copy(); tint[mask > 127] = (0, 0, 255)
            cv2.addWeighted(tint, .24, image, .76, 0, dst=image)
        raw_pts = raw_positions[index, :2, :].T.astype(np.int32)
        final_pts = final_positions[index, :2, :].T.astype(np.int32)
        if not np.array_equal(raw_pts, final_pts):
            cv2.polylines(image, [raw_pts.reshape(-1, 1, 2)], False,
                          (255, 255, 0), 1, cv2.LINE_AA)
        for seg_index, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            pts = final_pts[lo:hi + 1].reshape(-1, 1, 2)
            cv2.polylines(image, [pts], False, palette[seg_index % len(palette)], 2,
                          cv2.LINE_AA)
        for node in range(len(final_pts)):
            color = (0, 255, 255) if node in joint_node_indices else (255, 255, 255)
            cv2.circle(image, tuple(final_pts[node]), 2, color, -1, cv2.LINE_AA)
        row = extraction_qc[index]
        for raw_key, fixed_key in (("raw_tip_xy", "fixed_tip_xy"),
                                   ("raw_base_xy", "fixed_base_xy")):
            if raw_key not in row or fixed_key not in row:
                continue
            raw_endpoint = tuple(np.rint(row[raw_key]).astype(int))
            fixed_endpoint = tuple(np.rint(row[fixed_key]).astype(int))
            cv2.line(image, raw_endpoint, fixed_endpoint, (255, 255, 0), 1,
                     cv2.LINE_AA)
            cv2.drawMarker(image, raw_endpoint, (255, 255, 0), cv2.MARKER_TILTED_CROSS,
                           7, 1, cv2.LINE_AA)
            cv2.circle(image, fixed_endpoint, 3, (0, 255, 255), 1, cv2.LINE_AA)
        state = "INVALID" if row.get("hard_invalid") else (
            "CHECK" if row.get("suspicious") else "OK")
        cv2.putText(image, f"f{frame} {state} L={row.get('resampled_length_px', 0):.1f}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cells.append(image)
    if cells:
        cols = 4
        H, W = cells[0].shape[:2]
        rows = int(np.ceil(len(cells) / cols))
        canvas = np.zeros((rows * H, cols * W, 3), np.uint8)
        for k, image in enumerate(cells):
            r, c = divmod(k, cols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = image
        cv2.imwrite(os.path.join(qc_dir, "skeleton_overlays.png"), canvas)
    with open(os.path.join(qc_dir, "README.txt"), "w") as handle:
        handle.write("红色=输入mask；彩色线=各机器人段；黄色节点=关节；")
        handle.write("青色线(若出现)=时间自动修复前中心线。\n")
        handle.write("端部青色叉=细化原端点；黄色圆=双端端帽修正点；短青线=修正位移。\n")


def build_parser():
    pa = argparse.ArgumentParser(description="实物 mask+actions → transition npz（免标定）")
    pa.add_argument("--seq", required=True, help="raw 序列目录(取 actions6.csv + 默认 masks 路径)")
    pa.add_argument("--masks-dir", default=None,
                    help="mask 目录(默认 derived/<seq名>/masks)")
    pa.add_argument("--actions", default=None,
                    help="actions6.csv(默认 <seq>/actions6.csv)")
    pa.add_argument("--action-channels", default="auto",
                    help="模型动作对应的来源根通道；auto:按 channel_source6 推导，"
                         "无合同旧数据沿用 active_channel")
    pa.add_argument("--action-max", default=None,
                    help="每通道归一化上限(逗号分隔, kPa)；默认读 meta.json hi6[ch]")
    pa.add_argument("--n-points", type=int, default=15,
                    help="骨架节点数(默认 15; 实测降节点误差不大, 全管线按 N 分数自适应)")
    pa.add_argument("--skeleton-method",
                    choices=("skeletonize", "medial_axis", "row_centroid"),
                    default="skeletonize",
                    help="默认快速细化主路径；medial_axis更慢；row_centroid仅兼容旧数据")
    pa.add_argument("--segment-lengths", default="1,1",
                    help="tip→base各物理段相对长度；默认两段等长，15节点得到7+7区间")
    pa.add_argument("--base-anchor", default=None,
                    help="可选基座像素x,y；默认以主路径较上端为base")
    pa.add_argument("--tip-fix", action=argparse.BooleanOptionalAction, default=True,
                    help="仅旧row_centroid：末端node0垂直切片修正")
    pa.add_argument("--endpoint-fix", action=argparse.BooleanOptionalAction, default=True,
                    help="skeletonize/medial_axis双端端帽中心修正（默认开）")
    pa.add_argument("--skel-dev-thresh", type=float, default=80.0,
                    help="仅--legacy-global-outlier时使用的旧全局中位阈值")
    pa.add_argument("--legacy-global-outlier", action="store_true",
                    help="旧单通道数据兼容：按全序列中位修复；六通道通用流程禁止默认启用")
    pa.add_argument("--repair-suspicious", action="store_true",
                    help="除提取失败外，也插值修复时间QC可疑帧；默认只标记供QC")
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
    has_source_contract = "channel_source6" in meta
    sources = normalize_channel_sources(
        meta.get("channel_source6"), meta.get("channel_equalities", ()))
    equalities = channel_equalities_from_sources(sources)
    equality_tolerance = float(meta.get(
        "channel_equality_tolerance_kpa", EQUALITY_TOLERANCE_KPA))
    planarity_qc = load_planarity_qc(seq, args.planarity_qc)
    independent_channels = independent_channels_for_sources(sources)
    source_contract = sources if (has_source_contract or equalities) else None
    if args.action_channels == "auto":
        # identity source map 表示六路独立；无合同旧数据保持单通道兼容。
        channels = independent_channels if source_contract else (
            int(meta.get("active_channel", 0)),)
    else:
        channels = tuple(int(c.strip()) for c in args.action_channels.split(",")
                         if c.strip() != "")
    if source_contract and channels != independent_channels:
        raise ValueError(
            "带 channel_source6 的序列必须按来源根通道的固定顺序使用 "
            f"--action-channels {','.join(map(str, independent_channels))}")
    expansion = (action_expansion6(channels, channel_sources=source_contract)
                 if source_contract else None)

    segment_lengths = tuple(float(value) for value in args.segment_lengths.split(",")
                            if value.strip())
    base_anchor = (tuple(float(value) for value in args.base_anchor.split(","))
                   if args.base_anchor else None)
    if base_anchor is not None and len(base_anchor) != 2:
        raise ValueError("--base-anchor 必须是 x,y")
    print(f">>> 读 mask → 2D 骨架: {masks_dir}  method={args.skeleton_method} "
          f"segments={segment_lengths} endpoint_fix={args.endpoint_fix} "
          f"legacy_tip_fix={args.tip_fix}")
    positions, fs, extraction_qc = masks_to_positions(
        masks_dir, args.n_points, tip_fix=args.tip_fix, endpoint_fix=args.endpoint_fix,
        skeleton_method=args.skeleton_method, segment_lengths=segment_lengths,
        base_anchor_xy=base_anchor, return_qc=True)
    T = positions.shape[0]
    valid = int((positions[:, :2, :].sum(axis=(1, 2)) > 0).sum())   # 非空骨架帧
    print(f"    {T} 帧, 非空骨架 {valid} ({valid/T*100:.1f}%)")

    raw_positions = positions.copy()
    hard_invalid, suspicious = temporal_skeleton_qc(positions, extraction_qc)
    repair = hard_invalid | (suspicious if args.repair_suspicious else False)
    positions = interpolate_flagged_frames(positions, repair)
    if args.legacy_global_outlier:
        positions, n_legacy, legacy_bad = clean_outlier_skeletons(
            positions, args.skel_dev_thresh)
        repair |= legacy_bad
        print(f"    [legacy] 全局中位修复 {n_legacy} 帧；仅适用于旧单通道数据")
    print(f"    提取失败自动插值: {int(hard_invalid.sum())} 帧；"
          f"时间QC可疑: {int(suspicious.sum())} 帧"
          f"（{'已插值' if args.repair_suspicious else '仅标记'}）")
    outlier_path = os.path.join(out_root, "skeleton_outlier_frames.txt")
    os.makedirs(out_root, exist_ok=True)
    with open(outlier_path, "w") as f:
        f.write("# hard_invalid（自动插值）\n")
        f.write(" ".join(str(int(i)) for i in np.where(hard_invalid)[0]) + "\n")
        f.write("# suspicious（默认仅标记；--repair-suspicious才插值）\n")
        f.write(" ".join(str(int(i)) for i in np.where(suspicious)[0]) + "\n")

    layout = next((item for item in extraction_qc if item.get("segment_intervals")), {})
    segment_intervals = tuple(int(v) for v in layout.get("segment_intervals", ()))
    joint_nodes = tuple(int(v) for v in layout.get("joint_node_indices", ()))
    save_conversion_qc(out_root, seq, masks_dir, fs, extraction_qc,
                       raw_positions, positions, joint_nodes)

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
        assert len(maxes) == 6, (
            "六维 NPZ 的 --action-max 必须为六列，或为可按 channel_source6 展开的根通道列")
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
             endpoint_fix=args.endpoint_fix,
             channel_equalities=equalities, channel_sources=source_contract,
             pair_residual_max=pair_residual_max,
             planarity_qc=planarity_qc,
             model_action_channels=channels,
             action_expansion=expansion,
             skeleton_method=args.skeleton_method,
             segment_lengths=segment_lengths, segment_intervals=segment_intervals,
             joint_node_indices=joint_nodes)
    save_npz(os.path.join(out_root, "val", f"{seq_name}_val.npz"), pos_va, act_va,
             n_points=args.n_points, tip_fix=args.tip_fix,
             endpoint_fix=args.endpoint_fix,
             channel_equalities=equalities, channel_sources=source_contract,
             pair_residual_max=pair_residual_max,
             planarity_qc=planarity_qc,
             model_action_channels=channels,
             action_expansion=expansion,
             skeleton_method=args.skeleton_method,
             segment_lengths=segment_lengths, segment_intervals=segment_intervals,
             joint_node_indices=joint_nodes)

    print(f"\n>>> 完成。训练: --data_dir {os.path.join(out_root,'train')}")
    print(f"           验证: {os.path.join(out_root,'val')}")
    print(f"    raw_action_dim=6 model_action_dim={len(channels)} n_nodes={args.n_points}"
          "（Dataset 按合同投影）")


if __name__ == "__main__":
    main()
