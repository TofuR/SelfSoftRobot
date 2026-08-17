"""clean_transition_npz.py — 增强清洗实物 transition npz（绝对位置锚定静态段共识）。

用户选定方案（共识/均值）：静态段(关节以上)用跨帧中位共识替换。

背景（用户观测 + 实测验证）:
  - 双段臂，只驱动末端 1-DOF → 动作段 = node0(图底)..关节；关节及以上(近端段)静止。
    弯曲集中在 node0-5(col_std 5-10)，衰减到关节(实测 node~20)。
  - **关节 node id 跨帧漂移(实测 19-27, 中位 20, 仅 64% 帧落在 node20)**：骨架按弧长
    重采样到 31 点，提取臂长帧间变化(分割差异/管茬)→ 同一物理关节落到不同 node id。
    → 不能用固定 node id（用户提示"不能用相对 node id，会有偏差"），必须用关节**绝对位置**
    （实测稳定: row 95.4±2.4, col 311±2.4）每帧定位。
  - 手干扰帧(原 02313-02369)整帧离群已由现有 npz 的 clean_outlier_skeletons 时间插值修复
    （raw node0 col 跑到 145-205, npz 已平滑回 303-307）。

本脚本在现有 npz 上额外做（动作气压全保留，时序/切分不变）:
  1. detect_joint_xy: robust 估计关节绝对位置（排除末端弯曲，取跨帧局部 col 突偏峰值 node 的中位）。
  2. stabilize_static_region(joint_xy): 每帧关节 node=离绝对位置最近；静态段(关节..node30)
     弧长重采样→跨帧中位共识→按每帧弧长映射回；动作段(node0..关节)原值不动。
  3. 残余离群(动作段核心 nodes0-18 偏离时间中位>阈值)→整帧时间插值（兜底）。

输入: data/real_seq/<seq>/{train,val}/*.npz
输出: data/real_seq/<seq>_clean/{train,val}/*.npz  (不覆盖原文件) + qc/

用法:
  python scripts/real/clean_transition_npz.py --seq seq_20260627_163921
  python scripts/real/clean_transition_npz.py --seq ... --act-dev-thresh 60
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.real.masks_to_transition_npz import (  # noqa: E402
    stabilize_static_region, detect_joint_xy)


def interpolate_frames(positions, bad):
    """对 bad 帧整帧线性插值（前后最近有效帧）。"""
    T = positions.shape[0]
    good_idx = np.where(~bad)[0]
    out = positions.copy()
    if len(good_idx) == 0:
        return out
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
    return out


def clean_split(positions, joint_xy, act_dev_thresh, act_nodes=None):
    """增强清洗。act_nodes=None→0.6·N(动作段核心节点数, 用于残余离群检测), 任意 N 自适应。"""
    T, _, N = positions.shape
    if act_nodes is None:
        act_nodes = max(5, int(0.6 * N))
    raw = positions.copy()
    stab, jn, ccol, crow = stabilize_static_region(positions, joint_xy)
    act_xy = stab[:, :2, :act_nodes + 1]
    act_med = np.median(act_xy, axis=0, keepdims=True)
    act_dev = np.abs(act_xy - act_med).max(axis=(1, 2))
    bad = act_dev > act_dev_thresh
    cleaned = interpolate_frames(stab, bad) if bad.any() else stab
    return cleaned, raw, jn, ccol, crow, int(bad.sum()), bad


def draw_skel(img, skel3, color, pt=3, lw=2):
    if skel3 is None or np.abs(skel3[:2]).max() == 0:
        return
    pts = skel3[:2].T.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, lw, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), pt, color, -1, cv2.LINE_AA)


def qc_montage(raw_all, clean_all, frame_ids, cam0, masks_dir, out_path,
               mask_color=(0, 0, 255), raw_color=(255, 255, 0),
               clean_color=(0, 255, 255), mask_alpha=0.3):
    cells = []
    for fi in frame_ids:
        if fi >= len(raw_all):
            continue
        ip = os.path.join(cam0, f"{fi:05d}.png")
        mp = os.path.join(masks_dir, f"{fi:05d}.png")
        if not os.path.isfile(ip):
            continue
        img = cv2.imread(ip)
        if os.path.isfile(mp):
            m01 = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            if m01.any():
                ov = img.copy(); ov[m01 > 0] = mask_color
                cv2.addWeighted(ov, mask_alpha, img, 1 - mask_alpha, 0, dst=img)
        draw_skel(img, raw_all[fi], raw_color)        # 清洗前(青)
        draw_skel(img, clean_all[fi], clean_color)    # 清洗后(黄)
        cv2.putText(img, f"f{fi}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cells.append(img)
    if not cells:
        return
    h, w = cells[0].shape[:2]
    cols = 4
    rows = int(np.ceil(len(cells) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(cells):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    cv2.imwrite(out_path, canvas)


def process_npz(npz_path, out_path, act_dev_thresh, act_nodes, joint_xy=None):
    d = np.load(npz_path)
    pos = d['positions'].astype(np.float32)
    act = d['actions'].astype(np.float32)
    _meta = {k: d[k].item() for k in ('n_points', 'tip_fix') if k in d}   # 保留数据配置元数据
    if joint_xy is None:
        joint_xy, peak_node = detect_joint_xy(pos)
    else:
        _, peak_node = detect_joint_xy(pos)
    if act_nodes is None:
        act_nodes = max(5, int(0.6 * pos.shape[2]))
    cleaned, raw, jn, ccol, crow, n_out, bad = clean_split(
        pos, joint_xy, act_dev_thresh, act_nodes)
    T, _, N = pos.shape
    act_std_before = raw[:, :2, :act_nodes + 1].std(0).mean()
    act_std_after = cleaned[:, :2, :act_nodes + 1].std(0).mean()
    print(f"  {os.path.basename(npz_path)}: T={T}")
    print(f"    关节绝对位置: col={joint_xy[0]:.1f} row={joint_xy[1]:.1f} (检测峰 node{peak_node})")
    print(f"    每帧关节 node id: 范围[{jn.min()},{jn.max()}] 中位{int(np.median(jn))}"
          f" → 仅{int((jn==int(np.median(jn))).sum()/T*100)}%帧落在中位 node(故用绝对位置锚定)")
    print(f"    动作段(node0-{act_nodes}) 时间 std: 清洗前 {act_std_before:.2f} → 后 {act_std_after:.2f}px (应≈不变)")
    print(f"    残余离群(动作段>{act_dev_thresh}px)插值: {n_out} 帧")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kw = dict(positions=cleaned.astype(np.float32), actions=act.astype(np.float32))
    for k, v in _meta.items():           # 透传 n_points/tip_fix 元数据
        kw[k] = np.array(v)
    np.savez_compressed(out_path, **kw)
    print(f"    → {os.path.relpath(out_path)}")
    return cleaned, raw, bad, joint_xy


def main(argv=None):
    pa = argparse.ArgumentParser(description='增强清洗 transition npz（绝对位置锚定静态段共识）')
    pa.add_argument('--seq', required=True, help='序列名(如 seq_20260627_163921)')
    pa.add_argument('--in-root', default=None, help='输入根(默认 data/real_seq/<seq>)')
    pa.add_argument('--out-root', default=None, help='输出根(默认 data/real_seq/<seq>_clean)')
    pa.add_argument('--act-dev-thresh', type=float, default=60.0,
                    help='动作段残余离群阈值 px(默认 60: 实测真实极端弯曲最大偏离~48px,'
                         '腐败>80px 已被 clean_outlier 处理,故 60 只兜底捕获残留腐败,不误伤真实极端弯曲)')
    pa.add_argument('--act-nodes', type=int, default=None,
                    help='动作段核心节点数(用于残余离群检测;默认None=0.6·N 自适应)')
    pa.add_argument('--joint-col', type=float, default=None, help='手动指定关节 col(覆盖自动检测)')
    pa.add_argument('--joint-row', type=float, default=None, help='手动指定关节 row(覆盖自动检测)')
    pa.add_argument('--cam0', default=None, help='原图目录(默认 real_capture/data/raw/<seq>/cam0)')
    pa.add_argument('--masks-dir', default=None, help='mask 目录(QC;默认 derived/<seq>/masks)')
    args = pa.parse_args(argv)

    in_root = args.in_root or os.path.join(PROJECT_ROOT, 'data', 'real_seq', args.seq)
    out_root = args.out_root or os.path.join(PROJECT_ROOT, 'data', 'real_seq', args.seq + '_clean')
    cam0 = args.cam0 or os.path.join(PROJECT_ROOT, 'real_capture', 'data', 'raw', args.seq, 'cam0')
    masks_dir = args.masks_dir or os.path.join(
        PROJECT_ROOT, 'real_capture', 'data', 'derived', args.seq, 'masks')
    qc_dir = os.path.join(out_root, 'qc')
    os.makedirs(qc_dir, exist_ok=True)
    manual_joint = None
    if args.joint_col is not None and args.joint_row is not None:
        manual_joint = np.array([args.joint_col, args.joint_row], np.float64)

    raw_train, clean_train, joint_xy = None, None, None
    for split, subdir in [('train', 'train'), ('val', 'val')]:
        npzs = sorted(glob.glob(os.path.join(in_root, subdir, '*.npz')))
        if not npzs:
            print(f"[跳过] {subdir}: 无 npz")
            continue
        out_path = os.path.join(out_root, subdir, os.path.basename(npzs[0]))
        jxy = manual_joint if manual_joint is not None else joint_xy
        cleaned, raw, bad, joint_xy = process_npz(
            npzs[0], out_path, args.act_dev_thresh, args.act_nodes, joint_xy=jxy)
        if split == 'train':
            raw_train, clean_train = raw, cleaned
            with open(os.path.join(qc_dir, 'outlier_frames_train.txt'), 'w') as f:
                f.write(f"# 残余离群帧(动作段>{args.act_dev_thresh}px)，整帧时间插值\n")
                f.write(' '.join(str(int(i)) for i in np.where(bad)[0]) + '\n')

    if raw_train is not None and os.path.isdir(cam0):
        frame_ids = [f for f in [100, 1000, 2313, 2330, 4079, 4080, 5000, 6500, 7800, 7950]
                     if f < len(raw_train)]
        out_png = os.path.join(qc_dir, 'clean_qc_montage.png')
        qc_montage(raw_train, clean_train, frame_ids, cam0, masks_dir, out_png)
        print(f"\n  QC montage: {os.path.relpath(out_png)}  (青=清洗前, 黄=清洗后, 红=mask)")

    print(f"\n完成 → {out_root}")
    print(f"  训练: --data_dir {os.path.join(out_root,'train')}")


if __name__ == '__main__':
    main()
