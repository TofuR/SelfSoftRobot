"""repair_masks.py — mask 级修复（**独立于 node 轨道**, 不重骨架化）。

为什么需要: 形态预测目标是 mask, 但当前所有清洗(clean_outlier/stabilize_static/tip_fix)
都在 **node 层**(骨架化后)。mask 本身的分割误差未修——如 f4080 静态段顶部被截成 w=17
(非常态 31), 非封闭孔(binary_fill_holes 填不了), node 层用共识把偏掉的点拉回了但 mask 仍错,
不能当形态 GT。

修法(用户指定: base 在边缘、宽度近常量、无缺失):
  - 静态段(关节以上, 跨帧稳定) **逐行宽共识**: 每行跨帧取 [min_col,max_col] 中位 = 共识宽,
    每帧该行替换为共识(修 f4080 的 w=17→31)。关节行由"跨帧逐行质心 col 的 std"自动定位
    (静态段 std 低, 弯曲段 std 高)。
  - 动作段(关节以下)不动(保留真实弯曲)。
  - 全图填小洞(binary_fill_holes, 兜底)。
独立轨道: 只产 derived/<seq>/masks_repaired/, 不碰 node npz(那是另一条清洗线)。

用法:
  python scripts/real/repair_masks.py --seq seq_20260627_163921
  python scripts/real/repair_masks.py --seq ... --joint-row 95   # 手动指定关节行
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_masks(masks_dir):
    fs = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
    if not fs:
        sys.exit(f"无 mask: {masks_dir}")
    return np.stack([(cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) for f in fs]), fs


def detect_joint_row(masks):
    """关节行 = 静态段(顶部稳定宽)结束处, 由**宽度凸起**(管-臂合并处更宽)定位。

    用宽度而非质心 std: 顶部质心 std 受缺失噪声(f4080 类截断)干扰(反而偏大), 而关节处
    的宽度凸起(~36 vs 常态 31)是结构性的、跨帧稳定 → 鲁棒。实测关节 row~96, 与 node 层
    detect_joint_xy(row~95.7) 一致。
    """
    T, H, W = masks.shape
    idx = np.arange(W)
    rows_with = np.where(masks.any(axis=2).any(axis=0))[0]   # (H,) 任一帧有 mask 的行
    r_max = int(rows_with.max()) if len(rows_with) else H
    med_w = np.zeros(H)
    for r in range(H):
        row = masks[:, r, :]
        has = row.any(1)
        if has.sum() < max(3, T * 0.1):
            continue
        med_w[r] = float(np.median(row.sum(1)[has]))
    top_end = max(3, r_max // 6)
    static_w = float(np.median(med_w[1:top_end])) if np.any(med_w[1:top_end] > 0) else 31.0
    # 关节凸起: 上半区宽度峰值(管-臂合并处最宽); 确认显著宽于 static 才采信, 否则兜底
    lo, hi = max(8, r_max // 10), r_max // 2
    seg = med_w[lo:hi]
    peak = lo + int(np.argmax(seg)) if seg.size else max(8, r_max // 3)
    joint_row = peak if med_w[peak] >= static_w + 2.0 else max(8, r_max // 3)
    return joint_row, med_w


def repair_static_segment(masks, joint_row):
    """静态段(行 0..joint_row-1)逐行替换为跨帧宽共识; 动作段不动; 填洞。"""
    T, H, W = masks.shape
    out = masks.copy()
    idx = np.arange(W)
    for r in range(min(joint_row, H)):
        row = masks[:, r, :]                 # (T, W)
        has = row.any(1)
        if has.sum() < max(3, T * 0.3):      # 多数帧该行无内容 → 不动(避免凭空造)
            continue
        left = np.where(row, idx, W).min(1).astype(float)
        right = np.where(row, idx, -1).max(1).astype(float)
        left[~has] = np.nan
        right[~has] = np.nan
        cmin = int(np.nanmedian(left))
        cmax = int(np.nanmedian(right))
        if cmax <= cmin:
            continue
        out[:, r, :] = 0
        out[:, r, cmin:cmax + 1] = 1         # 每帧该行 = 共识宽
    for t in range(T):                        # 填小洞(兜底, 实测实物多无封闭孔)
        out[t] = binary_fill_holes(out[t]).astype(np.uint8)
    return out


# ----------------------------- 动作段半mask 修复(时间插值) -----------------------------
def _detect_cap_rows(masks, joint_row):
    """末端 cap 行(管末端自然变窄, 不应补全): 底部连续窄段(跨帧共识)。"""
    T, H, W = masks.shape
    med_w = np.zeros(H)
    for r in range(joint_row, H):
        row = masks[:, r, :]
        h = row.any(1)
        if h.sum() < max(2, T * 0.1):
            continue
        med_w[r] = float(np.median(row.sum(1)[h]))
    act_w = med_w[med_w > 0]
    base_w = float(np.median(act_w)) if len(act_w) else 31.0
    cap = set()
    rows_with = np.where(med_w > 0)[0]
    if len(rows_with) == 0:
        return cap
    r = int(rows_with.max())
    while r >= joint_row:
        if 0 < med_w[r] < 0.7 * base_w:
            cap.add(int(r))
            r -= 1
        else:
            break
    return cap


def _corrupt_rows(mask, joint_row, exp_w, cap_rows, arm_col_margin=3):
    """判定该帧动作段哪些行是"半mask腐败行"。

    腐败判据(全满足): width < 0.65·exp_w 且非 cap; 左右边都不贴图像边缘; 与紧邻 healthy 行
    列重叠≥4(真半mask与主体管共边连续; 离散噪声行不连续→丢弃)。
    """
    H, W = mask.shape
    widths = np.array([int((mask[r] > 0.5).sum()) for r in range(H)])
    exp = np.full(H, exp_w, dtype=float)

    def edges(r):
        c = np.where(mask[r] > 0.5)[0]
        return (int(c.min()), int(c.max())) if len(c) else (None, None)

    def overlap(r1, r2):
        a, b = edges(r1), edges(r2)
        if a[0] is None or b[0] is None:
            return 0
        return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)

    def is_healthy_anchor(rn):
        if rn < joint_row or rn >= H or rn in cap_rows or widths[rn] == 0:
            return False
        if not (0.8 * exp_w <= widths[rn] <= 1.6 * exp_w):
            return False
        cn = np.where(mask[rn] > 0.5)[0]
        if len(cn) == 0 or cn.min() < arm_col_margin or cn.max() > W - 1 - arm_col_margin:
            return False
        return True

    candidates = []
    for r in range(joint_row, H):
        if r in cap_rows or widths[r] == 0:
            continue
        cl, cr = edges(r)
        if cl is None or cl < arm_col_margin or cr > W - 1 - arm_col_margin:
            continue
        if widths[r] >= 0.65 * exp_w:
            continue
        candidates.append(r)

    corrupt, visited = [], set()
    cand_set = set(candidates)
    for start in candidates:
        if start in visited:
            continue
        seg = [start]; visited.add(start)
        j = start + 1
        while j in cand_set and overlap(j - 1, j) >= 4:
            seg.append(j); visited.add(j); j += 1
        k = start - 1
        while k in cand_set and overlap(k, k + 1) >= 4:
            seg.insert(0, k); visited.add(k); k -= 1
        lo, hi = seg[0], seg[-1]
        if any(is_healthy_anchor(hi + d) or is_healthy_anchor(lo - d) for d in (1, 2, 3)):
            corrupt.extend(seg)
    return np.array(sorted(corrupt), dtype=int), exp


def repair_actuated(masks, joint_row, neighbors=(-2, -1, 1, 2),
                    width_ratio=0.65, fill_holes=True, verbose=False):
    """动作段半mask/缺块修复(**时间插值**为主, 宽度补全兜底)。

    为什么用时间插值: 那块缺的硅胶在单帧图像里就是看不见(半透明+光照), SAM2/阈值法都补不回;
    只有邻帧里那块可见 → 用邻帧补。半mask 的质心偏向腐败侧不能当锚, 改用"边重合配准":
    当前帧必有至少一边(稳定边)与邻帧同侧重合, 固定该边、用邻帧宽补缺侧。

    只动 joint_row 以下的动作段; 静态段交由 repair_static_segment; 末端 cap 不动。
    在 f4902 上: r132-156 半mask(w19)→全宽(w31), 正常帧几乎不变(<0.3%)。
    Returns: repaired (T,H,W) uint8 0/1。
    """
    masks = (masks > 0.5).astype(np.uint8)
    T, H, W = masks.shape
    out = masks.copy()
    cap_rows = _detect_cap_rows(masks, joint_row)
    act_widths = []
    for r in range(joint_row, H):
        if r in cap_rows:
            continue
        row = masks[:, r, :]
        h = row.any(1)
        if h.sum() < max(2, T * 0.1):
            continue
        act_widths.extend(row.sum(1)[h].tolist())
    exp_w = float(np.median(act_widths)) if act_widths else 31.0

    stats = {"frames": 0, "rows": 0, "ti": 0, "w": 0}
    for t in range(T):
        corrupt, exp = _corrupt_rows(masks[t], joint_row, exp_w, cap_rows)
        if len(corrupt) == 0:
            continue
        touched = False
        for r in corrupt:
            ri = int(r)
            w_exp = exp[ri]
            w_now = int((masks[t, ri] > 0.5).sum())
            cur_cols = np.where(masks[t, ri] > 0.5)[0]
            if len(cur_cols) == 0:
                continue
            cur_l, cur_r = int(cur_cols.min()), int(cur_cols.max())
            filled = False
            # 方法1: 时间插值(主) — 边重合配准
            nb_lefts, nb_rights = [], []
            for d in neighbors:
                tn = t + d
                if tn < 0 or tn >= T:
                    continue
                if int((masks[tn, ri] > 0.5).sum()) < 0.8 * w_exp:
                    continue
                n_cols = np.where(masks[tn, ri] > 0.5)[0]
                if len(n_cols) == 0:
                    continue
                nb_lefts.append(int(n_cols.min())); nb_rights.append(int(n_cols.max()))
            if nb_lefts:
                nb_l, nb_r = int(np.median(nb_lefts)), int(np.median(nb_rights))
                nb_w = min(nb_r - nb_l + 1, int(round(1.6 * w_exp)))
                new_row = masks[t, ri].copy()
                if abs(cur_r - nb_r) <= abs(cur_l - nb_l):   # 右边稳→向左补
                    new_l = max(0, cur_r - nb_w + 1)
                    new_row[new_l:cur_r + 1] = 1
                else:                                         # 左边稳→向右补
                    new_r = min(W - 1, cur_l + nb_w - 1)
                    new_row[cur_l:new_r + 1] = 1
                if int(new_row.sum()) > w_now:
                    out[t, ri] = new_row; filled = True; touched = True
                    stats["ti"] += 1; stats["rows"] += 1
            if filled:
                continue
            # 方法2: 宽度补全(辅, 无健康邻帧) — 单边稳定扩展
            cur_cent = float(cur_cols.mean()); half = w_exp / 2.0
            new_row = masks[t, ri].copy()
            if abs((cur_r - cur_cent) - half) <= abs((cur_cent - cur_l) - half):
                new_l = max(0, int(round(cur_r - w_exp + 1))); new_row[new_l:cur_r + 1] = 1
            else:
                new_r = min(W - 1, int(round(cur_l + w_exp - 1))); new_row[cur_l:new_r + 1] = 1
            if int(new_row.sum()) > w_now:
                out[t, ri] = new_row; touched = True; stats["w"] += 1; stats["rows"] += 1
        if touched:
            stats["frames"] += 1

    if fill_holes:
        for t in range(T):
            out[t] = binary_fill_holes(out[t]).astype(np.uint8)
    if verbose:
        print(f"  [repair_actuated] exp_w={exp_w:.1f} frames touched={stats['frames']} "
              f"rows filled={stats['rows']} (时间插值={stats['ti']} 宽度补全={stats['w']})")
    return out


def montage(raw, rep, frame_ids, out_path):
    cells = []
    for fi in frame_ids:
        if fi >= len(raw):
            continue
        pair = np.hstack([raw[fi] * 255, rep[fi] * 255]).astype(np.uint8)
        pair = np.repeat(pair[:, :, None], 3, 2)
        cv2.putText(pair, f"f{fi} L=raw R=repaired", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(pair)
    if not cells:
        return
    h, w = cells[0].shape[:2]
    cols = 3
    rows = int(np.ceil(len(cells) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(cells):
        r, c = divmod(k, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    cv2.imwrite(out_path, canvas)


def main(argv=None):
    pa = argparse.ArgumentParser(description="mask 级修复(静态段宽共识+填洞, 独立于 node)")
    pa.add_argument("--seq", required=True, help="序列名(derived/<seq>/masks)")
    pa.add_argument("--masks-dir", default=None, help="mask 目录(默认 derived/<seq>/masks)")
    pa.add_argument("--out-dir", default=None, help="输出(默认 derived/<seq>/masks_repaired)")
    pa.add_argument("--joint-row", type=int, default=None, help="手动指定关节行(默认自动检测)")
    pa.add_argument("--actuated", action=argparse.BooleanOptionalAction, default=True,
                    help="动作段时间插值修复(半mask/缺块, 默认开; --no-actuated 关闭)")
    pa.add_argument("--limit", type=int, default=None, help="只处理前 N 帧(预览)")
    args = pa.parse_args(argv)

    masks_dir = args.masks_dir or os.path.join(
        PROJECT_ROOT, "real_capture", "data", "derived", args.seq, "masks")
    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "real_capture", "data", "derived", args.seq, "masks_repaired")
    masks, fs = load_masks(masks_dir)
    if args.limit:
        masks = masks[:args.limit]
        fs = fs[:args.limit]
    T, H, W = masks.shape
    print(f">>> {T} 帧 mask ({masks_dir})")

    joint_row, std = detect_joint_row(masks) if args.joint_row is None else (args.joint_row, None)
    print(f"  关节行(静态段以上) = row {joint_row} (静态段行 0..{joint_row - 1})")

    rep = repair_static_segment(masks, joint_row)
    if args.actuated:
        rep = repair_actuated(rep, joint_row, verbose=True)

    def width_std(ms):
        ws = []
        for r in range(min(joint_row, H)):
            row = ms[:, r, :]
            has = row.any(1)
            if has.sum() < max(3, T * 0.3):
                continue
            w = row.sum(1)
            ws.extend([int(x) for x in w[has]])
        return float(np.std(ws)) if ws else 0.0
    print(f"  静态段像素宽 std: 修前 {width_std(masks):.2f} → 修后 {width_std(rep):.2f}px (应↓)")
    diff = int((rep != masks).sum())
    print(f"  改动像素: {diff} ({diff / (T * H * W) * 100:.3f}%)")

    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(fs):
        cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), (rep[i] * 255).astype(np.uint8))
    print(f"  → {out_dir}")

    ids = [f for f in [100, 4079, 4080, 4085, 4902, 2330, T // 2, T - 1] if f < T]
    montage(masks, rep, ids, os.path.join(out_dir, "repair_qc_montage.png"))
    print(f"  QC: {os.path.join(out_dir, 'repair_qc_montage.png')} (左=raw 右=repaired)")


if __name__ == "__main__":
    main()
