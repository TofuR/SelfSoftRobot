"""compare_skeleton_methods.py — 比较 2D 骨架提取方法，聚焦"末端 node0 落到 mask 尖角"问题。

背景（实测确认, 见 2026-07-09 诊断）:
  当前 extract_skeleton_2d 是**逐行质心**: 每行白色像素列均值, 从底到顶, 弧长重采样。
  - 直管(03959/04079/04080): cap 在每行对称 → 最底行质心 = 中点 → node0 正确(误差 0-1px)。
  - 弯管(04085): 管体宽 [310,341](中点 325.5), 但最底几行因管倾斜而**变窄且偏移**
    (row282=[314,323], row283=[314,321], 质心 317.5) → 最底行抓到的是倾斜 cap 的**角落**,
    不是中点 → node0 落角落(317.5 vs 真 325.5, 偏 6px), 且 node0-1-2 都在非对称区 → 尖折角。
  根因 = 对倾斜形状做水平切片, 不是细化算法伪影。故形态学/细化类"更好骨架化"治不了本。

本脚本实现多种方法, 在帧样本上比较:
  指标:
    - tipColErr_pk: |node0.col - 真值tip.col| (主; 真值=尖端区 distance-transform 峰值点=
      离所有边界最远的 cap 中心=真中心线端点; 次 tipColErr_cap=cap 质心)
    - kink_deg: node0-1-2 处方向突变(°), 越小越平滑(软体臂应平滑曲线)
    - body_dev: 与当前法在中段(node5-25)的平均偏差(px), 防止方法把直管段搞坏(回归检查)
  合成: 选定帧上把各方法骨架叠在原图+mask 上(各色), 存 montage 直观对比。

方法:
  M0_cur     当前逐行质心(基线)
  M1_dwrow   逐行 distance-transform 加权质心(向 ridge 拉, 廉价去偏)
  M2_pca     PCA 主轴 + 垂直切片质心(对倾斜管切片对称 → 修 node0 + 折角)
  M3_medial  skimage medial_axis(真中轴) + 最长路径 + 弧长重采样
  M4_snap    M0 + cap-aware 末端修正(node0 重算为 cap 中心, 末端几点沿中心线方向重平滑)
  M5_morph   形态学闭运算圆角 + M0(对照: 治标)

用法:
  python scripts/real/compare_skeleton_methods.py
  python scripts/real/compare_skeleton_methods.py --n-sample 40 --frames 04085,03959
"""
import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.skeleton_2d import extract_skeleton_2d as _m0_current  # noqa: E402

SEQ = "seq_20260627_163921"
MASKS = os.path.join(PROJECT_ROOT, "real_capture", "data", "derived", SEQ, "masks")
CAM0 = os.path.join(PROJECT_ROOT, "real_capture", "data", "raw", SEQ, "cam0")
OUT = os.path.join(PROJECT_ROOT, "output", "skeleton_method_cmp")


# ----------------------------- 共用工具 -----------------------------
def resample_arc(pts, n_points):
    if len(pts) < 2:
        return np.zeros((n_points, 2), np.float32)
    pts = np.asarray(pts, np.float64)
    d = np.diff(pts, axis=0)
    seg = np.sqrt((d ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-6:
        return np.zeros((n_points, 2), np.float32)
    t = np.linspace(0, cum[-1], n_points)
    out = np.zeros((n_points, 2), np.float32)
    out[:, 0] = np.interp(t, cum, pts[:, 0])
    out[:, 1] = np.interp(t, cum, pts[:, 1])
    return out


def order_tip_first(pts):
    """重排使 tip(最大 row=图底) 在前, 与 extract_skeleton_2d 约定一致。"""
    pts = np.asarray(pts, np.float64)
    if len(pts) < 2:
        return pts
    tip = int(np.argmax(pts[:, 1]))
    if tip == 0:
        return pts
    if tip == len(pts) - 1:
        return pts[::-1]
    return pts[np.argsort(-pts[:, 1])]


# ----------------------------- 各方法 -----------------------------
def m0_current(mask, n):
    return _m0_current(mask, n)


def m1_dwrow(mask, n):
    """逐行 distance-transform 加权质心。向 ridge 拉, 缓解角落但仍是水平切片。"""
    from scipy.ndimage import distance_transform_edt
    H, W = mask.shape
    edt = distance_transform_edt(mask)
    coords = []
    for row in range(H - 1, -1, -1):
        cs = np.where(mask[row] > 0.5)[0]
        if len(cs):
            w = edt[row, cs] + 1e-6
            coords.append([(cs.astype(float) * w).sum() / w.sum(), float(row)])
    if len(coords) < 2:
        return np.zeros((n, 2), np.float32)
    return resample_arc(np.array(coords, np.float64), n)


def m2_pca(mask, n):
    """PCA 主轴 + 垂直切片质心。对倾斜管, 垂直切片对称 → cap 中点, 修 node0+折角。"""
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 5:
        return np.zeros((n, 2), np.float32)
    pts = np.column_stack([xs.astype(float), ys.astype(float)])  # (col,row)
    center = pts.mean(0)
    cov = np.cov(pts.T)
    _, vecs = np.linalg.eigh(cov)
    axis = vecs[:, -1]  # 最大特征值方向
    t = (pts - center) @ axis
    tmin, tmax = t.min(), t.max()
    span = tmax - tmin
    if span < 1e-6:
        return np.zeros((n, 2), np.float32)
    n_slab = 200
    half = span / n_slab * 1.6
    ts = np.linspace(tmin, tmax, n_slab)
    centerline = []
    for ti in ts:
        sel = np.abs(t - ti) < half
        if sel.sum() > 0:
            centerline.append(pts[sel].mean(0))
    centerline = np.array(centerline, np.float64)
    centerline = order_tip_first(centerline)
    return resample_arc(centerline, n)


def _longest_path(skel):
    """细骨架图(bool,1px)上的最长路径(直径), 返回有序点列表(col,row) 或 None。"""
    ys, xs = np.where(skel)
    if len(xs) < 2:
        return None
    idx = {p: i for i, p in enumerate(zip(xs.tolist(), ys.tolist()))}
    adj = {}
    for (x, y) in idx:
        nb = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in idx:
                    nb.append(q)
        adj[(x, y)] = nb
    ends = [p for p in adj if len(adj[p]) == 1]
    if not ends:
        ends = list(adj)[:1]
    from collections import deque

    def bfs(src):
        prev = {src: None}
        dq = deque([src])
        far = src
        while dq:
            cur = dq.popleft()
            far = cur
            for q in adj[cur]:
                if q not in prev:
                    prev[q] = cur
                    dq.append(q)
        path = []
        c = far
        while c is not None:
            path.append(c)
            c = prev[c]
        return path[::-1], far

    _, a = bfs(ends[0])
    path, _ = bfs(a)
    return np.array(path, np.float64) if len(path) > 1 else None


def m3_medial(mask, n):
    """skimage medial_axis(真中轴) + 最长路径 + 弧长重采样。中轴对圆/平 cap 端于中心。"""
    from skimage.morphology import medial_axis
    skel = medial_axis(mask > 0.5)
    path = _longest_path(skel)
    if path is None:
        return np.zeros((n, 2), np.float32)
    path = order_tip_first(path)
    return resample_arc(path, n)


def m4_snap(mask, n):
    """M0 + cap-aware 末端修正。node0 重算为 cap 中心(dist 峰值), 末端3点沿中心线方向重平滑。"""
    from scipy.ndimage import distance_transform_edt
    sk = _m0_current(mask, n).astype(np.float64)
    if np.abs(sk).max() == 0:
        return sk.astype(np.float32)
    ys, xs = np.where(mask > 0.5)
    r1 = ys.max()
    edt = distance_transform_edt(mask)
    lo = max(0, r1 - int(0.12 * r1))
    sub = edt[lo:r1 + 1]
    if sub.any():
        ly, lx = np.unravel_index(sub.argmax(), sub.shape)
        tip_col, tip_row = float(lx), float(ly + lo)
    else:
        tip_col, tip_row = sk[0, 0], sk[0, 1]
    sk[0] = [tip_col, tip_row]
    if n >= 5:
        anchor = sk[3]
        d = sk[0] - anchor
        L = np.hypot(*d)
        if L > 1e-6:
            sk[1] = sk[0] - d * (1 / 3.0)
            sk[2] = sk[0] - d * (2 / 3.0)
    return sk.astype(np.float32)


def m5_morph(mask, n):
    """形态学闭运算(圆角 cap 角落) + M0。对照: 只治标。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    return _m0_current(m, n)


def m6_perptip(mask, n):
    """M0 body + 垂直于局部轴的尖端切片质心(原理性修法)。

    根因是水平切片倾斜管; body 段水平切片正确(直管每行质心=中心线), 只有倾斜 tip cap 错。
    故: 保留 M0 的 body, 仅重算 tip —— 从 M0 body 节点估**局部轴方向**, 在尖端做**垂直于轴**
    的切片; 垂直切片对管是左右对称的 → 质心 = 局部中心线点 = cap 中点(修 corner + 折角)。
    比 M2(全局 PCA)更准: 用局部方向不偏, 且不动 body(body_dev≈0)。
    """
    sk = _m0_current(mask, n).astype(np.float64)
    if np.abs(sk).max() == 0:
        return sk.astype(np.float32)
    ys, xs = np.where(mask > 0.5)
    if len(xs) < 10:
        return sk.astype(np.float32)
    pts = np.column_stack([xs.astype(float), ys.astype(float)])  # (col,row)
    far = sk[min(7, n - 1)]    # body 节点(偏 base)
    near = sk[min(3, n - 1)]   # body 节点(偏 tip)
    seg = near - far           # 指向 tip 的局部轴方向
    L = float(np.hypot(*seg))
    if L < 1e-6:
        return sk.astype(np.float32)
    d = seg / L
    proj = (pts - far) @ d
    w = float(mask.sum(1).max())            # 管径估计(最大行宽)
    slab = proj >= proj.max() - 0.4 * w     # 尖端垂直切片
    if int(slab.sum()) < 3:
        return sk.astype(np.float32)
    node0 = pts[slab].mean(0)               # 垂直切片质心 = 中心线中点
    sk[0] = node0
    a = sk[min(3, n - 1)]                    # 沿 body→node0 重布 node1,2 消折角
    sk[1] = node0 + (a - node0) / 3.0
    sk[2] = node0 + (a - node0) * 2.0 / 3.0
    return sk.astype(np.float32)


METHODS = [
    ("M0_cur", m0_current, (255, 255, 0)),      # 青
    ("M1_dwrow", m1_dwrow, (0, 165, 255)),      # 橙
    ("M2_pca", m2_pca, (0, 0, 255)),            # 红
    ("M3_medial", m3_medial, (255, 0, 255)),    # 紫
    ("M4_snap", m4_snap, (0, 255, 255)),        # 黄
    ("M5_morph", m5_morph, (255, 255, 255)),    # 白
    ("M6_perptip", m6_perptip, (255, 0, 0)),    # 蓝(原理性修法)
]


# ----------------------------- 真值 + 指标 -----------------------------
def tip_truth(mask):
    """真值 tip = mask 尖端直边的中点(用户定义:"骨架最后应在 mask 尖端直边的中间")。
    truth_col = 尖端区(底部35%)**最靠近 tip 的满宽行**水平中点 = cap 处中心线 col。
    关键鲁棒点: 只在**尖端区**取宽度, 用**中位宽**(非 max)——否则关节 bulge(管-臂合并处
    比管体宽)会被当 max 宽度, 把"满宽行"误判为关节行(如 f4079 关节宽39>管体32→truth 跑到
    关节 col 316 而非 tip col 306)。仅由 mask 计算, 独立于所有候选方法/EDT。返回 (truth_col, tip_row, bend)。"""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0:
        return None
    rows = np.where(mask.any(1))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    rng = r1 - r0
    tip_lo = max(r0, r1 - int(0.35 * rng))           # 尖端区 = 底部 35%
    tip_rows = [r for r in range(tip_lo, r1 + 1) if mask[r].any()]
    widths = {r: int(mask[r].sum()) for r in tip_rows}
    if not widths:
        return None
    typ_w = float(np.median(list(widths.values())))   # 中位宽(robust to bulge)
    thr = 0.85 * typ_w
    full = [r for r in tip_rows if widths[r] >= thr]
    if not full:
        truth_col = float((xs.min() + xs.max()) / 2)
    else:
        r_tb = max(full)                              # 尖端区最靠近 tip 的满宽行
        lo, hi = max(r0, r_tb - 1), min(r1, r_tb + 1)
        seg = [np.where(mask[r] > 0.5)[0] for r in range(lo, hi + 1)]
        seg = [s for s in seg if len(s)]
        truth_col = float(np.mean([(s.min() + s.max()) / 2.0 for s in seg]))
    mids = [(r, (lambda c: (c.min() + c.max()) / 2.0)(np.where(mask[r] > 0.5)[0]))
            for r in tip_rows if widths[r] >= 0.6 * typ_w]
    if len(mids) >= 3:
        rr = np.array([a[0] for a in mids]); cc = np.array([a[1] for a in mids])
        line = np.interp(rr, [rr.min(), rr.max()], [cc[0], cc[-1]])
        bend = float(np.abs(cc - line).max())
    else:
        bend = 0.0
    return truth_col, float(r1), bend


def tip_kink_deg(sk):
    """node0-1-2 处方向突变(°): node0→node1 与 node1→node2 的转角。平滑≈0。"""
    if len(sk) < 3 or np.abs(sk).max() == 0:
        return np.nan
    a = sk[0] - sk[1]
    b = sk[1] - sk[2]
    la, lb = np.hypot(*a), np.hypot(*b)
    if la < 1e-6 or lb < 1e-6:
        return np.nan
    return float(np.degrees(np.arccos(np.clip((a @ b) / (la * lb), -1, 1))))


def body_dev(sk0, sk):
    """与 M0 在中段 node5-25 的平均偏差(px) — 回归检查。"""
    if np.abs(sk).max() == 0 or np.abs(sk0).max() == 0:
        return np.nan
    lo, hi = 5, min(25, len(sk) - 1)
    return float(np.hypot(*(sk[lo:hi] - sk0[lo:hi]).T).mean())


# ----------------------------- 合成 viz -----------------------------
def draw_skel(img, sk, color, r=3, lw=2):
    if sk is None or np.abs(sk).max() == 0:
        return
    pts = sk.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, color, lw, cv2.LINE_AA)
    for p in pts.reshape(-1, 2):
        cv2.circle(img, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)


def composite(mask, skels, cam0_path):
    img = cv2.imread(cam0_path) if os.path.isfile(cam0_path) else None
    if img is None:
        img = np.full((mask.shape[0], mask.shape[1], 3), 40, np.uint8)
    ov = img.copy()
    ov[mask > 0] = (0, 0, 255)
    cv2.addWeighted(ov, 0.25, img, 0.75, 0, dst=img)
    for name, sk, color in skels:
        draw_skel(img, sk, color)
    return img


# ----------------------------- 主流程 -----------------------------
def main(argv=None):
    pa = argparse.ArgumentParser()
    pa.add_argument("--n-sample", type=int, default=24)
    pa.add_argument("--frames", default=None, help="指定帧(逗号分隔), 覆盖采样")
    pa.add_argument("--n-points", type=int, default=31)
    args = pa.parse_args(argv)

    allf = sorted(int(os.path.splitext(f)[0]) for f in os.listdir(MASKS) if f.endswith(".png"))
    if args.frames:
        fs = [int(x) for x in args.frames.split(",")]
    else:
        idx = np.linspace(0, len(allf) - 1, args.n_sample).astype(int)
        fs = sorted(set(allf[i] for i in idx))
        for must in (3959, 4085, 4079, 4080):
            if must in allf and must not in fs:
                fs.append(must)
        fs = sorted(set(fs))
    # 排除已标记的离群/腐败帧(手干扰/管茬, 由 clean_outlier_skeletons 另行时间插值修复)——
    # 它们是数据腐败, 不是骨架化方法问题, 留在样本里会让所有方法都"失败"(err 50-100px), 污染对比。
    out_path = os.path.join(PROJECT_ROOT, "data", "real_seq", SEQ, "skeleton_outlier_frames.txt")
    outliers = set()
    if os.path.isfile(out_path):
        for ln in open(out_path):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                outliers.update(int(x) for x in ln.split())
    n_before = len(fs)
    fs = [f for f in fs if f not in outliers]
    print(f"评估 {len(fs)} 帧(排除 {n_before - len(fs)} 离群/腐败帧), 方法 {[m[0] for m in METHODS]}")

    os.makedirs(OUT, exist_ok=True)
    # 面积腐败预过滤: 手/管茬合并等使 mask 面积远超正常(~8500) → 真值与方法都失效, 非骨架化问题。
    # 排除面积 > 1.6×中位 的 blob(如 f1211 area=21091), 它们不在已标记离群列表里。
    areas = {}
    for f in fs:
        mp = os.path.join(MASKS, f"{f:05d}.png")
        m = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if m.any():
            areas[f] = int(m.sum())
    med_area = float(np.median(list(areas.values()))) if areas else 0.0
    n_area = sum(1 for a in areas.values() if a > 1.6 * med_area)
    print(f"  面积中位={med_area:.0f}, 额外排除面积腐败(>1.6×) {n_area} 帧")
    agg = {name: {"err": [], "kink": [], "body_dev": [], "fail": 0} for name, _, _ in METHODS}
    bends = []
    composite_cells = []
    worst_per_method = {name: [] for name, _, _ in METHODS}

    for f in fs:
        if areas.get(f, 0) > 1.6 * med_area:           # 面积腐败 blob 跳过
            continue
        mp = os.path.join(MASKS, f"{f:05d}.png")
        m = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if not m.any():
            continue
        truth = tip_truth(m)
        if truth is None:
            continue
        truth_col, tip_row, bend = truth
        bends.append(bend)
        sk0 = _m0_current(m, args.n_points)
        skels = []
        for name, fn, color in METHODS:
            try:
                sk = fn(m, args.n_points)
            except Exception:
                agg[name]["fail"] += 1
                worst_per_method[name].append((999.0, f, bend))
                skels.append((name, np.zeros((args.n_points, 2)), color))
                continue
            if np.abs(sk).max() == 0:
                agg[name]["fail"] += 1
                skels.append((name, sk, color))
                continue
            e = abs(sk[0, 0] - truth_col)
            agg[name]["err"].append((e, bend))
            agg[name]["kink"].append(tip_kink_deg(sk))
            agg[name]["body_dev"].append(body_dev(sk0, sk))
            worst_per_method[name].append((e, f, bend))
            skels.append((name, sk, color))
        if f in (3959, 4085, 4079, 4080) or len(composite_cells) < 8:
            img = composite(m, skels, os.path.join(CAM0, f"{f:05d}.png"))
            cv2.drawMarker(img, (int(truth_col), int(tip_row)), (0, 200, 0), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(img, f"f{f} bend{bend:.0f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            composite_cells.append((f, img))

    allb = sorted(bends)
    hi_cut = allb[int(0.6 * len(allb))] if allb else 0.0
    n_hi = sum(1 for b in bends if b >= hi_cut)
    print(f"\n=== 汇总 (高弯 = top40%, bend≥{hi_cut:.1f}px, n={n_hi}/{len(bends)}) ===")
    hdr = f"{'method':<10} {'err_all':>8} {'err_hiBend':>10} {'err_max':>8} {'kink_deg':>9} {'body_dev':>9} {'fail':>5}"
    print(hdr)
    print("-" * len(hdr))
    for name, _, _ in METHODS:
        a = agg[name]
        errs = a["err"]
        e_all = float(np.mean([e for e, _ in errs])) if errs else float("nan")
        e_hi = float(np.mean([e for e, b in errs if b >= hi_cut])) if errs else float("nan")
        e_max = max((e for e, _ in errs), default=float("nan"))
        kk = float(np.mean(a["kink"])) if a["kink"] else float("nan")
        bd = float(np.mean(a["body_dev"])) if a["body_dev"] else float("nan")
        print(f"{name:<10} {e_all:>8.2f} {e_hi:>10.2f} {e_max:>8.2f} {kk:>9.2f} {bd:>9.2f} {a['fail']:>5}")
    print("\n  绿× = tip 真值(尖端满宽行中点)。err_hiBend = 高弯帧末端误差(关键, corner 问题在此);"
          " kink_deg 越小越平滑; body_dev 大=方法把直管段改坏(回归)。")

    if composite_cells:
        h, w = composite_cells[0][1].shape[:2]
        cols = 4
        rows = int(np.ceil(len(composite_cells) / cols))
        canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
        for k, (f, im) in enumerate(composite_cells):
            r, c = divmod(k, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
        p = os.path.join(OUT, "methods_montage.png")
        cv2.imwrite(p, canvas)
        for f, im in composite_cells:
            if f in (4085, 3959):
                cv2.imwrite(os.path.join(OUT, f"frame_{f:05d}.png"), im)
        print(f"\n合成 montage: {p}")
        print("  (大图 frame_04085.png / frame_03959.png)")
        print("  M0_cur=青 M1_dwrow=橙 M2_pca=红 M3_medial=紫 M4_snap=黄 M5_morph=白 | 绿×=tip真值")


if __name__ == "__main__":
    main()
