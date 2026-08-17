"""segment_rd.py — 实物 RGB → 二值前景 分割方法扫描 + QC 可视化 (Phase 1a)。

数据：real_capture/data/raw/seq_20260627_163921/cam0
  - 白硅胶软臂（半透明、主体亮）+ 蓝色静态墙背景 + 白色气管（与臂同色）
  - 单通道 ch0 0–150 kPa 随机驱动 → 1-DOF 平面弯曲；底端=运动尖端，顶端=固定基座

目的：在采样帧上对比几条经典 CV 路线，输出 QC 对比大图 + 阈值扫描，便于
选定方法/阈值后再批量（Phase 1b）。SAM2 仅在此全部失败时作兜底。

核心观察（已实测）：
  - 跨 34 min 序列 74.5% 像素 std<5 → 背景(蓝墙+座)非常静态，
    per-pixel median 直接得到干净背景（不必单独拍 bg）。
  - 白臂在蓝底上对比极强（gray≈200 vs bg≈70），但白色气管与臂同色 →
    必须靠"动过"(bg-subtract)剔除静态管/座，再用形态学 OPEN 按宽度剔除细管。

路线（A→F 逐步精化，对应你"颜色提主体 + 骨架剪枝去管"的思路）:
  A  HSV 白      S<sat & V>val                 全部白色(臂+管+座+眩光)
  B  背景差      |gray - median_bg| > diff      "动过"区域(臂主体 + 摆动管)
  C  A ∩ dil(B)  白 且 动过                      去静态座/墙白斑/静态主气管
  D  C 形态学    OPEN(按宽度去细管) + CLOSE(填体)
  E  D 连通区    面积/高度过滤 → 留粗体机器人
  F  E 骨架      skeletonize 叠加(看是否还有管茬需剪枝)

用法:
  python scripts/real/segment_rd.py \
      --seq real_capture/data/raw/seq_20260627_163921 \
      --n-samples 8 \
      --out  real_capture/data/derived/seq_20260627_163921/qc
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from skimage.morphology import skeletonize  # noqa: E402
    HAVE_SK = True
except ImportError:  # pragma: no cover
    HAVE_SK = False


# -------------------------------------------------------------------- 背景
def build_median_bg(cam_dir: str, n_bg: int = 500) -> tuple[np.ndarray, list[str]]:
    """从 cam0 均匀采样 n_bg 帧灰度 → per-pixel median = 静态背景。

    机器人移动占据每像素 <50% 时间 → 中值趋近真实静态背景（蓝墙+座+静态管）。
    """
    fs = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
    if not fs:
        sys.exit(f"无帧: {cam_dir}")
    idx = np.linspace(0, len(fs) - 1, min(n_bg, len(fs))).astype(int)
    stack = np.stack([cv2.imread(fs[i], cv2.IMREAD_GRAYSCALE) for i in idx])
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg, fs


# -------------------------------------------------------------------- 基元
def mask_white(bgr: np.ndarray, sat: int, val: int) -> np.ndarray:
    """HSV 白：S<sat（低饱和）& V>val（高亮）。蓝墙高饱和被排除。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    return ((s < sat) & (v > val)).astype(np.uint8)


def mask_moved(gray: np.ndarray, bg: np.ndarray, diff: int) -> np.ndarray:
    """背景差：|gray - bg| > diff → 与静态背景不同的像素（臂+摆动管）。"""
    return (cv2.absdiff(gray, bg) > diff).astype(np.uint8)


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return mask
    return cv2.dilate(mask, np.ones((k, k), np.uint8))


def morph_clean(mask: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """OPEN（去细管/噪）→ CLOSE（填体）。kernel 尺寸介于管宽与臂宽之间。"""
    m = mask
    if open_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return m


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """binary fill holes（scipy）。"""
    from scipy.ndimage import binary_fill_holes
    return binary_fill_holes(mask > 0).astype(np.uint8)


def keep_robot(mask: np.ndarray, min_area_frac: float, min_height_frac: float,
               H: int, W: int) -> np.ndarray:
    """连通区过滤：留面积≥min_area_frac·Frame 且 高度≥min_height_frac·H 的最大区。

    臂是竖直长条（base 顶 → tip 底），h 占帧高比例大；管茬/噪块矮小。
    多个候选时取面积最大者（保险）。
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return np.zeros_like(mask)
    min_area = min_area_frac * H * W
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area and h >= min_height_frac * H:
            cands.append((area, i))
    if not cands:
        return np.zeros_like(mask)
    cands.sort(reverse=True)
    keep_i = cands[0][1]
    return (lbl == keep_i).astype(np.uint8)


# -------------------------------------------------------------------- 全管线
def segment_pipeline(bgr: np.ndarray, bg: np.ndarray, p: dict) -> dict:
    """跑完整 A→E 管线，返回各阶段掩码（供 QC 网格逐列展示）。"""
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    A = mask_white(bgr, p["sat"], p["val"])
    B = mask_moved(gray, bg, p["diff"])
    C = (A & dilate(B, p["dil"])).astype(np.uint8)
    D = morph_clean(C, p["open_k"], p["close_k"])
    Df = fill_holes(D)
    E = keep_robot(Df, p["min_area_frac"], p["min_h_frac"], H, W)
    out = {"A": A, "B": B, "C": C, "D": Df, "E": E}
    if HAVE_SK:
        out["sk"] = skeletonize(E.astype(bool)).astype(np.uint8)
    return out


# -------------------------------------------------------------------- 指标
def metrics(mask: np.ndarray, H: int, W: int) -> dict:
    fg = float(mask.mean())
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest = 0
    bbox = (0, 0, 0, 0)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        i = 1 + int(np.argmax(areas))
        largest = int(areas.max())
        bbox = tuple(int(x) for x in stats[i, :4])
    return {"fg_frac": round(fg, 4), "n_cc": int(n) - 1,
            "largest_area": largest, "largest_bbox_xywh": bbox}


# -------------------------------------------------------------------- 采样帧
def pick_sample_frames(fps: list[str], actions_csv: str, n: int) -> list[int]:
    """按 c0 压力百分位挑帧，覆盖 rest→high 全变形范围。返回帧索引。"""
    raw = np.atleast_2d(np.genfromtxt(actions_csv, delimiter=",", dtype=float))
    while raw.shape[0] and np.isnan(raw[0]).all():       # 跳表头
        raw = raw[1:]
    c0 = raw[:, 1]                                        # t, c0..c5 → c0
    n_all = min(len(fps), len(c0))
    qs = np.linspace(0, 1, n)                             # 0,1/(n-1),...,1
    order = np.argsort(c0[:n_all])                        # 按 c0 升序的帧索引
    pick = [int(order[min(len(order) - 1, int(round(q * (len(order) - 1))))])
            for q in qs]
    return sorted(set(pick))


# -------------------------------------------------------------------- QC 网格
def qc_grid(samples, bg, p, out_dir):
    """每行=一个采样帧；列= RGB | A | B | C | D | E | overlay(E+skeleton)。"""
    cols = ["RGB", "A HSV白", "B 背景差", "C 白∩动", "D 形态学", "E 最终", "E 叠加+骨架"]
    fig, axes = plt.subplots(len(samples), len(cols),
                             figsize=(2.4 * len(cols), 2.4 * len(samples)),
                             squeeze=False)
    for r, (idx, bgr) in enumerate(samples):
        res = segment_pipeline(bgr, bg, p)
        mets = metrics(res["E"], bgr.shape[0], bgr.shape[1])
        for c, name in enumerate(cols):
            ax = axes[r, c]
            if name == "RGB":
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            elif name == "E 叠加+骨架":
                ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                overlay = np.where(res["E"][..., None] > 0, [255, 0, 0], [0, 0, 0])
                ax.imshow(overlay, alpha=0.30)
                if "sk" in res and res["sk"].any():
                    ys, xs = np.where(res["sk"] > 0)
                    ax.plot(xs, ys, "c.", ms=1)
            else:
                ax.imshow(res[name.split()[0]], cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(f"frame {idx}\nfg={mets['fg_frac']:.3f}\nA={mets['largest_area']}",
                              fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = os.path.join(out_dir, "method_grid.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# -------------------------------------------------------------------- 阈值扫描
def sweep_sat_val(bgr, bg, out_dir, base_p):
    """扫描 sat × val 网格（看白色提取敏感性，以 E 最终掩码呈现）。"""
    sats = [80, 100, 120, 140]
    vals = [100, 120, 140, 160]
    fig, axes = plt.subplots(len(sats), len(vals),
                             figsize=(2.2 * len(vals), 2.2 * len(sats)),
                             squeeze=False)
    for r, s in enumerate(sats):
        for c, v in enumerate(vals):
            p = dict(base_p); p["sat"] = s; p["val"] = v
            res = segment_pipeline(bgr, bg, p)
            axes[r, c].imshow(res["E"], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(f"S<{s},V>{v}", fontsize=8)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("sat × val 扫描（E 最终掩码）", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_sat_val.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


def sweep_open_k(bgr, bg, out_dir, base_p):
    """扫描 OPEN kernel（按宽度去管的关键参数）。"""
    ks = [3, 5, 7, 9, 11, 13]
    fig, axes = plt.subplots(1, len(ks), figsize=(2.2 * len(ks), 2.4))
    for c, k in enumerate(ks):
        p = dict(base_p); p["open_k"] = k
        res = segment_pipeline(bgr, bg, p)
        m = metrics(res["E"], bgr.shape[0], bgr.shape[1])
        axes[c].imshow(res["E"], cmap="gray", vmin=0, vmax=1)
        axes[c].set_title(f"open_k={k}\nA={m['largest_area']}", fontsize=8)
        axes[c].set_xticks([]); axes[c].set_yticks([])
    fig.suptitle("OPEN kernel 扫描（去细管；太大伤臂）", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep_open_k.png")
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    return path


# -------------------------------------------------------------------- main
def build_parser():
    pa = argparse.ArgumentParser(description="实物 RGB 分割方法扫描 + QC")
    pa.add_argument("--seq", required=True, help="序列目录(含 cam0/, actions6.csv)")
    pa.add_argument("--out", default=None, help="QC 输出目录(默认 <seq>/../derived/<seq名>/qc)")
    pa.add_argument("--n-samples", type=int, default=8)
    pa.add_argument("--n-bg", type=int, default=500, help="中值背景采样帧数")
    # 默认参数（diag 校准后：机器人是 ~20-40px 细鞭状，半透明 → 用 white∩动过 + 小 OPEN）
    pa.add_argument("--sat", type=int, default=100)
    pa.add_argument("--val", type=int, default=120)
    pa.add_argument("--diff", type=int, default=25)
    pa.add_argument("--dil", type=int, default=35)
    pa.add_argument("--open-k", type=int, default=5)
    pa.add_argument("--close-k", type=int, default=15)
    pa.add_argument("--min-area-frac", type=float, default=0.003)
    pa.add_argument("--min-h-frac", type=float, default=0.15)
    return pa


def main():
    args = build_parser().parse_args()
    seq = args.seq.rstrip("/")
    seq_name = os.path.basename(seq)
    out_dir = args.out or os.path.join(os.path.dirname(seq), "..", "derived", seq_name, "qc")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    mask_dir = os.path.join(os.path.dirname(out_dir), "sample_masks")
    os.makedirs(mask_dir, exist_ok=True)

    cam_dir = os.path.join(seq, "cam0")
    print(f">>> 构建中值背景（采样 {args.n_bg} 帧）...")
    bg, fps = build_median_bg(cam_dir, args.n_bg)
    cv2.imwrite(os.path.join(out_dir, "bg_median.png"), bg)
    print(f"    bg: mean={bg.mean():.1f} std={bg.std():.1f}  → {out_dir}/bg_median.png")

    p = dict(sat=args.sat, val=args.val, diff=args.diff, dil=args.dil,
             open_k=args.open_k, close_k=args.close_k,
             min_area_frac=args.min_area_frac, min_h_frac=args.min_h_frac)

    actions_csv = os.path.join(seq, "actions6.csv")
    print(f">>> 按 c0 压力百分位挑 {args.n_samples} 帧样本...")
    idxs = pick_sample_frames(fps, actions_csv, args.n_samples)
    samples = [(i, cv2.imread(fps[i])) for i in idxs]
    print(f"    采样帧索引: {idxs}")

    print(">>> 生成方法对比网格...")
    grid_path = qc_grid(samples, bg, p, out_dir)
    print(f"    {grid_path}")

    # 阈值扫描（用中间样本帧）
    mid_bgr = samples[len(samples) // 2][1]
    print(">>> 阈值扫描 sat×val / open_k ...")
    sv = sweep_sat_val(mid_bgr, bg, out_dir, p); print(f"    {sv}")
    ok = sweep_open_k(mid_bgr, bg, out_dir, p); print(f"    {ok}")

    # 落地：各样本最终掩码 + 全部指标
    print(">>> 写出样本掩码 + 指标...")
    lines = []
    for idx, bgr in samples:
        res = segment_pipeline(bgr, bg, p)
        m = metrics(res["E"], bgr.shape[0], bgr.shape[1])
        cv2.imwrite(os.path.join(mask_dir, f"{idx:05d}_mask.png"), res["E"] * 255)
        lines.append(f"frame {idx}: {m}")
    metrics_txt = "\n".join(lines)
    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(metrics_txt + "\n")
    print(metrics_txt)

    meta = {"seq": seq, "params": p, "n_bg": args.n_bg,
            "sample_frames": idxs, "have_skeletonize": HAVE_SK}
    with open(os.path.join(out_dir, "segment_rd_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n>>> QC 完成。查看 {out_dir}/method_grid.png 选定方法/阈值。")


if __name__ == "__main__":
    main()
