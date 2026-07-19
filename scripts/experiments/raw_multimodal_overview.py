#!/usr/bin/env python3
"""raw_multimodal_overview.py — 实物同步多模态原始数据展示图(汇报用).

一个序列里 4 种模态共用 t_sec 时钟, 完美同步:
  cam0/NNNNN.png   RealSense RGB 帧   (frame_times.txt[i] = cam0/{i:05d}.png 的 t_sec)
  actions6.csv     c0..c5 气压指令 kPa(c0 = 激活通道; 其余 0)
  pressure.csv     p_active 实测气压 kPa(闭环跟踪验证)
  ndi.csv          末端 6DOF  x,y,z(mm) + 四元数(真值)

画法(一张汇报图):
  第0行: 时间上均匀采样的 N 帧 相机缩略图, 每帧标 t / p_active, 带序号 ①②③…
  第1行: 气压 指令 vs 实测 vs t,  在采样帧时刻画 竖虚线 + 序号
  第2行: NDI 末端 x / y / z(mm) vs t, 同样竖虚线(x 为主弯曲轴)
  第3行: 末端 2D 轨迹(x–y 平面 mm), 按气压着色, 采样帧末端标星 + 序号
序号把缩略图 ↔ 曲线时刻一一对应, 让"同步"一眼可见。

输出(output/raw_data_overview/):
  multimodal_overview_<seq>.png   主图
  multimodal_overview_<seq>.json  采样帧 / 概况

Usage:
  python scripts/experiments/raw_multimodal_overview.py                       # 默认主训练序列 163921
  python scripts/experiments/raw_multimodal_overview.py --seq seq_20260627_173114 --n_frames 6
  python scripts/experiments/raw_multimodal_overview.py --sample pressure     # 按气压分层取帧(看准静态形变)
"""
import os, sys, csv, json, argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW = os.path.join(ROOT, "real_capture", "data", "raw")
OUT = os.path.join(ROOT, "output", "raw_data_overview")
os.makedirs(OUT, exist_ok=True)

_CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
COL = {"cmd": "#8e44ad", "meas": "#e67e22", "x": "#3498db", "y": "#27ae60", "z": "#95a5a6"}


def load_csv(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[0]
    arr = []
    for r in rows[1:]:
        if len(r) != len(hdr):
            continue
        try:
            arr.append([float(x) for x in r])
        except ValueError:
            pass
    return hdr, np.array(arr)


def nearest_idx(arr, v):
    return int(np.argmin(np.abs(arr - v)))


def pick_frame_times(t_all, p_all, n, mode):
    """返回 n 个采样帧的索引(避开首尾 3%)。"""
    lo, hi = int(0.03 * len(t_all)), int(0.97 * len(t_all))
    pool = np.arange(lo, hi)
    if mode == "pressure":
        qs = np.linspace(0, 1, n)
        picked = [pool[nearest_idx(p_all[pool], np.quantile(p_all[pool], q))] for q in qs]
    else:  # time: 时间均匀
        picked = list(np.linspace(pool[0], pool[-1], n).astype(int))
    seen, out = set(), []
    for i in picked:
        if i not in seen:
            seen.add(i); out.append(int(i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="seq_20260627_163921")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--sample", default="time", choices=["time", "pressure"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    seq = args.seq
    d = os.path.join(RAW, seq)

    ft = np.array([float(x) for x in open(os.path.join(d, "frame_times.txt")).read().split()])
    _, act = load_csv(os.path.join(d, "actions6.csv"))      # t,c0..c5
    _, pre = load_csv(os.path.join(d, "pressure.csv"))      # t,p_active,reserved
    _, ndi = load_csv(os.path.join(d, "ndi.csv"))           # t,x,y,z,...
    meta = json.load(open(os.path.join(d, "meta.json")))

    t = ft
    c0 = act[:, 1]
    p = pre[:, 1]
    ndi_x, ndi_y, ndi_z = ndi[:, 1], ndi[:, 2], ndi[:, 3]
    if len(ndi) != len(t):
        ix = np.clip(np.searchsorted(ndi[:, 0], t), 0, len(ndi) - 1)
        ndi_x, ndi_y, ndi_z = ndi_x[ix], ndi_y[ix], ndi_z[ix]

    fidx = pick_frame_times(t, p, args.n_frames, args.sample)
    n = len(fidx)
    dur = t[-1] - t[0]
    fps = len(t) / dur if dur > 0 else float("nan")

    print(f"{seq}: {len(t)} 帧, dur={dur:.1f}s, ~{fps:.2f}fps, p=[{p.min():.0f},{p.max():.0f}]kPa, "
          f"tip x=[{ndi_x.min():.1f},{ndi_x.max():.1f}]mm, 采样 {n} 帧 ({args.sample})")

    # ── 画图 ──
    fig = plt.figure(figsize=(16, 10.5))
    gs = GridSpec(4, n, figure=fig, height_ratios=[3.0, 1.5, 1.5, 1.8],
                  hspace=0.34, wspace=0.06)

    # 第0行: 相机缩略图
    for j, fi in enumerate(fidx):
        axi = fig.add_subplot(gs[0, j])
        img = np.array(Image.open(os.path.join(d, "cam0", f"{fi:05d}.png")))
        axi.imshow(img)
        axi.set_xticks([]); axi.set_yticks([])
        for s in axi.spines.values():
            s.set_edgecolor("#bbbbbb")
        axi.set_title(f"t={t[fi]:.1f}s\np={p[fi]:.0f}kPa", fontsize=8.5)
        axi.text(0.02, 0.96, _CIRC[j], transform=axi.transAxes, fontsize=13,
                 va="top", ha="left", color="white",
                 bbox=dict(boxstyle="circle,pad=0.25", fc="#2c3e50", ec="none"))

    # 第1行: 气压 指令 vs 实测
    axp = fig.add_subplot(gs[1, :])
    axp.plot(t, c0, "-", color=COL["cmd"], lw=1.2, alpha=0.9, label="指令 c0 (kPa)")
    axp.plot(t, p, ":", color=COL["meas"], lw=1.0, alpha=0.8, label="实测 p_active (kPa)")
    for j, fi in enumerate(fidx):
        axp.axvline(t[fi], color="#34495e", ls="--", lw=0.8, alpha=0.6)
    ymax = axp.get_ylim()[1]
    for j, fi in enumerate(fidx):
        axp.text(t[fi], ymax, _CIRC[j], fontsize=10, ha="center", va="top", color="#2c3e50")
    axp.set_ylabel("气压 (kPa)", fontsize=11)
    axp.legend(fontsize=8.5, loc="upper right", ncol=2)
    axp.grid(True, alpha=0.25)
    axp.set_title(f"{seq}  ·  {meta.get('mode','?')} 模式  ·  {len(t)} 帧 @ ~{fps:.1f}fps  ·  "
                  f"时长 {dur:.0f}s  ·  激活通道 ch{meta.get('active_channel',0)}  ·  "
                  f"气压 0–150 kPa 激励",
                  fontsize=12, loc="left")

    # 第2行: NDI 末端 x/y/z
    axn = fig.add_subplot(gs[2, :], sharex=axp)
    axn.plot(t, ndi_x, "-", color=COL["x"], lw=1.3, label="NDI x")
    axn.plot(t, ndi_y, "-", color=COL["y"], lw=1.3, label="NDI y")
    axn.plot(t, ndi_z, "-", color=COL["z"], lw=1.1, alpha=0.7, label="NDI z")
    ymax = axn.get_ylim()[1]
    for j, fi in enumerate(fidx):
        axn.axvline(t[fi], color="#34495e", ls="--", lw=0.8, alpha=0.6)
        axn.text(t[fi], ymax, _CIRC[j], fontsize=10, ha="center", va="top", color="#2c3e50")
    axn.set_ylabel("末端位置 (mm)", fontsize=11)
    axn.legend(fontsize=8.5, loc="upper right", ncol=3)
    axn.grid(True, alpha=0.25)
    axn.set_xlabel("t (s)", fontsize=11)

    # 第3行: 末端 2D 轨迹(x–y), 按气压着色
    axt = fig.add_subplot(gs[3, :])
    sc = axt.scatter(ndi_x, ndi_y, c=p, cmap="plasma", s=4, alpha=0.5)
    cb = fig.colorbar(sc, ax=axt, pad=0.01, fraction=0.04)
    cb.set_label("气压 (kPa)", fontsize=10)
    for j, fi in enumerate(fidx):
        axt.scatter(ndi_x[fi], ndi_y[fi], marker="*", s=220, color="white",
                    edgecolors="black", linewidths=1.2, zorder=5)
        axt.text(ndi_x[fi], ndi_y[fi], _CIRC[j], fontsize=8, ha="center", va="center",
                 color="black", fontweight="bold", zorder=6)
    axt.set_xlabel("NDI x (mm)", fontsize=11)
    axt.set_ylabel("NDI y (mm)", fontsize=11)
    axt.set_title("末端 2D 轨迹(按气压着色, ★=采样帧末端)", fontsize=11, loc="left")
    axt.grid(True, alpha=0.25)
    axt.set_aspect("equal", adjustable="datalim")

    fig.suptitle("同步多模态原始数据采集  —  RealSense 相机 + 气压(指令/实测) + NDI 末端真值(共用 t_sec 时钟)",
                 fontsize=14, y=0.997)

    fpng = os.path.join(args.out, f"multimodal_overview_{seq}_{args.sample}.png")
    plt.savefig(fpng, dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "seq": seq, "mode": meta.get("mode"), "n_frames": len(t),
        "duration_s": float(dur), "fps_est": float(fps),
        "pressure_kPa": {"min": float(p.min()), "max": float(p.max())},
        "tip_mm": {"x": [float(ndi_x.min()), float(ndi_x.max())],
                   "y": [float(ndi_y.min()), float(ndi_y.max())],
                   "z": [float(ndi_z.min()), float(ndi_z.max())]},
        "sampled_frames": [{"badge": _CIRC[j], "frame_idx": int(fi),
                            "t_s": float(t[fi]), "p_kPa": float(p[fi]),
                            "tip_mm": [float(ndi_x[fi]), float(ndi_y[fi]), float(ndi_z[fi])]}
                           for j, fi in enumerate(fidx)],
    }
    fjson = os.path.join(args.out, f"multimodal_overview_{seq}_{args.sample}.json")
    json.dump(summary, open(fjson, "w"), indent=2, ensure_ascii=False)
    print(f"保存 → {os.path.relpath(fpng, ROOT)}")
    print(f"保存 → {os.path.relpath(fjson, ROOT)}")


if __name__ == "__main__":
    main()
