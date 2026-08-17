#!/usr/bin/env python3
"""exp5b_hysteresis_loop_real.py — 实物加载-卸载迟滞回线(仿真 exp5b 的实物对照).

仿真版(exp5b_hysteresis_loop.py)用 PyElastica 跑 0→+τ→0→-τ→0 四阶段, 画
扭矩↔末端位移的迟滞回线. 实物对应: 单通道(ch0)气压在 0↔150 kPa 做三角波加载-
卸载周期, NDI 末端 mm 作响应, 画 气压↔末端位移 的迟滞回线.

气动是单向的(只能 0→150, 不能负压), 故实物只有"正半周期"回线: 加载曲线与
卸载曲线不重合 → 迟滞(黏弹性滞后). 反复周期还能看循环演化(预条件化/Mullins).

输入(均在 real_capture/data/raw/<seq>/):
  actions6.csv : t_sec, c0..c5      (c0 = ch0 气压指令 kPa, 三角波 0↔150)
  pressure.csv : t_sec, p_active     (实测气压 kPa, 作 x 轴真实输入)
  ndi.csv      : t_sec, x,y,z,...    (末端 6DOF mm; x 为主弯曲响应)

输出(output/exp5b_hysteresis_loop/):
  hysteresis_loop_real.png   主图: 迟滞回线 + 首末周期对比
  time_series_real.png        压力 + 末端位移 时间序列

Usage:
  python scripts/experiments/exp5b_hysteresis_loop_real.py            # 默认 173114(准静态)
  python scripts/experiments/exp5b_hysteresis_loop_real.py --seq seq_20260627_172916
  python scripts/experiments/exp5b_hysteresis_loop_real.py --seq BOTH  # 两序列分别画
"""
import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# CJK 字体修复(避免"加载/卸载/半高宽"等中文变方框)
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RAW = os.path.join(ROOT, "real_capture", "data", "raw")
OUT = os.path.join(ROOT, "output", "exp5b_hysteresis_loop")
os.makedirs(OUT, exist_ok=True)

PHASE_COLORS = {"load": "#e74c3c", "unload": "#3498db"}  # 红=加载 蓝=卸载


def load_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
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


def load_seq(seq):
    """加载一个序列, 按 t_sec 对齐 actions/pressure/ndi."""
    _, act = load_csv(os.path.join(RAW, seq, "actions6.csv"))   # t,c0..c5
    _, pre = load_csv(os.path.join(RAW, seq, "pressure.csv"))   # t,p_active,reserved
    _, ndi = load_csv(os.path.join(RAW, seq, "ndi.csv"))        # t,x,y,z,...
    def align(base_t, arr_t, arr_data):
        idx = np.clip(np.searchsorted(arr_t, base_t), 0, len(arr_t) - 1)
        idx_left = np.clip(idx - 1, 0, len(arr_t) - 1)
        pick = np.where(np.abs(arr_t[idx] - base_t) < np.abs(arr_t[idx_left] - base_t),
                        idx, idx_left)
        return arr_data[pick]
    t = act[:, 0]
    return {
        "t": t, "c0": act[:, 1],
        "p": align(t, pre[:, 0], pre[:, 1]),
        "tip": align(t, ndi[:, 0], ndi[:, 1:4]),
        "n": len(t),
    }


def segment_cycles(c0):
    """三角波切成 [load(0→max), unload(max→0)] 周期."""
    d = np.diff(c0)
    s = np.sign(d); s[s == 0] = 1
    flips = np.where(np.diff(s) != 0)[0] + 1
    bounds = np.unique(np.r_[0, flips, len(c0)])
    starts, ends = bounds[:-1], bounds[1:]
    cycles = []
    for i in range(len(starts) - 1):
        a0, a1 = starts[i], ends[i]
        if c0[a1 - 1] > c0[a0]:                         # load 段(递增)
            u0, u1 = starts[i + 1], ends[i + 1]
            if c0[u1 - 1] < c0[u0]:                      # 紧跟的是 unload
                cycles.append({"load": (a0, a1), "unload": (u0, u1)})
    return cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="seq_20260627_173114",
                    help="raw 序列名; 或 BOTH 画两序列")
    ap.add_argument("--resp_axis", default="auto",
                    help="响应轴: auto(x,y 范围最大者) / x / y / z / |d|(到静止点距离)")
    args = ap.parse_args()

    seqs = ["seq_20260627_173114", "seq_20260627_172916"] if args.seq == "BOTH" else [args.seq]

    for seq in seqs:
        print(f"\n{'='*64}\n{seq}")
        D = load_seq(seq)
        p, tip = D["p"], D["tip"]
        if args.resp_axis == "auto":
            rng = tip[:, :2].max(0) - tip[:, :2].min(0)
            ax_i = int(np.argmax(rng))
            resp_name = ["x", "y"][ax_i]
            resp = tip[:, ax_i].copy()
            print(f"  响应轴 auto → {resp_name} (范围 {rng[ax_i]:.2f} mm vs {rng[1-ax_i]:.2f})")
        elif args.resp_axis == "|d|":
            rest_xy = tip[np.argsort(p)[:max(1, len(p)//10)]][:, :2].mean(0)
            resp = np.sqrt(((tip[:, :2] - rest_xy) ** 2).sum(1))
            resp_name = "|d|"
        else:
            ax_i = {"x": 0, "y": 1, "z": 2}[args.resp_axis]
            resp = tip[:, ax_i].copy(); resp_name = args.resp_axis
        # 居中到低气压(静止)响应均值
        resp = resp - resp[np.argsort(p)[:max(1, len(p)//10)]].mean()

        cyc = segment_cycles(D["c0"])
        print(f"  帧数={D['n']} p=[{p.min():.0f},{p.max():.0f}]kPa 周期={len(cyc)}")
        if not cyc:
            print("  !! 未识别到完整周期, 跳过"); continue

        # === Fig1: 迟滞回线 + 首末周期 ===
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        ax = axes[0]
        grid = np.linspace(0, 150, 60)
        load_curves, unload_curves = [], []
        for c in cyc:
            li, ui = slice(*c["load"]), slice(*c["unload"])
            ax.plot(p[li], resp[li], "-", color=PHASE_COLORS["load"], alpha=0.12, lw=1)
            ax.plot(p[ui], resp[ui], "-", color=PHASE_COLORS["unload"], alpha=0.12, lw=1)
            load_curves.append(np.interp(grid, p[li], resp[li]))
            unload_curves.append(np.interp(grid, p[ui][::-1], resp[ui][::-1]))
        load_m = np.mean(load_curves, 0)
        unload_m = np.mean(unload_curves, 0)
        ax.plot(grid, load_m, "-", color=PHASE_COLORS["load"], lw=2.5,
                label=f"Load (mean of {len(cyc)})")
        ax.plot(grid, unload_m, "-", color=PHASE_COLORS["unload"], lw=2.5,
                label=f"Unload (mean of {len(cyc)})")
        ax.fill(np.r_[grid, grid[::-1]], np.r_[load_m, unload_m[::-1]],
                alpha=0.18, color="purple", label="Hysteresis area")
        ax.plot(0, load_m[0], "k*", ms=14, zorder=10)
        ax.plot(150, load_m[-1], "k*", ms=14, zorder=10)
        mid = np.argmin(np.abs(grid - 75))
        hwidth = abs(load_m[mid] - unload_m[mid])
        ax.annotate(f"半高迟滞宽度\n≈{hwidth:.2f} mm @75kPa",
                    xy=(75, (load_m[mid] + unload_m[mid]) / 2), fontsize=9, ha="center",
                    xytext=(82, (load_m[mid] + unload_m[mid]) / 2),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        area = abs(np.trapz(load_m - unload_m, grid))
        ax.set_xlabel("Pressure p_active (kPa)", fontsize=13)
        ax.set_ylabel(f"Tip {resp_name} displacement (mm)", fontsize=13)
        ax.set_title(f"Real hysteresis loop — {seq}\n"
                     f"area={area:.2f} mm·kPa, 半高宽={hwidth:.2f} mm", fontsize=13)
        ax.legend(fontsize=9, loc="best"); ax.grid(True, alpha=0.3)

        ax = axes[1]
        c0c, cNc = cyc[0], cyc[-1]
        for tag, c, col in [("1st", c0c, "#e74c3c"), ("last", cNc, "#2ecc71")]:
            li, ui = slice(*c["load"]), slice(*c["unload"])
            ax.plot(p[li], resp[li], "-", color=col, lw=2, label=f"{tag} load")
            ax.plot(p[ui], resp[ui], "--", color=col, lw=2, label=f"{tag} unload")
        if len(cyc) >= 2:
            li, ui = slice(*cNc["load"]), slice(*cNc["unload"])
            drift = resp[ui.stop - 1] - resp[li.start]
            ax.set_title(f"First vs last cycle (preconditioning)\n"
                         f"末周期残余漂移 ≈ {drift:.2f} mm", fontsize=13)
        else:
            ax.set_title("First vs last cycle", fontsize=13)
        ax.set_xlabel("Pressure p_active (kPa)", fontsize=13)
        ax.set_ylabel(f"Tip {resp_name} displacement (mm)", fontsize=13)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        suffix = f"_{seq.split('_')[-1]}" if args.seq == "BOTH" else ""
        f1 = os.path.join(OUT, f"hysteresis_loop_real{suffix}.png")
        plt.savefig(f1, dpi=150); plt.close()
        print(f"  保存: {os.path.relpath(f1, ROOT)}  (area={area:.2f}, 半高宽={hwidth:.2f})")

        # === Fig2: 时间序列 ===
        fig, ax2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax = ax2[0]
        ax.plot(D["t"], D["c0"], "-", color="#8e44ad", lw=1.5, label="c0 command (kPa)")
        ax.plot(D["t"], p, ":", color="#e67e22", lw=1.2, label="p_active measured (kPa)")
        for c in cyc:
            ax.axvspan(D["t"][c["load"][0]], D["t"][c["unload"][1] - 1], alpha=0.06, color="#3498db")
        ax.set_ylabel("Pressure (kPa)", fontsize=12); ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3); ax.set_title(f"{seq} — 加载-卸载周期时间序列", fontsize=13)
        ax = ax2[1]
        ax.plot(D["t"], tip[:, 0], label="NDI x", lw=1.5)
        ax.plot(D["t"], tip[:, 1], label="NDI y", lw=1.5)
        ax.plot(D["t"], tip[:, 2], label="NDI z", lw=1.3, alpha=0.7)
        ax.set_xlabel("t (s)", fontsize=12); ax.set_ylabel("Tip position (mm)", fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        f2 = os.path.join(OUT, f"time_series_real{suffix}.png")
        plt.savefig(f2, dpi=150); plt.close()
        print(f"  保存: {os.path.relpath(f2, ROOT)}")

    with open(os.path.join(OUT, "summary.txt"), "a") as f:
        f.write(f"\n[real {args.seq}] resp={resp_name} area={area:.2f}mm*kPa 半高宽={hwidth:.2f}mm\n")
    print(f"\n=== 完成 → {OUT}/ ===")


if __name__ == "__main__":
    main()
