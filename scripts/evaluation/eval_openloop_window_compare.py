#!/usr/bin/env python3
"""eval_openloop_window_compare.py — open_loop window=1 vs window=40 决定性对比.

为什么这是决定性测试(而之前 gt 单步对比不是):
  gt 模式每步喂真实 s_{t-1} → s_{t-1} 已携带加载历史 → 动作窗口被盖住。
  open_loop 纯自回归 rollout(只 1 帧真实种子, 之后喂自身预测) → **动作窗口是唯一记忆来源**。
  所以 w1 vs w40 在 open_loop 下才真正检验"动作历史记忆有没有用"。

做什么(每模型, 8 种子 × 300 步纯自回归):
  - 漂移曲线: 平均节点误差、末端(node0)误差, ±跨种子 std, log-y
  - K_max: 平均曲线 & 末端曲线, 在多 mm 容差下的可信步数(+秒)
  - 分布: std / max across nodes(沿臂误差分布), 末步 per-node 误差
  - 单位: px 与 mm 双标(mm 由臂直径 10mm=33px → 0.302 mm/px)

输出(output/openloop_window_compare/):
  drift_curves.png     漂移曲线(mean实线/tip虚线, w1 vs w40, 容差线)  ← 核心
  kmax_bar.png         K_max 柱状(mean vs tip × w1 vs w40 @多mm)
  pernode_final.png    末步(k=300)沿臂各 node 误差(base→tip)
  err_band.png         mean 漂移 ±1σ 带
  compare_summary.json

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/eval_openloop_window_compare.py \
      --w1  train_log/open_loop_transition/exp_20260716_9/phase_open_loop_transition/model/best_model.pt \
      --w40 train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_CJK):
    font_manager.fontManager.addfont(_CJK)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from src.utils.model_loader import load_model
from src.evaluation.transition_metrics import build_action_window

MM_PER_PX = 0.302          # 臂直径 10mm ≈ 33px
FRAME_DT = 0.203           # 帧间隔 ~5fps
LABELS = {"w1": "open_loop window=1", "w40": "open_loop window=40"}
COLORS = {"w1": "#e74c3c", "w40": "#3498db"}


def rollout_pernode(model, actions_norm, positions, t0, max_k, ws, device):
    """纯自回归 rollout(1 真实种子→喂自身预测), 返回每步每节点 px 误差 (K,N)。"""
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()

    def to_norm(pos_3N):
        sk = pos_3N.T.astype(np.float32)
        return torch.from_numpy((sk - pc_center) / pc_scale).float().unsqueeze(0).to(device)

    s_roll = to_norm(positions[t0])
    s_prev = s_roll
    z = model.init_z_from_action(
        torch.from_numpy(build_action_window(actions_norm, t0, ws)).float().unsqueeze(0).to(device))
    K = min(max_k, positions.shape[0] - 1 - t0)
    N = positions.shape[2]
    errs = np.zeros((K, N), dtype=np.float32)
    with torch.no_grad():
        for k in range(1, K + 1):
            aw = torch.from_numpy(build_action_window(actions_norm, t0 + k, ws)).float().unsqueeze(0).to(device)
            out = model.forward(aw, s_roll, s_prev, z)
            z = out["latent_z"]
            s_pred = out["skeleton"]
            gt = to_norm(positions[t0 + k])
            p = (s_pred.squeeze(0).cpu().numpy() * pc_scale + pc_center)[:, :2]
            g = (gt.squeeze(0).cpu().numpy() * pc_scale + pc_center)[:, :2]
            errs[k - 1] = np.sqrt(((p - g) ** 2).sum(-1))   # (N,) px
            s_prev = s_roll
            s_roll = s_pred
    return errs


def characterize(ckpt, data_dir, max_steps, n_seeds, device):
    info = load_model(ckpt, data_dir=data_dir, device=device)
    m = info["model"]; m.eval()
    ws = info["window_size"]
    nf = info["norm_factor"] or 1.0
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    raw = np.load(files[0])
    actions = raw["actions"].astype(np.float32)
    positions = raw["positions"].astype(np.float32)
    T, _, N = positions.shape
    actions_norm = actions / nf
    seeds = np.linspace(1, max(2, T - max_steps - 2), n_seeds, dtype=int)
    acc = []
    for t0 in seeds:
        acc.append(rollout_pernode(m, actions_norm, positions, int(t0), max_steps, ws, device))
    K = min(acc[0].shape[0], max_steps)
    acc = np.stack([a[:K] for a in acc])            # (n_seeds, K, N)
    tip = acc[:, :, 0]                               # node0 = 末端
    meannode = acc.mean(2)
    maxnode = acc.max(2)
    stdnode = acc.std(2)
    return {
        "window_size": ws, "n_seeds": int(len(seeds)), "K": int(K),
        "tip_mean": tip.mean(0), "tip_std": tip.std(0),
        "mean_mean": meannode.mean(0), "mean_std": meannode.std(0),
        "max_mean": maxnode.mean(0), "stdnode_mean": stdnode.mean(0),
        "pernode_final": acc[:, -1, :].mean(0),       # 末步各 node (N,)
    }


def kmax_above(curve_px, thr_px):
    hits = np.where(curve_px > thr_px)[0]
    return int(hits[0] + 1) if len(hits) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w40", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="output/openloop_window_compare")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--n_seeds", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = {}
    for key, ckpt in (("w1", args.w1), ("w40", args.w40)):
        print(f"\n=== {key}: {ckpt} ===")
        res[key] = characterize(ckpt, args.data_dir, args.max_steps, args.n_seeds, device)
        r = res[key]
        print(f"  win={r['window_size']} K={r['K']} seeds={r['n_seeds']}")
        print(f"  末步 mean={r['mean_mean'][-1]:.2f}px({r['mean_mean'][-1]*MM_PER_PX:.2f}mm)  "
              f"tip={r['tip_mean'][-1]:.2f}px({r['tip_mean'][-1]*MM_PER_PX:.2f}mm)")

    mm_tols = [1.0, 1.5, 3.0, 6.0]
    px_tols = [m / MM_PER_PX for m in mm_tols]
    k = np.arange(1, res["w1"]["K"] + 1)

    # ── Fig 1: 漂移曲线(mean实线/tip虚线)w1 vs w40 ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ("w1", "w40"):
        ax.plot(k, res[key]["mean_mean"] * MM_PER_PX, "-", color=COLORS[key], lw=2,
                label=f"{LABELS[key]} — mean node")
        ax.plot(k, res[key]["tip_mean"] * MM_PER_PX, "--", color=COLORS[key], lw=1.5,
                label=f"{LABELS[key]} — tip (node0)")
    for mtol in (1.5, 3.0, 6.0):
        ax.axhline(mtol, color="gray", ls=":", lw=0.8)
        ax.text(res["w1"]["K"] * 0.99, mtol, f"{mtol}mm", fontsize=8, color="gray",
                ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("rollout step k(距上次观测的步数)"); ax.set_ylabel("误差 (mm, col-row 平面)")
    ax.set_title("open_loop window=1 vs 40 纯自回归漂移\n(mean 实线 / tip 虚线; 容差线 1.5/3/6mm)", fontsize=12)
    ax.legend(fontsize=8, loc="best"); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "drift_curves.png"), dpi=150); plt.close()

    # ── Fig 2: K_max 柱状 @ 多 mm 容差, mean vs tip × w1 vs w40 ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, curve_key, title in [(axes[0], "mean_mean", "平均节点"), (axes[1], "tip_mean", "末端 node0")]:
        x = np.arange(len(mm_tols)); w = 0.35
        for i, key in enumerate(("w1", "w40")):
            vals = []
            for px in px_tols:
                km = kmax_above(res[key][curve_key], px)
                vals.append(km if km is not None else res[key]["K"])
            ax.bar(x + (i - 0.5) * w, vals, w, color=COLORS[key], label=LABELS[key], alpha=0.85)
            for xi, v in zip(x + (i - 0.5) * w, vals):
                ax.text(xi, v + 2, f"{v}\n{v*FRAME_DT:.0f}s", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"{m}mm" for m in mm_tols])
        ax.set_ylabel("K_max(步)"); ax.set_title(f"K_max @容差 — {title}"); ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "kmax_bar.png"), dpi=150); plt.close()

    # ── Fig 3: 末步沿臂各 node 误差(node0=末端 ... node14=基座)──
    fig, ax = plt.subplots(figsize=(10, 5))
    nn = np.arange(res["w1"]["pernode_final"].shape[0])
    for key in ("w1", "w40"):
        ax.plot(nn, res[key]["pernode_final"] * MM_PER_PX, "-o", color=COLORS[key], lw=2,
                ms=4, label=LABELS[key])
    ax.invert_xaxis()  # node0(末端)在右, 基座在左
    ax.set_xlabel("node index(右=末端 node0, 左=基座 node14)")
    ax.set_ylabel("末步(k=300)误差 (mm)")
    ax.set_title("沿臂误差分布(k=300): 看 w1 是否在末端/某段更糟"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "pernode_final.png"), dpi=150); plt.close()

    # ── Fig 4: mean 漂移 ±1σ 带 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("w1", "w40"):
        m = res[key]["mean_mean"] * MM_PER_PX
        s = res[key]["mean_std"] * MM_PER_PX
        ax.plot(k, m, color=COLORS[key], lw=2, label=LABELS[key])
        ax.fill_between(k, m - s, m + s, color=COLORS[key], alpha=0.15)
    ax.set_yscale("log"); ax.set_xlabel("rollout step k"); ax.set_ylabel("平均节点误差 (mm, ±1σ)")
    ax.set_title("漂移带(跨8种子 ±1σ)"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "err_band.png"), dpi=150); plt.close()

    def serialize(r):
        return {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in r.items()}
    summary = {"mm_per_px": MM_PER_PX, "frame_dt": FRAME_DT, "data_dir": args.data_dir,
               "checkpoints": {"w1": args.w1, "w40": args.w40},
               "w1": serialize(res["w1"]), "w40": serialize(res["w40"]),
               "kmax_steps_mean": {f"{m}mm": {key: kmax_above(res[key]["mean_mean"], m / MM_PER_PX) for key in ("w1", "w40")} for m in mm_tols},
               "kmax_steps_tip": {f"{m}mm": {key: kmax_above(res[key]["tip_mean"], m / MM_PER_PX) for key in ("w1", "w40")} for m in mm_tols}}
    with open(os.path.join(args.out, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== K_max(步; 1步≈0.203s) ===")
    print(f"{'容差':<8}{'mean w1':>12}{'mean w40':>12}{'tip w1':>12}{'tip w40':>12}")
    for m in mm_tols:
        km1 = kmax_above(res["w1"]["mean_mean"], m / MM_PER_PX)
        km2 = kmax_above(res["w40"]["mean_mean"], m / MM_PER_PX)
        kt1 = kmax_above(res["w1"]["tip_mean"], m / MM_PER_PX)
        kt2 = kmax_above(res["w40"]["tip_mean"], m / MM_PER_PX)
        f = lambda v: (f"{v}({v*FRAME_DT:.0f}s)" if v else ">K")
        print(f"{m}mm     {f(km1):>12}{f(km2):>12}{f(kt1):>12}{f(kt2):>12}")
    print(f"\n末步(k={res['w1']['K']}) mean: w1={res['w1']['mean_mean'][-1]*MM_PER_PX:.2f}mm  "
          f"w40={res['w40']['mean_mean'][-1]*MM_PER_PX:.2f}mm  | "
          f"tip: w1={res['w1']['tip_mean'][-1]*MM_PER_PX:.2f}mm  w40={res['w40']['tip_mean'][-1]*MM_PER_PX:.2f}mm")
    print(f"\n已保存 → {args.out}/")


if __name__ == "__main__":
    main()
