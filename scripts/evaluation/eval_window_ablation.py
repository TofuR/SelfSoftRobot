#!/usr/bin/env python3
"""eval_window_ablation.py — window=1(无动作历史) vs window=40(有动作历史) gt 模型对比.

方向17 Exp2: 检验"动作历史记忆"对路径依赖形状预测的价值。
关键设计: 单步预测(gt 语义, 每步喂真实 s_{t-1}, z 跨帧演化), 消除 rollout 漂移干扰,
只看"给定真实上一帧 + 动作, 能不能预测下一帧形状"。按**动作趋势**(loading/unloading/
hold)分层误差。

假设(路径依赖):
  window=1 只看当前动作(+s_{t-1}) → 对同一动作的 load/unload 倾向预测同一形状 →
    方向反转/少数方向帧误差大, load vs unload 误差不对称。
  window=40 看动作历史 → 能区分 load/unload → 误差小且对称。

输入: 两个 checkpoint(同模式 gt, 同数据, 只差 window) + val 目录。
输出(output/window_ablation/):
  ablation_summary.json    定量(总体 + 按趋势分层 + load/unload gap)
  err_by_trend.png         分层误差柱状图(核心)
  err_vs_action.png        末端预测 vs 动作值, 按趋势着色(迟滞结构)
  err_over_time.png        逐帧误差时间序列 + 趋势底色
  skeleton_overlay.png     匹配动作的 load vs unload 帧预测叠加

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/eval_window_ablation.py \
      --w1  train_log/gt_transition/exp_20260716_8/phase_gt_transition/model/best_model.pt \
      --w40 train_log/gt_transition/exp_20260714_7/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val \
      --out output/window_ablation
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

TREND_COLORS = {"load": "#e74c3c", "unload": "#3498db", "hold": "#bbbbbb"}
LABELS = {"w1": "window=1 (no action history)", "w40": "window=40 (with history)"}


def px_node_err(pred_norm, gt_norm, pc_scale, pc_center):
    """单帧: norm (N,3) → 平均节点 px(col-row 平面)。"""
    p = pred_norm * pc_scale + pc_center
    g = gt_norm * pc_scale + pc_center
    return float(np.sqrt(((p[..., :2] - g[..., :2]) ** 2).sum(-1)).mean())


def eval_model(ckpt, data_dir, device):
    info = load_model(ckpt, data_dir=data_dir, device=device)
    m = info["model"]
    m.eval()
    ws = info["window_size"]
    nf = info["norm_factor"] or 1.0
    pc_center = m.pc_center.view(3).cpu().numpy()
    pc_scale = m.pc_scale.view(3).cpu().numpy()

    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    raw = np.load(files[0])
    actions = raw["actions"].astype(np.float32)      # (T, A)
    positions = raw["positions"].astype(np.float32)  # (T, 3, N)
    T = positions.shape[0]
    actions_norm = actions / nf

    def to_norm(pos_3N):
        sk = pos_3N.T.astype(np.float32)  # (N,3)
        return torch.from_numpy((sk - pc_center) / pc_scale).float().unsqueeze(0).to(device)

    errs, trends, acts, preds_xy, gts_xy = [], [], [], [], []
    z = m.init_z_from_action(
        torch.from_numpy(build_action_window(actions_norm, 1, ws)).float().unsqueeze(0).to(device))
    with torch.no_grad():
        for tt in range(2, T):
            aw = torch.from_numpy(build_action_window(actions_norm, tt, ws)).float().unsqueeze(0).to(device)
            out = m.forward(aw, to_norm(positions[tt - 1]), to_norm(positions[tt - 2]), z)
            z = out["latent_z"]
            pred = out["skeleton"].squeeze(0).cpu().numpy()   # (N,3) norm
            gt = to_norm(positions[tt]).squeeze(0).cpu().numpy()
            errs.append(px_node_err(pred, gt, pc_scale, pc_center))
            trends.append(float(actions_norm[tt, 0] - actions_norm[tt - 1, 0]))
            acts.append(float(actions_norm[tt, 0]))
            preds_xy.append((pred * pc_scale + pc_center)[:, :2])
            gts_xy.append((gt * pc_scale + pc_center)[:, :2])
    return {
        "errs": np.array(errs), "trends": np.array(trends), "acts": np.array(acts),
        "preds_xy": np.array(preds_xy), "gts_xy": np.array(gts_xy),
        "window_size": ws, "n_frames": T - 2,
    }


def classify(trend, thr):
    if trend > thr:
        return "load"
    if trend < -thr:
        return "unload"
    return "hold"


def summarize(res, thr):
    e = res["errs"]
    tr = np.array([classify(t, thr) for t in res["trends"]])
    by = {k: e[tr == k] for k in ("load", "unload", "hold")}
    s = {
        "window_size": res["window_size"],
        "n_frames": int(res["n_frames"]),
        "mean_px": float(e.mean()), "median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
        "n_load": int((tr == "load").sum()), "n_unload": int((tr == "unload").sum()), "n_hold": int((tr == "hold").sum()),
        "mean_load_px": float(by["load"].mean()) if by["load"].size else float("nan"),
        "mean_unload_px": float(by["unload"].mean()) if by["unload"].size else float("nan"),
        "mean_hold_px": float(by["hold"].mean()) if by["hold"].size else float("nan"),
        "load_unload_gap_px": float(abs(by["load"].mean() - by["unload"].mean())) if by["load"].size and by["unload"].size else float("nan"),
    }
    return s, tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1", required=True)
    ap.add_argument("--w40", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="output/window_ablation")
    ap.add_argument("--trend_thr", type=float, default=0.02,
                    help="动作趋势阈值(归一化 action 单位): |da|>thr=load/unload")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = {}
    for key, ckpt in (("w1", args.w1), ("w40", args.w40)):
        print(f"\n=== 评估 {key}: {ckpt} ===")
        res[key] = eval_model(ckpt, args.data_dir, device)
        print(f"  window={res[key]['window_size']} 帧={res[key]['n_frames']} "
              f"mean_err={res[key]['errs'].mean():.3f}px median={np.median(res[key]['errs']):.3f}px")

    summaries, trs = {}, {}
    for k in ("w1", "w40"):
        summaries[k], trs[k] = summarize(res[k], args.trend_thr)

    # ── Fig A: 按趋势分层误差(核心)──
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["load", "unload", "hold"]
    x = np.arange(len(cats))
    w = 0.35
    for i, k in enumerate(("w1", "w40")):
        vals = [summaries[k][f"mean_{c}_px"] for c in cats]
        ax.bar(x + (i - 0.5) * w, vals, w, label=LABELS[k],
               color=TREND_COLORS["load"] if k == "w1" else TREND_COLORS["unload"], alpha=0.8)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={summaries['w1'][f'n_{c}']})" for c in cats])
    ax.set_ylabel("单步预测误差 (px, 平均节点)")
    better_w40 = (summaries["w1"]["mean_load_px"] + summaries["w1"]["mean_unload_px"]) - \
                 (summaries["w40"]["mean_load_px"] + summaries["w40"]["mean_unload_px"])
    verdict = "动作历史有增益(路径依赖信号)" if better_w40 > 0.05 else "动作历史增益不显著(s_{t-1} 已携带记忆?)"
    ax.set_title(f"window=1 vs 40 单步误差(按动作趋势分层)\n{verdict}: "
                 f"w1 load/unload gap={summaries['w1']['load_unload_gap_px']:.2f}px, w40={summaries['w40']['load_unload_gap_px']:.2f}px")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "err_by_trend.png"), dpi=150); plt.close()

    # ── Fig B: 末端预测(node0) vs 动作值, 按趋势着色 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, k in zip(axes, ("w1", "w40")):
        tip_pred = res[k]["preds_xy"][:, 0, :]  # (T,2)
        tip_gt = res[k]["gts_xy"][:, 0, :]
        for tag, col in TREND_COLORS.items():
            msk = trs[k] == tag
            if not msk.any():
                continue
            ax.scatter(res[k]["acts"][msk], tip_pred[msk, 0], s=6, color=col, alpha=0.4, label=f"pred {tag}")
        ax.scatter(res[k]["acts"], tip_gt[:, 0], s=3, color="black", alpha=0.15, label="GT")
        ax.set_xlabel("action (归一化气压)"); ax.set_ylabel("node0 col (px)")
        ax.set_title(LABELS[k]); ax.legend(fontsize=7, loc="best"); ax.grid(alpha=0.3)
    fig.suptitle("末端预测 vs 动作值(按趋势着色): w1 呈 load/unload 双带(预测不出方向), w40 双带应更窄", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "err_vs_action.png"), dpi=150); plt.close()

    # ── Fig C: 逐帧误差时间序列 ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for ax, k in zip(axes, ("w1", "w40")):
        ax.plot(res[k]["errs"], lw=0.8, color=TREND_COLORS["load"] if k == "w1" else TREND_COLORS["unload"])
        ax.set_ylabel("px"); ax.set_title(LABELS[k]); ax.grid(alpha=0.3)
        um = trs[k] == "unload"
        for i in np.where(um)[0]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color=TREND_COLORS["unload"])
    axes[1].set_xlabel("val 帧序号")
    fig.suptitle("逐帧单步误差(蓝底=unload 帧): 看 w1 是否在 unload 段系统性偏高", fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "err_over_time.png"), dpi=150); plt.close()

    # ── Fig D: 匹配动作的 load vs unload 帧预测叠加 ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, k in zip(axes, ("w1", "w40")):
        acts = res[k]["acts"]; tr = trs[k]
        load_idx = np.where(tr == "load")[0]; unload_idx = np.where(tr == "unload")[0]
        if len(load_idx) and len(unload_idx):
            li = load_idx[len(load_idx) // 2]
            ui = unload_idx[np.argmin(np.abs(acts[unload_idx] - acts[li]))]
            for idx, col, tag in [(li, TREND_COLORS["load"], "loading"), (ui, TREND_COLORS["unload"], "unloading")]:
                ax.plot(res[k]["gts_xy"][idx, :, 0], res[k]["gts_xy"][idx, :, 1], "o-", color=col, ms=3, alpha=0.3)
                ax.plot(res[k]["preds_xy"][idx, :, 0], res[k]["preds_xy"][idx, :, 1], "x--", color=col, ms=4, label=f"{tag} (a={acts[idx]:.2f})")
            ax.set_title(f"{LABELS[k]}\n同动作 load vs unload: w1 预测应几乎重合, w40 应分开")
            ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "skeleton_overlay.png"), dpi=150); plt.close()

    out = {"trend_thr": args.trend_thr, "data_dir": args.data_dir,
           "checkpoints": {"w1": args.w1, "w40": args.w40},
           "w1": summaries["w1"], "w40": summaries["w40"]}
    with open(os.path.join(args.out, "ablation_summary.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("\n=== 定量对比 ===")
    print(f"{'指标':<20}{'win=1':>12}{'win=40':>12}")
    for met in ("mean_px", "median_px", "p90_px", "mean_load_px", "mean_unload_px", "mean_hold_px", "load_unload_gap_px"):
        print(f"{met:<20}{summaries['w1'][met]:>12.3f}{summaries['w40'][met]:>12.3f}")
    gap_diff = summaries["w1"]["load_unload_gap_px"] - summaries["w40"]["load_unload_gap_px"]
    print(f"\nload/unload 误差不对称: w1={summaries['w1']['load_unload_gap_px']:.3f}px  w40={summaries['w40']['load_unload_gap_px']:.3f}px  (差 {gap_diff:+.3f})")
    print(f"\n已保存 → {args.out}/")


if __name__ == "__main__":
    main()
