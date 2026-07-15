"""viz_control.py — 把方向1+2 的标量指标变成"看得见的模型行为"。

回答的问题:
  - 纯自回归 rollout 到底怎么发散?(open_loop vs gt, 把预测臂叠在真实臂上看)
  - 逆规划的轨迹长什么样?(planner / GT-actions / do-nothing 三条路径对比)

产出(output/viz/):
  horizon_rollout_grid.png — 2 行(open_loop/gt)× 6 列(k=1,20,40,80,160,300):
                              每格 预测骨架(色) vs GT 骨架(灰虚线), 直观看到发散
  plan_reach_compare.png   — 3 面板(planner/GT-actions/do-nothing):
                              s_init(蓝) → 轨迹(渐变色) → s_target(红) + 末态(绿/色)
  horizon_rollout_*_gif    — open_loop / gt rollout 动画(预测臂 vs 真实臂, 随 k 演化)
  plan_reach.gif           — planner 驱动 init→target 动画

帧间隔 0.203s(实测 frame_times), 故 K 步 ≈ K×0.2s: K=40→8.1s, K_max@10px=124→25s。

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/viz_control.py \
      --open_loop train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt \
      --gt train_log/gt_transition/exp_20260714_7/phase_gt_transition/model/best_model.pt \
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val \
      --plan_json output/inverse_plan/plan_result.json \
      --t0 500 --max_steps 300 --t_init 500 --t_target 540 --K 40
"""

import os
import sys
import glob
import json
import argparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
# CJK 字体: 系统 NotoSansCJK-Regular.ttc 在 matplotlib 里只注册为 "Noto Sans CJK JP"
# (CJK 统一表意文字中日韩共用, JP 变体也能渲染中文)。addfont 确保注册 + 用 JP 名。
from matplotlib import font_manager as _fm
for _p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.utils.model_loader import load_model
from src.evaluation.transition_metrics import build_action_window

FRAME_DT = 0.203   # 实测帧间隔(s)


def window_torch(buf, t, w):
    start = t - w + 1
    if start >= 0:
        return buf[start:t + 1]
    pad = torch.zeros((-start, buf.shape[1]), device=buf.device, dtype=buf.dtype)
    return torch.cat([pad, buf[0:t + 1]], 0)


def rollout_autoregressive(model, actions_norm_t, positions, t0, max_k, window_size, device):
    """纯自回归 rollout(1帧GT种子, 之后喂自身预测)。返回 preds/gt (K,N,3) 归一化。"""
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()
    def to_norm(pos_3N):
        skel = (pos_3N.T.astype(np.float32) - pc_center) / pc_scale
        return torch.from_numpy(skel).float().unsqueeze(0).to(device)
    s = to_norm(positions[t0]); s_prev = s
    z = model.init_z_from_action(window_torch(actions_norm_t, t0, window_size).unsqueeze(0))
    K = min(max_k, positions.shape[0] - 1 - t0)
    preds, gts = [], []
    with torch.no_grad():
        for k in range(1, K + 1):
            tt = t0 + k
            aw = window_torch(actions_norm_t, tt, window_size).unsqueeze(0)
            out = model.forward(aw, s, s_prev, z)
            s_pred = out["skeleton"]; z = out["latent_z"]
            preds.append(s_pred.squeeze(0).cpu().numpy())
            gts.append(to_norm(positions[tt]).squeeze(0).cpu().numpy())
            s_prev = s; s = s_pred
    return np.stack(preds, 0), np.stack(gts, 0)


def rollout_actions(model, history_t, s_init, actions_seq, K, window_size):
    """给定一段动作序列(接在 history 之后), rollout K 步。返回 preds (K,N,3) 归一化。"""
    t_start = history_t.shape[0] - 1
    buf = torch.cat([history_t, actions_seq], 0)
    s = s_init; s_prev = s_init
    z = model.init_z_from_action(window_torch(buf, t_start, window_size).unsqueeze(0))
    preds = []
    with torch.no_grad():
        for k in range(1, K + 1):
            aw = window_torch(buf, t_start + k, window_size).unsqueeze(0)
            out = model.forward(aw, s, s_prev, z)
            s_pred = out["skeleton"]; z = out["latent_z"]
            preds.append(s_pred.squeeze(0).cpu().numpy())
            s_prev = s; s = s_pred
    return np.stack(preds, 0)


def to_px(skel_norm, pc_center, pc_scale):
    return skel_norm * pc_scale + pc_center


def err_px(pred_norm, gt_norm, pc_center, pc_scale):
    p = to_px(pred_norm, pc_center, pc_scale)
    g = to_px(gt_norm, pc_center, pc_scale)
    return float(np.sqrt(((p[..., :2] - g[..., :2]) ** 2).sum(-1)).mean())


# ── 1. horizon rollout grid ──
def viz_horizon_grid(models, names, actions_norm_t, positions, t0, max_steps,
                     window_size, device, ks, out_path):
    pc_center = models[0].pc_center.view(3).cpu().numpy()
    pc_scale = models[0].pc_scale.view(3).cpu().numpy()
    fig, axes = plt.subplots(len(models), len(ks), figsize=(3.2 * len(ks), 3.6 * len(models)))
    if len(models) == 1:
        axes = axes[None, :]
    for r, (model, name) in enumerate(zip(models, names)):
        preds, gts = rollout_autoregressive(model, actions_norm_t, positions, t0,
                                            max_steps, window_size, device)
        for c, k in enumerate(ks):
            ax = axes[r, c]
            if k - 1 >= preds.shape[0]:
                ax.axis("off"); continue
            gt_px = to_px(gts[k - 1], pc_center, pc_scale)
            pr_px = to_px(preds[k - 1], pc_center, pc_scale)
            e = err_px(preds[k - 1], gts[k - 1], pc_center, pc_scale)
            ax.plot(gt_px[:, 0], gt_px[:, 1], "o--", color="gray", lw=2, ms=3, alpha=0.7, label="GT 真实")
            ax.plot(pr_px[:, 0], pr_px[:, 1], "s-", color="crimson" if r == 1 else "navy",
                    lw=2, ms=3, label="预测")
            ax.invert_yaxis(); ax.set_aspect("auto")  # auto: col范围小(50px)vs row(283px), equal会把x轴压成1/6; auto让x填满面板便于阅读(横向略拉伸, 但预测与GT同拉伸, 相对形变不失真)
            ax.set_title(f"k={k} ({k*FRAME_DT:.0f}s)  err={e:.1f}px", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=10, fontweight="bold")
            if r == 0 and c == 0:
                ax.legend(fontsize=7, loc="best")
            if c > 0:
                ax.sharex(axes[r, 0]); ax.sharey(axes[r, 0])
    fig.suptitle(f"纯自回归 rollout: 预测臂 vs 癟实臂 (种子 t0={t0}, 帧间隔 {FRAME_DT}s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()


# ── 2. horizon rollout GIF ──
def viz_horizon_gif(model, name, actions_norm_t, positions, t0, max_steps,
                    window_size, device, out_path, stride=4):
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()
    preds, gts = rollout_autoregressive(model, actions_norm_t, positions, t0,
                                        max_steps, window_size, device)
    K = preds.shape[0]
    frames = list(range(0, K, stride))
    fig, ax = plt.subplots(figsize=(6, 8))
    gt_px_all = to_px(gts, pc_center, pc_scale)
    lo = np.nanpercentile(gt_px_all.reshape(-1, 3)[:, :2], 1, 0)
    hi = np.nanpercentile(gt_px_all.reshape(-1, 3)[:, :2], 99, 0)
    pad = (hi - lo) * 0.1; lo, hi = lo - pad, hi + pad

    def draw(i):
        ax.clear()
        k = frames[i] + 1
        gt = gt_px_all[k - 1]
        pr = to_px(preds[k - 1], pc_center, pc_scale).copy()
        pr[:, 0] = np.clip(pr[:, 0], lo[0], hi[0]); pr[:, 1] = np.clip(pr[:, 1], lo[1], hi[1])
        ax.plot(gt[:, 0], gt[:, 1], "o--", color="gray", lw=3, ms=6, alpha=0.8, label="GT 真实")
        ax.plot(pr[:, 0], pr[:, 1], "s-", color="navy", lw=3, ms=6, label="预测")
        e = err_px(preds[k - 1], gts[k - 1], pc_center, pc_scale)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1]); ax.set_aspect("auto")
        ax.set_title(f"{name}  k={k}/{K} ({k*FRAME_DT:.0f}s)  err={e:.1f}px", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ani = FuncAnimation(fig, draw, frames=len(frames), interval=200)
    ani.save(out_path, writer=PillowWriter(fps=5)); plt.close()


# ── 3. plan reach compare ──
def viz_plan_compare(model, history_t, s_init, s_target, a_plan, gt_act, seed_last,
                     K, window_size, pc_center, pc_scale, device, out_path):
    preds_plan = rollout_actions(model, history_t, s_init, a_plan, K, window_size)
    preds_gt = rollout_actions(model, history_t, s_init, gt_act, K, window_size)
    preds_do = rollout_actions(model, history_t, s_init, seed_last.repeat(K, 1), K, window_size)
    init_np = s_init.squeeze(0).cpu().numpy()
    tgt_np = s_target.squeeze(0).cpu().numpy()
    panels = [("planner(优化动作)", preds_plan, "navy"),
              ("GT-actions(真实动作)", preds_gt, "darkgreen"),
              ("do-nothing(重复末动作)", preds_do, "darkorange")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    cmap = plt.cm.viridis
    for ax, (title, preds, col) in zip(axes, panels):
        for k in range(K):
            p = to_px(preds[k], pc_center, pc_scale)
            ax.plot(p[:, 0], p[:, 1], "-", color=cmap(k / max(K - 1, 1)), alpha=0.5, lw=1.0)
        pi = to_px(init_np, pc_center, pc_scale)
        pt = to_px(tgt_np, pc_center, pc_scale)
        pp = to_px(preds[-1], pc_center, pc_scale)
        fin_err = err_px(preds[-1], tgt_np, pc_center, pc_scale)
        ax.plot(pi[:, 0], pi[:, 1], "o-", color="blue", lw=2.5, ms=5, label="s_init")
        ax.plot(pt[:, 0], pt[:, 1], "s-", color="red", lw=2.5, ms=6, label="s_target")
        ax.plot(pp[:, 0], pp[:, 1], "^-", color=col, lw=2.5, ms=6, label=f"末态({fin_err:.1f}px)")
        ax.invert_yaxis(); ax.set_aspect("auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
        ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    fig.suptitle(f"逆规划轨迹对比 (K={K}步={K*FRAME_DT:.0f}s): 哪条路径到 s_target?", fontsize=13)
    plt.tight_layout(); plt.savefig(out_path, dpi=120); plt.close()


def viz_plan_gif(model, history_t, s_init, s_target, a_plan, K, window_size,
                 pc_center, pc_scale, device, out_path):
    preds = rollout_actions(model, history_t, s_init, a_plan, K, window_size)
    tgt_np = s_target.squeeze(0).cpu().numpy()
    init_np = s_init.squeeze(0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 8))
    allpts = to_px(np.concatenate([preds, init_np[None], tgt_np[None]], 0), pc_center, pc_scale)
    lo = np.nanpercentile(allpts.reshape(-1, 3)[:, :2], 1, 0)
    hi = np.nanpercentile(allpts.reshape(-1, 3)[:, :2], 99, 0)
    pad = (hi - lo) * 0.1; lo, hi = lo - pad, hi + pad
    pt = to_px(tgt_np, pc_center, pc_scale); pi = to_px(init_np, pc_center, pc_scale)

    def draw(k):
        ax.clear()
        p = to_px(preds[k], pc_center, pc_scale)
        e = err_px(preds[k], tgt_np, pc_center, pc_scale)
        for j in range(k + 1):
            pj = to_px(preds[j], pc_center, pc_scale)
            ax.plot(pj[:, 0], pj[:, 1], "-", color=plt.cm.viridis(j / max(K - 1, 1)), alpha=0.4, lw=1)
        ax.plot(pi[:, 0], pi[:, 1], "o-", color="blue", lw=2.5, ms=5, label="s_init")
        ax.plot(pt[:, 0], pt[:, 1], "s-", color="red", lw=2.5, ms=6, label="s_target")
        ax.plot(p[:, 0], p[:, 1], "^-", color="navy", lw=2.5, ms=6, label=f"当前({e:.1f}px)")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1]); ax.set_aspect("auto")
        ax.set_title(f"planner  k={k+1}/{K} ({(k+1)*FRAME_DT:.1f}s)  err={e:.1f}px", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
    ani = FuncAnimation(fig, draw, frames=K, interval=150)
    ani.save(out_path, writer=PillowWriter(fps=6)); plt.close()


def main():
    p = argparse.ArgumentParser(description="方向1+2 可视化")
    p.add_argument("--open_loop", type=str, required=True)
    p.add_argument("--gt", type=str, default=None)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--plan_json", type=str, default=None)
    p.add_argument("--t0", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--t_init", type=int, default=500)
    p.add_argument("--t_target", type=int, default=540)
    p.add_argument("--K", type=int, default=40)
    p.add_argument("--out", type=str, default="output/viz")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    raw = np.load(files[0])
    actions = raw["actions"].astype(np.float32)
    positions = raw["positions"].astype(np.float32)
    actions_norm_t = torch.from_numpy(actions).float().to(device)   # norm_factor=1.0

    ks = [k for k in [1, 20, 40, 80, 160, 300] if k <= args.max_steps]
    models, names = [], []

    print("加载 open_loop...")
    info = load_model(args.open_loop, data_dir=args.data_dir, device=device)
    ol = info["model"]; ol.eval()
    models.append(ol); names.append("open_loop")
    window_size = info["window_size"]
    pc_center = ol.pc_center.view(3).cpu().numpy()
    pc_scale = ol.pc_scale.view(3).cpu().numpy()

    gt = None
    if args.gt:
        print("加载 gt...")
        gt = load_model(args.gt, data_dir=args.data_dir, device=device)["model"]; gt.eval()
        models.append(gt); names.append("gt")

    print("画 horizon_rollout_grid.png ...")
    viz_horizon_grid(models, names, actions_norm_t, positions, args.t0, args.max_steps,
                     window_size, device, ks, os.path.join(args.out, "horizon_rollout_grid.png"))

    print("画 horizon_rollout_open_loop.gif ...")
    viz_horizon_gif(ol, "open_loop", actions_norm_t, positions, args.t0, args.max_steps,
                    window_size, device, os.path.join(args.out, "horizon_rollout_open_loop.gif"))
    if gt is not None:
        viz_horizon_gif(gt, "gt", actions_norm_t, positions, args.t0, min(120, args.max_steps),
                        window_size, device, os.path.join(args.out, "horizon_rollout_gt.gif"))

    if args.plan_json:
        print("画 plan_reach_compare.png + plan_reach.gif ...")
        D = actions.shape[1]
        t_a = args.t_init
        pr = json.load(open(args.plan_json))
        a_plan = torch.from_numpy(np.array(pr["a_plan"], dtype=np.float32)).to(device)
        hist_len = max(window_size, 2)
        history_np = actions[max(0, t_a - hist_len + 1):t_a + 1]
        if history_np.shape[0] < hist_len:
            history_np = np.concatenate(
                [np.zeros((hist_len - history_np.shape[0], D), dtype=np.float32), history_np], 0)
        history_t = torch.from_numpy(history_np).float().to(device)
        seed_last = history_t[-1:].clone()
        gt_act = torch.from_numpy(actions[t_a + 1:t_a + args.K + 1]).float().to(device)
        pc_c_t = torch.from_numpy(pc_center); pc_s_t = torch.from_numpy(pc_scale)
        def to_norm(pos_3N):
            skel = torch.from_numpy(pos_3N.T.astype(np.float32))
            return ((skel - pc_c_t) / pc_s_t).unsqueeze(0).to(device)
        s_init = to_norm(positions[args.t_init])
        s_target = to_norm(positions[args.t_target])
        viz_plan_compare(ol, history_t, s_init, s_target, a_plan, gt_act, seed_last,
                         args.K, window_size, pc_center, pc_scale, device,
                         os.path.join(args.out, "plan_reach_compare.png"))
        viz_plan_gif(ol, history_t, s_init, s_target, a_plan, args.K, window_size,
                     pc_center, pc_scale, device, os.path.join(args.out, "plan_reach.gif"))

    print(f"\n完成 → {args.out}/")


if __name__ == "__main__":
    main()
