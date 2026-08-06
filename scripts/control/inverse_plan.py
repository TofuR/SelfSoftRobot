"""inverse_plan.py — 方向2: 可微逆运动学形状规划(shooting 法)。

定位(方向1 认证后的下一步, 实物形状控制的核心):
  方向1(eval_horizon.py)已证明 open_loop 前向模型可作"规划级仿真器"(drift 1.7x @300步,
  K_max ~50-120步)。本脚本用该仿真器做**逆规划**: 给定初始形状 s_init + 目标形状 s_target,
  优化一段 K 步动作序列, 使 rollout 从 s_init 到达 s_target。

方法 = 可微 shooting(梯度法 MPC):
  - 动作序列 a_{1..K} 设为 requires_grad;
  - 拼 history(产生 s_init 的真实动作) + a_planned 成 buffer;
  - K 步 rollout 喂前向模型(带梯度), 末态 s_K 与 s_target 算到达 loss;
  - backprop 进 a, Adam 步, 投影 a 到真实动作范围;
  - 多起点(零/末动作重复/线性插值/随机)取最优, 缓解 IK 非凸。

变长 K(本版核心, 修"固定 40 步中间反复横跳"):
  模型单步位移上界 = delta_scale_max(=1.0)·pc_scale, 实测末端 ~4px/步。故首末差距 gap_px
  决定**最少需要步数** K ≈ ceil(gap / step_budget)。--auto_k 据此选 K(夹到 [k_min,k_max]),
  小差距→少步数→无多余中间步可横跳。--k_sweep 扫一组 K 实测"reach_px vs K", 经验确认最少步数。
  另加**路径 loss**(全程平均 err): 尽早到达并保持, 消灭残留 wandering。

Loss = w_reach·‖s_K-target‖² + w_path·mean_k(err_k) + w_mono·(err 不准上升) + w_smooth·Σ‖Δa‖²
       + (--obstacle 时) w_obs·避障(每步)

验证基线:
  - do-nothing: a = 重复末动作 → 不应到目标(对照)。
  - GT-actions: rollout 真实 actions[t_a..t_a+K] → 模型对真实轨迹的保真度(上界参考)。
  - planner: 优化后的 a → 应显著优于 do-nothing, 接近 GT-actions。

Usage:
  # 固定 K(旧行为)
  CUDA_VISIBLE_DEVICES=0 python scripts/control/inverse_plan.py \\
      --checkpoint train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt \\
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val \\
      --t_init 500 --t_target 900 --K 40 --n_iter 400 --out output/inverse_plan

  # 变长 K(推荐): 据首末差距自动选步数
  ... --auto_k --step_budget_px 4 --k_min 4 --k_max 40

  # K 扫描: 看 reach_px 随 K 变化, 找最少足够步数
  ... --auto_k --k_sweep
"""

import os
import sys
import math
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
# (CJK 统一表意文字中日韩共用, JP 变体也能渲染中文)。addfont 确保注册 + 用 JP 名, 避免方框。
from matplotlib import font_manager as _fm
for _p in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',):
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from src.utils.model_loader import load_model


# ── 梯度友好的动作窗口切片(保留 requires_grad) ──
def window_torch(buf, t, w):
    """buf[t-w+1..t], 前向 zero-pad, 保留梯度。buf: (L, D)。"""
    start = t - w + 1
    if start >= 0:
        return buf[start:t + 1]
    pad = torch.zeros((-start, buf.shape[1]), device=buf.device, dtype=buf.dtype)
    return torch.cat([pad, buf[0:t + 1]], 0)


def plan_rollout(model, buffer_t, t_start, K, window_size, s_init):
    """从 s_init 出发, 把 buffer_t[t_start+1..t_start+K] 喂模型 K 步 rollout(带梯度)。

    buffer_t: (L, D) 含 history(const) + planned(requires_grad)。
    t_start: s_init 对应动作在 buffer 的索引。
    s_init: (1, N, 3) 归一化种子。
    返回 preds (K, N, 3) 归一化, 带梯度。
    """
    s = s_init
    s_prev = s_init
    aw0 = window_torch(buffer_t, t_start, window_size).unsqueeze(0)
    z = model.init_z_from_action(aw0)
    preds = []
    for k in range(1, K + 1):
        aw = window_torch(buffer_t, t_start + k, window_size).unsqueeze(0)
        out = model.forward(aw, s, s_prev, z)
        s_pred = out["skeleton"]
        z = out["latent_z"]
        preds.append(s_pred.squeeze(0))
        s_prev = s
        s = s_pred
    return torch.stack(preds, 0)   # (K,N,3)


def rollout_eval(model, buffer_t, t_start, K, window_size, s_init):
    """no_grad rollout, 返回 preds (K,N,3) normalized(验证用)。"""
    with torch.no_grad():
        return plan_rollout(model, buffer_t, t_start, K, window_size, s_init)


def obstacle_loss(preds_norm, pc_center, pc_scale, obs_list):
    """避障惩罚: preds (K,N,3) 归一化 → px, 对每个 keep-out 圆(cxcy,r)罚穿透。
    obs_list: [(cx, cy, r_px), ...] in px (col,row)。

    ⚠️ 聚合口径 2026-07-28 从"对 k 求和"改为 mean-over-(K,N)(与工作台统一):
       同一 w_obs 的避障压强不再随 K 线性漂移,与 --auto_k 兼容。
       docs/reports/2026-07-14/15 中含障碍的 planner 数字与本实现不可比
       (该报告本来就含不可复现的随机重启分量 —— CLI 无 torch.manual_seed)。
    """
    from real_validation.obstacles import cli_obstacle_loss
    return cli_obstacle_loss(preds_norm, pc_center, pc_scale, obs_list)


def optimize_plan(model, history_t, s_init, s_target, K, window_size, a_lo, a_hi,
                  device, n_iter=400, lr=0.05, w_reach=1.0, w_smooth=0.01, w_mono=1.0,
                  w_path=0.0,
                  obs_list=None, pc_center=None, pc_scale=None, w_obs=1.0,
                  init_kind="zero", seed_last=None):
    """单起点 shooting 优化。返回 (a_opt(K,D) numpy, loss 曲线, 末态 reach loss)。"""
    D = history_t.shape[1]
    if init_kind == "zero" or seed_last is None:
        a0 = torch.zeros(K, D, device=device)
    elif init_kind == "repeat":
        a0 = seed_last.repeat(K, 1)
    elif init_kind == "interp" and seed_last is not None:
        a0 = torch.linspace(0, 1, K, device=device).unsqueeze(1) * (a_hi - seed_last) + seed_last
    else:
        a0 = torch.rand(K, D, device=device) * (a_hi - a_lo) + a_lo
    a = a0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([a], lr=lr)
    t_start = history_t.shape[0] - 1
    losses = []
    for it in range(n_iter):
        opt.zero_grad()
        buffer_t = torch.cat([history_t, a], 0)
        preds = plan_rollout(model, buffer_t, t_start, K, window_size, s_init)
        # errs: 每步到目标的 MSE (K,)
        errs = ((preds - s_target.squeeze(0)) ** 2).mean(dim=(1, 2))
        # ① 末态必须到达(主目标)
        L_terminal = errs[-1]
        # ② 路径: 最小化全程平均 err → 尽早到达并保持(消灭中间 wandering/横跳, 等效变长K)
        L_path = errs.mean()
        # ③ 单调性: 惩罚 errs 逐差上升(冗余双保险, 到达后不准跑开)
        L_mono = (torch.relu(errs[1:] - errs[:-1]) ** 2).mean()
        L_smooth = ((a[1:] - a[:-1]) ** 2).mean() if K > 1 else torch.zeros((), device=device)
        L = (w_reach * L_terminal + w_path * L_path
             + w_mono * L_mono + w_smooth * L_smooth)
        if obs_list:
            L = L + w_obs * obstacle_loss(preds, pc_center, pc_scale, obs_list)
        L.backward()
        torch.nn.utils.clip_grad_norm_([a], 1.0)
        opt.step()
        with torch.no_grad():
            a.clamp_(a_lo, a_hi)
        losses.append(errs[-1].item())   # 记录末态MSE(multi-start 选最小 = 最佳到达)
    return a.detach().cpu().numpy(), losses, losses[-1]


def node_err_px(pred_norm, gt_norm, pc_center, pc_scale):
    """pred/gt (N,3) 归一化 → 平均节点 L2 px + 末端(node0) L2 px。"""
    p = pred_norm * pc_scale + pc_center
    g = gt_norm * pc_scale + pc_center
    d = np.sqrt(((p[:, :2] - g[:, :2]) ** 2).sum(-1))  # (N,)
    return float(d.mean()), float(d[0])   # (mean_node_px, tip_px)


# ── 变长 K 辅助 ──
def tip_gap_px(s_init, s_target, pc_center, pc_scale):
    """init→target 末端(node0) px 距离 + 平均节点 px 距离。s_*: (1,N,3) 归一化。"""
    p = s_init.squeeze(0).cpu().numpy() * pc_scale + pc_center
    g = s_target.squeeze(0).cpu().numpy() * pc_scale + pc_center
    tip = float(np.hypot(*(p[0, :2] - g[0, :2])))
    mean = float(np.sqrt(((p[:, :2] - g[:, :2]) ** 2).sum(-1)).mean())
    return tip, mean


def select_k_by_gap(gap_tip_px, step_budget_px, k_min, k_max):
    """gap → K: 模型单步最大末端位移 ≈ delta_scale_max·pc_scale ≈ step_budget_px(px)。
    K = ceil(gap / step_budget), 夹到 [k_min, k_max]。步数与首末差距成正比, 小差距→少步数。"""
    k = int(math.ceil(gap_tip_px / max(step_budget_px, 1e-6)))
    return max(k_min, min(k_max, k))


def plan_multistart(model, history_t, s_init, s_target, K, window_size, a_lo, a_hi,
                    device, n_iter, lr, w_smooth, w_path, obs_list,
                    pc_center_t, pc_scale_t, w_obs, n_restarts, seed_last):
    """多起点 shooting: zero/repeat/interp/random, 取末态 reach 最小者。返回 best dict。"""
    inits = ["zero", "repeat", "interp"] + ["random"] * max(0, n_restarts - 3)
    best = None
    for ik in inits[:n_restarts]:
        a_opt, losses, final_reach = optimize_plan(
            model, history_t, s_init, s_target, K, window_size, a_lo, a_hi, device,
            n_iter=n_iter, lr=lr, w_smooth=w_smooth, w_path=w_path,
            obs_list=obs_list, pc_center=pc_center_t, pc_scale=pc_scale_t, w_obs=w_obs,
            init_kind=ik, seed_last=seed_last)
        if best is None or final_reach < best["final_reach"]:
            best = {"a": a_opt, "losses": losses, "final_reach": final_reach, "init": ik}
        print(f"    [restart {ik}] reach(MSE)={final_reach:.3e}")
    print(f"    → 最优起点: {best['init']}, reach={best['final_reach']:.3e}")
    return best


def evaluate_plan(model, history_t, s_init, s_target, a_plan_np, K, window_size,
                  actions_norm, t_a, seed_last, pc_center, pc_scale, device):
    """planner / do-nothing / GT-actions 三条 rollout 验证。
    返回 dict: preds_plan/preds_do/preds_gt (K,N,3)归一化 + 四组 px 误差 + tgt/init np。"""
    t_start = history_t.shape[0] - 1
    a_plan_t = torch.from_numpy(a_plan_np).float().to(device)
    preds_plan = rollout_eval(model, torch.cat([history_t, a_plan_t], 0),
                              t_start, K, window_size, s_init).cpu().numpy()
    a_do = seed_last.repeat(K, 1)
    preds_do = rollout_eval(model, torch.cat([history_t, a_do], 0),
                            t_start, K, window_size, s_init).cpu().numpy()
    n_gt = min(K, actions_norm.shape[0] - (t_a + 1))
    gt_act = torch.from_numpy(actions_norm[t_a + 1:t_a + 1 + n_gt]).float().to(device)
    if n_gt < K:   # 序列不足 K 步: 末动作补齐, 仍可 rollout K 步做保真参考
        gt_act = torch.cat([gt_act, seed_last.repeat(K - n_gt, 1)], 0)
    preds_gt = rollout_eval(model, torch.cat([history_t, gt_act], 0),
                            t_start, K, window_size, s_init).cpu().numpy()
    tgt_np = s_target.squeeze(0).cpu().numpy()
    init_np = s_init.squeeze(0).cpu().numpy()
    plan_mean, plan_tip = node_err_px(preds_plan[-1], tgt_np, pc_center, pc_scale)
    do_mean, do_tip = node_err_px(preds_do[-1], tgt_np, pc_center, pc_scale)
    gt_mean, gt_tip = node_err_px(preds_gt[-1], tgt_np, pc_center, pc_scale)
    init_mean, init_tip = node_err_px(init_np, tgt_np, pc_center, pc_scale)
    return dict(preds_plan=preds_plan, preds_do=preds_do, preds_gt=preds_gt,
                plan_mean=plan_mean, plan_tip=plan_tip, do_mean=do_mean, do_tip=do_tip,
                gt_mean=gt_mean, gt_tip=gt_tip, init_mean=init_mean, init_tip=init_tip,
                tgt_np=tgt_np, init_np=init_np)


def main():
    parser = argparse.ArgumentParser(description="方向2: 可微逆运动学形状规划")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--t_init", type=int, default=500, help="s_init 帧索引")
    parser.add_argument("--t_target", type=int, default=900, help="s_target 帧索引")
    parser.add_argument("--K", type=int, default=40, help="规划步数(固定K模式; ≤ K_max ≈ 50-120)")
    parser.add_argument("--n_iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--w_smooth", type=float, default=0.01)
    parser.add_argument("--w_path", type=float, default=0.5,
                        help="路径 loss 权重(全程平均err, 消灭中间 wandering); 0=关闭")
    parser.add_argument("--n_restarts", type=int, default=4)
    # 变长 K
    parser.add_argument("--auto_k", action="store_true",
                        help="据首末差距自动选 K(=ceil(gap/step_budget), 夹[k_min,k_max])")
    parser.add_argument("--step_budget_px", type=float, default=4.0,
                        help="每步末端位移预算(px)≈delta_scale_max·pc_scale; 越大→K越小")
    parser.add_argument("--k_min", type=int, default=4)
    parser.add_argument("--k_max", type=int, default=40)
    parser.add_argument("--k_sweep", action="store_true",
                        help="扫一组 K, 输出 reach_px vs K 表(找最少足够步数); 需配合 --auto_k")
    # 避障
    parser.add_argument("--obstacle", type=str, default=None,
                        help="避障圆 'cx,cy,r_px'(px, col-row); 多个用 | 分隔")
    parser.add_argument("--w_obs", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="output/inverse_plan")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info["model"]
    # 必须用 train() 模式: planning 要对空间 nn.GRU 做 backward(cuDNN RNN backward 仅
    # train 模式可用)。本模型无 Dropout/BN, train 与 eval 的 forward 完全等价。
    model.train()
    window_size = info["window_size"]
    norm_factor = info["norm_factor"]

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    raw = np.load(files[0])
    actions = raw["actions"].astype(np.float32)       # (T, D)
    positions = raw["positions"].astype(np.float32)   # (T, 3, N)
    T = positions.shape[0]
    actions_norm = actions / norm_factor
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()
    D = actions.shape[1]

    t_a, t_b = args.t_init, args.t_target
    assert 0 <= t_a < T and 0 <= t_b < T, f"t_init/t_target 超出 T={T}"

    def to_norm_t(pos_3N):
        skel = pos_3N.T.astype(np.float32)
        return torch.from_numpy((skel - pc_center) / pc_scale).float().unsqueeze(0).to(device)

    s_init = to_norm_t(positions[t_a])
    s_target = to_norm_t(positions[t_b])

    # 动作范围(投影用) + history(产生 s_init 的真实动作窗口)
    a_lo = torch.from_numpy(actions_norm.min(0)).float().to(device)
    a_hi = torch.from_numpy(actions_norm.max(0)).float().to(device)
    hist_len = max(window_size, 2)
    history_np = actions_norm[max(0, t_a - hist_len + 1):t_a + 1]
    if history_np.shape[0] < hist_len:
        pad = np.zeros((hist_len - history_np.shape[0], D), dtype=np.float32)
        history_np = np.concatenate([pad, history_np], 0)
    history_t = torch.from_numpy(history_np).float().to(device)
    seed_last = history_t[-1:].clone()

    obs_list = None
    if args.obstacle:
        obs_list = []
        for tok in args.obstacle.split("|"):
            cx, cy, r = [float(x) for x in tok.split(",")]
            obs_list.append((cx, cy, r))

    # ── 首末差距 + K 选择 ──
    gap_tip, gap_mean = tip_gap_px(s_init, s_target, pc_center, pc_scale)
    if args.auto_k:
        K_fixed = select_k_by_gap(gap_tip, args.step_budget_px, args.k_min, args.k_max)
        print(f"\n{'='*60}\n方向2 逆规划(变长K): t_init={t_a} → t_target={t_b}")
        print(f"  gap_tip={gap_tip:.1f}px  gap_mean={gap_mean:.1f}px  "
              f"step_budget={args.step_budget_px}px → K_hint={K_fixed}")
    else:
        K_fixed = args.K
        print(f"\n{'='*60}\n方向2 逆规划(固定K): t_init={t_a} → t_target={t_b}, K={K_fixed}步 "
              f"(gap_tip {gap_tip:.1f}px)")
    assert t_a + K_fixed <= T, f"t_init+K={t_a + K_fixed} > T={T}; 减小 K/k_max 或 t_init"

    pc_center_t = model.pc_center.view(3).to(device)
    pc_scale_t = model.pc_scale.view(3).to(device)

    def _plan_one(Kk):
        best = plan_multistart(model, history_t, s_init, s_target, Kk, window_size,
                               a_lo, a_hi, device, args.n_iter, args.lr, args.w_smooth,
                               args.w_path, obs_list, pc_center_t, pc_scale_t, args.w_obs,
                               args.n_restarts, seed_last)
        ev = evaluate_plan(model, history_t, s_init, s_target, best["a"], Kk, window_size,
                           actions_norm, t_a, seed_last, pc_center, pc_scale, device)
        return best, ev

    # ── K 扫描 or 单 K ──
    sweep_rows = []
    if args.k_sweep:
        ks = sorted(set([args.k_min, K_fixed, min(2 * K_fixed, args.k_max), args.k_max]
                        + [k for k in (8, 12, 20) if args.k_min <= k <= args.k_max]))
        print(f"  K 扫描: {ks}")
        results_by_k = {}
        for Kk in ks:
            print(f"\n  -- K={Kk} --")
            best, ev = _plan_one(Kk)
            results_by_k[Kk] = (best, ev)
            sweep_rows.append(dict(K=Kk, init_mean=ev["init_mean"], do_mean=ev["do_mean"],
                                   gt_mean=ev["gt_mean"], plan_mean=ev["plan_mean"],
                                   plan_tip=ev["plan_tip"], best_init=best["init"]))
            print(f"     plan={ev['plan_mean']:.1f}px(tip {ev['plan_tip']:.1f})  "
                  f"do={ev['do_mean']:.1f}  GT={ev['gt_mean']:.1f}  init={ev['init_mean']:.1f}")
        # 选"最少足够步数": reach_plan 在最优值 5%+0.5px 容差内的最小 K
        best_plan = min(r["plan_mean"] for r in sweep_rows)
        tol = best_plan * 1.05 + 0.5
        sufficient = [r for r in sweep_rows if r["plan_mean"] <= tol]
        chosen_K = min(sufficient, key=lambda r: r["K"])["K"]
        best, ev = results_by_k[chosen_K]
        K = chosen_K
        print(f"\n  → 最少足够 K = {chosen_K} (reach {ev['plan_mean']:.1f}px, "
              f"容差 {tol:.1f}px); 全程最优 reach={best_plan:.1f}px")
    else:
        K = K_fixed
        print(f"\n  -- K={K} --")
        best, ev = _plan_one(K)

    a_plan_np = best["a"]
    plan_mean, plan_tip = ev["plan_mean"], ev["plan_tip"]
    do_mean, do_tip = ev["do_mean"], ev["do_tip"]
    gt_mean, gt_tip = ev["gt_mean"], ev["gt_tip"]
    init_mean, init_tip = ev["init_mean"], ev["init_tip"]

    print(f"\n{'='*60}\n验证(末态 vs s_target, 平均节点px / 末端px)  [选用 K={K}]")
    print(f"  初始差距 s_init→s_target : {init_mean:6.2f}px (tip {init_tip:.2f})")
    print(f"  do-nothing(重复末动作)   : {do_mean:6.2f}px (tip {do_tip:.2f})  [对照, 应≈初始]")
    print(f"  GT-actions(真实动作rollout): {gt_mean:6.2f}px (tip {gt_tip:.2f})  [模型保真上界]")
    print(f"  planner(优化动作)        : {plan_mean:6.2f}px (tip {plan_tip:.2f})  [本方法]")
    verdict = "✓ 成功" if plan_mean < do_mean * 0.7 else ("≈ 部分成功" if plan_mean < do_mean else "✗ 失败")
    print(f"  → {verdict}: planner {'优于' if plan_mean < do_mean else '不优于'} do-nothing"
          f" ({plan_mean / max(do_mean, 1e-9):.2f}×)")

    result = {
        "checkpoint": args.checkpoint,
        "t_init": t_a, "t_target": t_b, "K": K,
        "auto_k": args.auto_k, "step_budget_px": args.step_budget_px,
        "gap_tip_px": gap_tip, "gap_mean_px": gap_mean,
        "best_init": best["init"],
        "init_gap_px": {"mean": init_mean, "tip": init_tip},
        "do_nothing_px": {"mean": do_mean, "tip": do_tip},
        "gt_actions_px": {"mean": gt_mean, "tip": gt_tip},
        "planner_px": {"mean": plan_mean, "tip": plan_tip},
        "loss_curve": best["losses"],
        "a_plan": a_plan_np.tolist(),
        "k_sweep": sweep_rows,
    }
    with open(os.path.join(args.out, "plan_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── 可视化: init→轨迹→target 叠加 ──
    preds_plan = ev["preds_plan"]
    tgt_np, init_np = ev["tgt_np"], ev["init_np"]
    fig, ax = plt.subplots(figsize=(8, 10))
    cmap = plt.cm.viridis
    for k in range(K):
        p = preds_plan[k] * pc_scale + pc_center
        ax.plot(p[:, 0], p[:, 1], "-", color=cmap(k / max(K - 1, 1)), alpha=0.5, lw=1.0)
    pi = init_np * pc_scale + pc_center
    pt = tgt_np * pc_scale + pc_center
    pp = preds_plan[-1] * pc_scale + pc_center
    ax.plot(pi[:, 0], pi[:, 1], "o-", color="blue", lw=2, ms=4, label="s_init")
    ax.plot(pt[:, 0], pt[:, 1], "s-", color="red", lw=2, ms=5, label="s_target")
    ax.plot(pp[:, 0], pp[:, 1], "^-", color="green", lw=2, ms=5, label="planner 末态")
    if obs_list:
        for (cx, cy, r) in obs_list:
            ax.add_patch(plt.Circle((cx, cy), r, color="orange", fill=False, lw=2, ls="--"))
    ax.invert_yaxis()
    ax.set_aspect("auto")   # col范围小 vs row大; auto 让横轴可读(预测/GT同拉伸, 相对形变不失真)
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ktag = f"auto K={K} (gap {gap_tip:.0f}px/{args.step_budget_px:.0f}px/步)" if args.auto_k else f"K={K}"
    ax.set_title(f"Inverse plan: t{t_a}→t{t_b}, {ktag}\n"
                 f"planner {plan_mean:.1f}px vs do-nothing {do_mean:.1f}px "
                 f"(GT-actions {gt_mean:.1f}px)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plan_trajectory.png"), dpi=130)
    plt.close()

    # ── K 扫描表图(若 sweep) ──
    if sweep_rows:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ks = [r["K"] for r in sweep_rows]
        ax.plot(ks, [r["plan_mean"] for r in sweep_rows], "o-", color="navy", lw=2, ms=7, label="planner reach")
        ax.plot(ks, [r["gt_mean"] for r in sweep_rows], "s--", color="darkgreen", lw=1.5, ms=6, label="GT-actions")
        ax.plot(ks, [r["do_mean"] for r in sweep_rows], "^:", color="darkorange", lw=1.5, ms=6, label="do-nothing")
        ax.axhline(init_mean, color="gray", ls=":", alpha=0.7, label=f"init gap {init_mean:.1f}px")
        ax.axvline(K, color="red", ls="--", alpha=0.5, label=f"选用 K={K}")
        ax.set_xlabel("规划步数 K"); ax.set_ylabel("末态误差 (px)")
        ax.set_title(f"reach_px vs K (gap_tip {gap_tip:.0f}px): 最少足够步数 = {K}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args.out, "plan_k_sweep.png"), dpi=130); plt.close()

    print(f"\n已保存: {args.out}/plan_result.json, plan_trajectory.png"
          + (", plan_k_sweep.png" if sweep_rows else ""))
    print("\n解读:")
    print("  - planner px << do-nothing px → 逆规划成功(模型作仿真器找到有效动作序列)")
    print("  - planner ≈ GT-actions → 接近模型保真上界(最优)")
    print("  - 变长K: K 与 gap 成正比, 消灭固定大K的中间横跳; k_sweep 经验确认最少步数")


if __name__ == "__main__":
    main()
