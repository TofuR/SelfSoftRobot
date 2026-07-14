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

Loss = w_reach·‖s_K - s_target‖² + w_smooth·Σ‖Δa‖² + (--obstacle 时) w_obs·避障
步数约束: K ≤ K_max(方向1), 文档化为变长 K 扩展。

验证基线:
  - do-nothing: a = 重复末动作 → 不应到目标(对照)。
  - GT-actions: rollout 真实 actions[t_a..t_a+K] → 模型对真实轨迹的保真度(上界参考)。
  - planner: 优化后的 a → 应显著优于 do-nothing, 接近 GT-actions。

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/control/inverse_plan.py \\
      --checkpoint train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt \\
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val \\
      --t_init 500 --t_target 900 --K 40 --n_iter 400 --out output/inverse_plan
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
    """
    p = preds_norm * pc_scale + pc_center   # (K,N,3)
    loss = preds_norm.new_zeros(())
    for k in range(p.shape[0]):
        for (cx, cy, r) in obs_list:
            d = torch.sqrt((p[k, :, 0] - cx) ** 2 + (p[k, :, 1] - cy) ** 2)  # (N,)
            loss = loss + torch.relu(r - d).pow(2).mean()
    return loss


def optimize_plan(model, history_t, s_init, s_target, K, window_size, a_lo, a_hi,
                  device, n_iter=400, lr=0.05, w_reach=1.0, w_smooth=0.01,
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
        L_reach = ((preds[-1] - s_target.squeeze(0)) ** 2).mean()
        L_smooth = ((a[1:] - a[:-1]) ** 2).mean() if K > 1 else torch.zeros((), device=device)
        L = w_reach * L_reach + w_smooth * L_smooth
        if obs_list:
            L = L + w_obs * obstacle_loss(preds, pc_center, pc_scale, obs_list)
        L.backward()
        torch.nn.utils.clip_grad_norm_([a], 1.0)
        opt.step()
        with torch.no_grad():
            a.clamp_(a_lo, a_hi)
        losses.append(L_reach.item())
    return a.detach().cpu().numpy(), losses, losses[-1]


def node_err_px(pred_norm, gt_norm, pc_center, pc_scale):
    """pred/gt (N,3) 归一化 → 平均节点 L2 px + 末端(node0) L2 px。"""
    p = pred_norm * pc_scale + pc_center
    g = gt_norm * pc_scale + pc_center
    d = np.sqrt(((p[:, :2] - g[:, :2]) ** 2).sum(-1))  # (N,)
    return float(d.mean()), float(d[0])   # (mean_node_px, tip_px)


def main():
    parser = argparse.ArgumentParser(description="方向2: 可微逆运动学形状规划")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--t_init", type=int, default=500, help="s_init 帧索引")
    parser.add_argument("--t_target", type=int, default=900, help="s_target 帧索引")
    parser.add_argument("--K", type=int, default=40, help="规划步数(≤ K_max ≈ 50-120)")
    parser.add_argument("--n_iter", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--w_smooth", type=float, default=0.01)
    parser.add_argument("--n_restarts", type=int, default=4)
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
    K = args.K
    assert t_a + K <= T, f"t_init+K={t_a + K} > T={T}, GT-actions 基线需要; 减小 K 或 t_init"

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

    print(f"\n{'='*60}\n方向2 逆规划: t_init={t_a} → t_target={t_b}, K={K}步")
    print(f"  s_init/s_target 取自真实序列(可GT验证)")

    # ── 多起点优化 ──
    inits = ["zero", "repeat", "interp"] + ["random"] * max(0, args.n_restarts - 3)
    best = None
    for ik in inits[:args.n_restarts]:
        a_opt, losses, final_reach = optimize_plan(
            model, history_t, s_init, s_target, K, window_size, a_lo, a_hi, device,
            n_iter=args.n_iter, lr=args.lr, w_smooth=args.w_smooth,
            obs_list=obs_list, pc_center=model.pc_center.view(3).to(device),
            pc_scale=model.pc_scale.view(3).to(device), w_obs=args.w_obs,
            init_kind=ik, seed_last=seed_last)
        if best is None or final_reach < best["final_reach"]:
            best = {"a": a_opt, "losses": losses, "final_reach": final_reach, "init": ik}
        print(f"  [restart {ik}] final reach loss(normalized MSE)={final_reach:.3e}")
    print(f"  → 最优起点: {best['init']}, reach={best['final_reach']:.3e}")

    a_plan = torch.from_numpy(best["a"]).float().to(device)

    # ── 验证: planner / do-nothing / GT-actions 三条 rollout ──
    t_start = history_t.shape[0] - 1
    buffer_plan = torch.cat([history_t, a_plan], 0)
    preds_plan = rollout_eval(model, buffer_plan, t_start, K, window_size, s_init).cpu().numpy()

    a_do = seed_last.repeat(K, 1)
    buffer_do = torch.cat([history_t, a_do], 0)
    preds_do = rollout_eval(model, buffer_do, t_start, K, window_size, s_init).cpu().numpy()

    gt_act = torch.from_numpy(actions_norm[t_a + 1:t_a + K + 1]).float().to(device)
    buffer_gt = torch.cat([history_t, gt_act], 0)
    preds_gt = rollout_eval(model, buffer_gt, t_start, K, window_size, s_init).cpu().numpy()

    tgt_np = s_target.squeeze(0).cpu().numpy()
    init_np = s_init.squeeze(0).cpu().numpy()

    plan_mean, plan_tip = node_err_px(preds_plan[-1], tgt_np, pc_center, pc_scale)
    do_mean, do_tip = node_err_px(preds_do[-1], tgt_np, pc_center, pc_scale)
    gt_mean, gt_tip = node_err_px(preds_gt[-1], tgt_np, pc_center, pc_scale)
    init_mean, init_tip = node_err_px(init_np, tgt_np, pc_center, pc_scale)

    print(f"\n{'='*60}\n验证(末态 vs s_target, 平均节点px / 末端px):")
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
        "best_init": best["init"],
        "init_gap_px": {"mean": init_mean, "tip": init_tip},
        "do_nothing_px": {"mean": do_mean, "tip": do_tip},
        "gt_actions_px": {"mean": gt_mean, "tip": gt_tip},
        "planner_px": {"mean": plan_mean, "tip": plan_tip},
        "loss_curve": best["losses"],
        "a_plan": best["a"].tolist(),
    }
    with open(os.path.join(args.out, "plan_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ── 可视化: init→轨迹→target 叠加 ──
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
    ax.set_aspect("equal")
    ax.set_xlabel("col (px)"); ax.set_ylabel("row (px)")
    ax.set_title(f"Inverse plan: t{t_a}→t{t_b}, K={K}\n"
                 f"planner {plan_mean:.1f}px vs do-nothing {do_mean:.1f}px "
                 f"(GT-actions {gt_mean:.1f}px)")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "plan_trajectory.png"), dpi=130)
    plt.close()

    print(f"\n已保存: {args.out}/plan_result.json, plan_trajectory.png")
    print(f"\n解读:")
    print("  - planner px << do-nothing px → 逆规划成功(模型作仿真器找到有效动作序列)")
    print("  - planner ≈ GT-actions → 接近模型保真上界(最优)")
    print("  - 若 planner 失败: 检查 K>K_max / 非凸局部最小(增 restarts) / 梯度饱和(tanh)")


if __name__ == "__main__":
    main()
