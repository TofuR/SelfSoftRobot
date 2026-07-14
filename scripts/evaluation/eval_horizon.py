"""eval_horizon.py — 方向1: 纯自回归 rollout 视野认证(找最大可用 K)。

定位(为方向2 逆运动学规划服务):
  方向2(逆规划)把候选动作序列喂前向模型 rollout 来评估——若模型长程漂移, 规划动作
  无法迁移到真机。故本脚本认证前向模型能否当"规划级仿真器", 给出可信视野上限 K_max,
  作为方向2规划视野的硬约束。同时监测 z(无GT迟滞潜变量)长程是否发散——z 失稳则整个
  控制议程建立在不稳定动力学上。

做什么:
  - 多种子(从 val 不同帧起 rollout)聚合 error-by-k 曲线(均值), 统计稳健;
  - 双轨误差: 归一化空间 MSE + 物理空间平均节点 L2(px, col-row 平面);
  - K_max 在多容差(相对 onestep 的 3/10/30×, 绝对 px 3/5/10/20)的取值;
  - 可传多个 checkpoint 叠加对比——关键问题: open_loop 训练是否延长可用视野 vs gt;
  - 输出 JSON + 图(error vs k, log-y, 容差线 + drift + z_norm)。

复用: src.evaluation.transition_metrics.build_action_window + 模型 init_z_from_action/forward。

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/eval_horizon.py \\
      --checkpoints train_log/open_loop_transition/exp_20260714_8/phase_open_loop_transition/model/best_model.pt \\
                    train_log/gt_transition/exp_20260714_7/phase_gt_transition/model/best_model.pt \\
      --data_dir data/real_seq/seq_20260627_163921_n15_sam2_clean/val \\
      --max_steps 300 --n_seeds 8 --out output/horizon
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
from src.evaluation.transition_metrics import build_action_window


def rollout_horizon(model, actions_norm, positions, t0, max_k, window_size, device):
    """从 t0 帧 GT 种子纯自回归 rollout max_k 步。

    rollout 路径喂模型自身预测(s+z 自演化); onestep 参考喂 GT(独立 z_tf 轨迹)。
    Returns: roll/one/gt (K,N,3) 归一化, z_norm (K,)。
    """
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()

    def to_norm(pos_3N):
        skel = pos_3N.T.astype(np.float32)
        return torch.from_numpy((skel - pc_center) / pc_scale).float().unsqueeze(0).to(device)

    s_roll = to_norm(positions[t0])
    s_prev_roll = s_roll
    aw0 = build_action_window(actions_norm, t0, window_size)
    z_roll = model.init_z_from_action(
        torch.from_numpy(aw0).float().unsqueeze(0).to(device))
    z_tf = z_roll.clone()

    K = min(max_k, positions.shape[0] - 1 - t0)
    roll, one, zn = [], [], []
    with torch.no_grad():
        for k in range(1, K + 1):
            tt = t0 + k
            aw = torch.from_numpy(
                build_action_window(actions_norm, tt, window_size)
            ).float().unsqueeze(0).to(device)

            # rollout: 喂自身上一步预测
            ro = model.forward(aw, s_roll, s_prev_roll, z_roll)
            s_pred = ro["skeleton"]
            z_roll = ro["latent_z"]
            roll.append(s_pred.squeeze(0).cpu().numpy())
            s_prev_roll = s_roll
            s_roll = s_pred

            # onestep 参考: 喂 GT(干净上界)
            prev_gt = to_norm(positions[tt - 1])
            prev_prev_gt = to_norm(positions[max(tt - 2, 0)])
            oo = model.forward(aw, prev_gt, prev_prev_gt, z_tf)
            z_tf = oo["latent_z"]
            one.append(oo["skeleton"].squeeze(0).cpu().numpy())

            zn.append(z_roll.norm().item())

    gts = np.stack(
        [to_norm(positions[t0 + k]).squeeze(0).cpu().numpy() for k in range(1, K + 1)], 0)
    return np.stack(roll, 0), np.stack(one, 0), gts, np.array(zn)


def px_node_err(pred_norm, gt_norm, pc_center, pc_scale):
    """平均节点 L2(px), col-row 平面(z=0不计)。pred/gt_norm: (K,N,3) 归一化 → (K,)px。"""
    p = pred_norm * pc_scale + pc_center   # (K,N,3) px [col,row,0]
    g = gt_norm * pc_scale + pc_center
    d = np.sqrt(((p[..., :2] - g[..., :2]) ** 2).sum(-1))  # (K,N)
    return d.mean(axis=1)  # (K,) 每步平均节点px


def kmax_above(curve, thr):
    """首个超过 thr 的步数(1-indexed); 全程未超 → None。"""
    hits = np.where(curve > thr)[0]
    return int(hits[0] + 1) if len(hits) > 0 else None


def characterize(model, ckpt, data_dir, max_steps, n_seeds, window_size,
                 norm_factor, device):
    """对一个 checkpoint 在多种子上聚合 error-by-k, 返回 summary + by_k。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    raw = np.load(files[0])
    actions = raw["actions"].astype(np.float32)
    positions = raw["positions"].astype(np.float32)  # (T,3,N)
    T = positions.shape[0]
    actions_norm = actions / norm_factor
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()

    seeds = np.linspace(1, max(2, T - max_steps - 2), n_seeds, dtype=int)
    K = min(max_steps, T - 2)
    roll_mse = np.zeros(K)
    one_mse = np.zeros(K)
    roll_px = np.zeros(K)
    z_n = np.zeros(K)
    cnt = 0
    for t0 in seeds:
        r, o, g, zn = rollout_horizon(model, actions_norm, positions, int(t0),
                                      max_steps, window_size, device)
        kk = r.shape[0]
        roll_mse[:kk] += ((r - g) ** 2).mean(axis=(1, 2))
        one_mse[:kk] += ((o - g) ** 2).mean(axis=(1, 2))
        roll_px[:kk] += px_node_err(r, g, pc_center, pc_scale)
        z_n[:kk] += zn
        cnt += 1
    roll_mse /= cnt
    one_mse /= cnt
    roll_px /= cnt
    z_n /= cnt
    drift = roll_mse / np.maximum(one_mse, 1e-8)

    summary = {
        "checkpoint": ckpt,
        "n_seeds": int(cnt),
        "K_evaluated": int(K),
        "rollout_mse_final": float(roll_mse[-1]),
        "onestep_mse_final": float(one_mse[-1]),
        "drift_final_x": float(drift[-1]),
        "roll_px_final": float(roll_px[-1]),
        "z_norm_start": float(z_n[0]),
        "z_norm_final": float(z_n[-1]),
        "Kmax_drift_3x": kmax_above(drift, 3.0),
        "Kmax_drift_10x": kmax_above(drift, 10.0),
        "Kmax_drift_30x": kmax_above(drift, 30.0),
        "Kmax_px_3": kmax_above(roll_px, 3.0),
        "Kmax_px_5": kmax_above(roll_px, 5.0),
        "Kmax_px_10": kmax_above(roll_px, 10.0),
        "Kmax_px_20": kmax_above(roll_px, 20.0),
    }
    by_k = {
        "rollout_mse": roll_mse.tolist(),
        "onestep_mse": one_mse.tolist(),
        "drift": drift.tolist(),
        "roll_px": roll_px.tolist(),
        "z_norm": z_n.tolist(),
    }
    return summary, by_k


def plot_comparison(all_by_k, summaries, out_path, max_steps):
    """3 子图: rollout px(对数y, 含容差线) / drift ratio / z_norm。每 checkpoint 一条线。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors

    for i, (label, bk) in enumerate(all_by_k.items()):
        c = colors[i % len(colors)]
        k = np.arange(1, len(bk["roll_px"]) + 1)
        axes[0].plot(k, bk["roll_px"], label=label, color=c, lw=1.5)
        axes[1].plot(k, bk["drift"], label=label, color=c, lw=1.5)
        axes[2].plot(k, bk["z_norm"], label=label, color=c, lw=1.5)

    # px 容差线
    for thr in (3, 5, 10, 20):
        axes[0].axhline(thr, color="gray", ls=":", lw=0.8, alpha=0.7)
        axes[0].text(max_steps * 0.99, thr, f"{thr}px", fontsize=7,
                     color="gray", ha="right", va="bottom")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rollout step k (距上次观测的步数)")
    axes[0].set_ylabel("平均节点误差 (px, col-row 平面)")
    axes[0].set_title("纯自回归 rollout 误差 vs 步数")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].axhline(1.0, color="green", ls="--", lw=0.8, alpha=0.7)
    axes[1].axhline(10.0, color="orange", ls=":", lw=0.8, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("rollout step k")
    axes[1].set_ylabel("drift ratio (rollout / onestep)")
    axes[1].set_title("漂移比(>1=误差累积)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].set_xlabel("rollout step k")
    axes[2].set_ylabel("‖z_t‖")
    axes[2].set_title("迟滞潜变量 z 范数轨迹(发散=失稳)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="方向1: 纯自回归 rollout 视野认证")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="一个或多个 best_model.pt(叠加对比; 自动检测 gt/open_loop)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="含 .npz 的数据目录(建议 val)")
    parser.add_argument("--max_steps", type=int, default=300,
                        help="每个种子最长 rollout 步数")
    parser.add_argument("--n_seeds", type=int, default=8,
                        help="种子数(从 val 不同帧起 rollout 后聚合)")
    parser.add_argument("--out", type=str, default="output/horizon",
                        help="输出目录(JSON + PNG)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    all_by_k = {}
    all_summaries = []
    for ckpt in args.checkpoints:
        print(f"\n{'='*60}\n认证: {ckpt}")
        info = load_model(ckpt, data_dir=args.data_dir, device=device)
        model = info["model"]
        model.eval()
        window_size = info["window_size"]
        norm_factor = info["norm_factor"]
        mtype = ("open_loop" if getattr(model, "open_loop_mode", None) is not None
                 else "gt" if getattr(model, "gt_observed_mode", None) is not None
                 else "state_transition")
        label = f"{mtype}"

        summary, by_k = characterize(model, ckpt, args.data_dir, args.max_steps,
                                     args.n_seeds, window_size, norm_factor, device)
        summary["model_type"] = mtype
        all_by_k[label] = by_k
        all_summaries.append(summary)

        print(f"  [{label}] n_seeds={summary['n_seeds']}, K={summary['K_evaluated']}")
        print(f"  最终: rollout_mse={summary['rollout_mse_final']:.3e}, "
              f"onestep_mse={summary['onestep_mse_final']:.3e}, "
              f"drift={summary['drift_final_x']:.1f}x, "
              f"px={summary['roll_px_final']:.2f}px")
        print(f"  z_norm: {summary['z_norm_start']:.2f} → {summary['z_norm_final']:.2f}")
        print(f"  K_max(漂移比): 3x={summary['Kmax_drift_3x']}, "
              f"10x={summary['Kmax_drift_10x']}, 30x={summary['Kmax_drift_30x']}")
        print(f"  K_max(绝对px): 3px={summary['Kmax_px_3']}, "
              f"5px={summary['Kmax_px_5']}, 10px={summary['Kmax_px_10']}, "
              f"20px={summary['Kmax_px_20']}")

    # 保存 JSON + 图
    json_path = os.path.join(args.out, "horizon_summary.json")
    with open(json_path, "w") as f:
        json.dump({"summaries": all_summaries, "by_k": all_by_k}, f, indent=2)
    png_path = os.path.join(args.out, "horizon_comparison.png")
    plot_comparison(all_by_k, all_summaries, png_path, args.max_steps)
    print(f"\n{'='*60}")
    print(f"已保存: {json_path}")
    print(f"已保存: {png_path}")

    # 一句话结论
    print("\n解读:")
    print("  - px 曲线缓增 + drift<10x → 模型可作规划仿真器, K_max 取容差交叉点")
    print("  - px 指数增长 / z_norm 发散 → 模型不可作长程仿真器, 需先改 open_loop 训练")
    print("  - open_loop 的 K_max 应 ≥ gt(open_loop 专为 rollout 训练)")


if __name__ == "__main__":
    main()
