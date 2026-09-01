"""eval_horizon.py — 方向1: 纯自回归 rollout 视野认证(找最大可用 K)。

定位(为方向2 逆运动学规划服务):
  方向2(逆规划)把候选动作序列喂前向模型 rollout 来评估——若模型长程漂移, 规划动作
  无法迁移到真机。故本脚本认证前向模型能否当"规划级仿真器", 给出可信视野上限 K_max,
  作为方向2规划视野的硬约束。同时监测 z(无GT迟滞潜变量)长程是否发散——z 失稳则整个
  控制议程建立在不稳定动力学上。

做什么:
  - 多种子(从 val 不同帧起 rollout)聚合 error-by-k 曲线(均值), 统计稳健;
  - 双轨误差: 归一化空间 MSE + 模型状态空间平均节点 L2;
  - K_max 在多容差(相对 onestep 的 3/10/30× + 状态单位绝对阈值)的取值;
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
from src.data.action_view import project_actions, resolve_action_contract
from src.evaluation.diameter_scale import (
    DEFAULT_ROBOT_DIAMETER_MM, resolve_diameter_scale)
from src.registry.paths import ProjectPaths
from src.registry.runs import create_analysis_run


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


def state_node_err(pred_norm, gt_norm, pc_center, pc_scale):
    """模型状态平面中的平均节点 L2；单位由 NPZ 合同声明。"""
    p = pred_norm * pc_scale + pc_center
    g = gt_norm * pc_scale + pc_center
    d = np.sqrt(((p[..., :2] - g[..., :2]) ** 2).sum(-1))  # (K,N)
    return d.mean(axis=1)


px_node_err = state_node_err


def certified_k(curve, threshold):
    """返回测试范围内连续满足 ``curve <= threshold`` 的最大步数。"""
    values = np.asarray(curve, dtype=float)
    hits = np.where(values > float(threshold))[0]
    return int(hits[0]) if len(hits) else int(len(values))


def characterize(model, ckpt, data_dir, max_steps, n_seeds, window_size,
                 norm_factor, action_dim, device, robot_diameter_mm,
                 robot_diameter_px):
    """对一个 checkpoint 在多种子上聚合 error-by-k, 返回 summary + by_k。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    raw = np.load(files[0])
    state_frame = (str(raw["state_coordinate_frame"].item())
                   if "state_coordinate_frame" in raw else "camera_pixel_v1")
    state_unit = (str(raw["state_length_unit"].item())
                  if "state_length_unit" in raw else "px")
    diameter_scale = resolve_diameter_scale(
        raw, data_dir, diameter_mm=robot_diameter_mm,
        diameter_px=robot_diameter_px)
    raw_actions = raw["actions"].astype(np.float32)
    action_contract = resolve_action_contract(data_dir, "auto")
    actions = project_actions(
        raw_actions, action_contract.model_action_channels).astype(np.float32)
    if actions.shape[1] != int(action_dim):
        raise ValueError(
            f"视野认证动作视图{actions.shape[1]}D与checkpoint {action_dim}D不一致")
    positions = raw["positions"].astype(np.float32)  # (T,3,N)
    T = positions.shape[0]
    actions_norm = actions / norm_factor
    pc_center = model.pc_center.view(3).cpu().numpy()
    pc_scale = model.pc_scale.view(3).cpu().numpy()

    seeds = np.linspace(1, max(2, T - max_steps - 2), n_seeds, dtype=int)
    K = min(max_steps, T - 2)
    roll_mse = np.zeros(K)
    one_mse = np.zeros(K)
    roll_state = np.zeros(K)
    z_n = np.zeros(K)
    cnt = 0
    for t0 in seeds:
        r, o, g, zn = rollout_horizon(model, actions_norm, positions, int(t0),
                                      max_steps, window_size, device)
        kk = r.shape[0]
        roll_mse[:kk] += ((r - g) ** 2).mean(axis=(1, 2))
        one_mse[:kk] += ((o - g) ** 2).mean(axis=(1, 2))
        roll_state[:kk] += state_node_err(r, g, pc_center, pc_scale)
        z_n[:kk] += zn
        cnt += 1
    roll_mse /= cnt
    one_mse /= cnt
    roll_state /= cnt
    z_n /= cnt
    drift = roll_mse / np.maximum(one_mse, 1e-8)
    roll_mm = (roll_state if state_unit == "mm" else
               roll_state * diameter_scale.mm_per_px)
    absolute_thresholds = (2.0, 4.0, 8.0, 16.0) if state_unit == "mm" else \
        (3.0, 5.0, 10.0, 20.0)

    summary = {
        "metric_semantics": (
            "forward-model rollout prediction versus recorded skeleton GT "
            "under the recorded dataset action sequence"),
        "aggregation": "per exact rollout step k across evaluation seeds and all skeleton nodes",
        "checkpoint": ckpt,
        "n_seeds": int(cnt),
        "raw_action_dim": int(raw_actions.shape[1]),
        "model_action_dim": int(actions.shape[1]),
        "model_action_channels": list(action_contract.model_action_channels),
        "K_evaluated": int(K),
        "rollout_mse_final": float(roll_mse[-1]),
        "onestep_mse_final": float(one_mse[-1]),
        "drift_final_x": float(drift[-1]),
        "state_coordinate_frame": state_frame,
        "state_length_unit": state_unit,
        "roll_state_final": float(roll_state[-1]),
        "roll_mm_final": float(roll_mm[-1]),
        "robot_diameter_mm": float(diameter_scale.diameter_mm),
        "robot_diameter_px": float(diameter_scale.diameter_px),
        "mm_per_px": float(diameter_scale.mm_per_px),
        "diameter_scale_source": diameter_scale.source,
        "z_norm_start": float(z_n[0]),
        "z_norm_final": float(z_n[-1]),
        "Kmax_drift_3x": certified_k(drift, 3.0),
        "Kmax_drift_10x": certified_k(drift, 10.0),
        "Kmax_drift_30x": certified_k(drift, 30.0),
        "Kmax_mm_2": certified_k(roll_mm, 2.0),
        "Kmax_mm_4": certified_k(roll_mm, 4.0),
        "Kmax_mm_8": certified_k(roll_mm, 8.0),
        "Kmax_mm_16": certified_k(roll_mm, 16.0),
    }
    summary["Kmax_state"] = {
        f"{threshold:g}{state_unit}": certified_k(roll_state, threshold)
        for threshold in absolute_thresholds
    }
    if state_unit == "px":
        summary.update({
            "Kmax_px_3": certified_k(roll_state, 3.0),
            "Kmax_px_5": certified_k(roll_state, 5.0),
            "Kmax_px_10": certified_k(roll_state, 10.0),
            "Kmax_px_20": certified_k(roll_state, 20.0),
        })
    by_k = {
        "rollout_mse": roll_mse.tolist(),
        "onestep_mse": one_mse.tolist(),
        "drift": drift.tolist(),
        "roll_state": roll_state.tolist(),
        "state_unit": state_unit,
        "roll_mm": roll_mm.tolist(),
        "z_norm": z_n.tolist(),
    }
    return summary, by_k


def plot_comparison(all_by_k, summaries, out_path, max_steps):
    """3 子图: 状态误差 / drift ratio / z_norm。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.tab10.colors

    for i, (label, bk) in enumerate(all_by_k.items()):
        c = colors[i % len(colors)]
        k = np.arange(1, len(bk["roll_state"]) + 1)
        axes[0].plot(k, bk["roll_state"], label=label, color=c, lw=1.5)
        axes[1].plot(k, bk["drift"], label=label, color=c, lw=1.5)
        axes[2].plot(k, bk["z_norm"], label=label, color=c, lw=1.5)

    state_unit = summaries[0].get("state_length_unit", "px") if summaries else "px"
    thresholds = (2, 4, 8, 16) if state_unit == "mm" else (3, 5, 10, 20)
    for thr in thresholds:
        axes[0].axhline(thr, color="gray", ls=":", lw=0.8, alpha=0.7)
        axes[0].text(max_steps * 0.99, thr, f"{thr}{state_unit}", fontsize=7,
                     color="gray", ha="right", va="bottom")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("rollout step k (距上次观测的步数)")
    axes[0].set_ylabel(f"平均节点误差 ({state_unit})")
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
    parser.add_argument("--out", type=str, default=None,
                        help="输出目录(JSON + PNG)；默认分配新的 workspace analysis run")
    parser.add_argument("--robot-diameter-mm", type=float,
                        default=DEFAULT_ROBOT_DIAMETER_MM)
    parser.add_argument("--robot-diameter-px", type=float, default=None,
                        help="可选像素直径覆盖值；默认读取NPZ或同序列骨架QC")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.out is None:
        args.out = str(create_analysis_run(ProjectPaths.load(), "horizon"))
    os.makedirs(args.out, exist_ok=True)

    all_by_k = {}
    all_summaries = []
    for ckpt in args.checkpoints:
        print(f"\n{'='*60}\n认证: {ckpt}")
        info = load_model(ckpt, data_dir=args.data_dir, device=device)
        first_npz = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))[0]
        with np.load(first_npz, allow_pickle=False) as contract_npz:
            data_frame = (str(contract_npz["state_coordinate_frame"].item())
                          if "state_coordinate_frame" in contract_npz else
                          "camera_pixel_v1")
        checkpoint_frame = (info.get("saved_config") or {}).get(
            "state_view", {}).get("state_coordinate_frame", "camera_pixel_v1")
        if checkpoint_frame != data_frame:
            raise ValueError(
                f"checkpoint state_coordinate_frame={checkpoint_frame} 与数据 {data_frame} 不一致")
        model = info["model"]
        model.eval()
        window_size = info["window_size"]
        norm_factor = info["norm_factor"]
        mtype = ("open_loop" if getattr(model, "open_loop_mode", None) is not None
                 else "gt" if getattr(model, "gt_observed_mode", None) is not None
                 else "state_transition")
        label = f"{mtype}"

        summary, by_k = characterize(model, ckpt, args.data_dir, args.max_steps,
                                     args.n_seeds, window_size, norm_factor,
                                     info["action_dim"], device,
                                     args.robot_diameter_mm,
                                     args.robot_diameter_px)
        summary["model_type"] = mtype
        all_by_k[label] = by_k
        all_summaries.append(summary)

        print(f"  [{label}] 前向模型rollout预测误差; "
              f"n_seeds={summary['n_seeds']}, K={summary['K_evaluated']}")
        print(f"  最终: rollout_mse={summary['rollout_mse_final']:.3e}, "
              f"onestep_mse={summary['onestep_mse_final']:.3e}, "
              f"drift={summary['drift_final_x']:.1f}x, "
              f"error={summary['roll_state_final']:.2f}{summary['state_length_unit']}"
              f"≈{summary['roll_mm_final']:.2f}mm")
        print(f"  直径尺度: {summary['robot_diameter_mm']:.2f}mm / "
              f"{summary['robot_diameter_px']:.2f}px = "
              f"{summary['mm_per_px']:.6f}mm/px")
        print(f"  z_norm: {summary['z_norm_start']:.2f} → {summary['z_norm_final']:.2f}")
        print(f"  K_max(漂移比): 3x={summary['Kmax_drift_3x']}, "
              f"10x={summary['Kmax_drift_10x']}, 30x={summary['Kmax_drift_30x']}")
        print(f"  K_max(绝对{summary['state_length_unit']}): " + ", ".join(
            f"{label}={value}" for label, value in summary["Kmax_state"].items()))
        print(f"  K_max(估计mm): 2mm={summary['Kmax_mm_2']}, "
              f"4mm={summary['Kmax_mm_4']}, 8mm={summary['Kmax_mm_8']}, "
              f"16mm={summary['Kmax_mm_16']}")

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
    print("  - 状态误差曲线缓增且 drift<10x：按任务容差交叉点选择 K_max")
    print("  - 状态误差快速增长或 z_norm 发散：缩短规划视野并调整 OpenLoop 训练")
    print("  - open_loop 的 K_max 应 ≥ gt(open_loop 专为 rollout 训练)")


if __name__ == "__main__":
    main()
