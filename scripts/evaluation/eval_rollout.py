"""eval_rollout.py — StateTransitionSpatialModel 的闭环 rollout 评估。

目的：
  闭环模型 s_t = F(s_{t-1}, a_t, z_{t-1}) 推理时需把自身预测喂回（autoregressive），
  这与训练（teacher forcing，prev_skeleton 取 GT）存在 train/inference gap。
  本脚本验证 rollout 误差累积是否可控，并监测潜变量 z 的漂移。

  当前项目没有任何闭环 rollout 评估脚本（现有 evaluate_3d.py 是逐帧独立的），
  本脚本填补这一空白。

指标：
  - 单步误差（GT 喂 prev）：参考上界，反映模型本身精度。
  - rollout 误差（自身预测喂回）：T 步累积漂移，核心指标。
  - ‖z_t‖ 范数轨迹：z 无界漂移会连带 s 失稳，需监测。
  - 发散步（误差 > 10× 中位数）：定位漂移失控点。

注意：
  rollout 在归一化空间进行（模型 forward 在归一化空间运算），最后再反归一化与
  GT 物理坐标对比。这样保证 prev_skeleton 的"喂回"与训练时的空间一致。

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_rollout.py \
        --checkpoint train_log/state_transition/exp_xxx/phase_state_transition/model/best_model.pt \
        --data_dir data/seq_rz_c2_sk --seq_idx 0 --max_steps 50
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.model_loader import load_model


def build_action_window(actions, t, window_size):
    """构造以 t 结尾的动作窗口，不足时前向 zero-pad（与 dataset._get_action_window 一致）。

    Args:
        actions: (T, D) 全序列动作。
        t: 当前时间步（窗口右端）。
        window_size: 窗口长度。
    Returns:
        (window_size, D) 动作窗口。
    """
    D = actions.shape[1]
    start = t - window_size + 1
    if start >= 0:
        return actions[start:t + 1].copy()
    pad = np.zeros((-start, D), dtype=actions.dtype)
    return np.concatenate([pad, actions[0:t + 1]], axis=0)


def rollout_one_sequence(model, actions, positions, window_size, norm_factor,
                         device, max_steps=None):
    """对一个序列做闭环 rollout（归一化空间），返回预测与指标。

    Args:
        model: StateTransitionSpatialModel（已 set_normalization，eval 模式）。
        actions: (T, D) 全序列动作（物理值，未归一化）。
        positions: (T, 3, N) GT 中心线（物理坐标）。
        window_size: 动作窗口长度。
        norm_factor: 动作归一化因子。
        device: 计算设备。
        max_steps: 最大 rollout 步数（None = 整个序列）。

    Returns:
        dict: {
            'pred_norm':   (T, N, 3) 归一化空间 rollout 预测,
            'gt_norm':     (T, N, 3) 归一化空间 GT,
            'rollout_mse': (T,) 每步 rollout 误差（自身预测喂回）,
            'onestep_mse': (T,) 每步单步误差（GT 喂 prev，参考上界）,
            'z_norm':      (T,) 每步 ‖z_t‖₂,
        }
    """
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)

    # 归一化参数（numpy，用于把 GT 中心线转到归一化空间）
    actions_norm = actions / norm_factor
    pc_center_np = model.pc_center.view(3).cpu().numpy()
    pc_scale_np = model.pc_scale.view(3).cpu().numpy()

    def to_norm_skel(pos_3N):
        # pos_3N: (3, N) → (N, 3) 归一化张量 (1, N, 3)
        skel = pos_3N.T.astype(np.float32)
        skel = (skel - pc_center_np) / pc_scale_np
        return torch.from_numpy(skel).float().unsqueeze(0).to(device)

    pred_norm_list = []
    gt_norm_list = []
    rollout_mse = []
    onestep_mse = []
    z_norm_list = []

    # ── rollout 初始化：首帧用 GT（冷启动）──
    s_t_norm = to_norm_skel(positions[0])   # (1, N, 3)
    s_prev_norm = s_t_norm                    # 占位（首步无前驱）
    aw0 = build_action_window(actions_norm, 0, window_size)
    z0 = model.init_z_from_action(
        torch.from_numpy(aw0).float().unsqueeze(0).to(device))  # (1, z_dim)
    # 两条独立的 z 轨迹（关键修复）：
    #   z_t   — rollout 路径：喂"自身预测 s"演化（autoregressive）
    #   z_tf  — onestep 参考路径：喂"GT s"演化（teacher forcing）
    # 此前共用单个 z_t → onestep 参考被 rollout 演化的 z 污染，漂移比不干净。
    z_t = z0
    z_tf = z0.clone()

    with torch.no_grad():
        for t in range(T):
            gt_norm = to_norm_skel(positions[t])  # (1, N, 3)
            aw_tensor = torch.from_numpy(
                build_action_window(actions_norm, t, window_size)
            ).float().unsqueeze(0).to(device)

            # ── 1. 单步误差：GT 前驱 + GT 演化的 z_tf（干净的 teacher-forcing 参考上界）──
            prev_gt = to_norm_skel(positions[max(t - 1, 0)])
            prev_prev_gt = to_norm_skel(positions[max(t - 2, 0)])
            onestep_out = model.forward(aw_tensor, prev_gt, prev_prev_gt, z_tf)
            z_tf = onestep_out['latent_z']   # 从 GT s 演化（保持参考干净）
            onestep_mse.append(F.mse_loss(onestep_out['skeleton'], gt_norm).item())

            # ── 2. rollout：自身上一步预测喂回（autoregressive），z_t 从预测 s 演化 ──
            roll_out = model.forward(aw_tensor, s_t_norm, s_prev_norm, z_t)
            s_pred = roll_out['skeleton']  # (1, N, 3)
            z_t = roll_out['latent_z']    # (1, z_dim)

            rollout_mse.append(F.mse_loss(s_pred, gt_norm).item())
            z_norm_list.append(z_t.norm().item())
            pred_norm_list.append(s_pred.squeeze(0).cpu())
            gt_norm_list.append(gt_norm.squeeze(0).cpu())

            # 更新 rollout 状态（喂回自身预测）
            s_prev_norm = s_t_norm
            s_t_norm = s_pred

    pred_norm = torch.stack(pred_norm_list, dim=0)  # (T, N, 3)
    gt_norm = torch.stack(gt_norm_list, dim=0)      # (T, N, 3)
    return {
        'pred_norm': pred_norm.numpy(),
        'gt_norm': gt_norm.numpy(),
        'rollout_mse': np.array(rollout_mse),
        'onestep_mse': np.array(onestep_mse),
        'z_norm': np.array(z_norm_list),
    }


def rollout_windowed_one_sequence(model, actions, positions, window_size,
                                   norm_factor, device, window_len,
                                   stride=None, max_steps=None):
    """窗口开环 rollout 评估（方向 15 的核心指标）。

    与 rollout_one_sequence（整序列单种子）的区别：每 window_len 步用 GT 重新种子
    （s = positions[t0-1]，z = init_z_from_action(aw[t0])），窗口内剩余步把模型自身预测
    喂回（s 与 z 在窗口内自演化）。窗口结束重新种子。这把 rollout 漂移约束在 K 步内，
    对应"观测一次 → 开环预测 K 步"的部署语义。

    返回窗口内位置 k=0..window_len-1 的误差曲线（聚合所有窗口的均值）——展示误差随
    "距上次观测步数"的增长，是迟滞衰减/漂移的核心可视化。

    Args:
        model: StateTransitionSpatialModel / OpenLoopTransitionModel（已 set_normalization）。
        actions: (T, D) 全序列动作（物理值）。
        positions: (T, 3, N) GT 中心线（物理坐标）。
        window_size: 动作窗口长度。
        norm_factor: 动作归一化因子。
        device: 计算设备。
        window_len: 开环窗口 K（每 K 步重新种子）。
        stride: 窗口步长（默认=window_len，非重叠；减小可增多样本）。
        max_steps: 评估的最大序列长度。

    Returns:
        dict: {
            'rollout_err_by_k': (K,) 每个窗口内位置 k 的 rollout MSE 均值,
            'onestep_err_by_k': (K,) 干净 teacher-forced 参考（独立 z_tf 轨迹）,
            'z_norm_by_k':      (K,) 窗口内 ‖z_t‖ 均值,
            'drift_by_k':       (K,) rollout/onestep 逐位漂移比,
            'n_windows':        int,
        }
    """
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)
    if stride is None:
        stride = window_len  # 非重叠窗口

    actions_norm = actions / norm_factor
    pc_center_np = model.pc_center.view(3).cpu().numpy()
    pc_scale_np = model.pc_scale.view(3).cpu().numpy()

    def to_norm_skel(pos_3N):
        skel = pos_3N.T.astype(np.float32)
        skel = (skel - pc_center_np) / pc_scale_np
        return torch.from_numpy(skel).float().unsqueeze(0).to(device)

    K = window_len
    roll_by_k = [[] for _ in range(K)]
    one_by_k = [[] for _ in range(K)]
    z_by_k = [[] for _ in range(K)]
    n_windows = 0

    # 窗口起点 t0 ∈ [1, T-K]（t0≥1 保证 positions[t0-1] 种子有效）
    t0 = 1
    with torch.no_grad():
        while t0 + K <= T:
            # 种子：s = GT positions[t0-1]，z = init_z_from_action(aw[t0])
            aw_seed = build_action_window(actions_norm, t0, window_size)
            z0 = model.init_z_from_action(
                torch.from_numpy(aw_seed).float().unsqueeze(0).to(device))
            s_roll = to_norm_skel(positions[t0 - 1])      # rollout 种子（GT）
            s_prev_roll = s_roll
            z_t = z0
            z_tf = z0.clone()                              # onestep 参考的独立 z 轨迹

            for k in range(K):
                tt = t0 + k
                gt_norm = to_norm_skel(positions[tt])
                aw_tensor = torch.from_numpy(
                    build_action_window(actions_norm, tt, window_size)
                ).float().unsqueeze(0).to(device)

                # onestep 参考：GT 前驱 + GT 演化的 z_tf（干净）
                prev_gt = to_norm_skel(positions[tt - 1])
                prev_prev_gt = to_norm_skel(positions[max(tt - 2, 0)])
                onestep_out = model.forward(aw_tensor, prev_gt, prev_prev_gt, z_tf)
                z_tf = onestep_out['latent_z']
                one_by_k[k].append(F.mse_loss(onestep_out['skeleton'], gt_norm).item())

                # rollout：自身预测喂回，z_t 从预测 s 演化
                roll_out = model.forward(aw_tensor, s_roll, s_prev_roll, z_t)
                s_pred = roll_out['skeleton']
                z_t = roll_out['latent_z']
                roll_by_k[k].append(F.mse_loss(s_pred, gt_norm).item())
                z_by_k[k].append(z_t.norm().item())

                s_prev_roll = s_roll
                s_roll = s_pred

            n_windows += 1
            t0 += stride

    roll_k = np.array([np.mean(x) if x else np.nan for x in roll_by_k])
    one_k = np.array([np.mean(x) if x else np.nan for x in one_by_k])
    z_k = np.array([np.mean(x) if x else np.nan for x in z_by_k])
    drift_k = roll_k / np.maximum(one_k, 1e-8)
    return {
        'rollout_err_by_k': roll_k,
        'onestep_err_by_k': one_k,
        'z_norm_by_k': z_k,
        'drift_by_k': drift_k,
        'n_windows': n_windows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data dir with .npz sequences")
    parser.add_argument("--seq_idx", type=int, default=0,
                        help="Which .npz file to rollout (sorted order)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max rollout steps (None = whole sequence)")
    # ── 窗口开环评估（方向 15）：每 K 步用 GT 重新种子 ──
    parser.add_argument("--windowed", action="store_true",
                        help="Windowed open-loop eval: re-seed GT every --window_len steps "
                             "(方向 15 核心指标；默认关，走整序列单种子 rollout)")
    parser.add_argument("--window_len", type=int, default=40,
                        help="Open-loop window K (only with --windowed)")
    parser.add_argument("--window_stride", type=int, default=None,
                        help="Window stride (default=window_len, non-overlapping)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型（auto-detect type，含归一化 buffer）
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']
    window_size = info['window_size']
    norm_factor = info['norm_factor']

    # 加载序列
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.seq_idx >= len(files):
        raise IndexError(f"seq_idx {args.seq_idx} >= {len(files)} files")
    raw = np.load(files[args.seq_idx])
    actions = raw['actions'].astype(np.float32)
    positions = raw['positions'].astype(np.float32)  # (T, 3, N)
    print(f"\nRollout: {files[args.seq_idx]}")
    print(f"  T={positions.shape[0]}, N={positions.shape[2]}, D={actions.shape[1]}")

    if args.windowed:
        wres = rollout_windowed_one_sequence(
            model, actions, positions, window_size, norm_factor, device,
            args.window_len, stride=args.window_stride, max_steps=args.max_steps)
        rk = wres['rollout_err_by_k']
        ok = wres['onestep_err_by_k']
        dk = wres['drift_by_k']
        zk = wres['z_norm_by_k']
        K = len(rk)
        valid = ~np.isnan(rk)
        print(f"\n{'='*60}")
        print(f"Windowed open-loop rollout (方向 15): K={K}, n_windows={wres['n_windows']}")
        print(f"{'k':>3} {'rollout_MSE':>14} {'onestep_MSE':>14} {'drift_ratio':>12} {'z_norm':>8}")
        # 打印采样位置（k=0, K/4, K/2, 3K/4, K-1）避免 K=40 时刷屏
        sample_ks = sorted(set([0, K // 4, K // 2, 3 * K // 4, K - 1]))
        for k in sample_ks:
            if np.isnan(rk[k]):
                continue
            print(f"{k:>3} {rk[k]:>14.3e} {ok[k]:>14.3e} {dk[k]:>12.2f} {zk[k]:>8.3f}")
        mean_drift = np.nanmean(dk[valid])
        final_drift = dk[K - 1] if not np.isnan(dk[K - 1]) else np.nan
        print(f"\n  mean drift ratio = {mean_drift:.2f}x | final-k drift = {final_drift:.2f}x")
        print(f"  z_norm: start={zk[0]:.3f}, mid={zk[K//2]:.3f}, end={zk[K-1]:.3f}")
        print(f"{'='*60}")
        print("\n解读（窗口开环）:")
        print("  - drift_by_k 应在 k=0≈1 并随 k 单调缓增（迟滞衰减）；K-1 处仍 <~30x 为健康")
        print("  - 若 drift 在窗口内指数增长 / z_norm 爆炸：z 在闭环下失稳，需退火或 z 收缩正则")
        print("  - 对比整序列 rollout（无 --windowed）：窗口重种子应显著降低漂移")
        return

    result = rollout_one_sequence(
        model, actions, positions, window_size, norm_factor, device, args.max_steps)

    rm = result['rollout_mse']
    om = result['onestep_mse']
    zn = result['z_norm']
    T = len(rm)

    # 发散检测：rollout 误差 > 10× 中位数
    median_err = np.median(rm)
    div_steps = np.where(rm > 10 * max(median_err, 1e-8))[0]
    div_step = int(div_steps[0]) if len(div_steps) > 0 else None

    print(f"\n{'='*50}")
    print(f"Rollout results ({T} steps):")
    print(f"  rollout MSE : mean={rm.mean():.6f}, final={rm[-1]:.6f}")
    print(f"  onestep MSE : mean={om.mean():.6f}, final={om[-1]:.6f}")
    print(f"  z norm      : mean={zn.mean():.4f}, max={zn.max():.4f}, final={zn[-1]:.4f}")
    print(f"  median rollout err = {median_err:.6f}")
    print(f"  divergence step     = {div_step}")
    print(f"  漂移比 rollout/onestep = {rm.mean()/max(om.mean(),1e-8):.2f}x")
    print(f"{'='*50}")
    print("\n解读:")
    print("  - 漂移比 ≈ 1: rollout 稳定，误差不累积（理想）")
    print("  - 漂移比 显著 > 1: 误差累积，需 Stage 1 的 scheduled sampling + 收缩正则")
    print("  - divergence step != None: rollout 在该步失控发散")
    print("  - z norm 持续增长: z 漂移失控，需对 z 转移加收缩约束")


if __name__ == "__main__":
    main()
