"""transition_metrics.py — 状态转移族（gt_transition / open_loop / base）的 rollout 评估指标。

供"训练中评估"（shape_evaluation.evaluate_transition_during_training）和"可视化"
（visualize_3d_shape.py）共用，单一事实源（DRY）。

评估语义（窗口开环 rollout，方向 15）:
  每个窗口 K 步：以 1 帧 GT 作种子（positions[t0-1]），窗口内 k=0..K-1 把模型自身预测喂回
  （s 与 z 在窗口内自演化）。同时维护一条干净 teacher-forced 参考（onestep，喂 GT + 独立 z_tf）。

指标分两类:
  归一化空间（自回归漂移故事）:
    rollout_mse_by_k   窗口内位置 k 的开环 rollout MSE（喂自身预测）
    onestep_mse_by_k   干净 teacher-forced 参考（独立 z_tf，喂 GT s）
    drift_by_k         rollout/onestep 逐位漂移比
    z_norm_by_k        窗口内 ‖z_t‖
  物理空间（部署精度故事，反归一化到米）:
    mean_node / endpoint / max / chamfer (mm)，逐节点 base/mid/tip (mm)
    在 rollout 预测上算（开环部署精度）。

外加 model_vs_copy = rollout_mse_mean / copy_mse_mean（copy = 永远预测种子；<1 = 优于不动）。
"""

import os
import glob
import numpy as np
import torch

from src.training.metrics_3d import evaluate_skeleton, node_errors

ARM_LENGTH = 0.5   # 臂长（米）
ROD_RADIUS = 0.015  # 杆半径（米）


def build_action_window(actions_norm, t, window_size):
    """以 t 结尾的动作窗口，不足前向 zero-pad（与 dataset._get_action_window 一致）。"""
    D = actions_norm.shape[1]
    start = t - window_size + 1
    if start >= 0:
        return actions_norm[start:t + 1].copy()
    pad = np.zeros((-start, D), dtype=actions_norm.dtype)
    return np.concatenate([pad, actions_norm[0:t + 1]], axis=0)


def _window_len(model, config):
    """开环窗口 K = model.episode_len（open_loop），否则 config window_size。"""
    K = getattr(model, 'episode_len', None)
    if K:
        return int(K)
    return int(config.get('temporal', {}).get('window_size', 40))


def rollout_one_window(model, actions_norm, positions, t0, K, window_size,
                        pc_center, pc_scale, device):
    """单窗口开环 rollout + 干净 onestep 参考。

    种子: s = GT positions[t0-1]，z = init_z_from_action(aw[t0])。
    窗口 k=0..K-1: rollout 喂自身预测（z_t 演化），onestep 喂 GT（独立 z_tf 演化）。

    Returns dict: roll/one/gt (K,N,3) 归一化, seed (N,3), z_norm (K,)。
    """
    def to_norm(pos_3N):
        skel = pos_3N.T.astype(np.float32)
        skel = (skel - pc_center) / pc_scale
        return torch.from_numpy(skel).float().unsqueeze(0).to(device)  # (1,N,3)

    aw_seed = build_action_window(actions_norm, t0, window_size)
    z0 = model.init_z_from_action(
        torch.from_numpy(aw_seed).float().unsqueeze(0).to(device))
    s_roll = to_norm(positions[t0 - 1])
    s_prev_roll = s_roll
    z_t = z0
    z_tf = z0.clone()

    roll_preds, one_preds, gts, z_norms = [], [], [], []
    with torch.no_grad():
        for k in range(K):
            tt = t0 + k
            gt = to_norm(positions[tt])
            aw = torch.from_numpy(
                build_action_window(actions_norm, tt, window_size)
            ).float().unsqueeze(0).to(device)

            # onestep 参考: GT 前驱 + GT 演化的 z_tf（干净 teacher-forcing 上界）
            prev_gt = to_norm(positions[tt - 1])
            prev_prev_gt = to_norm(positions[max(tt - 2, 0)])
            one_out = model.forward(aw, prev_gt, prev_prev_gt, z_tf)
            z_tf = one_out['latent_z']

            # rollout: 自身上一步预测喂回，z_t 从预测 s 演化
            roll_out = model.forward(aw, s_roll, s_prev_roll, z_t)
            s_pred = roll_out['skeleton']
            z_t = roll_out['latent_z']

            roll_preds.append(s_pred.squeeze(0))
            one_preds.append(one_out['skeleton'].squeeze(0))
            gts.append(gt.squeeze(0))
            z_norms.append(z_t.norm().item())
            s_prev_roll = s_roll
            s_roll = s_pred

    return {
        'roll': torch.stack(roll_preds, 0),   # (K,N,3) normalized
        'one': torch.stack(one_preds, 0),
        'gt': torch.stack(gts, 0),
        'seed': to_norm(positions[t0 - 1]).squeeze(0),  # (N,3)
        'z_norm': np.array(z_norms),           # (K,)
    }


def _divergence_step(drift_k, factor=10.0):
    """首个 drift > factor× 中位数的 k（漂移失控点）；无则 None。"""
    med = float(np.median(drift_k))
    hits = np.where(drift_k > factor * max(med, 1e-8))[0]
    return int(hits[0]) if len(hits) > 0 else None


def evaluate_transition_rollout(model, data_dir, config, device,
                                 n_seqs=5, windows_per_seq=2, K=None,
                                 arm_length=ARM_LENGTH, rod_radius=ROD_RADIUS):
    """对状态转移模型做窗口开环 rollout 评估，返回 {'summary': {...}, 'by_k': {...}}。

    Args:
        model: StateTransitionSpatialModel / GTObserved / OpenLoop（已 set_normalization）。
        data_dir: 含 .npz 的数据目录。
        config: 训练配置 dict（读 window_size / evaluation）。
        device: torch device。
        n_seqs: 评估序列数（均匀采样；≤0 = 全部）。
        windows_per_seq: 每序列窗口数。
        K: 窗口长度（None → model.episode_len 或 config window_size）。
    """
    was_training = model.training
    model.eval()
    if K is None:
        K = _window_len(model, config)
    window_size = int(config.get('temporal', {}).get('window_size', 40))

    files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
    if not files:
        if was_training:
            model.train()
        return None
    if n_seqs > 0 and len(files) > n_seqs:
        idx = np.linspace(0, len(files) - 1, n_seqs, dtype=int)
        files = [files[i] for i in idx]

    norm_factor = 1.0
    nf = getattr(model, 'action_norm_factor', None)
    if nf is not None:
        norm_factor = nf.item() if isinstance(nf, torch.Tensor) else float(nf)
    pc_center = model.pc_center.view(3).detach().cpu().numpy()
    pc_scale = model.pc_scale.view(3).detach().cpu().numpy()

    roll_mse_k = np.zeros(K)
    one_mse_k = np.zeros(K)
    copy_mse_k = np.zeros(K)
    z_k = np.zeros(K)
    roll_node_mm_k = np.zeros(K)
    n_win = 0
    all_roll_world, all_gt_world = [], []   # 物理空间聚合
    per_node_sum = None

    for f in files:
        d = np.load(f)
        if 'positions' not in d:
            continue
        actions = d['actions'].astype(np.float32)
        positions = d['positions'].astype(np.float32)  # (T,3,N)
        T = positions.shape[0]
        if T - 1 < K:
            continue
        actions_norm = actions / norm_factor
        max_t0 = T - K
        t0s = ([max(1, max_t0 // 2)] if windows_per_seq <= 1
               else list(np.linspace(1, max_t0, windows_per_seq, dtype=int)))
        for t0 in t0s:
            t0 = int(t0)
            r = rollout_one_window(model, actions_norm, positions, t0, K,
                                    window_size, pc_center, pc_scale, device)
            roll, one, gt, seed = r['roll'], r['one'], r['gt'], r['seed']
            # 归一化空间 per-k MSE
            roll_mse_k += ((roll - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
            one_mse_k += ((one - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
            copy_mse_k += ((seed.unsqueeze(0) - gt) ** 2).mean(dim=(1, 2)).cpu().numpy()
            z_k += r['z_norm']
            # 物理空间 per-k 平均节点误差 (mm)
            roll_world = roll.cpu().numpy() * pc_scale + pc_center  # (K,N,3) m
            gt_world = gt.cpu().numpy() * pc_scale + pc_center
            roll_t = torch.from_numpy(roll_world).float()
            gt_t = torch.from_numpy(gt_world).float()
            nk_k = node_errors(roll_t, gt_t).mean(dim=1).cpu().numpy() * 1000.0  # (K,) mm per-k
            roll_node_mm_k += nk_k
            # 物理聚合
            all_roll_world.append(roll_t)
            all_gt_world.append(gt_t)
            pn = node_errors(roll_t, gt_t).mean(dim=0).cpu().numpy()
            per_node_sum = pn.copy() if per_node_sum is None else per_node_sum + pn
            n_win += 1

    if n_win == 0:
        if was_training:
            model.train()
        return None

    roll_mse_k /= n_win
    one_mse_k /= n_win
    copy_mse_k /= n_win
    z_k /= n_win
    roll_node_mm_k /= n_win
    drift_k = roll_mse_k / np.maximum(one_mse_k, 1e-8)

    # 物理聚合（全窗口全步 flatten 成一个大 batch）
    agg = evaluate_skeleton(torch.cat(all_roll_world, 0), torch.cat(all_gt_world, 0),
                             arm_length, rod_radius)
    per_node = per_node_sum / n_win
    N = len(per_node)
    nb, nm = N // 3, 2 * N // 3

    summary = {
        'n_windows': n_win, 'K': K, 'n_seqs': len(files),
        # 归一化空间（漂移故事）
        'rollout_mse_mean': float(roll_mse_k.mean()),
        'onestep_mse_mean': float(one_mse_k.mean()),
        'copy_mse_mean': float(copy_mse_k.mean()),
        'mean_drift': float(drift_k.mean()),
        'mid_drift': float(drift_k[K // 2]),
        'final_drift': float(drift_k[-1]),
        'divergence_step': _divergence_step(drift_k),
        'z_norm_start': float(z_k[0]),
        'z_norm_mid': float(z_k[K // 2]),
        'z_norm_end': float(z_k[-1]),
        'model_vs_copy': float(roll_mse_k.mean() / max(copy_mse_k.mean(), 1e-12)),
        # 物理空间（部署精度，mm）
        'mean_node_mm': float(agg['mean_node_err'] * 1000),
        'endpoint_mm': float(agg['endpoint_err'] * 1000),
        'max_node_mm': float(agg['max_node_err'] * 1000),
        'chamfer_mm': float(agg['chamfer_distance'] * 1000),
        'mean_pct_arm': float(agg['mean_pct_arm']),
        'per_node_base_mm': float(per_node[:nb].mean() * 1000),
        'per_node_mid_mm': float(per_node[nb:nm].mean() * 1000),
        'per_node_tip_mm': float(per_node[nm:].mean() * 1000),
    }
    by_k = {
        'rollout_mse': roll_mse_k.tolist(),
        'onestep_mse': one_mse_k.tolist(),
        'drift_ratio': drift_k.tolist(),
        'z_norm': z_k.tolist(),
        'rollout_node_mm': roll_node_mm_k.tolist(),
    }
    if was_training:
        model.train()
    return {'summary': summary, 'by_k': by_k}


def format_summary_line(s):
    """单行摘要（训练日志/可视化打印共用）。"""
    return (f"drift mean={s['mean_drift']:.1f}x/final={s['final_drift']:.1f}x "
            f"| node {s['mean_node_mm']:.2f}mm (end {s['endpoint_mm']:.2f}, max {s['max_node_mm']:.2f}) "
            f"| z {s['z_norm_start']:.2f}->{s['z_norm_end']:.2f} "
            f"| model/copy {s['model_vs_copy']:.2f}x")
