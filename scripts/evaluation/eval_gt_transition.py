"""eval_gt_transition.py — 全 GT 驱动框架的观测驱动评估。

定位（与 eval_rollout.py 的根本区别）:
  eval_rollout.py 是"纯自回归"——s 和 z 都喂模型预测，模拟"无法每步观测"的未来场景。
  本脚本是"观测驱动"——**s_{t-1} 每步都喂真实 GT**（仿真 positions[t-1]；实物对应
  图像骨架化结果），只有 z 跨帧演化。这才是 GTObservedTransitionModel 的真实部署场景。

  关键性质：
    - s 每步重置为真实观测 → s 不累积漂移（与 train 完全一致）
    - 误差累积风险仅在 z（z 无 GT，跨帧演化）→ 本脚本核心就是监测 z 是否漂移
    - 单步误差 = 模型在"完美知道前一状态"下的精度（这里就是部署精度，不是上界）

指标:
  - per_step_mse: 每步 ŝ_t vs GT s_t（s_{t-1} 真实，z 演化）——部署精度
  - z_norm 轨迹: ‖z_t‖₂ 随步数变化，检测 z 漂移（核心风险指标）
  - z 是否收敛/有界: 若 z_norm 单调发散 → 需对 z 转移加收缩约束

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/eval_gt_transition.py \
        --checkpoint train_log/gt_transition/exp_xxx/phase_gt_transition/model/best_model.pt \
        --data_dir data/seq_rz_c2_sk --seq_idx 0
"""

import os
import sys
import glob
import argparse

import numpy as np
import torch

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.model_loader import load_model


def build_action_window(actions, t, window_size):
    """构造以 t 结尾的动作窗口，不足时前向 zero-pad（与 dataset._get_action_window 一致）。"""
    D = actions.shape[1]
    start = t - window_size + 1
    if start >= 0:
        return actions[start:t + 1].copy()
    pad = np.zeros((-start, D), dtype=actions.dtype)
    return np.concatenate([pad, actions[0:t + 1]], axis=0)


def observed_rollout(model, actions, positions, window_size, norm_factor,
                     device, max_steps=None):
    """观测驱动 rollout：s_{t-1} 每步喂真实 GT，z 跨帧演化。

    与 eval_rollout 的纯自归 rollout 区别：prev_skeleton 恒取 GT（而非模型预测），
    所以 s 不累积漂移；只有 z 通过 model.forward 的 latent_z 跨帧演化。

    Returns:
        dict: {
            'per_step_mse': (T,) 每步 ŝ_t vs GT（部署精度）,
            'z_norm':       (T,) 每步 ‖z_t‖₂（z 漂移监测）,
            'pred_norm':    (T, N, 3) 归一化空间预测,
            'gt_norm':      (T, N, 3) 归一化空间 GT,
        }
    """
    T = positions.shape[0]
    if max_steps is not None:
        T = min(T, max_steps)

    actions_norm = actions / norm_factor
    pc_center_np = model.pc_center.view(3).cpu().numpy()
    pc_scale_np = model.pc_scale.view(3).cpu().numpy()

    def to_norm_skel(pos_3N):
        skel = pos_3N.T.astype(np.float32)
        skel = (skel - pc_center_np) / pc_scale_np
        return torch.from_numpy(skel).float().unsqueeze(0).to(device)

    per_step_mse = []
    z_norm = []
    pred_list, gt_list = [], []

    # z 初始化：首步 action_window → z_init
    aw0 = torch.from_numpy(build_action_window(actions_norm, 0, window_size)
                           ).float().unsqueeze(0).to(device)
    z_t = model.init_z_from_action(aw0)

    with torch.no_grad():
        for t in range(T):
            gt_norm = to_norm_skel(positions[t])  # (1, N, 3)
            aw_t = torch.from_numpy(
                build_action_window(actions_norm, t, window_size)
            ).float().unsqueeze(0).to(device)

            # 观测驱动：prev_skeleton 恒取真实 GT（positions[t-1]）
            prev_gt = to_norm_skel(positions[max(t - 1, 0)])
            prev_prev_gt = to_norm_skel(positions[max(t - 2, 0)])

            out = model.forward(aw_t, prev_gt, prev_prev_gt, z_t)
            s_pred = out["skeleton"]      # (1, N, 3)
            z_t = out["latent_z"]         # z 跨帧演化（唯一无 GT 的状态）

            per_step_mse.append(((s_pred - gt_norm) ** 2).mean().item())
            z_norm.append(z_t.norm().item())
            pred_list.append(s_pred.squeeze(0).cpu())
            gt_list.append(gt_norm.squeeze(0).cpu())

    return {
        'per_step_mse': np.array(per_step_mse),
        'z_norm': np.array(z_norm),
        'pred_norm': torch.stack(pred_list, dim=0).numpy(),
        'gt_norm': torch.stack(gt_list, dim=0).numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--data_dir", type=str, required=True, help="Data dir with .npz")
    parser.add_argument("--seq_idx", type=int, default=0, help="Which .npz to evaluate (sorted)")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps (None=whole seq)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_model(args.checkpoint, data_dir=args.data_dir, device=device)
    model = info['model']
    window_size = info['window_size']
    norm_factor = info['norm_factor']

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.npz")))
    if args.seq_idx >= len(files):
        raise IndexError(f"seq_idx {args.seq_idx} >= {len(files)} files")
    raw = np.load(files[args.seq_idx])
    actions = raw['actions'].astype(np.float32)
    positions = raw['positions'].astype(np.float32)
    print(f"\nObserved-driven eval: {files[args.seq_idx]}")
    print(f"  T={positions.shape[0]}, N={positions.shape[2]}, D={actions.shape[1]}")

    r = observed_rollout(model, actions, positions, window_size, norm_factor,
                         device, args.max_steps)
    pm, zn = r['per_step_mse'], r['z_norm']
    T = len(pm)

    # z 漂移判定：末步 z_norm 是否显著大于首步（发散趋势）
    z_drift = zn[-1] / max(zn[0], 1e-8)

    print(f"\n{'='*50}")
    print(f"Observed-driven results ({T} steps, s_{{t-1}} always real GT):")
    print(f"  per-step MSE : mean={pm.mean():.6e}, final={pm[-1]:.6e}, max={pm.max():.6e}")
    print(f"  z norm       : start={zn[0]:.4f}, mean={zn.mean():.4f}, final={zn[-1]:.4f}")
    print(f"  z drift ratio (final/start) = {z_drift:.2f}x")
    print(f"{'='*50}")
    print("\n解读（全 GT 驱动框架，s 每步真实）:")
    print("  - per-step MSE 就是部署精度（前一状态来自真实观测，train/inference 一致）")
    print("  - z drift ratio ≈ 1: z 稳定有界，无累积漂移（理想）")
    print("  - z drift ratio 显著 > 1 且单调增长: z 漂移失控，需对 z 转移加收缩约束")
    print("  - 对比 eval_rollout.py 的自回归漂移：这里 s 不漂移，只有 z 是风险源")


if __name__ == "__main__":
    main()
