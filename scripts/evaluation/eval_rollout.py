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
    z_t = model.init_z_from_action(
        torch.from_numpy(aw0).float().unsqueeze(0).to(device))  # (1, z_dim)

    with torch.no_grad():
        for t in range(T):
            gt_norm = to_norm_skel(positions[t])  # (1, N, 3)
            aw_tensor = torch.from_numpy(
                build_action_window(actions_norm, t, window_size)
            ).float().unsqueeze(0).to(device)

            # ── 1. 单步误差：GT 前驱 teacher forcing（参考上界）──
            prev_gt = to_norm_skel(positions[max(t - 1, 0)])
            prev_prev_gt = to_norm_skel(positions[max(t - 2, 0)])
            onestep_out = model.forward(aw_tensor, prev_gt, prev_prev_gt, z_t)
            onestep_mse.append(F.mse_loss(onestep_out['skeleton'], gt_norm).item())

            # ── 2. rollout：自身上一步预测喂回（autoregressive）──
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
