"""test_ema_discriminative.py — 测试 EMA 编码器区分不同驱动参数序列的能力。

测试内容:
  1. 不同 action window 的编码余弦相似度矩阵（越低越好）
  2. 相同 action vs 不同 action 的编码距离分布对比
  3. 编码向量各维度的方差（信息利用率）
  4. 线性可分性测试：不同 action 区间是否能被线性分类器分开

用法:
    python scripts/tests/test_ema_discriminative.py --data_dir data/seq_rz_c6_sk
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.encoders.multi_scale_ema import MultiScaleEMA


def load_action_windows(data_dir, window_size=20, max_samples=200):
    """从数据集加载 action windows 和对应的时间步信息。"""
    import glob
    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not npz_files:
        print(f"  Error: no npz files in {data_dir}")
        return None, None, None

    # 加载所有 actions
    all_actions = []
    all_norm = []
    for f in npz_files:
        d = np.load(f)
        actions = d['actions']
        norm = float(np.max(np.abs(actions)))
        all_actions.append(actions)
        all_norm.append(norm)

    global_norm = max(all_norm)
    print(f"  Loaded {len(npz_files)} sequences, global_norm={global_norm:.4f}")

    # 采样 action windows
    windows = []
    labels = []  # (seq_id, timestep) 用于后续分析
    action_values = []  # 当前帧的 action 值

    for seq_id, actions in enumerate(all_actions):
        actions_normed = actions / global_norm
        T = len(actions_normed)
        for t in range(window_size - 1, T):
            if len(windows) >= max_samples:
                break
            start = t - window_size + 1
            windows.append(actions_normed[start:t + 1])
            labels.append((seq_id, t))
            action_values.append(actions_normed[t])
        if len(windows) >= max_samples:
            break

    windows = np.array(windows, dtype=np.float32)  # (N, window_size, action_dim)
    action_values = np.array(action_values, dtype=np.float32)  # (N, action_dim)

    print(f"  Sampled {len(windows)} windows, shape={windows.shape}")
    print(f"  Action value range: [{action_values.min():.4f}, {action_values.max():.4f}]")

    return windows, action_values, labels


def test_cosine_similarity_matrix(encoder, windows, n_samples=50):
    """测试不同 action window 编码之间的余弦相似度。"""
    n = min(n_samples, len(windows))
    indices = np.random.choice(len(windows), n, replace=False)
    selected = torch.from_numpy(windows[indices]).float()

    with torch.no_grad():
        encodings = encoder(selected)  # (n, hidden_dim)
        enc_norm = F.normalize(encodings, dim=-1)
        sim_matrix = torch.mm(enc_norm, enc_norm.T).numpy()

    # 排除对角线
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim_matrix[mask]

    print(f"\n{'='*60}")
    print(f"  余弦相似度矩阵 (n={n})")
    print(f"{'='*60}")
    print(f"  Mean:  {off_diag.mean():.4f}")
    print(f"  Std:   {off_diag.std():.4f}")
    print(f"  Min:   {off_diag.min():.4f}")
    print(f"  Max:   {off_diag.max():.4f}")
    print(f"  Median: {np.median(off_diag):.4f}")

    # 如果 mean > 0.8，说明编码器输出太相似
    if off_diag.mean() > 0.8:
        print(f"  ⚠️  WARNING: 编码器区分度很低！不同输入产生几乎相同的编码。")
    elif off_diag.mean() > 0.5:
        print(f"  ⚠️  编码器区分度中等，有改善空间。")
    else:
        print(f"  ✅ 编码器区分度良好。")

    return off_diag.mean()


def test_same_vs_different(encoder, windows, action_values, n_pairs=500):
    """对比相同 action 和不同 action 的编码距离分布。"""
    n = len(windows)
    windows_t = torch.from_numpy(windows).float()

    # 把 action 值离散化为几个区间
    action_dim = action_values.shape[1]
    n_bins = 3

    # 按 action[0] 分 bin
    bins = []
    for d in range(action_dim):
        thresholds = np.percentile(action_values[:, d], [33, 67])
        bin_idx = np.digitize(action_values[:, d], thresholds)
        bins.append(bin_idx)

    # 同 bin 对 vs 不同 bin 对的距离
    same_dist = []
    diff_dist = []

    with torch.no_grad():
        encodings = encoder(windows_t)  # (N, hidden_dim)

    enc_np = encodings.numpy()

    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        dist = np.linalg.norm(enc_np[i] - enc_np[j])

        # 检查两个样本的 action 是否在同一个 bin
        same_bin = all(bins[d][i] == bins[d][j] for d in range(action_dim))
        if same_bin:
            same_dist.append(dist)
        else:
            diff_dist.append(dist)

    same_dist = np.array(same_dist) if same_dist else np.array([0])
    diff_dist = np.array(diff_dist) if diff_dist else np.array([0])

    print(f"\n{'='*60}")
    print(f"  同类 vs 不同类编码距离 (n_pairs={n_pairs})")
    print(f"{'='*60}")
    print(f"  同 bin  action 的编码距离: mean={same_dist.mean():.4f}, std={same_dist.std():.4f}")
    print(f"  不同 bin action 的编码距离: mean={diff_dist.mean():.4f}, std={diff_dist.std():.4f}")
    print(f"  距离比 (diff/same): {diff_dist.mean() / max(same_dist.mean(), 1e-8):.2f}x")

    ratio = diff_dist.mean() / max(same_dist.mean(), 1e-8)
    if ratio < 1.2:
        print(f"  ⚠️  WARNING: 同类和不同类的编码距离几乎一样，编码器没有学到 action 的差异。")
    elif ratio < 1.5:
        print(f"  ⚠️  编码器有一定区分能力，但不强。")
    else:
        print(f"  ✅ 编码器能区分不同 action 区间。")

    return ratio


def test_encoding_variance(encoder, windows, n_samples=200):
    """测试编码向量各维度的方差（信息利用率）。"""
    n = min(n_samples, len(windows))
    indices = np.random.choice(len(windows), n, replace=False)
    selected = torch.from_numpy(windows[indices]).float()

    with torch.no_grad():
        encodings = encoder(selected).numpy()  # (n, hidden_dim)

    dim_var = np.var(encodings, axis=0)  # (hidden_dim,)
    total_var = np.sum(dim_var)

    # 有效维度：有多少维的方差 > 总方差的 1%
    threshold = total_var / len(dim_var) * 0.01
    active_dims = np.sum(dim_var > threshold)

    print(f"\n{'='*60}")
    print(f"  编码维度利用率 (hidden_dim={len(dim_var)})")
    print(f"{'='*60}")
    print(f"  总方差: {total_var:.6f}")
    print(f"  活跃维度 (>1% 平均方差): {active_dims}/{len(dim_var)}")
    print(f"  最大方差维度: {dim_var.max():.6f}")
    print(f"  最小方差维度: {dim_var.min():.6f}")
    print(f"  方差比 (max/min): {dim_var.max() / max(dim_var.min(), 1e-10):.1f}x")

    utilization = active_dims / len(dim_var)
    if utilization < 0.3:
        print(f"  ⚠️  WARNING: 只有 {utilization*100:.0f}% 维度被利用，信息集中在少数维度。")
    elif utilization < 0.6:
        print(f"  ⚠️  维度利用率中等: {utilization*100:.0f}%")
    else:
        print(f"  ✅ 维度利用率良好: {utilization*100:.0f}%")

    return utilization


def test_action_sensitivity(encoder, windows, action_values, n_test=100):
    """测试编码对 action 变化的敏感度。"""
    n = min(n_test, len(windows))
    windows_t = torch.from_numpy(windows[:n]).float()

    # 选择若干样本，给 action 加小扰动，看编码变化
    perturbations = [0.01, 0.05, 0.1, 0.2]

    print(f"\n{'='*60}")
    print(f"  编码对 action 扰动的敏感度")
    print(f"{'='*60}")

    with torch.no_grad():
        enc_orig = encoder(windows_t)  # (n_test, hidden_dim)

    for eps in perturbations:
        # 给最后一个时间步的 action 加扰动
        perturbed = windows_t.clone()
        perturbed[:, -1, :] += eps * (2 * torch.rand_like(perturbed[:, -1, :]) - 1)

        with torch.no_grad():
            enc_perturbed = encoder(perturbed)

        # 编码的平均相对变化
        delta = torch.norm(enc_perturbed - enc_orig, dim=-1)  # (n_test,)
        orig_norm = torch.norm(enc_orig, dim=-1)
        relative_change = (delta / orig_norm).mean().item()

        print(f"  扰动 ε={eps:.3f}: 平均编码相对变化 = {relative_change:.4f}")


def test_temporal_sensitivity(encoder, windows, window_size):
    """测试编码器对时序模式的敏感度。"""
    print(f"\n{'='*60}")
    print(f"  编码对时序模式的敏感度")
    print(f"{'='*60}")

    n = min(50, len(windows))
    windows_t = torch.from_numpy(windows[:n]).float()

    with torch.no_grad():
        enc_orig = encoder(windows_t)

    # Test 1: 打乱时间顺序
    shuffled = windows_t.clone()
    for b in range(n):
        perm = torch.randperm(window_size)
        shuffled[b] = windows_t[b, perm]

    with torch.no_grad():
        enc_shuffled = encoder(shuffled)

    cos_sim = F.cosine_similarity(enc_orig, enc_shuffled, dim=-1).mean().item()
    print(f"  原始 vs 打乱顺序: cos_sim = {cos_sim:.4f}")
    if cos_sim > 0.9:
        print(f"    ⚠️  编码器对时序顺序几乎不敏感！EMA 可能权重太均匀。")
    elif cos_sim > 0.7:
        print(f"    ⚠️  编码器对时序顺序的敏感度较低。")
    else:
        print(f"    ✅ 编码器能区分不同时序模式。")

    # Test 2: 反转时间顺序
    reversed_w = windows_t.flip(1)
    with torch.no_grad():
        enc_reversed = encoder(reversed_w)

    cos_sim_rev = F.cosine_similarity(enc_orig, enc_reversed, dim=-1).mean().item()
    print(f"  原始 vs 反转顺序: cos_sim = {cos_sim_rev:.4f}")

    # Test 3: 常数序列 vs 变化序列
    constant = windows_t.clone()
    constant[:, :, :] = windows_t[:, -1:, :]  # 所有时间步用最后一个值

    with torch.no_grad():
        enc_constant = encoder(constant)

    cos_sim_const = F.cosine_similarity(enc_orig, enc_constant, dim=-1).mean().item()
    print(f"  原始 vs 常数序列: cos_sim = {cos_sim_const:.4f}")
    if cos_sim_const > 0.8:
        print(f"    ⚠️  编码器主要依赖当前帧，历史信息利用不足。")

    # 打印当前学到的 decay rates
    decays = encoder.decays.detach().numpy()
    print(f"  当前 decay rates: {decays}")
    for i, d in enumerate(decays):
        effective_window = 1 / (1 - d)
        print(f"    Scale {i}: decay={d:.3f}, 有效窗口={effective_window:.1f} 步")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EMA 编码器区分能力测试")
    parser.add_argument("--data_dir", type=str,
                        default="data/seq_rz_c6_sk",
                        help="数据目录")
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_scales", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=200)
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    print(f"\nEMA 编码器区分能力测试")
    print(f"{'='*60}")
    print(f"  Data: {data_dir}")
    print(f"  Config: window_size={args.window_size}, hidden_dim={args.hidden_dim}, "
          f"n_scales={args.n_scales}")

    # 加载数据
    windows, action_values, labels = load_action_windows(
        data_dir, window_size=args.window_size, max_samples=args.n_samples)

    if windows is None:
        return

    action_dim = windows.shape[-1]
    print(f"  action_dim={action_dim}")

    # 创建 EMA 编码器（随机初始化，未训练）
    encoder = MultiScaleEMA(
        action_dim=action_dim,
        n_scales=args.n_scales,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
    )
    encoder.eval()

    # ── 测试 ──
    print(f"\n{'='*60}")
    print(f"  【随机初始化的 EMA 编码器】")
    print(f"{'='*60}")

    test_cosine_similarity_matrix(encoder, windows, n_samples=50)
    test_same_vs_different(encoder, windows, action_values, n_pairs=500)
    test_encoding_variance(encoder, windows)
    test_action_sensitivity(encoder, windows, action_values)
    test_temporal_sensitivity(encoder, windows, args.window_size)

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"  总结与建议")
    print(f"{'='*60}")
    print("""
  随机初始化的 EMA 编码器可能区分度不高，因为：
  1. MLP 权重随机 → 输出空间没有结构
  2. 这是正常的 — 编码器需要与下游任务一起训练

  关键问题是：EMA 结构本身是否有足够的区分能力？
  - EMA 是线性加权平均 → 表达力有限
  - 4 个不同衰减率提供 4 个视角
  - 速度特征 (Δa) 提供即时变化信息

  如果随机初始化的编码器区分度就很低（余弦相似度 > 0.9），
  说明 EMA 结构本身可能需要增强。
    """)


if __name__ == "__main__":
    main()
