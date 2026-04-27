#!/usr/bin/env python3
"""
方向5: 时序编码与粘弹性迟滞建模 — 验证与分析

验证 PyElastica 仿真器是否存在迟滞行为，实现 HA-EMA 并与标准 EMA 对比。

Usage:
    python scripts/experiments/exp5_hysteresis_analysis.py --gpu 0
    python scripts/experiments/exp5_hysteresis_analysis.py --gpu 0 --use_sim   # 需要仿真器
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--data_dir', type=str, default='data/seq_rr_3d')
parser.add_argument('--output_dir', type=str, default='output/exp5_hysteresis')
parser.add_argument('--use_sim', action='store_true', help='Use simulator (requires PyElastica)')
parser.add_argument('--train_epochs', type=int, default=60, help='Training epochs for HA-EMA comparison')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向5: 迟滞分析 ===')
print(f'Device: {device}')
print(f'Output: {args.output_dir}')

# ═══════════════════════════════════════════════════════════════
# Part 1: 仿真器迟滞行为验证
# ═══════════════════════════════════════════════════════════════

def run_simulator_hysteresis():
    """使用仿真器运行三角波协议，验证迟滞行为。"""
    print('\n--- Part 1: 仿真器迟滞验证 ---')
    try:
        from elastica_env import ContinuousSoftArmEnv
    except Exception as e:
        print(f'无法导入仿真器: {e}')
        return None

    results = {}
    for freq in [0.5, 1.0, 2.0]:
        env = ContinuousSoftArmEnv(dt=1e-4)
        period = 1.0 / freq
        sim_steps_per_action = 200
        n_cycles = 3
        n_steps = int(n_cycles * period / (sim_steps_per_action * env.dt))

        torques, tip_x, tip_z = [], [], []
        for step in range(n_steps):
            t = step * sim_steps_per_action * env.dt
            # 三角波: 线性增加再线性减少
            phase = (t % period) / period
            if phase < 0.5:
                torque = 0.3 * (2 * phase)
            else:
                torque = 0.3 * (2 - 2 * phase)

            action = np.array([torque, 0.0])
            env.set_action(action)
            for _ in range(sim_steps_per_action):
                env.step(steps=1)

            pos = env.simulator[0].rod.position_collection.numpy()
            tip_x.append(pos[0, -1])
            tip_z.append(pos[2, -1])
            torques.append(torque)

        results[freq] = {
            'torques': np.array(torques),
            'tip_x': np.array(tip_x),
            'tip_z': np.array(tip_z),
        }
        print(f'  频率 {freq} Hz: {n_steps} 步, tip_x range=[{min(tip_x):.4f}, {max(tip_x):.4f}]')

    # 绘制迟滞环
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (freq, data) in enumerate(results.items()):
        ax = axes[i]
        torques = data['torques']
        tip_x = data['tip_x']
        # 用颜色表示时间
        colors = plt.cm.viridis(np.linspace(0, 1, len(torques)))
        for j in range(len(torques) - 1):
            ax.plot(torques[j:j+2], tip_x[j:j+2], color=colors[j], linewidth=1.5)
        ax.set_xlabel('Torque (N·m)')
        ax.set_ylabel('Tip X (m)')
        ax.set_title(f'f={freq} Hz')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hysteresis Loops (Simulator Triangular Wave)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'sim_hysteresis_loops.png'), dpi=150)
    plt.close()
    print(f'  保存: sim_hysteresis_loops.png')

    # 计算迟滞环面积
    for freq, data in results.items():
        torques = data['torques']
        tip_x = data['tip_x']
        # 取最后一个完整周期
        period_steps = len(torques) // 3
        t_last = torques[-period_steps:]
        x_last = tip_x[-period_steps:]
        # 加载和卸载分支
        mid = period_steps // 2
        loading = tip_x[-period_steps:-period_steps+mid]
        loading_t = torques[-period_steps:-period_steps+mid]
        unloading = tip_x[-period_steps+mid:]
        unloading_t = torques[-period_steps+mid:]
        area = np.abs(np.trapz(loading, loading_t) + np.trapz(unloading, unloading_t))
        print(f'  f={freq} Hz 迟滞环面积: {area:.6f}')

    return results


def analyze_data_hysteresis():
    """从已有 3D 数据中分析迟滞行为。"""
    print('\n--- Part 1b: 数据迟滞分析 ---')
    import glob
    files = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    if not files:
        print(f'未找到数据: {args.data_dir}')
        return None

    all_torques, all_tip_x, all_tip_z = [], [], []
    for f in files[:3]:
        d = np.load(f, allow_pickle=True)
        actions = d['actions']  # (500, 2)
        positions = d['positions']  # (500, 3, 31)

        tip_x = positions[:, 0, -1]  # tip x 坐标
        tip_z = positions[:, 2, -1]  # tip z 坐标
        torques_x = actions[:, 0]    # x 方向扭矩

        all_torques.append(torques_x)
        all_tip_x.append(tip_x)
        all_tip_z.append(tip_z)

    torques = np.concatenate(all_torques)
    tip_x = np.concatenate(all_tip_x)
    tip_z = np.concatenate(all_tip_z)

    print(f'  数据量: {len(torques)} 帧')
    print(f'  扭矩范围: [{torques.min():.6f}, {torques.max():.6f}]')
    print(f'  Tip X 范围: [{tip_x.min():.4f}, {tip_x.max():.4f}]')

    # 迟滞分析：扭矩变化方向 vs 位置
    torque_diff = np.diff(torques)
    loading_mask = torque_diff > 0
    unloading_mask = torque_diff < 0

    # 绘制扭矩-位置图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 扭矩-位置散点图（按时间着色）
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(torques)))
    ax.scatter(torques, tip_x, c=np.arange(len(torques)), cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('Torque X')
    ax.set_ylabel('Tip X (m)')
    ax.set_title('Torque vs Tip Position (colored by time)')
    ax.grid(True, alpha=0.3)

    # 加载/卸载分支
    ax = axes[1]
    ax.scatter(torques[:-1][loading_mask], tip_x[:-1][loading_mask],
               c='red', s=1, alpha=0.3, label='Loading')
    ax.scatter(torques[:-1][unloading_mask], tip_x[:-1][unloading_mask],
               c='blue', s=1, alpha=0.3, label='Unloading')
    ax.set_xlabel('Torque X')
    ax.set_ylabel('Tip X (m)')
    ax.set_title('Loading vs Unloading')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 时间序列
    ax = axes[2]
    t = np.arange(len(torques))
    ax.plot(t, torques, 'r-', alpha=0.7, label='Torque')
    ax2 = ax.twinx()
    ax2.plot(t, tip_x, 'b-', alpha=0.7, label='Tip X')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Torque', color='red')
    ax2.set_ylabel('Tip X (m)', color='blue')
    ax.set_title('Time Series')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hysteresis Analysis from Data', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'data_hysteresis_analysis.png'), dpi=150)
    plt.close()
    print(f'  保存: data_hysteresis_analysis.png')

    return {'torques': torques, 'tip_x': tip_x, 'tip_z': tip_z}


# ═══════════════════════════════════════════════════════════════
# Part 2: HA-EMA 实现
# ═══════════════════════════════════════════════════════════════

class StandardEMA(nn.Module):
    """标准 MultiScaleEMA（基线）。"""
    def __init__(self, action_dim=2, n_scales=4, window_size=20, hidden_dim=128):
        super().__init__()
        self.n_scales = n_scales
        self.window_size = window_size
        self.raw_decays = nn.Parameter(torch.linspace(-1.5, 1.5, n_scales))
        self.state_mlp = nn.Sequential(
            nn.Linear(n_scales * action_dim + action_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action_window):
        B, K, D = action_window.shape
        decays = torch.sigmoid(self.raw_decays)
        ema_features = []
        for s in range(self.n_scales):
            state = torch.zeros(B, D, device=action_window.device)
            for t in range(K):
                state = decays[s] * state + (1 - decays[s]) * action_window[:, t]
            ema_features.append(state)
        ema_concat = torch.cat(ema_features, dim=-1)
        velocity = action_window[:, -1] - action_window[:, -2] if K > 1 else torch.zeros(B, D, device=action_window.device)
        features = torch.cat([ema_concat, action_window[:, -1], velocity], dim=-1)
        return self.state_mlp(features)


class HysteresisAwareEMA(nn.Module):
    """迟滞感知 EMA — 加载/卸载使用不同的 decay rate。"""
    def __init__(self, action_dim=2, n_scales=4, window_size=20, hidden_dim=128):
        super().__init__()
        self.n_scales = n_scales
        self.window_size = window_size
        # 两组 decay rate: 加载和卸载
        self.raw_decays_load = nn.Parameter(torch.linspace(-1.5, 1.0, n_scales))
        self.raw_decays_unload = nn.Parameter(torch.linspace(-1.0, 1.5, n_scales))
        # 方向门控网络
        self.direction_gate = nn.Sequential(
            nn.Linear(action_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, n_scales),
            nn.Sigmoid(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(n_scales * action_dim + action_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action_window):
        B, K, D = action_window.shape
        velocity = action_window[:, -1] - action_window[:, -2] if K > 1 else torch.zeros(B, D, device=action_window.device)
        # 方向门控
        gate_input = torch.cat([action_window[:, -1], velocity], dim=-1)
        gate = self.direction_gate(gate_input)  # (B, n_scales)
        decays_load = torch.sigmoid(self.raw_decays_load)    # (n_scales,)
        decays_unload = torch.sigmoid(self.raw_decays_unload)
        # 动态 decay: 加权混合
        decays = gate * decays_load.unsqueeze(0) + (1 - gate) * decays_unload.unsqueeze(0)  # (B, n_scales)

        ema_features = []
        for s in range(self.n_scales):
            state = torch.zeros(B, D, device=action_window.device)
            for t in range(K):
                alpha = decays[:, s].unsqueeze(1)  # (B, 1)
                state = alpha * state + (1 - alpha) * action_window[:, t]
            ema_features.append(state)

        ema_concat = torch.cat(ema_features, dim=-1)
        features = torch.cat([ema_concat, action_window[:, -1], velocity], dim=-1)
        return self.state_mlp(features)


# ═══════════════════════════════════════════════════════════════
# Part 3: 训练对比实验
# ═══════════════════════════════════════════════════════════════

def prepare_training_data():
    """准备时序训练数据。"""
    from src.data.dataset import SoftSequenceDataset
    ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True)

    # 收集所有数据
    all_windows, all_positions = [], []
    for batch in loader:
        windows = batch[0]  # (B, 20, 2)
        positions = batch[-1]  # (B, 3, 31)
        all_windows.append(windows)
        all_positions.append(positions)

    windows = torch.cat(all_windows, dim=0)
    positions = torch.cat(all_positions, dim=0)  # (N, 3, 31)
    positions = positions.permute(0, 2, 1)  # (N, 31, 3)

    print(f'  训练数据: {windows.shape[0]} 样本')
    return windows, positions


class SkeletonPredictor(nn.Module):
    """简单的骨架预测器：EMA → skeleton。"""
    def __init__(self, ema_module, hidden_dim=128, n_nodes=31):
        super().__init__()
        self.ema = ema_module
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_nodes * 3),
        )
        self.n_nodes = n_nodes

    def forward(self, action_window):
        state = self.ema(action_window)
        skeleton = self.head(state).reshape(-1, self.n_nodes, 3)
        return skeleton


def train_and_compare():
    """训练标准 EMA 和 HA-EMA，对比骨架预测精度。"""
    print('\n--- Part 3: EMA vs HA-EMA 训练对比 ---')
    windows, positions = prepare_training_data()
    windows, positions = windows.to(device), positions.to(device)

    n_train = int(0.8 * len(windows))
    train_w, test_w = windows[:n_train], windows[n_train:]
    train_p, test_p = positions[:n_train], positions[n_train:]
    print(f'  训练: {n_train}, 测试: {len(test_w)}')

    results = {}
    for name, EMAClass in [('StandardEMA', StandardEMA), ('HA-EMA', HysteresisAwareEMA)]:
        print(f'\n  训练 {name}...')
        ema = EMAClass(action_dim=2, n_scales=4, window_size=20, hidden_dim=128).to(device)
        model = SkeletonPredictor(ema, hidden_dim=128, n_nodes=31).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train_losses, test_losses = [], []
        for epoch in range(args.train_epochs):
            model.train()
            perm = torch.randperm(n_train)
            epoch_loss = 0
            n_batches = 0
            batch_size = 32
            for i in range(0, n_train, batch_size):
                idx = perm[i:i+batch_size]
                pred = model(train_w[idx])
                loss = F.mse_loss(pred, train_p[idx])
                # 平滑性正则化
                diff2 = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
                loss += 0.01 * (diff2 ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # 测试
            model.eval()
            with torch.no_grad():
                pred_test = []
                for i in range(0, len(test_w), 64):
                    pred_test.append(model(test_w[i:i+64]))
                pred_test = torch.cat(pred_test)
                test_loss = F.mse_loss(pred_test, test_p).item()
                mne = (pred_test - test_p).norm(dim=-1).mean().item()

            train_losses.append(epoch_loss / n_batches)
            test_losses.append(test_loss)
            if (epoch + 1) % 10 == 0:
                print(f'    Epoch {epoch+1}: train={train_losses[-1]:.6f}, '
                      f'test_mse={test_loss:.6f}, test_mne={mne:.6f}m')

        # 最终评估
        model.eval()
        with torch.no_grad():
            pred_test = []
            for i in range(0, len(test_w), 64):
                pred_test.append(model(test_w[i:i+64]))
            pred_test = torch.cat(pred_test)

            mne = (pred_test - test_p).norm(dim=-1).mean().item()
            tip_err = (pred_test[:, -1] - test_p[:, -1]).norm(dim=-1).mean().item()
            diff2 = pred_test[:, 2:] - 2 * pred_test[:, 1:-1] + pred_test[:, :-2]
            smooth = (diff2 ** 2).sum(-1).sqrt().mean().item()

        results[name] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'mne': mne,
            'tip_err': tip_err,
            'smooth': smooth,
            'model': model,
            'pred_test': pred_test.cpu().numpy(),
            'gt_test': test_p.cpu().numpy(),
        }
        print(f'  {name} 最终: MNE={mne:.6f}m, Tip={tip_err:.6f}m, Smooth={smooth:.6f}')

    return results


def plot_comparison(results):
    """绘制 EMA vs HA-EMA 对比图。"""
    print('\n--- Part 4: 结果可视化 ---')

    # 1. 训练曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in results.items():
        axes[0].plot(r['train_losses'], label=name)
        axes[1].plot(r['test_losses'], label=name)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test MSE')
    axes[1].set_title('Test Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # 2. 柱状图对比指标
    names = list(results.keys())
    metrics = ['mne', 'tip_err', 'smooth']
    metric_labels = ['MNE (m)', 'Tip Error (m)', 'Smoothness']
    x = np.arange(len(metrics))
    width = 0.3
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        axes[2].bar(x + i * width, vals, width, label=name, alpha=0.8)
    axes[2].set_xticks(x + width / 2)
    axes[2].set_xticklabels(metric_labels)
    axes[2].set_title('Metrics Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Standard EMA vs HA-EMA', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'ema_comparison.png'), dpi=150)
    plt.close()

    # 3. 骨架对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': '3d'})
    for idx, (name, r) in enumerate(results.items()):
        ax = axes[idx]
        gt = r['gt_test']
        pred = r['pred_test']
        n_show = min(5, len(gt))
        for i in range(n_show):
            offset = i * 0.02
            ax.plot(gt[i, :, 0] + offset, gt[i, :, 1], gt[i, :, 2],
                    'b-', alpha=0.4, linewidth=1)
            ax.plot(pred[i, :, 0] + offset, pred[i, :, 1], pred[i, :, 2],
                    'r-', alpha=0.6, linewidth=1.5)
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.15, 0.15)
        ax.set_zlim(0, 0.55)
        ax.set_title(f'{name}\nMNE={r["mne"]:.4f}m')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.suptitle('Skeleton Prediction: GT (blue) vs Pred (red)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'skeleton_comparison.png'), dpi=150)
    plt.close()

    # 4. 学习到的 decay rates
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, r in results.items():
        ema = r['model'].ema
        if hasattr(ema, 'raw_decays'):
            decays = torch.sigmoid(ema.raw_decays).detach().cpu().numpy()
            axes[0].bar(np.arange(len(decays)) - 0.15, decays, 0.3, label=name, alpha=0.8)
            axes[1].bar(np.arange(len(decays)) - 0.15, decays, 0.3, label=name, alpha=0.8)
        elif hasattr(ema, 'raw_decays_load'):
            d_load = torch.sigmoid(ema.raw_decays_load).detach().cpu().numpy()
            d_unload = torch.sigmoid(ema.raw_decays_unload).detach().cpu().numpy()
            x_pos = np.arange(len(d_load))
            axes[0].bar(x_pos - 0.15, d_load, 0.3, label='Loading', color='red', alpha=0.8)
            axes[0].bar(x_pos + 0.15, d_unload, 0.3, label='Unloading', color='blue', alpha=0.8)
            axes[1].bar(x_pos - 0.15, d_load, 0.3, label='Loading', color='red', alpha=0.8)
            axes[1].bar(x_pos + 0.15, d_unload, 0.3, label='Unloading', color='blue', alpha=0.8)

    axes[0].set_xlabel('Scale Index')
    axes[0].set_ylabel('Decay Rate (α)')
    axes[0].set_title('Learned Decay Rates')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 方向门控分布
    for name, r in results.items():
        ema = r['model'].ema
        if hasattr(ema, 'direction_gate'):
            with torch.no_grad():
                test_w_cpu = torch.tensor(r['gt_test'][:, :1, :]).to(device)  # dummy
            # 用实际测试数据
            from src.data.dataset import SoftSequenceDataset
            ds = SoftSequenceDataset(args.data_dir, seq_len=20, return_3d=True)
            sample = ds[0]
            aw = sample[0].unsqueeze(0).to(device)
            velocity = aw[:, -1] - aw[:, -2]
            gate_input = torch.cat([aw[:, -1], velocity], dim=-1)
            gate = ema.direction_gate(gate_input).cpu().numpy()[0]
            axes[1].bar(np.arange(len(gate)) + 0.15, gate, 0.3, label='Gate (HA-EMA)', color='green', alpha=0.8)

    axes[1].set_xlabel('Scale Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Direction Gate & Decay Rates')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'learned_parameters.png'), dpi=150)
    plt.close()

    # 保存摘要
    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write('=== 方向5: 迟滞分析结果 ===\n\n')
        for name, r in results.items():
            f.write(f'{name}:\n')
            f.write(f'  MNE:         {r["mne"]:.6f} m\n')
            f.write(f'  Tip Error:   {r["tip_err"]:.6f} m\n')
            f.write(f'  Smoothness:  {r["smooth"]:.6f}\n\n')

    print(f'  所有结果保存到: {args.output_dir}/')
    for name, r in results.items():
        print(f'  {name}: MNE={r["mne"]:.4f}m, Tip={r["tip_err"]:.4f}m')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Part 1: 迟滞验证
    if args.use_sim:
        sim_results = run_simulator_hysteresis()
    data_results = analyze_data_hysteresis()

    # Part 2-4: 训练对比
    results = train_and_compare()
    plot_comparison(results)

    print('\n=== 完成 ===')
    print(f'查看结果: ls {args.output_dir}/')
