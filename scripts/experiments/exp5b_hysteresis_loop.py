#!/usr/bin/env python3
"""
方向5b: 单轴迟滞效应可视化 — 加载-卸载循环

方法：
  固定 Y 轴扭矩 = 0，只在 X 轴做完整循环：
    阶段1: 0 → +τ_max     加载
    阶段2: +τ_max → 0     卸载
    阶段3: 0 → -τ_max     反向加载
    阶段4: -τ_max → 0     反向卸载

  画图: 横轴 = X轴扭矩, 纵轴 = X轴末端位移
  如果加载和卸载路径不重合 → 存在迟滞（黏弹性滞后）

Usage:
    python scripts/experiments/exp5b_hysteresis_loop.py --gpu 0
    python scripts/experiments/exp5b_hysteresis_loop.py --axis y --tau_max 0.003
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='output/exp5b_hysteresis_loop')
parser.add_argument('--n_steps_per_phase', type=int, default=80,
                    help='每个阶段的步数')
parser.add_argument('--sim_steps_per_action', type=int, default=400,
                    help='每个 action 对应的物理仿真子步数')
parser.add_argument('--settle_steps', type=int, default=500,
                    help='每次改扭矩后等待稳定的仿真步数')
parser.add_argument('--tau_max', type=float, default=0.005,
                    help='驱动轴最大扭矩 (N·m)，与 simulation.json action_max 一致')
parser.add_argument('--axis', type=str, default='x', choices=['x', 'y'],
                    help='驱动轴方向 (另一个轴固定为 0)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.makedirs(args.output_dir, exist_ok=True)

DRIVE = 0 if args.axis == 'x' else 1
RESPONSE = 1 if DRIVE == 0 else 0  # 扭矩 X → 弯曲 Y, 扭矩 Y → 弯曲 X
AXIS_NAME = 'X' if DRIVE == 0 else 'Y'
RESPONSE_NAME = 'Y' if RESPONSE == 1 else 'X'

print(f'=== 方向5b: 单轴迟滞效应可视化 ({AXIS_NAME}轴) ===')
print(f'Output: {args.output_dir}')


def generate_single_axis_cycle(n_steps, tau_max, drive_axis):
    """生成四阶段单轴循环扭矩序列。"""
    phase_labels = [
        f'Load: 0→+{tau_max}',
        f'Unload: +{tau_max}→0',
        f'Load: 0→-{tau_max}',
        f'Unload: -{tau_max}→0',
    ]

    targets = [tau_max, 0, -tau_max, 0]
    starts = [0, tau_max, 0, -tau_max]

    all_actions = []
    all_phase_ids = []

    for p, (s, e) in enumerate(zip(starts, targets)):
        t = np.linspace(0, 1, n_steps, endpoint=False)
        values = s + (e - s) * t
        actions = np.zeros((n_steps, 2))
        actions[:, drive_axis] = values
        all_actions.append(actions)
        all_phase_ids.append(np.full(n_steps, p))

    actions = np.concatenate(all_actions, axis=0)
    phase_ids = np.concatenate(all_phase_ids)
    return actions, phase_ids, phase_labels


def run_continuous_cycle(actions, sim_steps_per_action, dt=1e-4):
    """在同一个仿真器中连续执行扭矩序列，记录每步末端位置。"""
    from elastica_env import ContinuousSoftArmEnv

    env = ContinuousSoftArmEnv(dt=dt)

    tip_positions = []
    settle = getattr(args, 'settle_steps', 2000)
    for i, action in enumerate(actions):
        env.set_action(action)
        env.step(steps=sim_steps_per_action)

        # 等待臂稳定后记录
        env.step(steps=settle)
        pos = env.simulation[0].position_collection.copy()
        tip_positions.append([pos[0, -1], pos[1, -1], pos[2, -1]])

        if (i + 1) % args.n_steps_per_phase == 0:
            phase = i // args.n_steps_per_phase
            tau = action
            tip = tip_positions[-1]
            print(f'    phase {phase} end, step {i+1}/{len(actions)}: '
                  f'tau_{AXIS_NAME.lower()}={tau[DRIVE]:+.4f}  '
                  f'tip=({tip[0]:.5f}, {tip[1]:.5f}, {tip[2]:.5f})')

    return np.array(tip_positions)


PHASE_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']


def plot_hysteresis(tip_positions, phase_ids, actions, phase_labels):
    """绘制迟滞回线。"""
    n_total = len(tip_positions)
    n_per_phase = n_total // 4

    drive_disp = tip_positions[:, RESPONSE]
    torque = actions[:, DRIVE]

    # ── 图1: 核心迟滞图 ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for p in range(4):
        mask = phase_ids == p
        ax.plot(torque[mask], drive_disp[mask], '-o', color=PHASE_COLORS[p],
                markersize=2, linewidth=2, label=phase_labels[p])

    key_indices = [0, n_per_phase - 1, 2 * n_per_phase - 1, 3 * n_per_phase - 1]
    for idx in key_indices:
        ax.plot(torque[idx], drive_disp[idx], 'k*', markersize=12, zorder=10)

    # 迟滞面积
    load_disp = drive_disp[phase_ids == 0]
    unload_disp = drive_disp[phase_ids == 1][::-1]
    n_common = min(len(load_disp), len(unload_disp))
    area = 0
    if n_common > 2:
        load_tau = torque[phase_ids == 0][:n_common]
        loop_disp = np.concatenate([load_disp[:n_common], unload_disp[:n_common][::-1]])
        loop_tau = np.concatenate([load_tau, load_tau[::-1]])
        ax.fill(loop_tau, loop_disp, alpha=0.15, color='purple', label='Hysteresis area')
        area = np.abs(np.trapz(load_disp[:n_common] - unload_disp[:n_common],
                               load_tau[:n_common]))

    ax.set_xlabel(f'{AXIS_NAME}-axis Torque (N·m)', fontsize=13)
    ax.set_ylabel(f'{RESPONSE_NAME}-axis Tip Displacement (m)', fontsize=13)
    ax.set_title(f'Hysteresis Loop — {AXIS_NAME} torque → {RESPONSE_NAME} displacement', fontsize=14)
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # ── 图2: 加载 vs 反向半周期 ──
    ax = axes[1]
    load_half = (phase_ids == 0) | (phase_ids == 1)
    ax.plot(torque[load_half], drive_disp[load_half], '-o',
            color='#e74c3c', markersize=2, linewidth=2, label='Positive half (0→+τ→0)')

    reverse_half = (phase_ids == 2) | (phase_ids == 3)
    ax.plot(torque[reverse_half], drive_disp[reverse_half], '-o',
            color='#3498db', markersize=2, linewidth=2, label='Negative half (0→-τ→0)')

    ax.set_xlabel(f'{AXIS_NAME}-axis Torque (N·m)', fontsize=13)
    ax.set_ylabel(f'{RESPONSE_NAME}-axis Tip Displacement (m)', fontsize=13)
    ax.set_title('Positive vs Negative half cycles', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'hysteresis_loop.png'), dpi=150)
    plt.close()
    print(f'  保存: hysteresis_loop.png')

    # ── 图3: 时间序列 ──
    fig, axes2 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = np.arange(n_total)

    ax = axes2[0]
    ax.plot(t, actions[:, 0], label='τ_x', linewidth=1.5)
    ax.plot(t, actions[:, 1], label='τ_y', linewidth=1.5)
    for p in range(4):
        x_start = p * n_per_phase
        ax.axvspan(x_start, x_start + n_per_phase, alpha=0.08, color=PHASE_COLORS[p])
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Torque Input ({AXIS_NAME} axis driven, {RESPONSE_NAME}=0)', fontsize=13)

    ax = axes2[1]
    ax.plot(t, tip_positions[:, 0], label='Tip X', linewidth=1.5)
    ax.plot(t, tip_positions[:, 1], label='Tip Y', linewidth=1.5)
    ax.plot(t, tip_positions[:, 2], label='Tip Z', linewidth=1.5, alpha=0.7)
    for p in range(4):
        x_start = p * n_per_phase
        ax.axvspan(x_start, x_start + n_per_phase, alpha=0.08, color=PHASE_COLORS[p])
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Position (m)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'time_series.png'), dpi=150)
    plt.close()
    print(f'  保存: time_series.png')

    return area


if __name__ == '__main__':
    actions, phase_ids, phase_labels = generate_single_axis_cycle(
        args.n_steps_per_phase, args.tau_max, DRIVE)
    n_total = len(actions)

    print(f'\n驱动轴: {AXIS_NAME}, 最大扭矩: ±{args.tau_max} N·m')
    print(f'{RESPONSE_NAME}轴: 固定为 0')
    print(f'循环: 0 → +{args.tau_max} → 0 → -{args.tau_max} → 0')
    print(f'总步数: {n_total}, 每步仿真 {args.sim_steps_per_action} 子步')

    print(f'\n运行连续循环...')
    tip_positions = run_continuous_cycle(actions, args.sim_steps_per_action)

    drive_disp = tip_positions[:, RESPONSE]
    print(f'\n末端 {RESPONSE_NAME} 位移范围: [{drive_disp.min():.6f}, {drive_disp.max():.6f}]')

    residual = np.linalg.norm(tip_positions[-1] - tip_positions[0])
    print(f'  终点 vs 起点偏移: {residual:.6f} m')

    print(f'\n生成可视化...')
    area = plot_hysteresis(tip_positions, phase_ids, actions, phase_labels)
    print(f'  迟滞面积 (正半周期): {area:.8f} m*N*m')

    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write(f'=== 方向5b: 单轴迟滞回线分析 ({AXIS_NAME}轴) ===\n\n')
        f.write(f'参数: axis={args.axis}, tau_max={args.tau_max}, '
                f'n_steps_per_phase={args.n_steps_per_phase}, '
                f'sim_substeps={args.sim_steps_per_action}\n')
        f.write(f'驱动轴: {AXIS_NAME}, 从动轴: {RESPONSE_NAME}=0\n')
        f.write(f'循环: 0 → +{args.tau_max} → 0 → -{args.tau_max} → 0\n\n')
        f.write(f'末端位置范围:\n')
        f.write(f'  X: [{tip_positions[:,0].min():.6f}, {tip_positions[:,0].max():.6f}]\n')
        f.write(f'  Y: [{tip_positions[:,1].min():.6f}, {tip_positions[:,1].max():.6f}]\n')
        f.write(f'  Z: [{tip_positions[:,2].min():.6f}, {tip_positions[:,2].max():.6f}]\n')
        f.write(f'终点 vs 起点偏移: {residual:.6f} m\n')
        f.write(f'迟滞面积 (正半周期): {area:.8f} m*N*m\n')

    print(f'\n=== 完成 ===')
    print(f'查看结果: ls {args.output_dir}/')
