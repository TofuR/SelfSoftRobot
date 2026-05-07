#!/usr/bin/env python3
"""
方向5b: 迟滞效应可视化 — 连续加载-卸载-反向加载循环

方法：
  在同一个仿真器实例中（不重置），连续执行：
    阶段1: (0, 0) → (τ_max, τ_max)   加载
    阶段2: (τ_max, τ_max) → (-τ_max, -τ_max)  反向卸载+加载
    阶段3: (-τ_max, -τ_max) → (τ_max, τ_max)  再次反向
  全程记录尖端 (x, y) 坐标，绘制 tip_x vs tip_y。
  正反向路径不重合围出的区域 = 迟滞效应。

Usage:
    python scripts/experiments/exp5b_hysteresis_loop.py --gpu 0
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
parser.add_argument('--n_steps_per_phase', type=int, default=60,
                    help='每个阶段的步数')
parser.add_argument('--sim_steps_per_action', type=int, default=400,
                    help='每个 action 对应的物理仿真子步数')
parser.add_argument('--tau_x_max', type=float, default=0.4,
                    help='X 方向最大扭矩 (N·m)')
parser.add_argument('--tau_y_max', type=float, default=0.25,
                    help='Y 方向最大扭矩 (N·m)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.makedirs(args.output_dir, exist_ok=True)

print(f'=== 方向5b: 迟滞效应可视化 ===')
print(f'Output: {args.output_dir}')


# ═══════════════════════════════════════════════════════════════
# 生成连续循环扭矩序列
# ═══════════════════════════════════════════════════════════════

def ease_curve(n, exponent):
    """生成非线性缓动曲线 t^exponent，t ∈ [0,1]。

    exponent > 1: 先慢后快（ease-in）
    exponent < 1: 先快后慢（ease-out）
    """
    t = np.linspace(0, 1, n)
    return t ** exponent


def generate_cyclic_actions(n_steps, tau_x_max, tau_y_max):
    """生成三阶段连续循环扭矩序列。

    X 方向: 峰值 τ_x_max，先快后慢 (exponent=0.5, ease-out)
    Y 方向: 峰值 τ_y_max，先慢后快 (exponent=2.0, ease-in)

    阶段1: (0,0) → (τ_x_max, τ_y_max)
    阶段2: (τ_x_max, τ_y_max) → (-τ_x_max, -τ_y_max)
    阶段3: (-τ_x_max, -τ_y_max) → (τ_x_max, τ_y_max)

    Returns:
        actions: (3*n_steps, 2) 连续扭矩序列
        phase_ids: (3*n_steps,) 每步所属的阶段编号 (0,1,2)
    """
    # X: 先快后慢 (exponent < 1)
    ease_x = ease_curve(n_steps, exponent=0.5)
    # Y: 先慢后快 (exponent > 1)
    ease_y = ease_curve(n_steps, exponent=2.0)

    phase1 = np.column_stack([
        tau_x_max * ease_x,
        tau_y_max * ease_y,
    ])
    phase2 = np.column_stack([
        tau_x_max - 2 * tau_x_max * ease_x,
        tau_y_max - 2 * tau_y_max * ease_y,
    ])
    phase3 = np.column_stack([
        -tau_x_max + 2 * tau_x_max * ease_x,
        -tau_y_max + 2 * tau_y_max * ease_y,
    ])
    actions = np.concatenate([phase1, phase2, phase3], axis=0)
    phase_ids = np.concatenate([
        np.zeros(n_steps, dtype=int),
        np.ones(n_steps, dtype=int),
        np.full(n_steps, 2, dtype=int),
    ])
    return actions, phase_ids


# ═══════════════════════════════════════════════════════════════
# 仿真运行（单一连续实例）
# ═══════════════════════════════════════════════════════════════

def run_continuous_cycle(actions, sim_steps_per_action, dt=1e-4):
    """在同一个仿真器中连续执行扭矩序列，记录每步尖端坐标。

    Returns:
        tip_xy: (N, 2) 尖端 (x, y) 坐标序列
    """
    from elastica_env import ContinuousSoftArmEnv

    env = ContinuousSoftArmEnv(dt=dt)

    tip_xy = []
    for i, action in enumerate(actions):
        env.set_action(action)
        env.step(steps=sim_steps_per_action)

        pos = env.simulation[0].position_collection.copy()
        tip_xy.append([pos[0, -1], pos[1, -1]])

        if (i + 1) % len(actions) // 3 == 0:
            tau = action
            tip = tip_xy[-1]
            print(f'    step {i+1}/{len(actions)}: τ=({tau[0]:+.4f}, {tau[1]:+.4f})  '
                  f'tip=({tip[0]:.5f}, {tip[1]:.5f})')

    return np.array(tip_xy)


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

PHASE_COLORS = ['#e74c3c', '#3498db', '#2ecc71']
PHASE_LABELS = [
    'Phase 1: (0,0)→(τ_x_max,τ_y_max)',
    'Phase 2: (τ_x_max,τ_y_max)→(-τ_x_max,-τ_y_max)',
    'Phase 3: (-τ_x_max,-τ_y_max)→(τ_x_max,τ_y_max)',
]


def plot_hysteresis(tip_xy, phase_ids, actions):
    """绘制迟滞回线。"""
    n_total = len(tip_xy)
    n_per_phase = n_total // 3

    # ── 图1: tip_x vs tip_y，按阶段着色 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for p in range(3):
        mask = phase_ids == p
        pts = tip_xy[mask]
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=PHASE_COLORS[p], markersize=2.5,
                linewidth=1.8, label=PHASE_LABELS[p], zorder=3 + p)

    # 标记关键点
    ax.plot(*tip_xy[0], 'k*', markersize=14, label='Start', zorder=10)
    ax.plot(*tip_xy[n_per_phase - 1], 'r^', markersize=10, label='Peak (+)', zorder=10)
    ax.plot(*tip_xy[2 * n_per_phase - 1], 'bv', markersize=10, label='Peak (-)', zorder=10)
    ax.plot(*tip_xy[-1], 'gs', markersize=10, label='End', zorder=10)

    ax.set_xlabel('Tip X (m)', fontsize=12)
    ax.set_ylabel('Tip Y (m)', fontsize=12)
    ax.set_title('Hysteresis Loop — Tip X vs Tip Y', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # ── 图2: |扭矩| vs |位移|（经典迟滞图） ──
    ax = axes[1]
    tau_mag = np.linalg.norm(actions, axis=-1)
    disp_mag = np.linalg.norm(tip_xy, axis=-1)

    for p in range(3):
        mask = phase_ids == p
        ax.plot(tau_mag[mask], disp_mag[mask], '-o', color=PHASE_COLORS[p],
                markersize=2.5, linewidth=1.8, label=PHASE_LABELS[p])

    ax.set_xlabel('|Torque| (N·m)', fontsize=12)
    ax.set_ylabel('|Tip Displacement| (m)', fontsize=12)
    ax.set_title('Torque vs Displacement', fontsize=13)
    ax.legend(fontsize=8, loc='best')
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
    for p in range(3):
        x_start = p * n_per_phase
        ax.axvspan(x_start, x_start + n_per_phase, alpha=0.08, color=PHASE_COLORS[p])
    ax.set_ylabel('Torque (N·m)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Torque Input vs Tip Response', fontsize=13)

    ax = axes2[1]
    ax.plot(t, tip_xy[:, 0], label='Tip X', linewidth=1.5)
    ax.plot(t, tip_xy[:, 1], label='Tip Y', linewidth=1.5)
    disp = np.linalg.norm(tip_xy, axis=-1)
    ax.plot(t, disp, '--', label='|Tip|', linewidth=1.2, alpha=0.7)
    for p in range(3):
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

    # ── 图4: 放大迟滞回线 + 填充面积 ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for p in range(3):
        mask = phase_ids == p
        pts = tip_xy[mask]
        ax.plot(pts[:, 0], pts[:, 1], '-o', color=PHASE_COLORS[p], markersize=3,
                linewidth=2, label=PHASE_LABELS[p], zorder=3 + p)

    # 阶段2和阶段3之间围出的区域就是迟滞
    p2_pts = tip_xy[phase_ids == 2]
    p1_pts = tip_xy[phase_ids == 1]
    n_common = min(len(p2_pts), len(p1_pts))
    if n_common > 2:
        loop_x = np.concatenate([p1_pts[:n_common, 0], p2_pts[:n_common, 0][::-1]])
        loop_y = np.concatenate([p1_pts[:n_common, 1], p2_pts[:n_common, 1][::-1]])
        ax.fill(loop_x, loop_y, alpha=0.15, color='purple', label='Hysteresis area', zorder=2)

    ax.plot(*tip_xy[0], 'k*', markersize=16, label='Start', zorder=10)
    ax.set_xlabel('Tip X (m)', fontsize=13)
    ax.set_ylabel('Tip Y (m)', fontsize=13)
    ax.set_title('Hysteresis Loop — Detail', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'hysteresis_area_detail.png'), dpi=150)
    plt.close()
    print(f'  保存: hysteresis_area_detail.png')


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    actions, phase_ids = generate_cyclic_actions(args.n_steps_per_phase, args.tau_x_max, args.tau_y_max)
    n_total = len(actions)

    print(f'\nX方向: 峰值={args.tau_x_max}, 先快后慢 (√t)')
    print(f'Y方向: 峰值={args.tau_y_max}, 先慢后快 (t²)')
    print(f'扭矩序列: ({actions[0]}) → ({actions[n_total//3-1]}) '
          f'→ ({actions[2*n_total//3-1]}) → ({actions[-1]})')
    print(f'总步数: {n_total}, 每步仿真 {args.sim_steps_per_action} 子步')

    print(f'\n运行连续循环...')
    tip_xy = run_continuous_cycle(actions, args.sim_steps_per_action)

    print(f'\n尖端坐标范围:')
    print(f'  X: [{tip_xy[:,0].min():.5f}, {tip_xy[:,0].max():.5f}]')
    print(f'  Y: [{tip_xy[:,1].min():.5f}, {tip_xy[:,1].max():.5f}]')

    # 迟滞度量：起点和终点的偏移
    residual = np.linalg.norm(tip_xy[-1] - tip_xy[n_total // 3 - 1])
    print(f'  终点 vs 阶段1峰值偏差: {residual:.6f} m')

    plot_hysteresis(tip_xy, phase_ids, actions)

    with open(os.path.join(args.output_dir, 'summary.txt'), 'w') as f:
        f.write('=== 方向5b: 迟滞回线分析 ===\n\n')
        f.write(f'参数: n_steps_per_phase={args.n_steps_per_phase}, '
                f'tau_x_max={args.tau_x_max}, tau_y_max={args.tau_y_max}, '
                f'sim_substeps={args.sim_steps_per_action}\n')
        f.write(f'X方向: 峰值={args.tau_x_max}, 先快后慢 (√t)\n')
        f.write(f'Y方向: 峰值={args.tau_y_max}, 先慢后快 (t²)\n')
        f.write(f'扭矩路径: (0,0) → ({args.tau_x_max},{args.tau_y_max}) → '
                f'({-args.tau_x_max},{-args.tau_y_max}) → ({args.tau_x_max},{args.tau_y_max})\n\n')
        f.write(f'尖端范围:\n')
        f.write(f'  X: [{tip_xy[:,0].min():.6f}, {tip_xy[:,0].max():.6f}]\n')
        f.write(f'  Y: [{tip_xy[:,1].min():.6f}, {tip_xy[:,1].max():.6f}]\n')
        f.write(f'终点 vs 峰值偏差: {residual:.6f} m\n')

    print(f'\n=== 完成 ===')
    print(f'查看结果: ls {args.output_dir}/')
