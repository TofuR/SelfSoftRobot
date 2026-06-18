"""visualize_3d_shape.py — 交互式 3D 形状可视化工具。

用法:
    python scripts/evaluation/visualize_3d_shape.py
    python scripts/evaluation/visualize_3d_shape.py --device cuda:0
    python scripts/evaluation/visualize_3d_shape.py --output output/visualize

交互式选择模型 checkpoint → 数据文件 → 帧 → 查询并可视化。
支持所有模型类型:
  density 类 (MSTNF, CMSTNF, MS-SCNF) — 密度阈值 3D 点云
  SDF 类 (SDF, SkeletonSDF)           — marching cubes 三角网格
  skeleton 类 (SpatialSequence, PCSpatial) — 骨架曲线 lines+markers
  pointcloud 类 (FlowMatch)           — ODE 生成点云
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.model_loader import load_model
from src.evaluation.query import query_density_field, query_sdf_field, query_skeleton, query_pointcloud, query_skeleton_direct
from src.evaluation.render import (
    render_density_html, render_sdf_html, render_pointcloud_html, render_skeleton_html,
    render_animation, render_png, render_gif,
)
from src.evaluation.transition_metrics import (
    rollout_one_window, evaluate_transition_rollout, format_summary_line,
)


# ──────────────────────────── 交互工具 ────────────────────────────

def select_from_list(items, prompt, allow_custom=False):
    """交互式列表选择。"""
    if not items:
        if allow_custom:
            return _input_path(f"{prompt} (无候选项，手动输入路径)")
        print(f"  {prompt}: 无可用选项")
        return None

    print(f"\n{prompt}:")
    for i, item in enumerate(items):
        rel_path = os.path.relpath(item, PROJECT_ROOT) if item.startswith('/') else item
        # 对于 checkpoint，额外显示 (exp_id, phase) 以便识别
        if 'train_log' in rel_path and rel_path.endswith('.pt'):
            model_tag, exp_name, phase = parse_checkpoint_path(item)
            print(f"  [{i}] {exp_name} | {phase} → {rel_path}")
        else:
            print(f"  [{i}] {rel_path}")
    if allow_custom:
        print(f"  [c] 自定义路径")

    choice = input("  > ").strip()
    if allow_custom and choice == 'c':
        return _input_path("  路径")
    try:
        return items[int(choice)]
    except (ValueError, IndexError):
        return items[0]


def _input_path(prompt):
    path = input(f"  {prompt}: ").strip()
    return path if path else None


def input_int(prompt, default):
    val = input(f"  {prompt} [{default}]: ").strip()
    return int(val) if val else default


def input_float(prompt, default):
    val = input(f"  {prompt} [{default}]: ").strip()
    return float(val) if val else default


# ──────────────────────────── 数据工具 ────────────────────────────

def scan_checkpoints():
    patterns = [
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'best_model.pt'),
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'skeleton_best.pt'),
        os.path.join(PROJECT_ROOT, 'train_log', '**', 'canonical_best.pt'),
    ]
    ckpts = []
    for pat in patterns:
        ckpts.extend(glob.glob(pat, recursive=True))
    return sorted(set(ckpts))


def parse_checkpoint_path(ckpt_path):
    """从 checkpoint 路径提取 (model_tag, exp_name, phase_name)。

    例如: train_log/gt_transition/exp_20260616_3/phase_gt_transition/model/best_model.pt
    返回: ('gt_transition', 'exp_20260616_3', 'gt_transition')

    Args:
        ckpt_path: checkpoint 文件路径

    Returns:
        (model_tag, exp_name, phase_name) 元组
    """
    parts = Path(ckpt_path).parts
    try:
        train_log_idx = parts.index('train_log')
        model_tag = parts[train_log_idx + 1] if train_log_idx + 1 < len(parts) else 'unknown'
        exp_name = parts[train_log_idx + 2] if train_log_idx + 2 < len(parts) else 'unknown'
        phase_name = parts[train_log_idx + 3] if train_log_idx + 3 < len(parts) else ''
        phase_name = phase_name.replace('phase_', '') if phase_name.startswith('phase_') else phase_name
        return model_tag, exp_name, phase_name
    except (ValueError, IndexError):
        return 'unknown', 'unknown', ''


def scan_data_dirs():
    data_root = os.path.join(PROJECT_ROOT, 'data')
    dirs = []
    for d in sorted(os.listdir(data_root)):
        full = os.path.join(data_root, d)
        if os.path.isdir(full) and glob.glob(os.path.join(full, '*.npz')):
            dirs.append(full)
    return dirs


def scan_npz_files(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, '*.npz')))


def get_action_window(npz_path, frame_idx, window_size, norm_factor):
    d = np.load(npz_path)
    actions = d['actions'].astype(np.float32) / norm_factor
    start = max(0, frame_idx - window_size + 1)
    window = actions[start:frame_idx + 1]
    if len(window) < window_size:
        pad = np.zeros((window_size - len(window), actions.shape[1]), dtype=np.float32)
        window = np.concatenate([pad, window], axis=0)
    return torch.from_numpy(window).unsqueeze(0)  # (1, K, D)


def get_gt_skeleton(npz_path, frame_idx):
    d = np.load(npz_path)
    if 'positions' not in d:
        return None
    pos = d['positions']
    if pos.ndim == 3:
        return pos[frame_idx]  # (3, N)
    return pos


def compute_bounds(npz_path, frame_idx=None):
    d = np.load(npz_path)
    if 'positions' not in d:
        return [-0.1, 0.1, -0.1, 0.2, -0.1, 0.6]
    pos = d['positions']
    if pos.ndim == 3:
        pos = pos[frame_idx] if frame_idx is not None else pos.reshape(-1, 3)
    margin = 0.03
    return [
        float(pos[0].min() - margin), float(pos[0].max() + margin),
        float(pos[1].min() - margin), float(pos[1].max() + margin),
        float(pos[2].min() - margin), float(pos[2].max() + margin),
    ]


def prepare_gt_skeleton_tensor(gt_skeleton, device):
    """将 numpy (3, N) 转为模型需要的 (1, N, 3) tensor。"""
    if gt_skeleton is None:
        return None
    return torch.from_numpy(gt_skeleton.T.astype(np.float32)).unsqueeze(0).to(device)


def prepare_prev_skeleton_tensor(model, gt_skeleton, device):
    """(3, N) raw GT → (1, N, 3) 归一化空间（用模型自身的 pc_center/pc_scale）。

    state_transition（GT-observed 单步转移）的 warm-start 需要归一化空间的前一步骨架。
    必须用模型 buffer 归一化，与训练时数据集的归一化一致（否则 prev 与模型内部空间错位）。
    """
    if gt_skeleton is None:
        return None
    skel = gt_skeleton.T.astype(np.float32)  # (N, 3)
    center = model.pc_center.detach().cpu().numpy().reshape(3)  # (3,)
    scale = model.pc_scale.detach().cpu().numpy().reshape(3)    # (3,)
    skel_norm = (skel - center) / scale
    return torch.from_numpy(skel_norm).unsqueeze(0).to(device)


# ──────────────────────────── 主流程 ────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='3D Shape Visualizer')
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default=None)
    parser.add_argument('--grid_res', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--gif', action='store_true', help='生成 GIF 动画（默认跳过以加速）')
    args = parser.parse_args()

    from config.params import load_config
    eval_cfg = load_config("training").get("evaluation", {})
    device_str = args.device or 'cuda:0'
    default_grid_res = args.grid_res or eval_cfg.get("grid_res", 40)
    default_threshold = args.threshold or eval_cfg.get("density_threshold", 0.01)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    output_dir = args.output or os.path.join(PROJECT_ROOT, 'output', 'visualize')
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== 3D Shape Visualizer ===\n")

    # ── Step 1: 选 checkpoint → 加载模型 ──
    ckpts = scan_checkpoints()
    ckpt_path = select_from_list(ckpts, "[1] 选择模型 checkpoint", allow_custom=True)
    if ckpt_path is None:
        return

    print(f"\n加载模型...")
    info = load_model(ckpt_path, device=device)
    model = info['model']
    model_type = info['model_type']
    window_size = info['window_size']
    norm_factor = info['norm_factor']
    use_gt_skeleton = info.get('use_gt_skeleton', False)
    skeleton_mode = info.get('skeleton_mode')
    trained_phases = info.get('trained_phases', set())

    print(f"  模型: {model_type}, skeleton_mode={skeleton_mode}")
    print(f"  use_gt_skeleton={use_gt_skeleton}")

    # ── state_transition 族：rollout（开环自回归）vs warm-start（GT[t-1]）模式 ──
    # OpenLoopTransitionModel → 默认 rollout（方向 15 部署语义：seed 一次→滚 K 步）
    # 其他 state_transition（gt_transition / base）→ 菜单选择
    rollout_mode = False
    is_state_transition = (model_type == 'state_transition')
    is_open_loop = bool(getattr(model, 'open_loop_mode', torch.tensor(False)).item()) \
        if hasattr(model, 'open_loop_mode') else False
    if is_state_transition:
        if is_open_loop:
            print("  OpenLoopTransitionModel → 默认 rollout 模式（seed 一次→滚 K 步，开环）")
            ans = input("  切换到 warm-start（每帧喂 GT[t-1]）? [y/N]: ").strip().lower()
            rollout_mode = (ans != 'y')
        else:
            print("\n  state_transition 模型 — 选择推理模式:")
            print("    [1] rollout（开环自回归：seed 一次→滚 K 步，方向 15 部署语义）")
            print("    [2] warm-start（每帧喂 GT[t-1]，单步转移）")
            mc = input("  > [1]: ").strip()
            rollout_mode = (mc != '2')
        print(f"  → 模式: {'rollout（开环）' if rollout_mode else 'warm-start（GT[t-1]）'}")

    # ── Step 2: 选数据 ──
    data_dirs = scan_data_dirs()
    data_dir = select_from_list(data_dirs, "[2] 选择数据目录", allow_custom=True)
    if data_dir is None:
        return

    npz_files = scan_npz_files(data_dir)
    npz_path = select_from_list(npz_files, "[3] 选择数据文件", allow_custom=True)
    if npz_path is None:
        return

    # 检查 GT skeleton 是否可用
    has_positions = 'positions' in np.load(npz_path)
    print(f"  GT skeleton: {'可用' if has_positions else '不可用'}")

    # ── Step 3: 判断是否需要 GT skeleton ──
    need_gt = False
    if model_type == 'ms_scnf' and 'joint' in trained_phases and 'skeleton' not in trained_phases:
        need_gt = True
        print("  注意: skeleton_head 未训练（phase 1 被跳过），推理需要 GT skeleton")
    elif model_type == 'skeleton_sdf' and 'joint' in trained_phases and 'skeleton' not in trained_phases:
        need_gt = True
        print("  注意: skeleton_head 未训练（phase 1 被跳过），推理需要 GT skeleton")

    if need_gt and not has_positions:
        print("  错误: 模型需要 GT skeleton 但数据中没有 positions 字段")
        return

    # ── Step 4: 帧范围 ──
    d = np.load(npz_path)
    n_frames = d['actions'].shape[0]
    print(f"\n[4] 帧范围 (共 {n_frames} 帧)")
    start_frame = input_int("  起始帧", 0)
    end_frame = input_int(f"  结束帧 (含, max {n_frames-1})", min(start_frame + 99, n_frames - 1))
    step = input_int("  帧间隔", max(1, (end_frame - start_frame) // 50))
    frame_indices = list(range(start_frame, end_frame + 1, step))
    n_vis = len(frame_indices)
    print(f"  将可视化 {n_vis} 帧: {frame_indices[0]}-{frame_indices[-1]} (step={step})")

    # ── Step 5: 查询参数 ──
    grid_res = input_int("[5] 网格分辨率", default_grid_res)
    is_sdf = model_type in ('sdf', 'skeleton_sdf')
    is_pc = model_type == 'flowmatch'
    is_skeleton = model_type in ('spatial_sequence', 'pc_spatial', 'state_transition')
    threshold = default_threshold
    sdf_mode = 'mesh'

    if is_skeleton:
        print("  (骨架模型，直接前向推理，无需网格查询)")
    elif is_sdf:
        print("\n[6] SDF 可视化模式:")
        print("  1. mesh (marching cubes)")
        print("  2. pointcloud (SDF<=0)")
        mode_choice = input("  > [1]: ").strip()
        if mode_choice == '2':
            sdf_mode = 'pointcloud'
    elif is_pc:
        n_points = input_int("[6] ODE 生成点数", 1000)
        n_ode_steps = input_int("    ODE 积分步数", 50)
        print("  (Flow Matching 模型，直接生成点云，无需网格查询)")
    else:
        threshold = input_float("[6] 密度阈值", default_threshold)

    # ── Step 6: bounds ──
    bounds = compute_bounds(npz_path)
    print(f"\n空间: [{bounds[0]:.3f},{bounds[1]:.3f}] x "
          f"[{bounds[2]:.3f},{bounds[3]:.3f}] x [{bounds[4]:.3f},{bounds[5]:.3f}]")

    # ── Step 7: 逐帧查询 ──
    model_tag, exp_name, phase_name = parse_checkpoint_path(ckpt_path)
    # 输出文件名格式: {model_type}_{exp_name}_{phase}_frames{start}-{end}
    base_name = f"{model_type}_{exp_name}_{phase_name}_frames{start_frame}-{end_frame}"

    print(f"\n查询模型 ({model_type}), {n_vis} 帧...")
    all_results = []
    all_gt = []
    all_pred = []

    # 开环 rollout 模式：窗口化——每 K 步用 GT 重新播种（部署语义：
    # "观测一次→预测 K 步→重观测"），而非从单一种子连续滚到底（会累积漂移到崩）。
    rollout_world = None
    if is_skeleton and rollout_mode:
        if start_frame < 1:
            start_frame = 1
        n_roll = min(n_vis, n_frames - start_frame)
        K = int(getattr(model, 'episode_len', None) or window_size)   # 每窗 rollout 步数
        _d = np.load(npz_path)
        _actions_norm = _d['actions'].astype(np.float32) / norm_factor
        _positions = _d['positions'].astype(np.float32)
        _pc_c = model.pc_center.view(3).cpu().numpy()
        _pc_s = model.pc_scale.view(3).cpu().numpy()
        chunks, ws = [], start_frame
        while ws < start_frame + n_roll:
            k = min(K, start_frame + n_roll - ws)                     # 末窗不足 K 截短
            _r = rollout_one_window(model, _actions_norm, _positions, ws, k,
                                     window_size, _pc_c, _pc_s, device)
            chunks.append(_r['roll'].cpu().numpy() * _pc_s + _pc_c)   # (k,N,3) world
            ws += k
        rollout_world = np.concatenate(chunks, axis=0)                # (n_roll,N,3)
        frame_indices = list(range(start_frame, start_frame + n_roll))
        n_vis = n_roll
        print(f"  rollout: 窗口化（每 {K} 步用 GT 重播种），{len(chunks)} 窗 ≈ {n_roll} 步")

    for vis_i, fidx in enumerate(frame_indices):
        # 开环 rollout：从预计算轨迹取第 vis_i 步，跳过逐帧 warm-start 查询
        if rollout_world is not None:
            pts = rollout_world[vis_i]  # (N,3) world
            result = {'points': pts, 'skeleton': pts[None]}
            gt_skeleton = get_gt_skeleton(npz_path, fidx)
            all_results.append(result)
            all_gt.append(gt_skeleton)
            all_pred.append(None)
            print(f"  [{vis_i + 1}/{n_vis}] frame {fidx} (rollout step)", end='\r')
            continue
        action_window = get_action_window(npz_path, fidx, window_size, norm_factor).to(device)
        gt_skeleton = get_gt_skeleton(npz_path, fidx)
        gt_skel_tensor = None
        pred_skeleton = None

        # 准备 GT skeleton tensor
        if (need_gt or use_gt_skeleton) and gt_skeleton is not None:
            gt_skel_tensor = prepare_gt_skeleton_tensor(gt_skeleton, device)

        # 查询
        if is_skeleton:
            if model_type == 'state_transition':
                # GT-observed 单步转移：用真实 GT[t-1] 作 warm-start（冷启动只会输出
                # 一个 ≈0 的 Δ，退化成蓝点）。pred=ŝ_t，overlay 的 GT=GT[t]（目标）。
                prev_raw = get_gt_skeleton(npz_path, max(0, fidx - 1))
                prev_tensor = prepare_prev_skeleton_tensor(model, prev_raw, device)
                result = query_skeleton_direct(model, action_window,
                                                prev_skeleton=prev_tensor)
            else:
                result = query_skeleton_direct(model, action_window)
        elif is_sdf:
            result = query_sdf_field(model, action_window, bounds, grid_res, device,
                                      gt_skeleton=gt_skel_tensor)
        elif is_pc:
            result = query_pointcloud(model, action_window, n_points=n_points,
                                       n_steps=n_ode_steps)
        else:
            result = query_density_field(model, action_window, bounds, grid_res, device,
                                          gt_skeleton=gt_skel_tensor)

        # 骨架可视化（overlay GT）
        if is_skeleton:
            # 预测已是 result['points']，GT 作为 overlay
            pass
        elif gt_skeleton is not None:
            pred_skeleton = gt_skeleton
        elif model_type in ('ms_scnf', 'skeleton_sdf') and not need_gt:
            try:
                skel = query_skeleton(model, action_window)
                pred_skeleton = skel['fine'][0].T  # (3, N)
            except Exception:
                pass

        all_results.append(result)
        all_gt.append(gt_skeleton)
        all_pred.append(pred_skeleton)

        n_verts = len(result['vertices']) if result.get('vertices') is not None else 0
        n_pts = len(result.get('points', []))
        if result.get('density') is not None:
            n_pts = int((result['density'] > threshold).sum())
        print(f"  [{vis_i+1}/{n_vis}] frame {fidx}: {n_verts} verts, {n_pts} pts", end='\r')

    print(f"\n  完成 {n_vis} 帧查询")

    # ── Step 8: 输出 ──
    print(f"\n生成可视化...")

    # 动画 HTML（state_transition 族：算量化指标并嵌入标题 + 存 metrics.txt）
    html_path = os.path.join(output_dir, f"{base_name}.html")
    metrics_summary = None
    if is_state_transition:
        try:
            _res = evaluate_transition_rollout(
                model, data_dir, load_config("training"), device,
                n_seqs=5, windows_per_seq=2)
            if _res is not None:
                metrics_summary = _res['summary']
        except Exception as e:
            print(f"  (量化指标计算跳过: {e})")
    fig = render_animation(all_results, model_type, threshold, all_gt, all_pred,
                           frame_indices, sdf_mode=sdf_mode, output_path=None)
    if metrics_summary is not None:
        _extra = f"<br><sup><sub>{format_summary_line(metrics_summary)}</sub></sup>"
        fig.update_layout(
            title=f"{model_type.upper()} — Animation ({n_vis} frames){_extra}")
        with open(os.path.join(output_dir, f"{base_name}_metrics.txt"), 'w',
                  encoding='utf-8') as fm:
            fm.write(format_summary_line(metrics_summary) + "\n\n")
            for _k, _v in metrics_summary.items():
                fm.write(f"{_k}: {_v}\n")
        print(f"  指标: {format_summary_line(metrics_summary)}")
    fig.write_html(html_path)
    print(f"  HTML: {os.path.relpath(html_path)}")

    # 单帧 PNG
    mid = n_vis // 2
    png_path = os.path.join(output_dir, f"{base_name}_mid.png")
    if is_skeleton:
        fig = render_skeleton_html(all_results[mid], all_gt[mid], all_pred[mid])
    elif is_sdf:
        fig = render_sdf_html(all_results[mid], sdf_mode, all_gt[mid], all_pred[mid])
    elif is_pc:
        fig = render_pointcloud_html(all_results[mid], all_gt[mid], all_pred[mid])
    else:
        fig = render_density_html(all_results[mid], threshold, all_gt[mid], all_pred[mid])
    render_png(fig, png_path)

    # GIF（默认跳过，--gif 时才生成）
    if args.gif:
        gif_path = os.path.join(output_dir, f"{base_name}.gif")
        render_gif(all_results, model_type, threshold, all_gt, all_pred,
                   frame_indices, sdf_mode=sdf_mode, output_path=gif_path)
    else:
        print("  (跳过 GIF，使用 --gif 生成动画)")

    print(f"\n完成! 输出目录: {output_dir}")


if __name__ == "__main__":
    main()
