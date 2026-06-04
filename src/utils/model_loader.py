"""model_loader.py — 统一模型加载工具。

自动检测模型类型和训练阶段，加载权重并返回 eval 模式的模型。
优先从同目录 config.json 读取参数，找不到则从 state_dict 推断。
"""

import os
import json
import glob
import numpy as np
import torch

from config.params import load_config


def _find_config(checkpoint_path):
    """从 checkpoint 路径向上搜索 config.json。"""
    ckpt_dir = os.path.dirname(checkpoint_path)
    for _ in range(5):
        cfg_path = os.path.join(ckpt_dir, 'config.json')
        if os.path.exists(cfg_path):
            return cfg_path
        parent = os.path.dirname(ckpt_dir)
        if parent == ckpt_dir:
            break
        ckpt_dir = parent
    return None


def _load_config_json(checkpoint_path):
    """加载 checkpoint 附近的 config.json，返回 dict 或 None。"""
    cfg_path = _find_config(checkpoint_path)
    if cfg_path is None:
        return None
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _infer_action_dim(data_dir):
    """从数据目录的 npz 文件推断 action_dim。"""
    if data_dir is None:
        return 2
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        return 2
    try:
        d = np.load(files[0])
        return d['actions'].shape[1]
    except Exception:
        return 2


def _load_norm_factor(checkpoint_path, data_dir):
    """加载动作归一化系数，搜索顺序：checkpoint 同目录 → data_dir。"""
    candidates = []
    ckpt_dir = os.path.dirname(checkpoint_path)
    while ckpt_dir and ckpt_dir != '/':
        candidates.append(os.path.join(ckpt_dir, 'action_norm_factor.txt'))
        parent = os.path.dirname(ckpt_dir)
        if parent == ckpt_dir:
            break
        ckpt_dir = parent
    if data_dir:
        candidates.append(os.path.join(data_dir, 'action_norm_factor.txt'))

    for path in candidates:
        if os.path.exists(path):
            return float(np.loadtxt(path))

    print("Warning: action_norm_factor.txt not found, using 1.0")
    return 1.0


def _detect_skeleton_mode(state_dict):
    """从 checkpoint key 推断骨架参数化方式。"""
    keys = set(state_dict.keys())
    if 'skeleton_head.basis_matrix' in keys:
        return 'bspline'
    if 'skeleton_head.eval_matrix' in keys:
        w = state_dict.get('skeleton_head.head.weight')
        if w is not None and w.shape[0] % 3 == 0:
            n_param = w.shape[0] // 3
            if n_param % 2 == 1:
                return 'fourier'
        return 'catmullrom'
    return 'point'


def _detect_skeleton_n_ctrl(state_dict):
    """从 checkpoint 推断骨架控制点数（bspline/catmullrom 用）。"""
    w = state_dict.get('skeleton_head.head.weight')
    if w is not None and w.shape[0] % 3 == 0:
        return w.shape[0] // 3
    return 10


def _detect_model_type(state_dict):
    """通过 checkpoint key 判断模型类型和训练阶段。"""
    keys = set(state_dict.keys())

    # Flow Matching 模型: velocity_net + temporal（无 density/sdf/skeleton）
    has_velocity = any('velocity_net' in k for k in keys)
    has_temporal = any('temporal' in k for k in keys)
    if has_velocity and has_temporal:
        return 'flowmatch', 0

    # SDF 模型（无骨架）: coord_encoder
    if any('coord_encoder' in k for k in keys):
        return 'sdf', 0

    # 骨架 SDF 模型: skeleton_head + sdf_net（无 density）
    has_skel = any('skeleton_head' in k for k in keys)
    has_sdf_net = any('sdf_net' in k for k in keys)
    has_density = any('density' in k for k in keys)

    if has_skel and has_sdf_net and not has_density:
        return 'skeleton_sdf', 0

    # MS-SCNF phase 1: 只保存了 temporal + skeleton_head 的子模块
    if has_skel and not has_density and not has_sdf_net:
        return 'ms_scnf', 1

    # MS-SCNF phase 2: skeleton_head + density
    if has_skel and has_density:
        return 'ms_scnf', 2

    # CMSTNF: canonical 模块
    if any('canonical' in k for k in keys):
        return 'cmstnf', 2

    return 'mstnf', 0


def load_model(checkpoint_path, data_dir=None, device='cpu', window_size=None):
    """加载训练好的模型。

    自动检测模型类型和训练阶段，加载权重和归一化系数。

    Args:
        checkpoint_path: 模型权重文件路径 (.pt)。
        data_dir: 数据目录（用于推断 action_dim 和加载 norm_factor）。
        device: 计算设备。
        window_size: 时序窗口长度，默认从配置读取。

    Returns:
        dict: {
            'model': 已加载权重的模型 (eval 模式),
            'norm_factor': float,
            'model_type': str ('ms_scnf' / 'cmstnf' / 'mstnf'),
            'phase': int (0=单阶段, 1=phase1, 2=phase2),
            'action_dim': int,
            'window_size': int,
        }
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_type, phase = _detect_model_type(state_dict)

    # 读取 config.json（如有），获取保存的参数
    saved_cfg = _load_config_json(checkpoint_path)

    train_cfg = load_config('training')
    action_dim = _infer_action_dim(data_dir)
    norm_factor = _load_norm_factor(checkpoint_path, data_dir)
    if window_size is None:
        window_size = saved_cfg.get('window_size') if saved_cfg else None
        if window_size is None:
            window_size = train_cfg['temporal']['window_size']

    # 从 config.json 或 state_dict 推断 skeleton_mode
    skel_mode = None
    if saved_cfg and 'skeleton_mode' in saved_cfg:
        skel_mode = saved_cfg['skeleton_mode']
    if model_type in ('ms_scnf', 'skeleton_sdf'):
        skel_mode = skel_mode or _detect_skeleton_mode(state_dict)
    n_ctrl = _detect_skeleton_n_ctrl(state_dict) if skel_mode and skel_mode != 'point' else 10

    # 判断是否需要 GT skeleton（从 config.json 的 phases 信息）
    use_gt_skeleton = False
    trained_phases = set()
    if saved_cfg:
        for p in saved_cfg.get('phases', []):
            if isinstance(p, str):
                trained_phases.add(p)
            elif isinstance(p, dict):
                if p.get('trained', False) or p.get('use_gt_skeleton', False):
                    trained_phases.add(p.get('name', ''))
                if p.get('use_gt_skeleton', False):
                    use_gt_skeleton = True

    if model_type == 'sdf':
        from src.models.model_sdf import TemporalSDFModel
        model = TemporalSDFModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
        ).to(device)
        model.load_state_dict(state_dict)

    elif model_type == 'flowmatch':
        from src.models.model_flowmatch import FlowMatchPointCloudModel
        pc_cfg = train_cfg.get('pointcloud', {})
        model = FlowMatchPointCloudModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
            velocity_net_hidden=pc_cfg.get('velocity_net_hidden', 256),
            velocity_net_layers=pc_cfg.get('velocity_net_layers', 6),
            time_embed_dim=pc_cfg.get('time_embed_dim', 64),
            sigma=pc_cfg.get('sigma', 1.0),
            ode_steps=pc_cfg.get('ode_steps', 50),
            ode_solver=pc_cfg.get('ode_solver', 'euler'),
            n_points=pc_cfg.get('n_surface_points', 1000),
        ).to(device)
        # strict=False: 旧 checkpoint 没有 pc_center/pc_scale buffer
        model.load_state_dict(state_dict, strict=False)

    elif model_type == 'ms_scnf':
        from src.models.model_ms_scnf import MSSCNFModel
        ms_cfg = train_cfg.get('ms_scnf', {})

        model = MSSCNFModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
            d_filter=train_cfg['model']['d_filter'],
            n_freqs=train_cfg['model']['n_freqs'],
            n_coarse=ms_cfg.get('n_coarse', 4),
            n_medium=ms_cfg.get('n_medium', 10),
            n_fine=ms_cfg.get('n_fine', 31),
            deform_n_freqs=train_cfg['canonical']['deform_n_freqs'],
            skeleton_mode=skel_mode or 'point',
            fourier_n_freq=saved_cfg.get('fourier_n_freq', ms_cfg.get('fourier_n_freq', 8)) if saved_cfg else ms_cfg.get('fourier_n_freq', 8),
            bspline_n_ctrl=n_ctrl if skel_mode == 'bspline' else ms_cfg.get('bspline_n_ctrl', 10),
            catmullrom_n_ctrl=n_ctrl if skel_mode == 'catmullrom' else ms_cfg.get('catmullrom_n_ctrl', 10),
        ).to(device)

        if phase == 1:
            model.temporal.load_state_dict(state_dict['temporal'])
            model.skeleton_head.load_state_dict(state_dict['skeleton_head'])
        else:
            model.load_state_dict(state_dict)

    elif model_type == 'cmstnf':
        from src.models.model_cmstnf import CMSTNFModel
        model = CMSTNFModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
            d_filter=train_cfg['model']['d_filter'],
            n_freqs=train_cfg['model']['n_freqs'],
        ).to(device)
        model.load_state_dict(state_dict)

    elif model_type == 'skeleton_sdf':
        from src.models.model_skeleton_sdf import SkeletonSDFModel
        ms_cfg = train_cfg.get('ms_scnf', {})

        model = SkeletonSDFModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
            skeleton_mode=skel_mode or 'bspline',
            rod_radius=ms_cfg.get('rod_radius', 0.015),
        ).to(device)
        model.load_state_dict(state_dict)

    else:  # mstnf
        from src.models.model_mstnf import MSTNFModel
        model = MSTNFModel(
            action_dim=action_dim,
            window_size=window_size,
            n_scales=train_cfg['temporal']['n_scales'],
            hidden_dim=train_cfg['temporal']['hidden_dim'],
            d_filter=train_cfg['model']['d_filter'],
            n_freqs=train_cfg['model']['n_freqs'],
        ).to(device)
        model.load_state_dict(state_dict)

    model.eval()

    info = {
        'model': model,
        'norm_factor': norm_factor,
        'model_type': model_type,
        'phase': phase,
        'action_dim': action_dim,
        'window_size': window_size,
        'skeleton_mode': skel_mode,
        'use_gt_skeleton': use_gt_skeleton,
        'trained_phases': trained_phases,
        'saved_config': saved_cfg,
    }
    print(f"Loaded {model_type} (phase {phase}) from {checkpoint_path}")
    print(f"  action_dim={action_dim}, window_size={window_size}, norm_factor={norm_factor:.4f}")
    if skel_mode:
        print(f"  skeleton_mode={skel_mode}, use_gt_skeleton={use_gt_skeleton}")
    return info


def load_model_from_experiment(exp_dir, data_dir=None, device='cpu', phase_name=None):
    """从实验目录加载最佳模型。

    Args:
        exp_dir: 实验目录（含 config.json 和 phase_*/ 子目录）。
        data_dir: 数据目录。
        device: 计算设备。
        phase_name: 指定 phase（默认取最后一个有 best_model.pt 的）。

    Returns:
        dict: 同 load_model()。
    """
    # 搜索 best_model.pt
    if phase_name:
        ckpt_path = os.path.join(exp_dir, f"phase_{phase_name}", "model", "best_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    else:
        candidates = sorted(glob.glob(os.path.join(exp_dir, 'phase_*', 'model', 'best_model.pt')))
        if not candidates:
            raise FileNotFoundError(f"No best_model.pt in {exp_dir}")
        ckpt_path = candidates[-1]

    return load_model(ckpt_path, data_dir=data_dir, device=device)
