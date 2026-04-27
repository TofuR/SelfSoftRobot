"""model_loader.py — 统一模型加载工具。

自动检测模型类型和训练阶段，加载权重并返回 eval 模式的模型。
"""

import os
import glob
import numpy as np
import torch

from src.config.params import load_config


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


def _detect_model_type(state_dict):
    """通过 checkpoint key 判断模型类型和训练阶段。"""
    if 'temporal' in state_dict and 'skeleton_head' in state_dict:
        # MS-SCNF phase 1: 只保存了 temporal + skeleton_head
        return 'ms_scnf', 1
    if 'density.pos_encoder' in state_dict or 'skeleton_conditioned_density' in str(state_dict.keys()):
        # MS-SCNF phase 2: 完整 state_dict 含 density 模块
        return 'ms_scnf', 2
    # 检查是否有 density 相关的 key（full state_dict 格式）
    for key in state_dict:
        if 'skeleton_head' in key:
            return 'ms_scnf', 2
        if 'canonical' in key:
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

    train_cfg = load_config('training')
    action_dim = _infer_action_dim(data_dir)
    norm_factor = _load_norm_factor(checkpoint_path, data_dir)
    if window_size is None:
        window_size = train_cfg['temporal']['window_size']

    if model_type == 'ms_scnf':
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
    }
    print(f"Loaded {model_type} (phase {phase}) from {checkpoint_path}")
    print(f"  action_dim={action_dim}, window_size={window_size}, norm_factor={norm_factor:.4f}")
    return info
