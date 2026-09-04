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

    # StateTransition: 可学习迟滞潜变量模型（z_cell + state_encoder + delta_head）
    # 检测放在 spatial_sequence 之前：本模型有 gru 但无 slice_head（用 delta_head），
    # 不会误命中 spatial_sequence 分支，但显式检测更清晰。
    has_z_cell = any('z_cell' in k for k in keys)
    has_state_encoder = any('state_encoder' in k for k in keys)
    has_delta_head = any('delta_head' in k for k in keys)
    if has_z_cell and (has_state_encoder or has_delta_head):
        return 'state_transition', 0

    # HOV2.1 retains hereditary PI/Maxwell banks but replaces the free point
    # readout with a fixed POD/generalized-coordinate geometry decoder.
    if ({'bend_basis', 'generalized_coordinate_scale',
         'pi_mode_directions_raw', 'maxwell_mode_directions_raw'} <= keys and
            any(k.startswith('play.') for k in keys) and
            any(k.startswith('maxwell.') for k in keys)):
        return 'hereditary_geometry', 0

    # ISHSM has an explicit POD basis, frozen H0 reference buffers and stable
    # trainable modal time constants.  Detect before generic fallbacks.
    ishsm_required = {
        'bend_basis', 'reference_segment_lengths', 'reference_bend_dirs',
        'reference_length_dirs', 'raw_taus', 'excitation',
    }
    if ishsm_required.issubset(keys):
        return 'ishsm', 0

    # HereditaryOperatorModel: 显式迟滞算子组（play/maxwell 算子库）。
    # 算子网格（thresholds/taus/decays）是 registered buffer，随 state_dict
    # 精确恢复，无需从 config 重建网格。
    has_play_bank = any(k.startswith('play.') for k in keys)
    has_maxwell_bank = any(k.startswith('maxwell.') for k in keys)
    if has_play_bank and has_maxwell_bank:
        return 'hereditary', 0

    # SpatialSequence: gru + slice_head（无 correction）
    has_gru = any('gru' in k for k in keys)
    has_slice = any('slice_head' in k for k in keys)
    has_correction = any('correction' in k for k in keys)
    if has_gru and has_slice and not has_correction:
        return 'spatial_sequence', 0
    if has_gru and has_slice and has_correction:
        return 'pc_spatial', 0

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


def _migrate_gru_keys(state_dict):
    """透明迁移 GRUCell → nn.GRU 的 state_dict 键（向后兼容旧 checkpoint）。

    背景：S1 优化把 model_state_transition 的逐节点 GRUCell 改为单次 nn.GRU，
    state_dict 键由 gru.weight_ih/hh/bias_ih/bias_hh 变为 *_l0。
    旧（GRUCell）checkpoint 加载到新（nn.GRU）模型时，strict=False 会静默忽略
    这些键 → GRU 层保持随机初始化 → 输出全是噪声（蓝点）。本函数在加载前补上后缀。

    权重 shape 完全一致（GRUCell 与单层 nn.GRU 的 ih/hh 矩阵同形），仅键名不同。

    Args:
        state_dict: 原始 state_dict（可能为 GRUCell 或 nn.GRU 格式）。

    Returns:
        迁移后的 state_dict（已是新格式则原样返回）。
    """
    renames = {
        'gru.weight_ih': 'gru.weight_ih_l0',
        'gru.weight_hh': 'gru.weight_hh_l0',
        'gru.bias_ih': 'gru.bias_ih_l0',
        'gru.bias_hh': 'gru.bias_hh_l0',
    }
    # 已是新格式（含 _l0 键）→ 无需迁移
    if any(new in state_dict for new in renames.values()):
        return state_dict
    new_sd = dict(state_dict)
    migrated = []
    for old, new in renames.items():
        if old in new_sd:
            new_sd[new] = new_sd.pop(old)
            migrated.append(old)
    if migrated:
        print(f"  [migrate] GRUCell → nn.GRU keys: {migrated}")
    return new_sd


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
    if any('raw_alphas' in key for key in state_dict) and (
            (saved_cfg or {}).get('gl_kernel_alignment') !=
            'current_at_window_end'):
        raise ValueError(
            "fractional checkpoint 缺少当前 GL 时间对齐合同，"
            "需使用修正后的 w0↔当前动作实现重新训练。")

    train_cfg = load_config('training')
    # action_dim 优先取 checkpoint 自身 config.json（实物 ad=1 / 仿真 ad=2 都能正确还原），
    # 避免可视化等"未传 data_dir"场景下误用仿真默认 2 → 与实物 checkpoint 形状不符。
    # 兜底：config 无 action_dim 时再用 data_dir npz 推断，最后默认 2。
    if saved_cfg and saved_cfg.get('action_dim'):
        action_dim = saved_cfg['action_dim']
    else:
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
        # velocity_net 新增了 action_embed/interaction/z_embed 模块
        # 旧 checkpoint 无这些权重 → strict=False 允许部分加载
        # strict=False: 旧 checkpoint 没有 pc_center/pc_scale/action_norm_factor buffer
        model.load_state_dict(state_dict, strict=False)

        # 从 checkpoint 恢复 action_norm_factor（优先于文件）
        if 'action_norm_factor' in state_dict:
            norm_factor = state_dict['action_norm_factor'].item()

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

    elif model_type in ('spatial_sequence', 'pc_spatial'):
        # SpatialSequence / PCSpatial — skeleton-only 模型
        # n_orders 推断优先级: saved_cfg > state_dict 权重形状 > 全局默认
        n_nodes = saved_cfg.get('n_nodes', 31) if saved_cfg else 31
        if saved_cfg and 'n_scales' in saved_cfg:
            n_orders = saved_cfg['n_scales']
        else:
            n_orders = train_cfg['temporal'].get('n_scales', 4)
            # 从 state_dict 权重形状推断（覆盖默认值）
            for k, v in state_dict.items():
                if k == 'temporal.raw_alphas':
                    n_orders = v.shape[0]
                    break
                if k == 'temporal.order_weights':
                    n_orders = v.shape[0]
                    break
        if any('raw_alphas' in k for k in state_dict):
            encoder_type = 'fractional'
        elif any('temporal.k_offsets' in k or 'temporal.logit_lambdas' in k for k in state_dict):
            encoder_type = 'gamma'
        elif any('temporal.cls_token' in k for k in state_dict):
            encoder_type = 'transformer'
        elif any('temporal.tcn_layers' in k for k in state_dict):
            encoder_type = 'tcn'
        elif any('temporal.gru.weight' in k for k in state_dict):
            encoder_type = 'gru'
        elif any('raw_decays' in k for k in state_dict):
            encoder_type = 'ema'
        else:
            encoder_type = saved_cfg.get('encoder_type', 'ema') if saved_cfg else 'ema'
        hidden_dim = saved_cfg.get('hidden_dim', train_cfg['temporal']['hidden_dim']) if saved_cfg else train_cfg['temporal']['hidden_dim']

        if model_type == 'spatial_sequence':
            from src.models.model_spatial_sequence import SpatialSequenceModel
            model = SpatialSequenceModel(
                action_dim=action_dim, window_size=window_size,
                n_orders=n_orders, hidden_dim=hidden_dim,
                n_nodes=n_nodes, encoder_type=encoder_type,
            ).to(device)
        else:
            from src.models.model_pc_spatial import PCSpatialSequenceModel
            n_views = saved_cfg.get('n_views', 2) if saved_cfg else 2
            model = PCSpatialSequenceModel(
                action_dim=action_dim, window_size=window_size,
                n_orders=n_orders, hidden_dim=hidden_dim,
                n_nodes=n_nodes, encoder_type=encoder_type, n_views=n_views,
            ).to(device)

        model.load_state_dict(state_dict, strict=False)
        # norm_factor 从 checkpoint buffer 恢复
        if 'action_norm_factor' in state_dict:
            norm_factor = state_dict['action_norm_factor'].item()

    elif model_type == 'hereditary_geometry':
        if (not saved_cfg or
                saved_cfg.get('model') != 'HereditaryGeometryModel'):
            raise ValueError(
                "HOV2.1 checkpoint 必须有 config.json 且 "
                "model=HereditaryGeometryModel")
        from src.models.model_hereditary_geometry import HereditaryGeometryModel
        h0_reference = str(saved_cfg.get(
            'h0_reference', 'fit_only_frozen_linear'))
        reference_kind = h0_reference.removeprefix('fit_only_frozen_')
        residual_mode = saved_cfg.get('hov21_residual', 'memory')
        model = HereditaryGeometryModel(
            action_dim=action_dim,
            n_nodes=int(saved_cfg['n_nodes']),
            window_size=window_size,
            n_play=int(saved_cfg.get('n_play', 2)),
            n_maxwell=int(saved_cfg.get('n_maxwell', 6)),
            dt=float(saved_cfg.get('dt', 0.1)),
            tau_range=(3.0 * float(saved_cfg.get('dt', 0.1)),
                       float(saved_cfg.get('tau_max', 2.0))),
            burnin_mode=saved_cfg.get('burnin_mode', 'equilibrium'),
            drive_normalization=saved_cfg.get(
                'operator_drive_normalization', 'free'),
            n_bend_modes=int(saved_cfg.get('n_bend_modes', 8)),
            section_intervals=tuple(saved_cfg.get(
                'section_intervals', (7, 7))),
            bend_basis=state_dict['bend_basis'],
            generalized_coordinate_scale=
                state_dict['generalized_coordinate_scale'],
            reference_segment_lengths=
                state_dict['reference_segment_lengths'],
            reference_bend_bias=state_dict['reference_bend_bias'],
            reference_bend_dirs=state_dict['reference_bend_dirs'],
            reference_length_bias=state_dict['reference_length_bias'],
            reference_length_dirs=state_dict['reference_length_dirs'],
            reference_kind=reference_kind,
            reference_knots=state_dict.get('reference_knots'),
            reference_drive_weights=
                state_dict.get('reference_drive_weights'),
            bend_basis_kind=saved_cfg.get('bend_basis_kind', 'pod'),
            base_position=state_dict['base_position'],
            residual_mode=residual_mode,
            bend_residual_max_rad=float(saved_cfg.get(
                'hov21_bend_residual_max_rad', 0.05)),
            length_residual_max_log=float(saved_cfg.get(
                'hov21_length_residual_max_log', 0.02)),
            episode_len=int(saved_cfg.get('episode_len', 40)),
        ).to(device)
        model.load_state_dict(state_dict)
        # Runtime code may support a newer contract than the checkpoint.
        # Preserve the artifact's declared semantics for provenance/reporting.
        model.model_contract_version = int(saved_cfg.get(
            'model_contract_version', 4))
        if 'action_norm_factor' in state_dict:
            norm_factor = state_dict['action_norm_factor'].item()

    elif model_type == 'ishsm':
        if not saved_cfg or saved_cfg.get('model') != 'ISHSMModel':
            raise ValueError(
                "ISHSM checkpoint 必须有 config.json 且 model=ISHSMModel，"
                "不能仅凭权重形状猜测几何合同")
        from src.models.model_ishsm import ISHSMModel
        model = ISHSMModel(
            action_dim=action_dim,
            n_nodes=int(saved_cfg['n_nodes']),
            n_bend_modes=int(saved_cfg['n_bend_modes']),
            section_intervals=tuple(saved_cfg['section_intervals']),
            dt=float(saved_cfg['dt']),
            tau_range=(float(saved_cfg['tau_min']),
                       float(saved_cfg['tau_max'])),
            use_dynamic_length=bool(saved_cfg.get(
                'use_dynamic_length', True)),
            use_persistent_state=bool(saved_cfg.get(
                'use_persistent_state', False)),
            persistence_init=float(saved_cfg.get('persistence_init', 0.1)),
            observation_update=saved_cfg.get(
                'observation_update', 'hard'),
            observation_gain_init=float(saved_cfg.get(
                'observation_gain_init', 0.25)),
            training_reanchor_intervals=tuple(saved_cfg.get(
                'training_reanchor_intervals', [0])),
            tau_parameterization=saved_cfg.get(
                'tau_parameterization', 'independent'),
            observation_projection=saved_cfg.get(
                'observation_projection', 'modal'),
            tip_dls_lambda_mm2=float(saved_cfg.get(
                'tip_dls_lambda_mm2', 1.0)),
            reference_kind=(
                'monotone_spline'
                if saved_cfg.get('h0_reference') ==
                'fit_only_frozen_monotone_spline' else 'linear'),
            reference_knots=(state_dict.get('reference_knots')
                             if 'reference_knots' in state_dict else None),
            reference_drive_weights=(
                state_dict.get('reference_drive_weights')
                if 'reference_drive_weights' in state_dict else None),
            episode_len=int(saved_cfg.get('episode_len', 40)),
        ).to(device)
        # All geometry/H0/POD buffers are part of the scientific contract.
        model.load_state_dict(state_dict, strict=True)
        norm_factor = model.action_norm_factor.item()

    elif model_type == 'state_transition':
        # StateTransitionSpatialModel — 闭环状态转移 + 可学习潜变量 z
        required_contract = {
            'model_contract_version': 2,
            'node_order': 'base_to_tip',
            'spatial_propagation_direction': 'base_to_tip',
            'gl_kernel_alignment': 'current_at_window_end',
        }
        actual_contract = {
            key: (saved_cfg or {}).get(key) for key in required_contract
        }
        if actual_contract != required_contract:
            raise ValueError(
                "checkpoint 使用旧的时间/节点方向合同，需按新管线重新"
                "前处理并训练。"
                f" required={required_contract}, actual={actual_contract}")
        n_nodes = saved_cfg.get('n_nodes', 31) if saved_cfg else 31
        z_dim = saved_cfg.get('z_dim', 16) if saved_cfg else 16
        if saved_cfg and 'n_scales' in saved_cfg:
            n_orders = saved_cfg['n_scales']
        else:
            n_orders = train_cfg['temporal'].get('n_scales', 4)
            for k, v in state_dict.items():
                if k == 'temporal.raw_alphas':
                    n_orders = v.shape[0]
                    break
                if k == 'temporal.order_weights':
                    n_orders = v.shape[0]
                    break
        # encoder_type 推断（与 spatial_sequence 分支一致）
        if any('raw_alphas' in k for k in state_dict):
            encoder_type = 'fractional'
        elif any('temporal.k_offsets' in k or 'temporal.logit_lambdas' in k for k in state_dict):
            encoder_type = 'gamma'
        elif any('temporal.cls_token' in k for k in state_dict):
            encoder_type = 'transformer'
        elif any('temporal.tcn_layers' in k for k in state_dict):
            encoder_type = 'tcn'
        elif any('temporal.gru.weight' in k for k in state_dict):
            encoder_type = 'gru'
        elif any('raw_decays' in k for k in state_dict):
            encoder_type = 'ema'
        else:
            encoder_type = saved_cfg.get('encoder_type', 'ema') if saved_cfg else 'ema'
        hidden_dim = saved_cfg.get('hidden_dim', train_cfg['temporal']['hidden_dim']) if saved_cfg else train_cfg['temporal']['hidden_dim']
        episode_len = saved_cfg.get('episode_len', 20) if saved_cfg else 20

        # 按 config.json 的 model 字段区分子类（三类继承同一基类 → state_dict key 完全
        # 相同，无法靠 key 检测区分；子类在 config 里记录了类名）：
        #   GTObservedTransitionModel（方向 14，每步真实 s）
        #   OpenLoopTransitionModel（方向 15，窗口开环：1 帧 GT 种子 + K 步自回归）
        #   StateTransitionSpatialModel（方向 13 基类，默认）
        saved_model_name = (saved_cfg or {}).get('model')
        is_gt_observed = saved_model_name == 'GTObservedTransitionModel'
        is_open_loop = saved_model_name == 'OpenLoopTransitionModel'
        if is_gt_observed:
            from src.models.model_gt_transition import GTObservedTransitionModel
            model = GTObservedTransitionModel(
                action_dim=action_dim, window_size=window_size,
                n_orders=n_orders, hidden_dim=hidden_dim,
                n_nodes=n_nodes, encoder_type=encoder_type, z_dim=z_dim,
                episode_len=episode_len,
            ).to(device)
        elif is_open_loop:
            from src.models.model_open_loop_transition import OpenLoopTransitionModel
            model = OpenLoopTransitionModel(
                action_dim=action_dim, window_size=window_size,
                n_orders=n_orders, hidden_dim=hidden_dim,
                n_nodes=n_nodes, encoder_type=encoder_type, z_dim=z_dim,
                episode_len=episode_len,
            ).to(device)
        else:
            from src.models.model_state_transition import StateTransitionSpatialModel
            model = StateTransitionSpatialModel(
                action_dim=action_dim, window_size=window_size,
                n_orders=n_orders, hidden_dim=hidden_dim,
                n_nodes=n_nodes, encoder_type=encoder_type, z_dim=z_dim,
            ).to(device)
        # 透明迁移 GRUCell → nn.GRU 键（旧 checkpoint 兼容），再加载。
        # 注意：仅本模型（state_transition）用 nn.GRU；spatial_sequence 仍用 GRUCell，
        # 故迁移只在 state_transition 分支调用，避免误改其它模型。
        state_dict = _migrate_gru_keys(state_dict)
        model.load_state_dict(state_dict, strict=False)
        if 'action_norm_factor' in state_dict:
            norm_factor = state_dict['action_norm_factor'].item()

    elif model_type == 'hereditary':
        # HereditaryOperatorModel — 显式迟滞算子模型（PI play + Maxwell）。
        # 算子网格 buffer（thresholds/taus/decays）随 state_dict 恢复，
        # dt/n_play/n_maxwell 从 config.json 透传字段读（缺失时用构造默认）。
        # burnin_mode 缺省 'rest'：旧 checkpoint（F1 修复前）以静息烧入
        # 训练，须按其训练协议评估;新 checkpoint 显式记录该字段。
        from src.models.model_hereditary_operator import HereditaryOperatorModel
        model = HereditaryOperatorModel(
            action_dim=action_dim,
            n_nodes=(saved_cfg or {}).get('n_nodes', 31),
            window_size=window_size,
            n_play=(saved_cfg or {}).get('n_play', 8),
            n_maxwell=(saved_cfg or {}).get('n_maxwell', 6),
            dt=(saved_cfg or {}).get('dt', 0.1),
            burnin_mode=(saved_cfg or {}).get('burnin_mode', 'rest'),
            drive_normalization=(saved_cfg or {}).get(
                'operator_drive_normalization', 'free'),
            residual_scale_max=(saved_cfg or {}).get('residual_scale_max', 0.3),
            episode_len=(saved_cfg or {}).get('episode_len', 40),
        ).to(device)
        # strict 加载：算子网格 buffer 必须精确恢复，静默丢键 = 错误网格
        model.load_state_dict(state_dict)
        if 'action_norm_factor' in state_dict:
            norm_factor = state_dict['action_norm_factor'].item()

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
