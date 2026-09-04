"""train_transition.py — 统一状态转移训练入口。

gt 与 open_loop 是**同一个网络**（都派生自 StateTransitionSpatialModel，state_dict 完全
相同），差别仅在 teacher_forcing_ratio：
  - gt         每步喂真实 s_{t-1}（tf=1.0）→ s 不漂移，部署=每步观测。主线（方向 14）。
  - open_loop  窗口内喂自身预测（tf 退火到 0）→ 开环 rollout，部署=观测一次预测 K 步（方向 15）。
  - hereditary 显式迟滞算子模型（HereditaryOperatorModel，设计文档 Version B）：
               PI play（率无关）+ 广义 Maxwell（率相关）电平读出，无骨架反馈，
               gt/open_loop 对它等价。迟滞谱经 hysteresis_report() 定量读出。
本脚本用 --mode 区分，合并 train_gt_transition / train_open_loop_transition（二者现为薄封装）。

用法:
  # gt（主线，每步真实 s，零漂移）
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \\
      --mode gt --data_dir data/seq_rz_c2_sk

  # open_loop（热启动自最新 gt_transition + 纯闭环 tf=0）
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \\
      --mode open_loop --data_dir data/seq_rz_c2_sk

  # open_loop + tf 退火（drift>50× 才升级；staircase 优先）
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \\
      --mode open_loop --tf_ratio 1.0 --tf_anneal_epochs 15 --tf_schedule staircase

  # hereditary（显式算子;--encoder/--z_dim 被忽略）
  CUDA_VISIBLE_DEVICES=1 python scripts/training/train_transition.py \\
      --mode hereditary --data_dir data/real_seq/<seq>_clean --dt 0.1
"""

import argparse
import glob
import os
import random
import sys

# 默认 cuda1（按用户要求：测试实验用 cuda1）；须在 import torch 前设
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from src.config.args import (  # noqa: E402
    add_common_args, resolve_training_config, build_common_overrides)
from src.utils.data_detect import detect_n_nodes  # noqa: E402
from src.data.action_view import resolve_action_contract  # noqa: E402
from src.registry.paths import ProjectPaths  # noqa: E402
from src.evaluation.real_transition_validation import transition_state_unit  # noqa: E402
from src.training.spec import ValidationSpec  # noqa: E402
from src.training.trainer_unified import UnifiedTrainer  # noqa: E402
from src.training.validation_adapters import transition_validation_adapter  # noqa: E402


def _positive_int_tuple(value):
    try:
        parsed = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是逗号分隔的正整数") from exc
    if not parsed or any(part <= 0 for part in parsed):
        raise argparse.ArgumentTypeError("必须是逗号分隔的正整数")
    return parsed


def _nonnegative_int_tuple(value):
    try:
        parsed = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是逗号分隔的非负整数") from exc
    if not parsed or any(part < 0 for part in parsed):
        raise argparse.ArgumentTypeError("必须是逗号分隔的非负整数")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(description="统一状态转移训练（gt | open_loop）")
    add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
    parser.add_argument("--mode", choices=[
                            "gt", "open_loop", "hereditary",
                            "hereditary_geo", "ishsm"],
                        default="gt",
                        help="gt=每步真实s(tf=1.0,零漂移); open_loop=窗口开环(tf退火到0,喂自身预测); "
                             "hereditary=显式迟滞算子模型; "
                             "hereditary_geo=HOV2.1 算子状态+广义坐标几何读出; "
                             "ishsm=可观测空间基上的8+2低维迟滞状态")
    parser.add_argument("--encoder", type=str, default="fractional",
                        choices=["ema", "fractional", "gamma", "gru", "transformer", "tcn"],
                        help="Temporal encoder type")
    parser.add_argument("--n_nodes", type=int, default=None,
                        help="骨架节点数（None 自动探测）")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="可学习迟滞潜变量 z 的维度")
    parser.add_argument("--episode_len", type=int, default=40,
                        help="窗口/episode 长度 K（z 演化步数；open_loop 即 rollout 视野）")
    parser.add_argument("--dense_step_weight", type=str, default="uniform",
                        choices=["uniform", "linear"],
                        help="dense 监督权重: uniform(等权) | linear(递增,末步权重大)")
    parser.add_argument(
        "--action-channels", default="auto",
        help="Dataset 模型动作视图；auto 读取 NPZ 的 model_action_channels，"
             "也可显式写 0,1,3,4")
    parser.add_argument(
        "--experiment-dir", default=None,
        help="本阶段的实验根目录；流水线传入 stages/gt 或 stages/open_loop")
    parser.add_argument(
        "--val_dir", default=None,
        help="可选独立 val NPZ 目录；提供后由 engine 生成 best_eval_model.pt")
    parser.add_argument("--validation_interval", type=int, default=5,
                        help="有 --val_dir 时每多少 epoch 验证一次")
    parser.add_argument("--validation_max_steps", type=int, default=500,
                        help="每次主线验证最多评估多少帧")
    parser.add_argument("--validation_min_delta", type=float, default=0.0)
    parser.add_argument("--validation_warmup", type=int, default=2,
                        help="早停前不计 patience 的验证次数")
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                        help="按验证次数计；缺省关闭早停但仍按 val 选择 checkpoint")
    parser.add_argument(
        "--allow-early-stop-before-tf-anneal", action="store_true",
        help="允许 OpenLoop 在 teacher-forcing 退火完成前早停（默认禁止）")
    # ── open_loop 专属（gt 模式忽略）──
    parser.add_argument("--init_from", type=str, default=None,
                        help="[open_loop] 热启动 checkpoint（默认自动找最新 "
                             "workspace 和历史 train_log 中的 gt_transition）")
    parser.add_argument("--tf_ratio", type=float, default=0.0,
                        help="[open_loop] 稳态/退火起始 teacher forcing (0.0=纯闭环)")
    parser.add_argument("--tf_anneal_epochs", type=int, default=0,
                        help="[open_loop] tf_ratio→tf_min 退火 epoch 数 (0=不退火,固定 tf_ratio)")
    parser.add_argument("--tf_min", type=float, default=0.0,
                        help="[open_loop] 退火下限 teacher forcing")
    parser.add_argument("--tf_schedule", type=str, default="staircase",
                        choices=["linear", "staircase"],
                        help="[open_loop] 退火形状: staircase(前半 nominal/后半 tf_min) | linear")
    # ── hereditary 专属（其它模式忽略）──
    parser.add_argument("--n_play", type=int, default=2,
                        help="[hereditary] PI play 算子数 J（率无关迟滞容量，下限 1）。"
                             "F4: v1 分析显示本数据上 play bank 近乎死（LOO +0.32mm），"
                             "默认降为 2 仅保留 E1 可证伪的最小容量")
    parser.add_argument("--n_maxwell", type=int, default=6,
                        help="[hereditary] Maxwell 元件数 M（率相关迟滞容量）")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="[hereditary] 采样间隔秒（必须与数据合同一致; 实物 10Hz → 0.1）")
    parser.add_argument("--tau_max", type=float, default=2.0,
                        help="[hereditary] Maxwell 时间常数网格上界秒（下界固定 3·dt）。"
                             "F2: 默认 2.0 把全部元素收进 episode 时域内可辨识带"
                             "（40 步 x dt=4s 时 tau=2s 在 episode 内已基本弛豫）;"
                             "配合 F1 equilibrium 烧入消除慢模态伪静态 aliasing")
    parser.add_argument("--burnin_mode", choices=["equilibrium", "rest"], default="equilibrium",
                        help="[hereditary] 冷启动烧入方式: equilibrium=窗口首动作平衡态起烧"
                             "（F1 修复，消除慢 Maxwell 模态伪静态 aliasing）; rest=旧行为（A/B 对照）")
    parser.add_argument(
        "--operator_drive_normalization", choices=["free", "unit_range"],
        default="unit_range",
        help="[hereditary_geo] unit_range 固定每通道 e(1)=1，移除驱动幅值的"
             "重参数化；free 仅用于旧模型 A/B")
    parser.add_argument("--residual_scale_max", type=float, default=0.3,
                        help="[hereditary] 残差幅度上限（归一化骨架单位）。F5: 修 F1 后重训，"
                             "若 residual_scale 仍钉在此上限则是真容量信号（可上调做对照实验）")
    parser.add_argument(
        "--hov21_residual", choices=["none", "memory"], default="none",
        help="[hereditary_geo] none=纯 PI/Maxwell 广义读出；memory=严格零平衡的小记忆残差")
    parser.add_argument(
        "--hov21_bend_residual_max_rad", type=float, default=0.05,
        help="[hereditary_geo] 每个 POD 弯曲系数的记忆残差硬上界（rad）")
    parser.add_argument(
        "--hov21_length_residual_max_log", type=float, default=0.02,
        help="[hereditary_geo] 每段 log-length 记忆残差硬上界")
    parser.add_argument(
        "--hov21_residual_bend_weight", type=float, default=0.1,
        help="[hereditary_geo] 物理弯曲记忆残差平方惩罚权重")
    parser.add_argument(
        "--hov21_residual_length_weight", type=float, default=0.1,
        help="[hereditary_geo] log-length 记忆残差平方惩罚权重")
    # ── ISHSM 专属（其它模式忽略）──
    parser.add_argument("--n_bend_modes", type=int, default=8,
                        help="[ishsm] 固定 POD 弯曲空间基数量")
    parser.add_argument(
        "--bend_basis_kind", choices=["pod", "local"], default="pod",
        help="[hereditary_geo] pod=压缩空间基；local=每个骨架线段的局部"
             "弯曲角（要求 n_bend_modes=N-1）")
    parser.add_argument("--section_intervals", type=_positive_int_tuple,
                        default=(7, 7),
                        help="[ishsm] 每段包含的骨架线段数，默认 7,7")
    parser.add_argument("--tau_min", type=float, default=0.3,
                        help="[ishsm] 可学习有效时间常数下界（秒）")
    parser.add_argument("--disable_dynamic_length", action="store_true",
                        help="[ishsm消融] 移除2个分段动态伸长状态，仅保留H0长度")
    parser.add_argument("--bend_loss_weight", type=float, default=0.05,
                        help="[ishsm] 广义弯曲坐标损失权重")
    parser.add_argument("--length_loss_weight", type=float, default=0.1,
                        help="[ishsm] 分段长度坐标损失权重")
    parser.add_argument("--endpoint_loss_weight", type=float, default=1.0,
                        help="[ishsm] 归一化端点重建 guardrail 权重")
    parser.add_argument("--h0_reference", choices=["linear", "monotone_spline"],
                        default="monotone_spline",
                        help="[ishsm] fit-only 冻结的记忆无关参考类型")
    parser.add_argument("--h0_knots", type=int, default=5,
                        help="[ishsm] 单调驱动铰链节点数")
    parser.add_argument("--h0_fit_steps", type=int, default=500,
                        help="[ishsm] 单调 H0 的 fit-only 优化步数")
    parser.add_argument(
        "--h0_fit_objective", choices=["coordinate", "geometry"],
        default="geometry",
        help="[ishsm] H0 拟合目标；geometry 同时约束重建节点和端点")
    parser.add_argument("--h0_geometry_weight", type=float, default=1.0,
                        help="[ishsm] H0 重建节点相对损失权重")
    parser.add_argument("--h0_endpoint_weight", type=float, default=0.25,
                        help="[ishsm] H0 重建端点相对损失权重")
    persistence = parser.add_mutually_exclusive_group()
    persistence.add_argument(
        "--enable_persistent_state", dest="use_persistent_state",
        action="store_true",
        help="[ishsm实验] 启用由观测初始化、保持到下次重锚定的形状偏置")
    persistence.add_argument(
        "--disable_persistent_state", dest="use_persistent_state",
        action="store_false",
        help="[ishsm] 禁用持久形状偏置（v2 证据支持的默认值）")
    parser.set_defaults(use_persistent_state=False)
    parser.add_argument("--persistence_init", type=float, default=0.1,
                        help="[ishsm] 锚点残差初始分配到持久分量的比例")
    parser.add_argument(
        "--ishsm_observation_update", choices=["hard", "innovation"],
        default="innovation",
        help="[ishsm] 观测校正；innovation 使用两个有界弯曲/伸长增益")
    parser.add_argument(
        "--ishsm_observation_gain_init", type=float, default=0.25,
        help="[ishsm] innovation 观测增益初值，必须在 (0,1)")
    parser.add_argument(
        "--ishsm_training_reanchor_intervals",
        type=_nonnegative_int_tuple, default=(0, 5, 10, 20),
        help="[ishsm] 每个训练 batch 采样的因果重锚间隔；0=只用初始锚点")
    parser.add_argument(
        "--ishsm_tau_parameterization",
        choices=["independent", "shared_bending"],
        default="shared_bending",
        help="[ishsm] shared_bending 保留8个空间模态但共享一个弯曲时间常数")
    parser.add_argument(
        "--ishsm_observation_projection", choices=["modal", "tip_dls"],
        default="tip_dls",
        help="[ishsm] 观测到8+2状态的投影；tip_dls 在 modal 投影后用解析"
             "端点 Jacobian 做阻尼最小二乘校正")
    parser.add_argument(
        "--ishsm_tip_dls_lambda_mm2", type=float, default=1.0,
        help="[ishsm] tip_dls 阻尼系数，单位 mm^2，必须为正")
    parser.add_argument(
        "--ishsm_validation_protocol",
        choices=["single_anchor", "periodic_40"], default="single_anchor",
        help="[ishsm] validation-best 选择协议；periodic_40 每40帧重锚定一次")
    parser.add_argument(
        "--ishsm_reanchor_interval", type=int, default=40,
        help="[ishsm] periodic_40 验证的重锚定帧间隔")
    return parser


def _load_ishsm_fit_arrays(data_dir, action_channels, norm_factor):
    """Load only the declared fit directory for frozen H0/POD estimation."""
    from src.data.action_view import project_actions

    actions, skeletons = [], []
    paths = sorted(glob.glob(os.path.join(str(data_dir), "*.npz")))
    if not paths:
        raise FileNotFoundError(f"ISHSM fit 目录没有 NPZ: {data_dir}")
    for path in paths:
        with np.load(path, allow_pickle=False) as raw:
            actions.append(project_actions(
                raw["actions"], action_channels).astype(np.float32) /
                float(norm_factor))
            skeletons.append(
                raw["positions"].astype(np.float32).transpose(0, 2, 1))
    return np.concatenate(actions), np.concatenate(skeletons)


def configure_transition_validation(args, phase_spec, config):
    """Attach an explicit native-unit validator when ``--val_dir`` is given."""
    phase_spec.validation = None
    if not args.val_dir:
        return None, None
    if args.validation_max_steps <= 0:
        raise ValueError("validation_max_steps 必须为正整数")
    unit = transition_state_unit(args.val_dir)
    metric = f"validation.node_mean_{unit}"
    phase_spec.validation = ValidationSpec(
        selection_metric=metric,
        selection_mode="min",
        eval_interval_epochs=args.validation_interval,
        min_delta=args.validation_min_delta,
        warmup_evaluations=args.validation_warmup,
        early_stopping_patience_evaluations=args.early_stopping_patience,
        lr_scheduler_metric=metric,
        restore_best_at_end=True,
        allow_early_stop_before_tf_anneal=(
            args.allow_early_stop_before_tf_anneal),
    )
    config.setdefault("evaluation", {})[
        "transition_validation_max_steps"] = args.validation_max_steps
    if getattr(args, "mode", None) == "ishsm":
        interval = int(getattr(args, "ishsm_reanchor_interval", 40))
        if interval <= 0:
            raise ValueError("ISHSM validation 重锚定间隔必须为正整数")
        config["evaluation"]["ishsm_validation_protocol"] = getattr(
            args, "ishsm_validation_protocol", "single_anchor")
        config["evaluation"]["ishsm_reanchor_interval"] = interval
    return (
        {phase_spec.name: args.val_dir},
        {phase_spec.name: transition_validation_adapter},
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode in {"ishsm", "hereditary_geo"} and any(weight < 0 for weight in (
            args.bend_loss_weight, args.length_loss_weight,
            args.endpoint_loss_weight, args.h0_geometry_weight,
            args.h0_endpoint_weight)):
        raise ValueError("广义几何 loss/H0 objective 权重必须非负")
    if args.mode == "hereditary_geo":
        if args.hov21_bend_residual_max_rad <= 0:
            raise ValueError("HOV2.1 bend residual 上界必须为正")
        if args.hov21_length_residual_max_log <= 0:
            raise ValueError("HOV2.1 length residual 上界必须为正")
        if (args.hov21_residual_bend_weight < 0 or
                args.hov21_residual_length_weight < 0):
            raise ValueError("HOV2.1 residual loss 权重必须非负")
    if args.mode == "ishsm":
        if not 0.0 < args.ishsm_observation_gain_init < 1.0:
            raise ValueError("ISHSM observation gain 初值必须在 (0,1)")
        if args.ishsm_tip_dls_lambda_mm2 <= 0:
            raise ValueError("ISHSM tip DLS lambda 必须为正数")
        if (args.ishsm_observation_update == "innovation" and
                args.use_persistent_state):
            raise ValueError(
                "ISHSM innovation observer 不能与旧 persistent state 同时启用")
    config = resolve_training_config(build_common_overrides(args))
    seed = config.get("optimization", {}).get("seed")
    if seed is not None:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Reproducibility seed: {seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_contract = resolve_action_contract(args.data_dir, args.action_channels)
    action_dim = action_contract.model_action_dim
    config["action_view"] = action_contract.to_dict()
    n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)
    temp_cfg = config["temporal"]
    hidden_dim = temp_cfg["hidden_dim"]

    # Build the fit-only dataset before model construction so ISHSM's frozen
    # H0/POD priors use exactly the same action normalization as training.
    from src.data.dataset_spatial import StateTransitionDataset
    norm_dataset = StateTransitionDataset(
        args.data_dir, seq_len=temp_cfg["window_size"],
        episode_mode=True, episode_len=args.episode_len,
        action_channels=action_contract.model_action_channels)
    pc_center, pc_scale = norm_dataset.get_normalization_params()

    # ── 按 mode 构造模型（同网络，不同 training_spec + mode buffer）──
    if args.mode == "gt":
        from src.models.model_gt_transition import GTObservedTransitionModel
        model = GTObservedTransitionModel(
            action_dim=action_dim, n_nodes=n_nodes, hidden_dim=hidden_dim,
            window_size=temp_cfg["window_size"], n_orders=temp_cfg["n_scales"],
            encoder_type=args.encoder, z_dim=args.z_dim,
            episode_len=args.episode_len).to(device)
        spec = model.training_spec
        spec.phases[0].dense_step_weight = args.dense_step_weight
        model_tag = "gt_transition"
        tf_info = f"tf={spec.phases[0].teacher_forcing_ratio}"
    elif args.mode == "hereditary":
        # 显式迟滞算子模型: 算子状态经 latent_z 槽 BPTT，无骨架反馈
        # （--encoder/--z_dim 属于隐式潜变量家族，本模型无此构件，忽略）
        from src.models.model_hereditary_operator import HereditaryOperatorModel
        model = HereditaryOperatorModel(
            action_dim=action_dim, n_nodes=n_nodes,
            window_size=temp_cfg["window_size"],
            n_play=args.n_play, n_maxwell=args.n_maxwell, dt=args.dt,
            tau_range=(3.0 * args.dt, args.tau_max),
            burnin_mode=args.burnin_mode,
            residual_scale_max=args.residual_scale_max,
            episode_len=args.episode_len).to(device)
        spec = model.training_spec
        spec.phases[0].dense_step_weight = args.dense_step_weight
        model_tag = "hereditary"
        tf_info = (f"operators: J={args.n_play} plays, M={args.n_maxwell} maxwell, "
                   f"dt={args.dt}s (tf n/a — 无骨架反馈)")
    elif args.mode == "hereditary_geo":
        from src.models.model_hereditary_geometry import HereditaryGeometryModel
        from src.models.model_ishsm import fit_ishsm_priors_from_arrays
        if sum(args.section_intervals) != n_nodes - 1:
            raise ValueError(
                f"--section_intervals={args.section_intervals} 总和必须为 "
                f"n_nodes-1={n_nodes - 1}")
        if not 1 <= args.n_bend_modes <= n_nodes - 1:
            raise ValueError("--n_bend_modes 必须在 [1,n_nodes-1]")
        fit_actions, fit_skeletons = _load_ishsm_fit_arrays(
            args.data_dir, action_contract.model_action_channels,
            norm_dataset.norm_factor)
        priors = fit_ishsm_priors_from_arrays(
            fit_actions, fit_skeletons,
            n_bend_modes=args.n_bend_modes,
            section_intervals=args.section_intervals,
            reference_kind=args.h0_reference,
            n_reference_knots=args.h0_knots,
            reference_fit_steps=args.h0_fit_steps,
            reference_fit_objective=args.h0_fit_objective,
            reference_geometry_weight=args.h0_geometry_weight,
            reference_endpoint_weight=args.h0_endpoint_weight,
            bend_basis_kind=args.bend_basis_kind)
        model = HereditaryGeometryModel(
            action_dim=action_dim, n_nodes=n_nodes,
            window_size=temp_cfg["window_size"],
            n_play=args.n_play, n_maxwell=args.n_maxwell, dt=args.dt,
            tau_range=(3.0 * args.dt, args.tau_max),
            burnin_mode=args.burnin_mode,
            drive_normalization=args.operator_drive_normalization,
            n_bend_modes=args.n_bend_modes,
            section_intervals=args.section_intervals,
            bend_basis=priors["bend_basis"],
            generalized_coordinate_scale=
                priors["generalized_coordinate_scale"],
            reference_segment_lengths=priors["reference_segment_lengths"],
            reference_bend_bias=priors["reference_bend_bias"],
            reference_bend_dirs=priors["reference_bend_dirs"],
            reference_length_bias=priors["reference_length_bias"],
            reference_length_dirs=priors["reference_length_dirs"],
            reference_kind=priors["reference_kind"],
            reference_knots=priors["reference_knots"],
            reference_drive_weights=priors["reference_drive_weights"],
            bend_basis_kind=priors["bend_basis_kind"],
            base_position=priors["base_position"],
            residual_mode=args.hov21_residual,
            bend_residual_max_rad=args.hov21_bend_residual_max_rad,
            length_residual_max_log=args.hov21_length_residual_max_log,
            episode_len=args.episode_len).to(device)
        spec = model.training_spec
        spec.phases[0].dense_step_weight = args.dense_step_weight
        model_tag = "hereditary_geometry"
        tf_info = (
            f"HOV2.1 operators J={args.n_play}, M={args.n_maxwell}; "
            f"geometry={args.n_bend_modes}+{len(args.section_intervals)}; "
            f"residual={args.hov21_residual}")
        config["loss_weights"].update({
            "bend": args.bend_loss_weight,
            "length": args.length_loss_weight,
            "endpoint": args.endpoint_loss_weight,
            "residual_bend": args.hov21_residual_bend_weight,
            "residual_length": args.hov21_residual_length_weight,
        })
    elif args.mode == "ishsm":
        from src.models.model_ishsm import ISHSMModel, fit_ishsm_priors_from_arrays
        if sum(args.section_intervals) != n_nodes - 1:
            raise ValueError(
                f"--section_intervals={args.section_intervals} 总和必须为 "
                f"n_nodes-1={n_nodes - 1}")
        if not 1 <= args.n_bend_modes <= n_nodes - 1:
            raise ValueError("--n_bend_modes 必须在 [1,n_nodes-1]")
        if not (0 < args.tau_min <= args.tau_max):
            raise ValueError("ISHSM 要求 0 < tau_min <= tau_max")
        fit_actions, fit_skeletons = _load_ishsm_fit_arrays(
            args.data_dir, action_contract.model_action_channels,
            norm_dataset.norm_factor)
        priors = fit_ishsm_priors_from_arrays(
            fit_actions, fit_skeletons,
            n_bend_modes=args.n_bend_modes,
            section_intervals=args.section_intervals,
            reference_kind=args.h0_reference,
            n_reference_knots=args.h0_knots,
            reference_fit_steps=args.h0_fit_steps,
            reference_fit_objective=args.h0_fit_objective,
            reference_geometry_weight=args.h0_geometry_weight,
            reference_endpoint_weight=args.h0_endpoint_weight)
        model = ISHSMModel(
            action_dim=action_dim, n_nodes=n_nodes,
            n_bend_modes=args.n_bend_modes,
            section_intervals=args.section_intervals,
            dt=args.dt, tau_range=(args.tau_min, args.tau_max),
            bend_basis=priors["bend_basis"],
            reference_segment_lengths=priors["reference_segment_lengths"],
            reference_bend_bias=priors["reference_bend_bias"],
            reference_bend_dirs=priors["reference_bend_dirs"],
            reference_length_bias=priors["reference_length_bias"],
            reference_length_dirs=priors["reference_length_dirs"],
            reference_kind=priors["reference_kind"],
            reference_knots=priors["reference_knots"],
            reference_drive_weights=priors["reference_drive_weights"],
            base_position=priors["base_position"],
            use_dynamic_length=not args.disable_dynamic_length,
            use_persistent_state=args.use_persistent_state,
            persistence_init=args.persistence_init,
            observation_update=args.ishsm_observation_update,
            observation_gain_init=args.ishsm_observation_gain_init,
            training_reanchor_intervals=
                args.ishsm_training_reanchor_intervals,
            tau_parameterization=args.ishsm_tau_parameterization,
            observation_projection=args.ishsm_observation_projection,
            tip_dls_lambda_mm2=args.ishsm_tip_dls_lambda_mm2,
            episode_len=args.episode_len).to(device)
        spec = model.training_spec
        spec.phases[0].dense_step_weight = args.dense_step_weight
        model_tag = "ishsm"
        tf_info = (f"memory={model.z_dim}D over "
                   f"{args.n_bend_modes}+{model.n_length_states} spatial coordinates, "
                   f"tau=[{args.tau_min},{args.tau_max}]s, "
                   f"observer={args.ishsm_observation_update}, "
                   f"projection={args.ishsm_observation_projection}, "
                   f"train K={args.ishsm_training_reanchor_intervals}")
        config["loss_weights"]["bend"] = args.bend_loss_weight
        config["loss_weights"]["length"] = args.length_loss_weight
        config["loss_weights"]["endpoint"] = args.endpoint_loss_weight
    else:  # open_loop
        from src.models.model_open_loop_transition import OpenLoopTransitionModel
        model = OpenLoopTransitionModel(
            action_dim=action_dim, n_nodes=n_nodes, hidden_dim=hidden_dim,
            window_size=temp_cfg["window_size"], n_orders=temp_cfg["n_scales"],
            encoder_type=args.encoder, z_dim=args.z_dim,
            episode_len=args.episode_len).to(device)
        _warm_start_open_loop(model, args.init_from, device, action_dim=action_dim)
        p0 = model.training_spec.phases[0]
        p0.teacher_forcing_ratio = args.tf_ratio
        p0.tf_anneal_epochs = args.tf_anneal_epochs
        p0.tf_min = args.tf_min
        p0.tf_schedule = args.tf_schedule
        p0.dense_step_weight = args.dense_step_weight
        spec = model.training_spec
        model_tag = "open_loop_transition"
        tf_info = (f"tf_ratio={args.tf_ratio}, anneal={args.tf_anneal_epochs}ep, "
                   f"tf_min={args.tf_min}, schedule={args.tf_schedule}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_tag}（mode={args.mode}）")
    if args.mode in {"hereditary", "hereditary_geo"}:
        print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, "
              f"episode_len(K): {args.episode_len}, dt: {model.dt.item():g}s")
        print(f"  Play r-grid: {[f'{r:.3f}' for r in model.play.thresholds.tolist()]}")
        print(f"  Maxwell tau-grid: {[f'{t:.2f}' for t in model.maxwell.taus.tolist()]}s")
        if args.mode == "hereditary_geo":
            print(f"  Geometry: {model.n_bend_modes}+{model.n_sections}, "
                  f"POD fit energy: {priors['bend_explained_energy']:.4%}")
            print(f"  H0: {model.reference_kind}/{args.h0_fit_objective}, "
                  f"node/tip RMSE: {priors['reference_fit_node_rmse_mm']:.4f}/"
                  f"{priors['reference_fit_endpoint_rmse_mm']:.4f} mm")
    elif args.mode == "ishsm":
        print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, "
              f"state: {model.n_bend_modes}+{model.n_length_states}, "
              f"POD fit energy: {priors['bend_explained_energy']:.4%}")
        print(f"  Section intervals: {model.section_intervals}; "
              f"initial taus: {[round(v, 3) for v in model.taus.tolist()]}")
        print(f"  H0: {model.reference_kind}/{args.h0_fit_objective}, "
              f"fit MSE: {priors['reference_fit_mse']:.6g}, "
              f"node/tip RMSE: {priors['reference_fit_node_rmse_mm']:.4f}/"
              f"{priors['reference_fit_endpoint_rmse_mm']:.4f} mm; persistent: "
              f"{model.use_persistent_state}")
        print(f"  Observer: {model.observation_update}, "
              f"initial gains: {[round(v, 3) for v in model.observation_gains.tolist()]}; "
              f"projection: {model.observation_projection} "
              f"(lambda={model.tip_dls_lambda_mm2:g} mm^2); "
              f"tau parameterization: {model.tau_parameterization}")
    else:
        print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, Encoder: {args.encoder}, "
              f"z_dim: {args.z_dim}, episode_len(K): {args.episode_len}")
    print(f"  Action view: raw={action_contract.raw_action_dim}D -> "
          f"channels={action_contract.model_action_channels} -> model={action_dim}D")
    print(f"  {tf_info}, dense_step_weight: {args.dense_step_weight}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Active losses: {spec.phases[0].active_losses}")

    if args.mode == "hereditary":
        # 合同字段进 config.json（dt 必须可追溯——设计文档 §六）
        config["dt"] = args.dt
        config["n_play"] = args.n_play
        config["n_maxwell"] = args.n_maxwell
        config["tau_max"] = args.tau_max
        config["burnin_mode"] = args.burnin_mode
        config["residual_scale_max"] = args.residual_scale_max
    elif args.mode == "hereditary_geo":
        config.update({
            "dt": args.dt,
            "n_play": args.n_play,
            "n_maxwell": args.n_maxwell,
            "tau_max": args.tau_max,
            "burnin_mode": args.burnin_mode,
            "operator_drive_normalization":
                args.operator_drive_normalization,
            "bend_basis_kind": args.bend_basis_kind,
            "n_bend_modes": args.n_bend_modes,
            "section_intervals": list(args.section_intervals),
            "h0_reference": f"fit_only_frozen_{args.h0_reference}",
            "h0_knots": args.h0_knots,
            "h0_fit_steps": args.h0_fit_steps,
            "h0_fit_objective": args.h0_fit_objective,
            "h0_geometry_weight": args.h0_geometry_weight,
            "h0_endpoint_weight": args.h0_endpoint_weight,
            "reference_fit_mse": priors["reference_fit_mse"],
            "reference_fit_node_rmse_mm":
                priors["reference_fit_node_rmse_mm"],
            "reference_fit_endpoint_rmse_mm":
                priors["reference_fit_endpoint_rmse_mm"],
            "bend_explained_energy": priors["bend_explained_energy"],
            "generalized_coordinate_scale":
                priors["generalized_coordinate_scale"].tolist(),
            "hov21_residual": args.hov21_residual,
            "hov21_bend_residual_max_rad":
                args.hov21_bend_residual_max_rad,
            "hov21_length_residual_max_log":
                args.hov21_length_residual_max_log,
        })
    elif args.mode == "ishsm":
        config.update({
            "dt": args.dt,
            "n_bend_modes": args.n_bend_modes,
            "section_intervals": list(args.section_intervals),
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "use_dynamic_length": not args.disable_dynamic_length,
            "use_persistent_state": args.use_persistent_state,
            "persistence_init": args.persistence_init,
            "observation_update": args.ishsm_observation_update,
            "observation_gain_init": args.ishsm_observation_gain_init,
            "training_reanchor_intervals":
                list(args.ishsm_training_reanchor_intervals),
            "tau_parameterization": args.ishsm_tau_parameterization,
            "observation_projection": args.ishsm_observation_projection,
            "tip_dls_lambda_mm2": args.ishsm_tip_dls_lambda_mm2,
            "observation_protocol": "single_anchor_or_periodic_reanchor",
            "h0_reference": f"fit_only_frozen_{args.h0_reference}",
            "h0_knots": args.h0_knots,
            "h0_fit_steps": args.h0_fit_steps,
            "h0_fit_objective": args.h0_fit_objective,
            "h0_geometry_weight": args.h0_geometry_weight,
            "h0_endpoint_weight": args.h0_endpoint_weight,
            "endpoint_loss_weight": args.endpoint_loss_weight,
            "reference_fit_mse": priors["reference_fit_mse"],
            "reference_fit_node_rmse_mm":
                priors["reference_fit_node_rmse_mm"],
            "reference_fit_endpoint_rmse_mm":
                priors["reference_fit_endpoint_rmse_mm"],
            "bend_explained_energy": priors["bend_explained_energy"],
        })

    model.set_normalization(pc_center, pc_scale, norm_dataset.norm_factor)
    config["state_view"] = norm_dataset.get_state_contract()

    data_dirs = {"sequence": args.data_dir}
    validation_data_dirs, validation_adapters = configure_transition_validation(
        args, spec.phases[0], config)
    trainer = UnifiedTrainer(model, view_strategy=None, config=config,
                             model_tag=model_tag)
    trainer.train(
        data_dirs,
        exp_dir=args.experiment_dir,
        validation_data_dirs=validation_data_dirs,
        validation_adapters=validation_adapters,
    )


def _ckpt_action_dim(ckpt_path):
    """读 checkpoint 的 action_dim：优先 sibling config.json，否则从 state_mlp 权重推断。

    state_mlp.0.weight 形状 [hidden, 6*action_dim]（输入拼接 [cond, flatten(s_{t-1}), v]）。
    """
    import json
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path)))  # .../exp_X
    cfg = os.path.join(exp_dir, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg) as f:
                ad = json.load(f).get("action_dim")
            if ad is not None:
                return int(ad)
        except Exception:
            pass
    try:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        w = sd.get("temporal.state_mlp.0.weight")
        if w is not None and w.dim() == 2 and w.shape[1] % 6 == 0:
            return int(w.shape[1] // 6)
    except Exception:
        pass
    return None


def _warm_start_open_loop(model, init_from, device, action_dim=None):
    """open_loop 从 gt_transition checkpoint 热启动单步动力学。

    ⚠️ 必须 _migrate_gru_keys（gt ckpt 用旧 GRUCell 键名，strict=False 会静默丢弃整层 GRU）。
    ⚠️ 自动检测时按 action_dim 过滤候选——否则可能选到仿真 ckpt(ad=2)套到实物模型(ad=1)
       上 → state_mlp size mismatch 崩溃。传 action_dim 后只挑匹配的 checkpoint。
    """
    if init_from is None:
        cands = _default_gt_checkpoint_candidates()
        if action_dim is not None and cands:
            cands = [c for c in cands if _ckpt_action_dim(c) == action_dim]
        if cands:
            # exp_* 的数字后缀不能按字符串排序（exp_9 会排在 exp_13 后面）。
            # mtime 也能保证顺序训练时选到刚完成的 GT checkpoint。
            init_from = max(cands, key=os.path.getmtime)
            print(f"[warm-start] 自动检测 gt_transition checkpoint (action_dim={action_dim}): {init_from}")
    if init_from is None:
        print(f"[warm-start] 未找到 action_dim={action_dim} 的 gt_transition checkpoint — 从头冷启动。")
        return
    if not os.path.exists(init_from):
        raise FileNotFoundError(f"--init_from 不存在: {init_from}")
    from src.utils.model_loader import _load_config_json, _migrate_gru_keys
    saved_cfg = _load_config_json(init_from) or {}
    required_contract = {
        "model_contract_version": model.model_contract_version,
        "node_order": model.node_order,
        "spatial_propagation_direction": model.spatial_propagation_direction,
        "gl_kernel_alignment": model.gl_kernel_alignment,
    }
    actual_contract = {key: saved_cfg.get(key) for key in required_contract}
    if actual_contract != required_contract:
        raise ValueError(
            "GT 热启动 checkpoint 与当前时间/节点方向合同不一致；"
            f" required={required_contract}, actual={actual_contract}")
    sd = torch.load(init_from, map_location=device, weights_only=True)
    sd = _migrate_gru_keys(sd)
    incompatible = model.load_state_dict(sd, strict=False)
    # 仅允许已知 mode buffer 未匹配；任何 trained module 缺失 = 静默丢权重 = BLOCKER
    safe = {"gt_observed_mode", "open_loop_mode"}
    real_missing = [k for k in incompatible.missing_keys if k not in safe]
    assert not real_missing, (
        f"[warm-start BLOCKER] trained keys dropped (GRU etc.): {real_missing}. "
        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    # reset delta_scale 到收缩值：gt 训练的 delta_scale(~4)对 open_loop 太大→tf=0 rollout
    # 发散→BPTT 梯度 NaN。配合 model 的 delta_scale_max clamp，从 0.1 起在收缩区重新学。
    if hasattr(model, 'delta_scale'):
        with torch.no_grad():
            model.delta_scale.fill_(0.1)
        print(f"  reset delta_scale=0.1 (gt 值对开环太大→发散 NaN；clamp_max="
              f"{getattr(model, 'delta_scale_max', 'inf')})")
    print(f"[warm-start] loaded {init_from}")
    print(f"  missing(应仅 mode buffer)={incompatible.missing_keys}")
    print(f"  unexpected(应仅 gt_observed_mode)={incompatible.unexpected_keys}")


def _default_gt_checkpoint_candidates(paths=None):
    """按“统一 workspace + 历史只读根”收集 GT 热启动候选。

    路径由 ``ProjectPaths`` 解析，因此调用者的当前工作目录不会改变搜索范围。
    返回字符串是为了保持下游 ``os.path``/``torch.load`` 的现有接口不变。
    """
    paths = paths or ProjectPaths.load()
    study_roots = [paths.training_study("gt_transition")]
    study_roots.extend(
        paths.legacy_candidates("training", "gt_transition"))

    candidates = []
    seen = set()
    pattern = "*/phase_gt_transition/model/best_model.pt"
    for study_root in study_roots:
        for candidate in study_root.glob(pattern):
            resolved = candidate.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(str(candidate))
    return candidates


if __name__ == "__main__":
    main()
