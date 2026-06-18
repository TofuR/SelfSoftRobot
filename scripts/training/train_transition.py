"""train_transition.py — 统一状态转移训练入口（--mode gt | open_loop）。

gt 与 open_loop 是**同一个网络**（都派生自 StateTransitionSpatialModel，state_dict 完全
相同），差别仅在 teacher_forcing_ratio：
  - gt         每步喂真实 s_{t-1}（tf=1.0）→ s 不漂移，部署=每步观测。主线（方向 14）。
  - open_loop  窗口内喂自身预测（tf 退火到 0）→ 开环 rollout，部署=观测一次预测 K 步（方向 15）。
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
"""

import argparse
import glob
import os
import sys

# 默认 cuda1（按用户要求：测试实验用 cuda1）；须在 import torch 前设
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch  # noqa: E402

from src.config.args import (  # noqa: E402
    add_common_args, resolve_training_config, build_common_overrides)
from src.utils.data_detect import detect_action_dim, detect_n_nodes  # noqa: E402
from src.training.trainer_unified import UnifiedTrainer  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(description="统一状态转移训练（gt | open_loop）")
    add_common_args(parser, data_dir_default="data/seq_rz_c2_sk")
    parser.add_argument("--mode", choices=["gt", "open_loop"], default="gt",
                        help="gt=每步真实s(tf=1.0,零漂移); open_loop=窗口开环(tf退火到0,喂自身预测)")
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
    # ── open_loop 专属（gt 模式忽略）──
    parser.add_argument("--init_from", type=str, default=None,
                        help="[open_loop] 热启动 checkpoint（默认自动找最新 "
                             "train_log/gt_transition/*/phase_gt_transition/model/best_model.pt）")
    parser.add_argument("--tf_ratio", type=float, default=0.0,
                        help="[open_loop] 稳态/退火起始 teacher forcing (0.0=纯闭环)")
    parser.add_argument("--tf_anneal_epochs", type=int, default=0,
                        help="[open_loop] tf_ratio→tf_min 退火 epoch 数 (0=不退火,固定 tf_ratio)")
    parser.add_argument("--tf_min", type=float, default=0.0,
                        help="[open_loop] 退火下限 teacher forcing")
    parser.add_argument("--tf_schedule", type=str, default="staircase",
                        choices=["linear", "staircase"],
                        help="[open_loop] 退火形状: staircase(前半 nominal/后半 tf_min) | linear")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = resolve_training_config(build_common_overrides(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    action_dim = detect_action_dim(args.data_dir)
    n_nodes = args.n_nodes or detect_n_nodes(args.data_dir)
    temp_cfg = config["temporal"]
    hidden_dim = temp_cfg["hidden_dim"]

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
    else:  # open_loop
        from src.models.model_open_loop_transition import OpenLoopTransitionModel
        model = OpenLoopTransitionModel(
            action_dim=action_dim, n_nodes=n_nodes, hidden_dim=hidden_dim,
            window_size=temp_cfg["window_size"], n_orders=temp_cfg["n_scales"],
            encoder_type=args.encoder, z_dim=args.z_dim,
            episode_len=args.episode_len).to(device)
        _warm_start_open_loop(model, args.init_from, device)
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
    print(f"  Action dim: {action_dim}, N nodes: {n_nodes}, Encoder: {args.encoder}, "
          f"z_dim: {args.z_dim}, episode_len(K): {args.episode_len}")
    print(f"  {tf_info}, dense_step_weight: {args.dense_step_weight}")
    print(f"  Parameters: {n_params:,}")
    print(f"  Active losses: {spec.phases[0].active_losses}")

    # ── 归一化（episode 模式数据集，与训练一致）──
    from src.data.dataset_spatial import StateTransitionDataset
    norm_dataset = StateTransitionDataset(
        args.data_dir, seq_len=temp_cfg["window_size"],
        episode_mode=True, episode_len=args.episode_len)
    pc_center, pc_scale = norm_dataset.get_normalization_params()
    model.set_normalization(pc_center, pc_scale, norm_dataset.norm_factor)

    data_dirs = {"sequence": args.data_dir}
    trainer = UnifiedTrainer(model, view_strategy=None, config=config,
                             model_tag=model_tag)
    trainer.train(data_dirs)


def _warm_start_open_loop(model, init_from, device):
    """open_loop 从 gt_transition checkpoint 热启动单步动力学。

    ⚠️ 必须 _migrate_gru_keys（gt ckpt 用旧 GRUCell 键名，strict=False 会静默丢弃整层 GRU）。
    """
    if init_from is None:
        cands = sorted(glob.glob(os.path.join(
            "train_log", "gt_transition", "*", "phase_gt_transition", "model",
            "best_model.pt")))
        if cands:
            init_from = cands[-1]
            print(f"[warm-start] 自动检测 gt_transition checkpoint: {init_from}")
    if init_from is None:
        print("[warm-start] 未找到 gt_transition checkpoint — 从头冷启动。")
        return
    if not os.path.exists(init_from):
        raise FileNotFoundError(f"--init_from 不存在: {init_from}")
    from src.utils.model_loader import _migrate_gru_keys
    sd = torch.load(init_from, map_location=device, weights_only=True)
    sd = _migrate_gru_keys(sd)
    incompatible = model.load_state_dict(sd, strict=False)
    # 仅允许已知 mode buffer 未匹配；任何 trained module 缺失 = 静默丢权重 = BLOCKER
    safe = {"gt_observed_mode", "open_loop_mode"}
    real_missing = [k for k in incompatible.missing_keys if k not in safe]
    assert not real_missing, (
        f"[warm-start BLOCKER] trained keys dropped (GRU etc.): {real_missing}. "
        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")
    print(f"[warm-start] loaded {init_from}")
    print(f"  missing(应仅 mode buffer)={incompatible.missing_keys}")
    print(f"  unexpected(应仅 gt_observed_mode)={incompatible.unexpected_keys}")


if __name__ == "__main__":
    main()
