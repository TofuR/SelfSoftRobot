"""公共训练参数解析与配置合并工具。

将 argparse CLI 参数与 training.json 默认配置统一合并，避免每个脚本重复定义参数。
CLI 参数使用 None 作为默认值，resolve_training_config 会跳过 None 只覆盖用户显式指定的值。

提供四组工具:
  1. add_common_args(parser)     — 所有训练脚本共享的基础参数
  2. add_two_phase_args(parser)  — 两阶段训练脚本的额外参数
  3. resolve_training_config(overrides) — 将 CLI 覆盖合并到 training.json 默认配置
  4. build_common_overrides(args) — 从 args 提取通用覆盖项
  5. resolve_phase_epochs(spec, config, ...) — 两阶段 epoch 分配

用法:
    from src.config.args import (add_common_args, add_two_phase_args,
                                  resolve_training_config, build_common_overrides,
                                  resolve_phase_epochs)

    parser = argparse.ArgumentParser()
    add_common_args(parser, data_dir_default="data/sequence_data")
    add_two_phase_args(parser)
    # ... 添加脚本特有参数 ...
    args = parser.parse_args()

    config = resolve_training_config(build_common_overrides(args))

依赖:
  config.params.load_config   — 加载 YAML/JSON 配置文件
  src.utils.config_utils.resolve_config — 深拷贝 + 覆盖合并
"""

from config.params import load_config
from src.utils.config_utils import resolve_config


def add_common_args(parser, data_dir_default="data/sequence_data"):
    """添加所有训练脚本共享的 CLI 参数。

    参数默认值为 None（而非具体值），这样 resolve_training_config 会使用
    training.json 中的默认值，只有用户显式传参时才会覆盖。

    Args:
        parser: argparse.ArgumentParser 实例。
        data_dir_default (str): --data_dir 的默认值，各脚本可能不同
                                （如 train_sdf.py 用 "data/sequence_data_1d"）。

    添加的参数:
        --data_dir      (str)  : 训练数据目录路径
        --lr            (float): 学习率，覆盖 training.json 中 optimization.lr
        --n_epochs      (int)  : 训练轮数，覆盖 training.json 中 optimization.n_epochs
        --batch_size    (int)  : 批大小，覆盖 training.json 中 optimization.batch_size
        --num_workers   (int)  : DataLoader 进程数
        --window_size   (int)  : 时序窗口大小，覆盖 temporal.window_size
        --n_scales      (int)  : 时序编码器尺度数，覆盖 temporal.n_scales
        --hidden_dim    (int)  : 时序编码器隐层维度，覆盖 temporal.hidden_dim
        --eval_interval (int)  : 训练中评估间隔（epoch 数，0=关闭）
        --seed          (int)  : 随机种子（可复现性）
    """
    parser.add_argument("--data_dir", type=str, default=data_dir_default)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--n_scales", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None,
                        help="Evaluate every N epochs during training (0=off)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")


def add_two_phase_args(parser):
    """添加两阶段训练脚本（如 CMSTNF、MS-SCNF、SkeletonSDF）的 CLI 参数。

    用于支持单独运行 Phase 1 或 Phase 2（从已有 Phase 1 权重继续训练）。

    Args:
        parser: argparse.ArgumentParser 实例。

    添加的参数:
        --phase          (int)  : 指定运行阶段（1 或 2），不传则运行完整两阶段
        --exp_dir        (str)  : 已有实验目录（用于 Phase 2 续训）
        --phase1_epochs  (int)  : Phase 1 训练轮数，覆盖 training.json 中 canonical.phase1_epochs
        --phase2_epochs  (int)  : Phase 2 训练轮数，覆盖 training.json 中 canonical.phase2_epochs

    典型用法:
        # 完整两阶段训练
        python train_cmstnf.py

        # 只跑 Phase 2（从已有 Phase 1 权重）
        python train_cmstnf.py --phase 2 --exp_dir train_log/train_cmstnf/exp_20260519_0
    """
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)


def resolve_training_config(overrides):
    """加载 training.json 默认配置，合并 CLI 覆盖，返回最终配置。

    工作流程:
      1. 从 config/training.json 加载默认配置（深拷贝，不修改原文件）
      2. 遍历 overrides，跳过值为 None 的条目（用户未显式传参）
      3. 支持 "section.key" 点号格式覆盖嵌套字段
      4. 返回合并后的完整配置 dict

    Args:
        overrides (dict): CLI 参数组成的覆盖字典。
            key 格式为 "section.subkey"（如 "optimization.lr"），
            value 为 None 表示用户没传，使用 JSON 默认值。

    Returns:
        dict: 合并后的完整训练配置

    示例:
        # 只覆盖 lr，其余用 training.json 默认
        config = resolve_training_config({"optimization.lr": 1e-3})

        # 使用 build_common_overrides 自动提取通用参数
        config = resolve_training_config(build_common_overrides(args))
    """
    defaults = load_config("training")
    return resolve_config(defaults, overrides)


def build_common_overrides(args):
    """从 add_common_args 解析的 args 中提取所有通用覆盖项。

    避免每个脚本重复写 {"optimization.lr": args.lr, ...}。

    Args:
        args: argparse.Namespace，包含 add_common_args 定义的参数。

    Returns:
        dict: 可直接传给 resolve_training_config 的覆盖字典。
    """
    return {
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "optimization.batch_size": getattr(args, "batch_size", None),
        "optimization.num_workers": getattr(args, "num_workers", None),
        "temporal.window_size": getattr(args, "window_size", None),
        "temporal.n_scales": getattr(args, "n_scales", None),
        "temporal.hidden_dim": getattr(args, "hidden_dim", None),
        "evaluation.eval_interval": getattr(args, "eval_interval", None),
    }


def resolve_phase_epochs(spec, config, phase=None, n_epochs_override=None):
    """计算两阶段模型的 n_epochs_per_phase。

    统一处理两阶段 epoch 分配逻辑，消除各训练脚本中的重复代码。

    Args:
        spec: TrainingSpec 实例（model.training_spec）
        config: 训练配置 dict
        phase: int 或 None，CLI --phase 参数（1 或 2，表示只跑某阶段）
        n_epochs_override: int 或 None，CLI --n_epochs 参数

    Returns:
        dict: {phase_name: n_epochs} 或 None（单阶段模型且 phase=None）

    逻辑:
        - phase=None（完整训练）: Phase 1 用 canonical.phase1_epochs，Phase 2 用 n_epochs
        - phase=1 或 2: 只运行指定阶段，其他阶段设为 0
    """
    if not spec.is_two_phase and phase is None:
        return None

    can_cfg = config.get("canonical", {})
    total_epochs = n_epochs_override or config["optimization"]["n_epochs"]
    n_epochs_per_phase = {}

    if phase is not None:
        for i, p in enumerate(spec.phases):
            if i + 1 == phase:
                n_epochs_per_phase[p.name] = total_epochs
            else:
                n_epochs_per_phase[p.name] = 0
    else:
        for p in spec.phases:
            if p.name in ("canonical", "skeleton"):
                n_epochs_per_phase[p.name] = can_cfg.get("phase1_epochs", 50)
            else:
                n_epochs_per_phase[p.name] = total_epochs

    return n_epochs_per_phase
