"""公共训练参数解析与配置合并工具。

将 argparse CLI 参数与 training.json 默认配置统一合并，避免每个脚本重复定义参数。
CLI 参数使用 None 作为默认值，resolve_training_config 会跳过 None 只覆盖用户显式指定的值。

提供三组工具:
  1. add_common_args(parser)     — 所有训练脚本共享的基础参数
  2. add_two_phase_args(parser)  — 两阶段训练脚本的额外参数
  3. resolve_training_config(overrides) — 将 CLI 覆盖合并到 training.json 默认配置

用法:
    from src.config.args import add_common_args, add_two_phase_args, resolve_training_config

    parser = argparse.ArgumentParser()
    add_common_args(parser, data_dir_default="data/sequence_data")
    add_two_phase_args(parser)
    # ... 添加脚本特有参数 ...
    args = parser.parse_args()

    config = resolve_training_config({
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
        "canonical.phase1_epochs": args.phase1_epochs,
        "canonical.phase2_epochs": args.phase2_epochs,
    })

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
        --data_dir   (str)  : 训练数据目录路径
        --lr         (float): 学习率，覆盖 training.json 中 optimization.lr
        --n_epochs   (int)  : 训练轮数，覆盖 training.json 中 optimization.n_epochs
    """
    parser.add_argument("--data_dir", type=str, default=data_dir_default)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)


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
        dict: 合并后的完整训练配置，结构如:
            {
                "model": {...},
                "optimization": {"lr": ..., "batch_size": ..., ...},
                "temporal": {...},
                "canonical": {"phase1_epochs": ..., "phase2_epochs": ..., ...},
                "loss_weights": {...},
                "logging": {...},
            }

    示例:
        # 只覆盖 lr，其余用 training.json 默认
        config = resolve_training_config({"optimization.lr": 1e-3})

        # 多项覆盖
        config = resolve_training_config({
            "optimization.lr": 1e-3,
            "optimization.n_epochs": 100,
            "canonical.phase1_epochs": 30,
        })
    """
    defaults = load_config("training")
    return resolve_config(defaults, overrides)
