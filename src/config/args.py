"""公共训练参数解析与配置合并工具。

用法:
    from src.config.args import add_common_args, add_two_phase_args, resolve_training_config

    parser = argparse.ArgumentParser()
    add_common_args(parser, data_dir_default="data/sequence_data")
    # ... 添加脚本特有参数 ...
    args = parser.parse_args()

    config = resolve_training_config({
        "optimization.lr": args.lr,
        "optimization.n_epochs": args.n_epochs,
    })
"""

from config.params import load_config
from src.utils.config_utils import resolve_config


def add_common_args(parser, data_dir_default="data/sequence_data"):
    """添加所有训练脚本共享的参数：--data_dir, --lr, --n_epochs。"""
    parser.add_argument("--data_dir", type=str, default=data_dir_default)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)


def add_two_phase_args(parser):
    """添加两阶段训练脚本的参数：--phase, --exp_dir, --phase1/2_epochs。"""
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2])
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--phase1_epochs", type=int, default=None)
    parser.add_argument("--phase2_epochs", type=int, default=None)


def resolve_training_config(overrides):
    """加载 training.json 默认配置并合并 CLI 覆盖。

    Args:
        overrides: {"section.key": value} 字典，value 为 None 表示用默认值。

    Returns:
        合并后的完整配置 dict。
    """
    defaults = load_config("training")
    return resolve_config(defaults, overrides)
