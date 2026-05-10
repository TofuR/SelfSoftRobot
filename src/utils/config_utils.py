"""配置合并工具 — JSON 默认值 + CLI 覆盖 → 最终配置。

用法:
    from src.utils.config_utils import resolve_config

    defaults = load_config("training")
    config = resolve_config(defaults, {
        "lr": args.lr,          # None 则用 JSON 默认
        "n_epochs": args.n_epochs,
    })
    # config 是全新 dict，不修改原 JSON 文件
"""

import copy


def resolve_config(defaults, overrides):
    """合并 JSON 默认配置与 CLI 覆盖。

    Args:
        defaults: 从 training.json 加载的默认配置 dict。
        overrides: {key: value} 字典，value 为 None 表示用户没传，用默认值。
                   支持 "section.key" 格式覆盖嵌套字段，例如 "optimization.lr"。

    Returns:
        全新的配置 dict（深拷贝，不修改 defaults）。
    """
    config = copy.deepcopy(defaults)

    for key, val in overrides.items():
        if val is None:
            continue

        if "." in key:
            # 嵌套覆盖: "optimization.lr" → config["optimization"]["lr"]
            parts = key.split(".", 1)
            section, subkey = parts[0], parts[1]
            if section not in config:
                config[section] = {}
            config[section][subkey] = val
        else:
            config[key] = val

    return config
