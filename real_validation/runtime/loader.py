"""仅从 real_validation 目录加载 OpenLoop checkpoint。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import OpenLoopTransitionModel


def _find_config(checkpoint: Path) -> tuple[Path, dict]:
    current = checkpoint.parent
    for _ in range(6):
        candidate = current / "config.json"
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            if not isinstance(value, dict):
                raise ValueError(f"config.json 顶层必须是对象: {candidate}")
            return candidate, value
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"checkpoint 附近找不到 config.json；请把二者一起放入 models/<name>/: {checkpoint}")


def _migrate_gru_keys(state_dict):
    renames = {
        "gru.weight_ih": "gru.weight_ih_l0", "gru.weight_hh": "gru.weight_hh_l0",
        "gru.bias_ih": "gru.bias_ih_l0", "gru.bias_hh": "gru.bias_hh_l0",
    }
    if any(target in state_dict for target in renames.values()):
        return state_dict
    migrated = dict(state_dict)
    for source, target in renames.items():
        if source in migrated:
            migrated[target] = migrated.pop(source)
    return migrated


def load_openloop_model(checkpoint_path: str, device: str = "cpu") -> dict:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config_path, config = _find_config(checkpoint)
    if config.get("model") != "OpenLoopTransitionModel":
        raise ValueError(f"只允许 OpenLoopTransitionModel，当前为 {config.get('model')}")
    encoder = config.get("encoder_type", "fractional")
    if encoder != "fractional":
        raise ValueError(f"PC 精简运行时当前只包含 fractional encoder，当前为 {encoder}")
    required = ("action_dim", "n_nodes", "window_size")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"config.json 缺少部署字段: {missing}")
    state_dict = _migrate_gru_keys(torch.load(
        checkpoint, map_location=device, weights_only=True))
    model = OpenLoopTransitionModel(
        action_dim=int(config["action_dim"]), n_nodes=int(config["n_nodes"]),
        hidden_dim=int(config.get("hidden_dim", 128)),
        window_size=int(config["window_size"]),
        n_orders=int(config.get("n_scales", 4)), z_dim=int(config.get("z_dim", 16)),
    ).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"checkpoint 与本地运行时不兼容；missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}")
    model.eval()
    return {
        "model": model,
        "model_type": "state_transition",
        "model_class": "OpenLoopTransitionModel",
        "action_dim": int(config["action_dim"]),
        "n_nodes": int(config["n_nodes"]),
        "window_size": int(config["window_size"]),
        "norm_factor": float(model.action_norm_factor.item()),
        "config": config,
        "config_path": str(config_path),
    }
