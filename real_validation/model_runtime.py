"""训练 checkpoint 到工作台运行时的唯一加载入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import file_sha256
from .models import ModelDescriptor
from .runtime import load_openloop_model


def _nearby_config(checkpoint: Path) -> dict[str, Any]:
    current = checkpoint.parent
    for _ in range(6):
        candidate = current / "config.json"
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}
        if current.parent == current:
            break
        current = current.parent
    return {}


def _nearby_manifest(checkpoint: Path) -> dict[str, Any] | None:
    """向上 6 层找 deploy_manifest.json;缺失/损坏返回 None(字段留 None,由 preflight 阻断)。"""
    current = checkpoint.parent
    for _ in range(6):
        candidate = current / "deploy_manifest.json"
        if candidate.is_file():
            try:
                with candidate.open("r", encoding="utf-8") as stream:
                    value = json.load(stream)
                return value if isinstance(value, dict) else None
            except (OSError, ValueError):
                return None
        if current.parent == current:
            break
        current = current.parent
    return None


class ModelRuntime:
    """持有模型及其不可变部署元数据；切换 checkpoint 时创建新实例。"""

    def __init__(self, checkpoint: str, data_dir: str | None = None,
                 device: str = "cpu", k_safe: int | None = None):
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        info = load_openloop_model(str(checkpoint_path), device=device)
        config = _nearby_config(checkpoint_path)
        model = info["model"]
        n_nodes = int(info["n_nodes"])
        if n_nodes <= 0:
            raise ValueError("无法从 checkpoint/config 推断 n_nodes")
        history = int(info["window_size"])
        k_train_value = config.get(
            "k_train", config.get("rollout_horizon", config.get("episode_len")))
        self.model = model
        self.info = info
        self.device = device
        manifest_raw = _nearby_manifest(checkpoint_path)
        manifest = None
        if manifest_raw:
            from .deploy_manifest import DeployManifest
            try:
                manifest = DeployManifest.from_dict(manifest_raw)
            except ValueError:
                manifest = None   # manifest 残缺 → 字段留 None,由 preflight 阻断规划
        self.manifest = manifest
        self.descriptor = ModelDescriptor(
            checkpoint=str(checkpoint_path),
            checkpoint_hash=file_sha256(checkpoint_path),
            model_type=str(info["model_type"]),
            action_dim=int(info["action_dim"]),
            n_nodes=n_nodes,
            history_steps=history,
            model_class=str(info["model_class"]),
            k_train=int(k_train_value) if k_train_value is not None else None,
            k_safe=int(k_safe) if k_safe is not None else None,
            data_dir=str(Path(data_dir).resolve()) if data_dir else None,
            normalization={"action_norm_factor": float(info["norm_factor"])},
            action_scale_kpa=manifest.action_scale_kpa if manifest else None,
            channel_map=manifest.channel_map if manifest else None,
            train_dt_nominal_s=manifest.train_dt_nominal_s if manifest else None,
            train_dt_measured_s=manifest.train_dt_measured_s if manifest else None,
            train_dt_std_s=manifest.train_dt_std_s if manifest else None,
            mask_source=manifest.mask_source if manifest else None,
            mask_source_provenance=manifest.mask_source_provenance if manifest else None,
            segment_params=manifest.segment_params if manifest else None,
            camera_fingerprint=manifest.camera if manifest else None,
            reference_frame_hash=manifest.reference_frame_sha256 if manifest else None,
            k_safe_table_px=manifest.k_safe_table_px if manifest else None,
            registration_residual_max_px=manifest.registration_residual_max_px
                if manifest else 2.0,
        )

    def eval(self) -> None:
        self.model.eval()

    def clear(self) -> None:
        """显式释放大对象；调用方随后应丢弃本 runtime。"""
        self.model = None
        self.info = {}
        try:
            import torch
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass
