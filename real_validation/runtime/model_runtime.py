"""训练 checkpoint 到工作台运行时的唯一加载入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.io import file_sha256
from ..contracts.models import ModelDescriptor
from . import load_openloop_model


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
    """向上 6 层读取 deploy_manifest.json；缺失或损坏返回 None。"""
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


def _nearby_manifest_path(checkpoint: Path) -> Path | None:
    current = checkpoint.parent
    for _ in range(6):
        candidate = current / "deploy_manifest.json"
        if candidate.is_file():
            return candidate.resolve()
        if current.parent == current:
            break
        current = current.parent
    return None


def certified_k_safe(table: dict[str, int] | None) -> int | None:
    """取认证表中长度容差最严格的一项作为默认规划视野。"""
    entries = []
    for label, value in (table or {}).items():
        try:
            text = str(label).strip()
            for suffix in ("px", "mm"):
                if text.endswith(suffix):
                    text = text[:-len(suffix)]
                    break
            threshold = float(text)
            steps = int(value)
        except (TypeError, ValueError):
            continue
        if threshold > 0 and steps > 0:
            entries.append((threshold, steps))
    return min(entries)[1] if entries else None


class ModelLoadError(RuntimeError):
    """预期内的操作员级加载错误(路径/配置/契约),不应向 UI 抛 traceback。"""


class ModelRuntime:
    """持有模型及其不可变部署元数据；切换 checkpoint 时创建新实例。"""

    def __init__(self, checkpoint: str, data_dir: str | None = None,
                 device: str = "cpu", k_safe: int | None = None):
        checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise ModelLoadError(
                f"checkpoint 不存在:{checkpoint_path}\n"
                f"请从服务器复制 train_log/<tag>/<exp>/phase_*/model/best_model.pt 到 "
                f"{checkpoint_path.parent}/,并同时复制该实验的 config.json 与 "
                f"deploy_manifest.json(见 real_validation/checkpoints/README.md)。")
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
        self.manifest_path = _nearby_manifest_path(checkpoint_path)
        manifest_raw = _nearby_manifest(checkpoint_path)
        manifest = None
        if self.manifest_path is not None:
            if manifest_raw is None:
                raise ModelLoadError("deploy_manifest.json 无法解析为 JSON 对象")
            from ..contracts.deploy_manifest import DeployManifest
            try:
                manifest = DeployManifest.from_dict(manifest_raw)
            except ValueError as error:
                raise ModelLoadError(
                    f"deploy_manifest.json 不满足当前部署合同: {error}") from error
        checkpoint_hash = file_sha256(checkpoint_path)
        if manifest is not None and manifest.checkpoint_sha256 != checkpoint_hash:
            raise ModelLoadError(
                "deploy_manifest.json 的 checkpoint_sha256 与所选模型不一致；"
                "请导出同一次训练试次的部署包")
        effective_k_safe = (int(k_safe) if k_safe is not None else
                            certified_k_safe(
                                (manifest.k_safe_table or manifest.k_safe_table_px)
                                if manifest else None))
        self.manifest = manifest
        self.reference_frame_path = None
        if manifest is not None and manifest.reference_frame:
            reference = Path(manifest.reference_frame)
            if not reference.is_absolute() and self.manifest_path is not None:
                reference = self.manifest_path.parent / reference
            self.reference_frame_path = reference.resolve()
        self.descriptor = ModelDescriptor(
            checkpoint=str(checkpoint_path),
            checkpoint_hash=checkpoint_hash,
            model_type=str(info["model_type"]),
            action_dim=int(info["action_dim"]),
            n_nodes=n_nodes,
            history_steps=history,
            model_class=str(info["model_class"]),
            k_train=int(k_train_value) if k_train_value is not None else None,
            k_safe=effective_k_safe,
            data_dir=str(Path(data_dir).resolve()) if data_dir else None,
            normalization={"action_norm_factor": float(info["norm_factor"])},
            action_scale_kpa=manifest.action_scale_kpa if manifest else None,
            channel_map=manifest.channel_map if manifest else None,
            channel_source6=manifest.channel_source6 if manifest else (),
            channel_equalities=manifest.channel_equalities if manifest else (),
            action_expansion6=manifest.action_expansion6 if manifest else (),
            train_dt_nominal_s=manifest.train_dt_nominal_s if manifest else None,
            train_dt_measured_s=manifest.train_dt_measured_s if manifest else None,
            train_dt_std_s=manifest.train_dt_std_s if manifest else None,
            mask_source=manifest.mask_source if manifest else None,
            mask_source_provenance=manifest.mask_source_provenance if manifest else None,
            segment_params=manifest.segment_params if manifest else None,
            camera_fingerprint=manifest.camera if manifest else None,
            reference_frame_hash=manifest.reference_frame_sha256 if manifest else None,
            k_safe_table_px=manifest.k_safe_table_px if manifest else None,
            k_safe_table=manifest.k_safe_table if manifest else None,
            k_safe_unit=manifest.k_safe_unit if manifest else None,
            planning_displacement_px_p95=(
                manifest.planning_displacement_px_p95 if manifest else None),
            planning_displacement_p95=(
                manifest.planning_displacement_p95 if manifest else None),
            robot_diameter_mm=manifest.robot_diameter_mm if manifest else None,
            robot_diameter_px=manifest.robot_diameter_px if manifest else None,
            mm_per_px=manifest.mm_per_px if manifest else None,
            state_coordinate_frame=(manifest.state_coordinate_frame if manifest
                                    else config.get("state_view", {}).get(
                                        "state_coordinate_frame", "camera_pixel_v1")),
            state_length_unit=(manifest.state_length_unit if manifest
                               else config.get("state_view", {}).get(
                                   "state_length_unit", "px")),
            node_order=(manifest.node_order if manifest else
                        config.get("node_order")),
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
