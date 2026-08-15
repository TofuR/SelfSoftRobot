"""从现有 transition NPZ 构建可复现的离线 anchor。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.models import Anchor, ModelDescriptor


def anchor_from_npz(path: str | Path, frame_index: int, model: ModelDescriptor,
                    runtime_model, padding: str = "reject") -> Anchor:
    source = Path(path).resolve()
    with np.load(source) as data:
        if "positions" not in data or "actions" not in data:
            raise ValueError("NPZ 必须包含 positions 和 actions")
        positions = np.asarray(data["positions"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
    if positions.ndim != 3 or actions.ndim != 2 or len(positions) != len(actions):
        raise ValueError("期望 positions=(T,3,N)/(T,N,3)，actions=(T,D) 且 T 相同")
    if frame_index < 0 or frame_index >= len(positions):
        raise IndexError(f"frame_index={frame_index} 超出 0..{len(positions) - 1}")
    if actions.shape[1] != model.action_dim:
        raise ValueError(f"NPZ action_dim={actions.shape[1]} 与模型 {model.action_dim} 不同")
    state = positions[frame_index]
    if state.shape == (3, model.n_nodes):
        state = state.T
    elif state.shape != (model.n_nodes, 3):
        raise ValueError(f"NPZ 节点形状 {state.shape} 与模型 N={model.n_nodes} 不同")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("NPZ anchor/history 含 NaN/Inf")

    start = frame_index - model.history_steps + 1
    history = actions[max(0, start):frame_index + 1]
    if start < 0:
        missing = -start
        if padding == "reject":
            raise ValueError(f"当前帧缺少 {missing} 步历史；请选择更晚帧或显式 padding")
        if padding == "zero":
            prefix = np.zeros((missing, model.action_dim), dtype=np.float32)
        elif padding == "repeat_first":
            prefix = np.repeat(actions[:1], missing, axis=0)
        else:
            raise ValueError("padding 只能是 reject/zero/repeat_first")
        history = np.concatenate((prefix, history), axis=0)

    from .anchor_utils import float_rows, model_normalization, normalize_rows
    center, scale = model_normalization(runtime_model)
    normalized = normalize_rows(state, center, scale)
    prev_state = positions[frame_index - 1] if frame_index >= 1 else None
    if prev_state is not None:
        if prev_state.shape == (3, model.n_nodes):
            prev_state = prev_state.T
        prev_state = normalize_rows(prev_state, center, scale)
    return Anchor(
        state=float_rows(normalized),
        action_history=tuple(tuple(float(value) for value in action) for action in history),
        prev_state=(None if prev_state is None else float_rows(prev_state)),
        frame_id="model_normalized", state_space="model_normalized",
        action_units="model_normalized",   # B2:npz actions 已归一到 [0,1],不是 kPa
        source=f"{source}#frame={frame_index}", quality={"kind": "offline_npz", "score": 1.0})
