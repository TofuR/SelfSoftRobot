"""warmup 冷启动:训练分布内动作序列填满 ActionHistoryBuffer。

模型需要 H = history_steps 步真实动作历史才能建 anchor(模型从没见过零填充窗口,
且分数阶 GL 核把最大权重压在窗口最旧格)。warmup = 按 train_dt 下发一段训练分布内
动作(默认 0→0.8×scale 慢速 ramp),每步经 transport.send 下发,applied6 进 buffer。

本模块只做**纯后端**:动作序列生成(与传输/GUI 解耦)。传输用 CommandTransport
(MockCommandTransport 或 QtValveTransport,executor.py / hardware_session.py)。
"""

from __future__ import annotations

import math

import numpy as np


def warmup_actions(action_dim: int, history_steps: int, *, lo=0.0, hi=1.0,
                   kind: str = "ramp", seed: int | None = None,
                   channel_equalities=()) -> np.ndarray:
    """训练分布内 warmup 动作序列,形状 (history_steps, action_dim),值域 [lo, hi]。

    kind:
      - "ramp": 0 → 0.8·hi 慢速线性(单调,分布内,最稳)
      - "triangle": 0→0.8→0 往返(覆盖加载+卸载)
      - "hold": 恒定 0.4·hi(静态)
    """
    if history_steps <= 0 or action_dim <= 0:
        raise ValueError("history_steps 与 action_dim 必须为正")
    out = np.zeros((history_steps, action_dim), dtype=np.float32)
    if kind == "ramp":
        peak = lo + (hi - lo) * 0.8
        for t in range(history_steps):
            v = lo + (peak - lo) * (t + 1) / max(1, history_steps)
            out[t, :] = v
    elif kind == "triangle":
        peak = lo + (hi - lo) * 0.8
        half = max(1, history_steps // 2)
        for t in range(history_steps):
            phase = t % (2 * half)
            frac = (phase + 1) / max(1, half)
            v = lo + (peak - lo) * min(frac, 2.0 - frac)
            out[t, :] = v
    elif kind == "hold":
        out[:, :] = lo + (hi - lo) * 0.4
    else:
        raise ValueError(f"未知 warmup kind: {kind}")
    if seed is not None:   # 可选:在 ramp 上加小抖动,仍分布内
        rng = np.random.default_rng(seed)
        out += rng.uniform(-0.02, 0.02, out.shape).astype(np.float32)
        out = np.clip(out, lo, hi)
    if channel_equalities:
        from ..contracts.models import apply_channel_equalities
        if action_dim != 6:
            raise ValueError("channel_equalities warmup 当前要求 action_dim=6")
        out = np.asarray(
            [apply_channel_equalities(row, channel_equalities) for row in out],
            dtype=np.float32)
    return out


def expand_to_6ch(actions_model, channel_map, channel_equalities=()) -> np.ndarray:
    """(H, action_dim) 模型动作 → (H, 6) applied 命令。

    委托 planner_service.expand_model_actions(共享 6 通道展开逻辑,含 channel_map
    唯一性/范围校验)。
    """
    from ..planning.planner_service import expand_model_actions
    return np.asarray(expand_model_actions(
        actions_model, channel_map, channel_equalities), dtype=np.float64)
