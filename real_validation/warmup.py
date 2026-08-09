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
                   kind: str = "ramp", seed: int | None = None) -> np.ndarray:
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
    return out


def expand_to_6ch(actions_model, channel_map) -> np.ndarray:
    """(H, action_dim) 模型动作 → (H, 6) applied 命令。

    warmup 下发必须经 channel_map 展开成 6 通道(未映射通道 0),与 executor/
    ActionHistoryBuffer.append_applied6 的输入一致。
    """
    actions_model = np.asarray(actions_model, dtype=np.float64)
    h, a = actions_model.shape
    if len(channel_map) != a:
        raise ValueError("channel_map 长度必须等于动作维度")
    expanded = np.zeros((h, 6), dtype=np.float64)
    for i, ch in enumerate(channel_map):
        expanded[:, ch] = actions_model[:, i]
    return expanded


def apply_scale(actions_model, action_scale_kpa, action_norm_factor) -> np.ndarray:
    """模型单位 [0,1] → kPa(下发前换算;warmup 在 kPa 空间下发)。"""
    scale = np.asarray(action_scale_kpa, dtype=np.float64)
    return np.asarray(actions_model, dtype=np.float64) * float(action_norm_factor) * scale
