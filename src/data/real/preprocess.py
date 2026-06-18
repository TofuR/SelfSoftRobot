"""preprocess.py — 实物数据预处理：NaN 清洗 + 高频气压→相机帧时间对齐。

两件事，都是实物才有的问题（仿真数据天然干净、天然同步）：
  clean_nan_skeleton       三角化对遮挡/分割失败会产生 NaN，训练不处理 → 必须清洗。
  align_actions_to_frames  气压日志通常是高频带时间戳（>> 相机 fps），需重采样到每帧时刻。
"""

import numpy as np


def clean_nan_skeleton(skel):
    """清洗骨架 (N, J, 3) 中的 NaN。

    策略：逐帧逐坐标，沿节点轴线性插值补全 NaN（相邻节点空间接近）；
    整帧全 NaN（该帧三角化彻底失败）→ 置零（调用方可随后丢弃该帧）。

    Args:
        skel: (N, J, 3) 三角化骨架，含 NaN。

    Returns:
        (N, J, 3) float32，无 NaN。
    """
    s = np.array(skel, dtype=np.float32)
    if s.ndim != 3 or s.shape[-1] != 3:
        raise ValueError(f"期望 (N,J,3)，得到 {s.shape}")
    N, J, _ = s.shape
    for n in range(N):
        for c in range(3):
            y = s[n, :, c]
            bad = np.isnan(y)
            if not bad.any():
                continue
            if bad.all():
                s[n, :, c] = 0.0
            else:
                idx = np.arange(J)
                s[n, :, c] = np.interp(idx, idx[~bad], y[~bad])
    return s


def align_actions_to_frames(actions_log, frame_times):
    """把带时间戳的高频气压日志重采样到相机帧时刻。

    气压采样率通常远高于相机 fps（如 1000 Hz vs 30 Hz）。由于软臂机械时间常数
    τ≈0.5–3 s >> 帧间隔（1/fps≈33 ms），臂对帧内的高频气压波动无响应——
    因此按帧时刻插值气压，物理上不丢信息（正确降采样）。

    Args:
        actions_log: (M, 1+A)，第 0 列 = 时间戳（秒，与相机同一时钟），
                     其余列 = 各腔气压。
        frame_times: (N,) 相机帧时刻（秒，同一时钟）。

    Returns:
        (N, A) float32，每帧时刻的插值气压；帧时刻超出日志范围时用端点常数外推。
    """
    log = np.asarray(actions_log, float)
    if log.ndim != 2 or log.shape[1] < 2:
        raise ValueError("actions_log 应为 (M, 1+A)：第 0 列时间戳，其余气压")
    t = log[:, 0]
    P = log[:, 1:]
    ft = np.asarray(frame_times, float).reshape(-1)
    N, A = len(ft), P.shape[1]
    out = np.zeros((N, A), np.float32)
    order = np.argsort(t)                 # 时间戳须单调
    t, P = t[order], P[order]
    for a in range(A):
        out[:, a] = np.interp(ft, t, P[:, a])   # np.interp 自动端点常数外推
    return out
