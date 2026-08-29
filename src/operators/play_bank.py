"""PlayBank — Prandtl–Ishlinskii play 算子组（率无关 / 摩擦型迟滞）。

谱系: Prandtl 1928 / Ishlinskii 1944（play-stop 算子），Preisach 1935 的
对角化特例。每个算子是一个带死区的摩擦滑块:

    状态:   p_{t} = clamp(p_{t-1}, u_t − r_j, u_t + r_j)
    读出:   q_{t} = u_t − p_{t}            （stop 算子形式，|q| ≤ r_j）

关键设计（对应设计文档 §2.2 与 §3 对 v1 错误的修正）:
  - q 是"电平"而非"增量"——恒定输入下 q 收敛到常数（持续偏移，可表示
    迟滞残余变形），不会线性漂移（v1 的电平-当-增量 bug 在此构造性消除）。
  - 输入 u 用 MonotoneSplineDrive 的输出 e_c（伪应变代理），非原始动作。
  - 阈值 r_j 取固定对数网格（可辨识性协议的半格偏移实验在网格层面做，
    不做可学习 τ/r 的 v1）。
  - 逐通道非负权重 b_{c,j} ≥ 0（softplus 参数化）——PI 密度
    μ_c(r_j) ∝ b_{c,j}，即"迟滞指纹"的定量读出。模式形状 B_j（共享跨
    通道）放在 LocalFrameModeBank，符号自由度由形状吸收。
  - 率无关性: 更新只依赖输入序列的几何路径（clamp 无 Δt），同一路径
    不同速率 → 同一状态。这是 E1 实验判据的构造基础。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlayBank(nn.Module):
    """PI play 算子组: 逐通道、多阈值、率无关。

    Args:
        n_channels: 驱动通道数 C。
        n_operators: 算子个数 J（阈值网格点数）。
        r_range: (r_min, r_max) 死区阈值对数网格范围（驱动 e 的单位）。
        weight_init: softplus 后的初始逐通道权重。
    """

    def __init__(self, n_channels: int, n_operators: int = 8,
                 r_range: tuple = (0.02, 0.5), weight_init: float = 0.05):
        super().__init__()
        assert n_operators >= 1
        assert 0 < r_range[0] < r_range[1]
        self.n_channels = n_channels
        self.n_operators = n_operators
        log_r = torch.linspace(math.log(r_range[0]), math.log(r_range[1]),
                               n_operators)
        self.register_buffer("thresholds", log_r.exp())    # (J,)
        # 非负逐通道权重 b_{c,j}（公理 3），符号自由度交给模式形状
        raw_init = math.log(math.expm1(weight_init))
        self.raw_weights = nn.Parameter(
            torch.full((n_channels, n_operators), raw_init))

    @property
    def weights(self) -> torch.Tensor:
        """(C, J) 非负权重 b_{c,j}，PI 密度 μ_c(r_j) ∝ b_{c,j} / r_j。"""
        return F.softplus(self.raw_weights)

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        """静息初始化: p = 0（零压静止约定，需采集协议配合 E0 预条件）。"""
        return torch.zeros(batch_size, self.n_channels, self.n_operators,
                           device=device)

    def step(self, p_prev: torch.Tensor, u: torch.Tensor):
        """单步更新。

        Args:
            p_prev: (B, C, J) 上一步 play 状态。
            u: (B, C) 当前驱动输入 e_t。
        Returns:
            (p_new, q): p_new (B, C, J) 更新后状态；
            q (B, C, J) = u − p_new，|q| ≤ r_j，恒输入下收敛到常数偏移。
        """
        lo = u.unsqueeze(-1) - self.thresholds     # (B, C, J)
        hi = u.unsqueeze(-1) + self.thresholds
        p_new = torch.clamp(p_prev, lo, hi)
        q = u.unsqueeze(-1) - p_new
        return p_new, q
