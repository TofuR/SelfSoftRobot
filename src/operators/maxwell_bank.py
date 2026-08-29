"""MaxwellBank — 广义 Maxwell 元件组（率相关 / 粘弹型迟滞）。

谱系: Maxwell 1867 元件 → 广义 Maxwell 模型（标准线性固体推广）。
每个元件是一阶滞后追踪弹性目标:

    h_{k,t} = d_k · h_{k,t−1} + (1 − d_k) · e_t,    d_k = exp(−Δt/τ_k)

关键设计（对应设计文档 §2.2 与 §3 的显式 Euler 教训）:
  - 精确 ZOH 离散（d = exp(−Δt/τ)），非显式 Euler——后者在 Δt/τ ≥ 2
    时振荡发散。exp 参数化保证 0 < d < 1 恒成立，任意 Δt 无条件稳定。
  - τ_k 取固定对数网格，下界 τ_min = 3Δt（可辨识窗口下界；更快的模态
    与采样率不可分，且会被静态项吸收——可辨识性审查 C1 的约束）。
  - h 追踪的是"目标电平"e（单位增益），幅度与符号自由度全部放在读出
    权重 v_{c,k} 与模式形状 C_k 上，消除 G·C 冗余参数化。
  - 读出用亏量形式 (h − e)（模型层完成）: 平衡态 h = e 时贡献为零，
    构造性避免与静态项双重计数。
"""

import math

import torch
import torch.nn as nn


class MaxwellBank(nn.Module):
    """广义 Maxwell 元件组: 逐通道、多时间常数、精确 ZOH。

    Args:
        n_channels: 驱动通道数 C。
        n_elements: 元件个数 M（τ 网格点数）。
        dt: 采样间隔（秒）。必须与数据合同一致（实物 10 Hz → 0.1）。
        tau_range: (τ_min, τ_max) 对数网格范围。τ_min 自动夹到 ≥ 3·dt。
        weight_init: 逐通道读出权重 v_{c,k} 的初始化标准差（带符号）。
    """

    def __init__(self, n_channels: int, n_elements: int = 6, dt: float = 0.1,
                 tau_range: tuple = None, weight_init: float = 0.02):
        super().__init__()
        assert n_elements >= 1
        assert dt > 0, "dt 必须进合同（秒）"
        self.n_channels = n_channels
        self.n_elements = n_elements
        self.dt = float(dt)
        tau_min = 3.0 * dt
        if tau_range is None:
            tau_range = (tau_min, 10.0)
        tau_min = max(tau_range[0], 3.0 * dt)
        tau_max = max(tau_range[1], 2.0 * tau_min)
        self.tau_range = (tau_min, tau_max)
        log_t = torch.linspace(math.log(tau_min), math.log(tau_max),
                               n_elements)
        self.register_buffer("taus", log_t.exp())                 # (M,)
        self.register_buffer("decays", torch.exp(-dt / log_t.exp()))
        # 带符号逐通道读出权重 v_{c,k}（粘弹谱幅度；符号与模式形状 C_k
        # 存在 (v, C) ↔ (−v, −C) 冗余，无害，谱读出取 |v|）
        self.weights = nn.Parameter(
            weight_init * torch.randn(n_channels, n_elements))

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        """静息初始化: h = 0（零压静止约定）。"""
        return torch.zeros(batch_size, self.n_channels, self.n_elements,
                           device=device)

    def step(self, h_prev: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        """单步精确 ZOH 更新。

        Args:
            h_prev: (B, C, M) 上一步元件状态。
            drive: (B, C) 当前驱动目标 e_t。
        Returns:
            (B, C, M) 更新后状态 h_t。
        """
        return self.decays * h_prev + \
            (1.0 - self.decays) * drive.unsqueeze(-1)
