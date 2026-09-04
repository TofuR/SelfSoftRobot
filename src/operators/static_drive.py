"""MonotoneSplineDrive — 逐通道单调样条驱动（伪应变代理）。

作用: 把归一化动作 a_c ∈ [0,1] 映射为单调递增标量驱动 e_c(a_c)，
作为 play / Maxwell 算子的共同输入（"伪应变"代理——真实伪应变需通道
配对抗压辨识，v1 用记忆无关单调变换替代，公理安全）。

参数化: 非负铰链（ReLU hinge）基函数线性组合

    e_c(a) = Σ_k w_{c,k} · relu(a − knot_k),   w_{c,k} ≥ 0 (softplus 参数化)

性质（构造保证，非软约束）:
  - e(action_min) = 0（静息 = 零驱动）
  - 严格单调不减（w ≥ 0）
  - 记忆无关（纯函数，无状态）——公理 1 的通道级构件
  - 容量受限: n_knots 个铰链，无跨通道耦合
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneSplineDrive(nn.Module):
    """逐通道单调铰链样条驱动 e_c(a_c)，动作域 [action_min, action_max]。

    Args:
        n_channels: 独立驱动通道数 C。
        n_knots: 铰链节点数（容量上限，默认 5 —— 设计文档 §2.2 的 ≤5 节点约束）。
        action_min / action_max: 归一化动作域（实物数据合同为 [0, 1]）。
        weight_init: softplus 后的初始铰链权重（≈初始斜率量级）。
        output_normalization: ``free`` 保留历史可学幅值；``unit_range``
            保证 e(action_max)=1，使后续迟滞阈值具有固定尺度。
    """

    def __init__(self, n_channels: int, n_knots: int = 5,
                 action_min: float = 0.0, action_max: float = 1.0,
                 weight_init: float = 0.4,
                 output_normalization: str = "free"):
        super().__init__()
        assert n_knots >= 1, "至少一个铰链节点"
        assert action_max > action_min, "动作域必须非空"
        if output_normalization not in {"free", "unit_range"}:
            raise ValueError(
                "output_normalization 必须为 free 或 unit_range")
        self.n_channels = n_channels
        self.n_knots = n_knots
        self.action_max = float(action_max)
        self.output_normalization = output_normalization
        knots = torch.linspace(action_min, action_max, n_knots)
        self.register_buffer("knots", knots)
        # softplus 参数化保证 w ≥ 0（公理 3: 谱非负在通道级的落实）
        raw_init = math.log(math.expm1(weight_init))
        self.raw_weights = nn.Parameter(
            torch.full((n_channels, n_knots), raw_init))

    @property
    def weights(self) -> torch.Tensor:
        """(C, K) 非负铰链权重（谱读出用）。"""
        weights = F.softplus(self.raw_weights)
        if self.output_normalization == "unit_range":
            # Remove the otherwise free drive-amplitude gauge.  On the
            # declared action domain this makes e(action_max)=1 per channel,
            # so play thresholds and Maxwell deficits share a reproducible
            # dimensionless scale across seeds/checkpoints.
            spans = torch.relu(self.action_max - self.knots).to(weights)
            endpoint = (weights * spans).sum(dim=-1, keepdim=True)
            weights = weights / endpoint.clamp_min(1e-12)
        return weights

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        """计算逐通道驱动。

        Args:
            action: (B, C) 归一化动作。
        Returns:
            (B, C) 驱动 e，e(action_min)=0，单调不减。
        """
        hinges = torch.relu(
            action.unsqueeze(-1) - self.knots)          # (B, C, K)
        return (self.weights * hinges).sum(dim=-1)       # (B, C)
