"""可学习的多尺度指数移动平均 (EMA) 时序编码器。

对动作窗口计算 N 个不同衰减率的 EMA，拼接后经 MLP 映射为物理状态。
不同衰减率的 EMA 通道捕捉不同时间尺度的动力学：
  - 快衰减 (decay≈0.3): 捕捉即时响应
  - 中衰减 (decay≈0.7): 捕捉中短期弹性恢复
  - 慢衰减 (decay≈0.95): 捕捉长期迟滞记忆

EMA 是对输入的线性加权求和，天然满足 Lipschitz 连续性。
"""

import torch
import torch.nn as nn


class MultiScaleEMA(nn.Module):
    """可学习的多尺度指数移动平均时序编码器。

    Args:
        action_dim: 动作维度。
        n_scales: EMA 尺度数（默认 4）。
        window_size: 输入窗口长度。
        hidden_dim: 输出物理状态维度。
    """

    def __init__(self, action_dim, n_scales=4, window_size=20, hidden_dim=128):
        super().__init__()
        self.action_dim = action_dim
        self.n_scales = n_scales
        self.window_size = window_size

        # 可学习的衰减率，初始化为从 0.2 到 0.95 均匀分布
        # 用 sigmoid 保证在 (0, 1) 范围内
        init_decays = torch.linspace(0.2, 0.95, n_scales)
        self.raw_decays = nn.Parameter(torch.logit(init_decays))

        # 速率特征：Δa = a_t - a_{t-1}，显式告知模型变化率
        # 拼接 [ema_1, ema_2, ..., ema_N, current_action, velocity]
        mlp_input_dim = n_scales * action_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def decays(self):
        """返回 (0, 1) 范围内的衰减率。"""
        return torch.sigmoid(self.raw_decays)

    def forward(self, action_window):
        """计算多尺度 EMA 物理状态。

        Args:
            action_window: (Batch, Window, Action_Dim) 动作序列窗口。

        Returns:
            physics_state: (Batch, Hidden_Dim) 物理状态向量。
        """
        B, K, D = action_window.shape

        decays = self.decays  # (N_scales,)

        # 构建衰减权重矩阵: weights[s, k] = decay_s^(K-1-k)
        # k=0 是最早的时间步，k=K-1 是当前时间步
        powers = torch.arange(K, device=action_window.device).float()  # (K,)
        # weights: (N_scales, K)
        weights = decays.unsqueeze(1) ** (K - 1 - powers).unsqueeze(0)
        # 归一化
        weights = weights / weights.sum(dim=1, keepdim=True)

        # 加权求和: (N_scales, K) @ (B, K, D) -> (B, N_scales, D)
        # 用 einsum 更清晰
        ema_features = torch.einsum('sk,bkd->bsd', weights, action_window)
        ema_flat = ema_features.reshape(B, self.n_scales * D)

        # 当前动作 + 速率
        current_action = action_window[:, -1, :]  # (B, D)
        if K >= 2:
            velocity = action_window[:, -1, :] - action_window[:, -2, :]
        else:
            velocity = torch.zeros_like(current_action)

        features = torch.cat([ema_flat, current_action, velocity], dim=-1)
        return self.state_mlp(features)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        """计算相邻两帧物理状态之间的平滑性 loss。

        Args:
            action_windows_t: (B, K, D) 时间步 t 的动作窗口
            action_windows_t1: (B, K, D) 时间步 t+1 的动作窗口

        Returns:
            smooth_loss: 标量
        """
        s_t = self.forward(action_windows_t)
        s_t1 = self.forward(action_windows_t1)
        return torch.mean((s_t1 - s_t) ** 2)
