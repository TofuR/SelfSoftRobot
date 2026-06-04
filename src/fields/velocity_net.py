"""velocity_net.py — FiLM 条件速度网络。

Flow Matching 的核心组件：预测条件速度场 u_theta(X_t, t | c)。

架构:
  SinusoidalPositionEmbedding(t) → t_emb
  拼接 [physics_state, t_emb] → FiLM 参数 (gamma, beta)
  逐点 MLP + FiLM 调制 → 预测速度

FiLM (Feature-wise Linear Modulation):
  h' = gamma * h + beta
  通过仿射变换注入条件信息，比拼接方式参数更少、调制更灵活。
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionEmbedding(nn.Module):
    """标量时间 t 的正弦位置编码（与 Diffusion 模型相同）。

    将标量 t ∈ [0, 1] 映射为 dim 维向量:
      [sin(2π * f_1 * t), cos(2π * f_1 * t), ..., sin(2π * f_n * t), cos(2π * f_n * t)]
    其中 f_i = exp(i * log(10000) / (dim/2))。
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """Args:
            t: (B, 1) 时间标量。

        Returns:
            (B, dim) 时间嵌入向量。
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=t.dtype) * -emb)
        emb = t.squeeze(-1).unsqueeze(-1) * emb  # (B, 1) * (half_dim,) → (B, half_dim)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class FiLMVelocityNet(nn.Module):
    """FiLM 条件速度网络。

    逐点处理输入点云，通过 FiLM 层注入条件信息（动作编码 + 时间），
    预测每个点在 Flow Matching ODE 中的速度。

    Args:
        point_dim: 点坐标维度（默认 3D）。
        cond_dim: 条件向量维度（与 MultiScaleEMA hidden_dim 对齐）。
        time_dim: 时间嵌入维度。
        hidden_dim: MLP 隐层维度。
        n_layers: FiLM 调制层数。
    """

    def __init__(self, point_dim=3, cond_dim=128, time_dim=64,
                 hidden_dim=256, n_layers=6):
        super().__init__()
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # 时间编码
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Z 坐标嵌入：让每个点根据自身 z 位置获得不同的 FiLM 参数
        self.z_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        z_embed_dim = 32

        # 点坐标输入层
        self.point_in = nn.Linear(point_dim, hidden_dim)

        # FiLM 调制层: [cond + time_emb + z_emb] → (gamma, beta)
        # z_emb 是 per-point 的 → FiLM 输出也是 per-point
        film_input_dim = cond_dim + time_dim + z_embed_dim
        self.film_layers = nn.ModuleList([
            nn.Linear(film_input_dim, 2 * hidden_dim)
            for _ in range(n_layers)
        ])

        # MLP 中间层（FiLM 调制后接的线性层）
        self.mlp_layers = nn.ModuleList()
        for i in range(n_layers - 1):
            self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, point_dim)
        self.activation = nn.SiLU()

        # 初始化 FiLM 层: gamma≈1, beta≈0（恒等调制）
        for layer in self.film_layers:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            layer.bias.data[:self.hidden_dim] = 1.0

    def forward(self, x_t, t, cond):
        """预测条件速度场。

        Args:
            x_t:  (B, N, 3) 时间 t 处的带噪点云。
            t:    (B, 1) Flow 时间步 ∈ [0, 1]。
            cond: (B, cond_dim) 条件向量（MultiScaleEMA 输出）。

        Returns:
            velocity: (B, N, 3) 预测速度。
        """
        B, N, _ = x_t.shape

        # 时间嵌入
        t_emb = self.time_embed(t)  # (B, time_dim)

        # Z 坐标嵌入（per-point）
        z_coord = x_t[:, :, 2:3]  # (B, N, 1)
        z_emb = self.z_embed(z_coord)  # (B, N, 32)

        # FiLM 条件: [cond, t_emb, z_emb] → per-point (gamma, beta)
        film_input = torch.cat([
            cond.unsqueeze(1).expand(-1, N, -1),   # (B, N, cond_dim)
            t_emb.unsqueeze(1).expand(-1, N, -1),  # (B, N, time_dim)
            z_emb,                                   # (B, N, 32)
        ], dim=-1)  # (B, N, cond_dim + time_dim + 32)

        # 点特征
        h = self.point_in(x_t)  # (B, N, hidden_dim)

        # 逐层 FiLM 调制 + 残差连接（现在 per-point 调制）
        for i in range(self.n_layers):
            h_residual = h  # 保存用于残差连接

            gamma_beta = self.film_layers[i](film_input)  # (B, N, 2*hidden_dim)
            gamma = gamma_beta[:, :, :self.hidden_dim]    # (B, N, hidden_dim)
            beta = gamma_beta[:, :, self.hidden_dim:]      # (B, N, hidden_dim)

            h = gamma * h + beta  # per-point FiLM 调制

            if i < self.n_layers - 1:
                h = self.activation(h)
                h = self.mlp_layers[i](h)
                h = h + h_residual  # 残差连接

        return self.output_layer(h)  # (B, N, 3)
