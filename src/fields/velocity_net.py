"""velocity_net.py — FiLM 条件速度网络。

Flow Matching 的核心组件：预测条件速度场 u_theta(X_t, t | c)。

架构:
  SinusoidalPositionEmbedding(t) → t_emb
  Action → action_embed (per-point broadcast)
  Z坐标 → z_embed (per-point)
  action_embed × z_embed → interaction (物理耦合)
  拼接 [cond, t_emb, interaction] → FiLM 参数 (gamma, beta)
  逐点 MLP + FiLM 调制 → 预测速度

关键设计:
  - Action 直接注入 velocity net（不经过 EMA 瓶颈）
  - Action × Z 交互项：显式建模"不同位置对 action 的响应不同"
  - EMA cond 提供时序上下文（迟滞），action_embed 提供精确条件
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
    """FiLM 条件速度网络（带直接 action 注入 + action-z 交互）。

    与纯 FiLM 的区别:
      - 原方案: FiLM(cond, t_emb, z_emb) — cond 被 EMA 瓶颈压缩，不同 action 几乎无区别
      - 新方案: FiLM(cond, t_emb, action_z_interaction) — action 直接参与，与 z 位置耦合

    物理动机:
      软臂弯曲 δ(z) ∝ action × z²，即变形量 = action × 位置响应函数。
      交互项让网络直接看到"在 z 位置，当前 action 的效果"，而非从 EMA 压缩后间接推断。

    Args:
        point_dim: 点坐标维度（默认 3D）。
        cond_dim: EMA 条件向量维度（时序上下文/迟滞）。
        action_dim: 原始 action 维度。
        time_dim: 时间嵌入维度。
        hidden_dim: MLP 隐层维度。
        n_layers: FiLM 调制层数。
    """

    def __init__(self, point_dim=3, cond_dim=128, action_dim=2,
                 time_dim=64, hidden_dim=256, n_layers=6):
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

        # Action 嵌入：直接从原始 action 值生成 per-point 特征
        # 不经过 EMA 瓶颈，保留精确的 action 信息
        action_embed_dim = 32
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.SiLU(),
            nn.Linear(action_embed_dim, action_embed_dim),
        )

        # Z 坐标嵌入：per-point 位置感知
        z_embed_dim = 32
        self.z_embed = nn.Sequential(
            nn.Linear(1, z_embed_dim),
            nn.SiLU(),
            nn.Linear(z_embed_dim, z_embed_dim),
        )

        # Action × Z 交互：建模"位置 × 条件"的物理耦合
        # 软臂: 基底(z小)几乎不动，末端(z大)响应最大
        interaction_dim = 32
        self.interaction = nn.Sequential(
            nn.Linear(action_embed_dim + z_embed_dim, interaction_dim),
            nn.SiLU(),
            nn.Linear(interaction_dim, interaction_dim),
        )

        # 点坐标输入层
        self.point_in = nn.Linear(point_dim, hidden_dim)

        # FiLM 调制层: [cond(EMA) + t_emb + interaction] → (gamma, beta)
        # cond 提供时序上下文，interaction 提供精确 action-z 耦合
        film_input_dim = cond_dim + time_dim + interaction_dim
        self.film_layers = nn.ModuleList([
            nn.Linear(film_input_dim, 2 * hidden_dim)
            for _ in range(n_layers)
        ])

        # MLP 中间层
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

    def forward(self, x_t, t, cond, action=None):
        """预测条件速度场。

        Args:
            x_t:    (B, N, 3) 时间 t 处的带噪点云。
            t:      (B, 1) Flow 时间步 ∈ [0, 1]。
            cond:   (B, cond_dim) EMA 条件向量（时序上下文/迟滞）。
            action: (B, action_dim) 当前帧原始 action 值。None 时退化为纯 EMA 条件。

        Returns:
            velocity: (B, N, 3) 预测速度。
        """
        B, N, _ = x_t.shape

        # 时间嵌入
        t_emb = self.time_embed(t)  # (B, time_dim)

        # Z 坐标嵌入（per-point）
        z_coord = x_t[:, :, 2:3]  # (B, N, 1)
        z_emb = self.z_embed(z_coord)  # (B, N, z_embed_dim)

        # Action 嵌入 + 与 Z 交互
        if action is not None:
            a_emb = self.action_embed(action)  # (B, action_embed_dim)
            a_emb = a_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, action_embed_dim)
        else:
            # 退化模式：无 action 信息时用零填充
            a_emb = torch.zeros(B, N, 32, device=x_t.device)

        # Action × Z 交互: [a_emb, z_emb] → interaction
        interaction_input = torch.cat([a_emb, z_emb], dim=-1)  # (B, N, a_dim + z_dim)
        interaction = self.interaction(interaction_input)  # (B, N, interaction_dim)

        # FiLM 条件: [cond(EMA), t_emb, interaction(action×z)]
        film_input = torch.cat([
            cond.unsqueeze(1).expand(-1, N, -1),   # (B, N, cond_dim) — 时序上下文
            t_emb.unsqueeze(1).expand(-1, N, -1),  # (B, N, time_dim) — 时间
            interaction,                             # (B, N, interaction_dim) — action-z 耦合
        ], dim=-1)

        # 点特征
        h = self.point_in(x_t)  # (B, N, hidden_dim)

        # 逐层 FiLM 调制 + 残差连接
        for i in range(self.n_layers):
            h_residual = h

            gamma_beta = self.film_layers[i](film_input)  # (B, N, 2*hidden_dim)
            gamma = gamma_beta[:, :, :self.hidden_dim]
            beta = gamma_beta[:, :, self.hidden_dim:]

            h = gamma * h + beta  # per-point FiLM 调制

            if i < self.n_layers - 1:
                h = self.activation(h)
                h = self.mlp_layers[i](h)
                h = h + h_residual  # 残差连接

        return self.output_layer(h)  # (B, N, 3)
