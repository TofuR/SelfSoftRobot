"""Multi-Scale Temporal Neural Field (MSTNF).

核心思想：用可学习的多尺度指数移动平均 (EMA) 替代 LSTM 编码动作历史。
EMA 是对输入的线性加权求和，天然满足 Lipschitz 连续性——输入微小变化
不会导致输出大幅跳变，从根本上解决 LSTM 的高频敏感问题。

不同衰减率的 EMA 通道捕捉不同时间尺度的动力学：
  - 快衰减 (decay≈0.3): 捕捉即时响应
  - 中衰减 (decay≈0.7): 捕捉中短期弹性恢复
  - 慢衰减 (decay≈0.95): 捕捉长期迟滞记忆
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder
from src.training.spec import PhaseSpec, TrainingSpec


class MultiScaleEMA(nn.Module):
    """可学习的多尺度指数移动平均时序编码器。

    对动作窗口计算 N 个不同衰减率的 EMA，拼接后经 MLP 映射为物理状态。

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


class MSTNFModel(nn.Module):
    """Multi-Scale Temporal Neural Field 完整模型。

    时序编码: 动作窗口 → MultiScaleEMA → physics_state
    空间查询: (x,y,z) + physics_state → [visibility, density]
    渲染: 体渲染 → 2D 图像

    Args:
        action_dim: 动作维度。
        window_size: 时序窗口长度。
        n_scales: EMA 尺度数。
        hidden_dim: 物理状态/隐层维度。
        d_filter: 空间 MLP 隐层维度。
        n_freqs: 位置编码频率数。
    """

    training_spec = TrainingSpec(
        phases=[PhaseSpec("full", forward_attr="forward", data_mode="sequence",
                          active_losses=["recon", "depth", "smooth"])],
    )

    def __init__(
        self,
        action_dim,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        d_filter=128,
        n_freqs=10,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        # 时序编码器
        self.temporal = MultiScaleEMA(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # 空间编码
        self.pos_encoder = PositionalEncoder(d_input=3, n_freqs=n_freqs, log_space=True)
        pos_enc_dim = 3 * (1 + 2 * n_freqs)

        # 空间解码器: pos_enc + physics_state + current_action → [visibility, density]
        # action skip connection: 即使 EMA 编码尚未成熟，解码器也能直接感知当前动作
        self.decoder = MLPDecoder(
            input_dim=pos_enc_dim + hidden_dim + action_dim,
            d_filter=d_filter,
            output_size=2,
        )

        # density bias: 0.0 中性初始化
        # softplus 在负值区也有非零梯度，不需要负偏置来控制稀疏性
        with torch.no_grad():
            self.decoder.net[-1].bias[1] = 0.0

    def encode_temporal(self, action_window):
        """编码动作窗口为物理状态。

        Args:
            action_window: (Batch, Window, Action_Dim)

        Returns:
            physics_state: (Batch, Hidden_Dim)
        """
        return self.temporal(action_window)

    def decode_spatial(self, points, physics_state, current_action=None):
        """查询空间场。

        Args:
            points: (N, N_samples, 3) 3D 查询点。
            physics_state: (N, Hidden_Dim) 物理状态。
            current_action: (N, Action_Dim) 当前动作 (skip connection)。

        Returns:
            output: (N, N_samples, 2) [visibility, density]
        """
        n_samples = points.shape[1]

        x_pos = self.pos_encoder(points)
        state_expanded = physics_state.unsqueeze(1).expand(-1, n_samples, -1)

        parts = [x_pos, state_expanded]
        if current_action is not None:
            action_expanded = current_action.unsqueeze(1).expand(-1, n_samples, -1)
        else:
            action_expanded = torch.zeros(
                physics_state.shape[0], n_samples, self.action_dim,
                device=physics_state.device, dtype=physics_state.dtype,
            )
        parts.append(action_expanded)
        latent = torch.cat(parts, dim=-1)
        return self.decoder(latent)

    def forward(self, points, action_window):
        """统一前向接口，兼容 MultiViewTrainer 调用约定。

        Args:
            points: (N_rays, n_samples, 3) 3D 查询点。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            output: (B*N_rays, n_samples, 2) [visibility, density]
        """
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        physics_state = self.temporal(action_window)  # (B, Hidden)
        current_action = action_window[:, -1, :]       # (B, D)

        pts_expanded = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        state_expanded = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.hidden_dim)
        action_expanded = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.action_dim)

        return self.decode_spatial(pts_expanded, state_expanded, action_expanded)

    def compute_smoothness(self, action_windows_t, action_windows_t1):
        """时序平滑 loss（委托给 temporal encoder）。"""
        return self.temporal.compute_smoothness(action_windows_t, action_windows_t1)

    def get_learned_decays(self):
        """返回当前学到的衰减率（用于分析/可视化）。"""
        return self.temporal.decays.detach().cpu().numpy()
