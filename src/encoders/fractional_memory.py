"""分数阶记忆编码器（FractionalMemory）。

用分数阶微积分的 Grünwald-Letnikov 离散化替代指数衰减 EMA，
实现物理上有根据的迟滞建模。

软体材料（硅胶、聚合物）的粘弹性实验表明记忆核是幂律衰减 G(t)∝t^(-α)，
而非指数衰减 G(t)∝e^(-t/τ)。分数阶导数的 GL 离散化天然给出幂律权重序列。

参数：
  α ∈ (0, 1): 分数阶参数
    - α → 0: 无记忆（纯弹性）
    - α → 1: 完全记忆（纯粘性）
    - α ≈ 0.3-0.5: 软体材料的典型范围

与 MultiScaleEMA 的接口完全一致：
  forward(action_window) -> (B, hidden_dim)
  compute_smoothness(aw_t, aw_t1) -> scalar
  decays 属性（返回 alphas，兼容 TemporalMixin）
"""

import torch
import torch.nn as nn


class FractionalMemory(nn.Module):
    """分数阶记忆时序编码器。

    用 Grünwald-Letnikov 权重替代指数衰减权重，
    从材料物理出发建模软体机器人的迟滞行为。

    Args:
        action_dim: 动作维度。
        n_orders: 分数阶个数（类比 EMA 的 n_scales），默认 4。
        window_size: 输入窗口长度。
        hidden_dim: 输出物理状态维度。
    """

    def __init__(self, action_dim, n_orders=4, window_size=20, hidden_dim=128,
                 n_scales=None):
        super().__init__()
        # 兼容 MultiScaleEMA 的 n_scales 参数名
        if n_scales is not None:
            n_orders = n_scales
        self.action_dim = action_dim
        self.n_orders = n_orders
        self.window_size = window_size

        # 可学习的分数阶参数 α ∈ (0, 1)
        # 初始化为 [0.2, 0.4, 0.6, 0.8] 的均匀分布
        init_alphas = torch.linspace(0.2, 0.8, n_orders)
        self.raw_alphas = nn.Parameter(torch.logit(init_alphas))

        # 每个阶次的输出权重（可学习混合系数）
        self.order_weights = nn.Parameter(torch.ones(n_orders))

        # 拼接 [fractional_features, current_action, velocity] → hidden_dim
        mlp_input_dim = n_orders * action_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def alphas(self):
        """返回 (0, 1) 范围内的分数阶参数。"""
        return torch.sigmoid(self.raw_alphas)

    @property
    def decays(self):
        """兼容 TemporalMixin.get_learned_decays()，返回 alphas。"""
        return self.alphas

    def _compute_gl_weights(self, alpha, length):
        """计算单个分数阶的 Grünwald-Letnikov 权重。

        递推公式:
            w_0 = 1
            w_k = w_{k-1} × (k - 1 - α) / k

        Args:
            alpha: 标量，分数阶参数 ∈ (0, 1)
            length: 窗口长度

        Returns:
            (length,) 权重张量
        """
        weights = torch.ones(length, device=alpha.device)
        for k in range(1, length):
            weights[k] = weights[k - 1] * (k - 1 - alpha) / k
        return weights

    def forward(self, action_window):
        """计算分数阶记忆物理状态。

        Args:
            action_window: (B, K, D) 动作序列窗口。
                           B=批次大小, K=窗口长度, D=动作维度

        Returns:
            physics_state: (B, hidden_dim) 物理状态向量。
        """
        B, K, D = action_window.shape
        alphas = self.alphas  # (n_orders,)

        # 计算每个阶次的 GL 权重并加权求和
        frac_features = []
        for i in range(self.n_orders):
            w = self._compute_gl_weights(alphas[i], K)  # (K,)
            # 归一化权重（绝对值归一化，因为有负项）
            w_norm = w / (w.abs().sum() + 1e-8)
            # 加权求和: (K,) @ (B, K, D) → (B, D)
            feat = torch.einsum('k,bkd->bd', w_norm, action_window)
            frac_features.append(feat * self.order_weights[i])

        frac_flat = torch.cat(frac_features, dim=-1)  # (B, n_orders * D)

        # 当前动作 + 速率（与 MultiScaleEMA 保持一致）
        current_action = action_window[:, -1, :]  # (B, D)
        if K >= 2:
            velocity = action_window[:, -1, :] - action_window[:, -2, :]
        else:
            velocity = torch.zeros_like(current_action)

        features = torch.cat([frac_flat, current_action, velocity], dim=-1)
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
