"""Gamma/Laguerre 时序编码器——带延迟峰值的记忆核。

用 Gamma 分布离散权重替代 EMA 的指数衰减或 FractionalMemory 的幂律衰减。
Gamma 权重的核心特性是"先升后降"的钟罩形曲线：
  - k=1 时退化为指数衰减（与 EMA 等价）
  - k>1 时峰值出现在 t_peak = (k-1)/(-ln(λ))，即响应有延迟
  - 不同 (k, λ) 组合捕获不同时间尺度的延迟响应

物理对应：粘弹性材料对阶跃输入的响应是 S 形的（先慢后快再慢），
Gamma 分布的钟罩形权重天然匹配这种延迟峰值特性，而 EMA/GL 的单调递减
权重假设"当前 action 立即生效"，造成模型超前预测。

接口与 MultiScaleEMA / FractionalMemory 完全一致：
  forward(action_window) -> (B, hidden_dim)
  compute_smoothness(aw_t, aw_t1) -> scalar
  decays 属性（返回 lambdas，兼容 TemporalMixin）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GammaLaguerreMemory(nn.Module):
    """Gamma/Laguerre 多核时序编码器。

    用多组 Gamma 分布权重加权动作历史，替代指数衰减或幂律衰减。
    每组 (k, λ) 定义一个"延迟核"：
      w_t = t^(k-1) * λ^t / Z

    Args:
        action_dim: 动作维度。
        n_scales: Gamma 核数量（类比 EMA 的 n_scales），默认 6。
        window_size: 输入窗口长度。
        hidden_dim: 输出物理状态维度。
        n_orders: 兼容别名（等同 n_scales）。
    """

    def __init__(self, action_dim, n_scales=None, window_size=40, hidden_dim=128,
                 n_orders=None):
        super().__init__()
        # 兼容参数名：模型传 n_orders 也能工作
        if n_scales is None and n_orders is not None:
            n_scales = n_orders
        if n_scales is None:
            n_scales = 6

        self.action_dim = action_dim
        self.n_kernels = n_scales
        self.window_size = window_size

        # ── 可学习参数 ──

        # Gamma 阶次 k：控制峰值延迟位置
        # softplus 保证 k ≥ 1；初始化为 [1, 2, 3, 4, 5, 6] 覆盖不同延迟
        init_ks = torch.linspace(1.0, float(n_scales), n_scales)
        self.k_offsets = nn.Parameter(init_ks)

        # 衰减率 λ：控制记忆长度
        # sigmoid 映射到 (0, 1)；初始化让 λ 从 0.95 到 0.7
        init_lambdas = torch.linspace(0.95, 0.7, n_scales)
        self.logit_lambdas = nn.Parameter(torch.logit(init_lambdas))

        # 每个核的混合权重
        self.kernel_weights = nn.Parameter(torch.ones(n_scales))

        # ── 输出 MLP ──
        # 拼接 [gamma_features, current_action, velocity] → hidden_dim
        mlp_input_dim = n_scales * action_dim + 2 * action_dim
        self.state_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def alphas(self):
        """兼容 TemporalMixin 的 alphas 属性。"""
        return self.lambdas

    @property
    def lambdas(self):
        """返回 (0, 1) 范围内的衰减率。"""
        return torch.sigmoid(self.logit_lambdas)

    @property
    def ks(self):
        """返回 ≥ 1 的 Gamma 阶次。"""
        return F.softplus(self.k_offsets)

    @property
    def decays(self):
        """兼容 TemporalMixin.get_learned_decays()，返回 lambdas。"""
        return self.lambdas

    def _compute_weights(self, k, lam, length):
        """计算单个 Gamma 核的离散权重（log 空间，数值稳定）。

        w_t = t^(k-1) * λ^t / Z

        Args:
            k: Gamma 阶次（标量）
            lam: 衰减率（标量）
            length: 窗口长度

        Returns:
            (length,) 归一化权重张量
        """
        t = torch.arange(length, dtype=torch.float32, device=lam.device)

        # log 空间计算避免溢出
        log_w = (k - 1) * torch.log(t.clamp(min=1e-10)) \
                + t * torch.log(lam) \
                - torch.lgamma(k)

        # k > 1 时 t=0 处权重应为 0（Gamma 分布从 0 升起）
        if k > 1.5:
            log_w = log_w.clone()
            log_w[0] = -100.0

        # 减去最大值做数值稳定的 exp
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)

        # 归一化
        w = w / (w.abs().sum() + 1e-8)
        return w

    def forward(self, action_window):
        """计算多核 Gamma 加权物理状态。

        Args:
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            physics_state: (B, hidden_dim) 物理状态向量。
        """
        B, K, D = action_window.shape
        ks = self.ks
        lambdas = self.lambdas

        kernel_features = []
        for i in range(self.n_kernels):
            w = self._compute_weights(ks[i], lambdas[i], K)
            feat = torch.einsum('k,bkd->bd', w, action_window)
            kernel_features.append(feat * self.kernel_weights[i])

        gamma_flat = torch.cat(kernel_features, dim=-1)  # (B, n_kernels * D)

        # 当前动作 + 速率（与 FractionalMemory / MultiScaleEMA 保持一致）
        current_action = action_window[:, -1, :]
        if K >= 2:
            velocity = action_window[:, -1, :] - action_window[:, -2, :]
        else:
            velocity = torch.zeros_like(current_action)

        features = torch.cat([gamma_flat, current_action, velocity], dim=-1)
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
