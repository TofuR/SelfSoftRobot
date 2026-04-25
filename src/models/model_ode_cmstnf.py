"""ODE-CMSTNF — 用 Neural ODE 替代 EMA 的时序编码。

核心优势:
  1. ODE 积分保证物理状态轨迹连续——输入微小变化不会导致输出跳变
  2. 二阶动力学可捕捉阻尼振荡（EMA 是一阶系统，只能指数衰减）
  3. 可嵌入物理先验：阻尼弹簧模型 ds/dt = v, dv/dt = (F - c*v - k*s) / m

与 CMSTNF 的区别仅在时序编码器：ODETemporalEncoder 替代 MultiScaleEMA。
Canonical Field 和两阶段训练流程完全复用。
"""

import torch
import torch.nn as nn
from .layers import PositionalEncoder, MLPDecoder
from .model_cmstnf import CanonicalField


class DampedSpringODE(nn.Module):
    """阻尼弹簧 ODE: dstate/dt = f(state, action)。

    物理先验：软体机器人的弯曲响应近似为阻尼谐振子。
    状态 s ∈ R^hidden 分裂为 [position, velocity]：
      ds_pos/dt = s_vel
      ds_vel/dt = -k * s_pos - c * s_vel + B * action
    其中 k (刚度), c (阻尼), B (力矩阵) 可学习。

    这种结构保证:
      - 轨迹连续（ODE 积分）
      - 可以振荡（二阶系统）
      - 阻尼稳定（不会发散）
    """

    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.half_dim = hidden_dim // 2
        assert hidden_dim % 2 == 0, "hidden_dim must be even for position/velocity split"

        # 可学习的物理参数（log 空间保证正值）
        self.log_stiffness = nn.Parameter(torch.zeros(self.half_dim) - 1.0)  # k, init ~0.37
        self.log_damping = nn.Parameter(torch.zeros(self.half_dim))           # c, init ~1.0
        # 力矩阵：action → velocity 变化率
        self.force_matrix = nn.Linear(action_dim, self.half_dim, bias=True)

        # 残差网络：捕获物理模型无法覆盖的非线性
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 残差初始化为接近零
        with torch.no_grad():
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(self, state, action):
        """ODE 右端：ds/dt = f(s, a)。

        Args:
            state: (B, Hidden)
            action: (B, Action_Dim)
        Returns:
            dsdt: (B, Hidden)
        """
        s_pos = state[:, :self.half_dim]
        s_vel = state[:, self.half_dim:]

        k = torch.exp(self.log_stiffness)  # (Half,)
        c = torch.exp(self.log_damping)     # (Half,)

        # 物理部分：阻尼弹簧
        force = self.force_matrix(action)  # (B, Half)
        ds_pos = s_vel
        ds_vel = -k * s_pos - c * s_vel + force
        ds_physics = torch.cat([ds_pos, ds_vel], dim=-1)

        # 残差部分
        ds_residual = self.residual(torch.cat([state, action], dim=-1))

        return ds_physics + ds_residual


class ODETemporalEncoder(nn.Module):
    """Neural ODE 时序编码器：沿动作序列积分 ds/dt = f(s, a)。

    与 MultiScaleEMA 的关键区别：
      - EMA：对输入做加权平均（线性、一阶、无振荡）
      - ODE：积分动力学方程（可非线性、二阶、可振荡、轨迹连续）
    """

    def __init__(self, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = DampedSpringODE(hidden_dim, action_dim)
        # 初始状态：零位置 + 零速度
        self.initial_state = nn.Parameter(torch.zeros(1, hidden_dim))
        # 可学习的时间步长（动作序列中每步对应的物理时间）
        self.log_dt = nn.Parameter(torch.tensor(0.0))  # dt ≈ 1.0

    def forward(self, action_window):
        """沿动作窗口积分 ODE。

        Args:
            action_window: (B, K, D) 动作序列。

        Returns:
            state: (B, Hidden) 最终物理状态。
        """
        B, K, D = action_window.shape
        state = self.initial_state.expand(B, -1).clone()
        dt = torch.exp(self.log_dt)

        # RK4 积分（比 Euler 更精确）
        for k in range(K):
            a = action_window[:, k, :]
            # k1
            k1 = self.ode_func(state, a)
            # k2
            k2 = self.ode_func(state + 0.5 * dt * k1, a)
            # k3
            k3 = self.ode_func(state + 0.5 * dt * k2, a)
            # k4
            k4 = self.ode_func(state + dt * k3, a)
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        return state

    def compute_smoothness(self, aw_t, aw_t1):
        """时序平滑 loss。"""
        s_t = self.forward(aw_t)
        s_t1 = self.forward(aw_t1)
        return torch.mean((s_t1 - s_t) ** 2)


class ODECMSTNFModel(nn.Module):
    """ODE-CMSTNF：Neural ODE 时序编码 + Canonical/Deformation 架构。

    与 CMSTNFModel 完全相同的空间结构，仅时序编码器从 EMA 换为 ODE。
    """

    def __init__(self, action_dim, window_size=20, hidden_dim=128,
                 d_filter=128, n_freqs=10, deform_n_freqs=6):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.canonical = CanonicalField(d_filter=d_filter, n_freqs=n_freqs)

        # ODE 时序编码器（替代 MultiScaleEMA）
        self.ode_encoder = ODETemporalEncoder(action_dim=action_dim, hidden_dim=hidden_dim)

        # 变形场
        self.deform_encoder = PositionalEncoder(d_input=3, n_freqs=deform_n_freqs, log_space=True)
        deform_enc_dim = 3 * (1 + 2 * deform_n_freqs)
        self.deform_mlp = MLPDecoder(
            input_dim=deform_enc_dim + hidden_dim + action_dim,
            d_filter=d_filter, output_size=3,
        )
        with torch.no_grad():
            self.deform_mlp.net[-1].bias.zero_()

    def _compute_displacement(self, points, action_window):
        """计算变形位移。

        Returns:
            displacement: (B*N_rays, n_samples, 3)
            physics_state: (B, Hidden)
        """
        B, K, D = action_window.shape
        physics_state = self.ode_encoder(action_window)
        current_action = action_window[:, -1, :]

        N_rays = points.shape[0]
        n_samples = points.shape[1]

        pts_exp = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        x_deform = self.deform_encoder(pts_exp).reshape(-1, self.deform_encoder.d_output)

        state_exp = physics_state.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, self.hidden_dim)
        state_flat = state_exp.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, self.hidden_dim)

        action_exp = current_action.unsqueeze(1).expand(-1, N_rays, -1).reshape(B * N_rays, D)
        action_flat = action_exp.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, D)

        latent = torch.cat([x_deform, state_flat, action_flat], dim=-1)
        displacement = self.deform_mlp(latent).reshape(B * N_rays, n_samples, 3)
        return displacement, physics_state

    # ── 接口与 CMSTNFModel 一致 ──

    def forward_canonical(self, points):
        return self.canonical(points)

    def forward(self, points, action_window):
        B = action_window.shape[0]
        N_rays = points.shape[0]
        n_samples = points.shape[1]

        displacement, _ = self._compute_displacement(points, action_window)
        pts_exp = points.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * N_rays, n_samples, 3)
        return self.canonical(pts_exp + displacement)

    def compute_smoothness(self, aw_t, aw_t1):
        return self.ode_encoder.compute_smoothness(aw_t, aw_t1)

    def freeze_canonical(self):
        for p in self.canonical.parameters():
            p.requires_grad = False

    def unfreeze_canonical(self):
        for p in self.canonical.parameters():
            p.requires_grad = True

    def get_ode_params(self):
        """返回 ODE 物理参数（用于分析/可视化）。"""
        return {
            'stiffness': torch.exp(self.ode_encoder.ode_func.log_stiffness).detach().cpu().numpy(),
            'damping': torch.exp(self.ode_encoder.ode_func.log_damping).detach().cpu().numpy(),
            'dt': torch.exp(self.ode_encoder.log_dt).item(),
        }
