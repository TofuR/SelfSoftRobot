"""FlowMatchPointCloud — 基于 Flow Matching 的条件点云生成模型。

架构:
  Action Window → MultiScaleEMA → physics_state (条件 c, 迟滞上下文)
                                       |
  Current Action ─→ action_embed ─┐    |
                                   ├→ Action×Z 交互
  Z 坐标 ─→ z_embed ─────────────┘    |
                                       |
                                 FiLMVelocityNet
                                       |
  源噪声 X₀ ~ N(0, σ²I) —[ODE: dX/dt = u_θ(X_t, t | c, action)]→ 预测点云 X̂
                                       |
                                 与 GT 点云对比 (FM loss / CD)

关键设计:
  - EMA 提供时序上下文（迟滞效应建模）
  - Raw action 直接注入 velocity net（避免 EMA 瓶颈丢失精确条件）
  - Action × Z 交互项：基底不动，末端响应最大

训练:
  - 采样 t ~ U(0,1), X₀ ~ N(0,σ²I)
  - OT-sort: 按 z 排序建立单调对应
  - 插值 X_t = (1-t)·X₀ + t·X₁（X₁ 为 GT 点云）
  - 目标速度 u = X₁ - X₀
  - 损失: L_FM + L_CD + L_compact + L_smooth

推理:
  - 从 X₀ ~ N(0,σ²I) 出发
  - 用 Euler/RK4 积分 ODE 得到 X̂
  - 反归一化恢复物理坐标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins import TemporalMixin
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.encoders.fractional_memory import FractionalMemory
from src.fields.velocity_net import FiLMVelocityNet
from src.training.spec import TrainingSpec, PhaseSpec
from src.training.ode_solver import euler_solve, rk4_solve


class FlowMatchPointCloudModel(nn.Module, TemporalMixin):
    """Flow Matching 条件点云生成模型。

    继承 TemporalMixin 获得 encode() 和 compute_smoothness()。
    使用 UnifiedTrainer 的声明式训练（training_spec）。

    Args:
        action_dim: 驱动维度。
        window_size: 时序窗口长度。
        n_scales: EMA 尺度数。
        hidden_dim: 时序编码隐层维度。
        velocity_net_hidden: 速度网络隐层维度。
        velocity_net_layers: 速度网络层数。
        time_embed_dim: 时间嵌入维度。
        sigma: 源噪声标准差。
        ode_steps: 推理时 ODE 积分步数。
        ode_solver: 推理用积分器（"euler" | "rk4"）。
        n_points: 推理时生成的点数。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                "flowmatch",
                dataset_type="pointcloud",
                supervision_mode="pointcloud",
                active_losses=["fm", "cd", "smooth", "compact"],
                forward_attr="forward",
            ),
        ],
    )

    def __init__(
        self,
        action_dim=2,
        window_size=20,
        n_scales=4,
        hidden_dim=128,
        velocity_net_hidden=256,
        velocity_net_layers=6,
        time_embed_dim=64,
        sigma=1.0,
        ode_steps=50,
        ode_solver="euler",
        n_points=1000,
        encoder_type="ema",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.sigma = sigma
        self.ode_steps = ode_steps
        self.ode_solver = ode_solver
        self.n_points = n_points
        self.encoder_type = encoder_type

        # 点云归一化参数（由 set_normalization 设置，推理时反归一化）
        self.register_buffer('pc_center', torch.zeros(1, 1, 3))
        self.register_buffer('pc_scale', torch.ones(1, 1, 3))

        # Action 归一化因子（由 set_normalization 设置，推理时正确归一化输入）
        self.register_buffer('action_norm_factor', torch.tensor(1.0))

        # 时序编码器（TemporalMixin 依赖 self.temporal）
        # 支持选择 EMA 或 FractionalMemory
        EncoderClass = FractionalMemory if encoder_type == "fractional" else MultiScaleEMA
        self.temporal = EncoderClass(
            action_dim=action_dim,
            n_scales=n_scales,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # FiLM 条件速度网络（接收 action 直接注入）
        self.velocity_net = FiLMVelocityNet(
            point_dim=3,
            cond_dim=hidden_dim,
            action_dim=action_dim,
            time_dim=time_embed_dim,
            hidden_dim=velocity_net_hidden,
            n_layers=velocity_net_layers,
        )

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        """设置归一化参数（从数据集获取）。

        Args:
            center: (3,) numpy array 或 tensor，各轴中心。
            scale:  (3,) numpy array 或 tensor，各轴半范围。
            action_norm_factor: float，action 归一化因子（max |all_actions|）。
        """
        import numpy as np
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center).float()
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        self.pc_center = center.view(1, 1, 3)
        self.pc_scale = scale.view(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    @staticmethod
    def _sort_by_z(points):
        """沿 z 轴排序点云，建立 OT 近似对应关系。

        适用于沿主轴延伸的杆状结构：排序后相同 rank 的点
        占据相同的 z-band，产生不交叉的 ODE 轨迹。

        Args:
            points: (B, N, 3) 点云。

        Returns:
            sorted_pts: (B, N, 3) 按 z 排序后的点云。
            z_indices: (B, N) 排序索引。
        """
        z_indices = points[:, :, 2].argsort(dim=1)  # (B, N)
        sorted_pts = torch.gather(
            points, 1,
            z_indices.unsqueeze(-1).expand(-1, -1, 3))
        return sorted_pts, z_indices

    def forward(self, x_t, t, action_window):
        """预测条件速度场。

        Args:
            x_t:  (B, N, 3) 时间 t 处的带噪点云。
            t:    (B, 1) Flow 时间步 ∈ [0, 1]。
            action_window: (B, K, D) 动作序列窗口。

        Returns:
            velocity: (B, N, 3) 预测速度。
        """
        cond = self.encode(action_window)  # (B, hidden_dim) — EMA 时序上下文
        current_action = action_window[:, -1, :]  # (B, action_dim) — 精确当前 action
        return self.velocity_net(x_t, t, cond, action=current_action)

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """计算 Flow Matching 训练损失。

        Args:
            batch: dict，包含:
                "action_window":  (B, K, D) 动作序列
                "gt_pointcloud":  (B, N, 3) GT 点云
            phase_spec: 当前 PhaseSpec。

        Returns:
            dict[str, torch.Tensor]: loss 名到标量的映射。
        """
        # TemporalMixin 处理 "smooth" loss
        losses = super().compute_losses(batch, phase_spec)
        active = set(phase_spec.active_losses)

        device = next(self.parameters()).device
        action_window = batch["action_window"].to(device)
        gt_pc = batch["gt_pointcloud"].to(device)  # (B, N, 3)

        B, N, _ = gt_pc.shape
        cond = self.encode(action_window)  # (B, hidden_dim)
        current_action = action_window[:, -1, :]  # (B, action_dim)

        if "fm" in active:
            t = torch.rand(B, 1, device=device)
            X0 = torch.randn(B, N, 3, device=device) * self.sigma

            # OT-sort: 按 z 排序，建立单调不交叉对应
            X0, _ = self._sort_by_z(X0)
            gt_pc, _ = self._sort_by_z(gt_pc)

            t_expand = t.unsqueeze(-1)
            X_t = (1 - t_expand) * X0 + t_expand * gt_pc
            u_target = gt_pc - X0

            u_pred = self.velocity_net(X_t, t, cond, action=current_action)
            losses["fm"] = F.mse_loss(u_pred, u_target)

            # Endpoint CD 辅助损失
            if "cd" in active:
                from src.losses.pointcloud_losses import chamfer_distance_with_details
                losses["cd"] = chamfer_distance_with_details(
                    X0 + u_pred, gt_pc)["cd"]

            # CD 监控（不计入 total）
            with torch.no_grad():
                from src.losses.pointcloud_losses import chamfer_distance_with_details
                losses["cd_monitor"] = chamfer_distance_with_details(
                    X0 + u_pred, gt_pc)["cd"]

            # 紧密度损失：惩罚同一 z-band 内 x,y 过度扩散（攻击扇形）
            if "compact" in active:
                from src.losses.pointcloud_losses import compactness_loss
                losses["compact"] = compactness_loss(X0 + u_pred, n_z_bins=10)

        return losses

    @torch.no_grad()
    def predict_pointcloud(self, action_window, n_points=None, n_steps=None):
        """推理：通过 ODE 积分生成点云。

        Args:
            action_window: (B, K, D) 动作序列（已归一化或原始值）。
            n_points: 生成的点数（None 使用默认）。
            n_steps: ODE 积分步数（None 使用默认）。

        Returns:
            (B, N, 3) 预测点云（物理坐标）。
        """
        n_points = n_points or self.n_points
        n_steps = n_steps or self.ode_steps

        device = next(self.parameters()).device
        action_window = action_window.to(device)

        # 自动归一化：如果 action_norm_factor > 1，说明需要归一化
        norm = self.action_norm_factor.item()
        if norm > 1.01:
            action_window = action_window / norm

        B = action_window.shape[0]
        cond = self.encode(action_window)
        current_action = action_window[:, -1, :]  # (B, action_dim)

        X0 = torch.randn(B, n_points, 3, device=device) * self.sigma
        X0, _ = self._sort_by_z(X0)

        # ODE 积分：每步都传入 action
        solver = euler_solve if self.ode_solver == "euler" else rk4_solve
        pred_normalized = solver(
            self.velocity_net, X0, cond, n_steps,
            action=current_action)  # 传递 action
        return pred_normalized * self.pc_scale.to(device) + self.pc_center.to(device)
