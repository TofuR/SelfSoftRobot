"""StateTransitionSpatialModel — 闭环状态转移中心线预测模型。

架构（基于 SpatialSequenceModel 升级，从"前馈稳态推断"改为"一步状态转移"）:

  前一步状态 + 当前动作                          ── TemporalEncoder ──→ cond (B, hidden)
    │                                                                          │
  s_{t-1} (B,N,3) ──┐    z_{t-1} (B, z_dim) ──→ z_cell(GRUCell) ──→ z_t ──┐    │
  v=s_{t-1}-s_{t-2} ┤                                                       │    │
                    └→ StateEncoder ──→ state_seed (warm start 的 GRU 种子)  │    │
                                                                              ↓    ↓
            cond + z_pos_embed + z_proj 注入 → GRU(z₀→z_K) → 每节点 Δ_raw
                                                                              │
            s_t = s_{t-1} + delta_scale · tanh(Δ_raw)    （预测增量而非绝对坐标）

关键设计:
  - 闭环：模型学习状态转移函数 s_t = F(s_{t-1}, a_t, z_{t-1})，而非 action→state 的前馈映射。
    解决稳态假设在迟滞下的失效（同一 action，不同历史 → 不同状态）。
  - Δ 预测：输出增量 s_{t-1}+Δ，输出范围小、天然连续，且 delta_scale·tanh 保证收缩，
    控制 rollout 时的误差累积。
  - 可学习迟滞潜变量 z（方案 A）：z 无物理真值（实物上无 z 传感器），由模型自演化
    z_t = Φ_z(z_{t-1}, a_t, s_{t-1})，端到端从 skeleton loss 学。z 用 GRUCell 实现
    （门控提供选择性记忆，利于迟滞建模，且有界激活利于收缩）。
  - z 与 TemporalEncoder 的区别：cond 是"当前动作编码"（每步重算、无记忆）；
    z 是"演化中的迟滞潜状态"（带历史记忆）。二者职责分离，z 通过 z_proj 加性注入 GRU。

向后兼容:
  - forward 的 prev_skeleton/prev_z 默认 None → 冷启动回退到 init_hidden(cond) + z_init(cond)，
    即退化为带 z 初始化的前馈预测，与旧调用方式（只传 action_window）兼容。
  - 不修改 SpatialSequenceModel / PCSpatialSequenceModel，互不影响。

Stage 0 限制（重要）:
  UnifiedTrainer 是逐帧独立训练（per-frame, shuffle）。Stage 0 下 z 每步从 z_init(cond)
  重新初始化，无跨帧记忆 → z 退化为 cond 的函数，不是真正的迟滞潜变量。这是 per-frame
  训练的固有限制。Stage 0 只验证架构正确 + 管线跑通 + s 单步转移达标；z 成长为真正
  迟滞潜变量需要 Stage 1 的序列级训练（episode 内逐步 rollout，z 跨帧演化）。

训练:
  - L_skeleton:      预测中心线与 GT 的 MSE
  - L_spatial_smooth: 相邻节点位移连续性约束
  - L_smooth:        时序平滑（TemporalMixin 提供，依赖 action_window_next）
  - z 无 GT，不加 loss。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixins import TemporalMixin
from src.encoders.fractional_memory import FractionalMemory
from src.encoders.multi_scale_ema import MultiScaleEMA
from src.encoders.gamma_laguerre import GammaLaguerreMemory
from src.encoders.temporal_gru import TemporalGRU
from src.encoders.temporal_transformer import TemporalTransformer
from src.encoders.temporal_tcn import TemporalTCN
from src.training.spec import TrainingSpec, PhaseSpec
from src.data.dataset_pointcloud import _sample_surface


class StateTransitionSpatialModel(nn.Module, TemporalMixin):
    """闭环状态转移模型：前一步骨架 + 当前动作 → 当前骨架（预测增量）。

    继承 TemporalMixin 获得 encode() / compute_smoothness() / 默认 compute_losses()。

    Args:
        action_dim: 驱动维度。
        n_nodes: 中心线节点数（与 GT positions 的 N 一致）。
        hidden_dim: 隐层维度。
        window_size: 时序窗口长度。
        n_orders: FractionalMemory 的分数阶个数。
        encoder_type: 时序编码器类型（同 SpatialSequenceModel）。
        z_dim: 可学习迟滞潜变量 z 的维度（推荐 16–32）。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                name="state_transition",
                dataset_type="state_transition",
                supervision_mode="spatial_sequence",
                active_losses=["skeleton", "spatial_smooth", "smooth"],
                forward_attr="forward",
            ),
        ],
    )

    def __init__(
        self,
        action_dim=2,
        n_nodes=31,
        hidden_dim=128,
        window_size=20,
        n_orders=4,
        encoder_type="fractional",
        z_dim=16,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim
        self.encoder_type = encoder_type
        self.z_dim = z_dim

        # 点云归一化参数（由 set_normalization 设置）
        self.register_buffer('pc_center', torch.zeros(1, 1, 3))
        self.register_buffer('pc_scale', torch.ones(1, 1, 3))
        self.register_buffer('action_norm_factor', torch.tensor(1.0))

        # ── 时序编码器（TemporalMixin 依赖 self.temporal）──
        # action_window → cond（"当前动作编码"，每步重算，无记忆）
        _ENCODERS = {
            "ema": MultiScaleEMA,
            "fractional": FractionalMemory,
            "gamma": GammaLaguerreMemory,
            "gru": TemporalGRU,
            "transformer": TemporalTransformer,
            "tcn": TemporalTCN,
        }
        EncoderClass = _ENCODERS.get(encoder_type, FractionalMemory)
        self.temporal = EncoderClass(
            action_dim=action_dim,
            n_scales=n_orders,
            window_size=window_size,
            hidden_dim=hidden_dim,
        )

        # Z 位置嵌入：将 z 坐标（沿中心线，非潜变量 z）映射到 hidden_dim
        self.z_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ── 可学习迟滞潜变量 z_module（方案 A）──
        # z_init: 冷启动 z_0 = z_init(cond)
        self.z_init = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # z_cell: 演化 z_t = Φ_z(z_{t-1}, [cond, s_{t-1}])，GRUCell 门控利于迟滞建模
        self.z_cell = nn.GRUCell(
            input_size=hidden_dim + 3 * n_nodes,  # 拼接 cond 与 flatten(s_{t-1})
            hidden_size=z_dim,
        )
        # z_proj: 将 z_t 投影到 hidden，加性注入 per-node GRU 输入
        self.z_proj = nn.Linear(z_dim, hidden_dim)

        # ── 状态编码器 StateEncoder（warm start 的 GRU 种子，替代 init_hidden）──
        # 输入 [s_{t-1}, v=s_{t-1}-s_{t-2}]，输出 hidden 作为 GRU 初始状态
        self.state_encoder = nn.Sequential(
            nn.Linear(6 * n_nodes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU：沿 Z 轴的空间状态传播（悬臂梁因果性，同 SpatialSequenceModel）
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        # 每节点的增量预测头 Δ_raw（替代 SpatialSequenceModel 的绝对坐标 slice_head）
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # delta_scale: 可学习标量，约束 Δ 幅度，保证收缩、防 NaN
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

        # 冷启动回退：prev_skeleton=None 时用 cond 生成 GRU 种子（保持旧行为）
        self.init_hidden = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        """设置归一化参数（从数据集获取）。"""
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center).float()
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        self.pc_center = center.view(1, 1, 3)
        self.pc_scale = scale.view(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    def _get_z_positions(self, device):
        """获取归一化后的 z 位置序列 [-1, 1]（沿中心线，非潜变量 z）。"""
        return torch.linspace(-1, 1, self.n_nodes, device=device)

    def init_z_from_action(self, action_window):
        """从动作窗口初始化冷启动潜变量 z_0（rollout 首帧 / 序列训练首步用）。

        Args:
            action_window: (B, K, D) 动作窗口。
        Returns:
            z_0: (B, z_dim)。
        """
        cond = self.encode(action_window)
        return self.z_init(cond)

    def forward(self, batch_or_action_window, prev_skeleton=None,
                prev_prev_skeleton=None, prev_z=None):
        """预测下一步中心线状态 s_t = s_{t-1} + Δ（闭环状态转移）。

        Args:
            batch_or_action_window: dict batch（训练，含 action_window/prev_gt_skeleton）
                                    或 (B, K, D) action_window 张量（推理/旧调用）。
            prev_skeleton: (B, N, 3) 前一步骨架。None → 冷启动，回退 init_hidden(cond)。
            prev_prev_skeleton: (B, N, 3) 前两步骨架（用于速度 v）。None → v=0。
            prev_z: (B, z_dim) 前一步潜变量。None → 冷启动 z_0 = z_init(cond)。

        Returns:
            dict:
              'skeleton': (B, n_nodes, 3) 预测中心线（归一化空间）。
              'latent_z': (B, z_dim) z_t，供 rollout 喂回。
        """
        if isinstance(batch_or_action_window, dict):
            action_window = batch_or_action_window["action_window"]
        else:
            action_window = batch_or_action_window

        device = action_window.device
        B = action_window.shape[0]

        cond = self.encode(action_window)  # (B, hidden_dim)

        # ── 演化潜变量 z_t ──
        if prev_z is None:
            # 冷启动：z_0 = z_init(cond)
            z_t = self.z_init(cond)  # (B, z_dim)
        else:
            # warm start：z_t = GRUCell([cond, flatten(s_{t-1})], z_{t-1})
            if prev_skeleton is None:
                s_prev_flat = torch.zeros(B, 3 * self.n_nodes, device=device)
            else:
                s_prev_flat = prev_skeleton.reshape(B, -1)
            z_input = torch.cat([cond, s_prev_flat], dim=-1)  # (B, hidden + 3N)
            z_t = self.z_cell(z_input, prev_z)  # (B, z_dim)

        z_proj = self.z_proj(z_t)  # (B, hidden_dim)，加性注入 per-node GRU

        # ── 空间 GRU 初始状态 h_0 ──
        if prev_skeleton is None:
            # 冷启动：用 cond 生成种子（与 SpatialSequenceModel 旧行为一致）
            h = self.init_hidden(cond)
        else:
            # warm start：用 [s_{t-1}, v] 编码生成种子
            if prev_prev_skeleton is None:
                v = torch.zeros_like(prev_skeleton)
            else:
                v = prev_skeleton - prev_prev_skeleton
            state_input = torch.cat(
                [prev_skeleton.reshape(B, -1), v.reshape(B, -1)], dim=-1)  # (B, 6N)
            h = self.state_encoder(state_input)  # (B, hidden_dim)

        # ── 沿 Z 轴生成各节点增量 Δ ──
        z_positions = self._get_z_positions(device)
        skeleton = []

        for i in range(self.n_nodes):
            z_emb = self.z_embed(
                z_positions[i:i + 1].unsqueeze(0).expand(B, -1))  # (B, hidden)
            gru_input = cond + z_emb + z_proj
            h = self.gru(gru_input, h)
            delta_raw = self.delta_head(h)  # (B, 3)
            delta = self.delta_scale * torch.tanh(delta_raw)  # 收缩约束
            if prev_skeleton is None:
                # 冷启动首帧：s_{t-1} 视为零，s_t = Δ
                skeleton.append(delta)
            else:
                skeleton.append(prev_skeleton[:, i, :] + delta)

        skeleton = torch.stack(skeleton, dim=1)  # (B, n_nodes, 3)

        return {"skeleton": skeleton, "latent_z": z_t}

    def forward_sequence(self, action_windows, init_skeleton, teacher_states=None):
        """序列级前向：沿时间逐步 rollout，z 跨帧演化（Stage 1 序列训练用）。

        与单步 forward 的区别：
          - z 在序列内逐步演化（真正成为迟滞潜变量，而非每步从 cond 重初始化）
          - 梯度穿过所有时间步（BPTT），训练 z 的转移动力学
          - teacher_states 提供 scheduled sampling：每步的"前一步骨架"按需取 GT 或自身预测

        Args:
            action_windows: (B, T, seq_len, D) 每步动作窗口。
            init_skeleton: (B, N, 3) 首步前驱骨架（归一化空间，rollout 初始化）。
            teacher_states: (B, T, N, 3) | None。每步的 GT 骨架（归一化空间），
                           用于 scheduled sampling 决定 prev_skeleton 取 GT 还是自身预测。
                           None 时纯闭环（每步都喂自身预测）。

        Returns:
            dict:
              'skeletons': (B, T, N, 3) 每步预测中心线（归一化空间）。
              'final_z':   (B, z_dim) 序列末尾的 z（供后续分析）。
        """
        B, T = action_windows.shape[0], action_windows.shape[1]
        device = action_windows.device

        # 首步 z 从首步 action_window 初始化（冷启动）
        z_t = self.init_z_from_action(action_windows[:, 0])  # (B, z_dim)
        s_prev = init_skeleton                  # (B, N, 3)
        s_prev_prev = init_skeleton             # 首步无 t-2，v=0

        skeletons = []
        for t in range(T):
            out = self.forward(action_windows[:, t], s_prev, s_prev_prev, z_t)
            s_pred = out["skeleton"]            # (B, N, 3)
            z_t = out["latent_z"]               # z 跨帧演化（BPTT 梯度穿过）
            skeletons.append(s_pred)

            # scheduled sampling：更新下一步的 s_prev。
            # 若提供 teacher_states，下一步的 prev 用 GT（teacher forcing），
            # 否则用当前预测（闭环）——实际 mixing 由 trainer 按概率决定，这里只承接 trainer 传入的 prev。
            if teacher_states is not None:
                s_prev_prev = s_prev
                s_prev = teacher_states[:, t]   # teacher forcing：下一步 prev = GT
            else:
                s_prev_prev = s_prev
                s_prev = s_pred                 # 闭环：下一步 prev = 自身预测

        return {"skeletons": torch.stack(skeletons, dim=1), "final_z": z_t}

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """计算训练损失（warm start，从 batch 取 GT 前一步骨架作 teacher forcing）。

        z 无 GT，不加 loss（端到端从 skeleton loss 学）。
        """
        # TemporalMixin.compute_losses 处理 "smooth" loss（依赖 action_window_next）
        losses = super().compute_losses(batch, phase_spec)
        active = set(phase_spec.active_losses)

        device = next(self.parameters()).device
        action_window = batch["action_window"].to(device)
        gt_skeleton = batch["gt_skeleton"].to(device)
        prev_skeleton = batch.get("prev_gt_skeleton")
        prev_prev_skeleton = batch.get("prev_prev_gt_skeleton")
        if prev_skeleton is not None:
            prev_skeleton = prev_skeleton.to(device)
        if prev_prev_skeleton is not None:
            prev_prev_skeleton = prev_prev_skeleton.to(device)

        pred = self.forward(action_window, prev_skeleton, prev_prev_skeleton)
        pred_skeleton = pred["skeleton"]

        if "skeleton" in active:
            losses["skeleton"] = F.mse_loss(pred_skeleton, gt_skeleton)

        if "spatial_smooth" in active:
            pred_delta = pred_skeleton[:, 1:, :] - pred_skeleton[:, :-1, :]
            gt_delta = gt_skeleton[:, 1:, :] - gt_skeleton[:, :-1, :]
            losses["spatial_smooth"] = F.mse_loss(pred_delta, gt_delta)

        return losses

    @torch.no_grad()
    def predict_skeleton(self, action_window, prev_skeleton=None, prev_z=None,
                         prev_prev_skeleton=None):
        """推理：预测中心线（物理坐标）。

        单参调用（prev 全 None）→ 冷启动，向后兼容旧调用方式。
        rollout 调用 → 传入上一步的 pred_skeleton 和 latent_z。

        Args:
            action_window: (B, K, D)。
            prev_skeleton: (B, N, 3) 上一步骨架（归一化空间），可选。
            prev_z: (B, z_dim) 上一步 z，可选。
            prev_prev_skeleton: (B, N, 3) 前两步骨架（速度），可选。

        Returns:
            (B, N, 3) 预测中心线（物理坐标，反归一化）。
        """
        device = next(self.parameters()).device
        action_window = action_window.to(device)
        norm = self.action_norm_factor.item()
        if norm > 1.01:
            action_window = action_window / norm

        pred = self.forward(action_window, prev_skeleton, prev_prev_skeleton, prev_z)
        pred_skeleton = pred["skeleton"]
        return pred_skeleton * self.pc_scale.to(device) + self.pc_center.to(device)

    @torch.no_grad()
    def predict_pointcloud(self, action_window, n_points=1000, avg_radius=0.015):
        """从预测中心线采样表面点（兼容评估流程，冷启动模式）。"""
        skeleton = self.predict_skeleton(action_window)
        B = skeleton.shape[0]
        all_points = []
        for b in range(B):
            skel_np = skeleton[b].cpu().numpy().T
            pts, _ = _sample_surface(skel_np, avg_radius, n_points)
            if len(pts) < n_points:
                pad = np.tile(pts[-1:], (n_points - len(pts), 1))
                pts = np.concatenate([pts, pad], axis=0)
            all_points.append(torch.from_numpy(pts[:n_points]).float())
        return torch.stack(all_points, dim=0).to(skeleton.device)
