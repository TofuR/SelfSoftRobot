"""HereditaryOperatorModel — 显式迟滞算子模型（设计文档 Version B，最小外科）。

架构（docs/designs/2026-08-29_hereditary_operator_model.md §2.2）:

    e_t = MonotoneSplineDrive(a_t)                     逐通道单调驱动（伪应变）
    s̃_t = bias + Σ_c e_{c,t} · D_c                     静态平衡形状（无记忆）
    q_t = e_t − PlayBank 状态                           率无关摩擦偏移（|q|≤r）
    d_t = h_t − e_t                                     粘弹亏量（h: Maxwell 状态）
    s_t = s̃_t
        + Σ_{c,j} b_{c,j}·q_{c,j,t} · B_j(s̃_t)         play 贡献（局部标架模态）
        + Σ_{c,k} v_{c,k}·d_{c,k,t} · C_k(s̃_t)         Maxwell 贡献（亏量形式）
        + r_t                                           小残差（tanh 有界）

四公理的构造性落实:
  1. 静态通道无记忆 —— s̃ 只依赖 a_t，且经逐通道单调样条（无跨通道 MLP）。
  2. 严格可加性 —— q/d 只以加性项进入读出，永不喂回任何编码器/GRU。
  3. 谱非负 —— b ≥ 0 (softplus)；Maxwell 读出亏量形式，平衡态贡献为零。
  4. 输入合同 —— 只吃动作通道；无骨架反馈（gt 与 open_loop 对本模型等价）。

状态不变量（供 rollout / BPTT）:
  packed state（"latent_z" 槽）= [p (C·J), h (C·M)]，语义是
  "算子已消费到 t−1 步输入"。forward(aw_t, prev_z) 消费 a_t = aw_t[:, -1]
  恰好一次；init_z_from_action(aw_0) 烧入 aw_0[:, :-1]（不含当前步），
  留给首个 forward 消费——同一动作不被重复计入。

冷启动约定: 静息 p=0, h=0（零压静止），在窗口前 K−1 步烧入。慢模态
（τ ≫ K·Δt）初始化不足是已知限制（设计文档 §六 v2: TBPTT 跨 episode 续态）。

与 StateTransitionSpatialModel 的接口兼容（trainer 无需改动）:
  forward(action_window, prev_skeleton, prev_prev_skeleton, prev_z)
      → {"skeleton": (B,N,3), "latent_z": (B, C·(J+M))}
  init_z_from_action(action_window) / set_normalization / predict_skeleton
  prev_skeleton / prev_prev_skeleton 被忽略（遗传模型无骨架反馈）。

训练: episode 模式（BPTT 穿透 clamp/exp），L_skeleton + L_spatial_smooth
（trainer episode 路径内联计算，不走 compute_losses）。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.operators import (
    MonotoneSplineDrive, PlayBank, MaxwellBank, LocalFrameModeBank,
)
from src.training.spec import TrainingSpec, PhaseSpec


class HereditaryOperatorModel(nn.Module):
    """显式迟滞算子模型（电平读出，非增量）。

    Args:
        action_dim: 独立驱动通道数 C。
        n_nodes: 骨架节点数 N。
        window_size: 动作窗口长度 K（仅用于冷启动烧入步数）。
        n_play: play 算子数 J（率无关迟滞容量）。
        n_maxwell: Maxwell 元件数 M（率相关迟滞容量）。
        dt: 采样间隔（秒，必须与数据合同一致；实物 10 Hz → 0.1）。
        r_range: play 死区阈值对数网格范围。
        tau_range: Maxwell 时间常数对数网格范围（None → [3dt, 10s]）。
        residual_scale_max: 残差幅度上限（归一化骨架单位；残差必须保持
            "小"——超过此值说明结构欠拟合，应加容量而非放残差）。
    """

    training_spec = TrainingSpec(
        phases=[
            PhaseSpec(
                name="hereditary",
                dataset_type="state_transition",
                supervision_mode="spatial_sequence",
                # episode 路径只计算 skeleton + spatial_smooth（同
                # GTObservedTransitionModel；无 action_window_next 可用）
                active_losses=["skeleton", "spatial_smooth"],
                forward_attr="forward",
                # 算子状态经 latent_z 槽在 episode 内 BPTT 演化。
                # tf 对本模型无意义（无骨架反馈，GT/自回归等价），
                # 取 1.0 让 trainer 走纯 GT 路径（无 scheduled sampling 分支）。
                use_episode_mode=True,
                teacher_forcing_ratio=1.0,
                episode_len=40,
            ),
        ],
    )

    def __init__(
        self,
        action_dim=2,
        n_nodes=31,
        window_size=20,
        n_play=8,
        n_maxwell=6,
        dt=0.1,
        r_range=(0.02, 0.5),
        tau_range=None,
        residual_scale_max=0.3,
        episode_len=40,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.n_play = n_play
        self.n_maxwell = n_maxwell
        # 节点/合同属性（与 state_transition 家族同槽，供检查器识别）
        self.node_order = "base_to_tip"
        self.model_contract_version = 2
        # dt 必须进合同（设计文档 §六）;τ 网格派生自它
        self.register_buffer("dt", torch.tensor(float(dt)))

        # ── 归一化参数（set_normalization 设置，同 state_transition 家族）──
        self.register_buffer('pc_center', torch.zeros(1, 1, 3))
        self.register_buffer('pc_scale', torch.ones(1, 1, 3))
        self.register_buffer('action_norm_factor', torch.tensor(1.0))

        # ── 算子层（src/operators，各自独立可测）──
        self.drive = MonotoneSplineDrive(action_dim)          # e_c(a_c)
        self.play = PlayBank(action_dim, n_play, r_range)     # b_{c,j}, r_j
        self.maxwell = MaxwellBank(action_dim, n_maxwell, dt,
                                   tau_range)                 # v_{c,k}, τ_k
        self.play_modes = LocalFrameModeBank(n_play, n_nodes)      # B_j
        self.maxwell_modes = LocalFrameModeBank(n_maxwell, n_nodes)  # C_k

        # ── 静态读出（公理 1: 无记忆、逐通道、无跨通道 MLP）──
        # 归一化骨架空间零均值 → bias=0 即静息形状
        self.static_bias = nn.Parameter(torch.zeros(n_nodes, 3))
        self.static_dirs = nn.Parameter(
            0.02 * torch.randn(action_dim, n_nodes, 3))       # D_c

        # ── 残差（公理 2: 输入只有 a_t 与算子读出，无历史窗口/骨架）──
        residual_in = action_dim + action_dim * n_play + action_dim * n_maxwell
        self.residual = nn.Sequential(
            nn.Linear(residual_in, 32),
            nn.SiLU(),
            nn.Linear(32, n_nodes * 3),
            nn.Tanh(),                                        # 有界 ±1
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        self.residual_scale_max = float(residual_scale_max)
        # 允许构造时覆盖 spec 默认 episode_len（同 GTObserved 模式）
        self.episode_len = episode_len
        self.training_spec.phases[0].episode_len = episode_len

    # ── 状态打包（latent_z 槽 = 算子状态）──

    @property
    def operator_state_dim(self) -> int:
        return self.action_dim * (self.n_play + self.n_maxwell)

    def _pack_state(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """(B,C,J) + (B,C,M) → (B, C·(J+M))。"""
        B = p.shape[0]
        return torch.cat([p.reshape(B, -1), h.reshape(B, -1)], dim=-1)

    def _unpack_state(self, state: torch.Tensor):
        """(B, C·(J+M)) → (p (B,C,J), h (B,C,M))。"""
        B = state.shape[0]
        sizes = [self.action_dim * self.n_play,
                 self.action_dim * self.n_maxwell]
        p_flat, h_flat = state.split(sizes, dim=-1)
        return (p_flat.reshape(B, self.action_dim, self.n_play),
                h_flat.reshape(B, self.action_dim, self.n_maxwell))

    # ── 冷启动烧入 ──

    def _burn_in(self, action_window: torch.Tensor):
        """静息起烧入算子状态。

        Args:
            action_window: (B, K, D) —— 应传"当前步之前"的历史
                （K−1 个元素；当前步留给 forward 的正式 step 消费）。
        Returns:
            (p, h) 各 (B, C, J) / (B, C, M)。
        """
        B = action_window.shape[0]
        device = action_window.device
        p = self.play.init_state(B, device)
        h = self.maxwell.init_state(B, device)
        for k in range(action_window.shape[1]):
            e = self.drive(action_window[:, k])              # (B, C)
            p, _ = self.play.step(p, e)
            h = self.maxwell.step(h, e)
        return p, h

    def init_z_from_action(self, action_window: torch.Tensor) -> torch.Tensor:
        """从动作窗口初始化算子状态（rollout 首帧 / episode 首步用）。

        烧入窗口的前 K−1 步（状态语义 = "已消费到 t−1"），最后一步
        a_t 留给首个 forward 消费——与 trainer 的调用序
        z_0 = init_z(aw[:,0]); forward(aw[:,0], ..., z_0) 对齐，
        当前动作只计一次。
        """
        return self._pack_state(*self._burn_in(action_window[:, :-1]))

    # ── 前向 ──

    def forward(self, batch_or_action_window, prev_skeleton=None,
                prev_prev_skeleton=None, prev_z=None):
        """单步转移 s_t = 读出(算子状态, a_t)（电平，非增量）。

        Args:
            batch_or_action_window: dict batch（含 action_window）或
                (B, K, D) 动作窗口张量。只取 [:, -1] 作当前动作 a_t。
            prev_skeleton / prev_prev_skeleton: 兼容 trainer 签名，忽略
                （遗传模型无骨架反馈）。
            prev_z: (B, C·(J+M)) 上一步算子状态。None → 冷启动
                （烧入窗口前 K−1 步后按正常步进消费 a_t）。
        Returns:
            dict: 'skeleton' (B, N, 3)（归一化空间）；
                  'latent_z' (B, C·(J+M)) 更新后算子状态（喂回下一步）。
        """
        if isinstance(batch_or_action_window, dict):
            action_window = batch_or_action_window["action_window"]
        else:
            action_window = batch_or_action_window

        a_t = action_window[:, -1, :]                        # (B, C)
        B, device = a_t.shape[0], a_t.device

        # ── 算子状态推进（当前动作恰好消费一次）──
        if prev_z is None:
            p, h = self._burn_in(action_window[:, :-1])
        else:
            p, h = self._unpack_state(prev_z)
        e_t = self.drive(a_t)                                # (B, C)
        p, q = self.play.step(p, e_t)                        # q (B, C, J)
        h = self.maxwell.step(h, e_t)                        # (B, C, M)

        # ── 静态平衡形状 s̃_t（公理 1）──
        s_static = self.static_bias.unsqueeze(0) + \
            torch.einsum('bc,cnd->bnd', e_t, self.static_dirs)

        # ── play 贡献: Σ_j (Σ_c b_{c,j} q_{c,j}) · B_j(s̃) ──
        w_q = self.play.weights.unsqueeze(0) * q             # (B, C, J)
        play_modes = self.play_modes(s_static)               # (J, B, N, 3)
        contrib_play = torch.einsum('bcj,jbnf->bnf', w_q, play_modes)

        # ── Maxwell 贡献（亏量形式）: Σ_k (Σ_c v_{c,k} d_{c,k}) · C_k(s̃) ──
        d_h = h - e_t.unsqueeze(-1)                          # (B, C, M)
        w_d = self.maxwell.weights.unsqueeze(0) * d_h        # (B, C, M)
        mw_modes = self.maxwell_modes(s_static)              # (M, B, N, 3)
        contrib_mw = torch.einsum('bcm,mbnf->bnf', w_d, mw_modes)

        # ── 残差（有界，输入无历史）──
        res_in = torch.cat(
            [a_t, q.reshape(B, -1), d_h.reshape(B, -1)], dim=-1)
        scale = torch.clamp(self.residual_scale, max=self.residual_scale_max)
        residual = scale * self.residual(res_in).view(B, self.n_nodes, 3)

        skeleton = s_static + contrib_play + contrib_mw + residual
        return {"skeleton": skeleton, "latent_z": self._pack_state(p, h)}

    # ── 训练 / 推理辅助 ──

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        """设置归一化参数（从数据集获取；同 state_transition 家族合同）。"""
        if isinstance(center, np.ndarray):
            center = torch.from_numpy(center).float()
        if isinstance(scale, np.ndarray):
            scale = torch.from_numpy(scale).float()
        self.pc_center = center.view(1, 1, 3)
        self.pc_scale = scale.view(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        """逐帧回退路径（非 episode 训练 / 冒烟测试用）。

        episode 模式下 trainer 走 _compute_sequence_losses，不经此函数。
        逐帧模式 = 每个样本冷启动烧入窗口（K 步截断的已知限制）后单步预测。
        """
        losses = {}
        active = set(phase_spec.active_losses)
        device = next(self.parameters()).device
        action_window = batch["action_window"].to(device)
        gt_skeleton = batch["gt_skeleton"].to(device)

        pred = self.forward(action_window)
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
        """推理: 预测中心线（物理坐标，反归一化）。

        单参调用（prev 全 None）→ 冷启动烧入窗口。rollout 调用传入上
        一步 latent_z（算子状态）。签名与 state_transition 家族一致。
        """
        device = next(self.parameters()).device
        action_window = action_window.to(device)
        norm = self.action_norm_factor.item()
        if norm > 1.01:
            action_window = action_window / norm

        pred = self.forward(action_window, prev_z=prev_z)
        pred_skeleton = pred["skeleton"]
        return pred_skeleton * self.pc_scale.to(device) + \
            self.pc_center.to(device)

    # ── 定量谱读出（论文核心交付物）──

    def hysteresis_report(self) -> dict:
        """迟滞谱读出: PI 密度 μ_c(r_j) 与 Maxwell 谱 |v_{c,k}|(τ_k)。

        返回的两组 (网格, 逐通道权重) 即设计文档定义的"迟滞指纹"，
        可直接用于: E1 判据的谱质量检验、逐通道分解报告、方差归因。
        """
        return {
            "play_thresholds": self.play.thresholds.detach().cpu(),
            "play_weights": self.play.weights.detach().cpu(),
            "maxwell_taus": self.maxwell.taus.detach().cpu(),
            "maxwell_weights": self.maxwell.weights.detach().cpu().abs(),
            "dt": self.dt.item(),
        }
