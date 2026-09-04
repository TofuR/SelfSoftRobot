"""HOV2.1: PI/Maxwell hereditary state with deterministic geometry readout.

The operator recurrence is inherited from :class:`HereditaryOperatorModel`.
Only its free point-coordinate readout is replaced:

    xi = xi_eq(a) + W_PI q + W_M (h-e) + eps_mem(q, h-e)
    skeleton = deterministic_geometry(xi)

``xi`` contains eight coefficients in a fixed fit-only POD bending basis and
two section log-length offsets. Every learned structured readout direction is
unit norm in an immutable fit-residual metric, and all scalar gains are
nonnegative. The optional neural residual reads memory only and is exactly
zero when both operator branches are at equilibrium.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_hereditary_operator import HereditaryOperatorModel
from src.models.model_ishsm import (
    generalized_to_skeleton,
    skeleton_to_generalized,
)
from src.training.spec import PhaseSpec, TrainingSpec


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


class HereditaryGeometryModel(HereditaryOperatorModel):
    """Hereditary PI/Maxwell state decoded through observable shape geometry."""

    training_spec = TrainingSpec(phases=[PhaseSpec(
        name="hereditary_geometry",
        dataset_type="state_transition",
        supervision_mode="spatial_sequence",
        active_losses=[
            "skeleton", "spatial_smooth", "bend", "length", "endpoint",
            "residual_bend", "residual_length"],
        forward_attr="forward",
        use_episode_mode=True,
        teacher_forcing_ratio=1.0,
        episode_len=40,
    )])

    def __init__(
            self,
            action_dim: int = 4,
            n_nodes: int = 15,
            window_size: int = 40,
            n_play: int = 2,
            n_maxwell: int = 6,
            dt: float = 0.1,
            r_range=(0.02, 0.5),
            tau_range=(0.3, 2.0),
            burnin_mode: str = "equilibrium",
            n_bend_modes: int = 8,
            section_intervals: Iterable[int] = (7, 7),
            bend_basis=None,
            generalized_coordinate_scale=None,
            reference_segment_lengths=None,
            reference_bend_bias=None,
            reference_bend_dirs=None,
            reference_length_bias=None,
            reference_length_dirs=None,
            reference_kind: str = "linear",
            reference_knots=None,
            reference_drive_weights=None,
            base_position=None,
            residual_mode: str = "memory",
            bend_residual_max_rad: float = 0.05,
            length_residual_max_log: float = 0.02,
            episode_len: int = 40):
        super().__init__(
            action_dim=action_dim, n_nodes=n_nodes,
            window_size=window_size, n_play=n_play, n_maxwell=n_maxwell,
            dt=dt, r_range=r_range, tau_range=tau_range,
            burnin_mode=burnin_mode, residual_scale_max=0.0,
            episode_len=episode_len)

        # Remove every free point-coordinate readout from HOV2. The inherited
        # drive and PI/Maxwell state recurrence are deliberately retained.
        for name in (
                "static_bias", "static_dirs", "play_modes", "maxwell_modes",
                "residual", "residual_scale"):
            delattr(self, name)
        del self.maxwell.weights

        self.n_bend_modes = int(n_bend_modes)
        self.section_intervals = tuple(int(v) for v in section_intervals)
        self.n_sections = len(self.section_intervals)
        self.generalized_dim = self.n_bend_modes + self.n_sections
        self.reference_kind = str(reference_kind)
        self.residual_mode = str(residual_mode)
        self.bend_residual_max_rad = float(bend_residual_max_rad)
        self.length_residual_max_log = float(length_residual_max_log)
        self.spatial_propagation_direction = "generalized_geometry"
        self.gl_kernel_alignment = "current_at_window_end"
        self.model_contract_version = 3

        if self.n_nodes - 1 != sum(self.section_intervals):
            raise ValueError("section_intervals 与 n_nodes 不匹配")
        if not 1 <= self.n_bend_modes <= self.n_nodes - 1:
            raise ValueError("n_bend_modes 必须在 [1,n_nodes-1]")
        if self.residual_mode not in {"none", "memory"}:
            raise ValueError("residual_mode 必须为 none 或 memory")
        if self.bend_residual_max_rad <= 0:
            raise ValueError("bend_residual_max_rad 必须为正数")
        if self.length_residual_max_log <= 0:
            raise ValueError("length_residual_max_log 必须为正数")

        n_bend = self.n_nodes - 1
        if bend_basis is None:
            grid = torch.arange(n_bend, dtype=torch.float32).unsqueeze(1)
            modes = torch.arange(
                self.n_bend_modes, dtype=torch.float32).unsqueeze(0)
            bend_basis = torch.cos(
                math.pi * (grid + 0.5) * modes / n_bend)
            bend_basis, _ = torch.linalg.qr(bend_basis, mode="reduced")
        defaults = {
            "bend_basis": (bend_basis, (n_bend, self.n_bend_modes)),
            "generalized_coordinate_scale": (
                torch.ones(self.generalized_dim)
                if generalized_coordinate_scale is None
                else generalized_coordinate_scale,
                (self.generalized_dim,)),
            "reference_segment_lengths": (
                torch.ones(n_bend) if reference_segment_lengths is None
                else reference_segment_lengths, (n_bend,)),
            "reference_bend_bias": (
                torch.zeros(n_bend) if reference_bend_bias is None
                else reference_bend_bias, (n_bend,)),
            "reference_bend_dirs": (
                torch.zeros(self.action_dim, n_bend)
                if reference_bend_dirs is None else reference_bend_dirs,
                (self.action_dim, n_bend)),
            "reference_length_bias": (
                torch.zeros(self.n_sections)
                if reference_length_bias is None else reference_length_bias,
                (self.n_sections,)),
            "reference_length_dirs": (
                torch.zeros(self.action_dim, self.n_sections)
                if reference_length_dirs is None else reference_length_dirs,
                (self.action_dim, self.n_sections)),
            "base_position": (
                torch.zeros(3) if base_position is None else base_position,
                (3,)),
        }
        for name, (value, expected_shape) in defaults.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name} 应为 {expected_shape}，得到 {tuple(tensor.shape)}")
            if name in {"generalized_coordinate_scale",
                         "reference_segment_lengths"} and torch.any(tensor <= 0):
                raise ValueError(f"{name} 必须严格为正")
            self.register_buffer(name, tensor.clone())

        if self.reference_kind == "monotone_spline":
            knots = torch.as_tensor(reference_knots, dtype=torch.float32)
            weights = torch.as_tensor(
                reference_drive_weights, dtype=torch.float32)
            if knots.ndim != 1 or knots.numel() < 2:
                raise ValueError("monotone_spline reference_knots 合同无效")
            if weights.shape != (self.action_dim, knots.numel()):
                raise ValueError("reference_drive_weights 合同无效")
            if torch.any(weights < 0):
                raise ValueError("reference_drive_weights 必须非负")
            self.register_buffer("reference_knots", knots.clone())
            self.register_buffer("reference_drive_weights", weights.clone())
        elif self.reference_kind != "linear":
            raise ValueError(f"未知 reference_kind: {self.reference_kind!r}")

        # Each action/operator pair owns a direction in the same observable
        # 8+2 coordinate system. Unit normalization removes gain-mode scaling
        # freedom; nonnegative scalar gains leave the direction sign explicit.
        self.pi_mode_directions_raw = nn.Parameter(torch.randn(
            self.action_dim, self.n_play, self.generalized_dim))
        self.maxwell_mode_directions_raw = nn.Parameter(torch.randn(
            self.action_dim, self.n_maxwell, self.generalized_dim))
        with torch.no_grad():
            self.play.raw_weights.fill_(_inverse_softplus(0.5))
        self.maxwell_gain_raw = nn.Parameter(torch.full(
            (self.action_dim, self.n_maxwell), _inverse_softplus(0.5)))

        if self.residual_mode == "memory":
            residual_in = self.action_dim * (self.n_play + self.n_maxwell)
            self.memory_residual_net = nn.Sequential(
                nn.Linear(residual_in, 32),
                nn.SiLU(),
                nn.Linear(32, self.generalized_dim),
            )
            self.raw_residual_bend_scale = nn.Parameter(
                torch.tensor(torch.logit(torch.tensor(0.2)).item()))
            self.raw_residual_length_scale = nn.Parameter(
                torch.tensor(torch.logit(torch.tensor(0.2)).item()))

    @property
    def pi_mode_directions(self) -> torch.Tensor:
        return F.normalize(self.pi_mode_directions_raw, dim=-1, eps=1e-8)

    @property
    def maxwell_mode_directions(self) -> torch.Tensor:
        return F.normalize(
            self.maxwell_mode_directions_raw, dim=-1, eps=1e-8)

    @property
    def maxwell_gains(self) -> torch.Tensor:
        return F.softplus(self.maxwell_gain_raw)

    @property
    def residual_coordinate_scales(self) -> torch.Tensor:
        bend = self.bend_residual_max_rad * torch.sigmoid(
            self.raw_residual_bend_scale)
        length = self.length_residual_max_log * torch.sigmoid(
            self.raw_residual_length_scale)
        return torch.cat([
            bend.expand(self.n_bend_modes), length.expand(self.n_sections)])

    def _reference(self, action: torch.Tensor):
        reference_drive = action
        if self.reference_kind == "monotone_spline":
            hinges = torch.relu(action.unsqueeze(-1) - self.reference_knots)
            reference_drive = (
                hinges * self.reference_drive_weights).sum(dim=-1)
        bend = self.reference_bend_bias + torch.einsum(
            "bc,cn->bn", reference_drive, self.reference_bend_dirs)
        length = self.reference_length_bias + torch.einsum(
            "bc,cs->bs", reference_drive, self.reference_length_dirs)
        return bend, length

    def _memory_residual(
            self, q: torch.Tensor, deficit: torch.Tensor) -> torch.Tensor:
        batch = q.shape[0]
        if self.residual_mode == "none":
            return torch.zeros(
                batch, self.generalized_dim, dtype=q.dtype, device=q.device)
        memory = torch.cat([
            q.reshape(batch, -1), deficit.reshape(batch, -1)], dim=-1)
        zero = torch.zeros_like(memory)
        # Subtracting the same network at zero memory makes eps_mem(0,0)=0
        # constructively, even though the MLP contains biases.
        centered = 0.5 * (
            torch.tanh(self.memory_residual_net(memory)) -
            torch.tanh(self.memory_residual_net(zero)))
        return centered * self.residual_coordinate_scales.to(centered)

    def _structured_memory(
            self, q: torch.Tensor, deficit: torch.Tensor):
        scale = self.generalized_coordinate_scale
        pi_components = (
            self.play.weights[None, :, :, None]
            * q[:, :, :, None]
            * self.pi_mode_directions[None, :, :, :]
            * scale[None, None, None, :])
        maxwell_components = (
            self.maxwell_gains[None, :, :, None]
            * deficit[:, :, :, None]
            * self.maxwell_mode_directions[None, :, :, :]
            * scale[None, None, None, :])
        return (
            pi_components.sum(dim=(1, 2)),
            maxwell_components.sum(dim=(1, 2)),
            pi_components,
            maxwell_components,
        )

    def _decode_generalized(
            self, action: torch.Tensor,
            memory_generalized: torch.Tensor) -> torch.Tensor:
        bend_ref, length_ref = self._reference(action)
        bend = (
            bend_ref +
            memory_generalized[:, :self.n_bend_modes] @ self.bend_basis.T)
        length = length_ref + memory_generalized[:, self.n_bend_modes:]
        physical = generalized_to_skeleton(
            bend, length, self.reference_segment_lengths,
            self.section_intervals, self.base_position)
        return ((physical - self.pc_center.to(physical)) /
                self.pc_scale.to(physical))

    def decode_equilibrium(self, action: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(
            action.shape[0], self.generalized_dim,
            dtype=action.dtype, device=action.device)
        return self._decode_generalized(action, zeros)

    def forward(self, batch_or_action_window, prev_skeleton=None,
                prev_prev_skeleton=None, prev_z=None):
        del prev_skeleton, prev_prev_skeleton
        action_window = (
            batch_or_action_window["action_window"]
            if isinstance(batch_or_action_window, dict)
            else batch_or_action_window)
        action = action_window[:, -1]
        if prev_z is None:
            p, h = self._burn_in(action_window[:, :-1])
        else:
            p, h = self._unpack_state(prev_z)
        drive = self.drive(action)
        p, q = self.play.step(p, drive)
        h = self.maxwell.step(h, drive)
        deficit = h - drive.unsqueeze(-1)

        pi, maxwell, pi_components, maxwell_components = \
            self._structured_memory(q, deficit)
        residual = self._memory_residual(q, deficit)
        memory = pi + maxwell + residual
        skeleton = self._decode_generalized(action, memory)
        return {
            "skeleton": skeleton,
            "latent_z": self._pack_state(p, h),
            "memory_generalized": memory,
            "pi_generalized": pi,
            "maxwell_generalized": maxwell,
            "memory_residual_generalized": residual,
            "pi_generalized_components": pi_components,
            "maxwell_generalized_components": maxwell_components,
        }

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        device = self.dt.device
        action_window = batch["action_window"].to(device)
        gt = batch["gt_skeleton"].to(device)
        output = self.forward(action_window)
        pred = output["skeleton"]
        losses = {"skeleton": F.mse_loss(pred, gt)}
        if "spatial_smooth" in phase_spec.active_losses:
            losses["spatial_smooth"] = F.mse_loss(
                pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
        losses.update(self.compute_sequence_aux_losses(
            pred.unsqueeze(1), gt.unsqueeze(1)))
        losses.update(self.compute_rollout_aux_losses(
            pred.unsqueeze(1), gt.unsqueeze(1), {
                key: value.unsqueeze(1) for key, value in output.items()
                if key.endswith("_generalized")}))
        return losses

    def compute_sequence_aux_losses(
            self, pred_seq: torch.Tensor, gt_seq: torch.Tensor) -> dict:
        batch, steps, nodes, dims = pred_seq.shape
        pred = (pred_seq.reshape(batch * steps, nodes, dims) *
                self.pc_scale.to(pred_seq) + self.pc_center.to(pred_seq))
        gt = (gt_seq.reshape(batch * steps, nodes, dims) *
              self.pc_scale.to(gt_seq) + self.pc_center.to(gt_seq))
        pred_b, pred_l, _ = skeleton_to_generalized(
            pred, self.reference_segment_lengths, self.section_intervals)
        gt_b, gt_l, _ = skeleton_to_generalized(
            gt, self.reference_segment_lengths, self.section_intervals)
        bend_scale = gt_b.detach().std(dim=0).clamp_min(0.02)
        length_scale = gt_l.detach().std(dim=0).clamp_min(0.005)
        return {
            "bend": F.mse_loss(pred_b / bend_scale, gt_b / bend_scale),
            "length": F.mse_loss(pred_l / length_scale, gt_l / length_scale),
            "endpoint": F.mse_loss(
                pred_seq[:, :, -1], gt_seq[:, :, -1]),
        }

    def compute_rollout_aux_losses(
            self, pred_seq: torch.Tensor, gt_seq: torch.Tensor,
            rollout_outputs: dict) -> dict:
        del pred_seq, gt_seq
        residual = rollout_outputs["memory_residual_generalized"]
        if self.residual_mode == "none":
            zero = residual.sum() * 0.0
            return {"residual_bend": zero, "residual_length": zero}
        return {
            # These are radians^2 and log-length^2, not normalized point-space
            # amplitudes. Their weights therefore have an explicit unit basis.
            "residual_bend": residual[..., :self.n_bend_modes].square().mean(),
            "residual_length": residual[..., self.n_bend_modes:].square().mean(),
        }

    def hysteresis_report(self) -> dict:
        return {
            "play_thresholds": self.play.thresholds.detach().cpu(),
            "play_weights": self.play.weights.detach().cpu(),
            "maxwell_taus": self.maxwell.taus.detach().cpu(),
            "maxwell_weights": self.maxwell_gains.detach().cpu(),
            "pi_mode_directions": self.pi_mode_directions.detach().cpu(),
            "maxwell_mode_directions": (
                self.maxwell_mode_directions.detach().cpu()),
            "generalized_coordinate_scale": (
                self.generalized_coordinate_scale.detach().cpu()),
            "dt": self.dt.item(),
        }

    def geometry_report(self) -> dict:
        return {
            "n_bend_modes": self.n_bend_modes,
            "n_sections": self.n_sections,
            "section_intervals": self.section_intervals,
            "reference_kind": self.reference_kind,
            "residual_mode": self.residual_mode,
            "residual_coordinate_scales": (
                self.residual_coordinate_scales.detach().cpu()
                if self.residual_mode == "memory" else
                torch.zeros(self.generalized_dim)),
            "bend_basis": self.bend_basis.detach().cpu(),
        }
