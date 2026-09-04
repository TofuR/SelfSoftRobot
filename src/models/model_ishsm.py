"""Interpretable Spatiotemporal Hereditary Shape Model (ISHSM).

The model represents planar shape with observable generalized coordinates:

* 14 bending coordinates: base tangent angle plus 13 turning angles;
* 2 section-wise log length scales for the two seven-interval sections.

A frozen memoryless reference is fitted from the current action.  Eight fixed
POD bending modes plus two section length residuals form the recurrent state.
The state follows a stable first-order update and is initialized from one
observed skeleton.  An optional deterministic damped-least-squares correction
makes the low-rank observation projection consistent with the measured tip.
No free coordinate residual is present.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.spec import PhaseSpec, TrainingSpec


def _section_index(section_intervals: Iterable[int], n_segments: int,
                   device=None) -> torch.Tensor:
    intervals = tuple(int(v) for v in section_intervals)
    if not intervals or any(v <= 0 for v in intervals):
        raise ValueError("section_intervals 必须是正整数序列")
    if sum(intervals) != n_segments:
        raise ValueError(
            f"section_intervals 总和必须等于线段数 {n_segments}，得到 {intervals}")
    return torch.repeat_interleave(
        torch.arange(len(intervals), device=device),
        torch.tensor(intervals, device=device))


def skeleton_to_generalized(
        skeleton: torch.Tensor,
        reference_segment_lengths: torch.Tensor,
        section_intervals: Iterable[int]):
    """Convert planar skeletons to bending coordinates and section scales.

    Args:
        skeleton: ``(B, N, 3)`` physical coordinates.
        reference_segment_lengths: ``(N-1,)`` positive physical lengths.
        section_intervals: number of segments in every physical section.

    Returns:
        bend ``(B,N-1)`` (base tangent + wrapped turning angles), section log
        scales ``(B,S)``, and base position ``(B,3)``.
    """
    if skeleton.ndim != 3 or skeleton.shape[-1] != 3:
        raise ValueError("skeleton 必须为 (B,N,3)")
    n_segments = skeleton.shape[1] - 1
    ref = torch.as_tensor(
        reference_segment_lengths, dtype=skeleton.dtype,
        device=skeleton.device).reshape(-1)
    if ref.numel() != n_segments or torch.any(ref <= 0):
        raise ValueError("reference_segment_lengths 维度错误或包含非正值")
    section_ids = _section_index(
        section_intervals, n_segments, device=skeleton.device)

    delta = skeleton[:, 1:, :] - skeleton[:, :-1, :]
    lengths = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-8)
    angles = torch.atan2(delta[..., 1], delta[..., 0])
    turning_raw = angles[:, 1:] - angles[:, :-1]
    turning = torch.atan2(torch.sin(turning_raw), torch.cos(turning_raw))
    bend = torch.cat([angles[:, :1], turning], dim=1)

    scales = []
    for section in range(len(tuple(section_intervals))):
        mask = section_ids == section
        current = lengths[:, mask].sum(dim=1)
        baseline = ref[mask].sum()
        scales.append(torch.log(current / baseline))
    log_section_scale = torch.stack(scales, dim=1)
    return bend, log_section_scale, skeleton[:, 0, :]


def generalized_to_skeleton(
        bend: torch.Tensor,
        log_section_scale: torch.Tensor,
        reference_segment_lengths: torch.Tensor,
        section_intervals: Iterable[int],
        base_position: torch.Tensor) -> torch.Tensor:
    """Deterministically integrate bending and positive lengths to positions."""
    if bend.ndim != 2:
        raise ValueError("bend 必须为 (B,N-1)")
    batch, n_segments = bend.shape
    intervals = tuple(int(v) for v in section_intervals)
    section_ids = _section_index(intervals, n_segments, device=bend.device)
    if log_section_scale.shape != (batch, len(intervals)):
        raise ValueError("log_section_scale 维度与 section_intervals 不匹配")
    ref = torch.as_tensor(
        reference_segment_lengths, dtype=bend.dtype,
        device=bend.device).reshape(-1)
    if ref.numel() != n_segments or torch.any(ref <= 0):
        raise ValueError("reference_segment_lengths 维度错误或包含非正值")
    base = torch.as_tensor(
        base_position, dtype=bend.dtype, device=bend.device)
    if base.ndim == 1:
        base = base.unsqueeze(0).expand(batch, -1)
    if base.shape != (batch, 3):
        raise ValueError("base_position 必须为 (3,) 或 (B,3)")

    angles = torch.cumsum(bend, dim=1)
    scale_per_segment = log_section_scale[:, section_ids]
    lengths = ref.unsqueeze(0) * torch.exp(scale_per_segment.clamp(-0.25, 0.25))
    increments = torch.stack(
        [lengths * torch.cos(angles), lengths * torch.sin(angles),
         torch.zeros_like(lengths)], dim=-1)
    relative = torch.cumsum(increments, dim=1)
    return torch.cat([base.unsqueeze(1), base.unsqueeze(1) + relative], dim=1)


def _ridge_reference(actions: np.ndarray, values: np.ndarray,
                     ridge: float):
    x = np.concatenate(
        [np.ones((len(actions), 1), dtype=np.float64),
         actions.astype(np.float64)], axis=1)
    reg = ridge * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ values.astype(np.float64))
    pred = x @ weights
    return weights[0], weights[1:], values - pred


def _monotone_spline_reference(
        actions: np.ndarray,
        values: np.ndarray,
        n_knots: int,
        steps: int,
        ridge: float,
        *,
        fit_objective: str,
        skeletons: torch.Tensor,
        base_positions: torch.Tensor,
        reference_segment_lengths: torch.Tensor,
        section_intervals: Iterable[int],
        geometry_weight: float,
        endpoint_weight: float):
    """Fit monotone drives using coordinate or reconstructed-geometry loss."""
    if n_knots < 2 or steps < 1:
        raise ValueError("monotone_spline 要求至少2个 knots 和正 fit_steps")
    if fit_objective not in {"coordinate", "geometry"}:
        raise ValueError(f"未知 H0 fit objective: {fit_objective!r}")
    if geometry_weight < 0 or endpoint_weight < 0:
        raise ValueError("H0 geometry/endpoint 权重必须非负")
    action_t = torch.as_tensor(actions, dtype=torch.float32)
    value_t = torch.as_tensor(values, dtype=torch.float32)
    knots = torch.linspace(0.0, 1.0, n_knots)
    linear_bias, linear_dirs, linear_residual = _ridge_reference(
        actions, values, ridge)
    # Preserve the v2 coordinate fit as a nested warm start. Geometry mode
    # first reaches that solution, then fine-tunes it with physical losses.
    raw_init = math.log(math.expm1(0.4))
    raw_weights_init = torch.full(
        (actions.shape[1], n_knots), raw_init, dtype=torch.float32)
    raw_weights = nn.Parameter(torch.full(
        (actions.shape[1], n_knots), 0.0, dtype=torch.float32))
    with torch.no_grad():
        raw_weights.copy_(raw_weights_init)
    bias = nn.Parameter(torch.as_tensor(linear_bias, dtype=torch.float32))
    directions = nn.Parameter(torch.as_tensor(linear_dirs, dtype=torch.float32))
    scale = value_t.std(dim=0).clamp_min(1e-3)

    n_segments = reference_segment_lengths.numel()
    linear_prediction = torch.as_tensor(
        values - linear_residual, dtype=torch.float32)
    with torch.no_grad():
        linear_skeleton = generalized_to_skeleton(
            linear_prediction[:, :n_segments],
            linear_prediction[:, n_segments:],
            reference_segment_lengths, section_intervals, base_positions)
        linear_xy_error = (
            linear_skeleton[..., :2] - skeletons[..., :2]).square().sum(-1)
        node_mse_scale = linear_xy_error.mean().clamp_min(1e-6)
        endpoint_mse_scale = linear_xy_error[:, -1].mean().clamp_min(1e-6)

    parameters = [raw_weights, bias, directions]
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    geometry_steps = max(int(steps) // 2, 1) if fit_objective == "geometry" else 0
    total_steps = int(steps) + geometry_steps
    best_score = float("inf")
    best_parameters = None
    for step_index in range(total_steps):
        geometry_active = (
            fit_objective == "geometry" and step_index >= int(steps))
        if geometry_active and step_index == int(steps):
            optimizer = torch.optim.Adam(parameters, lr=0.003)
        optimizer.zero_grad()
        weights = F.softplus(raw_weights)
        hinges = torch.relu(action_t.unsqueeze(-1) - knots)
        drive = (hinges * weights.unsqueeze(0)).sum(dim=-1)
        prediction = bias + drive @ directions
        loss = (((prediction - value_t) / scale) ** 2).mean()
        if geometry_active:
            predicted_skeleton = generalized_to_skeleton(
                prediction[:, :n_segments], prediction[:, n_segments:],
                reference_segment_lengths, section_intervals, base_positions)
            xy_error = (
                predicted_skeleton[..., :2] - skeletons[..., :2]
            ).square().sum(-1)
            loss = loss + geometry_weight * xy_error.mean() / node_mse_scale
            loss = loss + endpoint_weight * (
                xy_error[:, -1].mean() / endpoint_mse_scale)
        loss = loss + ridge * (directions.square().mean() + weights.square().mean())
        if geometry_active and float(loss.detach()) < best_score:
            best_score = float(loss.detach())
            best_parameters = [parameter.detach().clone()
                               for parameter in parameters]
        loss.backward()
        if geometry_active:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
        optimizer.step()
    if best_parameters is not None:
        with torch.no_grad():
            for parameter, best in zip(parameters, best_parameters):
                parameter.copy_(best)
    with torch.no_grad():
        weights = F.softplus(raw_weights)
        hinges = torch.relu(action_t.unsqueeze(-1) - knots)
        drive = (hinges * weights.unsqueeze(0)).sum(dim=-1)
        prediction = bias + drive @ directions
    return (
        bias.detach().cpu().numpy(), directions.detach().cpu().numpy(),
        (value_t - prediction).cpu().numpy(), knots.cpu().numpy(),
        weights.cpu().numpy(), float(F.mse_loss(prediction, value_t).item()))


def fit_ishsm_priors_from_arrays(
        actions: np.ndarray,
        skeletons: np.ndarray,
        n_bend_modes: int = 8,
        section_intervals: Iterable[int] = (7, 7),
        ridge: float = 1e-5,
        reference_kind: str = "linear",
        n_reference_knots: int = 5,
        reference_fit_steps: int = 500,
        reference_fit_objective: str = "coordinate",
        reference_geometry_weight: float = 1.0,
        reference_endpoint_weight: float = 0.25) -> dict:
    """Fit frozen H0 reference and POD basis using fit-split arrays only."""
    actions = np.asarray(actions, dtype=np.float32)
    skeletons = np.asarray(skeletons, dtype=np.float32)
    if actions.ndim != 2 or skeletons.ndim != 3:
        raise ValueError("actions/skeletons 必须为 (T,C)/(T,N,3)")
    if len(actions) != len(skeletons):
        raise ValueError("actions 与 skeletons 帧数不同")
    n_segments = skeletons.shape[1] - 1
    _section_index(section_intervals, n_segments)
    if not 1 <= n_bend_modes <= n_segments:
        raise ValueError("n_bend_modes 必须在 [1,N-1]")
    if reference_fit_objective not in {"coordinate", "geometry"}:
        raise ValueError(
            f"未知 H0 fit objective: {reference_fit_objective!r}")

    segment_lengths = np.linalg.norm(
        skeletons[:, 1:, :] - skeletons[:, :-1, :], axis=-1)
    reference_lengths = segment_lengths.mean(axis=0).astype(np.float32)
    bend_t, length_t, base_t = skeleton_to_generalized(
        torch.from_numpy(skeletons), torch.from_numpy(reference_lengths),
        section_intervals)
    bend = bend_t.numpy()
    section_length = length_t.numpy()

    values = np.concatenate([bend, section_length], axis=1)
    if reference_kind == "linear":
        bias, directions, residual = _ridge_reference(actions, values, ridge)
        reference_knots = np.empty((0,), dtype=np.float32)
        reference_drive_weights = np.empty(
            (actions.shape[1], 0), dtype=np.float32)
        prediction = values - residual
        reference_fit_mse = float(np.mean((prediction - values) ** 2))
    elif reference_kind == "monotone_spline":
        (bias, directions, residual, reference_knots,
         reference_drive_weights, reference_fit_mse) = \
            _monotone_spline_reference(
                actions, values, n_reference_knots,
                reference_fit_steps, ridge,
                fit_objective=reference_fit_objective,
                skeletons=torch.from_numpy(skeletons),
                base_positions=base_t,
                reference_segment_lengths=torch.from_numpy(reference_lengths),
                section_intervals=section_intervals,
                geometry_weight=reference_geometry_weight,
                endpoint_weight=reference_endpoint_weight)
    else:
        raise ValueError(f"未知 reference_kind: {reference_kind!r}")
    prediction = values - residual
    bend_bias = bias[:n_segments]
    length_bias = bias[n_segments:]
    bend_dirs = directions[:, :n_segments]
    length_dirs = directions[:, n_segments:]
    bend_residual = residual[:, :n_segments]

    prediction_t = torch.from_numpy(np.asarray(prediction, dtype=np.float32))
    with torch.no_grad():
        reference_skeleton = generalized_to_skeleton(
            prediction_t[:, :n_segments], prediction_t[:, n_segments:],
            torch.from_numpy(reference_lengths), section_intervals, base_t)
        xy_error = (
            reference_skeleton[..., :2] -
            torch.from_numpy(skeletons)[..., :2]).square().sum(-1)
        reference_fit_node_rmse_mm = float(torch.sqrt(xy_error.mean()).item())
        reference_fit_endpoint_rmse_mm = float(
            torch.sqrt(xy_error[:, -1].mean()).item())

    _, singular, vt = np.linalg.svd(
        bend_residual - bend_residual.mean(axis=0, keepdims=True),
        full_matrices=False)
    basis = vt[:n_bend_modes].T
    for mode in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0:
            basis[:, mode] *= -1
    energy = singular ** 2
    explained = float(energy[:n_bend_modes].sum() / max(energy.sum(), 1e-12))
    bend_scores = bend_residual @ basis
    length_residual = residual[:, n_segments:]
    generalized_scale = np.concatenate([
        np.maximum(bend_scores.std(axis=0), 0.02),
        np.maximum(length_residual.std(axis=0), 0.005),
    ]).astype(np.float32)
    return {
        "bend_basis": basis.astype(np.float32),
        "bend_explained_energy": explained,
        # Standard deviations in the fixed POD-bending / section-log-length
        # coordinates. HOV2.1 uses this immutable metric to normalize learned
        # readout directions without changing the ISHSM state update.
        "generalized_coordinate_scale": generalized_scale,
        "reference_segment_lengths": reference_lengths,
        "reference_bend_bias": np.asarray(bend_bias, dtype=np.float32),
        "reference_bend_dirs": np.asarray(bend_dirs, dtype=np.float32),
        "reference_length_bias": np.asarray(length_bias, dtype=np.float32),
        "reference_length_dirs": np.asarray(length_dirs, dtype=np.float32),
        "reference_kind": reference_kind,
        "reference_knots": np.asarray(reference_knots, dtype=np.float32),
        "reference_drive_weights": np.asarray(
            reference_drive_weights, dtype=np.float32),
        "reference_fit_mse": reference_fit_mse,
        "reference_fit_objective": reference_fit_objective,
        "reference_fit_node_rmse_mm": reference_fit_node_rmse_mm,
        "reference_fit_endpoint_rmse_mm": reference_fit_endpoint_rmse_mm,
        "base_position": base_t.mean(dim=0).numpy().astype(np.float32),
    }


class ISHSMModel(nn.Module):
    """H1 ISHSM: frozen H0 + observable 8+2 stable recurrent state."""

    training_spec = TrainingSpec(phases=[PhaseSpec(
        name="ishsm",
        dataset_type="state_transition",
        supervision_mode="spatial_sequence",
        active_losses=[
            "skeleton", "spatial_smooth", "bend", "length", "endpoint"],
        forward_attr="forward",
        use_episode_mode=True,
        teacher_forcing_ratio=1.0,
        episode_len=40,
    )])

    def __init__(
            self,
            action_dim: int = 4,
            n_nodes: int = 15,
            n_bend_modes: int = 8,
            section_intervals: Iterable[int] = (7, 7),
            dt: float = 0.1,
            tau_range=(0.3, 2.0),
            bend_basis=None,
            reference_segment_lengths=None,
            reference_bend_bias=None,
            reference_bend_dirs=None,
            reference_length_bias=None,
            reference_length_dirs=None,
            reference_kind: str = "linear",
            reference_knots=None,
            reference_drive_weights=None,
            base_position=None,
            use_dynamic_length: bool = True,
            use_persistent_state: bool = False,
            persistence_init: float = 0.1,
            observation_update: str = "hard",
            observation_gain_init: float = 0.25,
            training_reanchor_intervals=(0,),
            tau_parameterization: str = "independent",
            observation_projection: str = "modal",
            tip_dls_lambda_mm2: float = 1.0,
            episode_len: int = 40):
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_nodes = int(n_nodes)
        self.n_bend_modes = int(n_bend_modes)
        self.section_intervals = tuple(int(v) for v in section_intervals)
        self.n_sections = len(self.section_intervals)
        self.use_dynamic_length = bool(use_dynamic_length)
        self.n_length_states = self.n_sections if self.use_dynamic_length else 0
        self.transient_state_dim = self.n_bend_modes + self.n_length_states
        self.use_persistent_state = bool(use_persistent_state)
        self.n_persistent_states = (
            self.transient_state_dim if self.use_persistent_state else 0)
        self.z_dim = self.transient_state_dim + self.n_persistent_states
        self.reference_kind = str(reference_kind)
        self.observation_update = str(observation_update)
        self.observation_gain_init = float(observation_gain_init)
        self.training_reanchor_intervals = tuple(
            int(value) for value in training_reanchor_intervals)
        self.tau_parameterization = str(tau_parameterization)
        self.observation_projection = str(observation_projection)
        self.tip_dls_lambda_mm2 = float(tip_dls_lambda_mm2)
        self.episode_len = int(episode_len)
        self.window_size = 40
        self.node_order = "base_to_tip"
        self.spatial_propagation_direction = "global_spatial_basis"
        self.gl_kernel_alignment = "current_at_window_end"
        self.model_contract_version = (
            3 if self.observation_projection != "modal" else
            2 if (self.observation_update == "innovation" or
                  self.tau_parameterization != "independent") else 1)
        if self.n_nodes - 1 != sum(self.section_intervals):
            raise ValueError("section_intervals 与 n_nodes 不匹配")
        tau_min, tau_max = map(float, tau_range)
        if not (0 < tau_min <= tau_max):
            raise ValueError("tau_range 必须满足 0 < min <= max")
        if self.observation_update not in {"hard", "innovation"}:
            raise ValueError(
                f"未知 observation_update: {self.observation_update!r}")
        if not 0.0 < self.observation_gain_init < 1.0:
            raise ValueError("observation_gain_init 必须在 (0,1)")
        if (not self.training_reanchor_intervals or
                any(value < 0 for value in self.training_reanchor_intervals)):
            raise ValueError("training_reanchor_intervals 必须是非负整数序列")
        if self.tau_parameterization not in {
                "independent", "shared_bending"}:
            raise ValueError(
                f"未知 tau_parameterization: {self.tau_parameterization!r}")
        if self.observation_projection not in {"modal", "tip_dls"}:
            raise ValueError(
                f"未知 observation_projection: "
                f"{self.observation_projection!r}")
        if self.tip_dls_lambda_mm2 <= 0:
            raise ValueError("tip_dls_lambda_mm2 必须为正数")
        if self.observation_update == "innovation" and self.use_persistent_state:
            raise ValueError(
                "innovation observer 与旧 constant persistent state 不可同时启用")
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.register_buffer("dt", torch.tensor(float(dt)))

        n_bend = self.n_nodes - 1
        if bend_basis is None:
            grid = torch.arange(n_bend, dtype=torch.float32).unsqueeze(1)
            modes = torch.arange(self.n_bend_modes, dtype=torch.float32).unsqueeze(0)
            bend_basis = torch.cos(math.pi * (grid + 0.5) * modes / n_bend)
            bend_basis, _ = torch.linalg.qr(bend_basis, mode="reduced")
        defaults = {
            "bend_basis": (bend_basis, (n_bend, self.n_bend_modes)),
            "reference_segment_lengths": (
                torch.ones(n_bend) if reference_segment_lengths is None
                else reference_segment_lengths, (n_bend,)),
            "reference_bend_bias": (
                torch.zeros(n_bend) if reference_bend_bias is None
                else reference_bend_bias, (n_bend,)),
            "reference_bend_dirs": (
                torch.zeros(self.action_dim, n_bend) if reference_bend_dirs is None
                else reference_bend_dirs, (self.action_dim, n_bend)),
            "reference_length_bias": (
                torch.zeros(self.n_sections) if reference_length_bias is None
                else reference_length_bias, (self.n_sections,)),
            "reference_length_dirs": (
                torch.zeros(self.action_dim, self.n_sections)
                if reference_length_dirs is None else reference_length_dirs,
                (self.action_dim, self.n_sections)),
            "base_position": (
                torch.zeros(3) if base_position is None else base_position, (3,)),
        }
        for name, (value, shape) in defaults.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} 应为 {shape}，得到 {tuple(tensor.shape)}")
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

        self.register_buffer("pc_center", torch.zeros(1, 1, 3))
        self.register_buffer("pc_scale", torch.ones(1, 1, 3))
        self.register_buffer("action_norm_factor", torch.tensor(1.0))

        n_tau_parameters = (
            self.transient_state_dim
            if self.tau_parameterization == "independent" else
            1 + self.n_length_states)
        init_taus = torch.logspace(
            math.log10(tau_min), math.log10(tau_max), n_tau_parameters)
        fraction = ((torch.log(init_taus) - math.log(tau_min)) /
                    max(math.log(tau_max) - math.log(tau_min), 1e-8))
        fraction = fraction.clamp(1e-4, 1 - 1e-4)
        self.raw_taus = nn.Parameter(torch.logit(fraction))
        self.excitation = nn.Parameter(torch.zeros(
            self.transient_state_dim, self.action_dim))
        if self.observation_update == "innovation":
            gain_groups = 1 + int(self.n_length_states > 0)
            gain_logit = torch.logit(torch.tensor(self.observation_gain_init))
            self.raw_observation_gains = nn.Parameter(torch.full(
                (gain_groups,), float(gain_logit)))
        if self.use_persistent_state:
            if not 0.0 < persistence_init < 1.0:
                raise ValueError("persistence_init 必须在 (0,1)")
            self.raw_persistence = nn.Parameter(torch.full(
                (self.transient_state_dim,),
                torch.logit(torch.tensor(float(persistence_init))).item()))
        self.training_spec.phases[0].episode_len = self.episode_len

    @property
    def taus(self) -> torch.Tensor:
        if self.tau_min == self.tau_max:
            parameter_taus = torch.full_like(self.raw_taus, self.tau_min)
        else:
            log_tau = (math.log(self.tau_min) + torch.sigmoid(self.raw_taus) *
                       (math.log(self.tau_max) - math.log(self.tau_min)))
            parameter_taus = torch.exp(log_tau)
        if self.tau_parameterization == "independent":
            return parameter_taus
        return torch.cat([
            parameter_taus[:1].expand(self.n_bend_modes),
            parameter_taus[1:],
        ])

    @property
    def decays(self) -> torch.Tensor:
        return torch.exp(-self.dt / self.taus)

    def _reference(self, action: torch.Tensor):
        if self.reference_kind == "monotone_spline":
            hinges = torch.relu(
                action.unsqueeze(-1) - self.reference_knots)
            action = (hinges * self.reference_drive_weights).sum(dim=-1)
        bend = self.reference_bend_bias + torch.einsum(
            "bc,cn->bn", action, self.reference_bend_dirs)
        length = self.reference_length_bias + torch.einsum(
            "bc,cs->bs", action, self.reference_length_dirs)
        return bend, length

    def _to_physical(self, skeleton: torch.Tensor) -> torch.Tensor:
        return skeleton * self.pc_scale.to(skeleton) + self.pc_center.to(skeleton)

    def _to_normalized(self, skeleton: torch.Tensor) -> torch.Tensor:
        return ((skeleton - self.pc_center.to(skeleton)) /
                self.pc_scale.to(skeleton))

    def init_z_from_action(self, action_window: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            action_window.shape[0], self.z_dim,
            dtype=action_window.dtype, device=action_window.device)

    @property
    def observation_gains(self) -> torch.Tensor:
        """Return per-state gains with only bend/length group parameters."""
        if self.observation_update == "hard":
            return torch.ones(
                self.transient_state_dim, dtype=self.dt.dtype,
                device=self.dt.device)
        grouped = torch.sigmoid(self.raw_observation_gains)
        values = [grouped[:1].expand(self.n_bend_modes)]
        if self.n_length_states:
            values.append(grouped[1:2].expand(self.n_length_states))
        return torch.cat(values)

    @property
    def persistence_fraction(self) -> torch.Tensor:
        if not self.use_persistent_state:
            return torch.zeros(
                self.transient_state_dim, dtype=self.dt.dtype,
                device=self.dt.device)
        return torch.sigmoid(self.raw_persistence)

    def pack_memory_state(self, transient: torch.Tensor,
                          persistent: torch.Tensor | None = None) -> torch.Tensor:
        if not self.use_persistent_state:
            return transient
        if persistent is None or persistent.shape != transient.shape:
            raise ValueError("persistent 与 transient 状态必须同形")
        return torch.cat([transient, persistent], dim=-1)

    def unpack_memory_state(self, state: torch.Tensor):
        if state.shape[-1] != self.z_dim:
            raise ValueError(
                f"memory state 最后一维必须为 {self.z_dim}")
        if not self.use_persistent_state:
            return state, torch.zeros_like(state)
        return state.split(self.transient_state_dim, dim=-1)

    def _state_from_observation(self, action_window: torch.Tensor,
                                observed_skeleton: torch.Tensor) -> torch.Tensor:
        """Project a causal t-1 observation into the 8+2 coordinates."""
        physical = self._to_physical(observed_skeleton)
        bend_obs, length_obs, _ = skeleton_to_generalized(
            physical, self.reference_segment_lengths, self.section_intervals)
        index = -2 if action_window.shape[1] >= 2 else -1
        bend_ref, length_ref = self._reference(action_window[:, index, :])
        z_bend = (bend_obs - bend_ref) @ self.bend_basis
        residual = z_bend
        if self.use_dynamic_length:
            z_length = length_obs - length_ref
            residual = torch.cat([z_bend, z_length], dim=1)
        if self.observation_projection == "tip_dls":
            residual = self._tip_consistent_projection(
                action_window[:, index, :], residual, physical)
        return residual

    def _tip_jacobian(self, current_action: torch.Tensor,
                      transient_state: torch.Tensor) -> torch.Tensor:
        """Analytic planar tip Jacobian w.r.t. the observable 8+2 state.

        The Jacobian is expressed in physical millimetres per generalized
        coordinate.  Bending columns follow cumulative turning-angle
        integration; length columns follow the two section log scales.
        """
        bend_ref, length_ref = self._reference(current_action)
        bend = (bend_ref + transient_state[:, :self.n_bend_modes] @
                self.bend_basis.T)
        angles = torch.cumsum(bend, dim=1)
        cumulative_basis = torch.cumsum(self.bend_basis, dim=0)

        if self.use_dynamic_length:
            log_scales = (length_ref +
                          transient_state[:, self.n_bend_modes:])
        else:
            log_scales = length_ref
        section_ids = _section_index(
            self.section_intervals, bend.shape[1], device=bend.device)
        clamped_scales = log_scales.clamp(-0.25, 0.25)
        lengths = self.reference_segment_lengths.unsqueeze(0) * torch.exp(
            clamped_scales[:, section_ids])

        dx_dtheta = -lengths * torch.sin(angles)
        dy_dtheta = lengths * torch.cos(angles)
        jacobian_bend = torch.stack([
            dx_dtheta @ cumulative_basis,
            dy_dtheta @ cumulative_basis,
        ], dim=1)
        if not self.use_dynamic_length:
            return jacobian_bend

        active = ((log_scales > -0.25) & (log_scales < 0.25)).to(lengths)
        x_increment = lengths * torch.cos(angles)
        y_increment = lengths * torch.sin(angles)
        length_columns = []
        for section in range(self.n_sections):
            mask = section_ids == section
            length_columns.append(torch.stack([
                x_increment[:, mask].sum(dim=1) * active[:, section],
                y_increment[:, mask].sum(dim=1) * active[:, section],
            ], dim=1))
        jacobian_length = torch.stack(length_columns, dim=2)
        return torch.cat([jacobian_bend, jacobian_length], dim=2)

    def _tip_consistent_projection(
            self, current_action: torch.Tensor,
            modal_state: torch.Tensor,
            observed_physical: torch.Tensor) -> torch.Tensor:
        """Apply one minimum-norm DLS correction to the modal observation.

        This is a deterministic observation operator, not an extra learned
        residual.  Only planar tip coordinates are corrected because the
        current ISHSM geometry is explicitly planar.
        """
        predicted = self._to_physical(
            self.decode_state(current_action, modal_state))
        tip_error = observed_physical[:, -1, :2] - predicted[:, -1, :2]
        jacobian = self._tip_jacobian(current_action, modal_state)
        identity = torch.eye(
            2, dtype=jacobian.dtype, device=jacobian.device).unsqueeze(0)
        normal = jacobian @ jacobian.transpose(1, 2)
        solved = torch.linalg.solve(
            normal + self.tip_dls_lambda_mm2 * identity,
            tip_error.unsqueeze(-1))
        correction = jacobian.transpose(1, 2) @ solved
        return modal_state + correction.squeeze(-1)

    def assimilate_observation(
            self, action_window: torch.Tensor,
            observed_skeleton: torch.Tensor,
            predicted_state: torch.Tensor | None = None) -> torch.Tensor:
        """Correct a predicted state with a bounded causal innovation."""
        residual = self._state_from_observation(
            action_window, observed_skeleton)
        if self.observation_update == "innovation":
            if predicted_state is None:
                predicted_state = torch.zeros_like(residual)
            if predicted_state.shape != residual.shape:
                raise ValueError(
                    "innovation predicted_state 必须与观测状态同形")
            gains = self.observation_gains.to(residual).unsqueeze(0)
            return predicted_state + gains * (residual - predicted_state)
        if not self.use_persistent_state:
            return residual
        fraction = self.persistence_fraction.unsqueeze(0)
        return self.pack_memory_state(
            (1.0 - fraction) * residual, fraction * residual)

    def init_rollout_state(self, action_window: torch.Tensor,
                           observed_skeleton: torch.Tensor) -> torch.Tensor:
        """Initialize from the one skeleton immediately before prediction."""
        return self.assimilate_observation(
            action_window, observed_skeleton, predicted_state=None)

    init_z_from_observation = init_rollout_state

    def forward(self, batch_or_action_window, prev_skeleton=None,
                prev_prev_skeleton=None, prev_z=None):
        if isinstance(batch_or_action_window, dict):
            action_window = batch_or_action_window["action_window"]
        else:
            action_window = batch_or_action_window
        current = action_window[:, -1, :]
        previous = action_window[:, -2, :] if action_window.shape[1] >= 2 else current
        if prev_z is None:
            prev_z = self.init_z_from_action(action_window)
        transient, persistent = self.unpack_memory_state(prev_z)
        transient = self.decays.unsqueeze(0) * transient + \
            (current - previous) @ self.excitation.T
        state = self.pack_memory_state(transient, persistent)
        skeleton = self.decode_state(current, state)
        return {"skeleton": skeleton, "latent_z": state}

    def decode_state(self, current_action: torch.Tensor,
                     state: torch.Tensor) -> torch.Tensor:
        """Decode an explicit modal state without changing or advancing it."""
        if state.shape[-1] != self.z_dim:
            raise ValueError(
                f"state 最后一维必须为 {self.z_dim}，得到 {state.shape[-1]}")

        transient, persistent = self.unpack_memory_state(state)
        effective = transient + persistent
        bend_ref, length_ref = self._reference(current_action)
        bend = bend_ref + effective[:, :self.n_bend_modes] @ self.bend_basis.T
        length = length_ref
        if self.use_dynamic_length:
            length = length + effective[:, self.n_bend_modes:]
        physical = generalized_to_skeleton(
            bend, length, self.reference_segment_lengths,
            self.section_intervals, self.base_position)
        return self._to_normalized(physical)

    def compute_losses(self, batch: dict, phase_spec) -> dict:
        device = self.dt.device
        action_window = batch["action_window"].to(device)
        gt = batch["gt_skeleton"].to(device)
        pred = self.forward(action_window)["skeleton"]
        losses = {"skeleton": F.mse_loss(pred, gt)}
        if "spatial_smooth" in phase_spec.active_losses:
            losses["spatial_smooth"] = F.mse_loss(
                pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1])
        losses.update(self.compute_sequence_aux_losses(
            pred.unsqueeze(1), gt.unsqueeze(1)))
        return losses

    def compute_sequence_aux_losses(self, pred_seq: torch.Tensor,
                                    gt_seq: torch.Tensor) -> dict:
        batch, steps, nodes, dims = pred_seq.shape
        pred = self._to_physical(pred_seq.reshape(batch * steps, nodes, dims))
        gt = self._to_physical(gt_seq.reshape(batch * steps, nodes, dims))
        pred_b, pred_l, _ = skeleton_to_generalized(
            pred, self.reference_segment_lengths, self.section_intervals)
        gt_b, gt_l, _ = skeleton_to_generalized(
            gt, self.reference_segment_lengths, self.section_intervals)
        bend_scale = gt_b.detach().std(dim=0).clamp_min(0.02)
        length_scale = gt_l.detach().std(dim=0).clamp_min(0.005)
        return {
            "bend": F.mse_loss(pred_b / bend_scale, gt_b / bend_scale),
            "length": F.mse_loss(pred_l / length_scale, gt_l / length_scale),
            # Normalized-coordinate endpoint guardrail.  Its CLI weight is
            # deliberately small because the endpoint already contributes to
            # the all-node skeleton loss.
            "endpoint": F.mse_loss(
                pred_seq[:, :, -1, :], gt_seq[:, :, -1, :]),
        }

    def set_normalization(self, center, scale, action_norm_factor=1.0):
        center = torch.as_tensor(center, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        self.pc_center = center.reshape(1, 1, 3)
        self.pc_scale = scale.reshape(1, 1, 3)
        self.action_norm_factor = torch.tensor(float(action_norm_factor))

    @torch.no_grad()
    def predict_skeleton(self, action_window, prev_skeleton=None, prev_z=None,
                         prev_prev_skeleton=None):
        device = self.dt.device
        action_window = action_window.to(device)
        if self.action_norm_factor.item() > 1.01:
            action_window = action_window / self.action_norm_factor
        out = self.forward(action_window, prev_z=prev_z)
        return self._to_physical(out["skeleton"])

    def state_report(self) -> dict:
        return {
            "taus": self.taus.detach().cpu(),
            "decays": self.decays.detach().cpu(),
            "excitation": self.excitation.detach().cpu(),
            "bend_basis": self.bend_basis.detach().cpu(),
            "n_bend_modes": self.n_bend_modes,
            "n_length_states": self.n_length_states,
            "n_persistent_states": self.n_persistent_states,
            "persistence_fraction": self.persistence_fraction.detach().cpu(),
            "observation_update": self.observation_update,
            "observation_gains": self.observation_gains.detach().cpu(),
            "training_reanchor_intervals": self.training_reanchor_intervals,
            "tau_parameterization": self.tau_parameterization,
            "observation_projection": self.observation_projection,
            "tip_dls_lambda_mm2": self.tip_dls_lambda_mm2,
            "tau_parameters": self.raw_taus.detach().cpu(),
            "reference_kind": self.reference_kind,
        }
