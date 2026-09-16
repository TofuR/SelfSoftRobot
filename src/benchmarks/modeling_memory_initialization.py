"""Closed-form initialization of geometry memory readouts from training windows.

Integration is opt-in: call ``initialize_memory_readout(model, x_train,
y_train_physical)`` after constructing training windows and before normalizing
targets or creating the optimizer. The caller must supply training data only;
split provenance cannot be inferred from tensors. This module performs no loss
optimization and changes only the active PI/Maxwell readout parameters.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch

from src.benchmarks.modeling_models import AblatedGeometry, GeometryWindow
from src.models.model_ishsm import skeleton_to_generalized

__all__ = ["initialize_memory_readout"]


def initialize_memory_readout(model, action_windows, physical_targets, ridge=1e-3,
                              batch_size=512):
    """Fit active readouts using independent training windows and physical targets.

    Supports GeometryWindow wrapping AblatedGeometry, including its
    CalibratedGeometry subclass. Inputs have shapes (N,H,action_dim) and
    (N,n_nodes,3). Operator state is recomputed through the ordinary core forward.
    Features are q=e-p and deficit=e-h; the existing core reads h-e, so Maxwell
    coefficients are negated when written back. Bend residuals are projected
    using the pseudoinverse of bend_basis; outputs use the core's fixed scales.

    Float64 sufficient statistics are accumulated on CPU. RMS scaling does not
    center features or add an intercept. Ridge solves (X.T X/N + ridge I) W =
    X.T Y/N in scaled feature coordinates. ridge=0 uses a minimum-norm solution.
    An existing neural memory residual is held fixed and subtracted from Y.
    Zero coefficient rows use a unit fallback direction and the smallest normal
    positive gain representable by the parameter dtype. Zero-memory contribution
    stays exactly zero. Disabled readouts, dynamics, reference parameters, module
    training flags, requires_grad flags and existing gradients are preserved.

    Returns JSON-serializable initialization metadata. Fully disabled memory and
    static windows return a skipped result without reading either input.
    """
    if not isinstance(model, GeometryWindow) or not isinstance(model.core, AblatedGeometry):
        raise TypeError("Expected GeometryWindow with an AblatedGeometry/CalibratedGeometry core")
    if isinstance(ridge, bool) or not math.isfinite(float(ridge)) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    core = model.core
    branches = []
    if not core.disable_play:
        branches.append(("play", core.pi_mode_directions_raw, core.play.raw_weights, 1.))
    if not core.disable_maxwell:
        branches.append(("maxwell", core.maxwell_mode_directions_raw, core.maxwell_gain_raw, -1.))
    metadata = dict(schema="modeling_memory_initialization_v1", ridge=float(ridge),
                    batch_size=int(batch_size), active_branches=[b[0] for b in branches],
                    data_role="train (caller supplied)", centered=False, intercept=False,
                    features="q=e-p; deficit=e-h", core_maxwell_convention="h-e",
                    solver="float64 CPU normalized normal equations",
                    modified_parameters=[], status="skipped")
    if model.static or not branches:
        return dict(metadata, reason="static_window" if model.static else "all_branches_disabled")

    windows = torch.as_tensor(action_windows)
    targets = torch.as_tensor(physical_targets)
    if (windows.ndim != 3 or windows.shape[0] < 1 or windows.shape[1] < 1 or
            windows.shape[2] != core.action_dim or
            targets.shape != (windows.shape[0], core.n_nodes, 3) or
            windows.is_complex() or targets.is_complex()):
        raise ValueError("Expected nonempty (N,H,action_dim) windows and physical (N,n_nodes,3) targets")
    n = len(windows)
    dimension = sum(gain.numel() for _, _, gain, _ in branches)
    gram = torch.zeros(dimension, dimension, dtype=torch.float64)
    cross = torch.zeros(dimension, core.generalized_dim, dtype=torch.float64)
    target_square = torch.zeros((), dtype=torch.float64)
    parameter = branches[0][1]
    device, dtype = parameter.device, parameter.dtype
    training_flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            basis_inverse = torch.linalg.pinv(core.bend_basis.detach().to("cpu", torch.float64))
            scale = core.generalized_coordinate_scale.detach().to("cpu", torch.float64)
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("generalized_coordinate_scale must be finite and positive")
            for start in range(0, n, batch_size):
                actions = windows[start:start + batch_size].detach().to(device=device, dtype=dtype)
                physical = targets[start:start + batch_size].detach().to(device="cpu", dtype=torch.float64)
                if not torch.isfinite(actions).all() or not torch.isfinite(physical).all():
                    raise ValueError("Training arrays must contain only finite values")
                output = core(actions)
                p, h = core._unpack_state(output["latent_z"])
                drive = core.drive(actions[:, -1]).unsqueeze(-1)
                q, deficit = drive - p, drive - h
                features = {"play": q, "maxwell": deficit}
                x = torch.cat([features[name].flatten(1) for name, *_ in branches], dim=1)
                x = x.to("cpu", torch.float64)
                bend, loglength, _ = skeleton_to_generalized(
                    physical, core.reference_segment_lengths, core.section_intervals)
                reference_bend, reference_length = core._reference(actions[:, -1])
                bend_residual = (bend - reference_bend.to("cpu", torch.float64)) @ basis_inverse.T
                y = torch.cat((bend_residual, loglength - reference_length.to("cpu", torch.float64)), dim=1)
                # The nonlinear residual remains unchanged; fit the remaining readout.
                y = (y - output["memory_residual_generalized"].to("cpu", torch.float64)) / scale
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError("Nonfinite memory features or generalized targets")
                gram.add_(x.T @ x)
                cross.add_(x.T @ y)
                target_square.add_(y.square().sum())

            gram.div_(n)
            cross.div_(n)
            rms = gram.diagonal().clamp_min(0).sqrt()
            divisor = torch.where(rms > 0, rms, torch.ones_like(rms))
            scaled_gram = gram / divisor[:, None] / divisor[None, :]
            scaled_cross = cross / divisor[:, None]
            if ridge:
                scaled_weights = torch.linalg.solve(
                    scaled_gram + float(ridge) * torch.eye(dimension, dtype=torch.float64), scaled_cross)
            else:
                scaled_weights = torch.linalg.pinv(scaled_gram, hermitian=True) @ scaled_cross
            weights = scaled_weights / divisor[:, None]
            weights[rms == 0] = 0
            if not torch.isfinite(weights).all():
                raise ValueError("Readout solve produced nonfinite coefficients")

            # Stage every update before writing, so an invalid fit cannot partially initialize.
            updates, offset = [], 0
            for name, directions, raw_gain, sign in branches:
                count = raw_gain.numel()
                coefficient = sign * weights[offset:offset + count]
                gain = torch.linalg.vector_norm(coefficient, dim=1)
                unit = coefficient / torch.where(gain > 0, gain, torch.ones_like(gain))[:, None]
                unit[gain == 0, 0] = 1.
                positive_gain = gain.clamp_min(torch.finfo(raw_gain.dtype).tiny)
                inverse_softplus = positive_gain + torch.log(-torch.expm1(-positive_gain))
                for destination, value in ((directions, unit), (raw_gain, inverse_softplus)):
                    value = value.reshape(destination.shape).to(destination)
                    if not torch.isfinite(value).all():
                        raise ValueError("Readout coefficients exceed parameter dtype range")
                    updates.append((destination, value))
                offset += count
                metadata["modified_parameters"].extend(
                    ["core.pi_mode_directions_raw", "core.play.raw_weights"] if name == "play" else
                    ["core.maxwell_mode_directions_raw", "core.maxwell_gain_raw"])
            for destination, value in updates:
                destination.copy_(value)
            mse = (target_square / n - 2 * (weights * cross).sum() +
                   (weights * (gram @ weights)).sum()).clamp_min(0) / core.generalized_dim
            metadata.update(status="initialized", n_windows=n, n_features=dimension,
                            n_outputs=core.generalized_dim, feature_rms=rms.tolist(),
                            unexcited_features=int((rms == 0).sum()),
                            zero_coefficient_rows=int((weights.norm(dim=1) == 0).sum()),
                            training_generalized_rmse=float(mse.sqrt()),
                            fixed_neural_residual_subtracted=core.residual_mode != "none",
                            bend_projection="pseudoinverse of bend_basis")
    finally:
        for module, training in training_flags:
            module.training = training
    return metadata
