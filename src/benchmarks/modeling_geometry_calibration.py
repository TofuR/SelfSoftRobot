"""Calibrate a fitted HOV reference through its observable geometry coefficients.

The default adds 72 trainable scalars for four actions and fifteen nodes:
14 bend biases, 4 x 14 bend directions, and two section log-length biases.
The fitted spline drives, spatial basis, base, and reference segment lengths
remain buffers. PI/Maxwell dynamics and deterministic decoding are inherited.

Integration in ``modeling_models.make_model`` (use a function-local import,
since this module inherits the benchmark's ``AblatedGeometry``)::

    from src.benchmarks.modeling_geometry_calibration import make_calibrated_geometry
    return make_calibrated_geometry(geometry_config, normalization, static=static)

The returned configuration can reconstruct the model before loading its state
dict. ``static=True`` fits the same reference coefficients with both memory
branches disabled, retaining trainable calibration parameters.
"""
from __future__ import annotations

import torch
from torch import nn

from src.benchmarks.modeling_models import AblatedGeometry, GeometryWindow


class CalibratedGeometry(AblatedGeometry):
    """An explicit hereditary model with a trainable fitted static reference.

    Reference biases and directions are signed physical coefficients, not raw
    softplus parameters. ``reference_drive_weights`` already contains positive
    spline weights and stays frozen, preserving its monotone action map. The
    inherited operator drive and gains retain their softplus parameterizations.
    Section lengths retain the inherited positive, bounded exponential decode.

    ``calibrate_length_directions`` optionally opens the section length slopes
    (eight additional scalars in the benchmark); by default only their biases
    are calibrated. Every promoted coefficient starts at its fitted value.

    ``reference_pair_interactions`` adds signed, zero-initialized coefficients
    for each raw normalized pressure product u_i*u_j, i<j, to local bends.
    ``reference_pair_length_interactions`` also adds section log-length terms.
    These describe static channel coupling and relax the strictly additive
    single-channel reference assumption. They do not depend on memory or change
    the monotonicity of the individual spline drives.
    """

    def __init__(self, *args, calibrate_length_directions=False,
                 reference_pair_interactions=False,
                 reference_pair_length_interactions=False,
                 residual_mode="none", **kwargs):
        if residual_mode != "none":
            raise ValueError("Geometry calibration requires residual_mode='none'")
        if reference_pair_length_interactions and not reference_pair_interactions:
            raise ValueError("Pair length terms require reference_pair_interactions=True")
        super().__init__(*args, residual_mode=residual_mode, **kwargs)
        self.calibrate_length_directions = bool(calibrate_length_directions)
        names = ["reference_bend_bias", "reference_bend_dirs",
                 "reference_length_bias"]
        if self.calibrate_length_directions:
            names.append("reference_length_dirs")
        self.calibration_parameter_names = tuple(names)
        for name in self.calibration_parameter_names:
            value = getattr(self, name).detach().clone()
            delattr(self, name)
            self.register_parameter(name, nn.Parameter(value))

        self.reference_pair_interactions = bool(reference_pair_interactions)
        self.reference_pair_length_interactions = bool(reference_pair_length_interactions)
        if self.reference_pair_interactions:
            if self.action_dim < 2:
                raise ValueError("Pair interactions require at least two action channels")
            pairs = torch.triu_indices(self.action_dim, self.action_dim, offset=1,
                                       device=self.reference_bend_bias.device)
            self.register_buffer("reference_pair_indices", pairs)
            self.reference_pair_coefficients = nn.Parameter(
                self.reference_bend_bias.new_zeros(pairs.shape[1], self.n_nodes - 1))
            names.append("reference_pair_coefficients")
            if self.reference_pair_length_interactions:
                self.reference_pair_length_coefficients = nn.Parameter(
                    self.reference_length_bias.new_zeros(pairs.shape[1], self.n_sections))
                names.append("reference_pair_length_coefficients")
            self.calibration_parameter_names = tuple(names)

        # Both branch readouts are zero in the static ablation. Only reference
        # coefficients need optimization; its operator drive has no output path.
        if self.disable_play and self.disable_maxwell:
            self.drive.requires_grad_(False)

    def _reference(self, action):
        bend, length = super()._reference(action)
        if self.reference_pair_interactions:
            i, j = self.reference_pair_indices
            products = action[:, i] * action[:, j]
            bend = bend + products @ self.reference_pair_coefficients
            if self.reference_pair_length_interactions:
                length = length + products @ self.reference_pair_length_coefficients
        return bend, length

    def geometry_report(self):
        report = super().geometry_report()
        report.update(
            calibration_parameter_names=list(self.calibration_parameter_names),
            calibration_parameter_count=sum(
                getattr(self, name).numel()
                for name in self.calibration_parameter_names),
        )
        if self.reference_pair_interactions:
            report.update(reference_pair_indices=self.reference_pair_indices.T.tolist(),
                          reference_pair_length_interactions=self.reference_pair_length_interactions,
                          reference_structure="single_channel_terms_plus_static_pressure_pairs")
        return report


def make_calibrated_geometry(geometry_config, normalization, *, static=False):
    """Return ``(window_model, reconstruction_config)`` for benchmark integration.

    ``geometry_config`` accepts the existing ``AblatedGeometry`` kwargs plus
    ``calibrate_length_directions``, ``reference_pair_interactions`` and
    ``reference_pair_length_interactions``. Branch flags are honored in all variants.
    ``normalization`` is the existing train-only ``(center_xyz, scalar_scale)``.
    The caller's configuration is copied, so it can also build frozen controls.
    """
    config = dict(geometry_config)
    config.setdefault("residual_mode", "none")
    config.setdefault("calibrate_length_directions", False)
    config.setdefault("reference_pair_interactions", False)
    config.setdefault("reference_pair_length_interactions", False)
    if static:
        config.update(disable_play=True, disable_maxwell=True)
    core = CalibratedGeometry(**config)
    center, scale = normalization
    core.set_normalization(
        torch.as_tensor(center, dtype=torch.float32),
        torch.full((3,), float(scale), dtype=torch.float32), 1.0)
    # GeometryWindow(static=True) freezes all parameters, including reference
    # calibration. Disabled branches instead make this ordinary window path
    # history-independent while allowing the static reference to be optimized.
    return GeometryWindow(core), config
