"""Trainable memory injected inside an MLP or the existing Koopman readout.

Runner interface::

    normalization = fit_plugin_normalization(train_windows)  # (N, 20, 4)
    torch.manual_seed(seed)
    model = InternalMemoryModel("mlp", "both", normalization)
    prediction = model(action_windows)  # (B, 15, 3), normalized targets
    count = model.parameter_count()

Normalization contains JSON lists ``input_mean/std`` (4) and dictionaries
``memory_mean/std[variant]`` (0/8/24/32/32). Statistics are population moments
of training-window features under the initial drive, fixed during optimization.
Output-coordinate normalization belongs to the runner, not this module.

Features are channel-major: path has two q=e-p entries per channel, time six
d=h-e entries per channel, both concatenates path then time, and static_capacity
has phi(current)**1..8 per channel. Every non-base encoder has exactly the same
20 trainable spline coefficients; thresholds and time constants are buffers.
The path/time states are initialized separately in each causal window at e0.

MLP: h1=tanh(L1(standard_current)), h2=tanh(L2(h1)+D*m), y=readout(h2).
Koopman: exactly KoopmanShape(128,16)(raw_actions) plus D*m in 45 normalized
output coordinates. The whole base is initialized before any plugin modules.
D is bias-free and zero initially, preserving each same-seed base prediction.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn

from src.benchmarks.modeling_foundations import KoopmanShape
from src.operators.static_drive import MonotoneSplineDrive

FEATURE_DIMS = {"base": 0, "path": 8, "time": 24, "both": 32, "static_capacity": 32}
VARIANTS = tuple(FEATURE_DIMS)
FAMILIES = ("mlp", "koopman")
STD_FLOOR = 1e-6


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    """Count stored trainable parameters (including all 4x5 drive coefficients)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


count_parameters = parameter_count


def _validate_variant(variant):
    if variant not in FEATURE_DIMS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")


def _validate_windows(actions, history):
    if not isinstance(actions, torch.Tensor) or not actions.is_floating_point():
        raise ValueError("actions must be a floating-point Tensor of shape (B, history, 4)")
    if actions.ndim != 3 or actions.shape[1:] != (history, 4):
        raise ValueError(f"actions must have shape (B, {history}, 4), got {tuple(actions.shape)}")


class MemoryEncoder(nn.Module):
    """Unstandardized causal features, shape (B, FEATURE_DIMS[variant]).

    ``drive`` is a trainable MonotoneSplineDrive with unit_range output for every
    non-base variant. ``thresholds`` = (.02,.5), ``taus`` = six log-spaced values
    .6..2 seconds, and ``alpha`` = exp(-dt/taus) are fixed buffers. ``output_dim``
    and ``feature_dim`` expose the interface width. Base has no drive parameters.
    Constant windows yield exactly zero path/time features; static powers need
    not vanish. A constant input after a nonconstant history can retain path state.
    """

    def __init__(self, variant, history=20, dt=.2):
        super().__init__()
        _validate_variant(variant)
        if not isinstance(history, int) or isinstance(history, bool) or history < 1:
            raise ValueError("history must be a positive integer")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self.variant = variant
        self.history = history
        self.dt = float(dt)
        self.n_channels, self.n_play, self.n_maxwell = 4, 2, 6
        self.output_dim = self.feature_dim = FEATURE_DIMS[variant]
        self.drive = (None if variant == "base" else MonotoneSplineDrive(
            4, n_knots=5, output_normalization="unit_range"))
        # Match the original banks' float32 grids without their unused weights.
        thresholds = torch.linspace(math.log(.02), math.log(.5), 2).exp()
        taus = torch.linspace(math.log(.6), math.log(2.), 6).exp()
        self.register_buffer("thresholds", thresholds)
        self.register_buffer("taus", taus)
        self.register_buffer("alpha", torch.exp(-self.dt / taus))
        self.register_buffer("lag_exponents", torch.arange(history-1, 0, -1))
        self.register_buffer("static_exponents", torch.arange(1, 9))

    def forward(self, actions):
        _validate_windows(actions, self.history)
        if self.variant == "base":
            return actions.new_empty((len(actions), 0))
        if self.variant == "static_capacity":
            e = self.drive(actions[:, -1])
            return e.unsqueeze(-1).pow(self.static_exponents).flatten(1)
        e = self.drive(actions)
        features = []
        if self.variant in ("path", "both"):
            p = e[:, 0, :, None].expand(-1, -1, self.n_play)
            for t in range(1, self.history):
                level = e[:, t, :, None]
                p = torch.clamp(p, level-self.thresholds, level+self.thresholds)
            features.append((e[:, -1, :, None]-p).flatten(1))
        if self.variant in ("time", "both"):
            delta_e = e[:, 1:]-e[:, :-1]
            # Most recent increment has exponent one; d0=0 under h0=e0.
            # Recompute powers in the current dtype to preserve double gradients.
            powers = self.alpha[None, :].pow(self.lag_exponents[:, None])
            deficit = -torch.einsum("btc,tk->bck", delta_e, powers)
            features.append(deficit.flatten(1))
        return features[0] if len(features) == 1 else torch.cat(features, dim=1)


def fit_plugin_normalization(train_actions, *, history=20, dt=.2):
    """Fit JSON-serializable fixed statistics from training actions alone.

    Primary input: (N,history,4) already constructed split-local training windows
    as a Tensor or array. A (T,4) array denotes ONE training sequence and forms
    stride-one windows locally. Pass pooled sequence windows in 3D to preserve
    boundaries. Current-input moments use scored window endpoints, not context
    frames. Standard deviations use ddof=0 and a 1e-6 floor. No labels or random
    draws are used, and this function does not alter the global torch RNG state.
    """
    encoder = MemoryEncoder("both", history=history, dt=dt)
    actions = torch.as_tensor(train_actions, dtype=torch.float32, device="cpu").detach()
    if actions.ndim == 2:
        if actions.shape[1] != 4 or len(actions) < history:
            raise ValueError("a training sequence must have shape (T>=history, 4)")
        actions = actions.unfold(0, history, 1).transpose(1, 2)
    _validate_windows(actions, history)
    if not len(actions) or not torch.isfinite(actions).all():
        raise ValueError("training windows must be nonempty and finite")

    def moments(values):
        if values.shape[1] == 0:
            return [], []
        values = values.double()
        return (values.mean(0).tolist(),
                values.std(0, unbiased=False).clamp_min(STD_FLOOR).tolist())

    with torch.no_grad():
        input_mean, input_std = moments(actions[:, -1])
        both = encoder(actions)
        e = encoder.drive(actions[:, -1])
        static = e.unsqueeze(-1).pow(encoder.static_exponents).flatten(1)
        values = {"base": actions.new_empty((len(actions), 0)), "path": both[:, :8],
                  "time": both[:, 8:], "both": both, "static_capacity": static}
        memory_mean, memory_std = {}, {}
        for variant, features in values.items():
            memory_mean[variant], memory_std[variant] = moments(features)
    return dict(schema="internal_memory_normalization_v1", history=history, dt=float(dt),
        training_windows=len(actions), std_floor=STD_FLOOR, std_ddof=0,
        input_mean=input_mean, input_std=input_std,
        memory_mean=memory_mean, memory_std=memory_std)


class _MLPBase(nn.Module):
    """4 -> 64 -> 64 -> 45; module creation order fixes the shared base seed."""

    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(4, 64)
        self.L2 = nn.Linear(64, 64)
        self.readout = nn.Linear(64, 45)

    def forward(self, standardized_current):
        return self.readout(torch.tanh(self.L2(torch.tanh(self.L1(standardized_current)))))


class InternalMemoryModel(nn.Module):
    """Internal memory plugin with a preserved base architecture and seed.

    Public modules: ``base``, ``encoder``, ``D`` (None for base). Public buffers:
    ``input_mean/std`` (4), ``memory_mean/std`` (memory_dim). ``standardized_memory``
    applies fixed training statistics; ``encoder`` itself always returns raw
    features. All buffers and parameters follow .to(), .double() and state_dict.

    Counts, in variant order base/path/time/both/static_capacity:
      mlp:     7405 / 7937 / 8961 / 9473 / 9473
      koopman: 3981 / 4361 / 5081 / 5441 / 5441
    The intended MLP study uses base/both/static_capacity; path/time also follow
    the same public interface. D maps memory_dim -> 64 for MLP and -> 45 for
    Koopman. Means/stds are never refit in forward or in train mode.
    """

    def __init__(self, family, variant, normalization):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
        _validate_variant(variant)
        if not isinstance(normalization, Mapping):
            raise ValueError("normalization must be a dict from fit_plugin_normalization")
        self.family, self.variant = family, variant
        self.history = normalization.get("history", 20)
        self.dt = normalization.get("dt", .2)
        self.output_dim, self.n_nodes = 45, 15
        self.hidden_dim = 64 if family == "mlp" else 128
        self.latent_dim = None if family == "mlp" else 16
        # All random base initialization precedes construction of every branch.
        self.base = _MLPBase() if family == "mlp" else KoopmanShape(hidden=128, latent=16)
        self.encoder = MemoryEncoder(variant, history=self.history, dt=self.dt)
        self.memory_dim = self.encoder.output_dim
        self.branch_dim = 64 if family == "mlp" else 45
        self.D = nn.Linear(self.memory_dim, self.branch_dim, bias=False) if self.memory_dim else None
        if self.D is not None:
            nn.init.zeros_(self.D.weight)
        try:
            values = dict(input_mean=normalization["input_mean"], input_std=normalization["input_std"],
                memory_mean=normalization["memory_mean"][variant],
                memory_std=normalization["memory_std"][variant])
        except (KeyError, TypeError) as exc:
            raise ValueError("normalization needs input_mean/std and memory_mean/std[variant]") from exc
        for name, value in values.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            expected = 4 if name.startswith("input") else self.memory_dim
            if tensor.shape != (expected,) or not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must have {expected} finite entries")
            if name.endswith("std") and not torch.all(tensor > 0):
                raise ValueError(f"{name} must be strictly positive")
            self.register_buffer(name, tensor)

    def standardized_memory(self, actions):
        """Return (B,memory_dim), using fixed initial-drive training statistics."""
        return (self.encoder(actions)-self.memory_mean)/self.memory_std

    def forward(self, actions):
        _validate_windows(actions, self.history)
        if self.family == "koopman":
            prediction = self.base(actions)  # Preserve raw-pressure lift and all H updates.
            if self.D is not None:
                prediction = prediction+self.D(self.standardized_memory(actions)).reshape(-1, 15, 3)
            return prediction
        current = (actions[:, -1]-self.input_mean)/self.input_std
        h1 = torch.tanh(self.base.L1(current))
        second = self.base.L2(h1)
        if self.D is not None:
            second = second+self.D(self.standardized_memory(actions))
        return self.base.readout(torch.tanh(second)).reshape(-1, 15, 3)

    def parameter_count(self, trainable_only=True):
        return parameter_count(self, trainable_only=trainable_only)

    def count_parameters(self, trainable_only=True):
        return self.parameter_count(trainable_only=trainable_only)
