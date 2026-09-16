"""Action-only whole-shape baselines under a common causal-history contract.

Literature-related implementations are adaptations; provenance and differences
are recorded in docs/paper/icra2027/modeling_baselines_sources.md.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.model_hereditary_geometry import HereditaryGeometryModel
from src.models.model_ishsm import fit_ishsm_priors_from_arrays

MODEL_NAMES = (
    'mean', 'linear', 'polynomial2', 'mlp', 'window_mlp', 'direction_mlp',
    'gru', 'lstm', 'tcn', 'chen_direction', 'chen_static', 'park_tcn', 'bezier_gru', 'oscillator', 'hov', 'hov_no_play',
    'hov_no_maxwell', 'hov_static', 'hov_linear_reference', 'hov_pod8',
    'hov_memory_residual', 'hov_no_memory', 'koopman', 'pcc',
)


def mlp(input_dim, output_dim, hidden):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_dim))


def last_direction(actions):
    """Most recent nonzero charging/discharging direction inside the window.

    Holds retain the last observed sign; unknown direction at cold start is zero.
    This uses only pressure commands and has no access to future actions.
    """
    direction = torch.zeros_like(actions[:, 0])
    for t in range(1, actions.shape[1]):
        delta = actions[:, t] - actions[:, t - 1]
        direction = torch.where(delta.abs() > 1e-7, delta.sign(), direction)
    return direction


class Polynomial(nn.Module):
    def __init__(self, kind, nodes=15):
        super().__init__()
        self.kind = kind
        dimension = {'mean': 1, 'linear': 5, 'polynomial2': 15}[kind]
        self.register_buffer('coefficients', torch.zeros(dimension, nodes * 3))
        self.nodes = nodes

    def features(self, actions):
        a = actions[:, -1]
        features = [torch.ones_like(a[:, :1])]
        if self.kind != 'mean':
            features.append(a)
        if self.kind == 'polynomial2':
            features.append(torch.stack([a[:, i] * a[:, j] for i in range(4) for j in range(i, 4)], -1))
        return torch.cat(features, -1)

    def fit(self, actions, targets, ridge=1e-4):
        x = self.features(actions).double()
        y = targets.flatten(1).double()
        penalty = torch.eye(x.shape[1], dtype=x.dtype, device=x.device) * ridge
        penalty[0, 0] = 0
        # Scaling by sample count makes ridge independent of corpus size.
        coefficient = torch.linalg.solve(x.T @ x / len(x) + penalty, x.T @ y / len(x))
        self.coefficients.copy_(coefficient.float())

    def forward(self, actions):
        return (self.features(actions) @ self.coefficients).reshape(-1, self.nodes, 3)


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(inputs, outputs, 3, dilation=dilation)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x):
        return torch.tanh(self.conv(F.pad(x, (self.pad, 0))) + self.skip(x))


class ParkBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.pad = 2 * dilation
        self.first = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, 3, dilation=dilation)

    def forward(self, x):
        h = F.relu(self.first(F.pad(x, (self.pad, 0))))
        return x + F.relu(self.second(F.pad(h, (self.pad, 0))))


class OfficialRNN(nn.Module):
    def __init__(self, kind, hidden, nodes=15):
        super().__init__()
        from src.benchmarks.vendor.sponge_rnn import GRU, LSTM
        self.kind, self.hidden, self.nodes = kind, hidden, nodes
        self.core = (GRU if kind == 'gru' else LSTM)(4, nodes*3, hidden, 1, 0.)

    def forward(self, actions):
        h = actions.new_zeros(1, len(actions), self.hidden)
        output = self.core(actions, h) if self.kind == 'gru' else self.core(actions, h, h.clone())
        return output[0].reshape(-1, self.nodes, 3)


class OscillatorShape(nn.Module):
    """VON-inspired latent oscillator, trained with shape supervision.

    Positive diagonal mass=1, stiffness/damping; symplectic Euler as in the
    source representation paper, equilibrium at the first pressure. The visual
    encoder and original visual losses are replaced by a learned skeleton head.
    """
    def __init__(self, dt, latent=8, nodes=15, force_hidden=32):
        super().__init__()
        self.dt, self.nodes = dt, nodes
        self.force = mlp(4, latent, force_hidden)
        self.raw_stiffness = nn.Parameter(torch.zeros(latent))
        self.raw_damping = nn.Parameter(torch.zeros(latent))
        self.readout = nn.Linear(2*latent, nodes*3)

    def forward(self, actions):
        stiffness, damping = F.softplus(self.raw_stiffness)+.01, F.softplus(self.raw_damping)+.01
        z = self.force(actions[:, 0])/stiffness
        velocity = torch.zeros_like(z)
        for t in range(1, actions.shape[1]):
            velocity = velocity + self.dt*(self.force(actions[:, t])-stiffness*z-damping*velocity)
            z = z + self.dt*velocity
        return self.readout(torch.cat([z, velocity], -1)).reshape(-1, self.nodes, 3)


def bezier_sections(control, samples=8):
    """Two connected quadratic curves, sampled at equal arc length per section."""
    t = torch.linspace(0, 1, 65, device=control.device, dtype=control.dtype)
    basis = torch.stack([(1-t)**2, 2*t*(1-t), t**2], -1)
    sections = []
    for start in (0, 2):
        dense = torch.einsum('nk,bkc->bnc', basis, control[:, start:start+3])
        length = torch.linalg.vector_norm(dense[:, 1:]-dense[:, :-1], dim=-1)
        arc = torch.cat([length.new_zeros(len(length), 1), length.cumsum(-1)], -1)
        query = arc[:, -1:] * torch.linspace(0, 1, samples, device=arc.device, dtype=arc.dtype)
        index = torch.searchsorted(arc.contiguous(), query.contiguous(), right=True).clamp(1, len(t)-1)
        lower = torch.gather(arc, 1, index-1)
        upper = torch.gather(arc, 1, index)
        ratio = ((query-lower)/(upper-lower).clamp_min(1e-8)).unsqueeze(-1)
        left = torch.gather(dense, 1, (index-1).unsqueeze(-1).expand(-1, -1, 3))
        right = torch.gather(dense, 1, index.unsqueeze(-1).expand(-1, -1, 3))
        points = left + ratio*(right-left)
        sections.append(points if start == 0 else points[:, 1:])
    return torch.cat(sections, 1)


class NeuralShape(nn.Module):
    def __init__(self, kind, history, hidden=64, nodes=15, base_normalized=None, chen_hidden=128, park_channels=4):
        super().__init__()
        self.kind, self.nodes, self.history = kind, nodes, history
        if kind in ('chen_direction', 'chen_static'):
            layers = []
            inputs = 8 if kind == 'chen_direction' else 4
            for _ in range(4):
                layers.extend([nn.Linear(inputs, chen_hidden), nn.ReLU()])
                inputs = chen_hidden
            self.network = nn.Sequential(*layers, nn.Linear(chen_hidden, nodes*3))
        elif kind in ('mlp', 'window_mlp', 'direction_mlp'):
            inputs = {'mlp': 4, 'window_mlp': 4 * history, 'direction_mlp': 8}[kind]
            self.network = mlp(inputs, nodes * 3, hidden)
        elif kind in ('gru', 'lstm', 'bezier_gru'):
            cell = nn.LSTM if kind == 'lstm' else nn.GRU
            self.encoder = cell(4, hidden, batch_first=True)
            outputs = 12 if kind == 'bezier_gru' else nodes * 3
            self.readout = nn.Linear(hidden, outputs)
            if kind == 'bezier_gru':
                self.register_buffer('base_normalized', torch.as_tensor(
                    base_normalized if base_normalized is not None else [0., 0., 0.], dtype=torch.float32))
        elif kind == 'park_tcn':
            blocks = math.ceil(math.log2(1+(history-1)/4))
            self.encoder = nn.Sequential(nn.Conv1d(4, park_channels, 1) if park_channels != 4 else nn.Identity(),
                                         *[ParkBlock(park_channels, 2**b) for b in range(blocks)])
            self.readout = nn.Linear(park_channels, nodes*3)
        elif kind == 'tcn':
            self.encoder = nn.Sequential(CausalConv(4, hidden, 1), CausalConv(hidden, hidden, 2),
                                         CausalConv(hidden, hidden, 4))
            self.readout = nn.Linear(hidden, nodes * 3)
        else:
            raise ValueError(kind)

    def forward(self, actions):
        if self.kind in ('mlp', 'chen_static'):
            output = self.network(actions[:, -1])
        elif self.kind == 'window_mlp':
            output = self.network(actions.flatten(1))
        elif self.kind in ('direction_mlp', 'chen_direction'):
            output = self.network(torch.cat([actions[:, -1], last_direction(actions)], -1))
        elif self.kind in ('tcn', 'park_tcn'):
            output = self.readout(self.encoder(actions.transpose(1, 2))[:, :, -1])
        else:
            states, _ = self.encoder(actions)
            output = self.readout(states[:, -1])
        if self.kind == 'bezier_gru':
            base = self.base_normalized.view(1, 1, 3).expand(len(actions), 1, 3)
            return bezier_sections(torch.cat([base, output.reshape(len(actions), 4, 3)], 1))
        return output.reshape(-1, self.nodes, 3)


class AblatedGeometry(HereditaryGeometryModel):
    """Branches are excluded throughout fitting and inference, then retrained.

    State dimensions are retained for checkpoint compatibility; excluded branch
    parameters are frozen and its readout is identically zero during training.
    """
    def __init__(self, *args, disable_play=False, disable_maxwell=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_play, self.disable_maxwell = disable_play, disable_maxwell
        if disable_play:
            self.pi_mode_directions_raw.requires_grad_(False)
            for parameter in self.play.parameters():
                parameter.requires_grad_(False)
        if disable_maxwell:
            self.maxwell_mode_directions_raw.requires_grad_(False)
            self.maxwell_gain_raw.requires_grad_(False)
            for parameter in self.maxwell.parameters():
                parameter.requires_grad_(False)

    def _structured_memory(self, q, deficit):
        if self.disable_play:
            q = torch.zeros_like(q)
        if self.disable_maxwell:
            deficit = torch.zeros_like(deficit)
        return super()._structured_memory(q, deficit)


class GeometryWindow(nn.Module):
    def __init__(self, core, static=False):
        super().__init__()
        self.core, self.static = core, static
        if static:
            for parameter in core.parameters():
                parameter.requires_grad_(False)

    def forward(self, actions):
        if self.static:
            return self.core.decode_equilibrium(actions[:, -1])
        return self.core(actions)['skeleton']


def fit_normalization(sequences):
    positions = np.concatenate([s['positions'] for s in sequences])
    center = positions.mean(axis=(0, 1)).astype('float32')
    scale = float(max(np.sqrt(np.mean(np.sum((positions - center)**2, axis=-1))), 1.))
    return center, scale


def make_model(name, config, train_sequences=None, normalization=None, geometry_config=None):
    """Return model plus reconstruction metadata; priors use training only."""
    if name not in MODEL_NAMES:
        raise ValueError(f'Unknown model {name}')
    if name in ('mean', 'linear', 'polynomial2'):
        return Polynomial(name), None
    if name == 'koopman':
        from src.benchmarks.modeling_foundations import KoopmanShape
        return KoopmanShape(config['hidden'],config.get('latent',8)), None
    if name == 'pcc':
        from src.benchmarks.modeling_foundations import PCCShape, pcc_priors
        if geometry_config is None:
            geometry_config = pcc_priors(train_sequences,config['history'],config.get('train_stride',1),config.get('max_train_windows'))
        return PCCShape(config['hidden'],geometry_config,normalization), geometry_config
    if name in ('gru', 'lstm'):
        return OfficialRNN(name, config['hidden']), None
    if name == 'oscillator':
        return OscillatorShape(config['dt'],latent=config.get('latent',8),force_hidden=config.get('force_hidden',32)), None
    if not name.startswith('hov'):
        base = -np.asarray(normalization[0])/normalization[1] if normalization else None
        return NeuralShape(name, config['history'], config['hidden'], base_normalized=base, chen_hidden=config.get('chen_hidden',128),
                           park_channels=config.get('park_channels',4)), None
    if geometry_config is None:
        if not train_sequences or normalization is None:
            raise ValueError('Geometry priors require training data and normalization')
        from src.benchmarks.modeling_data import CausalWindows
        windows = CausalWindows(train_sequences, config['history'], config.get('train_stride', 1),
                                config.get('max_train_windows'))
        actions = np.stack([train_sequences[i]['actions'][t] for i, t in windows.indices])
        positions = np.stack([train_sequences[i]['positions'][t] for i, t in windows.indices])
        local = name != 'hov_pod8'
        priors = fit_ishsm_priors_from_arrays(
            actions, positions, n_bend_modes=14 if local else 8,
            section_intervals=(7, 7), reference_kind='linear' if name == 'hov_linear_reference' else 'monotone_spline',
            reference_fit_steps=config['prior_steps'], reference_fit_objective='geometry',
            bend_basis_kind='local' if local else 'pod')
        allowed = inspect.signature(HereditaryGeometryModel.__init__).parameters
        kwargs = {key: value for key, value in priors.items() if key in allowed}
        kwargs.update(action_dim=4, n_nodes=15, window_size=config['history'], dt=config['dt'],
                      n_bend_modes=14 if local else 8, section_intervals=(7, 7),
                      n_play=config.get('n_play',2), n_maxwell=config.get('n_maxwell',6), tau_range=(3*config['dt'], 2.),
                      drive_normalization='unit_range', burnin_mode='equilibrium',
                      residual_mode='memory' if name == 'hov_memory_residual' else 'none',
                      disable_play=name in ('hov_no_play','hov_no_memory'), disable_maxwell=name in ('hov_no_maxwell','hov_no_memory'))
        geometry_config = {key: value.tolist() if isinstance(value, np.ndarray) else value
                           for key, value in kwargs.items()}
    if config.get('calibrate_reference',False):
        from src.benchmarks.modeling_geometry_calibration import make_calibrated_geometry
        calibrated_config=dict(geometry_config,
            calibrate_length_directions=config.get('calibrate_length_directions',False),
            reference_pair_interactions=config.get('reference_pair_interactions',False),
            reference_pair_length_interactions=config.get('reference_pair_length_interactions',False))
        return make_calibrated_geometry(calibrated_config,normalization,static=name in ('hov_static','hov_no_memory'))
    core = AblatedGeometry(**geometry_config)
    core.set_normalization(np.asarray(normalization[0], dtype=np.float32),
                           np.full(3, normalization[1], dtype=np.float32), 1.)
    return GeometryWindow(core, name in ('hov_static','hov_no_memory')), geometry_config
