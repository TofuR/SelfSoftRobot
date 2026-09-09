"""Frozen HOV geometry inference and exact local sensitivities on NumPy arrays.

This is an inference adapter for residual_mode='none', not a newly trained
model. Recurrence remains sequential; all channels, readouts, and sensitivity
directions are batched. Derivatives use the current piecewise-smooth branch.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize

from .hereditary_bounds import block_basis


def as_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64, copy=True)


class FrozenHereditary:
    def __init__(self, model):
        if model.residual_mode != "none":
            raise ValueError("analytic adapter requires residual_mode=none")
        if model.reference_kind not in {"linear", "monotone_spline"}:
            raise ValueError("unsupported reference mapping")
        self.channels = model.action_dim
        self.n_play, self.n_maxwell = model.n_play, model.n_maxwell
        self.n_p = model.action_dim * model.n_play
        self.z_dim = model.operator_state_dim
        self.n_bend = model.n_bend_modes
        self.n_nodes = model.n_nodes
        self.thresholds = as_numpy(model.play.thresholds)
        self.alpha = as_numpy(model.maxwell.decays)
        self.knots, self.drive_weights = as_numpy(model.drive.knots), as_numpy(model.drive.weights)
        scale = model.generalized_coordinate_scale
        self.wp = as_numpy(model.play.weights[:, :, None] * model.pi_mode_directions * scale).reshape(self.n_p, -1)
        self.wm = as_numpy(model.maxwell_gains[:, :, None] * model.maxwell_mode_directions * scale).reshape(self.z_dim-self.n_p, -1)
        self.memory_state = np.vstack([-self.wp, self.wm])
        self.memory_drive = (self.wp.reshape(self.channels, self.n_play, -1).sum(1)
                             - self.wm.reshape(self.channels, self.n_maxwell, -1).sum(1))
        self.basis = as_numpy(model.bend_basis)
        self.reference_kind = model.reference_kind
        self.reference_directions = np.concatenate([as_numpy(model.reference_bend_dirs),
                                                    as_numpy(model.reference_length_dirs)], axis=1)
        self.reference_bias = np.concatenate([as_numpy(model.reference_bend_bias),
                                              as_numpy(model.reference_length_bias)])
        if self.reference_kind == "monotone_spline":
            self.reference_knots = as_numpy(model.reference_knots)
            self.reference_weights = as_numpy(model.reference_drive_weights)
        self.section_ids = np.repeat(np.arange(model.n_sections), model.section_intervals)
        self.lengths = as_numpy(model.reference_segment_lengths)
        self.base = as_numpy(model.base_position)[:2]
        # Convert memory-mode coordinates to local bend / section log length.
        sections = model.n_sections
        self.to_generalized = np.zeros((self.n_bend+sections, self.n_nodes-1+sections))
        self.to_generalized[:self.n_bend, :self.n_nodes-1] = self.basis.T
        self.to_generalized[self.n_bend:, self.n_nodes-1:] = np.eye(sections)
        self.state_to_generalized = self.memory_state @ self.to_generalized
        self.drive_to_generalized = self.memory_drive @ self.to_generalized

    def drive(self, actions):
        hinges = np.asarray(actions)[..., None] - self.knots
        return ((np.maximum(hinges, 0) * self.drive_weights).sum(-1),
                ((hinges > 0) * self.drive_weights).sum(-1))

    def step(self, state, action, *, derivatives=False):
        state, action = np.asarray(state), np.asarray(action)
        drive, slope = self.drive(action)
        p = state[:self.n_p].reshape(self.channels, self.n_play)
        h = state[self.n_p:].reshape(self.channels, self.n_maxwell)
        lo, hi = drive[:, None]-self.thresholds, drive[:, None]+self.thresholds
        held = (p >= lo) & (p <= hi)
        new = np.concatenate([np.clip(p, lo, hi).ravel(),
                              (self.alpha*h + (1-self.alpha)*drive[:, None]).ravel()])
        if not derivatives:
            return new
        diagonal = np.concatenate([held.ravel().astype(float), np.tile(self.alpha, self.channels)])
        bu = np.zeros((self.z_dim, self.channels))
        for channel in range(self.channels):
            bu[channel*self.n_play:(channel+1)*self.n_play, channel] = (~held[channel])*slope[channel]
            begin = self.n_p+channel*self.n_maxwell
            bu[begin:begin+self.n_maxwell, channel] = (1-self.alpha)*slope[channel]
        return new, diagonal, bu

    def observe(self, states, actions, *, derivatives=False):
        single = np.asarray(states).ndim == 1
        states, actions = np.atleast_2d(states), np.atleast_2d(actions)
        drive, slope = self.drive(actions)
        ref, ref_slope = actions, np.ones_like(actions)
        if self.reference_kind == "monotone_spline":
            hinge = actions[..., None]-self.reference_knots
            ref = (np.maximum(hinge, 0)*self.reference_weights).sum(-1)
            ref_slope = ((hinge > 0)*self.reference_weights).sum(-1)
        generalized = (self.reference_bias + ref @ self.reference_directions
                       + states @ self.state_to_generalized + drive @ self.drive_to_generalized)
        segments = self.n_nodes-1
        bend, logs = generalized[:, :segments], generalized[:, segments:]
        angle = np.cumsum(bend, axis=1)
        per_segment = logs[:, self.section_ids]
        lengths = self.lengths * np.exp(np.clip(per_segment, -.25, .25))
        increments = np.stack([lengths*np.cos(angle), lengths*np.sin(angle)], axis=-1)
        output = np.empty((len(states), self.n_nodes, 2))
        output[:, 0] = self.base
        output[:, 1:] = self.base + np.cumsum(increments, axis=1)
        if not derivatives:
            return output[0] if single else output
        # d increment_i / d bend_j is nonzero for j <= i.
        coordinate_count = generalized.shape[1]
        derivative = np.zeros((len(states), segments, 2, coordinate_count))
        tangent = np.stack([-increments[:, :, 1], increments[:, :, 0]], axis=-1)
        derivative[:, :, :, :segments] = tangent[:, :, :, None] * np.tril(np.ones((segments, segments)))[None, :, None, :]
        length_active = ((per_segment >= -.25) & (per_segment <= .25))
        for section in range(logs.shape[1]):
            derivative[:, :, :, segments+section] = increments * (length_active & (self.section_ids == section))[..., None]
        geometry = np.zeros((len(states), self.n_nodes, 2, coordinate_count))
        geometry[:, 1:] = np.cumsum(derivative, axis=1)
        cz = geometry @ self.state_to_generalized.T
        direct = (ref_slope[:, :, None]*self.reference_directions[None]
                  + slope[:, :, None]*self.drive_to_generalized[None])
        cu = np.einsum('bnyg,bcg->bnyc', geometry, direct)
        if single:
            return output[0], cz[0], cu[0]
        return output, cz, cu

    def rollout(self, state, actions, *, action_directions=None, initial_directions=None):
        actions = np.asarray(actions)
        if len(actions) == 0:
            return np.empty((0, self.n_nodes, 2))
        differentiated = action_directions is not None or initial_directions is not None
        states = []
        if differentiated:
            directions = (initial_directions.shape[-1] if initial_directions is not None
                          else action_directions.shape[-1])
            dz = (np.array(initial_directions, copy=True) if initial_directions is not None
                  else np.zeros((self.z_dim, directions)))
            du = (np.asarray(action_directions) if action_directions is not None
                  else np.zeros((*actions.shape, directions)))
            sensitivities = []
        for k, action in enumerate(actions):
            if differentiated:
                state, az, bu = self.step(state, action, derivatives=True)
                dz = az[:, None]*dz + bu @ du[k]
                sensitivities.append(dz)
            else:
                state = self.step(state, action)
            states.append(state)
        states = np.asarray(states)
        if not differentiated:
            return self.observe(states, actions)
        shapes, cz, cu = self.observe(states, actions, derivatives=True)
        jac = np.einsum('hnyz,hzd->hnyd', cz, sensitivities) + np.einsum('hnyc,hcd->hnyd', cu, du)
        return shapes, jac


def constrained_update(jac, residual, old, previous, bounds, basis, *, regularization=2., trust=.08):
    """Same QP as the torch reference, with sparse-in-time rate assembly."""
    n, channels = old.shape
    j, b = jac.reshape(-1, basis.shape[1]), residual.reshape(-1)
    hessian = j.T @ j / len(b) + regularization*np.eye(j.shape[1])
    linear = j.T @ b / len(b)
    shaped = basis.reshape(n, channels, -1)
    rate_basis = shaped.copy()
    rate_basis[1:] -= shaped[:-1]
    delta = np.diff(np.vstack([previous, old]), axis=0)
    matrix = np.vstack([basis, rate_basis.reshape(-1, basis.shape[1])])
    low = np.concatenate([(bounds.lower-old).ravel(), (-bounds.fall-delta).ravel()])
    high = np.concatenate([(bounds.upper-old).ravel(), (bounds.rise-delta).ravel()])
    active = np.any(abs(matrix) > 1e-12, axis=1)
    if np.any(low[~active] > 2e-6) or np.any(high[~active] < -2e-6):
        raise ValueError("infeasible invariant block rates")
    # A constant correction block repeats the same box-constraint row for
    # every covered action. Intersect equal rows exactly instead of asking the
    # solver to process hundreds of redundant inequalities at every step.
    matrix, inverse = np.unique(matrix[active], axis=0, return_inverse=True)
    compact_low = np.full(len(matrix), -np.inf)
    compact_high = np.full(len(matrix), np.inf)
    np.maximum.at(compact_low, inverse, low[active]-1e-7)
    np.minimum.at(compact_high, inverse, high[active]+1e-7)
    if np.any(compact_low > compact_high):
        raise ValueError("inconsistent action-block constraints")
    solution = minimize(lambda v: .5*v@hessian@v+linear@v, np.zeros(j.shape[1]),
                        jac=lambda v: hessian@v+linear, method="SLSQP",
                        bounds=[(-trust, trust)]*j.shape[1],
                        constraints=[LinearConstraint(matrix, compact_low, compact_high)],
                        options={"maxiter": 80, "ftol": 1e-8})
    return (basis @ solution.x).reshape(old.shape), solution


def fast_suffix_b(engine, state, old, previous, reference, bounds, *, blocks=8,
                  regularization=2., trust=.08):
    started = time.perf_counter()
    if not bounds.valid(old, previous):
        raise ValueError("infeasible old suffix")
    basis = block_basis(len(old), engine.channels, blocks)
    tick = time.perf_counter()
    prediction, jac = engine.rollout(state, old, action_directions=basis.reshape(*old.shape, -1))
    derivative_ms = (time.perf_counter()-tick)*1000
    error = prediction[:, 1:]-reference[:, 1:]
    tick = time.perf_counter()
    delta, solver = constrained_update(jac[:, 1:], error, old, previous, bounds, basis,
                                       regularization=regularization, trust=trust)
    solver_ms = (time.perf_counter()-tick)*1000
    before = float(np.mean(error**2))
    result, score, accepted = old.copy(), before, False
    tick = time.perf_counter()
    if solver.success and np.isfinite(delta).all():
        for fraction in (1., .5, .25, .125):
            candidate = old+fraction*delta
            candidate_score = float(np.mean((engine.rollout(state, candidate)[:, 1:]-reference[:, 1:])**2))
            if bounds.valid(candidate, previous) and candidate_score < before-1e-8:
                result, score, accepted = candidate, candidate_score, True
                break
    return result, {"accepted": accepted, "solver_success": bool(solver.success),
                    "solver_message": str(solver.message), "horizon": len(old),
                    "mse_before_mm2": before, "mse_after_mm2": score,
                    "derivative_ms": derivative_ms, "solver_ms": solver_ms,
                    "validation_ms": (time.perf_counter()-tick)*1000,
                    "time_ms": (time.perf_counter()-started)*1000}


class CachedSuffixA:
    """Precompute response to state and *all* suffix actions at each boundary.

    Online b includes both corrected-state deviation and old-minus-nominal
    suffix. Projection and true nonlinear validation are timed online. A stale
    cache is flagged; this class does not silently fall back to B or refresh.
    """
    def __init__(self, engine, initial_state, nominal, reference, *, blocks=8,
                 regularization=2., max_model_discrepancy_mm=.5):
        self.engine = engine
        self.nominal = np.array(nominal, copy=True)
        self.reference = np.array(reference, copy=True)
        self.max_model_discrepancy_mm = max_model_discrepancy_mm
        self.cache = []
        z = np.array(initial_state, copy=True)
        self.nominal_states = []
        started = time.perf_counter()
        for k in range(len(nominal)):
            self.nominal_states.append(z.copy())
            old = self.nominal[k:]
            n, c = old.shape
            dim = engine.z_dim+n*c
            initial = np.zeros((engine.z_dim, dim))
            initial[:, :engine.z_dim] = np.eye(engine.z_dim)
            inputs = np.zeros((n, c, dim))
            inputs[:, :, engine.z_dim:] = np.eye(n*c).reshape(n, c, n*c)
            shapes, jac = engine.rollout(z, old, action_directions=inputs, initial_directions=initial)
            linear = jac[:, 1:].reshape(-1, dim)
            obs, response = linear[:, :engine.z_dim], linear[:, engine.z_dim:]
            basis = block_basis(n, c, blocks)
            td = response @ basis
            q = np.linalg.solve(td.T@td/len(td)+regularization*np.eye(td.shape[1]), td.T/len(td))
            bias = (shapes[:, 1:]-reference[k:, 1:]).ravel()
            self.cache.append((basis, obs, response, bias, q))
            z = engine.step(z, nominal[k])
        self.precompute_ms = (time.perf_counter()-started)*1000
        self.cache_bytes = sum(v.nbytes for row in self.cache for v in row)

    def correct(self, boundary, state, old, previous, bounds, *, trust=.08):
        started = time.perf_counter()
        if not bounds.valid(old, previous):
            raise ValueError("infeasible old suffix")
        basis, obs, response, bias, q = self.cache[boundary]
        approximation = bias + obs@(state-self.nominal_states[boundary]) + response@(old-self.nominal[boundary:]).ravel()
        eta = -q @ approximation
        delta = (basis @ eta).reshape(old.shape)
        delta *= min(1., trust/max(float(abs(delta).max()), 1e-12))
        mapping_ms = (time.perf_counter()-started)*1000
        actual = self.engine.rollout(state, old)[:, 1:]-self.reference[boundary:, 1:]
        discrepancy = float(np.sqrt(np.mean((actual.ravel()-approximation)**2)))
        before = float(np.mean(actual**2))
        result, score, accepted = old.copy(), before, False
        stale = discrepancy > self.max_model_discrepancy_mm
        projection_max = 0.
        if not stale:
            for fraction in (1., .5, .25, .125):
                raw = old+fraction*delta
                candidate = bounds.project(raw, previous)
                projection_max = max(projection_max, float(abs(candidate-raw).max()))
                value = float(np.mean((self.engine.rollout(state, candidate)[:, 1:]-self.reference[boundary:, 1:])**2))
                if bounds.valid(candidate, previous) and value < before-1e-8:
                    result, score, accepted = candidate, value, True
                    break
        return result, {"accepted": accepted, "cache_stale": stale,
                        "cache_discrepancy_mm": discrepancy,
                        "reason": "cache_stale" if stale else ("accepted" if accepted else "no_descent"),
                        "projection_max": projection_max, "mse_before_mm2": before,
                        "mse_after_mm2": score, "horizon": len(old),
                        "mapping_ms": mapping_ms, "time_ms": (time.perf_counter()-started)*1000}
