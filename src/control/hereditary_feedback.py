"""Offline hereditary observer and full-suffix local control primitives.

Actions use checkpoint-normalized units; state is post-action [p,h]. This
module has no hardware or recorded-reference access. Fixed diagonal state
regularization is deliberately not presented as a calibrated EKF.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch


def physical_shape(model, state, last_action):
    shape = model.observe_state(last_action[None], state[None])[0]
    return (shape * model.pc_scale.reshape(3) + model.pc_center.reshape(3))[:, :2]


def rollout(model, state, actions):
    """Each returned shape is after its corresponding action, including row zero."""
    shapes = []
    for action in actions:
        state = model.step_state(action[None], state[None])["latent_z"][0]
        shapes.append(physical_shape(model, state, action))
    return torch.stack(shapes) if shapes else actions.new_empty((0, model.n_nodes, 2))


def batched_rollout(model, state, actions):
    """Keep recurrent operators sequential, decode the complete horizon once.

    Differentiable on CPU and CUDA. This preserves the reference transition and
    supports its optional memory residual; no change to time step or horizon.
    """
    if not len(actions):
        return actions.new_empty((0, model.n_nodes, 2))
    drives = model.drive(actions)
    p, h = model._unpack_state(state[None])
    ps, hs = [], []
    for drive in drives:
        p, _ = model.play.step(p, drive[None])
        h = model.maxwell.step(h, drive[None])
        ps.append(p[0])
        hs.append(h[0])
    q = drives[..., None]-torch.stack(ps)
    deficit = torch.stack(hs)-drives[..., None]
    pi, maxwell, _, _ = model._structured_memory(q, deficit)
    memory = pi+maxwell+model._memory_residual(q, deficit)
    skeleton = model._decode_generalized(actions, memory)
    return (skeleton*model.pc_scale.reshape(3)+model.pc_center.reshape(3))[:, :, :2]


def project_state(model, state, action, action_lower, action_upper):
    p, h = model._unpack_state(state[None])
    drive = model.drive(action[None]).unsqueeze(-1)
    p = torch.maximum(drive - model.play.thresholds,
                      torch.minimum(drive + model.play.thresholds, p))
    lo = model.drive(action_lower[None]).unsqueeze(-1)
    hi = model.drive(action_upper[None]).unsqueeze(-1)
    # Include the rest initialization when it is outside the drive range.
    if model.burnin_mode == "rest":
        lo, hi = torch.minimum(lo, torch.zeros_like(lo)), torch.maximum(hi, torch.zeros_like(hi))
    h = torch.maximum(lo, torch.minimum(hi, h))
    return model._pack_state(p, h)[0]


def correct_state(model, state, action, residual, lower, upper, *,
                  prior_std=0.08, max_delta=0.12, huber=2.0):
    """One robust linearization, bounded update, nonlinear acceptance on fixed evidence.

    residual(z) returns dimensionless, noise-scaled residuals. A finite
    backtracking check never refreshes correspondences to manufacture acceptance.
    """
    started = time.perf_counter()
    state = state.detach().clone()
    value = residual(state).detach().reshape(-1)
    if not torch.isfinite(value).all():
        raise ValueError("nonfinite measurement residual")
    if value.numel() == 0:
        return state, {"accepted": False, "reason": "no_evidence", "count": 0,
                       "delta_norm": 0.0, "time_ms": (time.perf_counter()-started)*1000}
    jac = torch.autograd.functional.jacobian(residual, state, vectorize=True).reshape(-1, len(state)).detach()
    weights = (huber / value.abs().clamp_min(huber)).detach()
    hessian = jac.T @ (weights[:, None] * jac) + torch.eye(len(state), device=state.device) / prior_std**2
    delta = torch.linalg.solve(hessian, -jac.T @ (weights * value))
    delta *= min(1.0, max_delta / max(float(delta.abs().max()), 1e-12))

    def cost(v):
        a = v.abs()
        return torch.where(a <= huber, a.square(), 2*huber*a-huber**2).sum()

    before = float(cost(value))
    result, accepted, after = state, False, before
    for fraction in (1.0, 0.5, 0.25, 0.125):
        candidate = project_state(model, state + fraction*delta, action, lower, upper).detach()
        actual_delta = candidate-state
        v = residual(candidate).detach()
        objective = float(cost(v) + (actual_delta/prior_std).square().sum())
        if torch.isfinite(v).all() and objective < before - 1e-9:
            result, accepted, after = candidate, True, float(cost(v))
            break
    return result, {"accepted": accepted, "reason": "accepted" if accepted else "no_descent",
                    "count": len(value), "residual_before": before, "residual_after": after,
                    "delta_norm": float((result-state).norm()),
                    "time_ms": (time.perf_counter()-started)*1000}


from real_validation.runtime.hereditary_bounds import ActionBounds, block_basis


def correct_suffix(model, state, old, previous, reference, bounds, *,
                   blocks=8, regularization=2.0, trust=0.08, rollout_fn=rollout):
    """B: one current rollout/Jacobian and one constrained quadratic problem.

    SLSQP solves this explicit convex QP (fixed Hessian/linear constraints).
    A bounded nonlinear acceptance check retains the old feasible suffix on failure.
    """
    started = time.perf_counter()
    old = old.detach().clone()
    state = state.detach().clone()
    n, channels = old.shape
    if not n:
        return old, {"accepted": False, "reason": "empty_suffix", "time_ms": 0.0}
    old_np, previous_np = old.cpu().numpy(), previous.detach().cpu().numpy()
    if not bounds.valid(old_np, previous_np):
        raise ValueError("old suffix violates action contract")
    basis_np = block_basis(n, channels, blocks)
    basis = torch.as_tensor(basis_np, dtype=old.dtype, device=old.device)
    eta0 = old.new_zeros(basis.shape[1])

    def objective_residual(eta):
        actions = old + (basis @ eta).reshape_as(old)
        # Base node is fixed by geometry and provides no control objective.
        return (rollout_fn(model, state, actions)[:, 1:] - reference[:, 1:]).reshape(-1)

    value = objective_residual(eta0).detach()
    jac = torch.autograd.functional.jacobian(
        objective_residual, eta0, vectorize=True, strategy="forward-mode").detach()
    j, b = jac.cpu().double().numpy(), value.cpu().double().numpy()
    size = len(b)
    hess = j.T @ j / size + regularization * np.eye(len(eta0))
    linear = j.T @ b / size
    difference = np.eye(n)
    if n > 1:
        difference[np.arange(1,n), np.arange(n-1)] = -1
    rate_matrix = np.kron(difference, np.eye(channels))
    current_delta = rate_matrix @ old_np.reshape(-1)
    current_delta[:channels] -= previous_np
    matrix = np.vstack([basis_np, rate_matrix @ basis_np])
    lows = np.concatenate([(np.broadcast_to(bounds.lower, old_np.shape)-old_np).reshape(-1),
                           np.tile(-bounds.fall,n)-current_delta])
    highs = np.concatenate([(np.broadcast_to(bounds.upper,old_np.shape)-old_np).reshape(-1),
                            np.tile(bounds.rise,n)-current_delta])
    # Constant action blocks have zero rate-Jacobian rows internally. A stored
    # float32 suffix at a slew limit can differ from the double bound by ~1e-8;
    # passing 0 <= -1e-8 to SLSQP makes an otherwise feasible QP infeasible.
    # Check these invariant rows at the declared feasibility tolerance, then
    # omit them. Keep every constraint the correction can actually change.
    active = np.any(abs(matrix)>1e-12,axis=1)
    if np.any(lows[~active]>2e-6) or np.any(highs[~active]<-2e-6):
        raise ValueError("fixed action-block rates violate the feasible suffix")
    matrix,lows,highs = matrix[active],lows[active]-1e-7,highs[active]+1e-7
    solution = minimize(lambda x: 0.5*x@hess@x + linear@x, np.zeros(len(eta0)),
                        jac=lambda x: hess@x+linear, method="SLSQP",
                        bounds=[(-trust,trust)]*len(eta0),
                        constraints=[LinearConstraint(matrix,lows,highs)],
                        options={"maxiter": 80, "ftol": 1e-8})
    before = float(np.mean(b*b))
    result, after, accepted, fraction_used = old, before, False, 0.0
    if solution.success and np.isfinite(solution.x).all():
        update = (basis @ torch.as_tensor(solution.x,dtype=old.dtype,device=old.device)).reshape_as(old)
        for fraction in (1.0,0.5,0.25,0.125):
            candidate = old + fraction*update
            with torch.no_grad():
                error = rollout_fn(model,state,candidate)[:,1:] - reference[:,1:]
                score = float(error.square().mean())
            if bounds.valid(candidate.cpu().numpy(),previous_np) and score < before-1e-8:
                result,after,accepted,fraction_used = candidate.detach(),score,True,fraction
                break
    return result, {"accepted": accepted, "reason": "accepted" if accepted else "solver_or_descent_rejected",
                    "solver_success": bool(solution.success), "solver_message": str(solution.message),
                    "variables": len(eta0), "horizon": n,
                    "mse_before_mm2": before, "mse_after_mm2": after,
                    "delta_max": float((result-old).abs().max()), "step_fraction": fraction_used,
                    "time_ms": (time.perf_counter()-started)*1000}


def plan_initial(model, state, seed, previous, reference, bounds, *, iterations=8, blocks=8):
    """Recorded pressure is a seed; optimize a task plan before starting replay."""
    plan = torch.as_tensor(bounds.project(seed.detach().cpu().numpy(),previous.detach().cpu().numpy()),
                           dtype=seed.dtype,device=seed.device)
    trace=[]
    for _ in range(iterations):
        plan, info = correct_suffix(model,state,plan,previous,reference,bounds,blocks=blocks)
        trace.append(info)
        if not info["accepted"]:
            break
    return plan.detach(),trace
