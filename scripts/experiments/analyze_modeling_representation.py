#!/usr/bin/env python3
"""Explain the frozen HOV representation with algebra and train/test diagnostics.

Writes only the representation task's report and run directories. All five
formal seeds are retained; no fitting, checkpoint selection or label editing.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.benchmarks.modeling_models import make_model
from src.models.model_ishsm import generalized_to_skeleton

STUDY = ROOT / "workspace/runs/training/modeling_three_seq_20260913_001928"
REPORT = ROOT / "workspace/reports/modeling_paper_revision_20260913_002"
RUN = ROOT / "workspace/runs/analysis/modeling_paper_revision_20260913_002/representation"


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, Path):
        return str(x.relative_to(ROOT)) if x.is_relative_to(ROOT) else str(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_json(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(x), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(plain(rows))


def chart(id, title, rows, *, kind="line", x="index", y="value", color=None,
          unit="", description=""):
    return dict(id=id, title=title, kind=kind, rows=rows, x=x, y=y,
                color=color, unit=unit, description=description)


def pooled_summary(rows, keys, fields):
    result = []
    for values in sorted({tuple(r[k] for k in keys) for r in rows}):
        part = [r for r in rows if tuple(r[k] for k in keys) == values]
        out = dict(zip(keys, values))
        out["seeds"] = len(part)
        for field in fields:
            data = np.asarray([r[field] for r in part], dtype=float)
            out[field] = float(data.mean())
            out[field + "_sd"] = float(data.std(ddof=1)) if len(part) > 1 else 0.
        result.append(out)
    return result


def load_data(study):
    result, inventory = {}, []
    for split in ("train", "test"):
        groups = []
        for path in sorted((study / "data" / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as d:
                a = d["actions"].astype(np.float32)
                windows = np.stack([a[t - 19:t + 1] for t in range(19, len(a))])
                groups.append(dict(group=path.stem, windows=windows,
                                   target=d["positions"][19:].astype(np.float64),
                                   frames=d["frame_ids"][19:]))
                inventory.append(dict(split=split, group=path.stem, frames=len(a),
                                      windows=len(windows), path=path))
        result[split] = groups
    assert sum(len(g["windows"]) for g in result["train"]) == 8988
    assert sum(len(g["windows"]) for g in result["test"]) == 2958
    return result, inventory


def checkpoint(study, seed):
    path = study / "formal/hov" / f"seed_{seed}" / "best_eval_model.pt"
    c = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = make_model(c["model"], c["config"],
                          normalization=(c["center"], c["scale"]),
                          geometry_config=c["geometry_config"])
    model.load_state_dict(c["state_dict"], strict=True)
    model.eval()
    assert model.core.residual_mode == "none"
    assert model.core.burnin_mode == "equilibrium"
    return model.core, c, path


def geometry_arrays(core, action, memory):
    """Physical reference, exact displacement and analytic J_ref @ memory.

    Uses the calibrated reference, including static pressure pair terms, and
    exactly the decoder's piecewise bounded length convention.
    """
    with torch.inference_mode():
        b, ell = core._reference(torch.as_tensor(action, dtype=torch.float32))
    b, ell = b.numpy().astype(float), ell.numpy().astype(float)
    basis = core.bend_basis.detach().numpy().astype(float)
    reference_lengths = core.reference_segment_lengths.detach().numpy().astype(float)
    section = np.repeat(np.arange(core.n_sections), core.section_intervals)
    theta = np.cumsum(b, axis=1)
    lam = ell[:, section]
    lengths = reference_lengths * np.exp(np.clip(lam, -.25, .25))
    tangent = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    delta_theta = np.cumsum(memory[:, :core.n_bend_modes] @ basis.T, axis=1)
    delta_lam = memory[:, core.n_bend_modes:][:, section]
    derivative_mask = (lam > -.25) & (lam < .25)
    first_segments = lengths[:, :, None] * (
        tangent * (delta_lam * derivative_mask)[:, :, None]
        + normal * delta_theta[:, :, None])
    new_lengths = reference_lengths * np.exp(np.clip(lam + delta_lam, -.25, .25))
    new_theta = theta + delta_theta
    actual_segments = new_lengths[:, :, None] * np.stack(
        [np.cos(new_theta), np.sin(new_theta)], axis=-1) - lengths[:, :, None] * tangent

    def integrate(x):
        return np.concatenate([np.zeros((len(x), 1, 2)), np.cumsum(x, axis=1)], axis=1)

    unclipped = np.all((np.abs(lam) < .25) & (np.abs(lam + delta_lam) < .25), axis=1)
    # Integral Taylor remainder: |exp(z)-1-z| <= |z|^2 exp(max(Re z,0))/2.
    segment_bound = .5 * lengths * (delta_theta ** 2 + delta_lam ** 2) * np.exp(np.maximum(delta_lam, 0))
    bounds = np.column_stack([np.zeros(len(memory)), np.cumsum(segment_bound, axis=1)])
    reference = integrate(lengths[:, :, None] * tangent) + core.base_position.detach().numpy()[:2]
    return dict(reference=reference, exact=integrate(actual_segments), linear=integrate(first_segments),
                bounds=bounds, unclipped=unclipped, delta_theta=delta_theta,
                delta_lam=delta_lam, bend_reference=b, length_reference=ell)


def jacobian_checks(core, a, m, geo):
    """Independent torch autograd and finite difference checks in float64."""
    indices = np.linspace(0, len(a) - 1, 8, dtype=int)
    max_autograd = max_finite = max_decoder = 0.
    basis = core.bend_basis.detach().double()
    ref_lengths = core.reference_segment_lengths.detach().double()
    for idx in indices:
        b = torch.from_numpy(geo["bend_reference"][idx:idx + 1])
        ell = torch.from_numpy(geo["length_reference"][idx:idx + 1])
        mem = torch.from_numpy(m[idx])

        def decode(v):
            return generalized_to_skeleton(
                b + (v[:core.n_bend_modes] @ basis.T)[None],
                ell + v[core.n_bend_modes:][None], ref_lengths,
                core.section_intervals, core.base_position.detach().double())[0, :, :2]

        zero = torch.zeros_like(mem)
        jac = torch.autograd.functional.jacobian(decode, zero)
        auto = (jac @ mem).detach().numpy()
        eps = 1e-3
        finite = ((decode(eps * mem) - decode(-eps * mem)) / (2 * eps)).detach().numpy()
        exact = (decode(mem) - decode(zero)).detach().numpy()
        max_autograd = max(max_autograd, float(np.max(np.abs(auto - geo["linear"][idx]))))
        max_finite = max(max_finite, float(np.max(np.abs(finite - geo["linear"][idx]))))
        max_decoder = max(max_decoder, float(np.max(np.abs(exact - geo["exact"][idx]))))
    assert max_autograd < 1e-8
    assert max_finite < 1e-5
    assert max_decoder < 1e-8
    return dict(checked_windows=8, analytic_vs_autograd_max_mm=max_autograd,
                analytic_vs_finite_difference_max_mm=max_finite,
                numpy_vs_torch_decoder_max_mm=max_decoder)


def collect(core, groups, study, split, seed, out):
    pooled = {k: [] for k in ("q", "d", "memory", "pi", "time", "action", "e_last",
                              "delta_e", "q_previous", "ever_saturated", "reference", "exact",
                              "linear", "target", "pred", "frames", "group_index")}
    recurrence_errors, agreement = [], []
    for group_index, g in enumerate(groups):
        local_predictions = []
        for start in range(0, len(g["windows"]), 256):
            a = torch.from_numpy(g["windows"][start:start + 256])
            with torch.inference_mode():
                e = core.drive(a.flatten(0, 1)).reshape(len(a), 20, 4)
                p = e[:, 0, :, None].repeat(1, 1, core.n_play)
                h = e[:, 0, :, None].repeat(1, 1, core.n_maxwell)
                q = torch.zeros_like(p)
                saturated = torch.zeros_like(q, dtype=torch.bool)
                q_error = 0.
                for t in range(1, 20):
                    qprev = q
                    p, q = core.play.step(p, e[:, t])
                    qstop = torch.clamp(qprev + (e[:, t] - e[:, t - 1])[:, :, None],
                                       -core.play.thresholds, core.play.thresholds)
                    q_error = max(q_error, float((q - qstop).abs().max()))
                    h = core.maxwell.step(h, e[:, t])
                    saturated |= (q.abs() - core.play.thresholds).abs() < 1e-6
                outputs = core._state_output(a[:, -1], p, h, q, e[:, -1])
                d = h - e[:, -1, :, None]
                alpha = core.maxwell.decays.numpy().astype(float)
                de = np.diff(e.numpy().astype(float), axis=1)
                powers = alpha[None, :] ** np.arange(19, 0, -1)[:, None]
                expanded = -np.einsum("btc,tk->bck", de, powers)
                d_error = float(np.max(np.abs(expanded - d.numpy())))
                recurrence_errors.append(dict(group=g["group"], start=start,
                                              stop_identity_max=q_error, time_convolution_max=d_error))
                m = outputs["memory_generalized"].numpy().astype(float)
                geo = geometry_arrays(core, a[:, -1].numpy(), m)
                pred = outputs["skeleton"].numpy() * core.pc_scale.numpy() + core.pc_center.numpy()
                assert np.max(np.abs(geo["reference"] + geo["exact"] - pred[:, :, :2])) < .0002
                if start == 0:
                    direct = core(a)["skeleton"].numpy() * core.pc_scale.numpy() + core.pc_center.numpy()
                    assert np.max(np.abs(direct - pred)) < 1e-6
                for key, value in dict(q=q.numpy(), d=d.numpy(), memory=m,
                        pi=outputs["pi_generalized"].numpy(), time=outputs["maxwell_generalized"].numpy(),
                        action=a[:, -1].numpy(), e_last=e[:, -1].numpy(), delta_e=de[:, -1],
                        q_previous=qprev.numpy(), ever_saturated=saturated.numpy(),
                        reference=geo["reference"], exact=geo["exact"], linear=geo["linear"],
                        target=g["target"][start:start + len(a)], pred=pred,
                        frames=g["frames"][start:start + len(a)], group_index=np.full(len(a), group_index)).items():
                    pooled[key].append(value)
                local_predictions.append(pred)
        if split == "test":
            path = study / "evaluations/hov" / f"seed_{seed}" / f"{g['group']}_predictions.npz"
            with np.load(path) as d:
                assert np.array_equal(g["frames"], d["frame_ids"])
                assert np.allclose(g["target"], d["target_mm"], atol=1e-5)
                error = float(np.max(np.abs(np.concatenate(local_predictions) - d["prediction_mm"])))
                assert error < .002
                agreement.append(dict(group=g["group"], coordinate_max_difference_mm=error))
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}
    np.savez_compressed(out / f"seed_{seed}_{split}_states_geometry.npz", **pooled)
    assert max(r["stop_identity_max"] for r in recurrence_errors) < 2e-7
    assert max(r["time_convolution_max"] for r in recurrence_errors) < 2e-6
    return pooled, dict(seed=seed, split=split, windows=len(pooled["q"]),
                        stop_identity_max=max(r["stop_identity_max"] for r in recurrence_errors),
                        time_convolution_max=max(r["time_convolution_max"] for r in recurrence_errors),
                        saved_prediction_agreement=agreement)


def pca_analysis(train, test, label, seed):
    train = train.reshape(len(train), -1).astype(float)
    test = test.reshape(len(test), -1).astype(float)
    mean, sd = train.mean(axis=0), train.std(axis=0)
    active = sd > 1e-9
    x = (train[:, active] - mean[active]) / sd[active]
    z = (test[:, active] - mean[active]) / sd[active]
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    p = s ** 2 / (s ** 2).sum()
    cumulative = np.cumsum(p)
    k95, k99 = (int(np.searchsorted(cumulative, q) + 1) for q in (.95, .99))
    test_energy = float((z ** 2).sum())
    test_component = (z @ vt.T) ** 2
    test_cum = np.cumsum(test_component.sum(axis=0)) / test_energy
    summary = dict(seed=seed, family=label, dimensions=train.shape[1], active_dimensions=int(active.sum()),
                   participation_rank=float(1 / np.sum(p ** 2)), k95=k95, k99=k99,
                   test_energy_at_train_k95=float(test_cum[k95 - 1]),
                   test_energy_at_train_k99=float(test_cum[k99 - 1]),
                   condition_number=float(s[0] / s[-1]))
    rows = [dict(seed=seed, family=label, component=i + 1, train_cumulative=float(cumulative[i]),
                 test_cumulative=float(test_cum[i]), relative_singular=float(s[i] / s[0])) for i in range(len(s))]
    return summary, rows, dict(mean=mean, sd=sd, active=active, basis=vt, singular_values=s)


def play_analysis(data, thresholds, split, seed):
    moving = np.abs(data["delta_e"]) > 1e-6
    rows = []
    for j, r in enumerate(thresholds):
        previous = data["q_previous"][:, :, j]
        q = data["q"][:, :, j]
        raw = previous + data["delta_e"]
        interior = (np.abs(raw) < r - 1e-6) & moving
        saturated = (np.abs(raw) > r + 1e-6) & moving
        reversal = moving & (np.abs(np.abs(previous) - r) < 1e-6) & (previous * data["delta_e"] < 0)
        from_boundary_interior = reversal & interior
        # q is the stop readout; p, not q, is fixed in this unsaturated interval.
        response_error = np.abs((q - previous) - data["delta_e"])
        rows.append(dict(seed=seed, split=split, threshold=float(r), active_channel_updates=int(moving.sum()),
                         interior_fraction=float(interior.sum() / moving.sum()),
                         clipped_fraction=float(saturated.sum() / moving.sum()),
                         boundary_fraction=float(1 - (interior.sum() + saturated.sum()) / moving.sum()),
                         reversal_events=int(reversal.sum()),
                         reversal_to_interior_events=int(from_boundary_interior.sum()),
                         reversal_linear_max_error=float(response_error[from_boundary_interior].max())
                            if from_boundary_interior.any() else 0.,
                         never_saturated_window_fraction=float((~data["ever_saturated"][:, :, j])[moving].mean())))
    return rows


def geometry_analysis(core, data, split, seed):
    m = data["memory"]
    geo = geometry_arrays(core, data["action"], m)
    exact, linear = geo["exact"], geo["linear"]
    remainder = exact - linear
    node_error = np.linalg.norm(remainder, axis=-1)
    norm_exact = np.linalg.norm(exact, axis=-1)
    energy = np.sum(exact ** 2)
    nz = np.linalg.norm(exact.reshape(len(m), -1), axis=1) > 1e-6
    cosine = np.sum(exact * linear, axis=(1, 2))[nz] / (
        np.linalg.norm(exact.reshape(len(m), -1), axis=1)[nz] *
        np.linalg.norm(linear.reshape(len(m), -1), axis=1)[nz])
    unclip = geo["unclipped"]
    violation = float(np.max(node_error[unclip] - geo["bounds"][unclip])) if unclip.any() else None
    if violation is not None:
        assert violation < 1e-9
    row = dict(seed=seed, split=split, windows=len(m),
               memory_node_displacement_mm=float(norm_exact.mean()),
               memory_endpoint_displacement_mm=float(norm_exact[:, -1].mean()),
               linearization_node_error_mm=float(node_error.mean()),
               linearization_endpoint_error_mm=float(node_error[:, -1].mean()),
               linearization_node_p95_mm=float(np.quantile(node_error, .95)),
               relative_rms_error=float(np.sqrt(np.sum(remainder ** 2) / energy)),
               explained_displacement_energy=float(1 - np.sum(remainder ** 2) / energy),
               median_shape_cosine=float(np.median(cosine)),
               max_cumulative_bend_change_rad=float(np.max(np.abs(geo["delta_theta"]))),
               p95_cumulative_bend_change_rad=float(np.quantile(np.abs(geo["delta_theta"]), .95)),
               max_log_length_change=float(np.max(np.abs(geo["delta_lam"]))),
               clipped_frames=int((~unclip).sum()),
               bound_max_violation_mm=violation)
    profiles = [dict(seed=seed, split=split, node=i,
                     actual_memory_displacement_mm=float(norm_exact[:, i].mean()),
                     jacobian_displacement_mm=float(np.linalg.norm(linear[:, i], axis=-1).mean()),
                     remainder_mm=float(node_error[:, i].mean())) for i in range(15)]
    # First-order path/time displacement profiles sum exactly, even though the
    # finite nonlinear displacements of separate branches need not be additive.
    for name in ("pi", "time"):
        effect = geometry_arrays(core, data["action"], data[name])["linear"]
        for p in profiles:
            p[name + "_linear_rms_mm"] = float(np.sqrt(np.mean(np.sum(effect[:, p["node"]] ** 2, axis=-1))))
    return row, profiles, jacobian_checks(core, data["action"], m, geo)


def kernel_analysis(core, action, seed, channel):
    """Local geometric impulse kernel of time memory, in mm / unit Delta e.

    Current reference is held fixed, so this is the memory term's Jacobian
    kernel; it is not the total input response or a measured physical impulse.
    """
    weights = (core.maxwell_gains[:, :, None] * core.maxwell_mode_directions *
               core.generalized_coordinate_scale[None, None, :]).detach().numpy().astype(float)
    alpha = core.maxwell.decays.numpy().astype(float)
    lag = np.arange(51)
    kernels = -np.einsum("ckg,lk->lcg", weights, alpha[None, :] ** (lag[:, None] + 1))
    generalized = kernels[:, channel]
    geometric = geometry_arrays(core, np.repeat(action[None], len(lag), axis=0), generalized)["linear"]
    rows = [dict(seed=seed, lag_s=float(l * core.maxwell.dt), channel=channel,
                 node=node, lateral_gain_mm=float(geometric[l, node, 0]),
                 longitudinal_gain_mm=float(geometric[l, node, 1]))
            for l in lag for node in (3, 7, 11, 14)]
    frequency = np.linspace(0, 2.5, 126)
    z = np.exp(-2j * np.pi * frequency[:, None] * core.maxwell.dt)
    response = -alpha * (1 - z) / (1 - alpha * z)
    frequency_rows = [dict(seed=seed, frequency_hz=float(f), tau_s=float(tau),
                           amplitude=float(abs(response[i, k])))
                      for i, f in enumerate(frequency) for k, tau in enumerate(core.maxwell.taus.numpy())]
    return rows, frequency_rows, kernels


def build(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.run.mkdir(parents=True, exist_ok=True)
    args.report.mkdir(parents=True, exist_ok=True)
    groups, inventory = load_data(args.study)
    train_windows = np.concatenate([g["windows"] for g in groups["train"]])
    train_current = train_windows[:, -1]
    mean_action = train_current.mean(axis=0)
    reference_index = int(np.argmin(np.linalg.norm(train_current - mean_action, axis=1)))
    reference_action = train_current[reference_index]
    channel = int(np.argmax(np.var(np.diff(train_windows, axis=1), axis=(0, 1))))
    state_rows, eigens, play_rows, geometry_rows, profiles = [], [], [], [], []
    kernels, frequency_rows, kernel_matrices, checks, metadata = [], [], [], [], []
    for seed in range(5):
        core, saved, path = checkpoint(args.study, seed)
        data = {}
        for split in ("train", "test"):
            data[split], check = collect(core, groups[split], args.study, split, seed, args.run)
            play_rows += play_analysis(data[split], core.play.thresholds.numpy(), split, seed)
            row, profile, validation = geometry_analysis(core, data[split], split, seed)
            geometry_rows.append(row)
            profiles += profile
            check["geometry_checks"] = validation
            checks.append(check)
        families = {"路径记忆8维": (data["train"]["q"], data["test"]["q"]),
                    "时间记忆24维": (data["train"]["d"], data["test"]["d"]),
                    "双记忆32维": (np.column_stack([data["train"]["q"].reshape(-1, 8), data["train"]["d"].reshape(-1, 24)]),
                                    np.column_stack([data["test"]["q"].reshape(-1, 8), data["test"]["d"].reshape(-1, 24)]))}
        for c in range(4):
            families[f"时间记忆通道{c}六尺度"] = (data["train"]["d"][:, c], data["test"]["d"][:, c])
        basis_saved = {}
        for i, (label, (train, test)) in enumerate(families.items()):
            row, eigen, basis = pca_analysis(train, test, label, seed)
            state_rows.append(row)
            eigens += eigen
            basis_saved.update({f"family_{i}_{key}": value for key, value in basis.items()})
        np.savez_compressed(args.run / f"seed_{seed}_train_pca_basis.npz", **basis_saved)
        k, f, km = kernel_analysis(core, reference_action, seed, channel)
        kernels += k
        frequency_rows += f
        kernel_matrices.append(km)
        metadata.append(dict(seed=seed, checkpoint=path, selected_epoch=saved["selected_epoch"],
                             thresholds=core.play.thresholds.numpy(), taus_s=core.maxwell.taus.numpy(),
                             dt_s=core.maxwell.dt, residual_mode=core.residual_mode,
                             reference_kind=core.reference_kind,
                             static_pair_interactions=core.reference_pair_interactions))
        print(f"seed {seed}: test Jacobian remainder {geometry_rows[-1]['linearization_node_error_mm']:.6f} mm", flush=True)

    matrices = np.stack(kernel_matrices)
    mean_kernel = matrices.mean(axis=0)
    relative_kernel_deviation = [float(np.linalg.norm(m - mean_kernel) / np.linalg.norm(mean_kernel)) for m in matrices]
    np.savez_compressed(args.run / "time_composite_kernels.npz", physical_generalized_kernel=matrices,
                        reference_action=reference_action)
    state_summary = pooled_summary(state_rows, ["family"], ["participation_rank", "k95", "k99",
                                  "test_energy_at_train_k95", "test_energy_at_train_k99", "condition_number"])
    geometry_summary = pooled_summary(geometry_rows, ["split"], ["memory_node_displacement_mm",
        "linearization_node_error_mm", "linearization_endpoint_error_mm", "relative_rms_error",
        "explained_displacement_energy", "median_shape_cosine", "p95_cumulative_bend_change_rad"])
    eigen_summary = pooled_summary(eigens, ["family", "component"], ["train_cumulative", "test_cumulative", "relative_singular"])
    play_summary = pooled_summary(play_rows, ["split", "threshold"], ["interior_fraction", "clipped_fraction",
        "reversal_events", "reversal_to_interior_events", "never_saturated_window_fraction"])
    profile_summary = pooled_summary(profiles, ["split", "node"], ["actual_memory_displacement_mm",
        "jacobian_displacement_mm", "remainder_mm", "pi_linear_rms_mm", "time_linear_rms_mm"])
    kernel_summary = pooled_summary(kernels, ["lag_s", "channel", "node"], ["lateral_gain_mm", "longitudinal_gain_mm"])
    for name, rows in dict(state_dimension_seed=state_rows, state_spectrum_seed=eigens,
                           play_regimes_seed=play_rows, geometry_seed=geometry_rows,
                           geometry_profile_seed=profiles, time_kernel_seed=kernels).items():
        write_csv(args.run / (name + ".csv"), rows)

    charts = [chart("representation_state_spectrum", "训练历史状态的累计方差与测试投影",
        [dict(family=r["family"], component=r["component"], source=source,
              series=r["family"] + " / " + source, cumulative_pct=100 * r[field],
              seed_sd_pct=100 * r[field + "_sd"]) for r in eigen_summary
         if r["family"] in ("时间记忆24维", "双记忆32维")
         for source, field in (("训练", "train_cumulative"), ("测试", "test_cumulative"))],
        x="component", y="cumulative_pct", color="series", unit="%",
        description="每个seed仅用train做中心化、标准化和PCA；test使用冻结基。方差解释率描述表示冗余，不是预测精度。"),
        chart("representation_tau_spectrum", "每通道六个时间基的奇异值谱",
              [r for r in eigen_summary if "六尺度" in r["family"]],
              x="component", y="relative_singular", color="family", unit="相对首奇异值",
              description="均为训练数据；各时间基先按训练标准差标准化。较小奇异值提示单个时间尺度系数的条件性。"),
        chart("representation_play_regimes", "当前路径更新处于内部线性区或裁剪区的比例",
              [dict(split=r["split"], threshold=r["threshold"], regime=label,
                    series=r["split"] + " / " + label, fraction_pct=100 * r[key])
               for r in play_summary for label, key in (("内部线性区", "interior_fraction"), ("裁剪区", "clipped_fraction"))],
              kind="bar", x="threshold", y="fraction_pct", color="series", unit="%",
              description="仅当前驱动变化绝对值>1e-6的通道更新。阈值单位为归一化驱动e，不是kPa。等于边界的少数更新单独保存在表中。"),
        chart("representation_geometry_profile", "几何Jacobian解释测试集的沿臂记忆位移",
              [dict(node=r["node"], method=label, displacement_mm=r[key], seed_sd_mm=r[key + "_sd"])
               for r in profile_summary if r["split"] == "test"
               for label, key in (("实际模型位移", "actual_memory_displacement_mm"),
                                  ("Jacobian一阶位移", "jacobian_displacement_mm"), ("一阶余项", "remainder_mm"))],
              x="node", y="displacement_mm", color="method", unit="mm",
              description="比较G(reference+memory)-G(reference)与J(reference)memory；actual指固定模型的精确几何输出，不是独立实测的记忆形变量。"),
        chart("representation_geometry_seed", "全部五次训练的Jacobian解释能量",
              [dict(seed=r["seed"], split=r["split"], explained_pct=100 * r["explained_displacement_energy"])
               for r in geometry_rows], kind="bar", x="seed", y="explained_pct", color="split", unit="%",
              description="1-Σ||精确位移-一阶位移||²/Σ||精确位移||²；含全部节点与窗口。柱图纵轴应从0开始。"),
        chart("representation_branch_geometry", "两类记忆经局部几何映射产生的沿臂位移尺度",
              [dict(node=r["node"], branch=label, rms_mm=r[key]) for r in profile_summary if r["split"] == "test"
               for label, key in (("路径记忆", "pi_linear_rms_mm"), ("时间记忆", "time_linear_rms_mm"))],
              x="node", y="rms_mm", color="branch", unit="mm",
              description="固定参考处Jm_branch的RMS；分支向量可相互抵消，两个RMS不能相加成总位移。"),
        chart("representation_time_kernel", "时间记忆的合成几何卷积核", kernel_summary,
              x="lag_s", y="lateral_gain_mm", color="node", unit="mm / 单位Δe",
              description=f"通道{channel}由train输入增量方差最大预选，参考动作由train均值的最近实际动作选取。局部时间项核，未含当前输入参考项及路径项，不等价于实物脉冲响应。五seed均值，SD字段随数据提供。"),
        chart("representation_time_frequency", "固定时间基对输入电平的离散频率响应",
              [r for r in frequency_rows if r["seed"] == 0], x="frequency_hz", y="amplitude", color="tau_s",
              unit="|d/e|", description="由D/E=-α(1-z⁻¹)/(1-αz⁻¹)精确计算；dt=0.2s，Nyquist=2.5Hz；这是数学响应，不是物理频率辨识。")]
    result = dict(schema="selfsr.analysis.representation.v1", title="HOV记忆表示：数学结构与数值证据",
                  study=args.study, output=args.run, seeds=list(range(5)), inventory=inventory, metadata=metadata,
                  definitions=dict(
                    q="q=e-p；路径记忆读出。q_t=clip(q_(t-1)+Δe_t,-r,r)。",
                    d="d=h-e；时间记忆所使用的偏差。d_t=αd_(t-1)-αΔe_t。",
                    effective_rank="participation ratio=(Σσ²)²/Σσ⁴，训练中心化与逐列标准化后计算。",
                    test_pca="只在train求均值/标准差/基，test以同一中心的总二阶能量为分母；不是test自行PCA。",
                    geometry="显式几何G的Jacobian在当前输入参考形态处评价，包含静态通道耦合。",
                    statistical_unit="全部seed0..4；先池化各split的帧，再汇总五seed。无帧独立显著性检验。",
                    kernel="仅时间记忆对驱动增量的局部形变核；当前参考固定，不包含输入引起的全部形变。"),
                  tables=dict(state_dimension=state_summary, state_dimension_seed=state_rows,
                              play_regimes=play_summary, play_regimes_seed=play_rows,
                              geometry=geometry_summary, geometry_seed=geometry_rows,
                              data_inventory=inventory,
                              time_kernel_consistency=[dict(seed=i, relative_frobenius_deviation_from_mean=v)
                                                       for i, v in enumerate(relative_kernel_deviation)]),
                  charts=charts,
                  kernel_reference=dict(action_normalized=reference_action, train_pooled_index=reference_index,
                                        selected_channel=channel, selection="train input variance; no shape label/error"),
                  validation=dict(status="passed", checks=checks,
                      finite_values="JSON allow_nan=False; zero-variance columns explicitly excluded from PCA",
                      geometry_bound="Integral second-order bound checked on every train/test window that stays inside length cap",
                      scope="数学恒等验证与固定模型解释；不作物理参数唯一辨识、记忆实测真值或因果分离主张"),
                  provenance=["src/operators/play_bank.py", "src/operators/maxwell_bank.py",
                              "src/models/model_hereditary_geometry.py", "src/models/model_ishsm.py",
                              "src/benchmarks/modeling_geometry_calibration.py",
                              "workspace/reports/modeling_mechanisms_20260913_001/history_mechanisms.json",
                              "workspace/reports/modeling_mechanisms_20260913_001/time_memory.json"])
    recompute = []
    for seed in range(5):
        with np.load(args.run / f"seed_{seed}_test_states_geometry.npz") as saved_arrays:
            exact, linear = saved_arrays["exact"], saved_arrays["linear"]
            error = exact - linear
            row = next(v for v in geometry_rows if v["seed"] == seed and v["split"] == "test")
            mean_error = float(np.linalg.norm(error, axis=-1).mean())
            explained = float(1 - np.sum(error ** 2) / np.sum(exact ** 2))
            assert abs(mean_error - row["linearization_node_error_mm"]) < 1e-10
            assert abs(explained - row["explained_displacement_energy"]) < 1e-10
            recompute.append(dict(seed=seed, remainder_mean_mm=mean_error, explained_energy=explained,
                                  status="passed"))
    result["validation"]["independent_saved_array_recompute"] = recompute
    result["equations"] = dict(
        path=r"q_t=\operatorname{clip}(q_{t-1}+\Delta e_t,-r,r),\quad q_t=e_t-p_t",
        time=r"d_{k,t}=\alpha_k(d_{k,t-1}-\Delta e_t)=\alpha_k^t d_{k,0}-\sum_{j=1}^{t}\alpha_k^{t-j+1}\Delta e_j",
        time_kernel=r"m_t^{\mathrm{time}}=\sum_{\ell=0}^{t-1}K_\ell\Delta e_{t-\ell},\quad (K_\ell)_c=-\sum_k W_{c,k}^{\mathrm{time}}\alpha_k^{\ell+1}",
        geometry=r"G(\xi_{\mathrm{ref}}+m)-G(\xi_{\mathrm{ref}})=J(\xi_{\mathrm{ref}})m+R",
        segment_geometry=r"\Delta Y_n=\sum_{i\le n}L_i[v_i^\perp\delta\theta_i+v_i\delta\lambda_{s(i)}]+R_n",
    )
    result["findings"] = findings(result)
    write_json(args.report / "representation.json", result)
    write_json(args.run / "validation.json", result["validation"])
    write_json(args.run / "analysis_config.json", dict(study=args.study, seeds=list(range(5)),
        history=20, dt_s=.2, train_windows=8988, test_windows=2958, threads=args.threads,
        pca_scaling="train only, feature-wise unit SD", geometry_reference="current-input calibrated reference",
        selection="all frozen formal checkpoints; no refitting", command="python scripts/experiments/analyze_modeling_representation.py"))
    (args.report / "representation.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps(result["findings"], ensure_ascii=False, indent=2), flush=True)


def findings(result):
    geometry = next(r for r in result["tables"]["geometry"] if r["split"] == "test")
    dual = next(r for r in result["tables"]["state_dimension"] if r["family"] == "双记忆32维")
    return dict(
        stop="路径记忆等价于截断累积输入增量，内部区保存最近的play锚点p；从一侧饱和边界到另一侧需2r的单调反向驱动行程。",
        time="时间记忆偏差是带负号的指数增量卷积；τ固定，其合成读出形成可计算的局部时间核。",
        geometry=f"测试集实际模型记忆位移均值{geometry['memory_node_displacement_mm']:.4f}mm；Jacobian一阶余项{geometry['linearization_node_error_mm']:.4f}mm，解释位移能量{100 * geometry['explained_displacement_energy']:.4f}%。",
        dimensions=f"32维双记忆的训练参与率有效维数{dual['participation_rank']:.3f}；训练99%子空间平均需{dual['k99']:.1f}维，对测试状态解释{100 * dual['test_energy_at_train_k99']:.3f}%。",
        boundary="可解释的是算子记忆、合成时间核与几何传递；这些结果不把各τ或各阈值权重等同为唯一材料参数。")


def markdown(r):
    g = next(x for x in r["tables"]["geometry"] if x["split"] == "test")
    d = next(x for x in r["tables"]["state_dimension"] if x["family"] == "双记忆32维")
    time_channels = [x for x in r["tables"]["state_dimension_seed"] if "六尺度" in x["family"]]
    min_cond = min(x["condition_number"] for x in time_channels)
    max_cond = max(x["condition_number"] for x in time_channels)
    play = [x for x in r["tables"]["play_regimes"] if x["split"] == "test"]
    rows = "\n".join(f"| {x['threshold']:.3f} | {100*x['interior_fraction']:.2f}% | {100*x['clipped_fraction']:.2f}% | {x['reversal_to_interior_events']:.1f} |" for x in play)
    dims = "\n".join(f"| {x['family']} | {x['participation_rank']:.2f} | {x['k99']:.1f} | {100*x['test_energy_at_train_k99']:.2f}% |"
                     for x in r["tables"]["state_dimension"] if "六尺度" not in x["family"])
    return rf"""# HOV记忆表示：数学结构与数值证据

使用正式三序列5Hz研究的全部五个冻结HOV模型（seed 0–4），在8988个训练窗口及2958个测试窗口上分析。窗口长20步，首帧令$p_0=h_0=e_0$。以下分析解释固定模型内部表示，不进行再训练或重新选择测试划分。数值先池化对应split全部窗口，再对五seed求均值与样本标准差。

## 可用于论文的小节：历史信息的表示与几何传递

**路径记忆存储有界的驱动行程。** 对单个通道及阈值$r$，令$e_t$为归一化单调驱动、$p_t$为play变量。由$p_t=\operatorname{{clip}}(p_{{t-1}},e_t-r,e_t+r)$及$q_t=e_t-p_t$可直接得到

$$q_t=\operatorname{{clip}}\left(q_{{t-1}}+\Delta e_t,-r,r\right),\qquad\Delta e_t=e_t-e_{{t-1}}.$$

当$q_{{t-1}}+\Delta e_t$处于$(-r,r)$内时，$p_t=p_{{t-1}}$，而$q_t$随当前输入以单位斜率变化；越过边界后$q_t$被限制为$\pm r$，$p_t$随输入移动。因此，$p$保留由初始值或最近一次边界接触确定的锚点，$q$表达相对该锚点的有界驱动位移。若在$q=r$处开始单调反向运动，则在反向行程$s\in[0,2r]$内$q=r-s$，到$s=2r$才达到另一边界。保持输入时$q$保持不变。这一表示区分了反转方向以及反转后的累计行程，而不仅是当前加卸载符号。上述分段斜率是以先前状态固定、当前$e_t$为自变量的偏导；折点处不可微，且$r$的单位是归一化驱动而非压力单位。

正式模型使用$r\in\{{0.02,0.5\}}$。测试集中，有输入变化的通道更新落在内部区或裁剪区的比例如下。计数是窗口末步的状态事件，五seed共享输入，不代表独立物理重复。

| 阈值r | 内部线性区 | 裁剪区 | 从饱和边界反转至内部的事件数，五seed均值 |
|---|---:|---:|---:|
{rows}

这项统计说明实际模型使用了不同的局部更新区间；递推恒等及斜率本身属于结构保证，不应作为预测效果的独立实验证明。对于$r=0.5$，约{100*play[-1]['never_saturated_window_fraction']:.2f}%的有效通道更新所在窗口尚未触及边界；这类窗口满足$q_t=e_t-e_0$，其锚点仍由窗口初始化确定。

**时间记忆是多尺度的输入增量卷积。** 记$\alpha_k=\exp(-\Delta t/\tau_k)$、$d_{{k,t}}=h_{{k,t}}-e_t$，则实际实现

$$h_{{k,t}}=\alpha_k h_{{k,t-1}}+(1-\alpha_k)e_t$$

等价于

$$d_{{k,t}}=\alpha_kd_{{k,t-1}}-\alpha_k\Delta e_t
=\alpha_k^td_{{k,0}}-\sum_{{j=1}}^t\alpha_k^{{t-j+1}}\Delta e_j.$$

当前窗口采用$d_{{k,0}}=0$。因此每个时间基保留的是不同衰减尺度上的有符号输入变化；恒定输入下$d_{{k,t+n}}=\alpha_k^nd_{{k,t}}$。令$\widetilde e_t=e_t-e_0$，在零初始偏差和固定采样间隔下，其相对该输入扰动的离散传递函数为

$$\frac{{D_k(z)}}{{\widetilde E(z)}}=-\frac{{\alpha_k(1-z^{{-1}})}}{{1-\alpha_kz^{{-1}}}}.$$

该关系表达低频或恒定成分的衰减与历史变化的保留。时间分支的几何读出为$m_t^{{\mathrm{{time}}}}=\sum_{{c,k}}W_{{c,k}}^{{\mathrm{{time}}}}d_{{c,k,t}}$，从而可进一步写成增量的合成卷积

$$m_t^{{\mathrm{{time}}}}=\sum_{{\ell=0}}^{{t-1}}K_\ell\Delta e_{{t-\ell}},\qquad (K_\ell)_c=-\sum_k W_{{c,k}}^{{\mathrm{{time}}}}\alpha_k^{{\ell+1}}.$$

固定模型的六个$\tau$为0.600、0.763、0.971、1.236、1.572、2.000秒；它们构成预设时间基。训练数据逐通道标准化后的六尺度状态矩阵条件数范围为{min_cond:.1f}–{max_cond:.1f}，说明逐个尺度的系数需要谨慎解释。采用训练集中心、标准差与PCA基得到下表；测试列使用同一冻结投影。

| 状态族 | 参与率有效维数 | 训练99%方差所需维数，五seed均值 | 该子空间保留的测试能量 |
|---|---:|---:|---:|
{dims}

参与率定义为$(\sum_i\sigma_i^2)^2/\sum_i\sigma_i^4$。双记忆的32个数值状态具有约{d['participation_rank']:.2f}维的参与率有效维数；其训练99%子空间保留测试状态能量的{100*d['test_energy_at_train_k99']:.2f}%。该结果反映本数据激励下的状态相关性，并不意味着可以删除其余状态而保持相同预测精度。报告同时给出合成时间核及五seed差异，以展示各时间尺度经过学习读出后共同形成的响应，而非把单尺度权重解释为独立材料谱。

**几何Jacobian将历史状态联系到沿臂位移。** 令当前输入的参考坐标为$\xi_{{\mathrm{{ref}}}}(u_t)$，记忆修正为$m_t=m_t^{{\mathrm{{path}}}}+m_t^{{\mathrm{{time}}}}$，则骨架预测为$Y_t=G(\xi_{{\mathrm{{ref}}}}+m_t)$。参考坐标包含当前输入的单通道形变及静态通道耦合。对第$i$条线段，设切向角$\theta_i=\sum_{{j\le i}}b_j$、长度$L_i=L_i^0\exp(\lambda_{{s(i)}})$、切向量$v_i=(\cos\theta_i,\sin\theta_i)$，在长度限制未激活的区域中

$$\Delta Y_n=\sum_{{i\le n}}L_i\left(v_i^\perp\,\delta\theta_i+v_i\,\delta\lambda_{{s(i)}}\right)+R_n
=J_n(\xi_{{\mathrm{{ref}}}})m_t+R_n.$$

弯曲修正通过累计角度产生沿臂传播的法向位移，长度修正产生沿切向的伸缩位移。这给出从路径／时间状态，经局部弯曲与分段长度，到骨架位置的显式关系。对固定参考，$Jm_t^{{\mathrm{{path}}}}$与$Jm_t^{{\mathrm{{time}}}}$可相加，但各自位移范数或RMS不可直接相加。进一步，时间记忆的局部位置核就是$J(\xi_{{\mathrm{{ref}}}})K_\ell$。

在全部测试窗口，模型产生的完整记忆位移$G(\xi_{{\mathrm{{ref}}}}+m)-G(\xi_{{\mathrm{{ref}}}})$的节点平均幅值为 **{g['memory_node_displacement_mm']:.4f}±{g['memory_node_displacement_mm_sd']:.4f} mm**，一阶近似的平均余项为 **{g['linearization_node_error_mm']:.4f}±{g['linearization_node_error_mm_sd']:.4f} mm**，相对RMS误差为 **{100*g['relative_rms_error']:.3f}%**。按$1-\sum\|R\|^2/\sum\|\Delta Y\|^2$定义，一阶映射解释 **{100*g['explained_displacement_energy']:.4f}%** 的模型记忆位移能量。这说明在当前数据和学习到的记忆幅值范围内，沿臂形变能够由几何传递关系准确解释。这里的精确位移来自固定模型，不能当作分离出的真实机器人迟滞形变真值。

## 数学与实现核验

1. 对所有训练／测试窗口、所有seed，独立重写stop形式核对play更新；以显式指数加权和核对时间递推。
2. 重新计算测试预测，与保存的正式预测逐帧对齐；不更改标签或序列划分。
3. 每个seed、每个split按索引均匀取8个窗口，用双精度PyTorch自动微分和中心有限差分分别核对解析Jacobian，共80个窗口。
4. 对长度限制未激活的全部窗口，检查逐节点二阶余项上界。令$z_i=\delta\lambda_i+\mathrm{{i}}\delta\theta_i$，由积分Taylor余项得到

$$\|R_n\|\le\tfrac12\sum_{{i\le n}}L_i\exp(\max(\delta\lambda_i,0))\left(\delta\lambda_i^2+\delta\theta_i^2\right).$$

5. PCA只使用train拟合。图中的均值／SD涵盖全部五seed；频率响应为固定公式计算，仅画一份固定网格。

全部检查通过，详细最大差异、每seed指标、状态／位移及训练PCA基保存在`workspace/runs/analysis/modeling_paper_revision_20260913_002/representation/`。图表数据含8组通用charts及逐seed tables，供主报告直接导入。

## 解释范围

该分析支持“记忆状态具有明确的驱动历史含义，其学习读出可经显式几何映射为沿臂位移”。它不提供材料迟滞、气路动态与测量误差的唯一分离，也不把六个固定时间基当作辨识出的六种独立物理过程。窗口初始化限制了当前可访问的前史。几何一阶精度是当前数据范围内的数值发现；无需把它推广为全局静态近似或真实机器人线性规律。

复现：`/Data5/ddf/environments/conda_envs/selfsr/bin/python scripts/experiments/analyze_modeling_representation.py --threads 2`。
"""


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, default=STUDY)
    p.add_argument("--report", type=Path, default=REPORT)
    p.add_argument("--run", type=Path, default=RUN)
    p.add_argument("--threads", type=int, default=2)
    build(p.parse_args())
