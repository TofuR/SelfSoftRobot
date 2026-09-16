#!/usr/bin/env python3
"""Audit learned time kernels and their relation to held-out geometric errors.

Reuses five frozen models. Random readouts are structural reference draws,
not trained models or independent robot experiments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_modeling_representation import (
    ROOT, RUN, STUDY, checkpoint, geometry_arrays, write_csv, write_json,
)

OUT = ROOT / "workspace/runs/analysis/modeling_branch_review_20260913_003"


def energy_fraction(matrix, normalize=False):
    a = np.asarray(matrix, dtype=float)
    if normalize:
        norms = np.linalg.norm(a, axis=-1, keepdims=True)
        a = a / np.maximum(norms, 1e-12)
    s = np.linalg.svd(a, compute_uv=False)
    return float(s[0] ** 2 / np.sum(s ** 2))


def metrics(target, reference, correction):
    residual = target[..., :2] - reference
    remaining = residual - correction
    return dict(
        reference_node_mm=float(np.linalg.norm(residual, axis=-1).mean()),
        corrected_node_mm=float(np.linalg.norm(remaining, axis=-1).mean()),
        residual_energy_reduction=float(1 - np.sum(remaining ** 2) / np.sum(residual ** 2)),
        correction_residual_cosine=float(np.sum(correction * residual) /
            np.sqrt(np.sum(correction ** 2) * np.sum(residual ** 2))),
        diagnostic_projection_scale=float(np.sum(correction * residual) / np.sum(correction ** 2)),
    )


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(RUN / "time_composite_kernels.npz")
    action = np.load(RUN / "seed_0_train_states_geometry.npz")["action"]
    standardized = (action - action.mean(0)) / action.std(0)
    _, _, basis = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ basis[0]
    indices = [int(np.argmin(abs(scores - np.quantile(scores, q)))) for q in (.1, .3, .5, .7, .9)]
    references = np.vstack([saved["reference_action"], action[indices]])
    rows, residual_rows, curves, basis_rows, random_rows, projections = [], [], [], [], [], []
    max_kernel_difference = 0.
    for seed in range(5):
        core, _, _ = checkpoint(STUDY, seed)
        alpha = core.maxwell.decays.numpy().astype(float)
        scale = core.generalized_coordinate_scale.detach().numpy().astype(float)
        w_unit = (core.maxwell_gains[:, :, None] * core.maxwell_mode_directions).detach().numpy().astype(float)
        w = (core.maxwell_gains[:, :, None] * core.maxwell_mode_directions *
             core.generalized_coordinate_scale[None, None, :]).detach().numpy().astype(float)
        phi = alpha[:, None] ** np.arange(1, 52)[None, :]
        kernels = -np.einsum("ckg,kl->lcg", w, phi)
        max_kernel_difference = max(max_kernel_difference, float(np.max(abs(kernels - saved["physical_generalized_kernel"][seed]))))
        if seed == 0:
            for count in (19, 20, 51):
                basis_rows.append(dict(lag_samples=count, last_lag_s=.2 * (count - 1),
                    rank1_energy=energy_fraction(phi[:, :count]),
                    normalized_rank1_energy=energy_fraction(phi[:, :count], True)))
        for ref_index, ref in enumerate(references):
            # Exact analytic local x-position Jacobian, 14 non-base nodes x 16 coordinates.
            jac = geometry_arrays(core, np.repeat(ref[None], 16, axis=0), np.eye(16))["linear"][:, 1:, 0].T
            for channel in range(4):
                matrix = -jac @ w[channel].T @ phi
                for subset, node_indices in (("all_nonbase", np.arange(14)), ("plotted", np.array([2, 6, 10, 13]))):
                    for count in (19, 20, 51):
                        m = matrix[node_indices, :count]
                        rows.append(dict(seed=seed, reference=ref_index, channel=channel,
                            nodes=subset, lag_samples=count, rank1_energy=energy_fraction(m),
                            normalized_rank1_energy=energy_fraction(m, True)))
                if ref_index == 0 and channel == 3:
                    curves.extend(dict(seed=seed, node=int(i+1), lag_s=float(.2*l), gain=float(matrix[i,l]))
                                  for i in range(14) for l in range(51))
                # Match each time mode's standardized coefficient norm. Retain geometry and poles.
                # These draws illustrate a structural reference, not a null distribution for a p-value.
                if ref_index == 0:
                    rng = np.random.default_rng(20260913 + seed * 4 + channel)
                    for draw in range(100):
                        direction = rng.normal(size=w_unit[channel].shape)
                        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
                        randomized = direction * np.linalg.norm(w_unit[channel], axis=-1, keepdims=True) * scale[None, :]
                        m = -jac @ randomized.T @ phi[:, :19]
                        random_rows.append(dict(seed=seed, channel=channel, draw=draw,
                            rank1_energy=energy_fraction(m), normalized_rank1_energy=energy_fraction(m, True)))
        for split in ("train", "test"):
            with np.load(RUN / f"seed_{seed}_{split}_states_geometry.npz") as d:
                linear_path = geometry_arrays(core, d["action"], d["pi"])["linear"]
                linear_time = geometry_arrays(core, d["action"], d["time"])["linear"]
                p, t = linear_path.ravel(), linear_time.ravel()
                r = (d["target"][..., :2] - d["reference"]).ravel()
                gram = np.array([[p @ p, p @ t], [p @ t, t @ t]])
                coefficients = np.linalg.solve(gram, np.array([p @ r, t @ r]))
                projections.append(dict(seed=seed, split=split,
                    branch_cosine=float((p @ t) / np.sqrt((p @ p) * (t @ t))),
                    path_projection=float(coefficients[0]), time_projection=float(coefficients[1]),
                    overlap_energy_fraction=float(2 * (p @ t) / (r @ r))))
                for name, correction in (("joint_exact", d["exact"]), ("joint_linear", linear_path + linear_time),
                                         ("path_linear", linear_path), ("time_linear", linear_time)):
                    residual_rows.append(dict(seed=seed, split=split, correction=name,
                        **metrics(d["target"], d["reference"], correction)))
    assert max_kernel_difference < 1e-10, max_kernel_difference
    write_csv(OUT / "kernel_rank.csv", rows)
    write_csv(OUT / "structural_readout_reference.csv", random_rows)
    write_csv(OUT / "test_residual_alignment.csv", residual_rows)
    write_csv(OUT / "lateral_kernel.csv", curves)
    write_json(OUT / "branch_joint_projection.json", projections)
    write_json(OUT / "protocol.json", dict(
        source=RUN, checkpoints=STUDY / "formal/hov", training_seeds=list(range(5)),
        reference_actions=references, reference_selection="original plus train-input PCA score quantiles .1/.3/.5/.7/.9",
        kernel_units="mm / unit normalized drive increment; local time-memory term",
        primary_lags="0..18 at dt=.2: H20 contains 19 increments; largest increment age 3.6s",
        compatibility_lags="20 samples to reproduce prior 0..3.8s display; 51 samples is analytic extension to 10s",
        rank="uncentered SVD; optionally unit L2 normalization of each node curve; base node excluded",
        random_reference="100 fixed untrained coefficient-direction draws per seed/channel at original reference; same per-mode standardized coefficient norm, poles and geometry; no inferential p-value",
        residual_target="observed skeleton minus fitted reference; this is prediction error, not isolated physical hysteresis ground truth",
        projection_scale="diagnostic scalar only; never applied to predictions or selected models",
        max_saved_kernel_difference=max_kernel_difference,
    ))
    def summary(part, fields):
        return {k: dict(mean=float(np.mean([r[k] for r in part])),
                        min=float(min(r[k] for r in part)), max=float(max(r[k] for r in part))) for k in fields}
    rank_fields = ["rank1_energy", "normalized_rank1_energy"]
    selected = [r for r in rows if r["reference"] == 0 and r["channel"] == 3 and r["nodes"] == "plotted" and r["lag_samples"] == 19]
    allref = [r for r in rows if r["nodes"] == "all_nonbase" and r["lag_samples"] == 19]
    original = [r for r in allref if r["reference"] == 0]
    result = dict(exponential_basis=basis_rows, original_plotted=summary(selected, rank_fields),
        original_all_channels_nodes=summary(original, rank_fields),
        all_references_channels_nodes=summary(allref, rank_fields),
        random_readout_reference=summary(random_rows, rank_fields),
        residuals={f"{split}/{kind}": summary([r for r in residual_rows if r["split"] == split and r["correction"] == kind],
                    ["reference_node_mm", "corrected_node_mm", "residual_energy_reduction", "correction_residual_cosine", "diagnostic_projection_scale"])
                   for split in ("train", "test") for kind in ("joint_exact", "joint_linear", "path_linear", "time_linear")})
    write_json(OUT / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
